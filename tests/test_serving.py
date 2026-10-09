"""How the model server batches (d1a.serving.serve): the state-prefix cache, the retry after running out of device
memory, the permute demo endpoint and the API key. Also the pass-sizing it relies on: d1a.core.encoding.rows_per_pass,
d1a.backends.cuda_graphs, and DecisionModel._rows_hidden's copies of a cached state. No weights; stand-in models record
what the server asks of them."""
import itertools
import random
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer

import d1a.backends.torch as torch_backend
from d1a.backends.cuda_graphs import PASS_TOKENS, bucket, count_bucket, length_groups
from d1a.backends.device import out_of_memory
from d1a.core.encoding import ROW_PASS_TOKENS, rows_per_pass
from d1a.serving import serve
from d1a.serving.serve import PREFIX_MAX_TOKENS, PrefixCache, Server

STATES = [tuple(f"{name}{i}" for i in range(n)) for name, n in zip("abcdefgh", (1, 2, 3, 4, 6, 9, 12, 20))]


def packed(state, branch=5, media=None):
    """A packed request whose state is the token ids `state`, then one branch."""
    enc = {"ids": list(state) + [0] * branch, "seg": [0] * len(state) + [1] * branch}
    if media is not None:
        enc["media"] = media
    return enc


# --- the state-prefix cache ---------------------------------------------------------------------------------------------

def test_prefix_cache_keeps_exactly_what_its_batch_would_leave():
    """plan() keeps (copies) only the new states that would still be cached once the whole batch was stored: the batch's
    last distinct cacheable states within the count and token bounds. Copying any other one is wasted, since the batch
    would evict it itself. A state is cacheable when the cache is on, its length is within [min_tokens, max_tokens] and
    it carries no media (placeholder ids are the same for every photo)."""
    rng = random.Random(1)
    for _ in range(300):
        size, low, high = rng.randint(0, 4), rng.randint(0, 4), rng.randint(5, 25)
        batch = [packed(rng.choice(STATES), media=rng.choice([None, None, None, "photo"])) for _ in range(rng.randint(1, 8))]
        keys, cached, keep = PrefixCache(size, low, high).plan(batch)
        assert [k is not None for k in keys] == [bool(size) and low <= e["seg"].count(0) <= high and "media" not in e for e in batch]
        assert cached == [None] * len(batch)
        whole = PrefixCache(size, low, high)
        whole.store(keys, cached, [None if k is None else f"prefix {i}" for i, k in enumerate(keys)])
        assert {k for k, kept in zip(keys, keep) if kept} == set(whole.entries)


def test_prefix_cache_holds_the_most_recently_used_states_that_fit():
    """Across batches the cache serves repeated states and counts hits and misses. A state reused and stored again counts
    as most recent. The cache never holds more than `size` states or `max_tokens` state tokens: it holds the most
    recently used states that fit, newest first, until one does not. Checked against that rule replayed from the history
    of every state stored."""
    assert PrefixCache(4, 0).max_tokens == PREFIX_MAX_TOKENS == 65536
    rng = random.Random(2)
    for _ in range(100):
        cache = PrefixCache(rng.randint(1, 4), rng.randint(0, 3), rng.randint(6, 30))
        stored, hits, misses = [], 0, 0
        for _ in range(12):
            keys, cached, keep = cache.plan([packed(rng.choice(STATES)) for _ in range(rng.randint(1, 5))])
            assert all(c is (cache.entries[k] if k in cache.entries else None) for k, c in zip(keys, cached) if k is not None)
            prefixes = [f"prefix of {k}" if kept else None for k, kept in zip(keys, keep)]   # the model makes one per kept state
            cache.store(keys, cached, prefixes)
            hits += sum(k is not None and c is not None for k, c in zip(keys, cached))
            misses += sum(k is not None and c is None for k, c in zip(keys, cached))
            stored += [k for k, p in zip(keys, prefixes) if p is not None]
            newest, tokens = [], 0
            for key in dict.fromkeys(reversed(stored)):
                if len(newest) == cache.size or tokens + len(key) > cache.max_tokens:
                    break
                newest.append(key); tokens += len(key)
            assert list(cache.entries) == newest[::-1] and (cache.hits, cache.misses) == (hits, misses)


# --- running out of device memory ---------------------------------------------------------------------------------------

class PassTensors:
    """Stands for the tensors one forward pass allocates."""


class StandInModel:
    """What Server asks of a model. Each pass records the object standing for its tensors. `fail` makes passes raise:
    "cached" runs out of memory when any state comes from the cache, "always" runs out of memory every time, and "other"
    raises an error unrelated to memory."""
    prefix_min_tokens = 0

    def __init__(self):
        self.fail, self.calls, self.passes = None, 0, []

    def encode(self, tok, rec, **limits):
        return rec

    def probs_batch(self, encs, cached, keep):
        self.calls += 1
        tensors = PassTensors(); self.passes.append(weakref.ref(tensors))
        if self.fail == "always" or self.fail == "cached" and any(c is not None for c in cached):
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self.fail == "other":
            raise ValueError("shapes do not match")
        return [[torch.tensor([0.25, 0.75])] for _ in encs], [("prefix", self.calls) if k else None for k in keep]


def test_running_out_of_memory_with_states_cached_drops_them_and_runs_once_more():
    """Server._run: a pass that runs out of device memory while states are cached clears the cache and runs once more.
    #75: kept resident, the cache failed every later batch of that size. A second failure fails the batch and leaves the
    cache empty. Running out with nothing cached, or any other error, is not retried. A failed batch's pass is freed with
    it: the model thread keeps the exception, whose frames held the pass's tensors (142 MiB on an H100)."""
    model = StandInModel()
    server = Server(SimpleNamespace(release_date=lambda: "2026-01-01"), None, model, "cpu")
    request = packed("abc", branch=1)
    entries = lambda: list(server.prefix_cache.entries.values())
    try:
        probs, stats = server.probs(request)
        assert probs == [[0.25, 0.75]] and stats["prefix_cache_hit"] is False and entries() == [("prefix", 1)]
        assert len(server.batch_ms) == 1   # each batch's model time feeds /v1/models' latency
        model.fail, model.calls = "cached", 0
        probs, stats = server.probs(request)   # the cached state fails the pass, the retry runs it from scratch
        assert probs == [[0.25, 0.75]] and stats["prefix_cache_hit"] is False and model.calls == 2
        assert server.prefix_cache.oom_retries == 1 and entries() == [("prefix", 2)]
        model.fail, model.calls = "always", 0
        with pytest.raises(torch.OutOfMemoryError):
            server.probs(request)
        assert model.calls == 2 and entries() == [] and server.prefix_cache.oom_retries == 2
        server.wait_idle()
        assert all(tensors() is None for tensors in model.passes), "a failed batch's pass outlived it"
        model.calls = 0
        with pytest.raises(torch.OutOfMemoryError):
            server.probs(request)   # nothing cached: nothing to drop, no retry
        assert model.calls == 1 and server.prefix_cache.oom_retries == 2
        model.fail = None
        server.probs(request)
        model.fail, model.calls = "other", 0
        with pytest.raises(ValueError):
            server.probs(request)
        assert model.calls == 1 and len(entries()) == 1
    finally:
        server.close()


def test_make_room_evicts_before_the_pass_exactly_what_store_would_evict_after_it():
    """PrefixCache.make_room (from Kev 1d77363) drops, before a batch runs, the entries the batch's store() would evict
    anyway, so an old state never stays resident through the pass of the new one replacing it. Under the model's hit
    rule (a hit hands its cached prefix back), a cache that makes room and then stores ends identical, order and counts
    included, to one that only stores, and make_room drops exactly the old states that store would drop."""
    rng = random.Random(5)
    for _ in range(150):
        size, low, high = rng.randint(1, 4), rng.randint(0, 3), rng.randint(6, 30)
        plain, roomy = PrefixCache(size, low, high), PrefixCache(size, low, high)
        for _ in range(10):
            batch = [packed(rng.choice(STATES)) for _ in range(rng.randint(1, 5))]
            keys, cached, keep = plain.plan(batch)
            assert roomy.plan(batch) == (keys, cached, keep)
            prefixes = [c if c is not None else (f"prefix of {k}" if kept else None) for k, c, kept in zip(keys, cached, keep)]
            before = set(roomy.entries)
            roomy.make_room(keys, cached, keep)
            kept_through = set(roomy.entries)
            plain.store(keys, cached, prefixes); roomy.store(keys, cached, prefixes)
            assert list(roomy.entries.items()) == list(plain.entries.items()) and (roomy.hits, roomy.misses) == (plain.hits, plain.misses)
            assert kept_through == before & set(roomy.entries)   # it drops exactly the old states the store would drop


def test_the_out_of_memory_retry_still_fires_when_make_room_dropped_the_cached_state():
    """A batch whose hit make_room drops (the batch's own new state takes the only slot) still holds that hit's prefix
    for its pass. If the pass runs out of memory, it is retried once: the cache itself is already empty by then."""
    model = StandInModel()
    server = Server(SimpleNamespace(release_date=lambda: "2026-01-01"), None, model, "cpu")
    server.prefix_cache = PrefixCache(size=1, min_tokens=0)
    try:
        server.probs(packed("abc", branch=1))
        model.fail, model.calls = "cached", 0
        server._run([packed("abc", branch=1), packed("xyz", branch=1)])
        assert model.calls == 2 and server.prefix_cache.oom_retries == 1
        assert list(server.prefix_cache.entries) == [tuple("xyz")]
    finally:
        server.close()


def test_out_of_memory_is_what_cuda_and_mps_raise_when_their_allocator_runs_out():
    assert out_of_memory(torch.OutOfMemoryError("CUDA out of memory"))
    assert out_of_memory(RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"))
    assert not out_of_memory(RuntimeError("shape mismatch")) and not out_of_memory(ValueError("MPS backend out of memory"))


# --- the permute endpoint and the API key -------------------------------------------------------------------------------

@pytest.mark.parametrize("n_perm, status", [(0, 422), (-1, 422), (65, 422), (1, 200), (64, 200)])
def test_permute_answers_one_choice_question_under_each_order(n_perm, status, monkeypatch):
    """/v1/systemone/permute answers a choice question under n_perm option orders, one forward pass each. The first order
    is the request's own, every order is a permutation of the options, and counts outside 1..64 are refused before any
    pass (#30: 0 divided by nothing, and an unbounded count ran forever)."""
    passes = []

    def answer(req, log=True):   # favours whichever option comes first, so the choice moves with the order
        order = list(req.questions["q"].criteria); passes.append(order)
        probabilities = {k: 0.7 if k == order[0] else 0.1 for k in order}
        return {"answers": {"q": {"probabilities": probabilities, "choice": order[0]}}, "latency_ms": 1.0}
    monkeypatch.setattr(serve, "server", lambda: nullcontext(SimpleNamespace(answer=answer)))
    request = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None, "c": None, "d": None}}}}
    with TestClient(serve.app) as client:
        r = client.post("/v1/systemone/permute", json={"request": request, "question": "q", "n_perm": n_perm})
        assert client.post("/v1/systemone/permute", json={"request": request, "question": "nope"}).status_code == 422
    assert r.status_code == status
    if status != 200:
        assert passes == []
        return
    runs = r.json()["runs"]
    assert len(runs) == len(passes) == n_perm and runs[0]["order"] == ["a", "b", "c", "d"]
    assert all(run["order"] == order and sorted(order) == ["a", "b", "c", "d"] for run, order in zip(runs, passes))
    assert r.json()["argmax_stable"] == (len({run["choice"] for run in runs}) == 1)
    assert r.json()["spread"] == {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in "abcd"}


def test_api_key_gates_the_api_and_every_response_names_its_request(monkeypatch):
    """With D1A_API_KEY set (serve.API_KEY), /v1/* needs `Authorization: Bearer <key>` and the OpenAPI schema stays open.
    Every response, refusals included, carries the request id the TypeSafe clients read."""
    with TestClient(serve.app) as client:
        assert client.get("/openapi.json").headers["x-typesafe-request-id"]
        monkeypatch.setattr(serve, "API_KEY", "secret")
        for headers in ({}, {"authorization": "Bearer wrong"}, {"authorization": "secret"}):
            r = client.get("/v1/models", headers=headers)
            assert r.status_code == 401 and r.headers["x-typesafe-request-id"], headers
        assert client.get("/openapi.json").status_code == 200


# --- sizing passes ------------------------------------------------------------------------------------------------------

def test_rows_per_pass_fills_its_token_budget_and_never_drops_to_zero():
    """A pass takes as many causal rows as fit the budget, each counting the cached state it carries and its own tokens,
    and always at least one. A request's peak memory is then one maximal row however many questions it has."""
    rng = random.Random(3)
    for _ in range(300):
        rows = [[0] * rng.randint(1, 600) for _ in range(rng.randint(1, 80))]
        prefix, budget = rng.choice([0, 270, 4802, 8192, 20000]), rng.choice([ROW_PASS_TOKENS, 4096, 100])
        n, row = rows_per_pass(rows, prefix_len=prefix, budget=budget), prefix + max(map(len, rows))
        assert n >= 1 and (n == 1 or n * row <= budget) and (n + 1) * row > budget
    assert rows_per_pass([[0] * 8192], prefix_len=8192) == 1   # a maximal row exactly fills the default budget


def test_graph_buckets_pad_a_pass_by_little():
    """Batched passes are padded to a few shapes, so a CUDA graph is captured once per shape. A count goes up to the next
    of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 ..., adding under half. A token length grows by under a quarter, or under 16 tokens
    when short."""
    steps = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    assert [count_bucket(n) for n in range(1, 33)] == [min(s for s in steps if s >= n) for n in range(1, 33)]
    assert all(n <= count_bucket(n) < 1.5 * n for n in range(2, 500))
    assert all(n <= bucket(n) < max(1.25 * n, n + 16) and bucket(bucket(n)) == bucket(n) for n in range(1, 5000))


def test_length_groups_split_a_batch_into_its_cheapest_passes():
    """length_groups splits rows into passes of at most `cap` rows, in length order. It takes the fewest padded tokens,
    where a pass costs count_bucket(rows) x bucket(longest) and at least PASS_TOKENS. Checked against every split of the
    sorted lengths on small batches; on a tie the larger pass comes last."""
    cost = lambda groups, lengths: sum(max(PASS_TOKENS, count_bucket(len(g)) * bucket(max(lengths[i] for i in g))) for g in groups)
    rng = random.Random(4)
    for _ in range(80):
        lengths = [rng.choice([rng.randint(1, 64), rng.randint(65, 1200)]) for _ in range(rng.randint(1, 9))]
        cap, order = rng.randint(1, 6), sorted(range(len(lengths)), key=lengths.__getitem__)
        groups = length_groups(lengths, cap)
        assert [i for g in groups for i in g] == order and all(1 <= len(g) <= cap for g in groups)
        splits = ([order[a:b] for a, b in zip((0, *cuts), (*cuts, len(order)))]
                  for k in range(len(order)) for cuts in itertools.combinations(range(1, len(order)), k))
        assert cost(groups, lengths) == min(cost(s, lengths) for s in splits if all(len(g) <= cap for g in s))
    assert length_groups([40, 20, 35, 30, 25, 45], 32) == [[1, 4, 3, 2, 0, 5]]   # under PASS_TOKENS a pass stays whole
    assert length_groups([30] * 20 + [900], 32)[-1] == [20]                     # one long row does not pad the rest
    assert sorted(len(g) for g in length_groups([100] * 40, 16)) == [8, 16, 16]


def test_each_pass_of_rows_gets_its_own_copy_of_the_cached_state(monkeypatch):
    """DecisionModel._rows_hidden runs branch rows on a cached state, rows_per_pass rows at a time. Each pass gets the
    state widened to its rows (attention keys and values, the DeltaNet conv window and recurrent state), with the state's
    tokens marked real in the attention mask. The pass may write to its copy, and the caller's prefix stays exactly as it
    was, so the next request can reuse it."""
    attention, deltanet = DynamicLayer(), LinearAttentionLayer()
    attention.update(torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 2.0))
    deltanet.update_conv_state(torch.ones(1, 2, 2), conv_kernel_size=2)
    deltanet.update_recurrent_state(torch.full((1, 2, 2), 3.0))
    cache, passes = Cache(layers=[attention, deltanet]), []

    class LM(torch.nn.Module):
        def forward(self, input_ids, position_ids, attention_mask, past_key_values, use_cache):
            rows, (kv, linear) = len(input_ids), past_key_values.layers
            passes.append((rows, attention_mask.shape[1]))
            assert kv.keys.shape[0] == linear.conv_states[0].shape[0] == linear.recurrent_states[0].shape[0] == rows
            assert (kv.keys == 1).all() and (kv.values == 2).all() and (attention_mask[:, :2] == 1).all()
            assert (linear.conv_states[0] == 1).all() and (linear.recurrent_states[0] == 3).all()
            kv.update(torch.zeros(rows, 1, 1, 2), torch.zeros(rows, 1, 1, 2))   # what a forward pass writes
            linear.update_conv_state(torch.zeros(rows, 2, 1), conv_kernel_size=2)
            linear.update_recurrent_state(torch.zeros_like(linear.recurrent_states[0]))
            return SimpleNamespace(last_hidden_state=torch.ones(rows, input_ids.shape[1], 2))

    model = torch_backend.DecisionModel.__new__(torch_backend.DecisionModel)
    torch.nn.Module.__init__(model)
    model.lm, model.device, model.pad_id = LM(), "cpu", 0
    model.eval()
    monkeypatch.setattr(torch_backend, "rows_per_pass", lambda rows, prefix_len=0: 2)
    hidden = model._rows_hidden([([1, 2], [2, 3]), ([3], [2]), ([4], [2])], cache=cache, prefix_len=2)
    assert [tuple(h.shape) for h in hidden] == [(2, 2), (1, 2), (1, 2)] and passes == [(2, 4), (1, 3)]
    assert (attention.keys == 1).all() and (attention.values == 2).all()
    assert (deltanet.conv_states[0] == 1).all() and (deltanet.recurrent_states[0] == 3).all()
    assert deltanet.has_previous_state[0] and len(cache.layers) == 2
