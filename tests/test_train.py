"""d1a.training.train on the CPU: the loss, how micro-batches weigh records, that a run is reproducible from its seed (the
committed tiny Gemma 4 in tests/golden/tiny-gemma4), how an epoch is planned (--length_sort, the none-pair gate,
--pass_tokens_max, --row_budget), what a non-finite gradient does, the gradient-norm summary, and resume points (the
hybrid Qwen3.5 of tests/conftest.py)."""
import contextlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from d1a.backends.checkpoint import read_meta
from d1a.backends.torch import DecisionModel
from d1a.core.encoding import MAX_STATE, encode, load_tokenizer
from d1a.eval.suite import read_json
from d1a.training import train
from d1a.training.data import load_records, materialize

ROOT = Path(__file__).resolve().parents[1]


def test_the_loss_is_proper():
    """The expected loss under the true distribution is at its minimum there: its gradient is zero."""
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(p * train.question_loss(logits, {"label": y}, "cpu") for y, p in enumerate([0.2, 0.8]))
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6
    z = torch.tensor([2.0, 0.0, -2.0])
    soft = train.question_loss(z, {"target": [0.5, 0.5, 0.0], "label": 0}, "cpu")   # a soft target replaces the label
    assert soft.item() == pytest.approx(-(torch.log_softmax(z, -1)[:2] / 2).sum().item())


def test_uneven_micro_batches_weigh_every_record_alike():
    """A step's gradient is the mean over its records whatever the micro-batch sizes, including a short last step."""
    x = torch.arange(10, dtype=torch.float32)
    gradients = []
    for batch, accum in ((8, 1), (3, 3), (2, 4), (10, 1)):
        w = torch.tensor(1.0, requires_grad=True)
        for mb, start in enumerate(range(0, len(x), batch)):
            chunk = x[start:start + batch]
            ((w * chunk).sum() / train.accumulation_records(len(x), batch, accum, mb)).backward()
        gradients.append(w.grad.item())
    assert gradients[0] == gradients[2] and gradients[3] == pytest.approx(x.mean().item())
    assert [train.accumulation_records(10, 3, 3, mb) for mb in range(4)] == [9, 9, 9, 1]


def test_a_run_is_reproducible_from_its_seed(tmp_path, monkeypatch):
    """The same seed and data give the same adapter and head, bit for bit; another seed does not (CPU)."""
    rows = [{"state": "the customer is charged twice" + " again" * i, "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": None, "shipping": None, "refund": None}, "label": ["billing", "shipping", "refund"][i % 3]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": i % 2 == 0}}} for i in range(8)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.chdir(ROOT)

    def trained(name, seed):
        argv = ["d1a.training.train", "--base", "tests/golden/tiny-gemma4/base", "--data", str(tmp_path / "data.jsonl"), "--device", "cpu", "--lora", "4",
                "--batch", "2", "--accum", "2", "--p_none_pair", "0.5", "--seed", str(seed), "--out", str(tmp_path / name)]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
        from safetensors.torch import load_file
        adapter = load_file(str(tmp_path / name / "adapter_model.safetensors"))
        from d1a.backends.checkpoint import read_meta
        head = read_meta(tmp_path / name).head
        return adapter, head

    first, again, other = trained("a", 0), trained("b", 0), trained("c", 1)
    for x, y in ((first, again),):
        assert x[0].keys() == y[0].keys() and all(torch.equal(x[0][k], y[0][k]) for k in x[0])
        assert all(torch.equal(x[1][k], y[1][k]) for k in x[1])
    assert any(not torch.equal(first[0][k], other[0][k]) for k in first[0])


# --- planning an epoch ---------------------------------------------------------------------------------------------------

def knobs(**overrides):
    """The training arguments encode_batch, none_pairs and plan_shapes read: no augmentation, no pairs, rows through a
    shared prefix."""
    return SimpleNamespace(**{"seed": 0, "p_none": 0.0, "p_none_distract": 0.0, "p_distract": 0.0, "p_none_pair": 0.0, "none_pair_max_state": None,
                              "max_state": MAX_STATE, "row_budget": 0, "shared_prefix": 1, **overrides})


def lengths_of(plan):
    return [([len(r["state"]) for r in chunk], records, ends) for chunk, records, ends in plan]


def test_length_sort_deals_each_step_into_runs_of_similar_cost():
    """Without --length_sort a step is --accum runs of --batch consecutive records. With it, the same records stay in the
    same step but are cut, in cost order, into runs whose costliest pass is as small as possible, costliest run first."""
    reqs = [{"_meta": {"id": f"r{i}"}, "state": "x" * n, "questions": {"q": {"type": "noul", "instructions": "?"}}}
            for i, n in enumerate([1, 90, 5, 70, 3, 80, 2, 4, 6, 7])]
    sort = lambda on: train.microbatch_plan(reqs, SimpleNamespace(batch=2, accum=2, length_sort=on, shared_prefix=1))
    assert lengths_of(sort(0)) == [([1, 90], 4, False), ([5, 70], 4, True), ([3, 80], 4, False), ([2, 4], 4, True), ([6, 7], 2, True)]
    # the second step's three short states cost more than its long one: their three branches outweigh one long state
    assert lengths_of(sort(1)) == [([90, 70], 4, False), ([5, 1], 4, True), ([4, 3, 2], 4, False), ([80], 4, True), ([7, 6], 2, True)]


def test_the_none_pair_gate_picks_short_states_and_the_plan_pays_for_their_siblings(hybrid_base):
    """--none_pair_max_state: a record trains its two none-pair siblings when its state, counted as encode counts it, is
    within the gate and its own draw says so (the same draw on every call, a new one each epoch, whatever the other
    records); encode_batch gives exactly those records their siblings, and the plan costs them before encoding."""
    tok, reqs = load_tokenizer(str(hybrid_base / "base")), load_records(hybrid_base / "data.jsonl")
    tokens = train.state_token_counts(tok, reqs)
    assert [tokens[id(r)] for r in reqs] == [encode(tok, materialize(r))["seg"].count(0) for r in reqs] == list(range(6, 22))
    always = knobs(p_none_pair=1.0, none_pair_max_state=10)
    gated = train.none_pairs(always, reqs, 0, tokens)
    assert gated == {id(r) for r in reqs[:5]}                                         # states of 6 to 10 tokens
    yes_no_only = [{**r, "questions": {"angry": r["questions"]["angry"]}} for r in reqs]
    assert train.none_pairs(always, yes_no_only, 0, train.state_token_counts(tok, yes_no_only)) == set()   # nothing to pair
    half = knobs(p_none_pair=0.5, none_pair_max_state=64)
    epoch0, again, epoch1 = (train.none_pairs(half, reqs, epoch, tokens) for epoch in (0, 0, 1))
    assert epoch0 == again != epoch1 and 0 < len(epoch0) < len(reqs)
    assert train.none_pairs(half, reqs[::2], 0, tokens) == epoch0 & {id(r) for r in reqs[::2]}

    model = DecisionModel(str(hybrid_base / "base"), tok, "cpu")
    variants = Counter(v.request_id for v in train.encode_batch(model, tok, always, reqs, 0, gated))
    assert variants == {r["_meta"]["id"]: 3 if id(r) in gated else 1 for r in reqs}
    assert len(train.encode_batch(model, tok, knobs(), reqs, 0)) == len(reqs)          # no gate, --p_none_pair 0: no siblings
    costs = train.character_shapes(reqs, gated)
    assert [len(costs[id(r)]) for r in reqs] == [3] * 5 + [1] * 11 and costs[id(reqs[0])][1][0] == costs[id(reqs[0])][0][0]
    plan_knobs = SimpleNamespace(batch=2, accum=2, length_sort=1, shared_prefix=1)
    plain, paired = train.microbatch_plan(reqs, plan_knobs), train.microbatch_plan(reqs, plan_knobs, gated)
    assert train.microbatch_plan(reqs, plan_knobs, set()) == plain != paired
    assert sorted(id(r) for chunk, _, _ in paired for r in chunk) == sorted(map(id, reqs))


def test_plan_shapes_are_what_encode_batch_builds(hybrid_base):
    """--pass_tokens_max plans on plan_shapes before encoding: per record, the token shapes of every variant encode_batch
    then builds this epoch (augmented, siblings included), with the gate's pairs and with pairs drawn per record."""
    tok, reqs = load_tokenizer(str(hybrid_base / "base")), load_records(hybrid_base / "data.jsonl")
    model = DecisionModel(str(hybrid_base / "base"), tok, "cpu")
    tokens = train.state_token_counts(tok, reqs)
    a = knobs(p_none=0.3, p_none_distract=0.3, p_distract=0.3, p_none_pair=0.5, none_pair_max_state=12)
    for pairs in (train.none_pairs(a, reqs, 1, tokens), None):
        built = defaultdict(list)
        for v in train.encode_batch(model, tok, a, reqs, 1, pairs):
            built[v.request_id].append(train.shape(v.enc))
        planned = train.plan_shapes(model, tok, a, reqs, 1, pairs, tokens)
        assert {r["_meta"]["id"]: planned[id(r)] for r in reqs} == built
        assert {len(shapes) for shapes in planned.values()} == {1, 3}


def test_pass_tokens_max_adds_micro_batches_until_every_pass_fits():
    """With token shapes, a step whose costliest run is over --pass_tokens_max is cut into one more run until none is; the
    step keeps its own records and normaliser. A record over the ceiling on its own is refused by name. Without shapes
    (no ceiling plan) the plan is the uncapped one."""
    states = [400, 390, 12, 10, 30, 28, 26, 24, 395, 8, 6]   # steps of 4, 4 and 3 records; r4 and r5 carry two siblings
    reqs = [{"_meta": {"id": f"r{i}"}, "state": "x" * n, "questions": {"q": {"type": "noul"}}} for i, n in enumerate(states)]
    shapes = {id(r): [(n, [4])] + [(n, [5])] * 2 * (i in (4, 5)) for i, (r, n) in enumerate(zip(reqs, states))}
    a = SimpleNamespace(batch=2, accum=2, length_sort=1, shared_prefix=1, pass_tokens_max=450)
    cost = lambda chunk: train.pass_tokens([s for r in chunk for s in shapes[id(r)]], True)
    view = lambda plan: [([states[reqs.index(r)] for r in chunk], records, ends, cost(chunk)) for chunk, records, ends in plan]
    uncapped = train.microbatch_plan(reqs, SimpleNamespace(**{**vars(a), "pass_tokens_max": 0}), None, shapes)
    assert view(uncapped)[0] == ([400, 390], 4, False, 808)
    assert view(train.microbatch_plan(reqs, a, None, shapes)) == [
        ([400], 4, False, 404), ([390], 4, False, 394), ([12, 10], 4, True, 32),     # the first step needed a third run
        ([28, 26, 24], 4, False, 165), ([30], 4, True, 105), ([395], 3, False, 399), ([8, 6], 3, True, 24)]
    with pytest.raises(ValueError, match=r"2 record\(s\) .* \(the largest, r0, 404\)"):
        train.microbatch_plan(reqs, SimpleNamespace(**{**vars(a), "pass_tokens_max": 398}), None, shapes)
    assert train.microbatch_plan(reqs, a) == train.microbatch_plan(reqs, SimpleNamespace(**{**vars(a), "pass_tokens_max": 0}))


@pytest.mark.parametrize("arguments, accepted", [(("--length_sort", "1"), True), ((), False), (("--length_sort", "1", "--row_budget", "64"), False)])
def test_pass_tokens_max_needs_length_sort_and_no_row_budget(arguments, accepted, tmp_path, capsys):
    argv = ["--out", str(tmp_path / "run"), "--pass_tokens_max", "4096", *arguments]
    if accepted:
        assert train.parse_args(argv).pass_tokens_max == 4096
        return
    with pytest.raises(SystemExit):
        train.parse_args(argv)
    assert "--pass_tokens_max caps the passes --length_sort 1 plans" in capsys.readouterr().err


def test_pass_tokens_max_is_refused_on_an_attention_only_base(hybrid_base, train_hybrid, tmp_path):
    """pass_tokens measures rows and the shared prefix, which a hybrid backbone always runs; an attention-only one runs
    short records packed, a cost the ceiling does not see, so the run stops once the model is built. The hybrid trains."""
    from transformers import AutoConfig, Qwen3_5ForCausalLM
    config = AutoConfig.from_pretrained(hybrid_base / "base")
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).save_pretrained(tmp_path / "attention")
    load_tokenizer(str(hybrid_base / "base")).save_pretrained(tmp_path / "attention")
    capped = ("--lora", "4", "--max_steps", "1", "--length_sort", "1", "--pass_tokens_max", "160")
    with pytest.raises(SystemExit, match="--pass_tokens_max needs a hybrid backbone"):
        train_hybrid(tmp_path / "refused", "--base", str(tmp_path / "attention"), *capped)
    assert not (tmp_path / "refused" / "head.safetensors").exists()
    assert (train_hybrid(tmp_path / "hybrid", *capped) / "head.safetensors").exists()


@pytest.mark.parametrize("shared", [0, 1])
def test_row_budget_splits_the_passes_and_keeps_the_gradient(hybrid_base, shared):
    """--row_budget runs a micro-batch in passes of at most that many padded tokens, splitting a record by question when
    it must (each part carries its share of the record's mean): the summed gradient is the one-pass gradient."""
    tok = load_tokenizer(str(hybrid_base / "base"))
    model = DecisionModel(str(hybrid_base / "base"), tok, "cpu")
    model.train()
    reqs = load_records(hybrid_base / "data.jsonl")[:4]

    def gradient(budget):
        a = knobs(shared_prefix=shared, row_budget=budget)
        batch = train.encode_batch(model, tok, a, reqs, 0)
        passes = train.row_passes(batch, budget, shared)
        model.zero_grad()
        for part in passes:
            train.batch_loss(model, a, part, "cpu", contextlib.nullcontext())[0].backward()
        return [v.share for v in batch], len(passes), [p.grad.clone() for p in model.parameters() if p.grad is not None]

    whole_shares, whole_passes, whole = gradient(0)
    split_shares, split_passes, split = gradient(16)   # under one record's two rows: every question its own pass
    assert (whole_shares, whole_passes) == ([1.0] * 4, 1) and (split_shares, split_passes) == ([0.5] * 8, 8)
    scale = max(g.abs().max() for g in whole)
    assert len(whole) == len(split) and all(torch.allclose(x, y, atol=1e-5 * scale) for x, y in zip(whole, split))


# --- non-finite gradients, the gradient-norm summary ---------------------------------------------------------------------

class NanGradient(torch.autograd.Function):
    """The identity forward, a NaN backward: a finite loss whose gradient is poisoned."""
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("nan"))


def test_a_nan_gradient_skips_its_optimizer_step(train_hybrid, tmp_path, monkeypatch):
    """A finite loss passes batch_loss's check; when its gradient is NaN the step is dropped (AdamW never sees a non-finite
    gradient), counted in training_metrics.json, and the run goes on to its next step."""
    real_loss, real_step, losses, steps = train.batch_loss, torch.optim.AdamW.step, [], []

    def poisoned_first(*args, **kwargs):
        loss, terms = real_loss(*args, **kwargs)
        losses.append(loss.item())
        return (NanGradient.apply(loss) if len(losses) == 1 else loss), terms

    def checked_step(self, *args, **kwargs):
        steps.append(all(torch.isfinite(p.grad).all() for group in self.param_groups for p in group["params"] if p.grad is not None))
        return real_step(self, *args, **kwargs)
    monkeypatch.setattr(train, "batch_loss", poisoned_first)
    monkeypatch.setattr(torch.optim.AdamW, "step", checked_step)
    out = train_hybrid(tmp_path / "run", "--lora", "4", "--accum", "2", "--max_steps", "2")
    metrics = read_json(out / "training_metrics.json")
    assert steps == [True] and len(losses) == 4 and all(math.isfinite(x) for x in losses)
    assert metrics["nonfinite_skipped"] == {"steps": 1} and metrics["optimizer_steps"] == 2
    assert all(torch.isfinite(t).all() for t in load_file(out / "adapter_model.safetensors").values())


def test_grad_norm_summary_per_epoch():
    """Per epoch with steps: their count, the mean and max norm before clipping, and the steps clipped (norm over
    MAX_GRAD_NORM; a norm at it is not clipped)."""
    limit = train.MAX_GRAD_NORM
    assert train.grad_norm_summary([[limit / 2, 3 * limit], [], [limit, limit / 4]]) == [
        {"epoch": 0, "steps": 2, "mean": 1.75 * limit, "max": 3 * limit, "clipped_steps": 1},
        {"epoch": 2, "steps": 2, "mean": 0.625 * limit, "max": limit, "clipped_steps": 0}]


# --- resume points ---------------------------------------------------------------------------------------------------------

def run_module(arguments, out):
    """d1a.training.train in its own process (a resumed run starts from nothing but what is on disk). -> its stdout."""
    done = subprocess.run([sys.executable, "-m", "d1a.training.train", *arguments, "--out", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


RESUMED_RUNS = {
    "lora": (),
    "length_sort": ("--p_none_pair", "0.5", "--length_sort", "1", "--shared_prefix", "1"),
    "none_pair_gate": ("--p_none_pair", "0.5", "--length_sort", "1", "--shared_prefix", "1", "--none_pair_max_state", "10"),
    # 160 is just over the costliest record on its own (157 padded tokens with its siblings), so some steps get more runs
    "pass_tokens_max": ("--p_none_pair", "0.5", "--length_sort", "1", "--shared_prefix", "1", "--pass_tokens_max", "160"),
}


@pytest.mark.parametrize("name", RESUMED_RUNS)
def test_a_run_stopped_and_resumed_ends_with_the_same_bits(hybrid_base, tmp_path, name):
    """A LoRA run stopped after step 3 of two epochs and continued with --resume 1 in a new process ends with the uninterrupted
    run's adapter, head and gradient-norm record, bit for bit; the resume point is gone once the checkpoint is written."""
    arguments = ["--base", str(hybrid_base / "base"), "--data", str(hybrid_base / "data.jsonl"), "--device", "cpu", "--lora", "4",
                 "--batch", "2", "--accum", "2", "--lr", "1e-3", "--epochs", "2", *RESUMED_RUNS[name]]
    log = run_module(arguments, tmp_path / "whole")
    run_module([*arguments, "--save_every_steps", "3", "--stop_after", "3"], tmp_path / "split")
    assert (tmp_path / "split" / "resume" / "latest.json").exists() and not (tmp_path / "split" / "adapter_model.safetensors").exists()
    run_module([*arguments, "--resume", "1"], tmp_path / "split")
    assert not (tmp_path / "split" / "resume").exists()

    whole, split = (load_file(tmp_path / d / "adapter_model.safetensors") for d in ("whole", "split"))
    assert whole.keys() == split.keys() and all(torch.equal(whole[k], split[k]) for k in whole)
    heads = [read_meta(tmp_path / d).head for d in ("whole", "split")]
    assert all(torch.equal(heads[0][k], heads[1][k]) for k in heads[0])
    metrics = [read_json(tmp_path / d / "training_metrics.json") for d in ("whole", "split")]
    norms = [m["grad_norm"] for m in metrics]
    assert norms[0] == norms[1] and [e["epoch"] for e in norms[0]] == [0, 1] and sum(e["steps"] for e in norms[0]) == metrics[0]["optimizer_steps"]
    assert all(0 < e["mean"] <= e["max"] for e in norms[0])   # real norms from the run: a summary of zeros would pass the rest

    assert ("none pairs: " in log) == (name == "none_pair_gate")
    if name == "pass_tokens_max":   # epoch 0, where the split run stops, has more micro-batches than its 4 steps x --accum 2
        plans = re.findall(r"plan: (\d+) micro-batches for (\d+) steps \(--accum 2\); the plan's largest pass (\d+)", log)
        assert len(plans) == 2 and int(plans[0][0]) > 2 * int(plans[0][1]) and all(int(largest) <= 160 for _, _, largest in plans), log
