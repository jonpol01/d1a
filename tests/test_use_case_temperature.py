"""Per-use-case temperatures (#209): a checkpoint's {use case: temperature} map (d1a.training.calibrate --use-case), the
request's optional `use_case`, and the readout that serves each request at its own temperature, in both backends, in a
batch and through the prefix cache. The tiny Gemma 4 checkpoint (tests/golden/tiny-gemma4) stands in for real weights:
a copy at T 1.7 ("plain") and the same copy with a routing temperature of 0.85 ("mapped")."""
import json
import math
import platform
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from d1a.backends.checkpoint import Checkpoint, LoadOptions, Meta, mlx_available, read_meta, write_meta
from d1a.core.api import SystemOneRequest, to_answers, to_record
from d1a.core.encoding import SERVE_MAX_BRANCH, SERVE_MAX_STATE
from d1a.training import calibrate

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path("tests/golden/tiny-gemma4")   # its checkpoint names its base by this relative path
CHECKPOINT_T, ROUTING_T = 1.7, 0.85


@pytest.fixture(scope="module")
def golden():
    return json.loads((ROOT / FIXTURE / "golden.json").read_text(encoding="utf-8"))["records"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """{"plain": the fixture at CHECKPOINT_T, "mapped": the same with {"routing": ROUTING_T}}, both written by calibrate."""
    out = {}
    for name in ("plain", "mapped"):
        run = tmp_path_factory.mktemp(name) / "checkpoint"
        shutil.copytree(ROOT / FIXTURE / "checkpoint", run)
        calibrate.main(["--run", str(run), "--temperature", str(CHECKPOINT_T), "--reason", "test"])
        if name == "mapped":
            calibrate.main(["--run", str(run), "--use-case", "routing", "--temperature", str(ROUTING_T), "--reason", "test"])
        out[name] = run
    return out


def load(run, temperature=None):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        ck = Checkpoint(str(run))
        tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32, backend="torch", temperature=temperature))
    return ck, tok, model.eval()


@pytest.fixture(scope="module")
def models(runs):
    return {"plain": load(runs["plain"]), "mapped": load(runs["mapped"]), "raw": load(runs["plain"], temperature=1.0)}


def encoded(tok, model, request, use_case=None):
    rec, meta = to_record(SystemOneRequest.model_validate(request))
    enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
    return ({**enc, "use_case": use_case} if use_case else enc), rec, meta


def every_path(model, enc):
    """A request's probabilities by every readout the server and the benchmark use: the plain pass, a cache miss that
    keeps the prefix, a cache hit, and the causal-row form (forced)."""
    out = {"probs": model.probs(enc)}
    out["miss"], prefix = model.probs_and_prefix(enc)
    out["hit"] = model.probs_with_prefix(enc, prefix)
    model.rows_form = lambda encs: True
    try:
        out["rows"] = model.probs(enc)
        out["rows_logits"] = [F.softmax(z, -1) for z in model.forward(enc)]
    finally:
        del model.rows_form
    return out


def at(raw, enc, T):
    """softmax(z / T) of the raw (T = 1) logits, per question: what serving at T must give, exactly."""
    return [F.softmax(z / T, -1) for z in raw.forward(enc)]


def read_at(model, T):
    """every_path of a model whose own temperature is T: how main serves a checkpoint calibrated at T."""
    class At:
        def __init__(self): self.saved = model.head.temperature
        def __enter__(self): model.head.temperature = T
        def __exit__(self, *a): model.head.temperature = self.saved
    return At()


def same(got, want):
    return len(got) == len(want) and all(torch.equal(g, w) for g, w in zip(got, want))


def server(ck, tok, model):
    from d1a.serving.serve import Server
    s = Server(ck, tok, model, "cpu")
    s.prefix_cache.min_tokens = 0   # cache every state, so the batch and cache tests reach the hit path
    return s


def test_a_routing_request_is_read_at_the_routing_temperature(models, golden):
    """Exact probabilities: on every readout path, those of the same weights calibrated at 0.85, and on the plain pass
    softmax(raw logits / 0.85); then those answers through the server."""
    ck, tok, model = models["mapped"]
    raw = models["raw"][2]
    assert model.head.temperature == CHECKPOINT_T and model.head.use_case_temperatures == {"routing": ROUTING_T}
    for record in golden[:12]:
        enc, _, _ = encoded(tok, model, record["request"], "routing")
        got = every_path(model, enc)
        with read_at(raw, ROUTING_T): want = every_path(raw, encoded(tok, raw, record["request"])[0])
        assert same(got["probs"], at(raw, enc, ROUTING_T)), record["id"]
        for path in got:
            assert same(got[path], want[path]), (record["id"], path)
    s = server(ck, tok, model)
    try:
        for record in golden[:12]:
            enc, _, meta = encoded(tok, model, record["request"])
            body = s.answer(SystemOneRequest.model_validate({**record["request"], "use_case": "routing"}))
            assert body["answers"] == to_answers([p.tolist() for p in at(raw, enc, ROUTING_T)], meta), record["id"]
    finally:
        s.close()


@pytest.mark.parametrize("use_case", [None, "triage"])
def test_without_a_use_case_or_with_an_unknown_one_a_request_is_read_at_the_checkpoint_temperature_bit_for_bit(models, golden, use_case):
    """Absent: bit-identical to the same checkpoint without the map, and to softmax(z / T) as main computes it. An unknown
    use case (one a client names before the checkpoint carries it) is not an error: it gets the checkpoint's temperature."""
    raw = models["raw"][2]
    for record in golden:
        got = every_path(models["mapped"][2], encoded(*models["mapped"][1:], record["request"], use_case)[0])
        before = every_path(models["plain"][2], encoded(*models["plain"][1:], record["request"])[0])
        want = at(raw, encoded(*models["plain"][1:], record["request"])[0], CHECKPOINT_T)
        assert same(got["probs"], want), record["id"]
        for path in got:
            assert same(got[path], before[path]), (record["id"], path)


def test_a_batch_mixing_use_cases_reads_each_request_at_its_own_temperature(models, golden):
    """One Server batch (and the prefix cache across batches) with routing, plain and unknown requests on the same and on
    different states: each gets exactly what it gets alone in the same cache state (a miss, then a hit), and that is its
    use case's temperature."""
    ck, tok, model = models["mapped"]
    raw = models["raw"][2]
    requests = [(golden[i]["request"], u) for i in (0, 1, 2) for u in ("routing", None, "triage")]
    encs = [encoded(tok, model, r, u)[0] for r, u in requests]
    alone = []
    for e in encs:
        s = server(ck, tok, model)
        try: alone.append([s._run([e])[0][0], s._run([e])[0][0]])   # a miss that keeps the state, then a hit on it
        finally: s.close()
    s = server(ck, tok, model)
    try:
        miss = s._run(encs)
        hit = s._run(encs[::-1])[::-1]   # the second batch, in the other order, hits the states the first one cached
        assert s.prefix_cache.hits == len(encs) and all(not m[1]["prefix_cache_hit"] and h[1]["prefix_cache_hit"] for m, h in zip(miss, hit))
    finally:
        s.close()
    for e, (_, u), m, h, a in zip(encs, requests, miss, hit, alone):
        assert [m[0], h[0]] == a, u
        want = at(raw, e, ROUTING_T if u == "routing" else CHECKPOINT_T)
        assert all(torch.allclose(torch.tensor(q), w, atol=1e-5, rtol=0) for q, w in zip(h[0], want)), u


def test_the_graphed_batch_readout_reads_each_question_at_its_requests_temperature(models):
    """DecisionModel._readout_many (the CUDA-graph batch path) with mixed temperatures: each question equals the same batch
    read at its temperature alone, and the questions at the checkpoint's are bit-identical to a batch with no use cases."""
    model = models["mapped"][2]
    ks = [3, 2, 4, 2]
    X = torch.randn(sum(ks) + len(ks), model.head.q.in_features, generator=torch.Generator().manual_seed(0))
    temps = [ROUTING_T, CHECKPOINT_T, ROUTING_T, CHECKPOINT_T]
    mixed = model._readout_many(X, ks, temps)
    plain, routing = model._readout_many(X, ks), model._readout_many(X, ks, [ROUTING_T] * len(ks))
    assert not same(plain, routing)
    for q, t in enumerate(temps):
        assert torch.equal(mixed[q], (routing if t == ROUTING_T else plain)[q]), q
    assert same(model._readout_many(X, ks, [CHECKPOINT_T] * len(ks)), plain)


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64" or not mlx_available(), reason="MLX runs on Apple Silicon only")
def test_the_mlx_backend_reads_each_request_at_its_use_cases_temperature():
    """MLXDecisionModel on a tiny random Gemma 4 (tests/test_mlx_export.py's): routing at its temperature, absent and
    unknown at the checkpoint's, on the prefix and the row forms."""
    import mlx.core as mx
    from d1a.backends.mlx import MLXDecisionModel
    from test_mlx_export import fake_encoding, tiny_gemma4
    mx.random.seed(0)
    m = MLXDecisionModel.from_lm(tiny_gemma4(), pad_id=0, head_dim=16)
    enc = fake_encoding(np.random.default_rng(1), 20, [(5, 2), (14, 4), (9, 3)])
    raw = m.forward(enc)                                   # temperature 1.0: the raw logits
    m.head.temperature, m.head.use_case_temperatures = CHECKPOINT_T, {"routing": ROUTING_T}
    for use_case, T in ((None, CHECKPOINT_T), ("triage", CHECKPOINT_T), ("routing", ROUTING_T)):
        e = {**enc, "use_case": use_case} if use_case else enc
        want = [F.softmax(z / T, -1) for z in raw]
        for got in (m.probs(e), m.probs_with_prefix(e, m.prefix(e)), [F.softmax(z, -1) for z in m.forward_rows(e)]):
            assert all(torch.allclose(g, w, atol=1e-5, rtol=0) for g, w in zip(got, want)), use_case
        assert same(m.probs(e), want), use_case            # the prefix form computes exactly these logits


def test_the_mlx_export_carries_the_use_case_temperatures_and_the_fits(tmp_path):
    """export_config copies the map, temperature_fit (#196) and the use-case fits, and an export reads them back as a run
    does; an export written before them reads as no map."""
    from test_mlx_export import write_export
    from d1a.backends.checkpoint import export_config
    fits = {"temperature_fit": {"method": "manual", "reason": "pool"}, "use_case_temperatures": {"routing": ROUTING_T},
            "use_case_temperature_fits": {"routing": {"method": "manual", "reason": "factory"}}}
    source = SimpleNamespace(requested="r", path=str(tmp_path), weights_sha256=lambda: "a" * 64,
                             meta=Meta(base="b", head_dim=4, temperature=1.48, extra={**fits, "args": {}}))
    cfg = export_config(source, [2], [5, 6, 7, 8, 9], 0, 8, None, "bfloat16")
    assert {k: cfg[k] for k in fits} == fits
    _, cfg = write_export(tmp_path, **fits)
    ck = Checkpoint(tmp_path)
    assert ck.meta.use_case_temperatures == {"routing": ROUTING_T} and ck.meta.temperature == 1.48
    assert {k: ck.meta.extra[k] for k in fits} == fits
    old = tmp_path / "old"; old.mkdir()
    write_export(old, temperature_fit=None, use_case_temperatures={}, use_case_temperature_fits={})
    assert Checkpoint(old).meta.use_case_temperatures == {} and "temperature_fit" not in Checkpoint(old).meta.extra


def rows(true_T, n=300, seed=0):
    """Scored choice rows whose labels follow softmax(logits / true_T), as d1a.eval.benchmark writes them."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = float(rng.uniform(0.5, 6.0))
        out.append({"id": f"s/{i}", "group": f"s/{i}", "source": "s", "task": "custom", "type": "choice", "keys": ["a", "b"], "variant": "clean",
                    "question": "q", "label": int(rng.random() >= 1 / (1 + math.exp(-m / true_T))), "logits": [m, 0.0],
                    "p": [1 / (1 + math.exp(-m)), 1 - 1 / (1 + math.exp(-m))], "inference_temperature": 1.0})
    return out


def test_calibrate_use_case_writes_only_that_use_cases_temperature(tmp_path):
    """--use-case fits (or writes) into use_case_temperatures and use_case_temperature_fits only: the checkpoint's
    temperature and temperature_fit, and other use cases, stay as they were."""
    run = tmp_path / "ckpt"; run.mkdir()
    main_fit = {"method": "manual", "reason": "pool"}
    write_meta(run, Meta(base="b", temperature=CHECKPOINT_T, extra={"temperature_fit": main_fit}))
    (tmp_path / "cal").mkdir()
    (tmp_path / "cal" / "rows.json").write_text(json.dumps(rows(0.5)), encoding="utf-8")
    T = calibrate.main(["--run", str(run), "--use-case", "routing", "--rows", str(tmp_path / "cal" / "rows.json"), "--allow-in-distribution"])
    meta = read_meta(run)
    assert 0.3 < T < 0.8 and meta.use_case_temperatures == {"routing": T}
    assert (meta.temperature, meta.extra["temperature_fit"]) == (CHECKPOINT_T, main_fit)
    fit = meta.extra["use_case_temperature_fits"]["routing"]
    assert fit["n"] == 300 and fit["in_distribution"]["allowed"] and "cross_validation" in fit
    calibrate.main(["--run", str(run), "--use-case", "triage", "--temperature", "2.5", "--reason", "copied"])
    meta = read_meta(run)
    assert meta.use_case_temperatures == {"routing": T, "triage": 2.5} and meta.temperature == CHECKPOINT_T
    assert meta.extra["use_case_temperature_fits"]["triage"] == {"method": "manual", "reason": "copied"}
    assert meta.extra["use_case_temperature_fits"]["routing"] == fit
    for bad in (["--use-case", "", "--temperature", "1", "--reason", "x"], ["--use-case", "routing", "--temperature", "0", "--reason", "x"],
                ["--use-case", "routing", "--temperature", "nan", "--reason", "x"]):
        with pytest.raises(SystemExit):
            calibrate.main(["--run", str(run), *bad])
    assert read_meta(run).use_case_temperatures == {"routing": T, "triage": 2.5}


def test_a_checkpoint_with_a_bad_use_case_temperature_is_refused_at_load(tmp_path):
    for bad in ({"routing": 0}, {"routing": float("inf")}, {"": 1.0}, {"routing": "0.85"}, ["routing"]):
        write_meta(tmp_path, Meta(base="b", extra={"use_case_temperatures": bad}))
        with pytest.raises(ValueError, match="use_case_temperatures"):
            Checkpoint(tmp_path)


def test_models_lists_the_map_and_the_decision_log_records_the_use_case_and_temperature(models, golden, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from d1a.serving import serve
    from d1a.serving.media import OnDemand
    s = server(*models["mapped"])
    try:
        monkeypatch.setattr(serve.app.state, "models", OnDemand(lambda: s, idle_s=0, device="cpu"), raising=False)
        monkeypatch.setattr(serve.app.state, "card", serve.card(s), raising=False)
        monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl"))
        with TestClient(serve.app) as client:
            cards = client.get("/v1/models").json()["models"]
            assert all(c["use_case_temperatures"] == {"routing": ROUTING_T} and c["temperature"] == CHECKPOINT_T for c in cards)
            for use_case in ("routing", None, "triage"):
                body = {**golden[0]["request"], **({"use_case": use_case} if use_case else {})}
                assert client.post("/v1/systemone", json=body).status_code == 200
        logged = [e for e in serve.LEARNING.log.events() if e["kind"] == "decision"]
        assert [(e.get("use_case"), e["temperature"]) for e in logged] == [("routing", ROUTING_T), (None, CHECKPOINT_T), ("triage", CHECKPOINT_T)]
        assert "use_case" not in logged[1]
    finally:
        s.close()
    plain = server(*models["plain"])
    try: assert serve.card(plain)["use_case_temperatures"] == {}
    finally: plain.close()


def test_presets_lib_mcp_and_the_python_client_send_the_use_case(models, golden, monkeypatch):
    """The agent presets name the routing use case; D1A.decide, the MCP tools and d1a_client pass it on, and send nothing
    new without it."""
    monkeypatch.syspath_prepend(str(ROOT / "clients" / "python"))
    import d1a_client
    from d1a.agents import mcp_server
    from d1a.agents.presets import PRESETS, USE_CASES
    from d1a.serving.lib import D1A
    assert USE_CASES == {kind: "routing" for kind in PRESETS}
    ck, tok, model = models["mapped"]
    m = D1A(ck, tok, model, "cpu")
    request = golden[0]["request"]
    enc, _, meta = encoded(tok, model, request)
    for use_case, T in (("routing", ROUTING_T), (None, CHECKPOINT_T)):
        assert m.decide(request["state"], request["questions"], use_case=use_case) == to_answers([p.tolist() for p in at(models["raw"][2], enc, T)], meta)
    sent = []

    class Response:
        def __init__(self, req): sent.append(json.loads(req.data))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"answers": {"route": {"probabilities": {"small": 0.8, "medium": 0.1, "large": 0.1}}}}'
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: Response(req))
    monkeypatch.setattr(mcp_server, "RUN", None)
    mcp_server.d1a_route("hi")
    mcp_server.d1a_decide("hi", {"q": {"type": "noul", "instructions": "?"}})
    client = d1a_client.Client("http://x")
    client.decide("s", {"q": {"type": "noul"}}, use_case="routing")
    client.decide("s", {"q": {"type": "noul"}})
    assert [b.get("use_case") for b in sent] == ["routing", None, "routing", None]
    assert "use_case" not in sent[1] and "use_case" not in sent[3]
