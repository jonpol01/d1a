"""d1a.eval.predictors: scoring through a remote System One endpoint (RemotePredictor), test-time order averaging of
choice options (RotationAveraged), and how a local checkpoint (LocalPredictor) scores a record too long for the exact
kernels. No network: the endpoint is a stand-in; the local model is a one-step LoRA of tests/conftest.py's hybrid base
or a stand-in for the MLX backend."""
import json
import math

import pytest
import torch

from d1a.backends.checkpoint import LoadOptions
from d1a.core.api import question_keys
from d1a.core.encoding import ROW_PASS_TOKENS
from d1a.eval import predictors
from d1a.eval.benchmark import evaluate_records
from d1a.eval.predictors import RemotePredictor, RotationAveraged

RECORD = {"state": "s", "questions": {
    "team": {"type": "choice", "instructions": "Who handles it?", "criteria": {"billing": "B", "shipping": "S"}, "label": "billing", "src": "t"},
    "urgent": {"type": "noul", "instructions": "Is it urgent?", "label": True, "src": "t"}}}


class Endpoint:
    """A System One server that fails its first `failures` calls, then answers; keeps every request body it received."""
    def __init__(self, failures):
        self.failures, self.bodies = failures, []

    def __call__(self, request, timeout):
        self.bodies.append(json.loads(request.data))
        if len(self.bodies) <= self.failures:
            raise OSError("503 Service Unavailable")
        body = {"model": "d1a-e4b-v0.5", "usage": {"input_tokens": 9},
                "answers": {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.6, "shipping": 0.4}},
                            "urgent": {"type": "noul", "noul": 0.25}}}
        return type("Response", (), {"read": lambda self: json.dumps(body).encode(), "__enter__": lambda self: self,
                                     "__exit__": lambda self, *exc: False})()


def remote(monkeypatch, failures, retries):
    endpoint = Endpoint(failures)
    monkeypatch.setattr("urllib.request.urlopen", endpoint)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return RemotePredictor("http://endpoint.test/", retries=retries), endpoint


def test_a_remote_endpoint_is_scored_like_a_local_model_and_never_sees_a_label(monkeypatch):
    predictor, endpoint = remote(monkeypatch, failures=1, retries=2)
    out = predictor(RECORD)
    # a yes/no answer becomes the two-option distribution the benchmark scores; the reported model id is kept
    assert out["probabilities"] == {"team": {"billing": 0.6, "shipping": 0.4}, "urgent": {"true": 0.25, "false": 0.75}}
    assert predictor.served_model == "d1a-e4b-v0.5" and len(endpoint.bodies) == 2   # one failure, retried once
    sent = json.dumps(endpoint.bodies)
    assert endpoint.bodies[0]["model"] == "d1a-latest" and '"label"' not in sent and '"src"' not in sent


def test_a_remote_endpoint_that_keeps_failing_raises_after_its_retries(monkeypatch):
    predictor, endpoint = remote(monkeypatch, failures=5, retries=3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        predictor(RECORD)
    assert len(endpoint.bodies) == 3


def position_biased(record):
    """A predictor whose choice logits are each option's merit plus +2 for whatever option comes first; yes/no answers
    have a fixed order-free logit."""
    merit = {"a": 1.0, "b": 0.0, "c": -1.0}
    out = {"probabilities": {}, "logits": {}, "inference_temperature": 1.0, "latency_ms": 1.0}
    for qid, q in record["questions"].items():
        keys = question_keys(q["type"], q.get("criteria"))
        z = {k: (merit[k] + (2.0 if i == 0 else 0.0)) if q["type"] == "choice" else (0.5 if k == "true" else -0.5) for i, k in enumerate(keys)}
        total = sum(math.exp(v) for v in z.values())
        out["logits"][qid], out["probabilities"][qid] = z, {k: math.exp(v) / total for k, v in z.items()}
    return out


def test_averaging_over_every_rotation_removes_a_position_bias():
    record = {"state": "s", "questions": {"pick": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}, "yes": {"type": "noul"}}}
    turned = RotationAveraged.rotated(record, 2)
    assert list(turned["questions"]["pick"]["criteria"]) == ["c", "a", "b"] and turned["questions"]["yes"] == record["questions"]["yes"]
    averaged = RotationAveraged(position_biased, 3)
    one, other = averaged(record), averaged(turned)
    assert one["rotations"] == 3 and one["latency_ms"] == pytest.approx(3.0)   # one pass per rotation
    assert one["probabilities"]["pick"] == pytest.approx(other["probabilities"]["pick"])   # the listed order no longer matters
    z = one["logits"]["pick"]   # only merit is left between options: the bias added the same amount to each
    assert (z["a"] - z["b"], z["b"] - z["c"]) == pytest.approx((1.0, 1.0))
    assert one["probabilities"]["yes"] == pytest.approx(position_biased(record)["probabilities"]["yes"])   # yes/no keeps its order
    with pytest.raises(ValueError):
        RotationAveraged(position_biased, 1)


# --- LocalPredictor: a record longer than ROW_PASS_TOKENS ----------------------------------------------------------------

def suite_record(record):
    """A labelled request shaped like a frozen suite's record, which d1a.eval.benchmark scores."""
    meta = {"id": "long-1", "source": "tiny", "group_id": "g", "variant": "clean"}
    return {**record, "_meta": meta, "questions": {qid: {**q, "src": "tiny"} for qid, q in record["questions"].items()}}


def test_a_long_record_on_a_hybrid_torch_model_runs_its_state_once(train_hybrid, hybrid_base, tmp_path, monkeypatch):
    """A record whose longest row is over ROW_PASS_TOKENS runs through the shared prefix (its state once) with the row
    form's logits. Only on CUDA does it leave the fp32-exact kernels, and then the prediction, its benchmark rows and the
    report's long_rows say so; a run without a long record carries none of it."""
    from d1a.training.data import load_records
    run = train_hybrid(tmp_path / "checkpoint", "--lora", "4", "--max_steps", "1")
    local = predictors.LocalPredictor(str(run), "cpu", LoadOptions(dtype=torch.float32, temperature=1.0))
    record = suite_record(load_records(hybrid_base / "data.jsonl")[7])
    shared, real = [], local.model.forward_batch
    monkeypatch.setattr(local.model, "forward_batch", lambda encs, shared_prefix=False: shared.append(shared_prefix) or real(encs, shared_prefix))

    report, rows = evaluate_records([record], local, tmp_path / "short")
    assert shared == [False] and "long_rows" not in report and not any("kernels" in row for row in rows)
    exact = local(record)

    monkeypatch.setattr(predictors, "ROW_PASS_TOKENS", 8)   # this record's rows are about 20 tokens
    shared.clear()
    long = local(record)
    assert shared == [True] and "kernels" not in long and long["input_tokens"] == exact["input_tokens"]
    for qid, logits in exact["logits"].items():
        assert long["logits"][qid] == pytest.approx(logits, abs=1e-5)

    monkeypatch.setattr(local, "device", "cuda")             # the CUDA policy, run on CPU tensors
    monkeypatch.setattr(predictors, "sync", lambda device: None)
    report, rows = evaluate_records([record], local, tmp_path / "long")
    assert [row["kernels"] for row in rows] == [predictors.LONG_ROW_KERNELS] * len(record["questions"])
    assert report["long_rows"] == {"count": 2, "records": 1, "kernels": [predictors.LONG_ROW_KERNELS], "threshold": ROW_PASS_TOKENS}
    assert [x for row in rows for x in row["logits"]] == pytest.approx([x for q in exact["logits"].values() for x in q.values()], abs=1e-5)


class MLXStandIn:
    """What LocalPredictor uses of d1a.backends.mlx.MLXDecisionModel: encode and forward (which already runs a record's state
    once), and no forward_batch. Records the length of every encoding it is asked to score."""
    backend, hybrid = "mlx", True
    head = type("Head", (), {"temperature": 1.0})()

    def __init__(self):
        self.scored = []

    def encode(self, tok, rec, **limits):
        state = [0] * 40 + [1]                               # a 41-token state, then one question: <decide> and one option
        return {"ids": [5] * len(state) + [6, 7], "seg": state + [1, 1], "pos": list(range(len(state) + 2)),
                "decide_idx": [len(state)], "opt_idx": [[len(state) + 1]]}

    def forward(self, enc):
        self.scored.append(len(enc["ids"]))
        return [torch.zeros(1)]


@pytest.mark.parametrize("row_pass_tokens", [ROW_PASS_TOKENS, 8])
def test_on_mlx_every_record_goes_through_its_forward_unlabelled(row_pass_tokens, monkeypatch):
    local = object.__new__(predictors.LocalPredictor)          # a loaded checkpoint's attributes, without loading one
    local.tok, local.model, local.temperature, local.device = None, MLXStandIn(), 1.0, "mps"
    local.context = {"max_state": 64, "max_branch": 64, "max_packed": 64}
    monkeypatch.setattr(predictors, "ROW_PASS_TOKENS", row_pass_tokens)
    monkeypatch.setattr(predictors, "sync", lambda device: None)
    record = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": None}, "label": "only", "src": "t"}}}
    out = local(record)
    assert local.model.scored == [43] and out["probabilities"] == {"q": {"only": 1.0}} and "kernels" not in out
