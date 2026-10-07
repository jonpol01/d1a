"""d1a.eval.predictors: scoring through a remote System One endpoint (RemotePredictor) and test-time order averaging of
choice options (RotationAveraged). No model and no network: the endpoint and the predictor are stand-ins."""
import json
import math

import pytest

from d1a.core.api import question_keys
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
