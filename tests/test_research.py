# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); dropped two checks bound to the removed experiments/ registrations; the tests of the removed Modal app and report scripts left; the contrastive generator's tests left with it (paired_flip and the held-out structure guard keep theirs); the d1a.metrics tests rewritten as tests/test_metrics.py; the d1a.benchmark tests rewritten as tests/test_benchmark.py; the d1a.suite tests rewritten as tests/test_suite.py; the d1a.data tests rewritten as tests/test_data.py.

import pytest
import torch

from d1a.suite import record_digest
from d1a.train import question_loss


def choice_request():
    return {"state": "The shoes are the wrong size.", "questions": {"reason": {
        "type": "choice", "instructions": "Why return the shoes?",
        "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"},
        "label": "size", "src": "fixture",
    }}}


def test_score_loss_is_proper_at_true_distribution():
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(p * question_loss(logits, {"label": y, "qtype": "score"}, "cpu") for y, p in enumerate([0.2, 0.8]))
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6


def frozen_request(i=0):
    r = choice_request()
    r["_meta"] = {"id": f"item-{i}", "group_id": f"item-{i}", "source": "fixture", "variant": "clean"}
    return r


def test_strict_encoding_rejects_truncation():
    from types import SimpleNamespace
    from d1a.model import encode

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(range(len(text))))

        def convert_tokens_to_ids(self, text):
            return 1000

    rec = {"state": "abcdefgh", "questions": [{"instr": "q", "options": ["a", "b"], "label": 0}]}
    assert encode(Tokenizer(), rec, max_state=4)["state_truncated"]
    with pytest.raises(ValueError, match="state exceeds"):
        encode(Tokenizer(), rec, max_state=4, strict=True)


def test_batched_mask_matches_single_and_pads_are_invisible():
    from d1a.model import branch_mask, branch_mask_batch
    a, b = [0, 0, 1, 1, 2], [0, 1, 1]
    m = branch_mask_batch([a, b], "cpu")
    assert m.shape == (2, 1, 5, 5)
    assert torch.equal(m[0:1], branch_mask(a, "cpu"))
    assert torch.equal(m[1:2, :, :3, :3], branch_mask(b, "cpu"))
    allowed = m[1, 0] == 0
    assert not allowed[:3, 3:].any()          # real tokens never attend to padding
    assert allowed[3, 3] and allowed[4, 4]    # padded rows keep the diagonal, so softmax is finite
    assert not allowed[3, 1:3].any()          # pads belong to no question segment (state stays visible; rows are discarded)


def test_uneven_microbatches_have_equal_record_weight():
    from d1a.train import accumulation_records
    x = torch.arange(10, dtype=torch.float32)
    gradients = []
    for batch, accum in ((8, 1), (3, 3), (2, 4)):
        w = torch.tensor(1.0, requires_grad=True)
        for mb, start in enumerate(range(0, len(x), batch)):
            chunk = x[start:start + batch]
            ((w * chunk).sum() / accumulation_records(len(x), batch, accum, mb)).backward()
        gradients.append(w.grad.item())
    assert gradients[0] == gradients[2]
    assert accumulation_records(10, 3, 3, 2) == 9
    assert accumulation_records(10, 3, 3, 3) == 1

def test_remote_predictor_maps_system_one_answers_and_retries(monkeypatch):
    import io, json
    from d1a.predictors import RemotePredictor
    rec = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}, "label": "a", "src": "t"},
                                       "y": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}
    calls = []
    class Resp:
        def __init__(self, body): self.body = body
        def read(self): return json.dumps(self.body).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def urlopen(req, timeout):
        calls.append(json.loads(req.data))
        if len(calls) == 1: raise OSError("503")
        return Resp({"model": "openjev-x", "answers": {"q": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}, "y": {"type": "noul", "noul": 0.2}}, "usage": {"input_tokens": 12}})
    p = RemotePredictor("http://example.test/", retries=2); monkeypatch.setattr("urllib.request.urlopen", urlopen); monkeypatch.setattr("time.sleep", lambda s: None)
    out = p(rec)
    assert out["probabilities"] == {"q": {"a": 0.7, "b": 0.3}, "y": {"true": 0.2, "false": 0.8}} and p.served_model == "openjev-x" and len(calls) == 2
    assert calls[0]["model"] == "d1a-latest" and "label" not in json.dumps(calls[0])        # labels never leave the machine


def test_rotation_averaging_cancels_a_position_bias():
    import math
    from d1a.api import question_keys
    from d1a.predictors import RotationAveraged
    content = {"a": 1.0, "b": 0.0, "c": -1.0}; position = [2.0, 0.0, 0.0]         # the first slot is favoured by +2 logits

    def biased(record):
        out = {"probabilities": {}, "logits": {}, "inference_temperature": 1.0, "latency_ms": 1.0}
        for qid, q in record["questions"].items():
            keys = question_keys(q["type"], q.get("criteria"))
            z = {k: (content.get(k, 0.0) + position[i] if q["type"] == "choice" else float(i)) for i, k in enumerate(keys)}
            s = sum(math.exp(v) for v in z.values())
            out["logits"][qid] = z; out["probabilities"][qid] = {k: math.exp(v) / s for k, v in z.items()}
        return out

    record = {"state": "s", "questions": {"c": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}, "n": {"type": "noul"}}}
    shifted = RotationAveraged.rotated(record, 1)
    assert list(shifted["questions"]["c"]["criteria"]) == ["b", "c", "a"] and shifted["questions"]["n"] == record["questions"]["n"]
    avg = RotationAveraged(biased, 3)
    one, other = avg(record), avg(shifted)
    assert one["rotations"] == 3 and one["latency_ms"] == 3.0
    for k in "abc":                                                               # order no longer matters after a full cycle
        assert one["probabilities"]["c"][k] == pytest.approx(other["probabilities"]["c"][k])
    z = one["logits"]["c"]; assert z["a"] - z["b"] == pytest.approx(1.0) and z["b"] - z["c"] == pytest.approx(1.0)   # the bias is a constant
    assert one["probabilities"]["n"] == pytest.approx(biased(record)["probabilities"]["n"])   # noul untouched
    with pytest.raises(ValueError):
        RotationAveraged(biased, 1)
