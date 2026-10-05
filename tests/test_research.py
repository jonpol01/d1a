# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); dropped two checks bound to the removed experiments/ registrations; the tests of the removed Modal app and report scripts left; the contrastive generator's tests left with it (paired_flip and the held-out structure guard keep theirs); the d1a.metrics tests rewritten as tests/test_metrics.py; the d1a.benchmark tests rewritten as tests/test_benchmark.py; the d1a.data tests rewritten as tests/test_data.py.
import pathlib

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


def test_locked_split_and_hash_verification(tmp_path):
    import json
    from d1a.suite import digest, load_split, write_json, write_jsonl
    for name in ("development", "test"):
        write_jsonl(tmp_path / f"{name}.jsonl", [frozen_request()])
    manifest = {"files": {f"{name}.jsonl": {"sha256": digest(tmp_path / f"{name}.jsonl"), "records": 1} for name in ("development", "test")}}
    write_json(tmp_path / "manifest.json", manifest)
    assert len(load_split(tmp_path, "development")) == 1
    with pytest.raises(ValueError, match="locked test"):
        load_split(tmp_path, "test")
    write_jsonl(tmp_path / "development.jsonl", [{}])
    with pytest.raises(ValueError, match="checksum"):
        load_split(tmp_path, "development")


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

def test_v3_training_refuses_heldout_structure():
    from d1a.composition import DEV_SHAPES, SHAPES, canonical, push_negation, structure_keys
    from d1a.suite import validate_training
    r = {"_meta": {"source": "compositional", "family": "held_and_or"}}
    with pytest.raises(ValueError, match="held-out"):
        validate_training([r], {"trainable_sources": ["compositional"]})
    # a random tree is refused when its structure, up to operand order, numbering and De Morgan, is a held-out shape's
    assert canonical(("or", ("not", 0), ("and", 1, 2))) == canonical(SHAPES["held_or_not"])
    assert canonical(("not", ("and", 0, 1))) != canonical(("or", ("not", 0), ("not", 1)))
    assert canonical(push_negation(("not", ("and", 0, 1)))) == canonical(("or", ("not", 0), ("not", 1)))
    negated = canonical(push_negation(SHAPES["final_negation"]))           # held out only through De Morgan
    assert negated != canonical(SHAPES["final_negation"]) and negated in structure_keys(SHAPES["final_negation"])
    for held in (canonical(SHAPES[DEV_SHAPES[0]]), negated):
        with pytest.raises(ValueError, match="held-out"):
            validate_training([{"_meta": {"source": "compositional", "family": "rand:7", "structure": held}}], {"trainable_sources": ["compositional"]})
    validate_training([{"_meta": {"source": "compositional", "family": "nested_and"}}], {"trainable_sources": ["compositional"]})


def test_missing_partition_is_fetched_and_verified(tmp_path, monkeypatch):
    import json
    from d1a import suite as S
    evals = tmp_path / "evals" / "x" / "decision-x"; evals.mkdir(parents=True)
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    import hashlib
    S.write_json(evals / "manifest.json", {"files": {"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served = tmp_path / "served.jsonl"; served.write_bytes(payload)
    calls = []
    def fake_download(repo, path, repo_type, revision):
        calls.append((repo, path, repo_type, revision)); return str(served)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert len(S.load_split(evals, "train")) == 1
    assert calls == [(S.SUITES_DATASET, "x/decision-x/train.jsonl", "dataset", S.SUITES_REVISION)]
    # a tampered mirror is rejected by the manifest hash
    (evals / "train.jsonl").unlink(); served.write_bytes(b'{"tampered": 1}\n')
    with pytest.raises(ValueError, match="checksum"):
        S.load_split(evals, "train")

def test_private_suite_fetches_from_its_own_mirror(tmp_path, monkeypatch):
    import hashlib
    from huggingface_hub.errors import RepositoryNotFoundError
    from d1a import suite as S
    evals = tmp_path / "evals" / "held" / "docs-x"; evals.mkdir(parents=True)
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    S.write_json(evals / "manifest.json", {"mirror": {"dataset": S.PRIVATE_DATASET, "revision": "abc123"},
                                           "files": {"test.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served = tmp_path / "served.jsonl"; served.write_bytes(payload)
    calls = []
    def fake_download(repo, path, repo_type, revision):
        calls.append((repo, path, repo_type, revision)); return str(served)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert len(S.load_split(evals, "test", allow_test=True)) == 1
    assert calls == [(S.PRIVATE_DATASET, "held/docs-x/test.jsonl", "dataset", "abc123")]
    # without access the Hub says "not found"; the loader says why, instead of a bare 404
    (evals / "test.jsonl").unlink()
    import httpx
    def denied(*a, **k): raise RepositoryNotFoundError("404 Client Error", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", denied)
    with pytest.raises(PermissionError, match="kev-private-evals"):
        S.load_split(evals, "test", allow_test=True)

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


def test_frozen_suites_load_under_any_locale(tmp_path):
    """Issue #12: frozen partitions contain non-ASCII text and are sha256-checked byte for byte, so they must be read as
    UTF-8 whatever the platform's preferred encoding is (Windows cp936 in the report; an ASCII C locale here), and their
    line endings must survive checkout (.gitattributes pins *.json / *.jsonl to LF)."""
    import os, subprocess, sys
    from d1a.suite import digest, write_json, write_jsonl
    suite = tmp_path / "evals" / "x" / "decision-x"; suite.mkdir(parents=True)
    record = {**frozen_request(), "state": "Zwölf Boxkämpfer — ‘quotes’ and é"}
    write_jsonl(suite / "development.jsonl", [record])
    write_json(suite / "manifest.json", {"files": {"development.jsonl": {"sha256": digest(suite / "development.jsonl"), "records": 1}}})
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "utf-8"}   # stdio only; open() still defaults to the locale
    code = f"import locale; from d1a.suite import load_split; r = load_split({str(suite)!r}, 'development'); print(locale.getpreferredencoding(False), r[0]['state'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__)), encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith(record["state"]) and "UTF-8" not in out.stdout.split()[0].upper(), out.stdout
    attributes = (pathlib.Path(__file__).resolve().parents[1] / ".gitattributes").read_text(encoding="utf-8")
    assert "*.jsonl text eol=lf" in attributes and "*.json text eol=lf" in attributes


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


def test_jsonl_round_trips_unicode_line_separators(tmp_path):
    from d1a.suite import read_jsonl, write_jsonl
    recs = [{"state": "line one\u2028line two"}, {"state": "next\x85record\u2029end"}, {"state": "plain"}]
    write_jsonl(tmp_path / "x.jsonl", recs)
    assert read_jsonl(tmp_path / "x.jsonl") == recs
