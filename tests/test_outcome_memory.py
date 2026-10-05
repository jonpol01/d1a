"""The outcome memory (#149): d1a.serve unchanged without it, the blend, memory-fit's time-ordered gate on held-out groups,
the query sidecar, and the pointer-head queries d1a.serve captures (checked against the head on the committed tiny
Gemma 4)."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from d1a.feedback import FeedbackLog, OutcomeMemory, blend, decode_vector, encode_vector, held_out, memory_fit, memory_probs, prequential
from d1a.serve import Learning, QueryTap

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "tests/golden/tiny-gemma4"
KEYS = ["a", "b"]


def choice(p):
    return {"type": "choice", "choice": max(p, key=p.get), "confidence": 0.0, "probabilities": p}


class Req:
    """The two attributes Learning.decide reads from a SystemOneRequest."""
    def __init__(self):
        self.state, self.questions = "s", {}


def test_unchanged_without_a_memory(tmp_path):
    """No memory file: the served answers are the very object scored, and the model card gains no keys; a memory without
    a question's params leaves that question alone."""
    answers = {"route": choice({"a": 0.9, "b": 0.1})}
    for log in (None, tmp_path / "log.jsonl"):
        learning = Learning(log)
        served, _ = learning.decide(Req(), answers, "run", {"route": np.ones(4, np.float32)})
        assert served is answers and set(learning.card()) == {"log", "outcome_calibrator", "calibrated_questions"}
    mem = OutcomeMemory({"other": {"k": 3, "tau": 0.1, "lam": 0.5, "keys": KEYS, "center": encode_vector(np.zeros(4)), "bank": []}})
    assert mem.apply(answers, answers, {"route": np.ones(4, np.float32)}) is answers
    assert mem.probs("other", {"a": 0.5, "c": 0.5}, np.ones(4)) is None   # options that do not match the stored ones


def test_the_blend():
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    m = memory_probs(KEYS, np.array([1.0, 0.0]), bank, ["a", "b", "b"], k=2, tau=0.5)
    w = np.exp(np.array([1.0, 0.6]) / 0.5); w /= w.sum()                  # the two nearest: [1, 0] ("a") and [.6, .8] ("b")
    assert m == pytest.approx({"a": w[0], "b": w[1]})
    assert blend({"a": 0.9, "b": 0.1}, m, 0.4) == pytest.approx({"a": 0.6 * 0.9 + 0.4 * w[0], "b": 0.6 * 0.1 + 0.4 * w[1]})
    assert blend({"a": 0.9, "b": 0.1}, None, 0.4) == {"a": 0.9, "b": 0.1}
    assert memory_probs(KEYS, np.array([1.0, 0.0]), np.zeros((0, 2)), [], 3, 0.1) is None
    mem = OutcomeMemory({"route": {"k": 1, "tau": 0.1, "lam": 0.8, "keys": KEYS, "center": encode_vector(np.zeros(2)),
                                   "bank": [{"q": encode_vector([0.0, 1.0]), "label": "b", "outcome_ts": 1.0}]}})
    raw = {"route": choice({"a": 0.9, "b": 0.1})}
    served = mem.apply(raw, raw, {"route": np.array([0.0, 2.0], np.float32)})
    assert served["route"] == {"type": "choice", "choice": "b", "confidence": 0.64, "probabilities": {"a": 0.18, "b": 0.82}}   # the choice can change


def test_memory_uses_only_outcomes_known_before_a_decision():
    rows = [{"ts": t, "outcome_ts": t + 0.5, "q": np.array([1.0, 0.0]), "probs": {"a": 0.5, "b": 0.5}, "label": "b"} for t in (0.0, 1.0, 2.0)]
    out = prequential(rows, rows, np.zeros(2), k=3, tau=0.1, lam=1.0)
    assert out[0] == {"a": 0.5, "b": 0.5}                                 # nothing known yet: the model's answer
    assert out[1] == pytest.approx({"a": 0.0, "b": 1.0}) and out[2] == pytest.approx({"a": 0.0, "b": 1.0})


def planted_log(path, n=240, seed=0):
    """A choice question the model answers "a" with p 0.9 everywhere, while the truth follows the query's side: one group
    (a pull request) per decision pair, each outcome known before the next decision."""
    rng, log = random.Random(seed), FeedbackLog(path)
    for i in range(n):
        side = rng.random() < 0.5
        q = np.array([1.0 if side else -1.0, rng.gauss(0, 0.2), rng.gauss(0, 0.2)], np.float32)
        events = [{"kind": "decision", "id": f"d{i}", "ts": float(i), "run": "r", "state": f"s{i}", "questions": {"route": {"type": "choice"}},
                   "answers": {"route": choice({"a": 0.9, "b": 0.1})}, "meta": {}},
                  {"kind": "outcome", "id": f"d{i}", "ts": i + 0.5, "labels": {"route": "b" if side else "a"}, "meta": {"group": f"pr{i // 2}"}}]
        with open(path, "a", encoding="utf-8") as f: f.write("".join(json.dumps(e) + "\n" for e in events))
        log.queries(f"d{i}", {"route": q})
    return log


def test_a_planted_overconfident_question_is_promoted(tmp_path):
    log = planted_log(tmp_path / "log.jsonl")
    res, vectors = log.resolved(), log.query_vectors()
    assert not {held_out(d["group"]) for d in res if d["group"] == "pr7"} - {held_out("pr7")}   # a group is on one side only
    mem, report = memory_fit(res, vectors)
    r = report["route"]
    assert r["promote"] and r["lam"] > 0 and r["ci"][1] < 0, r
    assert r["accuracy"]["memory"] > r["accuracy"]["temperature"] == r["accuracy"]["raw"]   # a temperature never changes the choice
    assert r["log_loss"]["memory"] < r["log_loss"]["temperature"] and r["flips"]["wrong_to_right"] > r["flips"]["right_to_wrong"]
    assert set(mem.params) == {"route"} and len(mem.params["route"]["bank"]) == 240
    raw = {"route": choice({"a": 0.9, "b": 0.1})}
    served = mem.apply(raw, raw, {"route": np.array([1.0, 0.0, 0.0], np.float32)})
    assert served["route"]["choice"] == "b"


def test_a_question_the_model_gets_right_is_not_promoted(tmp_path):
    log = FeedbackLog(tmp_path / "log.jsonl")
    with open(log.path, "w", encoding="utf-8") as f:
        for i in range(120):
            label = "a" if i % 2 else "b"
            p = {"a": 0.95, "b": 0.05} if label == "a" else {"a": 0.05, "b": 0.95}
            f.write(json.dumps({"kind": "decision", "id": f"d{i}", "ts": float(i), "run": "r", "state": "s", "questions": {}, "answers": {"route": choice(p)}, "meta": {}}) + "\n")
            f.write(json.dumps({"kind": "outcome", "id": f"d{i}", "ts": i + 0.5, "labels": {"route": label}, "meta": {"group": f"pr{i}"}}) + "\n")
    for i in range(120): log.queries(f"d{i}", {"route": np.random.default_rng(i).normal(size=3).astype(np.float32)})
    mem, report = memory_fit(log.resolved(), log.query_vectors())
    assert not report["route"]["promote"] and mem.params == {}


def test_the_sidecar_leaves_the_log_alone(tmp_path):
    log = FeedbackLog(tmp_path / "decisions.jsonl")
    did = log.decision("s", {}, {"route": choice({"a": 0.9, "b": 0.1})}, run="r")
    before = log.path.read_text(encoding="utf-8")
    v = np.arange(256, dtype=np.float32) / 7
    log.queries(did, {"route": v})
    assert log.path.read_text(encoding="utf-8") == before and log.queries_path.name == "decisions.queries.jsonl"
    assert np.array_equal(log.query_vectors()[did]["route"], v) and np.array_equal(decode_vector(encode_vector(v)), v)


def test_captured_queries_are_the_heads(monkeypatch):
    """Every scoring path of the tiny Gemma 4 with the tap installed: the answers are bit for bit the same, and each
    question's matched query is q(h_decide) computed directly from the record's hidden states."""
    monkeypatch.chdir(ROOT)
    from d1a.api import SystemOneRequest, to_record
    from d1a.model import DecisionModel, SERVE_MAX_BRANCH, SERVE_MAX_STATE, load_tokenizer
    tok = load_tokenizer(FIXTURE + "/base")
    torch.manual_seed(0)
    model = DecisionModel(FIXTURE + "/base", tok, "cpu", lora=4).eval()
    reqs = [{"state": "the customer is charged twice", "questions": {"team": {"type": "choice", "criteria": {"billing": None, "shipping": None, "refund": None}},
                                                                     "angry": {"type": "noul"}}},
            {"state": "the box arrived crushed and wet", "questions": {"place": {"type": "choice", "criteria": {"door": None, "locker": None}}}}]
    encs = [model.encode(tok, to_record(SystemOneRequest.model_validate(r))[0], max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH) for r in reqs]
    with torch.no_grad():
        plain = model.probs_batch(encs, [None] * 2, [False] * 2)[0]
        tap = QueryTap(model.head)
        tapped = model.probs_batch(encs, [None] * 2, [False] * 2)[0]
        for e, p_plain, p_tapped in zip(encs, plain, tapped):
            assert all(torch.equal(a, b) for a, b in zip(p_plain, p_tapped))
            got = tap.match([x.tolist() for x in p_tapped])
            h = model.hidden(e)
            want = [model.head.q(h[d]).numpy() for d in e["decide_idx"]]
            assert all(g is not None and np.allclose(g, w, atol=1e-5) for g, w in zip(got, want))
        tap.calls.clear()
        single = model.probs(encs[0])
        assert all(g is not None for g in tap.match([x.tolist() for x in single]))
