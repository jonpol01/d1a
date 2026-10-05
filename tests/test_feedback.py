import numpy as np

from d1a.data import materialize
from d1a.feedback import FeedbackLog, OutcomeCalibrator, gate, log_loss, records

Q = {"resolved": {"type": "noul", "instr": "Does the patch fix the issue?", "criteria": {"true": "fixes it", "false": "does not"}}}


def test_log_joins_outcomes_exports_trainable_records_and_recalibrates_a_shifted_base_rate(tmp_path):
    """Decisions and later outcomes join by id (pending ones stay out); records() are labelled requests d1a.data
    materialises; the calibrator turns an overconfident yes/no (P ~0.8 when 20% are true) into calibrated probabilities."""
    log = FeedbackLog(tmp_path / "f.jsonl"); rng = np.random.default_rng(0)
    for i in range(400):
        y = rng.random() < 0.2
        p = 1 / (1 + np.exp(-(1.4 + (0.9 if y else 0.0) + rng.normal(0, 0.6))))   # ranks y, but far too high
        did = log.decision(f"state {i}", Q, {"resolved": {"type": "noul", "noul": float(p)}}, run="JohnP1/d1a-e4b@v0.4", meta={"repo": f"r{i % 7}"})
        if i < 380: log.outcome(did, {"resolved": bool(y)})
    res = log.resolved()
    assert len(res) == 380 and len(log.pending()) == 20
    recs = records(res)
    assert recs[0]["questions"]["resolved"]["label"] in (True, False) and recs[0]["meta"]["run"] == "JohnP1/d1a-e4b@v0.4"
    assert materialize(recs[0])["questions"][0]["label"] in (0, 1)
    p = np.array([d["answers"]["resolved"]["noul"] for d in res]); y = np.array([d["labels"]["resolved"] for d in res])
    cal = OutcomeCalibrator().fit(res)
    q = np.array([cal.p("resolved", x) for x in p])
    assert p.mean() > 0.7 and abs(q.mean() - y.mean()) < 0.03
    assert log_loss(y, q).mean() < log_loss(y, p).mean()
    cal.save(tmp_path / "c.json")
    assert OutcomeCalibrator.load(tmp_path / "c.json").apply({"resolved": {"type": "noul", "noul": 0.9}})["resolved"]["noul"] == round(cal.p("resolved", 0.9), 4)


def test_gate_promotes_only_a_clearly_better_candidate_without_suite_regressions():
    rng = np.random.default_rng(1); y = rng.random(600) < 0.3; groups = np.arange(600) % 30
    good = np.where(y, 0.7, 0.15); poor = np.full(600, 0.3)
    assert gate(y, good, poor, groups)["promote"]
    assert not gate(y, poor, good, groups)["promote"]
    assert not gate(y, poor, poor, groups)["promote"]          # no improvement is not a promotion
    blocked = gate(y, good, poor, groups, suite_deltas={"decision-v7": -0.03, "hard-v1": 0.0})
    assert not blocked["promote"] and "decision-v7" in blocked["regressions"]


def test_serve_logs_decisions_accepts_outcomes_and_applies_a_reloaded_calibrator(tmp_path, monkeypatch):
    """d1a.serve's self-learning hooks: with a log, every answer carries a decision_id and POST /v1/feedback records its
    outcome; with a calibrator file, yes/no answers are recalibrated and a rewritten file takes effect on the next answer;
    with neither, /v1/feedback is a 404."""
    import os
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a import serve
    from d1a.api import SystemOneRequest, to_record
    req = SystemOneRequest(state="issue + patch", questions=Q)
    _, meta = to_record(req)
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested="JohnP1/d1a-e4b-mlx-q8@v0.4"), tok=None)
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    cal_path = tmp_path / "cal.json"
    OutcomeCalibrator({"resolved": [1.0, -2.0, 50]}).save(cal_path)
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl", cal_path))
    body = serve.Server._body(fake, req, meta, [[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})
    expected = round(OutcomeCalibrator({"resolved": [1.0, -2.0, 50]}).p("resolved", 0.8), 4)
    assert body["answers"]["resolved"]["noul"] == expected < 0.8 and body["decision_id"]
    with TestClient(serve.app) as client:
        assert client.post("/v1/feedback", json={"decision_id": body["decision_id"], "labels": {"resolved": False}}).json() == {"ok": True}
    (d,) = serve.LEARNING.log.resolved()
    assert d["id"] == body["decision_id"] and d["labels"] == {"resolved": False} and d["run"] == "JohnP1/d1a-e4b-mlx-q8@v0.4"
    assert d["answers"]["resolved"]["noul"] == expected and d["questions"]["resolved"]["type"] == "noul"
    OutcomeCalibrator({"resolved": [1.0, 0.0, 50]}).save(cal_path)                 # a new calibrator, picked up without a restart
    os.utime(cal_path, (os.path.getmtime(cal_path) + 5,) * 2)
    again = serve.Server._body(fake, req, meta, [[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})
    assert again["answers"]["resolved"]["noul"] == 0.8
    monkeypatch.setattr(serve, "LEARNING", serve.Learning())
    plain = serve.Server._body(fake, req, meta, [[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})
    assert "decision_id" not in plain and plain["answers"]["resolved"]["noul"] == 0.8
    with TestClient(serve.app) as client:
        assert client.post("/v1/feedback", json={"decision_id": "x", "labels": {"resolved": True}}).status_code == 404
