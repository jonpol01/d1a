import json
import numpy as np

from d1a.training.data import materialize
from d1a.learning.feedback import FeedbackLog, OutcomeCalibrator, choice_log_loss, choice_pairs, gate, gate_choice, held_out, log_loss, main, pairs, promote, records, split

Q = {"resolved": {"type": "noul", "instr": "Does the patch fix the issue?", "criteria": {"true": "fixes it", "false": "does not"}}}


def test_log_joins_outcomes_exports_trainable_records_and_recalibrates_a_shifted_base_rate(tmp_path):
    """Decisions and later outcomes join by id (pending ones stay out); records() are labelled requests d1a.training.data
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


CQ = {"blast": {"type": "choice", "instructions": "How far a mistake spreads.", "criteria": {"contained": "one module", "moderate": "one subsystem", "broad": "shared code", "massive": "everything"}}}


def test_choice_temperature_fixes_overconfidence_keeps_the_choice_and_passes_the_gate(tmp_path):
    """A choice question twice as sure as it should be (P = softmax(2 x true logits)): the fitted inverse temperature comes
    out near 0.5, recalibrated answers keep their chosen option, and the gate promotes them over the raw answers on
    held-out outcomes, and not the other way round."""
    log = FeedbackLog(tmp_path / "f.jsonl"); rng = np.random.default_rng(2); opts = list(CQ["blast"]["criteria"])
    for i in range(1200):
        z = rng.normal(0, 1.5, len(opts)); truth = np.exp(z) / np.exp(z).sum(); sure = np.exp(2 * z) / np.exp(2 * z).sum()
        probs = dict(zip(opts, map(float, sure)))
        did = log.decision(f"pr {i}", CQ, {"blast": {"choice": max(probs, key=probs.get), "probabilities": probs}}, run="JohnP1/d1a-e4b-mlx-q8@v0.4")
        log.outcome(did, {"blast": str(rng.choice(opts, p=truth))}, {"src": "reviewer"})
    res = log.resolved(); fit, held = res[:800], res[800:]
    cal = OutcomeCalibrator().fit(fit)
    assert abs(cal.params["blast"]["s"] - 0.5) < 0.1 and cal.params["blast"]["n"] == 800
    raw, labels = choice_pairs(held, "blast")
    new = [cal.probs("blast", pr) for pr in raw]
    assert all(max(a, key=a.get) == max(b, key=b.get) for a, b in zip(raw, new)) and all(abs(sum(b.values()) - 1) < 1e-9 for b in new)
    assert choice_log_loss(new, labels).mean() < choice_log_loss(raw, labels).mean()
    groups = np.arange(len(labels)) % 20
    assert gate_choice(labels, new, raw, groups)["promote"] and not gate_choice(labels, raw, new, groups)["promote"]
    served = cal.apply(held[0]["answers"])["blast"]
    assert served["choice"] == held[0]["answers"]["blast"]["choice"] and served["probabilities"] != held[0]["answers"]["blast"]["probabilities"]
    cal.save(tmp_path / "c.json")
    assert OutcomeCalibrator.load(tmp_path / "c.json").apply(held[0]["answers"]) == cal.apply(held[0]["answers"])


def test_promote_updates_the_served_calibrator_only_where_held_out_groups_confirm_it(tmp_path, capsys):
    """`d1a.learning.feedback promote` on a live log: whole groups (two decisions per pull request) go to one side; a question twice
    as sure as it should be gets a calibrator that passes the gate on the held-out groups and is written for the server,
    while a question that is already calibrated keeps no parameters, and a log with nothing to promote leaves the file alone."""
    log = FeedbackLog(tmp_path / "f.jsonl"); rng = np.random.default_rng(3); opts = list(CQ["blast"]["criteria"])
    Q2 = {**CQ, "type": {"type": "choice", "instructions": "Change type.", "criteria": {"bug": "fix", "docs": "docs", "feature": "new"}}}
    for i in range(1000):
        z = rng.normal(0, 1.5, 4); truth = np.exp(z) / np.exp(z).sum(); sure = np.exp(2 * z) / np.exp(2 * z).sum()
        zt = rng.normal(0, 1.5, 3); pt = np.exp(zt) / np.exp(zt).sum()            # "type" is calibrated as it is
        did = log.decision(f"pr {i // 2}", Q2, {"blast": {"probabilities": dict(zip(opts, map(float, sure)))},
                                                 "type": {"probabilities": dict(zip(["bug", "docs", "feature"], map(float, pt)))}}, run="r@v0.4")
        log.outcome(did, {"blast": str(rng.choice(opts, p=truth)), "type": str(rng.choice(["bug", "docs", "feature"], p=pt))},
                    {"src": "reviewer", "group": f"pr{i // 2}"})
    res = log.resolved()
    fit, test = split(res)
    assert {d["group"] for d in fit}.isdisjoint({d["group"] for d in test}) and len(fit) + len(test) == 1000 and 200 < len(test) < 400
    cal, report = promote(res)
    assert report["blast"]["promote"] and "blast" in cal.params and abs(cal.params["blast"]["s"] - 0.5) < 0.12
    assert not report.get("type", {}).get("promote") and "type" not in cal.params
    path = tmp_path / "served.json"
    main(["promote", str(tmp_path / "f.jsonl"), "--calibrator", str(path)])
    assert json.loads(capsys.readouterr().out)["promoted"] == ["blast"] and OutcomeCalibrator.load(path).params["blast"] == cal.params["blast"]
    empty = FeedbackLog(tmp_path / "g.jsonl"); empty.outcome(empty.decision("s", CQ, {"blast": {"probabilities": {o: 0.25 for o in opts}}}, run="r"), {"blast": "broad"})
    main(["promote", str(tmp_path / "g.jsonl"), "--calibrator", str(tmp_path / "untouched.json")])
    assert not (tmp_path / "untouched.json").exists()


def test_a_human_label_beats_a_later_reviewer_label_per_question(tmp_path):
    """Outcomes merge per question: a person's label wins over another model's whatever arrived last, the reviewer's still
    fills the questions the person left alone, records() carry each label's source, and src= keeps one source only."""
    log = FeedbackLog(tmp_path / "f.jsonl")
    did = log.decision("pr", {**CQ, "type": {"type": "choice", "instructions": "Change type.", "criteria": {"bug": "fix", "docs": "docs"}}},
                       {"blast": {"choice": "contained", "probabilities": {"contained": 0.6, "moderate": 0.2, "broad": 0.1, "massive": 0.1}},
                        "type": {"choice": "bug", "probabilities": {"bug": 0.7, "docs": 0.3}}}, run="r@v0.4")
    log.outcome(did, {"type": "docs"}, {"src": "human"})
    log.outcome(did, {"type": "bug", "blast": "broad"}, {"src": "reviewer"})
    (d,) = log.resolved()
    assert d["labels"] == {"type": "docs", "blast": "broad"} and d["label_src"] == {"type": "human", "blast": "reviewer"}
    (rec,) = records([d])
    assert rec["questions"]["type"]["src"] == "human" and rec["questions"]["blast"]["src"] == "reviewer"
    assert log.resolved("reviewer")[0]["labels"] == {"type": "bug", "blast": "broad"}


def test_serve_logs_decisions_accepts_outcomes_and_applies_a_reloaded_calibrator(tmp_path, monkeypatch):
    """d1a.serving.serve's self-learning hooks: with a log, every answer carries a decision_id and POST /v1/feedback records its
    outcome; with a calibrator file, yes/no answers are recalibrated and a rewritten file takes effect on the next answer;
    with neither, /v1/feedback is a 404."""
    import os
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a.serving import serve
    from d1a.core.api import SystemOneRequest, to_record
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
        assert client.post("/v1/feedback", json={"decision_id": body["decision_id"], "labels": {"resolved": False}, "src": "human", "group": "psf/requests#1"}).json() == {"ok": True}
    (d,) = serve.LEARNING.log.resolved()
    assert d["id"] == body["decision_id"] and d["labels"] == {"resolved": False} and d["run"] == "JohnP1/d1a-e4b-mlx-q8@v0.4"
    assert d["label_src"] == {"resolved": "human"} and d["group"] == "psf/requests#1"
    assert d["answers"]["resolved"]["noul"] == 0.8 and d["meta"]["served"]["resolved"]["noul"] == expected   # the model's P is logged, the served one beside it
    assert d["questions"]["resolved"]["type"] == "noul"
    for i in range(30):                                                        # recalibrating on the live log fits the model's P, not the served one
        did = serve.Server._body(fake, req, meta, [[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})["decision_id"]
        serve.LEARNING.outcome(did, {"resolved": i < 6})
    p_logged, y_logged = pairs(serve.LEARNING.log.resolved(), "resolved")
    assert set(p_logged) == {0.8}
    refit = OutcomeCalibrator().fit(serve.LEARNING.log.resolved(), min_outcomes=10)
    assert abs(refit.p("resolved", 0.8) - y_logged.mean()) < 0.02            # applied to the raw P the next request gives, it lands on the outcome rate
    OutcomeCalibrator({"resolved": [1.0, 0.0, 50]}).save(cal_path)                 # a new calibrator, picked up without a restart
    os.utime(cal_path, (os.path.getmtime(cal_path) + 5,) * 2)
    again = serve.Server._body(fake, req, meta, [[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})
    assert again["answers"]["resolved"]["noul"] == 0.8
    monkeypatch.setattr(serve, "LEARNING", serve.Learning())
    plain = serve.Server._body(fake, req, meta, [[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})
    assert "decision_id" not in plain and plain["answers"]["resolved"]["noul"] == 0.8
    with TestClient(serve.app) as client:
        assert client.post("/v1/feedback", json={"decision_id": "x", "labels": {"resolved": True}}).status_code == 404
