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
