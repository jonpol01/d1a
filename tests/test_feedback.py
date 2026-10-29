import json
import os
import numpy as np
import pytest

from d1a.training.data import materialize
from d1a.learning.feedback import FeedbackLog, OutcomeCalibrator, choice_log_loss, choice_pairs, gate, gate_choice, log_loss, main, pairs, promote, records, split, split_time

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


def test_promote_fits_only_on_the_decisions_the_served_model_made(tmp_path, capsys):
    """A calibrator corrects one model's probabilities, so after a model switch `promote --run <served repo@tag>` fits on
    that model's decisions alone. Here v0.4 was twice as sure as it should have been and v0.5 is half as sure. With fewer
    v0.5 outcomes than min_outcomes nothing is written, however many v0.4 has. Once v0.5 has enough, its own correction is
    promoted (s near 2), not v0.4's (near 0.5) or a blend of the two."""
    log = FeedbackLog(tmp_path / "f.jsonl"); rng = np.random.default_rng(5); opts = list(CQ["blast"]["criteria"])

    def decide(run, n, power, first):   # the served probabilities are the truth sharpened (power > 1) or flattened
        for i in range(first, first + n):
            z = rng.normal(0, 1.5, 4); truth = np.exp(z) / np.exp(z).sum(); served = np.exp(power * z) / np.exp(power * z).sum()
            did = log.decision(f"pr {i // 2}", CQ, {"blast": {"probabilities": dict(zip(opts, map(float, served)))}}, run=run)
            log.outcome(did, {"blast": str(rng.choice(opts, p=truth))}, {"src": "reviewer", "group": f"pr{i // 2}"})
    decide("r@v0.4", 1000, 2.0, 0); decide("r@v0.5", 10, 0.5, 1000)
    assert [d["run"] for d in log.resolved(run="r@v0.5")] == ["r@v0.5"] * 10 and len(log.resolved()) == 1010
    path, promote_v05 = tmp_path / "served.json", ["promote", str(tmp_path / "f.jsonl"), "--calibrator", str(tmp_path / "served.json"), "--run", "r@v0.5"]
    main(promote_v05)
    assert json.loads(capsys.readouterr().out)["promoted"] == [] and not path.exists()
    decide("r@v0.5", 1000, 0.5, 1010)
    main(promote_v05)
    assert json.loads(capsys.readouterr().out)["promoted"] == ["blast"] and abs(OutcomeCalibrator.load(path).params["blast"]["s"] - 2.0) < 0.5


def test_serve_with_a_calibrator_path_not_written_yet_serves_the_model_and_takes_the_file_once_written(tmp_path, monkeypatch):
    """At a model switch D1A_OUTCOME_CALIBRATOR stays set, so promote keeps running with --run, while the old model's file
    is moved aside. A server whose calibrator path names no file yet serves exactly what one without a calibrator serves,
    yes/no and choice answers alike. The first answer after promote writes the file (whole, then renamed into place)
    goes through it, without a restart."""
    import os
    from types import SimpleNamespace
    from d1a.serving import serve
    from d1a.core.api import SystemOneRequest, to_record
    req = SystemOneRequest(state="issue + patch", questions={**Q, **CQ})
    _, meta = to_record(req)
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested="JohnP1/d1a-e4b-mlx-q8@v0.5"), tok=None)
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    answer = lambda: serve.Server._body(fake, req, meta, [[0.2, 0.8], [0.1, 0.2, 0.3, 0.4]], {"tokens": 10, "latency_ms": 1.0})
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "plain.jsonl"))
    plain = answer()
    path = tmp_path / "calibrator.json"
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl", path))
    waiting = answer()
    assert not path.exists() and waiting["answers"] == plain["answers"] and serve.LEARNING.card()["calibrated_questions"] == []
    promoted = OutcomeCalibrator({"resolved": [1.0, -2.0, 50]})
    promoted.save(path.with_name(path.name + ".tmp")); os.replace(path.with_name(path.name + ".tmp"), path)   # as promote writes it
    taken = answer()
    assert taken["answers"]["resolved"]["noul"] == round(promoted.p("resolved", 0.8), 4) < 0.8
    assert taken["answers"]["blast"] == plain["answers"]["blast"] and serve.LEARNING.card()["calibrated_questions"] == ["resolved"]
    path.rename(tmp_path / "calibrator-v0.4.json")             # rolled back by moving the file aside: the next answer is the model's own
    assert answer()["answers"] == plain["answers"] and serve.LEARNING.card()["calibrated_questions"] == []
    (tmp_path / "calibrator-v0.4.json").rename(path)
    assert answer()["answers"] == taken["answers"]

    def unreadable(p):
        raise PermissionError(13, "Permission denied", str(p))
    monkeypatch.setattr(serve.os.path, "getmtime", unreadable)   # any other error reading it keeps the calibrator it has
    assert answer()["answers"] == taken["answers"]


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


def _pr_decision(log):
    return log.decision("pr", {**CQ, "type": {"type": "choice", "instructions": "Change type.", "criteria": {"bug": "fix", "docs": "docs"}}},
                        {"blast": {"choice": "contained", "probabilities": {"contained": 0.6, "moderate": 0.2, "broad": 0.1, "massive": 0.1}},
                         "type": {"choice": "bug", "probabilities": {"bug": 0.7, "docs": 0.3}}}, run="r@v0.4")


def test_a_void_withdraws_one_sources_outcomes_and_the_next_source_decides(tmp_path):
    """A PR author's own labels logged as "human" outrank the reviewer; voiding them (with a reason, the log untouched
    before it) leaves the reviewer's label, and a decision whose only outcome is voided is pending again."""
    log = FeedbackLog(tmp_path / "f.jsonl")
    did, alone = _pr_decision(log), _pr_decision(log)
    log.outcome(did, {"type": "docs", "blast": "contained"}, {"src": "human"})
    log.outcome(did, {"type": "bug", "blast": "broad"}, {"src": "reviewer"})
    log.outcome(alone, {"type": "docs"}, {"src": "human"})
    before = (tmp_path / "f.jsonl").read_text(encoding="utf-8")
    main(["void", str(tmp_path / "f.jsonl"), "--ids", f"{did},{alone}", "--src", "human", "--reason", "PR-author labels, not corrections"])
    assert (tmp_path / "f.jsonl").read_text(encoding="utf-8").startswith(before)   # appended, never rewritten
    (d,) = log.resolved()
    assert d["labels"] == {"type": "bug", "blast": "broad"} and d["label_src"] == {"type": "reviewer", "blast": "reviewer"}
    assert [x["id"] for x in log.pending()] == [alone]
    with pytest.raises(ValueError): log.void([did], "human", "")


def test_per_question_sources_keep_the_reviewers_type_but_not_its_severity(tmp_path):
    """outcomes.sources per question (#252: the review bot's severity scale is not the maintainers'): with {"sev": ["human"],
    "*": ["reviewer", "human"]} a reviewer outcome teaches type and blast but never severity; a person's severity still counts."""
    from d1a.learning import settings
    log = FeedbackLog(tmp_path / "f.jsonl"); did = _pr_decision(log); other = _pr_decision(log)
    log.outcome(did, {"type": "docs", "blast": "broad", "sev": "P2"}, {"src": "reviewer"})
    log.outcome(other, {"sev": "P1"}, {"src": "human"})
    per_q = {"sev": ["human"], "*": ["reviewer", "human"]}
    a, b = log.resolved(per_q)
    assert a["labels"] == {"type": "docs", "blast": "broad"} and a["label_src"] == {"type": "reviewer", "blast": "reviewer"}
    assert b["labels"] == {"sev": "P1"}
    assert log.resolved()[0]["labels"]["sev"] == "P2"   # unfiltered, the reviewer's severity is still in the log
    path = tmp_path / "learning.json"; settings.init(path)
    settings.set_values(path, {"outcomes.sources": per_q})
    assert settings.load(path)[0]["outcomes"]["sources"] == per_q
    for bad in ({}, {"sev": []}, {"sev": "human"}, {"sev": ["human", "human"]}):
        with pytest.raises(ValueError): settings.set_values(path, {"outcomes.sources": bad})


def test_an_outcome_after_a_void_counts_again(tmp_path):
    """A void covers what came before it: a person correcting the same decision later still wins over the reviewer."""
    log = FeedbackLog(tmp_path / "f.jsonl"); did = _pr_decision(log)
    log.outcome(did, {"type": "docs"}, {"src": "human"}); log.outcome(did, {"type": "bug"}, {"src": "reviewer"})
    log.void([did], "human", "PR-author labels")
    log.outcome(did, {"type": "docs"}, {"src": "human"})
    (d,) = log.resolved()
    assert d["labels"]["type"] == "docs" and d["label_src"]["type"] == "human"


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


def test_feedback_takes_outcomes_only_for_decisions_in_the_log(tmp_path, monkeypatch):
    """POST /v1/feedback refuses a decision_id the log does not hold, and writes nothing (#320): an outcome for no decision
    only grows the log, and is what a forged or mistyped id sends. A decision another writer logged is found by reading the
    log, one still being written once its line ends, past a line a crash cut short; a replaced log is read again from its
    start. A body sent as text/plain, the form a cross-site page can send without CORS, is refused whatever it holds."""
    from fastapi.testclient import TestClient
    from d1a.serving import serve
    path = tmp_path / "log.jsonl"
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(path))
    decide = lambda: FeedbackLog(path).decision("issue + patch", {"resolved": {"type": "noul"}}, {"resolved": {"noul": 0.8}}, run="r")
    first = decide()
    with TestClient(serve.app) as client:
        post = lambda did: client.post("/v1/feedback", json={"decision_id": did, "labels": {"resolved": True}, "src": "human"})
        size = path.stat().st_size
        assert post("0" * 32).status_code == 422 and path.stat().st_size == size
        assert post(first).json() == {"ok": True}
        second = decide()                                                   # logged after the log was read
        assert post(second).json() == {"ok": True}
        size = path.stat().st_size
        r = client.post("/v1/feedback", content=json.dumps({"decision_id": second, "labels": {"resolved": True}}),
                        headers={"content-type": "text/plain"})
        assert r.status_code == 422 and path.stat().st_size == size
        partial = "p" * 32
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"kind": "decision", "id": "' + "c" * 32 + '", "st' + json.dumps({"kind": "outcome", "id": "x"}) + "\n")   # a crash mid-write, run into the next append
            f.write(json.dumps({"kind": "decision", "id": partial}))       # its line not ended yet
        assert post(partial).status_code == 422
        with open(path, "a", encoding="utf-8") as f: f.write("\n")
        assert post(partial).json() == {"ok": True}
        replaced = tmp_path / "new.jsonl"
        replaced.write_text("\n" * (path.stat().st_size + 10), encoding="utf-8")   # a longer file, so only its identity says it is new
        os.replace(replaced, path)
        third = decide()
        assert post(first).status_code == 422 and post(third).json() == {"ok": True}
    assert [json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if '"outcome"' in l] == [third]


def test_the_outcome_calibrator_keeps_answers_read_at_different_temperatures_apart(tmp_path, capsys, monkeypatch):
    """A checkpoint's use-case temperature (#209) reads one request's probabilities at another temperature than the next
    one's, and the log records which. Here answers read at the checkpoint's T 1.78 are far too sure (P ~0.8 when 20% are
    true) and routing answers read at T 0.85 are calibrated. Fit and promote each temperature on its own decisions, so only
    the 1.78 correction is promoted; the server applies an entry only to answers read at its temperature; and a file from
    before (entries without a temperature, fitted when every answer was read at the checkpoint's own) applies at the
    checkpoint's temperature only, never to routing answers."""
    import os
    from types import SimpleNamespace
    from d1a.serving import serve
    from d1a.core.api import SystemOneRequest, to_record
    OWN, ROUTING = 1.7818, 0.85
    log = FeedbackLog(tmp_path / "f.jsonl"); rng = np.random.default_rng(7)
    for i in range(1200):
        if i % 2:
            y = rng.random() < 0.2; p = 1 / (1 + np.exp(-(1.4 + (0.9 if y else 0.0) + rng.normal(0, 0.6))))
            did = log.decision(f"s {i}", Q, {"resolved": {"type": "noul", "noul": float(p)}}, run="r@v0.5", temperature=OWN)
        else:
            p = rng.uniform(0.05, 0.95); y = rng.random() < p
            did = log.decision(f"s {i}", Q, {"resolved": {"type": "noul", "noul": float(p)}}, run="r@v0.5", use_case="routing", temperature=ROUTING)
        log.outcome(did, {"resolved": bool(y)}, {"group": f"g{i // 2}"})
    res = log.resolved()
    cal = OutcomeCalibrator().fit(res)
    assert cal.params == {} and sorted(cal.temperatures) == [ROUTING, OWN]
    for T in (OWN, ROUTING):
        rows = [d for d in res if d["temperature"] == T]
        p, y = pairs(rows, "resolved")
        assert abs(np.mean([cal.at(T).p("resolved", x) for x in p]) - y.mean()) < 0.03, T
    path = tmp_path / "served.json"
    main(["promote", str(tmp_path / "f.jsonl"), "--calibrator", str(path)])
    assert json.loads(capsys.readouterr().out)["promoted"] == [f"resolved@{OWN!r}"]
    served = OutcomeCalibrator.load(path)
    assert served.params == {} and list(served.temperatures) == [OWN]
    p, y = pairs([d for d in res if d["temperature"] == OWN], "resolved")
    assert abs(np.mean([served.at(OWN).p("resolved", x) for x in p]) - y.mean()) < 0.03

    req = {use_case: SystemOneRequest(state="s", questions=Q, use_case=use_case) for use_case in (None, "routing")}
    _, meta = to_record(req[None])
    head = SimpleNamespace(temperature=OWN, temperature_for=lambda use_case: ROUTING if use_case == "routing" else OWN)
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested="r@v0.5"), tok=None, model=SimpleNamespace(head=head))
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    answer = lambda use_case: serve.Server._body(fake, req[use_case], meta, [[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})["answers"]["resolved"]["noul"]
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl", path))
    assert answer(None) == round(served.at(OWN).p("resolved", 0.8), 4) < 0.5 and answer("routing") == 0.8
    assert serve.LEARNING.card()["calibrated_at"] == {repr(OWN): ["resolved"]}
    OutcomeCalibrator({"resolved": [1.0, -2.0, 50]}).save(path)   # a calibrator written before decisions logged their temperature
    os.utime(path, (path.stat().st_mtime + 5,) * 2)
    assert answer(None) == round(OutcomeCalibrator({"resolved": [1.0, -2.0, 50]}).p("resolved", 0.8), 4) < 0.8 and answer("routing") == 0.8


def test_a_request_marked_decision_log_off_is_answered_but_never_logged(tmp_path, monkeypatch):
    """Demo and smoke traffic sends `x-d1a-decision-log: off`: no outcome ever follows it, so logged it only swelled the
    pending decisions the outcome collector and `status` count. Every answering endpoint honours it; without it, or with
    any other value, the answer is logged as before."""
    from concurrent.futures import Future
    from contextlib import asynccontextmanager, nullcontext
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a.serving import serve

    def submit(rec, media=None, use_case=None):
        f = Future(); f.set_result(([[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})); return f
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested="JohnP1/d1a-e4b-mlx-q8@v0.5"), tok=None, submit=submit,
                           probs=lambda rec, use_case=None: submit(rec).result())
    for name in ("answer", "answer_async", "_body"): setattr(fake, name, getattr(serve.Server, name).__get__(fake))
    monkeypatch.setattr(serve, "server", lambda: nullcontext(fake))
    @asynccontextmanager
    async def server_async():
        yield fake
    monkeypatch.setattr(serve, "server_async", server_async)
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl"))
    req = {"state": "issue + patch", "questions": Q}
    with TestClient(serve.app) as client:
        for path in ("/v1/systemone", "/v1/systemone/separate"):
            r = client.post(path, json=req, headers={"x-d1a-decision-log": "Off"})
            assert r.status_code == 200 and r.json()["answers"]["resolved"]["noul"] == 0.8 and "decision_id" not in r.json()
        assert serve.LEARNING.log.events() == []
        kept = [client.post("/v1/systemone", json=req, headers=h).json() for h in ({}, {"x-d1a-decision-log": "on"})]
    assert [e["id"] for e in serve.LEARNING.log.events()] == [k["decision_id"] for k in kept]


def test_a_recalibrated_choice_answer_carries_its_own_confidence_and_a_score_answer_is_left_alone():
    """#195: the served answer must not contradict itself. A choice answer's confidence is read off the recalibrated
    probabilities; a score answer, which also carries probabilities by level, is never rescaled as a choice."""
    from d1a.core.api import answer, choice_confidence
    cal = OutcomeCalibrator({"blast": {"s": 0.5, "n": 100}, "effort": {"s": 0.5, "n": 100}})
    choice = answer([0.7, 0.2, 0.1], {"type": "choice", "keys": ["contained", "moderate", "massive"]})
    score = answer([0.1, 0.2, 0.7], {"type": "score", "legend": "0-2"})
    out = cal.apply({"blast": choice, "effort": score})
    assert out["blast"]["choice"] == "contained" and out["blast"]["probabilities"]["contained"] < choice["probabilities"]["contained"]
    # read off the unrounded probabilities, as the server reads it: equal to the shown ones' up to their rounding
    assert out["blast"]["confidence"] == pytest.approx(choice_confidence(list(out["blast"]["probabilities"].values())), abs=2e-4)
    assert out["blast"]["confidence"] < choice["confidence"]
    assert out["effort"] == score


def test_promote_drops_a_choice_calibration_a_score_question_carried_from_before():
    """A calibrator fitted before #195 may hold choice-type params for a score question. apply() leaves score answers
    alone now, so promote drops the entry: what the file lists is what is applied."""
    from d1a.core.api import answer
    stale = OutcomeCalibrator({"effort": {"s": 0.5, "n": 100}, "resolved": [1.0, 0.0, 50]})
    decisions = [{"id": f"d{i}", "answers": {"effort": answer([0.2, 0.5, 0.3], {"type": "score", "legend": "0-2"})}, "labels": {"effort": "1"}} for i in range(4)]
    served, report = promote(decisions, stale)
    assert "effort" not in served.params and served.params["resolved"] == [1.0, 0.0, 50]
    assert report["effort"]["promote"] is False and "never applied" in report["effort"]["dropped"]


def test_the_promote_command_writes_a_drop_even_when_nothing_is_promoted(tmp_path, capsys):
    """hermes-prbot on #204: promote() dropped a stale score entry in memory, but the CLI wrote the file only when a
    question was promoted, so a drop alone never reached the server."""
    from d1a.core.api import answer
    path = tmp_path / "served.json"; OutcomeCalibrator({"effort": {"s": 0.5, "n": 100}, "resolved": [1.0, 0.0, 50]}).save(path)
    log = FeedbackLog(tmp_path / "f.jsonl")
    for i in range(4):
        did = log.decision(f"s {i}", {"effort": {"type": "score", "criteria": ["low", "mid", "high"]}},
                           {"effort": answer([0.2, 0.5, 0.3], {"type": "score", "legend": "0-2"})}, run="r@v0.5")
        log.outcome(did, {"effort": "1"}, {"src": "reviewer", "group": f"g{i}"})
    main(["promote", str(tmp_path / "f.jsonl"), "--calibrator", str(path)])
    assert json.loads(capsys.readouterr().out)["promoted"] == []
    assert OutcomeCalibrator.load(path).params == {"resolved": [1.0, 0.0, 50]}


def _mix_shift(n, seed):
    """Choice decisions from a model trained on a balanced mix, served where option "a" is three times as common: the
    model's probabilities are right within each option's share but blind to the mix."""
    rng = np.random.default_rng(seed); keys = ["a", "b", "c"]; out = []
    for i in range(n):
        y = rng.choice(3, p=[0.6, 0.2, 0.2]); logits = rng.normal(0, 1, 3); logits[y] += 1.2
        p = np.exp(logits) / np.exp(logits).sum()
        out.append({"id": f"d{i}", "ts": float(i), "group": f"g{i}", "temperature": 1.5,
                    "answers": {"q": {"type": "choice", "choice": keys[int(p.argmax())], "probabilities": dict(zip(keys, p.tolist()))}}, "labels": {"q": keys[y]}})
    return out


def test_the_vector_form_learns_a_label_mix_the_model_was_not_trained_on_and_the_default_stays_one_temperature():
    """On decisions after training, a per-option bias learned from outcomes corrects
    the label mix; one temperature cannot. The default form is unchanged."""
    fit, test = _mix_shift(1500, 0), _mix_shift(1500, 1)
    temp, vec = OutcomeCalibrator()._fit(fit), OutcomeCalibrator()._fit(fit, form="vector")
    assert set(temp.params["q"]) == {"s", "n"} and set(vec.params["q"]) == {"s", "b", "n"}
    b = vec.params["q"]["b"]; assert b["a"] > b["b"] + 0.5 and b["a"] > b["c"] + 0.5
    probs, labels = choice_pairs(test, "q")
    raw, t_loss, v_loss = (choice_log_loss([c.probs("q", p) for p in probs] if c else probs, labels).mean() for c in (None, temp, vec))
    assert v_loss < t_loss - 0.02 and v_loss < raw - 0.02


def test_a_served_choice_changes_only_with_serve_choice_and_never_away_from_a_held_option(tmp_path):
    """The default keeps the model's chosen option (only probabilities move). serve["choice"] serves the recalibrated top
    option, except that a held option (severity P0/P1, which a label-mix correction pushes down) stays the model's."""
    params = {"sev": {"s": 1.0, "b": {"P2": 3.0}, "n": 100}}
    ans = lambda choice: {"sev": {"type": "choice", "choice": choice, "confidence": 0.5,
                                  "probabilities": {"P0": 0.1, "P1": 0.3, "P2": 0.2, "P3": 0.4}}}
    keep = OutcomeCalibrator(params).apply(ans("P3"))["sev"]
    assert keep["choice"] == "P3" and max(keep["probabilities"], key=keep["probabilities"].get) == "P2"
    serve = OutcomeCalibrator(params, serve={"choice": True, "hold": {"sev": ["P0", "P1"]}})
    assert serve.apply(ans("P3"))["sev"]["choice"] == "P2"
    assert serve.apply(ans("P1"))["sev"]["choice"] == "P1"
    path = tmp_path / "cal.json"; serve.save(path); back = OutcomeCalibrator.load(path)
    assert back.serve == serve.serve and back.at(1.5).serve == serve.serve
    OutcomeCalibrator(params).save(path); assert "serve" not in json.loads(path.read_text(encoding="utf-8")) and not OutcomeCalibrator.load(path).serve["choice"]


def test_the_time_split_holds_out_the_newest_groups_and_orders_a_replay_by_its_original_decision():
    rows = [{"id": f"d{i}", "ts": float(i), "group": f"g{i}"} for i in range(10)]
    rows.append({"id": "r", "ts": 99.0, "decided_ts": 0.5, "group": "g-replayed", "replay_of": "old"})
    fit, held = split_time(rows, share=0.3)
    assert {d["group"] for d in held} == {"g7", "g8", "g9"} and "g-replayed" in {d["group"] for d in fit}
    assert {d["group"] for d in fit}.isdisjoint({d["group"] for d in held})


def test_x_d1a_audit_on_adds_the_audit_block_and_names_the_calibrated_questions(tmp_path, monkeypatch):
    """#32 through the endpoint: only `x-d1a-audit: on` adds the block, so the System One wire is unchanged otherwise. Its
    outcome_calibrator lists the questions the calibrator holds entries for at the answer's temperature (the served
    answers being a recalibrated copy proves nothing), and logged says whether the decision was written."""
    from concurrent.futures import Future
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a.serving import serve

    def submit(rec, media=None, use_case=None):
        f = Future(); f.set_result(([[0.2, 0.8]], {"tokens": 10, "state_tokens": 6, "latency_ms": 1.0, "prefix_cache_hit": False})); return f
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested="JohnP1/d1a-e4b-mlx-q8@v0.6", weights_sha256=lambda: "ab" * 32), tok=None, submit=submit,
                           model=SimpleNamespace(backend="mlx", dtype="bfloat16"), probs=lambda rec, use_case=None: submit(rec).result())
    for name in ("answer", "answer_async", "_body", "audit_block", "weights_digest"): setattr(fake, name, getattr(serve.Server, name).__get__(fake))
    @asynccontextmanager
    async def server_async():
        yield fake
    monkeypatch.setattr(serve, "server_async", server_async)
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    cal_path = tmp_path / "cal.json"
    OutcomeCalibrator({"resolved": [1.0, -2.0, 50]}).save(cal_path)
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl", cal_path))
    req = {"state": "issue + patch", "questions": Q}
    with TestClient(serve.app) as client:
        bodies = {h: client.post("/v1/systemone", json=req, headers={"x-d1a-audit": h} if h else {}).json() for h in (None, "off", "On")}
    assert "audit" not in bodies[None] and "audit" not in bodies["off"]
    a = bodies["On"]["audit"]
    assert a["run"] == "JohnP1/d1a-e4b-mlx-q8@v0.6" and a["weights_sha256"] == "ab" * 32 and a["backend"] == "mlx"
    assert a["outcome_calibrator"] == ["resolved"] and a["logged"] is True and bodies["On"]["decision_id"]
    assert a["passes"] == [{"tokens": 10, "prefix_cache_hit": False, "cached_tokens": 0}]
    assert "issue + patch" not in json.dumps(a)   # hashes only, never the request's text
    import os
    OutcomeCalibrator({"other": [1.0, -2.0, 50]}).save(cal_path)   # entries for another question only: this answer is not recalibrated
    os.utime(cal_path, (os.path.getmtime(cal_path) + 5,) * 2)
    with TestClient(serve.app) as client:
        assert client.post("/v1/systemone", json=req, headers={"x-d1a-audit": "on"}).json()["audit"]["outcome_calibrator"] == []


def test_the_gate_promotes_only_on_enough_held_out_groups():
    """#364: a candidate fitted on a few dozen outcomes and judged on a dozen held-out groups passes the bootstrap by chance
    far more often than its 95% level says, so the gate needs min_held_out groups (30 by default) before it may promote,
    even a gain this plain."""
    sure = [{"id": f"d{i}", "ts": float(i), "group": f"g{i}", "temperature": 1.5, "labels": {"q": "b" if i % 5 else "a"},
             "answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}}} for i in range(100)]
    _, few = promote(sure[:60], form="vector", split_by="time")
    assert not few["q@1.5"]["promote"] and "18 held-out groups, fewer than 30" in few["q@1.5"]["reasons"][-1]
    assert promote(sure[:60], form="vector", split_by="time", min_held_out=10)[1]["q@1.5"]["promote"]
    assert promote(sure, form="vector", split_by="time")[1]["q@1.5"]["promote"]
