"""d1a.learning.flags (#237): a flag is computed on raw probabilities, is served beside an answer without changing it,
holds only for its run, and the gate refuses a flag that catches no more than the top answer or raises too often."""
import copy
import json

import pytest

from d1a.learning import flags as F

RUN = "JohnP1/d1a-e4b-mlx-q8@v0.6"
SEV = {"question": "sev", "options": ["P0", "P1"], "t": 0.30, "space": "raw", "run": RUN, "budget": 0.15,
       "fitted_on": {"kit": "test", "created_max": "2026-10-08T17:40:54Z", "n": 10}}


def answers(p0, p1, top="P2"):
    probs = {"P0": p0, "P1": p1, "P2": 1 - p0 - p1 - 0.1, "P3": 0.1, "P4": 0.0}
    return {"sev": {"type": "choice", "choice": top, "confidence": 0.5, "probabilities": probs},
            "type": {"type": "choice", "choice": "type/bug", "confidence": 0.9, "probabilities": {"type/bug": 0.95, "type/docs": 0.05}}}


def test_no_flag_configured_leaves_the_answers_byte_identical():
    a = answers(0.2, 0.2); before = json.dumps(a, sort_keys=True)
    assert F.annotate(a, copy.deepcopy(a), {}, RUN) is a and json.dumps(a, sort_keys=True) == before


def test_a_flag_adds_only_its_field():
    a = answers(0.2, 0.2); raw = copy.deepcopy(a); out = F.annotate(a, raw, {"sev_severe": SEV}, RUN)
    assert out["sev"]["flags"]["sev_severe"] == {"on": True, "p": 0.4, "t": 0.30, "space": "raw"}
    assert {k: v for k, v in out["sev"].items() if k != "flags"} == raw["sev"] and out["type"] == raw["type"]


def test_the_top_answer_in_the_group_raises_the_flag_whatever_t_is():
    a = answers(0.05, 0.4, top="P1")
    assert F.annotate(a, copy.deepcopy(a), {"s": {**SEV, "t": 0.99}}, RUN)["sev"]["flags"]["s"]["on"]


def test_a_flag_fitted_for_another_run_is_not_served():
    a = answers(0.3, 0.3)
    assert "flags" not in F.annotate(a, copy.deepcopy(a), {"sev_severe": {**SEV, "run": "JohnP1/d1a-e2b-mlx-q8@v0.6"}}, RUN)["sev"]


def test_the_flag_reads_raw_probabilities_not_the_calibrated_answer():
    served = answers(0.05, 0.05); raw = answers(0.2, 0.2)   # an outcome calibrator moved the served probabilities down
    f = F.annotate(served, raw, {"sev_severe": SEV}, RUN)["sev"]["flags"]["sev_severe"]
    assert f["on"] and f["p"] == 0.4 and served["sev"]["probabilities"]["P0"] == 0.05


def rows(spec):
    """benchmark rows for the sev question from (p0, p1, top index >= 2, gold index): the rest of the mass on the top option."""
    keys = ["P0", "P1", "P2", "P3", "P4"]; out = []
    for i, (p0, p1, top, gold) in enumerate(spec):
        assert top >= 2 and 1 - p0 - p1 > max(p0, p1)
        p = [p0, p1, 0.0, 0.0, 0.0]; p[top] = 1 - p0 - p1
        out.append({"id": f"pr/{i}", "group": f"pr/{i}", "question": "sev", "variant": "clean", "keys": keys, "label": gold, "p": p})
    return out


def test_fit_sets_t_at_the_budget_quantile():
    rs = rows([(0.004 * i, 0.0, 2, 2) for i in range(100)])
    (name, f), = F.fit(rs, "sev_severe", "sev", ["P0", "P1"], 0.15, RUN, "test", "2026-10-08T17:40:54Z").items()
    assert name == "sev_severe" and abs(f["t"] - 0.3366) < 1e-3 and f["fitted_on"]["n"] == 100
    assert abs(F.measure(rs, f)["flag"]["rate"] - 0.15) < 0.02
    with pytest.raises(ValueError):
        F.fit(rs, "x", "sev", ["P0"], 0.9, RUN, "test", "t")


def test_gate_serves_a_flag_that_catches_what_the_top_answer_misses():
    severe = [(0.25, 0.20, 2, 1)] * 20                              # severe, top answer P2, group probability 0.45
    rest = [(0.02, 0.03, 2, 2)] * 150 + [(0.1, 0.1, 3, 3)] * 30      # not severe, low group probability
    assert F.gate(rows(severe + rest), {**SEV, "t": 0.40})["serve"]


def test_gate_refuses_a_flag_no_better_than_the_top_answer():
    severe = [(0.25, 0.20, 2, 1)] * 20; rest = [(0.02, 0.03, 2, 2)] * 180
    g = F.gate(rows(severe + rest), {**SEV, "t": 0.99})
    assert not g["serve"] and "not clearly higher" in g["reasons"][0]


def test_gate_refuses_a_flag_over_its_budget():
    severe = [(0.25, 0.20, 2, 1)] * 20; rest = [(0.02, 0.03, 2, 2)] * 180
    g = F.gate(rows(severe + rest), {**SEV, "t": 0.0})
    assert not g["serve"] and any("above the budget" in r for r in g["reasons"])


def test_rows_for_another_question_are_refused():
    with pytest.raises(ValueError):
        F.measure(rows([(0.1, 0.1, 2, 2)]), {**SEV, "question": "blast"})


def test_the_server_serves_a_configured_flag_beside_the_answer_and_logs_it(tmp_path, monkeypatch):
    """d1a.serving.serve: a flag in the D1A_LEARNING file for the served run is added to that answer and kept in the log's
    meta; the answer itself and the log's answers are the model's; another run's flag, or no file, changes nothing."""
    from types import SimpleNamespace
    from d1a.serving import serve
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.learning import settings
    q = {"sev": {"type": "choice", "instructions": "Severity?", "criteria": {k: None for k in ("P0", "P1", "P2", "P3", "P4")}}}
    req = SystemOneRequest(state="a pull request", questions=q); _, meta = to_record(req)
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested=RUN), tok=None)
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    ps = [[0.2, 0.15, 0.45, 0.15, 0.05]]
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl"))
    monkeypatch.setattr(serve, "FLAGS", serve.flags.Configured(None))
    plain = serve.Server._body(fake, req, meta, ps, {"tokens": 1, "latency_ms": 1.0})
    path = tmp_path / "learning.json"; settings.init(path)
    settings.set_values(path, {"flags.sev_severe": SEV})
    monkeypatch.setattr(serve, "FLAGS", serve.flags.Configured(path))
    body = serve.Server._body(fake, req, meta, ps, {"tokens": 1, "latency_ms": 1.0})
    assert body["answers"]["sev"]["flags"] == {"sev_severe": {"on": True, "p": 0.35, "t": 0.30, "space": "raw"}}
    assert {k: v for k, v in body["answers"]["sev"].items() if k != "flags"} == plain["answers"]["sev"]
    logged = [d for d in serve.LEARNING.log.events() if d["kind"] == "decision"][-1]
    assert "flags" not in logged["answers"]["sev"] and logged["meta"]["flags"]["sev"]["sev_severe"]["on"]
    settings.set_values(path, {"flags.sev_severe": {**SEV, "run": "JohnP1/d1a-e2b-mlx-q8@v0.6"}})
    import os; os.utime(path, (os.path.getmtime(path) + 5,) * 2)
    other = serve.Server._body(fake, req, meta, ps, {"tokens": 1, "latency_ms": 1.0})
    assert "flags" not in other["answers"]["sev"]


def test_a_shadow_flag_is_logged_but_not_served(tmp_path, monkeypatch):
    """#296: mode "shadow" keeps the response byte-identical to the one without the flag, and the flag in the log's meta."""
    from types import SimpleNamespace
    from d1a.serving import serve
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.learning import settings
    q = {"sev": {"type": "choice", "instructions": "Severity?", "criteria": {k: None for k in ("P0", "P1", "P2", "P3", "P4")}}}
    req = SystemOneRequest(state="a pull request", questions=q); _, meta = to_record(req)
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested=RUN), tok=None)
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl"))
    ps = [[0.2, 0.15, 0.45, 0.15, 0.05]]
    monkeypatch.setattr(serve, "FLAGS", serve.flags.Configured(None))
    plain = serve.Server._body(fake, req, meta, ps, {"tokens": 1, "latency_ms": 1.0})
    path = tmp_path / "learning.json"; settings.init(path); settings.set_values(path, {"flags.sev_severe": {**SEV, "mode": "shadow"}})
    monkeypatch.setattr(serve, "FLAGS", serve.flags.Configured(path))
    body = serve.Server._body(fake, req, meta, ps, {"tokens": 1, "latency_ms": 1.0})
    strip = lambda b: json.dumps({k: v for k, v in b.items() if k != "decision_id"}, sort_keys=True)
    assert strip(body) == strip(plain)
    logged = [d for d in serve.LEARNING.log.events() if d["kind"] == "decision"][-1]["meta"]["flags"]["sev"]["sev_severe"]
    assert logged == {"on": True, "p": 0.35, "t": 0.30, "space": "raw", "shadow": True}


TRACK = {"window": 100, "every": 20, "clamp": [0.2, 0.4], "alert_rate": 0.23}


def logged(ps, run=RUN, replay=False, state=None, prefix="pr"):
    """Decision events for the sev question with group probability p (split evenly over P0/P1), top answer P2."""
    out = []
    for i, p in enumerate(ps):
        probs = {"P0": p / 2, "P1": p / 2, "P2": 1 - p - 0.01, "P3": 0.01, "P4": 0.0}
        out.append({"kind": "decision", "id": f"d{i}", "ts": 1.79e9 + i, "run": run, "state": state or f"{prefix} {i}",
                    "answers": {"sev": {"type": "choice", "choice": "P2", "probabilities": probs}}, **({"replay_of": "x"} if replay else {})})
    return out


def test_track_refits_t_from_logged_scores_only_when_due_and_within_the_clamp():
    """#296: below `window` decisions the offline t stays; then t is the (1 - budget) quantile of the last `window` of the
    flag's run, refit only every `every` new decisions, clamped; replays, other runs and repeated states don't count."""
    f = {**SEV, "track": TRACK}
    noise = logged([0.9] * 50, replay=True, prefix="replayed") + logged([0.9] * 50, run="other@v1", prefix="other") + logged([0.9] * 50, state="same pr")
    assert F.track(noise + logged([0.25] * 98), f)["t"] is None                      # 99 decisions count: below the window
    ps = [0.21 + 0.001 * i for i in range(100)]                                       # 0.15 above the 0.85 quantile ...
    r = F.track(logged(ps), f)
    assert abs(r["t"]["t"] - round(float(F.np.quantile(ps, 0.85)), 4)) < 1e-9 and r["t"]["fitted_on"]["kit"] == "decision-log"
    assert F.track(logged(ps), f, refit_at=90)["t"] is None                           # ... only 10 new since the last refit
    assert F.track(logged([0.9] * 100), f)["t"]["t"] == 0.4                           # clamped,
    assert any("clamp" in a for a in F.track(logged([0.9] * 100), f)["alerts"])       # and reported
    hot = {**f, "t": 0.30}
    assert any("flag rate" in a for a in F.track(logged([0.1] * 150 + [0.35] * 100), {**hot, "track": {**TRACK, "every": 100, "window": 250}}, refit_at=250)["alerts"])


def test_tick_writes_a_tracked_t_and_alerts_once(tmp_path, monkeypatch):
    """#296: the loop's tick refits a tracked flag into the settings file (audited, and re-read by the server) and notifies
    an alert at its onset only."""
    from d1a.learning import loop, settings as S
    from d1a.learning.feedback import FeedbackLog
    path = tmp_path / "learning.json"; S.init(path)
    S.set_values(path, {"flags.sev_severe": {**SEV, "mode": "shadow", "track": TRACK}, "reports.notify": ["flag_alert"]})
    log = FeedbackLog(tmp_path / "log.jsonl")
    for e in logged([0.9] * 120): log._append(e)
    sent = []; monkeypatch.setattr(loop, "notify", lambda settings, text: sent.append(text) or True)
    line = loop.tick(log.path, tmp_path / "cal.json", run=RUN, settings_path=path)
    assert line["flags"]["sev_severe"]["t"] == 0.4 and S.load(path)[0]["flags"]["sev_severe"]["t"] == 0.4
    assert S.load(path)[0]["flags"]["sev_severe"]["mode"] == "shadow" and "tick:track" in path.with_name(path.name + ".audit.jsonl").read_text(encoding="utf-8")
    loop.tick(log.path, tmp_path / "cal.json", run=RUN, settings_path=path)
    assert len(sent) == 1 and "clamp" in sent[0]


def test_a_broken_settings_file_does_not_stop_the_tick(tmp_path):
    """#297 review: a due refit into a settings file that no longer parses is reported, and the tick still saves its status."""
    from d1a.learning import loop, settings as S
    from d1a.learning.feedback import FeedbackLog
    path = tmp_path / "learning.json"; S.init(path); S.set_values(path, {"flags.sev_severe": {**SEV, "track": TRACK}})
    path.write_text("{", encoding="utf-8")                                            # a half-written edit; last-valid holds the flag
    log = FeedbackLog(tmp_path / "log.jsonl")
    for e in logged([0.9] * 120): log._append(e)
    line = loop.tick(log.path, tmp_path / "cal.json", run=RUN, settings_path=path)
    assert "error" in line["flags"]["sev_severe"] and "t" not in line["flags"]["sev_severe"]
    assert loop.load_status(log.path)["reports"][-1]["flags"] == line["flags"] and path.read_text(encoding="utf-8") == "{"
