"""The self-learning loop (#233): settings, replay across model versions (fail closed), the data-triggered gate."""
import json
import time

import pytest

from d1a.learning import loop, settings as S
from d1a.learning.feedback import FeedbackLog, OutcomeCalibrator, main

OLD, NEW = "JohnP1/d1a-e4b-mlx-q8@v0.5", "JohnP1/d1a-e4b-mlx-q8@v0.6"
Q = {"type": {"type": "choice", "instr": "Primary change type", "criteria": {"bug": None, "docs": None}}}


def answer(p_bug):
    return {"type": {"type": "choice", "choice": "bug" if p_bug >= 0.5 else "docs", "probabilities": {"bug": p_bug, "docs": 1 - p_bug}}}


def decide(log, state, run, group, label="bug", p=0.7, src="reviewer"):
    did = log.decision(state, Q, answer(p), run)
    log.outcome(did, {"type": label}, {"src": src, **({"group": group} if group else {})})
    return did


def manifest(tmp_path, run=NEW, prs=(), states=()):
    path = tmp_path / "trained.json"
    data = {"run": run, "mixes": [{"mix": "train.jsonl"}], "pr_ids": sorted(loop.pr_key(p) for p in prs),
            "state_sha256": sorted(loop.state_sha256(s) for s in states)}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class FakeServer:
    """The live server as replay() sees it: /v1/models (run, queued batches) and /v1/systemone with the replay header."""

    def __init__(self, log, run=NEW, queued=0):
        self.log, self.run, self.queued, self.sent = log, run, queued, []

    def __call__(self, url, body=None, headers=None, timeout=60):
        if url.endswith("/v1/models"): return {"models": [{"run": self.run, "batches": {"queued": self.queued}}]}
        self.sent.append(headers[loop.REPLAY_HEADER])
        self.log.decision(body["state"], body["questions"], answer(0.6), self.run, replay_of=headers[loop.REPLAY_HEADER])
        return {"answers": answer(0.6)}


@pytest.fixture
def server(monkeypatch, tmp_path):
    log = FeedbackLog(tmp_path / "log.jsonl")
    fake = FakeServer(log)
    monkeypatch.setattr(loop, "_http", fake)
    return log, fake


def test_settings_reject_values_outside_the_agreed_ranges_and_unknown_keys():
    S.validate({})   # every default is valid
    bad = [{"promotion": {"min_outcomes": 9}}, {"promotion": {"held_out": 0.05}}, {"promotion": {"held_out": 0.6}},
           {"promotion": {"ci_level": 0.8}}, {"schedule": {"after_new_outcomes": 0}}, {"schedule": {"daily_at": "4:00"}},
           {"schedule": {"daily_at": "24:00"}}, {"promotion": {"min_outcome": 20}}, {"flags": {}}, {"promotion": {"auto": "yes"}},
           {"outcomes": {"sources": []}}, {"reports": {"notify": ["sometimes"]}}, {"version": 2}]
    for data in bad:
        with pytest.raises(S.SettingsError): S.validate(data)
    ok = S.validate({"promotion": {"min_outcomes": 10, "held_out": 0.5, "ci_level": 0.99, "questions": ["type"]},
                     "schedule": {"daily_at": "23:59", "after_new_outcomes": None}})
    assert ok["promotion"]["held_out"] == 0.5 and ok["replay"] == S.DEFAULTS["replay"]


def test_a_broken_file_falls_back_to_the_last_valid_one_and_set_refuses_without_writing(tmp_path):
    path = tmp_path / "learning.json"
    S.init(path)
    S.set_values(path, {"promotion.min_outcomes": 12})
    with pytest.raises(S.SettingsError): S.set_values(path, {"promotion.min_outcomes": 3, "schedule.daily_at": "05:00"})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["promotion"]["min_outcomes"] == 12 and saved["schedule"] == S.DEFAULTS["schedule"]
    path.write_text('{"promotion": {"min_outcomes": 3}}', encoding="utf-8")   # edited by hand, wrongly
    settings, error, source = S.load(path)
    assert settings["promotion"]["min_outcomes"] == 12 and "min_outcomes" in error and "last-valid" in source
    assert [(a["key"], a["new"]) for a in S.audit(path)][-1] == ("promotion.min_outcomes", 12)


def test_a_changed_setting_takes_effect_on_the_next_run_without_a_restart(tmp_path, monkeypatch):
    path = tmp_path / "learning.json"; monkeypatch.setenv(S.ENV, str(path))
    assert S.load()[0]["promotion"]["min_outcomes"] == 20   # no file yet: today's behaviour
    S.init(path); S.set_values(path, {"promotion.min_outcomes": 15}, "page")
    assert S.load()[0]["promotion"]["min_outcomes"] == 15


def test_replay_never_rescores_a_training_pr_and_fails_closed_without_the_manifest(tmp_path, server):
    """The sabotage test of #233: a training PR in the log must not be replayed, by its PR id or by its state alone; a
    decision with no PR id is left out by default; no manifest, or one for another model, replays nothing."""
    log, fake = server
    keep = decide(log, "owner PR, never trained", OLD, "jonpol01/d1a#232")
    decide(log, "a hermes-agent PR it trained on", OLD, "NousResearch/hermes-agent#40604")
    decide(log, "trained state, no group", OLD, None)
    decide(log, "trained state, other group", OLD, "jonpol01/d1a#999")
    decide(log, "no group at all", OLD, None)
    decide(log, "made by the served model itself", NEW, "jonpol01/d1a#300")
    m = manifest(tmp_path, prs=["hermes-agent#40604:ja"], states=["trained state, no group", "trained state, other group"])
    for bad in (None, tmp_path / "missing.json", manifest(tmp_path, run=OLD)):
        with pytest.raises(loop.ReplayRefused): loop.replay(log, "http://s", bad, sleep=lambda s: None)
        assert fake.sent == []
    m = manifest(tmp_path, prs=["hermes-agent#40604:ja"], states=["trained state, no group", "trained state, other group"])
    rep = loop.replay(log, "http://s", m, sleep=lambda s: None)
    assert fake.sent == [keep]
    assert rep["excluded"] == {"its pull request is in the served model's training": 1, "its state is in the served model's training": 2,
                               "no pull-request id to check against training": 1}
    assert loop.replay(log, "http://s", m, sleep=lambda s: None)["replayed"] == 0   # once per model


def test_a_replay_takes_its_originals_outcomes_and_is_never_pending(tmp_path, server):
    log, fake = server
    original = decide(log, "owner PR", OLD, "jonpol01/d1a#232", label="docs")
    log.decision("waiting", Q, answer(0.5), OLD)
    loop.replay(log, "http://s", manifest(tmp_path), sleep=lambda s: None)
    (r,) = log.resolved(run=NEW)
    assert r["replay_of"] == original and r["replayed_from"] == OLD and r["labels"] == {"type": "docs"} and r["group"] == "jonpol01/d1a#232"
    assert [e["state"] for e in log.pending()] == ["waiting"]


def test_replay_waits_for_live_traffic_and_keeps_to_its_cap(tmp_path, server):
    log, fake = server
    for i in range(5): decide(log, f"owner PR {i}", OLD, f"jonpol01/d1a#{i}")
    fake.queued = 1
    assert loop.replay(log, "http://s", manifest(tmp_path), sleep=lambda s: None)["paused_for_live_traffic"] and fake.sent == []
    fake.queued = 0
    rep = loop.replay(log, "http://s", manifest(tmp_path), cap=2, sleep=lambda s: None)
    assert rep["replayed"] == 2 and rep["left"] == 3


def test_the_gate_triggers_after_n_new_outcomes_or_at_the_daily_time():
    st = S.validate({"schedule": {"daily_at": None, "after_new_outcomes": 10}})
    assert loop.due(st, {"outcomes_at_last_gate": 5, "last_gate": 1}, 14, time.time()) is None
    assert "10 new outcomes" in loop.due(st, {"outcomes_at_last_gate": 5, "last_gate": 1}, 15, time.time())
    daily = S.validate({"schedule": {"daily_at": "04:00", "after_new_outcomes": None}})
    at = lambda h, m: time.mktime(time.strptime(f"2026-10-11 {h:02d}:{m:02d}", "%Y-%m-%d %H:%M"))
    assert loop.due(daily, {"last_gate": at(3, 0)}, 0, at(3, 59)) is None
    assert loop.due(daily, {"last_gate": at(3, 0)}, 0, at(4, 1)) == "daily at 04:00"
    assert loop.due(daily, {"last_gate": at(4, 1)}, 0, at(23, 0)) is None


def test_tick_replays_on_a_model_change_reports_power_and_with_auto_off_never_writes(tmp_path, server, monkeypatch):
    log, fake = server
    path = tmp_path / "learning.json"; monkeypatch.setenv(S.ENV, str(path)); S.init(path)
    S.set_values(path, {"promotion.auto": False, "promotion.min_outcomes": 10, "schedule.daily_at": None,
                        "schedule.after_new_outcomes": None, "replay.pause_s": 0})   # no trigger: only the first replay reports
    for i in range(40): decide(log, f"owner PR {i}", OLD, f"jonpol01/d1a#{i}", label="bug" if i % 2 else "docs", p=0.9)
    trained = tmp_path / "trained"; trained.mkdir()
    manifest(tmp_path).rename(loop.manifest_path(trained, NEW))
    cal = tmp_path / "calibrator.json"
    line = loop.tick(log.path, cal, server="http://s", trained_dir=trained)
    assert line["model_change"] == {"from": None, "to": NEW} and line["replay"]["replayed"] == 40
    gate = line["gate"]
    assert gate["reason"] == "the first replay after a model change" and gate["replayed"] == 40 and not gate["calibrator_written"]
    assert "detectable" in gate["power"] and not cal.exists()   # auto off: gated and reported, the calibrator untouched
    status = json.loads(loop.status_path(log.path).read_text(encoding="utf-8"))
    assert status["per_question"]["type"]["outcomes"] == 40 and status["reports"][-1]["gate"]["reason"]
    again = loop.tick(log.path, cal, server="http://s", trained_dir=trained)
    assert "gate" not in again and "replay" not in again   # nothing new: no replay, no gate


def test_the_promote_command_takes_its_defaults_from_the_settings_file(tmp_path, capsys, monkeypatch):
    log = FeedbackLog(tmp_path / "log.jsonl")
    for i in range(30): decide(log, f"PR {i}", NEW, f"o/r#{i}", label="bug" if i % 3 else "docs", p=0.95)
    path = tmp_path / "learning.json"; monkeypatch.setenv(S.ENV, str(path)); S.init(path)
    S.set_values(path, {"promotion.auto": False, "promotion.questions": ["sev"]})
    main(["promote", str(log.path), "--calibrator", str(tmp_path / "cal.json"), "--run", NEW])
    out = json.loads(capsys.readouterr().out)
    assert out["report"] == {} and not (tmp_path / "cal.json").exists()   # "type" does not learn; nothing written


def test_the_server_logs_a_replay_with_its_original_and_rejects_a_malformed_id(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from contextlib import asynccontextmanager, nullcontext
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from d1a.serving import serve

    def submit(rec, media=None, use_case=None):
        f = Future(); f.set_result(([[0.2, 0.8]], {"tokens": 10, "latency_ms": 1.0})); return f
    fake = SimpleNamespace(checkpoint=SimpleNamespace(requested=NEW), tok=None, submit=submit, probs=lambda rec, use_case=None: submit(rec).result())
    for name in ("answer", "answer_async", "_body"): setattr(fake, name, getattr(serve.Server, name).__get__(fake))
    monkeypatch.setattr(serve, "server", lambda: nullcontext(fake))
    @asynccontextmanager
    async def server_async():
        yield fake
    monkeypatch.setattr(serve, "server_async", server_async)
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl"))
    req = {"state": "PR", "questions": {"resolved": {"type": "noul", "instructions": "Fixed?"}}}
    original = "a" * 32
    with TestClient(serve.app) as client:
        assert client.post("/v1/systemone", json=req, headers={"x-d1a-replay-of": original}).status_code == 200
        assert client.post("/v1/systemone", json=req, headers={"x-d1a-replay-of": "../etc"}).status_code == 422
        client.post("/v1/systemone/separate", json=req, headers={"x-d1a-replay-of": original})   # replays only via /v1/systemone
    events = serve.LEARNING.log.events()
    assert [e.get("replay_of") for e in events] == [original, None] and events[0]["run"] == NEW
