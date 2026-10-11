"""d1a.learning.rules (#252): a decode rule changes only the served choice, only for its run and use case, only when the
model's choice is one of `from` and its raw probability of the option reaches t; the probabilities, the confidence and the
log's answers stay the model's, and the log records what was served. Settings refuse a rule that could not hold."""
import json
from types import SimpleNamespace

import pytest

from d1a.learning import settings as S

RUN = "JohnP1/d1a-e4b-mlx-q8@v0.6"
RULE = {"question": "sev", "option": "P2", "t": 0.20, "from": ["P3", "P4"], "space": "raw", "run": RUN, "use_case": "pr-labeler",
        "fitted_on": {"kit": "silver-run1", "n": 62, "tested_on": "silver-run2"}}
SEV = {"sev": {"type": "choice", "instructions": "How serious.", "criteria": {k: k for k in ("P0", "P1", "P2", "P3", "P4")}},
       "ok": {"type": "noul", "instructions": "Fine?", "criteria": {"true": "yes", "false": "no"}}}


def test_a_rule_changes_only_the_served_choice_for_its_run_and_use_case(tmp_path, monkeypatch):
    from d1a.serving import serve
    from d1a.core.api import SystemOneRequest, to_record
    monkeypatch.setattr(serve, "output_tokens", lambda tok, answers: 0)
    monkeypatch.setattr(serve, "FLAGS", SimpleNamespace(current=lambda: {}))
    monkeypatch.setattr(serve, "RULES", SimpleNamespace(current=lambda: {"sev_p2": RULE}))
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(tmp_path / "log.jsonl"))

    def answer(sev, use_case="pr-labeler", run=RUN):
        req = SystemOneRequest(state="pr", questions=SEV, use_case=use_case); _, meta = to_record(req)
        fake = SimpleNamespace(checkpoint=SimpleNamespace(requested=run), tok=None)
        return serve.Server._body(fake, req, meta, [sev, [0.3, 0.7]], {"tokens": 10, "latency_ms": 1.0})["answers"]

    model = answer([0.01, 0.10, 0.2213, 0.5687, 0.10], use_case=None)
    served = answer([0.01, 0.10, 0.2213, 0.5687, 0.10])
    assert model["sev"]["choice"] == "P3" and served["sev"]["choice"] == "P2"
    assert served["sev"]["rule"] == {"name": "sev_p2", "from": "P3", "p": 0.2213, "t": 0.2} and "rule" not in model["sev"]
    assert {k: v for k, v in served["sev"].items() if k not in ("choice", "rule")} == {k: v for k, v in model["sev"].items() if k != "choice"}
    assert served["ok"] == model["ok"]
    logged = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    assert logged["answers"]["sev"]["choice"] == "P3" and logged["meta"]["served"]["sev"]["choice"] == "P2"
    assert logged["meta"]["rules"] == {"sev": {"sev_p2": {"from": "P3", "to": "P2", "p": 0.2213, "t": 0.2}}}
    for sev, kw in (([0.01, 0.10, 0.19, 0.60, 0.10], {}),                         # below t
                    ([0.05, 0.50, 0.30, 0.10, 0.05], {}),                         # the model said P1: never lowered to P2
                    ([0.01, 0.10, 0.2213, 0.5687, 0.10], {"run": "JohnP1/d1a-e2b-mlx-q8@v0.6"}),
                    ([0.01, 0.10, 0.2213, 0.5687, 0.10], {"use_case": "routing"}),
                    ([0.01, 0.10, 0.2213, 0.5687, 0.10], {"use_case": None})):
        got = answer(sev, **kw)["sev"]
        assert got["choice"] == max(zip(sev, SEV["sev"]["criteria"]))[1] and "rule" not in got
    assert "rules" not in [json.loads(l) for l in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()][-1]["meta"]


def test_settings_refuse_a_rule_that_could_not_hold(tmp_path):
    assert S.validate({})["rules"] == {} and S.validate({"rules": {"sev_p2": RULE}})["rules"]["sev_p2"] == RULE
    for bad in ({**RULE, "space": "calibrated"}, {**RULE, "t": 1.5}, {k: v for k, v in RULE.items() if k != "run"},
                {**RULE, "run": "JohnP1/d1a-e4b-mlx-q8"}, {**RULE, "from": ["P2", "P3"]}, {**RULE, "use_case": ""}, {**RULE, "extra": 1},
                {**RULE, "fitted_on": {"kit": "silver-run1", "n": 62}}):
        with pytest.raises(S.SettingsError): S.validate({"rules": {"sev_p2": bad}})
    with pytest.raises(S.SettingsError, match="one rule per question"):
        S.validate({"rules": {"a": RULE, "b": {**RULE, "t": 0.3}}})
    assert S.validate({"rules": {"a": RULE, "b": {**RULE, "use_case": "other"}}})
    path = tmp_path / "learning.json"; S.init(path)
    assert S.set_values(path, {"rules.sev_p2": RULE}) == [("rules.sev_p2", None, RULE)]
    S.set_values(path, {"rules.sev_p2": None}); assert S.load(path)[0]["rules"] == {}
