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
