"""d1a.api: the System One request schema, the text the model reads for a request, and the answers built from the
model's distributions. The text formats are pinned exactly: the trained checkpoints read them, so any change is a change
to every answer."""
import math

import pytest
from pydantic import ValidationError

from d1a.api import (MAX_OPTIONS, SystemOneRequest, choice_confidence, date_facts, option_text, question_keys, render, round_prob,
                     score_confidence, to_answers, to_record, with_date_facts)


def req(questions, state="s"):
    return SystemOneRequest.model_validate({"state": state, "questions": questions})


# --- text the model reads -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, text", [
    ("plain", "plain"), (None, ""), (3, "3"), (2.5, "2.5"), (True, "True"),
    (["x", "y"], "- x\n- y"),
    ({"what": "A", "not_for": "B"}, "what: A\nnot_for: B"),
    ({"note": None}, "note: "),                                                   # a null field keeps its label
    ({"ticket": {"channel": "email", "body": "hi"}}, "ticket:\n  channel: email\n  body: hi"),
    ({"examples": ["a", "b"]}, "examples:\n  - a\n  - b"),
    ([{"a": 1, "b": 2}, ["c"]], "- a: 1\n  b: 2\n- - c"),                          # a list item's first line loses its indent
    ({"a": {"b": {"c": [1]}}}, "a:\n  b:\n    c:\n      - 1"),
])
def test_render_formats(value, text):
    assert render(value) == text


def test_option_text():
    assert option_text("calm", None) == option_text("calm", "") == "calm"
    assert option_text("angry", "Hostile") == "angry: Hostile"
    assert option_text("policy", {"what": "A"}) == "policy: what: A"


def test_to_record_lays_out_every_question_type():
    rec, meta = to_record(req({
        "billing": {"type": "noul", "instructions": "About billing?", "criteria": {"true": "Charges", "false": "Not charges"}},
        "plain": {"type": "noul", "instructions": {"ask": "Urgent?"}},
        "tone": {"type": "choice", "instructions": "Tone?", "criteria": {"calm": None, "angry": "Hostile"}},
        "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["can wait", {"when": "today"}]},
    }, state={"document": "I was charged twice."}))
    assert rec["state"] == "document: I was charged twice."
    assert rec["questions"] == [
        {"instr": "About billing?", "options": ["no: Not charges", "yes: Charges"], "label": 0},   # noul: "no" first, as "false" is
        {"instr": "ask: Urgent?", "options": ["no", "yes"], "label": 0},
        {"instr": "Tone?", "options": ["calm", "angry: Hostile"], "label": 0},
        {"instr": "Urgency?", "options": ["can wait", "when: today"], "label": 0}]
    assert meta == [{"id": "billing", "type": "noul", "keys": ["false", "true"]}, {"id": "plain", "type": "noul", "keys": ["false", "true"]},
                    {"id": "tone", "type": "choice", "keys": ["calm", "angry"]},
                    {"id": "urgency", "type": "score", "keys": ["0", "1"], "legend": {"0": "can wait", "1": "when: today"}}]
    assert question_keys("choice", {"b": 1, "a": 2}) == ["b", "a"] and question_keys("score", ["x", "y", "z"]) == ["0", "1", "2"]


# --- answers ------------------------------------------------------------------------------------------------------------

def test_answers_per_type():
    _, meta = to_record(req({"n": {"type": "noul"}, "c": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}},
                             "s": {"type": "score", "criteria": ["lo", "mid", "hi"]}}))
    answers = to_answers([[0.3, 0.7], [0.15, 0.8, 0.05], [0.1, 0.3, 0.6]], meta)
    assert answers["n"] == {"type": "noul", "noul": 0.7}                         # P(true), the second option
    assert answers["c"] == {"type": "choice", "choice": "b", "confidence": round((0.8 - 1 / 3) / (2 / 3), 4),
                            "probabilities": {"a": 0.15, "b": 0.8, "c": 0.05}}
    # expected level 0.3 + 1.2 = 1.5; mode 2, E|level - 2| = 0.2 + 0.3 = 0.5 against 2/3 for uniform: 1 - 0.75
    assert answers["s"] == {"type": "score", "score": 1.5, "legend": {"0": "lo", "1": "mid", "2": "hi"},
                            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.25}
    assert to_answers([[0.4, 0.4, 0.2]], [{"id": "t", "type": "choice", "keys": ["x", "y", "z"]}])["t"]["choice"] == "x"   # a tie: the first


@pytest.mark.parametrize("p", [[0.79] + [0.21 / 39] * 39, [1 / MAX_OPTIONS] * MAX_OPTIONS])
def test_rounded_distributions_stay_within_typesafes_tolerance(p):
    served = to_answers([p], [{"id": "t", "type": "choice", "keys": [f"o{i}" for i in range(len(p))]}])["t"]["probabilities"]
    assert len(served) == len(p) and abs(sum(served.values()) - 1) < 0.02
    assert round_prob(1 / 3) == 0.3333


def test_confidence_formulas():
    for confidence in (choice_confidence, score_confidence):
        assert confidence([1.0]) == 1.0                                          # one option: nowhere else to be
        assert confidence([0.0, 0.0, 0.0]) == 0.0                                # all zeros read as uniform
        assert all(math.isclose(confidence([1 / k] * k), 0.0, abs_tol=1e-12) for k in range(2, 12))
    assert math.isclose(choice_confidence([0.0, 1.0, 0.0]), 1.0) and math.isclose(choice_confidence([0.6, 0.4]), 0.2)
    assert score_confidence([0.0, 0.0, 1.0]) == 1.0
    assert score_confidence([0.5, 0.0, 0.5]) == 0.0                              # wider than uniform floors at 0
    assert score_confidence([3.0, 9.0, 0.0]) == score_confidence([0.25, 0.75, 0.0])   # read normalised


@pytest.mark.parametrize("p, shown", [
    ([0.0, 0.57, 0.43], 0.35), ([0.0, 0.14, 0.86, 0.0, 0.0], 0.89), ([0.0, 0.0, 0.48, 0.52], 0.52),
    ([0.0, 0.74, 0.26], 0.61), ([0.0, 0.0, 0.0, 1.0], 1.0)])
def test_score_confidence_on_typesafes_examples(p, shown):
    """docs.typesafe.ai/primitives/score.md shows these at two decimals, probabilities included, so equal within that."""
    assert abs(round(score_confidence(p), 2) - shown) < 0.011


# --- dates --------------------------------------------------------------------------------------------------------------

def test_date_facts():
    text = "Due July 4, 2026. Received June 26, 2026. Shipped 2026-07-01. Due July 4, 2026 again; not a day: 2026-02-30."
    assert date_facts(text) == ("June 26, 2026 is 8 days before July 4, 2026. 2026-07-01 is 3 days before July 4, 2026. "
                                "2026-07-01 is 5 days after June 26, 2026.")
    assert date_facts("May 1, 2026 and 2026-05-02") == "2026-05-02 is 1 day after May 1, 2026."
    assert date_facts("2026-05-01 or May 1, 2026") == "May 1, 2026 is the same day as 2026-05-01."
    assert date_facts("only May 1, 2026") == ""


def test_with_date_facts_per_state_shape():
    two = "from May 1, 2026 to May 3, 2026"
    facts = "May 3, 2026 is 2 days after May 1, 2026."
    assert with_date_facts(two) == f"{two}\n\ndate_facts: {facts}"
    assert with_date_facts({"case": two}) == {"case": two, "date_facts": facts}
    assert with_date_facts([two]) == [two, {"date_facts": facts}]
    assert with_date_facts({"case": "one date: May 1, 2026"}) == {"case": "one date: May 1, 2026"}


# --- schema -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("questions", [
    {}, {"q": {"type": "bogus"}}, {"q": {"type": "choice"}}, {"q": {"type": "choice", "criteria": {}}},
    {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(MAX_OPTIONS + 1)}}},
    {"q": {"type": "score", "criteria": []}}, {"q": {"type": "score", "criteria": list(range(MAX_OPTIONS + 1))}},
])
def test_invalid_requests_are_refused(questions):
    with pytest.raises(ValidationError):
        req(questions)


def test_valid_limits_and_defaults():
    r = req({"c": {"type": "choice", "criteria": {f"o{i}": None for i in range(MAX_OPTIONS)}}, "s": {"type": "score", "criteria": [1]}})
    assert r.model == "d1a-latest" and r.questions["s"].instructions is None
    with pytest.raises(ValidationError, match="criteria must have 1..255 options"):
        req({"q": {"type": "choice", "criteria": {}}})
