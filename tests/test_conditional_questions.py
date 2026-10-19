"""Conditional questions (#36): ask_if asks a question only when earlier answers allow it, in one request. Without ask_if a
request is answered in one pass, as before; ask_if is never model input."""
import pytest

from d1a.core.api import SystemOneRequest, answer_in_passes, ask_order, conditional, in_request_order, to_answers, to_record

INTENT = {"type": "choice", "instructions": "What does the customer want?", "criteria": {"refund": None, "track": None}}


def req(**questions):
    return SystemOneRequest(state="Order 4471 was charged twice.", questions=questions)


def test_ask_if_is_validated_and_ordered():
    r = req(intent=INTENT, amount={"type": "score", "criteria": ["0", "1", "2"], "ask_if": {"intent": ["refund"]}},
            urgent={"type": "noul", "ask_if": {"amount": {"min": 1}}}, tone={"type": "noul"})
    assert conditional(r) and ask_order(r) == [["intent", "tone"], ["amount"], ["urgent"]]
    assert not conditional(req(intent=INTENT))
    bad = [({"a": {"type": "noul", "ask_if": {"b": True}}}, "not in this request"),
           ({"a": {"type": "noul", "ask_if": {"a": True}}}, "itself"),
           ({"a": {"type": "noul", "ask_if": {"b": True}}, "b": {"type": "noul", "ask_if": {"a": False}}}, "cycle"),
           ({"i": INTENT, "b": {"type": "noul", "ask_if": {"i": ["cancel"]}}}, "option keys"),
           ({"i": INTENT, "b": {"type": "noul", "ask_if": {"i": True}}}, "option keys"),
           ({"n": {"type": "noul"}, "b": {"type": "noul", "ask_if": {"n": ["yes"]}}}, "true or false")]
    for questions, why in bad:
        with pytest.raises(ValueError, match=why): req(**questions)


def test_ask_if_is_never_model_input():
    """The record the model reads is the same with and without ask_if, so no trained answer can move."""
    plain = req(intent=INTENT, urgent={"type": "noul", "instructions": "Urgent?"})
    gated = req(intent=INTENT, urgent={"type": "noul", "instructions": "Urgent?", "ask_if": {"intent": ["refund"]}})
    assert to_record(plain) == to_record(gated)


def test_passes_ask_only_what_the_answers_allow():
    """Each pass asks the questions whose conditions hold on the answers so far; a failed condition, or a skipped question
    it names, skips the question; the answers keep the request's order."""
    r = req(intent=INTENT, amount={"type": "score", "criteria": ["0", "1", "2"], "ask_if": {"intent": ["refund"]}},
            reason={"type": "noul", "ask_if": {"intent": ["track"]}}, escalate={"type": "noul", "ask_if": {"reason": True}},
            big={"type": "noul", "ask_if": {"amount": {"min": 1.5}}})
    want = {"intent": [0.8, 0.2], "amount": [0.1, 0.2, 0.7], "big": [0.3, 0.7]}   # refund; amount score 1.6
    calls = []

    def run(sub):
        calls.append(list(sub.questions)); rec, meta = to_record(sub)
        return meta, [want[m["id"]] for m in meta], {"tokens": 1}
    meta, ps, stats = answer_in_passes(r, run)
    answers = in_request_order(r, to_answers(ps, meta))
    assert calls == [["intent"], ["amount"], ["big"]] and len(stats) == 3
    assert list(answers) == ["intent", "amount", "reason", "escalate", "big"]
    assert answers["intent"]["choice"] == "refund" and answers["big"]["noul"] == 0.7
    assert answers["reason"] == {"type": "noul", "skipped": True, "ask_if": {"intent": ["track"]}}
    assert answers["escalate"] == {"type": "noul", "skipped": True, "ask_if": {"reason": True}}


def test_server_and_library_answer_conditional_requests_alike(train_hybrid, tmp_path):
    """The server and d1a.D1A give the same answers to a conditional request: the condition on the unconditional answer is
    asked, the other one and the question that depends on it are skipped; the server logs one decision and sums usage."""
    import d1a
    from d1a.serving import serve
    train_hybrid(tmp_path / "ck", "--lora", "4", "--max_steps", "2")
    m = d1a.D1A.load(str(tmp_path / "ck"), device="cpu")
    state = "the customer is charged twice"
    team = {"type": "choice", "instructions": "which team", "criteria": {"billing": "billing", "shipping": "shipping"}}
    first = m.decide(state, {"team": team})["team"]["choice"]
    other = "shipping" if first == "billing" else "billing"
    questions = {"team": team,
                 "angry": {"type": "noul", "instructions": "is the customer angry", "ask_if": {"team": [first]}},
                 "late": {"type": "noul", "instructions": "is the parcel late", "ask_if": {"team": [other]}},
                 "refund": {"type": "noul", "instructions": "should we refund", "ask_if": {"late": True}}}
    got = m.decide(state, questions)
    assert list(got) == ["team", "angry", "late", "refund"]
    assert "noul" in got["angry"] and got["late"]["skipped"] and got["refund"]["skipped"]
    server = serve.Server(m.checkpoint, m.tok, m.model, "cpu")
    learning = serve.Learning(tmp_path / "log.jsonl")
    try:
        real = serve.LEARNING; serve.LEARNING = learning
        body = server.answer(serve.SystemOneRequest(state=state, questions=questions))
        single = server.answer(serve.SystemOneRequest(state=state, questions={"team": team}), log=False)
    finally:
        serve.LEARNING = real; server.close()
    assert body["answers"] == got
    assert body["usage"]["input_tokens"] > single["usage"]["input_tokens"]   # two passes
    decisions = [e for e in learning.log.events() if e["kind"] == "decision"]
    assert len(decisions) == 1 and set(decisions[0]["answers"]) == {"team", "angry"}
