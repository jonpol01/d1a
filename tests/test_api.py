"""The System One HTTP API as a TypeSafe client sees it (POST /v1/systemone, GET /v1/models): each question type's answer
shape, the requests refused with 422, branch isolation, the model names and their cards, the request id on every
response, and the TypeSafe SDK's sync and async clients.

By default the tests start d1a.serving.serve's app in this process on a free local port, serving the committed tiny
checkpoint (tests/golden/tiny-gemma4) on the CPU, so they run in the unit suite in a few seconds. With D1A_BASE_URL set
they check that server instead:  D1A_BASE_URL=http://127.0.0.1:8008 uv run python -m pytest tests/test_api.py -m server
The tiny model's answers are random, so nothing here depends on what a question is answered, only on how;
tests/test_conformance.py pins the tiny model's exact probabilities."""
import asyncio
import math
import os
import threading
import time
from pathlib import Path

import httpx
import pytest
import torch

pytestmark = pytest.mark.server   # `-m server` still selects exactly these tests

ROOT = Path(__file__).resolve().parents[1]
TINY = Path("tests/golden/tiny-gemma4/checkpoint")   # relative to ROOT: the checkpoint names its base by this path
EXTERNAL = os.environ.get("D1A_BASE_URL")
REQUEST_ID = "x-typesafe-request-id"                 # what every TypeSafe client reports as response.request_id
SUM_TOLERANCE = 0.02                                 # TypeSafe's own bound on |sum - 1| (d1a.core.api.round_prob)
# Branch isolation: the tiny model runs in fp32, so a question answers the same with or without siblings up to the 4
# decimals answers are served at. A GPU server batches in bf16, where a different batch shape moves answers by ~0.01.
ISOLATION_TOLERANCE = 0.011 if EXTERNAL else 2e-4
MODEL_NAMES = {"d1a-latest"}                               # the only name /v1/models lists
ANSWERED_NAMES = ["d1a-latest", "jev-latest", "any-other-name"]   # any name is answered; jev-latest is the SDK's default

TEAM = {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems"}
URGENCY = ["can wait", "this week", "today"]


def tiny_server():
    """d1a.serving.serve's app with the tiny checkpoint loaded as `serve.main` loads one (self-check included), on uvicorn
    in a background thread. -> its base URL; stopped and unloaded when the generator closes."""
    import uvicorn
    from d1a.backends.checkpoint import Checkpoint, LoadOptions
    from d1a.serving import serve
    from d1a.serving.media import OnDemand

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        checkpoint = Checkpoint(str(TINY))
        tok, model = checkpoint.load("cpu", LoadOptions(dtype=torch.float32, backend="torch"))
        loaded = serve.Server(checkpoint, tok, model, "cpu")
    try:
        serve.self_check(loaded.probs)
        serve.app.state.models, serve.app.state.card = OnDemand(lambda: loaded, 0, "cpu"), serve.card(loaded)
        web = uvicorn.Server(uvicorn.Config(serve.app, host="127.0.0.1", port=0, log_level="warning"))
        thread = threading.Thread(target=web.run, name="test-api-server", daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while not web.started:
            assert thread.is_alive() and time.monotonic() < deadline, "the test server did not start"
            time.sleep(0.01)
        try:
            yield "http://127.0.0.1:%d" % web.servers[0].sockets[0].getsockname()[1]
        finally:
            web.should_exit = True
            thread.join(timeout=30)
            del serve.app.state.models, serve.app.state.card
    finally:
        loaded.close()


@pytest.fixture(scope="module")
def base_url():
    if EXTERNAL:
        yield EXTERNAL.rstrip("/")
    else:
        yield from tiny_server()


@pytest.fixture(scope="module")
def api(base_url):
    with httpx.Client(base_url=base_url, timeout=120) as client:
        yield client


def ask(api, state, questions, model="d1a-latest", status=200):
    """POST /v1/systemone; checks the status and the request id, which every response must carry. -> the body."""
    r = api.post("/v1/systemone", json={"state": state, "model": model, "questions": questions})
    assert r.status_code == status, r.text
    assert r.headers.get(REQUEST_ID)
    return r.json()


def check_distribution(probabilities, keys):
    assert list(probabilities) == list(keys)   # one probability per option, in the request's order
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1) <= SUM_TOLERANCE


def check_choice(answer, criteria):
    assert set(answer) == {"type", "choice", "confidence", "probabilities"} and answer["type"] == "choice"
    p = answer["probabilities"]
    check_distribution(p, criteria)
    assert answer["choice"] == max(p, key=p.get)   # the most likely option (the first, on a tie)
    k = len(p)   # TypeSafe's choice confidence: how far the top option is above uniform, 0 at uniform and 1 at certainty
    assert math.isclose(answer["confidence"], 1.0 if k == 1 else (max(p.values()) - 1 / k) / (1 - 1 / k), abs_tol=1e-3)


def check_score(answer, levels):
    assert set(answer) == {"type", "score", "legend", "probabilities", "confidence"} and answer["type"] == "score"
    keys = [str(i) for i in range(len(levels))]
    assert answer["legend"] == dict(zip(keys, levels))
    check_distribution(answer["probabilities"], keys)
    assert math.isclose(answer["score"], sum(i * p for i, p in enumerate(answer["probabilities"].values())), abs_tol=1e-3)
    assert 0 <= answer["score"] <= len(levels) - 1 and 0 <= answer["confidence"] <= 1


def check_noul(answer):
    assert set(answer) == {"type", "noul"} and answer["type"] == "noul" and 0 <= answer["noul"] <= 1


# --- answer shapes ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ANSWERED_NAMES)
def test_a_choice_is_answered_with_a_distribution_over_its_criteria(api, model):
    """Any model name answers, and the response names the model the request asked for."""
    body = ask(api, "The parcel was marked delivered but never arrived.",
               {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": TEAM}}, model=model)
    assert body["model"] == model and set(body["answers"]) == {"team"}
    check_choice(body["answers"]["team"], TEAM)
    assert set(body["usage"]) == {"input_tokens", "output_tokens"} and min(body["usage"].values()) > 0


def test_five_questions_of_every_type_answer_in_one_request_and_criteria_may_be_null(api):
    mood = {"calm": None, "annoyed": None, "furious": None}
    questions = {
        "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": TEAM},
        "mood": {"type": "choice", "instructions": "How does the customer sound?", "criteria": mood},
        "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
        "urgency": {"type": "score", "instructions": "How soon must we reply?", "criteria": URGENCY},
        "item": {"type": "choice", "instructions": "What was ordered?", "criteria": {"shoes": "", "jacket": None, "bag": "A bag or backpack"}},
    }
    body = ask(api, "My jacket came a week late, the wrong colour, and I was charged twice. Refund me.", questions)
    assert list(body["answers"]) == list(questions)
    check_choice(body["answers"]["team"], TEAM)
    check_choice(body["answers"]["mood"], mood)
    check_noul(body["answers"]["refund"])
    check_score(body["answers"]["urgency"], URGENCY)
    check_choice(body["answers"]["item"], ["shoes", "jacket", "bag"])


def test_instructions_and_criteria_may_be_structured_json(api):
    criteria = {"policy": {"what": "Whether an item can be returned", "not_for": "A return already on its way",
                           "examples": ["Can I return worn shoes?", "How many days do I have?"]},
                "status": {"what": "A return already on its way", "examples": ["Did my parcel reach you?"], "priority": 2}}
    body = ask(api, ["Order 1182", {"message": "I posted the shoes back on Monday.", "attachments": None}],
               {"topic": {"type": "choice", "instructions": {"question": "Which returns topic is this?", "hints": ["read the whole message"]},
                          "criteria": criteria}})
    check_choice(body["answers"]["topic"], criteria)


def test_a_noul_and_a_score_about_an_object_state(api):
    body = ask(api, {"ticket": {"subject": "Double charge", "body": "Two charges for one order. Fix it today."}, "tier": "gold"},
               {"billing": {"type": "noul", "instructions": "Is this about billing?", "criteria": {"true": "About charges", "false": "Anything else"}},
                "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": URGENCY}})
    check_noul(body["answers"]["billing"])
    check_score(body["answers"]["urgency"], URGENCY)


def test_instructions_may_be_left_out_and_a_score_may_have_one_level(api):
    """The SDK leaves `instructions` out when none is given; a single level is certain (TypeSafe's confidence rule)."""
    body = ask(api, "Two charges for one order.",
               {"billing": {"type": "noul", "criteria": {"true": "About charges", "false": "Anything else"}},
                "team": {"type": "choice", "criteria": TEAM},
                "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["now"]}})
    check_noul(body["answers"]["billing"])
    check_choice(body["answers"]["team"], TEAM)
    assert body["answers"]["urgency"] == {"type": "score", "score": 0.0, "legend": {"0": "now"}, "probabilities": {"0": 1.0}, "confidence": 1.0}


def test_a_choice_may_have_255_options(api):
    criteria = {f"option {i}": None for i in range(255)}
    check_choice(ask(api, "Pick any.", {"q": {"type": "choice", "instructions": "Which one?", "criteria": criteria}})["answers"]["q"], criteria)


# --- refusals -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("questions", [
    {},
    {"q": {"type": "maybe", "instructions": "Is it?"}},
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {}}},
    {"q": {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": None for i in range(256)}}},
    {"q": {"type": "choice", "instructions": "Which?"}},
    {"q": {"type": "score", "instructions": "How much?", "criteria": []}},
    {"q": {"type": "score", "instructions": "How much?", "criteria": list(range(256))}},
], ids=["no questions", "unknown type", "choice without options", "choice with 256 options", "choice without criteria",
        "score without levels", "score with 256 levels"])
def test_a_malformed_request_is_refused_with_422(api, questions):
    ask(api, "Anything.", questions, status=422)


# --- branch isolation ---------------------------------------------------------------------------------------------------

def test_a_question_answers_the_same_with_or_without_its_siblings(api):
    """Questions share the state but never see each other (the block-causal mask): each question asked alone, and all of
    them through /v1/systemone/separate (one pass per question), answer as they do packed into one request, in any order."""
    state = "Sunny and warm today; the beach is crowded and the ice-cream van has a queue."
    questions = {"nice": {"type": "noul", "instructions": "Is the weather described as nice?"},
                 "season": {"type": "choice", "instructions": "Which season is it?", "criteria": {"summer": None, "winter": None, "unknown": None}},
                 "crowd": {"type": "score", "instructions": "How busy is it?", "criteria": ["empty", "some people", "packed"]}}

    def close(a, b):
        flat = lambda x: [x["noul"]] if x["type"] == "noul" else list(x["probabilities"].values())
        return a["type"] == b["type"] and max(abs(u - v) for u, v in zip(flat(a), flat(b))) <= ISOLATION_TOLERANCE

    packed = ask(api, state, questions)["answers"]
    reversed_order = ask(api, state, dict(reversed(questions.items())))["answers"]
    separate = api.post("/v1/systemone/separate", json={"state": state, "model": "d1a-latest", "questions": questions})
    assert separate.status_code == 200 and separate.headers.get(REQUEST_ID)
    for qid, question in questions.items():
        alone = ask(api, state, {qid: question})["answers"][qid]
        assert close(packed[qid], alone) and close(reversed_order[qid], alone) and close(separate.json()["answers"][qid], alone), qid


# --- models and request ids ---------------------------------------------------------------------------------------------

def test_models_lists_only_d1a_latest_with_a_card(api):
    """GET /v1/models lists d1a-latest and nothing else; TypeSafe's models.list() needs name, description and release_date."""
    r = api.get("/v1/models")
    assert r.status_code == 200 and r.headers.get(REQUEST_ID)
    cards = {card["name"]: card for card in r.json()["models"]}
    assert set(cards) == MODEL_NAMES
    assert all(isinstance(card["description"], str) and card["description"] and card["release_date"] for card in cards.values())


def test_a_request_id_sent_by_the_client_comes_back(api):
    r = api.get("/v1/models", headers={REQUEST_ID: "req-from-client-42"})
    assert r.headers[REQUEST_ID] == "req-from-client-42"
    fresh = {api.get("/v1/models").headers[REQUEST_ID] for _ in range(3)}
    assert len(fresh) == 3   # otherwise a new id per response


# --- the TypeSafe SDK ---------------------------------------------------------------------------------------------------

def test_the_sdk_client_reads_every_answer_type(base_url):
    sdk = pytest.importorskip("typesafe_sdk")
    with sdk.TypeSafeClient(api_key="local", base_url=base_url, model="any-other-name") as client:
        resp = client.system_one(state={"ticket": "Charged twice for one order. Please sort it out today."},
                                 questions={"billing": sdk.Noul(instructions="Is this about billing?"),
                                            "mood": sdk.Choice(instructions="How does the customer sound?", criteria={"calm": None, "annoyed": None}),
                                            "urgency": sdk.Score(instructions="How urgent is it?", criteria=URGENCY)})
    assert 0 <= resp.nouls["billing"].noul <= 1 and resp.request_id
    assert resp.choices["mood"].choice in {"calm", "annoyed"}
    assert 0 <= resp.scores["urgency"].score <= len(URGENCY) - 1
    assert resp.usage.input_tokens > 0 and resp.usage.output_tokens > 0


def test_the_sdk_lists_d1a_latest(base_url):
    """The SDK refuses a card without a name, a description or a release date, so listing them checks all three."""
    sdk = pytest.importorskip("typesafe_sdk")
    with sdk.TypeSafeClient(api_key="local", base_url=base_url) as client:   # no model: the SDK's default, jev-latest
        listed = client.models.list()
    assert {card.name for card in listed.models} == MODEL_NAMES
    assert all(card.description and card.release_date for card in listed.models)


def test_the_async_sdk_client_answers(base_url):
    sdk = pytest.importorskip("typesafe_sdk")

    async def billing():
        async with sdk.AsyncTypeSafeClient(api_key="local", base_url=base_url, model="d1a-latest") as client:
            return await client.system_one(state="Charged twice.", questions={"billing": sdk.Noul(instructions="Is this about billing?")})

    resp = asyncio.run(billing())
    assert 0 <= resp.nouls["billing"].noul <= 1 and resp.request_id
