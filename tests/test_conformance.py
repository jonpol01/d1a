"""Conformance against committed golden vectors (#64): a tiny Gemma 4 checkpoint (tests/golden/tiny-gemma4, weights in
git, built by tests/golden/build_tiny_gemma4.py) must give the token ids in golden.json exactly and every question's
fp32 CPU probabilities to 1e-5, through the model, through d1a.serving.serve, and through scripts/golden_vectors.py compare (the
check any other port runs). A change to the model, checkpoint or serving code that moves an answer fails here; one that
is meant to must rebuild the fixture in the same pull request and say why."""
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path("tests/golden/tiny-gemma4")   # relative: the checkpoint names its base by this path
TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def golden():
    return json.loads((ROOT / FIXTURE / "golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loaded():
    from d1a.backends.checkpoint import LoadOptions, load
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        tok, model = load(str(FIXTURE / "checkpoint"), "cpu", LoadOptions(dtype=torch.float32, backend="torch"))
    return tok, model.eval()


def encoded(model, tok, request):
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.backends.torch import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    rec, _ = to_record(SystemOneRequest.model_validate(request))
    return rec, model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)


def test_the_fixture_covers_what_it_should(golden):
    records = golden["records"]
    assert golden["format"] == "d1a-golden" and golden["version"] == 1 and len(records) == 40
    assert {q["type"] for r in records for q in r["questions"]} == {"choice", "noul", "score"}
    assert max(len(r["ids"]) for r in records) > 4 * 8 and max(len(r["questions"]) for r in records) >= 5   # far past the 4-token window


def test_token_ids_and_probabilities_match_the_golden_vectors(golden, loaded):
    tok, model = loaded
    worst = 0.0
    with torch.no_grad():
        for record in golden["records"]:
            rec, enc = encoded(model, tok, record["request"])
            assert {"state": rec["state"], "questions": [{"instr": q["instr"], "options": q["options"]} for q in rec["questions"]]} == record["input"], record["id"]
            assert enc["ids"] == record["ids"], record["id"]
            for probs, q in zip(model.probs(enc), record["questions"]):
                worst = max(worst, (probs.float() - torch.tensor(q["probs"])).abs().max().item())
    assert worst < TOLERANCE, worst


def test_served_answers_match_the_golden_vectors(golden, loaded):
    from d1a.core.api import SystemOneRequest
    from d1a.backends.checkpoint import Checkpoint
    from d1a.serving.serve import Server
    tok, model = loaded
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        server = Server(Checkpoint(str(FIXTURE / "checkpoint")), tok, model, "cpu")
    try:
        for record in golden["records"][:15]:
            answers = server.answer(SystemOneRequest.model_validate(record["request"]))["answers"]
            for q in record["questions"]:
                a = answers[q["qid"]]
                served = [1 - a["noul"], a["noul"]] if q["type"] == "noul" else [a["probabilities"][k] for k in q["keys"]]
                assert max(abs(s - g) for s, g in zip(served, q["probs"])) <= 5e-5 + TOLERANCE, (record["id"], q["qid"])   # 4-decimal answers
    finally:
        server.close()


def test_the_golden_vectors_compare_tool_passes_on_the_fixture(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import golden_vectors
    golden_vectors.compare(Namespace(golden=str(FIXTURE / "golden.json"), run=str(FIXTURE / "checkpoint"), device="cpu", backend="torch", out=str(tmp_path / "report.json")))
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["token_id_mismatches"] == [] and report["all"]["questions"] == 154
    assert report["all"]["max_dp"] < TOLERANCE and report["all"]["argmax_flips"] == 0
    capsys.readouterr()


def test_an_audited_answer_says_what_produced_it(golden, loaded, monkeypatch):
    """#32: with x-d1a-audit: on, an answer carries what produced it, so a port or a support ticket can prove it: the code,
    the weights, the text the model read, its temperature and where that came from, the self-learning steps applied, and
    the prefix cache per model pass. The answers themselves are the ones an unaudited request gets."""
    import hashlib
    import d1a
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.backends.checkpoint import Checkpoint
    from d1a.serving import serve
    tok, model = loaded
    monkeypatch.setattr(serve, "PREFIX_MIN_TOKENS", "0")   # the fixture's states are short: cache them anyway
    monkeypatch.setattr(serve, "LEARNING", serve.Learning())
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        server = serve.Server(Checkpoint(str(FIXTURE / "checkpoint")), tok, model, "cpu")
        weights = Checkpoint(str(FIXTURE / "checkpoint")).weights_sha256()
    try:
        req = SystemOneRequest.model_validate(golden["records"][0]["request"])
        plain, first, again = server.answer(req), server.answer(req, audit=True), server.answer(req, audit=True)
        assert "audit" not in plain and first["answers"] == plain["answers"] == again["answers"] and first["usage"] == plain["usage"]
        a = first["audit"]
        assert a["d1a"] == d1a.__version__ and a["run"] == str(FIXTURE / "checkpoint") and a["weights_sha256"] == weights
        assert a["state_sha256"] == hashlib.sha256(to_record(req)[0]["state"].encode("utf-8")).hexdigest()
        assert a["temperature"] == a["checkpoint_temperature"] == model.head.temperature and a["temperature_source"] == "checkpoint"
        assert a["outcome_calibrator"] == [] and a["rules"] == {} and a["flags"] == {} and a["logged"] is False
        assert len(a["passes"]) == 1 and a["passes"][0]["tokens"] > 0
        hit = again["audit"]["passes"][0]   # the same state again: read from the prefix cache
        assert hit["prefix_cache_hit"] is True and 0 < hit["cached_tokens"] < hit["tokens"]
        monkeypatch.setattr(model.head, "use_case_temperatures", {"routing": 2.0})
        routed = server.answer(req.model_copy(update={"use_case": "routing"}), audit=True)["audit"]
        assert routed["temperature"] == 2.0 and routed["temperature_source"] == "use_case:routing"
        first_q, second_q = list(req.questions)[:2]
        cond = req.model_copy(update={"questions": {first_q: req.questions[first_q],
                                                    second_q: req.questions[second_q].model_copy(update={"ask_if": {first_q: list(req.questions[first_q].criteria)}})}})
        assert len(server.answer(cond, audit=True)["audit"]["passes"]) == 2   # one entry per ask_if pass
    finally:
        server.close()
