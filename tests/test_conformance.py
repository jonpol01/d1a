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
