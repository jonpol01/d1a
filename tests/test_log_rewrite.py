"""D1A_LOG_REWRITE: the decision log keeps names it must not hold in general wording; a log written before it is
reworded by `feedback reword`, outcomes untouched, and the trained manifest hashes states the same way."""
import json

import pytest

from d1a.learning import feedback
from d1a.learning.feedback import FeedbackLog, load_rewrite


def files(tmp_path, rules, entries=("Northwind",)):
    r, b = tmp_path / "rewrite.json", tmp_path / "blocklist.txt"
    r.write_text(json.dumps(rules), encoding="utf-8"); b.write_text("".join(e + "\n" for e in entries), encoding="utf-8")
    return {"D1A_LOG_REWRITE": str(r), "D1A_DATA_BLOCKLIST": str(b)}


def test_rules_are_refused_unless_idempotent_and_covering_and_decisions_are_logged_reworded(tmp_path):
    with pytest.raises(ValueError, match="does not match again"):
        load_rewrite(files(tmp_path, [{"pattern": "kit", "replace": "the kit"}]))
    with pytest.raises(ValueError, match=r"entries \[0\]"):
        load_rewrite(files(tmp_path, [{"pattern": "acme", "replace": "a company"}]))
    rules = load_rewrite(files(tmp_path, [{"pattern": r"northwind[- ]?", "replace": ""}, {"pattern": "the  ", "replace": "the "}]))
    log = FeedbackLog(tmp_path / "log.jsonl", rewrite=rules)
    did = log.decision("Ask Northwind ops about the northwind-kanban card", {"q": {"type": "noul", "instructions": "Northwind?"}},
                       {"q": {"noul": 0.7}}, run="r@v1", meta={"group": "org/northwind#1"})
    e = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8"))
    assert e["id"] == did and e["state"] == "Ask ops about the kanban card" and e["questions"]["q"]["instructions"] == "?"
    assert e["meta"] == {"group": "org/northwind#1"}   # ids stay: outcomes are matched to PRs by them


def test_reword_rewrites_decisions_only_and_the_manifest_hashes_reworded_states(tmp_path, monkeypatch, capsys):
    from d1a.learning import loop
    old = tmp_path / "old.jsonl"
    lines = [{"kind": "decision", "id": "a", "ts": 1, "run": "r", "state": "the Northwind board", "questions": {}, "answers": {}, "meta": {}},
             {"kind": "outcome", "id": "a", "ts": 2, "labels": {"q": True}, "meta": {"note": "Northwind"}},
             {"kind": "void", "ids": ["a"], "src": "s", "reason": "x", "ts": 3}]
    old.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    for k, v in files(tmp_path, [{"pattern": r"northwind ?", "replace": ""}]).items(): monkeypatch.setenv(k, v)
    feedback.main(["reword", str(old), "--out", str(tmp_path / "new.jsonl")])
    new = (tmp_path / "new.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    src = old.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(new) == 3 and json.loads(new[0])["state"] == "the board" and json.loads(new[0])["id"] == "a"
    assert new[1:] == src[1:]   # outcomes and voids byte for byte (an outcome's meta is not a decision's text)
    mix, record = tmp_path / "mix.jsonl", json.dumps({"state": "the Northwind board", "id": "org/x#1"})
    mix.write_text(f"{record}\n", encoding="utf-8")
    m = loop.build_manifest([str(mix)], "r", rewrite=load_rewrite())
    assert m["state_sha256"] == [loop.state_sha256("the board")]
