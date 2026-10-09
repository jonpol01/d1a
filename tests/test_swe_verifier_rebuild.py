"""recipes/swe-verifier/rebuild_texts.py: the released self-improvement files ship without the nebius text and rebuild it
byte for byte, run by run, including runs that share a run_key but differ in their last steps. Synthetic parquet
shards stand in for nebius; no model, no download."""
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("swe_verifier_rebuild_texts", ROOT / "recipes/swe-verifier/rebuild_texts.py")
rebuild_texts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rebuild_texts)


def run(iid, tail, patch="diff"):
    return {"instance_id": iid, "exit_status": "submitted", "generated_patch": patch, "target": False,
            "trajectory": [{"role": "system", "text": "s"}, {"role": "user", "text": f"ISSUE:\n{iid} breaks at /Users/x\nINSTRUCTIONS:"},
                           {"role": "ai", "text": "look"}, {"role": "user", "text": tail}]}


@pytest.fixture
def released(tmp_path, monkeypatch):
    shards = {k: [run(f"o__r-{k}{i}", f"out {k}{i}") for i in range(4)] for k in rebuild_texts.SHARDS}
    shards[4][2] = run("o__r-30", "a different tail")   # the same run_key as shard 3 row 0, another state
    for k, rows in shards.items(): pq.write_table(pa.Table.from_pylist(rows), tmp_path / f"{k}.parquet")
    monkeypatch.setattr(rebuild_texts, "shard", lambda k: str(tmp_path / f"{k}.parquet"))
    d = tmp_path / "data" / "self-improve"; d.mkdir(parents=True)
    picks = [shards[4][2], shards[3][0], shards[5][1]]
    fb, recs = [], []
    for i, r in enumerate(picks):
        did = f"id{i}"
        fb.append({"kind": "decision", "id": did, "ts": 1.5 + i, "run": "r1", "state": rebuild_texts.state(r),
                   "questions": {"resolved": {"type": "noul"}}, "answers": {"resolved": {"type": "noul", "noul": 0.25}}, "meta": {"repo": "o__r"}})
        fb.append({"kind": "outcome", "id": did, "ts": 2.5 + i, "labels": {"resolved": False}, "meta": {}})
        recs.append({"state": rebuild_texts.state(r), "questions": {"resolved": {"label": False}}, "meta": {"repo": "o__r", "feedback_id": did}})
    (d / "feedback.jsonl").write_text("".join(json.dumps(e) + "\n" for e in fb), encoding="utf-8")
    for n, name in enumerate(rebuild_texts.ROUNDS):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs[: n + 1]), encoding="utf-8")
    return d


def test_slim_drops_the_text_and_rebuild_restores_it_byte_for_byte(released):
    full = {p.name: p.read_bytes() for p in released.iterdir()}
    rebuild_texts.slim(released)
    for p in list(released.iterdir()):
        if not p.name.endswith(".ids.jsonl"): p.unlink()
    assert not any(b"/Users/x" in p.read_bytes() or b"ISSUE" in p.read_bytes() for p in released.iterdir())
    rebuild_texts.rebuild(released)
    assert {n: (released / n).read_bytes() for n in full} == full


def test_rebuild_refuses_a_row_that_is_not_the_named_run(released):
    rebuild_texts.slim(released)
    p = released / "feedback.ids.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"row": 0,', '"row": 1,', 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="is not run"):
        rebuild_texts.rebuild(released)
