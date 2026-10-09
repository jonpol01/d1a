"""Put the nebius text back into the released self-improvement files (JohnP1/d1a-swe-verifier-eval).

    python recipes/swe-verifier/rebuild_texts.py <dataset dir>          # *.ids.jsonl -> feedback.jsonl, train-r*.jsonl
    python recipes/swe-verifier/rebuild_texts.py --slim <dataset dir>   # the reverse: how the *.ids.jsonl files were made

The verifier's input (issue, final patch, last steps of the run) is text from nebius/SWE-agent-trajectories, so the
dataset does not re-host it. self-improve/feedback.ids.jsonl is the decision log with each decision's `state` replaced by
the run it was built from (`nebius`: shard, row and run_key at zero_shot.REVISION); a training record in
train-r*.ids.jsonl drops its `state`, which is its decision's (meta.feedback_id). Rebuilding reads those rows at the pinned
revision and builds each state with zero_shot.state_of, as self_improve.py did. run_key alone is not enough: some runs
share it (same issue, patch and length) and differ in their last steps.
"""
import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zero_shot import run_key, shard, state_of  # noqa: E402

COLS = ["instance_id", "trajectory", "exit_status", "generated_patch"]
SHARDS = (3, 4, 5)   # self_improve.py --stream-shards
ROUNDS = ("train-r1", "train-r2", "train-r3")


def state(r): return state_of(r, 3000, 6000, 3000)   # self_improve.py's limits


def rebuild(d):
    rows, states, out = {}, {}, []
    for line in open(d / "feedback.ids.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e["kind"] == "decision":
            src = e["nebius"]
            if src["shard"] not in rows: rows[src["shard"]] = pq.read_table(shard(src["shard"]), columns=COLS).to_pylist()
            r = rows[src["shard"]][src["row"]]
            if run_key(r) != src["run_key"]: sys.exit(f"decision {e['id']}: shard {src['shard']} row {src['row']} is not run {src['run_key']}")
            states[e["id"]] = state(r)
            e = {("state" if k == "nebius" else k): (states[e["id"]] if k == "nebius" else v) for k, v in e.items()}
        out.append(json.dumps(e) + "\n")
    (d / "feedback.jsonl").write_text("".join(out), encoding="utf-8")
    for name in ROUNDS:
        recs = [json.loads(line) for line in open(d / f"{name}.ids.jsonl", encoding="utf-8")]
        (d / f"{name}.jsonl").write_text("".join(json.dumps({"state": states[x["meta"]["feedback_id"]], **x}) + "\n" for x in recs), encoding="utf-8")
    print(f"rebuilt {len(states)} decisions and {', '.join(ROUNDS)} from nebius/SWE-agent-trajectories")


def slim(d):
    where = {}
    for k in SHARDS:
        for i, r in enumerate(pq.read_table(shard(k), columns=COLS).to_pylist()):
            where.setdefault(state(r), {"shard": k, "row": i, "run_key": run_key(r)})
    out, states = [], {}
    for line in open(d / "feedback.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if e["kind"] == "decision":
            if e["state"] not in where: sys.exit(f"decision {e['id']}: no run in shards {SHARDS} gives its state")
            states[e["id"]] = e["state"]
            e = {("nebius" if k == "state" else k): (where[v] if k == "state" else v) for k, v in e.items()}
        out.append(json.dumps(e) + "\n")
    (d / "feedback.ids.jsonl").write_text("".join(out), encoding="utf-8")
    for name in ROUNDS:
        recs = [json.loads(line) for line in open(d / f"{name}.jsonl", encoding="utf-8")]
        if any(x["state"] != states[x["meta"]["feedback_id"]] for x in recs): sys.exit(f"{name}: a record's state is not its decision's")
        (d / f"{name}.ids.jsonl").write_text("".join(json.dumps({k: v for k, v in x.items() if k != "state"}) + "\n" for x in recs), encoding="utf-8")
    print(f"slimmed {len(states)} decisions and {', '.join(ROUNDS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", help="the dataset directory (its self-improve/ folder is read and written)")
    ap.add_argument("--slim", action="store_true", help="write the *.ids.jsonl files from the full ones")
    a = ap.parse_args()
    (slim if a.slim else rebuild)(Path(a.data) / "self-improve")
