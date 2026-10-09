"""How long are the verifier's inputs in tokens? Rebuilds a zero_shot.py selection and measures exactly what was scored.

    uv run python recipes/swe-verifier/token_lengths.py --shards 3 --split test --per-issue 8 \
        --check runs/swe-verifier/eval/<test file>.jsonl [--records runs/swe-verifier/data/train.jsonl --max-state N]

The selection is zero_shot.py's own (same --shards, --split, --per-issue, --mixed-only, --seed, dataset --revision;
zero_shot.per_issue_draw). --check compares it run_key by run_key, in order, with a saved zero_shot.py output and stops
when they differ, so the measured inputs are the scored ones. Per input, two lengths with the Gemma 4 tokenizer:
- document: the state as encode() packs it (the leading ids, <state>, the escaped state tokens);
- full: the whole packed request (state + the verifier question's instruction, options and <decide>), what D1A.decide
  encodes (d1a.core.api.to_record + d1a.core.encoding.encode at the serving limits).
--records also measures to_records.py training records (their state as is); with --max-state, how many records' state is
longer than that (a trainer with that --max_state cuts them). CPU only, no model; prints aggregates only.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zero_shot import QUESTION, REVISION, per_issue_draw, run_key, shard, split_of, state_of  # noqa: E402

from d1a.core.api import SystemOneRequest, to_record  # noqa: E402
from d1a.core.encoding import SERVE_MAX_BRANCH, SERVE_MAX_STATE, encode, layout, load_tokenizer, user_tokens  # noqa: E402

COLS = ["instance_id", "model_name", "target", "trajectory", "exit_status", "generated_patch"]


def select(shards, split, per_issue, mixed_only, seed, revision):
    """zero_shot.py's --per-issue selection, reading only the split's rows of each shard (same order, less memory)."""
    rows = []
    for k in range(shards):
        t = pq.read_table(shard(k, revision), columns=COLS)
        keep = [i for i, iid in enumerate(t.column("instance_id").to_pylist()) if split == "all" or split_of(iid.rsplit("-", 1)[0]) == split]
        rows += t.take(keep).to_pylist(); del t
    rows = per_issue_draw(rows, per_issue, mixed_only, random.Random(seed))
    for r in rows: r["run_key"] = run_key(r)
    return rows


def lengths(tok, states):
    head = len(layout(tok)[0]) + 1   # leading ids + <state>
    doc, full = [], []
    for s in states:
        rec, _ = to_record(SystemOneRequest(model="d1a-latest", state=s, questions=QUESTION))
        doc.append(head + len(user_tokens(tok, rec["state"])))
        full.append(len(encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)["ids"]))
    return np.array(doc), np.array(full)


def show(name, chars, doc, full):
    q = lambda v: f"median {np.median(v):,.0f}  mean {v.mean():,.0f}  p90 {np.percentile(v, 90):,.0f}  p99 {np.percentile(v, 99):,.0f}  max {v.max():,}  min {v.min():,}"
    print(f"\n{name}: {len(doc)} inputs")
    print(f"  characters      {q(np.array(chars))}")
    print(f"  document tokens {q(doc)}")
    print(f"  full tokens     {q(full)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", type=int, default=3); ap.add_argument("--split", choices=["all", "train", "dev", "test"], default="test")
    ap.add_argument("--per-issue", type=int, default=8); ap.add_argument("--mixed-only", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=REVISION, help="the dataset revision (default: zero_shot.REVISION)")
    ap.add_argument("--issue-chars", type=int, default=3000); ap.add_argument("--patch-chars", type=int, default=6000); ap.add_argument("--tail-chars", type=int, default=3000)
    ap.add_argument("--check", help="a zero_shot.py output of this selection: stop unless its run_keys match in order")
    ap.add_argument("--records", nargs="*", default=[], help="to_records.py files to measure as well")
    ap.add_argument("--max-state", type=int, help="with --records: count the records whose state is longer than this")
    ap.add_argument("--tokenizer", default="google/gemma-4-E4B"); ap.add_argument("--tokenizer-revision", default="411aa17b749aa952df1359d2dcea73917a544d9a")
    a = ap.parse_args()
    tok = load_tokenizer(a.tokenizer, revision=a.tokenizer_revision)
    print(f"tokenizer: {a.tokenizer}@{a.tokenizer_revision[:8]} ({type(tok).__name__}, vocab {len(tok)})")
    rows = select(a.shards, a.split, a.per_issue, a.mixed_only, a.seed, a.revision)
    name = f"--shards {a.shards} --split {a.split} --per-issue {a.per_issue}{' --mixed-only' if a.mixed_only else ''} --seed {a.seed}"
    if a.check:
        ref = [json.loads(l)["run_key"] for l in Path(a.check).read_text(encoding="utf-8").splitlines() if l.strip()]
        if [r["run_key"] for r in rows] != ref:
            sys.exit(f"{name}: the rebuilt selection ({len(rows)} runs) differs from {a.check} ({len(ref)} rows)")
        name += f" (all {len(ref)} run_keys match {a.check} in order)"
    states = [state_of(r, a.issue_chars, a.patch_chars, a.tail_chars) for r in rows]
    del rows
    show(name, [len(s) for s in states], *lengths(tok, states))
    for path in a.records:
        recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        doc, full = lengths(tok, [r["state"] for r in recs])
        show(f"records {path}", [len(r["state"]) for r in recs], doc, full)
        if a.max_state:
            print(f"  records whose document is longer than {a.max_state} tokens: {int((doc > a.max_state).sum())}")


if __name__ == "__main__":
    main()
