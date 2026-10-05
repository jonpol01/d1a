"""SWE-agent runs -> D1A training records for the verifier question ("does this patch resolve the issue?").

    uv run python recipes/swe-verifier/to_records.py --shards 3 --out runs/swe-verifier/data

Source: nebius/SWE-agent-trajectories (CC-BY-4.0); `eval_logs` (the test results) is never read. The state is the same
text zero_shot.py gives the model (issue, patch, last steps), so zero-shot and trained numbers compare like for like.
Split by repository (split_of: 20% test, 10% dev, 70% train), so no repository is in two splits: the scored setting is
unseen repositories. Each issue keeps at most --per-issue runs (positives first, so rare successes are not dropped),
and train is capped at --max-train records.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from zero_shot import DATASET, QUESTION, split_of, state_of


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", type=int, default=3); ap.add_argument("--per-issue", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--issue-chars", type=int, default=3000); ap.add_argument("--patch-chars", type=int, default=6000); ap.add_argument("--tail-chars", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    by_issue = defaultdict(list)
    for k in range(a.shards):
        for r in pq.read_table(hf_hub_download(DATASET, f"data/train-{k:05d}-of-00012.parquet", repo_type="dataset"),
                               columns=["instance_id", "target", "trajectory", "exit_status", "generated_patch"]).to_pylist():
            by_issue[r["instance_id"]].append(r)
    q = QUESTION["resolved"]
    out = defaultdict(list)
    for iid, runs in sorted(by_issue.items()):
        rng.shuffle(runs); runs.sort(key=lambda r: not r["target"])   # positives first, then a random mix
        repo = iid.rsplit("-", 1)[0]
        for r in runs[: a.per_issue]:
            out[split_of(repo)].append({"state": state_of(r, a.issue_chars, a.patch_chars, a.tail_chars),
                                        "questions": {"resolved": {"type": "noul", "instructions": q["instr"], "criteria": q["criteria"],
                                                                   "label": bool(r["target"]), "src": "swe-verifier"}},
                                        "meta": {"instance_id": iid, "repo": repo}})
    rng.shuffle(out["train"]); out["train"] = out["train"][: a.max_train]
    d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
    for split, recs in out.items():
        (d / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        lab = Counter(r["questions"]["resolved"]["label"] for r in recs)
        print(f"{split}: {len(recs)} records, {lab[True]} resolved ({lab[True] / max(len(recs), 1):.1%}), "
              f"{len({r['meta']['instance_id'] for r in recs})} issues, {len({r['meta']['repo'] for r in recs})} repos")


if __name__ == "__main__":
    main()
