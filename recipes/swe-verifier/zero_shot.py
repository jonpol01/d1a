"""Zero-shot D1A as a verifier for coding-agent runs: from the issue, the agent's final patch and the tail of its
trajectory, P(the patch resolves the issue), compared with simple trajectory heuristics.

    uv run --extra mlx python recipes/swe-verifier/zero_shot.py --n 600 --out runs/swe-verifier/zero-shot.jsonl

Data: nebius/SWE-agent-trajectories (CC-BY-4.0): SWE-agent runs on SWE-bench-style tasks with `target` (resolved by
the hidden tests), the trajectory and the generated patch. `eval_logs` holds the test results, so it is never read.
A seeded random sample of --n runs from the first --shards shards.

Reports AUROC, ECE (10 bins), Brier score and reliability bins for D1A's P(resolved), and the AUROC of each heuristic:
patch size, a clean `submitted` exit, trajectory length, error observations, and "a run printed success". A logistic
regression over the heuristics, cross-validated by repository (no repository in both train and test), is the
learned baseline D1A has to beat.
"""
import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET = "nebius/SWE-agent-trajectories"
ERROR = re.compile(r"Traceback \(most recent call last\)|introduced new syntax error|Error:|error:|FAILED|failed", re.MULTILINE)
SUCCESS = re.compile(r"completed successfully|All tests passed|\b\d+ passed\b(?![^\n]*failed)|^OK$", re.MULTILINE)
QUESTION = {"resolved": {"type": "noul",
                         "instr": "A coding agent produced this patch for the issue. Does the patch correctly fix the issue, so that the repository's tests for it would pass?",
                         "criteria": {"true": "The patch fixes the issue as described", "false": "The patch is wrong, incomplete, or fixes something else"}}}


def split_of(repo):
    """test (20%), dev (10%) or train (70%) by a hash of the repository, so a repository is in one split only."""
    h = int(hashlib.sha1(repo.encode()).hexdigest(), 16) % 10
    return "test" if h < 2 else "dev" if h < 3 else "train"


def issue_of(traj):
    u = next((m.get("text") or "" for m in traj if m["role"] == "user"), "")
    i, j = u.find("ISSUE:"), u.find("INSTRUCTIONS:")
    return (u[i + 6: j] if i >= 0 and j > i else u).strip()


def heuristics(r):
    traj = r["trajectory"]; obs = [m.get("text") or "" for m in traj if m["role"] == "user"][1:]
    patch = r["generated_patch"] or ""
    changed = sum(1 for l in patch.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---")))
    return {"patch_lines": changed, "clean_submit": float(r["exit_status"] == "submitted"), "steps": len(obs),
            "error_obs": sum(bool(ERROR.search(o)) for o in obs), "success_printed": float(any(SUCCESS.search(o) for o in obs[-10:])),
            "edits_tests": float(bool(re.search(r"^\+\+\+ b/\S*test", patch, re.MULTILINE)))}


def state_of(r, issue_chars, patch_chars, tail_chars):
    tail, used = [], 0
    for m in reversed(r["trajectory"]):
        if m["role"] == "system": break
        t = f"[{'agent' if m['role'] == 'ai' else 'environment'}] {(m.get('text') or '').strip()}"[:800]
        if used + len(t) > tail_chars: break
        tail.append(t); used += len(t)
    return (f"ISSUE:\n{issue_of(r['trajectory'])[:issue_chars]}\n\nPATCH:\n{(r['generated_patch'] or '(no patch)')[:patch_chars]}\n\n"
            f"LAST STEPS OF THE AGENT'S RUN:\n" + "\n".join(reversed(tail)))


def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):   # average ranks for ties
        idx = s == v; ranks[idx] = ranks[idx].mean()
    pos = y.sum(); neg = len(y) - pos
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else float("nan")


def calibration(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (p > i / bins) & (p <= (i + 1) / bins) if i else (p <= 1 / bins)
        if m.any():
            rows.append((f"{i / bins:.1f}-{(i + 1) / bins:.1f}", int(m.sum()), round(float(p[m].mean()), 3), round(float(y[m].mean()), 3)))
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece), float(((p - y) ** 2).mean()), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--shards", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="JohnP1/d1a-e4b-mlx-q8", help="a repo takes its latest version (d1a.versions)"); ap.add_argument("--no-d1a", action="store_true")
    ap.add_argument("--issue-chars", type=int, default=3000); ap.add_argument("--patch-chars", type=int, default=6000); ap.add_argument("--tail-chars", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for k in range(a.shards):
        rows += pq.read_table(hf_hub_download(DATASET, f"data/train-{k:05d}-of-00012.parquet", repo_type="dataset"),
                              columns=["instance_id", "model_name", "target", "trajectory", "exit_status", "generated_patch"]).to_pylist()
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    model = None
    if not a.no_d1a:
        from d1a import D1A
        from d1a.versions import latest
        a.run = latest(a.run)
        model = D1A.load(a.run)
    out = []
    for i, r in enumerate(rows):
        rec = {"instance_id": r["instance_id"], "repo": r["instance_id"].rsplit("-", 1)[0], "model": r["model_name"], "y": bool(r["target"]), **heuristics(r)}
        if model:
            t0 = time.time(); ans = model.decide(state_of(r, a.issue_chars, a.patch_chars, a.tail_chars), QUESTION)["resolved"]
            rec["p_d1a"] = float(ans["noul"]); rec["run"] = a.run; rec["seconds"] = round(time.time() - t0, 2)
        out.append(rec)
        if i % 50 == 0: print(i, rec.get("p_d1a"), rec["y"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    y = [r["y"] for r in out]
    print(f"\n{len(out)} runs, {sum(y)} resolved ({np.mean(y):.1%}), {len({r['instance_id'] for r in out})} issues, {len({r['repo'] for r in out})} repos;"
          f" models {dict(Counter(r['model'] for r in out))}")
    feats = ["patch_lines", "clean_submit", "steps", "error_obs", "success_printed", "edits_tests"]
    for f in feats: print(f"  heuristic {f:16s} AUROC {auroc(y, [r[f] for r in out]):.3f}")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = np.array([[np.log1p(r["patch_lines"]), r["clean_submit"], np.log1p(r["steps"]), np.log1p(r["error_obs"]), r["success_printed"], r["edits_tests"]] for r in out])
    groups = [r["repo"] for r in out]; p_lr = np.zeros(len(out))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        p_lr[te] = LogisticRegression(max_iter=1000).fit(X[tr], np.array(y)[tr]).predict_proba(X[te])[:, 1]
    ece, brier, _ = calibration(y, p_lr)
    print(f"  logistic regression over the heuristics, CV by repo: AUROC {auroc(y, p_lr):.3f}, ECE {ece:.3f}, Brier {brier:.3f}")
    if model:
        p = [r["p_d1a"] for r in out]; ece, brier, bins = calibration(y, p)
        print(f"  D1A zero-shot ({a.run}): AUROC {auroc(y, p):.3f}, ECE {ece:.3f}, Brier {brier:.3f} (base-rate Brier {np.mean(y) * (1 - np.mean(y)):.3f}); "
              f"mean p {np.mean(p):.3f}; median {np.median([r['seconds'] for r in out]):.2f} s per run")
        print("  reliability (bin, n, mean p, resolved rate):", bins)


if __name__ == "__main__":
    main()
