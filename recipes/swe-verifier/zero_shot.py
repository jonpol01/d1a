"""Zero-shot D1A as a verifier for coding-agent runs: from the issue, the agent's final patch and the tail of its
trajectory, P(the patch resolves the issue), compared with simple trajectory heuristics.

    uv run --extra mlx python recipes/swe-verifier/zero_shot.py --n 600 --out runs/swe-verifier/zero-shot.jsonl

Data: nebius/SWE-agent-trajectories (CC-BY-4.0): SWE-agent runs on SWE-bench-style tasks with `target` (resolved by
the hidden tests), the trajectory and the generated patch. `eval_logs` holds the test results, so it is never read.
A seeded random sample of --n runs from the first --shards shards, at the pinned dataset commit (--revision).
--source openhands scores nebius/SWE-rebench-openhands-trajectories (another agent) instead, mapped by from_openhands.py.

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
REVISION = "68195a1450865274106246d0d0296a1d6807b88e"   # the dataset commit every released selection was drawn from
ERROR = re.compile(r"Traceback \(most recent call last\)|introduced new syntax error|Error:|error:|FAILED|failed", re.MULTILINE)
SUCCESS = re.compile(r"completed successfully|All tests passed|\b\d+ passed\b(?![^\n]*failed)|^OK$", re.MULTILINE)
QUESTION = {"resolved": {"type": "noul",
                         "instr": "A coding agent produced this patch for the issue. Does the patch correctly fix the issue, so that the repository's tests for it would pass?",
                         "criteria": {"true": "The patch fixes the issue as described", "false": "The patch is wrong, incomplete, or fixes something else"}}}


def shard(k, revision=REVISION):
    """Local path of the dataset's k-th parquet shard (of 12) at `revision`, downloaded once into the Hugging Face cache."""
    return hf_hub_download(DATASET, f"data/train-{k:05d}-of-00012.parquet", repo_type="dataset", revision=revision)


def per_issue_draw(rows, per_issue, mixed_only, rng):
    """Up to per_issue runs of every issue, issues in sorted order, drawn with rng (a random.Random).

    Without mixed_only, each issue's runs are rng.sample'd: the draw the released dev and test files were made with
    (eval/dev-*.jsonl, eval/test-*.jsonl), so they rebuild run_key for run_key. With mixed_only, only issues whose runs
    include both outcomes, and each sample keeps both: one success and one failure first, then at random."""
    by_issue = {}
    for r in rows: by_issue.setdefault(r["instance_id"], []).append(r)
    if not mixed_only:
        return [r for iid in sorted(by_issue) for r in rng.sample(by_issue[iid], min(per_issue, len(by_issue[iid])))]
    by_issue = {k: v for k, v in by_issue.items() if 0 < sum(r["target"] for r in v) < len(v)}
    out = []
    for iid in sorted(by_issue):
        runs = by_issue[iid]; rng.shuffle(runs)
        keep = {id(next(x for x in runs if x["target"])), id(next(x for x in runs if not x["target"]))}
        runs.sort(key=lambda r: id(r) not in keep)
        out += runs[:per_issue]
    return out


def run_key(r):
    """One id per recorded run, the same in full and --cut-step outputs (budget_sim.py joins on it)."""
    return hashlib.sha1(json.dumps([r["instance_id"], r["exit_status"], r["generated_patch"], len(r["trajectory"])]).encode()).hexdigest()[:16]


def split_of(repo):
    """test (20%), dev (10%) or train (70%) by a hash of the repository, so a repository is in one split only."""
    h = int(hashlib.sha1(repo.encode()).hexdigest(), 16) % 10
    return "test" if h < 2 else "dev" if h < 3 else "train"


ANYTIME = {"resolved": {"type": "noul",
                        "instr": "A coding agent is part-way through this issue. Will its run end with a patch that correctly fixes the issue, so that the repository's tests for it pass?",
                        "criteria": {"true": "The run is on track to a correct fix", "false": "The run is stuck, off track, or heading to a wrong or no fix"}}}


def cut(r, step):
    """The run as it stood after its `step`-th agent action: the trajectory up to there, no final patch; None when the
    run had ended by then (a run that short has nothing left to decide)."""
    n = 0
    for i, m in enumerate(r["trajectory"]):
        if m["role"] == "ai":
            n += 1
            if n == step:
                rest = r["trajectory"][i + 1:]
                if not any(x["role"] == "ai" for x in rest): return None
                return {**r, "trajectory": r["trajectory"][: i + 2], "generated_patch": None, "exit_status": "in_progress"}   # the final exit is not known yet
    return None


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
    patch = "(the run is still in progress; no patch yet)" if r["generated_patch"] is None else (r["generated_patch"] or "(no patch)")[:patch_chars]
    return (f"ISSUE:\n{issue_of(r['trajectory'])[:issue_chars]}\n\nPATCH:\n{patch}\n\n"
            f"LAST STEPS OF THE AGENT'S RUN:\n" + "\n".join(reversed(tail)))


def auroc(y, s):
    """Mann-Whitney AUROC with tied scores sharing their average rank."""
    from scipy.stats import rankdata
    y, s = np.asarray(y, bool), np.asarray(s, float)
    ranks = rankdata(s); pos = y.sum(); neg = len(y) - pos
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
    ap.add_argument("--run", default="JohnP1/d1a-e4b-mlx-q8", help="a repo takes its latest version (d1a.core.versions)"); ap.add_argument("--no-d1a", action="store_true")
    ap.add_argument("--issue-chars", type=int, default=3000); ap.add_argument("--patch-chars", type=int, default=6000); ap.add_argument("--tail-chars", type=int, default=3000)
    ap.add_argument("--split", choices=["all", "train", "dev", "test"], default="all", help="only runs whose repository is in this split (split_of)")
    ap.add_argument("--mixed-only", action="store_true", help="with --per-issue: only issues whose runs include both outcomes (within-issue evaluation)")
    ap.add_argument("--per-issue", type=int, default=0, help="instead of a random sample: up to this many runs of every issue (for best_of_k.py)")
    ap.add_argument("--cut-step", type=int, default=0, help="anytime mode: score each run as it stood after this many agent actions (runs already over are left out)")
    ap.add_argument("--swegemma", help="score a competition-harness results directory (from_swegemma.py) instead of the public runs")
    ap.add_argument("--tasks", help="with --swegemma: the tasks.jsonl holding the issues")
    ap.add_argument("--source", choices=["swe-agent", "openhands"], default="swe-agent",
                    help="openhands: nebius/SWE-rebench-openhands-trajectories through from_openhands.py (its 12 shards are issue buckets)")
    ap.add_argument("--issues", type=int, default=0, help="with --per-issue: keep only this many of the drawn issues, picked with --seed")
    ap.add_argument("--exclude", help="a zero_shot.py output whose runs are left out before the draw (by trajectory_id when its rows have one, else run_key)")
    ap.add_argument("--keep-issue-md", action="store_true",
                    help="with --source openhands: keep the harness's root issue.md diff in the patch (a shortcut; to measure it)")
    ap.add_argument("--revision", help="the dataset revision (default: the source's pinned commit; for swe-agent, the one the released dev and test rows came from)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.swegemma and a.source != "swe-agent": ap.error("--swegemma reads its own runs; leave --source out")
    if a.issues and not a.per_issue: ap.error("--issues needs --per-issue")
    if a.keep_issue_md and a.source != "openhands": ap.error("--keep-issue-md needs --source openhands")
    oh = a.source == "openhands"
    if oh: import from_openhands
    revision = a.revision or (from_openhands.REVISION if oh else REVISION)
    rows = []
    for k in range(0 if a.swegemma or oh else a.shards):
        rows += pq.read_table(shard(k, revision), columns=["instance_id", "model_name", "target", "trajectory", "exit_status", "generated_patch"]).to_pylist()
    if oh:   # ids and outcomes only; the drawn runs are read in full below
        rows = [r for r in from_openhands.index(revision) if r["shard"] < a.shards]
    if a.swegemma:
        from from_swegemma import load_runs
        rows = load_runs(a.swegemma, a.tasks); a.n = len(rows)
    rows = [r for r in rows if a.split == "all" or split_of(r["instance_id"].rsplit("-", 1)[0]) == a.split]
    if a.exclude:   # e.g. a natural-rate calibration sample that shares no run with a mixed-only draw
        gone = [json.loads(l) for l in Path(a.exclude).read_text(encoding="utf-8").splitlines() if l.strip()]
        by_tid = bool(gone) and all("trajectory_id" in g for g in gone)
        if by_tid and not oh: ap.error("--exclude: trajectory_id rows come from --source openhands")
        ids = {g["trajectory_id"] if by_tid else g["run_key"] for g in gone}
        rows = [r for r in rows if (r["trajectory_id"] if by_tid else run_key(r)) not in ids]
    rng = random.Random(a.seed)
    rows = per_issue_draw(rows, a.per_issue, a.mixed_only, rng) if a.per_issue else rng.sample(rows, min(a.n, len(rows)))
    if a.issues:
        keep = set(rng.sample(sorted({r["instance_id"] for r in rows}), min(a.issues, len({r["instance_id"] for r in rows}))))
        rows = [r for r in rows if r["instance_id"] in keep]
    if oh: rows = from_openhands.hydrate(rows, revision, a.keep_issue_md)
    for r in rows: r["run_key"] = run_key(r)
    if a.cut_step:
        rows = [c for c in (cut(r, a.cut_step) for r in rows) if c is not None]
    question = ANYTIME if a.cut_step else QUESTION
    model = None
    if not a.no_d1a:
        from d1a import D1A
        from d1a.core.versions import latest
        a.run = a.run if Path(a.run).exists() else latest(a.run)   # a local run directory is used as is
        model = D1A.load(a.run)
    out = []
    for i, r in enumerate(rows):
        rec = {"run_key": r["run_key"], "instance_id": r["instance_id"], "repo": r.get("repo") or r["instance_id"].rsplit("-", 1)[0], "model": r["model_name"], "y": bool(r["target"]), **heuristics(r)}
        rec.update({k: r[k] for k in ("trajectory_id", "issue_md") if k in r})   # openhands: the run's id, and whether its patch carried issue.md
        if model:
            t0 = time.time(); ans = model.decide(state_of(r, a.issue_chars, a.patch_chars, a.tail_chars), question)["resolved"]
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
    if len(set(groups)) < 2 or len(set(y)) < 2:
        print("  (too few repositories or outcomes for a cross-validated logistic regression)"); return
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
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
