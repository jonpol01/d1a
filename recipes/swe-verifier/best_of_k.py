"""Verifier-guided best-of-k: an agent ran k times on an issue; submit the run a scorer ranks highest. How often is the
submitted run the resolved one? Compared with picking at random and with an oracle that picks a resolved run whenever
one exists.

    uv run python recipes/swe-verifier/best_of_k.py runs/swe-verifier/zero-shot/<test-split, --per-issue>.jsonl \
        --fit runs/swe-verifier/zero-shot/<600-run file>.jsonl --k 2,4,8

Input: zero_shot.py output with many runs per issue (`--split test --per-issue N`). Scorers: random; the heuristics'
logistic regression, fitted on the --fit file's train-split repositories only; each probability field in the input
(p_d1a, ...). For each issue with at least k runs, the expected resolve rate over --draws random k-subsets; ties are
broken at random. 95% intervals from a bootstrap over issues.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import heuristic_features  # noqa: E402
from zero_shot import split_of  # noqa: E402


def pick_rate(scores, resolved, k, draws, rng):
    """Expected resolved rate of the top-scored run among k drawn without replacement (ties at random)."""
    n = len(scores); hits = 0.0
    for _ in range(draws):
        s = rng.choice(n, k, replace=False)
        best = s[scores[s] == scores[s].max()]
        hits += resolved[best].mean()
    return hits / draws


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs"); ap.add_argument("--fit", required=True, help="rows for the heuristic LR (only train-split repositories are used)")
    ap.add_argument("--k", default="2,4,8"); ap.add_argument("--draws", type=int, default=200); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.runs).read_text(encoding="utf-8").splitlines() if l.strip()]
    fit = [r for r in (json.loads(l) for l in Path(a.fit).read_text(encoding="utf-8").splitlines() if l.strip()) if split_of(r["repo"]) == "train"]
    lr = LogisticRegression(max_iter=1000).fit(heuristic_features(fit), [r["y"] for r in fit])
    scorers = {"random": np.zeros(len(rows)), "heuristics (LR)": lr.predict_proba(heuristic_features(rows))[:, 1]}
    for key in sorted({k for r in rows for k in r if k.startswith("p_")}): scorers[key] = np.array([r[key] for r in rows])
    issues = {}
    for i, r in enumerate(rows): issues.setdefault(r["instance_id"], []).append(i)
    y = np.array([r["y"] for r in rows], float)
    rng = np.random.default_rng(a.seed)
    print(f"{len(rows)} runs, {len(issues)} issues, {y.mean():.1%} resolved; heuristic LR fitted on {len(fit)} train-split runs")
    for k in (int(x) for x in a.k.split(",")):
        ids = [iid for iid, ix in issues.items() if len(ix) >= k]
        per = {name: np.array([pick_rate(s[issues[iid]], y[issues[iid]], k, a.draws, rng) for iid in ids]) for name, s in scorers.items()}
        per["oracle"] = np.array([1.0 - np.mean([y[np.array(issues[iid])[rng.choice(len(issues[iid]), k, replace=False)]].max() == 0 for _ in range(a.draws)]) for iid in ids])
        print(f"\nk={k}: {len(ids)} issues with at least {k} runs")
        for name, v in per.items():
            b = np.sort([v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.boot)])
            print(f"  {name:18s} resolve rate {v.mean():.3f}  [{b[int(0.025 * a.boot)]:.3f}, {b[int(0.975 * a.boot) - 1]:.3f}]")
        for name in [n for n in scorers if n.startswith("p_")]:
            d = per[name] - per["heuristics (LR)"]
            b = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)])
            print(f"  {name} - heuristics (LR): {d.mean():+.3f}  [{b[int(0.025 * a.boot)]:+.3f}, {b[int(0.975 * a.boot) - 1]:+.3f}]")


if __name__ == "__main__":
    main()
