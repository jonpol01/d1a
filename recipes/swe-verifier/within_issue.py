"""Within-issue AUROC (runs of one issue ranked against each other) with a 95% interval from a bootstrap over issues.

    uv run python recipes/swe-verifier/within_issue.py --fit runs/swe-verifier/eval/heuristics-train.jsonl NAME=<zero_shot.py output> ...

Each output must come from the same --mixed-only selection so rows line up. Scorers: the heuristics' logistic regression
(fitted on --fit, training-split runs) and each file's p_d1a. Global AUROC is shown for contrast. Within-issue AUROC is
invariant to anything constant per issue, so recalibration does not change it.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import heuristic_features  # noqa: E402
from zero_shot import auroc  # noqa: E402


def load(path): return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def per_issue(y, p, issues):
    by = {}
    for i, t in enumerate(issues): by.setdefault(t, []).append(i)
    return {t: auroc(y[ix], p[ix]) for t, ix in ((t, np.array(ix)) for t, ix in by.items()) if 0 < y[ix].sum() < len(ix)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", required=True); ap.add_argument("runs", nargs="+"); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    files = {n: load(f) for n, f in (x.split("=", 1) for x in a.runs)}
    first = next(iter(files.values()))
    for n, rows in files.items(): assert [r["run_key"] for r in rows] == [r["run_key"] for r in first], f"{n}: rows differ"
    y = np.array([r["y"] for r in first], float); issues = np.array([r["instance_id"] for r in first])
    fit = load(a.fit)
    scores = {"heuristics (LR)": LogisticRegression(max_iter=1000).fit(heuristic_features(fit), [r["y"] for r in fit]).predict_proba(heuristic_features(first))[:, 1]}
    scores.update({n: np.array([r["p_d1a"] for r in rows]) for n, rows in files.items()})
    per = {n: per_issue(y, p, issues) for n, p in scores.items()}
    keys = sorted(next(iter(per.values()))); rng = np.random.default_rng(0)
    print(f"{len(first)} runs, {len(set(issues))} issues, {len(keys)} with both outcomes, {y.mean():.1%} resolved")
    for n, d in per.items():
        v = np.array([d[k] for k in keys]); b = np.sort([v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.boot)])
        print(f"  {n:18s} within-issue AUROC {v.mean():.3f} [{b[int(.025 * a.boot)]:.3f}, {b[int(.975 * a.boot) - 1]:.3f}]   global AUROC {auroc(y, scores[n]):.3f}")
    base = np.array([per["heuristics (LR)"][k] for k in keys])
    for n in [n for n in per if n != "heuristics (LR)"]:
        d = np.array([per[n][k] for k in keys]) - base; b = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)])
        print(f"  {n} - heuristics: {d.mean():+.3f} [{b[int(.025 * a.boot)]:+.3f}, {b[int(.975 * a.boot) - 1]:+.3f}]")


if __name__ == "__main__":
    main()
