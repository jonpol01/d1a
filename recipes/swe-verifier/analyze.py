"""Does D1A add to cheap trajectory heuristics? Reads a zero_shot.py output and reports, all cross-validated by
repository (no repository on both sides of a fold):

    uv run python recipes/swe-verifier/analyze.py runs/swe-verifier/zero-shot/<file>.jsonl

- a logistic regression over the heuristics (the baseline);
- D1A's probability recalibrated by Platt scaling on its logit (calibration only; the ranking is D1A's);
- heuristics + D1A's logit together;
with AUROC, ECE, Brier and reliability bins, and 95% intervals for AUROC differences from a bootstrap over repositories.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zero_shot import auroc, calibration  # noqa: E402


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def heuristic_features(rows):
    return np.array([[np.log1p(r["patch_lines"]), r["clean_submit"], np.log1p(r["steps"]), np.log1p(r["error_obs"]), r["success_printed"], r["edits_tests"]]
                     for r in rows])


def cross_validated(X, y, groups, folds=5):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        p[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def repo_bootstrap(y, a, b, groups, n=2000, seed=0):
    """AUROC(a) - AUROC(b) and its 95% interval, resampling whole repositories."""
    rng = np.random.default_rng(seed); repos = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in repos}
    d = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(repos, len(repos))])
        d.append(auroc(y[s], a[s]) - auroc(y[s], b[s]))
    d = np.sort(d)
    return auroc(y, a) - auroc(y, b), d[int(0.025 * n)], d[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results"); ap.add_argument("--key", default="p_d1a", help="the probability field to analyse")
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    y = np.array([r["y"] for r in rows]); groups = np.array([r["repo"] for r in rows])
    raw = np.array([r[a.key] for r in rows]); H = heuristic_features(rows); D = logit(raw)[:, None]
    models = {"heuristics (LR)": cross_validated(H, y, groups), f"{a.key} raw": raw,
              f"{a.key} recalibrated (Platt)": cross_validated(D, y, groups), f"heuristics + {a.key}": cross_validated(np.hstack([H, D]), y, groups)}
    print(f"{len(rows)} runs, {y.sum()} resolved ({y.mean():.1%}), {len(np.unique(groups))} repositories; base-rate Brier {y.mean() * (1 - y.mean()):.3f}")
    for name, p in models.items():
        e, b, _ = calibration(y, p); print(f"  {name:32s} AUROC {auroc(y, p):.3f}  ECE {e:.3f}  Brier {b:.3f}")
    for x, z in ((f"{a.key} raw", "heuristics (LR)"), (f"heuristics + {a.key}", "heuristics (LR)")):
        d, lo, hi = repo_bootstrap(y, models[x], models[z], groups)
        print(f"  AUROC({x}) - AUROC({z}) = {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("  reliability, recalibrated (bin, n, mean p, resolved rate):", calibration(y, models[f"{a.key} recalibrated (Platt)"])[2])


if __name__ == "__main__":
    main()
