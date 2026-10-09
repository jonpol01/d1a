"""Does refitting the heuristics on other rows help? The heuristics' logistic regression fitted on --train, on --dev and on
both, each scored on the same held-out rows.

    uv run python recipes/swe-verifier/refit_heuristics.py --train runs/swe-verifier/eval/heuristics-train.jsonl \
        --dev <dev file> --test <test file> [--mixed <--mixed-only file>]

Every file is a zero_shot.py output (D1A's scores are not used). The model is evaluate.py's: analyze.heuristic_features,
LogisticRegression(max_iter=1000). On --test: AUROC, within-issue AUROC, ECE, Brier, and each refit's AUROC difference
against the --train fit with a 95% interval from a bootstrap over repositories (analyze.repo_bootstrap). On --mixed:
within-issue AUROC with a 95% interval from a bootstrap over issues (as within_issue.py), the difference against the
--train fit, and global AUROC. CPU only, no model; prints aggregates only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import heuristic_features, repo_bootstrap  # noqa: E402
from evaluate import within_issue  # noqa: E402
from within_issue import per_issue  # noqa: E402
from zero_shot import auroc, calibration  # noqa: E402


def load(path): return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def ci(v, rng, boot):
    b = np.sort([v[rng.integers(0, len(v), len(v))].mean() for _ in range(boot)])
    return v.mean(), b[int(0.025 * boot)], b[int(0.975 * boot) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--mixed", help="a zero_shot.py --mixed-only output, for within-issue AUROC")
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train, dev = load(a.train), load(a.dev)
    fits = {"train": train, "dev": dev, "train + dev": train + dev}
    models = {n: LogisticRegression(max_iter=1000).fit(heuristic_features(r), [x["y"] for x in r]) for n, r in fits.items()}
    print("fitted on: " + "; ".join(f"{n} {len(r)} runs, {np.mean([x['y'] for x in r]):.1%} resolved" for n, r in fits.items()))

    test = load(a.test)
    y = np.array([r["y"] for r in test]); g = np.array([r["repo"] for r in test]); iss = np.array([r["instance_id"] for r in test])
    p = {n: m.predict_proba(heuristic_features(test))[:, 1] for n, m in models.items()}
    print(f"\ntest: {len(y)} runs, {int(y.sum())} resolved ({y.mean():.1%}), {len(set(g))} repositories")
    for n, s in p.items():
        e, b, _ = calibration(y, s); w, nw = within_issue(y, s, iss)
        print(f"  fitted on {n:12s} AUROC {auroc(y, s):.3f}  within-issue AUROC {w:.3f} ({nw} issues)  ECE {e:.3f}  Brier {b:.3f}")
    print(f"  AUROC difference against the train fit (95% CI, bootstrap over repositories, n={a.boot}, seed {a.seed}):")
    for n in ("dev", "train + dev"):
        d, lo, hi = repo_bootstrap(y, p[n], p["train"], g, n=a.boot, seed=a.seed)
        print(f"    {n:12s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    if a.mixed:
        mixed = load(a.mixed)
        ym = np.array([r["y"] for r in mixed], float); im = np.array([r["instance_id"] for r in mixed])
        pm = {n: m.predict_proba(heuristic_features(mixed))[:, 1] for n, m in models.items()}
        per = {n: per_issue(ym, s, im) for n, s in pm.items()}
        keys = sorted(per["train"]); rng = np.random.default_rng(a.seed)
        print(f"\nmixed: {len(ym)} runs, {len(set(im))} issues, {len(keys)} with both outcomes, {ym.mean():.1%} resolved")
        for n, d in per.items():
            m, lo, hi = ci(np.array([d[k] for k in keys]), rng, a.boot)
            print(f"  fitted on {n:12s} within-issue AUROC {m:.3f} [{lo:.3f}, {hi:.3f}]   global AUROC {auroc(ym, pm[n]):.3f}")
        base = np.array([per["train"][k] for k in keys])
        for n in ("dev", "train + dev"):
            m, lo, hi = ci(np.array([per[n][k] for k in keys]) - base, rng, a.boot)
            print(f"  {n} - train fit, within-issue: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
