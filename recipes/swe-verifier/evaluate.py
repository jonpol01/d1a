"""Held-out evaluation of verifier scores: calibrate on dev, test once, compare with the heuristics.

    uv run python recipes/swe-verifier/evaluate.py --train-heuristics runs/swe-verifier/eval/heuristics-train.jsonl \
        --dev zero-shot=<dev file> --dev r1=<dev file> --test zero-shot=<test file> --test r1=<test file>

Each --dev/--test is NAME=zero_shot.py output (same seed and selection for every model, so rows line up). For each
model: its raw P, P recalibrated by Platt scaling on its logit (fitted on dev), and heuristics + its logit (fitted on
dev). The heuristics' logistic regression is fitted on --train-heuristics only. On test: AUROC, ECE, Brier,
reliability, AUROC differences against the heuristics with 95% intervals from a bootstrap over repositories, the
risk-coverage of "submit when P >= t", and seconds per run. --merged writes the test rows with every model's
calibrated P (p_<name>) for best_of_k.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import heuristic_features, logit, repo_bootstrap  # noqa: E402
from zero_shot import auroc, calibration  # noqa: E402


def load(path): return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def named(pairs):
    out = {}
    for p in pairs:
        name, _, path = p.partition("="); out[name] = load(path)
    return out


def risk_coverage(y, p, targets=(0.5, 0.6, 0.7, 0.8)):
    """For each precision target: the lowest threshold whose submitted runs reach it, and the share of runs submitted."""
    order = np.argsort(-p); ys = y[order]; prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    rows = []
    for t in targets:
        ok = np.flatnonzero(prec >= t)
        if len(ok): i = ok.max(); rows.append((t, round(float(p[order][i]), 3), round(float((i + 1) / len(y)), 3), int(ys[: i + 1].sum())))
        else: rows.append((t, None, 0.0, 0))
    return rows


def filter_purity(y, p, yields=(0.1, 0.2, 0.3)):
    """Rejection-sampling fine-tuning keeps the top share of runs by a score: per yield, the share of kept runs that
    truly resolved, and how many resolved runs were kept (ties broken by order)."""
    order = np.argsort(-p, kind="stable"); out = []
    for f in yields:
        keep = order[: max(1, int(round(f * len(y))))]
        out.append((f, round(float(y[keep].mean()), 3), int(y[keep].sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-heuristics", required=True); ap.add_argument("--dev", action="append", default=[]); ap.add_argument("--test", action="append", default=[])
    ap.add_argument("--merged", help="write test rows with calibrated p_<model> fields here")
    a = ap.parse_args()
    train = load(a.train_heuristics); dev, test = named(a.dev), named(a.test)
    first = next(iter(test.values()))
    for name, rows in test.items():
        assert [r["instance_id"] for r in rows] == [r["instance_id"] for r in first], f"{name}: test rows differ from the others"
    y = np.array([r["y"] for r in first]); groups = np.array([r["repo"] for r in first])
    heur = LogisticRegression(max_iter=1000).fit(heuristic_features(train), [r["y"] for r in train])
    scores = {"heuristics (LR)": heur.predict_proba(heuristic_features(first))[:, 1]}
    for name, rows in test.items():
        d = dev[name]; yd = np.array([r["y"] for r in d])
        pd, pt = np.array([r["p_d1a"] for r in d]), np.array([r["p_d1a"] for r in rows])
        platt = LogisticRegression(max_iter=1000).fit(logit(pd)[:, None], yd)
        combo = LogisticRegression(max_iter=1000).fit(np.hstack([heuristic_features(d), logit(pd)[:, None]]), yd)
        scores[f"{name} raw"] = pt
        scores[f"{name} calibrated"] = platt.predict_proba(logit(pt)[:, None])[:, 1]
        scores[f"heuristics + {name}"] = combo.predict_proba(np.hstack([heuristic_features(rows), logit(pt)[:, None]]))[:, 1]
    print(f"test: {len(y)} runs, {y.sum()} resolved ({y.mean():.1%}), {len(set(groups))} repositories (none in train or dev); base-rate Brier {y.mean() * (1 - y.mean()):.3f}")
    for name, p in scores.items():
        e, b, _ = calibration(y, p); print(f"  {name:28s} AUROC {auroc(y, p):.3f}  ECE {e:.3f}  Brier {b:.3f}")
    print("differences against the heuristics (95% CI, bootstrap over repositories):")
    for name, p in scores.items():
        if name == "heuristics (LR)": continue
        d, lo, hi = repo_bootstrap(y, p, scores["heuristics (LR)"], groups)
        print(f"  {name:28s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for name in [n for n in scores if n.endswith("calibrated") or n.startswith("heuristics")]:
        print(f"reliability {name}: {calibration(y, scores[name])[2]}")
        print(f"risk-coverage {name} (precision target, threshold, share submitted, resolved submitted): {risk_coverage(y, scores[name])}")
    print("fine-tuning data filter (yield kept, precision of kept runs, resolved runs kept):")
    filters = {"no filter (random)": np.random.default_rng(0).random(len(y)), "agent's clean submit": np.array([r["clean_submit"] for r in first], float)
               + 1e-3 * np.random.default_rng(1).random(len(y)), **scores}
    for name, p in filters.items():
        if name.endswith(" raw"): continue
        print(f"  {name:28s} {filter_purity(y, p)}")
    for name, rows in test.items():
        s = [r["seconds"] for r in rows if "seconds" in r]
        if s: print(f"seconds per run, {name}: median {np.median(s):.2f}, p95 {np.percentile(s, 95):.2f}")
    if a.merged:
        out = [{**{k: v for k, v in r.items() if k not in ("p_d1a", "seconds", "run")}, **{f"p_{n.replace(' ', '_')}": float(s[i]) for n, s in scores.items() if not n.endswith("raw")}}
               for i, r in enumerate(first)]
        Path(a.merged).write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")


if __name__ == "__main__":
    main()
