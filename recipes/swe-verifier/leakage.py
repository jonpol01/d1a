"""How much does the train/test split inflate a learned coding-agent verifier?

    uv run python recipes/swe-verifier/leakage.py --shards 12 --per-issue 10 --out runs/swe-verifier/leakage

Agent runs of the same issue share most of their outcome (some issues are easy, some impossible), so a verifier
evaluated with runs of one issue on both sides of the split can score well by recognising the issue, not by judging the
run. Three 5-fold protocols on the same runs: random runs, grouped by issue, grouped by repository (the setting a
verifier meets on new code). Verifiers, all refitted per fold:
- issue prior: the resolved rate of the same issue's runs in the training folds (the global rate when it has none);
  pure leakage by construction;
- heuristics: logistic regression on cheap trajectory features (zero_shot.heuristics);
- text: TF-IDF of the issue and the patch, logistic regression;
- heuristics + text.
Reports per protocol: AUROC, ECE and Brier of the out-of-fold predictions; AUROC inflation (random or by-issue minus
by-repository) with a 95% interval from a bootstrap over repositories; and within-issue AUROC (runs of one issue
ranked against each other, averaged over issues with both outcomes), the quantity best-of-k selection depends on and an
issue prior cannot improve. Data: nebius/SWE-agent-trajectories (CC-BY-4.0); eval_logs never read.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import heuristic_features  # noqa: E402
from zero_shot import REVISION, auroc, calibration, heuristics, issue_of, shard  # noqa: E402

COLS = ["instance_id", "target", "trajectory", "exit_status", "generated_patch"]


def load(shards, per_issue, seed, revision=REVISION):
    """One compact row per run (features and text; the trajectory itself is dropped), at most per_issue runs per issue."""
    rng = random.Random(seed); by = {}
    for k in range(shards):
        for r in pq.read_table(shard(k, revision), columns=COLS).to_pylist():
            by.setdefault(r["instance_id"], []).append({"issue": r["instance_id"], "repo": r["instance_id"].rsplit("-", 1)[0], "y": bool(r["target"]),
                                                        **heuristics(r), "text": issue_of(r["trajectory"])[:3000] + "\n" + (r["generated_patch"] or "")[:6000]})
        print(f"shard {k}: {sum(map(len, by.values()))} runs, {len(by)} issues", file=sys.stderr, flush=True)
    rows = [r for iid in sorted(by) for r in rng.sample(by[iid], min(per_issue, len(by[iid])))]
    return rows


def folds(protocol, rows, k=5, seed=0):
    idx = np.arange(len(rows))
    if protocol == "random runs": return list(KFold(k, shuffle=True, random_state=seed).split(idx))
    key = "issue" if protocol == "by issue" else "repo"
    return list(GroupKFold(k).split(idx, groups=[r[key] for r in rows]))


def out_of_fold(rows, protocol, y):
    H = heuristic_features(rows)
    p = {name: np.zeros(len(rows)) for name in ("issue prior", "heuristics", "text", "heuristics + text")}
    for tr, te in folds(protocol, rows):
        rate, n = {}, {}
        for i in tr: rate[rows[i]["issue"]] = rate.get(rows[i]["issue"], 0) + y[i]; n[rows[i]["issue"]] = n.get(rows[i]["issue"], 0) + 1
        base = y[tr].mean()
        p["issue prior"][te] = [rate[rows[i]["issue"]] / n[rows[i]["issue"]] if rows[i]["issue"] in n else base for i in te]
        p["heuristics"][te] = LogisticRegression(max_iter=2000).fit(H[tr], y[tr]).predict_proba(H[te])[:, 1]
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        T_tr = vec.fit_transform([rows[i]["text"] for i in tr]); T_te = vec.transform([rows[i]["text"] for i in te])
        p["text"][te] = LogisticRegression(max_iter=2000, C=1.0).fit(T_tr, y[tr]).predict_proba(T_te)[:, 1]
        scale = H[tr].std(0) + 1e-9; Hs_tr, Hs_te = csr_matrix(H[tr] / scale), csr_matrix(H[te] / scale)
        p["heuristics + text"][te] = LogisticRegression(max_iter=2000).fit(hstack([T_tr, Hs_tr]).tocsr(), y[tr]).predict_proba(hstack([T_te, Hs_te]).tocsr())[:, 1]
    return p


def within_issue_auroc(rows, y, p):
    """Mean AUROC over issues that have both a resolved and an unresolved run: ranking runs of the same issue."""
    by = {}
    for i, r in enumerate(rows): by.setdefault(r["issue"], []).append(i)
    vals = [auroc(y[ix], p[ix]) for ix in map(np.array, by.values()) if 0 < y[ix].sum() < len(ix)]
    return float(np.mean(vals)), len(vals)


def repo_bootstrap_diff(y, a, b, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed); keys = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in keys}
    d = np.sort([auroc(y[s], a[s]) - auroc(y[s], b[s]) for s in (np.concatenate([idx[g] for g in rng.choice(keys, len(keys))]) for _ in range(n))])
    return auroc(y, a) - auroc(y, b), d[int(0.025 * n)], d[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", type=int, default=12); ap.add_argument("--per-issue", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=REVISION, help="the dataset revision (default: zero_shot.REVISION)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = load(a.shards, a.per_issue, a.seed, a.revision); y = np.array([r["y"] for r in rows], float); groups = np.array([r["repo"] for r in rows])
    print(f"{len(rows)} runs, {len({r['issue'] for r in rows})} issues, {len(set(groups))} repositories, {y.mean():.1%} resolved")
    preds = {proto: out_of_fold(rows, proto, y) for proto in ("random runs", "by issue", "by repo")}
    result = {"runs": len(rows), "issues": len({r["issue"] for r in rows}), "repos": len(set(groups)), "resolved": float(y.mean()), "protocols": {}}
    for proto, p in preds.items():
        print(f"\n{proto}:")
        result["protocols"][proto] = {}
        for name, q in p.items():
            e, b, _ = calibration(y, q); w, nw = within_issue_auroc(rows, y, q)
            result["protocols"][proto][name] = {"auroc": round(auroc(y, q), 4), "ece": round(e, 4), "brier": round(b, 4), "within_issue_auroc": round(w, 4)}
            print(f"  {name:18s} AUROC {auroc(y, q):.3f}  ECE {e:.3f}  Brier {b:.3f}  within-issue AUROC {w:.3f} ({nw} issues)")
    print("\nAUROC inflation against the by-repository split (95% CI, bootstrap over repositories):")
    result["inflation"] = {}
    for name in preds["by repo"]:
        for proto in ("random runs", "by issue"):
            d, lo, hi = repo_bootstrap_diff(y, preds[proto][name], preds["by repo"][name], groups)
            result["inflation"][f"{name} | {proto}"] = [round(float(d), 4), round(float(lo), 4), round(float(hi), 4)]
            print(f"  {name:18s} {proto:12s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(out / "predictions.npz", y=y, repo=groups, issue=np.array([r["issue"] for r in rows]),
                        **{f"{proto}|{name}": q for proto, p in preds.items() for name, q in p.items()})


if __name__ == "__main__":
    main()
