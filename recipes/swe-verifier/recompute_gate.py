"""The promotion gate recomputed offline from scored files: would d1a.learning.feedback.gate promote the candidate?

    uv run python recipes/swe-verifier/recompute_gate.py --incumbent <dev file> --candidate <dev file> [--per-issue 4] \
        [--loop-shards 0,1,2 --loop-per-issue 4]

--incumbent and --candidate are zero_shot.py outputs of the same selection (rows line up by run_key). Each model's
P(resolved) is recalibrated out of fold (GroupKFold by repository, so no repository calibrates its own rows), two ways:
Platt scaling on the logit (sklearn) and d1a.learning.feedback.OutcomeCalibrator, the calibrator self_improve.py serves
(min_outcomes 10). On each, gate(): the candidate's log loss minus the incumbent's, with a 95% interval from a bootstrap
over repositories. Optional extra sets, from the same scored rows:
- --per-issue N: a seeded subsample of at most N rows per issue;
- --loop-shards: self_improve.py's own dev set (self_improve.runs(shards, "dev", --loop-per-issue, --seed)), rebuilt from
  the public dataset at --revision and joined to the scored rows by run_key; when some of its runs have no scored row,
  it says so instead of gating a different set.
CPU only, no model; prints aggregates only.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import logit  # noqa: E402
from zero_shot import REVISION, run_key  # noqa: E402

from d1a.learning.feedback import OutcomeCalibrator, gate, log_loss  # noqa: E402


def load(path): return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def out_of_fold(p, y, groups, how, folds):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(p, y, groups):
        if how == "Platt (sklearn)":
            m = LogisticRegression(max_iter=1000).fit(logit(p[tr])[:, None], y[tr]); out[te] = m.predict_proba(logit(p[te])[:, None])[:, 1]
        else:
            c = OutcomeCalibrator().fit([{"labels": {"resolved": bool(t)}, "answers": {"resolved": {"noul": float(q)}}} for t, q in zip(y[tr], p[tr])], min_outcomes=10)
            out[te] = [c.p("resolved", float(q)) for q in p[te]]
    return out


def report(name, y, groups, p_inc, p_cand, folds, boot, seed):
    print(f"{name}: {len(y)} runs, {y.mean():.1%} resolved, {len(set(groups))} repositories")
    for how in ("Platt (sklearn)", "d1a OutcomeCalibrator"):
        i, c = out_of_fold(p_inc, y, groups, how, folds), out_of_fold(p_cand, y, groups, how, folds)
        g = gate(y, c, i, groups=groups, n=boot, seed=seed)
        print(f"  {how:22s} log loss incumbent {log_loss(y, i).mean():.4f} -> candidate {log_loss(y, c).mean():.4f}; "
              f"delta {g['delta_log_loss']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}] promote={g['promote']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--incumbent", required=True); ap.add_argument("--candidate", required=True)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-issue", type=int, default=0, help="also on a seeded subsample of at most this many rows per issue")
    ap.add_argument("--loop-shards", help="also on self_improve.py's dev set from these dataset shards (e.g. 0,1,2)")
    ap.add_argument("--loop-per-issue", type=int, default=4); ap.add_argument("--revision", default=REVISION, help="the dataset revision for --loop-shards")
    a = ap.parse_args()
    inc, cand = load(a.incumbent), load(a.candidate)
    assert [r["run_key"] for r in inc] == [r["run_key"] for r in cand], "the two files are not the same selection"
    y = np.array([r["y"] for r in inc], float); g = np.array([r["repo"] for r in inc])
    pi, pc = np.array([r["p_d1a"] for r in inc]), np.array([r["p_d1a"] for r in cand])
    report("all scored rows", y, g, pi, pc, a.folds, a.boot, a.seed)

    if a.per_issue:
        by = {}
        for i, r in enumerate(inc): by.setdefault(r["instance_id"], []).append(i)
        rng = random.Random(a.seed)
        ix = np.array(sorted(j for iid in sorted(by) for j in rng.sample(by[iid], min(a.per_issue, len(by[iid])))))
        report(f"\nat most {a.per_issue} rows per issue (seed {a.seed})", y[ix], g[ix], pi[ix], pc[ix], a.folds, a.boot, a.seed)

    if a.loop_shards:
        from self_improve import runs
        loop = runs([int(x) for x in a.loop_shards.split(",")], "dev", a.loop_per_issue, a.seed, a.revision)
        keys = [run_key(r) for r in loop]
        at = {}
        for i, r in enumerate(inc): at.setdefault(r["run_key"], []).append(i)
        hit = [k for k in keys if k in at]
        amb = sum(1 for k in set(hit) if len({(pi[i], pc[i]) for i in at[k]}) > 1)   # a run scored twice must score the same
        print(f"\nself_improve.py's dev set: {len(loop)} runs, {len({r['instance_id'] for r in loop})} issues; {len(hit)} have a scored row "
              f"({amb} run_keys with scored rows that disagree)")
        if len(hit) < len(loop):
            print("  -> not gated: score the missing runs first (this script runs no model)")
        else:
            ix = np.array([at[k][0] for k in keys])
            report("self_improve.py's dev set", y[ix], g[ix], pi[ix], pc[ix], a.folds, a.boot, a.seed)


if __name__ == "__main__":
    main()
