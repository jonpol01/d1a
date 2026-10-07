# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: kev/rounds.py's pooled_temperature_ci (Kev e52f812, #168) as interval(); its registered paired read (paired() over kev.metrics.paired_bootstrap, Kev 1dc2fbc, #112: micro Brier, 2,000 record-clustered resamples, seed 0, rows in (id, question) order) as primary(); round 28's registered temperature rule and confirmation stages (experiments/rounds/r28.json and PLAN.md, Kev ebc8024, #199) as rule() and confirmation(); as one module over d1a.eval.benchmark rows instead of round specs, arms and read-outs; panels are given per call (judge, guard, confirm, locked), not registered; rows keyed by their file, so D1A partitions' line ids never share a cluster, and a source name two panels share is resampled as two strata (Kev's one), the only case where primary()'s draw differs from Kev's on rows Kev's read accepts; primary() pairs each row with itself at the other temperature instead of matching (id, question), so it takes the rows of two partitions that share ids, ordered by (id, question) and then by keyed source; the confirmation stages run in the same call once the rule passes, instead of as a later `kev.rounds confirm`. Round 29, registered in the same Kev commit (a 9B checkpoint selection under round 24's audited rule), is not ported.
"""Whether a refitted temperature may replace the one a checkpoint ships (d1a.training.calibrate --judge, --guard, --confirm,
--locked).

A temperature fitted on a held-out pool is not better by construction: in Kev's round 28 read-out (Kev #203) one checkpoint's
pool fit was no better than its shipped temperature and the other's improved the pooled panels but got worse on three
in-domain suites. So the new temperature is adopted only when, on the same rows at both temperatures:
  - rule(): on the --judge panels pooled, the Brier difference's 95% paired cluster-bootstrap upper bound is below 0 (Brier
    is a proper score and adds up per question, so it carries the interval), and ECE is below the shipped temperature's (so
    a temperature cannot sharpen toward a better Brier and a worse reliability); on every --judge and --guard panel alone,
    ECE rises by at most TOLERANCE;
  - confirmation(), scored only once the rule passes (round 28's stages): on every --confirm panel (tests), ECE is below the
    shipped temperature's; on every --locked panel (locked), Brier rises by at most TOLERANCE and accuracy is identical.
The Brier interval is Kev's registered paired read: SAMPLES record-clustered resamples at seed SEED, the rows in (id, question)
order with every tie broken, so neither calibrate's --seed nor the order of the rows in a file moves a verdict. ECE is never optimised, only judged.
interval() is report only: the spread of the fit over cluster resamples, at the same SAMPLES and SEED.
"""
import json

import numpy as np

from d1a.eval.metrics import TEMPERATURE_FIT, brier, cluster_resamples, ece, nll_at_temperature, probabilities_at_temperature

GRID = np.exp(np.linspace(np.log(0.25), np.log(4), TEMPERATURE_FIT["points"]))   # fit_temperature's grid for TEMPERATURE_FIT
TOLERANCE = 0.005   # round 28: how far a panel's ECE (judge, guard) or a locked panel's Brier may rise under the new temperature
SAMPLES = 2000      # kev/rounds.py SAMPLES: the registered resample count
SEED = 0            # kev/rounds.py paired(): the registered read's seed, so calibrate's --seed (its folds) never moves a verdict


def keyed(rows, panel):
    """The rows with source and group prefixed by their panel. Rows of a D1A partition are all source "custom" and group
    "custom/<line>", so two partitions pooled would otherwise share clusters between unrelated records."""
    return [{**row, "source": f"{panel}|{row['source']}", "group": f"{panel}|{row['group']}"} for row in rows]


def interval(rows, samples=SAMPLES, level=0.9):
    """The percentile interval of the temperature fitted (TEMPERATURE_FIT's grid and objective) on cluster resamples of
    raw scored rows. Each question's NLL on the grid is computed once; a resample's fit is the grid point with the least
    mean NLL over its questions."""
    nll = np.asarray([[nll_at_temperature(row, float(t)) for t in GRID] for row in rows])
    fits = [GRID[int(np.argmin(nll[idx].mean(axis=0)))] for idx in cluster_resamples(rows, samples, SEED)]
    lower, upper = np.quantile(fits, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"level": level, "samples": samples, "seed": SEED, "lower": float(lower), "upper": float(upper), "questions": len(rows)}


def served(rows, temperature):
    """(top probability, correct, Brier) per row at `temperature`."""
    p = [probabilities_at_temperature(row, temperature) for row in rows]
    return (np.asarray([float(q.max()) for q in p]), np.asarray([q.argmax() == row["label"] for q, row in zip(p, rows)]),
            np.asarray([brier(q, row["label"]) for q, row in zip(p, rows)]))


def sides(rows, candidate, shipped):
    """{n, acc, ece, brier}: each metric of the same rows at the candidate and the shipped temperature. An empty panel is
    refused, as Kev's metrics() refuses an empty population: its ECE, 0 at both temperatures, would pass any ECE check."""
    if not rows:
        raise ValueError("cannot judge a temperature on an empty panel")
    out = {"n": len(rows)}
    for side, temperature in (("candidate", candidate), ("shipped", shipped)):
        top, right, scores = served(rows, temperature)
        for metric, value in (("acc", float(right.mean())), ("ece", ece(top, right)), ("brier", float(scores.mean()))):
            out.setdefault(metric, {})[side] = value
    return out


def primary(rows, candidate, shipped, samples=SAMPLES):
    """sides() of the rows, with the candidate minus shipped Brier and its 95% paired cluster-bootstrap interval."""
    # paired_bootstrap's (id, question) order, so the draw is Kev's when ids are unique; the keyed source breaks ties between
    # partitions sharing custom/<line> ids, and the group and the row itself the rest (devtools-v1 repeats ids across groups,
    # which Kev's read refuses): a total order, so no order of the rows in a file moves the bound
    rows = sorted(rows, key=lambda row: (row["id"], row["question"], row["source"], row["group"], json.dumps(row, sort_keys=True, default=str)))
    delta = served(rows, candidate)[2] - served(rows, shipped)[2]
    draws = [float(delta[idx].mean()) for idx in cluster_resamples(rows, samples, SEED)]
    return {**sides(rows, candidate, shipped), "brier_delta": float(delta.mean()), "brier_ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def rule(judge, guard, candidate, shipped, samples=SAMPLES):
    """The verdict on `candidate` against `shipped`. judge, guard: {panel name: raw scored rows, keyed}; the two are checked
    apart, so a guard named like a judge panel never stands in for its check."""
    pooled = primary([row for rows in judge.values() for row in rows], candidate, shipped, samples)
    panels = {role: {name: sides(rows, candidate, shipped) for name, rows in given.items()} for role, given in (("judge", judge), ("guard", guard))}
    failed = [] if pooled["brier_ci95"][1] < 0 else [f"judge panels pooled: Brier difference upper bound {pooled['brier_ci95'][1]:+.4f} is not below 0"]
    if not pooled["ece"]["candidate"] < pooled["ece"]["shipped"]:
        failed.append(f"judge panels pooled: ECE {pooled['ece']['candidate']:.4f} is not below the shipped {pooled['ece']['shipped']:.4f}")
    failed += [f"{role} {name}: ECE {s['ece']['candidate']:.4f} > shipped {s['ece']['shipped']:.4f} + {TOLERANCE}"
               for role, named in panels.items() for name, s in named.items() if s["ece"]["candidate"] > s["ece"]["shipped"] + TOLERANCE]
    return {"adopt": not failed, "failed": failed, "candidate": candidate, "shipped": shipped, "tolerance": TOLERANCE,
            "judge": pooled, "panels": panels, "samples": samples, "seed": SEED}


def confirmation(tests, locked, candidate, shipped):
    """Round 28's confirmation stages, for a candidate that passed the rule. tests, locked: {panel name: raw scored rows}.
    Temperature never moves an argmax, so a locked panel's accuracy differs only when its recorded p disagree with its logits."""
    stages = {stage: {name: sides(rows, candidate, shipped) for name, rows in given.items()} for stage, given in (("tests", tests), ("locked", locked))}
    failed = [f"confirm {name}: ECE {s['ece']['candidate']:.4f} is not below the shipped {s['ece']['shipped']:.4f}"
              for name, s in stages["tests"].items() if not s["ece"]["candidate"] < s["ece"]["shipped"]]
    for name, s in stages["locked"].items():
        if s["brier"]["candidate"] > s["brier"]["shipped"] + TOLERANCE:
            failed.append(f"locked {name}: Brier {s['brier']['candidate']:.4f} > shipped {s['brier']['shipped']:.4f} + {TOLERANCE}")
        if s["acc"]["candidate"] != s["acc"]["shipped"]:
            failed.append(f"locked {name}: accuracy {s['acc']['candidate']:.4f} is not the shipped {s['acc']['shipped']:.4f}")
    return {"passed": not failed, "failed": failed, **stages}
