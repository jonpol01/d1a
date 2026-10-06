"""Compare checkpoints on the suites they were scored on (d1a.eval.benchmark output folders), for a training run's
verdict (#167, #175): per suite, each run's accuracy and its paired difference from a reference with a 95% bootstrap
interval over records, ECE at each checkpoint's fitted temperature, and a pre-registered bar.

    python scripts/compare_checkpoints.py --ref v0.4=scores/v0.4 --run B0=scores/b0-300 --run skills-v1=scores/skills-v1 \\
        --fit B0=scores/b0-300/cal-dv7,scores/b0-300/cal-pr --bar hard-v1=3 --bar devtools-v1=3 --floor -1

Each run folder holds one subfolder per suite with the rows.json d1a.eval.benchmark writes; the suites compared are those
every run has (calibration folders, cal-*, are not suites). Accuracy is on d1a.eval.metrics.scored_rows (clean, knowable),
the rows the benchmark's own report scores. ECE is at the temperature fitted on a run's --fit rows (fit_temperature with
TEMPERATURE_FIT, the fit d1a.training.calibrate makes), else at the temperature its rows were served at. Differences are
paired: only questions both runs answered, resampled by record, since a record's questions share a state. The bar is on
point estimates in points (--bar SUITE=MIN for the suites the run must raise, --floor for every other suite), as #167
pre-registered it. Exit 0 when every run passes the bar (or none is given), 1 otherwise.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d1a.eval.metrics import (
    TEMPERATURE_FIT,
    fit_temperature,
    metrics,
    raw_row,
    recorded,
    scored_rows,
)
from d1a.eval.suite import read_json, write_json


def load(folder):
    """A suite folder's scored rows, as they were served."""
    return scored_rows(read_json(Path(folder) / "rows.json"))


def suites(folder):
    return {p.name for p in Path(folder).iterdir() if (p / "rows.json").exists() and not p.name.startswith("cal-")}


def correct(row):
    p = row["p"]
    return int(max(range(len(p)), key=p.__getitem__) == int(row["label"]))


def paired(ref, run, seed=0, draws=2000):
    """{n, ref, run, delta, lo, hi}: accuracy of each on the questions both answered, the difference (run - ref) and its 95%
    bootstrap interval, records resampled with all their questions."""
    a, b = {(r["id"], r["question"]): r for r in ref}, {(r["id"], r["question"]): r for r in run}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no question was answered by both runs")
    per = defaultdict(lambda: [0, 0, 0])   # record id -> [questions, ref correct, run correct]
    for key in common:
        s = per[key[0]]; s[0] += 1; s[1] += correct(a[key]); s[2] += correct(b[key])
    stats = np.array(list(per.values()), dtype=float)
    n, ra, rb = stats.sum(0)
    picks = np.random.default_rng(seed).integers(0, len(stats), (draws, len(stats)))
    boot = (stats[picks, 2].sum(1) - stats[picks, 1].sum(1)) / stats[picks, 0].sum(1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n": int(n), "ref": ra / n, "run": rb / n, "delta": (rb - ra) / n, "lo": float(lo), "hi": float(hi)}


def raw(rows):
    return [raw_row(recorded(r)) for r in rows]


def fitted_temperature(folders):
    """The temperature d1a.training.calibrate would fit on these folders' rows, pooled."""
    return fit_temperature([r for f in folders for r in raw(load(f))], **TEMPERATURE_FIT)


def served_temperature(rows):
    return recorded(rows[0])["inference_temperature"] if rows else 1.0


def ece(rows, temperature):
    return metrics(raw(rows), temperature)["ece"]


def bar_failures(results, bar, floor):
    """Each run's failures of the bar: a --bar suite below its minimum, any other suite below --floor (points)."""
    out = {}
    for run, per_suite in results.items():
        fails = []
        for suite, r in per_suite.items():
            need = bar.get(suite, floor)
            if need is not None and 100 * r["delta"] < need:
                fails.append(f"{suite}: {100 * r['delta']:+.2f} < {need:+g}")
        fails += [f"{suite}: not scored" for suite in bar if suite not in per_suite]
        out[run] = fails
    return out


def named(values):
    out = {}
    for v in values:
        name, sep, path = v.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=PATH, got {v!r}")
        out[name] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="NAME=FOLDER: the checkpoint every run is compared with")
    ap.add_argument("--run", action="append", required=True, help="NAME=FOLDER; repeatable")
    ap.add_argument("--fit", action="append", default=[], help="NAME=FOLDER[,FOLDER]: calibration rows to fit NAME's temperature on")
    ap.add_argument("--bar", action="append", default=[], help="SUITE=MIN: the least gain in points a run must show on SUITE")
    ap.add_argument("--floor", type=float, help="the least change in points on every suite not in --bar")
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args(argv)
    (ref_name, ref_dir), runs = next(iter(named([a.ref]).items())), named(a.run)
    fit = {k: v.split(",") for k, v in named(a.fit).items()}
    bar = {k: float(v) for k, v in named(a.bar).items()}
    common = sorted(set.intersection(suites(ref_dir), *(suites(d) for d in runs.values())))
    if not common:
        raise SystemExit("no suite was scored for every run")
    temps = {name: (fitted_temperature(fit[name]) if name in fit else None) for name in [ref_name, *runs]}
    results, eces = {name: {} for name in runs}, {}
    for suite in common:
        ref_rows = load(Path(ref_dir) / suite)
        t_ref = temps[ref_name] or served_temperature(ref_rows)
        eces[(ref_name, suite)] = ece(ref_rows, t_ref)
        for name, folder in runs.items():
            rows = load(Path(folder) / suite)
            results[name][suite] = paired(ref_rows, rows)
            eces[(name, suite)] = ece(rows, temps[name] or served_temperature(rows))
    head = f"| suite | n | {ref_name} |" + "".join(f" {name} (Δ, 95% CI) |" for name in runs) + f" ECE {ref_name} |" + "".join(f" ECE {name} |" for name in runs)
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for suite in common:
        first = next(iter(results.values()))[suite]
        cells = "".join(f" {100 * r['run']:.2f} ({100 * r['delta']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]) |"
                        for r in (results[name][suite] for name in runs))
        print(f"| {suite} | {first['n']} | {100 * first['ref']:.2f} |{cells} {eces[(ref_name, suite)]:.4f} |"
              + "".join(f" {eces[(name, suite)]:.4f} |" for name in runs))
    folders = {ref_name: ref_dir, **runs}
    shown = {name: temps[name] or served_temperature(load(Path(folders[name]) / common[0])) for name in folders}
    print("temperatures: " + ", ".join(f"{name} {'fitted' if temps[name] else 'served at'} {t:.3f}" for name, t in shown.items()))
    failures = bar_failures(results, bar, a.floor) if (bar or a.floor is not None) else {}
    for name, fails in failures.items():
        print(f"bar {name}: " + ("PASS" if not fails else "FAIL (" + "; ".join(fails) + ")"))
    if a.json:
        write_json(Path(a.json), {"ref": ref_name, "suites": common, "results": results, "temperatures": temps,
                                  "ece": {f"{k[0]}/{k[1]}": v for k, v in eces.items()}, "bar": failures})
    return 1 if any(failures.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
