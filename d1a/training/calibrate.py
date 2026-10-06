"""Calibrate a checkpoint: fit one temperature on held-out scored rows and write it into the checkpoint (d1a_config.json, and the head.pt D1A 0.3 reads), so
every loader serves calibrated probabilities. Argmax never changes, so accuracy is the same before and after.

    python -m d1a.training.calibrate --run runs/new --rows runs/cal/rows.json --rows runs/calpr/rows.json:src_a,src_b

--rows are d1a.eval.benchmark rows.json files, pooled; `path:source,...` keeps only those sources (a source the rows' suite
does not list is refused: a typo would silently shrink the fit set). --exclude_rows drops records by id. --transfer rows
are reported before and after, never fitted. --temperature T writes a value without fitting and needs --reason.

The fit rows must be held out from the checkpoint's training. A temperature fitted on held-out items of the checkpoint's
own training data is in distribution and comes out overconfident on anything else (Kev's round 19: T 0.955 on its own
development rows gave ECE 0.059 elsewhere; a pool of held-out datasets gave 0.0085). So the fit is refused when a rows
file (a) comes from a suite the checkpoint trained on, (b) pools a source it trained on, or (c) reads the calibration or
development partition of a training corpus, and when a rows file's suite or the checkpoint's training cannot be placed.
--allow-in-distribution fits anyway, warns, and records every reason in the checkpoint's temperature_fit.in_distribution.

What the checkpoint trained on comes from its metadata (d1a.training.train records the suite's manifest hash, its args and --data) or a
provenance.json beside it; where rows came from, from the report.json d1a.eval.benchmark writes beside them.
"""
import argparse
import functools
from dataclasses import dataclass
from pathlib import Path

from d1a.backends.checkpoint import read_meta, write_meta
from d1a.eval.metrics import TEMPERATURE_FIT, TEMPERATURE_FIT_METHOD, cross_validated_temperature, fit_temperature, metrics, raw_row, recorded, scored_rows
from d1a.eval.suite import digest, read_json, read_manifest, suite_key

ROOT = Path(__file__).resolve().parents[2]   # the repository: d1a/training/calibrate.py
HELD_OUT_SPLITS = ("calibration", "development")   # a training corpus's own held-out partitions: in distribution


@functools.cache
def _suites():
    """({manifest sha256: suite dir}, {suite dir name: [suite dirs]}) of every suite in this checkout."""
    by_digest, by_name = {}, {}
    for m in sorted((ROOT / "evals").rglob("manifest.json")):
        d = str(m.parent.relative_to(ROOT))
        by_digest[digest(m)] = d
        by_name.setdefault(m.parent.name, []).append(d)
    return by_digest, by_name


def manifest(suite):
    return read_manifest(ROOT / suite) if suite and (ROOT / suite / "manifest.json").exists() else None


def sources(m):
    """Every source a manifest names (trainable, held-out, eval-only, and its `sources` table)."""
    return {*m.get("trainable_sources", []), *m.get("holdout_sources", []), *m.get("eval_only_sources", []), *m.get("sources", [])}


def trainable(m):
    """The sources a suite offers for training; a suite with any is a training corpus."""
    table = m.get("sources", {})
    marked = {k for k, v in table.items() if isinstance(v, dict) and v.get("trainable")} if isinstance(table, dict) else set()
    return {*m.get("trainable_sources", []), *marked}


@dataclass(frozen=True)
class Training:
    """What a checkpoint trained on, as far as this checkout can tell."""
    suites: frozenset     # its training suite, the components that suite names, its --data file's suite
    sources: frozenset    # the sources those suites train
    unplaced: tuple       # training data whose sources this checkout cannot list


def training_of(suite_sha256=None, suite=None, data=None):
    """Training from what a trainer recorded, or None when no suite of this checkout matches. A --data file counts
    through its directory's suite; one outside evals/ cannot be checked, so it is unplaced, never dropped."""
    start = _suites()[0].get(suite_sha256) or (suite_key(suite) if suite and manifest(suite_key(suite)) else None)
    if start is None: return None
    data_suite = suite_key(Path(data).parent) if data else None
    seen, found, unplaced = set(), set(), [] if data_suite or not data else [f"--data {data} (outside evals/)"]
    todo = [d for d in (start, data_suite) if d]
    while todo:
        d = todo.pop(0)
        if d in seen: continue
        seen.add(d)
        m = manifest(d)
        own = (trainable(m) if "trainable_sources" in m else sources(m)) if m is not None else set()
        if not own: unplaced.append(d)
        found |= own
        inputs = (m or {}).get("inputs")
        for component in inputs.get("components", {}) if isinstance(inputs, dict) else ():   # e.g. documents-v1-train -> evals/documents-v1
            dirs = _suites()[1].get(component.removesuffix("-train").removesuffix("-extra"))
            if dirs: todo += dirs
            else: unplaced.append(f"{d} component {component}")
    return Training(frozenset(seen), frozenset(found), tuple(unplaced))


def checkpoint_training(run):
    """Training of a checkpoint: what d1a.training.train recorded in its metadata, else the provenance.json beside it."""
    extra = read_meta(run).extra
    args = extra.get("args") or {}
    training = training_of(extra.get("suite_sha256"), args.get("suite"), args.get("data"))
    provenance = Path(run).parent / "provenance.json"
    if training is None and provenance.exists():
        p = read_json(provenance)
        training = training_of(p.get("suite_sha256"), data=p.get("config", {}).get("data"))
    return training


def rows_origin(path):
    """(suite dir, partition) a rows.json was scored on, from the d1a.eval.benchmark report.json beside it; (None, None) for
    rows it does not place (custom --data rows, a suite this checkout lacks)."""
    report = Path(path).parent / "report.json"
    if report.exists() and "suite_sha256" in (r := read_json(report)):
        return _suites()[0].get(r["suite_sha256"]), r.get("split", "development")
    return None, None


def parse_rows(values):
    """--rows values (`path` or `path:source,...`) -> [(path, [sources] or None)]."""
    return [(path, srcs.split(",") if srcs else None) for path, _, srcs in (v.partition(":") for v in values)]


def select(reads, exclude=()):
    """Every row of the reads, each limited to its sources, minus the records whose id an exclude rows.json holds."""
    excluded = {r["id"] for path in exclude for r in read_json(path)}
    return [r for path, srcs in reads for r in read_json(path) if (srcs is None or r["source"] in srcs) and r["id"] not in excluded]


def allowlist_typos(reads):
    """A message per `path:source,...` naming a source its suite (or, unplaced, the rows file itself) does not contain."""
    out = []
    for path, srcs in reads:
        if not srcs: continue
        suite = rows_origin(path)[0]
        known = sources(manifest(suite)) if suite else {r["source"] for r in read_json(path)}
        if missing := sorted(set(srcs) - known): out.append(f"{path}: {missing} not among the sources of {suite or 'these rows'} {sorted(known)[:10]}")
    return out


def in_distribution(fitted, training, run):
    """Every reason the fit set is not held out from the checkpoint's training (empty = held out)."""
    out = [f"{f['rows']}: cannot tell which suite these rows were scored on (no d1a.eval.benchmark report.json with a suite of this checkout beside them)"
           for f in fitted if f["suite"] is None]
    if training is None:
        return out + [f"{run}: cannot tell what this checkpoint was trained on (its metadata names no suite of this checkout, no provenance.json beside it)"]
    for f in fitted:
        if f["suite"] is None: continue
        d, m = f["suite"], manifest(f["suite"])
        if m is None: out.append(f"{f['rows']}: its suite {d} has no manifest in this checkout"); continue
        if d in training.suites: out.append(f"{f['rows']}: {d} is training data of the checkpoint")
        pooled = set(f.get("sources") or sources(m))
        if not pooled: out.append(f"{f['rows']}: {d} lists no sources; give the rows a source allowlist")
        if shared := sorted(pooled & training.sources): out.append(f"{f['rows']}: it pools {len(shared)} training source(s) {shared[:8]}" + (" ..." if len(shared) > 8 else ""))
        if f["split"] in HELD_OUT_SPLITS and trainable(m): out.append(f"{f['rows']}: it reads the {f['split']} partition of {d}, a training corpus")
    return out + [f"cannot list the sources of {d}, training data of {run}" for d in training.unplaced]


def report_line(name, rows, T):
    raw, cal = metrics(rows), metrics(rows, T)
    return (f"{name:12} T={T:.2f}  acc {raw['acc']:.3f} -> {cal['acc']:.3f} | brier {raw['brier']:.3f} -> {cal['brier']:.3f} | ece {raw['ece']:.3f} -> {cal['ece']:.3f}"
            f" | conf-err {raw['confident_error_rate']:.3f} -> {cal['confident_error_rate']:.3f} | cov@5% {raw['coverage_at_5pct_error']:.2f} -> {cal['coverage_at_5pct_error']:.2f}")


def calibrate(run, rows, exclude=(), transfer=None, allow_in_distribution=False, folds=5, seed=0):
    """Fit, report and write the temperature of `run` (see the module docstring). -> the temperature written."""
    reads = parse_rows(rows)
    if typos := allowlist_typos(reads):
        raise SystemExit("refusing a sources allowlist that names sources its rows do not contain (a typo shrinks the fit set):\n  " + "\n  ".join(typos))
    fitted = []
    for path, srcs in reads:
        suite, split = rows_origin(path)
        fitted.append({"rows": path, "suite": suite, "split": split, **({"sources": srcs} if srcs else {}),
                       "questions": len(scored_rows(select([(path, srcs)], exclude)))})
    training = checkpoint_training(run)
    problems = in_distribution(fitted, training, run)
    if problems and not allow_in_distribution:
        raise SystemExit("refusing to fit a temperature on rows that are not held out from the checkpoint's training:\n  " + "\n  ".join(problems)
                         + "\nFit on held-out datasets, or pass --allow-in-distribution to fit anyway (recorded in the checkpoint).")
    for line in problems: print(f"!!! IN DISTRIBUTION (--allow-in-distribution): {line}", flush=True)
    fit = [raw_row(recorded(r)) for r in select(reads, exclude) if r["variant"] == "clean"]
    T = fit_temperature(fit, **TEMPERATURE_FIT)
    print(report_line("fit rows", fit, T))
    if transfer: print(report_line("transfer", [r for r in read_json(transfer) if r["variant"] == "clean"], T))
    cv = cross_validated_temperature(fit, folds=folds, seed=seed, **TEMPERATURE_FIT)
    ci = cv["ece_ci95"]
    print(f"fit rows     OOF T=[{', '.join(f'{t:.2f}' for t in cv['temperatures'])}] ece raw {cv['raw']['ece']:.3f} [{ci['raw'][0]:.3f}, {ci['raw'][1]:.3f}]"
          f" -> oof {cv['out_of_fold']['ece']:.3f} [{ci['out_of_fold'][0]:.3f}, {ci['out_of_fold'][1]:.3f}]  delta [{ci['delta'][0]:.3f}, {ci['delta'][1]:.3f}] separated={cv['separated']}")
    meta = read_meta(run)
    meta.temperature = T
    meta.extra["temperature_fit"] = {"rows": rows[0] if len(rows) == 1 else list(rows), **({"exclude_rows": list(exclude)} if exclude else {}),
                                     "n": len(fit), "method": TEMPERATURE_FIT_METHOD, "cross_validation": cv,
                                     "fit_rows": fitted, "training_suites": sorted(training.suites) if training else None,
                                     **({"in_distribution": {"allowed": True, "problems": problems}} if problems else {})}
    write_meta(run, meta)
    print(f"wrote temperature {T:.2f} to {run}/d1a_config.json")
    return T


def write_manual(run, temperature, reason):
    """Write a temperature without fitting, recording where it comes from."""
    meta = read_meta(run)
    meta.temperature = temperature
    meta.extra["temperature_fit"] = {"method": "manual", "reason": reason.strip()}
    write_meta(run, meta)
    print(f"wrote temperature {temperature:.2f} to {run}/d1a_config.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="checkpoint directory (its metadata is rewritten: d1a_config.json, and head.pt)")
    ap.add_argument("--rows", action="append", default=[], help="fit set: a rows.json, optionally path:source,...; repeat to pool")
    ap.add_argument("--exclude_rows", action="append", default=[], help="rows.json whose record ids are dropped from the fit set; repeatable")
    ap.add_argument("--transfer", help="out-of-domain rows.json, reported before and after (never fitted)")
    ap.add_argument("--temperature", type=float, help="write this value without fitting; needs --reason")
    ap.add_argument("--reason", help="with --temperature: where the value comes from (recorded in the checkpoint)")
    ap.add_argument("--allow-in-distribution", action="store_true", help="fit even on rows that share data with the checkpoint's training; warned and recorded")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.temperature is not None:
        if not (a.reason or "").strip(): ap.error("--temperature needs --reason: where the value comes from (recorded in the checkpoint)")
        return write_manual(a.run, a.temperature, a.reason)
    if not a.rows: ap.error("--rows is required unless --temperature is given")
    return calibrate(a.run, a.rows, a.exclude_rows, a.transfer, a.allow_in_distribution, a.folds, a.seed)


if __name__ == "__main__":
    main()
