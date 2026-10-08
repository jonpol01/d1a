"""Calibrate a checkpoint: fit one temperature on held-out scored rows and write it into the checkpoint (d1a_config.json, and the head.pt D1A 0.3 reads), so
every loader serves calibrated probabilities. Argmax never changes, so accuracy is the same before and after.

    python -m d1a.training.calibrate --run runs/new --rows runs/cal/rows.json --rows runs/calpr/rows.json:src_a,src_b

--rows are d1a.eval.benchmark rows.json files, pooled; `path:source,...` (here and in --judge, --guard, --confirm and
--locked) keeps only those sources (a source the rows' suite does not list is refused: a typo would silently shrink the
set). --exclude_rows drops records from the fit by id. --transfer rows are reported before and after, never fitted.
--temperature T writes a value without fitting or judging it, needs --reason, and refuses the rule's options.

The fit rows must be held out from the checkpoint's training. A temperature fitted on held-out items of the checkpoint's
own training data is in distribution and comes out overconfident on anything else (Kev's round 19: T 0.955 on its own
development rows gave ECE 0.059 elsewhere; a pool of held-out datasets gave 0.0085). So the fit is refused when a rows
file (a) comes from a suite the checkpoint trained on, (b) pools a source it trained on, or (c) reads the calibration or
development partition of a training corpus, and when a rows file's suite or the checkpoint's training cannot be placed.
--allow-in-distribution fits anyway, warns, and records every reason in the checkpoint's temperature_fit.in_distribution.

Every fit also records the 90% bootstrap interval of its temperature (report only). With --judge rows, the fitted temperature
replaces the checkpoint's current one only when it passes Kev's round 28 rule on them and the --guard rows, and then that
round's confirmation stages on the --confirm and --locked rows, scored only once the rule passes (d1a.training.temperature_gate);
otherwise nothing is written and the run names every criterion that failed. The rule judges the refit against the incumbent:
--incumbent <run|T> (a checkpoint's served temperature, or T itself); without it, for a fine-tune not calibrated since
training, the temperature its --init_from checkpoint serves (d1a.training.train writes every run at 1.0, so the run's own
would judge the refit against an uncalibrated model, #207); otherwise the run's own temperature. The incumbent and where it
came from are printed and recorded in temperature_fit.rule.incumbent. Before anything is fitted, a file in any of
these roles is refused when it overlaps the rows the fit reads (each --rows selection minus --exclude_rows), and so is a
confirmation file that overlaps a file the rule reads; files overlap when they share a row or the suite partition their
report.json names, so a copy, whole or part, counts as the file. A selection with no scored rows is refused too: an empty
judge or guard panel would pass its ECE check on nothing, and an empty --rows selection would shrink the fit unseen. The rule's bootstrap and the interval are Kev's registered reads:
seed 0 whatever --seed (the cross-validation's) is, and, for the rule, the rows in (id, question) order with every tie broken,
so reordering a rows file never moves a verdict.

    python -m d1a.training.calibrate --run runs/new --rows runs/pool/rows.json --judge runs/served/rows.json --guard runs/hard/rows.json --confirm runs/test/rows.json --locked runs/locked/rows.json

--use-case NAME fits (or with --temperature, writes) the temperature of one use case instead: the requests that name it
(SystemOneRequest.use_case) are served at it, every other request keeps the checkpoint's temperature, which this never
touches. It goes into the checkpoint's use_case_temperatures map (d1a_config.json's extra, carried by MLX exports), its
record into use_case_temperature_fits[NAME], with the same fit, report and refusals; --judge compares it with the
incumbent's temperature for that use case (its entry, else the checkpoint's): --incumbent's, or for a fine-tune with no
entry of its own its --init_from checkpoint's (even when its main temperature is fitted: that fit says nothing about this
use case), else the run's own. --incumbent T is used as given.

    python -m d1a.training.calibrate --run runs/v0.5 --use-case routing --rows <generic-development + handlabelled-45 rows>.json --transfer <factory-development rows>.json --allow-in-distribution
    python -m d1a.training.calibrate --run runs/v0.5 --use-case routing --temperature 0.85 --reason "fitted on ..., report ..."

The only held-out rows a routing checkpoint has are its own suite's development partitions (evals/d1a/routing), which the
held-out check refuses: the checkpoint trained on that suite's train partitions, so they are in distribution (and the check
cannot place D1A suite partitions yet, #196). Fit on them with --allow-in-distribution (recorded), and judge the result out
of sample: by the cross-fit line (OOF) below, and by fitting on some partitions and scoring another with --transfer.

What the checkpoint trained on comes from its metadata (d1a.training.train records the suite's manifest hash, its args and --data) or a
provenance.json beside it; where rows came from, from the report.json d1a.eval.benchmark writes beside them.
"""
import argparse
import functools
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from d1a.core.api import USE_CASE_MAX_LENGTH
from d1a.backends.checkpoint import USE_CASE_FITS, USE_CASE_TEMPERATURES, Checkpoint, checked_use_case_temperatures, read_meta, write_meta
from d1a.eval.metrics import TEMPERATURE_FIT, TEMPERATURE_FIT_METHOD, cross_validated_temperature, fit_temperature, metrics, raw_row, recorded, scored_rows
from d1a.eval.suite import digest, read_json, read_manifest, suite_key
from d1a.training.temperature_gate import confirmation, interval, keyed, rule

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


def panel(path, srcs, exclude=()):
    """A rows file's clean, knowable rows at T=1, keyed by the file (temperature_gate.keyed)."""
    return keyed([raw_row(recorded(r)) for r in scored_rows(select([(path, srcs)], exclude))], path)


def panels(reads):
    """{`path` or `path:source,...`: its panel}: two selections of one file are two panels."""
    return {path + (f":{','.join(srcs)}" if srcs else ""): panel(path, srcs) for path, srcs in reads}


def identity(path, srcs=None, exclude=()):
    """What a rows file holds (its rows select() keeps), as a set two files share an element of when they overlap: the
    sha256 of each row (a copy of any of them elsewhere, whole or part, re-indented or not, overlaps it), and the (suite or
    partition sha256, split) of the report.json beside it (another read of the same questions)."""
    report = Path(path).parent / "report.json"
    r = read_json(report) if report.exists() else {}
    return ({hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest() for row in select([(path, srcs)], exclude)}
            | ({(r["suite_sha256"], r.get("split", "development"))} if "suite_sha256" in r else set()))


def roles_problem(reads, judged, guarded, tests, held, exclude=()):
    """Why these rows files cannot play these roles together, or None. The fit counts only the rows it reads (each --rows
    selection minus --exclude_rows: Kev's round 28 guards the transfer-v4 records it drops from its pool); every other role
    counts its whole file."""
    if (guarded or tests or held) and not judged:
        return "--guard, --confirm and --locked need --judge (without the rule they would be ignored and the fit written)"
    fitted = set().union(*(identity(path, srcs, exclude) for path, srcs in reads))
    ids = {path: identity(path) for path, _ in judged + guarded + tests + held}
    def among(mine, theirs): return [path for path, _ in mine if any(ids[path] & ids[other] for other, _ in theirs)]
    if twice := [path for path, _ in judged + guarded + tests + held if ids[path] & fitted]:
        return f"refusing to judge a temperature on the rows it is fitted on: {twice}"
    if twice := among(tests + held, judged + guarded):
        return f"refusing confirmation rows the rule already reads: {twice}"
    if twice := [path for i, (path, _) in enumerate(judged) if among([(path, None)], judged[:i])]:
        return f"refusing judge rows given twice (shared rows would count twice in the pool; name one file's sources in one path:a,b): {twice}"
    return None


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


def init_of(run, meta):
    """(where to read it, as named) for the checkpoint a fine-tune started from (d1a.training.train --init_from, from the run's
    training_config.json, else its metadata), or None for a run from a base model."""
    path = Path(run) / "training_config.json"
    trained = read_json(path) if path.exists() else meta.extra
    source = trained.get("init_source") or {}
    named = (trained.get("args") or {}).get("init_from") or source.get("init_from")
    if not named: return None
    resolved = source.get("resolved")   # the snapshot it trained from, when this machine still has it
    return (resolved if resolved and Path(resolved).is_dir() else named), named


def served_temperature(checkpoint, flag, use_case=None):
    try:
        return serving(Checkpoint(checkpoint).meta, use_case)
    except Exception as e:
        raise SystemExit(f"{flag}: cannot read the temperature {checkpoint} serves ({type(e).__name__}: {e}); pass --incumbent <run|T>") from e


def incumbent_of(run, meta, given=None, use_case=None):
    """{temperature, source, name}: what the rule judges a refit against, with `use_case` that use case's temperature on the
    incumbent checkpoint (see the module docstring)."""
    if given is not None:
        if not Path(given).is_dir():
            try: T = float(given)
            except ValueError: T = None
            if T is not None:
                if not (math.isfinite(T) and T > 0): raise SystemExit(f"--incumbent {given}: a temperature is a positive number")
                return {"temperature": T, "source": "incumbent", "name": given}
        return {"temperature": served_temperature(given, f"--incumbent {given}", use_case), "source": "incumbent", "name": given}
    init = init_of(run, meta)
    own = use_case in meta.use_case_temperatures if use_case else False   # an entry is only ever written by calibrate
    # still training's 1.0, or (with a use case) no entry of its own: the init's temperature is the one it would replace; a
    # main fit says nothing about one use case's requests
    if init and (not own if use_case else "temperature_fit" not in meta.extra):
        return {"temperature": served_temperature(init[0], f"--init_from {init[1]}", use_case), "source": "init_from", "name": init[1]}
    return {"temperature": serving(meta, use_case), "source": "run", "name": str(run)}


def report_line(name, rows, T):
    raw, cal = metrics(rows), metrics(rows, T)
    return (f"{name:12} T={T:.2f}  acc {raw['acc']:.3f} -> {cal['acc']:.3f} | brier {raw['brier']:.3f} -> {cal['brier']:.3f} | ece {raw['ece']:.3f} -> {cal['ece']:.3f}"
            f" | conf-err {raw['confident_error_rate']:.3f} -> {cal['confident_error_rate']:.3f} | cov@5% {raw['coverage_at_5pct_error']:.2f} -> {cal['coverage_at_5pct_error']:.2f}")


def serving(meta, use_case):
    """The temperature requests of `use_case` (None: every request without one) are served at now."""
    return meta.use_case_temperatures.get(use_case, meta.temperature) if use_case else meta.temperature


def write(run, meta, T, fit, use_case):
    """Write T and its fit record: the checkpoint's temperature, or with `use_case` only that use case's entry (the
    checkpoint's temperature and temperature_fit stay as they are)."""
    if use_case:
        meta.extra[USE_CASE_TEMPERATURES] = checked_use_case_temperatures({**meta.use_case_temperatures, use_case: T})
        meta.extra[USE_CASE_FITS] = {**meta.extra.get(USE_CASE_FITS, {}), use_case: fit}
    else:
        meta.temperature = T
        meta.extra["temperature_fit"] = fit
    write_meta(run, meta)
    print(f"wrote temperature {T:.2f}" + (f" for use case {use_case!r}" if use_case else "") + f" to {run}/d1a_config.json")


def calibrate(run, rows, exclude=(), transfer=None, allow_in_distribution=False, folds=5, seed=0, judge=(), guard=(), confirm=(), locked=(), incumbent=None, use_case=None):
    """Fit, report and write the temperature of `run`, or with `use_case` that use case's (see the module docstring).
    -> the temperature it (that use case) now serves."""
    reads, judged, guarded, tests, held = (parse_rows(v) for v in (rows, judge, guard, confirm, locked))
    if problem := roles_problem(reads, judged, guarded, tests, held, exclude):
        raise SystemExit(problem)
    if typos := allowlist_typos(reads + judged + guarded + tests + held):
        raise SystemExit("refusing a sources allowlist that names sources its rows do not contain (a typo shrinks the selection):\n  " + "\n  ".join(typos))
    gate = {role: panels(given) for role, given in (("judge", judged), ("guard", guarded), ("confirm", tests), ("locked", held))}
    if empty := [f"--{role} {name}" for role, named in {"rows": panels(reads), **gate}.items() for name, scored in named.items() if not scored]:
        raise SystemExit("refusing a selection with no scored rows (a source its suite lists that the read lacks, or only noise or"
                         " unknowable rows):\n  " + "\n  ".join(empty))
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
    if judged:
        against = incumbent_of(run, read_meta(run), incumbent, use_case)
        print(f"incumbent    T={against['temperature']:.4f}" + (f" for use case {use_case!r}" if use_case else "") + ": " + {"incumbent": f"--incumbent {against['name']}",
              "init_from": f"the temperature --init_from {against['name']} serves (this fine-tune is not calibrated since training)",
              "run": f"{run}'s own"}[against["source"]], flush=True)
    fit = [raw_row(recorded(r)) for r in select(reads, exclude) if r["variant"] == "clean"]
    T = fit_temperature(fit, **TEMPERATURE_FIT)
    print(report_line("fit rows", fit, T))
    if transfer: print(report_line("transfer", [r for r in read_json(transfer) if r["variant"] == "clean"], T))
    units = [r for path, srcs in reads for r in panel(path, srcs, exclude)]
    cv = cross_validated_temperature(units, folds=folds, seed=seed, **TEMPERATURE_FIT)
    ci = cv["ece_ci95"]
    print(f"fit rows     OOF T=[{', '.join(f'{t:.2f}' for t in cv['temperatures'])}] ece raw {cv['raw']['ece']:.3f} [{ci['raw'][0]:.3f}, {ci['raw'][1]:.3f}]"
          f" -> oof {cv['out_of_fold']['ece']:.3f} [{ci['out_of_fold'][0]:.3f}, {ci['out_of_fold'][1]:.3f}]  delta [{ci['delta'][0]:.3f}, {ci['delta'][1]:.3f}] separated={cv['separated']}")
    spread = interval(units)
    print(f"fit rows     T={T:.4f} 90% interval [{spread['lower']:.4f}, {spread['upper']:.4f}] over {spread['samples']} cluster resamples")
    meta = read_meta(run)
    current = serving(meta, use_case)
    verdict = None
    if judged:
        shipped = against["temperature"]
        verdict = {**rule(gate["judge"], gate["guard"], T, shipped), "incumbent": against}
        j = verdict["judge"]
        print(f"rule         T={T:.4f} vs incumbent {shipped:.4f} on {j['n']} judge questions: Brier {j['brier_delta']:+.4f}"
              f" [{j['brier_ci95'][0]:+.4f}, {j['brier_ci95'][1]:+.4f}], ECE {j['ece']['candidate']:.4f} vs {j['ece']['shipped']:.4f}")
        if verdict["adopt"] and (tests or held):
            verdict["confirmation"] = confirmation(gate["confirm"], gate["locked"], T, shipped)
            for stage, named in (("confirm", verdict["confirmation"]["tests"]), ("locked", verdict["confirmation"]["locked"])):
                for name, s in named.items():
                    print(f"{stage:12} {name}: ECE {s['ece']['candidate']:.4f} vs {s['ece']['shipped']:.4f}, Brier {s['brier']['candidate']:.4f}"
                          f" vs {s['brier']['shipped']:.4f}, acc {s['acc']['candidate']:.4f} vs {s['acc']['shipped']:.4f}")
        elif tests or held:
            print("confirmation rows not scored: the rule failed")
        if failed := verdict["failed"] + verdict.get("confirmation", {}).get("failed", []):
            print(f"kept temperature {current:.4f}" + (f" for use case {use_case!r}" if use_case else "") + f" in {run}: nothing written\n  " + "\n  ".join(failed))
            if shipped != current:
                print(f"  to serve the incumbent: --temperature {shipped}" + (f" --use-case {use_case!r}" if use_case else "")
                      + f" --reason \"refit failed the rule against {against['name']}\"")
            return current
    write(run, meta, T, {"rows": rows[0] if len(rows) == 1 else list(rows), **({"exclude_rows": list(exclude)} if exclude else {}),
                         "n": len(fit), "method": TEMPERATURE_FIT_METHOD, "cross_validation": cv,
                         "interval": spread, **({"rule": verdict} if verdict else {}),
                         "fit_rows": fitted, "training_suites": sorted(training.suites) if training else None,
                         **({"in_distribution": {"allowed": True, "problems": problems}} if problems else {})}, use_case)
    return T


def write_manual(run, temperature, reason, use_case=None):
    """Write a temperature without fitting, recording where it comes from (with `use_case`, that use case's only)."""
    write(run, read_meta(run), temperature, {"method": "manual", "reason": reason.strip()}, use_case)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="checkpoint directory (its metadata is rewritten: d1a_config.json, and head.pt)")
    ap.add_argument("--rows", action="append", default=[], help="fit set: a rows.json, optionally path:source,...; repeat to pool")
    ap.add_argument("--exclude_rows", action="append", default=[], help="rows.json whose record ids are dropped from the fit set; repeatable")
    ap.add_argument("--transfer", help="out-of-domain rows.json, reported before and after (never fitted)")
    ap.add_argument("--temperature", type=float, help="write this value without fitting or judging it; needs --reason")
    ap.add_argument("--reason", help="with --temperature: where the value comes from (recorded in the checkpoint)")
    ap.add_argument("--allow-in-distribution", action="store_true", help="fit even on rows that share data with the checkpoint's training; warned and recorded."
                    " A routing --use-case fit on evals/d1a/routing's development partitions needs it: they come from the corpus the checkpoint"
                    " trained on (and #196: the check cannot place D1A suite partitions yet); judge it by the OOF line and --transfer")
    ap.add_argument("--use-case", help="fit or write the temperature of this use case only (requests with \"use_case\": NAME), into the"
                    " checkpoint's use_case_temperatures; the checkpoint's own temperature is left as it is; --judge compares"
                    " it with the incumbent's temperature for this use case (see --incumbent)")
    ap.add_argument("--judge", action="append", default=[], help="rows the new temperature must beat the current one on: pooled, Brier"
                    " (95%% upper bound below 0) and ECE lower; each one's ECE may rise by at most 0.005; path or path:source,...; repeatable")
    ap.add_argument("--guard", action="append", default=[], help="with --judge: rows whose ECE may rise by at most 0.005; path or path:source,...,"
                    " each selection its own panel; repeatable")
    ap.add_argument("--incumbent", help="with --judge: the checkpoint (local run or Hub id) whose served temperature, or the"
                    " temperature T, the refit must beat; default: for a fine-tune not calibrated since training, the temperature"
                    " its --init_from checkpoint serves, else the run's own; with --use-case, the checkpoint's temperature for that use case,"
                    " and the init stays the incumbent until the fine-tune has an entry for it;"
                    " recorded in temperature_fit.rule.incumbent (use_case_temperature_fits[NAME] with --use-case)")
    ap.add_argument("--confirm", action="append", default=[], help="with --judge, scored once the rule passes: rows whose ECE must fall (Kev round 28's tests stage); repeatable")
    ap.add_argument("--locked", action="append", default=[], help="with --judge, scored once the rule passes: locked test rows whose Brier may rise by at most 0.005,"
                    " accuracy unchanged (round 28's locked stage); repeatable")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="the cross-validation's folds and bootstrap; the rule's bootstrap and the interval are always seed 0 (Kev's registered read)")
    a = ap.parse_args(argv)
    if a.use_case is not None:
        if not a.use_case.strip(): ap.error("--use-case needs a name")
        if len(a.use_case.strip()) > USE_CASE_MAX_LENGTH: ap.error(f"--use-case: at most {USE_CASE_MAX_LENGTH} characters, as a request's use_case")
        a.use_case = a.use_case.strip()   # stored as the map stores it, so the fit and serving() find the same entry
    if a.incumbent is not None and not a.judge:
        ap.error("--incumbent needs --judge (without the rule there is nothing to judge against it)")
    if a.temperature is not None:
        if not (a.reason or "").strip(): ap.error("--temperature needs --reason: where the value comes from (recorded in the checkpoint)")
        if a.judge or a.guard or a.confirm or a.locked:
            ap.error("--temperature writes its value unjudged: --judge, --guard, --confirm and --locked would be ignored")
        if a.use_case and not (math.isfinite(a.temperature) and a.temperature > 0): ap.error("--temperature must be finite and positive")
        return write_manual(a.run, a.temperature, a.reason, a.use_case)
    if not a.rows: ap.error("--rows is required unless --temperature is given")
    return calibrate(a.run, a.rows, a.exclude_rows, a.transfer, a.allow_in_distribution, a.folds, a.seed, a.judge, a.guard, a.confirm, a.locked, a.incumbent, a.use_case)


if __name__ == "__main__":
    main()
