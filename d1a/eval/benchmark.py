"""Score a predictor (a checkpoint, or any System One endpoint) on a frozen suite partition or on your own labelled JSONL.

    uv run python -m d1a.eval.benchmark --run runs/<run>/checkpoint --suite evals/<v>/decision-<v> --out runs/<name>
    uv run python -m d1a.eval.benchmark --remote http://127.0.0.1:8008 --suite ... --out ...      # any System One endpoint

Each prediction becomes one row per question (prediction_rows), d1a.eval.metrics scores the rows (summarize), and
evaluate_records writes three files to --out: predictions.jsonl (each record's request hash, prediction and rows, written
as it is scored), rows.json and report.json. The predictors are in d1a.eval.predictors.
"""
import argparse
import functools
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from d1a.core.api import option_text, question_keys, with_date_facts
from d1a.backends.checkpoint import LoadOptions
from d1a.training.data import api_request, load_records
from d1a.backends.device import default_device
from d1a.eval.metrics import EPSILON, grouped_metrics, metrics, unknowable_report
from d1a.backends.torch import ROW_PASS_TOKENS, ContextOverflow
from d1a.eval.predictors import LocalPredictor, RemotePredictor, RotationAveraged
from d1a.eval.suites import resolve
from d1a.eval.suite import CONTEXT, ENCODING, SERVING_CONTEXT, digest, load_split, read_manifest, record_digest, write_json


# --- rows ---------------------------------------------------------------------------------------------------------------

def labels(q):
    """(the option keys of a labelled request question, the index of its label among them)."""
    keys = question_keys(q["type"], q.get("criteria"))
    return keys, (keys.index(q["label"]) if q["type"] == "choice" else int(q["label"]))


def validate_distribution(raw, keys):
    """A returned {key: probability} as an array in option order, renormalised, and its sum as returned. Refuses other
    keys, values outside [0, 1] or not finite, and a sum off 1 by more than rounding over that many options explains."""
    if set(raw) != set(keys):
        raise ValueError("probability keys do not match requested options")
    p = np.array([raw[key] for key in keys], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("non-finite or out-of-range probabilities")
    total = float(p.sum())
    tolerance = max(1e-5, len(keys) * 0.005 + 1e-8)   # 4-decimal answers (d1a.core.api.round_prob) may each be off by 0.00005
    if total <= 0 or abs(total - 1) > tolerance:
        raise ValueError(f"invalid probability sum: {total}")
    return p / total, total


def parent_of(meta):
    """The clean record a variant perturbs. Suites frozen before parent_id existed kept it in a variant's group_id."""
    return meta.get("parent_id") or (meta["id"] if meta["variant"] == "clean" else meta["group_id"])


def prediction_rows(record, prediction):
    """One row per question of a labelled record: its identity, the distribution in option order (renormalised), the
    label's index, and the raw logits and inference temperature when the prediction carries them."""
    if set(prediction["probabilities"]) != set(record["questions"]):
        raise ValueError("answer IDs differ from request IDs")
    meta, rows = record["_meta"], []
    for qid, q in record["questions"].items():
        keys, label = labels(q)
        p, total = validate_distribution(prediction["probabilities"][qid], keys)
        row = {"id": meta["id"], "group": meta["group_id"], "question": qid, "source": meta["source"], "task": q["src"], "type": q["type"],
               "variant": meta["variant"], "keys": keys, "label": label, "control_id": meta.get("control_id"),
               "pair_id": meta.get("pair_id"), "sibling": meta.get("sibling"), "parent": parent_of(meta),
               "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum())}
        if "logits" in prediction:
            logits = prediction["logits"][qid]
            if set(logits) != set(keys) or not all(math.isfinite(logits[key]) for key in keys):
                raise ValueError("logit keys or values do not match the requested options")
            row["logits"] = [float(logits[key]) for key in keys]
            row["inference_temperature"] = prediction["inference_temperature"]
        if "kernels" in prediction:
            row["kernels"] = prediction["kernels"]   # a long row scored off the fp32-exact kernels (d1a.eval.predictors.LocalPredictor)
        rows.append(row)
    return rows


# --- the report ---------------------------------------------------------------------------------------------------------

def paired_flip(rows):
    """Minimal-pair metrics over rows with a pair_id (contrastive suites: siblings "a" and "b" differ in one sentence).
    Over the pairs whose label changes: how often the answer changes too, and how often both are right. Over the pairs
    whose label stays: how often the answer stays. None without pairs. A model that ignores the state cannot flip."""
    pairs = {}
    for row in rows:
        if not row.get("pair_id"):
            continue
        siblings = pairs.setdefault((row["pair_id"], row.get("question", "decision")), {})
        if row["sibling"] in siblings:
            raise ValueError("duplicate contrastive sibling")
        siblings[row["sibling"]] = row
    if not pairs:
        return None
    if any(set(siblings) != {"a", "b"} for siblings in pairs.values()):
        raise ValueError("incomplete contrastive pair")
    answer = lambda row: row["keys"][max(range(len(row["p"])), key=row["p"].__getitem__)]
    truth = lambda row: row["keys"][row["label"]]
    flipping = [s for s in pairs.values() if truth(s["a"]) != truth(s["b"])]
    steady = [s for s in pairs.values() if truth(s["a"]) == truth(s["b"])]

    def both_right(group):
        return sum(all(answer(row) == truth(row) for row in s.values()) for s in group) / len(group) if group else None

    out = {"pairs": len(flipping),
           "flip_rate": sum(answer(s["a"]) != answer(s["b"]) for s in flipping) / len(flipping) if flipping else None,
           "both_correct_rate": both_right(flipping)}
    if steady:
        out["invariant_pairs"] = len(steady)
        out["invariance_rate"] = sum(answer(s["a"]) == answer(s["b"]) for s in steady) / len(steady)
        out["invariant_both_correct_rate"] = both_right(steady)
    return out


def permutation_shift(rows, clean):
    """For each permuted Choice row, against its clean parent with the options put back in order: the largest probability
    change and whether the top answer changed."""
    parent = {(row["id"], row["question"]): row for row in clean}
    shifts, flips = [], []
    for row in rows:
        if row["variant"] != "permuted" or row["type"] != "choice":
            continue
        original = parent[(row["parent"], row["question"])]
        realigned = [row["p"][row["keys"].index(key)] for key in original["keys"]]
        shifts.append(float(np.max(np.abs(np.array(realigned) - original["p"]))))
        flips.append(int(np.argmax(realigned) != np.argmax(original["p"])))
    return shifts, flips


def position_bias(rows):
    """How often the top answer of a clean choice question is its first option, against how often the label is: a model
    that favours the first slot picks it more often than it is right there (#38)."""
    rows = [row for row in rows if row["type"] == "choice" and len(row["keys"]) > 1]
    if not rows:
        return None
    first = float(np.mean([int(np.argmax(row["p"]) == 0) for row in rows]))
    label = float(np.mean([int(row["label"] == 0) for row in rows]))
    return {"n": len(rows), "first_slot_rate": first, "label_first_rate": label, "excess": first - label}


def identical_controls(records, limit):
    """Up to `limit` controls, one per clean choice question in record order: its state and instructions with every option
    the first option's text. Asked as a score question, whose levels may repeat, so the options are the same text; the
    only difference between them is their position, and the answer should be uniform."""
    out = []
    for record in records:
        if record["_meta"]["variant"] != "clean":
            continue
        for qid, q in record["questions"].items():
            if len(out) >= limit:
                return out
            if q["type"] != "choice" or len(q["criteria"]) < 2:
                continue
            text = option_text(*next(iter(q["criteria"].items())))
            out.append({"state": record["state"], "_meta": {**record["_meta"], "id": f"{record['_meta']['id']}#identical:{qid}"},
                        "questions": {qid: {"type": "score", "instructions": q.get("instructions"), "criteria": [text] * len(q["criteria"]),
                                            "label": 0, "src": q.get("src")}}})
    return out


def identical_report(probabilities):
    """Over the controls' answers ({level: p} each): how far from uniform, and how often the first slot is strictly the top one."""
    deviations, first, firsts = [], [], []
    for probs in probabilities:
        p = np.array([probs[str(i)] for i in range(len(probs))], dtype=float)
        p /= p.sum()
        deviations.append(float(np.max(np.abs(p - 1 / len(p)))))
        first.append(float(p[0] - 1 / len(p)))
        firsts.append(int(p[0] > p[1:].max()))   # strictly: a uniform answer is not a first-slot pick
    if not deviations:
        return None
    return {"n": len(deviations), "mean_max_deviation": float(np.mean(deviations)), "max_max_deviation": float(np.max(deviations)),
            "mean_first_minus_uniform": float(np.mean(first)), "first_slot_rate": float(np.mean(firsts))}


def objective_tasks(tasks):
    """The tasks the objective averages: every task except the unknowable ones (their controls count)."""
    return [m["nll"] for name, m in tasks.items() if not name.startswith("unknowable_") or name.startswith("unknowable_control")]


METRIC_POLICY = {"version": 2, "selective_ties": "whole_confidence_groups",
                 "coverage_at_error": "in-sample maximum over confidence thresholds; not a deployed error guarantee",
                 "aurc": "right-step integral over whole confidence groups",
                 "confident_error_rate": "high-confidence errors divided by all questions",
                 "error_rate_at_0_9": "errors divided by questions accepted at p_max >= 0.9",
                 "nll": "exact from logits when recorded; otherwise from floored probabilities"}


def summarize(rows, temperature=1.0, heldout_sources=()):
    """The report over benchmark rows: the objective (minus the task-macro mean NLL), per-task and per-variant metrics,
    the clean knowable questions raw and at `temperature`, minimal pairs, option-order sensitivity and the unknowable
    report. heldout_sources: sources the scored model never trained on; their tasks are also reported apart."""
    clean = [row for row in rows if row["variant"] == "clean"]
    tasks = grouped_metrics(clean, "task")
    variants = grouped_metrics(rows, "variant")
    shifts, flips = permutation_shift(rows, clean)
    knowable = [row for row in clean if row["source"] != "unknowable"]   # unknowable records are scored on confidence alone
    heldout = [row for row in clean if row["source"] in heldout_sources]
    return {"objective": -float(np.mean(objective_tasks(tasks))),
            "paired_flip": paired_flip(clean), "unknowable": unknowable_report(clean),
            "clean": metrics(knowable), "position_bias": position_bias(knowable), "tasks": tasks, "variants": variants,
            "heldout_tasks": grouped_metrics(heldout, "task") if heldout else {},
            "permutation": {"n": len(shifts), "mean_max_delta": float(np.mean(shifts)) if shifts else None,
                            "flip_rate": float(np.mean(flips)) if flips else None},
            "temperature": temperature, "calibrated_clean": metrics(knowable, temperature),
            "metric_policy": {**METRIC_POLICY, "nll_floor": EPSILON, "renormalize_returned_probabilities": True,
                              "raw_sums_outside_1e_5": sum(abs(row["raw_probability_sum"] - 1) > 1e-5 for row in rows),
                              "returned_zeros": sum(row["zero_count"] for row in rows)}}


# --- scoring ------------------------------------------------------------------------------------------------------------

def predictions(records, predictor):
    """Yield, in record order, a zero-argument callable that returns predictor(record) or raises what it raised. A
    predictor with `concurrency` > 1 (RemotePredictor: one independent HTTP request per call) keeps that many calls in
    flight on a thread pool; in-process predictors (one GPU) have no `concurrency` and run one record at a time. Either
    way the caller sees each outcome in the order of the plain sequential loop."""
    workers = getattr(predictor, "concurrency", 1)
    if workers <= 1 or len(records) <= 1:
        for record in records:
            yield functools.partial(predictor, record)
        return
    pool = ThreadPoolExecutor(max_workers=min(workers, len(records)))
    try:
        for future in [pool.submit(predictor, record) for record in records]:
            yield future.result
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def evaluate_records(records, predictor, directory, temperature=1.0, heldout_sources=(), skip_overlong=False, identical=0):
    """Score `records` into `directory` (which must not exist) and return (report, rows). The first record that fails
    stops the run, with failure.json saying which and how far it got. skip_overlong: for data never admitted to a context
    (--data, or an eval-only suite frozen as published), a record the predictor cannot encode is counted in
    coverage["rejected_records"] and listed in rejected.json instead; admitted suites never hit it. A report must say how
    rejected records enter any headline number. identical: how many identical-option controls to score after the records
    (identical_controls; 0 for none), reported apart and written to identical_options.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    coverage = {"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                "evaluated_records": 0, "evaluated_questions": 0, "rejected_records": 0, "truncated_records": 0}
    rows, latencies, rejected = [], [], []
    outcomes = predictions(records, predictor)
    with (directory / "predictions.jsonl").open("w", encoding=ENCODING) as log:
        for record in records:
            try:
                prediction = next(outcomes)()
                scored = prediction_rows(record, prediction)
            except Exception as error:
                coverage["rejected_records"] += 1
                if skip_overlong and isinstance(error, ContextOverflow):
                    rejected.append({"id": record["_meta"]["id"], "error": str(error)})
                    continue
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            log.write(json.dumps({"request_sha256": record_digest(api_request(record)), "id": record["_meta"]["id"],
                                  "prediction": prediction, "rows": scored}, allow_nan=False) + "\n")
            log.flush()
            rows += scored
            latencies.append(prediction["latency_ms"])
            coverage["evaluated_records"] += 1
            coverage["evaluated_questions"] += len(scored)
            if coverage["evaluated_records"] % 50 == 0:
                print(f"evaluated {coverage['evaluated_records']}/{len(records)}", flush=True)
    write_json(directory / "rows.json", rows)
    if rejected:
        write_json(directory / "rejected.json", rejected)
    report = summarize(rows, temperature, heldout_sources)
    report["coverage"] = coverage
    report["latency_ms"] = {"median": float(np.median(latencies)), "p95": float(np.quantile(latencies, .95))}
    report["calibration"] = {"inference_temperature": getattr(predictor, "temperature", None), "additional_temperature": temperature,
                             "logits_recorded": all("logits" in row for row in rows)}
    if identical:
        skipped = {r["id"] for r in rejected}
        controls = [c for c in identical_controls(records, identical + len(skipped)) if c["_meta"]["id"].split("#identical:")[0] not in skipped][:identical]
        kept, answers = [], []
        for control, call in zip(controls, predictions(controls, predictor)):
            try:
                answers.append(next(iter(call()["probabilities"].values())))
                kept.append(control)
            except ContextOverflow:   # the first option repeated K times can outgrow the row its question fit in
                pass
        # how many were asked and skipped sits beside the results, so a mostly skipped set never reads as a clean one
        report["identical_options"] = {"requested": len(controls), "skipped_overlong": len(controls) - len(kept),
                                       **(identical_report(answers) or {"n": 0})} if controls else None
        write_json(directory / "identical_options.json", [{"id": c["_meta"]["id"], "p": a} for c, a in zip(kept, answers)])
    long = [row for row in rows if "kernels" in row]
    if long:   # absent when every row ran the exact kernels, so those reports are unchanged
        report["long_rows"] = {"count": len(long), "records": len({row["id"] for row in long}), "kernels": sorted({row["kernels"] for row in long}),
                               "threshold": ROW_PASS_TOKENS}
    write_json(directory / "report.json", report)
    return report, rows


# --- command line -------------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="checkpoint dir or Hub id (local scoring)")
    ap.add_argument("--remote", help="base URL of a System One-compatible endpoint to score instead of a local checkpoint")
    ap.add_argument("--remote-model", default="d1a-latest")
    ap.add_argument("--remote-concurrency", type=int, default=1, help="requests kept in flight against --remote (1 = sequential); rows and their order do not depend on it")
    ap.add_argument("--suite", help="frozen suite directory (scores its development partition)")
    ap.add_argument("--data", help="your own labelled requests, one JSON object per line (d1a.training.data.load_records), or a D1A suite partition (evals/d1a/<suite>:<partition>); an alternative to --suite")
    ap.add_argument("--context", choices=["training", "serving"], default="training",
                    help="--data only: the length limits records are scored under. training: a checkpoint's training limits (384-token "
                         "states by default); longer records are skipped and counted. serving: what d1a.serving.serve accepts (up to 64k tokens)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=default_device())
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--split", choices=["development", "calibration", "train"], default="development",
                    help="suite partition to score (train: teacher predictions for distillation; --allow-test reads the locked test instead)")
    ap.add_argument("--date_facts", action="store_true", help="apply d1a.core.api.with_date_facts to every state before scoring (the opt-in serving preprocessor); reported in report.json")
    ap.add_argument("--rotations", type=int, default=1, help="average every Choice question over this many cyclic option rotations (d1a.eval.predictors.RotationAveraged); 1 = one order")
    ap.add_argument("--identical-options", type=int, default=100, help="identical-option controls scored after the suite (position bias, #38): "
                    "the first N clean choice questions asked with every option the same text; 0 for none")
    a = ap.parse_args(argv)
    if bool(a.run) == bool(a.remote): ap.error("give exactly one of --run or --remote")
    if a.rotations < 1: ap.error("--rotations must be >= 1")
    if a.remote_concurrency < 1: ap.error("--remote-concurrency must be >= 1")
    if bool(a.suite) == bool(a.data): ap.error("give exactly one of --suite or --data")
    return a


def inputs(a):
    """(records, held-out sources, split name, sha256 of what was read, the context to score under, skip_overlong)."""
    if a.data:
        path = resolve(a.data)   # a D1A suite partition (evals/d1a/<suite>:<partition>) is fetched pinned and verified
        context = SERVING_CONTEXT if a.context == "serving" else CONTEXT
        return load_records(path), [], "custom", digest(Path(path)), context, True
    split = "test" if a.allow_test else a.split
    records = load_split(a.suite, split, allow_test=a.allow_test)
    manifest = read_manifest(a.suite)
    return (records, manifest["holdout_sources"], split, digest(Path(a.suite) / "manifest.json"),
            manifest.get("context", CONTEXT), bool(manifest.get("eval_only")))


def main():
    a = parse_args()
    records, heldout, split, source_hash, context, skip_overlong = inputs(a)
    if a.date_facts:
        records = [{**record, "state": with_date_facts(record["state"])} for record in records]
    if a.remote:
        predictor = RemotePredictor(a.remote, a.remote_model, os.environ.get("D1A_REMOTE_API_KEY", "local"), concurrency=a.remote_concurrency)
    else:
        predictor = LocalPredictor(a.run, a.device, LoadOptions.from_env(), context=context)
    scorer = RotationAveraged(predictor, a.rotations) if a.rotations > 1 else predictor
    report, _ = evaluate_records(records, scorer, a.out, heldout_sources=tuple(heldout), skip_overlong=skip_overlong, identical=a.identical_options)
    report.update(suite_sha256=source_hash, data=a.data, date_facts=a.date_facts, rotations=a.rotations, run=a.run or a.remote, split=split,
                  calibration_applied=None if a.remote else predictor.temperature != 1.0,
                  remote={"base_url": a.remote, "requested_model": a.remote_model, "served_model": predictor.served_model,
                          "concurrency": a.remote_concurrency} if a.remote else None)
    write_json(Path(a.out) / "report.json", report)
    skipped = report["coverage"]["rejected_records"]
    if a.data and skipped:   # a silently smaller set would read as the whole file's score
        print(f"!!! skipped {skipped} of {report['coverage']['requested_records']} records longer than the {a.context} context "
              f"(listed in {Path(a.out) / 'rejected.json'})" + ("; --context serving scores them" if a.context == "training" else ""), flush=True)
    print(json.dumps({"objective": report["objective"], "clean": report["clean"], "coverage": report["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
