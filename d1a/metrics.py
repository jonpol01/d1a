"""How well a model's answers score, from benchmark rows alone (no model, no torch: saved rows.json files work).

A row is one answered question as d1a.benchmark.prediction_rows writes it: "p" (the returned distribution), "label" (the
right option's index), "type", "keys", "task", "source", "group", "variant", "id", "question", and optionally "logits"
(the raw scores behind p) and "inference_temperature" (the temperature p was served at).

- metrics(rows): accuracy; calibration (ECE, Brier, NLL, confidence bias, error rates in the top confidence bins);
  selective prediction (coverage at an error budget, the area under the risk-coverage curve); for score questions the
  mean absolute level error and the ranked probability score.
- Temperatures: fit_temperature (one temperature minimising the NLL on a log grid), the rows a calibrated predictor
  would have returned (tempered_row, raw_row, served_at, served), and out-of-fold calibration.
- Uncertainty: bootstraps that resample whole (source, group) clusters, so the questions and variants of one record move
  together (cluster_resamples, paired_bootstrap, cross_validated_temperature).

Every reduction here keeps the order of operations its reports were first computed with, so a saved report, a fitted
temperature and a bootstrap interval reproduce exactly.
"""
import math
from collections import defaultdict

import numpy as np

EPSILON = 1e-9   # the floor under a probability whose log is taken (rows that recorded no logits)
HIGH_CONFIDENCE = 0.9


# --- calibration --------------------------------------------------------------------------------------------------------

def ece(conf, correct, bins=10):
    """Expected calibration error over `bins` equal-width confidence bins (the last one closed at 1): the bin-weighted
    mean of |accuracy - mean confidence|."""
    conf, hit = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (conf >= lo) & ((conf <= hi) if hi >= 1 else (conf < hi))
        if inside.any():
            total += inside.mean() * abs(hit[inside].mean() - conf[inside].mean())
    return float(total)


def require_temperature(temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def shifted_logits(row, temperature):
    """The row's logits shifted so the largest is 0, divided by `temperature`. Rows without logits use the log of their
    probabilities, floored at EPSILON."""
    require_temperature(temperature)
    p = np.asarray(row["p"], dtype=float)
    z = np.asarray(row["logits"], dtype=float) if "logits" in row else np.log(np.maximum(p, EPSILON))
    if z.shape != p.shape or not np.isfinite(z).all():
        raise ValueError("logits must be finite and match the option count")
    return (z - z.max()) / temperature


def probabilities_at_temperature(row, temperature=1.0):
    """The row's distribution at `temperature`: its own "p" at 1, otherwise the softmax of its tempered logits."""
    require_temperature(temperature)
    if temperature == 1:
        return np.asarray(row["p"], dtype=float)
    weights = np.exp(shifted_logits(row, temperature))
    return weights / weights.sum()


def nll_at_temperature(row, temperature=1.0):
    """-log P(label) at `temperature`: exact from the logits when the row has them, else from the floored probability."""
    if "logits" in row:
        z = shifted_logits(row, temperature)
        return float(np.log(np.exp(z).sum()) - z[row["label"]])
    return -math.log(max(float(probabilities_at_temperature(row, temperature)[row["label"]]), EPSILON))


def brier(p, label):
    return float(((p - np.eye(len(p))[label]) ** 2).sum())


def question_scores(row):
    """The per-question values whose mean is one of metrics()' figures, at T=1 (the additive metrics paired_bootstrap
    resamples)."""
    p, label = np.asarray(row["p"], dtype=float), row["label"]
    top, right = float(p.max()), bool(p.argmax() == label)
    return {"acc": float(right), "nll": nll_at_temperature(row), "brier": brier(p, label), "mean_conf": top,
            "confident_error_rate": float(top >= HIGH_CONFIDENCE and not right), "coverage_at_0_9": float(top >= HIGH_CONFIDENCE),
            "confidence_bias": top - float(right)}


def share_wrong(correct):
    """The error rate of a non-empty selection, None for an empty one."""
    return float((~correct).mean()) if len(correct) else None


def metrics(rows, temperature=1.0):
    """The scores of `rows` at `temperature` (see the module docstring); the keys appear in a fixed order."""
    if not rows:
        raise ValueError("cannot score an empty population")
    nll, right, top, briers, level_errors, rps = [], [], [], [], [], []
    for row in rows:
        p, label = probabilities_at_temperature(row, temperature), row["label"]
        nll.append(nll_at_temperature(row, temperature))
        right.append(int(p.argmax() == label))
        top.append(float(p.max()))
        briers.append(brier(p, label))
        if row["type"] == "score":
            truth = np.eye(len(p))[label]
            level_errors.append(abs(float(p @ np.arange(len(p))) - label))
            rps.append(float(((p.cumsum()[:-1] - truth.cumsum()[:-1]) ** 2).mean()))
    confidence, correct = np.asarray(top), np.asarray(right, dtype=bool)
    high = confidence >= HIGH_CONFIDENCE
    out = {"n": len(rows), "nll": float(np.mean(nll)), "acc": float(np.mean(right)), "ece": ece(top, right),
           "brier": float(np.mean(briers)), "mean_conf": float(np.mean(top)),
           "confident_error_rate": float(np.mean(high & ~correct)), "coverage_at_0_9": float(high.mean()),
           "accuracy_at_0_9": float(correct[high].mean()) if high.any() else None,
           "coverage_at_5pct_error": coverage_at_error(confidence, correct, 0.05),
           "coverage_at_1pct_error": coverage_at_error(confidence, correct, 0.01),
           "aurc": area_under_risk_coverage(confidence, correct),
           "error_rate_at_0_9": share_wrong(correct[high]),
           # mean top probability minus accuracy: positive is over-confident (untrained readouts), outcome-trained readouts
           # sit near zero or below
           "confidence_bias": float(confidence.mean() - correct.mean()),
           "top_bins": {str(t): {"n": int((confidence >= t).sum()), "errors": int(((confidence >= t) & ~correct).sum()),
                                 "error_rate": share_wrong(correct[confidence >= t])} for t in (0.9, 0.95, 0.99)},
           "selective": {}}
    ranked = np.sort(confidence)
    for fraction in (0.5, 0.8):
        cutoff = ranked[-max(1, math.ceil(len(rows) * fraction))]   # the most confident `fraction` of questions, ties included
        kept = confidence >= cutoff
        out["selective"][str(fraction)] = {"coverage": float(kept.mean()), "accuracy": float(correct[kept].mean()), "confidence_cutoff": float(cutoff)}
    if level_errors:
        out["score_mae"] = float(np.mean(level_errors))
        out["ranked_probability_score"] = float(np.mean(rps))
    return out


def grouped_metrics(rows, key, temperature=1.0):
    """metrics() per value of row[key], in sorted order."""
    by_value = defaultdict(list)
    for row in rows:
        by_value[row[key]].append(row)
    return {value: metrics(group, temperature) for value, group in sorted(by_value.items())}


def unknowable_report(rows):
    """Whether the model knows when it cannot know: on records whose deciding evidence was removed (source "unknowable")
    against their intact controls ("unknowable_control"), the mean top probability and the share answered at >= 0.9.
    Accuracy on the unknowable records means nothing by construction. None when there are none."""
    blind = [row for row in rows if row["source"] == "unknowable"]
    if not blind:
        return None
    controls = [row for row in rows if row["source"] == "unknowable_control"]
    top = lambda group: [float(max(row["p"])) for row in group]
    control_of = {row["id"]: row for row in controls}
    pairs = [(max(row["p"]), max(control_of[row["control_id"]]["p"])) for row in blind if row.get("control_id") in control_of]
    either = lambda value: value if controls else None
    return {"n": len(blind), "mean_max_p": float(np.mean(top(blind))), "share_at_0_9": float(np.mean([c >= 0.9 for c in top(blind)])),
            "control_mean_max_p": either(controls and float(np.mean(top(controls)))),
            "control_share_at_0_9": either(controls and float(np.mean([c >= 0.9 for c in top(controls)]))),
            "control_acc": either(controls and float(np.mean([int(np.argmax(row["p"]) == row["label"]) for row in controls]))),
            "paired_confidence_drop": float(np.mean([kept - lost for lost, kept in pairs])) if pairs else None,
            "share_less_confident_than_control": float(np.mean([lost < kept for lost, kept in pairs])) if pairs else None}


# --- selective prediction -----------------------------------------------------------------------------------------------

def checked_selection(confidence, correct):
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct)
    if confidence.ndim != 1 or correct.shape != confidence.shape:
        raise ValueError("confidence and correctness must be equal-length vectors")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("confidence must be finite and in [0, 1]")
    if not np.isin(correct, [False, True]).all():
        raise ValueError("correctness must be boolean")
    return confidence, correct.astype(bool)


def risk_steps(confidence, correct):
    """The risk-coverage curve as arrays, one step per distinct confidence from the highest: (the confidence, how many
    questions are accepted at or above it, how many of those are wrong). Tied confidences are accepted together."""
    confidence, correct = checked_selection(confidence, correct)
    if not len(confidence):
        return confidence, np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-confidence, kind="stable")
    sorted_conf, wrong_so_far = confidence[order], np.cumsum(~correct[order])
    last_of_tie = np.r_[np.flatnonzero(sorted_conf[1:] != sorted_conf[:-1]), len(order) - 1]
    return sorted_conf[last_of_tie], last_of_tie + 1, wrong_so_far[last_of_tie]


def coverage_at_error(confidence, correct, budget):
    """The largest share of questions that can be accepted, most confident first, with the error among the accepted at
    most `budget` (jev-benchmarks' coverage at a fixed error budget). Honest probabilities score high; confidently wrong
    ones score low whatever their accuracy."""
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("error budget must be in [0, 1]")
    _, accepted, wrong = risk_steps(confidence, correct)
    within = np.flatnonzero(wrong <= budget * accepted)
    return float(accepted[within[-1]] / accepted[-1]) if len(within) else 0.0


def risk_coverage_curve(confidence, correct):
    thresholds, accepted, wrong = risk_steps(confidence, correct)
    return [{"threshold": float(t), "accepted": int(n), "errors": int(e), "coverage": float(n / accepted[-1]), "risk": float(e / n)}
            for t, n, e in zip(thresholds, accepted, wrong)]


def area_under_risk_coverage(confidence, correct):
    """AURC: the risk integrated over coverage, one step per distinct confidence (a tie group is one step)."""
    _, accepted, wrong = risk_steps(confidence, correct)
    if not len(accepted):
        return 0.0
    return float(np.sum(np.diff(np.r_[0, accepted]) * wrong / accepted) / accepted[-1])


def select_threshold(confidence, correct, budget, min_accepted=1):
    """The lowest confidence threshold whose accepted questions (at least `min_accepted`) are within the error budget,
    or None."""
    if not math.isfinite(budget) or not 0 <= budget <= 1 or min_accepted < 1:
        raise ValueError("invalid error budget or minimum accepted count")
    thresholds, accepted, wrong = risk_steps(confidence, correct)
    within = np.flatnonzero((wrong <= budget * accepted) & (accepted >= min_accepted))
    return float(thresholds[within[-1]]) if len(within) else None


def evaluate_threshold(confidence, correct, threshold):
    """What accepting every question at or above `threshold` gives (None: accept nothing)."""
    confidence, correct = checked_selection(confidence, correct)
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be in [0, 1] or None for abstain-all")
    taken = np.zeros(len(confidence), dtype=bool) if threshold is None else confidence >= threshold
    n, wrong = int(taken.sum()), int((taken & ~correct).sum())
    return {"threshold": threshold, "n": len(confidence), "accepted": n, "errors": wrong,
            "coverage": n / len(confidence) if len(confidence) else 0.0, "risk": wrong / n if n else None}


# --- long states --------------------------------------------------------------------------------------------------------

LENGTH_EDGES = (8192, 16384, 32768, 65536)   # state-token edges of calibration_by_length (long-context calibration)
LENGTH_METRICS = ("acc", "ece", "brier", "confident_error_rate")


def length_buckets(edges=LENGTH_EDGES):
    """[(name, lo, hi)] for states of lo <= tokens < hi (hi None: no bound): the disjoint buckets (under_8k, 8k_16k, 16k_32k,
    32k_64k, 64k_plus for the default edges), then the tail from each inner edge (8k_plus, 16k_plus, 32k_plus). Edges are
    increasing multiples of 1,024, which name them."""
    if not edges or list(edges) != sorted(set(edges)) or any(edge <= 0 or edge % 1024 for edge in edges):
        raise ValueError("length edges must be increasing positive multiples of 1024")
    name = lambda tokens: f"{tokens // 1024}k"
    buckets = []
    for lo, hi in zip((0, *edges), (*edges, None)):
        label = f"under_{name(hi)}" if lo == 0 else f"{name(lo)}_plus" if hi is None else f"{name(lo)}_{name(hi)}"
        buckets.append((label, lo, hi))
    return buckets + [(f"{name(lo)}_plus", lo, None) for lo in edges[:-1]]


def calibration_by_length(rows, lengths=None, edges=LENGTH_EDGES):
    """{bucket: {"n", acc, ece, brier, confident_error_rate}} of `rows`, scored as given (at T=1: pass them served), split by
    state tokens: the row's own `state_tokens` if it has them, else lengths[row id] ({record id: tokens}). A count is the
    encoded state segment, its <state> token included (d1a.model.encode). An empty bucket has n 0 and None values. The
    one home of per-length calibration."""
    def state_tokens(row):
        if row.get("state_tokens") is not None:
            return row["state_tokens"]
        if lengths is None or row["id"] not in lengths:
            raise KeyError(f"no state-token count for record {row['id']!r}")
        return lengths[row["id"]]
    counted = [(state_tokens(row), row) for row in rows]
    report = {}
    for label, lo, hi in length_buckets(edges):
        group = [row for tokens, row in counted if tokens >= lo and (hi is None or tokens < hi)]
        scores = metrics(group) if group else {}
        report[label] = {"n": len(group), **{m: scores.get(m) for m in LENGTH_METRICS}}
    return report


# --- temperatures -------------------------------------------------------------------------------------------------------

def tempered_row(row, temperature):
    """The row as a predictor calibrated at `temperature` would have returned it (probabilities and logits), so metrics()
    at T=1 scores it; per-row temperatures (out of fold) cannot go through metrics(rows, T). The recorded
    inference_temperature composes (served at T, tempered by T': T * T'), so raw_row always gets back to T=1."""
    return {**row, "p": probabilities_at_temperature(row, temperature).tolist(), "logits": shifted_logits(row, temperature).tolist(),
            "inference_temperature": row.get("inference_temperature", 1.0) * temperature}


def raw_row(row):
    """The row as the checkpoint returned it at T=1, which fit_temperature needs: served logits are z / T, so z = logits * T.
    A row served at T=1 comes back as it is."""
    temperature = row.get("inference_temperature", 1.0)
    require_temperature(temperature)
    if temperature == 1.0:
        return row
    if "logits" not in row:
        raise ValueError("cannot restore raw logits from a row that recorded none")
    z = np.asarray(row["logits"], dtype=float) * temperature
    weights = np.exp(z - z.max())
    return {**row, "p": (weights / weights.sum()).tolist(), "logits": z.tolist(), "inference_temperature": 1.0}


def recorded(row):
    """The row with its inference temperature explicit: rows saved before benchmarks recorded one (None) were served raw."""
    return row if row.get("inference_temperature") is not None else {**row, "inference_temperature": 1.0}


def scored_rows(rows):
    """The rows metrics are computed on: the clean variant of every knowable record."""
    return [row for row in rows if row["variant"] == "clean" and row["source"] != "unknowable"]


def served_at(rows, temperature):
    """The scored rows as a predictor at `temperature` would have returned them, whatever temperature they were saved at
    (their raw logits are restored first, so a Hub checkpoint's served rows and a trial's raw rows are read alike)."""
    return [tempered_row(raw_row(recorded(row)), temperature) for row in scored_rows(rows)]


def served(fit_rows, eval_rows, **fit_kwargs):
    """(the temperature fitted on fit_rows' raw logits, eval_rows served at it): how a checkpoint is calibrated, and how a
    comparison reads it, served against served. fit_kwargs override TEMPERATURE_FIT key by key."""
    temperature = fit_temperature(served_at(fit_rows, 1.0), **{**TEMPERATURE_FIT, **fit_kwargs})
    return temperature, served_at(eval_rows, temperature)


# How every released temperature is fitted (d1a.calibrate): the least mean NLL over a 121-point log grid on 0.25..4, each
# question weighted equally. fit_temperature's own default (81 points, per-task macro) is for quick reports, never shipped.
TEMPERATURE_FIT = {"aggregation": "micro", "points": 121}
TEMPERATURE_FIT_METHOD = f"min {TEMPERATURE_FIT['aggregation']} mean NLL over a {TEMPERATURE_FIT['points']}-point log grid 0.25..4"


def fit_temperature(rows, aggregation="macro", points=81):
    """The temperature with the least mean NLL (micro: over questions; macro: the mean of per-task means) on a
    `points`-point log grid from 0.25 to 4, fitted on raw (T=1) rows."""
    if aggregation not in ("micro", "macro"):
        raise ValueError("invalid calibration aggregation")
    rows = scored_rows(rows)
    if not rows:
        raise ValueError("cannot fit temperature without labelled calibration rows")
    if any(row.get("inference_temperature", 1.0) != 1.0 for row in rows):
        raise ValueError("fit temperature on raw logits, not previously calibrated outputs")
    grid = np.exp(np.linspace(np.log(0.25), np.log(4), points))
    if aggregation == "micro":
        weight = np.ones(len(rows))
    else:
        per_task = defaultdict(int)
        for row in rows:
            per_task[row["task"]] += 1
        weight = np.asarray([1.0 / per_task[row["task"]] for row in rows])
    loss = [np.average([nll_at_temperature(row, float(t)) for row in rows], weights=weight) for t in grid]
    return float(grid[int(np.argmin(loss))])


# --- clustered resampling -----------------------------------------------------------------------------------------------

def clusters_by_source(rows):
    """{source: [row indices of each (source, group) cluster]}, clusters and sources in order of first appearance."""
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[(row["source"], row["group"])].append(i)
    by_source = defaultdict(list)
    for (source, _), indices in members.items():
        by_source[source].append(np.asarray(indices, dtype=int))
    return by_source


def cluster_resamples(rows, samples, seed):
    """`samples` bootstrap resamples of `rows` as index arrays: within each source, its (source, group) clusters drawn with
    replacement, so a record's questions and variants move together. The resampling unit of every bootstrap here and of
    the research scripts that bootstrap their own statistic."""
    by_source, rng = clusters_by_source(rows), np.random.default_rng(seed)
    for _ in range(samples):
        drawn = []
        for clusters in by_source.values():
            drawn += [clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))]
        yield np.concatenate(drawn)


def grouped_folds(rows, folds, seed):
    """A fold id per row. Folds are disjoint in (source, group), so a record's questions and variants never straddle train
    and test, and dealt round-robin within each source, so every fold sees every source."""
    units = sorted({(row["source"], row["group"]) for row in rows})
    if len(units) < folds:
        raise ValueError("fewer distinct (source, group) units than folds")
    rng = np.random.default_rng(seed)
    fold_of, dealt = {}, 0
    for source in sorted({source for source, _ in units}):
        own = [unit for unit in units if unit[0] == source]
        for position, pick in enumerate(rng.permutation(len(own))):
            fold_of[own[pick]] = (dealt + position) % folds
        dealt += len(own)
    return np.asarray([fold_of[(row["source"], row["group"])] for row in rows])


def out_of_fold_rows(rows, folds=5, seed=0, **fit_kwargs):
    """Each fold's rows served at the temperature fitted on the other folds (fit_temperature(**fit_kwargs), raw rows
    required), so the result scores like any calibrated prediction. -> (the scored rows in input order, the temperature
    of each fold)."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rows = scored_rows(rows)
    fold_id = grouped_folds(rows, folds, seed)
    temperatures, result = [], list(rows)
    for fold in range(folds):
        held_out = fold_id == fold
        temperature = fit_temperature([row for row, out in zip(rows, held_out) if not out], **fit_kwargs)
        temperatures.append(temperature)
        for i in np.flatnonzero(held_out):
            result[i] = tempered_row(rows[i], temperature)
    return result, temperatures


def percentile_interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def cross_validated_temperature(rows, folds=5, seed=0, samples=1000, **fit_kwargs):
    """Out-of-fold calibration against the raw rows: their metrics, and a 95% cluster-bootstrap interval on raw and
    out-of-fold ECE and on their paired difference; `separated` when that difference's interval excludes zero."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    rows = scored_rows(rows)
    calibrated, temperatures = out_of_fold_rows(rows, folds, seed, **fit_kwargs)
    shown = ("n", "ece", "brier", "nll", "confident_error_rate", "coverage_at_5pct_error")
    before, after = metrics(rows), metrics(calibrated)
    correct = np.asarray([np.argmax(row["p"]) == row["label"] for row in rows])   # the argmax does not depend on the temperature
    top = np.asarray([[max(row["p"]) for row in group] for group in (rows, calibrated)])
    draws = np.asarray([[ece(top[0][idx], correct[idx]), ece(top[1][idx], correct[idx])] for idx in cluster_resamples(rows, samples, seed)])
    intervals = {"raw": percentile_interval(draws[:, 0]), "out_of_fold": percentile_interval(draws[:, 1]),
                 "delta": percentile_interval(draws[:, 1] - draws[:, 0])}
    return {"raw": {k: before[k] for k in shown}, "out_of_fold": {k: after[k] for k in shown}, "ece_ci95": intervals,
            "separated": bool(intervals["delta"][1] < 0 or intervals["delta"][0] > 0), "temperatures": temperatures,
            "folds": folds, "seed": seed, "samples": samples, "groups": sum(len(c) for c in clusters_by_source(rows).values()),
            "unit": "source-stratified (source, group); sibling questions and variants stay together"}


RECOMPUTED = {"coverage_at_5pct_error", "coverage_at_1pct_error", "aurc", "ece"}   # not means of question_scores


def paired_bootstrap(candidate, reference, samples=1000, seed=0, metric="nll", aggregation="macro"):
    """The candidate's `metric` minus the reference's on the same clean questions (micro: over questions; macro: the mean
    of per-task differences), with a 95% paired cluster-bootstrap interval. Additive metrics are means of
    question_scores; the others (RECOMPUTED) are recomputed on every resample."""
    additive = metric not in RECOMPUTED
    if (additive and metric not in question_scores({"p": [1.0], "label": 0})) or aggregation not in ("micro", "macro") or samples < 1:
        raise ValueError("unsupported bootstrap metric, aggregation, or sample count")

    def by_question(rows):
        found = {}
        for row in scored_rows(rows):
            key = row["id"], row["question"]
            if key in found:
                raise ValueError("duplicate paired example")
            found[key] = row
        return found

    mine, theirs = by_question(candidate), by_question(reference)
    if not mine or mine.keys() != theirs.keys():
        raise ValueError("paired comparison requires identical complete clean examples")
    order = sorted(mine)
    for key in order:
        a, b = mine[key], theirs[key]
        if a["keys"] != b["keys"] or a["label"] != b["label"]:
            raise ValueError("paired comparison labels or option order differ")
        if any(a[field] != b[field] for field in ("source", "group", "task", "type")):
            raise ValueError("paired comparison group or task metadata differ")
    aligned = [mine[key] for key in order]
    tasks = np.asarray([row["task"] for row in aligned])

    def columns(indexed):
        rows = [indexed[key] for key in order]
        return (np.asarray([max(row["p"]) for row in rows]), np.asarray([np.argmax(row["p"]) == row["label"] for row in rows]),
                np.asarray([question_scores(row)[metric] for row in rows]) if additive else None)
    sides = columns(mine), columns(theirs)

    def statistic(idx, side):
        top, correct, per_question = side
        if per_question is not None:
            return float(per_question[idx].mean())
        top, correct = top[idx], correct[idx]
        if metric == "aurc":
            return area_under_risk_coverage(top, correct)
        if metric == "ece":
            return ece(top, correct)
        return coverage_at_error(top, correct, 0.05 if metric == "coverage_at_5pct_error" else 0.01)

    def difference(idx):
        parts = [idx] if aggregation == "micro" else [idx[tasks[idx] == task] for task in np.unique(tasks[idx])]
        return float(np.mean([statistic(part, sides[0]) - statistic(part, sides[1]) for part in parts]))

    draws = [difference(idx) for idx in cluster_resamples(aligned, samples, seed)]
    return {f"{aggregation}_{metric}_delta": difference(np.arange(len(order))), "ci95": percentile_interval(draws),
            "samples": samples, "groups": sum(len(c) for c in clusters_by_source(aligned).values()), "aggregation": aggregation,
            "unit": "source-stratified original record; sibling questions stay together",
            "method": "paired cluster percentile bootstrap; full statistic recomputed in each resample"}
