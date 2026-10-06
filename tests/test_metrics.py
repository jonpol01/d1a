"""d1a.eval.metrics on hand-built rows: every figure checked against a value worked out by hand, the tie and edge-case
policies, the temperature round trips, and the cluster structure of the bootstraps."""
import math

import numpy as np
import pytest

from d1a.eval import metrics as M


def row(p, label, **extra):
    return {"p": list(p), "label": label, "type": "choice", "variant": "clean", "source": "s", "task": "t", "group": "g", **extra}


def softmax(z):
    e = np.exp(np.asarray(z, dtype=float) - max(z))
    return (e / e.sum()).tolist()


# --- figures ------------------------------------------------------------------------------------------------------------

def test_ece_by_hand():
    # bins [0.5, 0.6): conf 0.55, right; [0.9, 1.0]: conf 0.95 and 1.0 (the last bin is closed), one right
    assert M.ece([0.55, 0.95, 1.0], [True, True, False]) == pytest.approx(1 / 3 * 0.45 + 2 / 3 * abs(0.5 - 0.975))
    assert M.ece([0.3], [False], bins=2) == pytest.approx(0.3)


def test_metrics_by_hand():
    rows = [row([0.7, 0.2, 0.1], 0), row([0.2, 0.5, 0.3], 2), row([0.95, 0.05], 0, type="noul"), row([0.92, 0.08], 1, type="noul")]
    m = M.metrics(rows)
    assert m["n"] == 4 and m["acc"] == 0.5
    assert m["nll"] == pytest.approx(-np.mean(np.log([0.7, 0.3, 0.95, 0.08])))
    assert m["brier"] == pytest.approx(np.mean([0.09 + 0.04 + 0.01, 0.04 + 0.25 + 0.49, 0.0025 * 2, 0.92 ** 2 + 0.92 ** 2]))
    assert m["mean_conf"] == pytest.approx(np.mean([0.7, 0.5, 0.95, 0.92]))
    assert m["confident_error_rate"] == 0.25 and m["coverage_at_0_9"] == 0.5 and m["accuracy_at_0_9"] == 0.5 and m["error_rate_at_0_9"] == 0.5
    assert m["confidence_bias"] == pytest.approx(m["mean_conf"] - 0.5)
    assert list(m) == ["n", "nll", "acc", "ece", "brier", "mean_conf", "confident_error_rate", "coverage_at_0_9", "accuracy_at_0_9",
                       "coverage_at_5pct_error", "coverage_at_1pct_error", "aurc", "error_rate_at_0_9", "confidence_bias", "top_bins", "selective"]
    with pytest.raises(ValueError, match="empty"):
        M.metrics([])


def test_score_questions_add_level_error_and_rps():
    m = M.metrics([row([0.1, 0.3, 0.6], 1, type="score")])
    assert m["score_mae"] == pytest.approx(abs(0.3 + 1.2 - 1))                  # expected level 1.5 against 1
    assert m["ranked_probability_score"] == pytest.approx(((0.1 - 0) ** 2 + (0.4 - 1) ** 2) / 2)


def test_top_bins_and_confidence_bias():
    m = M.metrics([row([0.99, 0.01], 0), row([0.99, 0.01], 1), row([0.6, 0.4], 0)])
    assert m["top_bins"]["0.99"] == {"n": 2, "errors": 1, "error_rate": 0.5}
    assert m["top_bins"]["0.9"]["n"] == 2 and m["top_bins"]["0.95"]["n"] == 2
    assert m["confidence_bias"] == pytest.approx((0.99 + 0.99 + 0.6) / 3 - 2 / 3)
    assert M.metrics([row([0.6, 0.4], 0)])["top_bins"]["0.9"] == {"n": 0, "errors": 0, "error_rate": None}


def test_selective_fractions_keep_confidence_ties_together():
    m = M.metrics([row([0.99, 0.01], 0), row([0.99, 0.01], 1)])
    assert m["confident_error_rate"] == 0.5
    assert m["selective"]["0.5"] == {"coverage": 1.0, "accuracy": 0.5, "confidence_cutoff": 0.99}   # half would split a tie


def test_grouped_metrics_sorted_by_value():
    rows = [row([0.9, 0.1], 0, task="b"), row([0.9, 0.1], 1, task="a"), row([0.6, 0.4], 0, task="a")]
    grouped = M.grouped_metrics(rows, "task")
    assert list(grouped) == ["a", "b"] and grouped["a"]["n"] == 2 and grouped["b"]["acc"] == 1.0


def test_unknowable_report():
    rows = [row([0.95, 0.05], 0, id="u1", source="unknowable", control_id="c1"), row([0.5, 0.5], 0, id="u2", source="unknowable"),
            row([0.9, 0.1], 0, id="c1", source="unknowable_control")]
    report = M.unknowable_report(rows)
    assert report["n"] == 2 and report["mean_max_p"] == pytest.approx(0.725) and report["share_at_0_9"] == 0.5
    assert report["control_mean_max_p"] == pytest.approx(0.9) and report["control_acc"] == 1.0
    assert report["paired_confidence_drop"] == pytest.approx(0.9 - 0.95) and report["share_less_confident_than_control"] == 0.0
    assert M.unknowable_report([row([0.9, 0.1], 0)]) is None
    assert M.unknowable_report([row([0.9, 0.1], 0, id="u", source="unknowable")])["control_acc"] is None


# --- selective prediction -----------------------------------------------------------------------------------------------

def test_coverage_never_splits_a_confidence_tie():
    right = [True] * 90 + [False] * 10
    assert M.coverage_at_error([0.99] * 100, right, 0.05) == 0.0                 # the tie's 10% error exceeds 5% whatever the order
    assert M.coverage_at_error([0.99] * 100, right[::-1], 0.05) == 0.0
    assert M.coverage_at_error([0.99] * 100, right, 0.1) == 1.0
    assert M.coverage_at_error([], [], 0.05) == 0.0


def test_risk_coverage_curve_and_area():
    curve = M.risk_coverage_curve([0.99, 0.99, 0.9, 0.8], [True, False, True, True])
    assert [(c["threshold"], c["accepted"], c["errors"]) for c in curve] == [(0.99, 2, 1), (0.9, 3, 1), (0.8, 4, 1)]
    assert [c["risk"] for c in curve] == pytest.approx([0.5, 1 / 3, 0.25])        # risk need not fall monotonically with coverage
    assert M.coverage_at_error([0.99, 0.99, 0.9, 0.8], [True, False, True, True], 0.25) == 1.0
    # one step per distinct confidence, a tie group being one step
    assert M.area_under_risk_coverage([0.99, 0.9, 0.8], [True, True, False]) == pytest.approx(1 / 9)
    assert M.area_under_risk_coverage([0.99] * 10, [True] * 9 + [False]) == pytest.approx(0.1)
    assert M.area_under_risk_coverage([0.99] * 10, [False] + [True] * 9) == pytest.approx(0.1)
    assert M.area_under_risk_coverage([], []) == 0.0


@pytest.mark.parametrize("confidence, correct, budget", [
    ([float("nan")], [True], 0.05), ([1.1], [True], 0.05), ([-0.1], [True], 0.05), ([0.5], [], 0.05), ([[0.5]], [True], 0.05),
    ([0.5], [2], 0.05), ([0.5], [True], -0.1), ([0.5], [True], 1.5), ([0.5], [True], float("nan"))])
def test_selective_inputs_are_checked(confidence, correct, budget):
    with pytest.raises(ValueError):
        M.coverage_at_error(confidence, correct, budget)


def test_a_threshold_is_chosen_once_and_then_applied():
    threshold = M.select_threshold([0.99, 0.98, 0.97, 0.6], [True, True, True, False], 0.05)
    assert threshold == 0.97
    assert M.select_threshold([0.99, 0.98], [True, True], 0.0, min_accepted=3) is None
    applied = M.evaluate_threshold([0.99, 0.7, 0.6], [False, True, True], threshold)   # evaluation labels do not move it
    assert (applied["accepted"], applied["errors"], applied["risk"]) == (1, 1, 1.0) and applied["coverage"] == pytest.approx(1 / 3)
    nothing = M.evaluate_threshold([0.99], [True], None)
    assert nothing["coverage"] == 0 and nothing["risk"] is None
    with pytest.raises(ValueError):
        M.evaluate_threshold([0.5], [True], 1.5)


# --- long states --------------------------------------------------------------------------------------------------------

def test_length_buckets_and_calibration_by_length():
    assert [name for name, _, _ in M.length_buckets()] == ["under_8k", "8k_16k", "16k_32k", "32k_64k", "64k_plus", "8k_plus", "16k_plus", "32k_plus"]
    assert M.length_buckets((1024,)) == [("under_1k", 0, 1024), ("1k_plus", 1024, None)]
    for bad in ((), (1000,), (2048, 1024), (1024, 1024)):
        with pytest.raises(ValueError):
            M.length_buckets(bad)
    rows = [row([0.9, 0.1], 0, id="a", state_tokens=100), row([0.9, 0.1], 1, id="b"), row([0.6, 0.4], 0, id="c", state_tokens=9000)]
    report = M.calibration_by_length(rows, lengths={"b": 20000})
    assert report["under_8k"] == {"n": 1, "acc": 1.0, "ece": pytest.approx(0.1), "brier": pytest.approx(0.02), "confident_error_rate": 0.0}
    assert report["8k_plus"]["n"] == 2 and report["16k_32k"]["acc"] == 0.0
    assert report["64k_plus"] == {"n": 0, "acc": None, "ece": None, "brier": None, "confident_error_rate": None}
    with pytest.raises(KeyError, match="'b'"):
        M.calibration_by_length(rows)


# --- temperatures -------------------------------------------------------------------------------------------------------

def test_temperature_keeps_each_argmax_but_can_reorder_confidence_across_questions():
    rows = [{"p": [0.6, 0.2, 0.2]}, {"p": [0.55, 0.449, 0.001]}]
    hot = [M.probabilities_at_temperature(r, 2.0) for r in rows]
    assert max(rows[0]["p"]) > max(rows[1]["p"]) and hot[0].max() < hot[1].max()
    assert [int(np.argmax(r["p"])) for r in rows] == [int(p.argmax()) for p in hot]


def test_temperatures_read_the_logits_not_the_floored_probabilities():
    assert M.probabilities_at_temperature({"p": [1.0, 0.0], "logits": [0.0, -100.0]}, 2.0)[1] == pytest.approx(math.exp(-50), rel=1e-6, abs=0)
    assert M.nll_at_temperature({"p": [1.0, 0.0], "logits": [0.0, -100.0], "label": 1}, 2.0) == pytest.approx(50.0)
    assert M.nll_at_temperature({"p": [1.0, 0.0], "label": 1}) == pytest.approx(-math.log(M.EPSILON))   # no logits: floored
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            M.probabilities_at_temperature({"p": [0.5, 0.5]}, bad)
    with pytest.raises(ValueError, match="logits"):
        M.probabilities_at_temperature({"p": [0.5, 0.5], "logits": [0.0]}, 2.0)


def test_fit_temperature_on_raw_rows_only():
    one = row([1.0, 0.0], 1, logits=[0.0, -100.0], inference_temperature=1.0)
    assert M.fit_temperature([one], aggregation="micro") == pytest.approx(4.0)  # the wrong answer pushes to the grid's top
    with pytest.raises(ValueError, match="raw logits"):
        M.fit_temperature([{**one, "inference_temperature": 2.0}])
    with pytest.raises(ValueError, match="without labelled"):
        M.fit_temperature([{**one, "variant": "permuted"}])
    with pytest.raises(ValueError, match="aggregation"):
        M.fit_temperature([one], aggregation="median")


def test_served_rows_round_trip_to_raw():
    z = [2.0, -1.0, 0.5]
    raw = row(softmax(z), 0, logits=z, inference_temperature=1.0)
    served = M.tempered_row(raw, 2.3)
    back = M.raw_row(served)
    assert back["inference_temperature"] == 1.0 and np.allclose(back["p"], raw["p"])
    assert np.allclose(np.subtract(back["logits"], max(back["logits"])), np.subtract(z, max(z)))
    assert M.raw_row(raw) is raw
    twice = M.tempered_row(served, 1.5)                                        # temperatures compose
    assert twice["inference_temperature"] == pytest.approx(2.3 * 1.5) and np.allclose(M.raw_row(twice)["p"], raw["p"])
    M.fit_temperature([back], aggregation="micro")
    with pytest.raises(ValueError, match="raw logits"):
        M.fit_temperature([served])
    with pytest.raises(ValueError, match="recorded none"):
        M.raw_row({"p": [0.6, 0.4], "inference_temperature": 2.0})
    assert M.recorded({"inference_temperature": None})["inference_temperature"] == 1.0


def test_served_fits_raw_rows_however_they_were_saved():
    rng = np.random.default_rng(1)
    rows = []
    for g in range(20):
        for q in range(2):
            z = rng.normal(0, 1, 3); y = int(rng.integers(0, 3)); z[y] += 2.0
            rows.append(row(softmax(z), y, id=f"r{g}", question=f"q{q}", source="s" if g % 2 else "t", group=f"g{g}",
                            logits=z.tolist(), inference_temperature=None))       # None: saved before temperatures were recorded
    temperature, out = M.served(rows, rows)
    raw = [{**r, "inference_temperature": 1.0} for r in rows]
    assert temperature == M.fit_temperature(raw, **M.TEMPERATURE_FIT)
    assert out == [M.tempered_row(r, temperature) for r in raw] == M.served_at(rows, temperature)
    assert M.served(out, rows)[0] == temperature                                # served rows are restored to raw before fitting
    assert np.allclose([r["p"] for r in M.served_at(out, temperature)], [r["p"] for r in out])   # and never tempered twice


# --- clustered resampling -----------------------------------------------------------------------------------------------

def clustered_rows(sources=("a", "b"), groups=10, per_group=2):
    out = []
    for source in sources:
        for g in range(groups):
            for _ in range(per_group):
                i = len(out)
                out.append(row(softmax([3.0, 0.0]), 1 if i % 4 == 3 else 0, id=str(i), question="q", source=source,
                               group=f"g{g}", logits=[3.0, 0.0], inference_temperature=1.0))
    return out


def test_cluster_resamples_move_records_whole_and_keep_each_sources_size():
    rows = clustered_rows(groups=5, per_group=3)
    for idx in M.cluster_resamples(rows, 30, 0):
        drawn = [(rows[i]["source"], rows[i]["group"]) for i in idx]
        assert all(drawn.count(unit) % 3 == 0 for unit in set(drawn))
        assert sum(rows[i]["source"] == "a" for i in idx) == 15
    assert [list(i) for i in M.cluster_resamples(rows, 3, 7)] == [list(i) for i in M.cluster_resamples(rows, 3, 7)]


def test_folds_are_group_disjoint_and_see_every_source():
    rows = clustered_rows()
    fold_id = M.grouped_folds(rows, 5, 0)
    fold_of = {}
    for r, f in zip(rows, fold_id):
        assert fold_of.setdefault((r["source"], r["group"]), f) == f
    for source in ("a", "b"):
        assert {fold_of[(source, f"g{g}")] for g in range(10)} == set(range(5))
    with pytest.raises(ValueError, match="fewer"):
        M.grouped_folds(rows[:4], 5, 0)


def test_cross_validated_temperature():
    result = M.cross_validated_temperature(clustered_rows(), folds=5, samples=200)
    assert len(result["temperatures"]) == 5 and all(t > 1 for t in result["temperatures"])   # 3:0 logits are over-confident at 75%
    assert result["out_of_fold"]["ece"] < result["raw"]["ece"]
    lo, hi = result["ece_ci95"]["delta"]
    assert lo <= hi and isinstance(result["separated"], bool) and result["groups"] == 20
    assert result["raw"]["n"] == result["out_of_fold"]["n"] == 40
    oof, temperatures = M.out_of_fold_rows(clustered_rows(), folds=5)
    assert temperatures == result["temperatures"] and all(r["inference_temperature"] > 1 for r in oof)
    with pytest.raises(ValueError, match="fewer"):
        M.cross_validated_temperature(clustered_rows(groups=1, per_group=1), folds=5)
    with pytest.raises(ValueError):
        M.cross_validated_temperature(clustered_rows(), samples=0)


def test_paired_bootstrap_recomputes_nonadditive_metrics_and_aggregates_by_task():
    candidate, reference = [], []
    for i in range(20):
        common = dict(id=str(i), question="q", group="one-cluster", source="fixture", task="x" if i % 2 else "y", keys=["a", "b"])
        candidate.append(row([0.99, 0.01] if i < 18 else [0.4, 0.6], 0, **common))
        reference.append(row([0.6, 0.4] if i < 18 else [0.01, 0.99], 0, **common))
    assert M.metrics(candidate)["acc"] == M.metrics(reference)["acc"]
    coverage = M.paired_bootstrap(candidate, reference, samples=50, metric="coverage_at_5pct_error", aggregation="micro")
    assert coverage["micro_coverage_at_5pct_error_delta"] == pytest.approx(0.9) and coverage["ci95"] == pytest.approx([0.9, 0.9])
    assert coverage["groups"] == 1                                              # one cluster: every resample is the whole set
    assert M.paired_bootstrap(candidate, candidate, samples=50, metric="aurc", aggregation="micro")["ci95"] == [0, 0]
    macro = M.paired_bootstrap(candidate, reference, samples=20, metric="acc", aggregation="macro")
    assert macro["macro_acc_delta"] == 0.0
    with pytest.raises(ValueError, match="identical"):
        M.paired_bootstrap(candidate, reference[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        M.paired_bootstrap(candidate + candidate[:1], reference)
    with pytest.raises(ValueError, match="option order"):
        M.paired_bootstrap(candidate, [{**r, "keys": ["b", "a"]} for r in reference])
    with pytest.raises(ValueError, match="group"):
        M.paired_bootstrap(candidate, [{**r, "group": "other"} for r in reference])
    with pytest.raises(ValueError, match="unsupported"):
        M.paired_bootstrap(candidate, reference, metric="vibes")
