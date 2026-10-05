"""d1a.benchmark with stand-in predictors: rows from predictions, the report, evaluate_records' files and its failure
rules, and the command line."""
import copy
import json
import sys
import threading
import time

import pytest

from d1a import benchmark as B
from d1a.api import question_keys
from d1a.model import ContextOverflow
from d1a.suite import CONTEXT, SERVING_CONTEXT, read_json

CRITERIA = {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"}


def record(i=0, variant="clean", **meta):
    return {"state": f"The shoes are the wrong size ({i}).", "questions": {"reason": {
        "type": "choice", "instructions": "Why return the shoes?", "criteria": dict(CRITERIA), "label": "size", "src": "fixture"}},
        "_meta": {"id": f"item-{i}", "group_id": f"item-{i}", "source": "fixture", "variant": variant, **meta}}


def answer(p=(0.8, 0.1, 0.1), keys=tuple(CRITERIA), **extra):
    return {"probabilities": {"reason": dict(zip(keys, p))}, "latency_ms": 1.0, **extra}


# --- rows ---------------------------------------------------------------------------------------------------------------

def test_labels_per_question_type():
    assert B.labels({"type": "choice", "criteria": {"a": None, "b": None}, "label": "b"}) == (["a", "b"], 1)
    assert B.labels({"type": "noul", "label": True}) == (question_keys("noul", None), 1)   # P(true) is the second option
    assert B.labels({"type": "score", "criteria": ["lo", "hi"], "label": 0}) == (["0", "1"], 0)


def test_a_row_per_question():
    (row,) = B.prediction_rows(record(control_id="c"), answer((0.1, 0.1, 0.8), keys=("color", "damage", "size")))
    assert row == {"id": "item-0", "group": "item-0", "question": "reason", "source": "fixture", "task": "fixture", "type": "choice",
                   "variant": "clean", "keys": ["size", "damage", "color"], "label": 0, "control_id": "c", "pair_id": None, "sibling": None,
                   "parent": "item-0", "p": [0.8, 0.1, 0.1], "raw_probability_sum": 1.0, "zero_count": 0}   # p in option order
    assert list(row) == list(B.prediction_rows(record(), answer())[0])                                      # a fixed key order


def test_variants_point_at_their_clean_parent():
    assert B.prediction_rows(record(variant="permuted", parent_id="item-7"), answer())[0]["parent"] == "item-7"
    old_style = record(variant="permuted"); old_style["_meta"]["group_id"] = "item-3"   # frozen before parent_id: kept in group_id
    assert B.prediction_rows(old_style, answer())[0]["parent"] == "item-3"


def test_logits_are_recorded_in_option_order():
    with_logits = answer(logits={"reason": {"color": -2.0, "damage": -1.0, "size": 0.0}}, inference_temperature=1.0, kernels="flash")
    (row,) = B.prediction_rows(record(), with_logits)
    assert row["logits"] == [0.0, -1.0, -2.0] and row["inference_temperature"] == 1.0 and row["kernels"] == "flash"
    for bad in ({"size": float("inf"), "damage": 0.0, "color": 0.0}, {"size": 0.0, "damage": 0.0}):
        with pytest.raises(ValueError, match="logit"):
            B.prediction_rows(record(), {**with_logits, "logits": {"reason": bad}})


def test_returned_distributions_are_checked_and_renormalised():
    p, total = B.validate_distribution({"x": 0.3333, "y": 0.3333, "z": 0.3333}, ["x", "y", "z"])   # 4-decimal answers
    assert total == pytest.approx(0.9999) and p.sum() == pytest.approx(1.0)
    for raw, message in (({"x": 0.5}, "keys"), ({"x": float("nan"), "y": 0.5}, "non-finite"), ({"x": -0.1, "y": 1.1}, "out-of-range"),
                         ({"x": 0, "y": 0}, "sum"), ({"x": 0.6, "y": 0.6}, "sum")):
        with pytest.raises(ValueError, match=message):
            B.validate_distribution(raw, ["x", "y"])
    with pytest.raises(ValueError, match="answer IDs"):
        B.prediction_rows(record(), {"probabilities": {}})
    assert B.prediction_rows(record(), answer((1.0, 0.0, 0.0)))[0]["zero_count"] == 2


# --- the report ---------------------------------------------------------------------------------------------------------

def rows_of(*pairs):
    return [row for rec, ans in pairs for row in B.prediction_rows(rec, ans)]


def test_summary_objective_heldout_and_policy_counts():
    unknowable = record(9, source="unknowable"); unknowable["questions"]["reason"]["src"] = "unknowable_test"
    rows = rows_of((record(0), answer()), (record(1, source="new"), answer((0.2, 0.7, 0.1))), (unknowable, answer((0.5, 0.25, 0.25))))
    report = B.summarize(rows, temperature=2.0, heldout_sources=("new",))
    assert report["objective"] == pytest.approx(-report["tasks"]["fixture"]["nll"])     # unknowable_* tasks are not in it
    assert report["clean"]["n"] == report["calibrated_clean"]["n"] == 2                 # nor in either accuracy denominator
    assert report["heldout_tasks"]["fixture"]["n"] == 1 and B.summarize(rows)["heldout_tasks"] == {}
    assert report["unknowable"]["n"] == 1 and report["temperature"] == 2.0
    assert report["metric_policy"]["selective_ties"] == "whole_confidence_groups" and report["metric_policy"]["returned_zeros"] == 0
    assert list(report) == ["objective", "paired_flip", "unknowable", "clean", "tasks", "variants", "heldout_tasks", "permutation",
                            "temperature", "calibrated_clean", "metric_policy"]


def test_permuted_variants_are_realigned_to_their_parent():
    rows = []
    for i in range(2):
        clean = record(i); clean["_meta"]["group_id"] = "shared"                        # siblings share a bootstrap group
        permuted = copy.deepcopy(clean); permuted["questions"]["reason"]["criteria"] = dict(reversed(CRITERIA.items()))
        permuted["_meta"].update(id=f"item-{i}/permuted", parent_id=f"item-{i}", variant="permuted")
        shifted = (0.7, 0.2, 0.1) if i == 0 else (0.1, 0.2, 0.7)                        # in each record's own option order
        rows += rows_of((clean, answer((0.7, 0.2, 0.1))), (permuted, answer(shifted, keys=tuple(permuted["questions"]["reason"]["criteria"]))))
    permutation = B.summarize(rows)["permutation"]
    assert permutation == {"n": 2, "mean_max_delta": pytest.approx(0.3), "flip_rate": 0.5}   # item-0 moved its answer, item-1 kept it


def test_paired_flip_counts_state_driven_flips():
    keys = ["no", "yes"]
    row = lambda pair, sibling, y, p: {"pair_id": pair, "sibling": sibling, "keys": keys, "label": y, "p": p}
    perfect = [row(f"p{i}", s, y, [1.0 - y, float(y)]) for i in range(4) for s, y in (("a", i % 2), ("b", 1 - i % 2))]
    assert B.paired_flip(perfect) == {"pairs": 4, "flip_rate": 1.0, "both_correct_rate": 1.0}
    constant = [{**r, "p": [1.0, 0.0]} for r in perfect]                                 # ignores the state: never flips
    assert B.paired_flip(constant) == {"pairs": 4, "flip_rate": 0.0, "both_correct_rate": 0.0}
    steady = perfect + [row("q", "a", 1, [0.0, 1.0]), row("q", "b", 1, [1.0, 0.0])]
    assert B.paired_flip(steady)["invariant_pairs"] == 1 and B.paired_flip(steady)["invariance_rate"] == 0.0
    with pytest.raises(ValueError, match="incomplete"):
        B.paired_flip(perfect[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        B.paired_flip(perfect + perfect[:1])
    assert B.paired_flip([{"keys": keys, "label": 0, "p": [1.0, 0.0]}]) is None


def test_rows_feed_the_paired_bootstrap():
    from d1a.metrics import paired_bootstrap
    rows = B.prediction_rows(record(), answer())
    assert B.summarize(rows)["objective"] == pytest.approx(-B.summarize(rows)["clean"]["nll"])
    assert paired_bootstrap(rows, rows, samples=50)["ci95"] == [0, 0]
    for other, message in (([], "identical"), ([{**rows[0], "group": "other"}], "group")):
        with pytest.raises(ValueError, match=message):
            paired_bootstrap(rows, other)
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap(rows + rows, rows)


# --- evaluate_records ---------------------------------------------------------------------------------------------------

def test_files_and_report(tmp_path):
    records = [record(i) for i in range(3)]
    report, rows = B.evaluate_records(records, lambda r: answer(latency_ms=float(r["_meta"]["id"][-1])), tmp_path / "out", temperature=1.5)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["predictions.jsonl", "report.json", "rows.json"]
    assert read_json(tmp_path / "out" / "rows.json") == rows and read_json(tmp_path / "out" / "report.json") == json.loads(json.dumps(report))
    logged = [json.loads(line) for line in (tmp_path / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["id"] for entry in logged] == ["item-0", "item-1", "item-2"] and all(len(entry["request_sha256"]) == 64 for entry in logged)
    assert report["coverage"] == {"requested_records": 3, "requested_questions": 3, "evaluated_records": 3, "evaluated_questions": 3,
                                  "rejected_records": 0, "truncated_records": 0}
    assert report["latency_ms"] == {"median": 1.0, "p95": pytest.approx(1.9)}
    assert report["calibration"] == {"inference_temperature": None, "additional_temperature": 1.5, "logits_recorded": False}
    assert "long_rows" not in report
    with pytest.raises(FileExistsError):                                                # never overwrites a run
        B.evaluate_records(records, lambda r: answer(), tmp_path / "out")


def test_a_failed_prediction_leaves_no_score(tmp_path):
    def fail(record):
        raise ValueError("invalid prediction")
    with pytest.raises(ValueError):
        B.evaluate_records([record()], fail, tmp_path / "out")
    failure = read_json(tmp_path / "out" / "failure.json")
    assert failure == {"coverage": {"requested_records": 1, "requested_questions": 1, "evaluated_records": 0, "evaluated_questions": 0,
                                    "rejected_records": 1, "truncated_records": 0}, "record_id": "item-0", "error_type": "ValueError"}
    assert not (tmp_path / "out" / "report.json").exists()


def test_overlong_records_are_skipped_only_when_asked(tmp_path):
    records = [record(i) for i in range(3)]

    def predictor(r):
        if r["_meta"]["id"] == "item-1": raise ContextOverflow("branch too long: 1590 tokens with a 7000-token state (row limit 8192)")
        return answer()

    report, _ = B.evaluate_records(records, predictor, tmp_path / "published", skip_overlong=True)
    assert report["coverage"]["evaluated_records"] == 2 and report["coverage"]["rejected_records"] == 1
    assert read_json(tmp_path / "published" / "rejected.json") == [{"id": "item-1", "error": "branch too long: 1590 tokens with a 7000-token state (row limit 8192)"}]
    with pytest.raises(ContextOverflow):
        B.evaluate_records(records, predictor, tmp_path / "admitted")
    assert read_json(tmp_path / "admitted" / "failure.json")["error_type"] == "ContextOverflow"
    with pytest.raises(ValueError):                                                     # any other error aborts even when skipping
        B.evaluate_records(records, lambda r: {"probabilities": {}, "latency_ms": 1.0}, tmp_path / "other", skip_overlong=True)


def test_long_rows_are_reported_with_their_kernels(tmp_path):
    report, _ = B.evaluate_records([record(0), record(1)], lambda r: answer(kernels="sdpa") if r["_meta"]["id"] == "item-1" else answer(),
                                   tmp_path / "out")
    assert report["long_rows"] == {"count": 1, "records": 1, "kernels": ["sdpa"], "threshold": B.ROW_PASS_TOKENS}


def test_concurrent_scoring_matches_the_sequential_loop(tmp_path):
    records = [record(i) for i in range(6)]

    class Predictor:
        def __init__(self, concurrency, fail=None):
            self.concurrency, self.fail, self.running, self.peak, self.lock = concurrency, fail, 0, 0, threading.Lock()

        def __call__(self, r):
            with self.lock:
                self.running += 1; self.peak = max(self.peak, self.running)
            time.sleep(0.02)
            with self.lock:
                self.running -= 1
            i = int(r["_meta"]["id"].split("-")[1])
            if i == self.fail: raise ContextOverflow("too long")
            p = 0.5 + 0.05 * i
            return answer((p, 1 - p, 0.0), latency_ms=float(i))

    one, four = Predictor(1), Predictor(4)
    r1, rows1 = B.evaluate_records(records, one, tmp_path / "one")
    r4, rows4 = B.evaluate_records(records, four, tmp_path / "four")
    assert rows1 == rows4 and r1["coverage"] == r4["coverage"] and r1["latency_ms"] == r4["latency_ms"]
    assert (tmp_path / "one" / "predictions.jsonl").read_bytes() == (tmp_path / "four" / "predictions.jsonl").read_bytes()
    assert one.peak == 1 and four.peak > 1
    skipped, _ = B.evaluate_records(records, Predictor(4, fail=2), tmp_path / "skip", skip_overlong=True)
    assert skipped["coverage"]["evaluated_records"] == 5 and [r["id"] for r in read_json(tmp_path / "skip" / "rejected.json")] == ["item-2"]
    with pytest.raises(ContextOverflow):                                                # surfaces at the failing record's position
        B.evaluate_records(records, Predictor(4, fail=2), tmp_path / "abort")
    failure = read_json(tmp_path / "abort" / "failure.json")
    assert failure["record_id"] == "item-2" and failure["coverage"]["evaluated_records"] == 2


# --- command line -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("context", ["training", "serving"])
def test_data_is_scored_under_the_chosen_context_and_skips_are_announced(tmp_path, monkeypatch, capsys, context):
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps({k: v for k, v in record(i).items() if k != "_meta"}) + "\n" for i in range(3)), encoding="utf-8")
    seen = {}

    class Local:
        temperature = 1.0
        def __init__(self, run, device, opts, context): seen["context"] = context
        def __call__(self, r):
            if r["_meta"]["id"].endswith("1") and seen["context"] is CONTEXT: raise ContextOverflow("state exceeds 384 tokens")
            return answer((1 / 3, 1 / 3, 1 / 3))
    monkeypatch.setattr(B, "LocalPredictor", Local)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--run", "x", "--data", str(data), "--out", str(tmp_path / "out"), "--device", "cpu", "--context", context])
    B.main()
    report = read_json(tmp_path / "out" / "report.json")
    assert report["data"] == str(data) and report["split"] == "custom" and report["calibration_applied"] is False and report["remote"] is None
    if context == "serving":
        assert seen["context"] is SERVING_CONTEXT and report["coverage"]["rejected_records"] == 0
    else:
        assert seen["context"] is CONTEXT and report["coverage"]["rejected_records"] == 1
        assert "skipped 1 of 3 records" in capsys.readouterr().out


@pytest.mark.parametrize("args", [[], ["--run", "x", "--remote", "http://h", "--suite", "s"], ["--run", "x"], ["--run", "x", "--suite", "s", "--data", "d"],
                                  ["--run", "x", "--suite", "s", "--rotations", "0"], ["--remote", "http://h", "--suite", "s", "--remote-concurrency", "0"]])
def test_command_line_refuses_contradictory_arguments(args, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--out", str(tmp_path / "out"), *args])
    with pytest.raises(SystemExit) as stop:
        B.main()
    assert stop.value.code == 2 and not (tmp_path / "out").exists()
