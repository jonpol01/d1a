# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); dropped two checks bound to the removed experiments/ registrations; the tests of the removed Modal app and report scripts left; the contrastive generator's tests left with it (paired_flip and the held-out structure guard keep theirs).
import copy
import pathlib
import random

import pytest
import torch

from d1a.suite import record_digest
from d1a.train import question_loss


def choice_request():
    return {"state": "The shoes are the wrong size.", "questions": {"reason": {
        "type": "choice", "instructions": "Why return the shoes?",
        "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"},
        "label": "size", "src": "fixture",
    }}}


def test_score_loss_is_proper_at_true_distribution():
    logits = torch.tensor([0.2, 0.8]).log().requires_grad_()
    loss = sum(p * question_loss(logits, {"label": y, "qtype": "score"}, "cpu") for y, p in enumerate([0.2, 0.8]))
    loss.backward()
    assert logits.grad.abs().max().item() < 1e-6


def frozen_request(i=0):
    r = choice_request()
    r["_meta"] = {"id": f"item-{i}", "group_id": f"item-{i}", "source": "fixture", "variant": "clean"}
    return r


def test_source_sampling_does_not_depend_on_other_sources(monkeypatch):
    from d1a import data

    def convert(split, n, rng):
        value = rng.randrange(1000000)
        rng.origins = [{"row": value, "row_sha256": str(value), "text_sha256": str(value)}]
        return [choice_request()]

    monkeypatch.setattr(data, "SOURCES", {"agnews": (convert, "train", "test"), "mnli": (convert, "train", "test")})
    alone = data.build(1, only=["mnli"])
    together = data.build(1)
    assert alone[0] == next(r for r in together if r["_meta"]["source"] == "mnli")


def test_strict_encoding_rejects_truncation():
    from types import SimpleNamespace
    from d1a.model import encode

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=list(range(len(text))))

        def convert_tokens_to_ids(self, text):
            return 1000

    rec = {"state": "abcdefgh", "questions": [{"instr": "q", "options": ["a", "b"], "label": 0}]}
    assert encode(Tokenizer(), rec, max_state=4)["state_truncated"]
    with pytest.raises(ValueError, match="state exceeds"):
        encode(Tokenizer(), rec, max_state=4, strict=True)


def test_locked_split_and_hash_verification(tmp_path):
    import json
    from d1a.suite import digest, load_split, write_json, write_jsonl
    for name in ("development", "test"):
        write_jsonl(tmp_path / f"{name}.jsonl", [frozen_request()])
    manifest = {"files": {f"{name}.jsonl": {"sha256": digest(tmp_path / f"{name}.jsonl"), "records": 1} for name in ("development", "test")}}
    write_json(tmp_path / "manifest.json", manifest)
    assert len(load_split(tmp_path, "development")) == 1
    with pytest.raises(ValueError, match="locked test"):
        load_split(tmp_path, "test")
    write_jsonl(tmp_path / "development.jsonl", [{}])
    with pytest.raises(ValueError, match="checksum"):
        load_split(tmp_path, "development")


def test_api_payload_excludes_answers_and_metadata():
    from d1a.data import api_request
    clean = api_request(frozen_request())
    assert set(clean) == {"state", "questions"}
    assert set(clean["questions"]["reason"]) == {"type", "instructions", "criteria"}


def test_failed_prediction_cannot_produce_partial_score(tmp_path):
    from d1a.suite import read_json
    from d1a.benchmark import evaluate_records

    def fail(record):
        raise ValueError("invalid prediction")

    out = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        evaluate_records([frozen_request()], fail, out)
    failure = read_json(out / "failure.json")
    assert failure["coverage"]["requested_records"] == 1
    assert failure["coverage"]["evaluated_records"] == 0
    assert failure["coverage"]["rejected_records"] == 1
    assert not (out / "report.json").exists()


def test_overlong_records_are_rejected_only_for_suites_scored_as_published(tmp_path):
    """ContextOverflow from the encoder (any of its three limits) is a counted rejection under skip_overlong and an abort
    otherwise; every other ValueError still aborts either way."""
    from d1a.benchmark import evaluate_records
    from d1a.model import ContextOverflow
    from d1a.suite import read_json
    records = [frozen_request(0), frozen_request(1), frozen_request(2)]
    keys = list(records[0]["questions"]["reason"]["criteria"])

    def predictor(record):
        if record["_meta"]["id"] == "item-1": raise ContextOverflow("branch too long: 1590 tokens with a 7000-token state (row limit 8192)")
        return {"probabilities": {"reason": {k: 1.0 / len(keys) for k in keys}}, "latency_ms": 1.0}

    report, _ = evaluate_records(records, predictor, tmp_path / "published", skip_overlong=True)
    assert report["coverage"]["evaluated_records"] == 2 and report["coverage"]["rejected_records"] == 1
    assert [r["id"] for r in read_json(tmp_path / "published" / "rejected.json")] == ["item-1"]
    with pytest.raises(ContextOverflow):
        evaluate_records(records, predictor, tmp_path / "admitted")
    assert read_json(tmp_path / "admitted" / "failure.json")["error_type"] == "ContextOverflow"

    def other(record):
        raise ValueError("answer IDs differ from request IDs")
    with pytest.raises(ValueError):
        evaluate_records(records, other, tmp_path / "other", skip_overlong=True)


def test_missing_answers_and_nonfinite_probabilities_fail():
    from d1a.benchmark import prediction_rows, validate_distribution
    with pytest.raises(ValueError, match="answer IDs"):
        prediction_rows(frozen_request(), {"probabilities": {}})
    with pytest.raises(ValueError, match="non-finite"):
        validate_distribution({"x": float("nan"), "y": 0.5}, ["x", "y"])
    with pytest.raises(ValueError, match="sum"):
        validate_distribution({"x": 0, "y": 0}, ["x", "y"])


def test_task_macro_and_record_bootstrap():
    from d1a.benchmark import prediction_rows, summarize
    from d1a.metrics import paired_bootstrap
    pred = {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}}
    rows = prediction_rows(frozen_request(), pred)
    report = summarize(rows)
    assert report["objective"] == pytest.approx(-report["clean"]["nll"])
    assert paired_bootstrap(rows, rows, samples=50)["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(rows, [])


def test_batched_mask_matches_single_and_pads_are_invisible():
    from d1a.model import branch_mask, branch_mask_batch
    a, b = [0, 0, 1, 1, 2], [0, 1, 1]
    m = branch_mask_batch([a, b], "cpu")
    assert m.shape == (2, 1, 5, 5)
    assert torch.equal(m[0:1], branch_mask(a, "cpu"))
    assert torch.equal(m[1:2, :, :3, :3], branch_mask(b, "cpu"))
    allowed = m[1, 0] == 0
    assert not allowed[:3, 3:].any()          # real tokens never attend to padding
    assert allowed[3, 3] and allowed[4, 4]    # padded rows keep the diagonal, so softmax is finite
    assert not allowed[3, 1:3].any()          # pads belong to no question segment (state stays visible; rows are discarded)


def test_paired_flip_counts_state_driven_flips():
    """benchmark.paired_flip over a contrastive suite's rows: a constant model never flips; a perfect model flips every
    pair whose label changes and gets both siblings right; pairs whose label does not change measure invariance."""
    from d1a.benchmark import paired_flip
    keys = ["no", "yes"]
    row = lambda pair, sibling, y, p: {"pair_id": pair, "sibling": sibling, "keys": keys, "label": y, "p": p}
    perfect = [row(f"p{i}", s, y, [1.0 - y, float(y)]) for i in range(4) for s, y in (("a", i % 2), ("b", 1 - i % 2))]
    assert paired_flip(perfect) == {"pairs": 4, "flip_rate": 1.0, "both_correct_rate": 1.0}
    constant = [{**r, "p": [1.0, 0.0]} for r in perfect]
    assert paired_flip(constant) == {"pairs": 4, "flip_rate": 0.0, "both_correct_rate": 0.0}
    same = perfect + [row("q", "a", 1, [0.0, 1.0]), row("q", "b", 1, [1.0, 0.0])]
    assert paired_flip(same)["invariant_pairs"] == 1 and paired_flip(same)["invariance_rate"] == 0.0
    with pytest.raises(ValueError, match="incomplete"):
        paired_flip(perfect[:-1])
    assert paired_flip([{"keys": keys, "label": 0, "p": [1.0, 0.0]}]) is None


def test_permuted_variants_pair_with_their_parent_not_their_group():
    from d1a.benchmark import prediction_rows, summarize
    rows = []
    for i in range(2):
        r = frozen_request(i); r["_meta"]["group_id"] = "shared-pair"     # siblings share a bootstrap group
        permuted = copy.deepcopy(r); q = permuted["questions"]["reason"]   # a frozen suite's "permuted" variant
        q["criteria"] = dict(reversed(q["criteria"].items()))
        permuted["_meta"].update(id=f"item-{i}/permuted", parent_id=f"item-{i}", variant="permuted")
        for rec in (r, permuted):
            keys = list(rec["questions"]["reason"]["criteria"])
            rows += prediction_rows(rec, {"probabilities": {"reason": {k: (0.7 if k == rec["questions"]["reason"]["label"] else 0.3 / (len(keys) - 1)) for k in keys}}})
    report = summarize(rows)
    assert report["permutation"]["n"] == 2 and report["permutation"]["flip_rate"] == 0.0


def test_coverage_cannot_split_equal_confidence_ties():
    from d1a.metrics import coverage_at_error
    correct = [True] * 90 + [False] * 10
    assert coverage_at_error([0.99] * 100, correct, 0.05) == 0.0
    assert coverage_at_error([0.99] * 100, correct[::-1], 0.05) == 0.0
    assert coverage_at_error([0.99] * 100, correct, 0.1) == 1.0
    assert coverage_at_error([], [], 0.05) == 0.0


def test_risk_curve_thresholds_and_nonmonotone_risk():
    from d1a.metrics import coverage_at_error, risk_coverage_curve
    curve = risk_coverage_curve([0.99, 0.99, 0.9, 0.8], [True, False, True, True])
    assert [p["accepted"] for p in curve] == [2, 3, 4]
    assert [p["threshold"] for p in curve] == [0.99, 0.9, 0.8]
    assert [p["risk"] for p in curve] == pytest.approx([0.5, 1 / 3, 0.25])
    assert coverage_at_error([0.99, 0.99, 0.9, 0.8], [True, False, True, True], 0.25) == 1.0


@pytest.mark.parametrize("confidence,correct,budget", [
    ([float("nan")], [True], 0.05), ([1.1], [True], 0.05),
    ([0.5], [], 0.05), ([[0.5]], [True], 0.05), ([0.5], [True], -0.1),
])
def test_selective_metrics_reject_invalid_inputs(confidence, correct, budget):
    from d1a.metrics import coverage_at_error
    with pytest.raises(ValueError):
        coverage_at_error(confidence, correct, budget)


def test_fixed_threshold_does_not_reselect_using_evaluation_labels():
    from d1a.metrics import select_threshold, evaluate_threshold
    threshold = select_threshold([0.99, 0.98, 0.97, 0.6], [True, True, True, False], 0.05)
    assert threshold == 0.97
    result = evaluate_threshold([0.99, 0.7, 0.6], [False, True, True], threshold)
    assert result["accepted"] == 1 and result["errors"] == 1 and result["risk"] == 1.0
    assert result["coverage"] == pytest.approx(1 / 3)
    empty = evaluate_threshold([0.99], [True], None)
    assert empty["coverage"] == 0 and empty["risk"] is None


def test_global_coverage_bootstrap_recomputes_full_statistic():
    from d1a.metrics import metrics, paired_bootstrap
    candidate, reference = [], []
    for i in range(20):
        common = {"id": str(i), "group": "one-cluster", "source": "fixture", "task": "fixture",
                  "question": "q", "variant": "clean", "type": "choice", "keys": ["a", "b"], "label": 0}
        candidate.append({**common, "p": [0.99, 0.01] if i < 18 else [0.4, 0.6]})
        reference.append({**common, "p": [0.6, 0.4] if i < 18 else [0.01, 0.99]})
    assert metrics(candidate)["acc"] == metrics(reference)["acc"]
    result = paired_bootstrap(candidate, reference, samples=50, metric="coverage_at_5pct_error", aggregation="micro")
    assert result["micro_coverage_at_5pct_error_delta"] == pytest.approx(0.9)
    assert result["ci95"] == pytest.approx([0.9, 0.9])
    assert result["groups"] == 1
    assert paired_bootstrap(candidate, candidate, samples=50, metric="aurc", aggregation="micro")["ci95"] == [0, 0]


def test_bootstrap_rejects_duplicate_or_inconsistent_pairs():
    from d1a.benchmark import prediction_rows
    from d1a.metrics import paired_bootstrap
    rows = prediction_rows(frozen_request(), {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}})
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap(rows + rows, rows)
    with pytest.raises(ValueError, match="group"):
        paired_bootstrap(rows, [{**rows[0], "group": "different"}])


def test_temperature_preserves_argmax_but_not_cross_question_ranking():
    import numpy as np
    from d1a.metrics import probabilities_at_temperature
    rows = [{"p": [0.6, 0.2, 0.2]}, {"p": [0.55, 0.449, 0.001]}]
    calibrated = [probabilities_at_temperature(r, 2.0) for r in rows]
    assert max(rows[0]["p"]) > max(rows[1]["p"])
    assert calibrated[0].max() < calibrated[1].max()
    assert all(np.argmax(r["p"]) == p.argmax() for r, p in zip(rows, calibrated))


def test_calibration_uses_logits_without_probability_floor_distortion():
    import numpy as np
    from d1a.metrics import probabilities_at_temperature
    p = probabilities_at_temperature({"p": [1.0, 0.0], "logits": [0.0, -100.0]}, 2.0)
    assert p[1] == pytest.approx(np.exp(-50), rel=1e-6, abs=0)
    for temperature in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            probabilities_at_temperature({"p": [0.5, 0.5]}, temperature)


def test_unknowable_not_in_raw_or_calibrated_accuracy_denominator():
    from d1a.benchmark import prediction_rows, summarize
    rows = prediction_rows(frozen_request(), {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}}})
    rows.append({**rows[0], "id": "unk", "source": "unknowable", "task": "unknowable_test"})
    result = summarize(rows, temperature=2.0)
    assert result["clean"]["n"] == result["calibrated_clean"]["n"] == 1
    assert result["metric_policy"]["selective_ties"] == "whole_confidence_groups"


def test_logits_are_recorded_in_option_order():
    from d1a.benchmark import prediction_rows
    prediction = {"probabilities": {"reason": {"size": 0.8, "damage": 0.1, "color": 0.1}},
                  "logits": {"reason": {"color": -2.0, "damage": -2.0, "size": 0.0}}, "inference_temperature": 1.0}
    row = prediction_rows(frozen_request(), prediction)[0]
    assert row["logits"] == [0.0, -2.0, -2.0] and row["inference_temperature"] == 1.0
    prediction["logits"]["reason"]["size"] = float("inf")
    with pytest.raises(ValueError, match="logit"):
        prediction_rows(frozen_request(), prediction)


def test_risk_curve_area_has_explicit_tie_policy():
    from d1a.metrics import area_under_risk_coverage
    assert area_under_risk_coverage([0.99, 0.9, 0.8], [True, True, False]) == pytest.approx(1 / 9)
    assert area_under_risk_coverage([0.99] * 10, [True] * 9 + [False]) == pytest.approx(0.1)
    assert area_under_risk_coverage([0.99] * 10, [False] + [True] * 9) == pytest.approx(0.1)


def test_temperature_fit_requires_raw_rows_and_uses_true_logit_nll():
    from d1a.metrics import fit_temperature, nll_at_temperature
    row = {"variant": "clean", "source": "fixture", "task": "fixture", "p": [1.0, 0.0],
           "logits": [0.0, -100.0], "label": 1, "inference_temperature": 1.0}
    assert nll_at_temperature(row, 2.0) == pytest.approx(50.0)
    assert fit_temperature([row], aggregation="micro") == pytest.approx(4.0)
    with pytest.raises(ValueError, match="raw logits"):
        fit_temperature([{**row, "inference_temperature": 2.0}])


def test_cross_validated_temperature_is_group_disjoint_and_reports_intervals():
    import numpy as np
    from d1a.metrics import cross_validated_temperature
    weights = np.exp([3.0, 0.0]); p = (weights / weights.sum()).tolist()
    rows = []
    for source in ("a", "b"):
        for group in range(10):
            for _ in range(2):
                i = len(rows)
                rows.append({"id": str(i), "source": source, "group": f"g{group}", "task": "t", "type": "choice",
                             "variant": "clean", "question": "q", "keys": ["x", "y"],
                             "label": 1 if i % 4 == 3 else 0, "logits": [3.0, 0.0], "p": p, "inference_temperature": 1.0})
    from d1a.metrics import grouped_folds
    fold_id = grouped_folds(rows, 5, 0)
    fold_of = {(r["source"], r["group"]): f for r, f in zip(rows, fold_id)}
    assert all(fold_of[(r["source"], r["group"])] == f for r, f in zip(rows, fold_id))   # a group never straddles folds
    for source in ("a", "b"):
        assert {fold_of[(source, f"g{g}")] for g in range(10)} == set(range(5))   # every source in every fold
    result = cross_validated_temperature(rows, folds=5, samples=200)
    assert len(result["temperatures"]) == 5 and all(t > 1 for t in result["temperatures"])
    assert result["out_of_fold"]["ece"] < result["raw"]["ece"]
    lo, hi = result["ece_ci95"]["delta"]
    assert lo <= hi and isinstance(result["separated"], bool)
    assert result["raw"]["n"] == result["out_of_fold"]["n"] == 40


def test_raw_row_inverts_the_served_temperature():
    import numpy as np
    from d1a.metrics import fit_temperature, raw_row, tempered_row
    raw = {"variant": "clean", "source": "s", "task": "t", "type": "choice", "label": 0, "logits": [2.0, -1.0, 0.5],
           "p": (np.exp([2.0, -1.0, 0.5]) / np.exp([2.0, -1.0, 0.5]).sum()).tolist(), "inference_temperature": 1.0}
    served = tempered_row(raw, 2.3)
    back = raw_row(served)
    assert back["inference_temperature"] == 1.0 and np.allclose(back["p"], raw["p"])
    assert np.allclose(np.asarray(back["logits"]) - max(back["logits"]), np.asarray(raw["logits"]) - max(raw["logits"]))
    assert raw_row(raw) is raw
    fit_temperature([back], aggregation="micro")   # accepted as raw
    with pytest.raises(ValueError, match="raw logits"):
        fit_temperature([served])
    with pytest.raises(ValueError, match="recorded none"):
        raw_row({"p": [0.6, 0.4], "inference_temperature": 2.0})
    twice = tempered_row(served, 1.5)                        # a second temperature composes, and raw_row still restores T=1
    assert twice["inference_temperature"] == pytest.approx(2.3 * 1.5) and np.allclose(raw_row(twice)["p"], raw["p"])


def test_cross_validated_temperature_rejects_too_few_groups():
    from d1a.metrics import cross_validated_temperature
    rows = [{"id": str(i), "source": "a", "group": f"g{i}", "task": "t", "type": "choice", "variant": "clean",
             "label": 0, "logits": [1.0, 0.0], "p": [0.73, 0.27], "inference_temperature": 1.0} for i in range(3)]
    with pytest.raises(ValueError, match="fewer"):
        cross_validated_temperature(rows, folds=5)


def test_selective_metrics_include_confidence_ties():
    from d1a.metrics import metrics
    rows = [{"p": [0.99, 0.01], "label": y, "type": "noul"} for y in [0, 1]]
    report = metrics(rows)
    assert report["confident_error_rate"] == .5
    assert report["selective"]["0.5"] == {"coverage": 1.0, "accuracy": .5, "confidence_cutoff": .99}


def test_uneven_microbatches_have_equal_record_weight():
    from d1a.train import accumulation_records
    x = torch.arange(10, dtype=torch.float32)
    gradients = []
    for batch, accum in ((8, 1), (3, 3), (2, 4)):
        w = torch.tensor(1.0, requires_grad=True)
        for mb, start in enumerate(range(0, len(x), batch)):
            chunk = x[start:start + batch]
            ((w * chunk).sum() / accumulation_records(len(x), batch, accum, mb)).backward()
        gradients.append(w.grad.item())
    assert gradients[0] == gradients[2]
    assert accumulation_records(10, 3, 3, 2) == 9
    assert accumulation_records(10, 3, 3, 3) == 1

def test_v3_training_refuses_heldout_structure():
    from d1a.composition import DEV_SHAPES, SHAPES, canonical, push_negation, structure_keys
    from d1a.suite import validate_training
    r = {"_meta": {"source": "compositional", "family": "held_and_or"}}
    with pytest.raises(ValueError, match="held-out"):
        validate_training([r], {"trainable_sources": ["compositional"]})
    # a random tree is refused when its structure, up to operand order, numbering and De Morgan, is a held-out shape's
    assert canonical(("or", ("not", 0), ("and", 1, 2))) == canonical(SHAPES["held_or_not"])
    assert canonical(("not", ("and", 0, 1))) != canonical(("or", ("not", 0), ("not", 1)))
    assert canonical(push_negation(("not", ("and", 0, 1)))) == canonical(("or", ("not", 0), ("not", 1)))
    negated = canonical(push_negation(SHAPES["final_negation"]))           # held out only through De Morgan
    assert negated != canonical(SHAPES["final_negation"]) and negated in structure_keys(SHAPES["final_negation"])
    for held in (canonical(SHAPES[DEV_SHAPES[0]]), negated):
        with pytest.raises(ValueError, match="held-out"):
            validate_training([{"_meta": {"source": "compositional", "family": "rand:7", "structure": held}}], {"trainable_sources": ["compositional"]})
    validate_training([{"_meta": {"source": "compositional", "family": "nested_and"}}], {"trainable_sources": ["compositional"]})


def test_none_pair_is_minimal_and_relabelled():
    from d1a.data import none_pair, materialize
    req = {"state": "The shoes are the wrong size.", "questions": {"reason": {"type": "choice", "instructions": "Why?",
           "criteria": {"size": "Wrong size", "damage": "Damaged", "color": "Wrong color"}, "label": "size", "src": "t"}}}
    present, absent = none_pair(req, random.Random(3))
    pk, ak = list(present["questions"]["reason"]["criteria"]), list(absent["questions"]["reason"]["criteria"])
    assert len(pk) == 4 and present["questions"]["reason"]["label"] == "size"
    assert [k for k in pk if k != "size"] == ak                     # same order, true option removed, nothing else moved
    assert absent["questions"]["reason"]["label"] == ak[-1] or absent["questions"]["reason"]["label"] in ak
    assert absent["questions"]["reason"]["label"] not in req["questions"]["reason"]["criteria"]
    materialize(present); materialize(absent)
    assert none_pair({"state": "s", "questions": {"q": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}, random.Random(0)) == []

def test_missing_partition_is_fetched_and_verified(tmp_path, monkeypatch):
    import json
    from d1a import suite as S
    evals = tmp_path / "evals" / "x" / "decision-x"; evals.mkdir(parents=True)
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    import hashlib
    S.write_json(evals / "manifest.json", {"files": {"train.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served = tmp_path / "served.jsonl"; served.write_bytes(payload)
    calls = []
    def fake_download(repo, path, repo_type, revision):
        calls.append((repo, path, repo_type, revision)); return str(served)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert len(S.load_split(evals, "train")) == 1
    assert calls == [(S.SUITES_DATASET, "x/decision-x/train.jsonl", "dataset", S.SUITES_REVISION)]
    # a tampered mirror is rejected by the manifest hash
    (evals / "train.jsonl").unlink(); served.write_bytes(b'{"tampered": 1}\n')
    with pytest.raises(ValueError, match="checksum"):
        S.load_split(evals, "train")

def test_private_suite_fetches_from_its_own_mirror(tmp_path, monkeypatch):
    import hashlib
    from huggingface_hub.errors import RepositoryNotFoundError
    from d1a import suite as S
    evals = tmp_path / "evals" / "held" / "docs-x"; evals.mkdir(parents=True)
    payload = b'{"state": "s", "questions": {}, "_meta": {}}\n'
    S.write_json(evals / "manifest.json", {"mirror": {"dataset": S.PRIVATE_DATASET, "revision": "abc123"},
                                           "files": {"test.jsonl": {"sha256": hashlib.sha256(payload).hexdigest(), "records": 1}}})
    served = tmp_path / "served.jsonl"; served.write_bytes(payload)
    calls = []
    def fake_download(repo, path, repo_type, revision):
        calls.append((repo, path, repo_type, revision)); return str(served)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert len(S.load_split(evals, "test", allow_test=True)) == 1
    assert calls == [(S.PRIVATE_DATASET, "held/docs-x/test.jsonl", "dataset", "abc123")]
    # without access the Hub says "not found"; the loader says why, instead of a bare 404
    (evals / "test.jsonl").unlink()
    import httpx
    def denied(*a, **k): raise RepositoryNotFoundError("404 Client Error", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co")))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", denied)
    with pytest.raises(PermissionError, match="kev-private-evals"):
        S.load_split(evals, "test", allow_test=True)

def test_remote_predictor_maps_system_one_answers_and_retries(monkeypatch):
    import io, json
    from d1a.predictors import RemotePredictor
    rec = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}, "label": "a", "src": "t"},
                                       "y": {"type": "noul", "instructions": "i", "label": True, "src": "t"}}}
    calls = []
    class Resp:
        def __init__(self, body): self.body = body
        def read(self): return json.dumps(self.body).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def urlopen(req, timeout):
        calls.append(json.loads(req.data))
        if len(calls) == 1: raise OSError("503")
        return Resp({"model": "openjev-x", "answers": {"q": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}, "y": {"type": "noul", "noul": 0.2}}, "usage": {"input_tokens": 12}})
    p = RemotePredictor("http://example.test/", retries=2); monkeypatch.setattr("urllib.request.urlopen", urlopen); monkeypatch.setattr("time.sleep", lambda s: None)
    out = p(rec)
    assert out["probabilities"] == {"q": {"a": 0.7, "b": 0.3}, "y": {"true": 0.2, "false": 0.8}} and p.served_model == "openjev-x" and len(calls) == 2
    assert calls[0]["model"] == "d1a-latest" and "label" not in json.dumps(calls[0])        # labels never leave the machine

def test_concurrent_predictions_keep_record_order_and_sequential_failure_semantics(tmp_path):
    """A predictor with `concurrency` > 1 is scored on a thread pool; the rows, predictions.jsonl order and coverage must be
    what the sequential loop produces, and an exception surfaces at the failing record's position with the same coverage."""
    import threading, time
    from d1a.benchmark import evaluate_records
    from d1a.model import ContextOverflow
    from d1a.suite import read_json
    records = [frozen_request(i) for i in range(6)]
    keys = list(records[0]["questions"]["reason"]["criteria"])

    class Predictor:
        def __init__(self, concurrency, fail=None):
            self.concurrency, self.fail, self.in_flight, self.peak, self.lock = concurrency, fail, 0, 0, threading.Lock()
        def __call__(self, record):
            with self.lock:
                self.in_flight += 1; self.peak = max(self.peak, self.in_flight)
            time.sleep(0.02)
            with self.lock: self.in_flight -= 1
            i = int(record["_meta"]["id"].split("-")[1])
            if i == self.fail: raise ContextOverflow("too long")
            p = 0.5 + 0.05 * i
            return {"probabilities": {"reason": {keys[0]: p, keys[1]: 1 - p, **{k: 0.0 for k in keys[2:]}}}, "latency_ms": float(i)}

    seq = Predictor(1); par = Predictor(4)
    r1, rows1 = evaluate_records(records, seq, tmp_path / "seq"); r2, rows2 = evaluate_records(records, par, tmp_path / "par")
    assert rows1 == rows2 and r1["coverage"] == r2["coverage"] and r1["latency_ms"] == r2["latency_ms"]
    assert (tmp_path / "seq" / "predictions.jsonl").read_bytes() == (tmp_path / "par" / "predictions.jsonl").read_bytes()
    assert seq.peak == 1 and par.peak > 1

    r3, _ = evaluate_records(records, Predictor(4, fail=2), tmp_path / "skip", skip_overlong=True)
    assert r3["coverage"]["evaluated_records"] == 5 and [r["id"] for r in read_json(tmp_path / "skip" / "rejected.json")] == ["item-2"]
    with pytest.raises(ContextOverflow):
        evaluate_records(records, Predictor(4, fail=2), tmp_path / "abort")
    failure = read_json(tmp_path / "abort" / "failure.json")
    assert failure["record_id"] == "item-2" and failure["coverage"]["evaluated_records"] == 2

def test_top_bins_and_confidence_bias():
    from d1a.metrics import metrics
    rows = [{"p": [0.99, 0.01], "label": 0, "type": "noul"}, {"p": [0.99, 0.01], "label": 1, "type": "noul"}, {"p": [0.6, 0.4], "label": 0, "type": "noul"}]
    m = metrics(rows)
    assert m["top_bins"]["0.99"] == {"n": 2, "errors": 1, "error_rate": 0.5} and m["top_bins"]["0.9"]["n"] == 2
    assert abs(m["confidence_bias"] - ((0.99 + 0.99 + 0.6) / 3 - 2 / 3)) < 1e-9


def test_frozen_suites_load_under_any_locale(tmp_path):
    """Issue #12: frozen partitions contain non-ASCII text and are sha256-checked byte for byte, so they must be read as
    UTF-8 whatever the platform's preferred encoding is (Windows cp936 in the report; an ASCII C locale here), and their
    line endings must survive checkout (.gitattributes pins *.json / *.jsonl to LF)."""
    import os, subprocess, sys
    from d1a.suite import digest, write_json, write_jsonl
    suite = tmp_path / "evals" / "x" / "decision-x"; suite.mkdir(parents=True)
    record = {**frozen_request(), "state": "Zwölf Boxkämpfer — ‘quotes’ and é"}
    write_jsonl(suite / "development.jsonl", [record])
    write_json(suite / "manifest.json", {"files": {"development.jsonl": {"sha256": digest(suite / "development.jsonl"), "records": 1}}})
    env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "utf-8"}   # stdio only; open() still defaults to the locale
    code = f"import locale; from d1a.suite import load_split; r = load_split({str(suite)!r}, 'development'); print(locale.getpreferredencoding(False), r[0]['state'])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__)), encoding="utf-8")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith(record["state"]) and "UTF-8" not in out.stdout.split()[0].upper(), out.stdout
    attributes = (pathlib.Path(__file__).resolve().parents[1] / ".gitattributes").read_text(encoding="utf-8")
    assert "*.jsonl text eol=lf" in attributes and "*.json text eol=lf" in attributes


def test_rotation_averaging_cancels_a_position_bias():
    import math
    from d1a.api import question_keys
    from d1a.predictors import RotationAveraged
    content = {"a": 1.0, "b": 0.0, "c": -1.0}; position = [2.0, 0.0, 0.0]         # the first slot is favoured by +2 logits

    def biased(record):
        out = {"probabilities": {}, "logits": {}, "inference_temperature": 1.0, "latency_ms": 1.0}
        for qid, q in record["questions"].items():
            keys = question_keys(q["type"], q.get("criteria"))
            z = {k: (content.get(k, 0.0) + position[i] if q["type"] == "choice" else float(i)) for i, k in enumerate(keys)}
            s = sum(math.exp(v) for v in z.values())
            out["logits"][qid] = z; out["probabilities"][qid] = {k: math.exp(v) / s for k, v in z.items()}
        return out

    record = {"state": "s", "questions": {"c": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}, "n": {"type": "noul"}}}
    shifted = RotationAveraged.rotated(record, 1)
    assert list(shifted["questions"]["c"]["criteria"]) == ["b", "c", "a"] and shifted["questions"]["n"] == record["questions"]["n"]
    avg = RotationAveraged(biased, 3)
    one, other = avg(record), avg(shifted)
    assert one["rotations"] == 3 and one["latency_ms"] == 3.0
    for k in "abc":                                                               # order no longer matters after a full cycle
        assert one["probabilities"]["c"][k] == pytest.approx(other["probabilities"]["c"][k])
    z = one["logits"]["c"]; assert z["a"] - z["b"] == pytest.approx(1.0) and z["b"] - z["c"] == pytest.approx(1.0)   # the bias is a constant
    assert one["probabilities"]["n"] == pytest.approx(biased(record)["probabilities"]["n"])   # noul untouched
    with pytest.raises(ValueError):
        RotationAveraged(biased, 1)


def test_none_pair_leaves_soft_target_questions_alone():
    import random
    from d1a.data import none_pair
    q = {"type": "choice", "criteria": {"a": None, "b": None, "c": None}, "label": "a", "src": "s"}
    req = {"state": "x", "questions": {"q": q}}
    assert len(none_pair(req, random.Random(0))) == 2
    assert none_pair({"state": "x", "questions": {"q": {**q, "target": {"a": 0.5, "b": 0.5}}}}, random.Random(0)) == []


def test_served_fits_on_raw_rows_and_cluster_resamples_keep_groups_together():
    import numpy as np
    from d1a.metrics import TEMPERATURE_FIT, cluster_resamples, fit_temperature, served, served_at, tempered_row
    rng = np.random.default_rng(1)
    rows = []
    for g in range(20):
        for q in range(2):
            z = rng.normal(0, 1, 3); y = int(rng.integers(0, 3)); z[y] += 2.0; p = np.exp(z - z.max()); p /= p.sum()
            rows.append({"id": f"r{g}", "question": f"q{q}", "source": "s" if g % 2 else "t", "group": f"g{g}", "task": "k", "type": "choice",
                         "variant": "clean", "label": y, "logits": z.tolist(), "p": p.tolist(), "inference_temperature": None})
    temperature, out = served(rows, rows)
    raw = [{**r, "inference_temperature": 1.0} for r in rows]
    assert temperature == fit_temperature(raw, **TEMPERATURE_FIT)                      # None = recorded raw
    assert out == [tempered_row(r, temperature) for r in raw] == served_at(rows, temperature)
    assert served(out, rows)[0] == temperature                                         # fitting on served rows restores raw logits first
    assert np.allclose([r["p"] for r in served_at(out, temperature)], [r["p"] for r in out])   # re-serving served rows does not temper twice
    for idx in cluster_resamples(rows, 20, 0):
        drawn = [rows[i]["group"] for i in idx]
        assert all(drawn.count(g) % 2 == 0 for g in set(drawn))                        # both questions of a record move together
        assert sum(rows[i]["source"] == "s" for i in idx) == 20                        # stratified: each source keeps its size


def test_jsonl_round_trips_unicode_line_separators(tmp_path):
    from d1a.suite import read_jsonl, write_jsonl
    recs = [{"state": "line one\u2028line two"}, {"state": "next\x85record\u2029end"}, {"state": "plain"}]
    write_jsonl(tmp_path / "x.jsonl", recs)
    assert read_jsonl(tmp_path / "x.jsonl") == recs


@pytest.mark.parametrize("context", ["training", "serving"])
def test_data_scoring_context_and_skip_warning(tmp_path, monkeypatch, capsys, context):
    # --data scored under the training limits used to drop long records without a word; --context serving scores them
    import json as _json, sys
    import d1a.benchmark as bm
    from d1a.model import ContextOverflow
    from d1a.suite import CONTEXT, SERVING_CONTEXT
    data = tmp_path / "data.jsonl"
    data.write_text("".join(_json.dumps({k: v for k, v in frozen_request(i).items() if k != "_meta"}) + "\n" for i in range(3)), encoding="utf-8")
    seen = {}

    class Fake:
        temperature = 1.0
        def __init__(self, run, device, opts, context): seen["context"] = context
        def __call__(self, record):
            if record["_meta"]["id"].endswith("1") and seen["context"] is CONTEXT: raise ContextOverflow("state exceeds 384 tokens")
            keys = list(record["questions"]["reason"]["criteria"])
            return {"probabilities": {"reason": {k: 1.0 / len(keys) for k in keys}}, "latency_ms": 1.0}
    monkeypatch.setattr(bm, "LocalPredictor", Fake)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--run", "x", "--data", str(data), "--out", str(tmp_path / "out"), "--device", "cpu", "--context", context])
    bm.main()
    report = _json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    if context == "serving":
        assert seen["context"] is SERVING_CONTEXT and report["coverage"]["rejected_records"] == 0
    else:
        assert seen["context"] is CONTEXT and report["coverage"]["rejected_records"] == 1
        assert "skipped 1 of 3 records" in capsys.readouterr().out
