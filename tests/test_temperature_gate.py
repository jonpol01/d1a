"""d1a.training.calibrate's adopt-or-keep gate (--judge, --guard, --confirm, --locked; d1a.training.temperature_gate), the roles
a rows file may not play twice, the selections refused before anything is fitted, and the fit pool's clusters. Rows are
synthetic two-option questions; no model is loaded. Each rule and confirmation case fails exactly one criterion, so removing or
loosening that criterion fails its case."""
import json
import shutil

import numpy as np
import pytest

from d1a.backends.checkpoint import Meta, read_meta, write_meta
from d1a.eval.metrics import paired_bootstrap, served_at
from d1a.eval.suite import digest, read_json
from d1a.training import calibrate
from d1a.training.temperature_gate import confirmation, interval, keyed, rule

SHIPPED, CANDIDATE = 1.0, 2.0
# (margin, right, wrong) groups: logits [margin, 0], top probability sigmoid(margin / T); figures are CANDIDATE vs SHIPPED
WINS = (2.0, 292, 108)                         # accuracy 0.73, top 0.88 -> 0.73: ECE 0.151 -> 0.001, Brier -0.045 [-0.071, -0.020]
NOISY = (2.0, 73, 27)                          # the same gain on 100 questions: Brier -0.0455 [-0.0994, +0.0024], above 0 but within 0.005
SHARPER = ((4.0, 391, 109), (1.0, 366, 134))   # Brier -0.018 [-0.030, -0.007], ECE 0.1005 -> 0.1042: better Brier, worse ECE
WORSE = (0.16, 54, 46)                         # calibrated at SHIPPED: ECE +0.020 (beyond 0.005, within 0.05), Brier +0.0008
SLIGHTLY = (0.024, 253, 247)                   # calibrated at SHIPPED: ECE +0.003 (within 0.005)
SHIPPED_IS_RIGHT = (2.0, 88, 12)               # accuracy 0.88 = top at SHIPPED: ECE 0.001 -> 0.149, Brier +0.044 (beyond 0.005, within 0.05)


def questions(panel, *groups):
    """Raw scored rows keyed by `panel`: per (margin, right, wrong), questions whose label is the top option `right` times."""
    rows = []
    for margin, right, wrong in groups:
        top = float(1 / (1 + np.exp(-margin)))
        for label in [0] * right + [1] * wrong:
            rows.append({"id": f"custom/{len(rows)}", "group": f"custom/{len(rows)}", "source": "custom", "task": "custom", "type": "choice",
                         "keys": ["a", "b"], "variant": "clean", "question": "q", "label": label, "logits": [margin, 0.0], "p": [top, 1 - top]})
    return keyed(rows, panel)


def disagreeing(panel):
    """WORSE three times plus one question whose recorded p ranks the options against its logits (a corrupted read): its
    argmax at SHIPPED (= 1, the recorded p) is not its argmax at CANDIDATE. On 301 questions that flip moves accuracy by
    0.0033, within 0.005, and Brier by -0.0020."""
    rows = questions(panel, WORSE, WORSE, WORSE)
    return rows + [{**rows[0], "id": "custom/bad", "group": f"{panel}|custom/bad", "label": 0, "logits": [2.0, 0.0], "p": [0.3, 0.7]}]


@pytest.mark.parametrize("judge, guard, failed", [
    ({"a": [WINS]}, {"g": [SLIGHTLY]}, None),
    ({"a": [NOISY]}, {}, "judge panels pooled: Brier difference upper bound"),
    ({"a": SHARPER}, {}, "judge panels pooled: ECE"),
    ({"a": [WINS], "b": [WORSE]}, {}, "judge b: ECE"),
    ({"a": [WINS]}, {"g": [WORSE], "h": [SLIGHTLY]}, "guard g: ECE"),
    ({"a": [WINS], "b": [WORSE]}, {"b": [SLIGHTLY]}, "judge b: ECE"),   # a guard named like a judge panel never replaces its check
], ids=["adopted", "brier-interval", "pooled-ece", "judge-panel-ece", "guard-ece", "guard-named-like-a-judge"])
def test_each_criterion_of_the_rule_alone_keeps_the_shipped_temperature(judge, guard, failed):
    verdict = rule({n: questions(n, *g) for n, g in judge.items()}, {n: questions(n, *g) for n, g in guard.items()}, CANDIDATE, SHIPPED)
    assert verdict["adopt"] is (failed is None) and verdict["seed"] == 0
    assert [f.startswith(failed) for f in verdict["failed"]] == ([] if failed is None else [True]), verdict["failed"]


def kevs_paired_read(rows, candidate, shipped):
    """Kev's registered read (kev/rounds.py paired(): paired_bootstrap's micro Brier, 2,000 record-clustered resamples, seed 0)."""
    return paired_bootstrap(served_at(rows, candidate), served_at(rows, shipped), samples=2000, seed=0, metric="brier", aggregation="micro")["ci95"]


def test_the_brier_interval_is_kevs_registered_paired_read_in_any_row_order():
    rows = questions("a", WINS)
    assert rule({"a": rows[::-1]}, {}, CANDIDATE, SHIPPED)["judge"]["brier_ci95"] == pytest.approx(kevs_paired_read(rows, CANDIDATE, SHIPPED), abs=1e-12)
    # two partitions share every custom/<line> id (Kev's read refuses them): neither their rows' order nor theirs moves the bound
    other = questions("b", *SHARPER)
    bounds = [rule(judge, {}, CANDIDATE, SHIPPED)["judge"]["brier_ci95"] for judge in ({"a": rows, "b": other}, {"b": other[::-1], "a": rows[::-1]})]
    assert bounds[0] == bounds[1]
    # one file repeating an id, question and source under another group (devtools-v1 does): the file order moves nothing either
    twins = [{**t, "id": r["id"], "question": r["question"], "source": r["source"], "group": r["group"] + "/twin"} for r, t in zip(rows, other)]
    orders = (rows + twins, [row for pair in zip(twins, rows) for row in pair])
    assert len({tuple(rule({"a": order}, {}, CANDIDATE, SHIPPED)["judge"]["brier_ci95"]) for order in orders}) == 1


@pytest.mark.parametrize("tests, locked, failed", [
    ({"t": questions("t", WINS)}, {"l": questions("l", WORSE)}, None),
    ({"t": questions("t", SHIPPED_IS_RIGHT)}, {"l": questions("l", WORSE)}, "confirm t: ECE"),
    ({"t": questions("t", SLIGHTLY)}, {"l": questions("l", WORSE)}, "confirm t: ECE"),   # a rise within 0.005 is not a fall
    ({"t": questions("t", WINS)}, {"l": questions("l", SHIPPED_IS_RIGHT)}, "locked l: Brier"),
    ({"t": questions("t", WINS)}, {"l": disagreeing("l")}, "locked l: accuracy"),
], ids=["confirmed", "tests-ece", "tests-ece-within-tolerance", "locked-brier", "locked-accuracy"])
def test_each_confirmation_criterion_alone_keeps_the_shipped_temperature(tests, locked, failed):
    # Kev's round 28 stages: ECE falls on the test rows; on the locked rows Brier rises by at most 0.005 and accuracy holds
    confirmed = confirmation(tests, locked, CANDIDATE, SHIPPED)
    assert confirmed["passed"] is (failed is None)
    assert [f.startswith(failed) for f in confirmed["failed"]] == ([] if failed is None else [True]), confirmed["failed"]


def test_an_empty_panel_is_refused_not_judged():
    # Kev's metrics() refuses an empty population: an empty panel's ECE, 0 at both temperatures, would pass its check
    with pytest.raises(ValueError, match="empty panel"):
        rule({"a": questions("a", WINS)}, {"g": []}, CANDIDATE, SHIPPED)
    with pytest.raises(ValueError, match="empty panel"):
        confirmation({}, {"l": []}, CANDIDATE, SHIPPED)


def generated(true_temperature, n=200, seed=0, source="custom"):
    """`n` choice questions whose raw logits are [m, 0] and whose label follows softmax([m, 0] / T_true), as d1a.eval.benchmark
    writes a --data partition: id and group "<source>/<line>"."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        margin = float(rng.uniform(0.5, 6.0))
        label = int(rng.random() >= 1 / (1 + np.exp(-margin / true_temperature)))
        p0 = float(1 / (1 + np.exp(-margin)))
        rows.append({"id": f"{source}/{i}", "group": f"{source}/{i}", "source": source, "task": "custom", "type": "choice", "keys": ["a", "b"],
                     "variant": "clean", "question": "q", "label": label, "logits": [margin, 0.0], "p": [p0, 1 - p0], "inference_temperature": 1.0})
    return rows


def write(folder, rows, report=None):
    folder.mkdir()
    (folder / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    if report: (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return str(folder / "rows.json")


def checkpoint(tmp_path, temperature):
    run = tmp_path / "ckpt"; run.mkdir()
    write_meta(run, Meta(base="b", temperature=temperature))
    return run


def test_a_refit_is_written_only_when_the_rule_and_its_confirmation_pass(tmp_path, capsys):
    run = checkpoint(tmp_path, 0.6)
    fit, judge = write(tmp_path / "pool", generated(2.0, seed=1)), write(tmp_path / "served", generated(2.0, n=400, seed=2))   # 400: the Brier interval clears 0
    skill = write(tmp_path / "skill", generated(0.5, seed=3, source="a") + generated(2.0, seed=4, source="b"))
    breadth = write(tmp_path / "breadth", generated(2.0, seed=6))
    gate = ["--run", str(run), "--rows", fit, "--judge", judge, "--allow-in-distribution"]
    # two selections of one file are two guards: the failing one keeps 0.6 though the passing one is given after it; the
    # confirmation rows are then never scored
    assert calibrate.main(gate + ["--guard", f"{skill}:a", "--guard", f"{skill}:b", "--confirm", breadth, "--locked", breadth]) == 0.6
    out = capsys.readouterr().out
    assert f"guard {skill}:a: ECE" in out and "judge panels" not in out and "temperature_fit" not in read_meta(run).extra
    assert "confirmation rows not scored: the rule failed" in out and breadth not in out
    # the rule passes; rows calibrated at the shipped 0.6 then fail both confirmation stages, and nothing is written
    shipped_right = write(tmp_path / "test", generated(0.6, seed=5))
    assert calibrate.main(gate + ["--confirm", shipped_right, "--locked", shipped_right]) == 0.6
    out = capsys.readouterr().out
    assert f"confirm {shipped_right}: ECE" in out and f"locked {shipped_right}: Brier" in out and read_meta(run).temperature == 0.6
    adopted = calibrate.main(gate + ["--guard", f"{skill}:b", "--confirm", breadth, "--locked", breadth, "--seed", "7"])
    fitted = read_meta(run).extra["temperature_fit"]
    assert 1.6 < adopted == read_meta(run).temperature < 2.5 and fitted["rule"]["adopt"] and fitted["rule"]["confirmation"]["passed"]
    assert fitted["interval"]["lower"] <= adopted <= fitted["interval"]["upper"]
    # the interval and the rule's bootstrap are Kev's registered reads, seed 0, whatever --seed the cross-validation took:
    # [1.5511, 2.3511] is Kev's pooled_temperature_ci on the fit rows
    assert fitted["cross_validation"]["seed"] == 7 and fitted["interval"] == interval(calibrate.panel(fit, None))
    assert [fitted["interval"][k] for k in ("seed", "lower", "upper")] == pytest.approx([0, 1.5511, 2.3511], abs=1e-4)
    assert fitted["rule"]["judge"]["brier_ci95"] == pytest.approx(kevs_paired_read(read_json(judge), adopted, 0.6), abs=1e-12)


def test_a_rows_file_is_refused_in_a_second_role_before_anything_is_fitted(tmp_path):
    run = checkpoint(tmp_path, 1.0)
    fit, other, third = (write(tmp_path / name, generated(1.5, n=50, seed=seed)) for name, seed in (("a", 4), ("b", 5), ("c", 10)))
    copy = tmp_path / "copy"; copy.mkdir(); shutil.copy(fit, copy / "rows.json")
    part = write(tmp_path / "part", generated(1.5, n=50, seed=4)[:10] + generated(1.5, n=20, seed=9, source="other"))
    report = {"suite_sha256": "f" * 64, "split": "development"}
    pool, reread = write(tmp_path / "pool", generated(1.5, n=50, seed=6), report), write(tmp_path / "reread", generated(1.5, n=50, seed=7), report)
    transfer = generated(1.5, n=50, seed=11, source="transfer")
    pooled, guard = write(tmp_path / "pooled", generated(1.5, n=50, seed=12) + transfer[:5]), write(tmp_path / "transfer", transfer)
    for args, why in [
        (["--rows", fit, "--judge", fit], "fitted on"),
        (["--rows", pooled, "--judge", other, "--guard", guard], "fitted on"),   # the pooled file holds five of the guard's rows
        (["--rows", fit, "--judge", other, "--guard", str(copy / "rows.json")], "fitted on"),   # a copy elsewhere is the same rows
        (["--rows", fit, "--judge", part], "fitted on"),   # and so is a file holding some of them
        (["--rows", f"{part}:custom", "--judge", f"{part}:other"], "fitted on"),   # the fit's own file counts whole in another role, whatever its selection
        (["--rows", pool, "--judge", reread], "fitted on"),   # another read of the same partition (report.json): the same questions
        (["--rows", fit, "--judge", other, "--confirm", fit], "fitted on"),
        (["--rows", fit, "--judge", other, "--locked", fit], "fitted on"),
        (["--rows", fit, "--judge", other, "--confirm", other], "rule already reads"),
        (["--rows", fit, "--judge", other, "--locked", other], "rule already reads"),
        (["--rows", fit, "--judge", other, "--guard", third, "--confirm", third], "rule already reads"),
        (["--rows", fit, "--judge", other, "--guard", third, "--locked", third], "rule already reads"),
        (["--rows", fit, "--judge", other, "--judge", f"{other}:custom"], "given twice"),
        (["--rows", fit, "--guard", other], "need --judge"),
        (["--rows", fit, "--confirm", other], "need --judge"),
        (["--rows", fit, "--locked", other], "need --judge"),
    ]:
        with pytest.raises(SystemExit, match=why):
            calibrate.main(["--run", str(run), "--allow-in-distribution", *args])
    # Kev's round 28 pools "minus the transfer-v4 development records" and guards them: a row the fit does not read (dropped by
    # --exclude_rows, or outside the --rows sources) may be judged, so these pass the roles check and stop at the next one (rows
    # of no known suite are not shown to be held out)
    for fitting in (["--rows", pooled, "--exclude_rows", guard], ["--rows", f"{pooled}:custom"]):
        with pytest.raises(SystemExit, match="not held out"):
            calibrate.main(["--run", str(run), *fitting, "--judge", other, "--guard", guard])
    assert read_meta(run).temperature == 1.0 and "temperature_fit" not in read_meta(run).extra
    # another partition of the same suite is another file
    assert not calibrate.identity(pool) & calibrate.identity(write(tmp_path / "test", generated(1.5, n=50, seed=8), {**report, "split": "test"}))


def test_a_selection_is_refused_in_any_role_when_it_names_absent_sources_or_holds_no_scored_rows(tmp_path):
    run = checkpoint(tmp_path, 1.0)
    fit, judge = write(tmp_path / "a", generated(1.5, n=50, seed=4)), write(tmp_path / "b", generated(1.5, n=50, seed=5))
    blind = write(tmp_path / "blind", generated(1.5, n=20, seed=13) + generated(1.5, n=20, seed=14, source="unknowable"))
    noise, empty = write(tmp_path / "noise", [{**r, "variant": "noise"} for r in generated(1.5, n=20, seed=15)]), write(tmp_path / "empty", [])
    # a hard-v1 read without a hard_tradeoff row: the allowlist check passes, since hard-v1's manifest lists that source
    hard = write(tmp_path / "hard", generated(1.5, n=20, seed=16, source="hard_judge"),
                 {"suite_sha256": digest(calibrate.ROOT / "evals/hard-v1/manifest.json"), "split": "development"})
    for role in ("--rows", "--judge", "--guard", "--confirm", "--locked"):   # --rows too: an empty fit selection silently shrinks the fit
        judged = [] if role == "--judge" else ["--judge", judge]
        for selection, why in [(f"{blind}:nosuch", "names sources its rows do not contain"), (f"{blind}:unknowable", "no scored rows"),
                               (noise, "no scored rows"), (empty, "no scored rows"), (f"{hard}:hard_tradeoff", "no scored rows")]:
            with pytest.raises(SystemExit, match=why):
                calibrate.main(["--run", str(run), "--rows", fit, "--allow-in-distribution", *judged, role, selection])
    assert read_meta(run).temperature == 1.0 and "temperature_fit" not in read_meta(run).extra


def test_a_manual_temperature_refuses_the_rules_options_instead_of_ignoring_them(tmp_path, capsys):
    run = checkpoint(tmp_path, 1.0)
    judge = write(tmp_path / "b", generated(1.5, n=50, seed=5))
    for role in ("--judge", "--guard", "--confirm", "--locked"):
        with pytest.raises(SystemExit):
            calibrate.main(["--run", str(run), "--temperature", "3.5", "--reason", "copied", role, judge])
        assert "would be ignored" in capsys.readouterr().err
    assert read_meta(run).temperature == 1.0 and "temperature_fit" not in read_meta(run).extra


def test_pooled_partitions_keep_their_clusters(tmp_path):
    # two D1A partitions share every "custom/<line>" id: their records must stay separate resampling and fold units
    run = checkpoint(tmp_path, 1.0)
    first, second = write(tmp_path / "a", generated(1.5, n=120, seed=4)), write(tmp_path / "b", generated(1.5, n=80, seed=5))
    calibrate.main(["--run", str(run), "--rows", first, "--rows", second, "--allow-in-distribution"])
    assert read_meta(run).extra["temperature_fit"]["cross_validation"]["groups"] == 200
