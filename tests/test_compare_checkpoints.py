"""scripts/compare_checkpoints.py: a training run's verdict table (#175). Differences are paired over the questions both
runs answered and resampled by record; ECE uses a fitted temperature when calibration rows are given; the bar fails a run
on its own suites and on the floor. Synthetic benchmark rows stand in for real scores."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_checkpoints", ROOT / "scripts/compare_checkpoints.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)


def row(rid, q, right, conf=0.8, variant="clean", source="s", temperature=1.0):
    """A two-option choice row answered with `conf` on the label (right) or on the other option (wrong)."""
    p = [conf, 1 - conf] if right else [1 - conf, conf]
    return {"id": rid, "question": q, "label": 0, "p": p, "logits": [math.log(x) * temperature for x in p], "type": "choice",
            "variant": variant, "source": source, "inference_temperature": temperature, "task": "t"}


def write(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rows.json").write_text(json.dumps(rows), encoding="utf-8")


def test_paired_over_common_questions_resampled_by_record():
    ref = [row(f"r{i}", q, right=(i % 2 == 0)) for i in range(40) for q in ("a", "b")]
    run = [row(f"r{i}", q, right=True) for i in range(40) for q in ("a", "b")] + [row("extra", "a", True)]   # not in ref: ignored
    r = cmp.paired(ref, run)
    assert (r["n"], r["ref"], r["run"]) == (80, 0.5, 1.0) and r["delta"] == 0.5
    assert 0.3 < r["lo"] < 0.5 < r["hi"] < 0.7
    assert r == cmp.paired(ref, run)                                              # the same seed, the same interval
    same = cmp.paired(ref, ref)
    assert (same["delta"], same["lo"], same["hi"]) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="no question"):
        cmp.paired(ref, [row("other", "a", True)])


def test_records_move_together():
    """Two questions of one record are one unit: an interval over 20 records of 2 questions is wider than over 40 of 1."""
    flips = lambda n, k: [row(f"r{i}", q, right=(i % 2 == 0)) for i in range(n) for q in k]
    two = cmp.paired(flips(20, ("a", "b")), [row(f"r{i}", q, True) for i in range(20) for q in ("a", "b")])
    one = cmp.paired(flips(40, ("a",)), [row(f"r{i}", "a", True) for i in range(40)])
    assert two["delta"] == one["delta"] and (two["hi"] - two["lo"]) > (one["hi"] - one["lo"])


def test_only_scored_rows_count(tmp_path):
    write(tmp_path / "s", [row("r1", "a", True), row("r2", "a", False, variant="shuffled"), row("r3", "a", False, source="unknowable")])
    assert [r["id"] for r in cmp.load(tmp_path / "s")] == ["r1"]


def test_suites_are_folders_with_rows_and_not_calibration(tmp_path):
    for name in ("hard-v1", "cal-dv7", "empty"):
        (tmp_path / name).mkdir()
    write(tmp_path / "hard-v1", []); write(tmp_path / "cal-dv7", [])
    assert cmp.suites(tmp_path) == {"hard-v1"}


def test_bar_and_floor():
    res = {"B": {"hard-v1": {"delta": 0.031}, "devtools-v1": {"delta": 0.02}, "transfer-v4": {"delta": -0.011}}}
    fails = cmp.bar_failures(res, {"hard-v1": 3, "devtools-v1": 3, "documents-v1": 3}, -1)["B"]
    assert fails == ["devtools-v1: +2.00 < +3", "transfer-v4: -1.10 < -1", "documents-v1: not scored"]
    assert cmp.bar_failures({"B": {"hard-v1": {"delta": 0.05}}}, {"hard-v1": 3}, None) == {"B": []}


def test_ece_at_the_fitted_temperature(tmp_path):
    """Overconfident rows (0.95 on a 50% coin) fit a temperature above 1, and their ECE there is lower than as served."""
    rows = [row(f"r{i}", "a", right=(i % 2 == 0), conf=0.95) for i in range(200)]
    write(tmp_path / "cal", rows)
    t = cmp.fitted_temperature([tmp_path / "cal"])
    assert t > 2 and cmp.ece(rows, t) < cmp.ece(rows, 1.0)


def test_main_writes_the_table_and_fails_a_run_below_the_bar(tmp_path, capsys):
    for name, right in (("ref", lambda i: i % 2 == 0), ("good", lambda i: True), ("bad", lambda i: i % 3 == 0)):
        write(tmp_path / name / "hard-v1", [row(f"r{i}", "a", right(i)) for i in range(60)])
    args = ["--ref", f"v0.4={tmp_path / 'ref'}", "--run", f"good={tmp_path / 'good'}", "--bar", "hard-v1=3", "--json", str(tmp_path / "out.json")]
    assert cmp.main(args) == 0
    out = capsys.readouterr().out
    assert "| hard-v1 | 60 | 50.00 | 100.00 (+50.00" in out and "bar good: PASS" in out
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["bar"] == {"good": []}
    write(tmp_path / "good" / "cal-x", [row(f"c{i}", "a", right=(i % 2 == 0), conf=0.95) for i in range(200)])
    assert cmp.main(args + ["--fit", f"good={tmp_path / 'good' / 'cal-x'}"]) == 0
    assert "good fitted" in capsys.readouterr().out                               # --fit replaces the served temperature
    assert cmp.main(args[:2] + ["--run", f"bad={tmp_path / 'bad'}", "--bar", "hard-v1=3"]) == 1
    assert "bar bad: FAIL (hard-v1: -16.67 < +3)" in capsys.readouterr().out
