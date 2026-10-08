"""d1a.training.calibrate's held-out guard on D1A suite partitions and on a run trained on a mix (#196), and its grid-edge
warning. No model is loaded: the rows are written here, as d1a.eval.benchmark writes them."""
import json

import pytest

from d1a.backends.checkpoint import Meta, read_meta, write_meta
from d1a.eval.suite import digest
from d1a.training import calibrate

PR_LABELS = calibrate.ROOT / "evals/d1a/pr-labels/manifest.json"


def scored(directory, report, n=40, correct=0.73):
    """rows.json (p 0.73 on the first option; the first `correct` share labelled 0) and the report.json beside it."""
    directory.mkdir(parents=True)
    rows = [{"id": f"r{i}", "source": "pr", "task": "pr", "type": "choice", "keys": ["a", "b"], "variant": "clean", "group": i,
             "question": "q", "label": int(i >= correct * n), "p": [0.73, 0.27], "logits": [1.0, 0.0], "inference_temperature": 1.0}
            for i in range(n)]
    (directory / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return str(directory / "rows.json")


def partition(name):
    """The report.json of rows benchmark scored with --data evals/d1a/pr-labels:<name> (the partition file's sha256, split custom)."""
    return {"suite_sha256": json.loads(PR_LABELS.read_text(encoding="utf-8"))["partitions"][name]["sha256"], "split": "custom"}


# how a fine-tune's training is recorded: a skills mix sidecar (counts), a D2 builder sidecar (sources), --data a D1A partition
SIDECARS = {"counts": {"counts": {"evals/d1a/pr-labels:train": 120, "evals/d1a/pr-labels:train-blast": 0, "evals/hard-v1": 50}},
            "sources": {"sources": {"pr-labels:train": {"records": 120}, "hard-v1:train": {"records": 50}}}}


def trained(tmp_path, how):
    run = tmp_path / "run"; run.mkdir()
    write_meta(run, Meta(base="b"))
    data = "evals/d1a/pr-labels:train"
    if how in SIDECARS:
        data = str(tmp_path / "mix.jsonl")
        (tmp_path / "mix.jsonl.json").write_text(json.dumps(SIDECARS[how]), encoding="utf-8")
    (run / "training_config.json").write_text(json.dumps({"args": {"data": data, "suite": None, "extra_suites": ""}, "suite_sha256": None}), encoding="utf-8")
    return run


@pytest.fixture
def quick(monkeypatch):   # the guard is under test, not the fit
    monkeypatch.setattr(calibrate, "fit_temperature", lambda rows, **kw: 1.5)


@pytest.mark.parametrize("how", ["counts", "sources", "data"])
def test_a_d1a_eval_partition_of_a_trained_suite_fits_with_a_note(tmp_path, capsys, quick, how):
    run = trained(tmp_path, how)
    assert calibrate.main(["--run", str(run), "--rows", scored(tmp_path / "dev", partition("development"))]) == 1.5   # no --allow-in-distribution
    fit = read_meta(run).extra["temperature_fit"]
    assert "in_distribution" not in fit and "evals/d1a/pr-labels:train" in fit["training_partitions"]
    assert fit["fit_rows"][0]["suite"] == "evals/d1a/pr-labels" and fit["fit_rows"][0]["split"] == "development"
    assert "SAME CORPUS" in capsys.readouterr().out and "eval partition of evals/d1a/pr-labels" in fit["same_corpus"][0]


@pytest.mark.parametrize("how", ["counts", "sources", "data"])
def test_a_d1a_partition_the_run_trained_on_is_refused(tmp_path, quick, how):
    run = trained(tmp_path, how)
    with pytest.raises(SystemExit, match="evals/d1a/pr-labels:train is training data of the checkpoint"):
        calibrate.main(["--run", str(run), "--rows", scored(tmp_path / "train", partition("train"))])
    assert read_meta(run).temperature == 1.0 and "temperature_fit" not in read_meta(run).extra   # nothing written


def test_a_mixs_frozen_suites_keep_the_frozen_rule(tmp_path, quick):
    # the mix's sidecar names evals/hard-v1: its development rows are refused on the frozen rule's own grounds
    run = trained(tmp_path, "counts")
    rows = scored(tmp_path / "hard", {"suite_sha256": digest(calibrate.ROOT / "evals/hard-v1/manifest.json"), "split": "development"})
    with pytest.raises(SystemExit) as refused:
        calibrate.main(["--run", str(run), "--rows", rows])
    why = str(refused.value)
    assert "evals/hard-v1 is training data of the checkpoint" in why and "development partition of evals/hard-v1, a training corpus" in why
    assert "cannot tell what this checkpoint was trained on" not in why


def test_a_fit_on_the_grid_edge_is_warned_and_recorded(tmp_path, capsys):
    run = tmp_path / "run"; run.mkdir()
    write_meta(run, Meta(base="b"))
    always_right = scored(tmp_path / "edge", {}, correct=1.0)                  # every row right: the least NLL is below 0.25
    assert calibrate.main(["--run", str(run), "--rows", always_right, "--allow-in-distribution"]) == pytest.approx(0.25)
    assert "GRID EDGE" in capsys.readouterr().out
    assert read_meta(run).extra["temperature_fit"]["grid_edge"]["temperature"] == pytest.approx(0.25)
    calibrate.main(["--run", str(run), "--rows", scored(tmp_path / "inside", {}), "--allow-in-distribution"])   # p 0.73, 73% right: T near 1
    assert "GRID EDGE" not in capsys.readouterr().out and "grid_edge" not in read_meta(run).extra["temperature_fit"]
