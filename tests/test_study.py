"""d1a.training.study (#62): the steps run in order, a stopped study continues without redoing what it finished, and
publishing refuses a checkpoint that was never calibrated. The steps' processes are stood in for; no model is loaded."""
import json
from pathlib import Path

import pytest

from d1a.backends import checkpoint
from d1a.backends.checkpoint import Meta
from d1a.training import study

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/d1a-e2b.yaml"


class Machine:
    """Stands in for the step processes: records each call and writes what the real one would."""
    def __init__(self, out, temperature=0.9, fit=None):
        self.out, self.calls, self.meta = Path(out), [], Meta(base="b", temperature=1.0)
        self.fitted = Meta(base="b", temperature=temperature, extra={"temperature_fit": fit or {"n": 1200, "method": "grid"}})

    def __call__(self, name, argv):
        argv = [str(x) for x in argv]
        self.calls.append((name, argv))
        if name == "d1a.training.recipe":
            (self.out / "base").mkdir(parents=True)
            (self.out / "base" / "recipe.json").write_text(json.dumps({"recipe": {"name": "d1a-e2b"}, "recipe_sha256": "ab" * 32, "d1a_commit": "c0ffee00"}), encoding="utf-8")
        elif name == "d1a.eval.benchmark":
            out = Path(argv[argv.index("--out") + 1]); out.mkdir(parents=True)
            # as d1a.eval.benchmark writes it: "temperature" is the extra metrics temperature (1.0 from the CLI); the one served is under calibration
            report = {"temperature": 1.0, "calibration": {"inference_temperature": self.meta.temperature, "additional_temperature": 1.0},
                      "split": "development", "clean": {"n": 10, "acc": 0.8, "ece": 0.03, "nll": 0.5}}
            (out / "report.json").write_text(json.dumps(report), encoding="utf-8"); (out / "rows.json").write_text("[]", encoding="utf-8")
        elif name == "d1a.training.calibrate":
            self.meta = self.fitted
        return 0

    def names(self):
        return [(n, a[a.index("--out") + 1].split("/study/")[-1] if "--out" in a and n != "d1a.training.recipe" else None) for n, a in self.calls]


@pytest.fixture
def machine(tmp_path, monkeypatch):
    m = Machine(tmp_path / "run")
    monkeypatch.setattr(checkpoint, "read_meta", lambda run: m.meta)
    return m


def run(machine, **kw):
    return study.Study(RECIPE, machine.out, "cpu", ["evals/v7/decision-v7", "evals/d1a/pr-labels:development"],
                       ["evals/v7/decision-v7", "evals/hard-v1"], runner=machine, **kw).run()


def test_train_calibrate_evaluate_report(machine):
    s = run(machine)
    assert machine.names() == [("d1a.training.recipe", None),
                               ("d1a.eval.benchmark", "calibrate/v7_decision-v7"), ("d1a.eval.benchmark", "calibrate/d1a_pr-labels_development"),
                               ("d1a.training.calibrate", None),
                               ("d1a.eval.benchmark", "evaluate/v7_decision-v7"), ("d1a.eval.benchmark", "evaluate/hard-v1")]
    bench = [a for n, a in machine.calls if n == "d1a.eval.benchmark"]
    assert bench[0][bench[0].index("--split") + 1] == "calibration" and "--data" in bench[1] and bench[2][bench[2].index("--split") + 1] == "development"
    cal = machine.calls[3][1]
    assert cal[cal.index("--run") + 1].endswith("/run/base") and cal.count("--rows") == 2 and "--allow-in-distribution" not in cal
    assert s["temperature"] == 0.9 and [r["evaluation"] for r in s["evaluations"]] == ["evals/v7/decision-v7", "evals/hard-v1"]
    md = (machine.out / "study/study.md").read_text(encoding="utf-8")
    assert "Temperature 0.9000, fitted on 1200 held-out rows" in md and "| evals/hard-v1 | development | 10 | 0.8000 | 0.0300 | 0.5000 |" in md


def test_a_stopped_study_continues(machine):
    run(machine)
    machine.calls.clear()
    run(machine)
    assert machine.names() == [("d1a.training.calibrate", None)]                 # trained and scored: only the cheap fit again
    machine.calls.clear(); machine.fitted = Meta(base="b", temperature=1.1, extra={"temperature_fit": {"n": 5}})
    run(machine)
    assert machine.names() == [("d1a.training.calibrate", None), ("d1a.eval.benchmark", "evaluate/v7_decision-v7"), ("d1a.eval.benchmark", "evaluate/hard-v1")]


def test_no_calibration_no_study(machine):
    with pytest.raises(SystemExit, match="calibrated on held-out rows"):
        study.Study(RECIPE, machine.out, "cpu", [], ["evals/hard-v1"], runner=machine).run()
    assert [n for n, _ in machine.calls] == ["d1a.training.recipe"]


def test_dry_run_runs_nothing(machine, capsys):
    assert run(machine, dry_run=True) is None and machine.calls == []
    printed = capsys.readouterr().out
    assert "python -m d1a.training.recipe run" in printed and "--dry-run" in printed and "python -m d1a.training.calibrate --run" in printed


class Hub:
    def __init__(self): self.calls = []
    def create_repo(self, repo, **kw): self.calls.append(("create", repo, kw))
    def upload_folder(self, **kw): self.calls.append(("upload", kw["path_in_repo"], kw["folder_path"]))


def test_publishing_refuses_an_uncalibrated_checkpoint(machine):
    run(machine)
    machine.meta = Meta(base="b", temperature=1.0)                                 # e.g. a recipe rerun after the fit
    with pytest.raises(SystemExit, match="never fitted"):
        study.publish(machine.out, "o/r:p", api=Hub())
    machine.meta = Meta(base="b", temperature=0.95, extra={"temperature_fit": {"n": 9, "in_distribution": ["trained on decision-v7"]}})
    with pytest.raises(SystemExit, match="in distribution"):
        study.publish(machine.out, "o/r:p", api=Hub())
    hub = Hub()
    study.publish(machine.out, "o/r:p", allow_in_distribution=True, api=hub)
    assert [c[:2] for c in hub.calls] == [("create", "o/r"), ("upload", "p/checkpoint"), ("upload", "p/study")]
    with pytest.raises(SystemExit, match="owner/repo:prefix"):
        study.publish(machine.out, "o/r", allow_in_distribution=True, api=Hub())


def test_in_distribution_is_passed_to_the_fit(machine):
    run(machine, allow_in_distribution=True)
    assert "--allow-in-distribution" in machine.calls[3][1]


def test_references():
    assert study.slug("evals/v7/decision-v7") == "v7_decision-v7" and study.slug("evals/d1a/pr-labels:development") == "d1a_pr-labels_development"
    assert study.benchmark_argv("ck", "evals/hard-v1", "o", None, "calibration") == ["--run", "ck", "--suite", "evals/hard-v1", "--split", "calibration", "--out", "o"]
    assert study.benchmark_argv("ck", "evals/d1a/x:dev", "o", "mps", "calibration") == ["--run", "ck", "--data", "evals/d1a/x:dev", "--out", "o", "--device", "mps"]
