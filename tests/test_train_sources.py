"""#211: d1a.training.train refuses a fine-tune (--init_from) whose data leaves out a training source, before any weights
load. Coverage comes from the --data mix's sidecar <data>.json, in the shape of each mix writer (recipes/skills/mix.py,
recipes/pr-labeler/mix.py, the D2 builder), and from --suite with --replay. A source left out on purpose is named with a
reason, which training_config.json records. No model loads here: load_tokenizer, the first thing that touches weights,
raises."""
import json
import sys

import pytest

from d1a.eval.suites import SKILLS, train_sources
from d1a.training import train

DV7 = ["--suite", "evals/v7/decision-v7", "--replay", "5"]   # decision-v7 replayed by the trainer
ROUTING = ["evals/d1a/routing:factory-train", "evals/d1a/routing:generic-train"]


class Loading(Exception):
    """The run passed the source check and reached the first step that loads weights."""


def skills_mix(drop=()):
    """recipes/skills/mix.py's sidecar: plan and counts by source."""
    plan = [[s, 10] for s in train_sources() if s not in drop]
    return {"plan": plan, "counts": dict(plan), "total": 10 * len(plan)}, DV7


def pr_labeler_mix(drop=()):
    """recipes/pr-labeler/mix.py's sidecar: every input's sha256 and counts by group (the routing partitions are one pool)."""
    refs = [s for s in train_sources() if ":" in s]
    inputs = {ref: "0" * 64 for ref in refs} | {f"{s}:train (manifest sha256)": "0" * 64 for s in SKILLS}
    counts = {"english": 50, "extra": 20, "ja": 10, "routing": 0 if set(ROUTING) & set(drop) else 10, "skills": {s: 10 for s in SKILLS}}
    return {"inputs": inputs, "params": {}, "counts": counts, "records": 120}, DV7


def d2_builder(drop=()):
    """The D2 builder's sidecar: records by short source name, decision-v7 inside the mix (--replay 0)."""
    short = {s: s.removeprefix("evals/d1a/") if ":" in s else f"{s.rpartition('/')[2]}:train" for s in [*train_sources(), train.DECISION_V7]}
    return {"sources": {name: {"records": 10} for s, name in short.items() if s not in drop}, "dv7_replay_by_trainer": 0}, []


SHAPES = {"skills mix": skills_mix, "pr-labeler mix": pr_labeler_mix, "d2 builder": d2_builder}


def run(tmp_path, monkeypatch, *argv, sidecar=None, out="run", loads=False):
    """train.main() on a one-record --data file (with `sidecar` beside it), stopped where weights would load unless `loads`."""
    data = tmp_path / "mix.jsonl"
    data.write_text('{"state": "s", "questions": {}}\n', encoding="utf-8")
    if sidecar is not None:
        (tmp_path / "mix.jsonl.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["d1a.training.train", "--data", str(data), *argv, "--out", str(tmp_path / out)])
    monkeypatch.setattr(train, "load_tokenizer", (lambda *a, **k: None) if loads else lambda *a, **k: (_ for _ in ()).throw(Loading()))
    train.main()


@pytest.mark.parametrize("shape", SHAPES)
def test_a_mix_covering_every_source_trains(shape, tmp_path, monkeypatch):
    sidecar, extra = SHAPES[shape]()
    with pytest.raises(Loading):
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *extra, sidecar=sidecar)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_mix_without_routing_is_refused_and_named(shape, tmp_path, monkeypatch):
    sidecar, extra = SHAPES[shape](drop=ROUTING)
    with pytest.raises(SystemExit) as e:
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *extra, sidecar=sidecar)
    message = str(e.value)
    assert "leaves out 2 training source(s): " + ", ".join(ROUTING) + "." in message
    assert "recipes/skills/mix.py" in message and "--allow_missing_sources" in message
    assert not (tmp_path / "run").exists()   # refused before the run directory, let alone the weights


def test_data_without_a_sidecar_is_refused_for_a_fine_tune(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as e:
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *DV7)
    assert f"leaves out {len(train_sources())} training source(s): {', '.join(train_sources())}. " in str(e.value)   # decision-v7 is replayed
    assert "mix.jsonl has no sidecar mix.jsonl.json naming its sources" in str(e.value)


def test_a_source_left_out_with_a_reason_trains_and_is_recorded(tmp_path, monkeypatch):
    sidecar, extra = skills_mix(drop=ROUTING)
    allow = ["--allow_missing_sources", "evals/d1a/routing:factory-train,routing:generic-train", "--reason", "routing is retrained next"]
    monkeypatch.setattr(train, "build_model", lambda *a: (None, None, None))
    monkeypatch.setattr(train, "training_requests", lambda *a: [])

    class Configured(Exception):
        pass

    monkeypatch.setattr(train, "optimizer_and_schedule", lambda *a: (_ for _ in ()).throw(Configured()))   # training_config.json is written by now
    with pytest.raises(Configured):
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *extra, *allow, sidecar=sidecar, loads=True)
    config = json.loads((tmp_path / "run/training_config.json").read_text(encoding="utf-8"))
    assert config["sources"] == {"covered": [s for s in train.required_sources() if s not in ROUTING], "allowed_missing": ROUTING,
                                 "reason": "routing is retrained next"}
    assert config["args"]["allow_missing_sources"] == allow[1]
    with pytest.raises(SystemExit, match="names no training source: routing:typo"):   # a misspelt source is no exception
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *extra, "--allow_missing_sources", "routing:typo", "--reason", "r", sidecar=sidecar, out="typo")


@pytest.mark.parametrize("argv", [["--allow_missing_sources", "all"], ["--allow_missing_sources", "all", "--reason", " "], ["--reason", "why"]])
def test_an_exception_without_its_reason_is_refused(argv, tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *argv)
    assert e.value.code == 2 and "--allow_missing_sources and --reason go together" in capsys.readouterr().err


def test_a_run_from_the_base_model_is_not_checked(tmp_path, monkeypatch):
    with pytest.raises(Loading):
        run(tmp_path, monkeypatch)   # --data with no sidecar and no other source: fine without --init_from


@pytest.mark.parametrize("records", ["lots", 10.0, True, -1])
def test_a_record_count_that_is_not_a_whole_number_is_refused(records, tmp_path, monkeypatch):
    sidecar, extra = skills_mix()
    sidecar["counts"]["evals/hard-v1"] = records
    with pytest.raises(SystemExit, match="gives record counts that are not whole numbers: {'evals/hard-v1': "):
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *extra, sidecar=sidecar)


def test_the_log_names_what_is_left_out_on_purpose(tmp_path, monkeypatch, capsys):
    sidecar, extra = skills_mix(drop=ROUTING)
    with pytest.raises(Loading):
        run(tmp_path, monkeypatch, "--init_from", "JohnP1/d1a-e4b", *extra, "--allow_missing_sources", "all", "--reason", "routing is retrained next",
            sidecar=sidecar)
    assert capsys.readouterr().out.splitlines()[0] == (
        "sources left out on purpose (--allow_missing_sources all): " + ", ".join(ROUTING) + "; reason: routing is retrained next")
