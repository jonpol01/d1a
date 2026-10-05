"""d1a.training.recipe (#63): recipe files checked against d1a.training.train's own options, stages chained, and the recipe recorded beside
every checkpoint. Nothing trains here: the runner is replaced by a stand-in."""
import json
from pathlib import Path

import pytest

from d1a.training import recipe as R

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path, text, name="r.yaml"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path / name


TWO_STAGES = """format: d1a-recipe
version: 1
name: two
stages:
  - name: base
    train: {suite: evals/v7/decision-v7, epochs: 2, lr: 1.0e-4, extra_suites: [evals/hard-v1, evals/devtools-v1], checkpointing: true}
  - name: skills
    train: {suite: evals/v7/decision-v7, epochs: 1, lr: 2.0e-5}
"""


def test_the_shipped_recipe_is_the_readme_command():
    """recipes/d1a-e2b.yaml gives the command README's Training section documents (base pinned as the trainer's default)."""
    recipe, _ = R.load(ROOT / "recipes/d1a-e2b.yaml")
    [(name, argv)] = R.commands(recipe, "runs/d1a-e2b", {"device": "cuda"})
    assert name == "base" and argv == [
        "--base", "google/gemma-4-E2B", "--base_revision", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "--suite", "evals/v7/decision-v7", "--epochs", "2", "--lr", "0.0001", "--batch", "4", "--accum", "2", "--dtype", "bf16",
        "--weights_dtype", "bf16", "--checkpointing", "1", "--p_none_pair", "0.25", "--device", "cuda", "--out", "runs/d1a-e2b/base"]


def test_stages_chain_and_are_recorded(tmp_path):
    seen = []
    def runner(argv):
        out = Path(argv[argv.index("--out") + 1]); out.mkdir(parents=True); seen.append(argv)
        return 0
    steps = R.run(write(tmp_path, TWO_STAGES), tmp_path / "run", {"device": "cpu", "resume": 1}, runner=runner)
    assert [n for n, _ in steps] == ["base", "skills"] and seen == [a for _, a in steps]
    base, skills = (dict(zip(a[::2], a[1::2])) for _, a in steps)
    assert "--init_from" not in base and skills["--init_from"] == str(tmp_path / "run" / "base")
    assert base["--extra_suites"] == "evals/hard-v1,evals/devtools-v1" and base["--checkpointing"] == "1" and base["--resume"] == "1"
    rec = json.loads((tmp_path / "run" / "skills" / "recipe.json").read_text(encoding="utf-8"))
    assert rec["stage"] == "skills" and rec["recipe"]["name"] == "two" and rec["command"][3:] == steps[1][1] and len(rec["recipe_sha256"]) == 64
    with pytest.raises(SystemExit, match="stage base failed"):                       # a failed stage stops the recipe
        R.run(write(tmp_path, TWO_STAGES), tmp_path / "again", {"device": "cpu"}, runner=lambda argv: 3)
    assert not (tmp_path / "again" / "skills").exists()
    assert R.run(write(tmp_path, TWO_STAGES), tmp_path / "dry", {"device": "cpu"}, dry_run=True, runner=lambda a: 1/0) and not (tmp_path / "dry").exists()


@pytest.mark.parametrize("change, message", [
    (("format: d1a-recipe", "format: kev-recipe"), "not a d1a-recipe file"),
    (("version: 1", "version: 2"), "version 1-1"),
    (("epochs: 1,", "epochs: one,"), "stage skills: d1a.training.train refuses"),                # d1a.training.train's own type check
    (("lr: 2.0e-5}", "lr: 2.0e-5, warmup: 3}"), "stage skills: d1a.training.train refuses"),     # not one of its options
    (("lr: 2.0e-5}", "lr: 2.0e-5, device: cuda}"), r"\['device'\] cannot be set in train"),
    (("name: skills", "name: base"), "unique name"),
    (("  - name: base\n", "  - name: base\n    init_from: previous\n"), "it is the first stage"),
    (("name: two\n", "name: two\nowner: me\n"), "unknown top-level keys"),
])
def test_bad_recipes_are_refused_before_anything_runs(tmp_path, change, message):
    path = write(tmp_path, TWO_STAGES.replace(*change, 1))
    with pytest.raises(ValueError, match=message):
        R.run(path, tmp_path / "run", {"device": "cpu"}, runner=lambda argv: pytest.fail("ran"))
