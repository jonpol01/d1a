"""Training recipes: a run's stages in one versioned YAML file instead of a long command line, checked against d1a.training.train's
own options (and a fine-tune stage against its source check) before anything runs, and recorded next to every checkpoint it produces (#63).

    python -m d1a.training.recipe run recipes/d1a-e2b.yaml --out runs/d1a-e2b --device cuda
    python -m d1a.training.recipe run recipes/d1a-e2b.yaml --out runs/d1a-e2b --device cuda --dry-run   # the commands, nothing run

A recipe:

    format: d1a-recipe
    version: 1
    name: d1a-e2b
    base: google/gemma-4-E2B              # optional, with base_revision: d1a.training.train's defaults otherwise
    stages:                               # run in order, each one d1a.training.train process writing <out>/<stage name>
      - name: base
        train: {suite: evals/v7/decision-v7, epochs: 2, lr: 1.0e-4, batch: 4, accum: 2}
      - name: skills
        init_from: previous               # the previous stage's checkpoint (the default after the first stage), or a run
        train: {data: runs/mix/train.jsonl, suite: evals/v7/decision-v7, replay: 160, epochs: 1, lr: 2.0e-5}   # a recipes/skills/mix.py mix + decision-v7: every source (#211)

A stage's `train` keys are d1a.training.train's options without the dashes; the machine settings (--device, --out, --resume,
--save_every_minutes) come from this command, never from the recipe. Each stage's directory gets recipe.json: the recipe,
its sha256, the stage and the exact command.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

FORMAT, VERSION = "d1a-recipe", 1
MACHINE = ("device", "out", "resume", "save_every_minutes")   # how and where to run: given to the runner, never recorded in a recipe
STAGE_KEYS = {"name", "train", "init_from"}


def load(path):
    """A recipe file, checked: its format and version, every stage's keys, and every training option against d1a.training.train."""
    import yaml
    text = Path(path).read_text(encoding="utf-8")
    recipe = yaml.safe_load(text)
    if not isinstance(recipe, dict) or recipe.get("format") != FORMAT or recipe.get("version") not in range(1, VERSION + 1):
        raise ValueError(f"{path}: not a {FORMAT} file of version 1-{VERSION} (format {recipe.get('format') if isinstance(recipe, dict) else None!r})")
    unknown = set(recipe) - {"format", "version", "name", "description", "base", "base_revision", "stages"}
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    stages = recipe.get("stages")
    if not stages or not isinstance(stages, list):
        raise ValueError(f"{path}: no stages")
    names = [s.get("name") for s in stages]
    if any(not isinstance(n, str) or not n or "/" in n for n in names) or len(set(names)) != len(names):
        raise ValueError(f"{path}: every stage needs a unique name without '/' (got {names})")
    for s in stages:
        if set(s) - STAGE_KEYS:
            raise ValueError(f"{path}: stage {s['name']}: unknown keys {sorted(set(s) - STAGE_KEYS)}")
        bad = sorted(set(s.get("train") or {}) & set(MACHINE + ("init_from", "base", "base_revision")))
        if bad:
            raise ValueError(f"{path}: stage {s['name']}: {bad} cannot be set in train (machine settings come from the command; "
                             "base and init_from have their own keys)")
    return recipe, hashlib.sha256(text.encode("utf-8")).hexdigest()


def flag(key, value):
    """One option as d1a.training.train reads it: --key value (lists joined with commas, booleans as 1 and 0)."""
    if isinstance(value, bool):
        value = int(value)
    elif isinstance(value, (list, tuple)):
        value = ",".join(map(str, value))
    return [f"--{key}", str(value)]


def commands(recipe, out, machine):
    """[(stage name, d1a.training.train argv)] in order. Each argv is checked by d1a.training.train's own parser (types, choices, its rules)
    and, for a fine-tune stage, by its source check (#211)."""
    from d1a.training import train
    out, steps, previous = Path(out), [], None
    for stage in recipe["stages"]:
        argv = []
        for key in ("base", "base_revision"):
            if recipe.get(key):
                argv += flag(key, recipe[key])
        init = stage.get("init_from", "previous" if previous else None)
        if init == "previous":
            if previous is None:
                raise ValueError(f"stage {stage['name']}: init_from previous, but it is the first stage")
            init = str(out / previous)
        if init:
            argv += flag("init_from", init)
        for key, value in (stage.get("train") or {}).items():
            argv += flag(key, value)
        for key in MACHINE:
            if machine.get(key) not in (None, "", 0):
                argv += flag(key, machine[key])
        argv += flag("out", str(out / stage["name"]))
        try:
            a = train.parse_args(argv)
        except SystemExit as e:   # argparse reports the problem on stderr and exits
            raise ValueError(f"stage {stage['name']}: d1a.training.train refuses these options (see above): {' '.join(argv)}") from e
        try:
            train.source_coverage(a)   # a fine-tune stage that leaves out a training source fails here, not after the stages before it trained
        except SystemExit as e:
            raise ValueError(f"stage {stage['name']}: {e}") from e
        steps.append((stage["name"], argv))
        previous = stage["name"]
    return steps


def record(recipe, sha256, name, argv, directory):
    """recipe.json beside the stage's checkpoint: the recipe, its hash, the stage and the command that made it."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True).stdout.strip() or None
    from d1a.eval.suite import write_json
    write_json(Path(directory, "recipe.json"), {"recipe": recipe, "recipe_sha256": sha256, "stage": name,
                                                "command": ["python", "-m", "d1a.training.train", *argv], "d1a_commit": commit})


def run(path, out, machine, dry_run=False, runner=None):
    """Every stage in order; a stage that fails stops the recipe. runner(argv) -> exit code (default: a d1a.training.train process)."""
    recipe, sha256 = load(path)
    steps = commands(recipe, out, machine)
    for name, argv in steps:
        print("python -m d1a.training.train " + " ".join(argv), flush=True)
        if dry_run:
            continue
        code = (runner or (lambda a: subprocess.run([sys.executable, "-m", "d1a.training.train", *a]).returncode))(argv)
        if code:
            raise SystemExit(f"stage {name} failed (exit {code}); later stages not run")
        record(recipe, sha256, name, argv, Path(out) / name)
    return steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a recipe's stages in order")
    r.add_argument("recipe")
    r.add_argument("--out", required=True, help="the run's directory: each stage writes <out>/<stage name>")
    r.add_argument("--device", choices=["cpu", "mps", "cuda"])
    r.add_argument("--resume", type=int, choices=[0, 1], default=0)
    r.add_argument("--save_every_minutes", type=float, default=0)
    r.add_argument("--dry-run", action="store_true", help="print each stage's d1a.training.train command and stop")
    a = ap.parse_args(argv)
    try:
        run(a.recipe, a.out, {"device": a.device, "resume": a.resume, "save_every_minutes": a.save_every_minutes}, a.dry_run)
    except ValueError as e:
        raise SystemExit(f"d1a.training.recipe: {e}")


if __name__ == "__main__":
    main()
