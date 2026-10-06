"""A study, end to end: train a recipe, calibrate the checkpoint on held-out rows, evaluate it, write the report, and
publish it, refusing a checkpoint that was never calibrated (#62; D1A 0.2's E4B shipped at temperature 1.0).

    python -m d1a.training.study run recipes/d1a-e2b.yaml --out runs/d1a-e2b --device cuda \\
        --calibrate evals/v7/decision-v7 --evaluate evals/v7/decision-v7 --evaluate evals/hard-v1
    python -m d1a.training.study run ... --dry-run                  # every command, nothing run
    python -m d1a.training.study publish runs/d1a-e2b --to JohnP1/d1a-e2b-runs:r3

Steps, each a process like the commands it runs (d1a.training.recipe, d1a.eval.benchmark, d1a.training.calibrate):
1. train: the recipe's stages (d1a.training.recipe); the checkpoint is its last stage. Skipped once that stage is recorded.
2. calibrate: each --calibrate is scored raw (a suite: its calibration partition; an evals/d1a/<suite>:<partition>
   reference: that partition) and the temperature is fitted on the pooled rows by d1a.training.calibrate, which refuses
   rows the checkpoint trained on.
3. evaluate: each --evaluate is scored with that temperature (a suite: its development partition).
4. report: <out>/study/study.json and study.md, one row per evaluation (n, accuracy, ECE, NLL) with the temperature and
   its fit.
5. publish (--publish, or the publish command): the checkpoint and the report go to a Hub repo, private, under a prefix.
   Refused when the checkpoint has no fitted temperature, or one fitted in distribution unless --allow-in-distribution.
A step whose output exists is not run again, so a study that stopped continues where it stopped; an evaluation scored
at another temperature is scored again.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from d1a.eval.suite import read_json, write_json
from d1a.training import recipe as recipes

METRICS = ("n", "acc", "ece", "nll")


def slug(ref):
    """A folder name for an evaluation reference: evals/v7/decision-v7 -> v7_decision-v7, evals/d1a/pr-labels:development -> d1a_pr-labels_development."""
    return str(ref).removeprefix("evals/").replace("/", "_").replace(":", "_")


def benchmark_argv(ckpt, ref, out, device, split):
    """d1a.eval.benchmark's options for one reference: a suite directory (its `split` partition) or a D1A suite partition."""
    where = ["--data", ref] if ":" in str(ref) else ["--suite", ref, "--split", split]
    return ["--run", str(ckpt), *where, "--out", str(out), *(["--device", device] if device else [])]


def calibration(meta):
    """(calibrated, why not): the checkpoint's temperature fit as d1a.training.calibrate records it."""
    fit = meta.extra.get("temperature_fit")
    if not fit:
        return False, f"temperature {meta.temperature}, never fitted (python -m d1a.training.calibrate)"
    if fit.get("in_distribution"):
        return False, f"temperature {meta.temperature} was fitted in distribution: {fit['in_distribution']}"
    return True, ""


def module(name, argv):
    """Run `python -m <name> <argv>` -> exit code."""
    return subprocess.run([sys.executable, "-m", name, *map(str, argv)], check=False).returncode


class Study:
    def __init__(self, recipe_path, out, device=None, calibrate=(), evaluate=(), resume=0, dry_run=False, runner=module, allow_in_distribution=False):
        self.recipe_path, self.out, self.device, self.allow_in_distribution = recipe_path, Path(out), device, allow_in_distribution
        self.calibrate_refs, self.evaluate_refs, self.resume, self.dry_run, self.runner = list(calibrate), list(evaluate), resume, dry_run, runner
        recipe, _ = recipes.load(recipe_path)
        self.ckpt = self.out / recipe["stages"][-1]["name"]
        self.dir = self.out / "study"

    def call(self, name, argv):
        print(f"python -m {name} " + " ".join(map(str, argv)), flush=True)
        if self.dry_run:
            return
        code = self.runner(name, argv)
        if code:
            raise SystemExit(f"d1a.training.study: {name} failed (exit {code}); later steps not run")

    def score(self, ref, out, split, temperature=None):
        """Score one reference into `out`, unless a report there was scored at `temperature` (None: any)."""
        report = out / "report.json"
        if report.exists() and (temperature is None or read_json(report).get("temperature") == temperature):
            print(f"kept {report}", flush=True)
            return
        shutil.rmtree(out, ignore_errors=True)   # d1a.eval.benchmark refuses an existing --out
        out.parent.mkdir(parents=True, exist_ok=True)
        self.call("d1a.eval.benchmark", benchmark_argv(self.ckpt, ref, out, self.device, split))

    def run(self):
        if (self.ckpt / "recipe.json").exists():
            print(f"kept {self.ckpt} (trained)", flush=True)
        else:
            self.call("d1a.training.recipe", ["run", self.recipe_path, "--out", self.out, "--resume", self.resume,
                                              *(["--device", self.device] if self.device else []), *(["--dry-run"] if self.dry_run else [])])
        if not self.calibrate_refs:
            raise SystemExit("d1a.training.study: no --calibrate: a study's checkpoint is calibrated on held-out rows before it is evaluated")
        rows = []
        for ref in self.calibrate_refs:
            out = self.dir / "calibrate" / slug(ref)
            self.score(ref, out, "calibration")   # d1a.training.calibrate fits on the raw scores, whatever temperature they were served at
            rows.append(out / "rows.json")
        self.call("d1a.training.calibrate", ["--run", self.ckpt, *[x for r in rows for x in ("--rows", r)],
                                             *(["--allow-in-distribution"] if self.allow_in_distribution else [])])
        if self.dry_run:
            return None
        from d1a.backends.checkpoint import read_meta
        meta = read_meta(self.ckpt)
        for ref in self.evaluate_refs:
            self.score(ref, self.dir / "evaluate" / slug(ref), "development", temperature=meta.temperature)
        return self.report(meta)

    def report(self, meta):
        rows = []
        for ref in self.evaluate_refs:
            r = read_json(self.dir / "evaluate" / slug(ref) / "report.json")
            rows.append({"evaluation": ref, "split": r.get("split"), **{k: r["clean"].get(k) for k in METRICS}})
        stage = read_json(self.ckpt / "recipe.json")
        study = {"recipe": stage["recipe"]["name"], "recipe_sha256": stage["recipe_sha256"], "d1a_commit": stage.get("d1a_commit"),
                 "checkpoint": str(self.ckpt), "temperature": meta.temperature, "temperature_fit": meta.extra.get("temperature_fit"),
                 "calibrate": self.calibrate_refs, "evaluations": rows}
        write_json(self.dir / "study.json", study)
        fit = study["temperature_fit"] or {}
        lines = [f"# Study: {study['recipe']}", "",
                 f"Recipe sha256 `{study['recipe_sha256'][:12]}`, d1a `{(study['d1a_commit'] or 'unknown')[:8]}`; checkpoint `{self.ckpt.name}`.",
                 f"Temperature {meta.temperature:.4f}, fitted on {fit.get('n', '?')} held-out rows ({', '.join(self.calibrate_refs)}).", "",
                 "| evaluation | split | n | accuracy | ECE | NLL |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['evaluation']} | {r['split']} | {r['n']} | {r['acc']:.4f} | {r['ece']:.4f} | {r['nll']:.4f} |" for r in rows]
        (self.dir / "study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        return study


def publish(out, to, allow_in_distribution=False, api=None):
    """Upload a study's checkpoint and report to `to` ("owner/repo:prefix"): refused unless it is calibrated (calibration())."""
    from d1a.backends.checkpoint import read_meta
    out = Path(out)
    study = read_json(out / "study" / "study.json")
    ckpt = Path(study["checkpoint"])
    meta = read_meta(ckpt)
    ok, why = calibration(meta)
    if not ok and not (allow_in_distribution and meta.extra.get("temperature_fit")):
        raise SystemExit(f"d1a.training.study: not publishing {ckpt}: {why}")
    repo, _, prefix = to.partition(":")
    if not prefix:
        raise SystemExit("d1a.training.study: --to is owner/repo:prefix (the folder the checkpoint goes to)")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt), path_in_repo=f"{prefix}/checkpoint", repo_id=repo, commit_message=f"{prefix}: checkpoint",
                      ignore_patterns=["resume/*"])
    api.upload_folder(folder_path=str(out / "study"), path_in_repo=f"{prefix}/study", repo_id=repo, commit_message=f"{prefix}: study",
                      allow_patterns=["study.json", "study.md", "*/*/report.json"])
    print(f"published {ckpt} to {repo}/{prefix}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train, calibrate, evaluate and report (and publish with --publish)")
    r.add_argument("recipe")
    r.add_argument("--out", required=True)
    r.add_argument("--device", choices=["cpu", "mps", "cuda"])
    r.add_argument("--calibrate", action="append", default=[], help="held-out rows the temperature is fitted on: a suite (its calibration partition) or evals/d1a/<suite>:<partition>; repeat to pool")
    r.add_argument("--evaluate", action="append", default=[], help="a suite (its development partition) or evals/d1a/<suite>:<partition>; repeatable")
    r.add_argument("--resume", type=int, choices=[0, 1], default=0, help="passed to the recipe's training stages")
    r.add_argument("--publish", help="owner/repo:prefix to upload the calibrated checkpoint and its report to")
    r.add_argument("--dry-run", action="store_true", help="print every command and stop")
    p = sub.add_parser("publish", help="upload a finished study's checkpoint and report; refuses an uncalibrated checkpoint")
    p.add_argument("out")
    p.add_argument("--to", required=True, help="owner/repo:prefix")
    for x in (r, p):
        x.add_argument("--allow-in-distribution", action="store_true", help="fit and publish a temperature fitted in distribution (d1a.training.calibrate warns and records it)")
    a = ap.parse_args(argv)
    if a.cmd == "publish":
        return publish(a.out, a.to, a.allow_in_distribution)
    try:
        study = Study(a.recipe, a.out, a.device, a.calibrate, a.evaluate, a.resume, a.dry_run, allow_in_distribution=a.allow_in_distribution).run()
    except ValueError as e:
        raise SystemExit(f"d1a.training.study: {e}")
    if study and a.publish:
        publish(a.out, a.publish, a.allow_in_distribution)


if __name__ == "__main__":
    main()
