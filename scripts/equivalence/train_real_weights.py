"""d1a.train on real weights, main against the branch, on MPS: a capped probe, then the module at --ref twice (MPS
training is not bit-deterministic run to run, so the two give the noise floor), then the working tree's, each a few
steps from the same seed and data. Reports, per pair, the largest adapter and head differences and the median step time.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6 uv run python scripts/equivalence/train_real_weights.py --steps 4
"""
import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, module_at  # noqa: E402

import d1a.train as new  # noqa: E402


def train(module, argv):
    sys.argv = ["d1a.train", *argv]
    with contextlib.redirect_stdout(io.StringIO()):
        module.main()


def weights(out):
    from safetensors.torch import load_file
    from d1a.checkpoint import read_meta
    return load_file(str(Path(out) / "adapter_model.safetensors")), read_meta(out).head


def drift(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--records", type=int, default=32)
    a = ap.parse_args()
    os.chdir(ROOT)
    if not os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"):
        raise SystemExit("set PYTORCH_MPS_HIGH_WATERMARK_RATIO (the 09-26 freeze rule)")
    from d1a.suite import load_split, read_json
    work = Path(tempfile.mkdtemp())
    rows = [{k: v for k, v in r.items() if k != "_meta"} for r in load_split("evals/v7/decision-v7", "development") if r["_meta"]["variant"] == "clean"][: a.records]
    (work / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    common = ["--data", str(work / "data.jsonl"), "--device", "mps", "--lora", "16", "--batch", "1", "--accum", "4", "--checkpointing", "1", "--seed", "0"]
    old = module_at(a.ref, "d1a/train.py")
    train(new, common + ["--max_steps", "1", "--out", str(work / "probe")])   # capped probe: one step, then the real runs
    print("probe ok", read_json(work / "probe" / "training_metrics.json")["step_seconds"], flush=True)
    runs = {}
    for name, module in (("main_a", old), ("main_b", old), ("branch", new)):
        out = work / name
        train(module, common + ["--max_steps", str(a.steps), "--out", str(out)])
        metrics = read_json(out / "training_metrics.json")
        runs[name] = (*weights(out), statistics.median(metrics["step_seconds"]))
        print(f"{name}: median step {runs[name][2]:.2f} s", flush=True)
    report = {"steps": a.steps, "records": len(rows),
              "noise_floor": {"adapter": drift(runs["main_a"][0], runs["main_b"][0]), "head": drift(runs["main_a"][1], runs["main_b"][1])},
              "main_vs_branch": {"adapter": max(drift(runs[m][0], runs["branch"][0]) for m in ("main_a", "main_b")),
                                 "head": max(drift(runs[m][1], runs["branch"][1]) for m in ("main_a", "main_b"))},
              "median_step_seconds": {k: round(v[2], 3) for k, v in runs.items()}}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
