"""The verifier improving itself from outcomes (d1a.feedback), measured round by round on unseen repositories.

    uv run --extra mlx python recipes/swe-verifier/self_improve.py --start runs/swe-verifier/train/r1 --rounds 3 \
        --batch 300 --out runs/swe-verifier/self-improve

A stream of agent runs the model never saw (shards --stream-shards, train-split repositories) arrives in batches, as an
agent's patches would come back from the tests. Each round:
1. the incumbent scores the batch; each answer goes into a FeedbackLog, then its outcome (resolved or not);
2. the batch is split by repository: 80% become training records, 20% a calibration slice;
3. a candidate continues the incumbent's LoRA on all training records so far (d1a.train --init_from, --steps-per-round);
4. each model is recalibrated (OutcomeCalibrator) on the calibration slices so far, scored by itself: never on records
   it trained on, which would make it overconfident;
5. gate() on the dev repositories decides promotion (lower log loss, 95% interval clear of zero);
6. the incumbent, as served (its calibrator applied), is measured on the test repositories.
Round 0 is the starting model with no feedback. Writes curve.jsonl (one line per round) and the logs; only the
candidates' adapters are kept, and a rejected candidate's directory is deleted.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zero_shot import DATASET, QUESTION, auroc, calibration, split_of, state_of  # noqa: E402

from d1a.feedback import FeedbackLog, OutcomeCalibrator, gate, log_loss, records  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLS = ["instance_id", "target", "trajectory", "exit_status", "generated_patch"]


def runs(shards, split, per_issue, seed):
    rows = []
    for k in shards:
        rows += [r for r in pq.read_table(hf_hub_download(DATASET, f"data/train-{k:05d}-of-00012.parquet", repo_type="dataset"), columns=COLS).to_pylist()
                 if split_of(r["instance_id"].rsplit("-", 1)[0]) == split]
    by = {}
    for r in rows: by.setdefault(r["instance_id"], []).append(r)
    rng = random.Random(seed)
    out = [r for iid in sorted(by) for r in rng.sample(by[iid], min(per_issue, len(by[iid])))]
    rng.shuffle(out)
    return out


def repo(r): return r["instance_id"].rsplit("-", 1)[0]


class Scorer:
    """One loaded model at a time (the Mac holds one E4B); scores are cached per model and row. dry=True replaces the
    model with a noisy function of the outcome (to test the loop's plumbing without a GPU)."""

    def __init__(self, dry=False): self.name, self.model, self.cache, self.dry = None, None, {}, dry

    def free(self):
        import gc
        self.name, self.model = None, None; gc.collect()

    def __call__(self, run, rows):
        if self.dry:
            rng = np.random.default_rng(abs(hash(run)) % 2**32)
            return np.array([self.cache.setdefault((run, id(r)), float(np.clip(0.75 + 0.15 * r["target"] + rng.normal(0, 0.1), 0.01, 0.99))) for r in rows])
        if self.name != run:
            from d1a import D1A
            self.model = None; self.model, self.name = D1A.load(run), run
        out = []
        for r in rows:
            key = (run, id(r))
            if key not in self.cache:
                self.cache[key] = float(self.model.decide(state_of(r, 3000, 6000, 3000), QUESTION)["resolved"]["noul"])
            out.append(self.cache[key])
        return np.array(out)


def measure(y, p):
    e, b, _ = calibration(y, p)
    return {"auroc": round(auroc(y, p), 4), "ece": round(e, 4), "brier": round(b, 4), "log_loss": round(float(log_loss(y, p).mean()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", required=True); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--batch", type=int, default=300)
    ap.add_argument("--stream-shards", default="3,4,5"); ap.add_argument("--eval-shards", default="0,1,2"); ap.add_argument("--per-issue", type=int, default=4)
    ap.add_argument("--steps-per-round", type=int, default=30); ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="no model and no training: test the loop on stand-in scores")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stream = runs([int(x) for x in a.stream_shards.split(",")], "train", 2, a.seed)
    dev = runs([int(x) for x in a.eval_shards.split(",")], "dev", a.per_issue, a.seed)
    test = runs([int(x) for x in a.eval_shards.split(",")], "test", a.per_issue, a.seed)
    print(f"stream {len(stream)} runs, dev {len(dev)}, test {len(test)}", flush=True)
    y_dev, y_test = np.array([r["target"] for r in dev]), np.array([r["target"] for r in test])
    g_dev = np.array([repo(r) for r in dev])
    score, log = Scorer(a.dry_run), FeedbackLog(out / "feedback.jsonl")
    incumbent, cal_slices, train_recs = a.start, [], []
    calibrators = {incumbent: OutcomeCalibrator()}

    def served(run, rows):   # what the loop serves: the model's own score through its outcome calibrator
        c = calibrators[run]
        return np.array([c.p("resolved", p) for p in score(run, rows)])

    def report(rnd, extra):
        line = {"round": rnd, "incumbent": incumbent, "outcomes": len(log.resolved()), "test": measure(y_test, served(incumbent, test)), **extra,
                "time": time.strftime("%H:%M")}
        with open(out / "curve.jsonl", "a", encoding="utf-8") as f: f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)

    report(0, {"promoted": None})
    for rnd in range(1, a.rounds + 1):
        batch = stream[(rnd - 1) * a.batch: rnd * a.batch]
        if not batch: break
        p = score(incumbent, batch)
        for r, pr in zip(batch, p):   # the decision as made, then the outcome as reported
            did = log.decision(state_of(r, 3000, 6000, 3000), QUESTION, {"resolved": {"type": "noul", "noul": float(pr)}}, run=incumbent, meta={"repo": repo(r)})
            log.outcome(did, {"resolved": bool(r["target"])})
        resolved = log.resolved()[-len(batch):]
        held = {g for g in {d["meta"]["repo"] for d in resolved} if random.Random(f"{a.seed}{g}").random() < 0.2}
        cal_slices += [r for r in batch if repo(r) in held]
        train_recs += records([d for d in resolved if d["meta"]["repo"] not in held], src="swe-verifier-feedback")
        data = out / f"train-r{rnd}.jsonl"; data.write_text("".join(json.dumps(x) + "\n" for x in train_recs), encoding="utf-8")
        cand = str(out / f"candidate-r{rnd}")
        cmd = [sys.executable, "-m", "d1a.train", "--base", "google/gemma-4-E4B", "--base_revision", "411aa17b749aa952df1359d2dcea73917a544d9a",
               "--init_from", incumbent, "--data", str(data), "--max_state", "5600", "--device", "mps", "--lora", "16", "--batch", "1",
               "--accum", "8", "--lr", str(a.lr), "--weights_dtype", "bf16", "--checkpointing", "1", "--seed", str(a.seed + rnd),
               "--epochs", "1", "--max_steps", str(a.steps_per_round), "--out", cand]
        score.free()   # free the scorer's model before training on the same machine
        if a.dry_run:
            Path(cand).mkdir(parents=True, exist_ok=True); Path(cand, "head.pt").touch(); rc = 0
        else:
            with open(out / f"train-r{rnd}.log", "w") as f: rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not Path(cand, "head.pt").exists(): report(rnd, {"promoted": False, "error": f"training exit {rc}"}); continue
        y_cal = np.array([r["target"] for r in cal_slices])
        for run in (incumbent, cand):   # recalibrate each on the calibration slices, scored by itself
            res = [{"labels": {"resolved": bool(t)}, "answers": {"resolved": {"noul": float(q)}}} for t, q in zip(y_cal, score(run, cal_slices))]
            calibrators[run] = OutcomeCalibrator().fit(res, min_outcomes=10)
        g = gate(y_dev, served(cand, dev), served(incumbent, dev), groups=g_dev)
        if g["promote"]:
            incumbent = cand
        else:
            shutil.rmtree(cand, ignore_errors=True)
        report(rnd, {"promoted": g["promote"], "gate": {k: g[k] for k in ("delta_log_loss", "ci", "reasons")}, "calibration_slice": len(cal_slices),
                     "training_records": len(train_recs)})


if __name__ == "__main__":
    main()
