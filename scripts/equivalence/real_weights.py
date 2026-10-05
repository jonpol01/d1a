"""A rewrite of the loading or model code against the module at --ref, on real weights: the same checkpoint loaded
through each version in turn (one in memory at a time), the golden-vector record set (scripts/golden_vectors.py, ~209
requests) scored by both, every probability compared, and load time and per-request latency measured. John's rule for
model.py, checkpoint.py and train.py rewrites: identical answers and equal-or-better latency, main against the branch.

    uv run python scripts/equivalence/real_weights.py --module d1a/checkpoint.py --run JohnP1/d1a-e2b --device mps
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/: golden_vectors
from common import ROOT, module_at  # noqa: E402


def score(checkpoint_module, run, device, dtype, records, rounds):
    """-> (probabilities per record, load seconds, per-request milliseconds over `rounds` passes)."""
    from golden_vectors import encode
    from d1a.device import empty_cache, sync
    t = time.perf_counter()
    tok, model = checkpoint_module.Checkpoint(run).load(device, checkpoint_module.LoadOptions(backend="torch", dtype=dtype))
    sync(device)
    loaded = time.perf_counter() - t
    probs, times = [], []
    with torch.no_grad():
        for r, (_, _, rec, _) in enumerate(records):
            enc = encode(model, tok, rec)
            out = model.probs(enc)
            probs.append([p.float().tolist() for p in out])
        for _ in range(rounds):
            for _, _, rec, _ in records:
                enc = encode(model, tok, rec)
                sync(device); t = time.perf_counter(); model.probs(enc); sync(device)
                times.append((time.perf_counter() - t) * 1000)
    del model, tok
    gc.collect(); empty_cache(device)
    return probs, loaded, times


def interleaved(old_module, new_module, run, device, dtype, records, rounds):
    """Latency with both versions in memory at once, alternating per request and swapping which goes first on every item,
    so the machine's background load falls on both alike (a sequential main-then-branch run on a loaded Mac once showed a
    fake +43%). -> {"old": [ms], "new": [ms]}, and the load time of each."""
    from golden_vectors import encode
    from d1a.device import sync
    loaded, models = {}, {}
    for name, module in (("old", old_module), ("new", new_module)):
        t = time.perf_counter()
        models[name] = module.Checkpoint(run).load(device, module.LoadOptions(backend="torch", dtype=dtype))
        sync(device); loaded[name] = round(time.perf_counter() - t, 2)
    times = {"old": [], "new": []}
    with torch.no_grad():
        for r in range(rounds):
            for i, (_, _, rec, _) in enumerate(records):
                order = ("old", "new") if (i + r) % 2 == 0 else ("new", "old")
                for name in order:
                    tok, model = models[name]
                    enc = encode(model, tok, rec)
                    sync(device); t = time.perf_counter(); model.probs(enc); sync(device)
                    times[name].append((time.perf_counter() - t) * 1000)
    return times, loaded


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--module", default="d1a/checkpoint.py", help="the rewritten module whose old version is loaded from --ref")
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--run", default="JohnP1/d1a-e2b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N records (0: all)")
    ap.add_argument("--out", default="")
    ap.add_argument("--interleave", action="store_true", help="latency only: both versions in memory, alternating per request (needs twice the memory)")
    a = ap.parse_args()
    import os
    os.chdir(ROOT)
    from golden_vectors import record_set
    import d1a.checkpoint as new_checkpoint
    if a.module != "d1a/checkpoint.py":
        raise SystemExit("only d1a/checkpoint.py is wired so far; model.py and train.py swap the module the checkpoint loads through")
    old_checkpoint = module_at(a.ref, a.module)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    records = record_set()[: a.limit or None]
    if a.interleave:
        times, loaded = interleaved(old_checkpoint, new_checkpoint, a.run, a.device, dtype, records, a.rounds)
        report = {"module": a.module, "ref": a.ref, "run": a.run, "device": a.device, "dtype": a.dtype, "records": len(records), "mode": "interleaved",
                  "load_s": loaded, **{name: {"median_ms": round(statistics.median(t), 2), "p90_ms": round(sorted(t)[int(0.9 * len(t))], 2), "n": len(t)}
                                       for name, t in times.items()},
                  "new_over_old_median": round(statistics.median(times["new"]) / statistics.median(times["old"]), 4)}
        print(json.dumps(report, indent=2))
        if a.out: Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return
    results = {}
    for name, module in (("old", old_checkpoint), ("new", new_checkpoint), ("old again", old_checkpoint), ("new again", new_checkpoint)):
        probs, loaded, times = score(module, a.run, a.device, dtype, records, a.rounds)
        results[name] = {"probs": probs, "load_s": round(loaded, 2), "median_ms": round(statistics.median(times), 2),
                         "p90_ms": round(sorted(times)[int(0.9 * len(times))], 2)}
        print(f"{name:10s} load {loaded:6.1f} s  median {results[name]['median_ms']:7.2f} ms  p90 {results[name]['p90_ms']:7.2f} ms", flush=True)
    worst = max(abs(x - y) for a_, b_ in zip(results["old"]["probs"], results["new"]["probs"]) for pa, pb in zip(a_, b_) for x, y in zip(pa, pb))
    bitwise = results["old"]["probs"] == results["new"]["probs"]
    report = {"module": a.module, "ref": a.ref, "run": a.run, "device": a.device, "dtype": a.dtype, "records": len(records),
              "questions": sum(len(p) for p in results["old"]["probs"]), "identical_bitwise": bitwise, "max_abs_dp": worst,
              "timing": {k: {kk: v[kk] for kk in ("load_s", "median_ms", "p90_ms")} for k, v in results.items()}}
    print(json.dumps(report, indent=2))
    if a.out: Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
