"""The prefix-cache HIT path (probs_with_prefix) on real weights: main's d1a/backends/torch.py against this checkout's,
on one loaded checkpoint. real_weights.py scores model.probs() only and never a hit, so a change to the hit path needs
this script (#256).

For each golden record (scripts/golden_vectors.record_set) one miss with main's code (probs_and_prefix) builds THE
prefix every hit below reuses (the cache is cropped back after each hit), so two forward passes never differ by their
prefix: main's hit, this checkout's hit, then main's hit again. Main's two hits measure the run-to-run floor of the
device and dtype. Bar: max |main - this| <= max(1e-5, max |main - main again|), with every row's hit packed as served.
--rounds N then times both hits alternately, swapping which goes first (0: no timing).

    PYTHONPATH=<checkout> python scripts/equivalence/hit_real.py --ref <main sha> --run JohnP1/d1a-e4b@v0.6 --out <json>
"""
import argparse, json, random, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/equivalence"), str(ROOT / "scripts"), str(ROOT)]


def maxdiff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="main's commit"); ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="mps"); ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--rounds", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from common import module_at
    from golden_vectors import encode, record_set
    import d1a.backends.checkpoint as ck
    from d1a.backends.device import sync
    old = module_at(a.ref, "d1a/backends/torch.py").DecisionModel
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    tok, model = ck.Checkpoint(a.run).load(a.device, ck.LoadOptions(backend="torch", dtype=dtype))
    new, rows, rng = type(model), [], random.Random(0)
    with torch.no_grad():
        for rid, _, rec, _ in record_set():
            enc = encode(model, tok, rec)
            miss, pre = old.probs_and_prefix(model, enc)
            h_old = [p.float() for p in old.probs_with_prefix(model, enc, pre)]
            h_new = [p.float() for p in new.probs_with_prefix(model, enc, pre)]
            h_old2 = [p.float() for p in old.probs_with_prefix(model, enc, pre)]
            miss = [p.float() for p in miss]
            row = {"id": rid, "tokens": len(enc["ids"]), "state_tokens": pre[0], "rows_form": bool(model.rows_form([enc])),
                   "identical_old_new": all(torch.equal(x, y) for x, y in zip(h_old, h_new)),
                   "identical_old_old": all(torch.equal(x, y) for x, y in zip(h_old, h_old2)),
                   "dp_old_new": maxdiff(h_old, h_new), "dp_old_old": maxdiff(h_old, h_old2),
                   "dp_old_miss": maxdiff(h_old, miss), "dp_new_miss": maxdiff(h_new, miss)}
            t = {"old": [], "new": []}
            for r in range(a.rounds):
                order = [("old", old), ("new", new)]
                if (r + rng.random() > 0.5) % 2: order.reverse()
                for name, cls in order:
                    sync(a.device); s = time.perf_counter(); cls.probs_with_prefix(model, enc, pre); sync(a.device)
                    t[name].append((time.perf_counter() - s) * 1000)
            if a.rounds: row |= {"old_ms": statistics.median(t["old"]), "new_ms": statistics.median(t["new"])}
            rows.append(row); del pre
    floor = max(r["dp_old_old"] for r in rows); dp = max(r["dp_old_new"] for r in rows)
    summary = {"ref": a.ref, "run": a.run, "device": a.device, "dtype": a.dtype, "records": len(rows),
               "packed_hits": sum(not r["rows_form"] for r in rows),
               "identical_old_new": sum(r["identical_old_new"] for r in rows), "identical_old_old": sum(r["identical_old_old"] for r in rows),
               "max_dp_old_new": dp, "max_dp_old_old (floor)": floor,
               "max_dp_old_miss": max(r["dp_old_miss"] for r in rows), "max_dp_new_miss": max(r["dp_new_miss"] for r in rows),
               "bar": f"max|main - this| <= max(1e-5, floor): {'PASS' if dp <= max(1e-5, floor) else 'FAIL'}"}
    if a.rounds:
        ratio = [r["new_ms"] / r["old_ms"] for r in rows]
        boot = sorted(statistics.fmean(rng.choices(ratio, k=len(ratio))) for _ in range(2000))
        summary |= {"ratio_new_over_old_mean": round(statistics.fmean(ratio), 4), "ratio_ci95": [round(boot[49], 4), round(boot[1949], 4)]}
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    sys.exit(0 if "PASS" in summary["bar"] else 1)


if __name__ == "__main__":
    main()
