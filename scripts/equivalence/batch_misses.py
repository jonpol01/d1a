"""probs_batch's packed passes of cache misses (d1a-engine#33) against each request alone, on real weights: the golden-vector
record set (scripts/golden_vectors.py) in shuffled groups, every request a miss whose prefix is not kept, against probs()
one request at a time. Prints the largest probability difference, every flipped answer with the margin it had alone, and
how many requests the packed passes took (the rest ran alone, as before: a group of one, or a request past ROW_PASS_TOKENS).

    uv run python scripts/equivalence/batch_misses.py --run JohnP1/d1a-e4b-it-qat --device cuda --dtype fp32
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/: golden_vectors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a PyTorch checkpoint (Hub id[@rev] or folder)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--group", type=int, default=8, help="requests per probs_batch call")
    ap.add_argument("--seeds", type=int, default=2, help="shuffles of the record set")
    ap.add_argument("--out", help="write the result as JSON here")
    a = ap.parse_args()
    from golden_vectors import encode, record_set
    from d1a.backends.checkpoint import Checkpoint, LoadOptions
    tok, model = Checkpoint(a.run).load(a.device, LoadOptions(backend="torch", dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype], merge=False))
    type(model).batch_misses = property(lambda self: True)   # on any device, for this check
    taken, packed = [], model._batched_misses
    model._batched_misses = lambda encs, idx: (lambda out: taken.append(len(out)) or out)(packed(encs, idx))
    records = record_set()
    encs = [encode(model, tok, rec) for _, _, rec, _ in records]
    with torch.no_grad():
        alone = [[p.float() for p in model.probs(e)] for e in encs]
    worst, flips, total = 0.0, [], 0
    for seed in range(a.seeds):
        order = list(range(len(encs)))
        random.Random(seed).shuffle(order)
        for start in range(0, len(order), a.group):
            group = order[start:start + a.group]
            with torch.no_grad():
                got, _ = model.probs_batch([encs[i] for i in group], [None] * len(group), [False] * len(group))
            total += len(group)
            for i, ps in zip(group, got):
                for q, (p, r) in enumerate(zip(ps, alone[i])):
                    p = p.float()
                    worst = max(worst, (p - r).abs().max().item())
                    if p.argmax() != r.argmax():
                        top = r.sort(descending=True).values
                        flips.append({"record": records[i][0], "question": q, "seed": seed, "margin_alone": round((top[0] - top[1]).item(), 4)})
    result = {"run": a.run, "device": a.device, "dtype": a.dtype, "requests": len(encs), "scored": total, "packed": sum(taken),
              "packed_share": round(sum(taken) / total, 3), "max_abs_dp": worst, "flips": flips, "torch": torch.__version__}
    print(json.dumps(result, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
