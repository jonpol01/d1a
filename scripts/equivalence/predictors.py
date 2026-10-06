"""d1a.eval.predictors against the module at --ref, bit for bit (latency aside): LocalPredictor on the committed tiny Gemma 4
checkpoint (generated records of every question type, the long-row path, the CUDA kernel policy on CPU tensors, the
hybrid branch, context overflows, temperature checks); RemotePredictor on a stand-in endpoint (bodies, the request it
sends, retries and their waits, failures); RotationAveraged over stand-in predictors with and without logits.

    uv run python scripts/equivalence/predictors.py                    # against origin/main (before the rewrite)
"""
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, Checker, arguments, exact, module_at  # noqa: E402

import d1a.eval.predictors as new  # noqa: E402
from d1a.backends.checkpoint import LoadOptions  # noqa: E402

WORDS = "the customer is charged twice which team billing shipping refund angry how bad no yes order late lost box arrived wrong size".split()
CHECKPOINT = "tests/golden/tiny-gemma4/checkpoint"   # relative: the checkpoint names its base by a path from the repository


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def record(rng):
    questions = {}
    for i in range(rng.randint(1, 4)):
        kind = rng.choice(["choice", "noul", "score"])
        if kind == "choice":
            keys = [f"o{j}" for j in range(rng.randint(2, 6))]
            q = {"type": "choice", "instructions": text(rng, 4), "criteria": {k: text(rng, 3) for k in keys}, "label": rng.choice(keys)}
        elif kind == "noul":
            q = {"type": "noul", "instructions": text(rng, 4), "label": rng.random() < 0.5}
        else:
            levels = [text(rng, 2) for _ in range(rng.randint(2, 5))]
            q = {"type": "score", "instructions": text(rng, 4), "criteria": levels, "label": rng.randrange(len(levels))}
        questions[f"q{i}"] = {**q, "src": "s"}
    return {"state": text(rng, rng.randint(3, 40)), "questions": questions}


def without_latency(p):
    return {k: v for k, v in p.items() if k != "latency_ms"}


def local(check, old, rng, iterations):
    os.chdir(ROOT)
    opts = LoadOptions(dtype=torch.float32)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        check.same(f"temperature {bad}", lambda: old.LocalPredictor(CHECKPOINT, "cpu", LoadOptions(temperature=bad)),
                   lambda: new.LocalPredictor(CHECKPOINT, "cpu", LoadOptions(temperature=bad)))
    a, b = old.LocalPredictor(CHECKPOINT, "cpu", opts), new.LocalPredictor(CHECKPOINT, "cpu", opts)
    check.equal("loaded", (a.run, a.temperature, a.device, a.context), (b.run, b.temperature, b.device, b.context))
    records = [record(rng) for _ in range(iterations)]
    for i, r in enumerate(records):
        check.same(f"record {i}", lambda: without_latency(a(r)), lambda: without_latency(b(r)))
    saved = (old.ROW_PASS_TOKENS, new.ROW_PASS_TOKENS, old.sync, new.sync)
    try:
        old.ROW_PASS_TOKENS = new.ROW_PASS_TOKENS = 8                      # every record now has long rows
        for i, r in enumerate(records[:20]):
            check.same(f"long {i}", lambda: without_latency(a(r)), lambda: without_latency(b(r)))
        old.sync = new.sync = lambda device: None
        a.device = b.device = "cuda"                                         # the CUDA kernel policy, on CPU tensors
        for i, r in enumerate(records[:10]):
            check.same(f"efficient {i}", lambda: without_latency(a(r)), lambda: without_latency(b(r)))
        a.device = b.device = "cpu"
        calls = {"old": [], "new": []}
        for name, p in (("old", a), ("new", b)):                            # the hybrid branch, through a stand-in
            p.model.hybrid = True
            p.model.forward_batch = lambda encs, shared_prefix=False, p=p, name=name: (calls[name].append(shared_prefix), [p.model.forward(encs[0])])[1]
        for i, r in enumerate(records[:10]):
            check.same(f"hybrid {i}", lambda: without_latency(a(r)), lambda: without_latency(b(r)))
        check.equal("hybrid calls", calls["old"], calls["new"])
        for name, p in (("old", a), ("new", b)):
            p.model.hybrid = False
            del p.model.forward_batch
    finally:
        old.ROW_PASS_TOKENS, new.ROW_PASS_TOKENS, old.sync, new.sync = saved
    for limit in (5, 20, 60):                                                # context overflows
        for name, p in (("old", a), ("new", b)):
            p.context = {"max_state": 4096, "max_branch": 4096, "max_packed": limit}
        for i, r in enumerate(records[:15]):
            check.same(f"packed limit {limit} record {i}", lambda: without_latency(a(r)), lambda: without_latency(b(r)))


class Response:
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self.body


def remote(check, old, rng, iterations):
    real_urlopen, real_sleep = urllib.request.urlopen, time.sleep
    try:
        for i in range(iterations):
            r = record(rng)
            answers = {}
            for qid, q in r["questions"].items():
                if q["type"] == "noul":
                    answers[qid] = {"noul": rng.random()}
                elif q["type"] == "choice":
                    answers[qid] = {"probabilities": {k: rng.random() for k in q["criteria"]}}
                else:
                    answers[qid] = {"probabilities": {str(j): rng.random() for j in range(len(q["criteria"]))}}
            body = {"answers": answers}
            if rng.random() < 0.7: body["model"] = rng.choice(["d1a-latest", "x"])
            if rng.random() < 0.7: body["usage"] = {"input_tokens": rng.randint(1, 500)}
            failures = rng.choice([0, 0, 1, 2, 5])
            retries, model = rng.choice([1, 2, 3]), rng.choice(["d1a-latest", "m"])
            results = {}
            for name, mod in (("old", old), ("new", new)):
                seen, waits, attempts = [], [], [0]

                def urlopen(req, timeout, seen=seen, attempts=attempts):
                    seen.append((req.full_url, req.get_method(), req.data, sorted(req.header_items()), timeout))
                    attempts[0] += 1
                    if attempts[0] <= failures:
                        raise OSError(f"attempt {attempts[0]} failed")
                    return Response(json.dumps(body).encode())
                urllib.request.urlopen, time.sleep = urlopen, waits.append
                p = mod.RemotePredictor("http://example.test/", model=model, retries=retries, timeout=7)
                p.served_model = "before"
                result = ("ok", without_latency(p(r))) if failures < retries else None
                if result is None:
                    try: p(r); result = ("ok?",)
                    except Exception as e: result = ("error", type(e).__name__, str(e))
                results[name] = (exact(result), seen, waits, p.served_model)
            check.equal(f"remote {i}", results["old"], results["new"])
        check.same("concurrency 0", lambda: old.RemotePredictor("u", concurrency=0), lambda: new.RemotePredictor("u", concurrency=0))
    finally:
        urllib.request.urlopen, time.sleep = real_urlopen, real_sleep


def rotation(check, old, rng, iterations):
    def stand_in(with_logits, temperature):
        def predict(r):
            out = {"probabilities": {}, "latency_ms": 3.0}
            if with_logits:
                out["logits"] = {}
                if temperature is not None: out["inference_temperature"] = temperature
            for qid, q in r["questions"].items():
                keys = list(q["criteria"]) if q["type"] == "choice" else (["true", "false"] if q["type"] == "noul" else [str(j) for j in range(len(q["criteria"]))])
                z = {k: (hash((qid, k, position)) % 997) / 97.0 - 5 for position, k in enumerate(keys)}   # order-dependent, deterministic
                top = max(z.values()); e = {k: 2.718281828 ** (v - top) for k, v in z.items()}; s = sum(e.values())
                out["probabilities"][qid] = {k: v / s for k, v in e.items()}
                if rng.random() < 0.02: out["probabilities"][qid][keys[0]] = 0.0     # log of a zero probability
                if with_logits: out["logits"][qid] = z
            return out
        predict.temperature, predict.concurrency = temperature, 3
        return predict
    for i in range(iterations):
        r = record(rng)
        for with_logits in (True, False):
            for rotations in (2, 3, 6):
                temperature = rng.choice([None, 1.0, 1.7])
                state = rng.getstate()
                rng.setstate(state); a = old.RotationAveraged(stand_in(with_logits, temperature), rotations)
                oa = (exact(a(r)), a.temperature, a.concurrency)
                rng.setstate(state); b = new.RotationAveraged(stand_in(with_logits, temperature), rotations)
                check.equal(f"rotation {i} logits={with_logits} r={rotations}", oa, (exact(b(r)), b.temperature, b.concurrency))
        for k in range(8):
            check.same(f"rotated {i} {k}", lambda: old.RotationAveraged.rotated(r, k), lambda: new.RotationAveraged.rotated(r, k))
    check.same("one rotation", lambda: old.RotationAveraged(lambda r: r, 1), lambda: new.RotationAveraged(lambda r: r, 1))


def main():
    a = arguments(__doc__.split("\n")[0], "origin/main", 60)
    old, rng, check = module_at(a.ref, "d1a/eval/predictors.py"), random.Random(a.seed), Checker()
    check.equal("LONG_ROW_KERNELS", old.LONG_ROW_KERNELS, new.LONG_ROW_KERNELS)
    local(check, old, rng, a.iterations)
    remote(check, old, rng, a.iterations * 5)
    rotation(check, old, rng, a.iterations)
    print(f"d1a.eval.predictors: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
