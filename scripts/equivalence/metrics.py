"""d1a.eval.metrics against the module at --ref, bit for bit: every public function on generated rows (with and without
logits, saved temperatures, ties, score questions, unknowable records, invalid inputs), reports with their key order,
fitted temperatures, folds and bootstrap draws.

    uv run python scripts/equivalence/metrics.py                 # against e3c0eca8, the last commit before the rewrite
"""
import random
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, arguments, module_at  # noqa: E402

import d1a.eval.metrics as new  # noqa: E402


def main():
    a = arguments(__doc__.split("\n")[0], "e3c0eca8", 500)
    warnings.simplefilter("ignore")
    old, rng, check = module_at(a.ref, "d1a/eval/metrics.py"), random.Random(a.seed), Checker()
    for name in ("EPSILON", "TEMPERATURE_FIT", "TEMPERATURE_FIT_METHOD", "LENGTH_EDGES", "LENGTH_METRICS"):
        check.equal(name, getattr(old, name), getattr(new, name))

    def row(i):
        kind = rng.choice(["choice", "noul", "score"]); k = 2 if kind == "noul" else rng.randint(1, 6)
        z = [rng.choice([rng.gauss(0, 3), 0.0, 1.0]) for _ in range(k)]
        if rng.random() < 0.1: z = [z[0]] * k
        t = rng.choice([1.0, 1.0, 2.0, 0.5, None])
        e = np.exp((np.array(z) - max(z)) / (t or 1.0)); p = (e / e.sum()).tolist()
        if rng.random() < 0.05: p = [round(x, 2) for x in p]
        r = {"p": p, "label": rng.randrange(k), "type": kind, "keys": [str(j) for j in range(k)], "task": rng.choice("abc"),
             "source": rng.choice(["s1", "s2", "s3", "unknowable", "unknowable_control"]), "group": f"g{rng.randint(0, 6)}",
             "variant": rng.choice(["clean"] * 5 + ["permuted"]), "id": f"r{i}", "question": rng.choice(["q1", "q2"]),
             "state_tokens": rng.choice([None, 100, 9000, 20000, 70000])}
        if rng.random() < 0.7: r["logits"] = [x / (t or 1.0) for x in z]
        if t is not None or rng.random() < 0.5: r["inference_temperature"] = t
        if r["source"] == "unknowable" and rng.random() < 0.7: r["control_id"] = f"r{rng.randint(0, i + 1)}"
        return r

    def both(name, *args, **kwargs):
        run = lambda m: (lambda: (lambda out: list(out) if name == "cluster_resamples" else out)(getattr(m, name)(*args, **kwargs)))
        check.same(name, run(old), run(new))

    for _ in range(a.iterations):
        n = rng.randint(0, 40); rows = [row(i) for i in range(n)]
        for i in range(0, n, 7): rows[i]["inference_temperature"] = 1.0
        t = rng.choice([1.0, 0.7, 2.5, 0.0, float("inf")])
        for r in rows[:5]:
            both("probabilities_at_temperature", r, t); both("nll_at_temperature", r, t); both("tempered_row", r, rng.choice([0.5, 2.0]))
            both("raw_row", r); both("recorded", r)
        both("metrics", rows, t if t not in (0.0, float("inf")) else 1.0); both("metrics", rows)
        both("grouped_metrics", rows, "task"); both("unknowable_report", rows); both("scored_rows", rows)
        conf = [rng.choice([0.5, 0.9, 1.0, rng.random()]) for _ in range(n)]; ok = [rng.random() < 0.7 for _ in range(n)]
        both("ece", conf, ok)
        for budget in (0.0, 0.05, 0.01, 1.0, -1.0, float("nan")):
            both("coverage_at_error", conf, ok, budget); both("select_threshold", conf, ok, budget, rng.choice([1, 3, 0]))
        if n: both("risk_coverage_curve", conf, ok)
        both("area_under_risk_coverage", conf, ok); both("evaluate_threshold", conf, ok, rng.choice([None, 0.5, 0.9, 1.0, 2.0]))
        both("calibration_by_length", rows, {r["id"]: rng.randint(0, 80000) for r in rows}); both("calibration_by_length", rows)
        both("length_buckets"); both("length_buckets", (1024, 4096)); both("length_buckets", (1000,))
        both("served_at", rows, rng.choice([1.0, 1.7])); both("served", rows, rows); both("served", rows, rows, points=11, aggregation="macro")
        both("fit_temperature", rows)
        raw = [old.raw_row(old.recorded(r)) for r in rows if "logits" in r or r.get("inference_temperature") in (None, 1.0)]
        samples, seed = rng.randint(1, 30), rng.randint(0, 5)
        both("cluster_resamples", raw, samples, seed); both("grouped_folds", raw, rng.choice([2, 3, 5]), seed)
        both("out_of_fold_rows", raw, rng.choice([2, 3]), seed, points=11); both("cross_validated_temperature", raw, 3, seed, samples, points=11)
        other = [{**r, "p": r["p"][::-1]} if rng.random() < 0.5 else r for r in raw]
        for metric in ("nll", "acc", "brier", "ece", "aurc", "coverage_at_5pct_error", "coverage_at_1pct_error", "confidence_bias", "bogus"):
            both("paired_bootstrap", raw, other, samples, seed, metric, rng.choice(["micro", "macro"]))
    print(f"d1a.eval.metrics: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
