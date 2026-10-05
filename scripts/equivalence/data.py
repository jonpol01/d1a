"""d1a.training.data against the module at --ref: every converter and build() on stand-in datasets (no download), augment and
none_pair from seeded generators, load_records on generated files, materialize and api_request; records compared with
their key order, and each generator's state compared after every call so the draws stay in step.

    uv run python scripts/equivalence/data.py                    # against b02d3545, the last commit before the rewrite
"""
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, arguments, module_at  # noqa: E402

import d1a.training.data as new  # noqa: E402

WORDS = ["Shoes", "arrived", "late", "Ünï", "twice", "charged", "refund", "the", "  ", "?", "wrong size", "Straße"]


class Dataset:
    """A stand-in for a datasets split: indexable rows and a label feature."""

    def __init__(self, rows, names=None):
        self.rows, self.features = rows, {"label": SimpleNamespace(names=names or [])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return dict(self.rows[i])


def fake_datasets(rng):
    text = lambda: " ".join(rng.choices(WORDS, k=rng.randint(1, 30)))
    intents = [f"intent_{i}" for i in range(rng.randint(2, 6))]
    n = rng.randint(0, 12)
    return {
        "legacy-datasets/banking77": Dataset([{"text": text(), "label": rng.randrange(len(intents))} for _ in range(n)], intents),
        "google/boolq": Dataset([{"question": text() + rng.choice(["", "?", "??"]), "passage": text(), "answer": rng.random() < 0.5} for _ in range(n)]),
        "fancyzhx/ag_news": Dataset([{"text": text(), "label": rng.randrange(4)} for _ in range(n)]),
        "nyu-mll/multi_nli": Dataset([{"premise": text(), "hypothesis": text(), "label": rng.choice([0, 1, 2, -1])} for _ in range(n)]),
        "SetFit/sst5": Dataset([{"text": text(), "label": rng.randrange(5)} for _ in range(n)]),
        "Yelp/yelp_review_full": Dataset([{"text": text(), "label": rng.randrange(5)} for _ in range(n)]),
    }


def main():
    a = arguments(__doc__.split("\n")[0], "b02d3545", 300)
    old, rng, check = module_at(a.ref, "d1a/training/data.py"), random.Random(a.seed), Checker()
    for name in ("REPOS", "NONE", "NONE_OPTIONS", "DISTRACTORS", "AG", "MNLI", "SST5", "YELP", "BANK_TEMPLATES", "EVAL_ONLY"):
        check.equal(name, getattr(old, name), getattr(new, name))
    check.equal("SOURCES", {k: v[1:] for k, v in old.SOURCES.items()}, {k: v[1:] for k, v in new.SOURCES.items()})

    def in_step(label, call):
        """call(module, generator) on both modules from the same seed; results and the generators' states must match."""
        seed = rng.getrandbits(32)
        gens = [random.Random(seed), random.Random(seed)]
        check.same(label, lambda: call(old, gens[0]), lambda: call(new, gens[1]))
        check.equal(label + " draws", gens[0].getstate(), gens[1].getstate())

    for _ in range(a.iterations):
        datasets = fake_datasets(rng)
        for m in (old, new):
            m._dataset = lambda repo, split, src, d=datasets: d[repo]
        n, seed = rng.randint(0, 15), rng.randint(0, 99)
        for source in old.SOURCES:
            def convert(m, _, source=source):
                src = m.Source(old.source_seed(seed, source), "rev")
                records = m.SOURCES[source][0]("train", n, src)
                return records, src.origins, src.getstate()
            check.same(f"convert {source}", lambda: convert(old, None), lambda: convert(new, None))
        only = rng.sample(list(old.SOURCES), rng.randint(0, 3))
        split = rng.choice(["train", "test"])
        check.same("build", lambda: old.build(n, split, seed, only=only), lambda: new.build(n, split, seed, only=only))
        check.same("build test", lambda: old.build(n, "test", seed), lambda: new.build(n, "test", seed))
        check.same("build unknown", lambda: old.build(1, only=["nope"]), lambda: new.build(1, only=["nope"]))
        records = old.build(n, "train", seed)
        for r in records[:6]:
            if rng.random() < 0.2:
                q = next(iter(r["questions"].values()))
                if q["type"] == "choice": q["target"] = {k: rng.random() for k in list(q["criteria"])[:2]}
            probs = rng.choice([(0.1, 0.12, 0.15), (0.5, 0.25, 0.25), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.6, 0.6, 0.0), (-0.1, 0, 0)])
            in_step("augment", lambda m, g, r=r, probs=probs: m.augment(json.loads(json.dumps(r)), g, *probs))
            in_step("none_pair", lambda m, g, r=r: m.none_pair(json.loads(json.dumps(r)), g))
            check.same("api_request", lambda: old.api_request(r), lambda: new.api_request(r))
            check.same("materialize", lambda: old.materialize(json.loads(json.dumps(r))), lambda: new.materialize(json.loads(json.dumps(r))))
        path = Path(tempfile.mkdtemp()) / "data.jsonl"
        lines = [json.dumps({k: v for k, v in r.items() if k != "_meta" or rng.random() < 0.3}, ensure_ascii=rng.random() < 0.5) for r in records[:5]]
        if rng.random() < 0.2: lines.insert(rng.randint(0, len(lines)), "   ")
        if rng.random() < 0.1: lines.append(json.dumps({"state": "s", "questions": {"q": {"type": "noul"}}}))
        if rng.random() < 0.05: lines = []
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        check.same("load_records", lambda: old.load_records(path, "mine"), lambda: new.load_records(path, "mine"))
        check.equal("source_seed", old.source_seed(seed, "x"), new.source_seed(seed, "x"))
    print(f"d1a.training.data: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
