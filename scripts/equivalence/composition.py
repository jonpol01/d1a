"""d1a.training.composition against the module at --ref, bit for bit: the shapes, their splits and the held-out keys, then
every function (skeleton, sort_commutative, relabel, canonical, push_negation, structure_keys) on generated expression
trees of every operator, depth and atom numbering, the suite shapes included.

    uv run python scripts/equivalence/composition.py                   # against origin/main (before the rewrite)
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Checker, arguments, module_at  # noqa: E402

import d1a.training.composition as new  # noqa: E402

ARITY = {"not": 1, "and": 2, "or": 2, "unless": 2, "if": 3}


def tree(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return rng.randint(0, 6)
    op = rng.choice(list(ARITY))
    return (op, *(tree(rng, depth - 1) for _ in range(ARITY[op])))


def main():
    a = arguments(__doc__.split("\n")[0], "origin/main", 20000)
    old, rng, check = module_at(a.ref, "d1a/training/composition.py"), random.Random(a.seed), Checker()
    for name in ("SHAPES", "TRAIN_SHAPES", "DEV_SHAPES", "TEST_SHAPES"):
        check.equal(name, getattr(old, name), getattr(new, name))
    check.equal("HELD_OUT_KEYS", sorted(old.HELD_OUT_KEYS), sorted(new.HELD_OUT_KEYS))
    trees = list(old.SHAPES.values()) + [tree(rng, rng.randint(0, 6)) for _ in range(a.iterations)]
    for i, t in enumerate(trees):
        for fn in ("skeleton", "sort_commutative", "relabel", "canonical", "push_negation"):
            check.same(f"tree {i} {fn}", lambda: getattr(old, fn)(t), lambda: getattr(new, fn)(t))
        check.same(f"tree {i} structure_keys", lambda: sorted(old.structure_keys(t)), lambda: sorted(new.structure_keys(t)))
        check.same(f"tree {i} held out", lambda: bool(old.structure_keys(t) & old.HELD_OUT_KEYS), lambda: bool(new.structure_keys(t) & new.HELD_OUT_KEYS))
    print(f"d1a.training.composition: identical to {a.ref} on {check.count} comparisons")


if __name__ == "__main__":
    main()
