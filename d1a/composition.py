# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: only the rule shapes and structure keys d1a.suite.validate_training checks are kept; the record generator left with Kev's suite builder.
"""The compositional rule shapes of the frozen suites (decision-v4, decision-v7), and the structure keys that tell a
training record built on a held-out shape from one built on a training shape (d1a.suite.validate_training)."""

SHAPES = {
    "atom": 0,
    "negation": ("not", 0),
    "conjunction": ("and", 0, 1),
    "disjunction": ("or", 0, 1),
    "exception": ("unless", 0, 1),
    "conditional": ("if", 0, 1, 2),
    "nested_and": ("and", ("and", 0, 1), 2),
    "nested_or": ("or", 0, ("or", 1, 2)),
    "held_and_or": ("and", ("or", 0, 1), 2),
    "held_or_not": ("or", ("and", 0, 1), ("not", 2)),
    "held_conditional": ("if", 0, ("not", 1), 2),
    "final_combination": ("and", ("if", 0, 1, 2), 3),
    "final_negation": ("not", ("or", ("and", 0, 1), 2)),
    "final_exception": ("or", ("unless", 0, 1), 2),
}
TRAIN_SHAPES = tuple(list(SHAPES)[:8])
DEV_SHAPES = tuple(list(SHAPES)[8:11])
TEST_SHAPES = tuple(list(SHAPES)[11:])


def skeleton(tree):
    """Structure with leaf identities erased: used to order commutative children independently of atom numbering."""
    if isinstance(tree, int): return "_"
    op, *children = tree
    parts = [skeleton(c) for c in children]
    if op in ("and", "or"): parts = sorted(parts)
    return f"{op}({','.join(parts)})"


def sort_commutative(tree):
    if isinstance(tree, int): return tree
    op, *children = tree
    children = [sort_commutative(c) for c in children]
    if op in ("and", "or"): children = sorted(children, key=skeleton)
    return (op, *children)


def canonical(tree):
    """Structure key: commutative children ordered by skeleton, then leaves renumbered in traversal order, so
    (A and B) or not C and not C or (B and A) share one key."""
    t = relabel(sort_commutative(tree))
    def render(t):
        if isinstance(t, int): return str(t)
        return f"{t[0]}({','.join(render(c) for c in t[1:])})"
    return render(t)


def push_negation(tree):
    """De Morgan normal form, so a random tree equivalent to a held-out shape under negation pushing is also excluded."""
    if isinstance(tree, int): return tree
    op, *children = tree
    if op == "not":
        inner = children[0]
        if isinstance(inner, int): return tree
        iop, *ic = inner
        if iop == "not": return push_negation(ic[0])
        if iop in ("and", "or"): return ("or" if iop == "and" else "and", *[push_negation(("not", c)) for c in ic])
        if iop == "unless": return push_negation(("or", ("not", ic[0]), ic[1]))
        return ("not", push_negation(inner))
    if op == "unless": return ("and", push_negation(children[0]), push_negation(("not", children[1])))
    return (op, *[push_negation(c) for c in children])


def relabel(tree):
    """Renumber leaves in first-appearance order so structure keys do not depend on which atom index was drawn."""
    mapping = {}
    def walk(t):
        if isinstance(t, int):
            mapping.setdefault(t, len(mapping)); return mapping[t]
        return (t[0], *[walk(c) for c in t[1:]])
    return walk(tree)


def structure_keys(tree):
    return {canonical(tree), canonical(push_negation(tree))}


HELD_OUT_KEYS = set().union(*(structure_keys(SHAPES[s]) for s in DEV_SHAPES + TEST_SHAPES))
