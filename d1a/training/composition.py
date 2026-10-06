"""The rule shapes of the frozen compositional suites (decision-v4, decision-v7), as boolean expression trees, and the
structure keys that tell a training record built on a held-out shape from one built on a training shape
(d1a.eval.suite.validate_training).

A tree is an atom index (an int) or a tuple (operator, child, ...): ("not", a), ("and", a, b), ("or", a, b),
("unless", a, b) (a, unless b) and ("if", a, b, c) (if a then b else c). Two trees share a structure key when they differ
only in the order of an "and" / "or"'s children and in which atom numbers were drawn; a tree also carries the key of its
form with every negation pushed down to the atoms (De Morgan), so a random tree equivalent to a held-out shape in that
form is held out too.
"""

# The shapes in the order the suites split them: the first 8 train, the next 3 development, the last 3 test.
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
_NAMES = tuple(SHAPES)
TRAIN_SHAPES, DEV_SHAPES, TEST_SHAPES = _NAMES[:8], _NAMES[8:11], _NAMES[11:]

COMMUTATIVE = ("and", "or")
DUAL = {"and": "or", "or": "and"}


def is_atom(tree):
    return isinstance(tree, int)


def skeleton(tree):
    """The tree with every atom written "_" and commutative children sorted: what orders those children independently of
    the atom numbers drawn."""
    if is_atom(tree):
        return "_"
    op, children = tree[0], [skeleton(c) for c in tree[1:]]
    return f"{op}({','.join(sorted(children) if op in COMMUTATIVE else children)})"


def sort_commutative(tree):
    """The tree with each "and" / "or"'s children in skeleton order."""
    if is_atom(tree):
        return tree
    op, children = tree[0], [sort_commutative(c) for c in tree[1:]]
    return (op, *(sorted(children, key=skeleton) if op in COMMUTATIVE else children))


def relabel(tree):
    """The tree with its atoms renumbered in order of first appearance (depth first, left to right)."""
    numbers = {}

    def walk(t):
        if is_atom(t):
            return numbers.setdefault(t, len(numbers))
        return (t[0], *(walk(c) for c in t[1:]))
    return walk(tree)


def render(tree):
    return str(tree) if is_atom(tree) else f"{tree[0]}({','.join(render(c) for c in tree[1:])})"


def canonical(tree):
    """The structure key: commutative children sorted, then atoms renumbered, so (A and B) or not C and not C or (B and A)
    share one key."""
    return render(relabel(sort_commutative(tree)))


def push_negation(tree):
    """The tree with each negation pushed down to the atoms (De Morgan; "a unless b" is "a and not b"). A negated "if" stays
    negated, its branches pushed."""
    if is_atom(tree):
        return tree
    op, children = tree[0], tree[1:]
    if op == "unless":
        a, b = children
        return ("and", push_negation(a), push_negation(("not", b)))
    if op != "not":
        return (op, *(push_negation(c) for c in children))
    inner = children[0]
    if is_atom(inner):
        return tree
    iop, ichildren = inner[0], inner[1:]
    if iop == "not":
        return push_negation(ichildren[0])
    if iop in DUAL:
        return (DUAL[iop], *(push_negation(("not", c)) for c in ichildren))
    if iop == "unless":   # not (a unless b) = not (a and not b) = not a or b
        a, b = ichildren
        return push_negation(("or", ("not", a), b))
    return ("not", push_negation(inner))


def structure_keys(tree):
    """The tree's keys: its own and that of its negation-pushed form."""
    return {canonical(tree), canonical(push_negation(tree))}


# Every key a development or test shape has: a training record on any of them is refused.
HELD_OUT_KEYS = set().union(*(structure_keys(SHAPES[name]) for name in DEV_SHAPES + TEST_SHAPES))
