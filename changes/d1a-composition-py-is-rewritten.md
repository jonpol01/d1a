### Changed

- `d1a/training/composition.py` (the compositional suites' rule shapes and the structure keys that keep held-out shapes
  out of training) is rewritten in D1A's own code, no longer derived from Kev. `scripts/equivalence/composition.py` found
  it identical to the previous version on 140,103 comparisons: the shapes and their splits, and every function on the
  suite shapes and 20,000 generated trees.
