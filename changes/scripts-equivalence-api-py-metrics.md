### Added

- `scripts/equivalence/` (`api.py`, `metrics.py`, `benchmark.py`, `suite.py`): each runs a rewritten module and the same
  module at a git commit (`--ref`; the default is the last commit before its rewrite) on generated inputs, and stops at
  the first difference: types, float bits, key order, files byte for byte, error messages.
