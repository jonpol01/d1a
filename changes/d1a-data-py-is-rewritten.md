### Changed

- `d1a/data.py` is rewritten in D1A's own code, no longer derived from Kev. Every converter, `build`, `augment`,
  `none_pair`, `load_records` and `materialize` gives the same records from the same seeds, every random draw in the same
  order (`scripts/equivalence/data.py`, new). Its tests are rewritten too, as `tests/test_data.py`.
