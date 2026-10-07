### Changed

- Kev's `tests/test_research.py` is rewritten in D1A's own tests. The remote and rotation-averaged predictor tests are now
  `tests/test_predictors.py`; strict truncation and the batched mask's exact equality are in `tests/test_encoding.py`.
  The new tests catch every planted bug the old ones caught, plus real tokens seeing pads, which the old ones missed. Its
  unused fixtures are gone. No library code changed.
