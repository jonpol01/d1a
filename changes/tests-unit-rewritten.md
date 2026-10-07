### Changed

- The Kev-derived tests left in `tests/test_unit.py` are rewritten as D1A's own. Epoch planning (`--length_sort`, the
  none-pair gate, `--pass_tokens_max`, `--row_budget`), a NaN gradient, the gradient-norm summary and resume points are
  in `tests/test_train.py`. The shared-prefix tests are in the new `tests/test_shared_prefix.py`, and the long-record
  `LocalPredictor` tests in `tests/test_predictors.py`. The tiny hybrid Qwen3.5 they train on is built once per session
  in `tests/conftest.py`. A test whose checks `tests/test_data.py`, `tests/test_train.py` and `tests/test_system_one.py`
  already made is gone, and so are two unused stand-ins. Of 57 bugs planted in the code under test, the old tests caught
  50 and the new ones 54, including all 50; the other three are equivalent mutants. No library code changed.
