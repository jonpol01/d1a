### Changed

- `d1a/suite.py` is rewritten in D1A's own code, no longer derived from Kev. The pins, the partition checks, the Hub
  mirror rules, the file formats (byte for byte) and the training-source guard are unchanged (`scripts/equivalence/suite.py`).
  Its tests are rewritten too, as `tests/test_suite.py`.
