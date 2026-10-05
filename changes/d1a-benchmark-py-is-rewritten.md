### Changed

- `d1a/benchmark.py` is rewritten in D1A's own code, no longer derived from Kev. Its rows, reports, the files it writes
  (byte for byte), its failure and skip rules and its command line are unchanged: checked against the previous version
  on about 34,000 generated cases and end to end on a frozen suite. Its tests are rewritten too, as
  `tests/test_benchmark.py`.
