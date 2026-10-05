### Changed

- `d1a/metrics.py` is rewritten in D1A's own code, no longer derived from Kev. Every figure, report (key order included),
  fitted temperature and bootstrap interval is bit-identical to the previous version: checked on about 300,000 generated
  calls and on saved benchmark rows. Its tests are rewritten too, as `tests/test_metrics.py`.
