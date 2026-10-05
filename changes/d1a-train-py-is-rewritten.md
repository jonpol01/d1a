### Changed

- `d1a/train.py` is rewritten in D1A's own code, no longer derived from Kev. The run, its options, what it writes and how it
  resumes are unchanged: on the CPU the adapter, the head, the configs and the logs are identical to the previous version
  under 15 option sets, a stopped and resumed run included (`scripts/equivalence/train.py`, new). Its tests are rewritten
  too, as `tests/test_train.py`, with a check that a seed reproduces a run bit for bit.
