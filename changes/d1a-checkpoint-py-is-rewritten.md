### Changed

- `d1a/checkpoint.py` is rewritten in D1A's own code, no longer derived from Kev. Its decisions (backend, dtype, merge),
  refusals, metadata and on-disk formats are unchanged (`scripts/equivalence/checkpoint.py`, new), and `head.pt` is read
  with `weights_only=True` explicitly, so it can hold only tensors and plain data even under
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`. Its tests are rewritten too, as `tests/test_checkpoint.py`.
