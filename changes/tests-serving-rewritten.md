### Changed

- The tests of the server's batching are rewritten as `tests/test_serving.py`, no longer derived from Kev. They cover the
  state-prefix cache, the retry after running out of device memory, the permute endpoint and the API key, plus the pass
  sizing they rely on (`rows_per_pass`, the CUDA-graph buckets and length groups, the cached-state copies). The cache is
  now checked against its rule over random batches: a batch keeps exactly what it would leave, and the cache holds the
  most recently used states that fit.
