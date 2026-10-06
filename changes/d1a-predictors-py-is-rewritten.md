### Changed

- `d1a/eval/predictors.py` is rewritten in D1A's own code, no longer derived from Kev. `scripts/equivalence/predictors.py`
  found it identical to the previous version on 1,294 comparisons, latency aside:
  - the local predictor on the committed tiny Gemma 4 checkpoint (long rows, the CUDA kernel policy, the hybrid branch,
    context overflows);
  - the remote predictor (requests, retries, waits, failures);
  - rotation averaging, with and without logits.
