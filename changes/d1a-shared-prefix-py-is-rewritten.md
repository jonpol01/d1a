### Changed

- `d1a/backends/shared_prefix.py` (Qwen3.5 training through a shared state prefix) is rewritten in D1A's own code, no
  longer derived from Kev. `scripts/equivalence/shared_prefix.py` found it identical to the previous version on 200
  comparisons, each a batch's branch hidden states and every parameter's gradient, on a tiny random Qwen3.5: eager and
  SDPA attention, padded and unpadded states, gradient checkpointing on and off, plus the Prefix cache calls.
