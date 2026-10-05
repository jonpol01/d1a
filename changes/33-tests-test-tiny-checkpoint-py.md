### Added

- `tests/test_tiny_checkpoint.py` (in CI): a random 6-layer Gemma 4, with sliding and KV-shared layers, goes through
  `d1a.train`, `d1a.checkpoint` and `d1a.serve` with no download. Every scoring path (packed, rows, prefix miss and hit,
  serving batch) and the served answers must match an independent reference: each question as a plain causal row
  through transformers. Ten mutation checks must fail: off-by-one readouts, a leaky question mask, the sliding window
  ignored, positions that do not restart, a dropped `<bos>`, temperature ignored, and three misread answer types (#33).
