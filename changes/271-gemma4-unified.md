### Added

- Gemma 4's unified model (the 12B, `google/gemma-4-12B-it-qat-q4_0-unquantized`) as a base on both backends (#270).
  PyTorch runs it through transformers as it already ran the E-series. MLX loads it through mlx-lm's Gemma 4 classes
  (`d1a.backends.mlx.unified_classes`): the unified checkpoint's vision embedder is dropped on load, and the unified
  model's zero per-layer-embedding and KV-sharing sizes are stated rather than taken from mlx-lm's E-series defaults.
  No D1A-12B is trained yet.
