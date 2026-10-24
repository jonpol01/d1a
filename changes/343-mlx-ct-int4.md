### Added

- `scripts/export_mlx.py --ct-base` builds on a compressed-tensors QAT checkpoint (Google's gemma-4-*-it-qat-w4a16-ct),
  keeping its int4 Linears exactly as MLX 4-bit (#343); `--q-embedding-bits` sets the embeddings' own width.
