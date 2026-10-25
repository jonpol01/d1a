### Added

- `scripts/export_mlx.py --q-embedding-group-size` sets the embeddings' own group size (#349). QAT fixes the Linears' groups
  of 32, not the embeddings'; 64 saves 0.17 GB on D1A-E2B-it.

### Fixed

- An MLX export's `config.json` keeps the base's `vision_config` (#349). mlx-lm's save_config deletes it, so a
  vision-language loader failed on an export even though its `media/` held the encoders.
