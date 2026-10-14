### Added

- `scripts/export_mlx.py --q4-0` writes 4-bit weights in groups of 32 on llama.cpp's Q4_0 grid (scale = d, bias = −8·d;
  `d1a.backends.mlx.q4_0`, `q4_0_mlx`) instead of mlx-lm's min/max scales, for a base trained for Q4_0 such as
  Gemma 4 12B-it QAT (#270). On that checkpoint Q4_0's weight error is about 20% below mlx-lm's 4-bit group 32 and 30%
  below group 64. The export's `d1a_config.json` records `"grid": "q4_0"`.
