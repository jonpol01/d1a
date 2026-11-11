### Fixed

- MLX exports read the pad id's per-layer embedding row exactly at photo, video and voice positions (#394, #386).
  Gemma 4 looks that one row up hundreds of times per media request, and a 4-bit table's error in it added up: on
  D1A-E2B-it QAT v0.7 MLX the 28 media answers move from up to 0.140 off the fp32 model to 0.085, with 0 flipped
  answers (2 before). Text answers do not change. An export now writes `media_pad_row.safetensors`, an older export
  gets one with `scripts/media_pad_row.py`, and `GET /v1/models` reports `media_pad_row`.
