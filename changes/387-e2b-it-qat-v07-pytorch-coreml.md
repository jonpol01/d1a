### Changed

- D1A-E2B-it QAT v0.7 is on PyTorch (`JohnP1/d1a-e2b-it-qat`) and Core ML (`JohnP1/d1a-e2b-it-qat-coreml`) since 2026-11-10,
  with the MLX build's adapter and head byte for byte (#372). README, docs/MODELS.md and docs/APPLE.md (EN, JA, KO) name v0.7
  and its checks. The PyTorch build carries the routing temperature 1.59 (#385). Each card states the checks it missed: four
  of eleven on PyTorch, one of seven on Core ML (photos, voice and video against the MLX build; #386). Tag `v0.6` keeps the
  first version of both.
