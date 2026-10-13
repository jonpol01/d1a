### Added

- MLX conformance (#64): `tests/test_conformance_mlx.py` runs the committed golden vectors (tests/golden/tiny-gemma4)
  through the MLX backend, as a LoRA checkpoint and as an export folder (ids exact, probabilities within 1e-5) and as an
  8-bit export (ids exact, no flip on a question the reference decides). CI runs it on Apple Silicon.

### Fixed

- The MLX backend and `export_mlx` load a checkpoint whose base is a local directory: they took it for a Hub id
  (`<dir>@<revision>`) and failed. Hub bases load as before.
