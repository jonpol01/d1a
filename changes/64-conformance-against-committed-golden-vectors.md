### Added

- Conformance against committed golden vectors (#64): `tests/golden/tiny-gemma4/` holds a tiny Gemma 4 checkpoint
  (weights in git, 556 KB, built by `tests/golden/build_tiny_gemma4.py`) and `golden.json`, with 40 requests and 154
  questions. `tests/test_conformance.py` requires its token ids exactly and every probability to 1e-5, through the model,
  `d1a.serve` and `scripts/golden_vectors.py compare`.
