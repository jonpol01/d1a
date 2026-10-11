### Removed

- The flat module paths of D1A 0.3 (`d1a.serve`, `d1a.feedback`, `d1a.train`, and the 24 others; `python -m d1a.<old>`),
  shims since 0.4: import and run `d1a.serving.serve`, `d1a.learning.feedback`, `d1a.training.train`, ... instead.
  `d1a/_layout.py` keeps the old-to-new map for the provenance scripts. No caller of ours used an old path.

### Changed

- A new run is saved without `head.pt` (`d1a_config.json` and `head.safetensors` only), as 0.4 announced. A run from
  before 0.4 keeps its `head.pt`, rewritten with the new metadata (a recalibration, say), so its two copies never
  disagree; reading `head.pt` stays.
