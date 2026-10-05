### Changed

- The package is split by job (#60): `d1a.core` (the System One schema), `d1a.backends` (the torch and MLX models,
  checkpoints), `d1a.serving` (the server, media, in-process use), `d1a.learning` (learning from outcomes),
  `d1a.training` (training, data, recipes, calibration), `d1a.eval` (benchmarks, metrics, suites) and `d1a.agents`
  (presets, MCP). Commands move with them: `python -m d1a.serving.serve`, `python -m d1a.training.train`,
  `python -m d1a.eval.benchmark`, `python -m d1a.learning.feedback`, and so on. The code is unchanged.

### Deprecated

- The old flat module paths (`d1a.serve`, `d1a.model`, `python -m d1a.train`, ...) still work in D1A 0.4, as the very
  same modules, with a DeprecationWarning; D1A 0.5 removes them. `d1a/_layout.py` lists where each one went.
