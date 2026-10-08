### Added

- Per-use-case temperatures (#209). A System One request may name its use case (`"use_case": "routing"`). When the
  checkpoint carries a temperature for it, that request's probabilities are read at it, on torch and MLX alike, each
  request of a batch at its own. Without `use_case`, or with a name the checkpoint has none for, answers are exactly as
  before. The top answer never moves. `python -m d1a.training.calibrate --use-case routing` fits or writes the temperature
  into the checkpoint's `use_case_temperatures`, leaving its own temperature alone. `GET /v1/models` lists the map, and
  the decision log records each request's use case and temperature. The agent presets (`USE_CASES`), the MCP tools,
  `D1A.decide(..., use_case=...)` and both clients send it. For v0.5, routing at T 0.85 instead of 1.78 lowers held-out
  factory-routing ECE from 0.132 to 0.030 with the same accuracy.

### Fixed

- MLX exports keep where their temperature came from (#196): `scripts/export_mlx.py` now copies `temperature_fit`, and
  the use-case temperatures with their fits, into the export's `d1a_config.json`.
