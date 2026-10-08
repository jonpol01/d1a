### Added

- Per-use-case temperatures (#209). A System One request may name its use case (`"use_case": "routing"`). When the
  checkpoint carries a temperature for it, that request's probabilities are read at it, on torch and MLX alike, each
  request of a batch at its own. Without `use_case`, or with a name the checkpoint has none for, answers are exactly as
  before. The top answer never moves. `python -m d1a.training.calibrate --use-case routing` fits or writes the temperature
  into the checkpoint's `use_case_temperatures`, leaving its own temperature alone. With `--judge`, a use-case refit is judged
  against the incumbent's temperature for that use case (its entry, else its temperature): `--incumbent`'s, a fine-tune's
  `--init_from` until the fine-tune has an entry of its own, or the run's own. `GET /v1/models` lists the map, and
  the decision log records each request's use case and temperature. The agent presets (`USE_CASES`), the MCP tools,
  `D1A.decide(..., use_case=...)` and both clients send it. `use_case` is at most 64 characters (longer is a 422), and a
  use-case name in the map or in `--use-case` is stored without surrounding whitespace and refused beyond 64 characters; a
  request's own `use_case` is read stripped too. For v0.5, routing at T 0.85 instead of 1.78 lowers held-out
  factory-routing ECE from 0.132 to 0.030 with the same accuracy.
- The agent presets' advice follows the temperature an answer was read at: `advise(..., temperatures=, use_case=)` (the
  checkpoint's temperatures from `D1A.temperatures` or `/v1/models`, which the MCP tools read on each call) carries
  `fail_up`'s small threshold through the temperature ratio, 0.7 at v0.5's T 1.78 becoming 0.855 at the routing T 0.85, so
  routing does not send more cards too low (3 of 108 held-out factory cards, not 5). The other thresholds stay as they are.
- The outcome calibrator keeps temperatures apart: `d1a.learning.feedback` fits, promotes (`"question@T"` in its report) and
  serves each temperature's entries from the decisions read at it only. A calibrator file written before keeps working at
  the checkpoint's own temperature and is never applied to answers read at a use case's.

### Fixed

- MLX exports keep where their temperature came from (#196): `scripts/export_mlx.py` now copies `temperature_fit`, and
  the use-case temperatures with their fits, into the export's `d1a_config.json`.
