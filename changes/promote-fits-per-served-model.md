### Added

- `python -m d1a.learning.feedback promote --run <repo@tag>` (and `status`, `records`, `calibrate`) keeps only the
  decisions that model made, as the decision log records them per decision. A calibrator corrects one model's
  probabilities: after a model switch, fitted on the old model's decisions, it would correct the new model by the old
  one's errors. With `--run`, `min_outcomes` counts the served model's outcomes only, so nothing is promoted until it
  has enough. `FeedbackLog.resolved(run=...)` does the same in code.

### Fixed

- A calibrator file moved aside (`D1A_OUTCOME_CALIBRATOR`) now stops being applied from the next answer, so a rollback
  needs no restart. Before, the server kept applying the last one it had loaded. A path that names no file yet serves
  the model's own answers until promote writes it. Any other error reading the file keeps the last calibrator rather
  than silently dropping it.
