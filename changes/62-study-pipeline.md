### Added

- `python -m d1a.training.study` runs a study end to end (#62). It trains a recipe, calibrates the checkpoint on
  held-out rows (`--calibrate`, fitted by `d1a.training.calibrate`, which refuses rows the checkpoint trained on), scores
  each `--evaluate` at that temperature, and writes `study/study.json` and `study.md`. `--publish owner/repo:prefix` (or
  the `publish` command) uploads the checkpoint and the report, and refuses a checkpoint whose temperature was never
  fitted, or was fitted in distribution without `--allow-in-distribution`. A study that stopped continues where it
  stopped; an evaluation scored at another temperature is scored again.
