### Added

- `d1a.training.calibrate` warns when a fit lands on an end of its grid (0.25 or 4), where the best temperature may lie
  beyond it, and records it in `temperature_fit.grid_edge` (#196).

### Fixed

- `d1a.training.calibrate`'s held-out guard places D1A suites (#196). Rows scored on `evals/d1a/<suite>:<partition>` are
  matched to their partition by its sha256 and role, and the checkpoint's training is read from its
  `training_config.json` and the sidecar of its `--data` mix (the shapes `d1a.training.train` reads), with `--extra_suites`
  and the sources it recorded covering. A partition the checkpoint trained on is refused; an eval partition (development,
  test) of a suite it trained on is allowed with a printed SAME CORPUS note, recorded in `temperature_fit.same_corpus`.
  A D1A pool no longer needs `--allow-in-distribution`. Frozen suites keep their rule, now also for the frozen suites a mix
  replays.
