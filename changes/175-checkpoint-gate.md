### Added

- `scripts/quality_gate.py --head-run`: gates a new checkpoint against the one it replaces (#175).
  - The head server and the head suites load the new weights; the base keeps `--run`.
  - Changed answers are listed for review instead of failing, since a new checkpoint is meant to change some.
  - Still failures: lower accuracy or worse calibration on a suite, latency above the noise floor, or a request one
    side does not answer or answers with other options.
