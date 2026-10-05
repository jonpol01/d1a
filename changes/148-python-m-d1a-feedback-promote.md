### Added

- `python -m d1a.feedback promote <log> --calibrator <file>` (#148): the promotion gate on a live decision log. Whole
  groups (an outcome's optional `group`, e.g. a pull request; `POST /v1/feedback` takes it) go to the fit or the
  held-out side. A calibrator fitted on the fit side replaces the served one for a question only if it passes the gate
  on the held-out side; the file is written whole and renamed into place, so `d1a.serve` reloads it safely.
