### Added

- `D1A_LOG_REWRITE`: a private rules file the server applies to the state, questions and answers of every decision the
  self-learning log records, so names the log must not keep are written in general wording (ids, the run and outcome
  groups stay, so outcomes still match). `python -m d1a.learning.feedback reword` rewrites an existing log the same way,
  and `trained --reworded` hashes training states as a reworded log records them.
