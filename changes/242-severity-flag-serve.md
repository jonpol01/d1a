### Added

- The server serves learned flags (#237, part 2): a flag in the `D1A_LEARNING` file's `flags`, fitted for the served run,
  is added inside its question's answer (`flags.<name>`: `on`, `p`, `t`, `space`), computed on the model's own
  probabilities before the outcome calibrator, and kept in the decision log's meta. The answer never changes, and the file
  is re-read when it changes, so a flag is served or withdrawn without a restart.
