### Added

- `python -m d1a.training.calibrate --judge rows.json [--guard rows.json] [--confirm rows.json] [--locked rows.json]` (#189):
  a refitted temperature replaces the checkpoint's current one only when it passes Kev's round 28 rule and confirmation. On
  the judge rows pooled, its Brier score must be lower with a 95% interval below zero and its ECE lower, and no judge or
  guard panel's ECE may rise by more than 0.005; then, scored only once that passes, ECE must fall on every `--confirm` file,
  and on every `--locked` file Brier may rise by at most 0.005 with accuracy unchanged. Otherwise nothing is written and the
  run names each criterion that failed. Before anything is fitted, the run refuses a file in another role that shares a row
  with the rows the fit reads (rows `--exclude_rows` drops do not count), and a confirmation file that shares a row with a
  rule file; a whole or partial copy shares its rows, and another read of the same suite partition counts as the same file.
  In every role, `--rows` included, it also refuses a selection with no scored rows and a `path:source,...` naming a source its suite does not
  list, and `--temperature` refuses these options instead of ignoring them. The rule's bootstrap is Kev's registered paired
  read: seed 0 whatever `--seed` is, and the rows in (id, question) order with every tie broken (devtools-v1 repeats ids
  across groups), so reordering a rows file never moves a verdict.
  Every fit now records the 90% bootstrap interval of its temperature in `temperature_fit.interval`.

### Fixed

- `d1a.training.calibrate`'s cross-validation report no longer merges records of two D1A partitions that share a line
  number (`custom/<n>`) into one fold and resampling unit. The fitted temperature never changed; the recorded
  cross-validation numbers of a pool with two or more rows files can.
