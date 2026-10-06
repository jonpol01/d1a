### Added

- `scripts/compare_checkpoints.py`: a training run's verdict table from its benchmark folders (#175).
  - Per suite: each checkpoint's accuracy, and its difference from a reference with a paired 95% bootstrap interval that
    resamples whole records.
  - ECE at each checkpoint's fitted temperature, from the rows' logits.
  - A pre-registered bar (`--bar SUITE=MIN`, `--floor`): exit 1 when a run misses it.
