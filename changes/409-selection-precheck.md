### Added

- `python -m d1a.eval.selection precheck|report` (#409): for a use case where D1A chooses which of a day's candidate
  items belong in its report, one yes/no decision per candidate (question id `include`), checked against a
  leave-one-day-out rule on |z| and type, and against |z| alone. Per-day AUC, paired, with a day-resampled 95% interval;
  it passes on at least 14 teacher days with the lower bound above 0. `report` adds precision, recall and F1 at three
  thresholds, the teacher's agreement with its own second draft, and the candidate generator's misses.
