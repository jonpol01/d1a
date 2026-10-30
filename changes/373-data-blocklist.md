### Added

- `D1A_DATA_BLOCKLIST`: a private file of names D1A's data must never hold. When it is set, `python -m d1a.eval.suites
  freeze` refuses to pin a partition that holds one, loading a suite partition refuses it, and `d1a.training.train`
  refuses `--data` or any training record (replay included) that holds one, before training starts. A refusal names the
  lines, never the entry.

### Changed

- The routing and PR-label suites are re-pinned to dataset revisions reworded in general terms. Labels, ids (but one),
  and record counts are unchanged.
