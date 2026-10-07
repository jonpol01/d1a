### Added

- `scripts/quality_gate.py --all-suites` scores every suite: the five card suites and every evaluation partition of
  every D1A suite (pr-labels, routing, ja-jglue, night2, external; composites once through their parts), on top of
  every demo and the labeler replay. AGENTS.md makes it the gate for any model or checkpoint change, decided on the
  whole scorecard.

### Fixed

- The quality gate scores D1A partitions at the serving context. At the training context it skipped every record
  longer than that, 770 of the 953 PRs in `pr-labels:test`, and reported the short ones only.
