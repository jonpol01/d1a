### Changed

- `scripts/quality_gate.py --expect-flips` with `--head-learning` also fails a request a decode rule applies to that did not flip
  (its use case, the run, the base's choice in `from`, p(option) >= t), so the gate shows the rule fired everywhere it should,
  not only that every flip was eligible (#259).
- docs/SELF_LEARNING.md and the PR-labeler recipe: severity's decode rule (#259) is available, not active. On John's repositories a blind
  check found it right on 3 of the 6 states it fires on, against 2 for v0.6's own answer, so it stays unset there (#252).
