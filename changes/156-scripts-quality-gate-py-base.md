### Added

- `scripts/quality_gate.py --base <ref> [--head <ref>]` (#156): the quality gate for model, loading and serving changes.
  Two `d1a.serve` instances, one per version, on the same real weights answer every demo example (regenerated from a
  d1a-playground checkout with `--playground`) and the labeler replay, alternating which goes first. A base/base run
  measures the noise floor. It fails on a changed choice, a probability moving more than `--tol` (1e-6), a request only
  one side answers, latency above the floor, or, with `--suites`, a frozen suite whose accuracy or calibration gets
  worse; exit 1 on FAIL. AGENTS.md makes it the required step for model and serving PRs and playground pin moves.
