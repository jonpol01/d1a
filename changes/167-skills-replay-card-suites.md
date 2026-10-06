### Added

- Fine-tunes keep the skills (#167): `recipes/pr-labeler/mix.py --replay-skills N` (default 500 per suite) replays the
  hard-v1, devtools-v1 and documents-v1 train partitions; `--replay-skills 0` rebuilds the v0.3 and v0.4 mixes byte for
  byte. `scripts/quality_gate.py --card-suites` also scores the five card suites. `recipes/README.md` and AGENTS.md make
  both the rule for any run that starts from a trained checkpoint.

### Changed

- `scripts/quality_gate.py` runs when this environment meets both versions' requirements (dependencies and the server
  extras), instead of requiring them to be declared the same; any difference is recorded in the report.
