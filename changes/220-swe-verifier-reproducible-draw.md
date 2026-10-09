### Added

- `recipes/swe-verifier` analysis tools (#220), each run on saved `zero_shot.py` outputs, CPU only and with no model:
  `recompute_gate.py` (the promotion gate between two scored files, with Platt scaling and with D1A's
  `OutcomeCalibrator`, out of fold by repository), `refit_heuristics.py` (the heuristics' regression refitted on dev, on
  train and on both, scored on the same test rows), `token_lengths.py` (the token length of every scored input, after
  checking the rebuilt selection against the saved file run for run) and `latency.py` (seconds per run). `best_of_k.py
  --pair A:B` adds the paired difference between two scorers without moving its other numbers.
- `recipes/swe-verifier/reproduce.sh` (#220) downloads the study's scored rows and r1 adapter
  (`JohnP1/d1a-swe-verifier-eval`, `JohnP1/d1a-swe-verifier-r1`; private until the release, `REVISION` pins them),
  checks them against their manifest, reruns the scripts that wrote each output and compares every one byte for byte.

### Fixed

- `recipes/swe-verifier/zero_shot.py --per-issue` draws each issue's runs with `rng.sample` again, as the dev and test
  sets were scored (#220), so they rebuild run_key for run_key; `--mixed-only` keeps its draw. The draw is
  `zero_shot.per_issue_draw`, and the recipe reads `nebius/SWE-agent-trajectories` at a pinned commit
  (`zero_shot.REVISION`, `--revision` to override).
