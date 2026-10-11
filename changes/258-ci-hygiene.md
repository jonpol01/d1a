### Added

- Pull-request hygiene in CI, inside the existing required checks: `verified commits` refuses AI attribution in commit
  messages or the pull request body; `package builds and installs` refuses a pull request without its
  `changes/<PR>-<slug>.md` fragment (Dependabot and releases exempt) and any added line with a local machine path.
  `d1a/core/hygiene.py` holds the patterns; `d1a.training.study publish` and `scripts/golden_vectors.py --upload` refuse
  to send files with a local path to the Hugging Face Hub. The Apple Silicon job also runs `tests/test_mlx.py`'s
  weight-free LoRA-merge test, which no CI job ran before.
