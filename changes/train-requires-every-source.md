### Changed

- `d1a.training.train` refuses a fine-tune (`--init_from`) whose training data leaves out a training source (#211): every
  `d1a.eval.suites.train_sources()` entry and decision-v7 must be in the `--data` mix (read from its sidecar `<data>.json`,
  as `recipes/skills/mix.py`, `recipes/pr-labeler/mix.py` and the D2 builder write it), replayed with `--suite` and
  `--replay`, or in `--extra_suites`. The check runs before any weights load and names what is missing; `--data` without a
  sidecar is refused for a fine-tune. A source left out on purpose takes `--allow_missing_sources <source,...|all> --reason
  "<why>"`, which the run's log names at its start and `training_config.json` records under `sources`. A sidecar's record
  counts must be whole numbers. `d1a.training.recipe` (and so `d1a.training.study`) runs the same check on every fine-tune
  stage before the first stage trains. Resume points written before this change still resume. A run from the base model
  is unchanged.
