### Changed

- Fine-tunes keep every skill: `d1a.eval.suites.train_sources()` lists every training source (hard-v1, devtools-v1,
  documents-v1 and every D1A train partition: PR labels incl. blast and Japanese, routing, JGLUE). Both mix tools
  refuse a new mix that leaves one out. `recipes/skills/mix.py` now replays 300 records of each source its plan leaves
  out (`--replay-rest`; `--recorded` rebuilds v0.5's mix). `recipes/pr-labeler/mix.py` takes the train-blast PRs by
  default. v0.5's skills stage had no routing and only 1,350 PR records, and its labeler severity slipped (#187).
