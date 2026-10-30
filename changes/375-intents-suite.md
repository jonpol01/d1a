### Added

- `evals/d1a/intents` (private dataset `JohnP1/d1a-intents`): English and Japanese intents (MASSIVE 1.1 by Amazon, CC BY
  4.0) and dialogue acts, the data D1A-E2B-it QAT v0.7 trained on (#372). Two train partitions (massive-train,
  acts-train) and three evaluation partitions (massive-in, massive-heldout with two scenarios held out, acts-eval).
  `train_sources()` now lists the train partitions, so every later fine-tune replays them and every gate scores the
  evaluation partitions; `recipes/pr-labeler/mix.py` replays them (`--replay-intents`, 300 each).
