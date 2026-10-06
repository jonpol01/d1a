# Recipes

How D1A's published checkpoints were trained, as runnable files: `d1a-e2b.yaml` (a `d1a.training.recipe` file) and the
PR labeler's data and job scripts (`pr-labeler/`). `swe-verifier/` is documented in its own README.

## Fine-tunes keep the skills (#167)

A fine-tune is any run that starts from a trained checkpoint (`--init_from`, or a recipe stage after the first). Two rules,
from #167: D1A-E4B's PR-labeler fine-tunes cost up to 4 points on hard-v1, devtools-v1 and documents-v1, because their
mixes replayed none of them.

1. **Replay the skills.** The fine-tune's training mix replays the train partitions of `evals/hard-v1`,
   `evals/devtools-v1` and `evals/documents-v1` beside its own data and decision-v7. `recipes/pr-labeler/mix.py` does
   it by default (`--replay-skills 500` per suite); `--replay-skills 0` only rebuilds the v0.3 and v0.4 mixes.
2. **Gate on the card suites.** Before the new checkpoint replaces the one it started from, `scripts/quality_gate.py
   --card-suites` scores decision-v7, transfer-v4, hard-v1, devtools-v1 and documents-v1 on both, and the PR shows the
   table. A drop beyond run-to-run noise on any of them blocks it, like any other downgrade (AGENTS.md, Quality bar).
