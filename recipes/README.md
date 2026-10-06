# Recipes

How D1A's published checkpoints were trained, as runnable files: `d1a-e2b.yaml` (a `d1a.training.recipe` file) and the
PR labeler's data and job scripts (`pr-labeler/`), and the skills stage (`skills/`). `swe-verifier/` is documented in its own
README.

## Fine-tunes keep the skills (#167)

A fine-tune is any run that starts from a trained checkpoint (`--init_from`, or a recipe stage after the first). Two rules,
from #167: D1A-E4B's PR-labeler fine-tunes cost up to 4 points on hard-v1, devtools-v1 and documents-v1, because their
mixes replayed none of them.

1. **Replay the skills.** The fine-tune's training mix replays the train partitions of `evals/hard-v1`,
   `evals/devtools-v1` and `evals/documents-v1` beside its own data and decision-v7. `recipes/pr-labeler/mix.py` does
   it by default (`--replay-skills 500` per suite); `--replay-skills 0` only rebuilds the v0.3 and v0.4 mixes.
2. **Gate on the card suites.** Before the new checkpoint replaces the one it started from, `scripts/quality_gate.py
   --run <the served checkpoint> --head-run <the new one> --card-suites` scores decision-v7, transfer-v4, hard-v1,
   devtools-v1 and documents-v1 on both, runs every demo example and the labeler replay on both (changed answers are listed
   for review: a new checkpoint is meant to change some), and the PR shows the table. A drop beyond run-to-run noise on any of them blocks it, like any other downgrade (AGENTS.md, Quality bar).

## The skills stage (`skills/`, #175)

D1A-E4B never had a full skills stage: its first run mixed hard-v1, devtools-v1, documents-v1 and dates into about 1,400
records, where Kev-4B trained hard and devtools for about 1,915 steps (#167, phase A). The #167 B0 pilot (v0.4 + 300 steps,
2,400 records) raised hard-v1 and devtools-v1 by about 5 points each without losing transfer-v4 or decision-v7. This stage
is the full size:
- **Mix** (`mix.py`): every hard-v1 (6,000) and devtools-v1 (5,320) training record, plus replay of documents-v1 1,000,
  JGLUE 650 and PR labels 1,350. With decision-v7 replay 1,000 added by the trainer, that is 15,320 records, 1,915 steps
  of 8.
- **Job** (`train_job.sh`, `launch_hf_job.sh`): one HF L4, lr 2e-5, LoRA 16, bf16, 1 record per pass and 8 passes per step
  (2 per pass ran out of memory on the longest states), max_state 6,400 (the mix's longest
  state is 6,209 tokens). It runs 5 steps on the 40 longest states first, then trains, and calibrates the way v0.4 was.
  Resume points go to the HF bucket. Outputs go to the private run repo.
- **Scoring is separate**, on the same path as the checkpoints it is compared with (the five card suites plus pr-labels
  test and test-ja), then the gates above before it replaces anything.

    git push && ./skills/launch_hf_job.sh skills-v1 8h      # from the B0 skills checkpoint; INIT=JohnP1/d1a-e4b@v0.4 for v0.4
