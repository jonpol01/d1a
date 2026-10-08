# Recipes

How D1A's published checkpoints were trained, as runnable files: `d1a-e2b.yaml` (a `d1a.training.recipe` file) and the
PR labeler's data and job scripts (`pr-labeler/`), and the skills stage (`skills/`). `swe-verifier/` is documented in its own
README.

## Fine-tunes keep the skills (#167)

A fine-tune is any run that starts from a trained checkpoint (`--init_from`, or a recipe stage after the first). Two rules,
from #167: D1A-E4B's PR-labeler fine-tunes cost up to 4 points on hard-v1, devtools-v1 and documents-v1, because their
mixes replayed none of them.

1. **Replay every skill.** The fine-tune's mix trains on or replays every training source, beside decision-v7 (which
   the trainer replays): the train partitions of `evals/hard-v1`, `evals/devtools-v1` and `evals/documents-v1`, and every
   D1A train partition: PR labels (train, train-ja, train-blast), routing (factory-train, generic-train) and JGLUE.
   `d1a.eval.suites.train_sources()` lists them, and both mix tools refuse a new mix that leaves one out.
   - **The trainer enforces it too (#211).** With `--init_from`, `d1a.training.train` reads the `--data` mix's sidecar
     (`<data>.json`, as both mix tools and the D2 builder write it), adds `--suite` with `--replay` and `--extra_suites`, and
     refuses the run before any weights load if a training source or decision-v7 is missing, naming each. A hand-built mix
     needs a sidecar with its records by source; without one a fine-tune is refused. A source left out on purpose is named:
     `--allow_missing_sources <source,...|all> --reason "<why>"`, and both are recorded in the run's `training_config.json`
     (`sources`), beside what it covered.
   - `recipes/pr-labeler/mix.py` replays JGLUE 500, routing 300 and each skills suite 500 by default, and takes the
     train-ja and train-blast PRs; `--replay-skills 0` only rebuilds the v0.3 and v0.4 mixes.
   - `recipes/skills/mix.py` adds 300 records of each source its plan leaves out (`--replay-rest`); `--recorded` only
     rebuilds v0.5's mix.
   - v0.5 shows why. Its skills stage replayed no routing and only 1,350 PR records: routing's confidence drifted and the
     labeler's severity slipped (77.8 → 75.6 on the 953 test PRs, #187).
2. **Gate on the card suites.** Before the new checkpoint replaces the one it started from, `scripts/quality_gate.py
   --run <the served checkpoint> --head-run <the new one> --card-suites` scores decision-v7, transfer-v4, hard-v1,
   devtools-v1 and documents-v1 on both, runs every demo example and the labeler replay on both (changed answers are listed
   for review: a new checkpoint is meant to change some), and the PR shows the table. A drop beyond run-to-run noise on any of them blocks it, like any other downgrade (AGENTS.md, Quality bar).
   - With `--all-suites` (the rule for any checkpoint), `scripts/decide.py` turns the gate into one verdict (#202):
     `python scripts/decide.py --gate <gate> --labeler <dump> --base-name v0.5 --head-name <new> --human <owner labels>`.
     - **INCOMPLETE** comes first. It means a suite, a demo group, the replay, the latency or a labeler side is missing or
       cannot be scored. Rerun the gate; it is never a pass.
     - **VETO** comes next:
       - V1: any card suite with Δ < −2 points or its CI lower bound < −4;
       - V3: latency above the floor + 0.015;
       - V4: any suite of 150 or more questions down 5 points or more, with its CI below 0.
     - **BETTER** needs a pooled Δ ≥ 0 over every suite and the two live labeler lines, and either:
       - a pooled lower bound above 0;
       - or 3 significant wins on distinct sources and no significant loss. A `-ja` twin and `pr-labels_test:sev+offsets`
         count as their original's source.
     - **NOT BETTER** covers the rest. A near miss (pooled Δ > 0, 2 wins, no loss) earns one rerun with a new seed.
     - V2, a person reading the safety demos' flips, comes before any deploy. Every significant loss becomes a follow-up.

## The skills stage (`skills/`, #175)

D1A-E4B never had a full skills stage: its first run mixed hard-v1, devtools-v1, documents-v1 and dates into about 1,400
records, where Kev-4B trained hard and devtools for about 1,915 steps (#167, phase A). The #167 B0 pilot (v0.4 + 300 steps,
2,400 records) raised hard-v1 and devtools-v1 by about 5 points each without losing transfer-v4 or decision-v7. This stage
is the full size:
- **Mix** (`mix.py --recorded`, v0.5's): every hard-v1 (6,000) and devtools-v1 (5,320) training record, plus replay of
  documents-v1 1,000, JGLUE 650 and PR labels 1,350. With decision-v7 replay 1,000 added by the trainer, that is 15,320
  records, 1,915 steps of 8. A new stage (`mix.py` without `--recorded`) also replays 300 of each other training source.
- **Job** (`train_job.sh`, `launch_hf_job.sh`): one HF L4, lr 2e-5, LoRA 16, bf16, 1 record per pass and 8 passes per step
  (2 per pass ran out of memory on the longest states), max_state 6,400 (the mix's longest
  state is 6,209 tokens). It runs 5 steps on the 40 longest states first, then trains, and calibrates the way v0.4 was.
  Resume points go to the HF bucket. Outputs go to the private run repo.
- **Scoring is separate**, on the same path as the checkpoints it is compared with (the five card suites plus pr-labels
  test and test-ja), then the gates above before it replaces anything.
- **The verdict table:** `scripts/compare_checkpoints.py --ref v0.4=<scores> --run <name>=<scores> --fit <name>=<calibration
  rows> --bar hard-v1=3 --bar devtools-v1=3 --floor -1`. It gives per-suite accuracy with paired, record-resampled 95% intervals
  against the reference, ECE at each checkpoint's fitted temperature, and the bar.

    git push && ./skills/launch_hf_job.sh skills-v1 8h      # from the B0 skills checkpoint; INIT=JohnP1/d1a-e4b@v0.4 for v0.4
