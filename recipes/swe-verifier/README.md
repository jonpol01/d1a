# SWE verifier: should a coding agent submit?

D1A reads an issue, a coding agent's final patch and the last steps of its run, and answers one yes/no question,
"does this patch resolve the issue?", with a calibrated probability, in one forward pass. An agent cannot see the hidden
tests, so it has to guess when it is done; a calibrated P(resolved) lets it choose between submitting, retrying and
abstaining. This recipe is the study for the Gemma 4 Developer Agent paper track (Kaggle,
`gemma-4-developer-agent-paper`, deadline 2026-11-13 08:59 JST).

| Step | Script | Cost |
|---|---|---|
| Zero-shot D1A against trajectory heuristics | `zero_shot.py` | $0 (MLX on a Mac) |
| Training records, split by repository | `to_records.py` | $0 |
| LoRA on D1A-E4B (local first) | `python -m d1a.train --data runs/swe-verifier/data/train.jsonl ...` | $0 local; a paid job only ≤ $5 after asking |

## Data

[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (CC-BY-4.0, Nebius):
SWE-agent runs on SWE-bench-style tasks with `target` (resolved by the hidden tests), the trajectory and the generated
patch. The scripts never read `eval_logs`, which holds the test results. Competition data (tasks, snapshots,
reference patches) is not used here and never leaves the competition's private folders.

Splits are by repository (`zero_shot.split_of`: a hash of the repository name; 20% test, 10% dev, 70% train), so the
test numbers are on repositories the model never saw, as the scored test set is. Each issue keeps at most four runs,
successes first.

## Outputs (local, git-ignored)

```
runs/swe-verifier/
  data/        train.jsonl dev.jsonl test.jsonl     (to_records.py)
  baselines/   heuristics-*.jsonl                   (zero_shot.py --no-d1a)
  zero-shot/   <model>@<tag>-*.jsonl, .log          (zero_shot.py)
  train/       <run name>/                          (d1a.train; adapter and logs; intermediate checkpoints deleted)
```

Every result records the `repo@tag` it ran with (`d1a.versions.latest`). Nothing from this study is written to the
released model repositories (`JohnP1/d1a-e4b`, `-e2b` and their MLX builds) or their tags; a checkpoint that has to be
on the Hub goes to its own private repository first.
