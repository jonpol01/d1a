# Training

How to train, fine-tune, calibrate and score a D1A checkpoint; the data it trains on is described in [TRAINING_DATA.md](TRAINING_DATA.md).

The D1A recipe on Gemma 4 E2B (one L4 is enough, about 15 GB peak with a bf16 backbone). `google/gemma-4-E2B` at commit `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` is the trainer's default base, so `--base` can be left out:

```bash
uv run python -m d1a.training.train --suite evals/v7/decision-v7 \
    --epochs 2 --lr 1e-4 --batch 4 --accum 2 --dtype bf16 --weights_dtype bf16 --checkpointing 1 \
    --p_none_pair 0.25 --device cuda --out runs/d1a-e2b
```

The same run as a recipe, [`recipes/d1a-e2b.yaml`](../recipes/d1a-e2b.yaml): a versioned YAML file of stages (each one `d1a.training.train` run, a later stage starting from the one before), checked against `d1a.training.train`'s options before anything runs and recorded beside every checkpoint it writes (`recipe.json`: the recipe, its sha256 and the exact command). `--dry-run` prints the commands.

```bash
uv run python -m d1a.training.recipe run recipes/d1a-e2b.yaml --out runs/d1a-e2b --device cuda    # writes runs/d1a-e2b/base
```

- For E4B pass `--base google/gemma-4-E4B --base_revision <sha>`.
- To fine-tune on your own data, start from a checkpoint: `--data mine.jsonl --init_from JohnP1/d1a-e2b`.
- To fit an adapter to a quantized build's base, train on that base's own weights: `--base_ct <folder or Hub id@revision>` takes a compressed-tensors QAT checkpoint of the same model (Google's `gemma-4-*-it-qat-w4a16-ct`) and writes its weights over the base's before training, each pack-quantized Linear as s·q exactly and every other tensor as stored. Use it with `--weights_dtype fp32`, so the int4 grid stays exact. A base parameter the checkpoint does not set stops the run (d1a#341). For a base too big for fp32 in memory (Gemma 4 E4B), add `--embedding_dtype bf16`: the base loads in bf16 and everything but the embedding tables goes to fp32, so the grid stays exact and the forward is the fp32 model's, while the tables (3.5B of E4B's 7.5B text parameters) keep half the memory.
  - The run records the checkpoint as `base_ct` in `d1a_config.json` (a cache snapshot as `org/name@revision`), and everything that loads the run rebuilds that trunk:
    - the PyTorch path writes s·q over the base, which is exact in fp32 (`D1A_DTYPE=fp32`) and rounded in bf16;
    - on Apple Silicon such a run loads on PyTorch; on MLX it runs as its export, which `scripts/export_mlx.py --run <it> --unmerged` builds on the same checkpoint without `--ct-base` (docs/APPLE.md);
    - the PyTorch photo and voice server refuses it, since it runs the base's own trunk; its MLX export answers photos, voice and video through `media/` from the QAT checkpoint (`--media`).
  - A warm start may move between the base's trunk and a QAT one; it refuses two different QAT checkpoints.
- A fine-tune must train on or replay every D1A training source (`d1a.eval.suites.train_sources()` and decision-v7), read from the mix's sidecar `mine.jsonl.json` as `recipes/skills/mix.py` and `recipes/pr-labeler/mix.py` write it, or it is refused before any weights load; to leave sources out on purpose, add `--allow_missing_sources all --reason "<why>"` (or name the sources), which `training_config.json` records (#211, `recipes/README.md`).
- `--data` (in `d1a.eval.benchmark` too) also takes a D1A suite partition, `evals/d1a/<suite>:<partition>`: a dataset pinned to one Hub commit, fetched and checked against its sha256 before use, and refused for training when it is an eval partition (`python -m d1a.eval.suites --help`).
- To train on several frozen suites in one run, add `--extra_suites evals/hard-v1,evals/devtools-v1` (each suite's own manifest rules apply to its records).
- For long runs add `--save_every_minutes 30`: a crash then loses at most 30 minutes, and `--resume 1` with the same arguments continues bit for bit. A single non-finite loss or gradient skips its batch instead of ending the run (three in a row still stop it). `python -m d1a.training.train --help` lists every option.
- Before you publish or serve a trained checkpoint, write its calibration temperature into it with `python -m d1a.training.calibrate` (training leaves it at 1.0 on purpose; `d1a.serving.serve` warns and `/v1/models` reports `calibrated: false` until you do).
- To refit a calibrated checkpoint, add `--judge <rows.json>` (and `--guard <rows.json>` per suite whose ECE may rise by at most 0.005): the new temperature is written only when, on the judge rows pooled, its Brier score is lower with a 95% upper bound below 0 and its ECE is lower, and no judge or guard file's ECE rises by more than 0.005. With `--confirm <test rows.json>` and `--locked <locked test rows.json>`, scored only once that passes, its ECE must also fall on the test rows, and on the locked rows its Brier score may rise by at most 0.005 with accuracy unchanged (Kev's round 28 rule and confirmation). Otherwise nothing is written, and the run names each criterion that failed.
- The rule judges the refit against the incumbent: for a fine-tune not yet calibrated, the temperature its `--init_from` checkpoint serves (training writes every run at 1.0, so its own would be no fair comparison); otherwise the run's own; or what `--incumbent <run|T>` names. The output and the checkpoint's `temperature_fit.rule.incumbent` say which.
- `--use-case routing` fits (or, with `--temperature T --reason ...`, writes) the temperature of one use case instead, into the checkpoint's `use_case_temperatures`; the checkpoint's own temperature is left alone, `--judge` takes the incumbent's temperature for that use case (for a fine-tune, its `--init_from` checkpoint's until it has an entry of its own), and MLX exports carry the map (`python -m d1a.training.calibrate --help` explains why a routing fit needs `--allow-in-distribution`).
- `python -m d1a.training.study run <recipe> --out <dir> --calibrate <held-out suite> --evaluate <suite> ...` does it all in one command: it trains the recipe, fits the temperature on the held-out rows, scores each evaluation at that temperature, and writes `<dir>/study/study.md`. With `--publish owner/repo:prefix` it uploads the checkpoint and the report, and it refuses a checkpoint whose temperature was never fitted. A study that stopped continues where it stopped.

Score a checkpoint on the frozen suites:

```bash
uv run python -m d1a.eval.benchmark --run runs/d1a-e2b --suite evals/v4/transfer-v4 --out runs/d1a-e2b-transfer
```

The frozen suites (`evals/`) come from Kev. Their manifests and small partitions are in git; partitions over about 10 MB are fetched on first use from D1A's private copy of Kev's Hugging Face dataset [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites), or from Kev's dataset itself for an account without access to the copy, and verified by sha256 either way. A few held-out suites name a private mirror and cannot be loaded without access to it. The suites were admitted under Qwen tokenizers, so the trainer re-admits their records under Gemma's tokenizer with each suite's own rule (70 of decision-v7's 12,576 records are dropped).

Gemma's tokenizer has none of the Qwen delimiter tokens, so D1A uses Gemma's reserved `<unused0>`–`<unused4>` tokens and a leading `<bos>`. The sliding-window attention layers get their own copy of the packed mask, and only the text model is loaded.

## Known problems

- On one Windows/WSL2 machine with an NVIDIA GPU, PyTorch's fused SDPA attention kernel corrupted memory during Gemma training. Loading the model with eager attention fixed it. If training crashes or produces NaNs there, try eager attention first.
