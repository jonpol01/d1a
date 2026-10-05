# Upstream: Kev

D1A is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer, Copyright 2026 Jared Palmer, licensed under
the Apache License 2.0. This file records where D1A's code came from and what D1A changed, as Apache-2.0 §4 asks.

| | |
|---|---|
| Upstream repository | https://github.com/jaredpalmer/kev |
| Upstream base commit | `0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91` (`0fe8fc9`, jaredpalmer/kev `main`) |
| Fork commit the port starts from | `13374b3` (https://github.com/jonpol01/kev `main`: `0fe8fc9` + Gemma 4 support) |
| Import | merged with full history into this repository (the merge commit keeps every Kev commit and author) |

## What D1A Changed

- **Ported from later upstream Kev**: the suites mirror pin (`d1a/suite.py` `SUITES_REVISION` = kev-suites `cc4bac8`) from Kev commit `6b1da9d` (#198), which publishes the `hard-v1` and `documents-v1` training partitions. Every other partition is unchanged at that revision.

- **Gemma 4 support** (from the jonpol01/kev fork, by John Soliva): Gemma 4 E2B / E4B bases in `d1a/model.py` and
  `d1a/train.py` (Gemma's reserved `<unused0>`–`<unused4>` tokens as delimiters with a leading `<bos>`, a packed mask for
  the sliding-window layers, text-only loading, re-admission of suite records under Gemma's tokenizer) with tests in
  `tests/test_unit.py` and `tests/test_model.py`.
- **Training robustness**: `d1a/train.py` writes resume points for LoRA runs (`--save_every_steps`,
  `--save_every_minutes`, `--resume`; Kev had them for full-weight runs only; now D1A's own `d1a/resume.py`), and a non-finite loss or gradient skips
  its micro-batch or step (up to `MAX_NONFINITE` in a row) instead of ending the run.
- **MLX backend for Gemma 4 and MLX exports**: `d1a/mlx_model.py` runs Gemma 4 bases on Apple Silicon (row form on
  replicated plain and sliding-window caches, KV-shared layers) besides Kev's Qwen3.5 path, and writes merged, optionally
  quantized export folders (`scripts/export_mlx.py`) that `d1a/checkpoint.py` loads from `d1a_config.json` and
  `d1a/serve.py` serves; `backend="auto"` now picks MLX for Gemma 4 on Apple Silicon. Gemma 4's per-layer embeddings are read from the weight
  files per request (`FlashEmbedding`, #71) instead of held in memory.
- **Default base**: `d1a.train` defaults to `google/gemma-4-E2B` at commit `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`.
  Qwen bases, including the hybrid Qwen3.5 code paths, still work unchanged.
- **Renames**: the Python package `kev` is `d1a` (`python -m d1a.serve|train|benchmark|...`, all imports, pyproject
  name); `KEV_*` environment variables are `D1A_*`; the snapshot mirror defaults
  to `JohnP1/d1a-snapshots`.
- **Model names**: the server lists `d1a-latest`, and keeps `kev-latest` (what clients written against Kev send) and
  `jev-latest` (the TypeSafe SDK default) as accepted names; requests, the benchmark's remote mode and the clients
  default to `d1a-latest`.
- **Clients**: D1A's dependency-free clients live in `clients/python` (`d1a-client`, import `d1a_client`) and
  `clients/js` (`d1a-client`), so the name `d1a` belongs to the main package.
- **Removed** (Kev's research record and its own deployments, not part of D1A; all still in the history): `runs/`,
  `experiments/` (except `smoke.json`, the Modal smoke plan, and `sft-v1-lengths.json`, the SFT corpus token counts),
  `PLAN.md`, `space/` (the Kev Hugging Face Space), `.agents/`, `skills-lock.json`, `skills/` (kev-deploy, kev-finetune),
  Kev's `AGENTS.md` and `docs/autoresearch.md`, `docs/claims.json` + `scripts/verify_claims.py`, Kev's model cards and
  figures, and the release and round read-out scripts (`release_confirm`, `release_numbers`, `round4_deltas`,
  `compare_night2`, `compare_q35`, `plot_family`, `plot_tweet`, `jevbench_paired`, `modal_retry_probe`, `publish_space.sh`).
  Tests bound to them went too: `tests/test_rounds.py`, `tests/test_skill_scripts.py`, two registration checks in
  `tests/test_research.py` and the claims check in `tests/test_conventions.py`.
- **Removed later** (Kev's research tooling that D1A never runs): `modal_app.py`, `d1a/anchors.py`, `autoresearch.py`,
  `compare.py`, `jev.py`, `plot.py`, `publish.py`, `study_v3.py`, and the round, plot and audit scripts
  (`base_mmlu_probe`, `breadth_reads`, `breadth_report`, `calibration_audit`, `chartstyle`, `checkpoint_average`,
  `compare_typesafe`, `longdoc_report`, `longstate_report`, `plot_calibration_audit`, `private_rows`, `reliability_head`,
  `review_calibration_screen`, `screen_longdoc_v1`, `screen_overlap`, `serving_bench`, `sft_probe`,
  `temperature_groups`), with the tests that only exercised them. Then Kev's study harness, `d1a/rounds.py`,
  `experiment.py`, `evaluate.py`, `budget.py` and `mirror.py`, with their tests: calibration became D1A's own
  `d1a/calibrate.py` (same fitted temperature and the same in-distribution refusals as the script it replaces; the
  full-weight snapshot cap moved to `d1a/full_ft.py`). Then Kev's Next.js `playground/` and the `tools/review`
  label-review page, with `JevPredictor` in `d1a/predictors.py` (it ran the playground's `jev-evaluate.mjs`) and its tests;
  the demo app is now [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground), and
  [removed-tools.md](removed-tools.md) records what both tools did and how to restore them. Then `d1a.train`'s research
  options that no D1A checkpoint used: label smoothing, the Brier and focal terms, the ordinal RPS, the permutation KL,
  anchoring, option isolation (also in `d1a.model`, `d1a.serve`, `d1a.mlx_model`; such checkpoints are refused at load)
  and delimiter-embedding training, with their tests; removed-tools.md describes each. Then full-weight training:
  `d1a/full_ft.py` (MasterAdamW, FSDP2 across GPUs under torchrun, snapshots), `--full_ft` and the snapshot options,
  loading full-weight checkpoints (now refused), and the two scripts that only served them, `interpolate_checkpoint.py`
  and `merge_lora_checkpoint.py`, with their tests. Training is single-process LoRA; resume points became D1A's own
  `d1a/resume.py`. Then Kev's training-history suites (`decision-v1`, `public-pool-v5`/`v6`, `round3`-`round15`, `sft-v1`, `sft-v2` and
  its `r21`-`r26`, `transfer-v1`, `v3`, `v5`, `v6`, `v8`) and the builders that only wrote them or the two below
  (`build_long_states`, `build_soft_targets`, `build_night2_data`, `freeze_calibration_audit`, `freeze_semif`,
  `freeze_semif_external`). `external` (semif-v1, typesafe-v1, wanli-v1, wanli-v2) and `night2` moved, byte for byte,
  into D1A suites (`evals/d1a/external`, `evals/d1a/night2`, data in the private dataset `JohnP1/d1a-evals`). Then the
  builders of the suites D1A keeps frozen (`build_hard_v1` with `hard_v1_common`/`_families`/`_numeric`/`_policy`,
  `build_breadth_v1`, `build_devtools_v1` with `devtools_v1_licences.json`, `build_longdoc_v1` with
  `longdoc_v1_synthetic`, `build_documents_v1`/`_v2`, `freeze_documents_v1`, `label_documents_v1`,
  `build_binding_diagnostic`, and `d1a/transfer_v9.py`) with the tests that only exercised them; the suites stay as data
  pinned by sha256 (`tests/test_frozen_suites.py`), and [removed-tools.md](removed-tools.md) says where to rebuild them.
  Then `d1a.suite`'s own suite freezer (`python -m d1a.suite`) with `d1a/contrastive.py` and the record generator of
  `d1a/composition.py` (its rule shapes stay for `d1a.suite.validate_training`; `paired_flip` moved to `d1a/benchmark.py`),
  and `tests/test_generators.py`. Then the `d1a/data.py` converters only that freezer read (13 sources; `build()` keeps
  its six defaults) and `scripts/longdoc_serving.py` (CUDA long-document serving cost, run through the removed Modal app).
- **Rewritten from scratch** (no longer derived): `d1a/calibrate.py`, `scripts/calibrate_checkpoint.py` (now a wrapper
  around it).
- **Rewritten**: `README.md`, `AGENTS.md`, `NOTICE`, `.gitignore`.
- **Kept as is**: the frozen suites in `evals/` (large partitions still download from Kev's Hub dataset
  `jaredpalmer/kev-suites`), the suite builders in `scripts/`.

## Derived files

Files taken from Kev and modified. Each carries the notice "Modified from Kev (...)" and a one-line summary of its
changes at the top; `scripts/check_license.py` checks that this list and the headers match exactly.

```text
.github/workflows/ci.yml
.gitignore
d1a/api.py
d1a/benchmark.py
d1a/checkpoint.py
d1a/composition.py
d1a/cuda_graphs.py
d1a/data.py
d1a/fused_qwen35.py
d1a/metrics.py
d1a/mlx_model.py
d1a/model.py
d1a/predictors.py
d1a/serve.py
d1a/shared_prefix.py
d1a/suite.py
d1a/train.py
pyproject.toml
scripts/mlx_parity.py
tests/test_api.py
tests/test_conventions.py
tests/test_mlx.py
tests/test_model.py
tests/test_research.py
tests/test_unit.py
```

## Derived files without a header

Files taken from Kev and modified that cannot hold a comment (JSON, generated lockfiles).

```text
scripts/golden_presets.json
uv.lock
```

## Unmodified copies

Files carried over from Kev byte for byte (an entry ending in `/` covers a directory). The check fails if one of them
changes without moving to "Derived files".

```text
evals/
.gitattributes
.python-version
LICENSE
d1a/device.py
experiments/sft-v1-lengths.json
experiments/smoke.json
```
