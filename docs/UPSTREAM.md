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

- **Ported from later upstream Kev**: the suites mirror pin (`d1a/eval/suite.py` `SUITES_REVISION` = kev-suites `cc4bac8`) from Kev commit `6b1da9d` (#198), which publishes the `hard-v1` and `documents-v1` training partitions. Every other partition is unchanged at that revision. `PrefixCache.make_room` in `d1a/serving/serve.py` from Kev commit `1d77363` (#201): before a batch runs, the cache drops the states that batch's store would evict anyway, so an old long state no longer stays resident through the pass of the new one (#189).
  `d1a/training/temperature_gate.py`, which `d1a.training.calibrate --judge/--guard/--confirm/--locked` runs, ports Kev's
  validation of a refitted temperature: `kev/rounds.py` `pooled_temperature_ci` (Kev commit `e52f812`, #168) as every fit's
  90% bootstrap interval of its temperature; `kev/rounds.py` `paired()`, the registered paired read over
  `kev.metrics.paired_bootstrap` (Kev `1dc2fbc`, #112: micro, 2,000 record-clustered resamples, seed 0, rows in (id, question)
  order), as the Brier interval; and round 28's registered temperature rule and confirmation stages
  (`experiments/rounds/r28.json` and PLAN.md, Kev `ebc8024`, #199). A refitted temperature is written only when, on the judge
  rows pooled, its Brier score is lower with a 95% upper bound below 0 and its ECE is lower, no judge or guard panel's ECE
  rises by more than 0.005, and then, scored only once that passes, ECE falls on every `--confirm` file (the tests stage) and
  Brier rises by at most 0.005 with accuracy identical on every `--locked` file (the locked stage). D1A's changes: panels are
  rows files given per call, not a registered spec; rows are keyed by their file, so two D1A partitions' `custom/<line>` ids
  never share a cluster, and each row is paired with itself at the other temperature, ordered by (id, question) and then by
  file, so the Brier interval equals Kev's `paired()` wherever no source name spans two files (those are resampled as one
  stratum per file); the confirmation stages run in the same call, once the rule passes. Round 29, registered in the same Kev
  commit, is not ported: it selects among 9B checkpoints under round 24's audited rule, on accuracy, with a calibration guard of
  ECE at most the parent's + 0.01 and a locked stage of accuracy at least the parent's − 1 pp and Brier at most + 0.005.

- **Gemma 4 support** (from the jonpol01/kev fork, by John Soliva): Gemma 4 E2B / E4B bases in `d1a/backends/torch.py` and
  `d1a/training/train.py` (Gemma's reserved `<unused0>`–`<unused4>` tokens as delimiters with a leading `<bos>`, a packed mask for
  the sliding-window layers, text-only loading, re-admission of suite records under Gemma's tokenizer) with tests in
  `tests/test_encoding.py`, `tests/test_unit.py` and `tests/test_model.py`.
- **Training robustness**: `d1a/training/train.py` writes resume points for LoRA runs (`--save_every_steps`,
  `--save_every_minutes`, `--resume`; Kev had them for full-weight runs only; now D1A's own `d1a/training/resume.py`), and a non-finite loss or gradient skips
  its micro-batch or step (up to `MAX_NONFINITE` in a row) instead of ending the run; tested in `tests/test_train.py`
  (resumes, a NaN gradient) and `tests/test_unit.py` (a non-finite loss).
- **MLX backend for Gemma 4 and MLX exports**: `d1a/backends/mlx.py` runs Gemma 4 bases on Apple Silicon (row form on
  replicated plain and sliding-window caches, KV-shared layers) besides Kev's Qwen3.5 path, and writes merged, optionally
  quantized export folders (`scripts/export_mlx.py`) that `d1a/backends/checkpoint.py` loads from `d1a_config.json` and
  `d1a/serving/serve.py` serves; `backend="auto"` now picks MLX for Gemma 4 on Apple Silicon. Gemma 4's per-layer embeddings are read from the weight
  files per request (`FlashEmbedding`, #71) instead of held in memory.
- **Default base**: `d1a.training.train` defaults to `google/gemma-4-E2B` at commit `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`.
  Qwen bases, including the hybrid Qwen3.5 code paths, still work unchanged.
- **Renames**: the Python package `kev` is `d1a` (`python -m d1a.serving.serve|train|benchmark|...`, all imports, pyproject
  name); `KEV_*` environment variables are `D1A_*`; the snapshot mirror defaults
  to `JohnP1/d1a-snapshots`.
- **Model names**: the server lists only `d1a-latest` (the old `kev-latest` and `jev-latest` listings were removed) and still
  answers any model name a request sends; requests, the benchmark's remote mode and the clients default to `d1a-latest`.
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
  `d1a/training/calibrate.py` (same fitted temperature and the same in-distribution refusals as the script it replaces; the
  full-weight snapshot cap moved to `d1a/full_ft.py`). Then Kev's Next.js `playground/` and the `tools/review`
  label-review page, with `JevPredictor` in `d1a/eval/predictors.py` (it ran the playground's `jev-evaluate.mjs`) and its tests;
  the demo app is now [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground), and
  [removed-tools.md](removed-tools.md) records what both tools did and how to restore them. Then `d1a.training.train`'s research
  options that no D1A checkpoint used: label smoothing, the Brier and focal terms, the ordinal RPS, the permutation KL,
  anchoring, option isolation (also in `d1a.backends.torch`, `d1a.serving.serve`, `d1a.backends.mlx`; such checkpoints are refused at load)
  and delimiter-embedding training, with their tests; removed-tools.md describes each. Then full-weight training:
  `d1a/full_ft.py` (MasterAdamW, FSDP2 across GPUs under torchrun, snapshots), `--full_ft` and the snapshot options,
  loading full-weight checkpoints (now refused), and the two scripts that only served them, `interpolate_checkpoint.py`
  and `merge_lora_checkpoint.py`, with their tests. Training is single-process LoRA; resume points became D1A's own
  `d1a/training/resume.py`. Then Kev's training-history suites (`decision-v1`, `public-pool-v5`/`v6`, `round3`-`round15`, `sft-v1`, `sft-v2` and
  its `r21`-`r26`, `transfer-v1`, `v3`, `v5`, `v6`, `v8`) and the builders that only wrote them or the two below
  (`build_long_states`, `build_soft_targets`, `build_night2_data`, `freeze_calibration_audit`, `freeze_semif`,
  `freeze_semif_external`). `external` (semif-v1, typesafe-v1, wanli-v1, wanli-v2) and `night2` moved, byte for byte,
  into D1A suites (`evals/d1a/external`, `evals/d1a/night2`, data in the private dataset `JohnP1/d1a-evals`). Then the
  builders of the suites D1A keeps frozen (`build_hard_v1` with `hard_v1_common`/`_families`/`_numeric`/`_policy`,
  `build_breadth_v1`, `build_devtools_v1` with `devtools_v1_licences.json`, `build_longdoc_v1` with
  `longdoc_v1_synthetic`, `build_documents_v1`/`_v2`, `freeze_documents_v1`, `label_documents_v1`,
  `build_binding_diagnostic`, and `d1a/transfer_v9.py`) with the tests that only exercised them; the suites stay as data
  pinned by sha256 (`tests/test_frozen_suites.py`), and [removed-tools.md](removed-tools.md) says where to rebuild them.
  Then `d1a.eval.suite`'s own suite freezer (`python -m d1a.eval.suite`) with `d1a/contrastive.py` and the record generator of
  `d1a/training/composition.py` (its rule shapes stay for `d1a.eval.suite.validate_training`; `paired_flip` moved to `d1a/eval/benchmark.py`),
  and `tests/test_generators.py`. Then the `d1a/training/data.py` converters only that freezer read (13 sources; `build()` keeps
  its six defaults) and `scripts/longdoc_serving.py` (CUDA long-document serving cost, run through the removed Modal app).
- **Rewritten from scratch** (no longer derived), one file per line in path order, so rewrites landing in parallel touch
  different lines; each behaviour-identical rewrite is checked by `scripts/equivalence/` where one exists:
  - `d1a/core/api.py`: the same schema, texts and answers; its tests are `tests/test_system_one.py`.
  - `d1a/eval/benchmark.py`: the same rows, reports and files; its tests are `tests/test_benchmark.py`.
  - `d1a/training/calibrate.py`: replaces Kev's calibration script; `scripts/calibrate_checkpoint.py` is now a wrapper around it.
  - `d1a/backends/checkpoint.py`: the same decisions, refusals and on-disk formats; its tests are `tests/test_checkpoint.py`.
  - `d1a/backends/shared_prefix.py`: the same branch hidden states and gradients, bit for bit, on a tiny Qwen3.5 (eager and
    SDPA, padded and unpadded states, checkpointing on and off); checked by `scripts/equivalence/shared_prefix.py`; its tests
    are `tests/test_shared_prefix.py`.
  - `d1a/training/data.py`: the same records from the same seeds (every draw in the same order); its tests are `tests/test_data.py`.
  - `d1a/eval/metrics.py`: bit-identical results; its tests are `tests/test_metrics.py`.
  - `d1a/eval/predictors.py`: the same predictions, requests, retries and rotation averages; checked by
    `scripts/equivalence/predictors.py` (local, remote and rotation-averaged predictors); its tests are
    `tests/test_predictors.py`, which with `tests/test_encoding.py` replaced Kev's `tests/test_research.py`.
  - `d1a/backends/torch.py` (its encoding and pointer head now in `d1a/core/encoding.py` and `d1a/core/head.py`, #60): the same
    encodings, masks, probabilities and training gradients, bit for bit, on Gemma 4 and Qwen3.5; its encoding, mask and
    pointer-head tests are `tests/test_encoding.py`.
  - `d1a/eval/suite.py`: the same pins, partition checks, mirror rules and file formats; its tests are `tests/test_suite.py`.
  - `d1a/training/composition.py`: the same shapes, splits, held-out keys and structure keys; checked by
    `scripts/equivalence/composition.py`.
  - `d1a/training/train.py`: the same training, bit for bit on the CPU, resumes included; its tests are `tests/test_train.py`.
  - `tests/test_api.py`: the same HTTP API checks and more (tighter sums, the confidence and expected-score formulas, 422s
    at both option limits, isolation for every question in any order); it now serves the tiny checkpoint in process.
- **Rewritten**: `README.md`, `AGENTS.md`, `NOTICE`, `.gitignore`.
- **Kept as is**: the frozen suites in `evals/` (large partitions still download from Kev's Hub dataset
  `jaredpalmer/kev-suites`), the suite builders in `scripts/`.

## Derived files

Files taken from Kev and modified. Each carries the notice "Modified from Kev (...)" and a one-line summary of its
changes at the top; `scripts/check_license.py` checks that this list and the headers match exactly.

```text
.github/workflows/ci.yml
.gitignore
d1a/backends/cuda_graphs.py
d1a/backends/fused_qwen35.py
d1a/backends/mlx.py
d1a/serving/serve.py
d1a/training/temperature_gate.py
pyproject.toml
scripts/mlx_parity.py
tests/test_conventions.py
tests/test_mlx.py
tests/test_model.py
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
d1a/backends/device.py
experiments/sft-v1-lengths.json
experiments/smoke.json
```
