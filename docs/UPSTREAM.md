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
- **Training robustness**: `d1a/train.py` writes resume points for LoRA runs too (`--save_every_steps`,
  `--save_every_minutes`, `--resume`; Kev had them for full-weight runs only), and a non-finite loss or gradient skips
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
  to `JohnP1/d1a-snapshots`; the playground and the label-review tool are titled D1A and proxy the API under `/d1a`.
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
  `temperature_groups`), with the tests that only exercised them. `d1a/rounds.py`, `experiment.py`, `budget.py` and
  `transfer_v9.py` stay because `scripts/calibrate_checkpoint.py`, `d1a/full_ft.py` and two suite builders import them.
- **Rewritten**: `README.md`, `AGENTS.md`, `NOTICE`, `.gitignore`.
- **Kept as is**: the frozen suites in `evals/` (large partitions still download from Kev's Hub dataset
  `jaredpalmer/kev-suites`), the suite builders in `scripts/`, the playground and `tools/review` apart from branding.

## Derived files

Files taken from Kev and modified. Each carries the notice "Modified from Kev (...)" and a one-line summary of its
changes at the top; `scripts/check_license.py` checks that this list and the headers match exactly.

```text
.github/workflows/ci.yml
.gitignore
d1a/api.py
d1a/benchmark.py
d1a/budget.py
d1a/calibrate.py
d1a/checkpoint.py
d1a/cuda_graphs.py
d1a/evaluate.py
d1a/experiment.py
d1a/full_ft.py
d1a/fused_qwen35.py
d1a/metrics.py
d1a/mirror.py
d1a/mlx_model.py
d1a/model.py
d1a/predictors.py
d1a/rounds.py
d1a/serve.py
d1a/shared_prefix.py
d1a/suite.py
d1a/train.py
d1a/transfer_v9.py
playground/next.config.ts
playground/src/app/chess/page.tsx
playground/src/app/layout.tsx
playground/src/components/answer-card.tsx
playground/src/components/chess-game.tsx
playground/src/components/playground.tsx
playground/src/lib/chess.ts
playground/src/lib/d1a.ts
pyproject.toml
scripts/build_binding_diagnostic.py
scripts/build_breadth_v1.py
scripts/build_devtools_v1.py
scripts/build_documents_v1.py
scripts/build_documents_v2.py
scripts/build_hard_v1.py
scripts/build_long_states.py
scripts/build_longdoc_v1.py
scripts/build_night2_data.py
scripts/build_soft_targets.py
scripts/calibrate_checkpoint.py
scripts/freeze_calibration_audit.py
scripts/freeze_documents_v1.py
scripts/freeze_semif.py
scripts/freeze_semif_external.py
scripts/interpolate_checkpoint.py
scripts/label_documents_v1.py
scripts/longdoc_serving.py
scripts/merge_lora_checkpoint.py
scripts/mlx_parity.py
tests/test_api.py
tests/test_breadth_v1.py
tests/test_conventions.py
tests/test_devtools_v1.py
tests/test_documents_tools.py
tests/test_generators.py
tests/test_hard_v1.py
tests/test_longdoc_v1.py
tests/test_mlx.py
tests/test_model.py
tests/test_research.py
tests/test_unit.py
tools/review/README.md
tools/review/index.html
tools/review/src/App.tsx
```

## Derived files without a header

Files taken from Kev and modified that cannot hold a comment (JSON, generated lockfiles).

```text
playground/package-lock.json
playground/package.json
scripts/golden_presets.json
tools/review/package-lock.json
tools/review/package.json
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
d1a/composition.py
d1a/contrastive.py
d1a/data.py
d1a/device.py
experiments/sft-v1-lengths.json
experiments/smoke.json
playground/.gitignore
playground/AGENTS.md
playground/CLAUDE.md
playground/components.json
playground/eslint.config.mjs
playground/postcss.config.mjs
playground/public/file.svg
playground/public/globe.svg
playground/public/next.svg
playground/public/vercel.svg
playground/public/window.svg
playground/scripts/jev-evaluate.mjs
playground/src/app/favicon.ico
playground/src/app/globals.css
playground/src/app/page.tsx
playground/src/components/chess-board.tsx
playground/src/components/ui/badge.tsx
playground/src/components/ui/button.tsx
playground/src/components/ui/card.tsx
playground/src/components/ui/input.tsx
playground/src/components/ui/label.tsx
playground/src/components/ui/separator.tsx
playground/src/components/ui/switch.tsx
playground/src/components/ui/tabs.tsx
playground/src/components/ui/textarea.tsx
playground/src/lib/utils.ts
playground/tsconfig.json
scripts/devtools_v1_licences.json
scripts/hard_v1_common.py
scripts/hard_v1_families.py
scripts/hard_v1_numeric.py
scripts/hard_v1_policy.py
scripts/longdoc_v1_synthetic.py
tools/review/.gitignore
tools/review/public/sample.jsonl
tools/review/src/index.css
tools/review/src/main.tsx
tools/review/tsconfig.json
tools/review/vite.config.ts
```
