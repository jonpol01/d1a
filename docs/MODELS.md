# Models

Which D1A builds exist, where they are published, and what each version is for.

## Capabilities (2026-10-14)

What each D1A model line can do today. "Decisions" means D1A's System One API (`POST /v1/systemone`): yes/no, choice and score questions over a document, answered in one pass with calibrated probabilities. Numbers are from each model's card; "not measured" means the card has no number yet.

| | **D1A-E4B v0.6.1** | **D1A-E2B v0.6.1** | **D1A-E2B-it v0.6** | D1A-12B |
|---|---|---|---|---|
| Base model (trunk) | Gemma 4 E4B (pretrained base) | Gemma 4 E2B (pretrained base) | Gemma 4 E2B-it (instruction-tuned) | Gemma 4 12B-it QAT; **not trained yet** (jonpol01/d1a#270, shelved) |
| Decisions (`/v1/systemone`) | yes | yes | yes (experimental) | — |
| PR labeling, 487 PRs newer than all training: severity / change type | **79.1% / 90.6%**; P0/P1 found 13 of 30 | 77.2% / 87.5%; P0/P1 0 of 30 | 72.9% / 87.3%; P0/P1 3 of 30 ("not a PR labeller") | — |
| Skill suites (hard, developer tools, long documents, transfer, Japanese JGLUE) | measured (card) | measured (card) | not measured | — |
| Photos, voice and video (`/v1/systemone/media`, MLX builds with `media/`) | yes, zero-shot | yes; 25 of 28 demo answers match E4B | yes; 21 of 28 match E4B | — |
| Routing temperature (`"use_case": "routing"`) | yes, T 1.18 (from v0.6.1) | yes, T 0.95 (from v0.6.1) | no | — |
| Self-learning (outcome log, recalibration, learned flags) | yes, when served by `d1a.serving.serve` | yes | yes | — |
| **Chat / text generation** | **no**: base trunk; the MLX build has the adapter merged | **no** (same) | **possible, not served by d1a yet.** With the adapter off it is Gemma 4 E2B-it; works today as an unmerged adapter in the Mieru iPhone app; `d1a.serving.serve` gains chat in a later release | planned: adapter off = Gemma 4 12B-it |
| Embeddings | no | no | no | no |
| Builds | PyTorch `JohnP1/d1a-e4b`; MLX 8-bit `JohnP1/d1a-e4b-mlx-q8` (7.6 GB with 1.0 GB of media encoders) and mirror `mlx-community/d1a-e4b-8bit`; Core ML int8 preview `JohnP1/d1a-e4b-coreml` (text only, ≤512 tokens) | PyTorch `JohnP1/d1a-e2b`; MLX 8-bit `JohnP1/d1a-e2b-mlx-q8` (4.8 GB) and mirror `mlx-community/d1a-e2b-8bit`; Core ML int8 `JohnP1/d1a-e2b-coreml` (text only, ≤512 tokens) | PyTorch adapter `JohnP1/d1a-e2b-it`; MLX `JohnP1/d1a-e2b-it-mlx-q8` (4.8 GB) and `-mlx-q4` (3.6 GB), both merged; Core ML `JohnP1/d1a-e2b-it-coreml` and `-coreml-4bit` | 4-bit Q4_0 export ready (`scripts/export_mlx.py --q4-0`, #272) |
| Calibration temperature | 1.95 | 1.45 | 1.41 | — |
| Where it runs | the Mac mini playground and the PR labeler (MLX 8-bit, v0.6.1) | — | the Mieru iPhone app | — |

Older single-purpose builds: `JohnP1/d1a-e4b-pr-labeler-mlx-q8` (v1) and `JohnP1/d1a-e4b-routing` / `-routing-mlx-q8` (v0.1), superseded by D1A-E4B v0.6.1.

## Builds

| | |
|---|---|
| Training and serving on Gemma 4 E2B / E4B (and Qwen) | yes, this repo |
| Gemma 4 12B (the unified model, e.g. `google/gemma-4-12B-it-qat-q4_0-unquantized`) as a base | both backends load it (#270: PyTorch through transformers, MLX through mlx-lm's Gemma 4 classes); no D1A-12B is trained yet |
| D1A-E2B v0.6 (Gemma 4 E2B: every later D1A skill in one run, for on-device use) | [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6`: against v0.2 hard decisions 31% → 62%, developer tools 51% → 65%, long documents 73% → 86%, PR change type 40% → 85%; on 487 PRs newer than all its training, severity 77% and change type 88% (E4B v0.6: 79%, 91%), but it never predicts P0 or P1 there, so E4B stays the PR labeller. Released as an explicit exception: general decisions (decision-v7) −1.4 against its bar's −1 (#229). `v0.2` (general decisions, 2 epochs, calibrated) and `v0.1` stay available |
| D1A-E2B for Apple Silicon (MLX, 8-bit) | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6` (3.5 GB, plus 0.9 GB of photo, voice and video encoders) |
| D1A-E4B v0.6 (Gemma 4 E4B: v0.5 refreshed on pull requests newer than all earlier training, plus long documents and routing) | [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6`: on 487 PRs newer than all its training, severity 79% (v0.5: 75%) and change type 91% (87%). Released as an explicit exception to the release rules: against v0.5 it loses 2.7 points on hard decisions and 4.1 on the older PR test set's change type (#197, #198), so pin `v0.5` for those. `v0.4`, `v0.3`, `v0.2` (Kev's later stages + Japanese + agent routing) and `v0.1` stay available |
| D1A-E4B for Apple Silicon (MLX, 8-bit) | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6` (~6 GB, plus 1 GB of photo, voice and video encoders; what the Mac mini playground serves) |
| D1A-E4B for the Apple Neural Engine (Core ML, int8) | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6`, a preview: text only, inputs up to 512 tokens ([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| D1A-E2B for the Apple Neural Engine (Core ML, int8) | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6`: text only, inputs up to 512 tokens ([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| Thin clients (Python, JS) | [`clients/`](../clients) |

Model versions: one Hugging Face repository per size and format, and one tag per version (`v0.1`, `v0.2`, `v0.3`, `v0.4`, `v0.5`, `v0.6`, ...); each version continues training the one before it, and its model card lists what it was trained on. Load a version as `JohnP1/d1a-e4b@v0.3`. Older tag names (`v0.2-hybrid`, `v0.2-2epoch`, ...) still work.
