# Models

Which D1A builds exist, where they are published, and what each version is for.

## What each build can and cannot do (2026-10-25)

Every published D1A build, checked against its files on the Hub. **Decisions**: the System One API (`POST /v1/systemone`).
**Chat**: OpenAI-format `POST /v1/chat/completions`, which needs the adapter kept separate (an *unmerged* build): with it
switched off the model is the instruction-tuned Gemma 4 itself. **Photo, voice, video**: `POST /v1/systemone/media`,
zero-shot on every build (no media in training).

| Build | Format | Decisions | Chat | Photo | Voice | Video | Limits |
|---|---|---|---|---|---|---|---|
| [d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6.1` | PyTorch (adapter) | ✅ | ❌ base trunk | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6.1` | MLX 8-bit, merged | ✅ | ❌ base trunk | ✅ | ✅ | ✅ | 7.5 GB |
| [d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | preview: text only, inputs ≤ 512 tokens |
| [d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6.1` | PyTorch (adapter) | ✅ | ❌ base trunk | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6.1` | MLX 8-bit, merged | ✅ | ❌ base trunk | ✅ | ✅ | ✅ | 4.8 GB |
| [d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | preview: text only, inputs ≤ 512 tokens |
| [d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it) `v0.6` | PyTorch (adapter) | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e4b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8) `v0.6` | MLX 8-bit, merged | ✅ | ❌ merged³ | ✅ | ✅ | ✅ | 7.5 GB |
| [d1a-e4b-it-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4) `v0.6` | MLX 4-bit, merged | ✅ | ❌ merged³ | ✅ | ✅ | ✅ | 5.2 GB |
| [d1a-e4b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit) `v0.6` | Core ML 4-bit | ✅ | ❌ | ❌ | ❌ | ❌ | preview: text only, ≤ 512 tokens; 3 of 42 test answers move |
| [d1a-e2b-it](https://huggingface.co/JohnP1/d1a-e2b-it) `v0.6` | PyTorch (adapter) | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | |
| [d1a-e2b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q8) `v0.6` | MLX 8-bit, merged | ✅ | ❌ merged³ | ✅ | ✅ | ✅ | 4.8 GB |
| [d1a-e2b-it-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-mlx-q4) `v0.6` | MLX 4-bit, merged | ✅ | ❌ merged³ | ✅ | ✅ | ✅ | 3.6 GB; not scored on the suites |
| [d1a-e2b-it-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-coreml) `v0.6` | Core ML int8 | ✅ | ❌ | ❌ | ❌ | ❌ | preview: text only, ≤ 512 tokens |
| [d1a-e2b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e2b-it-coreml-4bit) `v0.6` | Core ML 4-bit | ✅ | ❌ | ❌ | ❌ | ❌ | preview: text only, ≤ 512 tokens |
| [d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4) | MLX 4-bit (Google's QAT int4, exact), **unmerged** | ✅ | ✅ | ✅ | ✅ | ✅ | the recommended E2B-it build, replacing D1A-E2B-it v0.6 by the owner's decision of 2026-10-25 despite transfer-v4 −5.2 points; 3.7 GB |
| [d1a-e2b-it-qat](https://huggingface.co/JohnP1/d1a-e2b-it-qat) | PyTorch (adapter; `base_ct`: Google's QAT int4 trunk, rebuilt on load) | ✅ | ✅² | ✅¹ | ✅¹ | ✅¹ | the recommended E2B-it on PyTorch; loads Gemma 4 E2B-it and the QAT checkpoint |
| [d1a-e2b-it-qat-coreml](https://huggingface.co/JohnP1/d1a-e2b-it-qat-coreml) | Core ML (Google's QAT int4, exact; the adapter as a switch) | ✅⁴ | ✅⁴ | ✅⁴ | ✅⁴ | ✅⁴ | the recommended E2B-it on Core ML; 4.6 GB; verified on a Mac (M1 Max, GPU), not yet on an iPhone |

¹ Through the separate PyTorch media server, `python -m d1a.serving.media --run <repo>`, which loads the full Gemma 4
model (bf16, about 10 GB for E2B) ([Photos, voice and video](USAGE.md)).
² Served with `D1A_MERGE=0`, which keeps the adapter unmerged so chat can switch it off ([Chat with D1A](USAGE.md#chat-with-d1a)).
³ Chat needs an unmerged build: `scripts/export_mlx.py --run JohnP1/d1a-e2b-it --unmerged --media` makes one
([Chat with D1A](USAGE.md#chat-with-d1a)).

⁴ Through the repository's Python reference runner (`reference/d1a_coreml_qat.py`, no server; `io.json` lists every model
input and output for a port); every model on the GPU, none on the Neural Engine yet; decisions up to 1,536 tokens, chat up to
2,048 positions ([APPLE.md](APPLE.md#d1a-e2b-it-qat-on-core-ml)).

D1A-E2B-it QAT is the only Core ML build with chat, photo, voice and video; the v0.6 Core ML builds are text only.

## Capabilities (2026-10-17)

What each D1A model line can do today. "Decisions" means D1A's System One API (`POST /v1/systemone`): yes/no, choice and score questions over a document, answered in one pass with calibrated probabilities. Numbers are from each model's card; "not measured" means the card has no number yet.

| | **D1A-E4B-it v0.6** | **D1A-E4B v0.6.1** | **D1A-E2B v0.6.1** | **D1A-E2B-it v0.6** | D1A-12B |
|---|---|---|---|---|---|
| Base model (trunk) | Gemma 4 E4B-it (instruction-tuned) | Gemma 4 E4B (pretrained base) | Gemma 4 E2B (pretrained base) | Gemma 4 E2B-it (instruction-tuned) | Gemma 4 12B-it QAT; **not trained yet** (jonpol01/d1a#270, shelved) |
| Decisions (`/v1/systemone`) | yes | yes | yes | yes (experimental) | — |
| PR labeling, 487 PRs newer than all training: severity / change type | not measured on this set; PR-labels test (953 PRs): 72.9% / 88.4% (E2B-it 70.5% / 87.8%) | **79.1% / 90.6%**; P0/P1 found 13 of 30 | 77.2% / 87.5%; P0/P1 0 of 30 | 72.9% / 87.3%; P0/P1 3 of 30 ("not a PR labeller") | — |
| Skill suites (hard, developer tools, long documents, transfer, Japanese JGLUE) | measured 2026-10-17 against E2B-it (card): up on 21 of 22 suites, e.g. hard 75.7%, developer tools 74.3%, long documents 88.6%, JGLUE 78.6% | measured (card) | measured (card) | measured 2026-10-15 against E4B v0.6.1 (not published) | — |
| Photos, voice and video (`/v1/systemone/media`, MLX builds with `media/`) | yes, zero-shot; 7 of 28 demo answers differ from E2B-it (6 better by hand read) | yes, zero-shot | yes; 25 of 28 demo answers match E4B | yes; 21 of 28 match E4B | — |
| Routing temperature (`"use_case": "routing"`) | no (fitted, not shipped: no better than the general T) | yes, T 1.18 (from v0.6.1) | yes, T 0.95 (from v0.6.1) | no | — |
| Self-learning (outcome log, recalibration, learned flags) | yes | yes, when served by `d1a.serving.serve` | yes | yes | — |
| **Chat / text generation** | **yes**: an unmerged build (`scripts/export_mlx.py --unmerged`); with the adapter off it is Gemma 4 E4B-it ([Chat with D1A](USAGE.md#chat-with-d1a)) | **no**: base trunk; the MLX build has the adapter merged | **no** (same) | **yes**: build it with `scripts/export_mlx.py --unmerged` and serve it; with the adapter off it is Gemma 4 E2B-it ([Chat with D1A](USAGE.md#chat-with-d1a)). Also runs as an unmerged adapter in the Mieru iPhone app | planned: adapter off = Gemma 4 12B-it |
| Embeddings | from a GGUF embedder served beside it (`D1A_EMBED_GGUF`) | — | — | from a GGUF embedder served beside it (`D1A_EMBED_GGUF`) | — |
| Builds | PyTorch adapter `JohnP1/d1a-e4b-it`; MLX `JohnP1/d1a-e4b-it-mlx-q8` (7.5 GB with media) and `-mlx-q4` (5.2 GB), both merged; Core ML 4-bit `JohnP1/d1a-e4b-it-coreml-4bit` (text only, ≤512 tokens; 3 of 42 test answers move, int8 not built) | PyTorch `JohnP1/d1a-e4b`; MLX 8-bit `JohnP1/d1a-e4b-mlx-q8` (7.5 GB with 1.0 GB of media encoders) and mirror `mlx-community/d1a-e4b-8bit`; Core ML int8 preview `JohnP1/d1a-e4b-coreml` (text only, ≤512 tokens) | PyTorch `JohnP1/d1a-e2b`; MLX 8-bit `JohnP1/d1a-e2b-mlx-q8` (4.8 GB) and mirror `mlx-community/d1a-e2b-8bit`; Core ML int8 `JohnP1/d1a-e2b-coreml` (text only, ≤512 tokens) | PyTorch adapter `JohnP1/d1a-e2b-it`; MLX `JohnP1/d1a-e2b-it-mlx-q8` (4.8 GB) and `-mlx-q4` (3.6 GB), both merged; Core ML `JohnP1/d1a-e2b-it-coreml` and `-coreml-4bit` | 4-bit Q4_0 export ready (`scripts/export_mlx.py --q4-0`, #272) |
| Calibration temperature | 1.55 | 1.95 | 1.45 | 1.41 | — |

Older single-purpose builds for PR labelling and routing were superseded by D1A-E4B v0.6.1.

## Builds

| | |
|---|---|
| Training and serving on Gemma 4 E2B / E4B (and Qwen) | yes, this repo |
| Gemma 4 12B (the unified model, e.g. `google/gemma-4-12B-it-qat-q4_0-unquantized`) as a base | both backends load it (#270: PyTorch through transformers, MLX through mlx-lm's Gemma 4 classes); no D1A-12B is trained yet |
| D1A-E2B v0.6 (Gemma 4 E2B: every later D1A skill in one run, for on-device use) | [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6`: against v0.2 hard decisions 31% → 62%, developer tools 51% → 65%, long documents 73% → 86%, PR change type 40% → 85%; on 487 PRs newer than all its training, severity 77% and change type 88% (E4B v0.6: 79%, 91%), but it never predicts P0 or P1 there, so E4B stays the PR labeller. Released as an explicit exception: general decisions (decision-v7) −1.4 against its bar's −1 (#229). `v0.6.1` adds a routing temperature (T 0.95) to the same weights. `v0.2` (general decisions, 2 epochs, calibrated) and `v0.1` stay available |
| D1A-E2B for Apple Silicon (MLX, 8-bit) | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6` and `v0.6.1` (3.8 GB, plus 1.0 GB of photo, voice and video encoders) |
| D1A-E4B v0.6 (Gemma 4 E4B: v0.5 refreshed on pull requests newer than all earlier training, plus long documents and routing) | [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6`: on 487 PRs newer than all its training, severity 79% (v0.5: 75%) and change type 91% (87%). Released as an explicit exception to the release rules: against v0.5 it loses 2.7 points on hard decisions and 4.1 on the older PR test set's change type (#197, #198), so pin `v0.5` for those. `v0.6.1` adds a routing temperature (T 1.18) to the same weights. `v0.4`, `v0.3`, `v0.2` (Kev's later stages + Japanese + agent routing) and `v0.1` stay available |
| D1A-E4B-it v0.6 (E2B v0.6's recipe on the instruction-tuned Gemma 4 E4B-it: decisions and chat in one model) | [JohnP1/d1a-e4b-it](https://huggingface.co/JohnP1/d1a-e4b-it) `v0.6`: against D1A-E2B-it v0.6 on every suite, accuracy up on 21 of 22 (pooled +2.5 points; hard decisions +6.6, developer tools +3.2, long documents +2.3), about 2.9× slower per decision; with the adapter off it is Gemma 4 E4B-it ([Chat with D1A](USAGE.md#chat-with-d1a)). MLX builds, adapter merged (decisions only): [JohnP1/d1a-e4b-it-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q8) and [-mlx-q4](https://huggingface.co/JohnP1/d1a-e4b-it-mlx-q4) |
| D1A-E4B-it for the Apple Neural Engine (Core ML, 4-bit) | [JohnP1/d1a-e4b-it-coreml-4bit](https://huggingface.co/JohnP1/d1a-e4b-it-coreml-4bit) `v0.6`, a preview: text only, inputs up to 512 tokens. Against the fp32 reference on 28 requests, 3 of 42 answers move (max \|Δp\| 0.46), p50 6.6 s on an M1 Max GPU; use the MLX 8-bit build where answers must not move. No int8 build |
| D1A-E2B-it QAT for Apple Silicon (MLX, 4-bit) | [JohnP1/d1a-e2b-it-qat-mlx-q4](https://huggingface.co/JohnP1/d1a-e2b-it-qat-mlx-q4): D1A-E2B-it v0.6 retrained on Google's QAT int4 trunk (`google/gemma-4-E2B-it-qat-w4a16-ct`), kept exact (2.9 GB, plus 0.9 GB of photo, voice and video encoders from the QAT checkpoint); decisions, chat, photos, voice and video. Against the v0.6 adapter on the same trunk +2.1 points pooled; against D1A-E2B-it v0.6 on bf16 +0.4 pooled, PR severity +3.5 and PR labels +1.7, but transfer-v4 −5.2. It replaces v0.6 as the recommended E2B-it build by the owner's decision of 2026-10-25, a release exception like D1A-E4B v0.6's (#341). Every suite: [the scorecard](reports/2026-10-25-e2b-it-qat-scorecard.md) |
| D1A-E4B for Apple Silicon (MLX, 8-bit) | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6` and `v0.6.1` (6.6 GB, plus 1.0 GB of photo, voice and video encoders) |
| D1A-E4B for the Apple Neural Engine (Core ML, int8) | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6`, a preview: text only, inputs up to 512 tokens ([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| D1A-E2B for the Apple Neural Engine (Core ML, int8) | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6`: text only, inputs up to 512 tokens ([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| Thin clients (Python, JS) | [`clients/`](../clients) |

Model versions: one Hugging Face repository per size and format, and one tag per version (`v0.1`, `v0.2`, `v0.3`, `v0.4`, `v0.5`, `v0.6`, `v0.6.1`, ...); each version continues training the one before it, and its model card lists what it was trained on. Load a version as `JohnP1/d1a-e4b@v0.3`. Older tag names (`v0.2-hybrid`, `v0.2-2epoch`, ...) still work.
