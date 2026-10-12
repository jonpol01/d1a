# Models

Which D1A builds exist, where they are published, and what each version is for.

## Builds

| | |
|---|---|
| Training and serving on Gemma 4 E2B / E4B (and Qwen) | yes, this repo |
| D1A-E2B v0.6 (Gemma 4 E2B: every later D1A skill in one run, for on-device use) | [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) `v0.6`: against v0.2 hard decisions 31% → 62%, developer tools 51% → 65%, long documents 73% → 86%, PR change type 40% → 85%; on 487 PRs newer than all its training, severity 77% and change type 88% (E4B v0.6: 79%, 91%), but it never predicts P0 or P1 there, so E4B stays the PR labeller. Released as an explicit exception: general decisions (decision-v7) −1.4 against its bar's −1 (#229). `v0.2` (general decisions, 2 epochs, calibrated) and `v0.1` stay available |
| D1A-E2B for Apple Silicon (MLX, 8-bit) | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) `v0.6` (3.5 GB, plus 0.9 GB of photo, voice and video encoders) |
| D1A-E4B v0.6 (Gemma 4 E4B: v0.5 refreshed on pull requests newer than all earlier training, plus long documents and routing) | [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) `v0.6`: on 487 PRs newer than all its training, severity 79% (v0.5: 75%) and change type 91% (87%). Released as an explicit exception to the release rules: against v0.5 it loses 2.7 points on hard decisions and 4.1 on the older PR test set's change type (#197, #198), so pin `v0.5` for those. `v0.4`, `v0.3`, `v0.2` (Kev's later stages + Japanese + agent routing) and `v0.1` stay available |
| D1A-E4B for Apple Silicon (MLX, 8-bit) | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) `v0.6` (~6 GB, plus 1 GB of photo, voice and video encoders; what the Mac mini playground serves) |
| D1A-E4B for the Apple Neural Engine (Core ML, int8) | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) `v0.6`, a preview: text only, inputs up to 512 tokens ([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| D1A-E2B for the Apple Neural Engine (Core ML, int8) | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) `v0.6`: text only, inputs up to 512 tokens ([Apple Neural Engine (Core ML)](APPLE.md#apple-neural-engine-core-ml)) |
| Thin clients (Python, JS) | [`clients/`](../clients) |

Model versions: one Hugging Face repository per size and format, and one tag per version (`v0.1`, `v0.2`, `v0.3`, `v0.4`, `v0.5`, `v0.6`, ...); each version continues training the one before it, and its model card lists what it was trained on. Load a version as `JohnP1/d1a-e4b@v0.3`. Older tag names (`v0.2-hybrid`, `v0.2-2epoch`, ...) still work.
