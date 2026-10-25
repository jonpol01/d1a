# Apple Silicon

D1A on Apple hardware: MLX on the GPU and a Core ML preview on the Neural Engine.

## Apple Silicon (MLX)

On Apple Silicon `d1a.serving.serve` runs Gemma 4 checkpoints through MLX (`d1a/backends/mlx.py`, mlx-lm 0.31.3): the adapter is merged into the bf16 base at load and every request runs as the state once plus one row per question. `scripts/export_mlx.py` writes that merged model as a folder which loads without the base or the adapter, optionally quantized:

```bash
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --out runs/exports/d1a-e2b-mlx-bf16
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --q-bits 4 --q-group-size 64 --out runs/exports/d1a-e2b-mlx-4bit
uv run --extra mlx python scripts/export_mlx.py --run <a Gemma 4 12B-it QAT checkpoint> --q4-0 --out runs/exports/d1a-12b-mlx-q4_0   # Q4_0 grid
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b-it --ct-base google/gemma-4-E2B-it-qat-w4a16-ct@<rev> --unmerged --out runs/exports/d1a-e2b-it-qat-mlx-q4   # a QAT checkpoint's own int4
uv run --extra serve python -m d1a.serving.serve --run runs/exports/d1a-e2b-mlx-4bit --port 8009
```

`--q4-0` writes 4 bits in groups of 32 on llama.cpp's Q4_0 grid (scale = d, bias = −8·d) instead of mlx-lm's min/max scales, for a base trained for Q4_0 such as `google/gemma-4-12B-it-qat-q4_0-unquantized`: on that checkpoint Q4_0's weight error is about 20% below mlx-lm's 4-bit group 32 and 30% below group 64 (#270).

`--ct-base` builds on a compressed-tensors QAT checkpoint of the same model instead of the run's base, such as `google/gemma-4-E2B-it-qat-w4a16-ct`. Its int4 Linears (symmetric, groups of 32, packed 8 per int32 from the low nibble up, each stored as q + 8) are already MLX's 4-bit layout, so they are kept exactly: the words as they are, scale s and bias −8·s, which dequantize to s·q. Its other tensors are kept as they are, and the embeddings are quantized as usual, or at their own width with `--q-embedding-bits` (8 keeps them closer to the checkpoint's bf16). It needs `--unmerged`, since merging the adapter would leave the grid. On D1A-E2B-it all 276 quantized Linears are exact (d1a#341). A run trained on such a checkpoint (`--base_ct`, docs/TRAINING.md) exports on it without `--ct-base`, and refuses a different one; `--media` is refused for it, as for `--ct-base`.

Parity is measured against golden vectors from the fp32 PyTorch path (`scripts/golden_vectors.py`: 209 records, 274 questions: decision-v7 development records, the playground presets and three long-state records). For JohnP1/d1a-e2b on an M1 Max, |Δp| being the largest change of any option's probability in a question:

| | Weights | Max \|Δp\| | Mean \|Δp\| | Argmax flips | Accuracy (dev, fp32 0.811) | ECE (fp32 0.056) |
|---|---|---|---|---|---|---|
| MLX bf16 | 8.7 GB | 0.043 | 0.003 | 5 (all on reference margins under 0.025) | 0.824 | 0.061 |
| MLX 4-bit, group 64, embeddings included | 2.5 GB | 0.363 | 0.054 | 23 | 0.811 | 0.051 |
| MLX 8-bit, per-layer embeddings 4-bit (`--q-bits 8 --q-per-layer-bits 4`) | 3.5 GB | 0.084 | 0.009 | 5 (margins under 0.053) | 0.824 | 0.062 |

The 4-bit export keeps the accuracy but moves individual probabilities too far to stand in for the fp32 model (8% of the answers change), so it is not published. The error comes from the linear layers: with them in bf16 and both embeddings at 4 bits the maximum is 0.089. The mixed export (8-bit linear layers and token embeddings, 4-bit per-layer embeddings, which are half of E2B's weights) stays close to bf16. Process footprint on the M1 Max: 3.1 GB after loading the 4-bit export and 4.2 GB for the mixed one, about 1 GB more after serving, and a 60-question request peaks at 5.3 / 6.4 GB; a 6-question request takes about 0.4 s and a 60-question one about 4.5 s with either.

Gemma 4's per-layer embeddings are read from the weight files per request instead of held in memory (each request needs only its own tokens' rows): the 8-bit E2B export uses 2.96 GB after loading instead of 4.22 GB, E4B 5.37 GB instead of 6.88 GB, with bit-identical probabilities and no slower (the lookup adds about 0.25 ms per pass). `D1A_PLE_FLASH=0` keeps them in memory.

## Apple Neural Engine (Core ML)

D1A also runs on the Apple Neural Engine through Core ML, as a preview for iPhone/iPad; on Macs MLX is faster; text only, ≤ 512 tokens (#224):
- The text model is converted in chunks of 6 or 7 layers (5 chunks for E2B, 7 for E4B), with int8 weights (per block of 64).
- The embedding tables stay on the host as row lookups.
- Each chunk runs in its own process.

Both Core ML builds give the same answers as the full-precision model on every clear-cut question. On a Mac, MLX on the GPU
is several times faster, so the serving host keeps MLX; Core ML is for iPhone and iPad, where it keeps the GPU free. The
photo, voice and video encoders are not converted yet. Each Core ML repository's `reference/run.py` answers a request from
the downloaded files.

Measured on 2026-10-11 on the same 28 real requests (42 questions, all ≤ 512 tokens), on an Apple M1 Max and an Apple M4.

| | E2B · MLX | E2B · Core ML | E4B · MLX | E4B · Core ML |
|---|---|---|---|---|
| Version | v0.6 | v0.6 | v0.6 | v0.6 |
| Repository | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) | [JohnP1/d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml) | [JohnP1/d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8) | [JohnP1/d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml) |
| Runs on | Mac GPU | Neural Engine (iPhone, iPad, M4-class Macs) | Mac GPU | Neural Engine |
| Text | yes | yes | yes | yes |
| Photo, voice, video (16 frames, no sound) | yes | not yet (next) | yes | not yet |
| Input length | long documents | ≤ 512 tokens | long documents | ≤ 512 tokens |
| Questions | yes/no, choice, score, with a calibrated probability for each option | same | same | same |
| Languages | questions in English; documents in English or Japanese | same | same | same |
| PR labels, 487 newest PRs: type / severity | 87.5% / 77.2%; 0 of 30 P0/P1 found | same model | 90.6% / 79.1%; 13 of 30 P0/P1 | same model |
| Size | 3.8 GB + 1.0 GB media | 3.8 GB (2.0 GB chunks + 1.7 GB embeddings) | 6.6 GB + 1.0 GB media | 6.5 GB (4.2 GB chunks + 2.3 GB embeddings) |
| Median request, M4 (p95) | **0.20 s** (0.26) | 0.96 s (0.97) | 0.63 s (0.89) | 1.80 s (1.91) |
| Median request, M1 Max (p95) | **0.17 s** (0.23) | 9.7 s (10.0) ² | 0.81 s (1.33) | 2.93 s (2.95) |
| Load, M4 / M1 Max | 2 s / 9 s | 40 s / 247 s | 3 s / – | 52 s / 106 s |
| Answer changes vs fp32 PyTorch ¹ | 0 of 37 clear-cut (M4: 1 near-tie of 42) | 0 of 37 clear-cut (M4: 1 near-tie of 42) | 0 of 38 clear-cut | 0 of 38 clear-cut |
| Photo / voice / video, median (M1 Max) | 0.83 s over the 14 demo requests | – | – | – |

¹ Each build is compared with the fp32 PyTorch rebuild of the same v0.6 model. "Clear-cut" means the reference's top
answer leads by more than 0.05; a near-tie can flip either way.
² On the M1 Max's Neural Engine, the E2B chunks fall back to the CPU: each chunk takes the same time with or without the
Neural Engine. The M4 runs the same files on its Neural Engine.
