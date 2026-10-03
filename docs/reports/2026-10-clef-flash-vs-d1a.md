# Clef-flash vs D1A-E4B

October 2026. Cloudflare released two open decision models on October 1: **Clef** (27B) and **Clef-flash** (9B). They work the way D1A does. Each reads the document once, scores every allowed option with a small head, returns calibrated probabilities, and speaks the same System One API. This report measures Clef-flash against D1A-E4B on the same records, through the same client, on one Mac.

## Summary

- **On the jobs D1A was trained for, D1A wins clearly:**
  - pull-request labeling: 77.7% vs 56.5% in English, 70.0% vs 49.7% in Japanese;
  - Japanese JGLUE: 84.6% vs 71.4%;
  - agent routing: 91.7% vs 83.8%.
  
  This is partly home advantage, because Clef has never seen these question sets. On 17 real pull requests that neither model trained on, D1A still leads, 60.8% vs 56.9%.
- **On general decisions neither model trained on, Clef-flash wins clearly:** 82.2% vs 70.9% on D1A's held-out transfer set. The gap is reasoning:
  - combining conditions (and/or/not, if-then): 91–94% vs 50–66%;
  - general knowledge (MMLU): 78% vs 51%;
  - deadline rules: 80% vs 53%.
- **Speed on this Mac** (M1 Max, both models 8-bit): D1A answers short inputs about 3–4× faster (0.27 s vs 0.98 s median). On long pull requests both take about 4–5 s.
- **Media:**
  - both read images;
  - only D1A reads audio, because Gemma 4 has an audio encoder and Qwen3.5 has none;
  - D1A also reads video, through Gemma 4's own video path.
- **Clef-flash vs Jev:** Clef-flash isn't smarter than TypeSafe's Jev overall. Cloudflare's own Decision Index numbers put it at 57.1 against Jev's 57.9. Clef (27B) scores 61.2. Clef-flash is 13× faster than Jev, though.

## Setup

| | D1A-E4B v0.3 | Clef-flash |
|---|---|---|
| Base | Gemma 4 E4B (~4B effective, 8B with per-layer embeddings) | Qwen3.5-9B |
| Build tested | `JohnP1/d1a-e4b-mlx-q8@v0.3`: MLX, 8-bit, per-layer embeddings 4-bit | `clef-flash` in Ollama 0.35.1: Q8_0, 10 GB |
| Server | `d1a.serve` | Ollama `/v1/systemone` |
| Inputs | text, image, audio, video | text, image |
| License | Apache-2.0 | Apache-2.0 |

- **Hardware:** Apple M1 Max, 64 GB. The models ran one at a time, with no other GPU work.
- **Client:** every request went through [`scripts/compare_systemone.py`](../../scripts/compare_systemone.py). It sends the record's state and its questions without labels and reads each server's answer the same way.
- **Metrics:**
  - accuracy: the top answer equals the label;
  - accuracy and coverage at p ≥ 0.7;
  - expected calibration error over 10 bins;
  - median request latency.

| Set | Records (questions) | Who trained on this kind of data |
|---|---|---|
| PR labels, English test | 953 (2,045) | D1A (v0.3 PR-labeler training) |
| PR labels, Japanese test | 92 (193) | D1A |
| 17 real PRs from the owner's repositories | 17 (51) | neither (labels drafted by Claude) |
| JGLUE development: JNLI, JCommonsenseQA, JSTS | 500 (500) | D1A (JGLUE train split) |
| Agent factory: intake, judge, tool gate | 214 (640) | D1A (routing data) |
| Model routing, generic development set | 270 (270) | D1A |
| Model routing, hand-labelled | 45 (45) | D1A |
| transfer-v4 development: 11 held-out sources | 764 (764) | neither |

## Results

| Set | D1A-E4B v0.3 | Clef-flash | Right at p ≥ 0.7 (D1A / Clef) | Median latency (D1A / Clef) |
|---|---|---|---|---|
| PR labels, English | **77.7%** | 56.5% | 93% on 57% / 89% on 28% | 4.4 s / 5.0 s |
| PR labels, Japanese | **70.0%** | 49.7% | 91% on 56% / 77% on 27% | 2.2 s / 4.4 s |
| 17 real PRs | **60.8%** | 56.9% | 81% on 51% / 76% on 33% | 2.0 s / 3.9 s |
| JGLUE (Japanese) | **84.6%** | 71.4% | 96% on 68% / 91% on 67% | 0.27 s / 0.98 s |
| Agent factory | **91.7%** | 83.8% | 97% on 79% / 92% on 73% | 0.84 s / 1.98 s |
| Model routing, generic | **99.3%** | 90.4% | 100% on 98% / 98% on 68% | 0.39 s / 0.78 s |
| Model routing, hand-labelled | **100%** | 95.6% | 100% on 96% / 100% on 78% | 0.32 s / 0.94 s |
| **transfer-v4 (held out from both)** | 70.9% | **82.2%** | 81% on 70% / 87% on 87% | 0.27 s / 0.94 s |

Per question, D1A / Clef-flash:

- **PR labels, English:**
  - change type 89.1 / 81.7
  - severity 69.8 / 33.6
  - blast radius 54.0 / 41.0
- **JGLUE:**
  - multiple choice 91.5 / 93.3
  - inference 94.7 / 86.5
  - similarity 67.3 / 33.9
- **transfer-v4, by source:**

| Source | D1A | Clef-flash |
|---|---|---|
| MMLU | 51% | **78%** |
| Logic: and/or, or/not, if-then | 50–66% | **91–94%** |
| Deadline rules | 53% | **80%** |
| Emotion | 56% | 62% |
| Paraphrase (PAWS) | 70% | 76% |
| Offensive tweets | **79%** | 76% |
| SciQ, QNLI, authorization rules | 96%, 90%, 100% | same |

Calibration error is similar overall (0.04–0.10 for both). Clef-flash is less often confident on our sets: at p ≥ 0.7 it covers 27–33% of PR labels, against 51–57% for D1A.

## What it means for D1A

1. **The gap is reasoning, not reading.** D1A matches Clef-flash on reading comprehension, policy rules and science questions. It loses on composed conditions and on knowledge. That's where a 9B base helps, and it's where Cloudflare's training data is aimed (the blog cites synthetic data with permuted fields, prompts and schemas).
2. **Training ideas worth trying, from Cloudflare's write-up:**
   - LoRA rank 256, where D1A uses 16;
   - label-smoothed cross-entropy plus a Brier term (`d1a.train --brier_w` already exists and is 0 by default);
   - an RL stage for calibration;
   - a joint head in which all questions and options attend to each other;
   - 64K-token context.
3. **Keep Gemma 4 as D1A's base.** Moving to Clef's Qwen base would lose audio, and with it Voice triage and the driver voice notes. Closing the reasoning gap through data looks more promising, for example more composition and deadline-style records like those transfer-v4 holds out. A Clef-style base remains an option for text-only deployments.
4. **Speed holds up.** On a Mac, D1A's MLX build is faster on short decisions and level on long documents. Cloudflare's 38.8 ms is on an NVIDIA H200, not comparable to these numbers.

## Caveats

- D1A trained on six of the eight sets' kinds of data, so those rows are its home turf. transfer-v4 and the 17 real PRs are the only neutral rows. Cloudflare's 43-benchmark Decision Index is the broad neutral comparison, and D1A has not been run on it.
- The Clef-flash numbers are for Ollama's Q8_0 build, not Cloudflare's bf16 weights on Workers AI.
- The 17 real PRs carry labels drafted by Claude, not by maintainers.
- Latency comes from one Mac running one request at a time, and includes HTTP.

## The PR-labeler round-2 candidate

The same runs scored the round-2 checkpoint, which continues v0.3 on 4,063 unseen English PRs plus 2,185 blast-labeled PRs.

| | Change |
|---|---|
| PR labels, English | +3.0 |
| PR labels, Japanese | +3.6 |
| 17 real PRs | +7.9 |
| JGLUE | −2.8 |
| transfer-v4 | −2.2 |
| Agent factory | −1.1 |
| Generic routing | −1.9 |

Its latency numbers were taken while other GPU work ran, so they're left out. The full comparison will go on its model card if it is published as v0.4.

## Reproduce

```bash
ollama pull clef-flash                                   # Ollama 0.35.1 or later
uv run --extra serve python -m d1a.serve --run JohnP1/d1a-e4b-mlx-q8@v0.3 --port 8019
python scripts/compare_systemone.py --endpoint http://127.0.0.1:8019 --model d1a-latest --out runs/compare/d1a \
  --data transfer=evals/v4/transfer-v4/development.jsonl
python scripts/compare_systemone.py --endpoint http://127.0.0.1:11434 --model clef-flash --out runs/compare/clef \
  --data transfer=evals/v4/transfer-v4/development.jsonl
```

The PR and routing sets are in the private datasets `JohnP1/d1a-pr-labels` and `JohnP1/d1a-routing`; JGLUE is in `JohnP1/d1a-ja-jglue`.

## Sources

- [Cloudflare: Introducing Clef](https://blog.cloudflare.com/clef-decision-models/), with its training recipe and Decision Index numbers
- [Cloudflare changelog](https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/)
- [Cloudflare/clef-flash model card](https://huggingface.co/Cloudflare/clef-flash)
- [Ollama: clef-flash](https://ollama.com/library/clef-flash)
- [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index): Kev-9B 38.5, Winnow-E4B (Gemma 4 E4B) 39.9, Jev 57.9 in edition 0.2.1
