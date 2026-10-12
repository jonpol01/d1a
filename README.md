# D1A

[![ci](https://github.com/jonpol01/d1a/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jonpol01/d1a/actions/workflows/ci.yml)
[![license check](https://github.com/jonpol01/d1a/actions/workflows/license.yml/badge.svg?branch=main)](https://github.com/jonpol01/d1a/actions/workflows/license.yml)
[![release](https://img.shields.io/github/v/release/jonpol01/d1a)](https://github.com/jonpol01/d1a/releases)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](pyproject.toml)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20model-JohnP1%2Fd1a--e4b-yellow)](https://huggingface.co/JohnP1/d1a-e4b)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MLX-black)](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8)
[![demos](https://img.shields.io/badge/demos-d1a--playground-7c3aed)](https://github.com/jonpol01/d1a-playground)

**A small decision model on Gemma 4.** One document and a set of typed questions in, a calibrated probability for every option out, in one forward pass. No text generation.

> **Built on Kev.** D1A is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer, licensed under the [Apache License 2.0](LICENSE). The model code, trainer, benchmark and frozen evaluation suites here started as a copy of Kev (upstream commit `0fe8fc9`); [docs/UPSTREAM.md](docs/UPSTREAM.md) lists every file taken from Kev and what D1A changed. D1A is an independent project. It is not affiliated with, sponsored by or endorsed by Jared Palmer or the Kev authors.

## Proven at

- PR labeling in daily use on a Mac mini: one request per PR, median 0.63 s per PR on an M4 (p95 0.89 s), free, local.
- D1A-E4B v0.6 on 487 PRs newer than all its training: severity 79% and change type 91%.
- The data, training job, scores and history: [recipes/pr-labeler](recipes/pr-labeler/README.md).

## Quick Start

You need Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jonpol01/d1a.git && cd d1a
uv sync --extra serve
uv run --extra serve python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6 --port 8009
```

That serves the MLX build on Apple Silicon. Elsewhere pass the PyTorch build, `--run JohnP1/d1a-e4b@v0.6` (CUDA if you have a GPU; `D1A_BACKEND=torch` runs PyTorch on MPS, see [Apple Silicon (MLX)](docs/APPLE.md#apple-silicon-mlx)). The first run downloads the model. `--run` also takes a local checkpoint directory or a Hub revision (`repo@rev`).

Every setting (server flags, `D1A_*` variables, self-learning, photos and voice, agents) is listed with its default and
when to change it in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

Send it a ticket:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "d1a-latest",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"returns": null, "shipping": null, "billing": null}},
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?"}
  }
}'
```

## Pick a Build

| | PyTorch | MLX 8-bit (Apple Silicon) | Core ML (preview) | Use when |
|---|---|---|---|---|
| D1A-E4B | [d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) | [d1a-e4b-mlx-q8](https://huggingface.co/JohnP1/d1a-e4b-mlx-q8), ~6 GB | [d1a-e4b-coreml](https://huggingface.co/JohnP1/d1a-e4b-coreml), 6.1 GB | PR labels, the most accurate build |
| D1A-E2B | [d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) | [d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8), 3.5 GB | [d1a-e2b-coreml](https://huggingface.co/JohnP1/d1a-e2b-coreml), 3.5 GB | on-device |

E4B for PR labels (E2B never predicted P0/P1 on newer PRs); E2B for on-device. All builds are at `v0.6`. Core ML is a preview for iPhone/iPad: text only, inputs up to 512 tokens, and MLX is faster on Macs ([docs/APPLE.md](docs/APPLE.md)). Sizes, exceptions and older versions: [docs/MODELS.md](docs/MODELS.md).

## Use It

`POST /v1/systemone` takes a `state` (the document) and `questions`, each a `noul` (yes/no), `choice` or `score`. The response has a probability for every option of every question. The same shape works in your own code, without a server:

```python
from d1a import D1A

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 3.0 GB); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instr": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instr": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

- Clients for Python and JS: [docs/USAGE.md](docs/USAGE.md#clients)
- MCP tools for agents: [docs/USAGE.md](docs/USAGE.md#from-agents-mcp)
- Photos, voice and video: [docs/USAGE.md](docs/USAGE.md#photos-voice-and-video)

## How It Works

1. **The input.** The document goes in once; each question follows as its own branch with its options and a `<decide>` token.
2. **The model.** A Gemma 4 backbone with a LoRA adapter and a small pointer head score each option, and a calibrated softmax turns the scores into probabilities.
3. **Two ways to run it.** Packed (one sequence under a block-causal mask) or rows (the document cached, one short row per question), with the same answers.
4. **Where it runs.** `d1a.serving.serve` speaks the System One API on MLX or PyTorch, and reads photos, voice notes and video with Gemma 4's own encoders.
5. **Learning from outcomes.** Decisions are logged with the model that made them; when outcomes come back they refit the probabilities.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/arch/model-dark.svg">
  <img src="docs/arch/model-light.svg" alt="Gemma 4 backbone with LoRA, pointer head, softmax over the options" width="100%">
</picture>

Prose and the other diagrams: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The exact rules: [docs/SPEC.md](docs/SPEC.md).

## Learns From Outcomes

D1A logs every decision. When outcomes come back it recalibrates its probabilities (used in production), and it can retrain behind a held-out gate (no retrain has passed the gate yet). Details and numbers: [docs/SELF_LEARNING.md](docs/SELF_LEARNING.md).

## Demos

Twelve use cases, each answered in one forward pass, are in [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground); the recordings of all twelve are in [docs/DEMOS.md](docs/DEMOS.md).

<img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/42b07a301d95ca0accd46e003a57dc3704011391/docs/gifs/en/prlabel.gif" alt="PR labeler demo running on D1A" width="100%">

## Train Your Own

```bash
uv run python -m d1a.training.recipe run recipes/d1a-e2b.yaml --out runs/d1a-e2b --device cuda
```

One L4 is enough. Fine-tuning, calibration and scoring: [docs/TRAINING.md](docs/TRAINING.md); the recipes: [recipes/README.md](recipes/README.md).

## Status

As of 2026-10-12:

- The PR labeler is in daily use on a Mac mini (D1A-E4B v0.6).
- The SWE verifier is no better than cheap trajectory heuristics within an issue (a null result); recalibration on outcomes works ([docs/SELF_LEARNING.md](docs/SELF_LEARNING.md)).
- The photo, voice and video demos are zero-shot: the checkpoints are trained on text only ([docs/DEMOS.md](docs/DEMOS.md)).
- Core ML is a preview: text only, inputs up to 512 tokens, slower than MLX on Macs ([docs/APPLE.md](docs/APPLE.md)).
- Options within one question can still influence each other; asking questions together or separately gives the same probabilities, but option order is not irrelevant.
- The benchmark numbers Kev publishes are for Kev's own Qwen checkpoints, not D1A.
- Reports are in [docs/reports/](docs/reports/): [D1A-E4B routing v0.1](docs/reports/2026-10-routing-v0.1.md) ([日本語](docs/reports/2026-10-routing-v0.1.ja.md)): model routing and agent-kit decisions, before and after training, with what the numbers do and do not show.

## Development

```bash
uv sync --extra serve
uv run python -m pytest tests --unit -q   # what CI runs: every test file but the few tests/conftest.py's OUTSIDE_UNIT_TESTS lists
python scripts/check_license.py          # license and provenance rules (see CONTRIBUTING.md)
uv run python scripts/equivalence/metrics.py   # a rewritten module against its last version before the rewrite (api, metrics, benchmark, suite)
```

These suites need no model weights (the tokenizer, about 10 MB, is downloaded from the Hub). `tests/test_model.py` and `tests/test_mlx.py` need real weights. `tests/test_api.py` serves the committed tiny checkpoint in process; set `D1A_BASE_URL` to run it against a running server instead (`pytest tests/test_api.py -m server`).

## Releases

D1A's code is versioned with [Semantic Versioning](https://semver.org) and released on [GitHub](https://github.com/jonpol01/d1a/releases); [CHANGELOG.md](CHANGELOG.md) says what changed in each version, and model checkpoints carry their own Hugging Face tags. How to cut a release: [CONTRIBUTING.md](CONTRIBUTING.md#releasing).

## License

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

- D1A's code comes from [Kev](https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0. Files taken from Kev and changed carry a notice at the top; [docs/UPSTREAM.md](docs/UPSTREAM.md) lists them and every change.
- D1A's changes are Copyright 2026 John Soliva, Apache-2.0.
- Gemma 4 is released by Google under Apache-2.0; the Qwen base models are also Apache-2.0. Training datasets have their own licenses.
- Authors: Kev by Jared Palmer ([@jaredpalmer](https://github.com/jaredpalmer)), built with [Devin](https://devin.ai); D1A by John Soliva ([@jonpol01](https://github.com/jonpol01)).
- Credits carried over from Kev: Kev thanks [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked) for the architecture write-up, [TypeSafe](https://docs.typesafe.ai/api) for the API design, [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B-Base) for the base models, [3x3xX3N0N](https://github.com/jaredpalmer/kev/issues/8) and [Radexito](https://github.com/Radexito) for their contributions. Related work: [Hydragen](https://arxiv.org/abs/2402.05099), [DeFT](https://arxiv.org/abs/2404.00242), [FIRST](https://arxiv.org/abs/2406.15657).
