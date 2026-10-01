# D1A

**A small decision model on Gemma 4.** One document and a set of typed questions in, a calibrated probability for every option out, in one forward pass. No text generation.

> **Built on Kev.** D1A is built on [Kev](https://github.com/jaredpalmer/kev) by Jared Palmer, licensed under the [Apache License 2.0](LICENSE). The model code, trainer, benchmark, frozen evaluation suites and playground here started as a copy of Kev (upstream commit `0fe8fc9`); [docs/UPSTREAM.md](docs/UPSTREAM.md) lists every file taken from Kev and what D1A changed. D1A is an independent project. It is not affiliated with, sponsored by or endorsed by Jared Palmer or the Kev authors.

## See It Running

Nine use cases, each answered by D1A-E2B in one forward pass (recorded live from the [playground](https://github.com/jonpol01/d1a-playground) on an M4 Mac mini, MLX 8-bit):

<table>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/routing.gif" alt="Model routing demo running on D1A" width="100%"><br><b>1. Model routing</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/guardrails.gif" alt="Guardrails demo running on D1A" width="100%"><br><b>2. Guardrails</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/tools.gif" alt="Tool-call gating demo running on D1A" width="100%"><br><b>3. Tool-call gating</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/inbox.gif" alt="Inbox triage demo running on D1A" width="100%"><br><b>4. Inbox triage</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/rerank.gif" alt="Reranking demo running on D1A" width="100%"><br><b>5. Reranking</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/evals.gif" alt="LLM evals demo running on D1A" width="100%"><br><b>6. LLM evals</b></td></tr>
<tr><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/labeling.gif" alt="Bulk labeling demo running on D1A" width="100%"><br><b>7. Bulk labeling</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/control.gif" alt="Real-time control demo running on D1A" width="100%"><br><b>8. Real-time control</b></td><td align="center" width="33%"><img src="https://raw.githubusercontent.com/jonpol01/d1a-playground/129a15d464f6731dcbeaf390d0cf54f3126cd643/docs/gifs/en/gate.gif" alt="Confidence gate demo running on D1A" width="100%"><br><b>9. Confidence gate</b></td></tr>
</table>

## What It Is

D1A answers yes/no, multiple-choice and rating questions about a piece of text, the way an API call answers a function: routing a support ticket, gating an agent's tool call, triaging an inbox, ranking passages, grading an LLM's answer. Every answer comes with probabilities, so your code can act on the confident cases and send the rest to a person.

A causal LM backbone with a LoRA adapter runs one prefill pass over the document and the questions under a block-causal mask (each question sees the document and itself, never the other questions). A small pointer head scores each option's end token against the question's `<decide>` token, and a softmax turns the scores into probabilities. D1A's backbones are Gemma 4 E2B and E4B; the Qwen bases Kev was built for still work. It speaks the TypeSafe System One API (`POST /v1/systemone`), so the TypeSafe SDK and existing clients work against it.

## What Works Today

| | |
|---|---|
| Training and serving on Gemma 4 E2B / E4B (and Qwen) | yes, this repo |
| D1A-E2B (Gemma 4 E2B, 2 epochs) | [JohnP1/d1a-e2b](https://huggingface.co/JohnP1/d1a-e2b) (tag `v0.2-2epoch`; `v0.1-1epoch` keeps the first version) |
| D1A-E2B for Apple Silicon (MLX, 8-bit) | [JohnP1/d1a-e2b-mlx-q8](https://huggingface.co/JohnP1/d1a-e2b-mlx-q8) (v0.2, 4.2 GB in memory) |
| D1A-E4B (Gemma 4 E4B, 2 epochs) | [JohnP1/d1a-e4b](https://huggingface.co/JohnP1/d1a-e4b) (tag `v0.1-2epoch`; the most accurate D1A so far) |
| Live demos of nine use cases | [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground) |
| Thin clients (Python, JS) | [`clients/`](clients) |

## Quick Start

You need Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jonpol01/d1a.git && cd d1a
uv sync --extra serve
uv run --extra serve python -m d1a.serve --run JohnP1/d1a-e2b --port 8009
```

This serves the prototype checkpoint: CUDA if you have a GPU, MLX on Apple Silicon (`D1A_BACKEND=torch` for PyTorch MPS; see [Apple Silicon (MLX)](#apple-silicon-mlx)). The first run downloads the adapter and the base model. `--run` also takes a local checkpoint directory or a Hub revision (`repo@rev`). Once the D1A weights are published, `--run JohnP1/d1a-e2b` serves them the same way.

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

### The System One API

`POST /v1/systemone` takes a `state` (the document), a `model` name and up to many `questions`, each a `noul` (yes/no), `choice` (one of named options) or `score` (an ordered scale). The response has one answer per question with a probability per option, plus token usage and latency. `GET /v1/models` lists the accepted model names with the serving details, including load: requests and batches served, the queue, prefix-cache hits and the recent batches' model time (p50, p95, max).

Model names: the server answers any name and lists `d1a-latest`, `kev-latest` and `jev-latest`. `d1a-latest` is D1A's own name. `kev-latest` stays accepted so clients written against Kev keep working unchanged, and `jev-latest` is the TypeSafe SDK's default, so an unconfigured SDK client works too. Set `D1A_API_KEY` to require `Authorization: Bearer <key>`.

With the TypeSafe SDK (included in `uv sync --extra serve`); it reads non-English text too:

```python
from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="d1a-latest")
r = client.system_one(
    state="注文した靴が2週間遅れて届き、サイズも間違っていました。",
    questions={"team": Choice(instructions="Which team should handle this?",
                              criteria={"returns": None, "shipping": None, "billing": None})},
)
print(r.choices["team"].choice, r.choices["team"].probabilities)
```

### In Your Own Code (No Server)

Load a checkpoint once and ask questions in-process. Questions and answers use the same shape as the System One API, so code moves between the two unchanged:

```python
from d1a import D1A

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 4.2 GB); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instr": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instr": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

### From Agents (MCP)

`d1a.mcp_server` gives agents (Hermes, Claude Code, any MCP client) the decisions of an agent factory as tools: `d1a_intake` (a new request: which worker, how large a model, ask the user first?), `d1a_judge` (a card and its latest report: done, blocked, needs a person, next step), `d1a_tier` (small, medium or large model for the implementation), `d1a_gate` (may this tool call run: allow, ask, deny), `d1a_route` (which model size answers a prompt) and `d1a_decide` (your own questions). Each returns the answers with probabilities plus `advice`, an action whose defaults fail safe: unsure answers route up, ask a person, or keep a card open (`d1a.presets.advise`).

```bash
pip install "d1a[mcp] @ git+https://github.com/jonpol01/d1a"
D1A_URL=http://127.0.0.1:8009 python -m d1a.mcp_server        # stdio; asks a running d1a.serve
```

Register it as a stdio MCP server (command `python`, args `-m d1a.mcp_server`, env `D1A_URL`) in the agent's MCP settings, and allow the tool names above. `D1A_RUN=<checkpoint>` loads a model in the MCP process instead of calling a server. The question sets are in `d1a/presets.py`, worded exactly as the routing training data asks them.

### Clients

Dependency-free clients for any System One server live in [`clients/python`](clients/python) (PyPI `d1a-client`, import `d1a_client`) and [`clients/js`](clients/js) (npm `d1a-client`):

```python
from d1a_client import Client

client = Client("http://127.0.0.1:8009")
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

### Playground

`playground/` is a Next.js app for trying questions by hand, comparing packed and separate answers, permuting options and playing chess against the model. Start a server on :8009, then `cd playground && npm install && npm run dev` (set `D1A_API` to point it elsewhere).

### Apple Silicon (MLX)

On Apple Silicon `d1a.serve` runs Gemma 4 checkpoints through MLX (`d1a/mlx_model.py`, mlx-lm 0.31.3): the adapter is merged into the bf16 base at load and every request runs as the state once plus one row per question. `scripts/export_mlx.py` writes that merged model as a folder which loads without the base or the adapter, optionally quantized:

```bash
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --out runs/exports/d1a-e2b-mlx-bf16
uv run --extra mlx python scripts/export_mlx.py --run JohnP1/d1a-e2b --q-bits 4 --q-group-size 64 --out runs/exports/d1a-e2b-mlx-4bit
uv run --extra serve python -m d1a.serve --run runs/exports/d1a-e2b-mlx-4bit --port 8009
```

Parity is measured against golden vectors from the fp32 PyTorch path (`scripts/golden_vectors.py`: 209 records, 274 questions: decision-v7 development records, the playground presets and three long-state records). For JohnP1/d1a-e2b on an M1 Max, |Δp| being the largest change of any option's probability in a question:

| | Weights | Max \|Δp\| | Mean \|Δp\| | Argmax flips | Accuracy (dev, fp32 0.811) | ECE (fp32 0.056) |
|---|---|---|---|---|---|---|
| MLX bf16 | 8.7 GB | 0.043 | 0.003 | 5 (all on reference margins under 0.025) | 0.824 | 0.061 |
| MLX 4-bit, group 64, embeddings included | 2.5 GB | 0.363 | 0.054 | 23 | 0.811 | 0.051 |
| MLX 8-bit, per-layer embeddings 4-bit (`--q-bits 8 --q-per-layer-bits 4`) | 3.5 GB | 0.084 | 0.009 | 5 (margins under 0.053) | 0.824 | 0.062 |

The 4-bit export keeps the accuracy but moves individual probabilities too far to stand in for the fp32 model (8% of the answers change), so it is not published. The error comes from the linear layers: with them in bf16 and both embeddings at 4 bits the maximum is 0.089. The mixed export (8-bit linear layers and token embeddings, 4-bit per-layer embeddings, which are half of E2B's weights) stays close to bf16. Process footprint on the M1 Max: 3.1 GB after loading the 4-bit export and 4.2 GB for the mixed one, about 1 GB more after serving, and a 60-question request peaks at 5.3 / 6.4 GB; a 6-question request takes about 0.4 s and a 60-question one about 4.5 s with either.

## Training

The D1A recipe on Gemma 4 E2B (one L4 is enough, about 15 GB peak with a bf16 backbone). `google/gemma-4-E2B` at commit `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` is the trainer's default base, so `--base` can be left out:

```bash
uv run python -m d1a.train --suite evals/v7/decision-v7 \
    --epochs 2 --lr 1e-4 --batch 4 --accum 2 --dtype bf16 --weights_dtype bf16 --checkpointing 1 \
    --p_none_pair 0.25 --device cuda --out runs/d1a-e2b
```

For E4B pass `--base google/gemma-4-E4B --base_revision <sha>`. To fine-tune on your own data, start from a checkpoint: `--data mine.jsonl --init_from JohnP1/d1a-e2b`. For long runs add `--save_every_minutes 30`: a crash then loses at most 30 minutes, and `--resume 1` with the same arguments continues bit for bit. A single non-finite loss or gradient skips its batch instead of ending the run (three in a row still stop it). `python -m d1a.train --help` lists every option. Before you publish or serve a trained checkpoint, write its calibration temperature into it with `scripts/calibrate_checkpoint.py` (training leaves it at 1.0 on purpose; `d1a.serve` warns and `/v1/models` reports `calibrated: false` until you do).

Score a checkpoint on the frozen suites:

```bash
uv run python -m d1a.benchmark --run runs/d1a-e2b/checkpoint --suite evals/v4/transfer-v4 --out runs/d1a-e2b-transfer
```

The frozen suites (`evals/`) come from Kev. Their manifests and small partitions are in git; partitions over about 10 MB are fetched from Kev's Hugging Face dataset [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) on first use and verified by sha256. A few held-out suites name a private mirror and cannot be loaded without access to it. The suites were admitted under Qwen tokenizers, so the trainer re-admits their records under Gemma's tokenizer with each suite's own rule (70 of decision-v7's 12,576 records are dropped).

Gemma's tokenizer has none of the Qwen delimiter tokens, so D1A uses Gemma's reserved `<unused0>`–`<unused4>` tokens and a leading `<bos>`. The sliding-window attention layers get their own copy of the packed mask, and only the text model is loaded.

## Status and Limitations

Reports: [D1A-E4B routing v0.1](docs/reports/2026-10-routing-v0.1.md) ([日本語](docs/reports/2026-10-routing-v0.1.ja.md)): model routing and agent-kit decisions, before and after training, with what the numbers do and do not show.

Early. On the development partitions of decision-v7:

| Model | Base | Accuracy: Trained Sources (dev) | Accuracy: New Sources (dev) |
|---|---|---|---|
| D1A-E4B (JohnP1/d1a-e4b), 2 epochs | Gemma-4-E4B | 0.853 | 0.680 |
| D1A-E2B v0.2 (JohnP1/d1a-e2b), 2 epochs | Gemma-4-E2B | 0.824 | 0.602 |
| D1A-E2B v0.1, 1 epoch | Gemma-4-E2B | 0.794 | 0.569 |
| Kev-0.8B base recipe, 2 epochs (reference) | Qwen3.5-0.8B-Base | 0.817 | 0.622 |

D1A-E4B is the most accurate so far, ahead of Jev (0.845) on sources it trained on and behind it (0.857) on new ones. Two epochs raised the calibration error from 0.040 to about 0.057 on both sizes, which is being looked at. E4B needed half E2B's learning rate (5e-5): at 1e-4 it diverged mid-epoch.

- On one Windows/WSL2 machine with an NVIDIA GPU, PyTorch's fused SDPA attention kernel corrupted memory during Gemma training. Loading the model with eager attention fixed it. If training crashes or produces NaNs there, try eager attention first.
- Options within one question can still influence each other; asking questions together or separately gives the same probabilities, but option order is not irrelevant.
- The benchmark numbers Kev publishes are for Kev's own Qwen checkpoints, not D1A.

## Development

```bash
uv sync --extra serve
uv run python -m pytest tests/test_unit.py tests/test_research.py tests/test_generators.py tests/test_conventions.py \
    tests/test_documents_tools.py tests/test_hard_v1.py tests/test_devtools_v1.py tests/test_breadth_v1.py tests/test_mlx_export.py -q
python scripts/check_license.py          # license and provenance rules (see CONTRIBUTING.md)
```

These suites need no model weights (the tokenizer, about 10 MB, is downloaded from the Hub). `tests/test_model.py`, `tests/test_mlx.py` and `tests/test_api.py` need weights or a running server. `modal_app.py` runs training and benchmarks on [Modal](https://modal.com) under your own workspace (app and volume names are `d1a-*`, set `D1A_APP_NAME` to change the app).

## License

Apache-2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

- D1A's code comes from [Kev](https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0. Files taken from Kev and changed carry a notice at the top; [docs/UPSTREAM.md](docs/UPSTREAM.md) lists them and every change.
- D1A's changes are Copyright 2026 John Soliva, Apache-2.0.
- Gemma 4 is released by Google under Apache-2.0; the Qwen base models are also Apache-2.0. Training datasets have their own licenses.
- Authors: Kev by Jared Palmer ([@jaredpalmer](https://github.com/jaredpalmer)), built with [Devin](https://devin.ai); D1A by John Soliva ([@jonpol01](https://github.com/jonpol01)).
- Credits carried over from Kev: Kev thanks [Archer Hume](https://archerhume.com/posts/jevs-architecture-unmasked) for the architecture write-up, [TypeSafe](https://docs.typesafe.ai/api) for the API design, [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B-Base) for the base models, [3x3xX3N0N](https://github.com/jaredpalmer/kev/issues/8) and [Radexito](https://github.com/Radexito) for their contributions. Related work: [Hydragen](https://arxiv.org/abs/2402.05099), [DeFT](https://arxiv.org/abs/2404.00242), [FIRST](https://arxiv.org/abs/2406.15657).
