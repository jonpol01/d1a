# Usage

How to call D1A: the System One API, the in-process library, the MCP tools for agents, the clients, and photos, voice and video.

## The System One API

`POST /v1/systemone` takes a `state` (the document), a `model` name and up to many `questions`, each a `noul` (yes/no), `choice` (one of named options) or `score` (an ordered scale). The response has one answer per question with a probability per option, plus token usage and latency. `GET /v1/models` lists the accepted model names with the serving details, including load: requests and batches served, the queue, prefix-cache hits and the recent batches' model time (p50, p95, max). `GET /metrics` reports the same in Prometheus text format (plus memory), for a scraper; like `/v1`, it needs the bearer key when `D1A_API_KEY` is set.

A request may also name its use case, `"use_case": "routing"` (optional, at most 64 characters, surrounding whitespace ignored; D1A's own field, which other System One servers ignore). When the checkpoint carries a temperature for that use case, the request's probabilities are read at it; without `use_case`, or with a name the checkpoint has no temperature for, the checkpoint's own temperature applies and the answers are exactly what they were. The top answer never changes, only how sure it is. `GET /v1/models` lists the checkpoint's `use_case_temperatures` (`{}` for most checkpoints), and the decision log records each request's use case and the temperature it was read at. The agent presets, the MCP tools, `D1A.decide(..., use_case=...)` and both clients pass it. 

**Conditional questions (`ask_if`, #36).** A question can be asked only when earlier answers allow it, in the same
request. `"ask_if": {"<question id>": <condition>}`, every condition required:

| The question it names | Condition |
|---|---|
| a `choice` | a list of its option keys: its answer is one of them |
| a `noul` | `true` or `false`: its answer (P(yes) ≥ 0.5) is that |
| a `score` | `{"min": .., "max": ..}`, either optional: its answer's `score` (the expected level, counted from 0) is in range |

```json
{"state": "Order 4471 was charged twice.",
 "questions": {"intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": {"refund": null, "track": null}},
               "amount": {"type": "score", "instructions": "How much is at stake?", "criteria": ["small", "medium", "large"], "ask_if": {"intent": ["refund"]}},
               "carrier": {"type": "choice", "instructions": "Which carrier?", "criteria": {"ups": null, "dhl": null}, "ask_if": {"intent": ["track"]}}}}
```

- **Passes:** the server answers in passes, the questions without `ask_if` first, then each question after the ones it
  names. The state is cached, so a later pass costs only its question rows.
- **Skipped questions:** a question whose condition fails, or that names a skipped question, comes back as
  `{"type": .., "skipped": true, "ask_if": {..}}`, in the request's order.
- **Which answers the conditions read:** the model's own answers, as the learned flags do (before any outcome
  calibration).
- **One request:** usage and `latency_ms` are summed over the passes, and the decision log keeps one entry.
- **Validation:** a condition naming an unknown question or itself, a cycle, or a condition that does not fit the
  question's type is a 422.
- **Never model input:** `ask_if` is D1A's own field, so a request without it is answered exactly as before.
  `/v1/systemone/permute` and `/v1/systemone/separate` refuse it, since they answer questions alone. `D1A.decide` takes
  the same questions.

**Routing temperature (v0.5-era numbers).** Routing is the first use case: v0.5 is served at T 1.78, which leaves its routing answers under-confident; on held-out factory routing rows a routing temperature of 0.85 lowers ECE from 0.132 to 0.030 and raises the share of answers at 0.9 confidence or more from 0.28 to 0.66, with the same accuracy (#209). v0.6 ships without a routing temperature yet: its own is to be fitted on the same rows and gated before it is added.

Model names: `GET /v1/models` lists only `d1a-latest`, but the server answers a request whatever model name it sends, so a client written for another name (an unconfigured TypeSafe SDK client, for one) keeps working. Set `D1A_API_KEY` to require `Authorization: Bearer <key>`.

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

## In Your Own Code (No Server)

Load a checkpoint once and ask questions in-process. Questions and answers use the same shape as the System One API, so code moves between the two unchanged:

```python
from d1a import D1A

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 3.0 GB); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instr": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instr": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

## From Agents (MCP)

`d1a.agents.mcp_server` gives agents (Hermes, Claude Code, any MCP client) the decisions of an agent factory as tools: `d1a_intake` (a new request: which worker, how large a model, ask the user first?), `d1a_judge` (a card and its latest report: done, blocked, needs a person, next step), `d1a_tier` (small, medium or large model for the implementation), `d1a_gate` (may this tool call run: allow, ask, deny), `d1a_route` (which model size answers a prompt) and `d1a_decide` (your own questions). Each returns the answers with probabilities plus `advice`, an action whose defaults fail safe: unsure answers route up, ask a person, or keep a card open (`d1a.agents.presets.advise`).

```bash
pip install "d1a[mcp] @ git+https://github.com/jonpol01/d1a"
D1A_URL=http://127.0.0.1:8009 python -m d1a.agents.mcp_server        # stdio; asks a running d1a.serving.serve
```

Register it as a stdio MCP server (command `python`, args `-m d1a.agents.mcp_server`, env `D1A_URL`) in the agent's MCP settings, and allow the tool names above. `D1A_RUN=<checkpoint>` loads a model in the MCP process instead of calling a server. The question sets are in `d1a/agents/presets.py`, worded exactly as the routing training data asks them, and every tool but `d1a_decide` sends them with `"use_case": "routing"` (`presets.USE_CASES`). `advise` takes the checkpoint's temperatures (`D1A.temperatures`, or the server's `/v1/models` entry, which the MCP tools read on each call) and the use case sent, so that `fail_up`'s small threshold (0.7 at the checkpoint's temperature) follows the temperature the answers were read at: carried through the ratio, `sigmoid(logit(0.7) × T_checkpoint / T_use_case)`, it is 0.855 for v0.5's routing T 0.85. The gate's, the judge's and the 0.5 thresholds stay as they are.

**Claude Code gate hook.** `d1a.agents.claude_code_gate` is a ready-made `PreToolUse` hook: before Claude Code runs a
tool call, D1A's gate decides.

```json
{"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "python -m d1a.agents.claude_code_gate", "timeout": 30}]}]}}
```

- **D1A's answers:** "deny" denies the call, and "ask" asks you. Each carries D1A's probabilities as the reason.
- **"allow" adds nothing by default:** Claude Code's own permission rules decide, as without the hook. Set
  `D1A_GATE_ALLOW=1` to let D1A's "allow" skip the prompt.
- **Fail-safe:** if D1A is unreachable, slower than `D1A_GATE_TIMEOUT` (default 10 s), or anything else fails, the hook asks you.
- **No forged fields:** the command and the card are each read on one line, with newlines escaped, so a command can't forge the gate's other fields.
- **Use `D1A_URL`:** with `D1A_RUN`, the hook loads the model on every call.
- **What it reads:** `D1A_URL`/`D1A_RUN`, as the MCP tools do, and `D1A_GATE_CARD` (the task the agent is on;
  default: the session's directory).
- **Other tools:** a `matcher` such as `"Bash|Write|Edit"` gates them too, read as the tool's name and input.
- **What it is, and isn't:** a second opinion on tool calls, not a prompt-injection defense. It never replaces Claude
  Code's own permissions. Measured on D1A-E4B-it v0.6:
  - on public prompt-injection detection (devtools-v1), it catches **60%** of injections (45 of 75), with no false alarms;
  - on held-out agent-factory tool calls (in-distribution: the same generator as its training data), the gate's top
    answer is 98% right, and with the fail-safe advice it allowed 0 of 33 calls that should have been asked or denied.

## Clients

Dependency-free clients for any System One server live in [`clients/python`](../clients/python) (PyPI `d1a-client`, import `d1a_client`) and [`clients/js`](../clients/js) (npm `d1a-client`):

```python
from d1a_client import Client

client = Client("http://127.0.0.1:8009")
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

## Chat with D1A

A D1A model on an **instruction-tuned** trunk is also a chat model: with its D1A adapter switched off it is exactly
that trunk (bit-identical hidden states). `d1a.serving.serve` uses this to answer OpenAI-format chat and embedding requests
from the same model in memory that answers System One decisions, so apps written for the OpenAI API (mem0, n8n,
chat UIs) can use D1A instead of a separate chat server (#274).

- **Which models can chat:** D1A-E2B-it and D1A-E4B-it today; D1A-12B when trained. The base-trunk models (D1A-E2B, D1A-E4B)
  cannot, and neither can a build whose adapter is merged into its weights (the published `-mlx-q8` / `-mlx-q4` repos).
- **What changes:** chat runs with the adapter off, on the server's model thread between decision requests. A decision
  that arrives during a chat is answered between two of the chat's tokens (or, while a long prompt is being read, between two chunks of it), with the adapter on for it and off again
  after, so it waits about one token rather than the whole answer, and the chat's answer is the same as without it.

**1. Build a chat-capable folder** from the published adapter (Apple Silicon; downloads the -it base, ~10 GB for E2B,
~16 GB for E4B; for E4B-it use `--run JohnP1/d1a-e4b-it@v0.6`):

```bash
uv run --extra serve python scripts/export_mlx.py --run JohnP1/d1a-e2b-it@v0.6 --q-bits 8 --q-group-size 64 \
    --unmerged --out runs/exports/d1a-e2b-it-chat
```

`--unmerged` keeps the adapter beside the base (`adapter/`) instead of folding it in, and the folder carries the base's
chat template. On the 28-request bench its decisions match fp32 (0/42 flips).

**2. Serve it**, with embeddings if your app needs them (mem0 does):

```bash
uv sync --extra serve --extra embed      # llama.cpp, for /v1/embeddings
D1A_EMBED_GGUF=<path to nomic-embed-text-v1.5.Q4_K_M.gguf> \
  uv run --extra serve python -m d1a.serving.serve --run runs/exports/d1a-e2b-it-chat --port 8009
```

The GGUF is `nomic-ai/nomic-embed-text-v1.5-GGUF`, file `nomic-embed-text-v1.5.Q4_K_M.gguf`: the file LM Studio serves.
It runs on the GPU, as LM Studio runs it, so its vectors are the ones LM Studio stored (cosine 1.00000 on the Mac mini), and existing memories stay comparable.

**3. Point an OpenAI client at it**:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8009/v1", api_key="unused")   # a real key when D1A_API_KEY is set
r = client.chat.completions.create(model="d1a-chat", messages=[{"role": "user", "content": "What is SQL injection?"}])
```

- **Endpoints:** `POST /v1/chat/completions` (messages, `max_tokens`, `temperature`, `top_p`, `stop`, `tools` and
  `tool_choice`, `response_format`, `stream`), `POST /v1/embeddings` (`input`, a string or a list; `encoding_format` `float` or `base64`, which Node
  OpenAI clients such as n8n send by default), and `GET /v1/models`,
  which lists `d1a-chat` and the embedder beside the decision model, also while an idle server has unloaded the
  model (`--idle-unload`).
- **Any model id is answered**, so an app configured for another model (an LM Studio id, say) needs only the URL.
- **Streaming** sends the answer token by token; with `tools` it comes as one chunk, once the tool calls are parsed.
- **Agent loops:** send the assistant's tool calls back as OpenAI gives them (`arguments` as a JSON string), followed
  by `tool` messages with the results. Each step reuses the earlier prompt's cache. Measured with D1A-E4B-it on an M1
  Max: 3 tool calls and a summary in 4 steps, 4.1 s cold, then 0.6–1.8 s a step. Arguments that are not a JSON object
  get a 422.
- **Prompt cache:** a chat whose prompt continues an earlier one (an agent's next step, the next turn of a conversation, a
  new session with the same system prompt) prefills only its new tokens; `usage.prompt_tokens_details.cached_tokens`
  says how many it took from the cache. Up to 16 prompts are kept in 1 GiB (`D1A_CHAT_CACHE_MB`, `0` turns it off),
  and they go with the model at an idle unload. Measured with D1A-E4B-it on an M1 Max: an agent step on a 20k-token prompt takes 1–1.7 s instead of 33 s, and a
  new session with the same 20k-token system prompt 2.2 s.
- **JSON mode** asks the model for JSON in the prompt; there is no constrained decoding.
- **A tool call cut off by `max_tokens`** is dropped: the answer keeps the text before it, with no tool call and
  `finish_reason` `"length"`, so retry with a larger `max_tokens`.
- **mem0:** set the OpenAI LLM and embedder base URL to the server. Measured with D1A-E2B-it: facts extracted, graph relations
  stored, search finds them; an add takes about 11 s on an M1 Max. Its graph search on questions finds no entities: the 2B
  model answers the question instead of calling the extraction tool, as it did under LM Studio.

## Photos, Voice and Video

The model server answers questions about an image, a short voice clip or a video with the same model, the same request shape plus a `media` field. On Apple Silicon, serve an MLX build that carries Gemma 4's vision and audio encoders (`media/`, about 1 GB, fetched on the first such request; `JohnP1/d1a-e4b-mlx-q8@v0.3` has them, and `scripts/export_mlx.py --media` adds them to your own export):

```bash
uv run --extra serve --extra media python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6 --port 8009 --idle-unload 600
```

```bash
curl -s localhost:8009/v1/systemone/media -H 'content-type: application/json' -d '{
  "media": {"type": "image", "data": "'"$(base64 -i parcel.jpg)"'"},
  "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}
}'
```

`--idle-unload 600` drops the model, encoders included, after 10 minutes without a request and loads it again on the next one (2 s for D1A-E4B on an M1 Max); `GET /v1/models` answers either way and says whether it is `loaded`. With a PyTorch checkpoint (NVIDIA, CPU) run `python -m d1a.serving.media --run JohnP1/d1a-e2b --port 8010` instead: it loads the full Gemma 4 model (bf16, about 10 GB for E2B) as its own process, on the first request, and drops it after 10 idle minutes.


`type` is `image` (JPEG, PNG, WebP), `audio` (WAV, FLAC or OGG, up to 30 s; any sample rate) or `video` (MP4, MOV or WebM, up to 32 MB: 16 frames sampled evenly, each read as a timestamped image; the sound track is not used). An optional `state` adds text next to the media. The results of the zero-shot tests are in [DEMOS.md](DEMOS.md#photo-voice-and-video-results).
