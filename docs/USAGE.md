# Usage

How to call D1A: the System One API, the in-process library, the MCP tools for agents, the clients, and photos, voice and video.

## The System One API

`POST /v1/systemone` takes a `state` (the document), a `model` name and up to many `questions`, each a `noul` (yes/no), `choice` (one of named options) or `score` (an ordered scale). The response has one answer per question with a probability per option, plus token usage and latency. `GET /v1/models` lists the accepted model names with the serving details, including load: requests and batches served, the queue, prefix-cache hits and the recent batches' model time (p50, p95, max). `GET /metrics` reports the same in Prometheus text format (plus memory), for a scraper; like `/v1`, it needs the bearer key when `D1A_API_KEY` is set.

A request may also name its use case, `"use_case": "routing"` (optional, at most 64 characters, surrounding whitespace ignored; D1A's own field, which other System One servers ignore). When the checkpoint carries a temperature for that use case, the request's probabilities are read at it; without `use_case`, or with a name the checkpoint has no temperature for, the checkpoint's own temperature applies and the answers are exactly what they were. The top answer never changes, only how sure it is. `GET /v1/models` lists the checkpoint's `use_case_temperatures` (`{}` for most checkpoints), and the decision log records each request's use case and the temperature it was read at. The agent presets, the MCP tools, `D1A.decide(..., use_case=...)` and both clients pass it. 

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
