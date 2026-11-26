<p>English · <a href="ja/USAGE.md">日本語</a> · <a href="ko/USAGE.md">한국어</a></p>

# Usage

How to call D1A: the System One API, the in-process library, the MCP tools for agents, the clients, and photos, voice and video.

## The System One API

`POST /v1/systemone` takes a `state` (the document), a `model` name and up to many `questions`, each a `noul` (yes/no), `choice` (one of named options) or `score` (an ordered scale). The response has one answer per question with a probability per option, plus token usage and latency. `GET /v1/models` lists the accepted model names with the serving details, including load: requests and batches served, the queue, prefix-cache hits and the recent batches' model time (p50, p95, max). `GET /metrics` reports the same in Prometheus text format (plus memory), for a scraper; like `/v1`, it needs the bearer key when `D1A_API_KEY` is set.

A request may also name its use case, `"use_case": "routing"` (optional, at most 64 characters, surrounding whitespace ignored; D1A's own field, which other System One servers ignore). When the checkpoint carries a temperature for that use case, the request's probabilities are read at it; without `use_case`, or with a name the checkpoint has no temperature for, the checkpoint's own temperature applies and the answers are exactly what they were. The top answer never changes, only how sure it is. `GET /v1/models` lists the checkpoint's `use_case_temperatures` (`{}` for most checkpoints), and the decision log records each request's use case and the temperature it was read at. The agent presets, the MCP tools, `D1A.decide(..., use_case=...)` and both clients pass it. 

**Japanese dates (`ja_dates`, #410).** D1A reads urgency from dates far better when they are written `Month D, YYYY`: on
Japanese documents whose dates alone were rewritten that way, urgency accuracy rose from 75.0% to 81.7% (83.0% for the same
documents in English). A request may send `"ja_dates": true` (D1A's own field, a JSON boolean) to have the server rewrite the
state's Japanese dates before reading it: `2026年8月8日`, `2026/8/8`, `令和8年8月8日` (also `令和元年`, `平成`, `昭和`),
full-width digits, a weekday kept as written (`（金）`), and ranges that start with a full date. Dates without a year, dates
that are not real days and relative terms (`来週`, `明日`) stay as written, and a state with no Japanese in it is never
touched. `false` keeps the dates as written even on a server started with `D1A_JA_DATES=1`; without the field, that
setting decides (off by default, so answers are unchanged). `GET /v1/models` shows whether a server rewrites: its card has
`"preprocessing": {"ja_dates": true, "date_facts": ...}` when the setting is on, and no `preprocessing` field when it rewrites
nothing. `D1A.decide(..., ja_dates=True)` and both clients pass it. To
do the same on the client side before sending, run `d1a.core.ja_dates.with_ja_dates` on the whole state; its docstring lists
the rules for a client in another language. Rewriting was measured on `年月日` and era dates; slash dates, ranges and
dates followed by a time are covered by unit tests only.

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

**Routing temperature (v0.5-era numbers).** Routing is the first use case: v0.5 is served at T 1.78, which leaves its routing answers under-confident; on held-out factory routing rows a routing temperature of 0.85 lowers ECE from 0.132 to 0.030 and raises the share of answers at 0.9 confidence or more from 0.28 to 0.66, with the same accuracy (#209). v0.6 shipped without one; v0.6.1 carries it, T 1.18 for D1A-E4B and 0.95 for D1A-E2B (#264), and every other request keeps the checkpoint's own temperature (1.95 and 1.45).

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

m = D1A.load("JohnP1/d1a-e2b-mlx-q8")   # Apple Silicon (MLX, 3.8 GB download); JohnP1/d1a-e2b on NVIDIA or CPU
answers = m.decide("Shoes arrived late and I was charged twice.",
                   {"team": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "returns", "shipping": "shipping", "billing": "billing"}},
                    "urgent": {"type": "noul", "instructions": "Is this urgent?"}})
print(answers["team"]["probabilities"], answers["urgent"]["noul"])
```

## From Agents (MCP)

`d1a.agents.mcp_server` gives agents (Hermes, Claude Code, any MCP client) the decisions of an agent factory as tools: `d1a_intake` (a new request: which worker, how large a model, ask the user first?), `d1a_judge` (a card and its latest report: done, blocked, needs a person, next step), `d1a_tier` (small, medium or large model for the implementation), `d1a_gate` (may this tool call run: allow, ask, deny), `d1a_route` (which model size should answer a prompt) and `d1a_decide` (your own questions). Each returns the answers with probabilities plus `advice`, the action to take, whose defaults fail safe: an unsure answer advises a larger model, asking a person, or keeping a card open (`d1a.agents.presets.advise`). The calling agent acts on it; to have d1a-serve forward chats itself, see [Route chats to other models](#route-chats-to-other-models).

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

Dependency-free clients for any System One server live in [`clients/python`](../clients/python) (`d1a-client`, import `d1a_client`) and [`clients/js`](../clients/js) (`d1a-client`). Neither is on PyPI or npm yet; each README says how to install it from the repository.

```python
from d1a_client import Client

client = Client("http://127.0.0.1:8009")
answer = client.decide("Shoes arrived late and I was charged twice.",
                       {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"returns": None, "shipping": None, "billing": None}}})
print(answer["answers"]["team"]["probabilities"])
```

Both also cover the rest of a D1A server: questions about a photo, a voice clip or a video (`decide_media` /
`decideMedia`), outcomes for the server to learn from (`feedback`), keeping a test request out of the decision log
(`log=False`), the served model's card (`models`), and OpenAI-format chat (`chat`). `tests/test_clients.py` checks that
every request they send is one the server's own request models accept, and that both clients send the same requests.

## Chat with D1A

A D1A model on an **instruction-tuned** trunk is also a chat model: with its D1A adapter switched off it is exactly
that trunk (bit-identical hidden states). `d1a.serving.serve` uses this to answer OpenAI-format chat and embedding requests
from the same model in memory that answers System One decisions, so apps written for the OpenAI API (mem0, n8n,
chat UIs) can use D1A instead of a separate chat server (#274).

- **Which models can chat:** D1A-E2B-it and D1A-E4B-it today; D1A-12B when trained. The base-trunk models (D1A-E2B, D1A-E4B)
  cannot, and neither can a build whose adapter is merged into its weights (the published `-mlx-q8` / `-mlx-q4` repos).
- **On PyTorch** (NVIDIA, CPU, or a Mac without MLX), serve the LoRA run itself with `D1A_MERGE=0`, which keeps the adapter
  unmerged so it can be switched off:
  `D1A_MERGE=0 python -m d1a.serving.serve --run JohnP1/d1a-e2b-it@v0.6 --port 8009`. Chat then generates on PyTorch: the
  same answers as transformers' own generation, without the MLX prompt cache, so a long conversation reads its whole prompt
  each turn. Tool calls parse as on MLX.
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
It runs on the GPU, as LM Studio runs it, so its vectors are the ones LM Studio stored (cosine 1.00000 on the serving host), and existing memories stay comparable.

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
- **The model's context is a hard limit:** a request whose prompt and `max_tokens` (1024 when not sent) exceed it
  (131,072 tokens for D1A-E4B-it, `max_position_embeddings` in the model's config) gets a 400 at once, OpenAI's
  `context_length_exceeded`, and is never run. Clients that compress on that error (Hermes does) shorten and resend.
- **A tool call cut off by `max_tokens`** is dropped: the answer keeps the text before it, with no tool call and
  `finish_reason` `"length"`, so retry with a larger `max_tokens`.
- **mem0:** set the OpenAI LLM and embedder base URL to the server. Measured with D1A-E2B-it: facts extracted, graph relations
  stored, search finds them; an add takes about 11 s on an M1 Max. Its graph search on questions finds no entities: the 2B
  model answers the question instead of calling the extraction tool, as it did under LM Studio.

## Route chats to other models

With `D1A_ROUTES` set, `POST /v1/chat/completions` picks the model that answers each chat (#419). A client points at D1A
alone:
- a request to review a pull request or a piece of code goes to a review service;
- a chat that needs a larger model goes to an OpenAI-compatible one;
- everything else is answered by D1A, as above.

Unset, the endpoint is unchanged.

```json
{"routes": [
  {"name": "review", "target": {"kind": "review", "url": "http://127.0.0.1:<port>/v1/review", "key_file": "<secret file>", "timeout_s": 300},
   "repos": ["<owner>/*"], "is_review_min_p": 0.8},
  {"name": "large", "target": {"kind": "openai", "url": "http://127.0.0.1:<port>/v1/chat/completions", "model": "<model name>", "key_file": "<secret file>", "timeout_s": 300},
   "max_in_flight": 2, "min_tier": "large"}]}
```

```bash
D1A_ROUTES=routes.json uv run --extra serve python -m d1a.serving.serve --run <chat-capable model> --port 8009
```

**Who is routed.** Every caller of the chat endpoint is routed: anyone who can reach the server, or a relay in front of
it, can send work to the targets. The server says so at startup. Two settings restrict it:
- `"client_key_file": "<secret file>"` at the top of the file: only a chat that sends `Authorization: Bearer <that key>`
  is routed. Any other chat is answered by D1A exactly as without routes, with no route header.
- `D1A_API_KEY`: the server then refuses every request without it, and routes every chat it accepts.

**Setup**
- Either route may be left out.
- Target URLs and keys come from the file only, never from a request. A key file holds one key on one line. It is sent
  as `Authorization: Bearer <key>` (never the client's own header) and is never logged.
- Startup checks the file and never contacts a target. Unknown keys, a URL that is not http(s) or carries credentials,
  and a key file of more than one line are refused.
- Redirects are not followed, and proxy settings in the environment are not used.
- Routing runs before D1A reads the request, so D1A's own limits (its context, `max_tokens`) never stop a chat meant for
  another model.

**Review.** Only a chat whose last message is the user's. One yes/no question is read on that message:

> Is this a request to review an existing pull request (read its diff and give feedback), rather than to write or change code?

It is read raw: no use case, no date rewriting, no calibrator, and it is kept out of the decision log and the prefix cache.
On 40 hand-written English and Japanese requests without links it was right on 38 (AUROC 0.96). Then, in this order, on
the text outside a fenced code block:
1. A GitHub pull-request link and P(yes) ≥ 0.5: that pull request is reviewed (`{"mode": "pr", "pr": "owner/repo#N"}`).
   A message that only mentions a pull request is answered by D1A. A known edge: "Can you summarise what <link>
   changed?" read 0.86 on the served model, so it gets a review.
2. A fenced code block (```` ``` ```` or `~~~`) is reviewed (`{"mode": "code", "files": {"<path>": "<the text inside the fence>"}}`)
   in either of two cases:
   - P(yes) ≥ 0.5;
   - a review verb (review, reviewing, look over, レビュー) when the block is code. Code here means a language tag that
     is not text, markdown, JSON or a log, or an untagged block under "… from `<path>`:".

   The path comes from "… from `<path>`:" (a file name with an extension; never a URL, an absolute path or `..`), else
   `snippet.py`. A fenced draft,
   config or log under "review this" stays with D1A.
3. P(yes) ≥ `is_review_min_p` (0.8) with a pull request named as `owner/repo#N`, or with a review verb and none named
   (then D1A asks which). A bare "continue" read 0.76, which is why a review verb is required when none is named.

The review service's `text` comes back verbatim, whatever its status, so its own "no review was written" replies reach
the user. D1A never writes review text itself; instead it sends one of these fixed replies:
- no pull request named, or two: "Which pull request? Send its link and I'll have it reviewed.";
- a repository not in `repos`: "That repository is not on the review list — no review was written.";
- a review already running (one runs at a time): "The reviewer is busy — no review was written. Try again in a few minutes.";
- no reply text (connection refused, timeout, a status without text): "reviewer unavailable — no review was written.".

A streamed review sends the role chunk at once, then a keep-alive comment every 10 s, then the text. A client that
leaves stops its review. `tools` are ignored.

**Large.** The route preset (`d1a.agents.presets`, use case `routing`, fail-up advice: an unsure answer goes up) is read on
the first user message and the last three. The largest tier wins:
- a "continue" mid-task stays with the large model;
- an agent's tool turn (the last message is a tool result) is read the same way, so a task stays with the model it
  started on.

A chat at `min_tier` or above goes to the target. The default `large` sends only hard requests; `medium` sends
everything but small ones (greetings, simple facts, short rewrites). Then:
- The request body goes to the target as the client sent it (`tools`, `stream_options` and all), with only `model`
  replaced. The target's response, streamed or not, comes back verbatim, its `model` included.
- A 4xx comes back as the target sent it.
- If the target cannot be reached, fails (5xx, a redirect) or sends nothing within `timeout_s` before its first byte,
  D1A answers instead. Its reply then starts with "(The large model is unavailable — answered by D1A.)".
- At most `max_in_flight` (default 2) chats go to the target at once. Another is answered by D1A, its reply starting
  with "(The large model is busy — answered by D1A.)".
- When D1A answers in the large model's place, a `max_tokens` past D1A's own limit (8192) is cut to it, so D1A answers
  rather than refusing the request.
- `timeout_s` bounds each read from the target, not the whole answer.

**Headers:** every chat answered with routes on carries `x-d1a-route` (`review`, `large` or `none`). A route that failed
also carries `x-d1a-route-error`: `which-pr`, `not-listed`, `busy`, `connect`, `timeout`, `status <code>`, `empty`,
`too-large`, `not-json` or `error`. A streamed review sends its headers before the review ends, so a failure there
shows only in the text.

**Cost:** a routed chat pays D1A's passes before it is answered or sent on: the review question on the last user
message, and the route preset on up to four user messages. They run in one model batch.

**Logs:** the chat log (`D1A_CHAT_LOG`) prints a line per chat: the route, P(yes), the tiers and the outcome. It never
prints message text or a key.

## Photos, Voice and Video

The model server answers questions about an image, a short voice clip or a video with the same model, the same request shape plus a `media` field. On Apple Silicon, serve an MLX build that carries Gemma 4's vision and audio encoders (`media/`, about 1 GB, fetched on the first such request; `JohnP1/d1a-e4b-mlx-q8@v0.3` has them, and `scripts/export_mlx.py --media` adds them to your own export):

```bash
uv run --extra serve --extra media python -m d1a.serving.serve --run JohnP1/d1a-e4b-mlx-q8@v0.6.1 --port 8009 --idle-unload 600
```

```bash
curl -s localhost:8009/v1/systemone/media -H 'content-type: application/json' -d '{
  "media": {"type": "image", "data": "'"$(base64 -i parcel.jpg)"'"},
  "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}}
}'
```

`--idle-unload 600` drops the model, encoders included, after 10 minutes without a request and loads it again on the next one (2 s for D1A-E4B on an M1 Max); `GET /v1/models` answers either way and says whether it is `loaded`. With a PyTorch checkpoint (NVIDIA, CPU) run `python -m d1a.serving.media --run JohnP1/d1a-e2b --port 8010` instead: it loads the full Gemma 4 model (bf16, about 10 GB for E2B) as its own process, on the first request, and drops it after 10 idle minutes.


`type` is `image` (JPEG, PNG or WebP only: another format, or a file that does not decode, gets a 422), `audio` (WAV, FLAC or OGG, up to 30 s; any sample rate; a clip that does not decode, or holds no samples, gets a 422) or `video` (MP4, MOV or WebM, up to 32 MB: 16 frames sampled evenly, each read as a timestamped image; the sound track is not used). An optional `state` adds text next to the media. The results of the zero-shot tests are in [DEMOS.md](DEMOS.md#photo-voice-and-video-results).
