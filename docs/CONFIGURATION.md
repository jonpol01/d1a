# Configuring D1A

Every setting D1A reads, in one place. It is written for people and for coding agents: each entry gives the default,
what the setting changes, and when to change it. `tests/test_configuration_doc.py` fails when the code reads a `D1A_*`
variable or a server flag that this page does not list.

D1A is configured three ways:
- **command-line flags** on `python -m d1a.serving.serve` and the other entry points;
- **environment variables** (`D1A_*`), read when the process starts;
- **files the server re-reads while it runs**: the outcome calibrator (`D1A_OUTCOME_CALIBRATOR`).

Nothing else changes D1A's behaviour. The defaults are the measured, released configuration. Leave a setting at its
default unless this page names a reason to change it.

## Check what is in force

`GET /v1/models` on a running server returns the effective configuration:
- the checkpoint (`run`), `device`, `backend` and `dtype`;
- the temperature and per-use-case temperatures, and whether it is `calibrated`;
- `learning`: whether decisions are logged, the calibrator file, and the questions it corrects;
- the prefix cache's size and limits, and recent batch latency.

```bash
curl -s 127.0.0.1:8009/v1/models | python -m json.tool
```

## I want to ...

| Goal | Setting |
|---|---|
| serve a released model | `--run JohnP1/d1a-e4b-mlx-q8@v0.6` (Apple Silicon) or `--run JohnP1/d1a-e4b@v0.6` (CUDA, CPU); pin the tag |
| reach the server from another machine or a container | `--host 0.0.0.0`, and set `D1A_API_KEY` |
| require a key | `D1A_API_KEY=<key>`; clients send `Authorization: Bearer <key>` |
| free memory when idle | `--idle-unload 600` |
| answer about photos, voice and video | install the `media` extra and serve an MLX export that carries `media/` |
| learn from outcomes | `D1A_FEEDBACK_LOG`, `POST /v1/feedback`, `python -m d1a.learning.feedback promote`, `D1A_OUTCOME_CALIBRATOR` (see Self-learning) |
| keep test or demo traffic out of the learning log | request header `x-d1a-decision-log: off` |
| reproduce reported numbers exactly | `D1A_DTYPE=fp32` (and `D1A_BACKEND=torch` on Apple Silicon) |
| give agents D1A as tools | `python -m d1a.agents.mcp_server` with `D1A_URL` or `D1A_RUN` |
| score a remote D1A endpoint | `python -m d1a.eval.benchmark --remote <url>` with `D1A_REMOTE_API_KEY` |

## The server: `python -m d1a.serving.serve`

Install with `uv sync --extra serve`. Add `--extra mlx` on Apple Silicon, `--extra media` for photos, voice and video,
and `--extra mcp` for the MCP server.

| Flag | Default | What it does |
|---|---|---|
| `--run` | `runs/d1a` | The checkpoint: a Hub id with an optional `@tag` (`JohnP1/d1a-e4b-mlx-q8@v0.6`), a local training run (holds `head.pt`) or an MLX export folder. Pin a tag in production. |
| `--fallback` | `runs/smoke` | Served when `--run` is a local path that does not exist. |
| `--host` | `127.0.0.1` | The interface to bind. `0.0.0.0` serves beyond this machine; set `D1A_API_KEY` when you do. |
| `--port` | `8008` | The port. The docs and the playground use 8009. |
| `--device` | `auto` | `auto` tries cuda, then mps, then cpu, and skips a device whose test kernel fails. Name a device to insist on it. |
| `--idle-unload` | `0` | Seconds without a request before the model is dropped from memory; the next request reloads it. `0` keeps it loaded. |

| Variable | Default | What it does | When to change it |
|---|---|---|---|
| `D1A_API_KEY` | unset (open) | Requires `Authorization: Bearer <key>` on `/v1/*` and `/metrics`. | Always when the server is reachable beyond localhost. Never commit the key. |
| `D1A_PREFIX_CACHE` | `4` | States (documents) kept in the prefix cache, so repeated questions on one document skip its prefill. `0` turns it off. | Raise it for many clients asking about different documents; `0` for an exact-memory budget. |
| `D1A_PREFIX_MAX_TOKENS` | `65536` | State tokens the cache holds in total, least recently used evicted first. A longer state is not cached. | Lower it on small-memory machines. |
| `D1A_PREFIX_MIN_TOKENS` | the model's (0 for MLX and hybrid backbones, 384 for attention-only torch models) | States shorter than this are not cached. | Rarely. |
| `D1A_DATE_FACTS` | `0` | `1` adds computed date facts to a state that holds two or more dates (`d1a.core.api.with_date_facts`). | When questions compare dates; measure first, as the reported numbers are without it. |
| `D1A_FEEDBACK_LOG` | unset | See Self-learning. | |
| `D1A_OUTCOME_CALIBRATOR` | unset | See Self-learning. | |

Fixed in code, not settings: up to 64 requests per batch (`MAX_BATCH`), a 75 s keep-alive, and the model list
(`/v1/models` lists `d1a-latest`, but any model name in a request is answered).

## Model loading (`D1A_*` load options)

`d1a.backends.checkpoint.LoadOptions.from_env` reads these for the command-line tools (the server, the benchmark).
Library code passes `LoadOptions(...)` instead. Without a setting, the server picks the defaults shown.

| Variable | Default | What it does | When to change it |
|---|---|---|---|
| `D1A_BACKEND` | `auto` in the server | `torch`, `mlx` or `auto`. `auto` uses MLX on Apple Silicon for Gemma 4 and Qwen3.5 bases when mlx-lm is installed and fp32 was not asked for. An MLX export always runs on MLX. | `torch` on a Mac to reproduce the reported (torch) numbers. |
| `D1A_DTYPE` | bf16 on cuda and mps, fp32 on cpu | `bf16`, `fp16` or `fp32`. fp32 is the exact path every reported number uses. bf16 halves memory and is 2–4.5× faster, with probabilities within ~0.01. Ignored on MLX. | `fp32` to reproduce numbers. |
| `D1A_ATTN` | sdpa on mps and cuda, eager on cpu | The attention kernel: `sdpa` or `eager`. Ignored on MLX. | Rarely. |
| `D1A_MERGE` | `1` | `0` keeps the LoRA adapter unmerged (torch only; MLX always merges). | Debugging only. |
| `D1A_LORA_SCALE` | `1` | Interpolates between the base model (`0`) and the fine-tuned weights (`1`). | Experiments only. |
| `D1A_TEMPERATURE` | the checkpoint's | Serves every request, use cases included, at this temperature. `1.0` gives raw logits. | Experiments only: the checkpoint's own temperature is fitted and released with it. |
| `D1A_CUDA_GRAPHS` | on for cuda in the server | `0` declines CUDA-graph replay of the hybrid backbone's passes. | Debugging on CUDA. |
| `D1A_FUSED` | on for cuda when available | `0` declines and `1` insists on the fused Qwen3.5 Triton kernels (needs flash-linear-attention at the pinned version). | Debugging on CUDA. |
| `D1A_PLE_FLASH` | on | MLX only: reads Gemma 4's per-layer embeddings from the weight files per request instead of holding them, about 1.3–1.5 GB less memory with identical probabilities. `0` keeps them in memory. | Rarely. |
| `D1A_SHAPE_BUCKET` | `64` | Torch on MPS: pads sequences to a multiple of this, so compiled kernels are reused. `1` disables it. | Rarely. |

## Self-learning

The loop: log each decision → report its real outcome → fit a calibrator on the outcomes → promote it only through
a statistical gate. Details are in README "Self-Learning" and in the `d1a.learning.feedback` docstring.

| Setting | Default | What it does |
|---|---|---|
| `D1A_FEEDBACK_LOG=<path>` | unset (off) | Logs every decision to this append-only JSONL file, and gives each answer a `decision_id`. Enables `POST /v1/feedback`. |
| header `x-d1a-decision-log: off` | logged | Answers the request but keeps it out of the log. Use it for demos, smoke tests and anything no outcome will follow. |
| `POST /v1/feedback` | — | `{"decision_id", "labels": {question_id: answer}, "src": "human" or "reviewer", "group": "<e.g. owner/repo#123>"}`. When sources disagree on a question, the most trusted one's label wins. Returns 404 when logging is off. |
| `D1A_OUTCOME_CALIBRATOR=<path>` | unset (off) | Applies this calibrator to yes/no and choice answers. The server reloads it when the file changes, so promotion needs no restart. |
| `D1A_LEARNING=<path>` | unset (today's behaviour) | The self-learning settings file (below), read by `promote`, `replay` and `tick` on every run, so a change needs no restart. The server itself does not read it: decision logging stays `D1A_FEEDBACK_LOG`, because the outcome poster and `/review` depend on that log. |
| header `x-d1a-replay-of: <decision id>` | — | Sent by `replay` only. The server logs the answer as a decision of the model it serves now, marked as a replay of that earlier decision, so the replay takes its outcomes. `/v1/systemone` only. |

The `python -m d1a.learning.feedback` commands each take the log as their first argument, and also `--src` (only
outcomes from one source) and `--run` (only decisions one model made):

| Command | What it does |
|---|---|
| `status <log>` | Counts resolved and pending decisions, per question. |
| `records <log> --out <file>` | Turns resolved decisions into training records for `d1a.training.train --data`. |
| `calibrate <log> --out <file> [--min-outcomes 20]` | Fits a calibrator on every outcome, with no gate. |
| `promote <log> --calibrator <file> [--min-outcomes 20] [--held-out 0.3] --run <served run>` | Fits on part of the outcomes, split by group. It gates each question on the rest and rewrites the calibrator only for the questions that pass: log loss better, with a bootstrap interval clear of zero. Flags not given come from the settings file. Each question's report gives `mde`, the smallest log-loss gain its held-out set can detect. |
| `config init\|show\|set key=value ...\|validate [--file <path>]` | Creates, shows (with where each value came from and the last changes), changes or checks the settings file (`$D1A_LEARNING`). `set` validates the whole file first and writes nothing when a value is out of range. |
| `trained --mix <mix.jsonl> [--mix ...] --run <repo@tag> --out <file> [--log <log>]` | The trained manifest of a model: every PR id and state hash in the mixes of its lineage (give every stage's mix). A mix that is not found makes it incomplete, and replay refuses it. `--log` gives records exported from a feedback log their PR id. |
| `replay <log> --server <url> --trained <file> [--cap 50] [--pause 1]` | Re-sends the decisions an earlier model made, that have outcomes, to the live server, one at a time and only while no request waits, so the served model starts with every earlier outcome. It skips a decision whose PR or state is in the manifest, or that has no PR id (by default), and refuses a missing, incomplete or other model's manifest. The idle check can race with one live request, which then waits one replay. With `D1A_DATE_FACTS=1` a replayed state gets today's date facts, not the original day's. |
| `tick <log> --calibrator <file> --server <url> --trained-dir <dir>` | One run of the loop, meant every few minutes: replay after a model change, then `promote` when it is due (`schedule`), a report line in `learning-status.json` beside the log, and a webhook message on a promotion or when a question first has enough outcomes. |

The settings file (`D1A_LEARNING`; `config show` prints it). Unset, or a key left out, means the default:

| Setting | Default | Allowed | What it does |
|---|---|---|---|
| `outcomes.enabled` | `true` | true, false | Whether outcome collectors (the playground's poster) run. |
| `outcomes.sources` | `null` (all) | a list, e.g. `["reviewer", "human"]` | The outcome sources that count for learning. |
| `promotion.auto` | `true` | true, false | `false`: the gate still runs and reports, and never writes the calibrator. |
| `promotion.questions` | `null` (all) | a list of question ids | The questions that learn. |
| `promotion.min_outcomes` | `20` | ≥ 10 | Fit outcomes a question needs before it is fitted at all. |
| `promotion.held_out` | `0.3` | 0.1 to 0.5 | The share of groups the gate holds out. |
| `promotion.ci_level` | `0.95` | 0.9, 0.95, 0.99 | The gate's bootstrap interval. |
| `promotion.tolerance`, `promotion.bootstrap` | `0.01`, `2000` | 0 to 0.1; 200 to 20,000 | Frozen-suite tolerance; bootstrap resamples. |
| `schedule.daily_at` | `"04:00"` | `HH:MM` local time, or `null` | `tick` runs the gate once a day after this time. |
| `schedule.after_new_outcomes` | `10` | ≥ 1, or `null` | `tick` also runs the gate after this many new outcomes. |
| `replay.enabled`, `replay.on_model_change` | `true`, `true` | true, false | Replay earlier outcomes when the served model changes. |
| `replay.exclude_no_pr_id` | `true` | true, false | Skip decisions with no PR id (they cannot be checked against training). |
| `replay.per_tick_cap`, `replay.pause_s` | `50`, `1.0` | 1 to 500; 0 to 60 s | Replays per tick, and the pause between them. |
| `reports.webhook_file` | `null` | a path | A file holding a webhook URL (Discord-compatible); the URL itself never goes in the settings. |
| `reports.notify` | `["promotion", "min_reached"]` | from `promotion`, `min_reached`, `every_gate` | What `tick` posts to the webhook. |

A file with an error is never used: jobs keep the last valid one (`<file>.last-valid.json`) and report the error, and
every accepted change is appended to `<file>.audit.jsonl`.

How to set it up on your own deployment:
1. Serve with `D1A_FEEDBACK_LOG=runs/feedback/decisions.jsonl` and `D1A_OUTCOME_CALIBRATOR=runs/feedback/calibrator.json`.
2. Send the outcome back to `POST /v1/feedback` whenever you learn it, with a `group` (the PR, ticket or document it belongs to) and a `src`.
3. `export D1A_LEARNING=runs/feedback/learning.json` and run `python -m d1a.learning.feedback config init`; change what you need with `config set`.
4. Write the served model's trained manifest (`trained --mix ... --run <served run>`) into a folder, then schedule `tick runs/feedback/decisions.jsonl --calibrator runs/feedback/calibrator.json --server http://127.0.0.1:8009 --trained-dir <folder>` every few minutes. It replays after a model change and gates when `schedule` says so; a calibrator fitted on another model's answers is never applied to this one's.
5. Read `runs/feedback/learning-status.json` (each question's outcomes and how many it still needs, and the last gate reports), and `GET /v1/models` → `learning.calibrated_questions` after a promotion.

A question needs `--min-outcomes` (20) outcomes before it is fitted, and the gate needs enough held-out groups to
decide. With a few outcomes a day, expect weeks rather than days.

## Photos, voice and video

Install the `media` extra. Serve an MLX export that carries `media/` (E4B: `v0.3` to `v0.6`; E2B: `v0.2`, not
`v0.2.1-2epoch-calibrated`), or a torch checkpoint (the encoders come from its base). `POST /v1/systemone/media` takes a
System One request plus `{"media": {"type": "image" | "audio" | "video", "data": <base64>}}`. A video is read as 16
frames; its sound is not used. There are no further settings.

## Agents (MCP)

`python -m d1a.agents.mcp_server` (the `mcp` extra) exposes D1A's agent decisions as MCP tools over stdio.

| Variable | Default | What it does |
|---|---|---|
| `D1A_URL` | `http://127.0.0.1:8009` | The D1A server the tools ask. |
| `D1A_RUN` | unset | Loads this checkpoint in the MCP process instead of asking a server. |

## Evaluation

`python -m d1a.eval.benchmark` scores a checkpoint (`--run`, using the load options above) or a running server
(`--remote <url> --remote-model <name>`).

| Variable | Default | What it does |
|---|---|---|
| `D1A_REMOTE_API_KEY` | `local` | The bearer key sent to `--remote`; match the server's `D1A_API_KEY`. |

## The Mac mini playground deployment

The playground (jonpol01/d1a-playground) runs D1A as launchd services through `mini.sh`, configured by `.demo/mini.env`
(`MODEL_RUN`, `FEEDBACK_LOG`, `OUTCOME_CALIBRATOR`, `LABEL_OUTCOMES`, ...). The playground's README documents those keys.
`./mini.sh update` applies a change.
