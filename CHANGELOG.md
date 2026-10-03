# Changelog

All notable changes to D1A's code (the `d1a` package, its server and its tools) are listed here, newest first.

- **Versions** follow [Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`. Before 1.0 a minor version may
  change an API; every such change is listed under *Upgrade notes*. The version lives in `pyproject.toml`.
- **Releases** are cut by pushing a `vX.Y.Z` tag. CI then checks that the tag, `pyproject.toml` and this file agree,
  runs the tests, builds the package and publishes a GitHub release whose text is that version's section below.
- **Model checkpoints are versioned separately**: one Hugging Face repository per size and format, one tag per model
  version (for example `JohnP1/d1a-e4b@v0.4`). Each release lists the checkpoints it was tested with; the table under
  *Model versions* at the end of this file lists them all.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Video.** `POST /v1/systemone/media` takes `{"type": "video"}`: an MP4, MOV or WebM clip up to 32 MB, of which 16
  frames are sampled evenly and read through Gemma 4's own video path (timestamped frames, the same vision encoder), by
  the same model on MLX and in `d1a.media`. The sound track is not used. Zero-shot: on single-scene clips D1A-E4B v0.3
  answers as it does for the still photo; questions about the order of events are not reliable yet. New dependency in
  the `media` extra: `av` (PyAV).
- **One model for text, photos and voice.** `d1a.serve` answers `POST /v1/systemone/media` (a request plus a photo or
  voice clip) with the model it already serves: Gemma 4's vision and audio encoders turn the media into soft tokens,
  and the same MLX language model reads them. The encoders (~1 GB for E4B) ship in an export's `media/` folder
  (`scripts/export_mlx.py --media`, or `--media-only` for an existing export), and are fetched and loaded on the first
  photo or voice request. On the playground's 10 samples D1A-E4B v0.3 matches its own PyTorch path (0 of 20 answers
  change, max |Δp| 0.065). `python -m d1a.media` stays for PyTorch checkpoints.
- `d1a.serve --idle-unload SECONDS` drops the model (and the media encoders) after that long without a request and
  loads it again on the next one: 2.0 s for D1A-E4B on an M1 Max, which goes from 6.4 GB in use (encoders
  loaded) to 1.3 GB. `GET /v1/models` answers without
  loading the model and reports `loaded` and `idle_unload_s`. Default 0 (always loaded), as before.
- `recipes/pr-labeler`: how D1A-E4B v0.3 learned to label pull requests, as scripts: fetch labeled PRs from GitHub,
  build records with the labeling job's own questions, translate them into other languages (on MLX or any
  OpenAI-compatible server), train and score on one GPU (Hugging Face Jobs or your own), and read the results.

### Changed

- **PR-length requests are about 30% faster on Apple Silicon, with identical answers.** The state pass (the document,
  run once before the questions) now stops before Gemma 4's KV-shared layers: those layers keep no cache of their own
  and nothing reads the state's own outputs, so for the state they were pure waste (18 of D1A-E4B's 42 layers). On an
  M1 Max, 60 held-out pull requests (median 1,584 state tokens): state pass 3,470 -> 2,137 ms, whole request 4,465 ->
  3,082 ms (medians), every probability bit-identical; photos, voice and video too (a 16-frame clip's state pass 1,805
  -> 1,079 ms). Other backbones run the full pass as before.

- Downloading a model from the Hub skips an export's `media/` folder (1 GB) until a photo or voice request needs it.

### Security

- torch 2.13.0 (from 2.8.0), which fixes three memory-corruption advisories: GHSA-vgrw-7cvw-pwgx (`unpack_sequence`,
  medium), GHSA-qfhq-4f3w-5fph (`lstm_cell`, low) and GHSA-rrmf-rvhw-rf47 (`torch.jit.script`, low). D1A calls none of
  these functions. On D1A-E2B's PyTorch path (Apple GPU) the answers are unchanged (0 of 40 top answers differ) and the
  median latency drops from 424–543 ms to 298–320 ms.

### Fixed

- `d1a.benchmark --data` scored your own records under the training length limit (384-token states) and skipped longer
  ones without saying so, so a file of long documents (pull requests, reports) was scored on its short ones only. It
  now prints how many it skipped, and `--context serving` scores them all, up to what `d1a.serve` accepts.
- Multi-process full fine-tuning on the CPU (FSDP2 over gloo) failed on torch 2.13, which reduces gradients with
  PREMUL_SUM once a divide factor is set; D1A now asks for plain sums, the same result for its factor of 1.

## [0.2.0] - 2026-10-02

The first D1A release: Kev's decision model moved onto Gemma 4, with trained checkpoints, an Apple Silicon backend,
agent tools, model routing, and photos and voice.

### What's new

- **Gemma 4 is the default base.** D1A trains and serves on Gemma 4 E2B and E4B. The Qwen bases Kev was built for
  still work.
- **Better models.** D1A-E4B v0.2 (hybrid) adds Kev's later training stages, Japanese decisions and agent routing on
  top of v0.1: Japanese 0.664 → 0.835, English on new sources 0.680 → 0.704, English on trained sources unchanged
  (0.848), agent-factory routing 0.933.
- **Fast on a Mac.** On Apple Silicon D1A runs on MLX with 8-bit exports that are checked against the PyTorch
  reference, and Gemma 4's per-layer embeddings are read from disk per request: E2B uses 3.0 GB instead of 4.2 GB.
- **Use it from agents.** Ready-made question sets (intake, judge, tier, gate, route) and an MCP server turn each
  decision into one tool call, and D1A can say whether a prompt needs a small, medium or large model.
- **Use it in your own code.** `from d1a import D1A` loads a checkpoint once and answers questions in-process, with no
  server.
- **Photos and voice notes.** `d1a.media` answers the same typed questions about an image or a short audio clip,
  using Gemma 4's own vision and audio encoders (no captioning or speech-to-text). It loads its model on the first
  request and frees it when idle.

### Added

- MLX backend for Gemma 4 (sliding-window and KV-shared layers), quantized MLX exports (`scripts/export_mlx.py`) and
  golden-vector parity checks (`scripts/golden_vectors.py`) ([#39](https://github.com/jonpol01/d1a/pull/39)).
- Gemma 4 per-layer embeddings read from the weight files per request instead of held in memory, with bit-identical
  results: 1.3 GB less for the 8-bit E2B export, 1.5 GB less for E4B (`D1A_PLE_FLASH=0` to decline)
  ([#90](https://github.com/jonpol01/d1a/pull/90)).
- `d1a.D1A`: load a checkpoint once and answer questions in-process ([#47](https://github.com/jonpol01/d1a/pull/47)).
- Agent-factory presets and an MCP server (`d1a.mcp_server`, extra `mcp`): `d1a_intake`, `d1a_judge`, `d1a_tier`,
  `d1a_gate`, `d1a_route`, `d1a_decide` ([#52](https://github.com/jonpol01/d1a/pull/52)).
- `d1a.media`: System One questions about a photo or a voice clip (`POST /v1/systemone/media`, extra `media`)
  ([#83](https://github.com/jonpol01/d1a/pull/83)), loaded on demand and dropped after `--idle-unload` seconds
  ([#85](https://github.com/jonpol01/d1a/pull/85)).
- `d1a.serve`: `--device` with a kernel probe and a startup self-check of the readout
  ([#43](https://github.com/jonpol01/d1a/pull/43)); recent batch latency in `/v1/models`
  ([#48](https://github.com/jonpol01/d1a/pull/48)); a warning when the checkpoint is not calibrated, and `calibrated`
  in `/v1/models` ([#54](https://github.com/jonpol01/d1a/pull/54)).
- Training: resume points for LoRA runs ([#46](https://github.com/jonpol01/d1a/pull/46)); `--extra_suites` to train
  on several frozen suites in one run ([#67](https://github.com/jonpol01/d1a/pull/67)).
- `d1a/backbone.py`: what differs between model families in one place, for the torch and MLX paths
  ([#70](https://github.com/jonpol01/d1a/pull/70), [#77](https://github.com/jonpol01/d1a/pull/77)).
- `scripts/kev_share.py`: how much of D1A is still Kev's code, by area ([#76](https://github.com/jonpol01/d1a/pull/76)).
- Reports: model routing v0.1 findings, English and Japanese ([#56](https://github.com/jonpol01/d1a/pull/56)); Apple's
  Core AI and Foundation Models (macOS and iOS 27) compared with D1A ([#86](https://github.com/jonpol01/d1a/pull/86)).
- README: GIFs of the eleven playground demos, and animated architecture diagrams generated by
  `docs/arch/make_svgs.py` ([#44](https://github.com/jonpol01/d1a/pull/44),
  [#83](https://github.com/jonpol01/d1a/pull/83), [#84](https://github.com/jonpol01/d1a/pull/84)).
- CI: Python 3.12 and 3.13, MLX tests on Apple Silicon, a lint pass, a package build and install check, and this
  release workflow.

### Changed

- The package, modules and settings are D1A's: `kev` → `d1a`, `KEV_*` → `D1A_*` environment variables, model name
  `d1a-latest` (`kev-latest` and `jev-latest` still answer) ([#1](https://github.com/jonpol01/d1a/pull/1)).
- The default training base is Gemma 4 E2B, and `backend="auto"` picks MLX for Gemma 4 on Apple Silicon
  ([#1](https://github.com/jonpol01/d1a/pull/1), [#39](https://github.com/jonpol01/d1a/pull/39)).
- Model routing sends a prompt to the small tier only at p(small) ≥ 0.7, so close calls go up a tier
  ([#55](https://github.com/jonpol01/d1a/pull/55)).
- The frozen suites are pinned at kev-suites `cc4bac8`, which adds the hard-v1 and documents-v1 training partitions
  ([#68](https://github.com/jonpol01/d1a/pull/68)).

### Fixed

- A proxy in front of `d1a.serve` (the playground's) could reuse a connection the server had just closed
  (ECONNRESET): the server now keeps idle connections for 75 s ([#53](https://github.com/jonpol01/d1a/pull/53)).
- A GPU whose driver or PyTorch build cannot run a kernel no longer crashes the server on its first request: the
  device is probed at startup and the server falls back to the CPU ([#43](https://github.com/jonpol01/d1a/pull/43)).
- A non-finite loss or gradient skips its batch instead of ending the run; three in a row still end it, because then
  the weights themselves are broken ([#46](https://github.com/jonpol01/d1a/pull/46)).

### Removed

- Kev's research tooling, deployments and records that D1A does not use: 26 files, about 5,000 lines
  ([#69](https://github.com/jonpol01/d1a/pull/69)).

### Models tested with this release

| Checkpoint | Hugging Face tag |
|---|---|
| D1A-E4B v0.2 (hybrid), PyTorch | `JohnP1/d1a-e4b@v0.2-hybrid` |
| D1A-E4B v0.2 (hybrid), MLX 8-bit | `JohnP1/d1a-e4b-mlx-q8@v0.2-hybrid` |
| D1A-E2B v0.2.1 (2 epochs, calibrated), PyTorch | `JohnP1/d1a-e2b@v0.2.1-2epoch-calibrated` |
| D1A-E2B v0.2, MLX 8-bit | `JohnP1/d1a-e2b-mlx-q8` |

### Upgrade notes (from Kev)

- Rename imports and commands: `python -m kev.serve` → `python -m d1a.serve`, `from kev...` → `from d1a...`, and
  `KEV_*` environment variables → `D1A_*`.
- On Apple Silicon the server now uses MLX for Gemma 4 by default; set `D1A_BACKEND=torch` for PyTorch MPS.
- Optional features are extras: `serve` (HTTP server), `mlx` (Apple Silicon), `media` (photos and voice), `mcp`
  (agent tools).

## 0.1.0 - 2026-09-30

Not released. The starting point: Kev imported at `jaredpalmer/kev@0fe8fc9` with the Gemma 4 port from
`jonpol01/kev@13374b3`.

## Model versions

Checkpoints on Hugging Face, newest first. Each version continues training the previous one of its size. Load one as
`repo@tag`; the MLX repositories (`-mlx-q8`) carry the same versions for Apple Silicon.

| Model | Tag | What it adds | Older tag name |
|---|---|---|---|
| JohnP1/d1a-e4b | `v0.4` | pull-request labeling round 2: severity, Japanese, a blast-radius answer no longer biased to "broad"; earlier skills 1–3 points lower than v0.3 | |
| JohnP1/d1a-e4b | `v0.3` | pull-request labeling (type, blast radius, severity), English and Japanese | |
| JohnP1/d1a-e4b | `v0.2` | Kev's later stages (dates, missing evidence, hard, tool-use and long-document decisions), Japanese (JGLUE), agent routing | `v0.2-hybrid` |
| JohnP1/d1a-e4b | `v0.1` | general decisions (decision-v7, 2 epochs, calibrated) | `v0.1.1-2epoch-calibrated` |
| JohnP1/d1a-e2b | `v0.2` | general decisions, 2 epochs, calibrated | `v0.2.1-2epoch-calibrated` |
| JohnP1/d1a-e2b | `v0.1` | general decisions, 1 epoch | `v0.1-1epoch` |

Retired, kept for reproducibility: JohnP1/d1a-e4b-routing and its MLX build (now part of v0.2), and
JohnP1/d1a-e4b-pr-labeler-mlx-q8 (now v0.3).

[Unreleased]: https://github.com/jonpol01/d1a/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jonpol01/d1a/releases/tag/v0.2.0
