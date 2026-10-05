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

- Conformance against committed golden vectors (#64): `tests/golden/tiny-gemma4/` holds a tiny Gemma 4 checkpoint
  (weights in git, 556 KB, built by `tests/golden/build_tiny_gemma4.py`) and `golden.json`, with 40 requests and 154
  questions. `tests/test_conformance.py` requires its token ids exactly and every probability to 1e-5, through the model,
  `d1a.serve` and `scripts/golden_vectors.py compare`.
- `scripts/equivalence/` (`api.py`, `metrics.py`, `benchmark.py`, `suite.py`): each runs a rewritten module and the same
  module at a git commit (`--ref`; the default is the last commit before its rewrite) on generated inputs, and stops at
  the first difference: types, float bits, key order, files byte for byte, error messages.
- `tests/test_tiny_checkpoint.py` (in CI): a random 6-layer Gemma 4, with sliding and KV-shared layers, goes through
  `d1a.train`, `d1a.checkpoint` and `d1a.serve` with no download. Every scoring path (packed, rows, prefix miss and hit,
  serving batch) and the served answers must match an independent reference: each question as a plain causal row
  through transformers. Ten mutation checks must fail: off-by-one readouts, a leaky question mask, the sliding window
  ignored, positions that do not restart, a dropped `<bos>`, temperature ignored, and three misread answer types (#33).

### Changed

- `d1a/suite.py` is rewritten in D1A's own code, no longer derived from Kev. The pins, the partition checks, the Hub
  mirror rules, the file formats (byte for byte) and the training-source guard are unchanged (`scripts/equivalence/suite.py`).
  Its tests are rewritten too, as `tests/test_suite.py`.
- CI runs every test file (`pytest tests --unit`) except the few `tests/conftest.py`'s `OUTSIDE_UNIT_TESTS` lists with
  why, instead of a list naming each file: a new test file runs without being listed, and two pull requests adding test
  files no longer conflict on that list. The release runs the same command. CI keeps the Hugging Face cache (the pinned
  tokenizers the tests read) between runs. `docs/UPSTREAM.md` lists rewritten files one per line.
- `d1a/api.py` is rewritten in D1A's own code, no longer derived from Kev. The schema, the text the model reads, the
  answers and the validation errors are unchanged: they were checked identical to the previous version on 166,718
  generated requests, distributions and dates. Its tests are rewritten too, as `tests/test_system_one.py`.
- `d1a/metrics.py` is rewritten in D1A's own code, no longer derived from Kev. Every figure, report (key order included),
  fitted temperature and bootstrap interval is bit-identical to the previous version: checked on about 300,000 generated
  calls and on saved benchmark rows. Its tests are rewritten too, as `tests/test_metrics.py`.
- `d1a/benchmark.py` is rewritten in D1A's own code, no longer derived from Kev. Its rows, reports, the files it writes
  (byte for byte), its failure and skip rules and its command line are unchanged: checked against the previous version
  on about 34,000 generated cases and end to end on a frozen suite. Its tests are rewritten too, as
  `tests/test_benchmark.py`.

### Removed

- `python -m d1a.suite`, the suite freezer inherited from Kev, with `d1a/contrastive.py` and the record generator of
  `d1a/composition.py`. D1A loads its frozen suites as data and never rebuilds them; `paired_flip` moved to
  `d1a.benchmark`, and the rule shapes that `d1a.suite.validate_training` checks stay. See docs/removed-tools.md (#65).
- The `d1a.data` converters only that freezer read (13 sources; `build()` keeps its six defaults) and
  `scripts/longdoc_serving.py`, which ran only through Kev's removed Modal app (#65).

### Fixed

- `scripts/golden_vectors.py compare` no longer crashes on a one-option question (a choice with one criterion or a
  one-level score), which has no second-best probability to measure a flip margin against.

## [0.3.0] - 2026-10-05

**D1A learns from outcomes.** Every decision can be logged with the model that made it; when the real outcome comes back
(the tests passed, a maintainer corrected a label), D1A recalibrates on it at once and promotes a retrained model only
through a statistical gate. Measured first on a coding-agent verifier: three rounds of 300 outcomes cut the calibration
error from 0.068 to 0.017, and the gate kept the model each time retraining was not clearly better. This release also
makes D1A more its own: Kev's playground, study harness, training-history suites and suite builders are gone (70% of the
code is Kev-derived, from 77%), and Gemma 4 runs about a third faster on torch and 20–30% faster on Apple Silicon.

### Added

- **`recipes/swe-verifier`: D1A as a verifier of coding-agent runs** (issue, patch and the end of the run → P(the patch resolves the
  issue)), on public SWE-agent runs (nebius/SWE-agent-trajectories, CC-BY-4.0) held out by repository: zero-shot and LoRA scoring,
  held-out evaluation with repository-bootstrap intervals and within-issue AUROC, best-of-k, a training-data filter, anytime
  restarts under a budget, a split-leakage study (random runs vs by issue vs by repository), the self-improvement rounds through
  `d1a.feedback`, competition-harness input (`from_swegemma.py`), verified references, and the paper draft.
- **`GET /metrics`** (`d1a.serve`): Prometheus text format: whether the model is loaded, requests and batches served, queue depth, recent batch latency (p50, p95, max), prefix-cache hits, misses and states, peak process memory and the model's device memory. Like `/v1/models`, it does not load an unloaded model.
- **README: the self-learning loop's measured rounds** (calibration error 0.068 → 0.017; the gate kept the model when retraining was not clearly better), also on the diagram.
- **Self-learning as a service** (`d1a.serve`): `D1A_FEEDBACK_LOG` logs every decision and returns a `decision_id`; `POST /v1/feedback` records its real outcome; `D1A_OUTCOME_CALIBRATOR` applies `d1a.feedback`'s calibrator to yes/no answers and re-reads it when the file changes. `/v1/models` reports the learning state.
- **README: Self-Learning.** A section on how D1A learns from outcomes (usage, CLI, the verifier's measured numbers), step 5 in How It Works, and an animated diagram of the loop (`docs/arch/learning-{light,dark}.svg` from `make_svgs.py`).
- **`d1a.feedback`: learning from outcomes.** A `FeedbackLog` records each decision and, later, what actually happened
  (a patch passed its tests, a label was corrected); `OutcomeCalibrator` refits each yes/no question on those outcomes
  (Platt scaling on the logit, which also corrects a shifted base rate that one temperature cannot); `records()` turns
  them into training data for a small LoRA update; `gate()` promotes a candidate only when its held-out log loss is
  lower with a bootstrap interval clear of zero and no frozen suite regressed. `python -m d1a.feedback status | records |
  calibrate`. Fitted values match scikit-learn's logistic regression on D1A-E4B v0.4's verifier outputs.
- **`d1a.versions`: the newest released checkpoint in one place.** `latest("JohnP1/d1a-e4b-mlx-q8")` gives
  `JohnP1/d1a-e4b-mlx-q8@v0.4`; the Space takes its default from it. `tests/test_versions.py` fails when `LATEST` and
  the first row per model of the "Model versions" table below disagree, so a release cannot leave a default behind.
- **A Hugging Face Space, ready to publish** (`space/`, `space/push.sh`; not hosted yet: a Gradio or ZeroGPU Space needs HF PRO): model routing,
  guardrails, tool-call gating, the PR labeler (paste a GitHub PR link) and zero-shot photo and voice checks on
  D1A-E4B v0.4. One `d1a.media.MediaModel` answers all of them: `MediaModel.probs` now also scores a text-only record
  (media None) on the same weights.
- **Kev's eval suites as D1A suites**: `external` (semif-v1, typesafe-v1, wanli-v1, wanli-v2) and `night2` (dates,
  unknowable facts, assertions) are `evals/d1a/external` and `evals/d1a/night2`, with the data byte for byte in the
  private dataset `JohnP1/d1a-evals`. `d1a.benchmark --data evals/d1a/external:semif-v1` scores the same as the Kev suite
  did (D1A-E4B: identical probabilities on all 252 rows). `python -m d1a.suites freeze --provenance` records where a
  suite's data came from.
- **D1A suites** (`d1a/suites.py`, `evals/d1a/`). Our own datasets, pinned: a manifest per dataset holds its Hub commit
  and each partition's sha256, record count and role (train or eval), while the text stays in the private dataset.
  `evals/d1a/<suite>:<partition>` works wherever `--data` does (`d1a.train`, `d1a.benchmark`). The partition is fetched
  at the pinned commit and checked before use, and `d1a.train` refuses an eval partition before loading the model.
  `pr-labels`, `ja-jglue` and `routing` are frozen. The PR-labeler recipe reads them through `recipes/pr-labeler/mix.py`,
  which replaces the job's inline mixing script (byte-identical mixes, round 2's included) and records the inputs'
  hashes, the parameters and the mix's sha256 beside it. `EXTRA` now names partitions (`train-ja train-blast`), and
  `DATA` is gone.
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

- **Served PR requests about 20% faster on Apple Silicon** (MLX). The question rows ran through all of Gemma 4's
  layers; its 18 KV-shared layers now run only at the positions the pointer head reads, each as a length-1 row at its
  own rope offset with an explicit causal (and sliding-window) mask over the shared keys
  (`MLXDecisionModel._picked_hidden`). D1A-E4B 8-bit on an M1 Max, 30 PR requests: 2,503 -> 1,955 ms per request
  (p50; the question rows 1,689 -> 1,013 ms). Exact in fp32 (D1A-E2B, 90 questions: max |dp| 2.3e-6); at 8 bits it
  moves answers less than running the rows one at a time instead of batched already does (max |dp| 0.015 vs 0.019).
- **Gemma 4 trains and scores about a third faster on torch** (the path HF Jobs uses). Its last 18 of 42 layers
  (E4B; KV-shared) take keys and values from earlier layers, so in the packed forward they now run only over the
  positions the pointer head reads (each question's `<decide>` and option ends, about 10-20 per record) instead of the
  whole document (`d1a.backbone.Gemma4.picked_hidden`). D1A-E4B v0.4 on 2 PR records (Apple M1 Max, bf16): scoring
  4.84 -> 3.00 s, a LoRA training step with checkpointing 12.9-14.0 -> 8.5-8.9 s. Exact in fp32 (D1A-E2B, 120
  questions: max |dp| 1.8e-6, gradients equal to 1e-5); in bf16 it moves answers less than changing the batch size
  does (438 questions: mean |dp| 0.0011 and 2 flips, against 0.0060 and 5 for batch 1 vs 2).
- **Model families behind one interface.** `d1a.backbone` now also decides whether the MLX backend runs a base (Gemma 4
  and Qwen3.5; it was a `model_type` list in `d1a.checkpoint`) and loads Qwen3.5's CUDA serving kernels
  (`d1a.fused_qwen35`, `d1a.cuda_graphs`) through `Qwen35.serve_cuda`, so Gemma 4 is the main path and Qwen3.5 a
  plug-in. `d1a.model.is_hybrid` and `d1a.model.sliding_window` are gone: use `d1a.backbone.for_config(config)`.
  Answers and latency are unchanged.
- **PR-length requests are about 30% faster on Apple Silicon, with identical answers.** The state pass (the document,
  run once before the questions) now stops before Gemma 4's KV-shared layers: those layers keep no cache of their own
  and nothing reads the state's own outputs, so for the state they were pure waste (18 of D1A-E4B's 42 layers). On an
  M1 Max, 60 held-out pull requests (median 1,584 state tokens): state pass 3,470 -> 2,137 ms, whole request 4,465 ->
  3,082 ms (medians), every probability bit-identical; photos, voice and video too (a 16-frame clip's state pass 1,805
  -> 1,079 ms). Other backbones run the full pass as before.

- Downloading a model from the Hub skips an export's `media/` folder (1 GB) until a photo or voice request needs it.

### Removed

- **Kev's suite builders** (about 6,900 lines, 93-100% Kev's): the hard-v1, breadth-v1, devtools-v1, longdoc-v1,
  documents-v1/v2 and transfer-v9 builders and generators, `build_binding_diagnostic.py`, `devtools_v1_licences.json`,
  and the tests that only exercised them. The suites stay as frozen data; `tests/test_frozen_suites.py` (new, in CI)
  checks every partition in git against its manifest's sha256 and record count, plus the eval-only suites' contracts.
  `docs/removed-tools.md` says which commit rebuilds each suite.

- **Kev's training-history suites** (20 of them: `decision-v1`, `public-pool-v5`/`v6`, `round3`-`round15`, `sft-v1`,
  `sft-v2` and its rounds, `transfer-v1`, `v3`, `v5`, `v6`, `v8`) and six builders that only served retired or moved
  suites. Nothing D1A trains or evaluates on depends on them; docs/removed-tools.md says how to restore them.
- **Full-weight training.** `d1a.train --full_ft`, its FSDP2 multi-GPU mode under torchrun and the snapshot options
  (`--snapshot_fractions`, `--snapshot_every_steps`, `--snapshot_dir`) are gone, with `d1a/full_ft.py`,
  `scripts/interpolate_checkpoint.py` and `scripts/merge_lora_checkpoint.py`. Training is single-process LoRA, and
  LoRA training is bit-identical to before. A full-weight checkpoint is refused at load. Resume points moved to D1A's
  own `d1a/resume.py`, one file per point: a point written before this change cannot be resumed (finish that run on the
  older version). `--shared_prefix` now defaults to 0 (it was on only with `--full_ft 1`). `training_metrics.json`
  drops `world_size` and `snapshots`.
- **Kev's unused training options.** `d1a.train` drops `--label_smoothing`, `--brier_w`, `--focal_gamma`, `--ord_w`,
  `--perm_kl`/`--perm_frac`, `--anchor`/`--anchor_w`/`--anchor_sources`, `--option_isolation` and
  `--special_embeddings`: no D1A checkpoint used any of them. Training with the remaining options is bit-identical
  (same weights and losses on tiny Qwen3.5 and Gemma 4 bases). A checkpoint trained with option isolation is now
  refused at load; token-trained adapters still load. [docs/removed-tools.md](docs/removed-tools.md) describes each
  option and how to restore it.
- Kev's in-repo `playground/` (Next.js: request editor, packed vs separate, option permutation, chess) and the
  `tools/review` label-review page, about 14,000 lines, and `JevPredictor` in `d1a/predictors.py`, which ran the
  playground's Jev script. The demo app is [jonpol01/d1a-playground](https://github.com/jonpol01/d1a-playground);
  [docs/removed-tools.md](docs/removed-tools.md) records what both tools did and how to restore them. The
  `/v1/systemone/separate` and `/v1/systemone/permute` endpoints stay.

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

[Unreleased]: https://github.com/jonpol01/d1a/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jonpol01/d1a/releases/tag/v0.3.0
[0.2.0]: https://github.com/jonpol01/d1a/releases/tag/v0.2.0
