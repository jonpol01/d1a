# D1A: notes for coding agents

D1A is a small decision model on Gemma 4, built on Kev (https://github.com/jaredpalmer/kev, Apache-2.0). A causal LM
backbone + LoRA runs one prefill pass under a block-causal mask; a pointer head scores each option's end token against
the question's `<decide>` token. README.md is the front page and docs/ is the user guide; docs/UPSTREAM.md records what came from Kev.

## Layout
- `d1a/` the package, by subpackage (#60; the old flat module paths were removed in 0.5, and `d1a/_layout.py` still maps
  them for scripts/kev_share.py and scripts/check_license.py):
  - `core/`: `api.py` (the System One request and answer schema), `encoding.py` (how a request is packed into tokens, the
    context limits, rows), `head.py` (the pointer head), `versions.py`, `settings.py` (every `D1A_*` setting, its default and
    reader), `hygiene.py` (what must never be published: local paths, AI-attribution lines);
  - `backends/`: `torch.py` (DecisionModel, masks; re-exports encoding and the head), `mlx.py` (Apple Silicon backend for Qwen3.5
    and Gemma 4, and the MLX export folders `scripts/export_mlx.py` writes), `checkpoint.py` (loading; `D1A_*` load options
    are read only by `LoadOptions.from_env`), `backbone.py`, `device.py`, `shared_prefix.py` (Qwen3.5 training through a
    shared state prefix), `fused_qwen35.py`, `cuda_graphs.py`, `q4_0.py` (llama.cpp's Q4_0 grid, in numpy);
  - `serving/`: `serve.py` (FastAPI System One server), `media.py`, `lib.py` (in-process `from d1a import D1A`),
    `openai_compat.py` (OpenAI-format chat and embeddings, #274);
  - `learning/`: `feedback.py` (the decision log, outcomes, the outcome calibrator and its promotion gate), `settings.py` (the
    `D1A_LEARNING` settings file), `loop.py` (the trained manifest, replay and the tick), `flags.py` (learned decision flags,
    #237), `rules.py` (served decode rules, #252);
  - `training/`: `train.py` (trainer; default base Gemma 4 E2B), `data.py`, `resume.py` (training resume points),
    `recipe.py` (versioned YAML stages run through d1a.training.train), `calibrate.py` (writes a checkpoint's temperature,
    or with --use-case one use case's, refusing fit rows that are not held out from its training), `study.py` (recipe -> calibrate -> evaluate -> report ->
    publish; publishing refuses an uncalibrated checkpoint), `composition.py`, `temperature_gate.py` (the rule a refitted
    temperature must pass);
  - `eval/`: `benchmark.py` (scores a checkpoint or a remote endpoint on a suite), `metrics.py`, `predictors.py`,
    `suite.py` (frozen suites, Hub-mirrored partitions), `suites.py` (D1A's own suites);
  - `agents/`: `presets.py`, `mcp_server.py`, `claude_code_gate.py` (the Claude Code PreToolUse hook), `demo.py` (one task
    through an agent's day).
- `evals/` frozen suites from Kev (manifests pin every partition's sha256; never edit them in place). `evals/d1a/` holds D1A's
  own suites (`d1a/eval/suites.py`): a manifest per dataset pins its Hub commit and each partition's sha256 and role
  (train/eval); the text stays in the private dataset. Re-pin with `python -m d1a.eval.suites freeze`, never by hand.
- `scripts/` suite builders and one-off tools;
  `clients/` the dependency-free Python and JS clients.

## Commands
- Env: `uv sync --extra serve` (Python 3.12 or 3.13).
- Unit tests (no weights): `uv run python -m pytest tests --unit -q`, every test file but `tests/conftest.py`'s `OUTSIDE_UNIT_TESTS`;
  a new test file runs in CI without being listed anywhere.
- License and provenance: `python scripts/check_license.py` (add `--skip-upstream` offline). See CONTRIBUTING.md.

## Rules
- A file taken from Kev and changed keeps its "Modified from Kev" header and stays listed in docs/UPSTREAM.md.
- Do not brand anything "Kev"; product, package, CLI and model names are D1A.
- Every setting is documented in docs/CONFIGURATION.md: each `D1A_*` variable the package reads and each server flag.
  A new setting adds its entry there; tests/test_configuration_doc.py fails until it does.
- A change adds a fragment, `changes/<PR number>-<slug>.md` (format in `changes/README.md`), instead of editing
  CHANGELOG.md; `scripts/release_notes.py assemble` writes the version's section at release.
- Never load a model or start training in a unit test (tiny random models built or committed under tests/ are fine).
- `tests/test_conformance.py` pins the answers of the committed tiny checkpoint (`tests/golden/tiny-gemma4`). A change
  that moves them on purpose rebuilds it (`tests/golden/build_tiny_gemma4.py`) in the same PR and says why.

## Development machine resources (mandatory)
The development machine's disk and memory are shared with other work, and a long job once ran its disk full and killed
the shell. The resource guard enforces the rules below; this machine's numbers and paths are in `AGENTS.local.md`
(untracked) where the checkout has one.
- Every long job (training, benchmark, gate, calibration, export, swe-verifier scoring) starts behind the guard's
  preflight: `d1a_preflight <name> <new GB> <peak RAM GB> [--dtype fp32 --params-b N] || exit 1`. It refuses when free
  disk is below the new data plus a headroom, when the peak would not fit the free memory less a margin, and while the
  guard holds paused jobs. Size downloads with `d1a_hf_size <repo> <rev>`; never guess.
- fp32 models of 4B or more need `--allow-fp32-big`: more free disk, no other D1A job, and the overnight window only.
- The guard pauses D1A jobs (SIGSTOP) when free disk runs low or swap grows fast while one runs, stops them (SIGTERM)
  when the disk is nearly full, and resumes them once both have recovered for a while.
- Delete Hugging Face cache entries only with `scan_cache_dir().delete_revisions(...).execute()`, never `rm` on a repo folder:
  the cache shares blobs (`hub/blobs/<xx>/<hash>` + `.refs`), so a hand delete frees nothing and leaves stale `.refs` behind.
  Where the tool no longer lists a repo, delete one store file only after its `.refs` names nothing else, and log it.

## Quality bar (mandatory)
The maintainer, 2026-10-06: every PR and every deploy is checked against the test suite, with numbers; nothing is downgraded or
stripped off, and new features come with their tests.
- **Numbers in every PR body**, from the PR's own head:
  - the unit run and the golden vectors (`tests/test_conformance.py`);
  - for a change to the model, loading or serving code, and before a playground pin move: **`scripts/quality_gate.py`**
    (required; its table and PASS line go in the PR). It runs base and head on real weights, interleaved: every demo
    example (`--playground <d1a-playground checkout>`) and the labeler replay (`runs/labeler-replay`, private, never
    committed), with flips, max |dp| and latency against a base/base floor, plus frozen suites with `--suites`. A change
    to how a request field is served also runs with the field sent: `--use-case routing` for the use-case temperature
    path (the routing requests carry `"use_case"`, both servers get the same requests). A decode rule (`rules` in the
    `D1A_LEARNING` file) runs with `--head-learning <file> --labeler-use-case pr-labeler --expect-flips <question>`:
    flips on that question only, every request the rule applies to flipped, max |dp| 0 everywhere, and 0 flips without
    the file. For a
    torch-only path, also `scripts/equivalence/real_weights.py --interleave`;
  - **the `quality-gate` check: off by default (the maintainer, 2026-10-06), turned on on demand,** for a pull request whose
    severity calls for it, and only when the local machine is free to run the gate (no training, scoring or other GPU
    work), so it never sits pending. To turn it on, run `gh workflow enable quality-gate` and add `quality-gate` to the
    `main` ruleset's required status checks; undo both afterwards. When on, a
    pull request that changes a module the model server loads, or the dependencies
    (`scripts/gate_required.py`, from the server's own imports), cannot merge until `scripts/quality_gate.py --post-status`
    passes on its exact head (`.github/workflows/quality-gate.yml` leaves it pending; any other pull request gets it as
    success). A new head starts pending again. A playground pin move posts to that pull request
    (`--post-status jonpol01/d1a-playground@<head>`, which checks that its mini.sh pins the gated head);
  - after a pin move, the serving host's demo smoke test (`./mini.sh update` ends with `scripts/demo_smoke.mjs` in
    jonpol01/d1a-playground).
- **No downgrade.** The answers stay identical, or the PR shows they are better on held-out data. Accuracy and
  calibration never drop, and latency stays within run-to-run noise (measured interleaved, against main).
- **No stripping.** No endpoint, demo, example, CLI flag or feature disappears silently. A removal updates a failing
  test on purpose, and the PR says why.
- **New features ship with their tests** in the same PR.
- **Fine-tunes keep every skill** (#167; `recipes/README.md`; the maintainer, 2026-10-07): a run that starts from a trained
  checkpoint trains on or replays every training source, as `d1a.eval.suites.train_sources()` lists them: hard-v1,
  devtools-v1, documents-v1 and every D1A train partition (PR labels incl. blast and Japanese, routing, JGLUE), plus
  decision-v7 through the trainer. Both mix tools refuse a new mix that leaves one out, and so does `d1a.training.train`
  with `--init_from` (#211): it reads the mix's sidecar `<data>.json` before any weights load. A source left out on
  purpose takes `--allow_missing_sources <source,...|all> --reason "<why>"`, recorded in `training_config.json`. Its gate
  adds `--all-suites` (below).
- **Every model is judged on every suite** (the maintainer, 2026-10-07). A new or changed checkpoint runs
  `scripts/quality_gate.py --head-run <new> --all-suites`. That scores, against the checkpoint it replaces:
  - the five card suites;
  - every D1A suite's evaluation partitions, at the serving context;
  - every demo and the labeler replay.
  The decision is made on that one scorecard, across every use case, never on one of them. Name each regression in the
  PR, and fix it in a follow-up rather than hiding it.
  `scripts/decide.py --gate <the gate's --out> --labeler <live labeler dump> --human <owner labels>` computes it (#202):
  - INCOMPLETE when any part of the gate is missing (never a pass);
  - VETO: V1, a card suite with Δ < −2 or CI lower < −4; V3, latency above the floor + 0.015; V4, a suite of n ≥ 150 with
    Δ ≤ −5 and its CI below 0;
  - BETTER needs no veto, a pooled Δ ≥ 0, and either a pooled lower bound > 0 or ≥ 3 significant wins on distinct sources
    with no significant loss;
  - V2 is a person reading the safety demos' flips before any deploy.
  Every significant loss is a follow-up issue.

## Project board (mandatory)
Every piece of D1A work is tracked on GitHub Project #14 "D1A" (https://github.com/users/jonpol01/projects/14), and its card
moves at every step, without being asked:
- Columns: **Backlog** (not planned yet) → **Ready** (scoped, planned next) → **In progress** → **In review** (PR open,
  waiting for CI and review) → **Done** (merged or closed).
- New work, a finding or a follow-up becomes an issue with a milestone and a card in Backlog or Ready as soon as it is agreed.
- Starting work moves its card to In progress. Opening a PR adds the PR's card to In review and moves the issue with it.
  Merging or closing moves both to Done, and closes or updates the issue the PR completes.
- Every PR gets a card. Before ending a session, check that no open PR sits outside In review and no merged PR or closed
  issue outside Done.

## Pull requests (mandatory)
Every PR is watched from the moment it opens until it is merged; the watch ends only with the merge:
- CI green and the reviewer (hermes-prbot) clean on the **exact head** → merge, then move the cards (above).
- Findings → verify them against the code, fix with a test that fails without the fix, push, reply on the PR, and keep
  watching for the re-review of the new head. Never leave a reviewed PR waiting.
