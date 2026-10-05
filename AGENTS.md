# D1A: notes for coding agents

D1A is a small decision model on Gemma 4, built on Kev (https://github.com/jaredpalmer/kev, Apache-2.0). A causal LM
backbone + LoRA runs one prefill pass under a block-causal mask; a pointer head scores each option's end token against
the question's `<decide>` token. README.md is the user guide; docs/UPSTREAM.md records what came from Kev.

## Layout
- `d1a/` the package: `model.py` (DecisionModel, encode, masks), `train.py` (trainer; default base Gemma 4 E2B),
  `serve.py` (FastAPI System One server), `benchmark.py` (scores a checkpoint or a remote endpoint on a suite),
  `checkpoint.py` (loading; `D1A_*` load options are read only by `LoadOptions.from_env`), `suite.py` (frozen suites,
  Hub-mirrored partitions), `api.py` (request schema), `resume.py` (training resume points), `shared_prefix.py` (Qwen3.5 training through a shared state prefix),
  `mlx_model.py` (Apple Silicon backend for Qwen3.5 and Gemma 4, and the MLX export folders `scripts/export_mlx.py` writes), `calibrate.py` (writes a checkpoint's temperature, refusing fit rows that are not held out from its training).
- `evals/` frozen suites from Kev (manifests pin every partition's sha256; never edit them in place). `evals/d1a/` holds D1A's
  own suites (`d1a/suites.py`): a manifest per dataset pins its Hub commit and each partition's sha256 and role
  (train/eval); the text stays in the private dataset. Re-pin with `python -m d1a.suites freeze`, never by hand.
- `scripts/` suite builders and one-off tools;
  `clients/` the dependency-free Python and JS clients.

## Commands
- Env: `uv sync --extra serve` (Python 3.12 or 3.13).
- Unit tests (no weights): `uv run python -m pytest tests --unit -q`, every test file but `tests/conftest.py`'s `OUTSIDE_UNIT_TESTS`;
  a new test file runs in CI without being listed anywhere.
- License and provenance: `python scripts/check_license.py` (add `--skip-upstream` offline). See CONTRIBUTING.md.

## Rules
- A file taken from Kev and changed keeps its "Modified from Kev" header and stays listed in docs/UPSTREAM.md.
- Do not brand anything "Kev"; product, package, CLI and model names are D1A. `kev-latest` is only a compatibility alias.
- Never load a model or start training in a unit test (tiny random models built or committed under tests/ are fine).
- `tests/test_conformance.py` pins the answers of the committed tiny checkpoint (`tests/golden/tiny-gemma4`). A change
  that moves them on purpose rebuilds it (`tests/golden/build_tiny_gemma4.py`) in the same PR and says why.

## Quality bar (mandatory)
John, 2026-10-06: every PR and every deploy is checked against the test suite, with numbers; nothing is downgraded or
stripped off, and new features come with their tests.
- **Numbers in every PR body**, from the PR's own head:
  - the unit run and the golden vectors (`tests/test_conformance.py`);
  - for a change to the model, loading or serving code: real-weight answers and interleaved latency
    (`scripts/equivalence/real_weights.py --interleave`);
  - the demo smoke test (`scripts/demo_smoke.mjs` in jonpol01/d1a-playground) and the labeler replay
    (`runs/labeler-replay`).
- **No downgrade.** The answers stay identical, or the PR shows they are better on held-out data. Accuracy and
  calibration never drop, and latency stays within run-to-run noise (measured interleaved, against main).
- **No stripping.** No endpoint, demo, example, CLI flag or feature disappears silently. A removal updates a failing
  test on purpose, and the PR says why.
- **New features ship with their tests** in the same PR.

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
