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
- `evals/` frozen suites from Kev (manifests pin every partition's sha256; never edit them in place).
- `scripts/` suite builders and one-off tools;
  `clients/` the dependency-free Python and JS clients.

## Commands
- Env: `uv sync --extra serve` (Python 3.12 or 3.13).
- Unit tests (no weights): the list in `.github/workflows/ci.yml`.
- License and provenance: `python scripts/check_license.py` (add `--skip-upstream` offline). See CONTRIBUTING.md.

## Rules
- A file taken from Kev and changed keeps its "Modified from Kev" header and stays listed in docs/UPSTREAM.md.
- Do not brand anything "Kev"; product, package, CLI and model names are D1A. `kev-latest` is only a compatibility alias.
- Never load a model or start training in a unit test.
