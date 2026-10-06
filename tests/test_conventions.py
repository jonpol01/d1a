# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); dropped the published-claims check and the removed space/ app; modal_app.py left the scanned sources; the rules of the removed Kev research modules (experiment, rounds) left with them, and calibration's moved to d1a.training.calibrate; CI runs every test file but tests/conftest.py's OUTSIDE_UNIT_TESTS (pytest tests --unit).
"""Source conventions: facts that have one canonical home must not be re-derived elsewhere.

Each rule is (what it guards, regex, files allowed to match). A failure means a second copy of a rule that already has
a home; call the canonical helper instead.
Run: uv run python -m pytest tests/test_conventions.py -q
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("d1a", "scripts", "tests")

RULES = [
    ("head.pt is read and written through d1a.backends.checkpoint (Meta, read_meta, write_meta)",
     r"torch\.(load|save)\([^\n]*head\.pt", {"d1a/backends/checkpoint.py", "tests/test_checkpoint.py"}),   # the test writes a hostile head.pt by hand
    ("D1A_DTYPE/D1A_MERGE/D1A_ATTN/D1A_LORA_SCALE/D1A_TEMPERATURE/D1A_BACKEND/D1A_CUDA_GRAPHS are read only by LoadOptions.from_env",
     r"environ(\.get)?\(?\[?\s*\"D1A_(DTYPE|MERGE|ATTN|LORA_SCALE|TEMPERATURE|BACKEND|CUDA_GRAPHS)\"", {"d1a/backends/checkpoint.py"}),
    ("whether a checkpoint is a LoRA adapter or full weights, and where its shards are, is d1a.backends.checkpoint.Checkpoint.full / shards (the loader rule)",
     r"glob\(\"model\*\.safetensors\"\)|adapter_config\.json\"\)\.exists\(\)", {"d1a/backends/checkpoint.py"}),
    ("a checkpoint becomes a model only through d1a.backends.checkpoint (Checkpoint.load picks the torch or MLX implementation)",
     r"MLXDecisionModel\(|merge_lora\(", {"d1a/backends/checkpoint.py", "d1a/backends/mlx.py", "tests/test_mlx.py"}),
    ("option keys come from d1a.core.api.question_keys",
     r"\[\s*\"false\"\s*,\s*\"true\"\s*\]|\[str\(i\) for i in range\(len\(", {"d1a/core/api.py", "tests/test_system_one.py", "tests/test_tiny_checkpoint.py"}),   # these tests pin the contract (the second re-derives it on purpose)
    ("the training context is d1a.core.encoding.MAX_STATE/MAX_BRANCH/MAX_PACKED, lifted only through d1a.core.encoding.training_context (d1a.eval.suite.CONTEXT in manifests), and d1a.core.encoding.fits",
     r"(?<![\w.])(>|<=|>=|<)\s*2048\b|\b2048\s*(<|>)|max_(branch|state|packed)\"?\s*[=:]\s*\d{3,}", {"d1a/core/encoding.py"}),
    ("the serving / long-state limits (SERVE_MAX_*, ROW_PASS_TOKENS, MAX_TRAIN_STATE) and the pre-64k aliases the frozen suites' builders "
     "rebuild byte for byte with (SERVE_MAX_*_8K, MAX_TRAIN_STATE_8K) are defined only in d1a.core.encoding",
     r"^\s*(SERVE_MAX_(STATE|BRANCH|PACKED)|ROW_PASS_TOKENS|MAX_TRAIN_STATE)(_8K)?\s*(=|,[^\n=]*=)|(?<![\w.])7552\b", {"d1a/core/encoding.py"}),
    ("the serving contexts manifests record (SERVING_CONTEXT, and SERVING_CONTEXT_8K for the suites frozen before 64k states) are defined only in d1a.eval.suite",
     r"^\s*SERVING_CONTEXT(_8K)?\s*=", {"d1a/eval/suite.py"}),
    ("text files are read and written as UTF-8 (d1a.eval.suite.read_json/read_jsonl/write_json/write_jsonl, or an explicit encoding=); "
     "the platform locale must never decide how a frozen partition is decoded (issue #12)",
     r"\.read_text\(\)|\.write_text\((?![^\n]*encoding=)|json\.loads?\(open\(|encoding=None|(?<![\w.])open\((?![^\n]*encoding=)(?![^\n]*\"[rwax]b\")", {"d1a/eval/suite.py"}),
    ("suite manifests are read through d1a.eval.suite.read_manifest",
     r"manifest\.json\"\)\.read_text\(\)", {"d1a/eval/suite.py"}),
    ("device selection, synchronize and empty_cache go through d1a.backends.device (the Space is a CUDA-only one-off)",
     r"is_available\(\) else|torch\.(mps|cuda)\.(synchronize|empty_cache|current_allocated_memory|max_memory_allocated)\(", {"d1a/backends/device.py"}),
    ("which partitions stay out of git is d1a.eval.suite.GIT_LIMIT",
     r"10 \* 1024 \* 1024", {"d1a/eval/suite.py"}),
    ("the pinned Qwen3.5 tokenizer suite builders admit records under is d1a.eval.suite.ADMISSION_TOKENIZER",
     r"1001bb4d826a52d1f399e183466143f4da7b741b", {"d1a/eval/suite.py"}),
    ("calibration by state-token length is d1a.eval.metrics.calibration_by_length (LENGTH_EDGES), and whether a temperature fit set "
     "shares data with a checkpoint's training is d1a.training.calibrate.in_distribution",
     r"\(8192, 16384, 32768, 65536\)|def (in_distribution|calibration_by_length)\(|HELD_OUT_SPLITS = ", {"d1a/eval/metrics.py", "d1a/training/calibrate.py"}),
    ("a state's normalised-text hash (text_sha256) is d1a.eval.suite.text_digest",
     r"\.casefold\(\)\.split\(\)\)\.encode\(\)", {"d1a/eval/suite.py", "d1a/training/data.py"}),   # d1a.eval.suite imports d1a.training.data, so d1a.training.data keeps its inline copy
]


def sources():
    for entry in SCANNED:
        path = ROOT / entry
        yield from (p for p in ([path] if path.is_file() else sorted(path.rglob("*.py"))) if "__pycache__" not in p.parts and p != Path(__file__))


def test_private_suites_keep_their_partitions_out_of_git():
    """A manifest that names its own mirror (d1a.eval.suite: a held-out suite) publishes hashes only: no partition in git."""
    import json, subprocess
    tracked = set(subprocess.run(["git", "ls-files", "evals"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split())
    for manifest in sorted((ROOT / "evals").rglob("manifest.json")):
        if "mirror" not in json.loads(manifest.read_text(encoding="utf-8")): continue
        leaked = [f for f in tracked if f.startswith(str(manifest.parent.relative_to(ROOT)) + "/") and f.endswith(".jsonl")]
        assert not leaked, f"{manifest.parent} names a private mirror but tracks partitions: {leaked}"


@pytest.mark.parametrize("what,pattern,allowed", RULES, ids=[r[0][:60] for r in RULES])
def test_single_home(what, pattern, allowed):
    regex = re.compile(pattern)
    offenders = []
    for path in sources():
        rel = str(path.relative_to(ROOT))
        if rel in allowed:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if regex.search(code):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, f"{what}\n" + "\n".join(offenders)


def unit_runs(workflow, job):
    """The pytest commands of one job of a workflow, parsed (an unquoted run: holding ": " is not valid YAML, and Actions
    would reject the workflow)."""
    import yaml
    steps = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8"))["jobs"][job]["steps"]
    return [step["run"] for step in steps if "pytest" in step.get("run", "")]


def test_ci_and_the_release_run_every_test_file():
    """CI's unit job and the release run `pytest tests --unit`: every test file except tests/conftest.py's
    OUTSIDE_UNIT_TESTS, each listed with why. A list naming every file went stale when files were removed (#123, which
    failed the v0.3.0 release) and made any two pull requests adding a test file conflict on its one line (#136, #138)."""
    import re
    from conftest import OUTSIDE_UNIT_TESTS
    assert unit_runs("ci.yml", "python") == ["uv run python -m pytest tests --unit -q"]
    assert unit_runs("release.yml", "release") == ["uv run python -m pytest tests --unit -q"]
    for workflow in ("ci.yml", "release.yml"):
        assert not re.search(r"(?<![A-Z_])UNIT_TESTS", (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8"))
    assert all((ROOT / path).is_file() and reason for path, reason in OUTSIDE_UNIT_TESTS.items()), OUTSIDE_UNIT_TESTS


def test_the_unit_run_skips_exactly_the_listed_files():
    """--unit leaves out the OUTSIDE_UNIT_TESTS files and nothing else; without it, every file is collected."""
    from types import SimpleNamespace
    import conftest
    skipped = lambda unit: {str(f.relative_to(ROOT)) for f in (ROOT / "tests").glob("test_*.py")
                            if conftest.pytest_ignore_collect(f, SimpleNamespace(getoption=lambda name: unit))}
    assert skipped(True) == set(conftest.OUTSIDE_UNIT_TESTS) and skipped(False) == set()
    assert conftest.pytest_ignore_collect(ROOT / "tests", SimpleNamespace(getoption=lambda name: True)) is None
