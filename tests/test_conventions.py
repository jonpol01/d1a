# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); dropped the published-claims check and the removed space/ app; modal_app.py left the scanned sources; the rules of the removed Kev research modules (experiment, rounds) left with them, and calibration's moved to d1a.training.calibrate; CI runs every test file but tests/conftest.py's OUTSIDE_UNIT_TESTS (pytest tests --unit); the facts, their scanner and the private-mirror check rewritten in D1A's own code.
"""Single homes: a fact D1A keeps in one place (a file format, a limit, a loader rule) must not be restated elsewhere.

Each Home names the fact, a regex that finds a restatement of it, and the files the fact lives in. A failure lists the
lines that restate it: call the owner instead.
    uv run python -m pytest tests/test_conventions.py -q
"""
import json
import re
import subprocess
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
Home = namedtuple("Home", "fact pattern owners")

HOMES = [
    Home("head.pt goes in and out only through d1a.backends.checkpoint's Meta, read_meta and write_meta",
         r"torch\.(load|save)\([^\n]*head\.pt",
         {"d1a/backends/checkpoint.py", "tests/test_checkpoint.py"}),   # the test writes a hostile head.pt by hand
    Home("the D1A_* load options (DTYPE, MERGE, ATTN, LORA_SCALE, TEMPERATURE, BACKEND, CUDA_GRAPHS) are read "
         "in one place, LoadOptions.from_env",
         r"environ(\.get)?\(?\[?\s*\"D1A_(DTYPE|MERGE|ATTN|LORA_SCALE|TEMPERATURE|BACKEND|CUDA_GRAPHS)\"",
         {"d1a/backends/checkpoint.py"}),
    Home("LoRA adapter or full weights, and which shards: only d1a.backends.checkpoint.Checkpoint.full and "
         ".shards decide (the loader rule)",
         r"glob\(\"model\*\.safetensors\"\)|adapter_config\.json\"\)\.exists\(\)",
         {"d1a/backends/checkpoint.py"}),
    Home("only d1a.backends.checkpoint turns a checkpoint into a model (Checkpoint.load chooses torch or MLX)",
         r"MLXDecisionModel\(|merge_lora\(",
         {"d1a/backends/checkpoint.py", "d1a/backends/mlx.py", "tests/test_mlx.py"}),
    Home("a question's option keys are d1a.core.api.question_keys",
         r"\[\s*\"false\"\s*,\s*\"true\"\s*\]|\[str\(i\) for i in range\(len\(",
         {"d1a/core/api.py", "tests/test_system_one.py", "tests/test_tiny_checkpoint.py"}),   # these tests pin the contract (the second re-derives it on purpose)
    Home("the training context lives in d1a.core.encoding (MAX_STATE, MAX_BRANCH, MAX_PACKED), raised only by "
         "its training_context (d1a.eval.suite.CONTEXT in manifests) and checked by its fits",
         r"(?<![\w.])(>|<=|>=|<)\s*2048\b|\b2048\s*(<|>)|max_(branch|state|packed)\"?\s*[=:]\s*\d{3,}",
         {"d1a/core/encoding.py"}),
    Home("the serving and long-state limits (SERVE_MAX_*, ROW_PASS_TOKENS, MAX_TRAIN_STATE) and the pre-64k "
         "aliases the frozen suites' builders need to rebuild byte for byte (SERVE_MAX_*_8K, "
         "MAX_TRAIN_STATE_8K) are set only in d1a.core.encoding",
         r"^\s*(SERVE_MAX_(STATE|BRANCH|PACKED)|ROW_PASS_TOKENS|MAX_TRAIN_STATE)(_8K)?\s*(=|,[^\n=]*=)|(?<![\w.])7552\b",
         {"d1a/core/encoding.py"}),
    Home("the serving contexts a manifest records (SERVING_CONTEXT, and SERVING_CONTEXT_8K for suites frozen "
         "before 64k states) are set only in d1a.eval.suite",
         r"^\s*SERVING_CONTEXT(_8K)?\s*=",
         {"d1a/eval/suite.py"}),
    Home("text is read and written as UTF-8, through d1a.eval.suite's read_json, read_jsonl, write_json and "
         "write_jsonl or an explicit encoding=, never the platform locale, which once decided how a frozen "
         "partition decoded (issue #12)",
         r"\.read_text\(\)|\.write_text\((?![^\n]*encoding=)|json\.loads?\(open\(|encoding=None|(?<![\w.])open\((?![^\n]*encoding=)(?![^\n]*\"[rwax]b\")",
         {"d1a/eval/suite.py"}),
    Home("a suite manifest is read through d1a.eval.suite.read_manifest",
         r"manifest\.json\"\)\.read_text\(\)",
         {"d1a/eval/suite.py"}),
    Home("device choice, synchronize and empty_cache go through d1a.backends.device (the Space was a CUDA-only "
         "one-off)",
         r"is_available\(\) else|torch\.(mps|cuda)\.(synchronize|empty_cache|current_allocated_memory|max_memory_allocated)\(",
         {"d1a/backends/device.py"}),
    Home("the size above which a partition stays out of git is d1a.eval.suite.GIT_LIMIT",
         r"10 \* 1024 \* 1024",
         {"d1a/eval/suite.py"}),
    Home("the tokenizer revision suite builders admit records under is d1a.eval.suite.ADMISSION_TOKENIZER",
         r"1001bb4d826a52d1f399e183466143f4da7b741b",
         {"d1a/eval/suite.py"}),
    Home("calibration by state length is d1a.eval.metrics.calibration_by_length (LENGTH_EDGES), and whether a "
         "temperature's fit set overlaps a checkpoint's training is d1a.training.calibrate.in_distribution",
         r"\(8192, 16384, 32768, 65536\)|def (in_distribution|calibration_by_length)\(|HELD_OUT_SPLITS = ",
         {"d1a/eval/metrics.py", "d1a/training/calibrate.py"}),
    Home("the normalised-text hash of a state (text_sha256) is d1a.eval.suite.text_digest",
         r"\.casefold\(\)\.split\(\)\)\.encode\(\)",
         {"d1a/eval/suite.py", "d1a/training/data.py"}),   # d1a.eval.suite imports d1a.training.data, so d1a.training.data keeps its inline copy
]


def scanned():
    """Every Python file under d1a/, scripts/ and tests/ except this one, as (its path in the repo, its text)."""
    for top in ("d1a", "scripts", "tests"):
        for path in sorted((ROOT / top).rglob("*.py")):
            if "__pycache__" not in path.parts and path != Path(__file__):
                yield str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")


@pytest.mark.parametrize("home", HOMES, ids=[home.fact[:60] for home in HOMES])
def test_each_fact_lives_in_one_place(home):
    restated = re.compile(home.pattern)   # a comment may name the fact; code may not restate it
    found = [f"{rel}:{n}: {line.strip()}" for rel, text in scanned() if rel not in home.owners
             for n, line in enumerate(text.splitlines(), 1) if restated.search(line.split("#", 1)[0])]
    assert not found, home.fact + "\n" + "\n".join(found)


def test_a_suite_with_a_private_mirror_tracks_no_partition():
    """A manifest naming its own mirror (a held-out suite, d1a.eval.suite) publishes hashes, never its partitions."""
    tracked = subprocess.run(["git", "ls-files", "evals"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    for manifest in sorted((ROOT / "evals").rglob("manifest.json")):
        if "mirror" in json.loads(manifest.read_text(encoding="utf-8")):
            folder = str(manifest.parent.relative_to(ROOT)) + "/"
            assert not [f for f in tracked if f.startswith(folder) and f.endswith(".jsonl")], f"{folder} names a private mirror but tracks partitions"


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
