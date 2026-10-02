# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); dropped the published-claims check and the removed space/ app; modal_app.py left the scanned sources.
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
    ("head.pt is read and written through d1a.checkpoint (Meta, read_meta, write_meta)",
     r"torch\.(load|save)\([^\n]*head\.pt", {"d1a/checkpoint.py"}),
    ("D1A_DTYPE/D1A_MERGE/D1A_ATTN/D1A_LORA_SCALE/D1A_TEMPERATURE/D1A_BACKEND/D1A_CUDA_GRAPHS are read only by LoadOptions.from_env",
     r"environ(\.get)?\(?\[?\s*\"D1A_(DTYPE|MERGE|ATTN|LORA_SCALE|TEMPERATURE|BACKEND|CUDA_GRAPHS)\"", {"d1a/checkpoint.py"}),
    ("whether a checkpoint is a LoRA adapter or full weights, and where its shards are, is d1a.checkpoint.Checkpoint.full / shards (the loader rule)",
     r"glob\(\"model\*\.safetensors\"\)|adapter_config\.json\"\)\.exists\(\)", {"d1a/checkpoint.py"}),
    ("a checkpoint becomes a model only through d1a.checkpoint (Checkpoint.load picks the torch or MLX implementation)",
     r"MLXDecisionModel\(|merge_lora\(", {"d1a/checkpoint.py", "d1a/mlx_model.py", "tests/test_mlx.py"}),
    ("option keys come from d1a.api.question_keys",
     r"\[\s*\"false\"\s*,\s*\"true\"\s*\]|\[str\(i\) for i in range\(len\(", {"d1a/api.py", "tests/test_unit.py"}),   # the unit test pins the contract
    ("the training context is d1a.model.MAX_STATE/MAX_BRANCH/MAX_PACKED, lifted only through d1a.model.training_context (d1a.suite.CONTEXT in manifests), and d1a.model.fits",
     r"(?<![\w.])(>|<=|>=|<)\s*2048\b|\b2048\s*(<|>)|max_(branch|state|packed)\"?\s*[=:]\s*\d{3,}", {"d1a/model.py"}),
    ("the serving / long-state limits (SERVE_MAX_*, ROW_PASS_TOKENS, MAX_TRAIN_STATE) and the pre-64k aliases the frozen suites' builders "
     "rebuild byte for byte with (SERVE_MAX_*_8K, MAX_TRAIN_STATE_8K) are defined only in d1a.model",
     r"^\s*(SERVE_MAX_(STATE|BRANCH|PACKED)|ROW_PASS_TOKENS|MAX_TRAIN_STATE)(_8K)?\s*(=|,[^\n=]*=)|(?<![\w.])7552\b", {"d1a/model.py"}),
    ("the serving contexts manifests record (SERVING_CONTEXT, and SERVING_CONTEXT_8K for the suites frozen before 64k states) are defined only in d1a.suite",
     r"^\s*SERVING_CONTEXT(_8K)?\s*=", {"d1a/suite.py"}),
    ("text files are read and written as UTF-8 (d1a.suite.read_json/read_jsonl/write_json/write_jsonl, or an explicit encoding=); "
     "the platform locale must never decide how a frozen partition is decoded (issue #12)",
     r"\.read_text\(\)|\.write_text\((?![^\n]*encoding=)|json\.loads?\(open\(|encoding=None|(?<![\w.])open\((?![^\n]*encoding=)(?![^\n]*\"[rwax]b\")", {"d1a/suite.py"}),
    ("suite manifests are read through d1a.suite.read_manifest",
     r"manifest\.json\"\)\.read_text\(\)", {"d1a/suite.py"}),
    ("device selection, synchronize and empty_cache go through d1a.device (the Space is a CUDA-only one-off)",
     r"is_available\(\) else|torch\.(mps|cuda)\.(synchronize|empty_cache|current_allocated_memory|max_memory_allocated)\(", {"d1a/device.py"}),
    ("the isolation sibling probe is d1a.experiment.ISOLATION_PROBE (fp32 mechanism check and served isolation read the same question)",
     r"CRANE-9274", {"d1a/experiment.py"}),
    ("which partitions stay out of git is d1a.suite.GIT_LIMIT",
     r"10 \* 1024 \* 1024", {"d1a/suite.py"}),
    ("the pinned Qwen3.5 tokenizer suite builders admit records under is d1a.suite.ADMISSION_TOKENIZER",
     r"1001bb4d826a52d1f399e183466143f4da7b741b", {"d1a/suite.py", "d1a/transfer_v9.py"}),   # transfer_v9 pins every Qwen3.5 base it scores
    ("a trial's served temperature (fitted on its own development rows), clean rows served with unknowable records kept, and "
     "the registered paired read (2,000 resamples, seed 0, micro) are d1a.rounds.temperature / served_clean / paired",
     r"development/rows\.json\"\), \[\]\)\[0\]|tempered_row\(raw_row\(recorded\(|SAMPLES = 2000|def (boot|knowable)\(|knowable = lambda",
     {"d1a/rounds.py", "d1a/metrics.py"}),   # d1a.metrics.served_at is the scored-rows form the helpers build on
    ("calibration by state-token length is d1a.metrics.calibration_by_length (LENGTH_EDGES), and whether a temperature fit set "
     "shares data with a checkpoint's training is d1a.rounds.pool_conflicts (round pools and scripts/calibrate_checkpoint.py alike)",
     r"\(8192, 16384, 32768, 65536\)|def (pool_conflicts|calibration_by_length)\(|FIT_SPLITS = ", {"d1a/metrics.py", "d1a/rounds.py"}),
    ("a state's normalised-text hash (text_sha256) is d1a.suite.text_digest",
     r"\.casefold\(\)\.split\(\)\)\.encode\(\)", {"d1a/suite.py", "d1a/data.py"}),   # d1a.suite imports d1a.data, so d1a.data keeps its inline copy
]


def sources():
    for entry in SCANNED:
        path = ROOT / entry
        yield from (p for p in ([path] if path.is_file() else sorted(path.rglob("*.py"))) if "__pycache__" not in p.parts and p != Path(__file__))


def test_private_suites_keep_their_partitions_out_of_git():
    """A manifest that names its own mirror (d1a.suite: a held-out suite) publishes hashes only: no partition in git."""
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
