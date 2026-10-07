"""Test-session settings shared by every test file, the one list of test files CI does not run, and the tiny hybrid
Qwen3.5 the training, shared-prefix and predictor tests run on.

    uv run python -m pytest tests --unit -q     # what CI and the release run: every test file except OUTSIDE_UNIT_TESTS
"""
import json
import os
import sys
from pathlib import Path

import pytest

# Test files the unit run (--unit) skips, and why. Every other tests/test_*.py runs in CI as soon as it exists, so two pull
# requests that each add a test file never edit the same line (they did when CI named every file, #136 and #138).
OUTSIDE_UNIT_TESTS = {
    "tests/test_api.py": "needs a running d1a.serving.serve (pytest -m server)",
    "tests/test_model.py": "needs real weights and the smoke checkpoint",
    "tests/test_mlx.py": "needs MLX and real weights; its weight-free tests run in the Apple Silicon job",
}
ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--unit", action="store_true", help="skip the test files in tests/conftest.py's OUTSIDE_UNIT_TESTS (CI's unit run)")


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--unit"):
        return None
    try:
        relative = Path(collection_path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    return True if relative in OUTSIDE_UNIT_TESTS else None


def pytest_configure(config):
    # GitHub's macOS runners are virtual machines whose emulated GPU runs MLX kernels very slowly (the MLX tests stalled
    # for 15+ minutes there; they take about a minute on a real Mac): CI runs MLX on the CPU instead
    if os.environ.get("D1A_TEST_MLX_DEVICE") == "cpu":
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.set_default_device(mx.cpu)


# --- a hybrid Qwen3.5 small enough for the CPU ---------------------------------------------------------------------------
# Gemma 4 (tests/golden/tiny-gemma4) is attention-only; the shared prefix, --pass_tokens_max and the predictor's long-row
# route need a hybrid backbone, whose Gated DeltaNet layers run every record as rows.

HYBRID_WORDS = ("the", "customer", "is", "charged", "twice", "it", "which", "team", "billing", "shipping", "refund", "angry")
TEAMS = ("billing", "shipping", "refund")


def hybrid_requests(n=16):
    """n labelled requests: a 3-way Choice and a yes/no each, and a state that grows by one token per request (6 + i)."""
    return [{"state": " ".join(["the customer is charged twice", *["it"] * i]), "questions": {
        "team": {"type": "choice", "instructions": "which team", "criteria": dict.fromkeys(TEAMS), "label": TEAMS[i % len(TEAMS)]},
        "angry": {"type": "noul", "instructions": "is the customer angry", "label": not i % 2}}} for i in range(n)]


@pytest.fixture(scope="session")
def hybrid_base(tmp_path_factory):
    """A folder with base/ (a random bf16 Qwen3.5: layer 0 Gated DeltaNet, layer 1 attention, saved like a Hub snapshot,
    with a word-level tokenizer that knows Qwen's delimiters) and data.jsonl (hybrid_requests)."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from d1a.core.encoding import SPECIAL
    folder = tmp_path_factory.mktemp("hybrid")
    vocabulary = ["<unk>", "<pad>", *SPECIAL, *HYBRID_WORDS]
    words = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="<unk>"))
    words.pre_tokenizer = pre_tokenizers.Whitespace()
    config = Qwen3_5TextConfig(vocab_size=len(vocabulary), pad_token_id=1, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
                               hidden_size=32, intermediate_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                               linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).to(torch.bfloat16).save_pretrained(folder / "base")
    PreTrainedTokenizerFast(tokenizer_object=words, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=SPECIAL).save_pretrained(folder / "base")
    with open(folder / "data.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in hybrid_requests())
    return folder


@pytest.fixture
def train_hybrid(hybrid_base, monkeypatch):
    """-> run(out, *arguments): d1a.training.train in this process on hybrid_base's base and data (CPU, --batch 2, --lr 1e-3);
    later arguments override these. Returns `out` as a Path."""
    from d1a.training import train

    def run(out, *arguments):
        monkeypatch.setattr(sys, "argv", ["d1a.training.train", "--base", str(hybrid_base / "base"), "--data", str(hybrid_base / "data.jsonl"),
                                          "--device", "cpu", "--batch", "2", "--lr", "1e-3", "--out", str(out), *arguments])
        train.main()
        return Path(out)
    return run
