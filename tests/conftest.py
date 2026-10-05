"""Test-session settings shared by every test file, and the one list of test files CI does not run.

    uv run python -m pytest tests --unit -q     # what CI and the release run: every test file except OUTSIDE_UNIT_TESTS
"""
import os
from pathlib import Path

# Test files the unit run (--unit) skips, and why. Every other tests/test_*.py runs in CI as soon as it exists, so two pull
# requests that each add a test file never edit the same line (they did when CI named every file, #136 and #138).
OUTSIDE_UNIT_TESTS = {
    "tests/test_api.py": "needs a running d1a.serve (pytest -m server)",
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
