"""Test-session settings shared by every test file."""
import os


def pytest_configure(config):
    # GitHub's macOS runners are virtual machines whose emulated GPU runs MLX kernels very slowly (the MLX tests stalled
    # for 15+ minutes there; they take about a minute on a real Mac): CI runs MLX on the CPU instead
    if os.environ.get("D1A_TEST_MLX_DEVICE") == "cpu":
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.set_default_device(mx.cpu)
