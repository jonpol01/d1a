"""llama.cpp's Q4_0 (#270: Gemma 4 12B-it QAT was trained for it) and its exact form in MLX's affine 4-bit, group 32
(d1a.backends.mlx.q4_0, q4_0_mlx). The packing tests need MLX on Apple Silicon.
    uv run python -m pytest tests/test_q4_0.py -v
"""
import numpy as np
import pytest

from d1a.backends.checkpoint import mlx_available

needs_mlx = pytest.mark.skipif(not mlx_available(), reason="MLX needs Apple Silicon")


def test_q4_0_rounds_as_llama_cpp():
    """d = the largest-magnitude value / -8, q = min(15, floor(w / d + 8.5)): the extreme maps to 0, its mirror to 15,
    zero to 8; an all-zero block has d = 0 and q = 8."""
    from d1a.backends.mlx import q4_0
    block = np.zeros(32, np.float32); block[:6] = [-1.0, 1.0, 0.5, -0.0625, 0.0, 0.875]
    q, d = q4_0(np.stack([block, np.zeros(32, np.float32)]))
    assert d.tolist() == [[0.125], [0.0]]
    assert q[0, :6].tolist() == [0, 15, 12, 8, 8, 15] and q[1].tolist() == [8] * 32   # 1.0/0.125 + 8.5 = 16.5 -> capped at 15


@needs_mlx
def test_the_mlx_form_dequantizes_to_q4_0_exactly_and_mx_quantize_does_not():
    import mlx.core as mx
    from d1a.backends.mlx import q4_0, q4_0_mlx
    w = (np.random.default_rng(0).standard_normal((16, 128)) * 0.02).astype(np.float32)
    q, d = q4_0(w)
    want = (q.astype(np.float32) - 8) * np.repeat(d, 32, axis=1)
    got = np.array(mx.dequantize(*q4_0_mlx(w, mx.float32), group_size=32, bits=4))
    assert np.array_equal(got, want)
    affine = np.array(mx.dequantize(*mx.quantize(mx.array(w), group_size=32, bits=4), group_size=32, bits=4))
    assert not np.array_equal(affine, want)   # why the export packs Q4_0 itself
