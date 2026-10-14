"""llama.cpp's Q4_0 (#270: Gemma 4 12B-it QAT was trained for it, d1a.backends.q4_0, numpy only) and its exact form in MLX's
affine 4-bit, group 32 (d1a.backends.mlx.q4_0_mlx). The MLX test needs Apple Silicon; the others run anywhere.
    uv run python -m pytest tests/test_q4_0.py -v
"""
import numpy as np
import pytest

from d1a.backends.checkpoint import mlx_available
from d1a.backends.q4_0 import pack, q4_0, unpack

needs_mlx = pytest.mark.skipif(not mlx_available(), reason="MLX needs Apple Silicon")


def test_q4_0_rounds_as_llama_cpp():
    """d = the largest-magnitude value / -8, q = min(15, floor(w / d + 8.5)): the extreme maps to 0, its mirror to 15
    (1.0 / 0.125 + 8.5 = 16.5, capped), zero to 8; an all-zero block has d = 0 and q = 8."""
    block = np.zeros(32, np.float32); block[:6] = [-1.0, 1.0, 0.5, -0.0625, 0.0, 0.875]
    q, d = q4_0(np.stack([block, np.zeros(32, np.float32)]))
    assert d.tolist() == [[0.125], [0.0]]
    assert q[0, :6].tolist() == [0, 15, 12, 8, 8, 15] and q[1].tolist() == [8] * 32


def test_pack_and_unpack_are_inverse():
    q = np.random.default_rng(0).integers(0, 16, (3, 64)).astype(np.uint8)
    words = pack(q)
    assert words.shape == (3, 8) and words.dtype == np.uint32 and np.array_equal(unpack(words), q)
    assert int(pack(np.arange(8, dtype=np.uint8)[None])[0, 0]) == 0x76543210   # element i at bits 4i


@needs_mlx
def test_mlx_reads_the_packing_as_q4_0_and_mx_quantize_does_not_land_on_it():
    """MLX dequantizes scale * q + bias; on the CPU that is two roundings, so the comparison allows one float32 rounding.
    A wrong nibble order or scale would be off by whole grid steps."""
    import mlx.core as mx
    from d1a.backends.mlx import q4_0_mlx
    w = (np.random.default_rng(0).standard_normal((16, 128)) * 0.02).astype(np.float32)
    q, d = q4_0(w)
    want = (q.astype(np.float32) - 8) * np.repeat(d, 32, axis=1)
    packed, scales, biases = q4_0_mlx(w, mx.float32)
    assert np.array_equal(unpack(np.array(packed)), q) and np.array_equal(np.array(scales), d) and np.array_equal(np.array(biases), -8 * d)
    got = np.array(mx.dequantize(packed, scales, biases, group_size=32, bits=4))
    assert np.allclose(got, want, rtol=0, atol=np.repeat(np.abs(d), 32, axis=1) * 1e-6)
    affine = np.array(mx.dequantize(*mx.quantize(mx.array(w), group_size=32, bits=4), group_size=32, bits=4))
    assert not np.allclose(affine, want, rtol=0, atol=np.repeat(np.abs(d), 32, axis=1) * 1e-3)   # why the export packs Q4_0 itself
