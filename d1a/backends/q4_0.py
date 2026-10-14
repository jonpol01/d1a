"""llama.cpp's Q4_0 (blocks of 32 along each row; its quantize_row_q4_0_ref), in numpy only, so it runs without MLX:
d = the block's largest-magnitude value / -8, q = min(15, int(w / d + 8.5)), w' = (q - 8) * d. d1a.backends.mlx.q4_0_mlx
holds it exactly in MLX's affine 4-bit, group 32 (#270: Gemma 4 12B-it QAT was trained for it)."""
import numpy as np


def q4_0(w):
    """[rows, cols] float array -> (q uint8 in 0..15, d float32 [rows, cols/32])."""
    w = np.asarray(w, np.float32)
    b = w.reshape(w.shape[0], -1, 32)
    d = np.take_along_axis(b, np.abs(b).argmax(-1)[..., None], -1)[..., 0] / -8.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    q = np.clip(np.floor(b * inv[..., None] + 8.5), 0, 15).astype(np.uint8)
    return q.reshape(w.shape), d.astype(np.float32)


def pack(q):
    """4-bit values [rows, cols] -> uint32 [rows, cols/8], as MLX packs them: 8 per word, from the low nibble up."""
    words = q.reshape(q.shape[0], -1, 8).astype(np.uint32) << (4 * np.arange(8, dtype=np.uint32))
    return words.sum(-1, dtype=np.uint32)


def unpack(words):
    """pack's inverse: uint32 [rows, cols/8] -> uint8 [rows, cols]."""
    words = np.asarray(words, np.uint32)
    return ((words[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).astype(np.uint8).reshape(words.shape[0], -1)
