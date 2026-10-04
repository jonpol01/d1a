"""MLX export folders (d1a_config.json) and the Gemma 4 MLX path, without weights: the export config round trip and
backend rules run anywhere; the cache test builds a tiny random Gemma 4 in mlx-lm (Apple Silicon only, a second).
Run: uv run python -m pytest tests/test_mlx_export.py -q
"""
import json
import platform
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from d1a.checkpoint import (EXPORT_CONFIG, EXPORT_FORMAT, EXPORT_HEAD, EXPORT_VERSION, Checkpoint, LoadOptions, Meta,
                            export_config, mlx_available, write_meta)


def write_export(tmp_path, **override):
    """An export folder as scripts/export_mlx.py writes it, minus mlx-lm's weights (never read by these tests)."""
    from safetensors.torch import save_file
    head = {"q.weight": torch.randn(4, 8), "q.bias": torch.randn(4), "k.weight": torch.randn(4, 8), "k.bias": torch.randn(4)}
    save_file(head, str(tmp_path / EXPORT_HEAD))
    source = SimpleNamespace(requested="JohnP1/d1a-e2b@v0.1-1epoch", path=str(tmp_path / "snapshots" / "71c14b2"), weights_sha256=lambda: "a" * 64,
                             meta=Meta(base="google/gemma-4-E2B", base_revision="d29ff6b", lora=16, head_dim=4, temperature=1.48, head=head))
    cfg = {**export_config(source, [2], [5, 6, 7, 8, 9], 0, 8, {"group_size": 64, "bits": 4, "mode": "affine", "embeddings": True}, "bfloat16"), **override}
    (tmp_path / EXPORT_CONFIG).write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8"); (tmp_path / "model.safetensors").write_bytes(b"")
    return head, cfg


def test_export_config_round_trip(tmp_path):
    head, cfg = write_export(tmp_path)
    assert (cfg["format"], cfg["format_version"], cfg["bos_id"], cfg["source_revision"]) == (EXPORT_FORMAT, EXPORT_VERSION, 2, "71c14b2")
    ck = Checkpoint(tmp_path)
    assert ck.export == cfg
    m = ck.meta
    assert (m.base, m.base_revision, m.lora, m.head_dim, m.temperature, m.weights) == ("google/gemma-4-E2B", "d29ff6b", 16, 4, 1.48, "mlx")
    assert all(torch.equal(m.head[k], head[k]) and m.head[k].dtype == torch.float32 for k in head)
    assert len(ck.weights_sha256()) == 64   # over the weight shards, as for full weights


def test_export_runs_on_mlx_only(tmp_path):
    write_export(tmp_path)
    ck = Checkpoint(tmp_path)
    for opts in (LoadOptions(), LoadOptions(backend="auto"), LoadOptions(backend="mlx"), LoadOptions(backend="auto", dtype=torch.float32)):
        assert ck.backend("mps", opts) == "mlx"
    with pytest.raises(ValueError, match="MLX export"):
        ck.backend("mps", LoadOptions(backend="torch"))
    with pytest.raises(ValueError, match="mlx-lm|lora_scale"):   # merged weights cannot be interpolated (or no mlx-lm here)
        ck._load_mlx(None, LoadOptions(lora_scale=0.5))
    with pytest.raises(ValueError, match="MLX export"):
        ck.warm_start(None, ck.meta)


@pytest.mark.parametrize("override", [{"format_version": EXPORT_VERSION + 1}, {"format": "mlx-lm"}, {"format_version": "1"}])
def test_export_format_is_checked(tmp_path, override):
    write_export(tmp_path, **override)
    with pytest.raises(ValueError, match="format"):
        Checkpoint(tmp_path)


def test_auto_backend_takes_gemma4_bases(tmp_path, monkeypatch):
    write_meta(tmp_path, Meta(base="google/gemma-4-E2B", head={}))
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    ck = Checkpoint(tmp_path)
    for model_type, layers, mlx in (("gemma4_text", ["sliding_attention", "full_attention"], True), ("qwen3", ["full_attention"], False),
                                    ("qwen3_5_text", ["linear_attention", "full_attention"], True)):
        monkeypatch.setattr(Checkpoint, "text_config", lambda self: SimpleNamespace(model_type=model_type, layer_types=layers))
        assert ck.backend("mps", LoadOptions(backend="auto")) == ("mlx" if mlx and mlx_available() else "torch")
        assert ck.backend("cuda", LoadOptions(backend="auto")) == "torch"


def tiny_gemma4():
    from mlx_lm.models import gemma4
    s, f = "sliding_attention", "full_attention"
    text = {"hidden_size": 64, "num_hidden_layers": 6, "intermediate_size": 128, "num_attention_heads": 2, "head_dim": 32, "global_head_dim": 64,
            "num_key_value_heads": 1, "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 16, "vocab_size_per_layer_input": 64,
            "sliding_window": 8, "layer_types": [s, s, f, s, s, f]}   # layers 4, 5 read the keys of layers 3, 2
    return gemma4.Model(gemma4.ModelArgs.from_dict({"model_type": "gemma4", "vocab_size": 64, "text_config": text}))


def fake_encoding(rng, state, branches):
    """An encoding (d1a.model.encode's layout) of random tokens: `state` tokens, then per question (tokens, options)."""
    ids, seg, decide, opts = [int(t) for t in rng.integers(1, 64, state)], [0] * state, [], []
    for k, (n, options) in enumerate(branches, start=1):
        start = len(ids); ids += [int(t) for t in rng.integers(1, 64, n)]; seg += [k] * n
        decide.append(start + n - 1); opts.append([start + 1 + 2 * j for j in range(options)])
    return {"ids": ids, "seg": seg, "pos": list(range(len(ids))), "decide_idx": decide, "opt_idx": opts}


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64" or not mlx_available(), reason="MLX runs on Apple Silicon only")
def test_gemma4_state_pass_skips_the_kv_shared_layers():
    """The state pass never runs Gemma 4's KV-shared layers (they keep no cache and the state's outputs are not read), and
    the questions answered on that cache equal each question run as its own full row."""
    import mlx.core as mx
    from d1a.mlx_model import MLXDecisionModel
    mx.random.seed(0)
    m = MLXDecisionModel.from_lm(tiny_gemma4(), pad_id=0, head_dim=16)
    enc = fake_encoding(np.random.default_rng(1), 20, [(5, 2), (14, 4)])
    rows = m.forward_rows(enc)
    shared = [i for i, p in enumerate(m.text.previous_kvs) if p != i]
    assert shared == [4, 5]
    saved = {i: m.text.layers[i] for i in shared}

    class Trap:   # keeps the layer type the mask builder reads; fails if the layer is run
        def __init__(self, layer): self.layer_type = layer.layer_type
        def __call__(self, *a, **k): raise AssertionError("a KV-shared layer ran during the state pass")
    for i in shared: m.text.layers[i] = Trap(saved[i])
    prefix = m.prefix(enc)
    for i, layer in saved.items(): m.text.layers[i] = layer
    for z, r in zip(m._branch_logits(enc, prefix[1]), rows):
        assert torch.allclose(z, r, atol=1e-4), (z - r).abs().max()


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64" or not mlx_available(), reason="MLX runs on Apple Silicon only")
def test_gemma4_prefix_form_matches_rows_across_the_sliding_window():
    """The serving form (state once into KV + rotating caches, the questions as right-padded rows on replicated copies)
    equals each question run as its own full row, for states shorter and longer than the window, in fp32; reusing the
    prefix leaves it intact."""
    import mlx.core as mx
    from d1a.mlx_model import MLXDecisionModel
    mx.random.seed(0)
    m = MLXDecisionModel.from_lm(tiny_gemma4(), pad_id=0, head_dim=16)
    assert not m.hybrid and m.dtype == "float32"
    rng = np.random.default_rng(0)
    for state in (5, 7, 20):
        enc = fake_encoding(rng, state, [(5, 2), (14, 4), (9, 3)])
        rows = m.forward_rows(enc)
        prefix = m.prefix(enc)
        before = [np.asarray(c.keys) for c in prefix[1]]
        for got in (m.forward(enc), [z for z in m._branch_logits(enc, prefix[1])], [z for z in m._branch_logits(enc, prefix[1])]):
            for z, r in zip(got, rows):
                assert torch.allclose(z, r, atol=1e-4), (state, (z - r).abs().max())
        assert all(np.array_equal(np.asarray(c.keys), b) for c, b in zip(prefix[1], before))
