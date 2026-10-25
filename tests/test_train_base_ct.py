"""--base_ct (d1a#341): the base weights from a compressed-tensors QAT checkpoint. Every pack-quantized Linear is written
as s q exactly (under its LoRA wrapper), every other tensor as stored, the adapter is left as it is, and a checkpoint that
leaves a base parameter unset is refused. A run that recorded base_ct loads on that trunk. On the committed tiny Gemma 4
fixture, no download."""
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURE = Path(__file__).resolve().parent / "golden" / "tiny-gemma4"


def ct_checkpoint(base, out, drop=None):
    """The tiny base as a compressed-tensors checkpoint would store it (model.language_model.* names; each Linear whose
    width is a multiple of 32 as weight_packed / weight_scale / weight_shape, q + 8 packed 8 per int32 from the low nibble
    up, scales on bf16's grid). -> {text-relative name: (q, scale)}"""
    from safetensors.numpy import load_file, save_file
    out.mkdir()
    stored, quantized = {}, {}
    for k, w in load_file(str(base / "model.safetensors")).items():
        name = "model.language_model." + k.removeprefix("model.")
        if drop and drop in name:
            continue
        if k.endswith("layer_scalar"):   # a buffer, which QAT changes: the model must take the checkpoint's value
            stored[name] = (w * 1.2).astype(w.dtype)
            continue
        if not k.endswith("_proj.weight") or w.ndim != 2 or w.shape[1] % 32:
            stored[name] = w
            continue
        groups = w.astype(np.float32).reshape(w.shape[0], -1, 32)
        scale = ((np.abs(groups).max(-1) / 7 + 1e-8).astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)
        q = np.clip(np.round(groups / scale[..., None]), -8, 7).astype(np.int8).reshape(w.shape)
        u = (q.astype(np.int32) + 8).reshape(w.shape[0], -1, 8).astype(np.uint32) << (4 * np.arange(8, dtype=np.uint32))
        stem = name[: -len(".weight")]
        stored |= {stem + ".weight_packed": u.sum(-1, dtype=np.uint32).view(np.int32), stem + ".weight_scale": scale,
                   stem + ".weight_shape": np.array(w.shape, np.int64)}
        quantized[stem.removeprefix("model.language_model.")] = (q, scale)
    save_file(stored, str(out / "model.safetensors"))
    return quantized


@pytest.fixture
def model():
    from d1a.backends.torch import DecisionModel
    from d1a.core.encoding import load_tokenizer
    base = FIXTURE / "base"
    return DecisionModel(str(base), load_tokenizer(str(base)), "cpu", lora=4, head_dim=16)


def test_the_base_takes_the_checkpoints_exact_grid_and_the_adapter_is_left_alone(model, tmp_path):
    from d1a.backends.checkpoint import load_ct_base
    quantized = ct_checkpoint(FIXTURE / "base", tmp_path / "ct")
    assert quantized
    lora = {n: p.detach().clone() for n, p in model.lm.named_parameters() if "lora_" in n}
    nq, nother = load_ct_base(model.lm, tmp_path / "ct")
    assert nq == len(quantized) and nother > 0
    modules = dict(model.lm.named_modules())
    for rel, (q, scale) in quantized.items():
        module = next(m for path, m in modules.items() if path.endswith("." + rel) or path == rel)
        weight = module.base_layer.weight if hasattr(module, "base_layer") else module.weight
        ref = (q.astype(np.float32).reshape(q.shape[0], -1, 32) * scale[..., None]).reshape(q.shape)
        assert np.array_equal(weight.detach().float().numpy(), ref), rel
    assert all(torch.equal(p, lora[n]) for n, p in model.lm.named_parameters() if "lora_" in n)
    from safetensors.numpy import load_file
    ct = load_file(str(tmp_path / "ct" / "model.safetensors"))
    scalars = {n: b for n, b in model.lm.named_buffers() if n.endswith("layer_scalar")}
    assert scalars and all(np.array_equal(b.float().numpy(), ct["model.language_model." + n.split("base_model.model.", 1)[-1]].astype(np.float32))
                           for n, b in scalars.items())


def test_a_checkpoint_that_leaves_a_base_parameter_unset_is_refused(model, tmp_path):
    from d1a.backends.checkpoint import load_ct_base
    ct_checkpoint(FIXTURE / "base", tmp_path / "ct", drop="layers.0.input_layernorm")
    with pytest.raises(ValueError, match="not in the checkpoint"):
        load_ct_base(model.lm, tmp_path / "ct")


def test_a_run_trained_on_a_qat_checkpoint_loads_and_warm_starts_on_it(tmp_path):
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, Meta, ct_spec, write_meta
    # recorded as a Hub id@revision when trained from a cache snapshot; a run from before the field says it in its arguments
    assert ct_spec("/h/hub/models--google--gemma-4-E2B-it-qat-w4a16-ct/snapshots/971342c0") == "google/gemma-4-E2B-it-qat-w4a16-ct@971342c0"
    assert Meta.from_dict({"base": "b", "args": {"base_ct": "g/q@1"}}).base_ct == "g/q@1"
    quantized = ct_checkpoint(FIXTURE / "base", tmp_path / "ct")
    run = tmp_path / "run"
    shutil.copytree(FIXTURE / "checkpoint", run)
    plain = Checkpoint(run).load("cpu", LoadOptions(merge=False))[1]
    meta = Checkpoint(run).meta
    meta.base_ct = str(tmp_path / "ct")
    write_meta(run, meta)
    ck = Checkpoint(run)
    assert ck.meta.base_ct == str(tmp_path / "ct") and ck.backend("mps") == "torch"   # never the MLX path that builds the base's trunk
    with pytest.raises(ValueError, match="QAT checkpoint"):
        ck.backend("mps", LoadOptions(backend="mlx"))
    model = ck.load("cpu", LoadOptions(merge=False))[1]
    modules = dict(model.lm.named_modules())
    for rel, (q, scale) in quantized.items():
        module = next(m for path, m in modules.items() if path.endswith("." + rel))
        ref = (q.astype(np.float32).reshape(q.shape[0], -1, 32) * scale[..., None]).reshape(q.shape)
        assert np.array_equal(module.base_layer.weight.detach().float().numpy(), ref), rel
    adapter = {n: p for n, p in plain.lm.named_parameters() if "lora_" in n}
    assert adapter and all(torch.equal(p, adapter[n]) for n, p in model.lm.named_parameters() if "lora_" in n)
    # a warm start may move between the base's trunk and a QAT one, never between two QAT checkpoints
    from d1a.backends.torch import DecisionModel
    from d1a.core.encoding import load_tokenizer
    fresh = DecisionModel(str(FIXTURE / "base"), load_tokenizer(str(FIXTURE / "base")), "cpu", lora=meta.lora, head_dim=meta.head_dim)
    ck.warm_start(fresh, dataclasses.replace(meta, base_ct=None))
    with pytest.raises(ValueError, match="base_ct"):
        ck.warm_start(fresh, dataclasses.replace(meta, base_ct="google/another-qat@0"))
