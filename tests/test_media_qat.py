"""qat_weights (d1a#341): the PyTorch media server puts a QAT run's own checkpoint on the loaded Gemma 4: the text model as
load_ct_base writes it (s q exactly) and the vision and audio encoders with their projections as the checkpoint stores them.
An encoder tensor the model lacks, or one the checkpoint leaves unset, is refused. On the committed tiny Gemma 4 text model
with stand-in encoders, no download."""
import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file

from test_train_base_ct import FIXTURE, ct_checkpoint


def gemma_with_encoders():
    from transformers import AutoModelForCausalLM
    inner = torch.nn.Module()
    inner.language_model = AutoModelForCausalLM.from_pretrained(str(FIXTURE / "base")).model
    inner.vision_tower, inner.audio_tower = torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    inner.embed_vision, inner.embed_audio = torch.nn.Module(), torch.nn.Module()
    inner.embed_vision.embedding_projection, inner.embed_audio.embedding_projection = torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)
    inner.audio_tower.register_buffer("input_max", torch.zeros(4))                       # stored, as Gemma 4's audio clipping ranges are
    inner.audio_tower.register_buffer("inv_timescales", torch.ones(4), persistent=False)   # computed at load, never stored
    return inner


def with_encoders(ct, drop=None):
    stored = load_file(str(ct / "model.safetensors"))
    encoders = {f"model.{n}": np.full(p.shape, 0.5, np.float32) for n, p in gemma_with_encoders().state_dict().items() if not n.startswith("language_model")}
    save_file({**stored, **{k: v for k, v in encoders.items() if k != drop}}, str(ct / "model.safetensors"))
    return encoders


def test_the_media_model_takes_the_qat_text_model_and_its_encoders(tmp_path):
    from d1a.serving.media import qat_weights
    quantized = ct_checkpoint(FIXTURE / "base", tmp_path / "ct")
    encoders = with_encoders(tmp_path / "ct")
    inner = gemma_with_encoders()
    assert qat_weights(inner, tmp_path / "ct") == len(encoders)
    params = inner.state_dict()
    assert "audio_tower.input_max" in params and all(torch.equal(params[k.removeprefix("model.")], torch.from_numpy(v)) for k, v in encoders.items())
    rel, (q, scale) = next(iter(quantized.items()))
    weight = params["language_model." + rel + ".weight"]
    assert np.array_equal(weight.detach().numpy(), (q.astype(np.float32).reshape(q.shape[0], -1, 32) * scale[..., None]).reshape(q.shape))
    ct_checkpoint(FIXTURE / "base", tmp_path / "ct2")
    with_encoders(tmp_path / "ct2", drop="model.embed_vision.embedding_projection.weight")
    with pytest.raises(ValueError, match="encoder parameters not in the checkpoint"):
        qat_weights(gemma_with_encoders(), tmp_path / "ct2")
