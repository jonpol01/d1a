"""export_media(weights=...) (d1a#341): an export on a compressed-tensors QAT checkpoint takes that checkpoint's vision and
audio encoders, whose vision projection QAT retrained, and only the encoders; without it, the base's. No download: the
base's config and processor lookups are stubbed."""
import json

import torch
from safetensors.torch import load_file, save_file

TENSORS = ("model.vision_tower.encoder.weight", "model.audio_tower.encoder.weight", "model.embed_vision.embedding_projection.weight",
           "model.embed_audio.embedding_projection.weight")


def checkpoint(folder, value):
    folder.mkdir()
    save_file({**{k: torch.full((2, 2), value) for k in TENSORS}, "model.language_model.layers.0.weight": torch.zeros(2, 2)},
              str(folder / "model.safetensors"))
    return folder


def test_the_media_folder_takes_the_qat_checkpoints_encoders(tmp_path, monkeypatch):
    import transformers
    from d1a.serving import media
    base = checkpoint(tmp_path / "base", 1.0)
    (base / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    qat = checkpoint(tmp_path / "qat", 2.0)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, name, revision=None: str(base / name))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: type("P", (), {"save_pretrained": lambda self, d: None})())
    for weights, value, out in ((None, 1.0, tmp_path / "plain"), (qat, 2.0, tmp_path / "with-qat")):
        written = load_file(str(tmp_path / out / media.MEDIA_DIR / "model.safetensors")) if media.export_media("base", None, out, weights=weights) else None
        assert sorted(written) == sorted(k.removeprefix("model.") for k in TENSORS)   # the encoders only, never the text model
        assert all(torch.equal(t, torch.full((2, 2), value)) for t in written.values())
