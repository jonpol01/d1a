"""Gemma 4's unified model (#270: the 12B, google/gemma-4-12B-it-qat-*): its text stack is the E-series one with per-layer
embeddings and KV-shared layers off, so d1a's torch backend runs it through transformers and the MLX backend through mlx-lm's
gemma4 classes (d1a.backends.mlx.unified_classes). A tiny random-weight unified text model, saved once as transformers'
text-only checkpoint and once in the multimodal layout the 12B ships (model.language_model.*, a text_config and the unified
vision embedder, which the MLX load drops), must give the same token ids and, through one pointer head, the same
probabilities on both backends within the fp32 bar.
    uv run python -m pytest tests/test_gemma4_unified.py -v
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from d1a.backends.checkpoint import mlx_available

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "golden" / "tiny-gemma4"   # its tokenizer and golden requests
FP32_TOLERANCE = 1e-5
needs_mlx = pytest.mark.skipif(not mlx_available(), reason="the MLX backend needs mlx-lm on Apple Silicon")
LAYERS = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"]


def text_config():
    from transformers import Gemma4UnifiedTextConfig
    return Gemma4UnifiedTextConfig(
        vocab_size=38, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=4, layer_types=LAYERS,
        rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                         "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
        pad_token_id=0, bos_token_id=2, eos_token_id=3, tie_word_embeddings=True, dtype="float32")


@pytest.fixture(scope="module")
def bases(tmp_path_factory):
    """(text-only dir, multimodal-layout dir) holding the same random weights."""
    from safetensors.numpy import load_file, save_file
    from transformers import Gemma4UnifiedForCausalLM
    tmp = tmp_path_factory.mktemp("gemma4-unified")
    text, multi = tmp / "text", tmp / "multimodal"
    torch.manual_seed(0)
    Gemma4UnifiedForCausalLM(text_config()).save_pretrained(text)
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copyfile(FIXTURE / "base" / name, text / name)
    shutil.copytree(text, multi)
    weights = load_file(str(text / "model.safetensors"))
    weights = {k.replace("model.", "model.language_model.", 1): v for k, v in weights.items() if k.startswith("model.")}
    weights["model.vision_embedder.patch_dense.weight"] = np.zeros((4, 4), np.float32)   # dropped on load, as in the 12B
    save_file(weights, str(multi / "model.safetensors"))
    tc = json.loads((text / "config.json").read_text(encoding="utf-8"))
    # transformers sizes the full-attention heads from the weights and saves no global_head_dim; mlx-lm defaults it to 512
    # (the 12B's config states its 512), so it is written here from the first full-attention layer's query projection
    full = LAYERS.index("full_attention")
    tc["global_head_dim"] = weights[f"model.language_model.layers.{full}.self_attn.q_proj.weight"].shape[0] // tc["num_attention_heads"]
    config = {"architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified", "text_config": tc, "tie_word_embeddings": True}
    (multi / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return text, multi


def test_the_unified_text_model_is_d1as_gemma4_backbone(bases):
    from d1a.backends.torch import DecisionModel, load_tokenizer
    m = DecisionModel(str(bases[0]), load_tokenizer(str(bases[0])), "cpu")
    assert m.lm.config.model_type == "gemma4_unified_text"
    assert m.backbone.name == "gemma4" and m.backbone.mlx and m.sliding_window == 4 and not m.hybrid


@needs_mlx
def test_torch_and_mlx_give_the_same_answers(bases):
    from d1a.backends.mlx import MLXDecisionModel, load_mlx_lm
    from d1a.backends.torch import SERVE_MAX_BRANCH, SERVE_MAX_STATE, DecisionModel, load_tokenizer
    from d1a.core.api import SystemOneRequest, to_record
    tok = load_tokenizer(str(bases[0]))
    torch.manual_seed(1)
    ref = DecisionModel(str(bases[0]), tok, "cpu").eval()
    lm, _ = load_mlx_lm(bases[1])   # strict apart from the dropped vision embedder: every parameter loaded
    mlx = MLXDecisionModel.from_lm(lm, ref.pad_id)
    mlx.head.load_state_dict(ref.head.state_dict())
    golden = json.loads((FIXTURE / "golden.json").read_text(encoding="utf-8"))
    worst, n = 0.0, 0
    with torch.no_grad():
        for record in golden["records"]:
            rec, _ = to_record(SystemOneRequest.model_validate(record["request"]))
            a = ref.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            b = mlx.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            assert a["ids"] == b["ids"], record["id"]
            for p, q in zip(ref.probs(a), mlx.probs(b), strict=True):
                worst = max(worst, float((p.float().cpu() - q.float().cpu()).abs().max())); n += 1
    print(f"questions {n}, max |dp| torch vs MLX {worst:.2e}")
    assert n >= 100 and worst <= FP32_TOLERANCE, (n, worst)
