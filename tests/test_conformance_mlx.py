"""MLX conformance (#64): the tiny Gemma 4 fixture (tests/golden/tiny-gemma4) through the MLX backend must give golden.json's
token ids exactly and its fp32 probabilities within FP32_TOLERANCE, both as a LoRA checkpoint (the adapter merged at load)
and as an export folder (d1a.backends.mlx.export_mlx, the path of every served build); an 8-bit export keeps the ids and
flips no answer the reference decides by more than twice its largest error (measured 10-11: max |dp| 0.064, mean 0.0076
over 154 questions). mlx-lm's Gemma 4 model is the text model under a multimodal wrapper whose photo and voice towers it
drops on load, so the fixture gets that wrapper here (key names and config only) and runs through the same code as the
served exports. The MLX tests need Apple Silicon; the base_dir test runs anywhere.
    uv run python -m pytest tests/test_conformance_mlx.py -v
"""
import json
import shutil
from pathlib import Path

import pytest
import torch

from d1a.backends.checkpoint import Meta, base_dir, mlx_available, read_meta, write_meta

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "golden" / "tiny-gemma4"
FP32_TOLERANCE = 1e-5   # the torch bar; measured 3.0e-6 on the CPU and 2.6e-6 on Metal (10-11)
needs_mlx = pytest.mark.skipif(not mlx_available(), reason="the MLX backend needs mlx-lm on Apple Silicon")


def test_a_local_base_directory_is_used_as_it_is(tmp_path, monkeypatch):
    from d1a.backends import checkpoint
    calls = []
    monkeypatch.setattr(checkpoint, "resolve_run", lambda run: calls.append(run) or "/hub/snapshot")
    assert base_dir(Meta(base=str(tmp_path), base_revision="abc")) == str(tmp_path) and not calls
    assert base_dir(Meta(base="google/gemma-4-E4B", base_revision="abc")) == "/hub/snapshot"
    assert calls == ["google/gemma-4-E4B@abc"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """The fixture's checkpoint, its base given the multimodal Gemma 4 layout (model.language_model.*, a text_config)."""
    from safetensors.numpy import load_file, save_file
    tmp = tmp_path_factory.mktemp("mlx-conformance")
    base, ck = tmp / "base", tmp / "checkpoint"
    base.mkdir()
    weights = load_file(str(FIXTURE / "base" / "model.safetensors"))
    save_file({k.replace("model.", "model.language_model.", 1): v for k, v in weights.items()}, str(base / "model.safetensors"))
    text = json.loads((FIXTURE / "base" / "config.json").read_text(encoding="utf-8"))
    # transformers sizes the full-attention heads from the weights when the config leaves global_head_dim out; mlx-lm
    # defaults it to 512, so it is written here from the first full-attention layer's query projection
    full = text["layer_types"].index("full_attention")
    text = {"global_head_dim": weights[f"model.layers.{full}.self_attn.q_proj.weight"].shape[0] // text["num_attention_heads"], **text}
    config = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "vocab_size": text["vocab_size"], "text_config": text}
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(FIXTURE / "base" / name, base / name)
    shutil.copytree(FIXTURE / "checkpoint", ck)
    meta = read_meta(ck)
    meta.base = str(base)   # absolute, where the committed checkpoint names its base relative to the repository
    write_meta(ck, meta)
    return ck


@pytest.fixture(scope="module")
def golden():
    return json.loads((FIXTURE / "golden.json").read_text(encoding="utf-8"))


def answers(run, golden):
    """Load `run` on the MLX backend; -> [(record id, golden question, probabilities)] over every question, after checking
    the token ids."""
    from d1a.backends.checkpoint import LoadOptions, load
    from d1a.backends.torch import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    from d1a.core.api import SystemOneRequest, to_record
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    out = []
    for record in golden["records"]:
        rec, _ = to_record(SystemOneRequest.model_validate(record["request"]))
        enc = model.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        assert enc["ids"] == record["ids"], record["id"]
        out += [(record["id"], q, p.float().cpu()) for p, q in zip(model.probs(enc), record["questions"], strict=True)]
    return out


def worst(found):
    return max((p - torch.tensor(q["probs"])).abs().max().item() for _, q, p in found)


@needs_mlx
def test_a_lora_checkpoint_matches_the_golden_vectors(checkpoint, golden):
    err = worst(answers(checkpoint, golden))
    assert err < FP32_TOLERANCE, err


@needs_mlx
def test_an_export_folder_matches_the_golden_vectors(checkpoint, golden, tmp_path):
    from d1a.backends.checkpoint import Checkpoint
    from d1a.backends.mlx import export_mlx
    export_mlx(Checkpoint(checkpoint), tmp_path / "export")
    err = worst(answers(tmp_path / "export", golden))
    assert err < FP32_TOLERANCE, err


@needs_mlx
def test_an_8bit_export_keeps_the_ids_and_every_decided_answer(checkpoint, golden, tmp_path):
    """The tiny model's 32-wide random weights quantize far worse than real ones (the README's parity table holds the
    real-weight numbers), so the bar here is ids and decided answers, with the error measured, not a fixed bound."""
    from d1a.backends.checkpoint import Checkpoint
    from d1a.backends.mlx import export_mlx
    export_mlx(Checkpoint(checkpoint), tmp_path / "q8", bits=8, group_size=32, embeddings=False)   # 32: the tiny hidden size
    found = answers(tmp_path / "q8", golden); err = worst(found)
    for rid, q, p in found:
        ref = torch.tensor(q["probs"])
        if ref.numel() > 1 and float(ref.topk(2).values.diff().abs()) > 2 * err:   # decided: the top-2 margin exceeds twice the error
            assert int(p.argmax()) == int(ref.argmax()), (rid, q["qid"], err)


@needs_mlx
def test_a_q4_0_export_holds_llama_cpps_q4_0_grid_exactly(checkpoint, golden, tmp_path):
    """--q4-0 (#270): every quantized weight stores Q4_0 of the unquantized export's weight exactly (its 4-bit values,
    scale d and bias -8 d), and the export loads and answers with the golden token ids."""
    import mlx.core as mx
    import numpy as np
    from d1a.backends.checkpoint import Checkpoint
    from d1a.backends.mlx import export_mlx
    from d1a.backends.q4_0 import q4_0, unpack
    export_mlx(Checkpoint(checkpoint), tmp_path / "bf16")
    cfg = export_mlx(Checkpoint(checkpoint), tmp_path / "q4", q4_0_grid=True)
    assert cfg["quantization"]["grid"] == "q4_0" and cfg["quantization"]["bits"] == 4 and cfg["quantization"]["group_size"] == 32
    plain, quant = mx.load(str(tmp_path / "bf16" / "model.safetensors")), mx.load(str(tmp_path / "q4" / "model.safetensors"))
    keys = [k.removesuffix(".scales") for k in quant if k.endswith(".scales")]
    assert keys
    for k in keys:
        w = plain[k + ".weight"]
        q, d = q4_0(np.array(w.astype(mx.float32)))
        assert np.array_equal(unpack(np.array(quant[k + ".weight"])), q), k
        assert np.array_equal(np.array(quant[k + ".scales"].astype(mx.float32)), np.array(mx.array(d).astype(w.dtype).astype(mx.float32))), k
        assert np.array_equal(np.array(quant[k + ".biases"].astype(mx.float32)), np.array(mx.array(-8 * d).astype(w.dtype).astype(mx.float32))), k
    answers(tmp_path / "q4", golden)   # checks every record's token ids


@needs_mlx
def test_an_unmerged_export_answers_like_the_merged_one_and_switches_back_to_the_base(checkpoint, golden, tmp_path):
    """#274: export_mlx(unmerged=True) keeps the adapter beside the base. With it on, the golden answers hold within the
    fp32 bar, as the merged export's do; with it off (adapter(False), the chat path) the hidden states are the base
    model's exactly; on again, they are unchanged."""
    import mlx.core as mx
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, base_dir, load
    from d1a.backends.mlx import export_mlx, load_mlx_lm
    cfg = export_mlx(Checkpoint(checkpoint), tmp_path / "unmerged", unmerged=True)
    assert cfg["adapter"] == "unmerged"
    err = worst(answers(tmp_path / "unmerged", golden))
    assert err < FP32_TOLERANCE, err
    tok, model = load(str(tmp_path / "unmerged"), "mps", LoadOptions(backend="mlx"))
    assert model.loras
    base, _ = load_mlx_lm(base_dir(read_meta(checkpoint)))
    ids = mx.array([tok("Order 4411 arrived two weeks late and the box was crushed.")["input_ids"]])
    on = model.text(ids)
    with model.adapter(False):
        off = model.text(ids)
    again = model.text(ids)
    assert mx.array_equal(off, base.language_model.model(ids))
    assert not mx.array_equal(on, off) and mx.array_equal(on, again)


@needs_mlx
def test_a_merged_model_has_no_adapter_switch(checkpoint, tmp_path):
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    export_mlx(Checkpoint(checkpoint), tmp_path / "merged")
    _, model = load(str(tmp_path / "merged"), "mps", LoadOptions(backend="mlx"))
    with pytest.raises(ValueError, match="merged"):
        with model.adapter(False):
            pass


@needs_mlx
def test_a_chat_runs_with_the_adapter_off_and_leaves_decisions_unchanged(checkpoint, golden, tmp_path):
    """#274: generate_chat (the /v1/chat/completions body) on an unmerged export: an OpenAI-shaped answer within its
    token limit, tool_calls present whenever tools are sent, and the adapter back on afterwards, so the golden answers
    are exactly what they were before the chat."""
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    from d1a.backends.torch import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.serving.openai_compat import ChatRequest, generate_chat
    run = tmp_path / "unmerged"
    export_mlx(Checkpoint(checkpoint), run, unmerged=True)
    (run / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")   # the tiny fixture has none
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    encs = [model.encode(tok, to_record(SystemOneRequest.model_validate(r["request"]))[0], max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            for r in golden["records"][:8]]
    before = [p for e in encs for p in model.probs(e)]
    req = ChatRequest(messages=[{"role": "user", "content": "Order 4411 arrived late."}], max_tokens=4, temperature=0.0,
                      tools=[{"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}])
    body = generate_chat(model, tok, req, run)
    choice = body["choices"][0]
    assert body["object"] == "chat.completion" and choice["message"]["role"] == "assistant" and choice["message"]["tool_calls"] == []
    assert 0 < body["usage"]["completion_tokens"] <= 4 and model.loras
    after = [p for e in encs for p in model.probs(e)]   # the same model: the adapter must be back on
    assert all(torch.equal(p, q) for p, q in zip(before, after))
