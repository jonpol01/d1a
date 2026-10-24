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


def ct_checkpoint(base, out):
    """`base` as a compressed-tensors QAT checkpoint would store it (pack-quantized int4, symmetric, groups of 32): every
    Linear weight whose width is a multiple of 32 becomes weight_packed (8 values per int32 from the low nibble up, each
    q + 8), weight_scale and weight_shape; the rest is kept. -> {name: (q int8, scale)}"""
    import numpy as np
    from safetensors.numpy import load_file, save_file
    out.mkdir()
    weights, stored, quantized = load_file(str(base / "model.safetensors")), {}, {}
    for k, w in weights.items():
        if not (k.endswith("_proj.weight") or k.endswith("_gate.weight") or k.endswith("_projection.weight")) or w.ndim != 2 or w.shape[1] % 32:
            stored[k] = w
            continue
        groups = w.astype(np.float32).reshape(w.shape[0], -1, 32)
        scale = (np.abs(groups).max(-1) / 7 + 1e-8).astype(w.dtype)
        q = np.clip(np.round(groups / scale.astype(np.float32)[..., None]), -8, 7).astype(np.int8).reshape(w.shape)
        u = (q.astype(np.int32) + 8).reshape(w.shape[0], -1, 8).astype(np.uint32) << (4 * np.arange(8, dtype=np.uint32))
        name = k[:-len(".weight")]
        stored |= {name + ".weight_packed": u.sum(-1, dtype=np.uint32).view(np.int32), name + ".weight_scale": scale,
                   name + ".weight_shape": np.array(w.shape, np.int64)}
        quantized[name] = (q, scale)
    save_file(stored, str(out / "model.safetensors"))
    shutil.copyfile(base / "config.json", out / "config.json")
    return quantized


@needs_mlx
def test_a_ct_base_export_holds_the_checkpoints_own_int4_grid_exactly(checkpoint, golden, tmp_path):
    """--ct-base (d1a#341): the export keeps a compressed-tensors QAT checkpoint's int4 Linears as it stores them (the
    words, scale s and bias -8 s, so each weight dequantizes to s q exactly) and its other tensors as they are, and it
    loads and answers. A merged adapter would leave the grid, so ct_base refuses it."""
    import mlx.core as mx
    import numpy as np
    from d1a.backends.checkpoint import Checkpoint, base_dir
    from d1a.backends.mlx import export_mlx
    base = Path(base_dir(Checkpoint(checkpoint).meta))
    quantized = ct_checkpoint(base, tmp_path / "ct")
    assert quantized
    with pytest.raises(ValueError, match="unmerged"):
        export_mlx(Checkpoint(checkpoint), tmp_path / "merged", ct_base=tmp_path / "ct")
    cfg = export_mlx(Checkpoint(checkpoint), tmp_path / "q4", unmerged=True, ct_base=tmp_path / "ct")
    assert cfg["quantization"]["grid"] == "ct-int4" and cfg["quantization"]["bits"] == 4 and cfg["quantization"]["group_size"] == 32
    out, ct = mx.load(str(tmp_path / "q4" / "model.safetensors")), mx.load(str(tmp_path / "ct" / "model.safetensors"))
    for name, (q, scale) in quantized.items():
        m = "language_model.model." + name.split("model.language_model.", 1)[1]
        assert mx.array_equal(out[m + ".weight"], ct[name + ".weight_packed"].view(mx.uint32)), name
        assert mx.array_equal(out[m + ".scales"], ct[name + ".weight_scale"]) and mx.array_equal(out[m + ".biases"], ct[name + ".weight_scale"] * -8), name
        deq = mx.dequantize(out[m + ".weight"], out[m + ".scales"].astype(mx.float32), out[m + ".biases"].astype(mx.float32), group_size=32, bits=4)
        ref = (q.astype(np.float32).reshape(q.shape[0], -1, 32) * scale.astype(np.float32)[..., None]).reshape(q.shape)
        assert np.array_equal(np.array(deq), ref), name
    norm = next(k for k in ct if k.endswith("layers.0.input_layernorm.weight"))
    assert mx.array_equal(out["language_model.model." + norm.split("model.language_model.", 1)[1]], ct[norm])
    answers(tmp_path / "q4", golden)   # loads, and checks every record's token ids
    cfg8 = export_mlx(Checkpoint(checkpoint), tmp_path / "q4e8", unmerged=True, ct_base=tmp_path / "ct", embedding_bits=8)
    widths = {k: v["bits"] for k, v in cfg8["quantization"].items() if isinstance(v, dict) and "bits" in v}
    assert widths and all(b == 8 for k, b in widths.items() if "embed" in k), widths   # the embeddings at 8 bits, the Linears as before
    out8 = mx.load(str(tmp_path / "q4e8" / "model.safetensors"))
    name = next(iter(quantized)); m = "language_model.model." + name.split("model.language_model.", 1)[1]
    assert mx.array_equal(out8[m + ".weight"], out[m + ".weight"])


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


@needs_mlx
def test_decisions_between_a_chats_tokens_leave_both_unchanged(checkpoint, golden, tmp_path):
    """#289: Server.between_tokens runs decisions inside a chat's adapter(False) block, between two of its tokens. The
    decisions must read the adapter (the probabilities of a model at rest) and the chat's tokens must be the ones it
    generates uninterrupted, greedy and sampled from a seed."""
    import mlx.core as mx
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    from d1a.backends.torch import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.serving.openai_compat import ChatRequest, generate_chat
    run = tmp_path / "unmerged"
    export_mlx(Checkpoint(checkpoint), run, unmerged=True)
    (run / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    encs = [model.encode(tok, to_record(SystemOneRequest.model_validate(r["request"]))[0], max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            for r in golden["records"][:4]]
    at_rest = [p for e in encs for p in model.probs(e)]
    for temperature in (0.0, 1.0):
        req = ChatRequest(messages=[{"role": "user", "content": "Order 4411 arrived late."}], max_tokens=12, temperature=temperature)
        mx.random.seed(7)
        alone = generate_chat(model, tok, req, run)
        mid_chat = []

        def between():
            with model.adapter(True):
                mid_chat.extend(p for e in encs for p in model.probs(e))
        mx.random.seed(7)
        interrupted = generate_chat(model, tok, req, run, between=between)
        tokens = interrupted["usage"]["completion_tokens"]
        assert interrupted["choices"] == alone["choices"] and tokens == alone["usage"]["completion_tokens"] > 1
        # at least once per token; the prefill's progress calls (#290) add a few more
        assert len(mid_chat) >= len(at_rest) * tokens and len(mid_chat) % len(at_rest) == 0
        assert all(torch.equal(p, at_rest[i % len(at_rest)]) for i, p in enumerate(mid_chat))


@needs_mlx
def test_decisions_between_a_chats_prefill_chunks_leave_both_unchanged(checkpoint, golden, tmp_path, monkeypatch):
    """#290: a long prompt is prefilled in PREFILL_STEP-token chunks (mlx-lm's own), and between() runs between them, so
    a decision waits for one chunk, not the whole prompt. Prefilled with cut-ins, the chat's tokens are the ones an
    uninterrupted prefill gives, and the decisions read the adapter."""
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    from d1a.backends.torch import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    from d1a.core.api import SystemOneRequest, to_record
    from d1a.serving import openai_compat
    from d1a.serving.openai_compat import ChatRequest, generate_chat
    run = tmp_path / "unmerged"
    export_mlx(Checkpoint(checkpoint), run, unmerged=True)
    (run / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    encs = [model.encode(tok, to_record(SystemOneRequest.model_validate(r["request"]))[0], max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            for r in golden["records"][:2]]
    at_rest = [p for e in encs for p in model.probs(e)]
    monkeypatch.setattr(openai_compat, "PREFILL_STEP", 8)   # the tiny prompt below spans several chunks
    req = ChatRequest(messages=[{"role": "user", "content": "Order 4411 arrived late and the box was crushed; the customer wants a refund today."}],
                      max_tokens=4, temperature=0.0)
    alone = generate_chat(model, tok, req, run)
    calls, mid_chat, first_token = [], [], []

    def between():
        calls.append(len(first_token))
        with model.adapter(True):
            mid_chat.extend(p for e in encs for p in model.probs(e))
    interrupted = generate_chat(model, tok, req, run, on_text=lambda piece: first_token.append(piece), between=between)
    assert interrupted["choices"] == alone["choices"]
    assert calls.count(0) >= 3   # before the first token: the start of the prefill and between its chunks
    assert all(torch.equal(p, at_rest[i % len(at_rest)]) for i, p in enumerate(mid_chat))


@needs_mlx
def test_a_chat_continuing_a_kept_prompt_prefills_only_its_new_tokens(checkpoint, tmp_path, monkeypatch):
    """The chat prompt cache: an agent's next step (its earlier prompt, the answer, a new message) takes the earlier
    prompt's cache and prefills only the rest, and builds the cache, and the answer, a cold prefill gives. The fixture's
    sliding window (4 tokens) is far shorter than these prompts, so the kept caches cannot be trimmed, as on E4B-it past
    512 tokens. A new session with the same system prompt takes the system prompt's cache; a prompt that only shares
    part of a kept one is prefilled cold."""
    import mlx.core as mx
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    from d1a.serving import openai_compat
    from d1a.serving.openai_compat import ChatRequest, PromptCache, generate_chat
    run = tmp_path / "unmerged"
    export_mlx(Checkpoint(checkpoint), run, unmerged=True)
    (run / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    monkeypatch.setattr(openai_compat, "PREFILL_STEP", 8)   # several chunks, and the new tokens start mid-chunk
    system = {"role": "system", "content": "You are the support desk's agent: check the order, then act on it.\n"}
    first = [system, {"role": "user", "content": "Order 4411 arrived late and the box was crushed; the customer wants a refund.\n"}]
    step = first + [{"role": "assistant", "content": "Refund issued for order 4411.\n"},
                    {"role": "user", "content": "Now email the customer and close the ticket.\n"}]
    other = [{"role": "user", "content": "Order 4411 arrived late and the box was crushed; the customer wants a replacement.\n"}]
    ask = lambda messages, cache=None: generate_chat(model, tok, ChatRequest(messages=messages, max_tokens=6, temperature=0.0), run, cache=cache)
    cache = PromptCache(1 << 30)
    n_first, n_step = (len(tok("".join(m["content"] for m in ms))["input_ids"]) for ms in (first, step))   # with <bos>, as the chat encodes
    assert ask(first, cache)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    warm = ask(step, cache)
    assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] == n_first - 1   # the kept prompt but its last token
    assert warm["choices"] == ask(step)["choices"]
    assert ask(step, cache)["usage"]["prompt_tokens_details"]["cached_tokens"] == n_step - 1   # the same prompt again
    assert ask(other, cache)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    session = [system, {"role": "user", "content": "Order 5120 never arrived.\n"}]
    fresh = ask(session, cache)
    assert fresh["usage"]["prompt_tokens_details"]["cached_tokens"] == len(tok(system["content"])["input_ids"])
    assert fresh["choices"] == ask(session)["choices"]
    # the kept cache of the step's prompt, built warm, gives the next-token logits a cold prefill gives
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    prompt = "".join(m["content"] for m in step)
    ids, cold = tok(prompt)["input_ids"], PromptCache(1 << 30)
    with model.adapter(False):
        list(openai_compat.pieces(model.lm, TokenizerWrapper(tok), prompt, 1, lambda x: mx.argmax(x, axis=-1), cache=cold))
        (a, rest), (b, _) = cache.fetch(ids), cold.fetch(ids)
        assert rest == ids[-1:]
        warm_logits, cold_logits = (model.lm(mx.array([rest]), cache=c) for c in (a, b))
    assert mx.allclose(warm_logits, cold_logits, atol=1e-4)


@needs_mlx
def test_an_agents_step_after_a_tool_call_answers_and_reuses_the_prompt_cache(checkpoint, tmp_path):
    """A real agent loop: step 1 asks with tools, step 2 sends back the assistant's tool call (OpenAI: arguments as a JSON
    string) and the tool's result. The template refuses string arguments as Gemma 4's does. Step 2 answers, and it takes
    step 1's prompt from the cache."""
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    from d1a.serving.openai_compat import ChatRequest, PromptCache, generate_chat
    run = tmp_path / "unmerged"
    export_mlx(Checkpoint(checkpoint), run, unmerged=True)
    template = ("{% for m in messages %}{% for c in m.get('tool_calls') or [] %}{% if c.function.arguments is string %}"
                "{{ raise_exception('tool_calls[].function.arguments must be a JSON object') }}{% endif %}"
                "{{ c.function.name }}({{ c.function.arguments.command }})\n{% endfor %}{{ m['content'] or '' }}\n{% endfor %}")
    (run / "chat_template.jinja").write_text(template, encoding="utf-8")
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    tools = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
    step1 = [{"role": "system", "content": "You are an operations agent on the serving Mac."}, {"role": "user", "content": "How full is the disk?"}]
    step2 = step1 + [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "df -h /"}'}}]},
                     {"role": "tool", "tool_call_id": "call_1", "content": "/dev/disk3s1s1 228Gi 13Gi 12Gi 52% /"}]
    cache = PromptCache(1 << 30)
    ask = lambda messages: generate_chat(model, tok, ChatRequest(messages=messages, tools=tools, max_tokens=4, temperature=0.0), run, cache=cache)
    assert ask(step1)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    second = ask(step2)
    assert second["choices"][0]["finish_reason"] in ("stop", "length", "tool_calls")
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


@needs_mlx
def test_an_empty_chat_answer_logs_its_context_without_text(checkpoint, tmp_path, monkeypatch, caplog):
    """generate_chat's log line for an answer with no text and no tool call ends with the roles before it; one cut off by
    max_tokens before any text says that instead (#314); any other answer's line is unchanged. No message text reaches the
    log."""
    import logging
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    from d1a.serving import openai_compat
    from d1a.serving.openai_compat import ChatRequest, generate_chat
    run = tmp_path / "unmerged"
    export_mlx(Checkpoint(checkpoint), run, unmerged=True)
    (run / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] or '' }}{% endfor %}", encoding="utf-8")
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    messages = [{"role": "user", "content": "the private question"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c", "content": "the private tool output"}]
    caplog.set_level(logging.INFO, logger="d1a.chat")
    monkeypatch.setattr(openai_compat, "pieces", lambda *a, **k: iter([("", 0, "stop")]))   # the model ends its turn at once
    body = generate_chat(model, tok, ChatRequest(messages=messages, max_tokens=4), run)
    assert body["choices"][0]["message"]["content"] == "" and body["usage"]["completion_tokens"] == 0
    line = caplog.records[-1].getMessage()
    assert line.endswith(", empty answer after user > assistant(tool_calls) > tool") and "private" not in line
    monkeypatch.setattr(openai_compat, "pieces", lambda *a, **k: iter([("", 4, "length")]))   # a tool call cut short at max_tokens
    generate_chat(model, tok, ChatRequest(messages=messages, max_tokens=4), run)
    line = caplog.records[-1].getMessage()
    assert line.endswith(", cut off by max_tokens before any text") and "empty answer" not in line
    monkeypatch.setattr(openai_compat, "pieces", lambda *a, **k: iter([("Fine.", 1, "stop")]))
    generate_chat(model, tok, ChatRequest(messages=messages, max_tokens=4), run)
    line = caplog.records[-1].getMessage()
    assert "empty answer" not in line and "cut off" not in line


@needs_mlx
def test_a_chat_past_the_models_context_is_refused_before_any_prefill(checkpoint, tmp_path, monkeypatch):
    """generate_chat refuses a prompt that, with its completion, exceeds the model's context, before the prefill: it
    raises ContextTooLong (served as a 400) and never runs the model. A prompt that fits answers as before."""
    from d1a.backends.checkpoint import Checkpoint, LoadOptions, load
    from d1a.backends.mlx import export_mlx
    from d1a.serving import openai_compat
    from d1a.serving.openai_compat import ChatRequest, ContextTooLong, generate_chat
    run = tmp_path / "unmerged"
    export_mlx(Checkpoint(checkpoint), run, unmerged=True)
    (run / "chat_template.jinja").write_text("{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")
    tok, model = load(str(run), "mps", LoadOptions(backend="mlx"))
    monkeypatch.setattr(openai_compat, "context_length", lambda model_dir, cap=None: 64)
    short = ChatRequest(messages=[{"role": "user", "content": "Order 4411 arrived late."}], max_tokens=4, temperature=0.0)
    assert generate_chat(model, tok, short, run)["usage"]["completion_tokens"] <= 4
    def no_prefill(*a, **k):
        raise AssertionError("the model ran on a prompt past its context")
    monkeypatch.setattr(openai_compat, "pieces", no_prefill)
    long = ChatRequest(messages=[{"role": "user", "content": "Order 4411 arrived late. " * 40}], max_tokens=4)
    with pytest.raises(ContextTooLong) as e:
        generate_chat(model, tok, long, run)
    assert e.value.context == 64 and e.value.prompt + e.value.completion > 64
