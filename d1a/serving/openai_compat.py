"""OpenAI-compatible chat and embeddings on D1A's server (#274), so apps that speak the OpenAI API (mem0, n8n, chat UIs)
use the same model D1A decides with. The model must have an UNMERGED adapter on an instruction-tuned base
(scripts/export_mlx.py --unmerged on D1A-E2B-it): chat runs with the adapter switched off, which is the base model
exactly (MLXDecisionModel.adapter), and decisions keep it on. Generation runs on the server's model thread, like every
forward pass, so a chat and a decision never overlap. Embeddings come from a separate small GGUF embedder run by
llama.cpp (D1A_EMBED_GGUF), the engine and file LM Studio served them with, so vectors stored earlier stay comparable.

    POST /v1/chat/completions   messages, max_tokens / max_completion_tokens, temperature, top_p, stop, tools,
                                tool_choice, response_format (no constrained decoding: the model is asked in the prompt),
                                stream (token by token; with tools, one chunk once the tool calls are parsed, then [DONE]);
                                a tool call cut off by max_tokens is dropped (finish_reason "length", #278)
    POST /v1/embeddings         input (a string or a list of strings), model, encoding_format ("float" or "base64");
                                `dimensions` is accepted and ignored
"""
import base64
import json
import logging
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("d1a.chat")
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS = 8192


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")   # OpenAI clients send fields this server has no use for (user, seed, ...)
    model: str | None = None
    messages: list[dict] = Field(min_length=1)
    max_tokens: int | None = Field(None, ge=1, le=MAX_TOKENS)
    max_completion_tokens: int | None = Field(None, ge=1, le=MAX_TOKENS)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    response_format: dict | None = None
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    input: str | list[str]
    dimensions: int | None = None        # mem0 always sends it; the embedder's own width is what it returns
    encoding_format: str | None = None


def text_of(content):
    """A message's content as text: OpenAI allows a string or a list of parts ({"type": "text", "text": ...})."""
    if content is None or isinstance(content, str):
        return content or ""
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def eos_ids(tok, model_dir):
    """The ids that end a turn: the base's generation_config.json eos_token_id (Gemma 4 -it: <eos>, <turn|> and one more),
    else the tokenizer's eos and <turn|>."""
    ids = set()
    cfg = Path(model_dir) / "generation_config.json" if model_dir else None
    if cfg is not None and cfg.exists():
        eos = json.loads(cfg.read_text(encoding="utf-8")).get("eos_token_id")
        ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    if tok.eos_token_id is not None: ids.add(tok.eos_token_id)
    turn = tok.convert_tokens_to_ids("<turn|>")
    if isinstance(turn, int) and turn != tok.unk_token_id: ids.add(turn)
    return ids


def tool_calls_of(text):
    """Gemma 4's <|tool_call>call:name{args}<tool_call|> blocks -> (the text before the first one, OpenAI tool_calls). A block
    cut off unfinished (max_tokens ran out inside it) is dropped, never returned as text: the markup means nothing to an
    OpenAI client, and finish_reason "length" already tells it to retry with more tokens (#278)."""
    from mlx_lm.tool_parsers import gemma4
    if gemma4.tool_call_start not in text:
        return text, []
    head = text.split(gemma4.tool_call_start, 1)[0]
    try:
        calls = gemma4.parse_tool_call(text)
    except ValueError:   # no complete call: only an unfinished one
        return head, []
    calls = calls if isinstance(calls, list) else [calls]
    return head, [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                   "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}} for c in calls]


def pieces(lm, tokenizer, prompt, limit, sampler):
    """mlx-lm's stream_generate loop (the same prompt encoding, end-of-turn check and streaming detokenizer) without its
    wired-memory context, which needs the Metal device and fails on the CPU one CI runs MLX on.
    -> (new text, tokens so far, None | "stop" | "length") per step; the last step names the finish."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    detok = tokenizer.detokenizer
    detok.reset()
    n = 0
    for n, (token, _) in enumerate(generate_step(mx.array(ids), lm, max_tokens=limit, sampler=sampler)):
        if token in tokenizer.eos_token_ids:
            detok.finalize(); yield detok.last_segment, n, "stop"; return
        detok.add_token(token)
        if n + 1 == limit:
            break
        yield detok.last_segment, n + 1, None
    detok.finalize()
    yield detok.last_segment, n + 1, "length"


def generate_chat(model, tok, req, model_dir=None, model_name="d1a", on_text=None):
    """One chat completion with the adapter off. Run on the model thread (Server.run_task). on_text, if given, receives
    each new piece of the answer as it is generated (streaming), never text past a stop string. -> the OpenAI response body."""
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    messages = [{**m, "content": text_of(m.get("content"))} for m in req.messages]
    if req.response_format and req.response_format.get("type") == "json_object":   # no constrained decoding: ask for it
        messages = [{"role": "system", "content": "Respond with a single valid JSON object only."}] + messages
    prompt = tok.apply_chat_template(messages, tools=req.tools or None, tokenize=False, add_generation_prompt=True)
    limit = req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS
    sampler = make_sampler(temp=req.temperature, top_p=req.top_p if req.top_p < 1.0 else 0.0)
    wrapper = TokenizerWrapper(tok, eos_token_ids=eos_ids(tok, model_dir))
    stops = [s for s in ([req.stop] if isinstance(req.stop, str) else req.stop or []) if s]
    t, text, sent, finish, completion_tokens = time.time(), "", 0, "length", 0
    with model.adapter(False):
        for piece, completion_tokens, ended in pieces(model.lm, wrapper, prompt, limit, sampler):
            text += piece
            if ended: finish = ended
            cut = min((text.index(s) for s in stops if s in text), default=None)
            if cut is not None:
                text, finish = text[:cut], "stop"
            if on_text is not None and len(text) > sent:
                on_text(text[sent:]); sent = len(text)
            if cut is not None:
                break
    prompt_tokens = len(tok(prompt, add_special_tokens=False)["input_ids"])
    log.info("chat: %d prompt tokens, %d completion tokens, %.1f s%s", prompt_tokens, completion_tokens, time.time() - t, ", tools" if req.tools else "")
    content, calls = tool_calls_of(text) if req.tools else (text, [])
    message = {"role": "assistant", "content": content.strip() or (None if calls else "")}
    if req.tools is not None:
        message["tool_calls"] = calls   # always a list when tools were sent: mem0's graph memory indexes it unconditionally
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(t), "model": req.model or model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}}


def chunk(body_id, created, model, delta, finish=None):
    """One OpenAI chat.completion.chunk event."""
    return "data: " + json.dumps({"id": body_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, ensure_ascii=False) + "\n\n"


def stream_chunks(body):
    """A finished completion as an OpenAI stream: one chunk with the whole message, one with the finish reason, [DONE].
    Used when `tools` are sent: tool calls can only be parsed from the whole answer (serve streams other chats token by
    token through generate_chat's on_text)."""
    choice = body["choices"][0]
    delta = {k: v for k, v in choice["message"].items() if v is not None}
    if "tool_calls" in delta:
        delta["tool_calls"] = [{**c, "index": i} for i, c in enumerate(delta["tool_calls"])]
    base = {k: body[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    yield f"data: {json.dumps(base | {'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(base | {'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice['finish_reason']}]})}\n\n"
    yield "data: [DONE]\n\n"


def encoded(vector, encoding_format):
    """An embedding as OpenAI returns it: a list of floats, or with encoding_format "base64" the base64 of its little-endian
    float32 bytes. The openai Node SDK (n8n) always asks for base64 and decodes it, so a list garbles its vectors (#282)."""
    if encoding_format == "base64":
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
    return vector


class Embedder:
    """A GGUF embedding model (nomic-embed-text-v1.5 by default) through llama.cpp, loaded on first use. llama.cpp runs
    its own Metal context, so it needs neither the model thread nor MLX; one lock serializes its calls."""

    def __init__(self, gguf, name):
        self.gguf, self.name, self.lock, self.llm = gguf, name, threading.Lock(), None

    def embed(self, texts):
        with self.lock:
            if self.llm is None:
                from llama_cpp import Llama
                # every layer on the GPU, as LM Studio runs it: llama.cpp's CPU path (llama-cpp-python's default) gives vectors
                # only ~0.99 cosine to the ones LM Studio stored; Metal gives the same vectors (cosine 1.00000)
                self.llm = Llama(model_path=str(self.gguf), embedding=True, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            return [self.llm.embed(t, normalize=True) for t in texts]   # unit length, as LM Studio served them: stored vectors are compared by distance, not only by angle

    def response(self, req):
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        vectors = self.embed(texts)
        tokens = sum(len(self.llm.tokenize(t.encode("utf-8"))) for t in texts)
        return {"object": "list", "model": req.model or self.name,
                "data": [{"object": "embedding", "index": i, "embedding": encoded(v, req.encoding_format)} for i, v in enumerate(vectors)],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}
