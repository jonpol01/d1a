"""OpenAI-compatible chat and embeddings on D1A's server (#274), so apps that speak the OpenAI API (mem0, n8n, chat UIs)
use the same model D1A decides with. The model must have an UNMERGED adapter on an instruction-tuned base
(scripts/export_mlx.py --unmerged on D1A-E2B-it): chat runs with the adapter switched off, which is the base model
exactly (MLXDecisionModel.adapter), and decisions keep it on. Generation runs on the server's model thread, like every
forward pass, so a chat and a decision never overlap; decisions that arrive during a chat run between two of its tokens
(#289) or between two chunks of its prompt's prefill (#290). A chat whose prompt starts with an earlier one's (an agent's
next step, a conversation's next turn) prefills only what is new: its prompt cache (PromptCache) keeps the KV cache of the
latest prompts. Embeddings come from a separate small GGUF embedder run by
llama.cpp (D1A_EMBED_GGUF), the engine and file LM Studio served them with, so vectors stored earlier stay comparable.

    POST /v1/chat/completions   messages, max_tokens / max_completion_tokens, temperature, top_p, stop, tools,
                                tool_choice, response_format (no constrained decoding: the model is asked in the prompt),
                                stream (token by token; with tools, one chunk once the tool calls are parsed, then [DONE]);
                                a tool call cut off by max_tokens is dropped (finish_reason "length", #278);
                                usage.prompt_tokens_details.cached_tokens: the prompt tokens taken from the prompt cache
    POST /v1/embeddings         input (a string or a list of strings), model, encoding_format ("float" or "base64");
                                `dimensions` is accepted and ignored
"""
import base64
import copy
import functools
import json
import logging
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger("d1a.chat")
DEFAULT_MAX_TOKENS = 1024
PREFILL_STEP = 2048   # tokens per prefill chunk: mlx-lm's default, so a chat's cache matches generate_step's own
MAX_TOKENS = 8192
CACHE_PROMPTS = 16         # prompts whose KV cache is kept. A chat keeps two (its system prompt, evicted last, and its
                           # whole prompt), so a few agents taking turns each keep both; the byte budget is the real
                           # limit (D1A_CHAT_CACHE_MB). E4B-it: ~8 KB a token (its 4 full-attention layers) + ~0.1 GB a
                           # prompt (20 sliding-window layers), so 1 GiB holds three or four 20k-token prompts


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

    @field_validator("messages")
    @classmethod
    def _tool_call_arguments(cls, messages):
        """An agent's next step sends back the assistant's earlier tool calls. OpenAI gives each call's `arguments` as a JSON
        string, and Gemma 4's chat template refuses anything but an object, so every step after a tool call failed (500).
        Each one is decoded here; arguments that are not a JSON object are refused (422)."""
        out = []
        for m in messages:
            calls = m.get("tool_calls") if isinstance(m, dict) else None
            if calls:
                m = {**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": arguments_of(c["function"].get("arguments"))}}
                                         if isinstance(c, dict) and isinstance(c.get("function"), dict) else c for c in calls]}
            out.append(m)
        return out


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    input: str | list[str]
    dimensions: int | None = None        # mem0 always sends it; the embedder's own width is what it returns
    encoding_format: str | None = None


def arguments_of(arguments):
    """A tool call's arguments as an object: OpenAI clients send a JSON string ("" or none: no arguments)."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise ValueError(f"tool_calls[].function.arguments is not JSON: {e}") from None
    if not isinstance(arguments, dict):
        raise ValueError("tool_calls[].function.arguments must be a JSON object")
    return arguments


ROLES = {"system", "developer", "user", "assistant", "tool"}   # the OpenAI chat roles


class ContextTooLong(ValueError):
    """A chat whose prompt and requested completion do not fit the model's context, refused before any prefill: past it
    the model reads positions it was never trained on, and a 237k-token prompt held the model thread 16 minutes."""

    def __init__(self, prompt, completion, context):
        self.prompt, self.completion, self.context = prompt, completion, context
        super().__init__(f"This model's maximum context length is {context} tokens. However, you requested {prompt + completion} tokens "
                         f"({prompt} in the messages, {completion} in the completion). Please reduce the length of the messages or completion.")

    def body(self):
        """OpenAI's error body: clients read code context_length_exceeded as "compress and retry" (Hermes does)."""
        return {"error": {"message": str(self), "type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"}}


@functools.lru_cache(maxsize=8)
def model_context(model_dir):
    """The model's context in tokens: config.json's max_position_embeddings (Gemma 4: under text_config); None when there
    is no config or it names none."""
    cfg = Path(model_dir) / "config.json" if model_dir else None
    if cfg is None or not cfg.exists():
        return None
    c = json.loads(cfg.read_text(encoding="utf-8"))
    n = (c.get("text_config") or {}).get("max_position_embeddings") or c.get("max_position_embeddings")
    return int(n) if n else None


def context_length(model_dir, cap=None):
    """The longest prompt plus completion a chat may ask for: the model's context, or `cap` (D1A_CHAT_CONTEXT) when that
    is lower. A prompt near the model's context still holds the one model thread for minutes (prefill ~230 tokens/s on
    an M4: 131k tokens ~9 min), and every other chat waits behind it. None: no limit."""
    model = model_context(model_dir)
    return min(x for x in (model, cap) if x) if (model or cap) else None


def chat_messages(req):
    """The request's messages as the chat template takes them: text content, and JSON mode asked for in a system message."""
    messages = [{**m, "content": text_of(m.get("content"))} for m in req.messages]
    if req.response_format and req.response_format.get("type") == "json_object":   # no constrained decoding: ask for it
        messages = [{"role": "system", "content": "Respond with a single valid JSON object only."}] + messages
    return messages


def completion_limit(req):
    return req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS


def check_context(tok, req, context, prompt=None):
    """-> the prompt's tokens; raises ContextTooLong when they and the completion asked for exceed `context` (None: no
    limit)."""
    prompt = prompt if prompt is not None else tok.apply_chat_template(chat_messages(req), tools=req.tools or None, tokenize=False, add_generation_prompt=True)
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    if context and n + completion_limit(req) > context:
        raise ContextTooLong(n, completion_limit(req), context)
    return n


def last_roles(messages, n=3):
    """The roles of the last messages, an assistant message that called tools marked: the context of an empty answer in
    the log, without any of its text. A role is the client's string, so one outside ROLES is logged as "other": it could
    carry text, or a newline that forges a log line."""
    role = lambda m: m["role"] if isinstance(m.get("role"), str) and m["role"] in ROLES else "other"
    return " > ".join(role(m) + ("(tool_calls)" if m.get("tool_calls") else "") for m in messages[-n:])


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
    from d1a.serving import gemma4_tools as gemma4   # mlx-lm's parser, importable without mlx-lm (PyTorch chat)
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


class PromptCache:
    """The KV caches of the latest chat prompts, adapter off (mlx-lm's LRUPromptCache, the one mlx_lm.server keeps). A
    prompt that starts with a kept one reuses its cache and prefills only the rest. Gemma 4's sliding-window layers
    cannot be cut back once a prompt passes their window, so a kept prompt is only reused whole, never trimmed: each
    cache is kept at its prompt's end, before any answer token, which is where the next turn's prompt continues it, and
    at the end of its system prompt, which a new session with the same system prompt continues. The system prompts are
    evicted last."""

    def __init__(self, nbytes, prompts=CACHE_PROMPTS):
        from mlx_lm.models.cache import LRUPromptCache
        self.lru = LRUPromptCache(max_size=prompts, max_bytes=nbytes)

    def fetch(self, ids):
        """-> (a copy of the cache of the longest usable kept prefix of ids, or None; the ids left to run, never empty)."""
        cache, rest = self.lru.fetch_nearest_cache("chat", ids)
        return (cache, rest) if rest else (None, ids)   # ids are a kept prompt exactly: its last token is not in it

    def keep(self, ids, cache, kind="assistant"):
        self.lru.insert_cache("chat", list(ids), copy.deepcopy(cache), cache_type=kind)


def pieces(lm, tokenizer, prompt, limit, sampler, between=None, cache=None, stats=None, system=None):
    """mlx-lm's stream_generate loop (the same prompt encoding, end-of-turn check and streaming detokenizer) without its
    wired-memory context, which needs the Metal device and fails on the CPU one CI runs MLX on. between, if given, is
    called between the prompt's prefill chunks (PREFILL_STEP tokens each, mlx-lm's own chunking, so the chat's cache is
    the one an uninterrupted prefill builds): a decision waits for one chunk of a long prompt, not all of it (#290).
    cache, a PromptCache, supplies the prompt's longest kept prefix and keeps this prompt, and its system prompt (system:
    the prompt's start that renders it); stats["cached"] is set to the prompt tokens it supplied. -> (new text, tokens so far, None | "stop" | "length") per step; the last step names the
    finish."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step, generation_stream
    from mlx_lm.models.cache import make_prompt_cache
    add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    ids = tokenizer.encode(prompt, add_special_tokens=add_special)
    kv, rest = cache.fetch(ids) if cache is not None else (None, ids)
    kv = kv if kv is not None else make_prompt_cache(lm)
    if stats is not None:
        stats["cached"] = len(ids) - len(rest)
    between = between or (lambda: None)
    # the prompt but its last token, as generate_step prefills it (its chunks, and one more cut at the system prompt's
    # end): the cache is kept there and here, before the last token's step starts the answer
    start, end = len(ids) - len(rest), len(ids) - 1
    cuts = set(range(start, end, PREFILL_STEP))
    mark = None
    if cache is not None and system:
        head = tokenizer.encode(system, add_special_tokens=add_special)
        mark = len(head) if start < len(head) < end and ids[:len(head)] == head else None
        cuts |= {mark} - {None}
    between()
    with mx.stream(generation_stream):
        for a, b in zip(sorted(cuts), sorted(cuts)[1:] + [end]):
            lm(mx.array(ids[a:b])[None], cache=kv)
            mx.eval([c.state for c in kv])
            mx.clear_cache()
            if b == mark:
                cache.keep(ids[:b], kv, "system")
            between()
    if cache is not None and end > start:
        cache.keep(ids[:-1], kv)
    detok = tokenizer.detokenizer
    detok.reset()
    n = 0
    steps = generate_step(mx.array(rest[-1:]), lm, max_tokens=limit, sampler=sampler, prompt_cache=kv)
    for n, (token, _) in enumerate(steps):
        if token in tokenizer.eos_token_ids:
            detok.finalize(); yield detok.last_segment, n, "stop"; return
        detok.add_token(token)
        if n + 1 == limit:
            break
        yield detok.last_segment, n + 1, None
    detok.finalize()
    yield detok.last_segment, n + 1, "length"


def sample(logits, temperature, top_p):
    """One token from a logit vector: greedy at temperature 0, else softmax at the temperature, kept to the top_p mass."""
    import torch
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, -1)
    if 0 < top_p < 1:
        p, order = probs.sort(descending=True)
        p = p * (p.cumsum(0) - p < top_p)   # the smallest set reaching top_p, its last token included
        probs = torch.zeros_like(probs).scatter(0, order, p)
    return int(torch.multinomial(probs / probs.sum(), 1))


def next_logits(lm, hidden):
    """A PyTorch text model's next-token logits from its last hidden state: the tied embeddings (DecisionModel keeps the text
    model, not its generation head) with the final soft-cap, as Gemma 4's generation head computes them. An untied
    base would need a head DecisionModel does not load, so it is refused."""
    import torch
    if not getattr(lm.config, "tie_word_embeddings", False):
        raise ValueError("chat on PyTorch needs a base whose generation head is its tied embeddings (Gemma 4)")
    logits = (hidden @ lm.get_input_embeddings().weight.T).float()
    cap = getattr(lm.config, "final_logit_softcapping", None)
    return torch.tanh(logits / cap) * cap if cap else logits


def pieces_torch(model, tok, prompt, limit, temperature, top_p, eos, between=None):
    """pieces for a PyTorch DecisionModel with its adapter unmerged: the prompt in PREFILL_STEP chunks with between() after
    each (a decision waits one chunk, not the whole prompt, #290), then one token per step on the model's own KV cache.
    -> the same steps as pieces."""
    import torch
    if model.hybrid:
        raise ValueError("chat on PyTorch runs attention-only bases (Gemma 4); a hybrid base chats on MLX")
    lm, between = model.lm, between or (lambda: None)
    device = lm.get_input_embeddings().weight.device
    add_special = tok.bos_token is None or not prompt.startswith(tok.bos_token)
    ids = tok.encode(prompt, add_special_tokens=add_special)
    past, out, text = None, [], ""
    with torch.no_grad():
        for a in range(0, len(ids), PREFILL_STEP):
            res = lm(input_ids=torch.tensor([ids[a:a + PREFILL_STEP]], device=device), past_key_values=past, use_cache=True)
            past = res.past_key_values
            between()
        for n in range(limit):
            token = sample(next_logits(lm, res.last_hidden_state[0, -1]), temperature, top_p)
            if token in eos:
                yield "", n, "stop"; return
            out.append(token)
            decoded = tok.decode(out)
            piece = "" if decoded.endswith("\ufffd") else decoded[len(text):]   # a character still incomplete waits for its next token
            text = text if decoded.endswith("\ufffd") else decoded
            if n + 1 == limit:
                yield piece, n + 1, "length"; return
            yield piece, n + 1, None
            res = lm(input_ids=torch.tensor([[token]], device=device), past_key_values=past, use_cache=True)
            past = res.past_key_values


def generate_chat(model, tok, req, model_dir=None, model_name="d1a", on_text=None, between=None, cache=None, context=None):
    """One chat completion with the adapter off. Run on the model thread (Server.run_task). on_text, if given, receives
    each new piece of the answer as it is generated (streaming), never text past a stop string. between, if given, is
    called after every token and between the prompt's prefill chunks, still inside the adapter switch
    (Server.between_tokens: the decisions waiting meanwhile, #289, #290). cache, a PromptCache, holds adapter-off caches
    only: never pass it one a decision wrote. context: the longest prompt plus completion allowed (context_length), the
    model's own when not given; a longer request raises ContextTooLong before any prefill. -> the OpenAI response body."""
    messages = chat_messages(req)
    prompt = tok.apply_chat_template(messages, tools=req.tools or None, tokenize=False, add_generation_prompt=True)
    prompt_tokens = check_context(tok, req, context or context_length(model_dir), prompt)   # before any prefill
    n_system = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(messages))
    system = tok.apply_chat_template(messages[:n_system], tools=req.tools or None, tokenize=False) if 0 < n_system < len(messages) else None
    system = system if system and prompt.startswith(system) else None   # a template that renders it differently alone: no mark
    limit = completion_limit(req)
    if getattr(model, "backend", None) == "torch":   # PyTorch: no mlx-lm, no prompt cache
        steps = lambda stats: pieces_torch(model, tok, prompt, limit, req.temperature, req.top_p, eos_ids(tok, model_dir), between)
    else:
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        sampler = make_sampler(temp=req.temperature, top_p=req.top_p if req.top_p < 1.0 else 0.0)
        wrapper = TokenizerWrapper(tok, eos_token_ids=eos_ids(tok, model_dir))
        steps = lambda stats: pieces(model.lm, wrapper, prompt, limit, sampler, between, cache, stats, system)
    stops = [s for s in ([req.stop] if isinstance(req.stop, str) else req.stop or []) if s]
    t, text, sent, finish, completion_tokens, stats = time.time(), "", 0, "length", 0, {"cached": 0}
    with model.adapter(False):
        for piece, completion_tokens, ended in steps(stats):
            text += piece
            if ended: finish = ended
            cut = min((text.index(s) for s in stops if s in text), default=None)
            if cut is not None:
                text, finish = text[:cut], "stop"
            if on_text is not None and len(text) > sent:
                on_text(text[sent:]); sent = len(text)
            if cut is not None:
                break
            if between is not None:
                between()
    content, calls = tool_calls_of(text) if req.tools else (text, [])
    # a reply cut off by max_tokens before any text is the client's limit, not an empty answer: logging it as one hid the
    # real ones among smoke tests that cut a tool call short on purpose (#314)
    empty = "" if content.strip() or calls else (", cut off by max_tokens before any text" if finish == "length"
                                                 else f", empty answer after {last_roles(req.messages)}")
    log.info("chat: %d prompt tokens (%d cached), %d completion tokens, %.1f s%s%s", prompt_tokens, stats["cached"], completion_tokens,
             time.time() - t, ", tools" if req.tools else "", empty)
    message = {"role": "assistant", "content": content.strip() or (None if calls else "")}
    if req.tools is not None:
        message["tool_calls"] = calls   # always a list when tools were sent: mem0's graph memory indexes it unconditionally
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(t), "model": req.model or model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens,
                      "prompt_tokens_details": {"cached_tokens": stats["cached"]}}}


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
