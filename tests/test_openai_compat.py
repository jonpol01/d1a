"""OpenAI-compatible chat and embeddings on D1A's server (#274): the model thread's task queue, the chat refusal for a
merged model, token-by-token streaming, Gemma 4 tool calls in OpenAI form, the embeddings endpoint and the OpenAI model
listing. No weights: stand-ins replace the model and the GGUF embedder (the MLX path runs in tests/test_conformance_mlx.py).
    uv run python -m pytest tests/test_openai_compat.py -v
"""
import json
import sys
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from d1a.backends.checkpoint import mlx_available
from d1a.serving import openai_compat, serve
from d1a.serving.serve import Server
from test_serving import StandInModel, packed

needs_mlx = pytest.mark.skipif(not mlx_available(), reason="mlx-lm is not installed")


def test_a_task_runs_alone_on_the_model_thread_between_decisions():
    """Chat generation must run on the model thread (MLX arrays belong to the thread that made them), never inside a
    decision batch, and leave decisions answering as before."""
    server = Server(SimpleNamespace(release_date=lambda: "2026-01-01"), "tok", StandInModel(), "cpu")
    try:
        before = server.probs(packed("abc", branch=1))
        seen = server.run_task(lambda model, tok: (threading.current_thread().name, tok)).result(timeout=10)
        after = server.probs(packed("abc", branch=1))
    finally:
        server.close()
    assert seen == ("d1a-model", "tok") and before[0] == after[0]


class SwitchedModel(StandInModel):
    """StandInModel with MLXDecisionModel's adapter switch; each pass records whether the adapter was on."""
    def __init__(self):
        super().__init__()
        self.on, self.passes_on = True, []

    @contextmanager
    def adapter(self, on=True):
        was, self.on = self.on, on
        try: yield
        finally: self.on = was

    def probs_batch(self, encs, cached, keep):
        self.passes_on.append(self.on)
        return super().probs_batch(encs, cached, keep)


def test_a_decision_during_a_chat_is_answered_between_its_tokens_with_the_adapter_on():
    """#289: a decision that arrives while a chat generates is answered at the chat's next token, with the adapter on,
    not after the whole answer; the chat then goes on with the adapter off."""
    model = SwitchedModel()
    server = Server(SimpleNamespace(release_date=lambda: "2026-01-01"), "tok", model, "cpu")
    started, decision = threading.Event(), {}

    def chat(model, tok):
        with model.adapter(False):
            started.set()
            for _ in range(200):   # a long answer: decision["future"] would wait for all of it without between_tokens
                if "future" in decision: break
                threading.Event().wait(0.01)
            server.between_tokens()
            return decision["future"].done(), model.on
    try:
        task = server.run_task(chat)
        assert started.wait(10)
        decision["future"] = server.submit(packed("abc", branch=1))
        answered_mid_chat, adapter_on_after = task.result(timeout=10)
        probs, _ = decision["future"].result(timeout=10)
    finally:
        server.close()
    assert answered_mid_chat and not adapter_on_after and model.passes_on == [True] and probs[0] == [0.25, 0.75]


def test_a_decision_taken_between_tokens_gets_the_error_when_the_adapter_switch_fails():
    """between_tokens takes the waiting decisions off the queue before switching the adapter on: if the switch raises,
    they must fail with its error, not wait forever, and the chat goes on."""
    class Broken(SwitchedModel):
        @contextmanager
        def adapter(self, on=True):
            if on: raise RuntimeError("switch failed")
            yield
    server = Server(SimpleNamespace(release_date=lambda: "2026-01-01"), "tok", Broken(), "cpu")
    try:
        pending = Future()
        def chat(model, tok):
            pending.set_result(server.submit(packed("abc", branch=1)))
            server.between_tokens()
            return "chat done"
        assert server.run_task(chat).result(timeout=10) == "chat done"
        with pytest.raises(RuntimeError, match="switch failed"):
            pending.result().result(timeout=10)
    finally:
        server.close()


def held(model):
    """server_async stand-in: one Server-like object holding `model`, whose run_task runs fn at once."""
    def run_task(fn):
        f = Future(); f.set_result(fn(model, "tok")); return f
    s = SimpleNamespace(model=model, checkpoint=SimpleNamespace(path="/nowhere"), run_task=run_task, between_tokens=lambda: None,
                        chat_prompts="the server's prompt cache")

    @asynccontextmanager
    async def server_async():
        yield s
    return server_async


def test_chat_on_a_merged_model_is_refused_with_501(monkeypatch):
    monkeypatch.setattr(serve, "server_async", held(SimpleNamespace(loras=[])))
    with TestClient(serve.app) as client:
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 501 and "--unmerged" in r.json()["detail"]


def test_a_stream_sends_each_piece_as_it_comes_then_the_finish_and_done(monkeypatch):
    def fake_generate(model, tok, req, model_dir=None, model_name="d1a", on_text=None, between=None, cache=None):
        assert cache == "the server's prompt cache"   # every chat of the server shares one, so a next step finds the last one's
        for piece in ("Hel", "lo"): on_text(piece)
        return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}]}
    monkeypatch.setattr(serve, "server_async", held(SimpleNamespace(loras=["one"])))
    monkeypatch.setattr(openai_compat, "generate_chat", fake_generate)
    with TestClient(serve.app) as client:
        r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    events = [line.removeprefix("data: ") for line in r.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(e) for e in events[:-1]]
    assert r.headers["content-type"].startswith("text/event-stream") and events[-1] == "[DONE]"
    assert [c["choices"][0]["delta"] for c in chunks] == [{"role": "assistant", "content": ""}, {"content": "Hel"}, {"content": "lo"}, {}]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and all(c["object"] == "chat.completion.chunk" for c in chunks)


@needs_mlx
def test_gemma_tool_calls_become_openai_tool_calls():
    text = 'Let me check.<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|>'
    head, calls = openai_compat.tool_calls_of(text)
    assert head == "Let me check." and len(calls) == 1
    assert calls[0]["type"] == "function" and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert openai_compat.tool_calls_of("No tools needed.") == ("No tools needed.", [])


@needs_mlx
def test_a_tool_call_cut_off_by_max_tokens_leaves_no_markup_in_the_text():
    """#278: an answer that runs out of tokens inside a tool call came back as content holding the raw, half-written Gemma
    block. The client gets the text before it and no tool call (finish_reason is already "length"), so it can retry."""
    cut = 'Saving it now.<|tool_call>call:write_file{path:<|"|>/tmp/a.md<|"|>,content:<|"|>WHERE TO POINT'
    assert openai_compat.tool_calls_of(cut) == ("Saving it now.", [])
    head, calls = openai_compat.tool_calls_of('<|tool_call>call:get_weather{city:<|"|>Tokyo<|"|>}<tool_call|><|tool_call>call:get_wea')
    assert head == "" and [c["function"]["name"] for c in calls] == ["get_weather"]   # the complete call survives, the cut one goes


class StubLlama:
    def __init__(self): self.normalize = []

    def embed(self, text, normalize=False):
        self.normalize.append(normalize); return [0.6, 0.8]

    def tokenize(self, data): return list(data)


def test_embeddings_are_off_without_a_gguf_and_unit_length_with_one(monkeypatch):
    monkeypatch.setattr(serve, "EMBED_GGUF", None)
    with TestClient(serve.app) as client:
        assert client.post("/v1/embeddings", json={"input": "hi"}).status_code == 501
        monkeypatch.setattr(serve, "EMBED_GGUF", "/stub.gguf")
        embedder = openai_compat.Embedder("/stub.gguf", "text-embedding-nomic-embed-text-v1.5"); embedder.llm = StubLlama()
        serve.app.state.embedder = embedder
        try:
            r = client.post("/v1/embeddings", json={"input": ["a b", "c"], "model": "x", "dimensions": 768}).json()
        finally:
            serve.app.state.embedder = None
    assert r["object"] == "list" and [d["index"] for d in r["data"]] == [0, 1] and r["data"][0]["embedding"] == [0.6, 0.8]
    assert embedder.llm.normalize == [True, True]   # LM Studio served unit-length vectors; stored ones are compared by distance


def test_the_embedder_runs_every_layer_on_the_gpu(monkeypatch):
    """llama.cpp on the CPU (llama-cpp-python's default, n_gpu_layers=0) gives vectors ~0.99 cosine to the ones LM Studio
    stored with Metal; all layers offloaded gives the same vectors. Measured on the Mac mini (#274)."""
    made = {}
    class Llama(StubLlama):
        def __init__(self, **kw): super().__init__(); made.update(kw)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Llama))
    assert openai_compat.Embedder("/stub.gguf", "e").embed(["x"]) == [[0.6, 0.8]]
    assert made["n_gpu_layers"] == -1 and made["embedding"] is True


def test_embeddings_asked_for_in_base64_come_as_little_endian_float32(monkeypatch):
    """#282: the openai Node SDK (n8n's embeddings node) always sends encoding_format "base64" and decodes each embedding as
    base64 float32; a float list came out as 192 numbers instead of 768. Float, or no format, stays a list."""
    import base64, struct
    monkeypatch.setattr(serve, "EMBED_GGUF", "/stub.gguf")
    embedder = openai_compat.Embedder("/stub.gguf", "e"); embedder.llm = StubLlama()
    serve.app.state.embedder = embedder
    try:
        with TestClient(serve.app) as client:
            b64 = client.post("/v1/embeddings", json={"input": ["a", "b"], "encoding_format": "base64"}).json()["data"]
            flt = client.post("/v1/embeddings", json={"input": "a", "encoding_format": "float"}).json()["data"]
            none = client.post("/v1/embeddings", json={"input": "a"}).json()["data"]
    finally:
        serve.app.state.embedder = None
    decoded = [list(struct.unpack("<2f", base64.b64decode(d["embedding"]))) for d in b64]
    assert decoded == [pytest.approx([0.6, 0.8])] * 2 and flt[0]["embedding"] == none[0]["embedding"] == [0.6, 0.8]


def test_chat_stays_listed_while_the_model_is_unloaded(monkeypatch):
    """#280: with --idle-unload an idle server holds no model, and d1a-chat dropped out of /v1/models, so a client that picks
    its model from the list (n8n) could not find it. A chat-capable checkpoint lists it loaded or not; a merged one never."""
    monkeypatch.setattr(serve, "EMBED_GGUF", None)
    monkeypatch.setattr(serve.app.state, "models", SimpleNamespace(model=None, idle_s=600), raising=False)
    monkeypatch.setattr(serve.app.state, "card", {"description": "d"}, raising=False)
    for can_chat, listed in ((True, True), (False, False)):
        monkeypatch.setattr(serve.app.state, "chat", can_chat, raising=False)
        assert (serve.CHAT_NAME in [d["id"] for d in serve.models()["data"]]) is listed


def test_the_openai_model_list_rides_beside_the_typesafe_one(monkeypatch):
    monkeypatch.setattr(serve, "EMBED_GGUF", "/stub.gguf")
    serve.app.state.models, serve.app.state.card = SimpleNamespace(model=None, idle_s=0), {"description": "d"}
    body = serve.models()
    ids = [d["id"] for d in body["data"]]
    assert body["object"] == "list" and body["models"][0]["name"] in ids and serve.EMBED_NAME in ids
    assert all(m["description"] == "d" for m in body["models"])   # the TypeSafe listing the playground reads is unchanged
