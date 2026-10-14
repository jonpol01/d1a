"""OpenAI-compatible chat and embeddings on D1A's server (#274): the model thread's task queue, the chat refusal for a
merged model, token-by-token streaming, Gemma 4 tool calls in OpenAI form, the embeddings endpoint and the OpenAI model
listing. No weights: stand-ins replace the model and the GGUF embedder (the MLX path runs in tests/test_conformance_mlx.py).
    uv run python -m pytest tests/test_openai_compat.py -v
"""
import json
import sys
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
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


def held(model):
    """server_async stand-in: one Server-like object holding `model`, whose run_task runs fn at once."""
    def run_task(fn):
        f = Future(); f.set_result(fn(model, "tok")); return f
    s = SimpleNamespace(model=model, checkpoint=SimpleNamespace(path="/nowhere"), run_task=run_task)

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
    def fake_generate(model, tok, req, model_dir=None, model_name="d1a", on_text=None):
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


def test_the_openai_model_list_rides_beside_the_typesafe_one(monkeypatch):
    monkeypatch.setattr(serve, "EMBED_GGUF", "/stub.gguf")
    serve.app.state.models, serve.app.state.card = SimpleNamespace(model=None, idle_s=0), {"description": "d"}
    body = serve.models()
    ids = [d["id"] for d in body["data"]]
    assert body["object"] == "list" and body["models"][0]["name"] in ids and serve.EMBED_NAME in ids
    assert all(m["description"] == "d" for m in body["models"])   # the TypeSafe listing the playground reads is unchanged
