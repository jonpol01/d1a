"""Chat routes (d1a.serving.routes, #419): with D1A_ROUTES set, POST /v1/chat/completions sends a pull-request review to a
review service and a hard chat to a larger model, and leaves the rest to D1A. No model: the passes are stand-ins, or a real
Server on a stand-in model, and the targets are local HTTP services on this machine."""
import json
import logging
import socket
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from d1a.serving import openai_compat, routes, serve
from test_serving import StandInModel, packed

KEY, D1A_TEXT, REVIEW_TEXT = "target-key-5f2c", "D1A's own answer", "Looks good: one nit in serve.py."
LINK = "https://github.com/jonpol01/d1a/pull/92"


class Target:
    """A local HTTP service standing in for the review service or the large model. `reply(handler, body)` -> (status,
    content type, bytes) or None after writing the response itself; every request is kept in `requests`."""

    def __init__(self, reply):
        self.reply, self.requests, self.got = reply, [], threading.Event()
        target = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                target.requests.append({"path": self.path, "headers": dict(self.headers), "body": body}); target.got.set()
                out = target.reply(self, body)
                if out is None: return
                status, kind, data = out
                self.send_response(status); self.send_header("content-type", kind); self.send_header("content-length", str(len(data))); self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args): pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown(); self.server.server_close()


def json_reply(status, body):
    return lambda handler, request: (status, "application/json", json.dumps(body).encode())


def closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


@pytest.fixture
def targets():
    made = []
    yield lambda reply: made.append(Target(reply)) or made[-1]
    for t in made: t.close()


def configure(tmp_path, monkeypatch, review_url=None, large_url=None, timeout_s=5, p=0.0, tiers=None, repos=("jonpol01/*",)):
    """Routes from a real D1A_ROUTES file, stand-in passes (P(yes) = p, tier by text from `tiers`, default small) and a
    stand-in D1A chat. -> the calls the passes and D1A's chat got."""
    key_file = tmp_path / "key"; key_file.write_text(KEY + "\n", encoding="utf-8")
    spec = []
    if review_url: spec.append({"name": "review", "target": {"kind": "review", "url": review_url, "key_file": str(key_file), "timeout_s": timeout_s}, "repos": list(repos)})
    if large_url: spec.append({"name": "large", "target": {"kind": "openai", "url": large_url, "model": "big-model", "key_file": str(key_file), "timeout_s": timeout_s}})
    (tmp_path / "routes.json").write_text(json.dumps({"routes": spec}), encoding="utf-8")
    monkeypatch.setattr(serve.app.state, "routes", serve.load_routes(str(tmp_path / "routes.json"), "server-key"))
    calls = {"p": [], "tier": [], "chat": []}

    async def p_review(text): calls["p"].append(text); return p
    async def tier(text): calls["tier"].append(text); return (tiers or {}).get(text, "small")

    def generate(model, tok, req, model_dir=None, model_name="d1a", on_text=None, between=None, cache=None, context=None):
        calls["chat"].append(req)
        if on_text: on_text(D1A_TEXT)
        return {"id": "chatcmpl-d1a", "object": "chat.completion", "created": 0, "model": req.model or model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": D1A_TEXT}, "finish_reason": "stop"}], "usage": {}}

    monkeypatch.setattr(serve, "is_review_p", p_review)
    monkeypatch.setattr(serve, "route_tier", tier)
    monkeypatch.setattr(openai_compat, "generate_chat", generate)
    monkeypatch.setattr(serve, "server_async", held())
    return calls


def held():
    """server_async stand-in for D1A's chat: a model that can chat, whose run_task runs fn at once."""
    from concurrent.futures import Future
    def run_task(fn):
        f = Future(); f.set_result(fn(None, None)); return f
    s = SimpleNamespace(model=SimpleNamespace(loras=["one"]), checkpoint=SimpleNamespace(path="/nowhere"), run_task=run_task,
                        between_tokens=lambda: None, chat_prompts=None, tok=None)

    @asynccontextmanager
    async def server_async():
        yield s
    return server_async


def chat(client, text, stream=False, **extra):
    body = {"model": "d1a-chat", "messages": [{"role": "user", "content": text}], "stream": stream} | extra
    return client.post("/v1/chat/completions", json=body)


def content(r):
    return r.json()["choices"][0]["message"]["content"]


def streamed(r):
    """(the text of a streamed answer, its SSE comment lines)."""
    data = [line.removeprefix("data: ") for line in r.text.splitlines() if line.startswith("data: ")]
    assert data[-1] == "[DONE]"
    return "".join(json.loads(d)["choices"][0]["delta"].get("content") or "" for d in data[:-1]), [line for line in r.text.splitlines() if line.startswith(":")]


# --- off, and what is never routed ----------------------------------------------------------------------------------------

def test_off_every_chat_reaches_d1a_untouched(monkeypatch, tmp_path):
    calls = configure(tmp_path, monkeypatch, review_url=f"http://127.0.0.1:{closed_port()}", p=0.99)
    monkeypatch.setattr(serve.app.state, "routes", None)
    with TestClient(serve.app) as client:
        r = chat(client, f"review {LINK}")
    assert r.status_code == 200 and content(r) == D1A_TEXT and "x-d1a-route" not in r.headers
    assert calls["p"] == calls["tier"] == [] and len(calls["chat"]) == 1


def test_a_turn_that_does_not_end_with_the_user_is_not_routed(monkeypatch, tmp_path, targets):
    """Mid tool loop (the last message is a tool result) the chat stays with D1A: no pass runs, nothing is forwarded."""
    large = targets(json_reply(200, {"choices": []}))
    calls = configure(tmp_path, monkeypatch, large_url=large.url, tiers={"debug the scheduler": "large"})
    messages = [{"role": "user", "content": "debug the scheduler"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "file text"}]
    with TestClient(serve.app) as client:
        r = client.post("/v1/chat/completions", json={"messages": messages, "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}]})
    assert r.status_code == 200 and r.headers["x-d1a-route"] == "none" and calls["tier"] == [] and large.requests == []


def test_small_and_medium_chats_stay_with_d1a(monkeypatch, tmp_path, targets):
    large = targets(json_reply(200, {"choices": []}))
    calls = configure(tmp_path, monkeypatch, large_url=large.url, tiers={"hi": "small", "summarise this": "medium"})
    with TestClient(serve.app) as client:
        for text in ("hi", "summarise this"):
            r = chat(client, text)
            assert content(r) == D1A_TEXT and r.headers["x-d1a-route"] == "none"
    assert large.requests == [] and len(calls["chat"]) == 2


def test_a_routed_chat_needs_the_api_key_like_any_other(monkeypatch, tmp_path, targets):
    review = targets(json_reply(200, {"text": REVIEW_TEXT}))
    calls = configure(tmp_path, monkeypatch, review_url=review.url, p=0.99)
    monkeypatch.setattr(serve, "API_KEY", "server-key")
    with TestClient(serve.app) as client:
        r = chat(client, f"review {LINK}")
        ok = client.post("/v1/chat/completions", headers={"authorization": "Bearer server-key"},
                         json={"messages": [{"role": "user", "content": f"review {LINK}"}]})
    assert r.status_code == 401 and calls["p"] == [f"review {LINK}"] * 1 and content(ok) == REVIEW_TEXT and len(review.requests) == 1


# --- review ---------------------------------------------------------------------------------------------------------------

def test_a_review_request_is_forwarded_and_its_text_returned_verbatim(monkeypatch, tmp_path, targets, caplog):
    """The pull request goes as owner/repo#n with the configured key, never the client's; the reply is the service's text.
    A proxy set in the environment is not used (the key goes to the configured URL only), and no log line holds the key or
    the message."""
    review = targets(json_reply(200, {"text": REVIEW_TEXT, "verdict": "approve"}))
    calls = configure(tmp_path, monkeypatch, review_url=review.url, p=0.6)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{closed_port()}"); monkeypatch.setenv("ALL_PROXY", f"http://127.0.0.1:{closed_port()}")
    message = f"このPRをレビューして {LINK}をお願い"
    with caplog.at_level(logging.INFO, logger="d1a.chat"), TestClient(serve.app) as client:
        r = client.post("/v1/chat/completions", headers={"authorization": "Bearer client-key"}, json={"messages": [{"role": "user", "content": message}]})
    assert r.status_code == 200 and content(r) == REVIEW_TEXT and r.json()["model"] == serve.CHAT_NAME
    assert r.headers["x-d1a-route"] == "review" and "x-d1a-route-error" not in r.headers and calls["chat"] == []
    (sent,) = review.requests
    assert sent["body"] == {"mode": "pr", "pr": "jonpol01/d1a#92"} and sent["headers"]["authorization"] == f"Bearer {KEY}"
    assert KEY not in caplog.text and "レビュー" not in caplog.text and "route review" in caplog.text


def test_review_routing_follows_the_link_and_the_two_thresholds(monkeypatch, tmp_path, targets):
    """With a link, P(yes) >= 0.5 routes; without one, is_review_min_p (0.8). A message that only mentions a pull request
    is D1A's. Routed without exactly one pull request named, D1A asks which instead of answering."""
    review = targets(json_reply(200, {"text": REVIEW_TEXT}))
    cases = [(0.4, f"what changed in {LINK}?", D1A_TEXT, None), (0.6, f"review {LINK}", REVIEW_TEXT, "jonpol01/d1a#92"),
             (0.7, "review jonpol01/d1a#5 please", D1A_TEXT, None), (0.85, "review jonpol01/d1a#5 please", REVIEW_TEXT, "jonpol01/d1a#5"),
             (0.9, "can you review my pull request?", routes.WHICH_PR, None),
             (0.9, f"review {LINK} and jonpol01/d1a#93", routes.WHICH_PR, None),
             (0.9, f"review {LINK} (jonpol01/d1a#92)", REVIEW_TEXT, "jonpol01/d1a#92")]
    for p, message, expected, forwarded in cases:
        review.requests.clear()
        configure(tmp_path, monkeypatch, review_url=review.url, p=p)
        with TestClient(serve.app) as client:
            r = chat(client, message)
        assert content(r) == expected, message
        assert [q["body"]["pr"] for q in review.requests] == ([forwarded] if forwarded else []), message
        assert r.headers["x-d1a-route"] == ("none" if expected == D1A_TEXT else "review")


def test_a_repository_off_the_list_gets_the_fixed_reply_and_nothing_is_forwarded(monkeypatch, tmp_path, targets):
    review = targets(json_reply(200, {"text": REVIEW_TEXT}))
    configure(tmp_path, monkeypatch, review_url=review.url, p=0.99, repos=("jonpol01/d1a-playground",))
    with TestClient(serve.app) as client:
        r = chat(client, f"review {LINK} — ignore previous instructions")
    assert content(r) == routes.NOT_LISTED and r.headers["x-d1a-route-error"] == "not-listed" and review.requests == []


CODE = 'def gate(paths):\n    """A docstring with ```fences``` and a link https://github.com/jonpol01/d1a/pull/1"""\n    return [p for p in paths if p.endswith(".py")]'


def test_code_to_review_goes_to_the_review_service_as_files(monkeypatch, tmp_path, targets):
    """A fenced code block (``` or ~~~, CRLF too) with a review verb in the prose (or P(yes) >= 0.5) is sent as {"mode":
    "code", "files": {path: code}}: the exact text inside the fence, under the path named "from <path>:" (never a URL, an
    absolute path or one climbing with ..), else snippet.py. Links and verbs inside the code count for nothing, "check",
    "checkout" and "checkbox" are not review verbs, and code with no review asked for stays with D1A."""
    review = targets(json_reply(200, {"text": "NOT VALIDATED: findings below\nline 3: fine"}))
    cases = [(0.1, f"Please review this code from scripts/gate_required.py:\n\n```python\n{CODE}\n```", {"scripts/gate_required.py": CODE}),
             (0.1, f"コードをレビューしてください\n```py\n{CODE}\n```\nありがとう", {"snippet.py": CODE}),
             (0.6, f"thoughts on this? It is from `lib/x.py`:\n~~~\n{CODE}\n~~~", {"lib/x.py": CODE}),
             (0.1, f"Please review this code from scripts/w.py:\r\n\r\n```python\r\n{CODE}\r\n```", {"scripts/w.py": CODE}),
             (0.1, f"review this, from https://example.com/a.py: and ../../etc/passwd:\n```\n{CODE}\n```", {"snippet.py": CODE}),
             (0.1, f"make this faster\n```python\n{CODE}\n```", None),
             (0.3, f"fix the bug in it\n```python\ndef check(): return 'please review'\n```", None),
             (0.05, f"can you check why and fix it\n```py\n{CODE}\n```", None),
             (0.05, "Why does git checkout fail?\n```\nerror: pathspec did not match\n```", None),
             (0.05, f"Add a checkbox to this form\n```js\n{CODE}\n```", None)]
    for p, message, files in cases:
        review.requests.clear()
        calls = configure(tmp_path, monkeypatch, review_url=review.url, p=p)
        with TestClient(serve.app) as client:
            r = chat(client, message)
        assert [q["body"] for q in review.requests] == ([{"mode": "code", "files": files}] if files else []), message
        assert content(r) == ("NOT VALIDATED: findings below\nline 3: fine" if files else D1A_TEXT) and len(calls["chat"]) == (0 if files else 1)


@pytest.mark.parametrize("reply, expected, error", [
    (None, routes.UNAVAILABLE, "connect"),
    (lambda h, b: time.sleep(1.5) or (200, "application/json", b'{"text": "late"}'), routes.UNAVAILABLE, "timeout"),
    (json_reply(500, {"detail": "boom"}), routes.UNAVAILABLE, "status 500"),
    (json_reply(401, {"detail": "missing key"}), routes.UNAVAILABLE, "status 401"),
    (json_reply(200, {"text": "  "}), routes.UNAVAILABLE, "empty"),
    (lambda h, b: (200, "application/json", json.dumps({"text": "x" * (routes.REVIEW_MAX_BYTES + 10)}).encode()), routes.UNAVAILABLE, "too-large"),
    (lambda h, b: (200, "text/html", b"<html>"), routes.UNAVAILABLE, "not-json"),
    (lambda h, b: (h.send_response(302), h.send_header("location", "http://example.invalid/"), h.send_header("content-length", "0"), h.end_headers()) and None, routes.UNAVAILABLE, "status 302"),
    (json_reply(422, {"error": "no review", "text": "Not fetchable — no review was written."}), "Not fetchable — no review was written.", "status 422"),
    (json_reply(503, {"error": "reviewer unavailable", "text": "reviewer unavailable — no review was written (queue full)."}), "reviewer unavailable — no review was written (queue full).", "status 503"),
])
def test_a_failed_review_never_comes_back_as_model_text(monkeypatch, tmp_path, targets, reply, expected, error):
    """A review that cannot be had gets the fixed reply, or the text the service's reply carries (its own "no review was
    written" on a 422 or 503), and the reason in x-d1a-route-error. D1A's chat is never asked, and a redirect is never
    followed."""
    url = targets(reply).url if reply else f"http://127.0.0.1:{closed_port()}"
    calls = configure(tmp_path, monkeypatch, review_url=url, p=0.99, timeout_s=0.5)
    with TestClient(serve.app) as client:
        r = chat(client, f"review {LINK}")
    assert r.status_code == 200 and content(r) == expected and r.headers["x-d1a-route-error"] == error and calls["chat"] == []


def test_a_second_review_while_one_runs_is_told_the_reviewer_is_busy(monkeypatch, tmp_path, targets):
    """One review at a time; the slot is free again once it ends, so the next one goes through."""
    release = threading.Event()
    review = targets(lambda h, b: release.wait(5) and (200, "application/json", json.dumps({"text": REVIEW_TEXT}).encode()))
    configure(tmp_path, monkeypatch, review_url=review.url, p=0.99)
    with TestClient(serve.app) as client:
        first = []
        t = threading.Thread(target=lambda: first.append(chat(client, f"review {LINK}"))); t.start()
        assert review.got.wait(5)
        busy = chat(client, f"review {LINK}")
        release.set(); t.join(5)
        again = chat(client, f"review {LINK}")
    assert content(busy) == routes.BUSY and busy.headers["x-d1a-route-error"] == "busy"
    assert content(first[0]) == content(again) == REVIEW_TEXT and len(review.requests) == 2


def test_a_streamed_review_sends_the_role_chunk_keep_alives_then_the_text(monkeypatch, tmp_path, targets):
    review = targets(lambda h, b: time.sleep(0.4) or (200, "application/json", json.dumps({"text": REVIEW_TEXT}).encode()))
    configure(tmp_path, monkeypatch, review_url=review.url, p=0.99)
    monkeypatch.setattr(routes, "KEEP_ALIVE_S", 0.05)
    with TestClient(serve.app) as client:
        r = chat(client, f"review {LINK}", stream=True)
    text, comments = streamed(r)
    first = json.loads(r.text.split("\n", 1)[0].removeprefix("data: "))
    assert r.headers["x-d1a-route"] == "review" and r.headers["content-type"].startswith("text/event-stream")
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""} and text == REVIEW_TEXT and comments


# --- large ----------------------------------------------------------------------------------------------------------------

AGENT_REQUEST = {"model": "d1a-chat", "max_tokens": 100_000, "stream_options": {"include_usage": True}, "parallel_tool_calls": False, "seed": 7,
                 "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
                 "messages": [{"role": "system", "content": "You are an agent."},
                              {"role": "user", "content": "Find why the scheduler deadlocks under load"},
                              {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"}}]},
                              {"role": "tool", "tool_call_id": "c1", "content": "def run(): ..."},
                              {"role": "user", "content": "continue"}]}
LARGE_REPLY = {"id": "chatcmpl-big", "object": "chat.completion", "created": 1, "model": "big-model",
               "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"b.py\"}"}}]}, "finish_reason": "tool_calls"}]}


def test_a_hard_chat_goes_whole_to_the_large_model_and_its_answer_comes_back_verbatim(monkeypatch, tmp_path, targets):
    """The body as the client sent it (tools, stream options, tool-call arguments as strings, a max_tokens past D1A's own
    limit), with only `model` replaced, and the configured key. The tier is sticky upward: the first user message is hard,
    so "continue" stays with the large model."""
    large = targets(json_reply(200, LARGE_REPLY))
    calls = configure(tmp_path, monkeypatch, large_url=large.url, tiers={"Find why the scheduler deadlocks under load": "large", "continue": "small"})
    with TestClient(serve.app) as client:
        r = client.post("/v1/chat/completions", headers={"authorization": "Bearer client-key"}, json=AGENT_REQUEST)
    assert r.status_code == 200 and r.json() == LARGE_REPLY and r.headers["x-d1a-route"] == "large" and calls["chat"] == []
    (sent,) = large.requests
    assert sent["body"] == AGENT_REQUEST | {"model": "big-model"} and sent["headers"]["authorization"] == f"Bearer {KEY}"
    assert sorted(calls["tier"]) == ["Find why the scheduler deadlocks under load", "continue"]


def test_a_streamed_hard_chat_passes_the_large_models_stream_through(monkeypatch, tmp_path, targets):
    events = b'data: {"choices":[{"delta":{"role":"assistant","content":"Proof:"}}]}\n\ndata: {"choices":[{"delta":{"content":" by induction"}}]}\n\ndata: [DONE]\n\n'
    def reply(h, b):   # sent in two writes, no content-length: the body ends when the target closes
        h.send_response(200); h.send_header("content-type", "text/event-stream"); h.end_headers()
        h.wfile.write(events[:40]); h.wfile.flush(); time.sleep(0.2); h.wfile.write(events[40:])
    large = targets(reply)
    configure(tmp_path, monkeypatch, large_url=large.url, tiers={"prove it": "large"})
    with TestClient(serve.app) as client:
        r = chat(client, "prove it", stream=True)
    assert r.content == events and r.headers["x-d1a-route"] == "large" and large.requests[0]["body"]["stream"] is True


def test_a_compressed_stream_from_the_large_model_reaches_the_client_as_text(monkeypatch, tmp_path, targets):
    """No compression is asked for, and a target that compresses anyway is decoded: content-encoding is not passed on."""
    import gzip
    events = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    def reply(h, b):
        data = gzip.compress(events)
        h.send_response(200); h.send_header("content-type", "text/event-stream"); h.send_header("content-encoding", "gzip")
        h.send_header("content-length", str(len(data))); h.end_headers(); h.wfile.write(data)
    large = targets(reply)
    configure(tmp_path, monkeypatch, large_url=large.url, tiers={"prove it": "large"})
    with TestClient(serve.app) as client:
        r = chat(client, "prove it", stream=True)
    assert r.content == events and "content-encoding" not in r.headers and large.requests[0]["headers"]["accept-encoding"] == "identity"


def test_a_4xx_from_the_large_model_comes_back_as_it_sent_it(monkeypatch, tmp_path, targets):
    large = targets(json_reply(400, {"error": {"message": "context too long"}}))
    calls = configure(tmp_path, monkeypatch, large_url=large.url, tiers={"prove it": "large"})
    with TestClient(serve.app) as client:
        r = chat(client, "prove it")
    assert r.status_code == 400 and r.json() == {"error": {"message": "context too long"}} and r.headers["x-d1a-route-error"] == "status 400" and calls["chat"] == []


REDIRECT = lambda h, b: (h.send_response(302), h.send_header("location", "http://example.invalid/"), h.send_header("content-length", "0"), h.end_headers()) and None


@pytest.mark.parametrize("reply, error", [(None, "connect"), (json_reply(503, {"error": "overloaded"}), "status 503"), (REDIRECT, "status 302"),
                                          (lambda h, b: time.sleep(1.5) or (200, "application/json", b"{}"), "timeout")])
@pytest.mark.parametrize("stream", [False, True])
def test_large_model_down_d1a_answers_with_the_note(monkeypatch, tmp_path, targets, reply, error, stream):
    url = targets(reply).url if reply else f"http://127.0.0.1:{closed_port()}"
    calls = configure(tmp_path, monkeypatch, large_url=url, tiers={"prove it": "large"}, timeout_s=0.5)
    with TestClient(serve.app) as client:
        r = chat(client, "prove it", stream=stream)
    text = streamed(r)[0] if stream else content(r)
    assert r.status_code == 200 and text == f"{routes.NOTE_LARGE_DOWN}\n{D1A_TEXT}" and len(calls["chat"]) == 1
    assert r.headers["x-d1a-route"] == "large" and r.headers["x-d1a-route-error"] == error


def test_a_failed_pass_leaves_the_chat_to_d1a(monkeypatch, tmp_path, targets):
    review = targets(json_reply(200, {"text": REVIEW_TEXT}))
    calls = configure(tmp_path, monkeypatch, review_url=review.url)
    async def broken(text): raise RuntimeError("the server is stopping")
    monkeypatch.setattr(serve, "is_review_p", broken)
    with TestClient(serve.app) as client:
        r = chat(client, f"review {LINK}")
    assert content(r) == D1A_TEXT and r.headers["x-d1a-route"] == "none" and r.headers["x-d1a-route-error"] == "error"
    assert review.requests == [] and len(calls["chat"]) == 1


def test_a_streamed_review_whose_client_leaves_stops_the_review():
    """The stream is closed when its client goes (StreamingResponse cancels it): the review is cancelled with it, so its
    slot is free at once rather than after timeout_s."""
    async def run():
        task = routes.asyncio.ensure_future(routes.asyncio.sleep(30))
        stream = routes._review_stream(task, {}, "d1a-chat")
        first = await anext(stream)
        await stream.aclose()
        await routes.asyncio.sleep(0)
        return first, task.cancelled()
    first, cancelled = routes.asyncio.run(run())
    assert '"role": "assistant"' in first and cancelled


# --- the passes on a server -----------------------------------------------------------------------------------------------

class PassModel(StandInModel):
    """A stand-in model for the routing passes: P(yes) for a yes/no question, `route` for a three-option one. Each encoding
    is recorded with whether the prefix cache could keep it."""

    def __init__(self, p_yes, route):
        super().__init__(); self.p_yes, self.route, self.seen = p_yes, route, []

    def encode(self, tok, rec, **limits):
        return packed(tuple(rec["state"])) | {"options": [len(q["options"]) for q in rec["questions"]]}

    def probs_batch(self, encs, cached, keep):
        self.seen += [(enc.get("uncached", False), k) for enc, k in zip(encs, keep)]
        dist = {2: [1 - self.p_yes, self.p_yes], 3: self.route}
        return [[torch.tensor(dist[n]) for n in enc["options"]] for enc in encs], [None] * len(encs)


def real_server(monkeypatch, model):
    server = serve.Server(SimpleNamespace(release_date=lambda: "2026-01-01", requested="stand-in"),
                          lambda text, add_special_tokens=False: SimpleNamespace(input_ids=text.split()), model, "cpu")
    held_now = []

    @asynccontextmanager
    async def server_async():
        held_now.append(1)
        try: yield server
        finally: held_now.pop()
    monkeypatch.setattr(serve, "server_async", server_async)
    real_server.last = server_async
    return server, held_now


def test_the_passes_run_on_the_model_unlogged_and_the_review_pass_uncached(monkeypatch, tmp_path):
    """The review question is read raw and never kept in the prefix cache; neither pass is written to the decision log; the
    route preset's tier follows its fail-up advice (a sure "small" stays small, an unsure answer goes up)."""
    log = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(serve, "LEARNING", serve.Learning(str(log)))
    server, _ = real_server(monkeypatch, PassModel(0.83, [0.9, 0.05, 0.05]))
    monkeypatch.setattr(serve.app.state, "card", {"temperature": 1.0, "use_case_temperatures": {}}, raising=False)
    try:
        assert serve.asyncio.run(serve.is_review_p("review my PR")) == 0.83
        assert serve.asyncio.run(serve.route_tier("hello")) == "small"
        server.model.route = [0.3, 0.1, 0.6]
        assert serve.asyncio.run(serve.route_tier("prove the theorem")) == "large"
    finally:
        server.close()
    assert server.model.seen[0] == (True, False) and all(not uncached for uncached, _ in server.model.seen[1:])
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


def test_the_model_thread_and_server_are_free_while_a_review_runs(monkeypatch, tmp_path, targets):
    """The review pass runs on the model; the forward runs after it let go of the server: a decision submitted while the
    review service is working is answered, and the server is not held."""
    is_review_p = serve.is_review_p
    server, held_now = real_server(monkeypatch, PassModel(0.95, [0.1, 0.1, 0.8]))
    during = []
    review = targets(lambda h, b: during.append((server.probs({"state": "ab", "questions": [{"options": ["no", "yes"]}]}), list(held_now))) or (200, "application/json", json.dumps({"text": REVIEW_TEXT}).encode()))
    configure(tmp_path, monkeypatch, review_url=review.url)
    monkeypatch.setattr(serve, "is_review_p", is_review_p)
    monkeypatch.setattr(serve, "server_async", real_server.last)
    try:
        with TestClient(serve.app) as client:
            r = chat(client, f"review {LINK}")
    finally:
        server.close()
    assert content(r) == REVIEW_TEXT and during and during[0][1] == [] and server.model.seen[0] == (True, False)


# --- the routes file and startup ------------------------------------------------------------------------------------------

def test_startup_refuses_routes_without_an_api_key_or_with_a_bad_file(tmp_path):
    key_file = tmp_path / "key"; key_file.write_text(KEY, encoding="utf-8")
    good = {"name": "review", "target": {"kind": "review", "url": "http://127.0.0.1:1/v1/review", "key_file": str(key_file)}, "repos": ["jonpol01/*"]}
    path = tmp_path / "routes.json"; path.write_text(json.dumps({"routes": [good]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="D1A_API_KEY"):
        serve.load_routes(str(path), None)
    assert serve.load_routes(None, None) is None and serve.load_routes(str(path), "k").review.target.timeout_s == 300
    bad = [good | {"is_review_minp": 0.9}, good | {"repos": ["*"]}, good | {"target": good["target"] | {"url": f"http://user:{KEY}@127.0.0.1/"}},
           good | {"target": good["target"] | {"url": "file:///etc/passwd"}}, good | {"target": good["target"] | {"kind": "shell"}},
           {"name": "large", "target": {"kind": "openai", "url": "http://127.0.0.1:1/v1/chat/completions"}}]
    for spec in bad:
        path.write_text(json.dumps({"routes": [spec]}), encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            serve.load_routes(str(path), "k")
        assert KEY not in str(e.value)
    key_file.write_text(f"{KEY}\nsecond-line", encoding="utf-8"); path.write_text(json.dumps({"routes": [good]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="one key on one line") as e:
        serve.load_routes(str(path), "k")
    key_file.write_text(KEY, encoding="utf-8")
    loaded = serve.load_routes(str(path), "k")
    assert KEY not in repr(loaded) and KEY not in json.dumps(loaded.card()) and "url" not in json.dumps(loaded.card())


def test_pull_requests_are_read_from_links_and_references_only():
    found = lambda text: routes.pull_requests(text)
    assert found(f"このPRをレビューして {LINK}をお願い") == ([("jonpol01", "d1a", "92")], True)
    assert found("レビューしてhttps://github.com/a/b/pull/7 と github.com/c/d/pull/8") == ([("a", "b", "7"), ("c", "d", "8")], True)
    assert found("see jonpol01/d1a#12 and https://GitHub.com/JonPol01/D1A/pull/12/files") == ([("JonPol01", "D1A", "12")], True)
    assert found("x.github.com/a/b/pull/3 user@github.com/a/b/pull/4 notgithub.com/a/b/pull/5 https://evil.com/github.com/a/b/pull/6") == ([], False)
    assert found("issue #12, a/b#0, a/b/pull/9, ../..#3") == ([], False)
