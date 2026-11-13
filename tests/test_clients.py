"""The dependency-free clients (clients/python, clients/js) against the server's own request models: every request they send
goes to a route the server has and validates as its endpoint's pydantic model (SystemOneRequest, MediaRequest, Feedback,
ChatRequest), with headers the server reads as meant; both clients send the same requests; a refusal comes back with the
server's reason; and the API key never follows a redirect to another host. No model: stand-in HTTP servers record what
arrives."""
import base64
import importlib.util
import json
import shutil
import subprocess
import threading
import urllib.error
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from d1a.core.api import SystemOneRequest
from d1a.serving import openai_compat, serve
from d1a.serving.media import MediaRequest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d1a_client", ROOT / "clients/python/d1a_client/__init__.py")
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)

MODELS = {"/v1/systemone": SystemOneRequest, "/v1/systemone/media": MediaRequest, "/v1/feedback": serve.Feedback,
          "/v1/chat/completions": openai_compat.ChatRequest}
QUESTIONS = {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": None, "billing": None}},
             "urgent": {"type": "noul", "instructions": "Does it need an answer today?"}}
MEDIA = b"\x89PNG\r\n\x1a\n" + bytes(range(256))   # never decoded: the server is a stand-in
DECISION = "ab" * 16
KEPT_HEADERS = ("content-type", "authorization", serve.LOG_HEADER)
REFUSAL = {"detail": [{"loc": ["body", "questions"], "msg": "Field required"}]}

# The same calls as python_calls, through clients/js (Node 18+: fetch, Blob), then its refusals.
JS_CALLS = """
const fs = require("fs");
const { Client, D1AError } = require(process.argv[2]);
const [url, mediaPath] = process.argv.slice(3);
const questions = JSON.parse(process.argv[5]);
(async () => {
  const c = new Client(url, { apiKey: "k" });
  await c.decide("Charged twice.", questions);
  await c.decide("Charged twice.", questions, { useCase: "routing", log: false });
  await c.decideMedia("image", fs.readFileSync(mediaPath), questions, { state: "from the driver" });
  await c.decideMedia("audio", new Blob([fs.readFileSync(mediaPath)]), questions);
  await c.feedback("%s", { team: "billing", urgent: true }, { src: "human", group: "pr-1" });
  await c.feedback("%s", { urgent: false });
  await c.models();
  await c.chat([{ role: "user", content: "hi" }], { max_tokens: 8 });
  await new Client(url + "/moved", { apiKey: "k" }).models();
  for (const [path, status, detail] of [["/refuse", 422, "Field required"], ["/text", 500, "boom"]]) {
    try { await new Client(url + path).decide("x", questions); process.exit(3); }
    catch (e) { if (!(e instanceof D1AError) || e.status !== status || !e.detail.includes(detail)) throw e; }
  }
  try { await c.chat([{ role: "user", content: "hi" }], { stream: true }); process.exit(4); }
  catch (e) { if (e instanceof D1AError) throw e; }
})().catch((e) => { console.error(e); process.exit(2); });
""" % (DECISION, DECISION)


class Recorder(BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply(None)

    def do_POST(self):
        self.reply(json.loads(self.rfile.read(int(self.headers["content-length"]))))

    def reply(self, body):
        extra = {}
        if self.path.startswith("/refuse"):
            status, ctype, out = 422, "application/json", json.dumps(REFUSAL)
        elif self.path.startswith("/text"):
            status, ctype, out = 500, "text/plain", "boom\n"
        elif self.path == "/moved/v1/models":   # to the other server: another host as far as a client can tell
            status, ctype, out = 302, "text/plain", ""
            extra["location"] = self.server.moved_to + "/v1/models"
        else:
            headers = {k.lower(): v for k, v in self.headers.items()}
            self.server.seen.append((self.command, self.path, {k: headers[k] for k in KEPT_HEADERS if k in headers}, body))
            status, ctype, out = 200, "application/json", json.dumps({"ok": True})
        self.send_response(status)
        self.send_header("content-type", ctype)
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(out.encode())

    def log_message(self, *args):
        pass


def start():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    httpd.seen = []
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def url(httpd):
    return f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def server():
    httpd, other = start(), start()
    httpd.other, httpd.moved_to = other, url(other)
    yield httpd
    for s in (httpd, other):
        s.shutdown()
        s.server_close()


def python_calls(base, media_path):
    c = client.Client(base, api_key="k")
    c.decide("Charged twice.", QUESTIONS)
    c.decide("Charged twice.", QUESTIONS, use_case="routing", log=False)
    c.decide_media("image", media_path, QUESTIONS, state="from the driver")
    c.decide_media("audio", MEDIA, QUESTIONS)
    c.feedback(DECISION, {"team": "billing", "urgent": True}, src="human", group="pr-1")
    c.feedback(DECISION, {"urgent": False})
    c.models()
    c.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    client.Client(base + "/moved", api_key="k").models()


def test_clients_call_routes_the_server_has():
    routes = {(m, r.path) for r in serve.app.routes for m in getattr(r, "methods", None) or ()}
    assert {("POST", p) for p in MODELS} | {("GET", "/v1/models")} <= routes


def test_python_client_requests_match_the_server(server, tmp_path):
    (tmp_path / "photo.png").write_bytes(MEDIA)
    python_calls(url(server), tmp_path / "photo.png")
    seen = server.seen
    assert [(m, p) for m, p, _, _ in seen] == [("POST", "/v1/systemone")] * 2 + [("POST", "/v1/systemone/media")] * 2 + \
        [("POST", "/v1/feedback")] * 2 + [("GET", "/v1/models"), ("POST", "/v1/chat/completions")]
    for method, path, headers, body in seen:
        assert headers["authorization"] == "Bearer k"
        if method == "POST":
            assert headers["content-type"] == "application/json"
            MODELS[path].model_validate(body)   # what the endpoint parses the body as: a refused field fails here
        else:
            assert body is None and "content-type" not in headers
    (_, _, h1, plain), (_, _, h2, routed) = seen[:2]
    assert "use_case" not in plain and serve.logged(SimpleNamespace(headers=h1)) is True   # without the options: as before
    assert routed["use_case"] == "routing" and serve.logged(SimpleNamespace(headers=h2)) is False and plain["model"] == "d1a-latest"
    photo, voice = seen[2][3], seen[3][3]
    assert base64.b64decode(photo["media"]["data"]) == MEDIA and photo["media"]["type"] == "image" and photo["state"] == "from the driver"
    assert voice["media"]["type"] == "audio" and voice["state"] is None
    assert seen[4][3] == {"decision_id": DECISION, "labels": {"team": "billing", "urgent": True}, "src": "human", "group": "pr-1"}
    assert seen[5][3] == {"decision_id": DECISION, "labels": {"urgent": False}}
    assert seen[7][3] == {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}   # no model: the server names its chat model
    assert server.other.seen == [("GET", "/v1/models", {}, None)]   # the redirect was followed, without the key


def test_python_client_refusal_carries_the_reason(server):
    with pytest.raises(urllib.error.HTTPError) as refused:   # still urllib's error, for code written before D1AError
        client.Client(url(server) + "/refuse").decide("x", QUESTIONS)
    assert isinstance(refused.value, client.D1AError) and refused.value.code == 422
    assert refused.value.reason == HTTPStatus(422).phrase and json.loads(refused.value.read()) == REFUSAL   # as urllib's
    assert "Field required" in refused.value.detail and "Field required" in str(refused.value)
    with pytest.raises(client.D1AError) as failed:
        client.Client(url(server) + "/text").models()
    assert failed.value.code == 500 and failed.value.detail == "boom"
    with pytest.raises(ValueError):
        client.Client(url(server)).chat([{"role": "user", "content": "hi"}], stream=True)
    assert server.seen == []


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_js_client_sends_what_the_python_client_sends(server, tmp_path):
    version = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout
    if int(version.lstrip("v").split(".")[0]) < 18:
        pytest.skip(f"the JS client needs Node 18+ (fetch), found {version.strip()}")
    (tmp_path / "photo.png").write_bytes(MEDIA)
    (tmp_path / "calls.js").write_text(JS_CALLS, encoding="utf-8")
    python_calls(url(server), tmp_path / "photo.png")
    from_python = server.seen, server.other.seen
    server.seen, server.other.seen = [], []
    run = subprocess.run(["node", str(tmp_path / "calls.js"), str(ROOT / "clients/js/index.js"), url(server),
                          str(tmp_path / "photo.png"), json.dumps(QUESTIONS)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert (server.seen, server.other.seen) == from_python
