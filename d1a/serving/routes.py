"""Chat routes (#419): with D1A_ROUTES set, POST /v1/chat/completions picks the model that answers each chat. A request to
review a pull request goes to a review service, a hard request to a larger OpenAI-compatible model, and everything else is
answered by D1A as before. Off (D1A_ROUTES unset), requests reach the chat endpoint untouched.

D1A_ROUTES names a JSON file. Target URLs and keys come from that file only, never from a request:

    {"routes": [
      {"name": "review", "target": {"kind": "review", "url": "http://127.0.0.1:<port>/v1/review", "key_file": "<secret file>", "timeout_s": 300},
       "repos": ["<owner>/*", "<owner>/<repo>"], "is_review_min_p": 0.8},
      {"name": "large", "target": {"kind": "openai", "url": "http://127.0.0.1:<port>/v1/chat/completions", "model": "<name>", "key_file": "<secret file>", "timeout_s": 300}}]}

Either route may be left out. Only a chat whose last message is the user's is routed; mid tool loop (last message from a
tool) it stays with D1A. Routing runs before the chat endpoint parses the request, so D1A's own limits (its context,
max_tokens) never stop a chat meant for another model.

- review: one yes/no pass (IS_REVIEW) on the last user message, then review_request: a GitHub pull-request link with
  P(yes) >= 0.5 asks for that pull request's review; a fenced code block with a review verb (or P(yes) >= 0.5) for that
  code's; otherwise P(yes) >= is_review_min_p asks for the pull request named as owner/repo#n. The review service's text
  comes back verbatim. D1A never writes review text: a request naming no pull request or two, a repository not in
  `repos`, a busy or failed reviewer each get a fixed reply. `tools` are ignored. One review runs at a time.
- large: the route preset (d1a.agents.presets, use case routing, fail-up advice) read on the first and on the last user
  message; the larger tier wins, so a "continue" mid-task stays with the large model. On "large" the request body is sent
  as the client sent it (tools, stream options and all), with only `model` replaced, and the target's response comes back
  verbatim, streamed when the client streams. If the target cannot be reached, times out or fails (5xx) before its first
  byte, D1A answers instead, its reply starting with NOTE_LARGE_DOWN. A 4xx comes back as the target sent it.

Every chat answered with routes on carries `x-d1a-route` (the route's name, or "none"), and a route that failed also
`x-d1a-route-error` (ERRORS). The serving side (the passes, the middleware's place in the app) is in d1a.serving.serve;
this module loads no model.
"""
import asyncio
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from starlette.responses import JSONResponse, Response, StreamingResponse

from d1a.serving import openai_compat

log = logging.getLogger("d1a.chat")   # the chat log, printed by serve.log_chats: route, probabilities and outcome, never text or a key

PATH = "/v1/chat/completions"
IS_REVIEW = "Is this a request to review an existing pull request (read its diff and give feedback), rather than to write or change code?"   # the wording the zero-shot check measured (38/40, AUROC 0.96): rewording it voids that number
LINK_MIN_P = 0.5   # with a pull-request link in the message, the yes/no answer itself decides
TIERS = ("small", "medium", "large")

WHICH_PR = "Which pull request? Send its link and I'll have it reviewed."
NOT_LISTED = "That repository is not on the review list — no review was written."
BUSY = "The reviewer is busy — no review was written. Try again in a few minutes."
UNAVAILABLE = "reviewer unavailable — no review was written."   # the review service's own wording for this case
NOTE_LARGE_DOWN = "(The large model is unavailable — answered by D1A.)"
# x-d1a-route-error values: which-pr, not-listed, busy, connect, timeout, status <code>, empty, too-large, not-json, error.
# Fixed words and a status number only: never an exception's text, which can carry a URL.

REVIEW_MAX_BYTES = 256 * 1024   # the review service's reply
LARGE_MAX_BYTES = 16 << 20   # a whole (unstreamed) completion from the large model
KEEP_ALIVE_S = 10            # a streamed review sends an SSE comment this often until its text arrives

_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_REPO = r"[A-Za-z0-9._-]{1,100}"
_NUMBER = r"[0-9]{1,9}"
# ASCII classes: a link followed by Japanese ("…/pull/92をレビュー") ends at the number, and one written right after Japanese
# text still starts at its scheme. The lookbehind keeps "x.github.com", "notgithub.com" and "user@github.com" out.
PR_LINK = re.compile(r"(?<![A-Za-z0-9._@/-])(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s]+)/pull/([0-9]+)(?![0-9])", re.IGNORECASE)
PR_REF = re.compile(r"(?<![A-Za-z0-9._/#-])([A-Za-z0-9-]+)/([A-Za-z0-9._-]+)#([0-9]+)(?![0-9])")
# Code to review: everything from the first fence's line to the last closing fence, as the review service counts its lines;
# its path from "… from <path>:" in the prose, else CODE_PATH (a .py name: the service splits on def and class).
FENCE = re.compile(r"^ {0,3}(```|~~~)[^\r\n]*\r?\n(.*?)\r?\n {0,3}\1[ \t]*\r?$(?![\s\S]*^ {0,3}\1[ \t]*\r?$)", re.S | re.M)
FROM_PATH = re.compile(r"\bfrom\s+[`'\"]?([^\s`'\"]*[A-Za-z0-9_])[`'\"]?:")
# whole words only, and no "check": "checkout", "checkbox" and "check why and fix it" are not review requests (p >= 0.5
# still routes one that is)
REVIEW_VERB = re.compile(r"\b(?:review|reviewing|look over)\b|レビュー", re.IGNORECASE)
CODE_PATH = "snippet.py"


@dataclass(frozen=True)
class Target:
    kind: str        # "review" or "openai"
    url: str
    timeout_s: float
    model: str | None = None                           # openai: the model name sent to the target
    key: str | None = field(default=None, repr=False)   # never printed


@dataclass(frozen=True)
class Route:
    name: str
    target: Target
    repos: tuple[str, ...] = ()   # review: "owner/repo" or "owner/*", lower case
    is_review_min_p: float = 0.8

    def allows(self, pr):
        owner, repo, _ = pr
        return any(p == f"{owner}/{repo}".lower() or p == f"{owner.lower()}/*" for p in self.repos)


@dataclass(frozen=True)
class Routes:
    review: Route | None
    large: Route | None

    def card(self):
        """What GET /v1/models shows: no URL, no key."""
        return [{"name": r.name, "kind": r.target.kind} | ({"repos": list(r.repos), "is_review_min_p": r.is_review_min_p} if r is self.review else {"model": r.target.model})
                for r in (self.review, self.large) if r is not None]


def _keys(obj, where, required, optional):
    if not isinstance(obj, dict):
        raise ValueError(f"D1A_ROUTES: {where} must be an object")
    if missing := [k for k in required if k not in obj]:
        raise ValueError(f"D1A_ROUTES: {where} is missing {', '.join(missing)}")
    if unknown := sorted(set(obj) - set(required) - set(optional)):
        raise ValueError(f"D1A_ROUTES: {where} has unknown keys {', '.join(unknown)}")   # a misspelt key never silently falls back to a default


def _number(value, where, low, high, default):
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low < value <= high:
        raise ValueError(f"D1A_ROUTES: {where} must be a number above {low} and at most {high}")
    return float(value)


def _target(t, where):
    kind = t.get("kind") if isinstance(t, dict) else None
    if kind not in ("review", "openai"):
        raise ValueError(f"D1A_ROUTES: {where}'s target kind must be \"review\" or \"openai\"")
    _keys(t, f"{where}'s target", ("kind", "url") + (("model",) if kind == "openai" else ()), ("key_file", "timeout_s"))
    url = urlsplit(t["url"]) if isinstance(t["url"], str) else None
    if url is None or url.scheme not in ("http", "https") or not url.hostname or url.username is not None or url.password is not None:
        raise ValueError(f"D1A_ROUTES: {where}'s target url must be http(s)://host[:port]/path, without credentials")
    if kind == "openai" and (not isinstance(t["model"], str) or not t["model"]):
        raise ValueError(f"D1A_ROUTES: {where}'s target needs a model name")
    key = None
    if t.get("key_file") is not None:
        try: key = Path(t["key_file"]).read_text(encoding="utf-8").strip()
        except (OSError, TypeError) as e: raise ValueError(f"D1A_ROUTES: cannot read {where}'s key file: {getattr(e, 'strerror', None) or type(e).__name__}") from None
        if not key or any(c.isspace() for c in key):
            raise ValueError(f"D1A_ROUTES: {where}'s key file must hold one key on one line")
    return Target(kind, t["url"], _number(t.get("timeout_s"), f"{where}'s timeout_s", 0, 3600, 300), t.get("model"), key)


def load(path):
    """The routes in the D1A_ROUTES file at `path`, checked: -> Routes. Raises ValueError naming what is wrong, never a key.
    At most one route of each kind: "review" (the review route) and "openai" (the large route)."""
    try: spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e: raise ValueError(f"D1A_ROUTES: cannot read {path}: {e}") from None
    _keys(spec, "the file", ("routes",), ())
    if not isinstance(spec["routes"], list) or not spec["routes"]:
        raise ValueError("D1A_ROUTES: routes must be a non-empty list")
    by_kind, names = {}, set()
    for i, r in enumerate(spec["routes"]):
        name = r.get("name") if isinstance(r, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", name) or name == "none":
            raise ValueError(f'D1A_ROUTES: route {i} needs a name of lower-case letters, digits, "-" or "_" (not "none")')
        if name in names: raise ValueError(f"D1A_ROUTES: two routes are named {name}")
        names.add(name)
        target = _target(r.get("target"), name)
        if target.kind in by_kind: raise ValueError(f"D1A_ROUTES: more than one route of kind {target.kind}")
        if target.kind == "review":
            _keys(r, name, ("name", "target", "repos"), ("is_review_min_p",))
            repos, pattern = r["repos"], re.compile(rf"{_OWNER}/(?:\*|{_REPO})")
            if not isinstance(repos, list) or not repos or not all(isinstance(p, str) and pattern.fullmatch(p) for p in repos):
                raise ValueError(f'D1A_ROUTES: {name}\'s repos must be a non-empty list of "owner/repo" or "owner/*"')
            by_kind["review"] = Route(name, target, tuple(p.lower() for p in repos), _number(r.get("is_review_min_p"), f"{name}'s is_review_min_p", 0, 1, 0.8))
        else:
            _keys(r, name, ("name", "target"), ())
            by_kind["openai"] = Route(name, target)
    return Routes(by_kind.get("review"), by_kind.get("openai"))


def _valid(owner, repo, number):
    return bool(re.fullmatch(_OWNER, owner) and re.fullmatch(_REPO, repo) and repo not in (".", "..")
                and re.fullmatch(_NUMBER, number) and int(number) > 0)


def pull_requests(text):
    """-> (the pull requests a message names, as (owner, repo, number) in order of mention without repeats, compared
    case-insensitively; whether any came as a GitHub link). Links count, and so do explicit "owner/repo#123" references."""
    found, linked = {}, False
    for regex, is_link in ((PR_LINK, True), (PR_REF, False)):
        for owner, repo, number in regex.findall(text):
            if not _valid(owner, repo, number): continue
            linked |= is_link
            found.setdefault((owner.lower(), repo.lower(), int(number)), (owner, repo, str(int(number))))
    return list(found.values()), linked


def user_texts(chat):
    """(the first user message's text, the last message's text) when the chat's last message is the user's, else None (an
    agent's tool round, a prefilled reply, a body the endpoint will refuse)."""
    messages = chat.get("messages") if isinstance(chat, dict) else None
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages) or messages[-1].get("role") != "user":
        return None
    try:
        first = next(m for m in messages if m.get("role") == "user")
        return openai_compat.text_of(first.get("content")), openai_compat.text_of(messages[-1].get("content"))
    except (AttributeError, TypeError):   # content the endpoint will refuse
        return None


def _reason(e):
    import httpx
    if isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException)): return "timeout"
    if isinstance(e, (httpx.ConnectError, OSError)): return "connect"
    return "error"


def _client(target):
    """An HTTP client for `target`: redirects not followed, and no proxy or .netrc from the environment, so the key goes to
    the configured URL and nowhere else. timeout_s bounds each read (connect: 10 s at most)."""
    import httpx
    return httpx.AsyncClient(follow_redirects=False, trust_env=False, timeout=httpx.Timeout(target.timeout_s, connect=min(10.0, target.timeout_s)))


def _headers(target):
    # the configured key, never the client's; identity encoding, so a streamed answer passes through as the text it is
    return {"content-type": "application/json", "accept-encoding": "identity"} | ({"authorization": f"Bearer {target.key}"} if target.key else {})


async def _read(resp, limit):
    body = bytearray()
    async for part in resp.aiter_bytes():
        body += part
        if len(body) > limit: return None
    return bytes(body)


@dataclass
class Outcome:
    text: str
    error: str | None = None   # x-d1a-route-error


def _relative_path(path):
    """A repository-relative file path such as scripts/x.py: not a URL, an absolute or home path, or one climbing with ..."""
    return ("/" in path or "." in path) and "://" not in path and not path.startswith(("/", "~", "\\")) and ".." not in path.split("/")


def review_request(text, p, min_p):
    """What the review route makes of a user message, given P(yes) of IS_REVIEW about it. Checked in order, on the prose
    (the text outside a fenced code block): a GitHub pull-request link and p >= 0.5; a fenced code block with a review
    verb in the prose or p >= 0.5; p >= min_p. -> ("pr", (owner, repo, number)), ("code", {path: code}), ("which", None)
    when a pull request is wanted but none or several are named, or None: not a review request."""
    fence = FENCE.search(text)
    prose = text[:fence.start()] + text[fence.end():] if fence else text
    code = fence.group(2) if fence else None
    prs, linked = pull_requests(prose)
    if linked and p >= LINK_MIN_P:
        return ("pr", prs[0]) if len(prs) == 1 else ("which", None)
    if fence and (REVIEW_VERB.search(prose) or p >= LINK_MIN_P):
        path = next((m for m in FROM_PATH.findall(text[:fence.start()]) if _relative_path(m)), CODE_PATH)
        return "code", {path: code}
    if p >= min_p:
        return ("pr", prs[0]) if len(prs) == 1 else ("which", None)
    return None


async def review(target, body):
    """POST `body` ({"mode": "pr", "pr": "owner/repo#n"} or {"mode": "code", "files": {path: code}}) to the review service.
    -> Outcome: the `text` its reply carries, verbatim, whatever the status (on a 422 or 503 that is the service's own "no
    review was written"), else UNAVAILABLE. The whole call is bounded by timeout_s and the reply by REVIEW_MAX_BYTES."""
    async def call():
        async with _client(target) as client, client.stream("POST", target.url, json=body, headers=_headers(target)) as resp:
            status, data = resp.status_code, await _read(resp, REVIEW_MAX_BYTES)
        if data is None: return None, "too-large"
        try: text, parsed = json.loads(data).get("text"), True
        except (ValueError, AttributeError): text, parsed = None, False
        text = text if isinstance(text, str) and text.strip() else None
        return text, None if status == 200 and text else f"status {status}" if status != 200 else "empty" if parsed else "not-json"

    try: text, why = await asyncio.wait_for(call(), target.timeout_s)
    except Exception as e: text, why = None, _reason(e)
    return Outcome(UNAVAILABLE if text is None else text, why)


def completion(chat, model_name, content):
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": chat.get("model") or model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _with_error(headers, error):
    return headers | ({"x-d1a-route-error": error} if error else {})


class ChatRoutes:
    """ASGI middleware in front of POST /v1/chat/completions. `routes()` gives the Routes in force (None: off, and every
    request passes straight through); `p_review(text)` and `tier(text)` are the server's passes (async); `model_name()` the
    name D1A's chat answers as. Placed inside the auth and CORS middleware, so a routed chat needs the API key too."""

    def __init__(self, app, routes, p_review, tier, model_name):
        self.app, self.routes, self.p_review, self.tier, self.model_name = app, routes, p_review, tier, model_name
        self.slot = threading.Lock()   # one review in flight; taken without waiting, so a second request is told the reviewer is busy
        self.tasks = set()             # the reviews running, held until done

    async def __call__(self, scope, receive, send):
        routes = self.routes() if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == PATH else None
        if routes is None:
            return await self.app(scope, receive, send)
        body, more = bytearray(), True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect": return
            body += message.get("body", b""); more = message.get("more_body", False)
        body, sent = bytes(body), False

        async def replay():   # the endpoint reads the body again; after it, the client's own messages (a disconnect)
            nonlocal sent
            if sent: return await receive()
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        try: chat = json.loads(body)
        except ValueError: chat = None
        texts = user_texts(chat)
        route, why, t = None, "no user turn", time.time()
        if texts is not None and texts[1].strip():
            first, last = texts
            jobs, ask = {}, None
            if routes.review: jobs["p"] = self.p_review(last)
            if routes.large: jobs |= {"last": self.tier(last)} | ({"first": self.tier(first)} if first != last and first.strip() else {})
            try: got = dict(zip(jobs, await asyncio.gather(*jobs.values())))   # one model batch
            except Exception as e:   # the passes failed (the server stopping, say): D1A's chat answers, or fails, as without routes
                log.info("route none (passes failed: %s)", type(e).__name__)
                return await self.app(scope, replay, _tagged(send, {"x-d1a-route": "none", "x-d1a-route-error": "error"}))
            tier = max((got[k] for k in ("first", "last") if k in got), key=TIERS.index, default=None)
            if "p" in got and (ask := review_request(last, got["p"], routes.review.is_review_min_p)): route = routes.review
            elif tier == "large": route = routes.large
            why = " ".join(f"{k} {v:.2f}" if k == "p" else f"tier {v}" if k == "last" else f"first {v}" for k, v in got.items()) + (f", {ask[0]}" if ask else "")
        if route is None:
            log.info("route none (%s)", why)
            return await self.app(scope, replay, _tagged(send, {"x-d1a-route": "none"}))
        stream = chat.get("stream") is True
        if route is routes.review:
            response = await self._review(route, chat, ask, stream)
        else:
            response = await self._large(route, chat, stream)
        if isinstance(response, str):   # the large model failed before its first byte: D1A answers, with the note
            log.info("route %s (%s): unavailable (%s), answered by D1A, %.1f s", route.name, why, response, time.time() - t)
            return await self.app(scope, replay, _noted(send, {"x-d1a-route": route.name, "x-d1a-route-error": response}, chat.get("model") or self.model_name()))
        log.info("route %s (%s): %s", route.name, why, response.headers.get("x-d1a-route-error") or "sent")
        try: await response(scope, receive, send)
        finally:
            if close := getattr(response, "close_upstream", None): await close()

    async def _review(self, route, chat, ask, stream):
        headers = {"x-d1a-route": route.name}
        reply = lambda outcome: _completion_response(completion(chat, self.model_name(), outcome.text), stream, _with_error(headers, outcome.error))
        kind, what = ask
        if kind == "which": return reply(Outcome(WHICH_PR, "which-pr"))
        if kind == "pr" and not route.allows(what): return reply(Outcome(NOT_LISTED, "not-listed"))
        body = {"mode": "pr", "pr": f"{what[0]}/{what[1]}#{what[2]}"} if kind == "pr" else {"mode": "code", "files": what}
        if not self.slot.acquire(blocking=False): return reply(Outcome(BUSY, "busy"))
        try:
            task = asyncio.ensure_future(review(route.target, body))
            self.tasks.add(task)
        except BaseException:
            self.slot.release(); raise
        task.add_done_callback(lambda done: (self.tasks.discard(done), self.slot.release()))   # done, failed or cancelled: a client gone before its stream started included
        if not stream: return reply(await task)
        response = StreamingResponse(_review_stream(task, chat, self.model_name()), media_type="text/event-stream", headers=headers)
        response.close_upstream = lambda: _cancel(task)   # a client gone before or during the stream frees the slot now, not at timeout_s
        return response

    async def _large(self, route, chat, stream):
        """The large model's response, or the reason it failed before its first byte (a str): D1A then answers."""
        target, client, upstream, keep = route.target, _client(route.target), None, False
        headers = {"x-d1a-route": route.name}
        try:
            request = client.build_request("POST", target.url, content=json.dumps(chat | {"model": target.model}).encode(), headers=_headers(target))
            upstream = await client.send(request, stream=True)
            status, kind = upstream.status_code, upstream.headers.get("content-type", "application/json")
            if status != 200 and not 400 <= status < 500:   # 5xx, and a redirect, which is not followed
                return f"status {status}"
            if status != 200 or not stream:   # a 4xx verbatim, or the whole completion
                body = await _read(upstream, LARGE_MAX_BYTES)
                if body is None: return "too-large"
                if status == 200 and not body: return "empty"
                return Response(body, status, _with_error(headers, None if status == 200 else f"status {status}"), media_type=kind)
            parts = upstream.aiter_bytes()   # decoded, should a target compress anyway (content-encoding is not passed on)
            first = await anext(parts, b"")
            if not first: return "empty"

            async def passthrough():   # the target's SSE bytes as they come
                yield first
                async for part in parts: yield part
            response = StreamingResponse(passthrough(), 200, headers, media_type=kind)
            response.close_upstream, keep = lambda: _close(upstream, client), True   # __call__ closes it once the client is done or gone
            return response
        except Exception as e:
            return _reason(e)
        finally:
            if not keep: await _close(upstream, client)


async def _close(upstream, client):
    if upstream is not None: await upstream.aclose()
    await client.aclose()


async def _cancel(task):
    task.cancel()


def _completion_response(body, stream, headers):
    if stream: return StreamingResponse(openai_compat.stream_chunks(body), media_type="text/event-stream", headers=headers)
    return JSONResponse(body, headers=headers)


async def _review_stream(task, chat, model_name):
    """The role chunk at once, a keep-alive comment every KEEP_ALIVE_S while the review runs, then its text. The headers
    went out with the first chunk, so a failed review shows only in the text."""
    body_id, created, name = f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), chat.get("model") or model_name
    try:
        yield openai_compat.chunk(body_id, created, name, {"role": "assistant", "content": ""})
        while not (await asyncio.wait({task}, timeout=KEEP_ALIVE_S))[0]:
            yield ": keep-alive\n\n"
    finally:
        task.cancel()   # no-op once done; with the client gone, the review stops and its slot is free
    yield openai_compat.chunk(body_id, created, name, {"content": task.result().text})
    yield openai_compat.chunk(body_id, created, name, {}, "stop")
    yield "data: [DONE]\n\n"


def _add_headers(message, headers):
    return message | {"headers": list(message.get("headers", [])) + [(k.encode(), v.encode()) for k, v in headers.items()]}


def _tagged(send, headers):
    """`send` with `headers` added to the response's."""
    async def tagged(message):
        await send(_add_headers(message, headers) if message["type"] == "http.response.start" else message)
    return tagged


def _noted(send, headers, model_name):
    """`send` for D1A's answer after the large model failed: NOTE_LARGE_DOWN starts the reply (a whole completion is
    rewritten, a stream gets a first chunk carrying it), and `headers` are added. An error response passes unchanged."""
    start, parts = None, []

    async def noted(message):
        nonlocal start
        if message["type"] == "http.response.start":
            start = _add_headers(message, headers)
            if start["status"] != 200: await send(start); start = False
            elif dict(start["headers"]).get(b"content-type", b"").startswith(b"text/event-stream"):
                await send(start); start = False
                note = openai_compat.chunk(f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), model_name, {"role": "assistant", "content": NOTE_LARGE_DOWN + "\n"})
                await send({"type": "http.response.body", "body": note.encode(), "more_body": True})
            return
        if start is False or message["type"] != "http.response.body":
            return await send(message)
        parts.append(message.get("body", b""))
        if message.get("more_body"): return
        body = b"".join(parts)
        try:
            data = json.loads(body)
            msg = data["choices"][0]["message"]
            msg["content"] = NOTE_LARGE_DOWN + ("\n" + msg["content"] if msg.get("content") else "")
            body = json.dumps(data, ensure_ascii=False).encode()
        except (ValueError, KeyError, IndexError, TypeError):
            pass
        await send({**start, "headers": [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"] + [(b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
    return noted
