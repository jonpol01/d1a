"""Chat routes (#419): with D1A_ROUTES set, POST /v1/chat/completions picks the model that answers each chat. A request to
review a pull request goes to a review service, a hard request to a larger OpenAI-compatible model, and everything else is
answered by D1A as before. Off (D1A_ROUTES unset), requests reach the chat endpoint untouched.

D1A_ROUTES names a JSON file. Target URLs and keys come from that file only, never from a request:

    {"client_key_file": "<secret file>",
     "routes": [
      {"name": "review", "target": {"kind": "review", "url": "http://127.0.0.1:<port>/v1/review", "key_file": "<secret file>", "timeout_s": 300},
       "repos": ["<owner>/*", "<owner>/<repo>"], "is_review_min_p": 0.8},
      {"name": "large", "target": {"kind": "openai", "url": "http://127.0.0.1:<port>/v1/chat/completions", "model": "<name>", "key_file": "<secret file>", "timeout_s": 300},
       "max_in_flight": 2, "min_tier": "large"}]}

Either route may be left out, and so may client_key_file. Without it every caller of the chat endpoint is routed: anyone
who can reach the server can send work to the targets. With it, only a chat carrying `Authorization: Bearer <client key>`
is routed (every chat when D1A_API_KEY is set, which the server then requires of every request), and any other chat is
answered by D1A exactly as without routes. Routing runs before the chat endpoint parses the request, so D1A's own limits
(its context, max_tokens) never stop a chat meant for another model.

- review: only a chat whose last message is the user's. One yes/no pass (IS_REVIEW) on that message, then review_request:
  a GitHub pull-request link with P(yes) >= 0.5 asks for that pull request's review; a fenced code block with P(yes) >=
  0.5, or with a review verb when the block is code, for that code's; otherwise P(yes) >= is_review_min_p asks for the
  pull request named as owner/repo#n, or, with a review verb but none named, which one. The review service's text comes back verbatim. D1A never writes review text: a
  request naming no pull request or two, a repository not in `repos`, a busy or failed reviewer each get a fixed reply.
  `tools` are ignored. One review runs at a time.
- large: the route preset (d1a.agents.presets, use case routing, fail-up advice) read on the first user message and the
  last three (tier_texts); the largest tier wins, so a "continue" mid-task stays with the large model, and so does an
  agent's tool turn (last message from a tool), whose tier is read the same way. At min_tier or above ("large" by default;
  "medium" sends everything but small requests) the request body is sent as the client sent it (tools, stream options
  and all), with only `model` replaced, and the target's response comes back verbatim, streamed when the client streams.
  If the target cannot be reached, times out or fails (5xx) before its first byte, D1A answers instead, its reply
  starting with NOTE_LARGE_DOWN. A 4xx comes back as the target sent it. At most max_in_flight (2) chats are forwarded at
  once; another is answered by D1A, its reply starting with NOTE_LARGE_BUSY.

Every chat answered with routes on carries `x-d1a-route` (the route's name, or "none"), and a route that failed also
`x-d1a-route-error` (ERRORS). The serving side (the passes, the middleware's place in the app) is in d1a.serving.serve;
this module loads no model.
"""
import asyncio
import hmac
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
NOTE_LARGE_BUSY = "(The large model is busy — answered by D1A.)"
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
FENCE = re.compile(r"^ {0,3}(```|~~~)([^\r\n]*)\r?\n(.*?)\r?\n {0,3}\1[ \t]*\r?$(?![\s\S]*^ {0,3}\1[ \t]*\r?$)", re.S | re.M)
NOT_CODE = {"", "text", "txt", "plain", "plaintext", "markdown", "md", "json", "log", "console", "output", "csv"}   # fences a review verb alone does not make code to review
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
    max_in_flight: int = 1        # forwards at once: one review; large: max_in_flight in the file (2)
    min_tier: str = "large"       # large: the smallest tier sent to the target
    slots: threading.BoundedSemaphore = field(init=False, repr=False, compare=False)   # taken without waiting: a chat over the cap is told so

    def __post_init__(self):
        object.__setattr__(self, "slots", threading.BoundedSemaphore(self.max_in_flight))

    def allows(self, pr):
        owner, repo, _ = pr
        return any(p == f"{owner}/{repo}".lower() or p == f"{owner.lower()}/*" for p in self.repos)


@dataclass(frozen=True)
class Routes:
    review: Route | None
    large: Route | None
    client_key: str | None = field(default=None, repr=False)   # a chat carrying it is routed; never printed

    def card(self):
        """What GET /v1/models shows: no URL, no key."""
        return [{"name": r.name, "kind": r.target.kind} | ({"repos": list(r.repos), "is_review_min_p": r.is_review_min_p} if r is self.review else {"model": r.target.model, "max_in_flight": r.max_in_flight, "min_tier": r.min_tier})
                for r in (self.review, self.large) if r is not None] + ([{"client_key": True}] if self.client_key else [])


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


def _key(path, where):
    """The key in a key file: one line, no spaces. Raises ValueError naming the file's role, never the key."""
    try: key = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, TypeError) as e: raise ValueError(f"D1A_ROUTES: cannot read {where}: {getattr(e, 'strerror', None) or type(e).__name__}") from None
    if not key or any(c.isspace() for c in key):
        raise ValueError(f"D1A_ROUTES: {where} must hold one key on one line")
    return key


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
    key = _key(t["key_file"], f"{where}'s key file") if t.get("key_file") is not None else None
    return Target(kind, t["url"], _number(t.get("timeout_s"), f"{where}'s timeout_s", 0, 3600, 300), t.get("model"), key)


def load(path):
    """The routes in the D1A_ROUTES file at `path`, checked: -> Routes. Raises ValueError naming what is wrong, never a key.
    At most one route of each kind: "review" (the review route) and "openai" (the large route)."""
    try: spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e: raise ValueError(f"D1A_ROUTES: cannot read {path}: {e}") from None
    _keys(spec, "the file", ("routes",), ("client_key_file",))
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
            _keys(r, name, ("name", "target"), ("max_in_flight", "min_tier"))
            cap, min_tier = r.get("max_in_flight", 2), r.get("min_tier", "large")
            if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 64:
                raise ValueError(f"D1A_ROUTES: {name}'s max_in_flight must be a whole number from 1 to 64")
            if min_tier not in ("medium", "large"):
                raise ValueError(f'D1A_ROUTES: {name}\'s min_tier must be "medium" or "large"')
            by_kind["openai"] = Route(name, target, max_in_flight=cap, min_tier=min_tier)
    client_key = _key(spec["client_key_file"], "the client key file") if spec.get("client_key_file") is not None else None
    return Routes(by_kind.get("review"), by_kind.get("openai"), client_key)


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


def user_turns(chat):
    """(the texts of the chat's user messages in order, whether its last message is the user's), or None for a body the
    endpoint will refuse."""
    messages = chat.get("messages") if isinstance(chat, dict) else None
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        return None
    try: texts = [openai_compat.text_of(m.get("content")) for m in messages if m.get("role") == "user"]
    except (AttributeError, TypeError): return None   # content the endpoint will refuse
    return texts, messages[-1].get("role") == "user"


def tier_texts(texts):
    """The user messages the tier is read on: the first and the last three, without repeats (at most four passes)."""
    return [t for t in dict.fromkeys(texts[:1] + texts[-3:]) if t.strip()]


def _carries(scope, key):
    """Whether the request's Authorization header is exactly "Bearer <key>", compared in constant time."""
    if not key: return False
    given = next((v for k, v in scope.get("headers", ()) if k.lower() == b"authorization"), b"")
    return hmac.compare_digest(given, f"Bearer {key}".encode())


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
    """A repository-relative file path such as scripts/x.py: a file name with an extension that has a letter in it (not
    "v1.2"), and not a URL, an absolute or home path, or one climbing with ..."""
    return (re.fullmatch(r"[^/]+\.[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*", path.rsplit("/", 1)[-1]) is not None and len(path.rsplit(".", 1)[-1]) <= 8
            and "://" not in path and not path.startswith(("/", "~", "\\")) and ".." not in path.split("/"))


def review_request(text, p, min_p):
    """What the review route makes of a user message, given P(yes) of IS_REVIEW about it. Checked in order, on the prose
    (the text outside a fenced code block): a GitHub pull-request link and p >= 0.5; a fenced code block with p >= 0.5,
    or with a review verb in the prose when the block is code (a language tag that is not text, markdown, JSON or a log,
    or a "from <path>:" naming it); p >= min_p with a pull request named as owner/repo#n, or with a review verb. -> ("pr", (owner, repo, number)), ("code", {path: code}), ("which", None)
    when a pull request is wanted but none or several are named, or None: not a review request."""
    fence = FENCE.search(text)
    prose = text[:fence.start()] + text[fence.end():] if fence else text
    code = fence.group(3) if fence else None
    prs, linked = pull_requests(prose)
    if linked and p >= LINK_MIN_P:
        return ("pr", prs[0]) if len(prs) == 1 else ("which", None)
    if fence:
        path = next((m for m in FROM_PATH.findall(text[:fence.start()]) if _relative_path(m)), None)
        tag = (fence.group(2).split() or [""])[0].lower()
        code_like = tag not in NOT_CODE or not tag and path is not None   # a text, JSON or log tag is never code, whatever path is named
        if p >= LINK_MIN_P or code_like and REVIEW_VERB.search(prose):
            return "code", {path or CODE_PATH: code}
    if p >= min_p and (prs or REVIEW_VERB.search(prose)):   # naming no pull request, it must say "review": a bare "continue" read 0.76
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
    request passes straight through); `keyed()` whether the server requires D1A_API_KEY (the auth middleware outside this
    one has then checked it); `p_review(text)` and `tier(text)` are the server's passes (async); `model_name()` the name
    D1A's chat answers as. A chat that is not authenticated (see the module docstring) passes straight through too."""

    def __init__(self, app, routes, keyed, p_review, tier, model_name):
        self.app, self.routes, self.keyed, self.p_review, self.tier, self.model_name = app, routes, keyed, p_review, tier, model_name
        self.tasks = set()   # the reviews running, held until done

    async def __call__(self, scope, receive, send):
        routes = self.routes() if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == PATH else None
        if routes is None or routes.client_key and not (self.keyed() or _carries(scope, routes.client_key)):
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
        turns = user_turns(chat)
        route, ask, why, t = None, None, "no user message", time.time()
        texts, user_last = turns if turns is not None else ([], False)
        last = texts[-1] if user_last else ""   # review: only a chat whose last message is the user's
        jobs = {"p": self.p_review(last)} if routes.review and last.strip() else {}
        if routes.large:   # mid tool loop too, so an agent's task stays with the model it started on
            jobs |= {f"tier{i}": self.tier(text) for i, text in enumerate(tier_texts(texts))}
        if jobs:
            try: got = dict(zip(jobs, await asyncio.gather(*jobs.values())))   # one model batch
            except Exception as e:   # the passes failed (the server stopping, say): D1A's chat answers, or fails, as without routes
                for job in jobs.values(): job.close()   # those never awaited, if gather failed before starting them
                log.info("route none (passes failed: %s)", type(e).__name__)
                return await self.app(scope, replay, _tagged(send, {"x-d1a-route": "none", "x-d1a-route-error": "error"}))
            tiers = [v for k, v in got.items() if k != "p"]
            tier = max(tiers, key=TIERS.index, default=None)
            if "p" in got and (ask := review_request(last, got["p"], routes.review.is_review_min_p)): route = routes.review
            elif tier is not None and TIERS.index(tier) >= TIERS.index(routes.large.min_tier): route = routes.large
            why = ", ".join(([f"p {got['p']:.2f}"] if "p" in got else []) + ([f"tiers {' '.join(tiers)}"] if tiers else [])
                            + ([ask[0]] if ask else []) + ([] if user_last else ["tool turn"]))
        if route is None:
            log.info("route none (%s)", why)
            return await self.app(scope, replay, _tagged(send, {"x-d1a-route": "none"}))
        stream = chat.get("stream") is True
        name = chat.get("model") or self.model_name()
        if route is routes.review:
            response = await self._review(route, chat, ask, stream)
        elif not route.slots.acquire(blocking=False):   # the large model has max_in_flight chats already: D1A answers, with the note
            log.info("route %s (%s): busy, answered by D1A", route.name, why)
            body = _within_d1a_limits(chat, body)
            return await self.app(scope, replay, _noted(send, {"x-d1a-route": route.name, "x-d1a-route-error": "busy"}, name, NOTE_LARGE_BUSY))
        else:
            try: response = await self._large(route, chat, stream)
            except BaseException: route.slots.release(); raise
            if isinstance(response, StreamingResponse):   # the slot is held until the stream ends or its client goes
                async def close_and_release(close_upstream=response.close_upstream):
                    try: await close_upstream()
                    finally: route.slots.release()
                response.close_upstream = close_and_release
            else:
                route.slots.release()
        if isinstance(response, str):   # the large model failed before its first byte: D1A answers, with the note
            log.info("route %s (%s): unavailable (%s), answered by D1A, %.1f s", route.name, why, response, time.time() - t)
            body = _within_d1a_limits(chat, body)
            return await self.app(scope, replay, _noted(send, {"x-d1a-route": route.name, "x-d1a-route-error": response}, name, NOTE_LARGE_DOWN))
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
        if not route.slots.acquire(blocking=False): return reply(Outcome(BUSY, "busy"))
        try:
            task = asyncio.ensure_future(review(route.target, body))
            self.tasks.add(task)
        except BaseException:
            route.slots.release(); raise
        task.add_done_callback(lambda done: (self.tasks.discard(done), route.slots.release()))   # done, failed or cancelled: a client gone before its stream started included
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


def _within_d1a_limits(chat, body):
    """The request D1A answers in the large model's place: a max_tokens past D1A's own limit (openai_compat.MAX_TOKENS: an
    agent asked the larger model for more) cut to that limit, so D1A answers instead of refusing the request; otherwise
    the bytes as sent."""
    capped = {k: openai_compat.MAX_TOKENS for k in ("max_tokens", "max_completion_tokens")
              if isinstance(chat.get(k), int) and not isinstance(chat.get(k), bool) and chat[k] > openai_compat.MAX_TOKENS}
    return json.dumps(chat | capped).encode() if capped else body


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


def _noted(send, headers, model_name, note):
    """`send` for D1A's answer in the large model's place: `note` starts the reply (a whole completion is rewritten, a
    stream gets a first chunk carrying it), and `headers` are added. An error response passes unchanged."""
    start, parts = None, []

    async def noted(message):
        nonlocal start
        if message["type"] == "http.response.start":
            start = _add_headers(message, headers)
            if start["status"] != 200: await send(start); start = False
            elif dict(start["headers"]).get(b"content-type", b"").startswith(b"text/event-stream"):
                await send(start); start = False
                first = openai_compat.chunk(f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), model_name, {"role": "assistant", "content": note + "\n"})
                await send({"type": "http.response.body", "body": first.encode(), "more_body": True})
            return
        if start is False or message["type"] != "http.response.body":
            return await send(message)
        parts.append(message.get("body", b""))
        if message.get("more_body"): return
        body = b"".join(parts)
        try:
            data = json.loads(body)
            msg = data["choices"][0]["message"]
            msg["content"] = note + ("\n" + msg["content"] if msg.get("content") else "")
            body = json.dumps(data, ensure_ascii=False).encode()
        except (ValueError, KeyError, IndexError, TypeError):
            pass
        await send({**start, "headers": [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"] + [(b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
    return noted
