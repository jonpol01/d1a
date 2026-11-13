"""Client for a D1A (or Kev) System One server: typed questions in, a probability per option out."""
import base64
import json
import os
import urllib.error
import urllib.request

__version__ = "0.1.0"

LOG_HEADER = "x-d1a-decision-log"   # "off": a D1A server answers but keeps the decision out of its decision log


class D1AError(urllib.error.HTTPError):
    """A request the server refused: its status (`code`) and its reason (`detail`, FastAPI's "detail"). An HTTPError, so code
    that caught urllib's error keeps catching it."""

    def __init__(self, error, detail):
        super().__init__(error.url, error.code, f"{error.reason}: {detail}" if detail else error.reason, error.headers, None)
        self.detail = detail


class Client:
    def __init__(self, base_url="http://127.0.0.1:8009", api_key=None, model="d1a-latest", timeout=120):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    def decide(self, state, questions, use_case=None, log=True):
        """POST /v1/systemone. `questions` maps an id to {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}.
        `use_case` (e.g. "routing"), sent only when given: a D1A server reads the answers at the checkpoint's temperature for
        that use case, if it has one. `log=False` keeps the decision out of the server's decision log (tests, demos: no
        outcome will follow)."""
        return self._send("/v1/systemone", self._request(state, questions, use_case), log)

    def decide_media(self, media_type, media, questions, state=None, use_case=None, log=True):
        """POST /v1/systemone/media: the same questions about a photo, a voice clip or a video. `media_type` is "image",
        "audio" or "video"; `media` is the file's bytes or its path. `state` is optional text beside it (a caption, a note).
        A D1A server answers these when it serves an MLX export with its media/ folder."""
        if isinstance(media, (str, os.PathLike)):
            with open(media, "rb") as f:
                media = f.read()
        request = self._request(state, questions, use_case)
        request["media"] = {"type": media_type, "data": base64.b64encode(media).decode("ascii")}
        return self._send("/v1/systemone/media", request, log)

    def feedback(self, decision_id, labels, src=None, group=None):
        """POST /v1/feedback: what actually happened after a decision, for the server to learn from. `decision_id` is the one
        its answer carried (a server started with D1A_FEEDBACK_LOG adds it); `labels` maps a question id to the outcome:
        True or False for a yes/no question, the option key for a choice. `src` names where the outcome came from ("human"
        outranks e.g. "reviewer"), `group` what the decision belongs to (e.g. a pull request)."""
        body = {"decision_id": decision_id, "labels": labels}
        body.update({k: v for k, v in (("src", src), ("group", group)) if v is not None})
        return self._send("/v1/feedback", body)

    def models(self):
        """GET /v1/models: the served model's card (its run, backend, temperature and use-case temperatures, learning
        settings). It answers without loading an idle model."""
        return self._send("/v1/models")

    def chat(self, messages, **options):
        """POST /v1/chat/completions, OpenAI format, answered whole (not streamed); `options` (max_tokens, temperature, tools,
        ...) are sent as given. The reply is choices[0]["message"]. Only a model on an instruction-tuned trunk chats
        (D1A-E2B-it, D1A-E4B-it); for streaming, use an OpenAI client with base_url=<server>/v1."""
        return self._send("/v1/chat/completions", {"messages": messages, **options})

    def _request(self, state, questions, use_case):
        request = {"state": state, "model": self.model, "questions": questions}
        if use_case is not None: request["use_case"] = use_case
        return request

    def _send(self, path, body=None, log=True):
        headers = {"content-type": "application/json"} if body is not None else {}
        if self.api_key: headers["authorization"] = f"Bearer {self.api_key}"
        if not log: headers[LOG_HEADER] = "off"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise D1AError(e, detail_of(e.read())) from None


def detail_of(raw):
    """FastAPI's "detail" off an error body (a string, or a validation error's list, as JSON), else the body's text."""
    text = raw.decode("utf-8", "replace").strip()
    try:
        detail = json.loads(text)["detail"]
    except (ValueError, KeyError, TypeError):
        return text
    return detail if isinstance(detail, str) else json.dumps(detail)
