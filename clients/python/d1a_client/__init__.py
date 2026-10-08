"""Client for a D1A (or Kev) System One server: typed questions in, a probability per option out."""
import json
import urllib.request

__version__ = "0.0.1"


class Client:
    def __init__(self, base_url="http://127.0.0.1:8009", api_key=None, model="d1a-latest", timeout=120):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    def decide(self, state, questions, use_case=None):
        """POST /v1/systemone. `questions` maps an id to {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}.
        `use_case` (e.g. "routing"), sent only when given: a D1A server reads the answers at the checkpoint's temperature for
        that use case, if it has one."""
        request = {"state": state, "model": self.model, "questions": questions}
        if use_case is not None: request["use_case"] = use_case
        body = json.dumps(request).encode()
        headers = {"content-type": "application/json"}
        if self.api_key: headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)
