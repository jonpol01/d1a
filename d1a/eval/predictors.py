"""Predictors for d1a.eval.benchmark: each is a callable record -> {"probabilities": {question id: {option key: p}},
"latency_ms", "input_tokens", ...}.

- LocalPredictor scores a checkpoint in this process (and adds the raw logits and the temperature it served at);
- RemotePredictor scores any System One endpoint (POST /v1/systemone), D1A's own server or another implementation
  (scripts/compare_systemone.py compares two servers request by request);
- RotationAveraged wraps either and averages each choice question over cyclic rotations of its options.
"""
import contextlib
import json
import math
import time
import urllib.request

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from d1a.backends.checkpoint import Checkpoint, LoadOptions
from d1a.backends.device import sync
from d1a.core.api import question_keys
from d1a.core.encoding import ROW_PASS_TOKENS, ContextOverflow, rows_of
from d1a.core.head import QueryTap
from d1a.eval.suite import CONTEXT
from d1a.learning.feedback import encode_vector
from d1a.training.data import api_request, materialize

# What a prediction (and the benchmark rows made from it) carries when it ran on SDPA's flash / memory-efficient kernels.
LONG_ROW_KERNELS = "efficient"
EFFICIENT_KERNELS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def keyed(keys, values):
    """{question id: {option key: value}} from each question's keys and a tensor of its values, in the same order."""
    return {qid: dict(zip(keys[qid], v.tolist())) for qid, v in zip(keys, values)}


class LocalPredictor:
    """A checkpoint scored in this process, fp32-exact: on CUDA, TF32 and the fused attention kernels are switched off.

    One exception keeps very long records possible. On CUDA with the torch backend, a record whose longest row (its state
    plus one question) is longer than d1a.core.encoding.ROW_PASS_TOKENS cannot hold the exact kernel's L x L scores (about
    200 GB per layer at 64k tokens), so it runs on the flash / memory-efficient kernels and its prediction carries
    "kernels": LONG_ROW_KERNELS; d1a.eval.benchmark copies that onto its rows and counts them in the report's long_rows.
    Shorter records carry nothing. On CPU and MPS attention stays exact everywhere. A long record on a hybrid torch backbone
    also runs its state once, its questions continuing from it (d1a.backends.shared_prefix); MLX does that for every record.
    """

    tap = None   # a QueryTap on the head when built with queries=True

    def __init__(self, run, device, opts=LoadOptions(), context=CONTEXT, queries=False):
        """opts.temperature: None serves the checkpoint's own temperature, 1.0 the raw logits. context: the limits
        (max_state, max_branch, max_packed) a record must encode within: a suite manifest's context, the training context
        by default. queries: each prediction also carries every question's pointer-head query q(h_decide) (base64 fp32,
        d1a.learning.feedback.encode_vector), the outcome memory's key (#149)."""
        t = opts.temperature
        if t is not None and not (math.isfinite(t) and t > 0):
            raise ValueError("temperature must be finite and positive")
        self.checkpoint = Checkpoint(run)
        self.run = self.checkpoint.path
        if device == "cuda":   # exact fp32: TF32's 10-bit mantissa moves probabilities by about 1e-3
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.tok, self.model = self.checkpoint.load(device, opts)
        self.temperature = self.model.head.temperature
        self.device, self.context = device, context
        self.tap = QueryTap(self.model.head) if queries else None

    def _encode(self, record):
        limits = self.context
        enc = self.model.encode(self.tok, materialize(record), max_state=limits["max_state"], max_branch=limits["max_branch"], strict=True)
        if len(enc["ids"]) > limits["max_packed"]:
            raise ContextOverflow(f"packed request exceeds the {limits['max_packed']}-token limit")
        return enc

    def _logits(self, enc):
        """-> (one logit tensor per question, whether the efficient kernels ran)."""
        state, _, rows = rows_of(enc)
        long = len(state) + max(len(r["ids"]) for r in rows) > ROW_PASS_TOKENS
        on_torch = self.model.backend == "torch"
        efficient = long and on_torch and self.device == "cuda"
        with sdpa_kernel(EFFICIENT_KERNELS) if efficient else contextlib.nullcontext():
            if long and on_torch and self.model.hybrid:
                return self.model.forward_batch([enc], shared_prefix=True)[0], efficient
            return self.model.forward(enc), efficient

    @torch.no_grad()
    def __call__(self, record):
        enc = self._encode(record)
        sync(self.device)
        start = time.perf_counter()
        if self.tap is not None: self.tap.calls.clear()
        logits, efficient = self._logits(enc)
        probs = [torch.softmax(z, -1).cpu() for z in logits]
        raw = [z.float().cpu() for z in logits]
        sync(self.device)
        keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in record["questions"].items()}
        out = {"probabilities": keyed(keys, probs), "logits": keyed(keys, raw), "inference_temperature": self.temperature,
               "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": len(enc["ids"])}
        if efficient:
            out["kernels"] = LONG_ROW_KERNELS
        if self.tap is not None:
            found = self.tap.match([p.numpy() for p in probs])
            out["queries"] = {qid: None if q is None else encode_vector(q) for qid, q in zip(keys, found)}
        return out


class RemotePredictor:
    """A System One endpoint (POST <base_url>/v1/systemone) scored on frozen records. Its probabilities are taken as
    returned (the benchmark renormalises every predictor's alike), and the model id it reports is kept in `served_model`
    so a report can name what was scored. `concurrency`: how many requests d1a.eval.benchmark may keep in flight (each
    call is one request with its own retries); 1 scores in order."""

    def __init__(self, base_url, model="d1a-latest", api_key="local", timeout=120, retries=3, concurrency=1):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key, self.timeout, self.retries = model, api_key, timeout, retries
        self.concurrency = concurrency
        self.served_model = None

    def _post(self, record):
        """-> (the response body, latency in ms). A failed attempt waits 2**attempt s before the next; after `retries`
        failures the last error is raised (the benchmark counts the record as rejected)."""
        body = json.dumps({**api_request(record), "model": self.model}).encode()
        request = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, method="POST",
                                         headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
        error = None
        for attempt in range(self.retries):
            try:
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    answer = json.loads(response.read())
                return answer, 1000 * (time.perf_counter() - start)
            except Exception as e:   # 5xx, timeouts, a dropped connection
                error = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"remote endpoint failed after {self.retries} attempts: {error}")

    @staticmethod
    def _probabilities(question, answer):
        if question["type"] == "noul":
            return {"true": float(answer["noul"]), "false": 1 - float(answer["noul"])}
        return {str(k): float(v) for k, v in answer["probabilities"].items()}

    def __call__(self, record):
        body, latency = self._post(record)
        self.served_model = body.get("model", self.served_model)
        probs = {qid: self._probabilities(q, body["answers"][qid]) for qid, q in record["questions"].items()}
        return {"probabilities": probs, "latency_ms": latency, "input_tokens": (body.get("usage") or {}).get("input_tokens")}


class RotationAveraged:
    """Order averaging at test time: a record is scored under the first `rotations` cyclic rotations of every choice
    question's options (rotation r of a K-option question starts at option r mod K; yes/no and score questions keep their
    order, which carries meaning), never more rotations than the widest question has options. Each option's logits are
    averaged; the softmax of mean logits is the geometric mean of the probabilities, so probabilities and logits stay
    consistent. A predictor that returns no logits is averaged in log-probability space. One forward pass per rotation."""

    def __init__(self, predictor, rotations):
        if rotations < 2:
            raise ValueError("rotation averaging needs at least 2 rotations")
        self.predictor, self.rotations = predictor, rotations
        self.temperature = getattr(predictor, "temperature", None)
        self.concurrency = getattr(predictor, "concurrency", 1)

    @staticmethod
    def rotated(record, r):
        """The record with every choice question's options rotated by r (mod its option count)."""
        def turn(q):
            if q["type"] != "choice":
                return q
            keys = list(q["criteria"])
            k = r % len(keys)
            return {**q, "criteria": {key: q["criteria"][key] for key in keys[k:] + keys[:k]}}
        return {**record, "questions": {qid: turn(q) for qid, q in record["questions"].items()}}

    @staticmethod
    def _softmax(z):
        top = max(z.values())
        e = {k: math.exp(v - top) for k, v in z.items()}
        total = sum(e.values())
        return {k: v / total for k, v in e.items()}

    def __call__(self, record):
        widest = max((len(q["criteria"]) for q in record["questions"].values() if q["type"] == "choice"), default=1)
        preds = [self.predictor(self.rotated(record, r)) for r in range(min(self.rotations, widest))]
        with_logits = all("logits" in p for p in preds)
        out = {"probabilities": {}, "latency_ms": sum(p["latency_ms"] for p in preds), "rotations": len(preds)}
        if with_logits:
            out["logits"] = {}
            out["inference_temperature"] = preds[0].get("inference_temperature", 1.0)
        for qid in record["questions"]:
            keys = list(preds[0]["probabilities"][qid])
            if with_logits:
                z = {k: sum(p["logits"][qid][k] for p in preds) / len(preds) for k in keys}
            else:
                z = {k: sum(math.log(max(p["probabilities"][qid][k], 1e-12)) for p in preds) / len(preds) for k in keys}
            out["probabilities"][qid] = self._softmax(z)
            if with_logits:
                out["logits"][qid] = z
        return out
