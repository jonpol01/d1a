# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); the remote model name defaults to d1a-latest; the Jev AI-SDK predictor removed with Kev's playground (docs/removed-tools.md).
"""Predictors: callables record -> {"probabilities": {qid: {key: p}}, "latency_ms", "input_tokens", ...} for d1a.benchmark.

LocalPredictor scores a checkpoint in-process; RemotePredictor any TypeSafe System One-compatible endpoint (scripts/
compare_systemone.py compares servers, Jev included, request by request).
"""
import contextlib
import json
import math
import time
import urllib.request

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from d1a.api import question_keys
from d1a.checkpoint import Checkpoint, LoadOptions
from d1a.data import api_request, materialize
from d1a.device import sync
from d1a.model import ROW_PASS_TOKENS, ContextOverflow, rows_of
from d1a.suite import CONTEXT


LONG_ROW_KERNELS = "efficient"   # the label of a prediction (and its benchmark rows) that took the long-row kernels below


class LocalPredictor:
    """Scores a checkpoint in-process. Evaluation is fp32-exact (no TF32, no fused SDPA kernels on CUDA), with one
    exception: on CUDA with the torch backend, a record whose longest row (state + one question) exceeds
    d1a.model.ROW_PASS_TOKENS runs under SDPA's flash / memory-efficient kernels, because the exact math kernel's L x L
    score matrix does not fit (~200 GB per layer pass at 64k). Such a prediction carries `"kernels": LONG_ROW_KERNELS`,
    d1a.benchmark copies it onto the record's rows and counts them in report.json's `long_rows`; shorter records carry
    nothing, so their rows and reports are unchanged. The efficient path is CUDA-only: on CPU / MPS attention stays eager
    and exact and still materialises L x L. A long record on a hybrid torch backbone also runs its state once, its
    questions continuing from it (d1a.shared_prefix), instead of once per question; the MLX backend's forward already
    runs the state once."""

    def __init__(self, run, device, opts=LoadOptions(), context=CONTEXT):
        """opts.temperature=None scores with the temperature the checkpoint carries; 1.0 scores raw logits. context: the
        max_state / max_branch / max_packed a record must encode within (a suite manifest's `context`; the training
        context by default, the serving limits for external suites frozen without admission)."""
        if opts.temperature is not None and not (math.isfinite(opts.temperature) and opts.temperature > 0):
            raise ValueError("temperature must be finite and positive")
        checkpoint = self.checkpoint = Checkpoint(run)
        self.run = checkpoint.path
        if device == "cuda":
            # evaluation is fp32-exact: TF32 (10-bit mantissa) moves probabilities by ~1e-3, the isolation gate's tolerance
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.tok, self.model = checkpoint.load(device, opts)
        self.temperature = self.model.head.temperature
        self.device = device
        self.context = context

    @torch.no_grad()
    def __call__(self, record):
        enc = self.model.encode(self.tok, materialize(record), max_state=self.context["max_state"], max_branch=self.context["max_branch"], strict=True)
        if len(enc["ids"]) > self.context["max_packed"]:
            raise ContextOverflow(f"packed request exceeds the {self.context['max_packed']}-token limit")
        sync(self.device)
        start = time.perf_counter()
        state, _, rows = rows_of(enc)
        long = len(state) + max(len(r["ids"]) for r in rows) > ROW_PASS_TOKENS
        torch_backend = self.model.backend == "torch"
        efficient = long and torch_backend and self.device == "cuda"
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]) if efficient else contextlib.nullcontext():
            if long and torch_backend and self.model.hybrid: logits = self.model.forward_batch([enc], shared_prefix=True)[0]
            else: logits = self.model.forward(enc)
        ps = [torch.softmax(z, -1).cpu() for z in logits]
        zs = [z.float().cpu() for z in logits]
        sync(self.device)
        keys = {qid: question_keys(q["type"], q.get("criteria")) for qid, q in record["questions"].items()}
        return {"probabilities": {qid: dict(zip(keys[qid], p.tolist())) for qid, p in zip(keys, ps)},
                "logits": {qid: dict(zip(keys[qid], z.tolist())) for qid, z in zip(keys, zs)},
                "inference_temperature": self.temperature,
                "latency_ms": 1000 * (time.perf_counter() - start), "input_tokens": len(enc["ids"]), **({"kernels": LONG_ROW_KERNELS} if efficient else {})}


class RemotePredictor:
    """Score any TypeSafe System One-compatible endpoint (POST <base_url>/v1/systemone) on frozen records. Probabilities are
    taken from the response as returned (renormalised by validate_distribution like every other predictor). Records the
    server-reported model id so the manifest can pin what was scored. `concurrency` is how many requests d1a.benchmark may
    keep in flight at once (each call is independent: one request, its own retries); 1 scores sequentially."""

    def __init__(self, base_url, model="d1a-latest", api_key="local", timeout=120, retries=3, concurrency=1):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.base_url, self.model, self.api_key, self.timeout, self.retries = base_url.rstrip("/"), model, api_key, timeout, retries
        self.concurrency = concurrency
        self.served_model = None

    def __call__(self, record):
        payload = json.dumps({**api_request(record), "model": self.model}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/systemone", data=payload, method="POST",
                                    headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(self.retries):
            try:
                start = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                latency = 1000 * (time.perf_counter() - start)
                break
            except Exception as error:   # 5xx / timeouts: retry with backoff; anything persistent surfaces as a rejected record
                last = error; time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"remote endpoint failed after {self.retries} attempts: {last}")
        self.served_model = body.get("model", self.served_model)
        probs = {}
        for qid, q in record["questions"].items():
            a = body["answers"][qid]
            if q["type"] == "noul": probs[qid] = {"true": float(a["noul"]), "false": 1 - float(a["noul"])}
            else: probs[qid] = {str(k): float(v) for k, v in a["probabilities"].items()}
        return {"probabilities": probs, "latency_ms": latency, "input_tokens": (body.get("usage") or {}).get("input_tokens")}




class RotationAveraged:
    """Test-time order averaging (PLAN.md round 4, item 4.4): score the record under the first `rotations` cyclic
    rotations of every Choice question's options (rotation r of a K-option question is r mod K; Noul and Score keep their
    order, which is part of their meaning) and average the logits per option key. The geometric mean of softmax
    probabilities is the softmax of the mean logits, so the returned probabilities and logits stay consistent; predictors
    that return no logits are averaged in log-probability space. Costs `rotations` forward passes per record."""

    def __init__(self, predictor, rotations):
        if rotations < 2:
            raise ValueError("rotation averaging needs at least 2 rotations")
        self.predictor, self.rotations = predictor, rotations
        self.temperature = getattr(predictor, "temperature", None)
        self.concurrency = getattr(predictor, "concurrency", 1)

    @staticmethod
    def rotated(record, r):
        def rotate(q):
            if q["type"] != "choice": return q
            keys = list(q["criteria"]); k = r % len(keys)
            return {**q, "criteria": {key: q["criteria"][key] for key in keys[k:] + keys[:k]}}
        return {**record, "questions": {qid: rotate(q) for qid, q in record["questions"].items()}}

    def __call__(self, record):
        widest = max((len(q["criteria"]) for q in record["questions"].values() if q["type"] == "choice"), default=1)
        preds = [self.predictor(self.rotated(record, r)) for r in range(min(self.rotations, widest))]
        field = "logits" if all("logits" in p for p in preds) else "probabilities"
        out = {"probabilities": {}, "latency_ms": sum(p["latency_ms"] for p in preds), "rotations": len(preds)}
        if field == "logits":
            out["logits"] = {}; out["inference_temperature"] = preds[0].get("inference_temperature", 1.0)
        for qid in record["questions"]:
            keys = list(preds[0]["probabilities"][qid])
            if field == "logits":
                z = {k: sum(p["logits"][qid][k] for p in preds) / len(preds) for k in keys}
            else:
                z = {k: sum(math.log(max(p["probabilities"][qid][k], 1e-12)) for p in preds) / len(preds) for k in keys}
            top = max(z.values()); e = {k: math.exp(v - top) for k, v in z.items()}; s = sum(e.values())
            out["probabilities"][qid] = {k: v / s for k, v in e.items()}
            if field == "logits": out["logits"][qid] = z
        return out
