# Modified from Kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache-2.0.
# Changes for D1A Copyright 2026 John Soliva: package renamed kev -> d1a (imports, module paths, KEV_* -> D1A_* environment variables); serves d1a-latest and keeps kev-latest and jev-latest as compatibility names; model cards name D1A; serves MLX export folders; --device with a GPU usability probe; a startup self-check of the readout and a warning when serving an uncalibrated checkpoint; recent batch latency in /v1/models; a 75 s keep-alive for proxies in front of it; --idle-unload; photos and voice clips through the same model (POST /v1/systemone/media); prefix cache keyed by state ids only (option isolation removed); self-learning hooks (decision log, POST /v1/feedback, outcome calibrator); GET /metrics.
"""FastAPI sidecar for the playground: loads one checkpoint, exposes prefill-only decisions.

Run: uv run --extra serve python -m d1a.serve --run runs/d1a --port 8008

TypeSafe-compatible: POST /v1/systemone, GET /v1/models, the `x-typesafe-request-id` response header, and bearer auth
when D1A_API_KEY is set (unset = open server, the local default). Demo extras: POST /v1/systemone/permute (one Choice
under several option orders) and POST /v1/systemone/separate (each question in its own pass, for the packed-vs-separate
comparison), self-learning (D1A_FEEDBACK_LOG=<path>: every answer gets a decision_id and is logged, POST /v1/feedback
{decision_id, labels} records what actually happened; D1A_OUTCOME_CALIBRATOR=<file>: yes/no and choice answers recalibrated by d1a.feedback's
calibrator, re-read when the file changes), POST /v1/systemone/media (a /v1/systemone request plus a photo, voice clip or video, d1a.media.MediaRequest,
answered by the same model: an MLX export with its media/ folder, scripts/export_mlx.py --media).

--idle-unload N drops the model (and the media encoders) after N seconds without a request and loads it again on the
next one; GET /v1/models answers either way without loading it, and so does GET /metrics (Prometheus text: loaded, requests,
batches, queue depth, batch latency, prefix cache, memory). D1A_PREFIX_CACHE / D1A_PREFIX_MIN_TOKENS / D1A_PREFIX_MAX_TOKENS size the state-prefix cache; D1A_DATE_FACTS=1 opts into the
date preprocessing (api.with_date_facts). Backend and precision follow LoadOptions (D1A_BACKEND, D1A_DTYPE, ...): on Apple
Silicon the hybrid Qwen3.5 checkpoints run on MLX by default, elsewhere on torch in bf16.
"""
import argparse, asyncio, atexit, hmac, math, os, queue, random, subprocess, sys, threading, time, traceback, uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from collections import deque
from concurrent.futures import Future
import torch
from dataclasses import dataclass, field, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from .api import SystemOneRequest, to_record, to_answers, output_tokens, with_date_facts
from .checkpoint import EXPORT_CONFIG, Checkpoint, LoadOptions, fused_available, is_hub_id
from .device import allocated_bytes, default_device, empty_cache, out_of_memory, sync
from .feedback import FeedbackLog, OutcomeCalibrator
from .media import MEDIA_DIR, MediaEncoder, MediaRequest, OnDemand, with_media
from .model import SERVE_MAX_BRANCH, SERVE_MAX_STATE, layout

PREFIX_CACHE_SIZE = int(os.environ.get("D1A_PREFIX_CACHE", "4"))          # states kept (KV + DeltaNet states; attention-only backbones also the state's hidden states); 0 disables
PREFIX_MIN_TOKENS = os.environ.get("D1A_PREFIX_MIN_TOKENS")               # states shorter than this are not cached; default = the model's prefix_min_tokens (0 for hybrid backbones and MLX, 384 for attention-only torch models)
PREFIX_MAX_TOKENS = int(os.environ.get("D1A_PREFIX_MAX_TOKENS", "65536"))  # state tokens the cache holds in all (least recently used evicted first); a longer state is not cached.
                                                                         # One 64k state (Kev-27B: ~1.3 GB of keys, values and DeltaNet states), or four 16k ones, not four 64k ones
DATE_FACTS = os.environ.get("D1A_DATE_FACTS", "0") == "1"
FEEDBACK_LOG = os.environ.get("D1A_FEEDBACK_LOG")                         # set = log every decision (d1a.feedback) and accept outcomes at POST /v1/feedback
OUTCOME_CALIBRATOR = os.environ.get("D1A_OUTCOME_CALIBRATOR")             # set = apply this d1a.feedback calibrator to yes/no and choice answers, reloaded when the file changes
API_KEY = os.environ.get("D1A_API_KEY")                                  # unset = open server; set = require Authorization: Bearer <key>, as the TypeSafe clients always send
MAX_BATCH = 64                                                           # requests the model thread takes at once (d1a.cuda_graphs splits them to fit its buffers)
KEEP_ALIVE_S = 75                                                        # idle keep-alive; above Node's pooled-socket reuse window, so a proxy (the playground's Next.js rewrite) never reuses a socket uvicorn just closed (ECONNRESET, #42); uvicorn's default is 5
MODEL_NAMES = ("d1a-latest", "kev-latest", "jev-latest")                 # all name this checkpoint (any name is served): kev-latest for clients written against Kev, jev-latest is the TypeSafe SDK default model, so an unconfigured client works


@dataclass
class PrefixCache:
    """State prefixes kept across requests, least recently used first: state token ids -> prefix.
    At most `size` states and `max_tokens` state tokens in all; states shorter than min_tokens or longer than max_tokens
    are not cached. A batch keeps (copies) only the new states that will still be here after it, its last distinct ones
    within both bounds: the rest would be evicted by the batch itself."""
    size: int
    min_tokens: int
    max_tokens: int = PREFIX_MAX_TOKENS
    entries: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    oom_retries: int = 0   # batches that ran out of device memory with states cached, dropped them and ran again (Server._run)

    def plan(self, encs):
        """-> (key per request, None when its state is not cached; its cached prefix or None; whether to keep a new one)."""
        lengths = [enc["seg"].count(0) for enc in encs]
        keys = [tuple(enc["ids"][:n]) if self.size and self.min_tokens <= n <= self.max_tokens and enc.get("media") is None else None   # media: same placeholder ids for different photos
                for enc, n in zip(encs, lengths)]
        survivors, tokens = set(), 0
        for key in dict.fromkeys(k for k in reversed(keys) if k is not None):   # most recent first, as store() keeps them
            if len(survivors) == self.size or tokens + len(key) > self.max_tokens: break
            survivors.add(key); tokens += len(key)
        return keys, [self.entries.get(k) if k is not None else None for k in keys], [k in survivors for k in keys]

    def store(self, keys, cached, prefixes):
        """Record hits and misses, and (re)insert the batch's prefixes in order: most recently used last."""
        for key, old, new in zip(keys, cached, prefixes):
            if key is None: continue
            self.hits += old is not None; self.misses += old is None
            if new is None: continue
            self.entries.pop(key, None); self.entries[key] = new
            while len(self.entries) > self.size or sum(len(k) for k in self.entries) > self.max_tokens: self.entries.pop(next(iter(self.entries)))

    def clear(self):
        self.entries.clear()


@dataclass
class Server:
    """The loaded checkpoint, the state-prefix cache (PrefixCache), and the one model thread that runs every forward pass.

    Request threads encode their record and queue it; the model thread takes everything queued when it becomes free and
    runs it as one batch (model.probs_batch: with CUDA graphs, shared state and row passes; otherwise one request at a
    time), then answers each request. It also captures pending CUDA graphs when the graphs say so (capture_due). `lock` is
    held around each batch and capture: hold it to use the model directly."""
    checkpoint: Checkpoint
    tok: object
    model: object
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    batches: int = 0
    batched_requests: int = 0
    release_date: str = field(default="")   # for the TypeSafe model card; resolved once (may ask the Hub)
    batch_ms: deque = field(default_factory=lambda: deque(maxlen=512))   # model time of the most recent batches (latency_summary)

    def __post_init__(self):
        self.release_date = self.release_date or self.checkpoint.release_date()
        self.prefix_cache = PrefixCache(PREFIX_CACHE_SIZE, int(PREFIX_MIN_TOKENS) if PREFIX_MIN_TOKENS else self.model.prefix_min_tokens)
        self.queue, self.stopping = queue.Queue(), threading.Event()
        # the model thread gives up the GIL at every CUDA sync and waits to get it back while the event loop parses and
        # answers requests; at Python's default 5 ms switch interval those waits stretched a batch's model time ~2x.
        # Process-wide, so close() puts it back.
        self.switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)
        self.thread = threading.Thread(target=self._work, name="d1a-model", daemon=True)
        self.thread.start()
        atexit.register(self.close)   # a daemon thread killed inside a CUDA call at interpreter exit aborts the process
        self.media, self.media_lock = None, threading.Lock()

    def close(self):
        """Stop the model thread after its current batch; requests still queued fail, and so do later ones (submit)."""
        if self.stopping.is_set(): return
        atexit.unregister(self.close)   # the registration holds the server, and with it the model, alive
        self.stopping.set(); self.thread.join()
        self.media = None
        while not self.queue.empty():
            self.queue.get_nowait()[1].set_exception(RuntimeError("the server stopped")); self.queue.task_done()
        sys.setswitchinterval(self.switch_interval)

    def submit(self, rec, media=None):
        """Queue one record for the model thread. -> a Future of (probabilities, stats). The state prefix (tokens up to the
        first question) is cached across requests, so a repeated state only pays for its question rows. latency_ms is the
        model time of the batch the request ran in (not its wait in the queue). media: MediaEncoder.encode's output, put
        right after <state>."""
        if self.stopping.is_set(): raise HTTPException(503, "the server is stopping")
        try:
            enc = self.model.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
            if media is not None: enc = with_media(enc, len(layout(self.tok)[0]) + 1, *media)
        except ValueError as e: raise HTTPException(422, str(e))
        done = Future()
        self.queue.put((enc, done))
        return done

    def probs(self, rec):
        return self.submit(rec).result()

    def _work(self):
        graphs = getattr(self.model, "graphs", None)
        while not self.stopping.is_set():
            try: batch = [self.queue.get(timeout=0.05)]
            except queue.Empty:
                if graphs is not None and graphs.capture_due(idle=True):
                    with self.lock: graphs.capture_pending(limit=1)
                continue
            while len(batch) < MAX_BATCH:
                try: batch.append(self.queue.get_nowait())
                except queue.Empty: break
            try:
                with self.lock: results = self._run([enc for enc, _ in batch])
            except Exception as e:   # every request of the batch gets the error; the thread lives on
                traceback.clear_frames(e.__traceback__)   # the batch's futures keep the exception, and its frames held the failed pass's tensors (142 MiB, a 3,578-token state on Kev-0.8B); the traceback keeps its lines
                results = [e] * len(batch)
            for (_, done), result in zip(batch, results):
                (done.set_exception if isinstance(result, Exception) else done.set_result)(result)
                self.queue.task_done()
            if graphs is not None and graphs.capture_due(idle=False):
                with self.lock: graphs.capture_pending(limit=1)

    def _run(self, encs):
        """One batch through model.probs_batch, with the prefix cache. -> per request (probs, stats). A pass out of device
        memory while states are cached drops the cache and runs once more: the cache only saves time, and kept resident
        it failed every later batch of that size (#75, @vtxyer)."""
        sync(self.device); t = time.time()
        for retry in (False, True):
            keys, cached, keep = self.prefix_cache.plan(encs)
            try: ps, prefixes = self.model.probs_batch(encs, cached, keep); break
            except Exception as e:
                if retry or not self.prefix_cache.entries or not out_of_memory(e): raise
            cached = None; self.prefix_cache.clear(); self.prefix_cache.oom_retries += 1   # after the except: its traceback holds the failed pass's tensors
            empty_cache(self.device)
        sync(self.device); dt = round((time.time() - t) * 1000, 1)
        self.batch_ms.append(dt)
        self.prefix_cache.store(keys, cached, prefixes)
        self.batches += 1; self.batched_requests += len(encs)
        return [([q.tolist() for q in p], {"tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0), "latency_ms": dt, "prefix_cache_hit": c is not None})
                for enc, p, c in zip(encs, ps, cached)]

    def wait_idle(self):
        """Block until every submitted request is answered and no CUDA graph waits to be captured (benchmarks, warm-up)."""
        self.queue.join()
        graphs = getattr(self.model, "graphs", None)
        while graphs is not None and graphs.capture_due(idle=True): time.sleep(0.01)
        with self.lock: pass                                   # a capture in progress finishes

    def answer(self, req):
        """The /v1/systemone response body for one request."""
        rec, meta = to_record(prepare(req))
        return self._body(req, meta, *self.probs(rec))

    async def answer_async(self, req):
        """answer() for the event loop: a request waiting on the model thread holds no worker thread, so a container takes
        as many concurrent requests as its batches can absorb (FastAPI runs sync endpoints on a 40-thread pool)."""
        rec, meta = to_record(prepare(req))
        return self._body(req, meta, *await asyncio.wrap_future(self.submit(rec)))

    def media_encoder(self):
        """Gemma 4's vision and audio encoders for this model, loaded on the first photo or voice request: the export's
        media/ folder, fetched from the Hub then if the run is a Hub id."""
        if self.model.backend != "mlx":
            raise HTTPException(501, "photos and voice run on an MLX export here; for a torch checkpoint use python -m d1a.media")
        with self.media_lock:
            if self.media is None:
                folder = Path(self.checkpoint.path) / MEDIA_DIR
                if not folder.is_dir() and is_hub_id(self.checkpoint.requested):
                    from huggingface_hub import snapshot_download
                    repo, _, rev = self.checkpoint.requested.partition("@")
                    folder = Path(snapshot_download(repo, revision=rev or None, allow_patterns=[f"{MEDIA_DIR}/*"])) / MEDIA_DIR
                if not folder.is_dir():
                    raise HTTPException(501, f"{self.checkpoint.requested} has no {MEDIA_DIR}/ folder: export it with scripts/export_mlx.py --media")
                self.media = MediaEncoder(str(folder), self.device)
            return self.media

    async def answer_media_async(self, req):
        """answer_async() for a request with a photo or a voice clip: the encoders run on this request's thread, the
        language model on the model thread as for any request."""
        rec, meta = to_record(prepare(req))
        try: media = await asyncio.to_thread(lambda: self.media_encoder().encode(req.media))
        except ValueError as e: raise HTTPException(422, str(e))
        return self._body(req, meta, *await asyncio.wrap_future(self.submit(rec, media)))

    def _body(self, req, meta, ps, m):
        answers, did = LEARNING.decide(req, to_answers(ps, meta), self.checkpoint.requested)
        body = {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": output_tokens(self.tok, answers)}, "latency_ms": m["latency_ms"]}
        if did is not None: body["decision_id"] = did
        return body


class Learning:
    """Self-learning hooks: the decision log and the outcome calibrator. The calibrator is re-read when its file changes, so
    `python -m d1a.feedback calibrate <log> --out <file>` takes effect on the next request without a restart."""

    def __init__(self, log_path=None, calibrator_path=None):
        self.log = FeedbackLog(log_path) if log_path else None
        self.calibrator_path, self._mtime, self._cal, self._lock = calibrator_path, None, None, threading.Lock()

    def calibrator(self):
        if not self.calibrator_path: return None
        try: mtime = os.path.getmtime(self.calibrator_path)
        except OSError: return self._cal
        with self._lock:
            if mtime != self._mtime: self._cal, self._mtime = OutcomeCalibrator.load(self.calibrator_path), mtime
            return self._cal

    def decide(self, req, answers, run):
        """Answers as served (recalibrated when a calibrator is set), and the decision id when logging is on. The log keeps the
        model's own answers, which the next `d1a.feedback calibrate` must fit; what was served is kept beside them in meta."""
        cal = self.calibrator()
        served = cal.apply(answers) if cal is not None else answers
        if self.log is None: return served, None
        questions = {qid: q.model_dump(exclude_none=True) for qid, q in req.questions.items()}
        with self._lock: did = self.log.decision(req.state, questions, answers, run=run, meta={"served": served} if served is not answers else None)
        return served, did

    def outcome(self, did, labels, meta=None):
        with self._lock: self.log.outcome(did, labels, meta)

    def card(self):
        cal = self.calibrator()
        return {"log": self.log is not None, "outcome_calibrator": self.calibrator_path, "calibrated_questions": sorted(cal.params) if cal else []}


LEARNING = Learning(FEEDBACK_LOG, OUTCOME_CALIBRATOR)


def latency_summary(ms):
    """Model time of recent batches: count, p50, p95 and max in ms (nearest rank), or None before the first batch."""
    if not ms: return None
    xs = sorted(ms); rank = lambda q: xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]
    return {"recent": len(xs), "p50_ms": rank(0.5), "p95_ms": rank(0.95), "max_ms": xs[-1]}


def prepare(req):
    """Opt-in preprocessing applied to every request before the model sees it."""
    return req.model_copy(update={"state": with_date_facts(req.state)}) if DATE_FACTS else req


app = FastAPI(title="d1a")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["x-typesafe-request-id", "server-timing"])


PROTECTED = ("/v1", "/metrics")   # paths that need the bearer key when D1A_API_KEY is set: the API and the operational metrics


@app.middleware("http")
async def typesafe(request, call_next):
    """Bearer auth (when API_KEY is set) and the request id every TypeSafe client reads off the response."""
    started = time.perf_counter()
    if API_KEY and request.url.path.startswith(PROTECTED) and not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {API_KEY}"):
        resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <D1A_API_KEY>"}, 401, {"www-authenticate": "Bearer"})
    else:
        resp = await call_next(request)
    resp.headers["x-typesafe-request-id"] = request.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
    resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.1f}"   # time inside this process, for telling it from the network
    return resp


@contextmanager
def server():
    """The Server for one request, loaded if it was unloaded (--idle-unload); it is not unloaded while a request holds it."""
    with app.state.models.use() as s:
        yield s


@asynccontextmanager
async def server_async():
    """server() for the event loop: a load (seconds) runs on a worker thread, not on the loop."""
    s = await asyncio.to_thread(app.state.models.acquire)
    try:
        yield s
    finally:
        app.state.models.release()


@app.post("/v1/systemone")
async def systemone(req: SystemOneRequest):
    """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
    async with server_async() as s:
        return await s.answer_async(req)


@app.post("/v1/systemone/media")
async def systemone_media(req: MediaRequest):
    """/v1/systemone with a photo, a voice clip or a video (d1a.media.MediaRequest), answered by the same model."""
    async with server_async() as s:
        return await s.answer_media_async(req)


class Feedback(BaseModel):
    decision_id: str
    labels: dict[str, bool | str | int]   # {question id: what actually happened}: bool for yes/no, the option key for choice
    src: str | None = None                # where the outcome came from (d1a.feedback.PREFER: "human" beats e.g. "reviewer")


@app.post("/v1/feedback")
def feedback(f: Feedback):
    """Report the real outcome of an earlier decision (its decision_id from /v1/systemone), for d1a.feedback to learn from."""
    if LEARNING.log is None:
        raise HTTPException(404, "feedback is off: start d1a.serve with D1A_FEEDBACK_LOG=<path>")
    LEARNING.outcome(f.decision_id, f.labels, {"src": f.src} if f.src else None)
    return {"ok": True}


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)   # each order is a forward pass; 0 divided by nothing, unbounded counts ran forever (#30)
    seed: int = 0


@app.post("/v1/systemone/permute")
def systemone_permute(r: PermuteSystemOne):
    """Re-run one Choice question under n_perm option orders. Returns per-order probabilities keyed by option name."""
    q = r.request.questions.get(r.question)
    if q is None or q.type != "choice": raise HTTPException(422, "question must be an existing choice question")
    rng = random.Random(r.seed); keys = list(q.criteria); runs = []
    with server() as s:
        for i in range(r.n_perm):
            order = list(keys)
            if i > 0: rng.shuffle(order)
            one = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
            resp = s.answer(one); a = resp["answers"][r.question]
            runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": resp["latency_ms"]})
    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
    with server() as s:
        parts = [s.answer(req.model_copy(update={"questions": {qid: q}})) for qid, q in req.questions.items()]
        answers = {qid: a for p in parts for qid, a in p["answers"].items()}
        out_tokens = output_tokens(s.tok, answers)
    return {"model": req.model, "answers": answers,
            "usage": {"input_tokens": sum(p["usage"]["input_tokens"] for p in parts), "output_tokens": out_tokens},
            "latency_ms": round(sum(p["latency_ms"] for p in parts), 1)}


@app.get("/v1/models")
def models():
    """One TypeSafe model card (name, description, release_date) per accepted model name, plus the D1A serving details
    a client may ignore: the run, the base, the device, the backend and precision, the temperature, whether the model is
    in memory (--idle-unload), and while it is, prefix-cache and batch stats. Answers without loading the model, so a
    client polling it does not keep an idle model in memory."""
    od = app.state.models
    s, card = od.model, dict(app.state.card)
    card["loaded"], card["idle_unload_s"], card["learning"] = s is not None, od.idle_s, LEARNING.card()
    if s is not None:
        card.update({"cuda_graphs": graphs.stats() if (graphs := getattr(s.model, "graphs", None)) else None,
                     "prefix_cache": {"size": s.prefix_cache.size, "min_state_tokens": s.prefix_cache.min_tokens, "max_tokens": s.prefix_cache.max_tokens, "hits": s.prefix_cache.hits,
                                      "misses": s.prefix_cache.misses, "cached_states": len(s.prefix_cache.entries), "oom_retries": s.prefix_cache.oom_retries},
                     "batches": {"count": s.batches, "requests": s.batched_requests, "queued": s.queue.qsize(), "latency": latency_summary(s.batch_ms)}})
    return {"models": [{"name": name, **card} for name in MODEL_NAMES]}


def peak_rss_bytes():
    """The process's peak resident memory (ru_maxrss is bytes on macOS, kilobytes on Linux)."""
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if sys.platform == "darwin" else v * 1024


def device_bytes(s):
    """Memory the loaded model holds on its device: MLX active memory, or torch's allocator (d1a.device.allocated_bytes)."""
    if s.model.backend == "mlx":
        import mlx.core as mx
        return (mx.get_active_memory() if hasattr(mx, "get_active_memory") else mx.metal.get_active_memory())
    return allocated_bytes(s.device)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus text format: whether the model is loaded, requests and batches served, the queue, recent batch latency,
    the prefix cache and memory. Answers without loading the model, like /v1/models."""
    s, lines = app.state.models.model, []
    def m(name, value, help_, kind="gauge", labels=""):
        if value is None: return
        if not any(l.startswith(f"# HELP {name} ") for l in lines): lines.extend([f"# HELP {name} {help_}", f"# TYPE {name} {kind}"])
        lines.append(f"{name}{labels} {value}")
    m("d1a_loaded", int(s is not None), "1 when the model is in memory (--idle-unload frees it)")
    m("d1a_process_peak_rss_bytes", peak_rss_bytes(), "peak resident memory of the server process")
    if s is not None:
        m("d1a_requests_total", s.batched_requests, "requests answered", "counter")
        m("d1a_batches_total", s.batches, "batches (forward passes) run", "counter")
        m("d1a_queue_depth", s.queue.qsize(), "requests waiting for the model")
        lat = latency_summary(s.batch_ms)
        if lat:
            for q, key in (("0.5", "p50_ms"), ("0.95", "p95_ms"), ("1", "max_ms")):
                m("d1a_batch_latency_ms", lat[key], f"model time of the last {lat['recent']} batches", labels=f'{{quantile="{q}"}}')
        pc = s.prefix_cache
        m("d1a_prefix_cache_hits_total", pc.hits, "state-prefix cache hits", "counter")
        m("d1a_prefix_cache_misses_total", pc.misses, "state-prefix cache misses", "counter")
        m("d1a_prefix_cache_states", len(pc.entries), "states held in the prefix cache")
        try: m("d1a_device_memory_bytes", device_bytes(s), "memory the model holds on its device")
        except Exception: pass
    return "\n".join(lines) + "\n"


def card(s):
    """The parts of the model card that stay the same while the model is unloaded, from a loaded Server."""
    ck, meta, T = s.checkpoint, s.checkpoint.meta, s.model.head.temperature
    return {"description": f"D1A pointer head on {meta.base}, serving {ck.requested} at temperature {T:.2f}", "release_date": s.release_date,
            "run": ck.requested, "base": meta.base, "lora": meta.lora, "device": s.device, "backend": s.model.backend, "dtype": s.model.dtype,
            "temperature": T, "calibrated": T != 1.0}


def usable(device):
    """Whether a real kernel runs on `device`, tried in a child process: a GPU whose driver or torch wheel lacks the local
    architecture can pass is_available() and then kill the process on its first kernel, which no try/except catches."""
    if device == "cpu": return True
    code = f"import torch; torch.ones(8, device={device!r}).mul(2).sum().item()"
    try: return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=300).returncode == 0
    except subprocess.TimeoutExpired: return False


SELF_CHECK = SystemOneRequest(model="d1a-latest", state="Startup self-check: the order shipped late.",
                              questions={"c": {"type": "choice", "instr": "Was the order late?", "criteria": {"yes": "Yes", "no": "No"}},
                                         "n": {"type": "noul", "instr": "Did the order ship late?"}})


def self_check(probs):
    """Refuse to serve a model that would answer wrongly without an error: one request through the real encode and
    readout path must return finite probabilities that sum to 1 (a broken export or precision fails here, not on a
    user's request). `probs(rec) -> (probabilities per question, stats)`. Raises SystemExit naming the failure."""
    ps, _ = probs(to_record(SELF_CHECK)[0])
    for q, p in zip(SELF_CHECK.questions, ps):
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p) - 1) > 1e-3:
            raise SystemExit(f"self-check failed: question {q!r} got probabilities {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/d1a")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind; 0.0.0.0 to serve beyond this machine (a container, a VM behind a proxy)")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto", help="accelerator; auto = cuda, then mps, then cpu, skipping one that cannot run a kernel")
    ap.add_argument("--idle-unload", type=int, default=0, help="seconds without a request before the model is dropped from memory and loaded again on the next one; 0 keeps it loaded")
    a = ap.parse_args()
    run = a.run if is_hub_id(a.run) or os.path.exists(f"{a.run}/head.pt") or os.path.exists(f"{a.run}/{EXPORT_CONFIG}") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    dev = default_device() if a.device == "auto" else a.device
    if not usable(dev):
        if a.device != "auto": raise SystemExit(f"--device {dev}: a test kernel failed on this machine")
        print(f"{dev} is available but a test kernel failed on it; serving on cpu (pass --device to insist)"); dev = "cpu"
    opts = LoadOptions.from_env()
    if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")   # serving default on Apple GPUs (parity measured)
    if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)   # serving default: 2-4.5x faster than fp32 on an L4, same answers (LoadOptions.dtype); D1A_DTYPE=fp32 for the exact path
    if dev == "cuda" and opts.cuda_graphs is None: opts = replace(opts, cuda_graphs=True)   # serving default: a pass is ~2,000 kernel launches, so replaying graphs cuts warm latency several-fold (d1a.cuda_graphs); D1A_CUDA_GRAPHS=0 to decline
    fused_default = dev == "cuda" and opts.fused is None
    if fused_default: opts = replace(opts, fused=fused_available())   # serving default: fused Qwen3.5 kernels, ~1/3 less GPU time per batch (d1a.fused_qwen35), when fla is installed; D1A_FUSED=0 to decline, D1A_FUSED=1 to insist
    if opts.backend is None: opts = replace(opts, backend="auto")   # serving default: MLX for the hybrid Qwen3.5 checkpoints on Apple Silicon (LoadOptions.backend); D1A_BACKEND=torch to decline
    ck = Checkpoint(run)

    def load():
        t0 = time.time()
        tok, model = ck.load(dev, opts)
        s = Server(ck, tok, model, dev)
        try: self_check(s.probs)
        except BaseException: s.close(); raise
        print(f"loaded {ck.requested} in {time.time() - t0:.1f} s", flush=True)
        return s

    app.state.models = OnDemand(load, a.idle_unload, dev)
    with app.state.models.use() as s:   # loaded now, so a bad checkpoint fails at startup, not on a user's request
        model = s.model; app.state.card = card(s)
    app.state.models.run_reaper()
    if fused_default and not opts.fused and model.hybrid: print("fused Qwen3.5 kernels off: install the flash-linear-attention version d1a/fused_qwen35.py pins (FLA_VERSION) to turn them on")
    if model.head.temperature == 1.0:   # 1.0 = never calibrated: d1a.train and the study harness leave head.pt at 1.0 on purpose
        print("!!! serving an uncalibrated checkpoint (temperature 1.0): probabilities will be overconfident. Fit one with "
              "python -m d1a.calibrate before publishing; thresholds on probabilities assume it.", flush=True)
    print(f"serving {ck.requested} ({ck.path}) on {dev} via {model.backend} ({model.dtype}) {a.host}:{a.port}"   # /v1/models reports the run as given, not the resolved cache path
          + (f"; unloads after {a.idle_unload} s idle" if a.idle_unload > 0 else ""), flush=True)
    del model, s
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, timeout_keep_alive=KEEP_ALIVE_S)


if __name__ == "__main__":
    main()
