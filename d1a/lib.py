"""D1A as a library: load a checkpoint once, then ask typed questions about a document, in-process, with no HTTP server.

    from d1a import D1A
    m = D1A.load("JohnP1/d1a-e2b-mlx-q8")          # a Hub id or a local run / MLX export folder
    m.decide("Order 4471 was charged twice.", {
        "intent": {"type": "choice", "instr": "What does the customer want?", "criteria": {"refund": "a refund", "track": "tracking"}},
        "urgent": {"type": "noul", "instr": "Is this urgent?"},
    })
    # -> {"intent": {"type": "choice", "choice": "refund", "probabilities": {...}, ...}, "urgent": {"type": "noul", "noul": 0.71}}

Questions and answers are the System One API's (d1a.api), so code written against a D1A server moves in-process
unchanged. Loading follows d1a.serve's defaults (bf16 on a GPU, MLX for Gemma 4 on Apple Silicon); D1A_* variables
(LoadOptions.from_env) override them. One call runs one forward pass; the object is not thread-safe.
"""
from dataclasses import replace

import torch

from .api import SystemOneRequest, to_answers, to_record
from .checkpoint import Checkpoint, LoadOptions
from .device import default_device
from .model import SERVE_MAX_BRANCH, SERVE_MAX_STATE


class D1A:
    def __init__(self, checkpoint, tok, model, device):
        self.checkpoint, self.tok, self.model, self.device = checkpoint, tok, model, device

    @classmethod
    def load(cls, run, device="auto"):
        dev = default_device() if device == "auto" else device
        opts = LoadOptions.from_env()
        if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")
        if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)
        if opts.backend is None: opts = replace(opts, backend="auto")
        ck = Checkpoint(run)
        tok, model = ck.load(dev, opts)
        return cls(ck, tok, model, dev)

    def decide(self, state, questions):
        """state: the document; questions: {id: question} as in a /v1/systemone request. -> {id: answer}."""
        rec, meta = to_record(SystemOneRequest(model="d1a-latest", state=state, questions=questions))
        enc = self.model.encode(self.tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH)
        (ps,), _ = self.model.probs_batch([enc], [None], [False])
        return to_answers([p.tolist() for p in ps], meta)
