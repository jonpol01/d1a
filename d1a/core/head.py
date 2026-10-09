"""The pointer head, the backend-independent core of D1A's readout (#60): the torch module whose weights every backend
(d1a.backends.torch, d1a.backends.mlx) applies to the hidden states at <decide> and at each option's </opt>."""
import math

import numpy as np
import torch
from torch import nn


class PointerHead(nn.Module):
    """Scores a question's options: a scaled dot product of the projected <decide> state (q) with each projected </opt>
    state (k). In eval mode the logits are divided by `temperature` (1.0: raw), the calibration a checkpoint carries
    (d1a.training.calibrate); training always sees 1, so a fitted value stays meaningful, and the argmax never changes.

    `use_case_temperatures` ({use case: temperature}, d1a.training.calibrate --use-case, #209) are the temperatures a request
    may select by its use case (d1a.core.api.SystemOneRequest.use_case); the backends read each request's probabilities at
    temperature_for(its use case), so requests at different temperatures can share a batch."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)   # dp: the pointer dimension
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0
        self.use_case_temperatures = {}

    def temperature_for(self, use_case=None):
        """The temperature a request of `use_case` is read at: its entry in use_case_temperatures, else the checkpoint's
        `temperature` (no use case, or one this checkpoint has no entry for: never an error, so clients can name a use
        case before a checkpoint carries it)."""
        return self.use_case_temperatures.get(use_case, self.temperature) if use_case is not None else self.temperature

    def _tempered(self, z, temperature=None):
        t = self.temperature if temperature is None else temperature
        return z if self.training or (not torch.is_tensor(t) and t == 1.0) else z / t

    def forward(self, h_decide, h_opts, temperature=None):   # [d], [K, d] -> logits [K]; temperature None: the head's own
        return self._tempered((self.k(h_opts) @ self.q(h_decide)) * self.scale, temperature)

    def many(self, h_decide, h_opts, owner, temperature=None):   # [Q, d], [sum K, d], each option's question [sum K] -> logits [sum K]
        """forward() for many questions in one pass (serving batches). temperature: None (the head's own), one value for
        the whole batch, or one per option ([sum K]) when the batch's requests are read at different temperatures."""
        return self._tempered((self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale, temperature)


class QueryTap:
    """Installed on a PointerHead, records each call's query q(h_decide) beside the probabilities its logits give, so a
    question's query can be found by its answer (match): the backends call the head in no fixed order (d1a.backends.mlx
    once per question, d1a.backends.torch's batches through many). The head's results are untouched; a question whose
    answer matches no call, or calls with different queries, gets None. The outcome memory's keys (#149, #158):
    d1a.eval.benchmark --queries stores them with each row."""

    def __init__(self, head):
        self.head, self.calls, self._forward, self._many = head, [], head.forward, head.many
        head.forward, head.many = self.forward, self.many

    def _record(self, q, z):
        self.calls.append((q.detach().float().cpu().numpy(), torch.softmax(z.detach().float(), -1).cpu().numpy()))

    def forward(self, h_decide, h_opts, temperature=None):
        z = self._forward(h_decide, h_opts, temperature)
        self._record(self.head.q(h_decide), z)
        return z

    def many(self, h_decide, h_opts, owner, temperature=None):
        z = self._many(h_decide, h_opts, owner, temperature)
        q = self.head.q(h_decide)
        for j in range(q.shape[0]): self._record(q[j], z[owner == j])
        return z

    def match(self, probs, tol=1e-5):
        """One query (or None) per question's probabilities, among the calls recorded since `calls` was last cleared."""
        out = []
        for p in probs:
            p = np.asarray(p, np.float32)
            hits = [q for q, c in self.calls if c.shape == p.shape and np.abs(c - p).max() <= tol]
            out.append(hits[0] if hits and all(np.allclose(h, hits[0], atol=1e-6) for h in hits) else None)
        return out
