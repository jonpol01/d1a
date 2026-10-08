"""The pointer head, the backend-independent core of D1A's readout (#60): the torch module whose weights every backend
(d1a.backends.torch, d1a.backends.mlx) applies to the hidden states at <decide> and at each option's </opt>."""
import math

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
