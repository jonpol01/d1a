"""The pointer head, the backend-independent core of D1A's readout (#60): the torch module whose weights every backend
(d1a.backends.torch, d1a.backends.mlx) applies to the hidden states at <decide> and at each option's </opt>."""
import math

from torch import nn


class PointerHead(nn.Module):
    """Scores a question's options: a scaled dot product of the projected <decide> state (q) with each projected </opt>
    state (k). In eval mode the logits are divided by `temperature` (1.0: raw), the calibration a checkpoint carries
    (d1a.training.calibrate); training always sees 1, so a fitted value stays meaningful, and the argmax never changes."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)   # dp: the pointer dimension
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0

    def _tempered(self, z):
        return z if self.training or self.temperature == 1.0 else z / self.temperature

    def forward(self, h_decide, h_opts):   # [d], [K, d] -> logits [K]
        return self._tempered((self.k(h_opts) @ self.q(h_decide)) * self.scale)

    def many(self, h_decide, h_opts, owner):   # [Q, d], [sum K, d], each option's question [sum K] -> logits [sum K]
        """forward() for many questions in one pass (serving batches)."""
        return self._tempered((self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale)
