"""NLinear-style last-value normalised linear forecaster."""

from __future__ import annotations

import torch
from torch import nn


class NLinearModel(nn.Module):
    """Apply independent linear temporal maps after subtracting each variable's last value."""

    def __init__(self, input_size: int, lookback: int) -> None:
        super().__init__()
        self.linear = nn.Linear(lookback, 1)
        self.feature_head = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor = x[:, -1:, :].detach()
        normalized = x - anchor
        mapped = self.linear(normalized.transpose(1, 2)).squeeze(-1) + anchor.squeeze(1)
        return self.feature_head(mapped).squeeze(-1)
