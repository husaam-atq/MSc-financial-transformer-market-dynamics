"""DLinear-style seasonal/trend decomposition linear forecaster."""

from __future__ import annotations

import torch
from torch import nn


class MovingAverage(nn.Module):
    """Replicate-padded moving average retaining the input sequence length."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        left = x[:, :1].expand(-1, padding, -1)
        right = x[:, -1:].expand(-1, padding, -1)
        padded = torch.cat([left, x, right], dim=1)
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class DLinearModel(nn.Module):
    """Linear heads applied separately to decomposed temporal trend and residual."""

    def __init__(self, input_size: int, lookback: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.decomposition = MovingAverage(min(kernel_size if kernel_size % 2 else kernel_size + 1, lookback if lookback % 2 else lookback - 1))
        self.seasonal = nn.Linear(lookback, 1)
        self.trend = nn.Linear(lookback, 1)
        self.feature_head = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomposition(x)
        seasonal = x - trend
        scalar = self.seasonal(seasonal.transpose(1, 2)).squeeze(-1) + self.trend(trend.transpose(1, 2)).squeeze(-1)
        return self.feature_head(scalar).squeeze(-1)
