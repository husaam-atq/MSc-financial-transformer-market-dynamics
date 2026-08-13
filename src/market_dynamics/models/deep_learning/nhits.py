"""N-HiTS-inspired multi-rate interpolation hierarchy."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class _NHiTSBlock(nn.Module):
    def __init__(self, input_size: int, lookback: int, hidden_size: int, pool_kernel: int, dropout: float) -> None:
        super().__init__()
        self.pool_kernel = min(pool_kernel, lookback)
        pooled_length = max(1, lookback // self.pool_kernel)
        self.net = nn.Sequential(
            nn.Linear(input_size * pooled_length, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.forecast = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = functional.avg_pool1d(x.transpose(1, 2), self.pool_kernel, self.pool_kernel).flatten(start_dim=1)
        return self.forecast(self.net(pooled))


class NHiTSModel(nn.Module):
    """Hierarchical multi-rate N-HiTS-inspired scalar forecasting stack."""

    def __init__(self, input_size: int, lookback: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _NHiTSBlock(input_size, lookback, hidden_size, kernel, dropout) for kernel in (1, 2, 4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([block(x).squeeze(-1) for block in self.blocks], dim=0).sum(dim=0)
