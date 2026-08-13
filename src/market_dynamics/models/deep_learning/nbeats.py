"""N-BEATS residual stack with generic fully connected blocks."""

from __future__ import annotations

import torch
from torch import nn


class _NBeatsBlock(nn.Module):
    def __init__(self, size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = size
        for _ in range(4):
            layers.extend([nn.Linear(current, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden_size
        self.layers = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_size, size)
        self.forecast = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.layers(x)
        return self.backcast(theta), self.forecast(theta)


class NBeatsModel(nn.Module):
    """Generic residual N-BEATS blocks operating on flattened observed windows."""

    def __init__(self, input_size: int, lookback: int, hidden_size: int, num_blocks: int, dropout: float) -> None:
        super().__init__()
        self.size = input_size * lookback
        self.blocks = nn.ModuleList([_NBeatsBlock(self.size, hidden_size, dropout) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.flatten(start_dim=1)
        forecast = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        for block in self.blocks:
            backcast, increment = block(residual)
            residual = residual - backcast
            forecast = forecast + increment
        return forecast.squeeze(-1)
