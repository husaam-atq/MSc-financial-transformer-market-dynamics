"""Bidirectional LSTM over a fully observed historical window."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMModel(nn.Module):
    """Bidirectional encoder; bidirectionality never extends beyond input time t."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Dropout(dropout), nn.Linear(hidden_size * 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1]).squeeze(-1)
