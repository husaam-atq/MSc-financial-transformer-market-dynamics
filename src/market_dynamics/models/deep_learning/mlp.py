"""Flattened-window MLP sequence baseline."""

from __future__ import annotations

import torch
from torch import nn


class MLPSequenceModel(nn.Module):
    """An MLP baseline that sees the complete ordered feature window."""

    def __init__(self, input_size: int, lookback: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        flattened_size = input_size * lookback
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Return the learned window representation in the common encoder format."""
        representation = x
        for layer in list(self.network)[:-1]:
            representation = layer(representation)
        return representation, representation, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, representation, _ = self.encode(x)
        return self.network[-1](representation).squeeze(-1)
