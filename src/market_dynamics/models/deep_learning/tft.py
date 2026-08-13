"""Compact Temporal Fusion Transformer for observed covariate windows."""

from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning.transformer_encoder import PositionalEncoding


class GatedResidualNetwork(nn.Module):
    """TFT-style nonlinear residual and gated skip pathway."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ELU(), nn.Linear(hidden_size, hidden_size))
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.skip = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.hidden(x)
        return self.norm(self.skip(x) + self.dropout(transformed * self.gate(transformed)))


class TemporalFusionTransformerModel(nn.Module):
    """TFT-inspired variable selection, LSTM temporal encoding and interpretable attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_length: int = 1024,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.variable_logits = nn.Linear(input_size, input_size)
        self.variable_values = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(input_size)])
        self.variable_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.position = PositionalEncoding(hidden_size, max_length=max_length)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.post_attention = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.variable_logits(x), dim=-1)
        values = torch.stack([layer(x[..., index : index + 1]) for index, layer in enumerate(self.variable_values)], dim=-2)
        selected = (weights.unsqueeze(-1) * values).sum(dim=-2)
        selected = self.variable_grn(selected)
        encoded, _ = self.lstm(selected)
        attended, _ = self.attention(self.position(encoded), self.position(encoded), encoded, need_weights=False)
        return self.head(self.post_attention(attended)[:, -1]).squeeze(-1)
