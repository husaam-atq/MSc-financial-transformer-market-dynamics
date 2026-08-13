"""Vanilla Transformer Encoder for temporal feature sequences."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding supporting batch-first tensors."""

    def __init__(self, d_model: int, max_length: int = 1024) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding max_length "
                f"{self.encoding.size(1)}"
            )
        return x + self.encoding[:, : x.size(1)]


class TransformerEncoderModel(nn.Module):
    """Time-token Transformer Encoder with a bounded configurable readout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_multiplier: int = 2,
        max_length: int = 1024,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if pooling not in {"last", "temporal_attention"}:
            raise ValueError("pooling must be 'last' or 'temporal_attention'")
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size, max_length=max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pooling = pooling
        self.pool_score = nn.Linear(hidden_size, 1, bias=False) if pooling == "temporal_attention" else None
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return temporal states, pooled representation and diagnostic attention weights."""
        embedded = self.position(self.input_projection(x))
        states = self.encoder(embedded)
        if self.pooling == "temporal_attention":
            if self.pool_score is None:  # Defensive invariant for serialized model configs.
                raise RuntimeError("Temporal attention pooling is missing its score layer")
            weights = torch.softmax(self.pool_score(states).squeeze(-1), dim=1)
            summary = torch.sum(states * weights.unsqueeze(-1), dim=1)
            return states, summary, weights
        else:
            summary = states[:, -1, :]
        return states, summary, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar task head applied to the encoded temporal summary."""
        _, summary, _ = self.encode(x)
        return self.head(summary).squeeze(-1)
