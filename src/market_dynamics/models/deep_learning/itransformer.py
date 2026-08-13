"""Optional iTransformer-style inverted-variate encoder."""

from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning.transformer_encoder import PositionalEncoding


class ITransformerModel(nn.Module):
    """Treat each feature trajectory as a token, inspired by iTransformer."""

    def __init__(
        self,
        input_size: int,
        lookback: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.temporal_projection = nn.Linear(lookback, hidden_size)
        self.position = PositionalEncoding(hidden_size, max_length=input_size)
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
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.temporal_projection(x.transpose(1, 2))
        encoded = self.encoder(self.position(tokens))
        return self.head(encoded.mean(dim=1)).squeeze(-1)
