"""Channel-independent PatchTST encoder for multivariate historical windows."""

from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning.transformer_encoder import PositionalEncoding


class PatchTSTModel(nn.Module):
    """Patch each variable's history and share a Transformer across channels.

    The model uses PatchTST's channel-independence principle: each feature
    trajectory is tokenised along time with shared weights, then channel states
    are aggregated only at the scalar forecasting head. The input contains only
    observations through the forecast timestamp, so patching introduces no
    look-ahead.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        patch_length: int,
        patch_stride: int,
        dropout: float,
        ff_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if patch_length < 1 or patch_stride < 1:
            raise ValueError("patch_length and patch_stride must be positive")
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.patch_projection = nn.Linear(patch_length, hidden_size)
        self.position = PositionalEncoding(hidden_size)
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
        if x.size(1) < self.patch_length:
            raise ValueError("lookback must be at least patch_length")
        patches = x.transpose(1, 2).unfold(dimension=2, size=self.patch_length, step=self.patch_stride)
        batch, channels, patch_count, patch_length = patches.shape
        tokens = self.patch_projection(patches.reshape(batch * channels, patch_count, patch_length))
        encoded = self.encoder(self.position(tokens)).mean(dim=1).reshape(batch, channels, -1)
        return self.head(encoded.mean(dim=1)).squeeze(-1)
