"""Autoformer-inspired decomposition and auto-correlation aggregation."""

from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning.dlinear import MovingAverage


class AutoCorrelationLayer(nn.Module):
    """FFT cross-correlation aggregation over a fixed observed input sequence."""

    def __init__(self, hidden_size: int, top_k: int = 3) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self.query(x), self.key(x), self.value(x)
        # Execute FFT correlation in float32 so AMP works for non-power-of-two
        # lookbacks such as the preregistered daily 60 observations.
        query_spectrum = torch.fft.rfft(query.float(), dim=1)
        key_spectrum = torch.fft.rfft(key.float(), dim=1)
        corr = torch.fft.irfft(query_spectrum * torch.conj(key_spectrum), n=x.size(1), dim=1).mean(dim=(0, 2))
        lag_indices = torch.topk(corr, k=min(self.top_k, corr.numel())).indices
        weights = torch.softmax(corr[lag_indices], dim=0)
        aggregate = torch.zeros_like(value)
        for lag, weight in zip(lag_indices, weights, strict=True):
            aggregate = aggregate + weight.to(value.dtype) * torch.roll(value, shifts=-int(lag.item()), dims=1)
        return aggregate


class AutoformerModel(nn.Module):
    """Decomposition architecture with auto-correlation temporal aggregation."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.decomposition = MovingAverage(25)
        self.input = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList([AutoCorrelationLayer(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomposition(x)
        encoded = self.input(x - trend)
        for layer in self.layers:
            encoded = self.norm(encoded + self.dropout(layer(encoded)))
        trend_state = self.input(trend[:, -1])
        return self.head(encoded[:, -1] + trend_state).squeeze(-1)
