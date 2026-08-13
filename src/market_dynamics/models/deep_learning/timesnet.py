"""TimesNet-inspired frequency-period convolutional encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class TimesNetModel(nn.Module):
    """Use dominant FFT periods to aggregate multiscale convolutional temporal views."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float, top_k: int = 3) -> None:
        super().__init__()
        self.top_k = top_k
        self.projection = nn.Linear(input_size, hidden_size)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1), nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(x)
        # cuFFT does not support arbitrary sequence lengths in float16. Period
        # discovery is numerically stable in float32 and carries no gradients.
        spectrum = torch.fft.rfft(encoded.float(), dim=1).abs().mean(dim=(0, 2))
        valid = spectrum[1:] if spectrum.numel() > 1 else spectrum
        indices = torch.topk(valid, k=min(self.top_k, valid.numel())).indices + (1 if spectrum.numel() > 1 else 0)
        period_views = []
        for frequency in indices:
            frequency_value = max(int(frequency.item()), 1)
            period = max(1, x.size(1) // frequency_value)
            padded_length = ((x.size(1) + period - 1) // period) * period
            padded = functional.pad(encoded, (0, 0, 0, padded_length - x.size(1)))
            filtered = self.conv(padded.transpose(1, 2)).transpose(1, 2)[:, : x.size(1)]
            period_views.append(filtered)
        mixed = torch.stack(period_views).mean(dim=0) if period_views else encoded
        return self.head(self.dropout(self.norm(mixed + encoded))[:, -1]).squeeze(-1)
