"""Causal temporal convolutional network."""

from __future__ import annotations

import torch
from torch import nn


class _Chomp1d(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.size] if self.size > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.residual(x))


class TCNModel(nn.Module):
    """Causal dilated-convolution model over a historical feature window."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for layer in range(num_layers):
            blocks.append(
                _TemporalBlock(
                    in_channels=input_size if layer == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Return batch-first temporal states and the final-timestep representation."""
        channel_first_states = self.network(x.transpose(1, 2))
        states = channel_first_states.transpose(1, 2)
        return states, channel_first_states[:, :, -1], None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, representation, _ = self.encode(x)
        return self.head(representation).squeeze(-1)
