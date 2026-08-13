"""FEDformer-inspired frequency-domain decomposition transformer."""

from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning.dlinear import MovingAverage
from market_dynamics.models.deep_learning.transformer_encoder import PositionalEncoding


class FourierBlock(nn.Module):
    """Learn complex frequency filters for the strongest Fourier coefficients."""

    def __init__(self, hidden_size: int, modes: int = 16) -> None:
        super().__init__()
        self.modes = modes
        self.weight_real = nn.Parameter(torch.randn(modes, hidden_size) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(modes, hidden_size) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use float32 for FFT support under CUDA AMP with arbitrary lookbacks.
        spectrum = torch.fft.rfft(x.float(), dim=1)
        use_modes = min(self.modes, spectrum.size(1))
        complex_weight = torch.complex(self.weight_real[:use_modes], self.weight_imag[:use_modes]).unsqueeze(0)
        filtered = torch.zeros_like(spectrum)
        filtered[:, :use_modes] = spectrum[:, :use_modes] * complex_weight
        return torch.fft.irfft(filtered, n=x.size(1), dim=1).to(dtype=x.dtype)


class FEDformerModel(nn.Module):
    """Frequency-enhanced decomposed transformer encoder for observed windows."""

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
        self.decomposition = MovingAverage(25)
        self.input = nn.Linear(input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size, max_length=max_length)
        self.fourier = nn.ModuleList([FourierBlock(hidden_size) for _ in range(num_layers)])
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, 2 * hidden_size, dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomposition(x)
        seasonal = self.position(self.input(x - trend))
        for block in self.fourier:
            seasonal = seasonal + block(seasonal)
        encoded = self.encoder(seasonal)
        return self.head(encoded[:, -1] + self.input(trend[:, -1])).squeeze(-1)
