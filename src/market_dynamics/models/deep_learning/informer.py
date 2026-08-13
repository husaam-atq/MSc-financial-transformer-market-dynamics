"""Informer-inspired sparse-attention encoder for long observed sequences."""

from __future__ import annotations

import math

import torch
from torch import nn

from market_dynamics.models.deep_learning.transformer_encoder import PositionalEncoding


class ProbSparseAttention(nn.Module):
    """Query-sampled approximation of Informer's ProbSparse attention mechanism."""

    def __init__(self, hidden_size: int, heads: int, dropout: float, sample_factor: int = 5) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.sample_factor = sample_factor
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        sample_count = min(length, max(1, int(self.sample_factor * math.log(length + 1))))
        sampled = torch.linspace(0, length - 1, sample_count, device=x.device).long()
        logits = torch.matmul(q, k[:, :, sampled].transpose(-2, -1)) / math.sqrt(self.head_dim)
        values = torch.matmul(self.dropout(torch.softmax(logits, dim=-1)), v[:, :, sampled])
        return self.out(values.transpose(1, 2).reshape(batch, length, hidden))


class InformerModel(nn.Module):
    """Encoder-only Informer-style model for scalar classification or regression output."""

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
        self.input = nn.Linear(input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size, max_length=max_length)
        self.attention = nn.ModuleList([ProbSparseAttention(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, 2 * hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden_size, hidden_size)) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.position(self.input(x))
        for attention, ffn, norm in zip(self.attention, self.ffn, self.norms, strict=True):
            encoded = norm(encoded + attention(encoded))
            encoded = norm(encoded + ffn(encoded))
        return self.head(encoded[:, -1]).squeeze(-1)
