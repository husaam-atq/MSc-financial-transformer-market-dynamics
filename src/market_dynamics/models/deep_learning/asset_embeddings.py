"""Static asset conditioning for pooled models."""

from __future__ import annotations

import torch
from torch import nn


class AssetConditionedModel(nn.Module):
    """Append a learned identifier embedding to every known historical timestep."""

    def __init__(self, base_model: nn.Module, num_assets: int, embedding_dim: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.asset_embedding = nn.Embedding(num_assets, embedding_dim)

    def conditioned_input(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """Append the fixed asset embedding for diagnostics or base-model inference."""
        embedding = self.asset_embedding(asset_ids).unsqueeze(1).expand(-1, x.size(1), -1)
        return torch.cat([x, embedding], dim=-1)

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.conditioned_input(x, asset_ids))


class AssetAgnosticModel(nn.Module):
    """Accept pooled batches while deliberately ignoring their asset identifiers."""

    def __init__(self, base_model: nn.Module, zero_channels: int = 0) -> None:
        super().__init__()
        if zero_channels < 0:
            raise ValueError("zero_channels must be non-negative")
        self.base_model = base_model
        self.zero_channels = zero_channels

    def conditioned_input(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        """Append fixed zero channels without encoding asset identity."""
        del asset_ids
        if self.zero_channels == 0:
            return x
        zeros = x.new_zeros(*x.shape[:-1], self.zero_channels)
        return torch.cat([x, zeros], dim=-1)

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        return self.base_model(self.conditioned_input(x, asset_ids))


class FixedPriorResidualModel(nn.Module):
    """Add immutable training-period asset-prior logits to a dynamic residual."""

    def __init__(self, base_model: nn.Module, prior_logits: torch.Tensor) -> None:
        super().__init__()
        if prior_logits.ndim != 1:
            raise ValueError("prior_logits must be one-dimensional")
        self.base_model = base_model
        self.register_buffer("prior_logits", prior_logits.detach().float().clone())

    def residual_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return the dynamic component without the fixed prior."""
        return self.base_model(x)

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        residual = self.residual_logits(x)
        return residual + self.prior_logits[asset_ids]


class ZeroChannelSequenceModel(nn.Module):
    """Append fixed zero channels while retaining a one-argument model API."""

    def __init__(self, base_model: nn.Module, zero_channels: int) -> None:
        super().__init__()
        if zero_channels < 0:
            raise ValueError("zero_channels must be non-negative")
        self.base_model = base_model
        self.zero_channels = int(zero_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_channels == 0:
            return self.base_model(x)
        zeros = x.new_zeros(*x.shape[:-1], self.zero_channels)
        return self.base_model(torch.cat([x, zeros], dim=-1))
