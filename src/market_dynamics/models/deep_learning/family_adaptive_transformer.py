"""Family-adaptive shared Transformer for bounded cross-asset generalisation tests."""

from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning.transformer_encoder import PositionalEncoding


class FamilyAdaptiveTransformer(nn.Module):
    """Shared temporal encoder with static asset/family context and residual heads.

    A single backbone learns common temporal states. Asset and family embeddings
    condition each timestep, while small family-specific residual heads can adjust
    the shared logit. This is deliberately lower-capacity than a mixture of
    experts and keeps every observation in the same shared representation space.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_assets: int,
        asset_to_family: torch.Tensor,
        num_families: int,
        asset_embedding_dim: int = 12,
        family_embedding_dim: int = 8,
        ff_multiplier: int = 2,
        max_length: int = 1024,
        pooling: str = "temporal_attention",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if pooling not in {"last", "temporal_attention"}:
            raise ValueError("pooling must be 'last' or 'temporal_attention'")
        mapping = torch.as_tensor(asset_to_family, dtype=torch.long).flatten()
        if len(mapping) != int(num_assets):
            raise ValueError("asset_to_family length must equal num_assets")
        if mapping.numel() == 0 or int(mapping.min()) < 0 or int(mapping.max()) >= int(num_families):
            raise ValueError("asset_to_family contains an invalid family id")
        self.register_buffer("asset_to_family", mapping, persistent=True)
        self.pooling = str(pooling)
        self.asset_embedding = nn.Embedding(int(num_assets), int(asset_embedding_dim))
        self.family_embedding = nn.Embedding(int(num_families), int(family_embedding_dim))
        conditioned_size = int(input_size) + int(asset_embedding_dim) + int(family_embedding_dim)
        self.input_projection = nn.Linear(conditioned_size, int(hidden_size))
        self.position = PositionalEncoding(int(hidden_size), max_length=int(max_length))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_size),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_size) * int(ff_multiplier),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
        self.pool_score = nn.Linear(int(hidden_size), 1, bias=False) if self.pooling == "temporal_attention" else None
        self.shared_head = nn.Sequential(nn.LayerNorm(int(hidden_size)), nn.Dropout(float(dropout)), nn.Linear(int(hidden_size), 1))
        self.family_adapter = nn.Sequential(
            nn.Linear(int(hidden_size) + int(family_embedding_dim), int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.family_heads = nn.ModuleList([nn.Linear(int(hidden_size), 1) for _ in range(int(num_families))])

    def family_ids(self, asset_ids: torch.Tensor) -> torch.Tensor:
        """Map validated pooled asset ids to the fixed family ids."""
        ids = asset_ids.long()
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(self.asset_to_family)):
            raise IndexError("Asset id is outside the configured family map")
        return self.asset_to_family[ids]

    def encode(self, x: torch.Tensor, asset_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return encoded states, pooled state, family ids and attention weights."""
        families = self.family_ids(asset_ids)
        asset_context = self.asset_embedding(asset_ids.long()).unsqueeze(1).expand(-1, x.size(1), -1)
        family_context = self.family_embedding(families).unsqueeze(1).expand(-1, x.size(1), -1)
        embedded = self.input_projection(torch.cat([x, asset_context, family_context], dim=-1))
        states = self.encoder(self.position(embedded))
        if self.pooling == "temporal_attention":
            if self.pool_score is None:  # Defensive invariant for serialized configuration.
                raise RuntimeError("Temporal attention pooling is missing its score layer")
            attention = torch.softmax(self.pool_score(states).squeeze(-1), dim=1)
            summary = torch.sum(states * attention.unsqueeze(-1), dim=1)
            return states, summary, families, attention
        return states, states[:, -1, :], families, None

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        _, summary, families, _ = self.encode(x, asset_ids)
        shared = self.shared_head(summary).squeeze(-1)
        adapted = self.family_adapter(torch.cat([summary, self.family_embedding(families)], dim=-1))
        all_residuals = torch.cat([head(adapted) for head in self.family_heads], dim=1)
        residual = all_residuals.gather(1, families.unsqueeze(1)).squeeze(1)
        return shared + residual
