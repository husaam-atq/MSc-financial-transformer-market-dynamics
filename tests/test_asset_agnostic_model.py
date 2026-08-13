from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning import AssetAgnosticModel


class _MeanModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2))


def test_asset_agnostic_model_ignores_identifier() -> None:
    model = AssetAgnosticModel(_MeanModel())
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    left = model(x, torch.tensor([0, 1]))
    right = model(x, torch.tensor([5, 7]))

    assert torch.equal(left, right)
