from __future__ import annotations

import pytest
import torch
from torch import nn

from market_dynamics.models.deep_learning.asset_embeddings import AssetAgnosticModel
from market_dynamics.models.deep_learning.lstm import LSTMModel
from market_dynamics.models.deep_learning.mlp import MLPSequenceModel
from market_dynamics.models.deep_learning.tcn import TCNModel


@pytest.mark.parametrize(
    ("model", "state_shape", "representation_shape"),
    [
        (MLPSequenceModel(5, 7, 12, 0.0), (3, 6), (3, 6)),
        (LSTMModel(5, 10, 2, 0.0), (3, 7, 10), (3, 10)),
        (TCNModel(5, 8, 2, 3, 0.0), (3, 7, 8), (3, 8)),
    ],
)
def test_encode_shapes_and_forward_equivalence(
    model: nn.Module,
    state_shape: tuple[int, ...],
    representation_shape: tuple[int, ...],
) -> None:
    model.eval()
    x = torch.randn(3, 7, 5)

    states, representation, diagnostics = model.encode(x)
    head = model.network[-1] if isinstance(model, MLPSequenceModel) else model.head

    assert states.shape == state_shape
    assert representation.shape == representation_shape
    assert diagnostics is None
    torch.testing.assert_close(model(x), head(representation).squeeze(-1), rtol=0.0, atol=0.0)


class _WidthCheckingModel(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_size
        return x.sum(dim=(1, 2))


def test_capacity_matched_asset_agnostic_model_appends_id_independent_zeros() -> None:
    model = AssetAgnosticModel(_WidthCheckingModel(input_size=7), zero_channels=3)
    x = torch.randn(2, 6, 4)
    first_ids = torch.tensor([0, 1])
    second_ids = torch.tensor([8, 3])

    conditioned = model.conditioned_input(x, first_ids)

    assert conditioned.shape == (2, 6, 7)
    torch.testing.assert_close(conditioned[..., :4], x)
    assert torch.count_nonzero(conditioned[..., 4:]).item() == 0
    torch.testing.assert_close(model(x, first_ids), model(x, second_ids), rtol=0.0, atol=0.0)


def test_asset_agnostic_model_default_preserves_original_input_width() -> None:
    model = AssetAgnosticModel(_WidthCheckingModel(input_size=4))
    x = torch.randn(2, 6, 4)

    assert model.conditioned_input(x, torch.tensor([0, 1])) is x
    assert model(x, torch.tensor([0, 1])).shape == (2,)
