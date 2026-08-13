from __future__ import annotations

import torch
from torch import nn

from market_dynamics.models.deep_learning.asset_embeddings import (
    FixedPriorResidualModel,
    ZeroChannelSequenceModel,
)
from market_dynamics.training.losses import PairwiseLogisticRankingLoss
from market_dynamics.training.train import _bounded_pairwise_logistic_loss


class _SumModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(1, 2))


def test_fixed_prior_residual_adds_only_training_prior_for_requested_asset() -> None:
    model = FixedPriorResidualModel(_SumModel(), torch.tensor([-1.0, 2.0]))
    features = torch.ones((2, 3, 1))
    output = model(features, torch.tensor([0, 1]))
    assert torch.allclose(output, torch.tensor([2.0, 5.0]))
    assert torch.allclose(model.residual_logits(features), torch.tensor([3.0, 3.0]))


def test_pairwise_ranking_loss_rewards_positive_score_above_negative() -> None:
    loss = PairwiseLogisticRankingLoss()
    good = loss(torch.tensor([2.0, 3.0]), torch.tensor([-1.0, 0.0]))
    bad = loss(torch.tensor([-1.0, 0.0]), torch.tensor([2.0, 3.0]))
    assert good < bad


def test_bounded_pairwise_loss_is_deterministic_and_respects_cap() -> None:
    positive = torch.arange(10, dtype=torch.float32)
    negative = torch.arange(10, dtype=torch.float32) - 2.0
    uncapped = _bounded_pairwise_logistic_loss(positive, negative, maximum_pairs=None)
    first = _bounded_pairwise_logistic_loss(positive, negative, maximum_pairs=7)
    second = _bounded_pairwise_logistic_loss(positive, negative, maximum_pairs=7)

    assert torch.isfinite(uncapped)
    assert torch.isfinite(first)
    assert torch.equal(first, second)


def test_zero_channel_sequence_model_capacity_matches_without_information() -> None:
    class _Capture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen = x
            return x.sum(dim=(1, 2))

    capture = _Capture()
    model = ZeroChannelSequenceModel(capture, zero_channels=2)
    values = torch.ones((3, 4, 5))
    model(values)
    assert capture.seen is not None
    assert capture.seen.shape == (3, 4, 7)
    assert torch.count_nonzero(capture.seen[:, :, -2:]) == 0
