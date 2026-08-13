"""Tests for the final bounded Transformer interpretation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from market_dynamics.interpretability.ifddrp_final import (
    _feature_groups,
    _integrated_gradients,
    _markdown_table,
)


class _AdditiveSequenceModel(nn.Module):
    def forward(self, features: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        del asset_ids
        return features.sum(dim=(1, 2))


def test_final_feature_groups_cover_each_input_once() -> None:
    features = ["return_close", "high_low_range", "volatility_20d", "DFF"]
    definitions = {
        "returns_momentum": ["return_close"],
        "range_dispersion": ["high_low_range"],
        "volatility": ["volatility_20d"],
        "macro_context": ["DFF"],
    }
    groups = _feature_groups(features, definitions)
    covered = sorted(index for indices in groups.values() for index in indices)
    assert covered == list(range(len(features)))
    assert groups["macro_context"] == [len(features) - 1]


def test_integrated_gradients_matches_additive_model() -> None:
    features = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    attribution = _integrated_gradients(
        _AdditiveSequenceModel(),
        features,
        torch.tensor([0]),
        torch.device("cpu"),
        steps=16,
    )
    np.testing.assert_allclose(attribution, features.numpy(), atol=1e-7)


def test_markdown_table_has_no_optional_dependency() -> None:
    rendered = _markdown_table(pd.DataFrame([{"component": "A|B", "support": "partial"}]))
    assert "| component | support |" in rendered
    assert "A\\|B" in rendered
