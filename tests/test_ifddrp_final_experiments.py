from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from market_dynamics.experiments import ifddrp_final_experiments as final_experiments
from market_dynamics.experiments.ifddrp_final_experiments import (
    _date_block_equal_asset_mae_difference,
    _fast_within_auc,
)
from market_dynamics.experiments.run_phase6 import (
    PerturbedDataset,
    _temporal_transform,
    _within_group_auc,
)


def test_fast_within_auc_matches_reference_with_ties() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 1, 0], dtype=int)
    probability = np.asarray([0.1, 0.8, 0.8, 0.8, 0.2, 0.7, 0.7, 0.4], dtype=float)
    groups = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    reference = _within_group_auc(labels, probability, groups)["pair_weighted_within_asset_roc_auc"]
    assert _fast_within_auc(labels, probability, groups) == reference


def test_within_auc_weights_assets_by_positive_negative_pairs() -> None:
    labels = np.asarray([0, 1, 0, 0, 1, 1], dtype=int)
    probability = np.asarray([0.1, 0.9, 0.9, 0.8, 0.2, 0.1], dtype=float)
    groups = np.asarray([0, 0, 1, 1, 1, 1], dtype=int)

    result = _within_group_auc(labels, probability, groups)

    assert result["per_asset_macro_roc_auc"] == pytest.approx(0.5)
    assert result["pair_weighted_within_asset_roc_auc"] == pytest.approx(0.2)
    assert result["eligible_assets"] == 2


@pytest.mark.parametrize("method", ["reverse", "deterministic_permutation", "circular_shift"])
def test_temporal_perturbations_move_complete_rows_and_preserve_metadata(method: str) -> None:
    lookback = 6
    features = torch.stack(
        [torch.arange(lookback, dtype=torch.float32), torch.arange(lookback, dtype=torch.float32) + 100.0],
        dim=1,
    )
    base = TensorDataset(
        features.unsqueeze(0),
        torch.tensor([1.0]),
        torch.tensor([17]),
        torch.tensor([4]),
    )
    perturbed = PerturbedDataset(base, _temporal_transform(method, lookback, seed=7))

    changed, target, source_index, asset_id = perturbed[0]

    assert torch.equal(torch.sort(changed[:, 0]).values, torch.arange(lookback, dtype=torch.float32))
    assert torch.equal(changed[:, 1] - changed[:, 0], torch.full((lookback,), 100.0))
    assert target.item() == 1.0
    assert source_index.item() == 17
    assert asset_id.item() == 4


def test_deterministic_temporal_permutation_reuses_registered_order() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    first = _temporal_transform("deterministic_permutation", 6, seed=20260712)(features)
    second = _temporal_transform("deterministic_permutation", 6, seed=20260712)(features)

    assert torch.equal(first, second)


def test_paired_mae_difference_is_positive_when_changed_predictions_are_worse(monkeypatch) -> None:
    monkeypatch.setattr(final_experiments, "_bootstrap_jobs", lambda: 1)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    base = pd.DataFrame(
        {
            "Date": np.repeat(dates, 2),
            "source_index": np.arange(40),
            "asset_id": np.tile([0, 1], 20),
            "asset_ticker": np.tile(["A", "B"], 20),
            "y_true": np.zeros(40),
            "prediction": np.zeros(40),
        }
    )
    changed = base.copy()
    changed["y_true"] = changed["y_true"].astype(np.float32)
    changed["prediction"] = 0.25
    result = _date_block_equal_asset_mae_difference(
        base,
        changed,
        iterations=16,
        block_size=4,
        seed=7,
        difference="changed_minus_base",
    )
    assert result["estimate"] == 0.25
    assert result["ci_lower"] == 0.25
