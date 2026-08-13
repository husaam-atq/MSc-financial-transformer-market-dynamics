from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.datasets.pairwise import (
    WithinAssetPairDataset,
    build_outcome_disjoint_pair_registry,
)
from market_dynamics.datasets.pooled_window_dataset import PooledWindowDataset, _AssetSplitData


def _dataset() -> PooledWindowDataset:
    dates = pd.date_range("2020-01-01", periods=24, freq="D")
    parts = []
    for asset_id in [0, 1]:
        target = np.asarray([(index + asset_id) % 3 == 0 for index in range(24)], dtype=np.float32)
        parts.append(
            _AssetSplitData(
                features=np.arange(48, dtype=np.float32).reshape(24, 2),
                target=target,
                dates=pd.DatetimeIndex(dates),
                asset_id=asset_id,
                endpoints=np.arange(24, dtype=np.int64),
                source_indices=np.arange(asset_id * 100, asset_id * 100 + 24, dtype=np.int64),
            )
        )
    return PooledWindowDataset(parts, lookback=1)


def test_pair_registry_is_same_asset_outcome_disjoint_and_deterministic() -> None:
    dataset = _dataset()
    first = build_outcome_disjoint_pair_registry(
        dataset,
        horizon=5,
        maximum_pairs_per_asset=20,
        seed=17,
        split="train",
    )
    second = build_outcome_disjoint_pair_registry(
        dataset,
        horizon=5,
        maximum_pairs_per_asset=20,
        seed=17,
        split="train",
    )

    assert first.sha256 == second.sha256
    assert len(first.pairs) == 40
    assert first.pairs["origin_distance_sessions"].ge(5).all()
    assert first.pairs["split"].eq("train").all()
    pair_dataset = WithinAssetPairDataset(dataset, first)
    for item in [0, len(pair_dataset) - 1]:
        _, _, asset_id, weight = pair_dataset[item]
        assert int(asset_id) in {0, 1}
        assert float(weight) > 0.0


def test_pair_registry_reports_raw_and_disjoint_support() -> None:
    registry = build_outcome_disjoint_pair_registry(
        _dataset(),
        horizon=10,
        maximum_pairs_per_asset=7,
        seed=9,
        split="validation",
    )

    assert registry.audit["raw_comparable_pairs"].gt(registry.audit["outcome_disjoint_pairs"]).all()
    assert registry.audit["selected_pairs"].eq(7).all()
    assert registry.audit["minimum_origin_distance"].ge(10).all()
