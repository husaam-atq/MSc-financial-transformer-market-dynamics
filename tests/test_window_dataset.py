from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.datasets.window_dataset import build_window_datasets
from market_dynamics.experiments.run_large_scale_screening import _loaders
from market_dynamics.splits.temporal import chronological_split


def _frame(n: int = 180) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
    return pd.DataFrame(
        {
            "feature_a": np.arange(n, dtype=float),
            "feature_b": np.arange(n, dtype=float) ** 2,
            "target_direction_5d": (np.arange(n) % 2).astype(float),
        },
        index=index,
    )


def test_window_dataset_shapes_and_target_alignment() -> None:
    frame = _frame()
    split = chronological_split(frame, purge=5, embargo=1)
    bundle = build_window_datasets(
        frame,
        ["feature_a", "feature_b"],
        "target_direction_5d",
        split,
        lookback=12,
    )

    x, y, source_index = bundle.train[0]
    endpoint = split.train_idx[11]
    assert x.shape == (12, 2)
    assert y.item() == frame.iloc[endpoint]["target_direction_5d"]
    assert source_index.item() == endpoint
    assert len(bundle.train) == len(split.train_idx) - 12 + 1


def test_no_window_crosses_split_boundaries() -> None:
    frame = _frame()
    split = chronological_split(frame, purge=5, embargo=2)
    bundle = build_window_datasets(
        frame,
        ["feature_a", "feature_b"],
        "target_direction_5d",
        split,
        lookback=10,
    )

    for dataset, indices in [
        (bundle.train, split.train_idx),
        (bundle.validation, split.val_idx),
        (bundle.test, split.test_idx),
    ]:
        for endpoint in dataset.endpoints:
            source_window = dataset.source_indices[endpoint - dataset.lookback + 1 : endpoint + 1]
            assert source_window.min() >= indices.min()
            assert source_window.max() <= indices.max()


def test_train_loader_shuffles_but_validation_and_test_are_sequential() -> None:
    frame = _frame()
    split = chronological_split(frame, purge=4, embargo=1)
    bundle = build_window_datasets(
        frame,
        ["feature_a", "feature_b"],
        "target_direction_5d",
        split,
        lookback=8,
    )
    train_loader, validation_loader, test_loader = _loaders(
        bundle.train,
        bundle.validation,
        bundle.test,
        {"batch_size": 4, "num_workers": 0},
        seed=42,
    )

    train_indices = np.concatenate([batch[2].numpy() for batch in train_loader])
    validation_indices = np.concatenate([batch[2].numpy() for batch in validation_loader])
    test_indices = np.concatenate([batch[2].numpy() for batch in test_loader])
    assert not np.array_equal(train_indices, np.sort(train_indices))
    assert np.array_equal(validation_indices, np.sort(validation_indices))
    assert np.array_equal(test_indices, np.sort(test_indices))
