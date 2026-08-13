"""Sliding-window PyTorch datasets that enforce temporal split boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from market_dynamics.preprocessing.scaling import TrainOnlyPreprocessor
from market_dynamics.splits.temporal import TemporalSplit


class WindowedTimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """A single-split sliding-window dataset.

    Every input ends at timestamp ``t`` and its label is the precomputed
    forward-looking target at ``t``. The constructor accepts rows from one
    split only, so no window can cross train/validation/test boundaries.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
        lookback: int,
        source_indices: np.ndarray | None = None,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must have shape [rows, features], got {features.shape}")
        if len(features) != len(targets) or len(features) != len(dates):
            raise ValueError("features, targets and dates must have equal lengths")
        if lookback < 1:
            raise ValueError("lookback must be at least one")
        if len(features) < lookback:
            raise ValueError(
                f"Split has {len(features)} rows but needs at least {lookback} for one window"
            )

        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        self.dates = pd.DatetimeIndex(dates)
        self.lookback = int(lookback)
        self.source_indices = (
            np.arange(len(features), dtype=np.int64)
            if source_indices is None
            else np.asarray(source_indices, dtype=np.int64)
        )
        if len(self.source_indices) != len(features):
            raise ValueError("source_indices must match feature row count")
        if len(self.source_indices) > 1 and not np.all(np.diff(self.source_indices) == 1):
            raise ValueError("source_indices must be contiguous within a split")

        self.endpoints = np.arange(self.lookback - 1, len(self.features), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = int(self.endpoints[item])
        start = end - self.lookback + 1
        window = torch.from_numpy(self.features[start : end + 1])
        target = torch.tensor(self.targets[end], dtype=torch.float32)
        source_index = torch.tensor(self.source_indices[end], dtype=torch.long)
        return window, target, source_index

    @property
    def window_end_dates(self) -> pd.DatetimeIndex:
        """Dates corresponding to target and final input row for each window."""
        return self.dates[self.endpoints]


@dataclass
class WindowDataBundle:
    """Windowed datasets plus train-fitted preprocessing state."""

    train: WindowedTimeSeriesDataset
    validation: WindowedTimeSeriesDataset
    test: WindowedTimeSeriesDataset
    preprocessor: TrainOnlyPreprocessor
    feature_columns: list[str]
    target_column: str
    split: TemporalSplit


def build_window_datasets(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split: TemporalSplit,
    lookback: int,
) -> WindowDataBundle:
    """Fit preprocessing on train rows and create independent split datasets."""
    if target_column not in frame.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Window datasets require a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Window datasets require sorted time order")
    if frame[target_column].isna().any():
        raise ValueError("Target frame must have null targets removed before window construction")

    preprocessor = TrainOnlyPreprocessor(scale=True)
    preprocessor.fit(frame.iloc[split.train_idx][feature_columns])

    train = _make_split_dataset(frame, feature_columns, target_column, split.train_idx, preprocessor, lookback)
    validation = _make_split_dataset(
        frame, feature_columns, target_column, split.val_idx, preprocessor, lookback
    )
    test = _make_split_dataset(frame, feature_columns, target_column, split.test_idx, preprocessor, lookback)
    return WindowDataBundle(
        train=train,
        validation=validation,
        test=test,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        target_column=target_column,
        split=split,
    )


def _make_split_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    indices: np.ndarray,
    preprocessor: TrainOnlyPreprocessor,
    lookback: int,
) -> WindowedTimeSeriesDataset:
    if len(indices) < lookback:
        raise ValueError(
            f"Split has {len(indices)} rows but lookback={lookback}. "
            "Reduce lookback or increase the split length."
        )
    if len(indices) > 1 and not np.all(np.diff(indices) == 1):
        raise ValueError("Temporal split indices must be contiguous for strict window boundaries")

    subset = frame.iloc[indices]
    return WindowedTimeSeriesDataset(
        features=preprocessor.transform(subset[feature_columns]),
        targets=subset[target_column].to_numpy(dtype=np.float32),
        dates=subset.index,
        lookback=lookback,
        source_indices=indices,
    )
