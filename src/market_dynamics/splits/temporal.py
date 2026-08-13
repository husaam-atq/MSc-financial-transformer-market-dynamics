"""Chronological and walk-forward validation splits with purge/embargo gaps."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Index positions for one leakage-aware temporal split."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    val_start: pd.Timestamp | None = None
    val_end: pd.Timestamp | None = None
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None
    purge: int = 0
    embargo: int = 0


@dataclass(frozen=True)
class GlobalDateSplit:
    """Globally shared date boundaries for a pooled multi-asset panel."""

    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purge: int = 0
    embargo: int = 0

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def val_end(self) -> pd.Timestamp:
        return self.val_dates[-1]

    @property
    def test_end(self) -> pd.Timestamp:
        return self.test_dates[-1]


def chronological_split(
    frame: pd.DataFrame,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    purge: int = 0,
    embargo: int = 0,
    min_samples: int = 1,
) -> TemporalSplit:
    """Create one train/validation/test chronological split.

    ``purge`` removes observations before validation/test boundaries whose
    overlapping labels could reach into the next split. ``embargo`` creates a
    post-boundary buffer before the next split begins.
    """
    _validate_order(frame)
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    n = len(frame)
    if n < min_samples:
        raise ValueError(f"Need at least {min_samples} samples, got {n}")

    train_boundary = int(np.floor(n * train_size))
    val_boundary = int(np.floor(n * (train_size + val_size)))
    gap_before = max(0, int(purge))
    gap_after = max(0, int(embargo))

    train_idx = np.arange(0, max(0, train_boundary - gap_before))
    val_idx = np.arange(
        min(n, train_boundary + gap_after),
        max(train_boundary + gap_after, val_boundary - gap_before),
    )
    test_idx = np.arange(min(n, val_boundary + gap_after), n)

    _ensure_non_empty(train_idx, val_idx, test_idx)
    return _build_split(frame, train_idx, val_idx, test_idx, purge=purge, embargo=embargo)


def walk_forward_splits(
    frame: pd.DataFrame,
    train_length: int,
    val_length: int,
    test_length: int,
    step_length: int | None = None,
    expanding: bool = True,
    purge: int = 0,
    embargo: int = 0,
) -> Iterator[TemporalSplit]:
    """Yield expanding- or rolling-window walk-forward splits."""
    _validate_order(frame)
    n = len(frame)
    step = step_length or test_length
    start = 0

    while True:
        train_start = 0 if expanding else start
        train_boundary = start + train_length
        val_start = train_boundary + embargo
        val_end = val_start + val_length
        test_start = val_end + embargo
        test_end = test_start + test_length
        if test_end > n:
            break

        train_idx = np.arange(train_start, max(train_start, train_boundary - purge))
        val_idx = np.arange(val_start, max(val_start, val_end - purge))
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) and len(val_idx) and len(test_idx):
            yield _build_split(frame, train_idx, val_idx, test_idx, purge=purge, embargo=embargo)
        start += step


def global_chronological_split(
    panel: pd.DataFrame,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    purge: int = 0,
    embargo: int = 0,
) -> GlobalDateSplit:
    """Split a pooled panel on shared calendar dates rather than row positions.

    Purge drops the last ``purge`` global dates before validation and test. The
    embargo drops the first ``embargo`` dates in the following segment. Thus a
    target that overlaps h future periods cannot carry information across a
    global panel boundary.
    """
    _validate_order(panel.sort_index())
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    dates = pd.DatetimeIndex(panel.index.unique()).sort_values()
    n = len(dates)
    train_boundary = int(np.floor(n * train_size))
    val_boundary = int(np.floor(n * (train_size + val_size)))
    train_dates = dates[: max(0, train_boundary - purge)]
    val_dates = dates[train_boundary + embargo : max(train_boundary + embargo, val_boundary - purge)]
    test_dates = dates[val_boundary + embargo :]
    if not len(train_dates) or not len(val_dates) or not len(test_dates):
        raise ValueError("Global date split is empty after purge/embargo")
    return GlobalDateSplit(train_dates, val_dates, test_dates, purge=purge, embargo=embargo)


def global_walk_forward_splits(
    panel: pd.DataFrame,
    train_length: int,
    val_length: int,
    test_length: int,
    step_length: int | None = None,
    expanding: bool = True,
    purge: int = 0,
    embargo: int = 0,
) -> Iterator[GlobalDateSplit]:
    """Yield pooled walk-forward folds using common chronological date cutoffs."""
    dates = pd.DatetimeIndex(panel.index.unique()).sort_values()
    step = step_length or test_length
    start = 0
    while True:
        train_start = 0 if expanding else start
        train_boundary = start + train_length
        val_start = train_boundary + embargo
        val_end = val_start + val_length
        test_start = val_end + embargo
        test_end = test_start + test_length
        if test_end > len(dates):
            break
        train_dates = dates[train_start : max(train_start, train_boundary - purge)]
        val_dates = dates[val_start : max(val_start, val_end - purge)]
        test_dates = dates[test_start:test_end]
        if len(train_dates) and len(val_dates) and len(test_dates):
            yield GlobalDateSplit(train_dates, val_dates, test_dates, purge=purge, embargo=embargo)
        start += step


def required_global_purge_for_asset_horizon(
    panel: pd.DataFrame,
    identifier: str,
    horizon: int,
) -> int:
    """Translate an asset-session horizon into a safe number of global dates.

    A pooled panel can combine seven-day crypto with business-day instruments.
    Purging only ``horizon`` global dates is then too short for a label spanning
    ``horizon`` observed ETF sessions. The returned span is conservative across
    every represented asset calendar in the supplied target-eligible panel.
    """
    if identifier not in panel.columns:
        raise KeyError(f"Panel identifier not found: {identifier}")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    dates = pd.DatetimeIndex(panel.index.unique()).sort_values()
    positions = pd.Series(np.arange(len(dates), dtype=np.int64), index=dates)
    required = int(horizon)
    for _, asset in panel.groupby(identifier, observed=True, sort=False):
        asset_dates = pd.DatetimeIndex(asset.index.unique()).sort_values()
        if len(asset_dates) <= horizon:
            continue
        global_positions = positions.loc[asset_dates].to_numpy(dtype=np.int64)
        required = max(
            required,
            int(np.max(global_positions[horizon:] - global_positions[:-horizon])),
        )
    return required


def required_global_purge_at_boundaries(
    panel: pd.DataFrame,
    identifier: str,
    horizon: int,
    boundaries: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> int:
    """Return the exact global-date purge needed at fixed split boundaries.

    Each boundary is ``(unpurged_source_end, next_split_start)``. Historical
    gaps far from a boundary do not inflate the result, while every source
    endpoint whose ``horizon``-th observed asset session reaches the next split
    is removed.
    """
    if identifier not in panel.columns:
        raise KeyError(f"Panel identifier not found: {identifier}")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    global_dates = pd.DatetimeIndex(panel.index.unique()).sort_values()
    global_positions = pd.Series(np.arange(len(global_dates), dtype=np.int64), index=global_dates)
    required = 0
    for source_end, next_start in boundaries:
        source_end_position = int(global_positions.loc[pd.Timestamp(source_end)])
        for _, asset in panel.groupby(identifier, observed=True, sort=False):
            asset_dates = pd.DatetimeIndex(asset.index.unique()).sort_values()
            if len(asset_dates) <= horizon:
                continue
            endpoints = asset_dates[:-horizon]
            future_ends = asset_dates[horizon:]
            crossing = endpoints[(endpoints <= source_end) & (future_ends >= next_start)]
            if len(crossing):
                earliest_position = int(global_positions.loc[crossing.min()])
                required = max(required, source_end_position - earliest_position + 1)
    return required


def local_split_from_global_dates(frame: pd.DataFrame, split: GlobalDateSplit) -> TemporalSplit:
    """Project pooled global cutoffs onto one asset without changing their time order."""
    _validate_order(frame)
    train_idx = np.flatnonzero(frame.index.isin(split.train_dates))
    val_idx = np.flatnonzero(frame.index.isin(split.val_dates))
    test_idx = np.flatnonzero(frame.index.isin(split.test_dates))
    _ensure_non_empty(train_idx, val_idx, test_idx)
    return _build_split(frame, train_idx, val_idx, test_idx, split.purge, split.embargo)


def _validate_order(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Temporal splits require a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Temporal splits require index sorted in ascending time order")


def _ensure_non_empty(*arrays: np.ndarray) -> None:
    if any(len(array) == 0 for array in arrays):
        raise ValueError(
            "Split produced an empty train, validation or test segment. "
            "Reduce purge/embargo or adjust split lengths."
        )


def _build_split(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    purge: int,
    embargo: int,
) -> TemporalSplit:
    return TemporalSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_start=frame.index[train_idx[0]],
        train_end=frame.index[train_idx[-1]],
        val_start=frame.index[val_idx[0]],
        val_end=frame.index[val_idx[-1]],
        test_start=frame.index[test_idx[0]],
        test_end=frame.index[test_idx[-1]],
        purge=purge,
        embargo=embargo,
    )
