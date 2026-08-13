from __future__ import annotations

import pandas as pd

from market_dynamics.splits.temporal import chronological_split, walk_forward_splits


def _frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {"x": range(n)},
        index=pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n, freq="D"), name="Date"),
    )


def test_chronological_split_preserves_order() -> None:
    frame = _frame()
    split = chronological_split(frame, purge=5, embargo=2)

    assert split.train_idx.max() < split.val_idx.min()
    assert split.val_idx.max() < split.test_idx.min()
    assert frame.index[split.train_idx].max() < frame.index[split.val_idx].min()
    assert frame.index[split.val_idx].max() < frame.index[split.test_idx].min()


def test_purge_embargo_creates_gap_around_boundaries() -> None:
    frame = _frame()
    split = chronological_split(frame, train_size=0.6, val_size=0.2, test_size=0.2, purge=5, embargo=2)

    assert split.val_idx.min() - split.train_idx.max() - 1 == 7
    assert split.test_idx.min() - split.val_idx.max() - 1 == 7


def test_walk_forward_splits_expand_training_window() -> None:
    frame = _frame(120)
    splits = list(
        walk_forward_splits(
            frame,
            train_length=40,
            val_length=10,
            test_length=10,
            step_length=10,
            expanding=True,
            purge=2,
            embargo=1,
        )
    )

    assert len(splits) > 1
    assert splits[1].train_idx.min() == 0
    assert splits[1].train_idx.max() > splits[0].train_idx.max()
