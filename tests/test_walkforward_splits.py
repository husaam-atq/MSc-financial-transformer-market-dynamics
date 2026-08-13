from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from market_dynamics.experiments.run_walkforward_robustness import _three_walkforward_folds
from market_dynamics.splits.temporal import (
    global_walk_forward_splits,
    required_global_purge_at_boundaries,
    required_global_purge_for_asset_horizon,
)


def test_global_walkforward_dates_preserve_order_and_purge_gaps() -> None:
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    panel = pd.DataFrame({"Ticker": ["A"] * 80 + ["B"] * 80}, index=dates.append(dates))
    folds = list(global_walk_forward_splits(panel, train_length=30, val_length=15, test_length=10, purge=3, embargo=2))

    assert len(folds) >= 2
    first = folds[0]
    assert first.train_end < first.val_dates[0] < first.test_dates[0]
    assert len(pd.date_range(first.train_end, first.val_dates[0], freq="D")) >= 1 + first.purge + first.embargo


def test_global_purge_expands_for_mixed_asset_calendars() -> None:
    global_dates = pd.date_range("2024-01-01", periods=100, freq="D")
    business_dates = pd.bdate_range("2024-01-01", periods=70)
    panel = pd.concat(
        [
            pd.DataFrame({"Ticker": "CRYPTO"}, index=global_dates),
            pd.DataFrame({"Ticker": "ETF"}, index=business_dates),
        ]
    ).sort_index()

    purge = required_global_purge_for_asset_horizon(panel, "Ticker", horizon=10)

    assert purge >= 14
    assert purge > 10

    fold = next(
        global_walk_forward_splits(
            panel,
            train_length=50,
            val_length=20,
            test_length=20,
            purge=purge,
            embargo=0,
        )
    )
    for _, asset in panel.groupby("Ticker", observed=True):
        asset_dates = pd.DatetimeIndex(asset.index.unique()).sort_values()
        train_endpoints = asset_dates[asset_dates.isin(fold.train_dates)]
        validation_start = fold.val_dates.min()
        for endpoint in train_endpoints:
            position = asset_dates.get_loc(endpoint)
            if position + 10 < len(asset_dates):
                assert asset_dates[position + 10] < validation_start


def test_boundary_purge_ignores_remote_history_gap() -> None:
    global_dates = pd.date_range("2020-01-01", periods=220, freq="D")
    regular = pd.bdate_range("2020-01-01", periods=155)
    remote_gap = regular.delete(slice(20, 50))
    panel = pd.concat(
        [
            pd.DataFrame({"Ticker": "CRYPTO"}, index=global_dates),
            pd.DataFrame({"Ticker": "ETF"}, index=remote_gap),
        ]
    ).sort_index()
    fold = next(global_walk_forward_splits(panel, 140, 35, 35, purge=0, embargo=1))

    boundary_purge = required_global_purge_at_boundaries(
        panel,
        "Ticker",
        10,
        [(fold.train_dates.max(), fold.val_dates.min()), (fold.val_dates.max(), fold.test_dates.min())],
    )
    global_maximum = required_global_purge_for_asset_horizon(panel, "Ticker", 10)

    assert boundary_purge < global_maximum
    assert boundary_purge >= 10


def test_final_split_config_has_exact_purge_embargo_and_no_label_crossings() -> None:
    phase6 = yaml.safe_load(
        Path("configs/phase6_config.yaml").read_text(encoding="utf-8")
    )["phase6"]
    robustness = yaml.safe_load(
        Path("configs/phase2c_robustness_config.yaml").read_text(encoding="utf-8")
    )["phase2c"]
    global_dates = pd.date_range("2020-01-01", periods=1_000, freq="D")
    business_dates = pd.bdate_range(global_dates.min(), global_dates.max())
    panel = pd.concat(
        [
            pd.DataFrame({"Ticker": "BTC-USD"}, index=global_dates),
            pd.DataFrame({"Ticker": "SPY"}, index=business_dates),
        ]
    ).sort_index()
    panel["target_stress_10d"] = 0

    folds = _three_walkforward_folds(
        panel,
        target="target_stress_10d",
        lookback=int(phase6["lookback"]),
        robustness=robustness,
        purge_override=int(phase6["corrected_purge"]),
    )

    assert int(phase6["corrected_purge"]) == 18
    assert int(robustness["embargo"]) == 1
    assert len(folds) == 3
    for fold in folds:
        assert fold.purge == 18
        assert fold.embargo == 1
        boundaries = (
            (fold.train_dates, fold.val_dates.min()),
            (fold.val_dates, fold.test_dates.min()),
        )
        for _, asset in panel.groupby("Ticker", observed=True):
            asset_dates = pd.DatetimeIndex(asset.index.unique()).sort_values()
            for source_dates, next_start in boundaries:
                endpoints = asset_dates[asset_dates.isin(source_dates)]
                crossing_count = 0
                for endpoint in endpoints:
                    position = int(asset_dates.get_loc(endpoint))
                    if position + 10 < len(asset_dates):
                        crossing_count += int(asset_dates[position + 10] >= next_start)
                assert crossing_count == 0
