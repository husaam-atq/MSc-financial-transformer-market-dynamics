from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_dynamics.experiments.run_large_scale_screening import (
    _has_training_class_variation,
    _screening_block_resolved,
)
from market_dynamics.splits.temporal import chronological_split
from market_dynamics.targets.make_targets import (
    add_future_maximum_loss_target,
    add_targets,
    fit_asset_relative_stress_target,
)


def _featured(close: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": close,
            "Volume": 1000,
            "Ticker": "TEST",
            "Provider": "unit",
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_direction_target_shifting_uses_future_close_only() -> None:
    frame = _featured([100, 101, 102, 103, 104, 105, 90, 91, 92, 93, 94])
    targets = add_targets(frame, {"targets": {"direction_horizons": [1, 5, 10], "volatility_horizons": [5, 10]}})

    assert np.isclose(targets.iloc[0]["target_forward_return_1d"], 0.01)
    assert targets.iloc[0]["target_direction_1d"] == 1.0
    assert np.isclose(targets.iloc[0]["target_forward_return_5d"], 0.05)
    assert targets.iloc[0]["target_direction_5d"] == 1.0
    assert np.isclose(targets.iloc[0]["target_forward_return_10d"], -0.06)
    assert targets.iloc[0]["target_direction_10d"] == 0.0
    assert targets.iloc[5]["target_direction_1d"] == 0.0
    assert np.isnan(targets.iloc[-1]["target_direction_1d"])


def test_realized_volatility_uses_t_plus_1_through_t_plus_h() -> None:
    close = [100, 101, 103, 106, 110, 115, 121]
    frame = _featured(close)
    targets = add_targets(frame, {"targets": {"direction_horizons": [1], "volatility_horizons": [5]}})
    prices = pd.Series(close, dtype=float)
    log_returns = np.log(prices).diff()
    expected = np.sqrt(np.sum(np.square(log_returns.iloc[1:6])))

    assert np.isclose(targets.iloc[0]["target_realized_vol_5d"], expected)
    assert np.isclose(targets.iloc[0]["target_log_realized_vol_5d"], np.log(expected + 1e-12))


def test_stress_target_is_future_only_and_has_the_configured_horizon() -> None:
    frame = _featured([100.0] * 10 + [90.0] + [90.0] * 10)
    targets = add_targets(
        frame,
        {"targets": {"direction_horizons": [1], "volatility_horizons": [5], "stress": {"enabled": True, "horizon": 10, "large_negative_return": -0.05}}},
    )

    assert targets.iloc[0]["target_stress_10d"] == 1.0
    assert np.isnan(targets.iloc[-1]["target_stress_10d"])


def _stress_config(**overrides: float) -> dict[str, object]:
    stress = {
        "enabled": True,
        "horizon": 10,
        "large_negative_return": -0.05,
        "drawdown_threshold": -0.07,
        "volatility_spike_multiplier": 2.0,
    }
    stress.update(overrides)
    return {"targets": {"direction_horizons": [10], "volatility_horizons": [10], "stress": stress}}


def test_stress_target_terminal_return_branch_uses_t_plus_10() -> None:
    history = [100.0 + 0.1 * (-1) ** index for index in range(20)] + [100.0]
    future = np.linspace(99.4, 94.0, 10).tolist()
    targets = add_targets(
        _featured(history + future),
        _stress_config(drawdown_threshold=-1.0, volatility_spike_multiplier=1e9),
    )

    assert targets.iloc[20]["target_forward_return_10d"] == pytest.approx(-0.06)
    assert targets.iloc[20]["target_stress_10d"] == 1.0


def test_stress_target_drawdown_branch_uses_minimum_over_next_ten_sessions() -> None:
    history = [100.0 + 0.1 * (-1) ** index for index in range(20)] + [100.0]
    future = [99.0, 97.0, 92.0, 95.0, 97.0, 99.0, 100.0, 100.0, 100.0, 100.0]
    targets = add_targets(
        _featured(history + future),
        _stress_config(large_negative_return=-1.0, volatility_spike_multiplier=1e9),
    )

    assert targets.iloc[20]["target_forward_return_10d"] == pytest.approx(0.0)
    assert targets.iloc[20]["target_stress_10d"] == 1.0


def test_stress_target_volatility_branch_compares_future_with_past_only() -> None:
    history = [100.0 + 0.05 * (-1) ** index for index in range(20)] + [100.0]
    future = [102.0, 100.0] * 5
    frame = _featured(history + future + [25.0])
    targets = add_targets(
        frame,
        _stress_config(large_negative_return=-1.0, drawdown_threshold=-1.0),
    )
    changed_after_horizon = frame.copy()
    changed_after_horizon.iloc[-1, changed_after_horizon.columns.get_loc("Close")] = 400.0
    changed_after_horizon.iloc[-1, changed_after_horizon.columns.get_loc("Adj Close")] = 400.0
    changed_targets = add_targets(
        changed_after_horizon,
        _stress_config(large_negative_return=-1.0, drawdown_threshold=-1.0),
    )

    assert targets.iloc[20]["target_stress_10d"] == 1.0
    assert changed_targets.iloc[20]["target_stress_10d"] == targets.iloc[20]["target_stress_10d"]


def test_direction_and_volatility_targets_prefer_adjusted_close() -> None:
    frame = _featured([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    frame["Adj Close"] = [200.0, 190.0, 180.0, 170.0, 160.0, 150.0]
    targets = add_targets(
        frame,
        {"targets": {"direction_horizons": [1], "volatility_horizons": [5]}},
    )
    expected = np.sqrt(np.square(np.log(pd.Series(frame["Adj Close"], dtype=float)).diff().iloc[1:6]).sum())

    assert targets.iloc[0]["target_forward_return_1d"] == pytest.approx(-0.05)
    assert targets.iloc[0]["target_direction_1d"] == 0.0
    assert targets.iloc[0]["target_realized_vol_5d"] == pytest.approx(expected)


def test_targets_use_observed_sessions_not_union_calendar_placeholders() -> None:
    observed_dates = pd.bdate_range("2024-01-01", periods=25)
    all_dates = pd.date_range(observed_dates.min(), observed_dates.max(), freq="D")
    frame = pd.DataFrame(index=pd.DatetimeIndex(all_dates, name="Date"))
    frame["Ticker"] = "ETF"
    frame["Close"] = np.nan
    frame["Adj Close"] = np.nan
    prices = pd.Series(np.arange(100.0, 125.0), index=observed_dates)
    frame.loc[observed_dates, "Close"] = prices
    frame.loc[observed_dates, "Adj Close"] = prices

    result = add_targets(
        frame,
        {
            "targets": {
                "direction_horizons": [10],
                "volatility_horizons": [10],
                "stress": {"enabled": True, "horizon": 10},
            }
        },
    )

    observed = result.loc[observed_dates]
    assert observed["target_direction_10d"].notna().sum() == 15
    assert observed["target_stress_10d"].notna().sum() == 15
    assert result.loc[result["Close"].isna(), "target_stress_10d"].isna().all()


def test_single_class_training_split_is_identified_before_model_fitting() -> None:
    frame = _featured([100.0] * 80)
    frame["target_stress_10d"] = 0.0
    split = chronological_split(frame, train_size=0.6, val_size=0.2, test_size=0.2, purge=0, embargo=0)

    assert not _has_training_class_variation(frame, split, "target_stress_10d")


def test_screening_resume_only_skips_resolved_blocks() -> None:
    progress = pd.DataFrame(
        [
            {"track": "daily", "scope": "local", "asset": "SPY", "target": "target_direction_5d", "status": "completed"},
            {"track": "daily", "scope": "local", "asset": "SPY", "target": "target_direction_5d", "status": "completed"},
            {"track": "daily", "scope": "local", "asset": "IWM", "target": "target_direction_5d", "status": "failed"},
            {"track": "daily", "scope": "local", "asset": "UUP", "target": "target_stress_10d", "status": "skipped"},
        ]
    )

    assert _screening_block_resolved(progress, "daily", "local", "SPY", "target_direction_5d")
    assert _screening_block_resolved(progress, "daily", "local", "UUP", "target_stress_10d")
    assert not _screening_block_resolved(progress, "daily", "local", "IWM", "target_direction_5d")
    assert not _screening_block_resolved(progress, "daily", "pooled", "__pooled__", "target_direction_5d")


def test_asset_relative_target_fits_threshold_on_training_dates_only() -> None:
    frame = _featured([100, 99, 98, 97, 96, 95, 100, 99, 98, 97, 96, 95, 94, 93, 92, 70, 60, 50])
    dates = pd.DatetimeIndex(frame.index)
    fitted, definition = fit_asset_relative_stress_target(
        frame,
        dates[:8],
        horizon=3,
        quantile=0.90,
    )
    original_threshold = float(definition.thresholds["TEST"])
    changed = frame.copy()
    changed[["Close", "Adj Close"]] = changed[["Close", "Adj Close"]].astype(float)
    future_values = np.linspace(90.0, 1.0, 7)
    changed.loc[dates[11]:, "Close"] = future_values
    changed.loc[dates[11]:, "Adj Close"] = future_values
    _, changed_definition = fit_asset_relative_stress_target(
        changed,
        dates[:8],
        horizon=3,
        quantile=0.90,
    )
    assert float(changed_definition.thresholds["TEST"]) == pytest.approx(original_threshold)
    assert fitted.loc[dates[-3]:, "target_asset_relative_max_loss_10d_q90"].isna().all()


def test_asset_relative_target_uses_only_t_plus_one_through_horizon() -> None:
    frame = _featured([100.0, 90.0, 95.0, 80.0, 100.0, 100.0, 100.0, 100.0])
    dates = pd.DatetimeIndex(frame.index)
    targeted, _ = fit_asset_relative_stress_target(
        frame,
        dates[:4],
        horizon=2,
        quantile=0.5,
    )
    assert targeted.loc[dates[0], "target_future_maximum_loss_10d"] == pytest.approx(0.10)
    assert targeted.loc[dates[1], "target_future_maximum_loss_10d"] == pytest.approx(1.0 - 80.0 / 90.0)


def test_continuous_maximum_loss_target_uses_future_observed_sessions_only() -> None:
    frame = _featured([100.0, 90.0, 95.0, 80.0, 110.0, 105.0])
    dates = pd.DatetimeIndex(frame.index)
    targeted = add_future_maximum_loss_target(frame, horizon=3)

    assert targeted.loc[dates[0], "target_future_maximum_loss_3d"] == pytest.approx(0.20)
    assert targeted.loc[dates[1], "target_future_maximum_loss_3d"] == pytest.approx(1.0 - 80.0 / 90.0)
    assert targeted.loc[dates[2], "target_future_maximum_loss_3d"] == pytest.approx(1.0 - 80.0 / 95.0)
    assert targeted.iloc[-3:]["target_future_maximum_loss_3d"].isna().all()


def test_continuous_maximum_loss_target_does_not_backfill_missing_calendar_rows() -> None:
    frame = _featured([100.0, 90.0, 80.0, 70.0, 60.0])
    absent = frame.index[1]
    frame.loc[absent, ["Close", "Adj Close"]] = np.nan
    targeted = add_future_maximum_loss_target(frame, horizon=2)

    assert pd.isna(targeted.loc[absent, "target_future_maximum_loss_2d"])
    assert targeted.loc[frame.index[0], "target_future_maximum_loss_2d"] == pytest.approx(0.30)


def test_continuous_maximum_loss_uses_one_price_source_per_asset() -> None:
    frame = _featured([100.0, 95.0, 90.0, 85.0, 80.0])
    frame["Adj Close"] = [200.0, np.nan, 180.0, 170.0, 160.0]
    targeted = add_future_maximum_loss_target(frame, horizon=2)

    assert targeted.iloc[0]["target_future_maximum_loss_2d"] == pytest.approx(0.10)
