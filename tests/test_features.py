from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.features.engineering import add_features


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Adj Close": close,
            "Volume": np.linspace(1000, 2000, len(close)),
            "Ticker": "TEST",
            "Provider": "unit",
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_features_do_not_change_when_future_price_changes() -> None:
    base = _ohlcv(np.linspace(100, 140, 80))
    changed = base.copy()
    changed.loc[changed.index[-1], "Adj Close"] = 999.0
    changed.loc[changed.index[-1], "Close"] = 999.0

    base_features = add_features(base)
    changed_features = add_features(changed)

    columns = ["return_close", "volatility_20d", "cum_return_20d", "ma_distance_20d"]
    pd.testing.assert_series_equal(
        base_features.loc[base.index[40], columns],
        changed_features.loc[base.index[40], columns],
        check_names=False,
    )


def test_feature_output_shape_preserves_rows() -> None:
    frame = _ohlcv(np.linspace(100, 140, 80))
    features = add_features(frame)
    assert len(features) == len(frame)
    assert {"return_close", "volatility_20d", "negative_return_streak"}.issubset(features.columns)


def test_features_do_not_backward_fill_future_values() -> None:
    frame = _ohlcv(np.linspace(100, 140, 80))
    frame.loc[frame.index[30], "Volume"] = 0.0
    frame.loc[frame.index[31], "Volume"] = 1000000.0

    features = add_features(frame)

    assert np.isnan(features.loc[frame.index[30], "volume_change"])
    assert not np.isclose(
        features.loc[frame.index[30], "volume_ma_ratio_20d"],
        features.loc[frame.index[31], "volume_ma_ratio_20d"],
        equal_nan=False,
    )


def test_rolling_features_ignore_union_calendar_placeholders() -> None:
    observed_dates = pd.bdate_range("2024-01-01", periods=30)
    all_dates = pd.date_range(observed_dates.min(), observed_dates.max(), freq="D")
    frame = pd.DataFrame(index=pd.DatetimeIndex(all_dates, name="Date"))
    frame["Ticker"] = "ETF"
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        frame[column] = np.nan
    price = np.arange(100.0, 130.0)
    frame.loc[observed_dates, "Open"] = price
    frame.loc[observed_dates, "High"] = price + 1.0
    frame.loc[observed_dates, "Low"] = price - 1.0
    frame.loc[observed_dates, "Close"] = price
    frame.loc[observed_dates, "Adj Close"] = price
    frame.loc[observed_dates, "Volume"] = 1000.0

    result = add_features(frame)

    assert result.loc[observed_dates, "volatility_20d"].notna().sum() > 0
    assert result.loc[result["Close"].isna(), "volatility_20d"].isna().all()
