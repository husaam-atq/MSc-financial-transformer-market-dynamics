from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.targets.hourly_targets import add_hourly_targets


def test_hourly_direction_and_realized_volatility_use_future_hours_only() -> None:
    close = np.arange(100.0, 140.0)
    frame = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 100.0, "Ticker": "BTC/USDT"},
        index=pd.date_range("2024-01-01", periods=len(close), freq="h", name="Date"),
    )
    targetted = add_hourly_targets(frame)
    expected_return = close[4] / close[0] - 1.0
    expected_volatility = np.sqrt(np.sum(np.log(close[1:5] / close[:4]) ** 2))

    assert np.isclose(targetted.iloc[0]["target_forward_return_4h"], expected_return)
    assert targetted.iloc[0]["target_direction_4h"] == 1.0
    assert np.isclose(targetted.iloc[0]["target_realized_vol_4h"], expected_volatility)
    assert np.isnan(targetted.iloc[-1]["target_direction_1h"])
