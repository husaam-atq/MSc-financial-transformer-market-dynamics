from __future__ import annotations

import pandas as pd

from market_dynamics.models.volatility import _garch_return_column, parse_volatility_target


def test_garch_uses_hourly_return_column_for_hourly_targets() -> None:
    spec = parse_volatility_target("target_log_realized_vol_4h")
    frame = pd.DataFrame({"hourly_log_return": [0.01, -0.02, 0.005]})

    assert spec is not None
    assert _garch_return_column(frame, spec) == "hourly_log_return"


def test_garch_uses_daily_return_column_for_daily_targets() -> None:
    spec = parse_volatility_target("target_log_realized_vol_5d")
    frame = pd.DataFrame({"log_return": [0.01, -0.02, 0.005]})

    assert spec is not None
    assert _garch_return_column(frame, spec) == "log_return"
