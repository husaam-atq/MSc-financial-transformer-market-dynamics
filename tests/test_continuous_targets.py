import numpy as np
import pandas as pd

from market_dynamics.targets.continuous import (
    fit_ordinal_severity,
    future_downside_realized_volatility,
    future_maximum_adverse_loss,
    future_realized_volatility,
    future_rolling_maximum,
    future_state_change,
)


def test_continuous_targets_use_t_plus_one_through_h() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    price = pd.Series([100.0, 90.0, 95.0, 80.0, 120.0, 60.0], index=index)
    loss = future_maximum_adverse_loss(price, horizon=2)
    assert np.isclose(loss.iloc[0], 0.10)
    assert np.isclose(loss.iloc[1], 1.0 - 80.0 / 90.0)
    changed = price.copy()
    changed.iloc[0] = 1000.0
    assert np.isclose(future_maximum_adverse_loss(changed, horizon=2).iloc[0], 0.91)
    outside_horizon = price.copy()
    outside_horizon.iloc[3] = 1.0
    assert np.isclose(future_maximum_adverse_loss(outside_horizon, horizon=2).iloc[0], loss.iloc[0])


def test_future_volatility_alignment() -> None:
    returns = pd.Series([0.0, -0.1, 0.2, -0.3, 0.4])
    downside = future_downside_realized_volatility(returns, horizon=2)
    total = future_realized_volatility(returns, horizon=2)
    assert np.isclose(downside.iloc[0], 0.1)
    assert np.isclose(total.iloc[0], np.sqrt(0.1**2 + 0.2**2))


def test_future_state_helpers_and_train_only_ordinal_cutoffs() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    state = pd.Series(np.arange(8, dtype=float), index=index)
    assert future_rolling_maximum(state, horizon=2).iloc[0] == 2.0
    assert future_state_change(state, horizon=2).iloc[0] == 2.0
    labels, cutoffs = fit_ordinal_severity(state, index[:4], quantiles=(0.5, 0.75))
    assert cutoffs == (1.5, 2.25)
    assert labels.iloc[-1] == 2.0
