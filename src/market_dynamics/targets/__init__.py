"""Supervised financial target builders."""

from market_dynamics.targets.continuous import (
    fit_ordinal_severity,
    future_downside_realized_volatility,
    future_maximum_adverse_loss,
    future_realized_volatility,
    future_rolling_maximum,
    future_state_change,
)
from market_dynamics.targets.hourly_targets import add_hourly_targets, hourly_target_columns
from market_dynamics.targets.make_targets import add_targets

__all__ = [
    "add_hourly_targets",
    "add_targets",
    "fit_ordinal_severity",
    "future_downside_realized_volatility",
    "future_maximum_adverse_loss",
    "future_realized_volatility",
    "future_rolling_maximum",
    "future_state_change",
    "hourly_target_columns",
]
