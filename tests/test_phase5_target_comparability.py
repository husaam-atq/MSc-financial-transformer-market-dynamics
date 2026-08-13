from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.evaluation.target_comparability import (
    reconstruct_stress_components,
    summarize_target_comparability,
)
from market_dynamics.targets.make_targets import add_targets


def _asset_frame(ticker: str, prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Ticker": ticker,
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Adj Close": prices,
            "Volume": 100.0,
        }
    ).set_index("Date")


def test_reconstructed_components_exactly_match_stored_stress_target() -> None:
    prices = [100.0, 100.0, 99.0, 98.0, 88.0, 87.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0]
    source = _asset_frame("A", prices).reset_index()
    targeted = add_targets(
        source.set_index("Date"),
        {
            "targets": {
                "direction_horizons": [],
                "volatility_horizons": [],
                "stress": {
                    "enabled": True,
                    "horizon": 3,
                    "large_negative_return": -0.05,
                    "drawdown_threshold": -0.07,
                    "volatility_spike_multiplier": 2.0,
                },
            }
        },
    )
    components = reconstruct_stress_components(
        targeted,
        target_column="target_stress_3d",
        horizon=3,
        large_negative_return=-0.05,
        drawdown_threshold=-0.07,
        volatility_spike_multiplier=2.0,
        past_volatility_window=20,
    )
    valid = components[["stored_target", "reconstructed_target"]].dropna()
    assert not valid.empty
    assert np.array_equal(valid["stored_target"].to_numpy(), valid["reconstructed_target"].to_numpy())


def test_target_summary_uses_only_given_split_dates() -> None:
    components = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset_ticker": ["A", "A", "A"],
            "stored_target": [0.0, 1.0, 1.0],
            "future_return": [0.01, -0.08, -0.1],
            "future_drawdown": [0.0, -0.08, -0.1],
            "future_volatility": [0.1, 0.3, 0.4],
            "past_volatility": [0.1, 0.1, 0.1],
            "volatility_ratio": [1.0, 3.0, 4.0],
            "negative_return_component": [0.0, 1.0, 1.0],
            "drawdown_component": [0.0, 1.0, 1.0],
            "volatility_spike_component": [0.0, 1.0, 1.0],
        }
    )
    mapping = pd.DataFrame({"ticker": ["A"], "asset_class": ["Crypto"], "family": ["Crypto"]})
    summary = summarize_target_comparability(
        components,
        mapping,
        {"train": pd.DatetimeIndex([pd.Timestamp("2024-01-01")]), "validation": pd.DatetimeIndex([pd.Timestamp("2024-01-02")])},
        minimum_family_observations=1,
    )
    train = summary[(summary["split"] == "train") & (summary["family"] == "Crypto")].iloc[0]
    validation = summary[(summary["split"] == "validation") & (summary["family"] == "Crypto")].iloc[0]
    assert train["n_obs"] == 1 and train["stress_prevalence"] == 0.0
    assert validation["n_obs"] == 1 and validation["stress_prevalence"] == 1.0
