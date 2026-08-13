from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_dynamics.features.market_structure import (
    downside_breadth_features,
    equal_family_market_return,
    fit_downside_thresholds,
    fit_upper_tail_threshold,
    future_max_loss_target,
    rolling_market_structure_features,
    upper_tail_event,
)


def _correlated_returns(*, collapsed: bool, rows: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    market = rng.normal(size=rows)
    family_a = rng.normal(size=rows)
    family_b = rng.normal(size=rows)
    values: dict[str, np.ndarray] = {}
    for asset in range(6):
        family = family_a if asset < 3 else family_b
        if collapsed:
            signal = 0.95 * market + 0.05 * family
        else:
            signal = 0.10 * market + 0.85 * family
        values[f"A{asset}"] = 0.01 * (signal + 0.10 * rng.normal(size=rows))
    index = pd.bdate_range("2020-01-01", periods=rows, name="Date")
    return pd.DataFrame(values, index=index)


def test_spectral_features_identify_dimensional_collapse_and_preserve_time() -> None:
    families = {f"A{i}": "left" if i < 3 else "right" for i in range(6)}
    block = rolling_market_structure_features(
        _correlated_returns(collapsed=False), window=60, families=families
    )
    collapsed = rolling_market_structure_features(
        _correlated_returns(collapsed=True), window=60, families=families
    )

    assert block.index.equals(_correlated_returns(collapsed=False).index)
    assert block.iloc[:59].isna().all().all()
    assert collapsed["normalized_effective_rank"].iloc[-1] < block[
        "normalized_effective_rank"
    ].iloc[-1]
    assert collapsed["largest_eigenvalue_share"].iloc[-1] > block[
        "largest_eigenvalue_share"
    ].iloc[-1]
    assert block["fixed_family_weighted_modularity"].iloc[-1] > collapsed[
        "fixed_family_weighted_modularity"
    ].iloc[-1]
    assert block["cross_family_positive_mixing"].iloc[-1] < collapsed[
        "cross_family_positive_mixing"
    ].iloc[-1]
    assert np.isfinite(block["correlation_turnover"].iloc[-1])
    assert np.isfinite(block["mst_mean_edge_distance"].iloc[-1])
    assert np.isfinite(block["normalized_effective_rank_change_20d"].iloc[-1])


def test_rolling_structure_rejects_incomplete_or_short_matrices() -> None:
    returns = _correlated_returns(collapsed=False, rows=20)
    with pytest.raises(ValueError, match="at least 60"):
        rolling_market_structure_features(returns, window=60)

    returns.loc[returns.index[5], "A0"] = np.nan
    with pytest.raises(ValueError, match="no missing"):
        rolling_market_structure_features(returns, window=10)


def test_rolling_structure_does_not_use_future_returns() -> None:
    returns = _correlated_returns(collapsed=False, rows=100)
    changed = returns.copy()
    changed.iloc[-1] = 100.0

    base_features = rolling_market_structure_features(returns, window=40)
    changed_features = rolling_market_structure_features(changed, window=40)

    cutoff = returns.index[-2]
    pd.testing.assert_series_equal(
        base_features.loc[cutoff], changed_features.loc[cutoff], check_names=False
    )


def test_downside_thresholds_are_training_only_and_breadth_is_pair_normalized() -> None:
    index = pd.bdate_range("2024-01-01", periods=6, name="Date")
    returns = pd.DataFrame(
        {
            "A": [-0.03, -0.02, 0.01, -0.50, 0.02, 0.03],
            "B": [-0.04, -0.01, 0.02, -0.60, 0.01, 0.04],
            "C": [-0.02, -0.01, 0.01, 0.02, 0.03, 0.04],
        },
        index=index,
    )
    thresholds = fit_downside_thresholds(returns.iloc[:3], quantile=0.25)
    thresholds_after_future_change = fit_downside_thresholds(
        returns.assign(A=returns["A"].where(returns.index < index[3], -99.0)).iloc[:3],
        quantile=0.25,
    )
    pd.testing.assert_series_equal(thresholds, thresholds_after_future_change)

    features = downside_breadth_features(returns, thresholds, change_lag=2)
    assert features.loc[index[3], "downside_breadth"] == pytest.approx(2 / 3)
    assert features.loc[index[3], "downside_coexceedance"] == pytest.approx(1 / 3)
    assert features.iloc[:2, 2:].isna().all().all()


def test_equal_family_return_weights_families_not_asset_count() -> None:
    index = pd.bdate_range("2024-01-01", periods=2, name="Date")
    returns = pd.DataFrame({"A": 0.10, "B": 0.10, "C": 0.00}, index=index)
    result = equal_family_market_return(returns, {"A": "equity", "B": "equity", "C": "bond"})

    assert result.name == "equal_family_market_return"
    assert result.index.equals(index)
    assert np.allclose(result, 0.05)


def test_future_max_loss_uses_only_t_plus_one_through_horizon() -> None:
    index = pd.date_range("2024-01-01", periods=5, name="Date")
    simple_returns = pd.Series([0.50, 0.10, -0.20, 0.05, 0.10], index=index)
    target = future_max_loss_target(simple_returns, horizon=2, returns_are_log=False)

    # At t0 the future path is +10%, then -12% relative to the t0 origin.
    assert target.loc[index[0]] == pytest.approx(0.12)
    # At t1 the future path starts at -20%, then recovers to -16%.
    assert target.loc[index[1]] == pytest.approx(0.20)
    assert target.iloc[-2:].isna().all()

    changed = simple_returns.copy()
    changed.iloc[0] = -0.90
    changed_target = future_max_loss_target(changed, horizon=2, returns_are_log=False)
    assert changed_target.loc[index[0]] == pytest.approx(target.loc[index[0]])


def test_upper_tail_event_threshold_is_fit_on_training_target_only() -> None:
    index = pd.bdate_range("2024-01-01", periods=6, name="Date")
    target = pd.Series([0.01, 0.02, 0.03, 0.04, 1.0, np.nan], index=index, name="loss")
    threshold = fit_upper_tail_threshold(target.iloc[:4], quantile=0.75)
    event = upper_tail_event(target, threshold)

    assert threshold == pytest.approx(0.0325)
    assert event.loc[index[3]] == 1.0
    assert event.loc[index[0]] == 0.0
    assert np.isnan(event.loc[index[-1]])
