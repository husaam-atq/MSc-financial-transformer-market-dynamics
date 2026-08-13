from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.evaluation.market_dynamics import (
    apply_volume_momentum_states,
    benjamini_hochberg,
    breadth_dispersion_results,
    fit_volume_momentum_thresholds,
    lead_lag_results,
)


def test_benjamini_hochberg_is_bounded_and_monotonic_in_rank_order() -> None:
    values = benjamini_hochberg([0.001, 0.02, 0.04, np.nan])
    assert np.all((values[:3] >= 0.0) & (values[:3] <= 1.0))
    assert np.isnan(values[3])
    assert values[0] <= values[1] <= values[2]


def test_volume_momentum_thresholds_are_fit_on_training_only() -> None:
    train = pd.DataFrame(
        {
            "family": ["Equities"] * 4,
            "volume_ma_ratio_20d": [0.0, 1.0, 2.0, 3.0],
            "cum_return_20d": [-0.3, -0.1, 0.1, 0.3],
        }
    )
    validation = pd.DataFrame(
        {
            "family": ["Equities"],
            "volume_ma_ratio_20d": [100.0],
            "cum_return_20d": [-100.0],
        }
    )
    thresholds = fit_volume_momentum_thresholds(train, "volume_ma_ratio_20d", "cum_return_20d", 0.25, 0.75)
    assert thresholds.loc[0, "volume_upper"] == 2.25
    applied = apply_volume_momentum_states(validation, thresholds, "volume_ma_ratio_20d", "cum_return_20d")
    assert applied.loc[0, "volume_momentum_state"] == "high_volume_negative_momentum"


def test_lead_lag_marks_only_sign_replicated_adjusted_associations_as_robust() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rows = []
    for split in ("train", "validation"):
        for index, date in enumerate(dates):
            signal = float(index)
            for family in ("A", "B"):
                rows.append(
                    {
                        "split": split,
                        "Date": date,
                        "family": family,
                        "mean_momentum": signal if family == "A" else -signal,
                        "future_stress_breadth": signal if family == "B" else -signal,
                    }
                )
    results = lead_lag_results(pd.DataFrame(rows), [1], minimum_observations=30)
    target = results[
        results["source_family"].eq("A")
        & results["destination_family"].eq("B")
        & results["lag_observed_sessions"].eq(1)
    ]
    assert target["robust_train_validation"].all()
    assert target["train_rho"].notna().all()
    assert target["validation_rho"].notna().all()


def test_breadth_summary_excludes_label_derived_stress_concentration() -> None:
    rows = []
    for index, date in enumerate(pd.date_range("2024-01-01", periods=40, freq="D")):
        for family, asset in (("A", "A1"), ("B", "B1")):
            rows.append(
                {
                    "split": "train",
                    "Date": date,
                    "family": family,
                    "asset_ticker": asset,
                    "return_close": 0.01 * (-1) ** index,
                    "cum_return_20d": 0.02 * (-1) ** index,
                    "downside_move_indicator": float(index % 2),
                    "target": float(index % 2),
                    "volume_momentum_state": "other",
                }
            )
    _, summary = breadth_dispersion_results(pd.DataFrame(rows), "target", "cum_return_20d")
    assert "family_stress_concentration" not in set(summary["measure"])
    assert {"train_rho", "validation_rho", "robust_train_validation"}.issubset(summary.columns)
