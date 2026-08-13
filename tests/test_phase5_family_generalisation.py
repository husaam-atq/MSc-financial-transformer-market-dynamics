from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.evaluation.family_generalisation import (
    attach_family_mapping,
    chronological_validation_partitions,
    fit_family_postprocessors,
    summarize_family_predictions,
)


def _validation_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for family, offset in (("Crypto", 0.15), ("Equities", 0.05)):
        for index, date in enumerate(dates):
            y_true = int(index % 4 == 0)
            rows.append(
                {
                    "Date": date,
                    "asset_ticker": f"{family}_{index % 2}",
                    "family": family,
                    "y_true": y_true,
                    "raw_probability": min(0.95, max(0.05, offset + 0.6 * y_true + 0.01 * (index % 3))),
                }
            )
    return pd.DataFrame(rows)


def test_attach_family_mapping_rejects_unknown_asset() -> None:
    frame = pd.DataFrame({"asset_ticker": ["A", "B"]})
    mapping = pd.DataFrame({"ticker": ["A"], "asset_class": ["Crypto"], "family": ["Crypto"]})
    try:
        attach_family_mapping(frame, mapping)
    except ValueError as exc:
        assert "B" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("Unknown assets must fail family attachment")


def test_nested_validation_partitions_are_chronological_and_disjoint() -> None:
    partitions = chronological_validation_partitions(_validation_frame())
    observed_dates = [set(part["Date"]) for part in partitions.values()]
    assert observed_dates[0].isdisjoint(observed_dates[1])
    assert observed_dates[1].isdisjoint(observed_dates[2])
    assert max(observed_dates[0]) < min(observed_dates[1]) < min(observed_dates[2])


def test_family_postprocessing_uses_validation_only_and_produces_decisions() -> None:
    options = {
        "calibration_methods": ["raw", "platt"],
        "calibration_bins": 5,
        "threshold_metric": "f1",
        "threshold_grid_size": 21,
        "minimum_family_observations": 5,
        "minimum_family_positive_examples": 1,
    }
    postprocessors, calibration, thresholds, partitions = fit_family_postprocessors(_validation_frame(), options)
    assert set(postprocessors) == {
        "global_raw",
        "family_raw_threshold",
        "global_calibrated",
        "family_calibrated",
        "family_calibrated_global_threshold",
    }
    selected = postprocessors["family_calibrated"].apply(partitions["selection"])
    assert np.isfinite(selected["selected_probability"]).all()
    assert selected["decision"].isin([0, 1]).all()
    summary = summarize_family_predictions(selected, "family_calibrated", "validation_selection")
    assert {"family_macro", "non_crypto_example_weighted", "worst_family"}.issubset(summary["aggregation"])
    assert not calibration.empty and not thresholds.empty


def test_family_postprocessing_handles_non_contiguous_prediction_indices() -> None:
    options = {
        "calibration_methods": ["raw", "platt"],
        "calibration_bins": 5,
        "threshold_metric": "f1",
        "threshold_grid_size": 21,
        "minimum_family_observations": 5,
        "minimum_family_positive_examples": 1,
    }
    postprocessors, _, _, partitions = fit_family_postprocessors(_validation_frame(), options)
    non_contiguous = partitions["selection"].copy()
    non_contiguous.index = np.arange(100, 100 + 3 * len(non_contiguous), 3)
    selected = postprocessors["family_calibrated"].apply(non_contiguous)
    assert len(selected) == len(non_contiguous)
    assert np.isfinite(selected["selected_probability"]).all()
