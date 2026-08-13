from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.evaluation.post_freeze import (
    aligned_probability_ensemble,
    asset_block_bootstrap_ci,
    asset_block_bootstrap_difference,
    fit_calibration_candidates,
    fit_variance_scale,
    gaussian_ensemble,
    gaussian_uncertainty_metrics,
)


def test_calibration_candidates_are_ranked_from_validation_probability_quality() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.25, 0.70, 0.80, 0.90])
    candidates = fit_calibration_candidates(y, probabilities, ["raw", "temperature", "platt"])
    assert {candidate.method for candidate in candidates} == {"raw", "temperature", "platt"}
    assert all(0.0 <= candidate.threshold <= 1.0 for candidate in candidates)
    assert candidates[0].validation_metrics["brier_score"] <= candidates[-1].validation_metrics["brier_score"]


def test_probability_ensemble_requires_identical_endpoints_and_labels() -> None:
    base = pd.DataFrame(
        {
            "split": ["test", "test"],
            "Date": pd.date_range("2024-01-01", periods=2),
            "source_index": [1, 2],
            "asset_id": [0, 0],
            "y_true": [0.0, 1.0],
            "raw_probability": [0.2, 0.8],
        }
    )
    other = base.copy()
    other["raw_probability"] = [0.4, 0.6]
    ensemble = aligned_probability_ensemble([base, other], "raw_probability")
    assert np.allclose(ensemble["ensemble_probability"], [0.3, 0.7])
    invalid = other.iloc[:1].copy()
    try:
        aligned_probability_ensemble([base, invalid], "raw_probability")
    except ValueError as exc:
        assert "identical endpoint membership" in str(exc)
    else:  # pragma: no cover - guards against silent partial joins
        raise AssertionError("Expected strict ensemble membership validation")


def test_gaussian_ensemble_and_validation_variance_scale() -> None:
    first = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "Date": pd.date_range("2024-01-01", periods=2),
            "source_index": [1, 2],
            "asset_id": [0, 0],
            "y_true": [0.0, 2.0],
            "prediction_mean": [0.0, 1.0],
            "prediction_variance": [1.0, 1.0],
        }
    )
    second = first.copy()
    second["prediction_mean"] = [0.0, 3.0]
    ensemble = gaussian_ensemble([first, second])
    assert np.allclose(ensemble["ensemble_mean"], [0.0, 2.0])
    assert np.allclose(ensemble["ensemble_variance"], [1.0, 2.0])
    scale = fit_variance_scale(ensemble["y_true"], ensemble["ensemble_mean"], ensemble["ensemble_variance"])
    assert scale > 0.0
    metrics = gaussian_uncertainty_metrics(ensemble["y_true"], ensemble["ensemble_mean"], ensemble["ensemble_variance"] * scale)
    assert 0.0 <= metrics["interval_95_coverage"] <= 1.0


def test_asset_block_bootstrap_ci_returns_ordered_interval() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": np.repeat([0, 1], 6),
            "Date": list(pd.date_range("2024-01-01", periods=6)) * 2,
            "value": np.arange(12, dtype=float),
        }
    )
    interval = asset_block_bootstrap_ci(frame, ("value",), lambda value: float(np.mean(value)), iterations=25, block_size=2)
    assert interval["ci_lower"] <= interval["estimate"] <= interval["ci_upper"]


def test_asset_block_bootstrap_difference_has_bounded_p_value() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": np.repeat([0, 1], 6),
            "Date": list(pd.date_range("2024-01-01", periods=6)) * 2,
            "candidate": np.arange(12, dtype=float),
            "baseline": np.arange(12, dtype=float) - 1.0,
        }
    )
    result = asset_block_bootstrap_difference(
        frame,
        ("candidate", "baseline"),
        lambda candidate, baseline: float(np.mean(candidate - baseline)),
        iterations=25,
        block_size=2,
    )
    assert 0.0 < result["bootstrap_two_sided_p"] <= 1.0
