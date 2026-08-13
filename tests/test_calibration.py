from __future__ import annotations

import numpy as np

from market_dynamics.evaluation.calibration import calibration_summary


def test_calibration_summary_reports_brier_and_ece() -> None:
    summary, reliability = calibration_summary(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.6, 0.2]), bins=4)
    assert 0.0 <= summary["brier_score"] <= 1.0
    assert summary["expected_calibration_error"] >= 0.0
    assert reliability["counts"].sum() == 4
