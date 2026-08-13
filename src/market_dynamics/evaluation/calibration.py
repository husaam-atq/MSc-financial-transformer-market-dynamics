"""Probability calibration statistics and reliability figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss


def calibration_summary(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return Brier score, ECE and bin-level reliability data."""
    actual = np.asarray(y_true, dtype=float)
    probability = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    valid = np.isfinite(actual) & np.isfinite(probability)
    actual, probability = actual[valid], probability[valid]
    if len(actual) == 0:
        raise ValueError("No finite classification probabilities for calibration")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, bins - 1)
    counts = np.bincount(bucket, minlength=bins)
    mean_prediction = np.full(bins, np.nan)
    observed_frequency = np.full(bins, np.nan)
    ece = 0.0
    for index in range(bins):
        mask = bucket == index
        if counts[index]:
            mean_prediction[index] = probability[mask].mean()
            observed_frequency[index] = actual[mask].mean()
            ece += counts[index] / len(actual) * abs(mean_prediction[index] - observed_frequency[index])
    return (
        {"brier_score": float(brier_score_loss(actual, probability)), "expected_calibration_error": float(ece), "n": float(len(actual))},
        {"counts": counts, "mean_prediction": mean_prediction, "observed_frequency": observed_frequency, "bin_edges": edges},
    )


def plot_reliability(reliability: dict[str, np.ndarray], path: str | Path, title: str) -> Path:
    """Create a reliability diagram from calibration_summary output."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mean_prediction = reliability["mean_prediction"]
    observed = reliability["observed_frequency"]
    valid = np.isfinite(mean_prediction) & np.isfinite(observed)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(mean_prediction[valid], observed[valid], marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=150)
    plt.close()
    return target
