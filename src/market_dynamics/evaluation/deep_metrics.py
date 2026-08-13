"""Metrics and figures for Phase 2 deep model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Phase 2 batches often run headlessly from a hidden process on Windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_dynamics.evaluation.metrics import classification_metrics, regression_metrics


def evaluate_deep_predictions(
    task: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Evaluate probabilities for classification or values for regression."""
    if task == "classification":
        y_pred = (np.asarray(predictions) >= 0.5).astype(int)
        return classification_metrics(y_true, y_pred, np.asarray(predictions))
    if task == "regression":
        return regression_metrics(y_true, np.asarray(predictions))
    raise ValueError(f"Unsupported task: {task}")


def plot_training_history(
    train_losses: list[float],
    validation_losses: list[float],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save train/validation loss curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_test_predictions(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: np.ndarray,
    task: str,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save chronological test prediction diagnostics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 4.5))
    if task == "classification":
        plt.plot(dates, predictions, label="Predicted probability", linewidth=1.0)
        plt.scatter(dates, y_true, label="Observed class", s=8, alpha=0.55)
        plt.ylim(-0.05, 1.05)
        plt.ylabel("Probability / class")
    else:
        plt.plot(dates, y_true, label="Actual", linewidth=1.2)
        plt.plot(dates, predictions, label="Prediction", linewidth=1.0)
        plt.ylabel("Target value")
    plt.xlabel("Date")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def prediction_frame(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: np.ndarray,
    source_indices: np.ndarray,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Create a durable prediction artefact with chronological identifiers."""
    frame = pd.DataFrame(
        {
            "Date": dates,
            "source_index": source_indices,
            "y_true": y_true,
            "prediction": predictions,
        }
    )
    for key, value in metadata.items():
        frame[key] = value
    return frame
