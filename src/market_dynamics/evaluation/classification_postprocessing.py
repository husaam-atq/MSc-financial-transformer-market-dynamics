"""Validation-only threshold tuning and probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score


def validation_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
    grid_size: int = 101,
) -> tuple[float, float]:
    """Choose a classification threshold using validation data only."""
    actual = np.asarray(y_true).astype(int)
    prob = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(prob)
    actual, prob = actual[valid], prob[valid]
    if len(actual) == 0:
        raise ValueError("No finite validation probabilities for threshold tuning")
    thresholds = np.linspace(0.0, 1.0, int(grid_size))
    scorer = f1_score if metric == "f1" else balanced_accuracy_score
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        predicted = (prob >= threshold).astype(int)
        score = float(scorer(actual, predicted, zero_division=0)) if metric == "f1" else float(scorer(actual, predicted))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


class Calibrator(Protocol):
    """Small protocol for post-hoc probability calibrators."""

    method: str

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities."""


@dataclass
class IdentityCalibrator:
    method: str = "none"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(probabilities, dtype=float)


@dataclass
class PlattCalibrator:
    model: LogisticRegression
    method: str = "platt"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_probability_logits(probabilities).reshape(-1, 1))[:, 1]


@dataclass
class IsotonicCalibrator:
    model: IsotonicRegression
    method: str = "isotonic"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict(_clip_probabilities(probabilities))


@dataclass
class TemperatureCalibrator:
    temperature: float
    method: str = "temperature"

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        logits = _probability_logits(probabilities) / self.temperature
        return 1.0 / (1.0 + np.exp(-logits))


def fit_probability_calibrator(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    method: str,
) -> Calibrator:
    """Fit a post-hoc calibrator using validation probabilities only."""
    method = method.lower()
    actual = np.asarray(y_true).astype(int)
    prob = _clip_probabilities(probabilities)
    valid = np.isfinite(prob)
    actual, prob = actual[valid], prob[valid]
    if method in {"none", "identity"}:
        return IdentityCalibrator()
    if len(np.unique(actual)) < 2:
        return IdentityCalibrator(method=f"{method}_skipped_single_class")
    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(_probability_logits(prob).reshape(-1, 1), actual)
        return PlattCalibrator(model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(prob, actual)
        return IsotonicCalibrator(model)
    if method == "temperature":
        return TemperatureCalibrator(_fit_temperature(actual, prob))
    raise ValueError(f"Unsupported calibration method: {method}")


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)


def _probability_logits(probabilities: np.ndarray) -> np.ndarray:
    prob = _clip_probabilities(probabilities)
    return np.log(prob / (1.0 - prob))


def _fit_temperature(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    logits = _probability_logits(probabilities)
    temperatures = np.linspace(0.5, 5.0, 91)
    best_temperature = 1.0
    best_loss = np.inf
    for temperature in temperatures:
        calibrated = 1.0 / (1.0 + np.exp(-(logits / temperature)))
        loss = -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1.0 - calibrated))
        if loss < best_loss:
            best_loss = float(loss)
            best_temperature = float(temperature)
    return best_temperature
