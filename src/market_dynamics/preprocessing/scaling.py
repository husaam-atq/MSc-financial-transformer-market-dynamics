"""Preprocessing utilities that fit statistics on training data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainOnlyPreprocessor:
    """Median-impute and optionally standardise features using train data only."""

    scale: bool = True

    def __post_init__(self) -> None:
        # Keep an all-missing training feature as an explicit zero-imputed column.
        # This preserves the configured model schema when a point-in-time-safe
        # macro series starts only after the first training period.
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        self.pipeline = Pipeline(steps)
        self.feature_columns: list[str] | None = None

    def fit(self, X_train: pd.DataFrame) -> TrainOnlyPreprocessor:
        self.feature_columns = list(X_train.columns)
        self.pipeline.fit(_clean_features(X_train))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return self.pipeline.transform(_clean_features(X[self.feature_columns]))

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @property
    def imputer_statistics_(self) -> np.ndarray:
        return self.pipeline.named_steps["imputer"].statistics_

    @property
    def scaler_mean_(self) -> np.ndarray | None:
        scaler = self.pipeline.named_steps.get("scaler")
        return None if scaler is None else scaler.mean_


def make_preprocessing_pipeline(scale: bool = True) -> Pipeline:
    """Return a scikit-learn pipeline suitable for model-specific training."""
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)
