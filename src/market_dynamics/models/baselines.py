"""Classical Phase 1 baseline models for classification and regression."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
_WARNED_OPTIONAL: set[str] = set()


@dataclass
class ModelPrediction:
    """Container for a model's predictions on one evaluation split."""

    model_name: str
    y_pred: np.ndarray
    y_score: np.ndarray | None = None


def classification_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    config: dict[str, Any],
) -> list[ModelPrediction]:
    """Fit Phase 1 classifiers on training data and predict an evaluation set."""
    random_state = int(config.get("models", {}).get("random_state", 42))
    y_train = y_train.astype(int)
    predictions: list[ModelPrediction] = [
        _majority_class_prediction(y_train, len(X_eval)),
        _previous_direction_prediction(X_eval, y_train),
    ]

    if y_train.nunique() < 2:
        LOGGER.warning("Skipping learned classifiers because y_train has one class")
        return predictions

    model_config = config.get("models", {})
    logit_config = model_config.get("logistic_regression", {})
    rf_config = model_config.get("random_forest", {})
    optional = model_config.get("optional_models", {})

    classifiers: list[tuple[str, Any, bool]] = [
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=int(logit_config.get("max_iter", 3000)),
                class_weight=logit_config.get("class_weight", "balanced"),
                random_state=random_state,
            ),
            True,
        ),
        (
            "elastic_net_logistic",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=1.0,
                max_iter=int(logit_config.get("max_iter", 3000)),
                class_weight=logit_config.get("class_weight", "balanced"),
                random_state=random_state,
            ),
            True,
        ),
        (
            "random_forest_classifier",
            RandomForestClassifier(
                n_estimators=int(rf_config.get("n_estimators", 300)),
                max_depth=rf_config.get("max_depth", 6),
                min_samples_leaf=int(rf_config.get("min_samples_leaf", 20)),
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            False,
        ),
    ]

    classifiers.extend(_optional_classifiers(optional, random_state))
    for name, estimator, scale in classifiers:
        prediction = _fit_predict_estimator(name, estimator, X_train, y_train, X_eval, scale=scale)
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def regression_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    config: dict[str, Any],
) -> list[ModelPrediction]:
    """Fit Phase 1 regression baselines on training data and predict an evaluation set."""
    random_state = int(config.get("models", {}).get("random_state", 42))
    rf_config = config.get("models", {}).get("random_forest", {})
    optional = config.get("models", {}).get("optional_models", {})
    predictions: list[ModelPrediction] = [
        ModelPrediction("historical_mean", np.full(len(X_eval), float(y_train.mean())))
    ]

    regressors: list[tuple[str, Any, bool]] = [
        (
            "random_forest_regressor",
            RandomForestRegressor(
                n_estimators=int(rf_config.get("n_estimators", 300)),
                max_depth=rf_config.get("max_depth", 6),
                min_samples_leaf=int(rf_config.get("min_samples_leaf", 20)),
                random_state=random_state,
                n_jobs=-1,
            ),
            False,
        )
    ]
    regressors.extend(_optional_regressors(optional, random_state))
    for name, estimator, scale in regressors:
        prediction = _fit_predict_estimator(name, estimator, X_train, y_train, X_eval, scale=scale)
        if prediction is not None:
            predictions.append(prediction)
    return predictions


def _majority_class_prediction(y_train: pd.Series, n_eval: int) -> ModelPrediction:
    majority = int(y_train.mode().iloc[0])
    score = np.full(n_eval, float(y_train.mean()))
    return ModelPrediction("majority_class", np.full(n_eval, majority), score)


def _previous_direction_prediction(X_eval: pd.DataFrame, y_train: pd.Series) -> ModelPrediction:
    majority = int(y_train.mode().iloc[0])
    if "return_close" not in X_eval.columns:
        return ModelPrediction("previous_direction", np.full(len(X_eval), majority))
    previous = X_eval["return_close"]
    y_pred = np.where(previous.notna(), (previous > 0.0).astype(int), majority)
    y_score = np.where(previous.notna(), (previous > 0.0).astype(float), float(y_train.mean()))
    return ModelPrediction("previous_direction", y_pred.astype(int), y_score.astype(float))


def _fit_predict_estimator(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    scale: bool,
) -> ModelPrediction | None:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    try:
        pipeline.fit(_clean(X_train), y_train)
        y_pred = pipeline.predict(_clean(X_eval))
        y_score = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(_clean(X_eval))
            if proba.ndim == 2 and proba.shape[1] > 1:
                y_score = proba[:, 1]
        return ModelPrediction(name, np.asarray(y_pred), y_score)
    except Exception as exc:
        LOGGER.warning("Skipping model %s after fit/predict failure: %s", name, exc)
        return None


def _optional_classifiers(optional: dict[str, Any], random_state: int) -> list[tuple[str, Any, bool]]:
    models: list[tuple[str, Any, bool]] = []
    if optional.get("xgboost", True):
        try:
            from xgboost import XGBClassifier

            models.append(
                (
                    "xgboost_classifier",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    False,
                )
            )
        except ImportError:
            _warn_optional_missing("xgboost", "XGBoost classifier")

    if optional.get("lightgbm", True):
        try:
            from lightgbm import LGBMClassifier

            models.append(
                (
                    "lightgbm_classifier",
                    LGBMClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    False,
                )
            )
        except ImportError:
            _warn_optional_missing("lightgbm", "LightGBM classifier")

    if optional.get("catboost", True):
        try:
            from catboost import CatBoostClassifier

            models.append(
                (
                    "catboost_classifier",
                    CatBoostClassifier(
                        iterations=300,
                        depth=4,
                        learning_rate=0.03,
                        loss_function="Logloss",
                        random_seed=random_state,
                        verbose=False,
                    ),
                    False,
                )
            )
        except ImportError:
            _warn_optional_missing("catboost", "CatBoost classifier")

    return models


def _optional_regressors(optional: dict[str, Any], random_state: int) -> list[tuple[str, Any, bool]]:
    models: list[tuple[str, Any, bool]] = []
    if optional.get("xgboost", True):
        try:
            from xgboost import XGBRegressor

            models.append(
                (
                    "xgboost_regressor",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    False,
                )
            )
        except ImportError:
            _warn_optional_missing("xgboost", "XGBoost regressor")

    if optional.get("lightgbm", True):
        try:
            from lightgbm import LGBMRegressor

            models.append(
                (
                    "lightgbm_regressor",
                    LGBMRegressor(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    False,
                )
            )
        except ImportError:
            _warn_optional_missing("lightgbm", "LightGBM regressor")

    if optional.get("catboost", True):
        try:
            from catboost import CatBoostRegressor

            models.append(
                (
                    "catboost_regressor",
                    CatBoostRegressor(
                        iterations=300,
                        depth=4,
                        learning_rate=0.03,
                        loss_function="RMSE",
                        random_seed=random_state,
                        verbose=False,
                    ),
                    False,
                )
            )
        except ImportError:
            _warn_optional_missing("catboost", "CatBoost regressor")
    return models


def _warn_optional_missing(package: str, model_label: str) -> None:
    key = f"{package}:{model_label}"
    if key not in _WARNED_OPTIONAL:
        LOGGER.warning(
            "Optional dependency %s is not installed; skipping %s",
            package,
            model_label,
        )
        _WARNED_OPTIONAL.add(key)


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)
