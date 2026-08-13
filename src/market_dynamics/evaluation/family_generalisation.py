"""Family-aware, validation-only post-processing for Phase 5.

The routines operate on persisted historical predictions. They never read the sealed
fresh holdout and keep calibration fitting, threshold fitting, and strategy selection
in chronological subsegments of historical validation data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss

from market_dynamics.evaluation.calibration import calibration_summary
from market_dynamics.evaluation.classification_postprocessing import (
    Calibrator,
    IdentityCalibrator,
    fit_probability_calibrator,
    validation_optimal_threshold,
)
from market_dynamics.evaluation.metrics import classification_metrics
from market_dynamics.evaluation.post_freeze import calibration_slope_intercept


@dataclass
class FamilyPostprocessor:
    """Fixed validation-fitted calibrators and decision thresholds."""

    strategy: str
    global_calibrator: Calibrator
    global_threshold: float
    family_calibrators: dict[str, Calibrator]
    family_thresholds: dict[str, float]
    family_methods: dict[str, str]
    fallback_families: set[str]

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply fixed family-aware post-processing without fitting on `frame`."""
        required = {"family", "y_true", "raw_probability"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Prediction frame missing columns: {sorted(missing)}")
        # GroupBy exposes index labels rather than positional offsets.  Resetting
        # here ensures that the vector writes below remain correct for persisted
        # prediction frames whose original index is non-contiguous.
        output = frame.copy().reset_index(drop=True)
        probability = np.full(len(output), np.nan, dtype=float)
        threshold = np.full(len(output), np.nan, dtype=float)
        for family, positions in output.groupby("family", observed=True).groups.items():
            indices = positions.to_numpy(dtype=int)
            calibrator = self.family_calibrators.get(str(family), self.global_calibrator)
            probability[indices] = calibrator.predict(output.iloc[indices]["raw_probability"].to_numpy(dtype=float))
            threshold[indices] = self.family_thresholds.get(str(family), self.global_threshold)
        output["selected_probability"] = np.clip(probability, 0.0, 1.0)
        output["selected_threshold"] = threshold
        output["decision"] = (output["selected_probability"] >= output["selected_threshold"]).astype(int)
        output["strategy"] = self.strategy
        return output


def attach_family_mapping(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach a verified family mapping to a prediction or endpoint frame."""
    if "asset_ticker" not in frame.columns:
        raise KeyError("Family attachment requires an asset_ticker column")
    required = {"ticker", "asset_class", "family"}
    missing = required.difference(mapping.columns)
    if missing:
        raise KeyError(f"Family mapping missing columns: {sorted(missing)}")
    output = frame.merge(
        mapping[["ticker", "asset_class", "family"]],
        how="left",
        left_on="asset_ticker",
        right_on="ticker",
        validate="many_to_one",
    )
    unknown = sorted(output.loc[output["family"].isna(), "asset_ticker"].dropna().unique())
    if unknown:
        raise ValueError(f"No family mapping for assets: {unknown}")
    return output.drop(columns=["ticker"])


def chronological_validation_partitions(
    frame: pd.DataFrame,
    calibration_fit_fraction: float = 0.50,
    threshold_fit_end_fraction: float = 0.75,
) -> dict[str, pd.DataFrame]:
    """Create non-overlapping calibration, threshold and strategy-validation periods."""
    if "Date" not in frame.columns:
        raise KeyError("Validation frame must include Date")
    working = frame.copy()
    working["Date"] = pd.to_datetime(working["Date"])
    dates = np.sort(working["Date"].dropna().unique())
    if len(dates) < 12:
        raise ValueError("Validation period has too few distinct dates for nested post-processing")
    if not 0.0 < calibration_fit_fraction < threshold_fit_end_fraction < 1.0:
        raise ValueError("Nested validation fractions must satisfy 0 < calibration < threshold_end < 1")
    first = max(1, int(np.floor(len(dates) * calibration_fit_fraction)))
    second = max(first + 1, int(np.floor(len(dates) * threshold_fit_end_fraction)))
    if second >= len(dates):
        second = len(dates) - 1
    labels = {
        "calibration_fit": dates[:first],
        "threshold_fit": dates[first:second],
        "selection": dates[second:],
    }
    result = {name: working[working["Date"].isin(values)].copy() for name, values in labels.items()}
    if any(part.empty for part in result.values()):
        raise ValueError("Nested validation partition is empty")
    return result


def fit_family_postprocessors(
    validation: pd.DataFrame,
    options: dict[str, Any],
) -> tuple[dict[str, FamilyPostprocessor], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit bounded global/family post-processing using chronological validation only."""
    required = {"Date", "family", "y_true", "raw_probability"}
    missing = required.difference(validation.columns)
    if missing:
        raise KeyError(f"Validation frame missing columns: {sorted(missing)}")
    partitions = chronological_validation_partitions(
        validation,
        calibration_fit_fraction=float(options.get("calibration_fit_fraction", 0.50)),
        threshold_fit_end_fraction=float(options.get("threshold_fit_end_fraction", 0.75)),
    )
    calibration_fit = partitions["calibration_fit"]
    threshold_fit = partitions["threshold_fit"]
    methods = [str(method) for method in options["calibration_methods"]]
    bins = int(options["calibration_bins"])
    threshold_metric = str(options["threshold_metric"])
    threshold_grid_size = int(options["threshold_grid_size"])
    minimum_n = int(options["minimum_family_observations"])
    minimum_positive = int(options["minimum_family_positive_examples"])

    global_calibrator, global_method, global_candidate_rows = _select_calibrator(
        calibration_fit,
        threshold_fit,
        methods,
        bins,
        family="__global__",
    )
    global_raw_threshold = _threshold_for_frame(threshold_fit, IdentityCalibrator(), threshold_metric, threshold_grid_size)
    global_calibrated_threshold = _threshold_for_frame(threshold_fit, global_calibrator, threshold_metric, threshold_grid_size)
    family_raw_thresholds: dict[str, float] = {}
    family_calibrators: dict[str, Calibrator] = {}
    family_thresholds: dict[str, float] = {}
    family_methods: dict[str, str] = {}
    fallback_families: set[str] = set()
    candidate_rows = list(global_candidate_rows)
    threshold_rows = [
        _threshold_record("global_raw", "__global__", "raw", global_raw_threshold, False, threshold_fit),
        _threshold_record("global_calibrated", "__global__", global_method, global_calibrated_threshold, False, threshold_fit),
    ]
    families = sorted(validation["family"].astype(str).unique())
    for family in families:
        fit_part = calibration_fit[calibration_fit["family"].eq(family)]
        threshold_part = threshold_fit[threshold_fit["family"].eq(family)]
        raw_threshold = _threshold_for_frame_or_fallback(
            threshold_part,
            IdentityCalibrator(),
            threshold_metric,
            threshold_grid_size,
            global_raw_threshold,
        )
        family_raw_thresholds[family] = raw_threshold
        eligible = _family_is_eligible(fit_part, threshold_part, minimum_n, minimum_positive)
        if eligible:
            calibrator, method, rows = _select_calibrator(fit_part, threshold_part, methods, bins, family)
            threshold = _threshold_for_frame(threshold_part, calibrator, threshold_metric, threshold_grid_size)
            family_calibrators[family] = calibrator
            family_thresholds[family] = threshold
            family_methods[family] = method
            candidate_rows.extend(rows)
        else:
            fallback_families.add(family)
            family_calibrators[family] = global_calibrator
            family_thresholds[family] = global_calibrated_threshold
            family_methods[family] = f"global_fallback:{global_method}"
        threshold_rows.extend(
            [
                _threshold_record("family_raw_threshold", family, "raw", raw_threshold, not eligible, threshold_part),
                _threshold_record("family_calibrated", family, family_methods[family], family_thresholds[family], not eligible, threshold_part),
            ]
        )

    identity = IdentityCalibrator()
    global_raw = FamilyPostprocessor("global_raw", identity, global_raw_threshold, {}, {}, {}, set())
    family_raw = FamilyPostprocessor("family_raw_threshold", identity, global_raw_threshold, {}, family_raw_thresholds, {family: "raw" for family in families}, set())
    global_calibrated = FamilyPostprocessor("global_calibrated", global_calibrator, global_calibrated_threshold, {}, {}, {}, set())
    family_calibrated = FamilyPostprocessor(
        "family_calibrated",
        global_calibrator,
        global_calibrated_threshold,
        family_calibrators,
        family_thresholds,
        family_methods,
        fallback_families,
    )
    family_calibrated_global_threshold = FamilyPostprocessor(
        "family_calibrated_global_threshold",
        global_calibrator,
        global_calibrated_threshold,
        family_calibrators,
        {family: global_calibrated_threshold for family in families},
        family_methods,
        fallback_families,
    )
    return (
        {
            item.strategy: item
            for item in [global_raw, family_raw, global_calibrated, family_calibrated, family_calibrated_global_threshold]
        },
        pd.DataFrame(candidate_rows),
        pd.DataFrame(threshold_rows),
        partitions,
    )


def summarize_family_predictions(frame: pd.DataFrame, source: str, split: str) -> pd.DataFrame:
    """Produce example-weighted, family-macro, asset-macro and worst-family rows."""
    required = {"family", "asset_ticker", "y_true", "selected_probability", "decision"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Cannot summarise family predictions; missing {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for family, part in frame.groupby("family", observed=True):
        rows.append({"source": source, "split": split, "aggregation": "family_example_weighted", "family": family, "n_assets": int(part["asset_ticker"].nunique()), **decision_metrics(part)})
    rows.append({"source": source, "split": split, "aggregation": "global_example_weighted", "family": "__all__", "n_assets": int(frame["asset_ticker"].nunique()), **decision_metrics(frame)})
    non_crypto = frame[~frame["family"].eq("Crypto")]
    if not non_crypto.empty:
        rows.append({"source": source, "split": split, "aggregation": "non_crypto_example_weighted", "family": "Non-crypto aggregate", "n_assets": int(non_crypto["asset_ticker"].nunique()), **decision_metrics(non_crypto)})
    per_asset: list[dict[str, object]] = []
    for asset, part in frame.groupby("asset_ticker", observed=True):
        per_asset.append({"asset_ticker": asset, "family": part["family"].iloc[0], **decision_metrics(part)})
    asset_metrics = pd.DataFrame(per_asset)
    family_rows = pd.DataFrame([row for row in rows if row["aggregation"] == "family_example_weighted"])
    if not asset_metrics.empty:
        for family, part in asset_metrics.groupby("family", observed=True):
            row = {"source": source, "split": split, "aggregation": "asset_macro", "family": family, "n_assets": int(len(part))}
            for metric in _MACRO_METRICS:
                row[metric] = float(part[metric].mean())
            rows.append(row)
    if not family_rows.empty:
        macro = {"source": source, "split": split, "aggregation": "family_macro", "family": "__all__", "n_assets": int(family_rows["family"].nunique())}
        worst = {"source": source, "split": split, "aggregation": "worst_family", "family": "__all__", "n_assets": int(family_rows["family"].nunique())}
        for metric in _MACRO_METRICS:
            macro[metric] = float(family_rows[metric].mean())
            worst[metric] = float(family_rows[metric].min())
        rows.extend([macro, worst])
    return pd.DataFrame(rows)


def decision_metrics(frame: pd.DataFrame, calibration_bins: int = 10) -> dict[str, float | bool]:
    """Return probability metrics while preserving precomputed family decisions."""
    y = frame["y_true"].to_numpy(dtype=float)
    probability = frame["selected_probability"].to_numpy(dtype=float)
    decision = frame["decision"].to_numpy(dtype=int)
    finite = np.isfinite(y) & np.isfinite(probability)
    y, probability, decision = y[finite].astype(int), probability[finite], decision[finite]
    if len(y) == 0:
        raise ValueError("No finite labels and probabilities for family diagnostics")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A single label was found.*", category=UserWarning)
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        base = classification_metrics(y, decision, probability)
        base["pr_auc"] = float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else np.nan
    calibration, _ = calibration_summary(y, probability, bins=int(calibration_bins))
    base.update(calibration)
    base["log_loss"] = float(log_loss(y, np.clip(probability, 1e-6, 1.0 - 1e-6), labels=[0, 1])) if len(np.unique(y)) == 2 else np.nan
    intercept, slope = calibration_slope_intercept(y, probability)
    base["calibration_intercept"] = intercept
    base["calibration_slope"] = slope
    base["true_negative"] = float(np.sum((y == 0) & (decision == 0)))
    base["false_positive"] = float(np.sum((y == 0) & (decision == 1)))
    base["false_negative"] = float(np.sum((y == 1) & (decision == 0)))
    base["true_positive"] = float(np.sum((y == 1) & (decision == 1)))
    base["degenerate_prediction"] = bool(
        base["prediction_positive_rate"] <= 0.05
        or base["prediction_positive_rate"] >= 0.95
        or base["prediction_unique_values"] < 2
    )
    return {key: float(value) if isinstance(value, np.floating) else value for key, value in base.items()}


def select_postprocessing_strategy(summary: pd.DataFrame) -> str:
    """Choose one strategy from the final validation-selection partition only."""
    required = {"source", "aggregation", "f1", "balanced_accuracy", "pr_auc", "brier_score"}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"Strategy summary missing columns: {sorted(missing)}")
    non_crypto = summary[summary["aggregation"].eq("non_crypto_example_weighted")].set_index("source")
    macro = summary[summary["aggregation"].eq("family_macro")].set_index("source")
    candidates = sorted(set(non_crypto.index).intersection(macro.index))
    if not candidates:
        raise ValueError("No complete validation strategy summaries available")
    ranked = sorted(
        candidates,
        key=lambda strategy: (
            -_nan_to_negative(non_crypto.loc[strategy, "f1"]),
            -_nan_to_negative(macro.loc[strategy, "balanced_accuracy"]),
            -_nan_to_negative(non_crypto.loc[strategy, "pr_auc"]),
            _nan_to_positive(non_crypto.loc[strategy, "brier_score"]),
            strategy,
        ),
    )
    return str(ranked[0])


def _select_calibrator(
    calibration_fit: pd.DataFrame,
    threshold_fit: pd.DataFrame,
    methods: list[str],
    bins: int,
    family: str,
) -> tuple[Calibrator, str, list[dict[str, object]]]:
    candidates: list[tuple[Calibrator, str, dict[str, float | bool]]] = []
    rows: list[dict[str, object]] = []
    y_fit = calibration_fit["y_true"].to_numpy(dtype=float)
    p_fit = calibration_fit["raw_probability"].to_numpy(dtype=float)
    y_threshold = threshold_fit["y_true"].to_numpy(dtype=float)
    p_threshold = threshold_fit["raw_probability"].to_numpy(dtype=float)
    for raw_method in methods:
        method = str(raw_method).lower()
        calibrator: Calibrator = IdentityCalibrator() if method == "raw" else fit_probability_calibrator(y_fit, p_fit, method)
        probability = calibrator.predict(p_threshold)
        metrics = _probability_metrics(y_threshold, probability, bins)
        candidates.append((calibrator, method, metrics))
        rows.append({"family": family, "calibration_method": method, "stage": "threshold_fit", "selected": False, **metrics})
    candidates.sort(key=lambda item: (_nan_to_positive(item[2]["brier_score"]), _nan_to_positive(item[2]["log_loss"]), _nan_to_positive(item[2]["expected_calibration_error"]), item[1]))
    calibrator, method, _ = candidates[0]
    for row in rows:
        row["selected"] = row["calibration_method"] == method
    return calibrator, method, rows


def _threshold_for_frame(frame: pd.DataFrame, calibrator: Calibrator, metric: str, grid_size: int) -> float:
    if frame.empty:
        raise ValueError("Cannot fit threshold on an empty validation partition")
    probability = calibrator.predict(frame["raw_probability"].to_numpy(dtype=float))
    threshold, _ = validation_optimal_threshold(frame["y_true"].to_numpy(dtype=float), probability, metric=metric, grid_size=grid_size)
    return float(threshold)


def _threshold_for_frame_or_fallback(frame: pd.DataFrame, calibrator: Calibrator, metric: str, grid_size: int, fallback: float) -> float:
    if frame.empty or frame["y_true"].nunique() < 2:
        return float(fallback)
    return _threshold_for_frame(frame, calibrator, metric, grid_size)


def _threshold_record(strategy: str, family: str, method: str, threshold: float, fallback: bool, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "strategy": strategy,
        "family": family,
        "calibration_method": method,
        "threshold": float(threshold),
        "global_fallback": bool(fallback),
        "threshold_fit_n": int(len(frame)),
        "threshold_fit_positive_count": int(frame["y_true"].sum()) if not frame.empty else 0,
    }


def _family_is_eligible(fit: pd.DataFrame, threshold: pd.DataFrame, minimum_n: int, minimum_positive: int) -> bool:
    return bool(
        len(fit) >= minimum_n
        and len(threshold) >= max(1, minimum_n // 2)
        and fit["y_true"].sum() >= minimum_positive
        and (len(fit) - fit["y_true"].sum()) >= minimum_positive
        and threshold["y_true"].nunique() == 2
    )


def _probability_metrics(y_true: np.ndarray, probability: np.ndarray, bins: int) -> dict[str, float | bool]:
    threshold = 0.5
    frame = pd.DataFrame({"y_true": y_true, "selected_probability": probability, "decision": (np.asarray(probability) >= threshold).astype(int)})
    return decision_metrics(frame, calibration_bins=bins)


def _nan_to_negative(value: object) -> float:
    return float(value) if value is not None and np.isfinite(value) else -np.inf


def _nan_to_positive(value: object) -> float:
    return float(value) if value is not None and np.isfinite(value) else np.inf


_MACRO_METRICS = ["f1", "balanced_accuracy", "roc_auc", "pr_auc", "brier_score", "prediction_positive_rate"]
