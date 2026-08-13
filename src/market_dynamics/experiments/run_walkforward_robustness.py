"""Phase 2C multi-seed walk-forward, uncertainty and calibration benchmark."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from market_dynamics.datasets.pooled_window_dataset import build_pooled_window_datasets
from market_dynamics.datasets.window_dataset import build_window_datasets
from market_dynamics.evaluation.bootstrap import bootstrap_metric_difference
from market_dynamics.evaluation.calibration import calibration_summary, plot_reliability
from market_dynamics.evaluation.metrics import classification_metrics, regression_metrics
from market_dynamics.evaluation.regime_diagnostics import regression_regime_diagnostics
from market_dynamics.evaluation.statistical_tests import (
    diebold_mariano_test,
    paired_accuracy_difference,
)
from market_dynamics.experiments.run_large_scale_screening import (
    _feature_columns,
    _has_training_class_variation,
    _run_deep_bundle,
    _run_pooled_deep,
    _target_horizon,
    load_partitioned_panel,
)
from market_dynamics.experiments.run_pooled_models import _with_models
from market_dynamics.models.baselines import classification_predictions, regression_predictions
from market_dynamics.models.classical_extended import (
    arima_direct_prediction,
    arimax_direct_prediction,
)
from market_dynamics.models.deep_learning.registry import deep_model_registry_frame
from market_dynamics.models.volatility import garch_predictions, volatility_baseline_predictions
from market_dynamics.splits.temporal import (
    global_walk_forward_splits,
    local_split_from_global_dates,
    required_global_purge_at_boundaries,
)
from market_dynamics.training.losses import task_from_target
from market_dynamics.utils.artifact_paths import artifact_safe_name
from market_dynamics.utils.data_versioning import write_run_metadata
from market_dynamics.utils.experiment_progress import (
    mark_stage_complete,
    prepare_run_directory,
    write_metric_snapshot,
)

LOGGER = logging.getLogger(__name__)


def run_walkforward_robustness(config: dict[str, Any], run_dir: str | Path | None = None) -> pd.DataFrame:
    """Run preregistered Phase 2C folds using models selected only on Layer 1 validation."""
    selection_path = Path(config["paths"]["reports_tables"]) / "phase2b_model_selection.csv"
    baseline_path = Path(config["paths"]["reports_tables"]) / "phase2b_baseline_selection.csv"
    if not selection_path.exists() or not baseline_path.exists():
        raise FileNotFoundError("Phase 2C requires validation-only Phase 2B model and baseline selection tables")
    selection, baselines = pd.read_csv(selection_path), pd.read_csv(baseline_path)
    run_dir = prepare_run_directory(config["paths"]["results_runs"], "phase2c", run_dir)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    deep_model_registry_frame().to_csv(run_dir / "deep_model_registry.csv", index=False)
    progress_path = run_dir / "walkforward_metrics_progress.csv"
    rows = _load_existing_progress(progress_path)
    completed_groups = _completed_progress_groups(rows)
    if completed_groups:
        LOGGER.info("Resuming Phase 2C from %s with %d metric rows across %d completed fold/scope groups", progress_path, len(rows), len(completed_groups))
    metadata_frames: list[pd.DataFrame] = []
    for track in ("daily", "crypto_hourly"):
        panel_root = Path(config["paths"]["processed"]) / ("daily_global_panel" if track == "daily" else "crypto_hourly_panel")
        panel = load_partitioned_panel(panel_root)
        volatility_column = "volatility_20d" if track == "daily" else "hourly_realized_volatility_24h"
        metadata_frames.append(
            panel.reset_index()[["Date", "Ticker", "asset_class", volatility_column]].rename(
                columns={"Ticker": "asset", volatility_column: "past_volatility"}
            )
        )
        features = _feature_columns(panel, track)
        targets = config["phase2b"]["screening"]["daily_targets" if track == "daily" else "hourly_targets"]
        lookbacks = config["phase2c"]["daily_lookbacks" if track == "daily" else "hourly_lookbacks"]
        for target in targets:
            if target not in panel.columns:
                continue
            task = task_from_target(target)
            target_panel = panel.dropna(subset=[target])
            for lookback in lookbacks:
                folds = _three_walkforward_folds(target_panel, target, int(lookback), config["phase2c"])
                for fold_number, global_split in enumerate(folds, start=1):
                    fold_dir = run_dir / f"fold_{fold_number}_lookback_{lookback}"
                    (fold_dir / "predictions").mkdir(parents=True, exist_ok=True)
                    local_key = _progress_group_key(track, target, int(lookback), fold_number, "local")
                    if local_key in completed_groups:
                        LOGGER.info("Skipping completed Phase 2C group: %s", local_key)
                    else:
                        rows.extend(_run_local_fold(config, track, target_panel, features, target, task, int(lookback), global_split, fold_number, selection, baselines, fold_dir))
                        completed_groups.add(local_key)
                        write_metric_snapshot(progress_path, rows)

                    pooled_key = _progress_group_key(track, target, int(lookback), fold_number, "pooled")
                    if pooled_key in completed_groups:
                        LOGGER.info("Skipping completed Phase 2C group: %s", pooled_key)
                    else:
                        rows.extend(_run_pooled_fold(config, track, target_panel, features, target, task, int(lookback), global_split, fold_number, selection, fold_dir))
                        completed_groups.add(pooled_key)
                        write_metric_snapshot(progress_path, rows)
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("Phase 2C did not produce metric rows")
    table_dir = Path(config["paths"]["reports_tables"])
    metrics.to_csv(table_dir / "phase2c_walkforward_metrics.csv", index=False)
    metrics.to_csv(run_dir / "walkforward_metrics.csv", index=False)
    _write_seed_summary(metrics, table_dir / "phase2c_seed_summary.csv")
    prediction_files = list(run_dir.glob("**/predictions/*.parquet"))
    prediction_panel = _load_predictions(prediction_files)
    bootstrap, statistics = _comparison_uncertainty(prediction_panel, config["phase2c"])
    bootstrap.to_csv(table_dir / "phase2c_bootstrap_confidence_intervals.csv", index=False)
    statistics.to_csv(table_dir / "phase2c_statistical_comparison.csv", index=False)
    calibration = _calibration_rows(prediction_panel, table_dir.parent / "figures" / "phase2c_calibration", int(config["phase2c"]["calibration"]["bins"]))
    calibration.to_csv(table_dir / "phase2c_calibration_summary.csv", index=False)
    regime = _write_regime_diagnostics(prediction_panel, pd.concat(metadata_frames, ignore_index=True), table_dir.parent / "figures" / "phase2c_regime_diagnostics")
    regime.to_csv(run_dir / "regime_diagnostics.csv", index=False)
    write_run_metadata(run_dir / "run_metadata.json", config, ["torch", "catboost", "xgboost", "lightgbm", "arch", "statsmodels"])
    mark_stage_complete(
        run_dir,
        "phase2c_walkforward_robustness",
        {
            "metric_rows": int(len(metrics)),
            "bootstrap_rows": int(len(bootstrap)),
            "statistical_test_rows": int(len(statistics)),
            "bootstrap_iterations": int(config["phase2c"]["bootstrap"]["iterations"]),
            "bootstrap_max_comparisons": int(config["phase2c"]["bootstrap"].get("max_comparisons", 0) or 0),
        },
    )
    return metrics


def _load_existing_progress(path: Path) -> list[dict[str, object]]:
    """Load a Phase 2C progress snapshot so interrupted runs can continue."""
    if not path.exists():
        return []
    progress = pd.read_csv(path)
    if progress.empty:
        return []
    required = {"track", "target", "lookback", "fold", "scope"}
    missing = required.difference(progress.columns)
    if missing:
        raise ValueError(f"Cannot resume Phase 2C because progress file is missing columns: {sorted(missing)}")
    return progress.to_dict("records")


def _progress_group_key(track: str, target: str, lookback: int, fold: int, scope: str) -> tuple[str, str, int, int, str]:
    return (str(track), str(target), int(lookback), int(fold), str(scope))


def _completed_progress_groups(rows: list[dict[str, object]]) -> set[tuple[str, str, int, int, str]]:
    """Return fold/scope groups that were flushed after a completed local or pooled run."""
    groups: set[tuple[str, str, int, int, str]] = set()
    for row in rows:
        try:
            groups.add(_progress_group_key(str(row["track"]), str(row["target"]), int(float(row["lookback"])), int(float(row["fold"])), str(row["scope"])))
        except (KeyError, TypeError, ValueError):
            continue
    return groups


def _three_walkforward_folds(
    panel: pd.DataFrame,
    target: str,
    lookback: int,
    robustness: dict[str, Any],
    purge_override: int | None = None,
) -> list[Any]:
    dates = pd.DatetimeIndex(panel.index.unique()).sort_values()
    n = len(dates)
    test_length = max(lookback + 5, n // 12)
    val_length = max(lookback + 5, n // 12)
    train_length = n - val_length - 3 * test_length - 2 * int(robustness.get("embargo", 1))
    if train_length < max(3 * lookback, 100):
        raise ValueError(f"Insufficient global history for three folds at lookback {lookback}")
    expanding = bool(robustness.get("expanding", True))
    horizon = _target_horizon(target)
    embargo = int(robustness.get("embargo", 1))
    if purge_override is not None:
        purge = int(purge_override)
    elif robustness.get("purge_by_horizon", True):
        unpurged = list(global_walk_forward_splits(panel, train_length, val_length, test_length, step_length=test_length, expanding=expanding, purge=0, embargo=embargo))
        boundaries = [
            boundary
            for fold in unpurged
            for boundary in [
                (fold.train_dates.max(), fold.val_dates.min()),
                (fold.val_dates.max(), fold.test_dates.min()),
            ]
        ]
        purge = required_global_purge_at_boundaries(panel, "Ticker", horizon, boundaries)
    else:
        purge = 0
    folds = list(global_walk_forward_splits(panel, train_length, val_length, test_length, step_length=test_length, expanding=expanding, purge=purge, embargo=embargo))
    if len(folds) < 3:
        raise ValueError("Unable to construct three walk-forward folds")
    return folds[:3]


def _run_local_fold(config: dict[str, Any], track: str, panel: pd.DataFrame, features: list[str], target: str, task: str, lookback: int, global_split: Any, fold: int, selection: pd.DataFrame, baselines: pd.DataFrame, fold_dir: Path) -> list[dict[str, object]]:
    selected = selection[(selection["track"] == track) & (selection["target"] == target) & (selection["scope"] == "local")]["model"].tolist()
    baseline_names = baselines[(baselines["track"] == track) & (baselines["target"] == target)]["model"].tolist()
    rows: list[dict[str, object]] = []
    for asset, frame in panel.groupby("Ticker", observed=True, sort=True):
        frame = frame.sort_index()
        try:
            local_split = local_split_from_global_dates(frame, global_split)
            if task == "classification" and not _has_training_class_variation(frame, local_split, target):
                rows.append({"track": track, "scope": "local", "asset": asset, "target": target, "task": task, "fold": fold, "lookback": lookback, "model": "all", "status": "skipped", "reason": "training split contains one class"})
                continue
            bundle = build_window_datasets(frame, features, target, local_split, lookback)
        except Exception as exc:
            rows.append({"track": track, "scope": "local", "asset": asset, "target": target, "fold": fold, "lookback": lookback, "model": "all", "status": "skipped", "reason": str(exc)})
            continue
        for seed in config["phase2c"]["seeds"]:
            for model in selected:
                run_rows = _run_deep_bundle(_with_models(config, [model]), track, "local", asset, target, task, lookback, int(seed), bundle, fold_dir)
                rows.extend({**row, "fold": fold, "lookback": lookback} for row in run_rows)
            rows.extend(_aligned_baseline_rows(track, asset, frame, features, target, task, bundle, baseline_names, int(seed), fold, lookback, fold_dir))
    return rows


def _run_pooled_fold(config: dict[str, Any], track: str, panel: pd.DataFrame, features: list[str], target: str, task: str, lookback: int, global_split: Any, fold: int, selection: pd.DataFrame, fold_dir: Path) -> list[dict[str, object]]:
    selected = selection[(selection["track"] == track) & (selection["target"] == target) & (selection["scope"] == "pooled")]["model"].tolist()
    if not selected:
        return []
    bundle = build_pooled_window_datasets(panel, "Ticker", features, target, global_split, lookback)
    rows: list[dict[str, object]] = []
    for seed in config["phase2c"]["seeds"]:
        for model in selected:
            run_rows = _run_pooled_deep(_with_models(config, [model]), track, target, task, lookback, model, int(seed), bundle, fold_dir)
            rows.extend({**row, "fold": fold, "lookback": lookback} for row in run_rows)
    return rows


def _aligned_baseline_rows(track: str, asset: str, frame: pd.DataFrame, features: list[str], target: str, task: str, bundle: Any, selected_names: list[str], seed: int, fold: int, lookback: int, fold_dir: Path) -> list[dict[str, object]]:
    endpoints = bundle.test.source_indices[bundle.test.endpoints]
    X_train, y_train = frame.iloc[bundle.split.train_idx][features], frame.iloc[bundle.split.train_idx][target]
    X_test, y_test = frame.iloc[endpoints][features], frame.iloc[endpoints][target]
    model_config = {"models": {"random_state": seed, "optional_models": {"xgboost": True, "lightgbm": True, "catboost": True}, "garch": {"enabled": True, "refit_frequency": 126, "max_train_size": 1500}}}
    predictions = classification_predictions(X_train, y_train.astype(int), X_test, model_config) if task == "classification" else regression_predictions(X_train, y_train, X_test, model_config)
    if task == "regression":
        predictions.extend(volatility_baseline_predictions(X_train, y_train, X_test, target))
        garch = garch_predictions(frame, endpoints, target, model_config)
        if garch is not None:
            predictions.append(garch)
    predictions.extend(item for item in [arima_direct_prediction(y_train, len(X_test), task == "classification"), arimax_direct_prediction(X_train, y_train, X_test, task == "classification")] if item is not None)
    output: list[dict[str, object]] = []
    for prediction in predictions:
        if prediction.model_name not in selected_names:
            continue
        values = classification_metrics(y_test, prediction.y_pred, prediction.y_score) if task == "classification" else regression_metrics(y_test, prediction.y_pred)
        values_to_save = prediction.y_score if task == "classification" and prediction.y_score is not None else prediction.y_pred
        data = pd.DataFrame({"Date": bundle.test.window_end_dates, "y_true": y_test.to_numpy(), "prediction": values_to_save, "track": track, "scope": "local", "asset": asset, "target": target, "task": task, "model": prediction.model_name, "seed": seed, "fold": fold, "lookback": lookback})
        filename = "_".join(artifact_safe_name(value) for value in ["baseline", track, asset, target, prediction.model_name, seed]) + ".parquet"
        data.to_parquet(fold_dir / "predictions" / filename, index=False)
        output.append({"track": track, "scope": "local", "asset": asset, "target": target, "task": task, "fold": fold, "lookback": lookback, "model": prediction.model_name, "seed": seed, "split": "test", "status": "completed", **values})
    return output


def _write_seed_summary(metrics: pd.DataFrame, path: Path) -> None:
    completed = metrics[(metrics["status"] == "completed") & (metrics["split"] == "test")].copy()
    if completed.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    values = [column for column in ["balanced_accuracy", "accuracy", "rmse", "mae", "pearson_corr"] if column in completed]
    completed.groupby(["track", "scope", "target", "model", "lookback"], observed=True)[values].agg(["mean", "std", "count"]).reset_index().to_csv(path, index=False)


def _load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        parent = path.parent.parent.name
        if parent.startswith("fold_"):
            _, fold, _, lookback = parent.split("_")
            frame["fold"] = frame.get("fold", int(fold))
            frame["lookback"] = frame.get("lookback", int(lookback))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _comparison_uncertainty(predictions: pd.DataFrame, robustness: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    bootstrap_config = robustness.get("bootstrap", {})
    iterations = int(bootstrap_config.get("iterations", 1000))
    confidence_level = float(bootstrap_config.get("confidence_level", 0.95))
    max_comparisons = int(bootstrap_config.get("max_comparisons", 0) or 0)
    max_per_stratum = int(bootstrap_config.get("max_comparisons_per_stratum", 0) or 0)
    sampling_seed = int(bootstrap_config.get("sampling_seed", 42))
    jobs_by_stratum: dict[tuple[str, str, int, str], list[dict[str, object]]] = {}
    seen_by_stratum: dict[tuple[str, str, int, str], int] = {}
    rng = np.random.default_rng(sampling_seed)
    for (track, target, asset, seed, fold, lookback), group in predictions[predictions["scope"] == "local"].groupby(["track", "target", "asset", "seed", "fold", "lookback"], observed=True):
        task = group["task"].iloc[0]
        baseline = group[group["model"].isin(group["model"].unique()) & group["model"].str.contains("baseline|majority|previous|logistic|forest|boost|arima|ewma|har|garch|historical_mean", case=False, regex=True)]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        base_frame = group[group["model"] == base["model"]][["Date", "y_true", "prediction"]]
        for model, candidate in group.groupby("model", observed=True):
            if model == base["model"]:
                continue
            joined = candidate[["Date", "y_true", "prediction"]].merge(base_frame, on="Date", suffixes=("_candidate", "_baseline"))
            if len(joined) < 10:
                continue
            stratum = (str(track), str(target), int(lookback), str(task))
            seen_by_stratum[stratum] = seen_by_stratum.get(stratum, 0) + 1
            job = {
                "track": track,
                "target": target,
                "asset": asset,
                "seed": seed,
                "fold": fold,
                "lookback": lookback,
                "task": task,
                "candidate_model": model,
                "baseline_model": base["model"],
                "y_true": joined["y_true_candidate"].to_numpy(),
                "candidate": joined["prediction_candidate"].to_numpy(),
                "baseline": joined["prediction_baseline"].to_numpy(),
            }
            bucket = jobs_by_stratum.setdefault(stratum, [])
            if max_per_stratum <= 0 or len(bucket) < max_per_stratum:
                bucket.append(job)
            else:
                replacement = int(rng.integers(0, seen_by_stratum[stratum]))
                if replacement < max_per_stratum:
                    bucket[replacement] = job
    jobs = [job for bucket in jobs_by_stratum.values() for job in bucket]
    total_seen = int(sum(seen_by_stratum.values()))
    if max_comparisons > 0 and len(jobs) > max_comparisons:
        keep = rng.choice(len(jobs), size=max_comparisons, replace=False)
        jobs = [jobs[int(index)] for index in np.sort(keep)]
    bootstrap_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    LOGGER.info(
        "Running bounded Phase 2C bootstrap on %d/%d candidate comparisons (%d iterations each)",
        len(jobs),
        total_seen,
        iterations,
    )
    for job in jobs:
        task = str(job["task"])
        metric = _binary_balanced_accuracy if task == "classification" else (lambda y, p: mean_absolute_error(y, p))
        common = {
            "track": job["track"],
            "target": job["target"],
            "asset": job["asset"],
            "seed": job["seed"],
            "fold": job["fold"],
            "lookback": job["lookback"],
            "candidate_model": job["candidate_model"],
            "baseline_model": job["baseline_model"],
            "comparison_sampling": "bounded_stratified_reservoir",
            "total_candidate_comparisons": total_seen,
            "sampled_candidate_comparisons": len(jobs),
        }
        try:
            summary = bootstrap_metric_difference(
                np.asarray(job["y_true"]),
                np.asarray(job["candidate"]),
                np.asarray(job["baseline"]),
                metric,
                iterations=iterations,
                confidence_level=confidence_level,
                seed=int(job["seed"]),
            )
            bootstrap_rows.append({"metric": "balanced_accuracy" if task == "classification" else "mae", **common, **summary})
            if task == "classification":
                test_rows.append(
                    {
                        **common,
                        "test": "mcnemar",
                        **paired_accuracy_difference(
                            np.asarray(job["y_true"]) == (np.asarray(job["candidate"]) >= 0.5),
                            np.asarray(job["y_true"]) == (np.asarray(job["baseline"]) >= 0.5),
                        ),
                    }
                )
            else:
                test_rows.append(
                    {
                        **common,
                        "test": "diebold_mariano",
                        **diebold_mariano_test(
                            np.abs((np.asarray(job["y_true"]) - np.asarray(job["candidate"])).astype(float))
                            - np.abs((np.asarray(job["y_true"]) - np.asarray(job["baseline"])).astype(float))
                        ),
                    }
                )
        except Exception as exc:
            test_rows.append({**common, "test": "failed", "reason": str(exc)})
    return pd.DataFrame(bootstrap_rows), pd.DataFrame(test_rows)


def _binary_balanced_accuracy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Warning-free balanced accuracy for bootstrap samples that may contain one class."""
    actual = np.asarray(y_true).astype(int)
    predicted = (np.asarray(probabilities) >= 0.5).astype(int)
    recalls = []
    for label in (0, 1):
        mask = actual == label
        if mask.any():
            recalls.append(float((predicted[mask] == label).mean()))
    if not recalls:
        raise ValueError("Balanced accuracy requires at least one labelled observation")
    return float(np.mean(recalls))


def _calibration_rows(predictions: pd.DataFrame, figure_dir: Path, bins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if predictions.empty:
        return pd.DataFrame(rows)
    for keys, part in predictions[predictions["task"] == "classification"].groupby(["track", "scope", "target", "model"], observed=True):
        try:
            summary, reliability = calibration_summary(part["y_true"].to_numpy(), part["prediction"].to_numpy(), bins=bins)
            plot_reliability(reliability, figure_dir / f"{'_'.join(str(item) for item in keys)}.png", "Reliability")
            rows.append({"track": keys[0], "scope": keys[1], "target": keys[2], "model": keys[3], **summary})
        except Exception as exc:
            rows.append({"track": keys[0], "scope": keys[1], "target": keys[2], "model": keys[3], "status": "failed", "reason": str(exc)})
    return pd.DataFrame(rows)


def _write_regime_diagnostics(predictions: pd.DataFrame, metadata: pd.DataFrame, figure_dir: Path) -> pd.DataFrame:
    """Save residual/error diagnostics by known asset class and pre-forecast volatility regime."""
    regression = predictions[(predictions["scope"] == "local") & (predictions["task"] == "regression")].copy()
    if regression.empty:
        return pd.DataFrame()
    joined = regression.merge(metadata, on=["Date", "asset"], how="left")
    diagnostic = regression_regime_diagnostics(joined, "past_volatility", ["track", "target", "model", "asset_class"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = diagnostic.groupby("volatility_regime", observed=True)["mae"].mean()
    plt.figure(figsize=(6, 4))
    aggregate.plot(kind="bar", color="#3b82a0")
    plt.ylabel("Mean absolute error")
    plt.title("Regression Error by Pre-Forecast Volatility Regime")
    plt.tight_layout()
    plt.savefig(figure_dir / "error_by_volatility_regime.png", dpi=150)
    plt.close()
    return diagnostic
