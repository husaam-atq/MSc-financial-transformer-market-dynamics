"""Layer 1 preregistered model-family screening for Phase 2B."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from market_dynamics.datasets.pooled_window_dataset import build_pooled_window_datasets
from market_dynamics.datasets.window_dataset import build_window_datasets
from market_dynamics.evaluation.deep_metrics import (
    evaluate_deep_predictions,
    plot_training_history,
    prediction_frame,
)
from market_dynamics.evaluation.metrics import classification_metrics, regression_metrics
from market_dynamics.features.engineering import all_feature_columns
from market_dynamics.features.hourly_engineering import HOURLY_FEATURE_COLUMNS
from market_dynamics.models.baselines import classification_predictions, regression_predictions
from market_dynamics.models.classical_extended import (
    arima_direct_prediction,
    arimax_direct_prediction,
)
from market_dynamics.models.deep_learning import AssetConditionedModel, build_deep_model
from market_dynamics.models.deep_learning.registry import deep_model_registry_frame
from market_dynamics.models.volatility import garch_predictions, volatility_baseline_predictions
from market_dynamics.splits.temporal import (
    chronological_split,
    global_chronological_split,
    required_global_purge_at_boundaries,
)
from market_dynamics.training.losses import build_loss, task_from_target
from market_dynamics.training.sampling import make_weighted_binary_sampler
from market_dynamics.training.train import fit_model, predict_loader
from market_dynamics.utils.artifact_paths import artifact_safe_name
from market_dynamics.utils.data_versioning import write_run_metadata
from market_dynamics.utils.experiment_progress import (
    mark_stage_complete,
    prepare_run_directory,
    write_metric_snapshot,
)
from market_dynamics.utils.gpu_monitoring import runtime_memory_summary
from market_dynamics.utils.torch_utils import resolve_device, set_torch_seed

LOGGER = logging.getLogger(__name__)


def load_partitioned_panel(root: str | Path) -> pd.DataFrame:
    """Load deterministic ticker partitions created by the Phase 2B builders."""
    files = sorted(Path(root).rglob("data.parquet"))
    if not files:
        raise FileNotFoundError(f"No ticker partitions found under {root}; run the matching panel builder first")
    panel = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    return panel.sort_values(["Ticker", "Date"]).set_index("Date")


def run_large_scale_screening(
    config: dict[str, Any],
    tracks: tuple[str, ...] = ("daily", "crypto_hourly"),
    run_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run every requested model on the stratified development subset and three seeds."""
    phase = config["phase2b"]
    run_dir = prepare_run_directory(config["paths"]["results_runs"], "phase2b", run_dir)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    deep_model_registry_frame().to_csv(run_dir / "deep_model_registry.csv", index=False)
    progress_path = run_dir / "model_screening_progress.csv"
    progress = _load_existing_screening_progress(progress_path, tracks)
    rows: list[dict[str, object]] = progress.to_dict(orient="records")
    for track in tracks:
        if track not in {"daily", "crypto_hourly"}:
            raise ValueError(f"Unsupported screening track: {track}")
        screening = phase["screening"]
        panel_root = Path(config["paths"]["processed"]) / ("daily_global_panel" if track == "daily" else "crypto_hourly_panel")
        panel = load_partitioned_panel(panel_root)
        assets = screening["daily_assets" if track == "daily" else "crypto_assets"]
        targets = screening["daily_targets" if track == "daily" else "hourly_targets"]
        lookback = int(screening["daily_lookback" if track == "daily" else "hourly_lookback"])
        available_assets = [asset for asset in assets if asset in set(panel["Ticker"])]
        missing_assets = sorted(set(assets).difference(available_assets))
        if missing_assets:
            LOGGER.warning("Configured %s screening assets unavailable: %s", track, missing_assets)
        subset = panel[panel["Ticker"].isin(available_assets)].copy()
        features = _feature_columns(subset, track)
        for target in targets:
            if target not in subset.columns:
                LOGGER.warning("Skipping absent %s target %s", track, target)
                continue
            task = task_from_target(target)
            for asset in available_assets:
                if _screening_block_resolved(progress, track, "local", asset, target):
                    LOGGER.info("Skipping completed screening block: %s/local/%s/%s", track, asset, target)
                    continue
                asset_frame = subset[subset["Ticker"] == asset].sort_index().dropna(subset=[target])
                block_rows = _screen_local_target(config, track, asset, asset_frame, features, target, task, lookback, run_dir)
                rows.extend(block_rows)
                progress = write_metric_snapshot(progress_path, block_rows)
            if _screening_block_resolved(progress, track, "pooled", "__pooled__", target):
                LOGGER.info("Skipping completed screening block: %s/pooled/%s", track, target)
                continue
            block_rows = _screen_pooled_target(config, track, subset, features, target, task, lookback, run_dir)
            rows.extend(block_rows)
            progress = write_metric_snapshot(progress_path, block_rows)
    metrics = _load_existing_screening_progress(progress_path, tracks)
    if metrics.empty and rows:
        metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("Screening produced no metrics; inspect data availability and target coverage")
    metrics = metrics.sort_values(["track", "scope", "task", "asset", "target", "model", "seed", "split"])
    table = Path(config["paths"]["reports_tables"]) / "phase2b_model_screening.csv"
    run_metrics = metrics.copy()
    if table.exists():
        previous = pd.read_csv(table)
        previous = previous[~previous["track"].isin(tracks)]
        metrics = pd.concat([previous, metrics], ignore_index=True).sort_values(["track", "scope", "task", "asset", "target", "model", "seed", "split"])
    metrics.to_csv(table, index=False)
    run_metrics.to_csv(run_dir / "model_screening.csv", index=False)
    selection = select_models_from_validation(metrics, int(phase.get("selection", {}).get("deep_models_per_task", 3)))
    selection.to_csv(run_dir / "model_selection_decisions.csv", index=False)
    selection.to_csv(Path(config["paths"]["reports_tables"]) / "phase2b_model_selection.csv", index=False)
    baseline_selection = select_baselines_from_validation(metrics)
    baseline_selection.to_csv(Path(config["paths"]["reports_tables"]) / "phase2b_baseline_selection.csv", index=False)
    (run_dir / "model_selection_decisions.json").write_text(json.dumps(selection.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")
    write_run_metadata(
        run_dir / "run_metadata.json",
        config,
        ["torch", "catboost", "xgboost", "lightgbm", "arch", "statsmodels", "ccxt", "yfinance"],
    )
    mark_stage_complete(run_dir, "phase2b_layer1_screening", {"metric_rows": int(len(run_metrics)), "tracks": list(tracks)})
    return metrics


def _load_existing_screening_progress(path: str | Path, tracks: tuple[str, ...]) -> pd.DataFrame:
    """Load deduplicated progress for requested tracks so interrupted screening runs resume by block."""
    progress_path = Path(path)
    if not progress_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(progress_path)
    if frame.empty or "track" not in frame.columns:
        return frame
    return frame[frame["track"].isin(tracks)].copy()


def _screening_block_resolved(progress: pd.DataFrame, track: str, scope: str, asset: str, target: str) -> bool:
    """Return True when a screening block has already ended in completed or intentional skipped rows."""
    if progress.empty:
        return False
    required = {"track", "scope", "asset", "target", "status"}
    if not required.issubset(progress.columns):
        return False
    block = progress[
        (progress["track"] == track)
        & (progress["scope"] == scope)
        & (progress["asset"] == asset)
        & (progress["target"] == target)
    ]
    if block.empty:
        return False
    statuses = set(block["status"].dropna().astype(str))
    return bool(statuses) and statuses.issubset({"completed", "skipped"})


def select_models_from_validation(metrics: pd.DataFrame, per_task: int = 3) -> pd.DataFrame:
    """Predefined deep-model selection using validation only, never final test scores."""
    deep_models = {
        "mlp", "lstm", "gru", "bilstm", "tcn", "transformer_encoder", "patchtst", "itransformer",
        "tft", "timesnet", "informer", "autoformer", "fedformer", "nbeats", "nhits", "dlinear", "nlinear",
    }
    frame = metrics[(metrics["split"] == "validation") & metrics["model"].isin(deep_models)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["track", "target", "scope", "model", "selection_metric", "validation_score", "rank"])
    output: list[pd.DataFrame] = []
    for (track, target, scope, task), part in frame.groupby(["track", "target", "scope", "task"], observed=True):
        metric = "balanced_accuracy" if task == "classification" else "rmse"
        ordered = part.groupby("model", as_index=False)[metric].mean().sort_values(metric, ascending=task != "classification").head(per_task)
        ordered["track"] = track
        ordered["target"] = target
        ordered["scope"] = scope
        ordered["selection_metric"] = metric
        ordered["validation_score"] = ordered[metric]
        ordered["rank"] = range(1, len(ordered) + 1)
        output.append(ordered[["track", "target", "scope", "model", "selection_metric", "validation_score", "rank"]])
    return pd.concat(output, ignore_index=True)


def select_baselines_from_validation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose one relevant classical/econometric comparator from validation only."""
    deep_models = set(metrics[metrics["model"].isin(["mlp", "lstm", "gru", "bilstm", "tcn", "transformer_encoder", "patchtst", "itransformer", "tft", "timesnet", "informer", "autoformer", "fedformer", "nbeats", "nhits", "dlinear", "nlinear"])]["model"])
    frame = metrics[(metrics["split"] == "validation") & (metrics["scope"] == "local") & ~metrics["model"].isin(deep_models) & (metrics["status"] == "completed")].copy()
    output: list[dict[str, object]] = []
    for (track, target, task), part in frame.groupby(["track", "target", "task"], observed=True):
        metric = "balanced_accuracy" if task == "classification" else "rmse"
        score = part.groupby("model", as_index=False)[metric].mean().sort_values(metric, ascending=task != "classification").iloc[0]
        output.append({"track": track, "target": target, "task": task, "model": score["model"], "selection_metric": metric, "validation_score": score[metric], "selection_split": "validation"})
    return pd.DataFrame(output)


def _screen_local_target(
    config: dict[str, Any],
    track: str,
    asset: str,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    task: str,
    lookback: int,
    run_dir: Path,
) -> list[dict[str, object]]:
    if len(frame) < max(250, 3 * lookback):
        return [_status_row(track, "local", asset, target, "all", pd.NA, "skipped", f"insufficient rows ({len(frame)})")]
    try:
        split = chronological_split(frame, **_split_kwargs(config, track, target))
        if task == "classification" and not _has_training_class_variation(frame, split, target):
            return [_status_row(track, "local", asset, target, "all", pd.NA, "skipped", "training split contains one class")]
        bundle = build_window_datasets(frame, features, target, split, lookback)
    except Exception as exc:
        return [_status_row(track, "local", asset, target, "all", pd.NA, "failed", str(exc))]
    rows: list[dict[str, object]] = []
    for seed in config["phase2b"]["screening"]["seeds"]:
        rows.extend(_run_deep_bundle(config, track, "local", asset, target, task, lookback, int(seed), bundle, run_dir, sampling_config=config["phase2b"].get("screening")))
        rows.extend(_run_classical_local(config, track, asset, frame, features, target, task, split, int(seed), sampling_config=config["phase2b"].get("screening")))
    return rows


def _screen_pooled_target(
    config: dict[str, Any],
    track: str,
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    task: str,
    lookback: int,
    run_dir: Path,
) -> list[dict[str, object]]:
    target_panel = panel.dropna(subset=[target])
    try:
        split = global_chronological_split(
            target_panel,
            **_split_kwargs(config, track, target, target_panel),
        )
        bundle = build_pooled_window_datasets(target_panel, "Ticker", features, target, split, lookback)
    except Exception as exc:
        return [_status_row(track, "pooled", "__pooled__", target, "all", pd.NA, "failed", str(exc))]
    rows: list[dict[str, object]] = []
    for seed in config["phase2b"]["screening"]["seeds"]:
        for model in config["phase2b"]["models"]["deep"]:
            rows.extend(_run_pooled_deep(config, track, target, task, lookback, model, int(seed), bundle, run_dir, sampling_config=config["phase2b"].get("screening")))
        rows.extend(_run_pooled_classical(config, track, target, task, lookback, int(seed), bundle, run_dir, sampling_config=config["phase2b"].get("screening")))
    return rows


def _run_deep_bundle(
    config: dict[str, Any], track: str, scope: str, asset: str, target: str, task: str, lookback: int, seed: int, bundle: Any, run_dir: Path, sampling_config: dict[str, Any] | None = None
) -> list[dict[str, object]]:
    device = resolve_device(str(config["phase2b"]["training"].get("device", "auto")))
    output: list[dict[str, object]] = []
    training_config = _screen_training_config(config["phase2b"]["training"], sampling_config)
    for model_name in config["phase2b"]["models"]["deep"]:
        try:
            set_torch_seed(seed, deterministic=bool(config["phase2b"]["training"].get("deterministic", True)))
            model = build_deep_model(model_name, len(bundle.feature_columns), lookback, {"phase2": config["phase2b"]})
            train_loader, val_loader, test_loader = _loaders(
                bundle.train,
                bundle.validation,
                bundle.test,
                training_config,
                seed,
                sampling_config=sampling_config,
            )
            result = fit_model(model, train_loader, val_loader, build_loss(task, bundle.train.targets, training_config, device), training_config, device)
            output.extend(_prediction_metric_rows(track, scope, asset, target, task, lookback, model_name, seed, model, val_loader, test_loader, device, result.best_epoch + 1, result.best_validation_loss))
            _save_local_predictions(run_dir, track, scope, asset, target, model_name, seed, bundle, model, test_loader, task, device)
            _save_training_artifacts(config, run_dir, track, scope, asset, target, model_name, seed, model, result)
        except Exception as exc:
            LOGGER.exception("Deep screening failure: %s/%s/%s", asset, target, model_name)
            output.append(_status_row(track, scope, asset, target, model_name, seed, "failed", str(exc)))
        finally:
            if "model" in locals():
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return output


def _run_pooled_deep(config: dict[str, Any], track: str, target: str, task: str, lookback: int, model_name: str, seed: int, bundle: Any, run_dir: Path, sampling_config: dict[str, Any] | None = None) -> list[dict[str, object]]:
    device = resolve_device(str(config["phase2b"]["training"].get("device", "auto")))
    try:
        set_torch_seed(seed, deterministic=bool(config["phase2b"]["training"].get("deterministic", True)))
        training_config = _screen_training_config(config["phase2b"]["training"], sampling_config)
        embedding_dim = int(config["phase2b"]["model"].get("asset_embedding_dim", 12))
        base = build_deep_model(model_name, len(bundle.feature_columns) + embedding_dim, lookback, {"phase2": config["phase2b"]})
        model = AssetConditionedModel(base, len(bundle.asset_to_id), embedding_dim)
        train_loader, val_loader, test_loader = _loaders(
            bundle.train,
            bundle.validation,
            bundle.test,
            training_config,
            seed,
            sampling_config=sampling_config,
        )
        result = fit_model(model, train_loader, val_loader, build_loss(task, np.array([part.target[end] for part in bundle.train.assets for end in part.endpoints]), training_config, device), training_config, device)
        rows = _prediction_metric_rows(track, "pooled", "__pooled__", target, task, lookback, model_name, seed, model, val_loader, test_loader, device, result.best_epoch + 1, result.best_validation_loss)
        _save_pooled_predictions(run_dir, track, target, model_name, seed, bundle, model, test_loader, task, device)
        _save_training_artifacts(config, run_dir, track, "pooled", "__pooled__", target, model_name, seed, model, result)
        return rows
    except Exception as exc:
        LOGGER.exception("Pooled deep screening failure: %s/%s", target, model_name)
        return [_status_row(track, "pooled", "__pooled__", target, model_name, seed, "failed", str(exc))]
    finally:
        if "model" in locals():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_pooled_classical(config: dict[str, Any], track: str, target: str, task: str, lookback: int, seed: int, bundle: Any, run_dir: Path, sampling_config: dict[str, Any] | None = None) -> list[dict[str, object]]:
    """Fit pooled tabular ML with a train-known asset identifier one-hot feature."""
    train_dataset = _stratified_window_subset(
        bundle.train,
        None if sampling_config is None else sampling_config.get("max_train_windows_per_asset"),
    )
    validation_dataset = _stratified_window_subset(
        bundle.validation,
        None if sampling_config is None else sampling_config.get("max_evaluation_windows_per_asset"),
    )
    test_dataset = _stratified_window_subset(
        bundle.test,
        None if sampling_config is None else sampling_config.get("max_evaluation_windows_per_asset"),
    )
    X_train, y_train = _pooled_tabular(train_dataset, bundle.feature_columns, len(bundle.asset_to_id))
    output: list[dict[str, object]] = []
    baseline_config = {"models": {"random_state": seed, "optional_models": {"xgboost": True, "lightgbm": True, "catboost": True}}}
    for split_name, dataset in (("validation", validation_dataset), ("test", test_dataset)):
        X_eval, y_eval = _pooled_tabular(dataset, bundle.feature_columns, len(bundle.asset_to_id))
        train_series = pd.Series(y_train)
        predictions = classification_predictions(X_train, train_series.astype(int), X_eval, baseline_config) if task == "classification" else regression_predictions(X_train, train_series, X_eval, baseline_config)
        # The pooled table contains per-asset scaled features. Previous-direction
        # needs the raw return sign, so it is evaluated in the local framework
        # rather than misinterpreting a standardised value here.
        predictions = [item for item in predictions if item.model_name != "previous_direction"]
        metadata = _pooled_endpoint_metadata(dataset)
        for prediction in predictions:
            values = classification_metrics(y_eval, prediction.y_pred, prediction.y_score) if task == "classification" else regression_metrics(y_eval, prediction.y_pred)
            output.append({"track": track, "scope": "pooled", "asset": "__pooled__", "target": target, "task": task, "lookback": lookback, "model": prediction.model_name, "seed": seed, "split": split_name, "status": "completed", **values})
            if split_name == "test":
                saved = metadata.copy()
                saved["y_true"] = y_eval
                saved["prediction"] = prediction.y_score if task == "classification" and prediction.y_score is not None else prediction.y_pred
                saved["track"], saved["scope"], saved["target"], saved["task"], saved["model"], saved["seed"] = track, "pooled", target, task, prediction.model_name, seed
                target_path = run_dir / "predictions" / f"{track}_pooled_{target}_{prediction.model_name}_{seed}.parquet"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                saved.to_parquet(target_path, index=False)
    return output


def _pooled_tabular(dataset: Any, feature_columns: list[str], asset_count: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Use each strict sequence endpoint for pooled tabular ML without date randomisation."""
    feature_rows: list[np.ndarray] = []
    targets: list[float] = []
    for index in range(len(dataset)):
        window, target, _, asset_id = dataset[index]
        one_hot = np.zeros(asset_count, dtype=np.float32)
        one_hot[int(asset_id)] = 1.0
        feature_rows.append(np.concatenate([window[-1].numpy(), one_hot]))
        targets.append(float(target))
    columns = [*feature_columns, *[f"asset_id_{index}" for index in range(asset_count)]]
    return pd.DataFrame(np.asarray(feature_rows), columns=columns), np.asarray(targets, dtype=float)


def _pooled_endpoint_metadata(dataset: Any) -> pd.DataFrame:
    """Return metadata for a pooled dataset or deterministic Subset thereof."""
    if isinstance(dataset, Subset):
        return dataset.dataset.endpoint_metadata().iloc[dataset.indices].reset_index(drop=True)
    return dataset.endpoint_metadata()


def _run_classical_local(config: dict[str, Any], track: str, asset: str, frame: pd.DataFrame, features: list[str], target: str, task: str, split: Any, seed: int, sampling_config: dict[str, Any] | None = None) -> list[dict[str, object]]:
    train_cap = None if sampling_config is None else sampling_config.get("max_train_windows_per_asset")
    evaluation_cap = None if sampling_config is None else sampling_config.get("max_evaluation_windows_per_asset")
    train_idx = split.train_idx if train_cap is None else split.train_idx[-int(train_cap) :]
    X_train, y_train = frame.iloc[train_idx][features], frame.iloc[train_idx][target]
    rows: list[dict[str, object]] = []
    for name, raw_indices in {"validation": split.val_idx, "test": split.test_idx}.items():
        indices = raw_indices if evaluation_cap is None else _uniform_indices(len(raw_indices), int(evaluation_cap), raw_indices)
        X_eval, y_eval = frame.iloc[indices][features], frame.iloc[indices][target]
        baseline_config = {"models": {"random_state": seed, "optional_models": {"xgboost": True, "lightgbm": True, "catboost": True}, "garch": {"enabled": True, "refit_frequency": 126, "max_train_size": 1500}}}
        predictions = classification_predictions(X_train, y_train.astype(int), X_eval, baseline_config) if task == "classification" else regression_predictions(X_train, y_train, X_eval, baseline_config)
        if task == "regression":
            predictions.extend(volatility_baseline_predictions(X_train, y_train, X_eval, target))
            garch = garch_predictions(frame, indices, target, baseline_config)
            if garch is not None:
                predictions.append(garch)
        arima = arima_direct_prediction(y_train, len(X_eval), task == "classification")
        arimax = arimax_direct_prediction(X_train, y_train, X_eval, task == "classification")
        predictions.extend(item for item in [arima, arimax] if item is not None)
        for prediction in predictions:
            values = classification_metrics(y_eval, prediction.y_pred, prediction.y_score) if task == "classification" else regression_metrics(y_eval, prediction.y_pred)
            rows.append({"track": track, "scope": "local", "asset": asset, "target": target, "task": task, "lookback": 1, "model": prediction.model_name, "seed": seed, "split": name, "status": "completed", **values})
    return rows


def _prediction_metric_rows(track: str, scope: str, asset: str, target: str, task: str, lookback: int, model: str, seed: int, network: torch.nn.Module, val_loader: DataLoader, test_loader: DataLoader, device: torch.device, best_epoch: int, best_validation_loss: float) -> list[dict[str, object]]:
    output = []
    for split_name, loader in (("validation", val_loader), ("test", test_loader)):
        actual, prediction, _ = predict_loader(network, loader, task, device)
        metrics = evaluate_deep_predictions(task, actual, prediction)
        output.append({"track": track, "scope": scope, "asset": asset, "target": target, "task": task, "lookback": lookback, "model": model, "seed": seed, "split": split_name, "status": "completed", "best_epoch": best_epoch, "best_validation_loss": best_validation_loss, **runtime_memory_summary(), **metrics})
    return output


def _save_local_predictions(run_dir: Path, track: str, scope: str, asset: str, target: str, model: str, seed: int, bundle: Any, network: torch.nn.Module, loader: DataLoader, task: str, device: torch.device) -> None:
    actual, prediction, indices = predict_loader(network, loader, task, device)
    endpoint_sources = bundle.test.source_indices[bundle.test.endpoints]
    source_to_date = pd.Series(bundle.test.window_end_dates.to_numpy(), index=endpoint_sources)
    dates = pd.DatetimeIndex(source_to_date.loc[indices].to_numpy())
    filename = "_".join(artifact_safe_name(value) for value in [track, scope, asset, target, model, seed]) + ".parquet"
    prediction_frame(dates, actual, prediction, indices, {"track": track, "scope": scope, "asset": asset, "target": target, "model": model, "seed": seed}).to_parquet(run_dir / "predictions" / filename, index=False)


def _save_pooled_predictions(run_dir: Path, track: str, target: str, model: str, seed: int, bundle: Any, network: torch.nn.Module, loader: DataLoader, task: str, device: torch.device) -> None:
    actual, prediction, indices = predict_loader(network, loader, task, device)
    metadata = bundle.test.endpoint_metadata().set_index("source_index").loc[indices].reset_index()
    metadata["y_true"], metadata["prediction"] = actual, prediction
    metadata["track"], metadata["scope"], metadata["target"], metadata["task"], metadata["model"], metadata["seed"] = track, "pooled", target, task, model, seed
    metadata.to_parquet(run_dir / "predictions" / f"{track}_pooled_{target}_{model}_{seed}.parquet", index=False)


def _save_training_artifacts(config: dict[str, Any], run_dir: Path, track: str, scope: str, asset: str, target: str, model_name: str, seed: int, model: torch.nn.Module, result: Any) -> None:
    """Persist model state and loss curves for every completed deep experiment."""
    name = f"{track}_{scope}_{asset}_{target}_{model_name}_seed{seed}".replace("/", "_").replace("-", "_")
    checkpoint = run_dir / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "track": track, "scope": scope, "asset": asset, "target": target, "model": model_name, "seed": seed, "best_epoch": result.best_epoch, "best_validation_loss": result.best_validation_loss}, checkpoint)
    plot_training_history(result.train_losses, result.validation_losses, Path(config["paths"]["reports_figures"]) / "phase2b_training_curves" / f"{name}.png", name)


def _feature_columns(panel: pd.DataFrame, track: str) -> list[str]:
    candidates = all_feature_columns() if track == "daily" else HOURLY_FEATURE_COLUMNS
    macro = [column for column in ["DFF", "DGS2", "DGS10", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2", "DTWEXBGS"] if column in panel]
    output = [column for column in [*candidates, *macro] if column in panel]
    if not output:
        raise ValueError(f"No feature columns recognised for {track}")
    return output


def _split_kwargs(
    config: dict[str, Any],
    track: str,
    target: str | None = None,
    panel: pd.DataFrame | None = None,
) -> dict[str, object]:
    split = config["phase2b"]["split"]
    horizon = _target_horizon(target) if target else (10 if track == "daily" else 24)
    purge = max(int(split.get("purge", 0)), horizon)
    if panel is not None:
        unpurged = global_chronological_split(
            panel,
            train_size=float(split["train_size"]),
            val_size=float(split["val_size"]),
            test_size=float(split["test_size"]),
            purge=0,
            embargo=int(split.get("embargo", 0)),
        )
        boundary_purge = required_global_purge_at_boundaries(
            panel,
            "Ticker",
            horizon,
            [
                (unpurged.train_dates.max(), unpurged.val_dates.min()),
                (unpurged.val_dates.max(), unpurged.test_dates.min()),
            ],
        )
        purge = max(purge, boundary_purge)
    return {"train_size": float(split["train_size"]), "val_size": float(split["val_size"]), "test_size": float(split["test_size"]), "purge": purge, "embargo": int(split.get("embargo", 0))}


def _target_horizon(target: str | None) -> int:
    """Extract the label horizon used to purge overlapping forward targets."""
    match = re.search(r"_(\d+)[dh]$", target or "")
    if match is None:
        raise ValueError(f"Cannot infer purge horizon from target: {target}")
    return int(match.group(1))


def _has_training_class_variation(frame: pd.DataFrame, split: Any, target: str) -> bool:
    """Avoid fitting or scoring local classification models with one training class."""
    values = pd.Series(frame.iloc[split.train_idx][target]).dropna()
    return values.nunique() >= 2


def _loaders(
    train: Dataset[Any],
    validation: Dataset[Any],
    test: Dataset[Any],
    training: dict[str, Any],
    seed: int,
    sampling_config: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct loaders, optionally applying deterministic Layer 1 screen caps."""
    train_dataset = _stratified_window_subset(
        train,
        None if sampling_config is None else sampling_config.get("max_train_windows_per_asset"),
    )
    validation_dataset = _stratified_window_subset(
        validation,
        None if sampling_config is None else sampling_config.get("max_evaluation_windows_per_asset"),
    )
    test_dataset = _stratified_window_subset(
        test,
        None if sampling_config is None else sampling_config.get("max_evaluation_windows_per_asset"),
    )
    kwargs = {"batch_size": int(training.get("batch_size", 512)), "num_workers": int(training.get("num_workers", 0)), "pin_memory": torch.cuda.is_available()}
    sampler = None
    weighted_sampling = training.get("weighted_sampling", {})
    if bool(weighted_sampling.get("enabled", False)):
        sampler = make_weighted_binary_sampler(
            train_dataset,
            seed=seed,
            positive_target_rate=float(weighted_sampling.get("target_positive_rate", 0.30)),
        )
    return (
        DataLoader(
            train_dataset,
            shuffle=sampler is None,
            sampler=sampler,
            generator=torch.Generator().manual_seed(seed) if sampler is None else None,
            **kwargs,
        ),
        DataLoader(validation_dataset, shuffle=False, **kwargs),
        DataLoader(test_dataset, shuffle=False, **kwargs),
    )


def _stratified_window_subset(dataset: Dataset[Any], maximum_per_asset: int | None) -> Dataset[Any]:
    """Uniformly subsample each asset's endpoints for the Layer 1 development screen."""
    if maximum_per_asset is None or int(maximum_per_asset) <= 0 or len(dataset) <= int(maximum_per_asset):
        return dataset
    cap = int(maximum_per_asset)
    if not hasattr(dataset, "references") or not hasattr(dataset, "assets"):
        return Subset(dataset, _uniform_indices(len(dataset), cap).tolist())
    grouped: dict[int, list[int]] = {}
    for position, (asset_position, _) in enumerate(dataset.references):
        asset_id = int(dataset.assets[asset_position].asset_id)
        grouped.setdefault(asset_id, []).append(position)
    indices = [selected for positions in grouped.values() for selected in _uniform_indices(len(positions), cap, positions)]
    return Subset(dataset, sorted(indices))


def _uniform_indices(length: int, cap: int, source: list[int] | None = None) -> np.ndarray:
    """Return deterministic evenly spaced source indices, retaining both endpoints."""
    values = np.arange(length, dtype=int) if source is None else np.asarray(source, dtype=int)
    if len(values) <= cap:
        return values
    positions = np.linspace(0, len(values) - 1, cap, dtype=int)
    return values[positions]


def _screen_training_config(training: dict[str, Any], sampling_config: dict[str, Any] | None) -> dict[str, Any]:
    """Override only the explicit Layer 1 epoch budget; Layer 2 remains full-length."""
    active = dict(training)
    if sampling_config is not None and "epochs" in sampling_config:
        active["epochs"] = int(sampling_config["epochs"])
    return active


def _status_row(track: str, scope: str, asset: str, target: str, model: str, seed: object, status: str, reason: str) -> dict[str, object]:
    return {"track": track, "scope": scope, "asset": asset, "target": target, "model": model, "seed": seed, "status": status, "reason": reason, "split": pd.NA}
