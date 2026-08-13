"""Layer 2 full-universe local and pooled benchmark runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from market_dynamics.datasets.pooled_window_dataset import build_pooled_window_datasets
from market_dynamics.datasets.window_dataset import build_window_datasets
from market_dynamics.experiments.run_large_scale_screening import (
    _feature_columns,
    _has_training_class_variation,
    _load_existing_screening_progress,
    _run_classical_local,
    _run_deep_bundle,
    _run_pooled_classical,
    _run_pooled_deep,
    _screening_block_resolved,
    _split_kwargs,
    load_partitioned_panel,
)
from market_dynamics.models.deep_learning.registry import deep_model_registry_frame
from market_dynamics.splits.temporal import chronological_split, global_chronological_split
from market_dynamics.training.losses import task_from_target
from market_dynamics.utils.data_versioning import write_run_metadata
from market_dynamics.utils.experiment_progress import (
    mark_stage_complete,
    prepare_run_directory,
    write_metric_snapshot,
)

LOGGER = logging.getLogger(__name__)


def run_full_universe_benchmark(config: dict[str, Any], run_dir: str | Path | None = None) -> pd.DataFrame:
    """Run validation-selected deep models plus aligned classical baselines on full panels."""
    selection_path = Path(config["paths"]["reports_tables"]) / "phase2b_model_selection.csv"
    if not selection_path.exists():
        raise FileNotFoundError("Layer 2 requires phase2b_model_selection.csv from the validation-only Layer 1 screen")
    selection = pd.read_csv(selection_path)
    run_dir = prepare_run_directory(config["paths"]["results_runs"], "phase2b", run_dir)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    deep_model_registry_frame().to_csv(run_dir / "deep_model_registry.csv", index=False)
    progress_path = run_dir / "metrics_progress.csv"
    progress = _load_existing_screening_progress(progress_path, ("daily", "crypto_hourly"))
    rows: list[dict[str, object]] = progress.to_dict(orient="records")
    for track in ("daily", "crypto_hourly"):
        panel_root = Path(config["paths"]["processed"]) / ("daily_global_panel" if track == "daily" else "crypto_hourly_panel")
        panel = load_partitioned_panel(panel_root)
        features = _feature_columns(panel, track)
        targets = config["phase2b"]["screening"]["daily_targets" if track == "daily" else "hourly_targets"]
        lookback = int(config["phase2b"]["screening"]["daily_lookback" if track == "daily" else "hourly_lookback"])
        for target in targets:
            if target not in panel:
                LOGGER.warning("Target missing from %s panel: %s", track, target)
                continue
            task = task_from_target(target)
            target_panel = panel.dropna(subset=[target])
            block_rows, progress = _run_local_full(config, track, target_panel, features, target, task, lookback, run_dir, selection, progress, progress_path)
            rows.extend(block_rows)
            if _screening_block_resolved(progress, track, "pooled", "__pooled__", target):
                LOGGER.info("Skipping completed Layer 2 pooled block: %s/%s", track, target)
                continue
            block_rows = _run_pooled_full(config, track, target_panel, features, target, task, lookback, run_dir, selection)
            rows.extend(block_rows)
            progress = write_metric_snapshot(progress_path, block_rows)
    metrics = _load_existing_screening_progress(progress_path, ("daily", "crypto_hourly"))
    if metrics.empty and rows:
        metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No Layer 2 metrics were produced")
    metrics.to_csv(Path(config["paths"]["reports_tables"]) / "phase2b_large_scale_metrics.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    comparison = _local_vs_pooled_comparison(metrics)
    comparison.to_csv(Path(config["paths"]["reports_tables"]) / "phase2b_local_vs_pooled_comparison.csv", index=False)
    comparison.to_csv(run_dir / "local_vs_pooled_comparison.csv", index=False)
    write_run_metadata(run_dir / "run_metadata.json", config, ["torch", "catboost", "xgboost", "lightgbm", "arch", "statsmodels"])
    mark_stage_complete(run_dir, "phase2b_layer2_full_universe", {"metric_rows": int(len(metrics))})
    return metrics


def _run_local_full(
    config: dict[str, Any],
    track: str,
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    task: str,
    lookback: int,
    run_dir: Path,
    selection: pd.DataFrame,
    progress: pd.DataFrame,
    progress_path: Path,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    selected = _selected_models(selection, track, target, "local")
    rows: list[dict[str, object]] = []
    for asset, frame in panel.groupby("Ticker", observed=True, sort=True):
        asset = str(asset)
        if _screening_block_resolved(progress, track, "local", asset, target):
            LOGGER.info("Skipping completed Layer 2 local block: %s/local/%s/%s", track, asset, target)
            continue
        frame = frame.sort_index()
        try:
            split = chronological_split(frame, **_split_kwargs(config, track, target))
            if task == "classification" and not _has_training_class_variation(frame, split, target):
                asset_rows = [{"track": track, "scope": "local", "asset": asset, "target": target, "task": task, "lookback": lookback, "model": "all", "seed": pd.NA, "split": pd.NA, "status": "skipped", "reason": "training split contains one class"}]
                rows.extend(asset_rows)
                progress = write_metric_snapshot(progress_path, asset_rows)
                continue
            bundle = build_window_datasets(frame, features, target, split, lookback)
        except Exception as exc:
            asset_rows = [{"track": track, "scope": "local", "asset": asset, "target": target, "task": task, "lookback": lookback, "model": "all", "seed": pd.NA, "split": pd.NA, "status": "skipped", "reason": str(exc)}]
            rows.extend(asset_rows)
            progress = write_metric_snapshot(progress_path, asset_rows)
            continue
        asset_rows: list[dict[str, object]] = []
        for seed in config["phase2b"]["screening"]["seeds"]:
            for model in selected:
                active = _with_models(config, [model])
                asset_rows.extend(_run_deep_bundle(active, track, "local", asset, target, task, lookback, int(seed), bundle, run_dir))
            asset_rows.extend(_run_classical_local(config, track, asset, frame, features, target, task, split, int(seed)))
        rows.extend(asset_rows)
        progress = write_metric_snapshot(progress_path, asset_rows)
    return rows, progress


def _run_pooled_full(config: dict[str, Any], track: str, panel: pd.DataFrame, features: list[str], target: str, task: str, lookback: int, run_dir: Path, selection: pd.DataFrame) -> list[dict[str, object]]:
    selected = _selected_models(selection, track, target, "pooled")
    if not selected:
        return []
    split = global_chronological_split(panel, **_split_kwargs(config, track, target, panel))
    bundle = build_pooled_window_datasets(panel, "Ticker", features, target, split, lookback)
    rows: list[dict[str, object]] = []
    for seed in config["phase2b"]["screening"]["seeds"]:
        for model in selected:
            rows.extend(_run_pooled_deep(_with_models(config, [model]), track, target, task, lookback, model, int(seed), bundle, run_dir))
        rows.extend(_run_pooled_classical(config, track, target, task, lookback, int(seed), bundle, run_dir))
    return rows


def _selected_models(selection: pd.DataFrame, track: str, target: str, scope: str) -> list[str]:
    return selection[(selection["track"] == track) & (selection["target"] == target) & (selection["scope"] == scope)].sort_values("rank")["model"].tolist()


def _with_models(config: dict[str, Any], models: list[str]) -> dict[str, Any]:
    active = {**config, "phase2b": {**config["phase2b"], "models": {**config["phase2b"]["models"], "deep": models}}}
    return active


def _local_vs_pooled_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    completed = metrics[(metrics["status"] == "completed") & (metrics["split"] == "test")].copy()
    if completed.empty:
        return pd.DataFrame()
    metric = completed.apply(lambda row: row.get("balanced_accuracy") if row.get("task") == "classification" else row.get("rmse"), axis=1)
    completed["comparison_metric"] = metric
    return completed.groupby(["track", "target", "task", "scope", "model"], observed=True)["comparison_metric"].agg(["mean", "std", "count"]).reset_index()
