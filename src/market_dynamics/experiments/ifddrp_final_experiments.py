"""Final bounded static/dynamic, within-asset and continuous-risk studies."""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from market_dynamics.datasets.pairwise import (
    WithinAssetPairDataset,
    build_outcome_disjoint_pair_registry,
)
from market_dynamics.datasets.pooled_window_dataset import (
    PooledWindowDataBundle,
    build_pooled_window_datasets,
)
from market_dynamics.evaluation.post_freeze import (
    binary_probability_metrics,
    fit_calibration_candidates,
)
from market_dynamics.evaluation.prior_neutral import (
    equal_group_weights,
    fit_group_priors,
    grouped_binary_metrics,
)
from market_dynamics.experiments.run_phase6 import (
    PerturbedDataset,
    Phase6Context,
    _bundle,
    _prediction_frame,
    _split_manifest,
    _within_group_auc,
)
from market_dynamics.experiments.run_phase6 import (
    build_context as build_phase6_context,
)
from market_dynamics.models.deep_learning import (
    AssetAgnosticModel,
    FixedPriorResidualModel,
    ZeroChannelSequenceModel,
    build_deep_model,
)
from market_dynamics.targets.make_targets import add_future_maximum_loss_target
from market_dynamics.training.losses import build_loss
from market_dynamics.training.sampling import dataset_targets, make_balanced_group_sampler
from market_dynamics.training.train import (
    fit_model,
    fit_model_with_explicit_pairs,
    predict_loader,
)
from market_dynamics.utils.torch_utils import set_torch_seed

MODEL_FAMILIES = ("mlp", "transformer_encoder")
STATIC_VARIANTS = (
    "no_id_temporal_model",
    "family_prior_dynamic_residual",
    "asset_prior_dynamic_residual",
)
OBJECTIVES = (
    "pooled_bce",
    "equal_asset_bce",
    "within_asset_pairwise",
    "bce_plus_within_asset_pairwise",
)
CONTINUOUS_VARIANTS = ("no_id_temporal", "asset_intercept_dynamic_residual")


@dataclass
class FinalExperimentContext:
    """Shared corrected Phase 6 data contract and final-study paths."""

    phase6: Phase6Context
    bundle: PooledWindowDataBundle
    static_options: dict[str, Any]
    objective_options: dict[str, Any]
    continuous_options: dict[str, Any]
    run_dir: Path
    table_dir: Path


class TargetScaledDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Expose globally train-standardised targets without changing windows."""

    def __init__(self, base: Dataset[Any], mean: float, scale: float) -> None:
        self.base = base
        self.mean = float(mean)
        self.scale = float(scale)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, target, source_index, asset_id = self.base[item]
        scaled = (target - self.mean) / self.scale
        return features, scaled, source_index, asset_id

    def endpoint_metadata(self) -> pd.DataFrame:
        return self.base.endpoint_metadata()


def build_final_context(
    config: dict[str, Any],
    phase6_config: dict[str, Any],
    static_config: dict[str, Any],
    objective_config: dict[str, Any],
    continuous_config: dict[str, Any],
    *,
    phase6_run_dir: str | Path,
    run_dir: str | Path,
) -> FinalExperimentContext:
    """Reconstruct the authoritative corrected bundle and enforce protocol identity."""
    phase6 = build_phase6_context(config, phase6_config, phase6_run_dir)
    bundle = _bundle(phase6, "corrected_asset_conditioned")
    static = dict(static_config.get("ifddrp_static_dynamic", static_config))
    objective = dict(objective_config.get("ifddrp_within_asset_objective", objective_config))
    continuous = dict(continuous_config.get("ifddrp_continuous_downside", continuous_config))
    for name, options in [("static", static), ("objective", objective), ("continuous", continuous)]:
        if int(options["lookback"]) != int(phase6.options["lookback"]):
            raise RuntimeError(f"{name} lookback differs from corrected Phase 6")
        if int(options["fold"]) != int(phase6.options["fold"]):
            raise RuntimeError(f"{name} fold differs from corrected Phase 6")
        if int(options["purge"]) != int(phase6.split.purge):
            raise RuntimeError(f"{name} purge differs from corrected Phase 6")
    active = Path(run_dir).resolve()
    for child in ["checkpoints", "predictions", "logs", "pair_registries", "manifests"]:
        (active / child).mkdir(parents=True, exist_ok=True)
    context = FinalExperimentContext(
        phase6=phase6,
        bundle=bundle,
        static_options=static,
        objective_options=objective,
        continuous_options=continuous,
        run_dir=active,
        table_dir=Path(config["paths"]["reports_tables"]),
    )
    _write_shared_manifest(context)
    return context


def run_static_dynamic(context: FinalExperimentContext) -> dict[str, pd.DataFrame]:
    """Train fixed-prior residual arms and run preregistered controls."""
    options = context.static_options
    training_rows: list[dict[str, object]] = []
    priors = _classification_priors(context.bundle, context.phase6.family_map)
    _write_prior_manifest(context, priors)
    for model_name in MODEL_FAMILIES:
        for variant in STATIC_VARIANTS:
            cell = f"static_dynamic__{model_name}__{variant}"
            for seed in _seeds(options):
                prediction_path = context.run_dir / "predictions" / f"{cell}_seed{seed}.parquet"
                log_path = context.run_dir / "logs" / f"{cell}_seed{seed}.json"
                if prediction_path.exists() and log_path.exists():
                    training_rows.append(json.loads(log_path.read_text(encoding="utf-8")))
                    continue
                set_torch_seed(seed, deterministic=True)
                model = _build_static_model(context, model_name, variant, priors).to(context.phase6.device)
                train_loader, validation_loader, test_loader = _endpoint_loaders(
                    context.bundle,
                    options["training"],
                    seed,
                    sampling="pooled_shuffle",
                )
                criterion = build_loss(
                    "classification",
                    dataset_targets(context.bundle.train),
                    options["training"],
                    context.phase6.device,
                )
                started = perf_counter()
                fitted = fit_model(
                    model,
                    train_loader,
                    validation_loader,
                    criterion,
                    options["training"],
                    context.phase6.device,
                )
                prediction = _classification_predictions(
                    model,
                    [("validation", validation_loader, context.bundle.validation), ("test", test_loader, context.bundle.test)],
                    context,
                )
                prediction.to_parquet(prediction_path, index=False)
                row = {
                    "study": "static_dynamic",
                    "cell": cell,
                    "model": model_name,
                    "variant": variant,
                    "seed": seed,
                    "status": "completed",
                    "runtime_seconds": perf_counter() - started,
                    "best_epoch": fitted.best_epoch + 1,
                    "best_validation_loss": fitted.best_validation_loss,
                }
                _save_checkpoint(context, model, cell, seed, row, priors=priors)
                log_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                training_rows.append(row)
                pd.DataFrame(training_rows).to_csv(context.run_dir / "static_dynamic_training.csv", index=False)
            _write_classification_ensemble(context, cell, _seeds(options), options)
    comparator_predictions = _static_dynamic_comparators(context, priors)
    for name, frame in comparator_predictions.items():
        frame.to_parquet(context.run_dir / "predictions" / f"static_dynamic__{name}_ensemble.parquet", index=False)
    results = _classification_result_table(context, "static_dynamic", options)
    order = _classification_order_tests(context, "static_dynamic", options, priors=priors)
    verdict = _static_dynamic_verdict(context, results, order)
    pd.DataFrame(training_rows).to_csv(context.run_dir / "static_dynamic_training.csv", index=False)
    results.to_csv(context.table_dir / "ifddrp_static_dynamic_results.csv", index=False)
    order.to_csv(context.table_dir / "ifddrp_static_dynamic_order_tests.csv", index=False)
    (context.table_dir / "ifddrp_static_dynamic_verdict.md").write_text(verdict, encoding="utf-8")
    return {"training": pd.DataFrame(training_rows), "results": results, "order": order}


def run_within_asset_objectives(context: FinalExperimentContext) -> dict[str, pd.DataFrame]:
    """Train four matched no-ID objectives with explicit pair support."""
    options = context.objective_options
    pair_settings = options["pair_construction"]
    train_registry = build_outcome_disjoint_pair_registry(
        context.bundle.train,
        horizon=int(options["horizon"]),
        maximum_pairs_per_asset=int(pair_settings["maximum_pairs_per_asset_per_batch"]),
        seed=int(pair_settings["registry_seed"]),
        split="train",
    )
    validation_registry = build_outcome_disjoint_pair_registry(
        context.bundle.validation,
        horizon=int(options["horizon"]),
        maximum_pairs_per_asset=int(pair_settings["maximum_pairs_per_asset_per_batch"]),
        seed=int(pair_settings["registry_seed"]) + 1,
        split="validation",
    )
    train_registry.pairs.to_parquet(context.run_dir / "pair_registries" / "train_pairs.parquet", index=False)
    validation_registry.pairs.to_parquet(context.run_dir / "pair_registries" / "validation_pairs.parquet", index=False)
    pair_audit = pd.concat([train_registry.audit, validation_registry.audit], ignore_index=True)
    pair_audit["asset_ticker"] = pair_audit["asset_id"].map(_reverse_assets(context.bundle))
    pair_audit["family"] = pair_audit["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
    pair_audit["registry_sha256"] = pair_audit["split"].map(
        {"train": train_registry.sha256, "validation": validation_registry.sha256}
    )
    pair_audit.to_csv(context.table_dir / "ifddrp_within_asset_pair_audit.csv", index=False)
    training_rows: list[dict[str, object]] = []
    for model_name in MODEL_FAMILIES:
        for objective_name in OBJECTIVES:
            cell = f"within_objective__{model_name}__{objective_name}"
            objective = options["objectives"][objective_name]
            for seed in _seeds(options):
                prediction_path = context.run_dir / "predictions" / f"{cell}_seed{seed}.parquet"
                log_path = context.run_dir / "logs" / f"{cell}_seed{seed}.json"
                if prediction_path.exists() and log_path.exists():
                    training_rows.append(json.loads(log_path.read_text(encoding="utf-8")))
                    continue
                set_torch_seed(seed, deterministic=True)
                model = _build_no_id_model(context, model_name, options).to(context.phase6.device)
                train_loader, validation_loader, test_loader = _endpoint_loaders(
                    context.bundle,
                    options["training"],
                    seed,
                    sampling=str(objective["sampling"]),
                )
                started = perf_counter()
                if float(objective["pairwise_coefficient"]) > 0.0:
                    pair_workers = int(options["training"].get("pair_num_workers", 4))
                    pair_loader_options = {
                        "num_workers": pair_workers,
                        "persistent_workers": pair_workers > 0,
                        "pin_memory": torch.cuda.is_available(),
                    }
                    pair_train_loader = DataLoader(
                        WithinAssetPairDataset(context.bundle.train, train_registry),
                        batch_size=int(options["training"]["batch_size"]),
                        shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        **pair_loader_options,
                    )
                    pair_validation_loader = DataLoader(
                        WithinAssetPairDataset(context.bundle.validation, validation_registry),
                        batch_size=int(options["training"]["batch_size"]),
                        shuffle=False,
                        **pair_loader_options,
                    )
                    fitted = fit_model_with_explicit_pairs(
                        model,
                        pair_train_loader,
                        pair_validation_loader,
                        options["training"],
                        context.phase6.device,
                        pointwise_coefficient=float(objective["pointwise_coefficient"]),
                        pairwise_coefficient=float(objective["pairwise_coefficient"]),
                    )
                else:
                    criterion = build_loss(
                        "classification",
                        dataset_targets(context.bundle.train),
                        options["training"],
                        context.phase6.device,
                    )
                    fitted = fit_model(
                        model,
                        train_loader,
                        validation_loader,
                        criterion,
                        options["training"],
                        context.phase6.device,
                    )
                prediction = _classification_predictions(
                    model,
                    [("validation", validation_loader, context.bundle.validation), ("test", test_loader, context.bundle.test)],
                    context,
                )
                prediction.to_parquet(prediction_path, index=False)
                row = {
                    "study": "within_objective",
                    "cell": cell,
                    "model": model_name,
                    "variant": objective_name,
                    "seed": seed,
                    "status": "completed",
                    "runtime_seconds": perf_counter() - started,
                    "best_epoch": fitted.best_epoch + 1,
                    "best_validation_loss": fitted.best_validation_loss,
                    "train_pair_registry_sha256": train_registry.sha256,
                    "validation_pair_registry_sha256": validation_registry.sha256,
                }
                _save_checkpoint(context, model, cell, seed, row)
                log_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                training_rows.append(row)
                pd.DataFrame(training_rows).to_csv(context.run_dir / "within_objective_training.csv", index=False)
            _write_classification_ensemble(context, cell, _seeds(options), options)
    results = _classification_result_table(context, "within_objective", options)
    order = _classification_order_tests(context, "within_objective", options)
    verdict = _within_objective_verdict(context, results, order, pair_audit)
    pd.DataFrame(training_rows).to_csv(context.run_dir / "within_objective_training.csv", index=False)
    results.to_csv(context.table_dir / "ifddrp_within_asset_objective_results.csv", index=False)
    order.to_csv(context.table_dir / "ifddrp_within_asset_objective_order_tests.csv", index=False)
    (context.table_dir / "ifddrp_within_asset_objective_verdict.md").write_text(verdict, encoding="utf-8")
    return {"training": pd.DataFrame(training_rows), "pairs": pair_audit, "results": results, "order": order}


def run_continuous_downside(context: FinalExperimentContext) -> dict[str, pd.DataFrame]:
    """Fit one continuous ten-session maximum-loss target under fixed controls."""
    options = context.continuous_options
    target = str(options["target"]["name"])
    horizon = int(options["target"]["horizon"])
    target_panel = add_future_maximum_loss_target(
        context.phase6.panel,
        horizon=horizon,
        target_column=target,
    ).dropna(subset=[target])
    bundle = build_pooled_window_datasets(
        target_panel,
        "Ticker",
        context.bundle.feature_columns,
        target,
        context.phase6.split,
        int(options["lookback"]),
    )
    training_target = dataset_targets(bundle.train).astype(float)
    target_mean = float(np.mean(training_target))
    target_scale = float(np.std(training_target, ddof=0))
    if not np.isfinite(target_scale) or target_scale <= 0.0:
        raise RuntimeError("Continuous training target has no finite variation")
    scaled = {
        "train": TargetScaledDataset(bundle.train, target_mean, target_scale),
        "validation": TargetScaledDataset(bundle.validation, target_mean, target_scale),
        "test": TargetScaledDataset(bundle.test, target_mean, target_scale),
    }
    intercepts = _continuous_asset_intercepts(bundle, target_mean, target_scale)
    _write_continuous_manifest(context, bundle, target_panel, target_mean, target_scale, intercepts)
    training_rows: list[dict[str, object]] = []
    for model_name in MODEL_FAMILIES:
        for variant in CONTINUOUS_VARIANTS:
            cell = f"continuous_downside__{model_name}__{variant}"
            for seed in _seeds(options):
                prediction_path = context.run_dir / "predictions" / f"{cell}_seed{seed}.parquet"
                log_path = context.run_dir / "logs" / f"{cell}_seed{seed}.json"
                if prediction_path.exists() and log_path.exists():
                    training_rows.append(json.loads(log_path.read_text(encoding="utf-8")))
                    continue
                set_torch_seed(seed, deterministic=True)
                model = _build_continuous_model(context, model_name, variant, bundle, intercepts).to(context.phase6.device)
                sampler = make_balanced_group_sampler(bundle.train, None, seed, "equal_asset")
                loader_kwargs = {
                    "batch_size": int(options["training"]["batch_size"]),
                    "num_workers": 0,
                    "pin_memory": torch.cuda.is_available(),
                }
                train_loader = DataLoader(scaled["train"], sampler=sampler, shuffle=False, **loader_kwargs)
                validation_loader = DataLoader(scaled["validation"], shuffle=False, **loader_kwargs)
                test_loader = DataLoader(scaled["test"], shuffle=False, **loader_kwargs)
                criterion = build_loss(
                    "regression",
                    (training_target - target_mean) / target_scale,
                    options["training"],
                    context.phase6.device,
                )
                started = perf_counter()
                fitted = fit_model(
                    model,
                    train_loader,
                    validation_loader,
                    criterion,
                    options["training"],
                    context.phase6.device,
                )
                frames = [
                    _continuous_prediction_frame(
                        model,
                        validation_loader,
                        bundle.validation,
                        bundle,
                        "validation",
                        target_mean,
                        target_scale,
                        context.phase6.device,
                    ),
                    _continuous_prediction_frame(
                        model,
                        test_loader,
                        bundle.test,
                        bundle,
                        "test",
                        target_mean,
                        target_scale,
                        context.phase6.device,
                    ),
                ]
                prediction = pd.concat(frames, ignore_index=True)
                prediction.to_parquet(prediction_path, index=False)
                row = {
                    "study": "continuous_downside",
                    "cell": cell,
                    "model": model_name,
                    "variant": variant,
                    "seed": seed,
                    "status": "completed",
                    "runtime_seconds": perf_counter() - started,
                    "best_epoch": fitted.best_epoch + 1,
                    "best_validation_loss": fitted.best_validation_loss,
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                }
                _save_checkpoint(
                    context,
                    model,
                    cell,
                    seed,
                    row,
                    target_scaling={"mean": target_mean, "scale": target_scale},
                    asset_mapping=bundle.asset_to_id,
                )
                log_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                training_rows.append(row)
                pd.DataFrame(training_rows).to_csv(context.run_dir / "continuous_downside_training.csv", index=False)
            _write_regression_ensemble(context, cell, _seeds(options))
    baseline_frames = _continuous_baselines(context, bundle, target_panel)
    for name, frame in baseline_frames.items():
        frame.to_parquet(context.run_dir / "predictions" / f"continuous_downside__{name}_ensemble.parquet", index=False)
    results = _continuous_result_table(context, bundle, target_panel, options)
    robustness = _continuous_order_and_robustness(
        context,
        bundle,
        scaled,
        target_panel,
        target_mean,
        target_scale,
        intercepts,
        options,
    )
    verdict, practical = _continuous_verdict(context, results, robustness)
    pd.DataFrame(training_rows).to_csv(context.run_dir / "continuous_downside_training.csv", index=False)
    results.to_csv(context.table_dir / "ifddrp_continuous_downside_results.csv", index=False)
    robustness.to_csv(context.table_dir / "ifddrp_continuous_downside_robustness.csv", index=False)
    (context.table_dir / "ifddrp_continuous_downside_verdict.md").write_text(verdict, encoding="utf-8")
    (context.table_dir / "ifddrp_continuous_downside_practical_use.csv").write_text(practical, encoding="utf-8")
    return {"training": pd.DataFrame(training_rows), "results": results, "robustness": robustness}


def _seeds(options: dict[str, Any]) -> list[int]:
    return [int(value) for value in options["seeds"]]


def _reverse_assets(bundle: PooledWindowDataBundle) -> dict[int, str]:
    return {value: key for key, value in bundle.asset_to_id.items()}


def _classification_priors(bundle: PooledWindowDataBundle, family_map: dict[str, str]) -> dict[str, Any]:
    training = _classification_endpoint_frame(bundle.train, bundle, "train", family_map)
    global_prior = float(training["y_true"].mean())
    asset = fit_group_priors(training, "asset_ticker", smoothing=1.0)
    family = fit_group_priors(training, "family", smoothing=1.0)
    asset_vector = np.asarray(
        [float(asset.get(ticker, global_prior)) for ticker, _ in sorted(bundle.asset_to_id.items(), key=lambda item: item[1])],
        dtype=float,
    )
    family_vector = np.asarray(
        [
            float(family.get(family_map.get(ticker, "Unknown"), global_prior))
            for ticker, _ in sorted(bundle.asset_to_id.items(), key=lambda item: item[1])
        ],
        dtype=float,
    )
    return {
        "global": global_prior,
        "asset": asset,
        "family": family,
        "asset_vector": asset_vector,
        "family_vector": family_vector,
    }


def _model_settings(options: dict[str, Any], model_name: str) -> dict[str, Any]:
    settings = deepcopy(options["models"][model_name])
    settings.setdefault("model", model_name)
    return settings


def _build_static_model(
    context: FinalExperimentContext,
    model_name: str,
    variant: str,
    priors: dict[str, Any],
) -> nn.Module:
    options = context.static_options
    zero_channels = int(options["capacity_control"]["zero_channels"])
    settings = _model_settings(options, model_name)
    base = build_deep_model(
        model_name,
        len(context.bundle.feature_columns) + zero_channels,
        int(options["lookback"]),
        {"model": settings},
    )
    if variant == "no_id_temporal_model":
        return AssetAgnosticModel(base, zero_channels=zero_channels)
    dynamic = ZeroChannelSequenceModel(base, zero_channels=zero_channels)
    if variant == "family_prior_dynamic_residual":
        vector = priors["family_vector"]
    elif variant == "asset_prior_dynamic_residual":
        vector = priors["asset_vector"]
    else:
        raise ValueError(f"Unknown static/dynamic variant: {variant}")
    prior_logits = torch.tensor(logit(np.clip(vector, 1e-6, 1.0 - 1e-6)), dtype=torch.float32)
    return FixedPriorResidualModel(dynamic, prior_logits)


def _build_no_id_model(context: FinalExperimentContext, model_name: str, options: dict[str, Any]) -> nn.Module:
    zero_channels = int(options["capacity_control"]["zero_channels"])
    base = build_deep_model(
        model_name,
        len(context.bundle.feature_columns) + zero_channels,
        int(options["lookback"]),
        {"model": _model_settings(options, model_name)},
    )
    return AssetAgnosticModel(base, zero_channels=zero_channels)


def _build_continuous_model(
    context: FinalExperimentContext,
    model_name: str,
    variant: str,
    bundle: PooledWindowDataBundle,
    intercepts: np.ndarray,
) -> nn.Module:
    options = context.continuous_options
    zero_channels = 12
    base = build_deep_model(
        model_name,
        len(bundle.feature_columns) + zero_channels,
        int(options["lookback"]),
        {"model": _model_settings(options, model_name)},
    )
    if variant == "no_id_temporal":
        return AssetAgnosticModel(base, zero_channels=zero_channels)
    if variant == "asset_intercept_dynamic_residual":
        return FixedPriorResidualModel(
            ZeroChannelSequenceModel(base, zero_channels=zero_channels),
            torch.tensor(intercepts, dtype=torch.float32),
        )
    raise ValueError(f"Unknown continuous variant: {variant}")


def _endpoint_loaders(
    bundle: PooledWindowDataBundle,
    training: dict[str, Any],
    seed: int,
    *,
    sampling: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    kwargs = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    if sampling == "equal_asset":
        sampler = make_balanced_group_sampler(bundle.train, None, seed, "equal_asset")
        train = DataLoader(bundle.train, sampler=sampler, shuffle=False, **kwargs)
    elif sampling == "pooled_shuffle":
        train = DataLoader(
            bundle.train,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")
    return (
        train,
        DataLoader(bundle.validation, shuffle=False, **kwargs),
        DataLoader(bundle.test, shuffle=False, **kwargs),
    )


def _classification_predictions(
    model: nn.Module,
    parts: list[tuple[str, DataLoader, Dataset[Any]]],
    context: FinalExperimentContext,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, loader, dataset in parts:
        y, probability, indices = predict_loader(model, loader, "classification", context.phase6.device)
        frames.append(
            _prediction_frame(
                dataset,
                indices,
                y,
                probability,
                context.bundle.asset_to_id,
                split,
            )
        )
    return pd.concat(frames, ignore_index=True)


def _write_classification_ensemble(
    context: FinalExperimentContext,
    cell: str,
    seeds: list[int],
    options: dict[str, Any],
) -> pd.DataFrame:
    frames = [pd.read_parquet(context.run_dir / "predictions" / f"{cell}_seed{seed}.parquet") for seed in seeds]
    ensemble = _mean_prediction_frames(frames, "raw_probability")
    validation = ensemble[ensemble["split"].eq("validation")]
    selected = fit_calibration_candidates(
        validation["y_true"].to_numpy(),
        validation["raw_probability"].to_numpy(),
        options["calibration"]["methods"],
        threshold_metric=str(options["calibration"]["threshold_metric"]),
        threshold_grid_size=int(options["calibration"]["threshold_grid_size"]),
        calibration_bins=int(options["calibration"]["bins"]),
    )[0]
    ensemble["selected_probability"] = selected.calibrator.predict(ensemble["raw_probability"].to_numpy())
    ensemble["calibration_method"] = selected.method
    ensemble["selected_threshold"] = selected.threshold
    ensemble.to_parquet(context.run_dir / "predictions" / f"{cell}_ensemble.parquet", index=False)
    return ensemble


def _mean_prediction_frames(frames: list[pd.DataFrame], value_column: str) -> pd.DataFrame:
    if not frames:
        raise ValueError("At least one prediction frame is required")
    keys = ["split", "Date", "source_index", "asset_id", "asset_ticker", "y_true"]
    ordered = [frame.sort_values(keys[:4]).reset_index(drop=True) for frame in frames]
    for frame in ordered[1:]:
        if not ordered[0][keys].equals(frame[keys]):
            raise RuntimeError("Seed prediction endpoints do not align")
    output = ordered[0][keys].copy()
    output[value_column] = np.mean(
        np.stack([frame[value_column].to_numpy(dtype=np.float64) for frame in ordered]),
        axis=0,
        dtype=np.float64,
    )
    return output


def _static_dynamic_comparators(
    context: FinalExperimentContext,
    priors: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    parts = [
        _classification_endpoint_frame(context.bundle.validation, context.bundle, "validation", context.phase6.family_map),
        _classification_endpoint_frame(context.bundle.test, context.bundle, "test", context.phase6.family_map),
    ]
    endpoint = pd.concat(parts, ignore_index=True)
    outputs["global_prior"] = _score_frame(endpoint, np.full(len(endpoint), priors["global"]))
    outputs["family_prior"] = _score_frame(
        endpoint,
        endpoint["family"].map(priors["family"]).fillna(priors["global"]).to_numpy(dtype=float),
    )
    outputs["asset_prior"] = _score_frame(
        endpoint,
        endpoint["asset_ticker"].map(priors["asset"]).fillna(priors["global"]).to_numpy(dtype=float),
    )
    historical = {
        "existing_mlp_asset_conditioned": (
            context.phase6.config["paths"]["results_runs"],
            "prp1_fixed_cross_model_20260715",
            "mlp_asset_conditioned_seed{seed}.parquet",
        ),
        "existing_mlp_no_id": (
            context.phase6.config["paths"]["results_runs"],
            "prp1_fixed_cross_model_20260715",
            "mlp_no_explicit_asset_id_seed{seed}.parquet",
        ),
        "existing_transformer_asset_conditioned": (
            context.phase6.config["paths"]["results_runs"],
            "phase6_transformer_falsification_20260712",
            "corrected_asset_conditioned_seed{seed}.parquet",
        ),
        "existing_transformer_no_id": (
            context.phase6.config["paths"]["results_runs"],
            "phase6_transformer_falsification_20260712",
            "no_explicit_asset_id_seed{seed}.parquet",
        ),
        "flattened_logistic_no_id": (
            context.phase6.config["paths"]["results_runs"],
            "prp1_fixed_cross_model_20260715",
            "flattened_logistic_no_explicit_asset_id_seed{seed}.parquet",
        ),
    }
    for name, (root, run, pattern) in historical.items():
        prediction_dir = Path(root) / run / "predictions"
        frames = [pd.read_parquet(prediction_dir / pattern.format(seed=seed)) for seed in [7, 42, 123]]
        outputs[name] = _mean_prediction_frames(frames, "raw_probability")
    return outputs


def _score_frame(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    output = frame[["split", "Date", "source_index", "asset_id", "asset_ticker", "y_true"]].copy()
    output["raw_probability"] = np.asarray(probability, dtype=float)
    return output


def _classification_endpoint_frame(
    dataset: Dataset[Any],
    bundle: PooledWindowDataBundle,
    split: str,
    family_map: dict[str, str],
) -> pd.DataFrame:
    frame = dataset.endpoint_metadata().copy()
    frame["asset_ticker"] = frame["asset_id"].map(_reverse_assets(bundle))
    frame["family"] = frame["asset_ticker"].map(family_map).fillna("Unknown")
    frame["y_true"] = dataset_targets(dataset)
    frame["split"] = split
    return frame


def _classification_result_table(
    context: FinalExperimentContext,
    study: str,
    options: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = sorted((context.run_dir / "predictions").glob(f"{study}__*_ensemble.parquet"))
    for path in paths:
        cell = path.stem.removesuffix("_ensemble")
        frame = pd.read_parquet(path)
        frame["family"] = frame["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
        validation = frame[frame["split"].eq("validation")].copy()
        if "selected_probability" not in frame:
            selected = fit_calibration_candidates(
                validation["y_true"].to_numpy(),
                validation["raw_probability"].to_numpy(),
                options["calibration"]["methods"],
                threshold_metric=str(options["calibration"]["threshold_metric"]),
                threshold_grid_size=int(options["calibration"]["threshold_grid_size"]),
                calibration_bins=int(options["calibration"]["bins"]),
            )[0]
            frame["selected_probability"] = selected.calibrator.predict(frame["raw_probability"].to_numpy())
            frame["calibration_method"] = selected.method
            frame["selected_threshold"] = selected.threshold
        threshold = float(frame["selected_threshold"].iloc[0])
        calibration_method = str(frame["calibration_method"].iloc[0])
        for split in ["validation", "test"]:
            part = frame[frame["split"].eq(split)].copy()
            for score_type, column in [("raw", "raw_probability"), ("validation_calibrated", "selected_probability")]:
                row = _classification_metric_record(part, column, threshold)
                rows.append(
                    {
                        "study": study,
                        "cell": cell,
                        "seed": "ensemble",
                        "split": split,
                        "score_type": score_type,
                        "calibration_method": calibration_method,
                        "threshold": threshold,
                        **row,
                    }
                )
        for seed_path in sorted((context.run_dir / "predictions").glob(f"{cell}_seed*.parquet")):
            seed = int(seed_path.stem.rsplit("seed", 1)[1])
            seed_frame = pd.read_parquet(seed_path)
            seed_frame["family"] = seed_frame["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
            for split in ["validation", "test"]:
                part = seed_frame[seed_frame["split"].eq(split)]
                rows.append(
                    {
                        "study": study,
                        "cell": cell,
                        "seed": seed,
                        "split": split,
                        "score_type": "raw",
                        "calibration_method": "none",
                        "threshold": 0.5,
                        **_classification_metric_record(part, "raw_probability", 0.5),
                    }
                )
    return pd.DataFrame(rows)


def _classification_metric_record(frame: pd.DataFrame, probability_column: str, threshold: float) -> dict[str, object]:
    y = frame["y_true"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    metrics = binary_probability_metrics(y, probability, threshold, calibration_bins=10)
    within = _within_group_auc(y, probability, frame["asset_id"].to_numpy(dtype=int))
    grouped = grouped_binary_metrics(frame, probability_column, "asset_ticker", threshold)
    eligible = grouped.dropna(subset=["roc_auc"])
    weights = equal_group_weights(frame, "asset_ticker")
    metrics.update(within)
    metrics["comparable_pairs"] = int(eligible["comparable_pairs"].sum()) if len(eligible) else 0
    metrics["equal_asset_roc_auc"] = float(roc_auc_score(y, probability, sample_weight=weights)) if len(np.unique(y)) == 2 else np.nan
    metrics["per_asset_macro_pr_auc"] = float(eligible["pr_auc"].mean()) if len(eligible) else np.nan
    return metrics


def _classification_order_tests(
    context: FinalExperimentContext,
    study: str,
    options: dict[str, Any],
    *,
    priors: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ensemble_paths = sorted((context.run_dir / "predictions").glob(f"{study}__*_ensemble.parquet"))
    for original_path in ensemble_paths:
        cell = original_path.stem.removesuffix("_ensemble")
        components = cell.split("__")
        if len(components) != 3 or components[1] not in MODEL_FAMILIES:
            continue
        model_name, variant = components[1], components[2]
        original = pd.read_parquet(original_path)
        for method in options["perturbations"]["methods"]:
            perturbed_path = context.run_dir / "predictions" / f"{cell}_ensemble_{method}.parquet"
            if perturbed_path.exists():
                perturbed = pd.read_parquet(perturbed_path)
            else:
                seed_frames: list[pd.DataFrame] = []
                for seed in _seeds(options):
                    model = _load_classification_cell_model(
                        context,
                        study,
                        model_name,
                        variant,
                        seed,
                        options,
                        priors,
                    )
                    transform = _history_preserving_transform(
                        str(method),
                        int(options["lookback"]),
                        int(options["perturbations"]["seed"]),
                    )
                    parts: list[pd.DataFrame] = []
                    for split, dataset in [("validation", context.bundle.validation), ("test", context.bundle.test)]:
                        loader = DataLoader(
                            PerturbedDataset(dataset, transform),
                            batch_size=int(options["training"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                        )
                        y, probability, indices = predict_loader(model, loader, "classification", context.phase6.device)
                        parts.append(
                            _prediction_frame(
                                dataset,
                                indices,
                                y,
                                probability,
                                context.bundle.asset_to_id,
                                split,
                            )
                        )
                    seed_frames.append(pd.concat(parts, ignore_index=True))
                perturbed = _mean_prediction_frames(seed_frames, "raw_probability")
                perturbed.to_parquet(perturbed_path, index=False)
            rows.extend(
                _order_comparison_rows(
                    context,
                    study,
                    cell,
                    str(method),
                    original,
                    perturbed,
                    options,
                )
            )
        if study == "static_dynamic" and "prior_dynamic_residual" in variant:
            if priors is None:
                raise RuntimeError("Residual controls require fitted priors")
            prior_kind = "asset" if variant.startswith("asset") else "family"
            rows.extend(_residual_control_rows(context, cell, original, priors, prior_kind, options))
    return pd.DataFrame(rows)


def _load_classification_cell_model(
    context: FinalExperimentContext,
    study: str,
    model_name: str,
    variant: str,
    seed: int,
    options: dict[str, Any],
    priors: dict[str, Any] | None,
) -> nn.Module:
    if study == "static_dynamic":
        if priors is None:
            raise RuntimeError("Static/dynamic model reconstruction requires priors")
        model = _build_static_model(context, model_name, variant, priors)
    else:
        model = _build_no_id_model(context, model_name, options)
    checkpoint = torch.load(
        context.run_dir / "checkpoints" / f"{study}__{model_name}__{variant}_seed{seed}.pt",
        map_location=context.phase6.device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(context.phase6.device).eval()


def _history_preserving_transform(method: str, lookback: int, seed: int) -> Any:
    if lookback < 2:
        raise ValueError("Order tests require at least two timesteps")
    if method == "reverse":
        return lambda x: torch.cat([torch.flip(x[:-1], dims=[0]), x[-1:]], dim=0)
    if method == "circular_shift":
        return lambda x: torch.cat([torch.roll(x[:-1], shifts=max(1, (lookback - 1) // 2), dims=0), x[-1:]], dim=0)
    if method == "deterministic_permutation":
        order = torch.from_numpy(np.random.default_rng(seed).permutation(lookback - 1).astype(np.int64))
        return lambda x: torch.cat([x[:-1].index_select(0, order), x[-1:]], dim=0)
    raise ValueError(f"Unknown temporal perturbation: {method}")


def _order_comparison_rows(
    context: FinalExperimentContext,
    study: str,
    cell: str,
    method: str,
    original: pd.DataFrame,
    perturbed: pd.DataFrame,
    options: dict[str, Any],
    *,
    compute_uncertainty: bool = False,
) -> list[dict[str, object]]:
    keys = ["split", "Date", "source_index", "asset_id", "asset_ticker", "y_true"]
    left = original.sort_values(keys[:4]).reset_index(drop=True)
    right = perturbed.sort_values(keys[:4]).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise RuntimeError(f"Order perturbation endpoint mismatch for {cell}/{method}")
    rows: list[dict[str, object]] = []
    for split in ["validation", "test"]:
        mask = left["split"].eq(split).to_numpy()
        frame = left.loc[mask, keys].copy()
        frame["original_probability"] = left.loc[mask, "raw_probability"].to_numpy(dtype=float)
        frame["perturbed_probability"] = right.loc[mask, "raw_probability"].to_numpy(dtype=float)
        original_metric = _classification_metric_record(frame, "original_probability", 0.5)
        perturbed_metric = _classification_metric_record(frame, "perturbed_probability", 0.5)
        uncertainty = (
            _date_block_within_difference(
                frame,
                block_size=int(options["evaluation"]["date_block_length"]),
                iterations=int(options["evaluation"]["bootstrap_iterations"]),
                seed=int(options["evaluation"]["bootstrap_seed"]) + zlib.crc32(f"{cell}/{method}/{split}".encode()),
            )
            if compute_uncertainty
            else {"ci_lower": np.nan, "ci_upper": np.nan, "valid_draw_fraction": np.nan}
        )
        rows.append(
            {
                "study": study,
                "cell": cell,
                "split": split,
                "control": method,
                "original_pooled_roc_auc": original_metric["roc_auc"],
                "perturbed_pooled_roc_auc": perturbed_metric["roc_auc"],
                "pooled_auc_drop": original_metric["roc_auc"] - perturbed_metric["roc_auc"],
                "original_within_asset_roc_auc": original_metric["pair_weighted_within_asset_roc_auc"],
                "perturbed_within_asset_roc_auc": perturbed_metric["pair_weighted_within_asset_roc_auc"],
                "within_asset_auc_drop": original_metric["pair_weighted_within_asset_roc_auc"] - perturbed_metric["pair_weighted_within_asset_roc_auc"],
                "within_asset_auc_drop_ci_lower": uncertainty["ci_lower"],
                "within_asset_auc_drop_ci_upper": uncertainty["ci_upper"],
                "bootstrap_valid_draw_fraction": uncertainty["valid_draw_fraction"],
                "prediction_spearman": float(frame["original_probability"].corr(frame["perturbed_probability"], method="spearman")),
                "mean_absolute_probability_change": float(np.mean(np.abs(frame["original_probability"] - frame["perturbed_probability"]))),
            }
        )
    return rows


def _residual_control_rows(
    context: FinalExperimentContext,
    cell: str,
    original: pd.DataFrame,
    priors: dict[str, Any],
    prior_kind: str,
    options: dict[str, Any],
) -> list[dict[str, object]]:
    vector = np.asarray(priors[f"{prior_kind}_vector"], dtype=float)
    seed_paths = [context.run_dir / "predictions" / f"{cell}_seed{seed}.parquet" for seed in _seeds(options)]
    seed_frames = [pd.read_parquet(path).sort_values(["split", "Date", "source_index", "asset_id"]).reset_index(drop=True) for path in seed_paths]
    shuffled_seed_frames: list[pd.DataFrame] = []
    for seed, frame in zip(_seeds(options), seed_frames, strict=True):
        output = frame[["split", "Date", "source_index", "asset_id", "asset_ticker", "y_true"]].copy()
        prior_logits = logit(np.clip(vector[frame["asset_id"].to_numpy(dtype=int)], 1e-6, 1.0 - 1e-6))
        residual = logit(np.clip(frame["raw_probability"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)) - prior_logits
        shuffled = residual.copy()
        for (split, asset_id), indices in frame.groupby(["split", "asset_id"], sort=True).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            rng = np.random.default_rng(int(options["perturbations"]["seed"]) + seed + 1009 * int(asset_id) + zlib.crc32(str(split).encode()))
            shuffled[positions] = residual[positions][rng.permutation(len(positions))]
        output["raw_probability"] = expit(prior_logits + shuffled)
        shuffled_seed_frames.append(output)
    shuffled = _mean_prediction_frames(shuffled_seed_frames, "raw_probability")
    zero = original[["split", "Date", "source_index", "asset_id", "asset_ticker", "y_true"]].copy()
    zero["raw_probability"] = vector[zero["asset_id"].to_numpy(dtype=int)]
    zero.to_parquet(context.run_dir / "predictions" / f"{cell}_ensemble_zero_dynamic_component.parquet", index=False)
    shuffled.to_parquet(context.run_dir / "predictions" / f"{cell}_ensemble_within_asset_shuffled_dynamic_component.parquet", index=False)
    rows: list[dict[str, object]] = []
    for name, control in [("zero_dynamic_component", zero), ("within_asset_shuffled_dynamic_component", shuffled)]:
        rows.extend(_order_comparison_rows(context, "static_dynamic", cell, name, original, control, options))
    return rows


def _date_block_within_difference(
    frame: pd.DataFrame,
    *,
    block_size: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    ordered_dates = pd.DatetimeIndex(frame["Date"].drop_duplicates().sort_values())
    groups = {date: part.index.to_numpy(dtype=int) for date, part in frame.reset_index(drop=True).groupby("Date", sort=False)}
    work = frame.reset_index(drop=True)
    y = work["y_true"].to_numpy(dtype=int)
    asset = work["asset_id"].to_numpy(dtype=int)
    original = work["original_probability"].to_numpy(dtype=float)
    perturbed = work["perturbed_probability"].to_numpy(dtype=float)
    blocks = int(np.ceil(len(ordered_dates) / block_size))
    starts_by_draw = np.random.default_rng(seed).integers(
        0,
        len(ordered_dates),
        size=(iterations, blocks),
    )

    def _draw(starts: np.ndarray) -> float:
        sampled = [ordered_dates[(int(start) + offset) % len(ordered_dates)] for start in starts for offset in range(block_size)]
        indices = np.concatenate([groups[date] for date in sampled[: len(ordered_dates)]])
        base = _fast_within_auc(y[indices], original[indices], asset[indices])
        changed = _fast_within_auc(y[indices], perturbed[indices], asset[indices])
        return base - changed

    draws = np.asarray(
        joblib.Parallel(n_jobs=_bootstrap_jobs(), prefer="processes")(
            joblib.delayed(_draw)(starts) for starts in starts_by_draw
        ),
        dtype=float,
    )
    valid = draws[np.isfinite(draws)]
    return {
        "ci_lower": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
        "ci_upper": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
        "valid_draw_fraction": float(len(valid) / iterations),
    }


def _fast_within_auc(y: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> float:
    """Compute pair-weighted within-group AUC using the rank-sum identity."""
    aucs: list[float] = []
    pair_counts: list[int] = []
    for group in np.unique(groups):
        selected = groups == group
        labels = y[selected].astype(int, copy=False)
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives == 0 or negatives == 0:
            continue
        ranks = rankdata(probability[selected], method="average")
        u_statistic = float(ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        pairs = positives * negatives
        aucs.append(u_statistic / pairs)
        pair_counts.append(pairs)
    return float(np.average(aucs, weights=pair_counts)) if aucs else np.nan


def _fast_date_block_bootstrap_within_auc(
    frame: pd.DataFrame,
    probability_column: str,
    block_size: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Circular global-date bootstrap using a low-overhead grouped AUC kernel."""
    work = frame.reset_index(drop=True)
    dates = pd.DatetimeIndex(work["Date"].drop_duplicates().sort_values())
    date_rows = {date: part.index.to_numpy(dtype=int) for date, part in work.groupby("Date", sort=False)}
    y = work["y_true"].to_numpy(dtype=int)
    probability = work[probability_column].to_numpy(dtype=float)
    asset = work["asset_id"].to_numpy(dtype=int)
    estimate = _fast_within_auc(y, probability, asset)
    block_count = int(np.ceil(len(dates) / block_size))
    starts_by_draw = np.random.default_rng(seed).integers(
        0,
        len(dates),
        size=(iterations, block_count),
    )

    def _draw(starts: np.ndarray) -> float:
        sampled_dates = [dates[(int(start) + offset) % len(dates)] for start in starts for offset in range(block_size)]
        indices = np.concatenate([date_rows[date] for date in sampled_dates[: len(dates)]])
        return _fast_within_auc(y[indices], probability[indices], asset[indices])

    draws = np.asarray(
        joblib.Parallel(n_jobs=_bootstrap_jobs(), prefer="processes")(
            joblib.delayed(_draw)(starts) for starts in starts_by_draw
        ),
        dtype=float,
    )
    valid = draws[np.isfinite(draws)]
    return {
        "estimate": estimate,
        "ci_lower": float(np.quantile(valid, 0.025)) if len(valid) else np.nan,
        "ci_upper": float(np.quantile(valid, 0.975)) if len(valid) else np.nan,
        "valid_draw_fraction": float(len(valid) / iterations),
    }


def _bootstrap_jobs() -> int:
    """Bound bootstrap parallelism to physical-core-scale resource use."""
    return max(1, min(12, os.cpu_count() or 1))


def _continuous_asset_intercepts(
    bundle: PooledWindowDataBundle,
    target_mean: float,
    target_scale: float,
) -> np.ndarray:
    frame = bundle.train.endpoint_metadata().copy()
    frame["y_true"] = dataset_targets(bundle.train)
    means = frame.groupby("asset_id", observed=True)["y_true"].mean()
    return np.asarray(
        [(float(means.get(asset_id, target_mean)) - target_mean) / target_scale for asset_id in range(len(bundle.asset_to_id))],
        dtype=float,
    )


def _continuous_prediction_frame(
    model: nn.Module,
    loader: DataLoader,
    base_dataset: Dataset[Any],
    bundle: PooledWindowDataBundle,
    split: str,
    target_mean: float,
    target_scale: float,
    device: torch.device,
) -> pd.DataFrame:
    y_scaled, prediction_scaled, indices = predict_loader(model, loader, "regression", device)
    metadata = base_dataset.endpoint_metadata().set_index("source_index").loc[indices].reset_index()
    output = metadata[["Date", "source_index", "asset_id"]].copy()
    output["asset_ticker"] = output["asset_id"].map(_reverse_assets(bundle))
    output["y_true"] = np.clip(target_mean + target_scale * y_scaled, 0.0, 1.0)
    output["prediction"] = np.clip(target_mean + target_scale * prediction_scaled, 0.0, 1.0)
    output["split"] = split
    return output


def _write_regression_ensemble(context: FinalExperimentContext, cell: str, seeds: list[int]) -> pd.DataFrame:
    frames = [pd.read_parquet(context.run_dir / "predictions" / f"{cell}_seed{seed}.parquet") for seed in seeds]
    output = _mean_prediction_frames(frames, "prediction")
    output.to_parquet(context.run_dir / "predictions" / f"{cell}_ensemble.parquet", index=False)
    return output


def _continuous_baselines(
    context: FinalExperimentContext,
    bundle: PooledWindowDataBundle,
    target_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    horizon = int(context.continuous_options["target"]["horizon"])
    target = str(context.continuous_options["target"]["name"])
    panel = (
        target_panel.reset_index()
        .sort_values(["Ticker", "Date"], kind="stable")
        .set_index("Date")
    )
    panel["known_target_lag10"] = panel.groupby("Ticker", observed=True)[target].shift(horizon)
    frames = {
        "train": _continuous_endpoint_frame(bundle.train, bundle, "train"),
        "validation": _continuous_endpoint_frame(bundle.validation, bundle, "validation"),
        "test": _continuous_endpoint_frame(bundle.test, bundle, "test"),
    }
    feature_columns = [
        "volatility_20d",
        "downside_volatility_20d",
        "rolling_drawdown_60d",
        "cum_return_20d",
        "known_target_lag10",
    ]
    panel_values = panel.reset_index()[["Date", "Ticker", *feature_columns]].rename(columns={"Ticker": "asset_ticker"})
    for split, frame in frames.items():
        frames[split] = frame.merge(panel_values, on=["Date", "asset_ticker"], how="left", validate="one_to_one")
    training = frames["train"]
    asset_means = training.groupby("asset_ticker", observed=True)["y_true"].mean()
    global_mean = float(training["y_true"].mean())
    medians = training[feature_columns].median()
    for frame in frames.values():
        frame[feature_columns] = frame[feature_columns].fillna(medians)
    outputs: dict[str, pd.DataFrame] = {}
    endpoint = pd.concat([frames["validation"], frames["test"]], ignore_index=True)
    outputs["training_asset_mean"] = _continuous_score_frame(
        endpoint,
        endpoint["asset_ticker"].map(asset_means).fillna(global_mean).to_numpy(dtype=float),
    )
    outputs["recent_volatility_20d"] = _continuous_score_frame(
        endpoint,
        np.sqrt(horizon) * endpoint["volatility_20d"].to_numpy(dtype=float),
    )
    outputs["recent_drawdown_60d"] = _continuous_score_frame(
        endpoint,
        np.maximum(0.0, -endpoint["rolling_drawdown_60d"].to_numpy(dtype=float)),
    )
    outputs["last_observable_target"] = _continuous_score_frame(
        endpoint,
        endpoint["known_target_lag10"].fillna(endpoint["asset_ticker"].map(asset_means)).fillna(global_mean).to_numpy(dtype=float),
    )
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    train_weights = equal_group_weights(training, "asset_ticker")
    ridge.fit(
        training[feature_columns].to_numpy(dtype=float),
        training["y_true"].to_numpy(dtype=float),
        ridge__sample_weight=train_weights,
    )
    joblib.dump(ridge, context.run_dir / "checkpoints" / "continuous_downside__ridge_sequence_summary.joblib")
    outputs["ridge_sequence_summary"] = _continuous_score_frame(
        endpoint,
        ridge.predict(endpoint[feature_columns].to_numpy(dtype=float)),
    )
    return outputs


def _continuous_endpoint_frame(dataset: Dataset[Any], bundle: PooledWindowDataBundle, split: str) -> pd.DataFrame:
    frame = dataset.endpoint_metadata().copy()
    frame["asset_ticker"] = frame["asset_id"].map(_reverse_assets(bundle))
    frame["y_true"] = dataset_targets(dataset).astype(float)
    frame["split"] = split
    return frame


def _continuous_score_frame(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    output = frame[["split", "Date", "source_index", "asset_id", "asset_ticker", "y_true"]].copy()
    output["prediction"] = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    return output


def _continuous_result_table(
    context: FinalExperimentContext,
    bundle: PooledWindowDataBundle,
    target_panel: pd.DataFrame,
    options: dict[str, Any],
) -> pd.DataFrame:
    del target_panel
    train_target = dataset_targets(bundle.train).astype(float)
    tail_threshold = float(np.quantile(train_target, 0.90))
    rows: list[dict[str, object]] = []
    paths = sorted((context.run_dir / "predictions").glob("continuous_downside__*_ensemble.parquet"))
    for path in paths:
        cell = path.stem.removesuffix("_ensemble")
        frame = pd.read_parquet(path)
        frame["family"] = frame["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
        for split in ["validation", "test"]:
            part = frame[frame["split"].eq(split)].copy()
            rows.append(
                {
                    "study": "continuous_downside",
                    "cell": cell,
                    "seed": "ensemble",
                    "split": split,
                    **_continuous_metric_record(part, tail_threshold),
                }
            )
        for seed_path in sorted((context.run_dir / "predictions").glob(f"{cell}_seed*.parquet")):
            seed = int(seed_path.stem.rsplit("seed", 1)[1])
            seed_frame = pd.read_parquet(seed_path)
            seed_frame["family"] = seed_frame["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
            for split in ["validation", "test"]:
                part = seed_frame[seed_frame["split"].eq(split)]
                rows.append(
                    {
                        "study": "continuous_downside",
                        "cell": cell,
                        "seed": seed,
                        "split": split,
                        **_continuous_metric_record(part, tail_threshold),
                    }
                )
    return pd.DataFrame(rows)


def _continuous_metric_record(frame: pd.DataFrame, tail_threshold: float) -> dict[str, object]:
    y = frame["y_true"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    asset_mae = frame.assign(error=np.abs(y - prediction)).groupby("asset_ticker", observed=True)["error"].mean()
    tail = y >= tail_threshold
    predicted_cutoff = float(np.quantile(prediction, 0.90))
    selected = prediction >= predicted_cutoff
    rho = spearmanr(y, prediction).statistic if np.std(y) > 0.0 and np.std(prediction) > 0.0 else np.nan
    return {
        "n_obs": len(frame),
        "n_assets": int(frame["asset_ticker"].nunique()),
        "n_dates": int(frame["Date"].nunique()),
        "mae": float(mean_absolute_error(y, prediction)),
        "equal_asset_mae": float(asset_mae.mean()),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "spearman": float(rho),
        "tail_threshold_training_q90": tail_threshold,
        "tail_n": int(tail.sum()),
        "tail_mae": float(mean_absolute_error(y[tail], prediction[tail])) if tail.any() else np.nan,
        "top_decile_actual_mean": float(np.mean(y[selected])) if selected.any() else np.nan,
        "overall_actual_mean": float(np.mean(y)),
        "top_decile_risk_enrichment": float(np.mean(y[selected]) / max(np.mean(y), 1e-12)) if selected.any() else np.nan,
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def _continuous_order_and_robustness(
    context: FinalExperimentContext,
    bundle: PooledWindowDataBundle,
    scaled: dict[str, TargetScaledDataset],
    target_panel: pd.DataFrame,
    target_mean: float,
    target_scale: float,
    intercepts: np.ndarray,
    options: dict[str, Any],
) -> pd.DataFrame:
    del target_panel
    rows: list[dict[str, object]] = []
    tail_threshold = float(np.quantile(dataset_targets(bundle.train).astype(float), 0.90))
    neural_paths = sorted((context.run_dir / "predictions").glob("continuous_downside__*__*_ensemble.parquet"))
    all_paths = sorted((context.run_dir / "predictions").glob("continuous_downside__*_ensemble.parquet"))
    for path in neural_paths:
        cell = path.stem.removesuffix("_ensemble")
        _, model_name, variant = cell.split("__")
        original = pd.read_parquet(path)
        for method in options["perturbations"]["methods"]:
            perturbed_path = context.run_dir / "predictions" / f"{cell}_ensemble_{method}.parquet"
            if perturbed_path.exists():
                perturbed = pd.read_parquet(perturbed_path)
            else:
                seed_frames: list[pd.DataFrame] = []
                transform = _history_preserving_transform(
                    str(method),
                    int(options["lookback"]),
                    int(options["perturbations"]["seed"]),
                )
                for seed in _seeds(options):
                    model = _build_continuous_model(context, model_name, variant, bundle, intercepts)
                    checkpoint = torch.load(
                        context.run_dir / "checkpoints" / f"{cell}_seed{seed}.pt",
                        map_location=context.phase6.device,
                        weights_only=False,
                    )
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    model = model.to(context.phase6.device).eval()
                    parts: list[pd.DataFrame] = []
                    for split in ["validation", "test"]:
                        base = bundle.validation if split == "validation" else bundle.test
                        loader = DataLoader(
                            PerturbedDataset(scaled[split], transform),
                            batch_size=int(options["training"]["batch_size"]),
                            shuffle=False,
                            num_workers=0,
                        )
                        parts.append(
                            _continuous_prediction_frame(
                                model,
                                loader,
                                base,
                                bundle,
                                split,
                                target_mean,
                                target_scale,
                                context.phase6.device,
                            )
                        )
                    seed_frames.append(pd.concat(parts, ignore_index=True))
                perturbed = _mean_prediction_frames(seed_frames, "prediction")
                perturbed.to_parquet(perturbed_path, index=False)
            for split in ["validation", "test"]:
                left = original[original["split"].eq(split)].sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
                right = perturbed[perturbed["split"].eq(split)].sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
                if not left[["Date", "source_index", "asset_id", "y_true"]].equals(right[["Date", "source_index", "asset_id", "y_true"]]):
                    raise RuntimeError(f"Continuous perturbation endpoint mismatch for {cell}/{method}")
                base_metric = _continuous_metric_record(left, tail_threshold)
                changed_metric = _continuous_metric_record(right, tail_threshold)
                paired = (
                    _date_block_equal_asset_mae_difference(
                        left,
                        right,
                        iterations=int(options["evaluation"]["bootstrap_iterations"]),
                        block_size=int(options["evaluation"]["date_block_length"]),
                        seed=int(options["evaluation"]["bootstrap_seed"]) + zlib.crc32(f"{cell}/{method}".encode()),
                        difference="changed_minus_base",
                    )
                    if split == "test"
                    else {"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                )
                rows.append(
                    {
                        "analysis": "temporal_order",
                        "cell": cell,
                        "split": split,
                        "group": method,
                        "n_obs": len(left),
                        "base_equal_asset_mae": base_metric["equal_asset_mae"],
                        "changed_equal_asset_mae": changed_metric["equal_asset_mae"],
                        "equal_asset_mae_deterioration": changed_metric["equal_asset_mae"] - base_metric["equal_asset_mae"],
                        "base_spearman": base_metric["spearman"],
                        "changed_spearman": changed_metric["spearman"],
                        "prediction_spearman": float(left["prediction"].corr(right["prediction"], method="spearman")),
                        "paired_difference_estimate": paired["estimate"],
                        "paired_difference_ci_lower": paired["ci_lower"],
                        "paired_difference_ci_upper": paired["ci_upper"],
                    }
                )
    for path in all_paths:
        cell = path.stem.removesuffix("_ensemble")
        frame = pd.read_parquet(path)
        frame["family"] = frame["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
        test = frame[frame["split"].eq("test")].copy()
        for family, part in test.groupby("family", observed=True):
            rows.append(
                {
                    "analysis": "family",
                    "cell": cell,
                    "split": "test",
                    "group": str(family),
                    **_continuous_metric_record(part, tail_threshold),
                }
            )
        for start, end in options["evaluation"]["predefined_subperiods"]:
            part = test[test["Date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            if len(part):
                rows.append(
                    {
                        "analysis": "subperiod",
                        "cell": cell,
                        "split": "test",
                        "group": f"{start}_to_{end}",
                        **_continuous_metric_record(part, tail_threshold),
                    }
                )
        ordered = test.sort_values(["asset_ticker", "Date"]).copy()
        ordered["asset_position"] = ordered.groupby("asset_ticker", observed=True).cumcount()
        for offset in range(int(options["evaluation"]["nonoverlap_stride"])):
            part = ordered[ordered["asset_position"].mod(int(options["evaluation"]["nonoverlap_stride"])).eq(offset)]
            rows.append(
                {
                    "analysis": "nonoverlap_offset",
                    "cell": cell,
                    "split": "test",
                    "group": str(offset),
                    **_continuous_metric_record(part, tail_threshold),
                }
            )
    result_frame = pd.DataFrame(rows)
    validation = _continuous_result_table(context, bundle, pd.DataFrame(), options)
    validation = validation[(validation["seed"].eq("ensemble")) & (validation["split"].eq("validation"))]
    baselines = validation[~validation["cell"].str.contains("__mlp__|__transformer_encoder__", regex=True)]
    if len(baselines):
        best_baseline = str(baselines.sort_values("equal_asset_mae").iloc[0]["cell"])
        comparator = pd.read_parquet(context.run_dir / "predictions" / f"{best_baseline}_ensemble.parquet")
        comparator = comparator[comparator["split"].eq("test")]
        for path in neural_paths:
            cell = path.stem.removesuffix("_ensemble")
            model = pd.read_parquet(path)
            model = model[model["split"].eq("test")]
            paired = _date_block_equal_asset_mae_difference(
                model,
                comparator,
                iterations=int(options["evaluation"]["bootstrap_iterations"]),
                block_size=int(options["evaluation"]["date_block_length"]),
                seed=int(options["evaluation"]["bootstrap_seed"]) + zlib.crc32(cell.encode()),
                difference="changed_minus_base",
            )
            result_frame = pd.concat(
                [
                    result_frame,
                    pd.DataFrame(
                        [
                            {
                                "analysis": "best_baseline_comparison",
                                "cell": cell,
                                "split": "test",
                                "group": best_baseline,
                                "paired_difference_estimate": paired["estimate"],
                                "paired_difference_ci_lower": paired["ci_lower"],
                                "paired_difference_ci_upper": paired["ci_upper"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return result_frame


def _date_block_equal_asset_mae_difference(
    base: pd.DataFrame,
    changed: pd.DataFrame,
    *,
    iterations: int,
    block_size: int,
    seed: int,
    difference: str,
) -> dict[str, float]:
    if difference != "changed_minus_base":
        raise ValueError("Only changed_minus_base is supported")
    endpoint_keys = ["Date", "source_index", "asset_id", "asset_ticker"]
    keys = [*endpoint_keys, "y_true"]
    left = base.sort_values(endpoint_keys[:3]).reset_index(drop=True)
    right = changed.sort_values(endpoint_keys[:3]).reset_index(drop=True)
    targets_match = np.allclose(
        left["y_true"].to_numpy(dtype=float),
        right["y_true"].to_numpy(dtype=float),
        atol=1e-7,
        rtol=0.0,
    )
    if not left[endpoint_keys].equals(right[endpoint_keys]) or not targets_match:
        raise RuntimeError("Paired MAE frames do not share endpoints")
    work = left[keys].copy()
    work["base_error"] = np.abs(left["prediction"].to_numpy() - left["y_true"].to_numpy())
    work["changed_error"] = np.abs(right["prediction"].to_numpy() - right["y_true"].to_numpy())
    estimate = _equal_asset_error(work, "changed_error") - _equal_asset_error(work, "base_error")
    dates = pd.DatetimeIndex(work["Date"].drop_duplicates().sort_values())
    groups = {date: part.index.to_numpy(dtype=int) for date, part in work.groupby("Date", sort=False)}
    block_count = int(np.ceil(len(dates) / block_size))
    starts_by_draw = np.random.default_rng(seed).integers(0, len(dates), size=(iterations, block_count))

    def _draw(starts: np.ndarray) -> float:
        sampled_dates = [dates[(int(start) + offset) % len(dates)] for start in starts for offset in range(block_size)]
        indices = np.concatenate([groups[date] for date in sampled_dates[: len(dates)]])
        sample = work.iloc[indices]
        return _equal_asset_error(sample, "changed_error") - _equal_asset_error(sample, "base_error")

    draws = np.asarray(
        joblib.Parallel(n_jobs=_bootstrap_jobs(), prefer="processes")(
            joblib.delayed(_draw)(starts) for starts in starts_by_draw
        ),
        dtype=float,
    )
    return {
        "estimate": float(estimate),
        "ci_lower": float(np.nanquantile(draws, 0.025)),
        "ci_upper": float(np.nanquantile(draws, 0.975)),
    }


def _date_block_regression_intervals(
    frame: pd.DataFrame,
    *,
    iterations: int,
    block_size: int,
    seed: int,
) -> dict[str, float]:
    """Return date-block intervals for equal-asset MAE and Spearman correlation."""
    work = frame.reset_index(drop=True).copy()
    work["absolute_error"] = np.abs(work["prediction"] - work["y_true"])
    dates = pd.DatetimeIndex(work["Date"].drop_duplicates().sort_values())
    groups = {date: part.index.to_numpy(dtype=int) for date, part in work.groupby("Date", sort=False)}
    block_count = int(np.ceil(len(dates) / block_size))
    starts_by_draw = np.random.default_rng(seed).integers(0, len(dates), size=(iterations, block_count))

    def _draw(starts: np.ndarray) -> tuple[float, float]:
        sampled_dates = [dates[(int(start) + offset) % len(dates)] for start in starts for offset in range(block_size)]
        indices = np.concatenate([groups[date] for date in sampled_dates[: len(dates)]])
        sample = work.iloc[indices]
        mae = _equal_asset_error(sample, "absolute_error")
        rho = spearmanr(sample["y_true"], sample["prediction"]).statistic
        return mae, float(rho)

    draws = np.asarray(
        joblib.Parallel(n_jobs=_bootstrap_jobs(), prefer="processes")(
            joblib.delayed(_draw)(starts) for starts in starts_by_draw
        ),
        dtype=float,
    )
    return {
        "equal_asset_mae_ci_lower": float(np.nanquantile(draws[:, 0], 0.025)),
        "equal_asset_mae_ci_upper": float(np.nanquantile(draws[:, 0], 0.975)),
        "spearman_ci_lower": float(np.nanquantile(draws[:, 1], 0.025)),
        "spearman_ci_upper": float(np.nanquantile(draws[:, 1], 0.975)),
    }


def _equal_asset_error(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("asset_ticker", observed=True)[column].mean().mean())


def _static_dynamic_verdict(
    context: FinalExperimentContext,
    results: pd.DataFrame,
    order: pd.DataFrame,
) -> str:
    options = context.static_options
    ensemble = results[
        results["seed"].eq("ensemble")
        & results["score_type"].eq("raw")
        & results["split"].eq("validation")
        & results["cell"].str.contains("prior_dynamic_residual")
    ].copy()
    if ensemble.empty:
        raise RuntimeError("No static/dynamic residual results were available")
    winner = ensemble.sort_values("pair_weighted_within_asset_roc_auc", ascending=False).iloc[0]
    cell = str(winner["cell"])
    _attach_selected_classification_uncertainty(
        context,
        results,
        order,
        cell,
        options,
        controls=[*options["perturbations"]["methods"], "within_asset_shuffled_dynamic_component"],
    )
    winner = results.loc[winner.name]
    model_name = cell.split("__")[1]
    no_id = ensemble[ensemble["cell"].eq(f"static_dynamic__{model_name}__no_id_temporal_model")]
    if no_id.empty:
        no_id = results[
            results["cell"].eq(f"static_dynamic__{model_name}__no_id_temporal_model")
            & results["seed"].eq("ensemble")
            & results["split"].eq("validation")
            & results["score_type"].eq("raw")
        ]
    simple = results[
        results["cell"].eq("static_dynamic__flattened_logistic_no_id")
        & results["seed"].eq("ensemble")
        & results["split"].eq("validation")
        & results["score_type"].eq("raw")
    ]
    comparator_auc = max(
        float(no_id.iloc[0]["pair_weighted_within_asset_roc_auc"]) if len(no_id) else -np.inf,
        float(simple.iloc[0]["pair_weighted_within_asset_roc_auc"]) if len(simple) else -np.inf,
    )
    order_rows = order[
        order["cell"].eq(cell)
        & order["split"].eq("validation")
        & order["control"].isin(options["perturbations"]["methods"])
    ]
    shuffled_row = order[
        order["cell"].eq(cell)
        & order["split"].eq("validation")
        & order["control"].eq("within_asset_shuffled_dynamic_component")
    ]
    seed_rows = results[
        results["cell"].eq(cell)
        & results["split"].eq("validation")
        & results["score_type"].eq("raw")
        & results["seed"].ne("ensemble")
    ]
    gate = options["promotion_gate"]
    prior_name = "asset_prior" if "asset_prior" in cell else "family_prior"
    prior_row = results[
        results["cell"].eq(f"static_dynamic__{prior_name}")
        & results["seed"].eq("ensemble")
        & results["split"].eq("validation")
        & results["score_type"].eq("raw")
    ].iloc[0]
    candidate_frame = pd.read_parquet(context.run_dir / "predictions" / f"{cell}_ensemble.parquet")
    comparator_candidates = []
    for comparator_cell in [
        f"static_dynamic__{model_name}__no_id_temporal_model",
        "static_dynamic__flattened_logistic_no_id",
    ]:
        path = context.run_dir / "predictions" / f"{comparator_cell}_ensemble.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            validation_frame = frame[frame["split"].eq("validation")]
            comparator_candidates.append(
                (
                    comparator_cell,
                    _fast_within_auc(
                        validation_frame["y_true"].to_numpy(dtype=int),
                        validation_frame["raw_probability"].to_numpy(dtype=float),
                        validation_frame["asset_id"].to_numpy(dtype=int),
                    ),
                    frame,
                )
            )
    comparator_cell, comparator_auc, comparator_frame = max(comparator_candidates, key=lambda item: item[1])
    paired_frame = _paired_classification_frame(
        candidate_frame,
        comparator_frame,
        split="validation",
    )
    paired_lift = _date_block_within_difference(
        paired_frame,
        block_size=int(options["evaluation"]["date_block_length"]),
        iterations=int(options["evaluation"]["bootstrap_iterations"]),
        seed=int(options["evaluation"]["bootstrap_seed"]) + zlib.crc32(f"{cell}/{comparator_cell}/lift".encode()),
    )
    zero_path = context.run_dir / "predictions" / f"{cell}_ensemble_zero_dynamic_component.parquet"
    prior_path = context.run_dir / "predictions" / f"static_dynamic__{prior_name}_ensemble.parquet"
    zero = pd.read_parquet(zero_path).sort_values(["split", "Date", "source_index", "asset_id"])
    static_prior = pd.read_parquet(prior_path).sort_values(["split", "Date", "source_index", "asset_id"])
    zero_exact = bool(np.array_equal(zero["raw_probability"].to_numpy(), static_prior["raw_probability"].to_numpy()))
    conditions = {
        "within_auc": float(winner["pair_weighted_within_asset_roc_auc"]) >= float(gate["minimum_test_within_asset_auc"]),
        "within_auc_lower_bound": float(winner["within_asset_ci_lower"]) > float(gate["minimum_lower_95ci_within_asset_auc"]),
        "macro_auc": float(winner["per_asset_macro_roc_auc"]) >= 0.55,
        "dynamic_lift": float(winner["pair_weighted_within_asset_roc_auc"]) - comparator_auc >= float(gate["minimum_lift_over_simple_dynamic_auc"]) and float(paired_lift["ci_lower"]) > 0.0,
        "proper_scores": float(winner["brier_score"]) < float(prior_row["brier_score"]) and float(winner["log_loss"]) < float(prior_row["log_loss"]),
        "order_sensitivity": len(order_rows) == len(options["perturbations"]["methods"]) and order_rows["within_asset_auc_drop"].ge(float(gate["minimum_each_order_auc_drop"])).all() and order_rows["within_asset_auc_drop_ci_lower"].gt(0.0).all(),
        "shuffled_residual": len(shuffled_row) == 1 and float(shuffled_row.iloc[0]["within_asset_auc_drop"]) >= float(gate["require_shuffled_residual_auc_drop"]) and float(shuffled_row.iloc[0]["within_asset_auc_drop_ci_lower"]) > 0.0,
        "seed_stability": seed_rows["pair_weighted_within_asset_roc_auc"].gt(0.50).all() and int(seed_rows["pair_weighted_within_asset_roc_auc"].ge(float(gate["minimum_seed_within_asset_auc"])).sum()) >= int(gate["minimum_passing_seeds"]),
        "zero_residual_identity": zero_exact,
    }
    passed = all(conditions.values())
    condition_text = "\n".join(f"- {name}: **{'pass' if value else 'fail'}**" for name, value in conditions.items())
    return "\n".join(
        [
            "# IFDDRP Static-Prior + Dynamic-Residual Verdict",
            "",
            "Evidence class: historical held-out but adaptive. The validation period and historical test have been opened in prior phases; this result cannot provide independent confirmation.",
            "",
            f"The validation-selected residual arm was `{cell}` with pair-weighted within-asset ROC-AUC `{float(winner['pair_weighted_within_asset_roc_auc']):.4f}` and per-asset macro ROC-AUC `{float(winner['per_asset_macro_roc_auc']):.4f}`.",
            f"Its strongest matched dynamic comparator was `{comparator_cell}` at `{comparator_auc:.4f}`. The paired date-block lift interval was `[{float(paired_lift['ci_lower']):.4f}, {float(paired_lift['ci_upper']):.4f}]`.",
            "",
            condition_text,
            "",
            f"Overall promotion gate: **{'passed' if passed else 'failed'}**.",
            "",
            "A fixed additive prior cannot itself alter ranking within one asset. Any within-asset difference is attributable to the fitted residual and training interaction. Pooled AUC is not used as evidence of timing skill.",
        ]
    )


def _within_objective_verdict(
    context: FinalExperimentContext,
    results: pd.DataFrame,
    order: pd.DataFrame,
    pair_audit: pd.DataFrame,
) -> str:
    options = context.objective_options
    validation = results[
        results["seed"].eq("ensemble")
        & results["score_type"].eq("raw")
        & results["split"].eq("validation")
    ].copy()
    candidates = validation[~validation["cell"].str.endswith("pooled_bce")]
    winner = candidates.sort_values("pair_weighted_within_asset_roc_auc", ascending=False).iloc[0]
    cell = str(winner["cell"])
    _attach_selected_classification_uncertainty(
        context,
        results,
        order,
        cell,
        options,
        controls=list(options["perturbations"]["methods"]),
    )
    winner = results.loc[winner.name]
    model_name = cell.split("__")[1]
    pooled = validation[validation["cell"].eq(f"within_objective__{model_name}__pooled_bce")].iloc[0]
    order_rows = order[
        order["cell"].eq(cell)
        & order["split"].eq("validation")
        & order["control"].isin(options["perturbations"]["methods"])
    ]
    seeds = results[
        results["cell"].eq(cell)
        & results["split"].eq("validation")
        & results["score_type"].eq("raw")
        & results["seed"].ne("ensemble")
    ]
    gate = options["promotion_gate"]
    candidate_frame = pd.read_parquet(context.run_dir / "predictions" / f"{cell}_ensemble.parquet")
    pooled_frame = pd.read_parquet(
        context.run_dir / "predictions" / f"within_objective__{model_name}__pooled_bce_ensemble.parquet"
    )
    paired_frame = _paired_classification_frame(candidate_frame, pooled_frame, split="validation")
    paired_lift = _date_block_within_difference(
        paired_frame,
        block_size=int(options["evaluation"]["date_block_length"]),
        iterations=int(options["evaluation"]["bootstrap_iterations"]),
        seed=int(options["evaluation"]["bootstrap_seed"]) + zlib.crc32(f"{cell}/pooled_bce/lift".encode()),
    )
    conditions = {
        "within_auc": float(winner["pair_weighted_within_asset_roc_auc"]) >= float(gate["minimum_test_within_asset_auc"]),
        "within_auc_lower_bound": float(winner["within_asset_ci_lower"]) > float(gate["minimum_lower_95ci_within_asset_auc"]),
        "lift_over_pooled_bce": float(winner["pair_weighted_within_asset_roc_auc"] - pooled["pair_weighted_within_asset_roc_auc"]) >= float(gate["minimum_lift_over_pooled_bce_auc"]) and float(paired_lift["ci_lower"]) > 0.0,
        "order_sensitivity": len(order_rows) == len(options["perturbations"]["methods"]) and order_rows["within_asset_auc_drop"].ge(float(gate["minimum_each_order_auc_drop"])).all() and order_rows["within_asset_auc_drop_ci_lower"].gt(0.0).all(),
        "seed_stability": seeds["pair_weighted_within_asset_roc_auc"].gt(0.50).all() and int(seeds["pair_weighted_within_asset_roc_auc"].ge(float(gate["minimum_seed_within_asset_auc"])).sum()) >= int(gate["minimum_passing_seeds"]),
    }
    passed = all(conditions.values())
    selected_pairs = pair_audit.groupby("split", observed=True)["selected_pairs"].sum().to_dict()
    condition_text = "\n".join(f"- {name}: **{'pass' if value else 'fail'}**" for name, value in conditions.items())
    return "\n".join(
        [
            "# IFDDRP Within-Asset Objective Verdict",
            "",
            "Evidence class: historical held-out but adaptive.",
            "",
            f"The frozen pair registries contain `{int(selected_pairs.get('train', 0)):,}` training pairs and `{int(selected_pairs.get('validation', 0)):,}` validation pairs. These are weighting support, not independent observations.",
            "",
            f"The validation-selected aligned objective was `{cell}` with within-asset ROC-AUC `{float(winner['pair_weighted_within_asset_roc_auc']):.4f}` versus `{float(pooled['pair_weighted_within_asset_roc_auc']):.4f}` for its matched pooled-BCE control.",
            f"The paired date-block lift interval over the matched control was `[{float(paired_lift['ci_lower']):.4f}, {float(paired_lift['ci_upper']):.4f}]`.",
            "",
            condition_text,
            "",
            f"Overall promotion gate: **{'passed' if passed else 'failed'}**.",
        ]
    )


def _paired_classification_frame(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Align two prediction frames exactly before paired dependence-aware inference."""
    endpoint_keys = ["split", "Date", "source_index", "asset_id", "asset_ticker"]
    keys = [*endpoint_keys, "y_true"]
    left = candidate[candidate["split"].eq(split)].sort_values(endpoint_keys[:4]).reset_index(drop=True)
    right = comparator[comparator["split"].eq(split)].sort_values(endpoint_keys[:4]).reset_index(drop=True)
    labels_match = np.array_equal(
        left["y_true"].to_numpy(dtype=int),
        right["y_true"].to_numpy(dtype=int),
    )
    if not left[endpoint_keys].equals(right[endpoint_keys]) or not labels_match:
        raise RuntimeError(f"Paired prediction endpoint mismatch for split {split}")
    output = left[keys].copy()
    output["original_probability"] = left["raw_probability"].to_numpy(dtype=float)
    output["perturbed_probability"] = right["raw_probability"].to_numpy(dtype=float)
    return output


def _attach_selected_classification_uncertainty(
    context: FinalExperimentContext,
    results: pd.DataFrame,
    order: pd.DataFrame,
    cell: str,
    options: dict[str, Any],
    *,
    controls: list[str],
) -> None:
    """Attach full block intervals only to the validation-selected ensemble."""
    original = pd.read_parquet(context.run_dir / "predictions" / f"{cell}_ensemble.parquet")
    study = cell.split("__", 1)[0]
    report_stem = {
        "static_dynamic": "ifddrp_static_dynamic",
        "within_objective": "ifddrp_within_asset_objective",
    }[study]
    existing_results_path = context.table_dir / f"{report_stem}_results.csv"
    existing_order_path = context.table_dir / f"{report_stem}_order_tests.csv"
    existing_results = pd.read_csv(existing_results_path) if existing_results_path.exists() else pd.DataFrame()
    existing_order = pd.read_csv(existing_order_path) if existing_order_path.exists() else pd.DataFrame()
    block_size = int(options["evaluation"]["date_block_length"])
    iterations = int(options["evaluation"]["bootstrap_iterations"])
    base_seed = int(options["evaluation"]["bootstrap_seed"])
    for split in ["validation", "test"]:
        part = original[original["split"].eq(split)]
        mask = (
            results["cell"].eq(cell)
            & results["seed"].eq("ensemble")
            & results["split"].eq(split)
            & results["score_type"].eq("raw")
        )
        prior = existing_results[
            existing_results.get("cell", pd.Series(dtype=str)).eq(cell)
            & existing_results.get("seed", pd.Series(dtype=str)).astype(str).eq("ensemble")
            & existing_results.get("split", pd.Series(dtype=str)).eq(split)
            & existing_results.get("score_type", pd.Series(dtype=str)).eq("raw")
        ] if not existing_results.empty else pd.DataFrame()
        current_estimate = float(results.loc[mask, "pair_weighted_within_asset_roc_auc"].iloc[0])
        reusable = (
            len(prior) == 1
            and np.isclose(float(prior.iloc[0]["pair_weighted_within_asset_roc_auc"]), current_estimate, atol=1e-12)
            and np.isfinite(float(prior.iloc[0].get("within_asset_ci_lower", np.nan)))
        )
        interval = (
            {
                "estimate": float(prior.iloc[0].get("within_asset_estimate", current_estimate)),
                "ci_lower": float(prior.iloc[0]["within_asset_ci_lower"]),
                "ci_upper": float(prior.iloc[0]["within_asset_ci_upper"]),
                "valid_draw_fraction": float(prior.iloc[0].get("within_asset_valid_draw_fraction", 1.0)),
            }
            if reusable
            else _fast_date_block_bootstrap_within_auc(
                part,
                probability_column="raw_probability",
                block_size=block_size,
                iterations=iterations,
                seed=base_seed + zlib.crc32(f"{cell}/{split}/selected".encode()),
            )
        )
        for key, value in interval.items():
            results.loc[mask, f"within_asset_{key}"] = value
        for control in controls:
            path = context.run_dir / "predictions" / f"{cell}_ensemble_{control}.parquet"
            if not path.exists():
                raise RuntimeError(f"Missing selected-model control predictions: {path}")
            order_mask = order["cell"].eq(cell) & order["split"].eq(split) & order["control"].eq(control)
            prior_order = existing_order[
                existing_order.get("cell", pd.Series(dtype=str)).eq(cell)
                & existing_order.get("split", pd.Series(dtype=str)).eq(split)
                & existing_order.get("control", pd.Series(dtype=str)).eq(control)
            ] if not existing_order.empty else pd.DataFrame()
            current_drop = float(order.loc[order_mask, "within_asset_auc_drop"].iloc[0])
            reusable_order = (
                len(prior_order) == 1
                and np.isclose(float(prior_order.iloc[0]["within_asset_auc_drop"]), current_drop, atol=1e-12)
                and np.isfinite(float(prior_order.iloc[0].get("within_asset_auc_drop_ci_lower", np.nan)))
            )
            if reusable_order:
                difference = {
                    "ci_lower": float(prior_order.iloc[0]["within_asset_auc_drop_ci_lower"]),
                    "ci_upper": float(prior_order.iloc[0]["within_asset_auc_drop_ci_upper"]),
                    "valid_draw_fraction": float(prior_order.iloc[0].get("bootstrap_valid_draw_fraction", 1.0)),
                }
            else:
                paired = _paired_classification_frame(original, pd.read_parquet(path), split=split)
                difference = _date_block_within_difference(
                    paired,
                    block_size=block_size,
                    iterations=iterations,
                    seed=base_seed + zlib.crc32(f"{cell}/{control}/{split}".encode()),
                )
            order.loc[order_mask, "within_asset_auc_drop_ci_lower"] = difference["ci_lower"]
            order.loc[order_mask, "within_asset_auc_drop_ci_upper"] = difference["ci_upper"]
            order.loc[order_mask, "bootstrap_valid_draw_fraction"] = difference["valid_draw_fraction"]


def _continuous_verdict(
    context: FinalExperimentContext,
    results: pd.DataFrame,
    robustness: pd.DataFrame,
) -> tuple[str, str]:
    options = context.continuous_options
    validation = results[(results["seed"].eq("ensemble")) & (results["split"].eq("validation"))]
    neural = validation[validation["cell"].str.contains("__mlp__|__transformer_encoder__", regex=True)]
    baselines = validation[~validation.index.isin(neural.index)]
    winner = neural.sort_values("equal_asset_mae").iloc[0]
    baseline = baselines.sort_values("equal_asset_mae").iloc[0]
    cell = str(winner["cell"])
    test = results[
        results["cell"].eq(cell)
        & results["seed"].eq("ensemble")
        & results["split"].eq("test")
    ].iloc[0]
    baseline_test = results[
        results["cell"].eq(str(baseline["cell"]))
        & results["seed"].eq("ensemble")
        & results["split"].eq("test")
    ].iloc[0]
    order = robustness[
        robustness["analysis"].eq("temporal_order")
        & robustness["cell"].eq(cell)
        & robustness["split"].eq("test")
    ]
    seeds = results[
        results["cell"].eq(cell)
        & results["seed"].ne("ensemble")
        & results["split"].eq("test")
    ]
    test_frame = pd.read_parquet(context.run_dir / "predictions" / f"{cell}_ensemble.parquet")
    test_frame = test_frame[test_frame["split"].eq("test")]
    uncertainty = _date_block_regression_intervals(
        test_frame,
        iterations=int(options["evaluation"]["bootstrap_iterations"]),
        block_size=int(options["evaluation"]["date_block_length"]),
        seed=int(options["evaluation"]["bootstrap_seed"]) + zlib.crc32(f"{cell}/test_metrics".encode()),
    )
    test_mask = (
        results["cell"].eq(cell)
        & results["seed"].eq("ensemble")
        & results["split"].eq("test")
    )
    for name, value in uncertainty.items():
        results.loc[test_mask, name] = value
    test = results.loc[test.name]
    paired_baseline = robustness[
        robustness["analysis"].eq("best_baseline_comparison")
        & robustness["cell"].eq(cell)
        & robustness["split"].eq("test")
    ]
    reduction = (float(baseline_test["equal_asset_mae"]) - float(test["equal_asset_mae"])) / float(baseline_test["equal_asset_mae"])
    gate = options["promotion_gate"]
    conditions = {
        "mae_reduction": reduction >= float(gate["minimum_mae_reduction_over_best_simple_fraction"]) and len(paired_baseline) == 1 and float(paired_baseline.iloc[0]["paired_difference_ci_lower"]) > 0.0,
        "spearman": float(test["spearman"]) >= float(gate["minimum_spearman"]) and float(test["spearman_ci_lower"]) > float(gate["minimum_spearman_lower_95ci"]),
        "order_sensitivity": len(order) == len(options["perturbations"]["methods"]) and (
            order["equal_asset_mae_deterioration"] / order["base_equal_asset_mae"]
        ).ge(float(gate["minimum_each_order_mae_deterioration_fraction"])).all() and order["paired_difference_ci_lower"].gt(0.0).all(),
        "seed_improvement": int(seeds["equal_asset_mae"].lt(float(baseline_test["equal_asset_mae"])).sum()) >= int(gate["minimum_seed_improvement_count"]),
    }
    passed = all(conditions.values())
    condition_text = "\n".join(f"- {name}: **{'pass' if value else 'fail'}**" for name, value in conditions.items())
    verdict = "\n".join(
        [
            "# IFDDRP Continuous Downside-Risk Verdict",
            "",
            "Evidence class: historical held-out but adaptive. The outcome is maximum origin-to-path loss, not peak-to-trough drawdown.",
            "",
            f"Validation selected `{cell}`. On the historical test its equal-asset MAE was `{float(test['equal_asset_mae']):.6f}` (95% date-block interval `[{float(test['equal_asset_mae_ci_lower']):.6f}, {float(test['equal_asset_mae_ci_upper']):.6f}]`) versus `{float(baseline_test['equal_asset_mae']):.6f}` for `{baseline['cell']}`, a relative reduction of `{reduction:.2%}`. Test Spearman was `{float(test['spearman']):.4f}` with interval `[{float(test['spearman_ci_lower']):.4f}, {float(test['spearman_ci_upper']):.4f}]`.",
            "",
            condition_text,
            "",
            f"Overall promotion gate: **{'passed' if passed else 'failed'}**.",
            "",
            "Practical monitoring remains descriptive and cannot be interpreted as a trading strategy.",
        ]
    )
    practical = pd.DataFrame(
        [
            {
                "status": "unlocked_descriptive_only" if passed else "locked",
                "selected_cell": cell,
                "promotion_gate_passed": passed,
                "permitted_use": "risk-monitoring analysis only; no trading strategy" if passed else "none",
                "reason": "all preregistered gates passed" if passed else "one or more temporal-skill gates failed",
            }
        ]
    ).to_csv(index=False)
    return verdict, practical


def _save_checkpoint(
    context: FinalExperimentContext,
    model: nn.Module,
    cell: str,
    seed: int,
    row: dict[str, object],
    *,
    priors: dict[str, Any] | None = None,
    target_scaling: dict[str, float] | None = None,
    asset_mapping: dict[str, int] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "cell": cell,
        "seed": seed,
        "feature_columns": context.bundle.feature_columns,
        "asset_to_id": asset_mapping or context.bundle.asset_to_id,
        "split": _split_manifest(context.phase6.split),
        "training_summary": row,
        "endpoint_sha256": _endpoint_sha256(context.bundle),
    }
    if priors is not None:
        payload["priors"] = {
            "global": float(priors["global"]),
            "asset": {str(key): float(value) for key, value in priors["asset"].items()},
            "family": {str(key): float(value) for key, value in priors["family"].items()},
        }
    if target_scaling is not None:
        payload["target_scaling"] = target_scaling
    torch.save(payload, context.run_dir / "checkpoints" / f"{cell}_seed{seed}.pt")


def _endpoint_sha256(bundle: PooledWindowDataBundle) -> str:
    parts: list[pd.DataFrame] = []
    for split, dataset in [("train", bundle.train), ("validation", bundle.validation), ("test", bundle.test)]:
        frame = dataset.endpoint_metadata().copy()
        frame["split"] = split
        frame["y_true"] = dataset_targets(dataset)
        parts.append(frame)
    payload = pd.concat(parts, ignore_index=True).to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_shared_manifest(context: FinalExperimentContext) -> None:
    payload = {
        "evidence_class": "historical_held_out_but_adaptive",
        "target": context.bundle.target_column,
        "lookback": int(context.static_options["lookback"]),
        "split": _split_manifest(context.phase6.split),
        "features": context.bundle.feature_columns,
        "asset_to_id": context.bundle.asset_to_id,
        "endpoint_sha256": _endpoint_sha256(context.bundle),
        "train_windows": len(context.bundle.train),
        "validation_windows": len(context.bundle.validation),
        "test_windows": len(context.bundle.test),
        "device": str(context.phase6.device),
    }
    (context.run_dir / "manifests" / "shared_contract.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_prior_manifest(context: FinalExperimentContext, priors: dict[str, Any]) -> None:
    payload = {
        "global": float(priors["global"]),
        "asset": {str(key): float(value) for key, value in priors["asset"].items()},
        "family": {str(key): float(value) for key, value in priors["family"].items()},
        "fit": "training_endpoints_only",
        "smoothing": 1.0,
    }
    (context.run_dir / "manifests" / "classification_priors.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_continuous_manifest(
    context: FinalExperimentContext,
    bundle: PooledWindowDataBundle,
    target_panel: pd.DataFrame,
    target_mean: float,
    target_scale: float,
    intercepts: np.ndarray,
) -> None:
    source_rows: list[dict[str, object]] = []
    for ticker, asset in context.phase6.panel.groupby("Ticker", observed=True, sort=True):
        observed = asset[asset["Close"].notna()]
        adjusted = observed.get("Adj Close")
        use_adjusted = bool(adjusted is not None and adjusted.notna().all() and adjusted.gt(0.0).all())
        source_rows.append(
            {
                "asset_ticker": ticker,
                "price_source": "Adj Close" if use_adjusted else "Close",
                "observed_rows": len(observed),
                "target_rows": int(target_panel[target_panel["Ticker"].eq(ticker)][bundle.target_column].notna().sum()),
            }
        )
    pd.DataFrame(source_rows).to_csv(context.table_dir / "ifddrp_continuous_downside_price_source_audit.csv", index=False)
    payload = {
        "target": bundle.target_column,
        "definition": "maximum origin-to-path loss over t+1 through t+10",
        "target_mean_training": target_mean,
        "target_scale_training": target_scale,
        "asset_intercepts_scaled": intercepts.tolist(),
        "asset_to_id": bundle.asset_to_id,
        "features": bundle.feature_columns,
        "split": _split_manifest(bundle.split),
        "train_windows": len(bundle.train),
        "validation_windows": len(bundle.validation),
        "test_windows": len(bundle.test),
    }
    (context.run_dir / "manifests" / "continuous_contract.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
