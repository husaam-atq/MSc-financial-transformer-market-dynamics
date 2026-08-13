"""Execute the frozen PRP-1 fixed cross-model falsification study."""

from __future__ import annotations

import hashlib
import json
import pickle
import zlib
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from market_dynamics.datasets.pooled_window_dataset import PooledWindowDataBundle
from market_dynamics.evaluation.post_freeze import (
    aligned_probability_ensemble,
    binary_probability_metrics,
    fit_calibration_candidates,
)
from market_dynamics.evaluation.prior_neutral import (
    equal_group_weights,
    fit_group_priors,
)
from market_dynamics.experiments.run_large_scale_screening import _loaders
from market_dynamics.experiments.run_phase6 import (
    PerturbedDataset,
    Phase6Context,
    _bundle,
    _family_ids_by_asset,
    _prediction_frame,
    _split_manifest,
    _temporal_transform,
    _within_group_auc,
    build_context,
)
from market_dynamics.models.deep_learning import (
    AssetAgnosticModel,
    AssetConditionedModel,
    build_deep_model,
)
from market_dynamics.training.losses import build_loss
from market_dynamics.training.sampling import dataset_targets
from market_dynamics.training.train import fit_model, predict_loader
from market_dynamics.utils.torch_utils import set_torch_seed

NEURAL_MODELS = ("mlp", "lstm", "tcn", "transformer_encoder")
IDENTITY_VARIANTS = ("asset_conditioned", "no_explicit_asset_id")
HISTORICAL_VARIANTS = {
    "asset_conditioned": "corrected_asset_conditioned",
    "no_explicit_asset_id": "no_explicit_asset_id",
}


@dataclass(frozen=True)
class FixedCrossModelContext:
    """Immutable inputs and paths for the registered comparison."""

    phase6: Phase6Context
    options: dict[str, Any]
    bundle: PooledWindowDataBundle
    run_dir: Path
    table_dir: Path
    config_path: Path
    config_sha256: str
    endpoint_sha256: str
    represented_asset_ids: tuple[int, ...]


def build_fixed_cross_model_context(
    base_config: dict[str, Any],
    phase6_config: dict[str, Any],
    cross_model_config: dict[str, Any],
    run_dir: str | Path,
) -> FixedCrossModelContext:
    """Reconstruct the Phase 6 data contract and enforce the frozen protocol."""
    options = dict(cross_model_config.get("prp1_fixed_cross_model", cross_model_config))
    active = Path(run_dir).resolve()
    for child in ("checkpoints", "predictions", "logs", "manifests", "cache"):
        (active / child).mkdir(parents=True, exist_ok=True)
    phase6 = build_context(base_config, phase6_config, active)
    bundle = _bundle(phase6, "corrected_asset_conditioned")
    _validate_registered_contract(phase6, options, bundle)
    endpoint_hash = endpoint_fingerprint(bundle)
    represented = tuple(sorted({int(part.asset_id) for part in bundle.train.assets}))
    config_path = Path(cross_model_config.get("_meta", {}).get("config_path", "")).resolve()
    config_bytes = config_path.read_bytes() if config_path.is_file() else json.dumps(options, sort_keys=True).encode()
    return FixedCrossModelContext(
        phase6=phase6,
        options=options,
        bundle=bundle,
        run_dir=active,
        table_dir=phase6.table_dir,
        config_path=config_path,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        endpoint_sha256=endpoint_hash,
        represented_asset_ids=represented,
    )


def endpoint_fingerprint(bundle: PooledWindowDataBundle) -> str:
    """Hash exact endpoint membership, labels and sparse asset IDs for all splits."""
    digest = hashlib.sha256()
    for split_name, dataset in (("train", bundle.train), ("validation", bundle.validation), ("test", bundle.test)):
        frame = dataset.endpoint_metadata().copy()
        frame["y_true"] = dataset_targets(dataset)
        frame.insert(0, "split", split_name)
        frame = frame.sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
        digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def run_preflight(context: FixedCrossModelContext) -> pd.DataFrame:
    """Validate real implementations, capacity, historical reuse and endpoints."""
    rows: list[dict[str, object]] = []
    counts: list[int] = []
    for model_name in NEURAL_MODELS:
        for identity_variant in IDENTITY_VARIANTS:
            model = build_fixed_neural_model(context, model_name, identity_variant)
            parameter_count = count_parameters(model)
            counts.append(parameter_count)
            features, _, _, asset_ids = context.bundle.validation[0]
            with torch.no_grad():
                output = model(features.unsqueeze(0), asset_ids.unsqueeze(0))
            if output.shape != (1,):
                raise RuntimeError(f"{model_name}/{identity_variant} returned {tuple(output.shape)}")
            rows.append(
                {
                    "model": model_name,
                    "identity_variant": identity_variant,
                    "status": "passed",
                    "parameter_count": parameter_count,
                    "real_implementation": type(model.base_model).__name__,
                    "endpoint_sha256": context.endpoint_sha256,
                }
            )
    ratio = max(counts) / min(counts)
    maximum = float(context.options["gates"]["maximum_neural_parameter_ratio"])
    if ratio > maximum:
        raise RuntimeError(f"Neural parameter ratio {ratio:.6f} exceeds frozen gate {maximum:.6f}")
    for identity_variant in IDENTITY_VARIANTS:
        for seed in _seeds(context):
            _load_transformer_checkpoint(context, identity_variant, seed, validate_predictions=True)
    rows.append(
        {
            "model": "all_neural",
            "identity_variant": "all",
            "status": "passed",
            "parameter_count": np.nan,
            "real_implementation": "parameter_ratio_gate",
            "endpoint_sha256": context.endpoint_sha256,
            "parameter_ratio": ratio,
        }
    )
    output = pd.DataFrame(rows)
    output.to_csv(context.run_dir / "manifests" / "preflight.csv", index=False)
    _write_contract_manifest(context, output)
    return output


def run_smoke(context: FixedCrossModelContext) -> pd.DataFrame:
    """Exercise training, checkpoint and perturbation paths on bounded subsets."""
    rows: list[dict[str, object]] = []
    training = deepcopy(context.options["training"])
    training.update({"epochs": 1, "early_stopping_patience": 1, "mixed_precision": False, "batch_size": 32})
    train = Subset(context.bundle.train, range(min(128, len(context.bundle.train))))
    validation = Subset(context.bundle.validation, range(min(64, len(context.bundle.validation))))
    test = Subset(context.bundle.test, range(min(64, len(context.bundle.test))))
    for model_name in NEURAL_MODELS[:-1]:
        model = build_fixed_neural_model(context, model_name, "asset_conditioned").to(context.phase6.device)
        loaders = _loaders(train, validation, test, training, seed=7, sampling_config=None)
        criterion = build_loss("classification", dataset_targets(train), training, context.phase6.device)
        fitted = fit_model(model, loaders[0], loaders[1], criterion, training, context.phase6.device)
        y, probability, indices = predict_loader(model, loaders[2], "classification", context.phase6.device)
        if not (len(y) == len(probability) == len(indices) == len(test)):
            raise RuntimeError(f"Smoke prediction length mismatch for {model_name}")
        rows.append({"model": model_name, "status": "passed", "best_epoch": fitted.best_epoch + 1, "rows": len(y)})
    model = _load_transformer_checkpoint(context, "asset_conditioned", 7, validate_predictions=True)
    perturb = PerturbedDataset(test, _temporal_transform("reverse", int(context.options["lookback"]), int(context.options["perturbations"]["seed"])))
    loader = DataLoader(perturb, batch_size=32, shuffle=False)
    y, probability, _ = predict_loader(model.to(context.phase6.device), loader, "classification", context.phase6.device)
    rows.append({"model": "transformer_encoder", "status": "passed", "best_epoch": "reused", "rows": len(y)})
    output = pd.DataFrame(rows)
    output.to_csv(context.run_dir / "manifests" / "smoke.csv", index=False)
    return output


def run_training(context: FixedCrossModelContext, include_logistic: bool = True) -> pd.DataFrame:
    """Train or resume every registered non-Transformer arm and reference Transformer arms."""
    rows: list[dict[str, object]] = []
    for model_name in NEURAL_MODELS:
        for identity_variant in IDENTITY_VARIANTS:
            for seed in _seeds(context):
                if model_name == "transformer_encoder":
                    checkpoint, source = _historical_checkpoint_paths(context, identity_variant, seed)
                    _load_transformer_checkpoint(context, identity_variant, seed, validate_predictions=False)
                    row = _execution_row(context, model_name, identity_variant, seed, "reused", checkpoint, source)
                else:
                    row = _train_neural_arm(context, model_name, identity_variant, seed)
                rows.append(row)
                _write_execution_manifest(context, rows)
    if include_logistic:
        for identity_variant in IDENTITY_VARIANTS:
            for seed in _seeds(context):
                rows.append(_train_logistic_arm(context, identity_variant, seed))
                _write_execution_manifest(context, rows)
    output = pd.DataFrame(rows)
    _write_execution_manifest(context, rows)
    return output


def run_evaluation(context: FixedCrossModelContext) -> pd.DataFrame:
    """Evaluate every completed seed, ensembles and train-only static priors."""
    assert_execution_complete(context)
    rows: list[dict[str, object]] = []
    for model_name in (*NEURAL_MODELS, "flattened_logistic"):
        seed_frames: list[pd.DataFrame] = []
        for identity_variant in IDENTITY_VARIANTS:
            seeds = _seeds(context)
            seed_frames.clear()
            for seed in seeds:
                frame = load_prediction_frame(context, model_name, identity_variant, seed)
                if frame is None:
                    rows.append(_failure_metric_row(model_name, identity_variant, seed, "missing_prediction"))
                    continue
                seed_frames.append(frame)
                rows.extend(_evaluate_frame(context, frame, model_name, identity_variant, str(seed)))
            if len(seed_frames) == len(seeds) and len(seed_frames) > 1:
                ensemble = aligned_probability_ensemble(seed_frames, "raw_probability")
                reverse = {value: key for key, value in context.bundle.asset_to_id.items()}
                ensemble["asset_ticker"] = ensemble["asset_id"].map(reverse)
                rows.extend(_evaluate_frame(context, ensemble.rename(columns={"ensemble_probability": "raw_probability"}), model_name, identity_variant, "ensemble"))
    rows.extend(_static_prior_rows(context))
    output = pd.DataFrame(rows)
    output.to_csv(context.run_dir / "fixed_cross_model_results.csv", index=False)
    output.to_csv(context.table_dir / "prp1_fixed_cross_model_results.csv", index=False)
    return output


def run_temporal_order(context: FixedCrossModelContext) -> pd.DataFrame:
    """Apply the three frozen order-destruction transforms to neural checkpoints."""
    assert_execution_complete(context)
    rows: list[dict[str, object]] = []
    methods = [str(value) for value in context.options["perturbations"]["methods"]]
    for model_name in (*NEURAL_MODELS, "flattened_logistic"):
        for identity_variant in IDENTITY_VARIANTS:
            for seed in _seeds(context):
                original = load_prediction_frame(context, model_name, identity_variant, seed)
                if original is None:
                    rows.append(_failure_metric_row(model_name, identity_variant, seed, "missing_prediction"))
                    continue
                test = original[original["split"].eq("test")]
                test = test.sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
                original_probability = test["raw_probability"].to_numpy(dtype=float)
                model = None if model_name == "flattened_logistic" else load_fixed_neural_checkpoint(context, model_name, identity_variant, seed).to(context.phase6.device)
                for method in methods:
                    perturbed = PerturbedDataset(
                        context.bundle.test,
                        _temporal_transform(method, int(context.options["lookback"]), int(context.options["perturbations"]["seed"])),
                    )
                    if model_name == "flattened_logistic":
                        matrix, y, indices = materialize_flattened_dataset(context, perturbed, identity_variant, f"test_{method}")
                        raw = _load_logistic_model(context, identity_variant, seed).predict_proba(matrix)[:, 1]
                    else:
                        loader = DataLoader(perturbed, batch_size=int(context.options["training"]["batch_size"]), shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
                        y, raw, indices = predict_loader(model, loader, "classification", context.phase6.device)
                    frame = _prediction_frame(context.bundle.test, indices, y, raw, context.bundle.asset_to_id, "test")
                    frame = frame.sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
                    ordered_test = test.sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
                    if not frame[["Date", "source_index", "asset_id", "y_true"]].equals(ordered_test[["Date", "source_index", "asset_id", "y_true"]]):
                        raise RuntimeError(f"Perturbation changed endpoint identity for {model_name}/{identity_variant}/{seed}/{method}")
                    perturbation_path = context.run_dir / "predictions" / f"{model_name}_{identity_variant}_seed{seed}_{method}.parquet"
                    frame.to_parquet(perturbation_path, index=False)
                    aligned_y = frame["y_true"].to_numpy(dtype=int)
                    probability = frame["raw_probability"].to_numpy(dtype=float)
                    metrics = binary_probability_metrics(aligned_y, probability, 0.5, int(context.options["calibration"]["bins"]))
                    original_metrics = binary_probability_metrics(aligned_y, original_probability, 0.5, int(context.options["calibration"]["bins"]))
                    within = _within_group_auc(aligned_y, probability, frame["asset_id"].to_numpy(dtype=int))
                    original_within = _within_group_auc(aligned_y, original_probability, frame["asset_id"].to_numpy(dtype=int))
                    rows.append(
                        {
                            "model": model_name,
                            "identity_variant": identity_variant,
                            "seed": seed,
                            "method": method,
                            "n_obs": len(aligned_y),
                            "original_roc_auc": original_metrics["roc_auc"],
                            "perturbed_roc_auc": metrics["roc_auc"],
                            "roc_auc_drop": original_metrics["roc_auc"] - metrics["roc_auc"],
                            "original_within_asset_roc_auc": original_within["pair_weighted_within_asset_roc_auc"],
                            "perturbed_within_asset_roc_auc": within["pair_weighted_within_asset_roc_auc"],
                            "within_asset_auc_drop": original_within["pair_weighted_within_asset_roc_auc"] - within["pair_weighted_within_asset_roc_auc"],
                            "prediction_spearman": float(pd.Series(original_probability).corr(pd.Series(probability), method="spearman")),
                            "mean_absolute_probability_displacement": float(np.mean(np.abs(original_probability - probability))),
                            "status": "completed",
                        }
                    )
                    pd.DataFrame(rows).to_csv(context.run_dir / "fixed_cross_model_temporal_order.csv", index=False)
    rows.extend(_temporal_ensemble_rows(context))
    output = pd.DataFrame(rows)
    output.to_csv(context.run_dir / "fixed_cross_model_temporal_order.csv", index=False)
    output.to_csv(context.table_dir / "prp1_fixed_cross_model_temporal_order.csv", index=False)
    return output


def run_identity_swap(context: FixedCrossModelContext) -> pd.DataFrame:
    """Score conditioned neural models after cycling only represented asset IDs."""
    assert_execution_complete(context)
    rows: list[dict[str, object]] = []
    for model_name in (*NEURAL_MODELS, "flattened_logistic"):
        for seed in _seeds(context):
            original = load_prediction_frame(context, model_name, "asset_conditioned", seed)
            if original is None:
                continue
            if model_name == "flattened_logistic":
                matrix, y, indices = materialize_flattened_dataset(context, context.bundle.test, "asset_conditioned", "test")
                swapped_matrix = _swap_logistic_identity_columns(context, matrix, context.bundle.test)
                probability = _load_logistic_model(context, "asset_conditioned", seed).predict_proba(swapped_matrix)[:, 1]
            else:
                model = load_fixed_neural_checkpoint(context, model_name, "asset_conditioned", seed).to(context.phase6.device)
                loader = DataLoader(context.bundle.test, batch_size=int(context.options["training"]["batch_size"]), shuffle=False, num_workers=0)
                y, probability, indices = _predict_with_represented_id_swap(model, loader, context.phase6.device, context.represented_asset_ids)
            frame = _prediction_frame(context.bundle.test, indices, y, probability, context.bundle.asset_to_id, "test")
            base = original[original["split"].eq("test")].sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
            frame = frame.sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
            if not frame[["Date", "source_index", "asset_id", "y_true"]].equals(base[["Date", "source_index", "asset_id", "y_true"]]):
                raise RuntimeError("ID swap changed endpoint membership")
            aligned_y = frame["y_true"].to_numpy(dtype=int)
            swapped_probability = frame["raw_probability"].to_numpy(dtype=float)
            base_probability = base["raw_probability"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "n_obs": len(frame),
                    "original_roc_auc": _safe_auc(aligned_y, base_probability),
                    "swapped_roc_auc": _safe_auc(aligned_y, swapped_probability),
                    "roc_auc_change": _safe_auc(aligned_y, swapped_probability) - _safe_auc(aligned_y, base_probability),
                    "prediction_spearman": float(pd.Series(base_probability).corr(pd.Series(swapped_probability), method="spearman")),
                    "mean_absolute_probability_displacement": float(np.mean(np.abs(base_probability - swapped_probability))),
                    "represented_ids": len(context.represented_asset_ids),
                    "unused_ids_introduced": False,
                    "status": "completed",
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(context.run_dir / "fixed_cross_model_identity_swap.csv", index=False)
    output.to_csv(context.table_dir / "prp1_fixed_cross_model_identity_swap.csv", index=False)
    return output


def run_representation_probes(context: FixedCrossModelContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit leakage-safe train representations and evaluate identity/state probes on test."""
    assert_execution_complete(context)
    identity_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = [
        {
            "status": "not_executed",
            "reason": "dynamic-state labels and controls were not frozen in the fixed cross-model protocol",
            "evidence_class": "scientific_stop_before_outcome_inspection",
        }
    ]
    per_asset = int(context.options["probes"]["samples_per_asset_per_split"])
    train_indices = balanced_dataset_indices(context.bundle.train, per_asset)
    test_indices = balanced_dataset_indices(context.bundle.test, per_asset)
    family_by_asset = _family_ids_by_asset(context.bundle.asset_to_id, context.phase6.family_map)
    for model_name in NEURAL_MODELS:
        for identity_variant in IDENTITY_VARIANTS:
            for seed in _seeds(context):
                model = load_fixed_neural_checkpoint(context, model_name, identity_variant, seed).to(context.phase6.device)
                train_repr, train_y, train_asset = encode_representations(model, Subset(context.bundle.train, train_indices), context.phase6.device)
                test_repr, test_y, test_asset = encode_representations(model, Subset(context.bundle.test, test_indices), context.phase6.device)
                for label_name, train_label, test_label in (
                    ("asset_id", train_asset, test_asset),
                    ("family_id", family_by_asset[train_asset], family_by_asset[test_asset]),
                ):
                    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
                    probe.fit(train_repr, train_label)
                    predicted = probe.predict(test_repr)
                    identity_rows.append(
                        {
                            "model": model_name,
                            "identity_variant": identity_variant,
                            "seed": seed,
                            "probe_label": label_name,
                            "train_rows": len(train_repr),
                            "test_rows": len(test_repr),
                            "accuracy": float(accuracy_score(test_label, predicted)),
                            "balanced_accuracy": float(balanced_accuracy_score(test_label, predicted)),
                            "chance_accuracy": float(1.0 / len(np.unique(train_label))),
                            "status": "completed",
                        }
                    )
    identity = pd.DataFrame(identity_rows)
    states = pd.DataFrame(state_rows)
    identity.to_csv(context.run_dir / "fixed_cross_model_identity_probes.csv", index=False)
    states.to_csv(context.run_dir / "fixed_cross_model_dynamic_state_probes.csv", index=False)
    identity.to_csv(context.table_dir / "prp1_fixed_cross_model_identity_probes.csv", index=False)
    states.to_csv(context.table_dir / "prp1_fixed_cross_model_dynamic_state_probes.csv", index=False)
    return identity, states


def build_fixed_neural_model(context: FixedCrossModelContext, model_name: str, identity_variant: str) -> torch.nn.Module:
    """Build one capacity-controlled registered neural model."""
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"Unregistered neural model: {model_name}")
    if identity_variant not in IDENTITY_VARIANTS:
        raise ValueError(f"Unregistered identity variant: {identity_variant}")
    config = deepcopy(context.options["model"])
    config["hidden_size"] = int(config["hidden_sizes"][model_name])
    config["num_layers"] = int(config["layers"][model_name])
    embedding_dim = int(config["asset_embedding_dim"])
    base = build_deep_model(
        model_name,
        len(context.bundle.feature_columns) + embedding_dim,
        int(context.options["lookback"]),
        {"model": config},
    )
    if identity_variant == "asset_conditioned":
        return AssetConditionedModel(base, len(context.bundle.asset_to_id), embedding_dim)
    return AssetAgnosticModel(base, zero_channels=embedding_dim)


def load_fixed_neural_checkpoint(context: FixedCrossModelContext, model_name: str, identity_variant: str, seed: int) -> torch.nn.Module:
    """Load a frozen or newly trained neural checkpoint without substitution."""
    if model_name == "transformer_encoder":
        return _load_transformer_checkpoint(context, identity_variant, seed, validate_predictions=False)
    path = _checkpoint_path(context, model_name, identity_variant, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_local_payload(context, payload, model_name, identity_variant, seed)
    model = build_fixed_neural_model(context, model_name, identity_variant)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def load_prediction_frame(context: FixedCrossModelContext, model_name: str, identity_variant: str, seed: int) -> pd.DataFrame | None:
    """Load one raw prediction artifact from the exact registered source."""
    if model_name == "transformer_encoder":
        _, prediction_path = _historical_checkpoint_paths(context, identity_variant, seed)
    else:
        prediction_path = _prediction_path(context, model_name, identity_variant, seed)
    if not prediction_path.exists():
        return None
    frame = pd.read_parquet(prediction_path)
    validate_prediction_frame(context, frame)
    return frame


def validate_prediction_frame(context: FixedCrossModelContext, frame: pd.DataFrame) -> None:
    """Validate exact split endpoints, labels and finite raw probabilities."""
    required = {"Date", "source_index", "asset_id", "asset_ticker", "y_true", "raw_probability", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Prediction artifact is missing columns: {sorted(missing)}")
    if set(frame["split"].astype(str)) != {"validation", "test"}:
        raise RuntimeError("Prediction artifact must contain exactly validation and test splits")
    if frame.duplicated(["split", "Date", "source_index", "asset_id"]).any():
        raise RuntimeError("Prediction artifact contains duplicate endpoint keys")
    reverse = {value: key for key, value in context.bundle.asset_to_id.items()}
    for split_name, dataset in (("validation", context.bundle.validation), ("test", context.bundle.test)):
        expected = dataset.endpoint_metadata().copy()
        expected["asset_ticker"] = expected["asset_id"].map(reverse)
        expected["y_true"] = dataset_targets(dataset)
        expected = expected.sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
        actual = frame[frame["split"].eq(split_name)].sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
        key_columns = ["Date", "source_index", "asset_id", "asset_ticker"]
        keys_match = actual[key_columns].equals(expected[key_columns])
        labels_match = np.array_equal(
            actual["y_true"].to_numpy(dtype=np.float32),
            expected["y_true"].to_numpy(dtype=np.float32),
        )
        if not keys_match or not labels_match:
            raise RuntimeError(f"Prediction artifact endpoint mismatch for {split_name}")
    probability = frame["raw_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise RuntimeError("Prediction artifact contains invalid probabilities")


def _load_logistic_model(context: FixedCrossModelContext, identity_variant: str, seed: int) -> LogisticRegression:
    path = context.run_dir / "checkpoints" / f"flattened_logistic_{identity_variant}_seed{seed}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("endpoint_sha256") != context.endpoint_sha256 or payload.get("config_sha256") != context.config_sha256:
        raise RuntimeError(f"Stale logistic checkpoint: {path}")
    model = payload.get("model")
    if not isinstance(model, LogisticRegression):
        raise TypeError(f"Unexpected logistic checkpoint model: {type(model)!r}")
    return model


def _swap_logistic_identity_columns(
    context: FixedCrossModelContext,
    matrix: np.ndarray,
    dataset: Any,
) -> np.ndarray:
    """Return a test matrix whose one-hot IDs cycle over represented IDs only."""
    output = np.array(matrix, dtype=np.float32, copy=True)
    base_width = int(context.options["lookback"]) * len(context.bundle.feature_columns)
    output[:, base_width:] = 0.0
    represented = context.represented_asset_ids
    swap = {asset_id: represented[(position + 1) % len(represented)] for position, asset_id in enumerate(represented)}
    asset_ids = dataset.endpoint_metadata()["asset_id"].to_numpy(dtype=int)
    swapped = np.asarray([swap[int(value)] for value in asset_ids], dtype=int)
    output[np.arange(len(output)), base_width + swapped] = 1.0
    return output


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def assert_execution_complete(context: FixedCrossModelContext) -> None:
    """Refuse analysis until all 30 registered model/variant/seed cells exist."""
    missing: list[str] = []
    for model_name in (*NEURAL_MODELS, "flattened_logistic"):
        for identity_variant in IDENTITY_VARIANTS:
            for seed in _seeds(context):
                if load_prediction_frame(context, model_name, identity_variant, seed) is None:
                    missing.append(f"{model_name}/{identity_variant}/seed{seed}")
                    continue
                if model_name == "flattened_logistic":
                    _load_logistic_model(context, identity_variant, seed)
                else:
                    load_fixed_neural_checkpoint(context, model_name, identity_variant, seed)
    if missing:
        raise RuntimeError(f"Registered execution is incomplete ({len(missing)} missing): {missing[:8]}")


def balanced_dataset_indices(dataset: Any, per_asset: int) -> list[int]:
    """Select deterministic equally bounded rows per represented asset."""
    metadata = dataset.endpoint_metadata().reset_index(names="dataset_index")
    selected: list[int] = []
    for _, part in metadata.groupby("asset_id", observed=True):
        positions = part["dataset_index"].to_numpy(dtype=int)
        if len(positions) > per_asset:
            positions = positions[np.linspace(0, len(positions) - 1, per_asset, dtype=int)]
        selected.extend(positions.tolist())
    return sorted(selected)


@torch.no_grad()
def encode_representations(model: torch.nn.Module, dataset: Dataset[Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one fixed representation per window without probe-side fitting."""
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    representations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    assets: list[np.ndarray] = []
    model.eval()
    for features, target, _, asset_id in loader:
        features_device = features.to(device)
        asset_device = asset_id.to(device)
        conditioned = model.conditioned_input(features_device, asset_device)
        _, summary, _ = model.base_model.encode(conditioned)
        representations.append(summary.detach().cpu().numpy())
        labels.append(target.numpy().astype(int))
        assets.append(asset_id.numpy().astype(int))
    return np.concatenate(representations), np.concatenate(labels), np.concatenate(assets)


def _validate_registered_contract(phase6: Phase6Context, options: dict[str, Any], bundle: PooledWindowDataBundle) -> None:
    expected = {
        "target": str(phase6.options["target"]),
        "lookback": int(phase6.options["lookback"]),
        "fold": int(phase6.options["fold"]),
        "purge": int(phase6.options["corrected_purge"]),
    }
    actual = {key: options[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"Cross-model contract differs from Phase 6: {actual!r} != {expected!r}")
    if tuple(options["models"]) != ("flattened_logistic", "mlp", "lstm", "tcn", "transformer_encoder"):
        raise RuntimeError("Registered model list changed")
    if tuple(options["identity_variants"]) != IDENTITY_VARIANTS:
        raise RuntimeError("Registered identity variants changed")
    if int(bundle.split.purge) != int(options["purge"]):
        raise RuntimeError("Bundle purge differs from frozen protocol")
    if len(bundle.feature_columns) != 34:
        raise RuntimeError(f"Expected 34 frozen features, found {len(bundle.feature_columns)}")


def _train_neural_arm(context: FixedCrossModelContext, model_name: str, identity_variant: str, seed: int) -> dict[str, object]:
    checkpoint_path = _checkpoint_path(context, model_name, identity_variant, seed)
    prediction_path = _prediction_path(context, model_name, identity_variant, seed)
    log_path = _log_path(context, model_name, identity_variant, seed)
    if checkpoint_path.exists() and prediction_path.exists() and log_path.exists():
        row = json.loads(log_path.read_text(encoding="utf-8"))
        _validate_local_payload(context, torch.load(checkpoint_path, map_location="cpu", weights_only=False), model_name, identity_variant, seed)
        validate_prediction_frame(context, pd.read_parquet(prediction_path))
        return row
    training = deepcopy(context.options["training"])
    set_torch_seed(seed, deterministic=bool(training["deterministic"]))
    model = build_fixed_neural_model(context, model_name, identity_variant).to(context.phase6.device)
    train_loader, validation_loader, test_loader = _loaders(context.bundle.train, context.bundle.validation, context.bundle.test, training, seed, sampling_config=None)
    criterion = build_loss("classification", dataset_targets(context.bundle.train), training, context.phase6.device)
    started = perf_counter()
    fitted = fit_model(model, train_loader, validation_loader, criterion, training, context.phase6.device)
    runtime = perf_counter() - started
    y_val, p_val, i_val = predict_loader(model, validation_loader, "classification", context.phase6.device)
    y_test, p_test, i_test = predict_loader(model, test_loader, "classification", context.phase6.device)
    frame = pd.concat(
        [
            _prediction_frame(context.bundle.validation, i_val, y_val, p_val, context.bundle.asset_to_id, "validation"),
            _prediction_frame(context.bundle.test, i_test, y_test, p_test, context.bundle.asset_to_id, "test"),
        ],
        ignore_index=True,
    )
    payload = {
        "state_dict": model.state_dict(),
        "model": model_name,
        "identity_variant": identity_variant,
        "seed": seed,
        "features": context.bundle.feature_columns,
        "split": _split_manifest(context.phase6.split),
        "training": training,
        "model_options": _model_options(context, model_name),
        "asset_to_id": context.bundle.asset_to_id,
        "endpoint_sha256": context.endpoint_sha256,
        "config_sha256": context.config_sha256,
        "parameter_count": count_parameters(model),
        "best_epoch": fitted.best_epoch,
        "best_validation_loss": fitted.best_validation_loss,
    }
    torch.save(payload, checkpoint_path)
    frame.to_parquet(prediction_path, index=False)
    row = _execution_row(context, model_name, identity_variant, seed, "completed", checkpoint_path, prediction_path)
    row.update({"runtime_seconds": runtime, "best_epoch": fitted.best_epoch + 1, "best_validation_loss": fitted.best_validation_loss})
    log_path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row


def _train_logistic_arm(context: FixedCrossModelContext, identity_variant: str, seed: int) -> dict[str, object]:
    model_path = context.run_dir / "checkpoints" / f"flattened_logistic_{identity_variant}_seed{seed}.pkl"
    prediction_path = _prediction_path(context, "flattened_logistic", identity_variant, seed)
    log_path = _log_path(context, "flattened_logistic", identity_variant, seed)
    if model_path.exists() and prediction_path.exists() and log_path.exists():
        _load_logistic_model(context, identity_variant, seed)
        validate_prediction_frame(context, pd.read_parquet(prediction_path))
        return json.loads(log_path.read_text(encoding="utf-8"))
    started = perf_counter()
    train_x, train_y, _ = materialize_flattened_dataset(context, context.bundle.train, identity_variant, "train")
    validation_x, validation_y, validation_indices = materialize_flattened_dataset(context, context.bundle.validation, identity_variant, "validation")
    test_x, test_y, test_indices = materialize_flattened_dataset(context, context.bundle.test, identity_variant, "test")
    logistic = context.options["logistic"]
    model = LogisticRegression(
        solver=str(logistic["solver"]), penalty=str(logistic["penalty"]), C=float(logistic["C"]),
        max_iter=int(logistic["max_iter"]), tol=float(logistic["tolerance"]), random_state=seed,
    )
    model.fit(train_x, train_y)
    validation_probability = model.predict_proba(validation_x)[:, 1]
    test_probability = model.predict_proba(test_x)[:, 1]
    frame = pd.concat(
        [
            _prediction_frame(context.bundle.validation, validation_indices, validation_y, validation_probability, context.bundle.asset_to_id, "validation"),
            _prediction_frame(context.bundle.test, test_indices, test_y, test_probability, context.bundle.asset_to_id, "test"),
        ], ignore_index=True,
    )
    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "endpoint_sha256": context.endpoint_sha256, "config_sha256": context.config_sha256}, handle)
    frame.to_parquet(prediction_path, index=False)
    row = _execution_row(context, "flattened_logistic", identity_variant, seed, "completed", model_path, prediction_path)
    row.update({"runtime_seconds": perf_counter() - started, "iterations": int(np.max(model.n_iter_))})
    log_path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row


def materialize_flattened_dataset(
    context: FixedCrossModelContext,
    dataset: Any,
    identity_variant: str,
    split_name: str,
) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    """Materialize one deterministic float32 window matrix as an ignored memmap."""
    identity_width = len(context.bundle.asset_to_id)
    base_width = int(context.options["lookback"]) * len(context.bundle.feature_columns)
    width = base_width + identity_width
    path = context.run_dir / "cache" / f"flattened_{identity_variant}_{split_name}_{context.endpoint_sha256[:12]}_{context.config_sha256[:12]}.dat"
    shape = (len(dataset), width)
    mode = "r+" if path.exists() and path.stat().st_size == np.prod(shape) * np.dtype(np.float32).itemsize else "w+"
    matrix = np.memmap(path, mode=mode, dtype=np.float32, shape=shape)
    labels = np.empty(len(dataset), dtype=np.int64)
    indices = np.empty(len(dataset), dtype=np.int64)
    write = mode == "w+"
    for position in range(len(dataset)):
        features, target, source_index, asset_id = dataset[position]
        if write:
            matrix[position, :base_width] = features.numpy().reshape(-1)
            matrix[position, base_width:] = 0.0
            if identity_variant == "asset_conditioned":
                matrix[position, base_width + int(asset_id)] = 1.0
        labels[position] = int(target.item())
        indices[position] = int(source_index.item())
    if write:
        matrix.flush()
    return matrix, labels, indices


def _evaluate_frame(
    context: FixedCrossModelContext,
    frame: pd.DataFrame,
    model_name: str,
    identity_variant: str,
    seed: str,
) -> list[dict[str, object]]:
    validation = frame[frame["split"].eq("validation")].copy()
    test = frame[frame["split"].eq("test")].copy()
    validation["family"] = validation["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
    test["family"] = test["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
    selected = fit_calibration_candidates(
        validation["y_true"].to_numpy(), validation["raw_probability"].to_numpy(),
        context.options["calibration"]["methods"], threshold_metric="f1",
        threshold_grid_size=int(context.options["calibration"]["threshold_grid_size"]),
        calibration_bins=int(context.options["calibration"]["bins"]),
    )[0]
    test["selected_probability"] = selected.calibrator.predict(test["raw_probability"].to_numpy())
    rows: list[dict[str, object]] = []
    for aggregation, weights in (
        ("pooled", None),
        ("equal_asset", equal_group_weights(test, "asset_ticker")),
        ("equal_date", equal_group_weights(test, "Date")),
        ("equal_family", equal_group_weights(test, "family")),
    ):
        metrics = _weighted_binary_metrics(test, "raw_probability", "selected_probability", selected.threshold, weights)
        rows.append(
            {
                "model": model_name,
                "identity_variant": identity_variant,
                "seed": seed,
                "aggregation": aggregation,
                "calibration_method": selected.method,
                "threshold": selected.threshold,
                "status": "completed",
                **metrics,
            }
        )
    within = _within_group_auc(
        test["y_true"].to_numpy(dtype=int), test["raw_probability"].to_numpy(dtype=float), test["asset_id"].to_numpy(dtype=int)
    )
    rows[0].update(within)
    if seed == "ensemble":
        uncertainty = date_block_bootstrap_within_auc(
            test,
            probability_column="raw_probability",
            block_size=int(context.options["evaluation"]["block_size_dates"]),
            iterations=int(context.options["evaluation"]["bootstrap_iterations"]),
            seed=int(context.options["evaluation"]["bootstrap_seed"]) + zlib.crc32(f"{model_name}/{identity_variant}".encode()),
        )
        rows[0].update({f"within_asset_{key}": value for key, value in uncertainty.items()})
    return rows


def date_block_bootstrap_within_auc(
    frame: pd.DataFrame,
    probability_column: str,
    block_size: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap pair-weighted within-asset AUC using circular global-date blocks."""
    frame = frame.reset_index(drop=True)
    ordered_dates = pd.DatetimeIndex(frame["Date"].drop_duplicates().sort_values())
    groups = {date: part.index.to_numpy(dtype=int) for date, part in frame.groupby("Date", sort=False)}
    y = frame["y_true"].to_numpy(dtype=int)
    probability = frame[probability_column].to_numpy(dtype=float)
    asset = frame["asset_id"].to_numpy(dtype=int)
    estimate = _within_group_auc(y, probability, asset)["pair_weighted_within_asset_roc_auc"]
    rng = np.random.default_rng(seed)
    draws = np.full(iterations, np.nan, dtype=float)
    blocks = int(np.ceil(len(ordered_dates) / block_size))
    for draw in range(iterations):
        starts = rng.integers(0, len(ordered_dates), size=blocks)
        sampled_dates = [ordered_dates[(int(start) + offset) % len(ordered_dates)] for start in starts for offset in range(block_size)]
        sampled_dates = sampled_dates[: len(ordered_dates)]
        indices = np.concatenate([groups[date] for date in sampled_dates])
        draws[draw] = _within_group_auc(y[indices], probability[indices], asset[indices])["pair_weighted_within_asset_roc_auc"]
    valid = draws[np.isfinite(draws)]
    if len(valid) == 0:
        return {"estimate": float(estimate), "ci_lower": np.nan, "ci_upper": np.nan, "valid_draw_fraction": 0.0}
    return {
        "estimate": float(estimate),
        "ci_lower": float(np.quantile(valid, 0.025)),
        "ci_upper": float(np.quantile(valid, 0.975)),
        "valid_draw_fraction": float(len(valid) / iterations),
    }


def _temporal_ensemble_rows(context: FixedCrossModelContext) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name in (*NEURAL_MODELS, "flattened_logistic"):
        for identity_variant in IDENTITY_VARIANTS:
            original_frames = [load_prediction_frame(context, model_name, identity_variant, seed) for seed in _seeds(context)]
            if any(frame is None for frame in original_frames):
                continue
            original = aligned_probability_ensemble([frame for frame in original_frames if frame is not None], "raw_probability")
            original = original[original["split"].eq("test")].sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
            for method in context.options["perturbations"]["methods"]:
                paths = [context.run_dir / "predictions" / f"{model_name}_{identity_variant}_seed{seed}_{method}.parquet" for seed in _seeds(context)]
                if not all(path.exists() for path in paths):
                    continue
                perturbed_frames = [pd.read_parquet(path) for path in paths]
                perturbed = aligned_probability_ensemble(perturbed_frames, "raw_probability")
                perturbed = perturbed.sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
                key_columns = ["split", "Date", "source_index", "asset_id", "y_true"]
                if not original[key_columns].equals(perturbed[key_columns]):
                    raise RuntimeError(f"Temporal ensemble endpoint mismatch for {model_name}/{identity_variant}/{method}")
                comparison = original[key_columns].copy()
                comparison["asset_ticker"] = comparison["asset_id"].map({value: key for key, value in context.bundle.asset_to_id.items()})
                comparison["original_probability"] = original["ensemble_probability"].to_numpy(dtype=float)
                comparison["perturbed_probability"] = perturbed["ensemble_probability"].to_numpy(dtype=float)
                y = comparison["y_true"].to_numpy(dtype=int)
                assets = comparison["asset_id"].to_numpy(dtype=int)
                original_within = _within_group_auc(y, comparison["original_probability"].to_numpy(), assets)
                perturbed_within = _within_group_auc(y, comparison["perturbed_probability"].to_numpy(), assets)
                uncertainty = _date_block_bootstrap_within_difference(
                    comparison,
                    int(context.options["evaluation"]["block_size_dates"]),
                    int(context.options["evaluation"]["bootstrap_iterations"]),
                    int(context.options["evaluation"]["bootstrap_seed"]) + zlib.crc32(f"{model_name}/{identity_variant}/{method}".encode()),
                )
                original_auc = _safe_auc(y, comparison["original_probability"].to_numpy())
                perturbed_auc = _safe_auc(y, comparison["perturbed_probability"].to_numpy())
                rows.append(
                    {
                        "model": model_name,
                        "identity_variant": identity_variant,
                        "seed": "ensemble",
                        "method": method,
                        "n_obs": len(comparison),
                        "original_roc_auc": original_auc,
                        "perturbed_roc_auc": perturbed_auc,
                        "roc_auc_drop": original_auc - perturbed_auc,
                        "original_within_asset_roc_auc": original_within["pair_weighted_within_asset_roc_auc"],
                        "perturbed_within_asset_roc_auc": perturbed_within["pair_weighted_within_asset_roc_auc"],
                        "within_asset_auc_drop": original_within["pair_weighted_within_asset_roc_auc"] - perturbed_within["pair_weighted_within_asset_roc_auc"],
                        "within_asset_auc_drop_ci_lower": uncertainty["ci_lower"],
                        "within_asset_auc_drop_ci_upper": uncertainty["ci_upper"],
                        "bootstrap_valid_draw_fraction": uncertainty["valid_draw_fraction"],
                        "prediction_spearman": float(comparison["original_probability"].corr(comparison["perturbed_probability"], method="spearman")),
                        "mean_absolute_probability_displacement": float(np.mean(np.abs(comparison["original_probability"] - comparison["perturbed_probability"]))),
                        "status": "completed",
                    }
                )
    return rows


def _date_block_bootstrap_within_difference(
    frame: pd.DataFrame,
    block_size: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    frame = frame.reset_index(drop=True)
    ordered_dates = pd.DatetimeIndex(frame["Date"].drop_duplicates().sort_values())
    groups = {date: part.index.to_numpy(dtype=int) for date, part in frame.groupby("Date", sort=False)}
    y = frame["y_true"].to_numpy(dtype=int)
    original = frame["original_probability"].to_numpy(dtype=float)
    perturbed = frame["perturbed_probability"].to_numpy(dtype=float)
    asset = frame["asset_id"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    draws = np.full(iterations, np.nan, dtype=float)
    blocks = int(np.ceil(len(ordered_dates) / block_size))
    for draw in range(iterations):
        starts = rng.integers(0, len(ordered_dates), size=blocks)
        sampled_dates = [ordered_dates[(int(start) + offset) % len(ordered_dates)] for start in starts for offset in range(block_size)]
        indices = np.concatenate([groups[date] for date in sampled_dates[: len(ordered_dates)]])
        base_auc = _within_group_auc(y[indices], original[indices], asset[indices])["pair_weighted_within_asset_roc_auc"]
        perturbed_auc = _within_group_auc(y[indices], perturbed[indices], asset[indices])["pair_weighted_within_asset_roc_auc"]
        draws[draw] = base_auc - perturbed_auc
    valid = draws[np.isfinite(draws)]
    if len(valid) == 0:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "valid_draw_fraction": 0.0}
    return {
        "ci_lower": float(np.quantile(valid, 0.025)),
        "ci_upper": float(np.quantile(valid, 0.975)),
        "valid_draw_fraction": float(len(valid) / iterations),
    }


def _weighted_binary_metrics(
    frame: pd.DataFrame,
    ranking_column: str,
    probability_column: str,
    threshold: float,
    weights: np.ndarray | None,
) -> dict[str, object]:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss

    y = frame["y_true"].to_numpy(dtype=int)
    ranking = frame[ranking_column].to_numpy(dtype=float)
    probability = frame[probability_column].to_numpy(dtype=float)
    predicted = (probability >= threshold).astype(int)
    return {
        "n_obs": len(y),
        "positives": int(y.sum()),
        "prevalence": float(np.average(y, weights=weights)),
        "roc_auc": float(roc_auc_score(y, ranking, sample_weight=weights)),
        "pr_auc": float(average_precision_score(y, ranking, sample_weight=weights)),
        "f1": float(f1_score(y, predicted, sample_weight=weights, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted, sample_weight=weights)),
        "brier_score": float(brier_score_loss(y, probability, sample_weight=weights)),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6), sample_weight=weights, labels=[0, 1])),
        "prediction_positive_rate": float(np.average(predicted, weights=weights)),
        "degenerate_prediction": bool(np.average(predicted, weights=weights) <= 0.05 or np.average(predicted, weights=weights) >= 0.95),
    }


def _static_prior_rows(context: FixedCrossModelContext) -> list[dict[str, object]]:
    metadata = context.bundle.train.endpoint_metadata().copy()
    metadata["y_true"] = dataset_targets(context.bundle.train).astype(int)
    metadata["asset_ticker"] = metadata["asset_id"].map({value: key for key, value in context.bundle.asset_to_id.items()})
    metadata["family"] = metadata["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
    test = context.bundle.test.endpoint_metadata().copy()
    test["y_true"] = dataset_targets(context.bundle.test).astype(int)
    test["asset_ticker"] = test["asset_id"].map({value: key for key, value in context.bundle.asset_to_id.items()})
    test["family"] = test["asset_ticker"].map(context.phase6.family_map).fillna("Unknown")
    global_prior = float(metadata["y_true"].mean())
    mappings = {
        "static_global_prior": np.full(len(test), global_prior),
        "static_family_prior": test["family"].map(fit_group_priors(metadata, "family")).fillna(global_prior).to_numpy(),
        "static_asset_prior": test["asset_ticker"].map(fit_group_priors(metadata, "asset_ticker")).fillna(global_prior).to_numpy(),
    }
    rows: list[dict[str, object]] = []
    for model_name, probability in mappings.items():
        test["selected_probability"] = probability
        metrics = _weighted_binary_metrics(test, "selected_probability", "selected_probability", 0.5, None)
        within = _within_group_auc(test["y_true"].to_numpy(), probability, test["asset_id"].to_numpy())
        rows.append({"model": model_name, "identity_variant": "train_only_prior", "seed": "na", "aggregation": "pooled", "calibration_method": "none", "threshold": 0.5, "status": "completed", **metrics, **within})
    return rows


def _load_transformer_checkpoint(
    context: FixedCrossModelContext,
    identity_variant: str,
    seed: int,
    validate_predictions: bool,
) -> torch.nn.Module:
    checkpoint_path, prediction_path = _historical_checkpoint_paths(context, identity_variant, seed)
    if not checkpoint_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"Missing frozen Transformer artifacts: {checkpoint_path} / {prediction_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_variant = HISTORICAL_VARIANTS[identity_variant]
    if payload.get("variant") != expected_variant or int(payload.get("seed")) != seed:
        raise RuntimeError("Historical Transformer checkpoint identity mismatch")
    if list(payload.get("features", [])) != list(context.bundle.feature_columns):
        raise RuntimeError("Historical Transformer feature manifest mismatch")
    if payload.get("split") != _split_manifest(context.phase6.split):
        raise RuntimeError("Historical Transformer split manifest mismatch")
    if payload.get("training") != context.options["training"]:
        raise RuntimeError("Historical Transformer training configuration mismatch")
    model = build_fixed_neural_model(context, "transformer_encoder", identity_variant)
    state = dict(payload["state_dict"])
    if identity_variant == "no_explicit_asset_id":
        key = "base_model.input_projection.weight"
        expected = model.state_dict()[key]
        historical = state[key]
        if historical.shape[1] == len(context.bundle.feature_columns) and expected.shape[1] > historical.shape[1]:
            padded = torch.zeros_like(expected)
            padded[:, : historical.shape[1]] = historical
            state[key] = padded
    model.load_state_dict(state, strict=True)
    model.eval()
    if validate_predictions:
        _validate_historical_predictions(context, model, prediction_path)
    return model


def _validate_historical_predictions(context: FixedCrossModelContext, model: torch.nn.Module, prediction_path: Path) -> None:
    frozen = pd.read_parquet(prediction_path)
    for split_name, dataset in (("validation", context.bundle.validation), ("test", context.bundle.test)):
        loader = DataLoader(dataset, batch_size=int(context.options["training"]["batch_size"]), shuffle=False, num_workers=0)
        y, probability, indices = predict_loader(model.to(context.phase6.device), loader, "classification", context.phase6.device)
        rebuilt = _prediction_frame(dataset, indices, y, probability, context.bundle.asset_to_id, split_name).sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
        reference = frozen[frozen["split"].eq(split_name)].sort_values(["Date", "source_index", "asset_id"]).reset_index(drop=True)
        if not rebuilt[["Date", "source_index", "asset_id", "asset_ticker", "y_true"]].equals(reference[["Date", "source_index", "asset_id", "asset_ticker", "y_true"]]):
            raise RuntimeError(f"Historical Transformer endpoint reconstruction failed for {split_name}")
        if not np.allclose(rebuilt["raw_probability"], reference["raw_probability"], atol=2e-6, rtol=1e-5):
            maximum = float(np.max(np.abs(rebuilt["raw_probability"] - reference["raw_probability"])))
            raise RuntimeError(f"Historical Transformer probabilities failed reconstruction: max abs {maximum}")


def _validate_local_payload(context: FixedCrossModelContext, payload: dict[str, Any], model_name: str, identity_variant: str, seed: int) -> None:
    checks = {
        "model": model_name,
        "identity_variant": identity_variant,
        "seed": seed,
        "features": context.bundle.feature_columns,
        "split": _split_manifest(context.phase6.split),
        "asset_to_id": context.bundle.asset_to_id,
        "endpoint_sha256": context.endpoint_sha256,
        "config_sha256": context.config_sha256,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"Checkpoint {key} mismatch for {model_name}/{identity_variant}/{seed}")


@torch.no_grad()
def _predict_with_represented_id_swap(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    represented_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    swap = {asset_id: represented_ids[(position + 1) % len(represented_ids)] for position, asset_id in enumerate(represented_ids)}
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    model.eval()
    for features, target, source_index, asset_id in loader:
        swapped = torch.tensor([swap[int(value)] for value in asset_id], dtype=torch.long, device=device)
        logits = model(features.to(device), swapped)
        labels.append(target.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        indices.append(source_index.numpy())
    return np.concatenate(labels), np.concatenate(probabilities), np.concatenate(indices)


def _write_contract_manifest(context: FixedCrossModelContext, preflight: pd.DataFrame) -> None:
    manifest = {
        "config_sha256": context.config_sha256,
        "endpoint_sha256": context.endpoint_sha256,
        "features": context.bundle.feature_columns,
        "asset_to_id": context.bundle.asset_to_id,
        "represented_asset_ids": context.represented_asset_ids,
        "skipped_assets": context.bundle.skipped_assets,
        "split": _split_manifest(context.phase6.split),
        "windows": {"train": len(context.bundle.train), "validation": len(context.bundle.validation), "test": len(context.bundle.test)},
        "preflight_rows": len(preflight),
    }
    (context.run_dir / "manifests" / "contract.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _write_execution_manifest(context: FixedCrossModelContext, rows: Iterable[dict[str, object]]) -> None:
    frame = pd.DataFrame(list(rows))
    frame.to_csv(context.run_dir / "fixed_cross_model_execution_manifest.csv", index=False)
    public = frame.copy()
    project_root = context.table_dir.parents[1]
    for column in ("checkpoint", "prediction"):
        if column in public:
            public[column] = public[column].map(
                lambda value: Path(str(value)).resolve().relative_to(project_root.resolve()).as_posix()
                if pd.notna(value) and str(value)
                else value
            )
    public.to_csv(context.table_dir / "prp1_fixed_cross_model_execution_manifest.csv", index=False)


def _execution_row(
    context: FixedCrossModelContext,
    model: str,
    identity_variant: str,
    seed: int,
    status: str,
    checkpoint: Path,
    prediction: Path,
) -> dict[str, object]:
    neural = model != "flattened_logistic"
    return {
        "model": model,
        "identity_variant": identity_variant,
        "seed": seed,
        "status": status,
        "parameter_count": count_parameters(build_fixed_neural_model(context, model, identity_variant)) if neural else np.nan,
        "train_windows": len(context.bundle.train),
        "validation_windows": len(context.bundle.validation),
        "test_windows": len(context.bundle.test),
        "checkpoint": str(checkpoint),
        "prediction": str(prediction),
        "config_sha256": context.config_sha256,
        "endpoint_sha256": context.endpoint_sha256,
    }


def _model_options(context: FixedCrossModelContext, model_name: str) -> dict[str, Any]:
    output = deepcopy(context.options["model"])
    output["hidden_size"] = int(output["hidden_sizes"][model_name])
    output["num_layers"] = int(output["layers"][model_name])
    return output


def _historical_checkpoint_paths(context: FixedCrossModelContext, identity_variant: str, seed: int) -> tuple[Path, Path]:
    root = Path("results/runs/phase6_transformer_falsification_20260712").resolve()
    variant = HISTORICAL_VARIANTS[identity_variant]
    return root / "checkpoints" / f"{variant}_seed{seed}.pt", root / "predictions" / f"{variant}_seed{seed}.parquet"


def _checkpoint_path(context: FixedCrossModelContext, model: str, identity_variant: str, seed: int) -> Path:
    return context.run_dir / "checkpoints" / f"{model}_{identity_variant}_seed{seed}.pt"


def _prediction_path(context: FixedCrossModelContext, model: str, identity_variant: str, seed: int) -> Path:
    return context.run_dir / "predictions" / f"{model}_{identity_variant}_seed{seed}.parquet"


def _log_path(context: FixedCrossModelContext, model: str, identity_variant: str, seed: int) -> Path:
    return context.run_dir / "logs" / f"{model}_{identity_variant}_seed{seed}.json"


def _seeds(context: FixedCrossModelContext) -> tuple[int, ...]:
    return tuple(int(value) for value in context.options["seeds"])


def _safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else np.nan


def _failure_metric_row(model: str, identity_variant: str, seed: object, reason: str) -> dict[str, object]:
    return {"model": model, "identity_variant": identity_variant, "seed": seed, "aggregation": "pooled", "status": "failed", "failure_reason": reason}
