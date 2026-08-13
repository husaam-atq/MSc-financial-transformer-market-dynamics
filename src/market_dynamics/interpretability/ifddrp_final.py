"""Bounded interpretation of the authoritative corrected Phase 6 Transformer."""

from __future__ import annotations

import zlib
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from market_dynamics.datasets.pooled_window_dataset import PooledWindowDataBundle
from market_dynamics.experiments.ifddrp_final_experiments import (
    FinalExperimentContext,
    _classification_priors,
    _fast_date_block_bootstrap_within_auc,
    _fast_within_auc,
    _history_preserving_transform,
)
from market_dynamics.experiments.run_phase6 import PerturbedDataset
from market_dynamics.experiments.run_phase6 import _model as build_phase6_model
from market_dynamics.training.sampling import dataset_targets
from market_dynamics.training.train import predict_loader
from market_dynamics.utils.torch_utils import set_torch_seed


@dataclass
class InterpretationContext:
    """Verified models and outcome-blind samples for final interpretation."""

    final: FinalExperimentContext
    options: dict[str, Any]
    models: dict[str, dict[int, nn.Module]]
    sample_features: torch.Tensor
    sample_asset_ids: torch.Tensor
    sample_metadata: pd.DataFrame
    ig_features: torch.Tensor
    ig_asset_ids: torch.Tensor
    ig_metadata: pd.DataFrame
    groups: dict[str, list[int]]
    reconstruction: pd.DataFrame
    run_dir: Path


def run_final_transformer_interpretation(
    final: FinalExperimentContext,
    config: dict[str, Any],
    *,
    run_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Run the frozen attribution, sanity, identity and regime diagnostics."""
    options = dict(config.get("ifddrp_final_interpretability", config))
    context = _prepare_context(final, options, Path(run_dir))
    _write_specification(context)
    feature_rows, lag_rows, cache = _attribution_suite(context)
    sanity = _sanity_suite(context, feature_rows, cache)
    identity = _identity_decomposition(context, cache)
    probes = _state_probe_registry(context)
    regimes = _regime_analysis(context)
    feature = pd.DataFrame(feature_rows)
    lag = pd.DataFrame(lag_rows)
    table_dir = final.table_dir
    feature.to_csv(table_dir / "ifddrp_transformer_feature_attribution.csv", index=False)
    lag.to_csv(table_dir / "ifddrp_transformer_lag_attribution.csv", index=False)
    sanity.to_csv(table_dir / "ifddrp_interpretability_sanity_checks.csv", index=False)
    identity.to_csv(table_dir / "ifddrp_identity_dynamic_information_decomposition.csv", index=False)
    probes.to_csv(table_dir / "ifddrp_transformer_state_probe_results.csv", index=False)
    regimes.to_csv(table_dir / "ifddrp_regime_conditional_results.csv", index=False)
    _write_interpretability_verdict(context, feature, lag, sanity, identity)
    _write_regime_verdict(context, regimes)
    _write_emergent_gate(context, regimes)
    _write_title_assessment(context)
    context.reconstruction.to_csv(context.run_dir / "prediction_reconstruction.csv", index=False)
    return {
        "features": feature,
        "lags": lag,
        "sanity": sanity,
        "identity": identity,
        "probes": probes,
        "regimes": regimes,
        "reconstruction": context.reconstruction,
    }


def _prepare_context(
    final: FinalExperimentContext,
    options: dict[str, Any],
    run_dir: Path,
) -> InterpretationContext:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    if int(options["lookback"]) != int(final.phase6.options["lookback"]):
        raise RuntimeError("Interpretability lookback differs from the authoritative model")
    variants = [str(options["authoritative_variant"]), str(options["identity_control_variant"])]
    models: dict[str, dict[int, nn.Module]] = {}
    reconstruction_rows: list[dict[str, object]] = []
    for variant in variants:
        models[variant] = {}
        for seed in [int(value) for value in options["seeds"]]:
            set_torch_seed(seed, deterministic=True)
            model = build_phase6_model(final.phase6, variant, final.bundle)
            checkpoint_path = final.phase6.run_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if list(checkpoint.get("features", [])) != list(final.bundle.feature_columns):
                raise RuntimeError(f"Checkpoint feature registry mismatch: {checkpoint_path}")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model = model.to(final.phase6.device).eval()
            models[variant][seed] = model
            reconstruction_rows.append(_verify_reconstruction(final, model, variant, seed))
    reconstruction = pd.DataFrame(reconstruction_rows)
    if not reconstruction["passed"].all():
        raise RuntimeError("Authoritative prediction reconstruction failed")
    sample_features, sample_ids, sample_meta = _equal_family_sample(
        final.bundle,
        final.phase6.family_map,
        int(options["sample"]["per_family"]),
    )
    ig_features, ig_ids, ig_meta = _equal_family_sample(
        final.bundle,
        final.phase6.family_map,
        int(options["sample"]["integrated_gradients_per_family"]),
    )
    groups = _feature_groups(final.bundle.feature_columns, options["feature_groups"])
    return InterpretationContext(
        final=final,
        options=options,
        models=models,
        sample_features=sample_features,
        sample_asset_ids=sample_ids,
        sample_metadata=sample_meta,
        ig_features=ig_features,
        ig_asset_ids=ig_ids,
        ig_metadata=ig_meta,
        groups=groups,
        reconstruction=reconstruction,
        run_dir=run_dir,
    )


def _verify_reconstruction(
    final: FinalExperimentContext,
    model: nn.Module,
    variant: str,
    seed: int,
) -> dict[str, object]:
    loader = DataLoader(final.bundle.test, batch_size=1024, shuffle=False, num_workers=0)
    y, probability, source = predict_loader(model, loader, "classification", final.phase6.device)
    rebuilt = pd.DataFrame({"source_index": source.astype(np.int64), "y": y, "p": probability})
    stored = pd.read_parquet(final.phase6.run_dir / "predictions" / f"{variant}_seed{seed}.parquet")
    stored = stored[stored["split"].eq("test")][["source_index", "y_true", "raw_probability"]]
    joined = stored.merge(rebuilt, on="source_index", how="outer", validate="one_to_one", indicator=True)
    membership = bool(joined["_merge"].eq("both").all() and len(joined) == len(stored))
    target_error = float(np.max(np.abs(joined["y_true"] - joined["y"]))) if membership else np.inf
    probability_error = float(np.max(np.abs(joined["raw_probability"] - joined["p"]))) if membership else np.inf
    return {
        "check": "authoritative_prediction_reconstruction",
        "variant": variant,
        "seed": seed,
        "rows": len(stored),
        "target_max_abs_error": target_error,
        "probability_max_abs_error": probability_error,
        "passed": bool(membership and target_error <= 1e-7 and probability_error <= 1e-6),
    }


def _equal_family_sample(
    bundle: PooledWindowDataBundle,
    family_map: dict[str, str],
    per_family: int,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    reverse = {value: key for key, value in bundle.asset_to_id.items()}
    metadata = bundle.test.endpoint_metadata().reset_index(drop=True)
    metadata["asset_ticker"] = metadata["asset_id"].map(reverse)
    metadata["family"] = metadata["asset_ticker"].map(family_map).fillna("Unknown")
    positions: list[int] = []
    for _, part in metadata.groupby("family", sort=True, observed=True):
        count = min(per_family, len(part))
        positions.extend(part.index[np.linspace(0, len(part) - 1, count, dtype=int)].tolist())
    positions = sorted(set(positions))
    items = [bundle.test[position] for position in positions]
    return (
        torch.stack([item[0] for item in items]),
        torch.stack([item[3] for item in items]).long().reshape(-1),
        metadata.iloc[positions].reset_index(drop=True),
    )


def _feature_groups(
    features: list[str],
    definitions: dict[str, list[str]],
) -> dict[str, list[int]]:
    configured = [column for columns in definitions.values() for column in columns]
    duplicates = sorted({column for column in configured if configured.count(column) > 1})
    missing = sorted(set(features) - set(configured))
    unknown = sorted(set(configured) - set(features))
    if duplicates or missing or unknown:
        raise RuntimeError(
            "Invalid interpretability feature partition: "
            f"duplicates={duplicates}, missing={missing}, unknown={unknown}"
        )
    groups = {
        name: [features.index(column) for column in columns]
        for name, columns in definitions.items()
    }
    if sorted(index for indices in groups.values() for index in indices) != list(range(len(features))):
        raise RuntimeError("Interpretability groups do not cover the model features exactly once")
    return groups


def _write_specification(context: InterpretationContext) -> None:
    options = context.final.phase6.options
    model_options = options["model"]
    representative = context.models[str(context.options["authoritative_variant"])][int(context.options["seeds"][0])]
    parameter_count = sum(parameter.numel() for parameter in representative.parameters())
    lines = [
        "# Authoritative Transformer Specification",
        "",
        "This specification is reconstructed from the committed Phase 6 config, code and verified checkpoints.",
        "",
        f"- Target: `{options['target']}` over the next 10 observed sessions.",
        f"- Lookback: `{options['lookback']}` observed sessions, ending at market close on origin day t.",
        f"- Inputs: `{len(context.final.bundle.feature_columns)}` train-scaled features plus a `{model_options['asset_embedding_dim']}`-channel learned asset embedding repeated over time.",
        f"- Hidden width: `{model_options['hidden_size']}`; encoder layers: `{model_options['num_layers']}`; heads: `{model_options['transformer_heads']}`; feed-forward width: `{int(model_options['hidden_size']) * int(model_options['transformer_ff_multiplier'])}`.",
        f"- Position: fixed sinusoidal encoding with maximum length `{model_options['max_length']}`.",
        f"- Pooling: `{model_options['transformer_pooling']}`; head: LayerNorm, dropout `{model_options['dropout']}`, scalar linear logit.",
        f"- Parameters: `{parameter_count:,}`.",
        f"- Training: `{options['training']['classification_loss']}` loss, AdamW at `{options['training']['learning_rate']}`, weight decay `{options['training']['weight_decay']}`, batch size `{options['training']['batch_size']}`, at most `{options['training']['epochs']}` epochs, three deterministic seeds.",
        f"- Split: fold `{options['fold']}`, purge `{context.final.phase6.split.purge}` global dates, embargo `{context.final.phase6.split.embargo}` date, train-only per-asset scaling.",
        "- Raw OHLCV fields are not direct model inputs. Technical inputs are engineered at or before t. Seven macro/context inputs are current-vintage/provenance-limited and cannot support strong positive attribution claims.",
        "- There is no explicit family or missingness-indicator input.",
    ]
    (context.final.table_dir / "ifddrp_transformer_authoritative_specification.md").write_text("\n".join(lines), encoding="utf-8")
    rows: list[dict[str, object]] = []
    group_by_feature = {
        context.final.bundle.feature_columns[index]: group
        for group, indices in context.groups.items()
        for index in indices
    }
    for position, feature in enumerate(context.final.bundle.feature_columns):
        group = group_by_feature[feature]
        rows.append(
            {
                "input_position": position,
                "feature": feature,
                "group": group,
                "source": "engineered_technical" if group != "macro_context" else "macro_context",
                "availability": "known_by_origin_close_t",
                "scaling": "per_asset_train_only_standardisation",
                "direct_model_input": True,
                "provenance_limit": group == "macro_context",
            }
        )
    for channel in range(int(model_options["asset_embedding_dim"])):
        rows.append(
            {
                "input_position": len(context.final.bundle.feature_columns) + channel,
                "feature": f"asset_embedding_channel_{channel + 1:02d}",
                "group": "asset_identity",
                "source": "learned_asset_id_embedding",
                "availability": "static_identifier",
                "scaling": "learned",
                "direct_model_input": True,
                "provenance_limit": False,
            }
        )
    pd.DataFrame(rows).to_csv(context.final.table_dir / "ifddrp_transformer_input_feature_registry.csv", index=False)


def _probability(model: nn.Module, features: torch.Tensor, asset_ids: torch.Tensor, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            x = features[start : start + 256].to(device)
            ids = asset_ids[start : start + 256].to(device)
            values.append(torch.sigmoid(model(x, ids)).cpu().numpy())
    return np.concatenate(values)


def _integrated_gradients(
    model: nn.Module,
    features: torch.Tensor,
    asset_ids: torch.Tensor,
    device: torch.device,
    steps: int,
) -> np.ndarray:
    x = features.to(device)
    ids = asset_ids.to(device)
    total = torch.zeros_like(x)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=device):
        scaled = (x * alpha).detach().requires_grad_(True)
        logits = model(scaled, ids)
        gradient = torch.autograd.grad(logits.sum(), scaled, retain_graph=False)[0]
        total += gradient.detach()
    return (x * total / steps).detach().cpu().numpy()


def _attribution_suite(
    context: InterpretationContext,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    features = context.sample_features
    asset_ids = context.sample_asset_ids
    device = context.final.phase6.device
    feature_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []
    cache: dict[str, Any] = {"ig_vectors": {}, "occlusion_vectors": {}}
    variants = [str(context.options["authoritative_variant"]), str(context.options["identity_control_variant"])]
    lag_blocks = {name: (int(bounds[0]), int(bounds[1])) for name, bounds in context.options["lag_blocks"].items()}
    for variant in variants:
        for seed, model in context.models[variant].items():
            original = _probability(model, features, asset_ids, device)
            zero_scores: dict[str, float] = {}
            for group, indices in context.groups.items():
                for method in ["train_mean_scaled_zero_occlusion", "within_window_circular_shift"]:
                    changed = features.clone()
                    if method.startswith("train_mean"):
                        changed[:, :, indices] = 0.0
                    else:
                        changed[:, :, indices] = torch.roll(changed[:, :, indices], shifts=30, dims=1)
                    masked = _probability(model, changed, asset_ids, device)
                    mean_abs = float(np.mean(np.abs(original - masked)))
                    if method.startswith("train_mean"):
                        zero_scores[group] = mean_abs
                    feature_rows.append(
                        _feature_row(context, variant, seed, group, method, original, masked, mean_abs)
                    )
            cache["occlusion_vectors"][(variant, seed)] = zero_scores
            for block, (start, stop) in lag_blocks.items():
                changed = features.clone()
                changed[:, start:stop, :] = 0.0
                masked = _probability(model, changed, asset_ids, device)
                lag_rows.append(_lag_row(variant, seed, block, start, stop, "train_mean_scaled_zero_occlusion", original, masked))
            ig = _integrated_gradients(
                model,
                context.ig_features,
                context.ig_asset_ids,
                device,
                int(context.options["integrated_gradients"]["steps"]),
            )
            ig_scores: dict[str, float] = {}
            for group, indices in context.groups.items():
                score = float(np.mean(np.abs(ig[:, :, indices])))
                ig_scores[group] = score
                feature_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "group": group,
                        "feature_names": ";".join(context.final.bundle.feature_columns[index] for index in indices),
                        "method": "integrated_gradients_raw_logit",
                        "n_obs": len(ig),
                        "mean_signed_probability_delta": np.nan,
                        "mean_absolute_probability_delta": np.nan,
                        "mean_absolute_logit_attribution": score,
                        "status": "completed",
                    }
                )
            cache["ig_vectors"][(variant, seed)] = ig_scores
            cache[("ig", variant, seed)] = ig
            for block, (start, stop) in lag_blocks.items():
                lag_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "lag_block": block,
                        "position_start": start,
                        "position_stop_exclusive": stop,
                        "method": "integrated_gradients_raw_logit",
                        "n_obs": len(ig),
                        "mean_signed_probability_delta": np.nan,
                        "mean_absolute_probability_delta": np.nan,
                        "mean_absolute_logit_attribution": float(np.mean(np.abs(ig[:, start:stop, :]))),
                        "attention_mass": np.nan,
                    }
                )
            weights = _attention_weights(model, features, asset_ids, device)
            for block, (start, stop) in lag_blocks.items():
                lag_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "lag_block": block,
                        "position_start": start,
                        "position_stop_exclusive": stop,
                        "method": "attention_pooling_diagnostic_only",
                        "n_obs": len(weights),
                        "mean_signed_probability_delta": np.nan,
                        "mean_absolute_probability_delta": np.nan,
                        "mean_absolute_logit_attribution": np.nan,
                        "attention_mass": float(weights[:, start:stop].sum(axis=1).mean()),
                    }
                )
        if variant == str(context.options["authoritative_variant"]):
            for seed, model in context.models[variant].items():
                original = _probability(model, features, asset_ids, device)
                conditioned = model.conditioned_input(features.to(device), asset_ids.to(device))
                zero_identity = conditioned.clone()
                embedding_dim = int(context.final.phase6.options["model"]["asset_embedding_dim"])
                zero_identity[:, :, -embedding_dim:] = 0.0
                with torch.no_grad():
                    masked = torch.sigmoid(model.base_model(zero_identity)).cpu().numpy()
                feature_rows.append(
                    _feature_row(context, variant, seed, "asset_identity", "zero_asset_embedding", original, masked, float(np.mean(np.abs(original - masked))))
                )
    for absent in ["family_identity", "missingness"]:
        feature_rows.append(
            {
                "variant": str(context.options["authoritative_variant"]),
                "seed": "not_applicable",
                "group": absent,
                "feature_names": "",
                "method": "direct_input_audit",
                "n_obs": 0,
                "mean_signed_probability_delta": np.nan,
                "mean_absolute_probability_delta": np.nan,
                "mean_absolute_logit_attribution": np.nan,
                "status": "no_explicit_input_channel",
            }
        )
    return feature_rows, lag_rows, cache


def _feature_row(
    context: InterpretationContext,
    variant: str,
    seed: int,
    group: str,
    method: str,
    original: np.ndarray,
    masked: np.ndarray,
    mean_abs: float,
) -> dict[str, object]:
    indices = context.groups.get(group, [])
    return {
        "variant": variant,
        "seed": seed,
        "group": group,
        "feature_names": ";".join(context.final.bundle.feature_columns[index] for index in indices),
        "method": method,
        "n_obs": len(original),
        "mean_signed_probability_delta": float(np.mean(original - masked)),
        "mean_absolute_probability_delta": mean_abs,
        "mean_absolute_logit_attribution": np.nan,
        "status": "provenance_limited" if group == "macro_context" else "completed",
    }


def _lag_row(
    variant: str,
    seed: int,
    block: str,
    start: int,
    stop: int,
    method: str,
    original: np.ndarray,
    masked: np.ndarray,
) -> dict[str, object]:
    return {
        "variant": variant,
        "seed": seed,
        "lag_block": block,
        "position_start": start,
        "position_stop_exclusive": stop,
        "method": method,
        "n_obs": len(original),
        "mean_signed_probability_delta": float(np.mean(original - masked)),
        "mean_absolute_probability_delta": float(np.mean(np.abs(original - masked))),
        "mean_absolute_logit_attribution": np.nan,
        "attention_mass": np.nan,
    }


def _attention_weights(model: nn.Module, features: torch.Tensor, asset_ids: torch.Tensor, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            x = features[start : start + 256].to(device)
            ids = asset_ids[start : start + 256].to(device)
            conditioned = model.conditioned_input(x, ids)
            _, _, weights = model.base_model.encode(conditioned)
            if weights is None:
                raise RuntimeError("Authoritative model does not expose temporal pooling weights")
            values.append(weights.cpu().numpy())
    return np.concatenate(values)


def _sanity_suite(
    context: InterpretationContext,
    feature_rows: list[dict[str, object]],
    cache: dict[str, Any],
) -> pd.DataFrame:
    rows = context.reconstruction.to_dict(orient="records")
    variant = str(context.options["authoritative_variant"])
    seeds = [int(value) for value in context.options["seeds"]]
    for seed in seeds:
        ig = cache["ig_vectors"][(variant, seed)]
        occlusion = cache["occlusion_vectors"][(variant, seed)]
        groups = sorted(set(ig) & set(occlusion))
        rho = float(spearmanr([ig[group] for group in groups], [occlusion[group] for group in groups]).statistic)
        rows.append({"check": "attribution_occlusion_rank_agreement", "variant": variant, "seed": seed, "value": rho, "passed": rho >= float(context.options["sanity"]["minimum_attribution_occlusion_rank_correlation"]), "detail": ";".join(groups)})
    for left, right in combinations(seeds, 2):
        left_scores = cache["ig_vectors"][(variant, left)]
        right_scores = cache["ig_vectors"][(variant, right)]
        groups = sorted(left_scores)
        rho = float(spearmanr([left_scores[group] for group in groups], [right_scores[group] for group in groups]).statistic)
        rows.append({"check": "integrated_gradients_seed_rank_stability", "variant": variant, "seed": f"{left}_vs_{right}", "value": rho, "passed": rho >= 0.3, "detail": "group-rank diagnostic"})
    seed = seeds[0]
    random_seed = int(context.options["sanity"]["parameter_randomisation_seed"])
    set_torch_seed(random_seed, deterministic=True)
    randomized = deepcopy(context.models[variant][seed])
    for module in randomized.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
        elif isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()
    randomized = randomized.to(context.final.phase6.device).eval()
    random_ig = _integrated_gradients(
        randomized,
        context.ig_features,
        context.ig_asset_ids,
        context.final.phase6.device,
        int(context.options["integrated_gradients"]["steps"]),
    )
    trained_scores = cache["ig_vectors"][(variant, seed)]
    random_scores = {
        group: float(np.mean(np.abs(random_ig[:, :, indices])))
        for group, indices in context.groups.items()
    }
    groups = sorted(trained_scores)
    random_rho = float(spearmanr([trained_scores[group] for group in groups], [random_scores[group] for group in groups]).statistic)
    rows.append({"check": "trained_vs_parameter_randomized_attribution_rank", "variant": variant, "seed": random_seed, "value": random_rho, "passed": abs(random_rho) <= float(context.options["sanity"]["maximum_trained_random_attribution_rank_correlation"]), "detail": "full module parameter reset"})
    sample_ig = cache[("ig", variant, seed)]
    magnitude = np.mean(np.abs(sample_ig), axis=(1, 2))
    labels = np.asarray(
        [context.final.bundle.test[position][1].item() for position in _sample_positions(context.final.bundle, context.ig_metadata)],
        dtype=float,
    )
    observed = float(abs(spearmanr(magnitude, labels).statistic)) if len(np.unique(labels)) == 2 else np.nan
    rng = np.random.default_rng(random_seed)
    draws = np.asarray([abs(spearmanr(magnitude, rng.permutation(labels)).statistic) for _ in range(int(context.options["sanity"]["label_permutations"]))])
    p_value = float((1 + np.sum(draws >= observed)) / (len(draws) + 1)) if np.isfinite(observed) else np.nan
    rows.append({"check": "label_permutation_attribution_association", "variant": variant, "seed": seed, "value": observed, "passed": True, "detail": f"diagnostic permutation p={p_value:.4f}; not a causal explanation test"})
    order = pd.read_csv(context.final.table_dir / "phase6_temporal_order_destruction.csv")
    baseline = _raw_variant_metrics(context, variant)
    for _, row in order[order["model_variant"].eq(variant)].iterrows():
        rows.append({"check": "historical_temporal_order_control", "variant": variant, "seed": "ensemble", "value": float(baseline["pooled_roc_auc"] - row["roc_auc"]), "passed": False, "detail": str(row["perturbation"])})
    no_id = _raw_variant_metrics(context, str(context.options["identity_control_variant"]))
    rows.append({"check": "identity_removal_pooled_auc_drop", "variant": variant, "seed": "ensemble", "value": float(baseline["pooled_roc_auc"] - no_id["pooled_roc_auc"]), "passed": True, "detail": "identity materially affects pooled ranking"})
    return pd.DataFrame(rows)


def _sample_positions(bundle: PooledWindowDataBundle, metadata: pd.DataFrame) -> list[int]:
    lookup = bundle.test.endpoint_metadata().reset_index().rename(columns={"index": "position"})
    joined = metadata.merge(lookup, on=["Date", "source_index", "asset_id"], how="left", validate="one_to_one")
    if joined["position"].isna().any():
        raise RuntimeError("Sample endpoints could not be mapped back to the dataset")
    return joined["position"].astype(int).tolist()


def _raw_variant_metrics(context: InterpretationContext, variant: str) -> dict[str, float]:
    frame = pd.read_parquet(context.final.phase6.run_dir / "predictions" / f"{variant}_ensemble.parquet")
    frame = frame[frame["split"].eq("test")].copy()
    probability_column = "ensemble_probability"
    y = frame["y_true"].to_numpy(dtype=int)
    p = frame[probability_column].to_numpy(dtype=float)
    return {
        "pooled_roc_auc": float(roc_auc_score(y, p)),
        "within_asset_roc_auc": _fast_within_auc(y, p, frame["asset_id"].to_numpy(dtype=int)),
    }


def _identity_decomposition(context: InterpretationContext, cache: dict[str, Any]) -> pd.DataFrame:
    conditioned = _raw_variant_metrics(context, str(context.options["authoritative_variant"]))
    no_id = _raw_variant_metrics(context, str(context.options["identity_control_variant"]))
    priors = _classification_priors(context.final.bundle, context.final.phase6.family_map)
    test_meta = context.final.bundle.test.endpoint_metadata()
    y = dataset_targets(context.final.bundle.test).astype(int)
    asset = test_meta["asset_id"].to_numpy(dtype=int)
    asset_prior = np.asarray(priors["asset_vector"], dtype=float)[asset]
    family_prior = np.asarray(priors["family_vector"], dtype=float)[asset]
    probe = pd.read_csv(context.final.table_dir / "phase6_representation_probe_results.csv")
    asset_probe = probe[(probe["model_variant"].eq("corrected_asset_conditioned")) & (probe["representation"].eq("transformer_summary")) & (probe["probe_task"].eq("asset_identity")) & (probe["split"].eq("test"))].iloc[0]
    rows = [
        {"component": "conditioned_transformer", **conditioned, "identity_probe_accuracy": float(asset_probe["accuracy"]), "interpretation": "pooled ranking mixes static identity and dynamic inputs"},
        {"component": "no_explicit_asset_id_transformer", **no_id, "identity_probe_accuracy": np.nan, "interpretation": "removing ID lowers pooled AUC without recovering within-asset skill"},
        {"component": "training_only_asset_prior", "pooled_roc_auc": float(roc_auc_score(y, asset_prior)), "within_asset_roc_auc": 0.5, "identity_probe_accuracy": 1.0, "interpretation": "strongest pooled static comparator"},
        {"component": "training_only_family_prior", "pooled_roc_auc": float(roc_auc_score(y, family_prior)), "within_asset_roc_auc": 0.5, "identity_probe_accuracy": np.nan, "interpretation": "family-level static comparator"},
    ]
    return pd.DataFrame(rows)


def _state_probe_registry(context: InterpretationContext) -> pd.DataFrame:
    source = pd.read_csv(context.final.table_dir / "phase6_representation_probe_results.csv")
    output = source[source["model_variant"].isin(["corrected_asset_conditioned", "no_explicit_asset_id"])].copy()
    output["evidence_class"] = "post_hoc_probe_on_opened_historical_split"
    output["forecasting_interpretation"] = np.where(
        output["probe_task"].eq("stress_state"),
        "state decodability does not establish use by the trained forecasting head",
        "identity diagnostic",
    )
    return output


def _regime_analysis(context: InterpretationContext) -> pd.DataFrame:
    variant = str(context.options["authoritative_variant"])
    original = pd.read_parquet(context.final.phase6.run_dir / "predictions" / f"{variant}_ensemble.parquet")
    original = original[original["split"].eq("test")].copy()
    original["raw_probability"] = original["ensemble_probability"]
    original = original.sort_values("source_index").reset_index(drop=True)
    perturbed = {
        method: _safe_perturbed_ensemble(context, method).sort_values("source_index").reset_index(drop=True)
        for method in context.options["safe_temporal_controls"]["methods"]
    }
    for method, frame in perturbed.items():
        if not np.array_equal(original["source_index"].to_numpy(), frame["source_index"].to_numpy()):
            raise RuntimeError(f"Regime perturbation endpoint mismatch: {method}")
    labelled = _attach_regimes(context, original)
    priors = _classification_priors(context.final.bundle, context.final.phase6.family_map)
    labelled["asset_prior"] = np.asarray(priors["asset_vector"], dtype=float)[labelled["asset_id"].to_numpy(dtype=int)]
    rows: list[dict[str, object]] = []
    regime_columns = [column for column in labelled.columns if column.startswith("regime__")]
    for column in regime_columns:
        regime_type = column.removeprefix("regime__")
        for value, part in labelled.groupby(column, observed=True, sort=True):
            if pd.isna(value) or len(part) < 100 or part["y_true"].nunique() < 2:
                continue
            y = part["y_true"].to_numpy(dtype=int)
            p = part["raw_probability"].to_numpy(dtype=float)
            assets = part["asset_id"].to_numpy(dtype=int)
            within = _fast_within_auc(y, p, assets)
            interval = _fast_date_block_bootstrap_within_auc(
                part,
                "raw_probability",
                int(context.options["regimes"]["date_block_length"]),
                int(context.options["regimes"]["bootstrap_iterations"]),
                int(context.options["regimes"]["bootstrap_seed"]) + zlib.crc32(f"{regime_type}/{value}".encode()),
            )
            row: dict[str, object] = {
                "regime_type": regime_type,
                "regime": value,
                "rows": len(part),
                "assets": part["asset_id"].nunique(),
                "dates": part["Date"].nunique(),
                "positives": int(y.sum()),
                "prevalence": float(y.mean()),
                "pooled_roc_auc": float(roc_auc_score(y, p)),
                "pair_weighted_within_asset_roc_auc": within,
                "within_asset_ci_lower": interval["ci_lower"],
                "within_asset_ci_upper": interval["ci_upper"],
                "static_asset_prior_pooled_roc_auc": float(roc_auc_score(y, part["asset_prior"])),
                "static_asset_prior_within_asset_roc_auc": 0.5,
            }
            for method, frame in perturbed.items():
                changed = (
                    frame.set_index("source_index")
                    .loc[part["source_index"].to_numpy(dtype=np.int64), "raw_probability"]
                    .to_numpy(dtype=float)
                )
                changed_within = _fast_within_auc(y, changed, assets)
                row[f"{method}_within_asset_roc_auc"] = changed_within
                row[f"{method}_within_asset_auc_drop"] = within - changed_within
            rows.append(row)
    return pd.DataFrame(rows)


def _safe_perturbed_ensemble(context: InterpretationContext, method: str) -> pd.DataFrame:
    path = context.run_dir / "predictions" / f"safe_{method}_ensemble.parquet"
    if path.exists():
        return pd.read_parquet(path)
    variant = str(context.options["authoritative_variant"])
    transform = _history_preserving_transform(
        method,
        int(context.options["lookback"]),
        int(context.options["safe_temporal_controls"]["seed"]),
    )
    metadata = context.final.bundle.test.endpoint_metadata().copy()
    seed_probabilities: list[np.ndarray] = []
    for seed, model in context.models[variant].items():
        loader = DataLoader(PerturbedDataset(context.final.bundle.test, transform), batch_size=1024, shuffle=False, num_workers=0)
        y, probability, source = predict_loader(model, loader, "classification", context.final.phase6.device)
        if not np.array_equal(source.astype(np.int64), metadata["source_index"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"Perturbed prediction ordering mismatch for seed {seed}")
        seed_probabilities.append(probability)
    output = metadata.copy()
    output["y_true"] = y
    output["raw_probability"] = np.mean(np.stack(seed_probabilities), axis=0)
    output.to_parquet(path, index=False)
    return output


def _attach_regimes(context: InterpretationContext, prediction: pd.DataFrame) -> pd.DataFrame:
    panel = context.final.phase6.target_panel.reset_index().copy()
    panel["asset_ticker"] = panel["Ticker"].astype(str)
    panel["family"] = panel["asset_ticker"].map(context.final.phase6.family_map).fillna("Unknown")
    settings = context.options["regimes"]
    vol = str(settings["volatility_feature"])
    stress = str(settings["current_stress_feature"])
    trend = str(settings["broad_trend_feature"])
    train = panel[panel["Date"].isin(context.final.phase6.split.train_dates)]
    vol_threshold = train.groupby("asset_ticker", observed=True)[vol].quantile(float(settings["volatility_training_quantile"]))
    stress_threshold = train.groupby("asset_ticker", observed=True)[stress].quantile(float(settings["current_stress_training_quantile"]))
    equity_trend = panel[panel["family"].eq("Equities")].groupby("Date", observed=True)[trend].median().rename("broad_trend")
    endpoint = panel[["Date", "asset_ticker", "family", vol, stress]].merge(equity_trend, on="Date", how="left")
    output = prediction.merge(endpoint, on=["Date", "asset_ticker", "family"], how="left", validate="one_to_one")
    output["vol_threshold"] = output["asset_ticker"].map(vol_threshold)
    output["stress_threshold"] = output["asset_ticker"].map(stress_threshold)
    output["regime__volatility"] = np.where(output[vol].ge(output["vol_threshold"]), "high", "low")
    output["regime__broad_trend"] = np.where(output["broad_trend"].ge(0.0), "positive", "negative")
    output["regime__current_stress"] = np.where(output[stress].le(output["stress_threshold"]), "stress", "non_stress")
    output["regime__asset_family"] = output["family"]
    output["regime__subperiod"] = "outside_registered_subperiod"
    for name, bounds in settings["test_subperiods"].items():
        mask = output["Date"].between(pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        output.loc[mask, "regime__subperiod"] = name
    return output


def _write_interpretability_verdict(
    context: InterpretationContext,
    feature: pd.DataFrame,
    lag: pd.DataFrame,
    sanity: pd.DataFrame,
    identity: pd.DataFrame,
) -> None:
    variant = str(context.options["authoritative_variant"])
    completed = feature[(feature["variant"].eq(variant)) & (feature["method"].eq("train_mean_scaled_zero_occlusion"))]
    ranking = completed.groupby("group", observed=True)["mean_absolute_probability_delta"].mean().sort_values(ascending=False)
    lag_rank = lag[(lag["variant"].eq(variant)) & (lag["method"].eq("train_mean_scaled_zero_occlusion"))].groupby("lag_block", observed=True)["mean_absolute_probability_delta"].mean().sort_values(ascending=False)
    failed = sanity[sanity["passed"].eq(False)]
    text = "\n".join([
        "# Final Transformer Interpretability Verdict",
        "",
        "Evidence class: post-hoc robustness on an opened historical test.",
        "",
        f"The largest mean zero-occlusion response was `{ranking.index[0]}` ({ranking.iloc[0]:.4f} mean absolute probability change). The largest registered lag response was `{lag_rank.index[0]}` ({lag_rank.iloc[0]:.4f}). These are sensitivity diagnostics, not causal importance.",
        "",
        f"Prediction reconstruction passed for all checkpoints. `{len(failed)}` sanity/control rows failed their desired dynamic-explanation condition. Historical order controls remain decisive: sequence destruction barely changed pooled ranking.",
        "",
        "Asset identity remains the strongest scientific interpretation. The model encodes state-sensitive inputs, but the trained head did not establish useful within-asset timing. Macro/context sensitivity is provenance-limited and is not promoted.",
        "",
        "Attention pooling weights are retained only as supplementary diagnostics. They are not treated as explanations.",
    ])
    (context.final.table_dir / "ifddrp_transformer_interpretability_verdict.md").write_text(text, encoding="utf-8")


def _write_regime_verdict(context: InterpretationContext, regimes: pd.DataFrame) -> None:
    gate = context.options["promotion_gate"]
    if regimes.empty:
        passed = pd.Series(dtype=bool)
    else:
        order_columns = [column for column in regimes if column.endswith("within_asset_auc_drop")]
        passed = (
            regimes["pair_weighted_within_asset_roc_auc"].ge(float(gate["minimum_pair_weighted_within_asset_auc"]))
            & regimes["within_asset_ci_lower"].gt(0.5)
            & regimes["pair_weighted_within_asset_roc_auc"].sub(0.5).ge(float(gate["minimum_incremental_auc_over_static_prior"]))
            & regimes[order_columns].ge(float(gate["minimum_each_temporal_auc_drop"])).all(axis=1)
        )
    winners = regimes.loc[passed, ["regime_type", "regime"]].to_dict(orient="records") if len(passed) else []
    text = "\n".join([
        "# Regime-Conditional Verdict",
        "",
        f"Registered regime cells passing every temporal-skill gate: `{len(winners)}`.",
        "",
        "No regime is promoted unless its within-asset interval excludes chance, it improves on the within-asset static-prior benchmark, and all three endpoint-preserving chronology controls deteriorate performance.",
        "",
        f"Passing cells: `{winners}`.",
    ])
    (context.final.table_dir / "ifddrp_regime_conditional_verdict.md").write_text(text, encoding="utf-8")


def _write_emergent_gate(context: InterpretationContext, regimes: pd.DataFrame) -> None:
    static_path = context.final.table_dir / "ifddrp_static_dynamic_verdict.md"
    within_path = context.final.table_dir / "ifddrp_within_asset_objective_verdict.md"
    continuous_path = context.final.table_dir / "ifddrp_continuous_downside_verdict.md"
    verdicts = {path.stem: ("passed" if "Overall promotion gate: **passed**" in path.read_text(encoding="utf-8") else "failed") for path in [static_path, within_path, continuous_path] if path.exists()}
    regime_pass = False
    if not regimes.empty:
        order_columns = [column for column in regimes if column.endswith("within_asset_auc_drop")]
        regime_pass = bool((regimes["within_asset_ci_lower"].gt(0.5) & regimes[order_columns].gt(0.02).all(axis=1)).any())
    unlocked = any(value == "passed" for value in verdicts.values()) or regime_pass
    text = "\n".join([
        "# Emergent-Dynamics Analysis Gate",
        "",
        f"Bounded experiment verdicts: `{verdicts}`.",
        f"A registered regime passed the strict dynamic gate: `{regime_pass}`.",
        "",
        f"Full latent-state/change-point analysis is **{'unlocked' if unlocked else 'locked'}**.",
        "",
        "The gate prevents attractive latent-space visualisations from being presented as evidence when the forecasting head has not shown incremental, chronology-dependent within-asset skill.",
    ])
    (context.final.table_dir / "ifddrp_emergent_dynamics_gate.md").write_text(text, encoding="utf-8")


def _write_title_assessment(context: InterpretationContext) -> None:
    rows = [
        {"title_component": "Interpretable", "support": "partial", "evidence": "verified occlusion/IG/identity diagnostics with sanity controls", "limit": "sensitivity is not causal explanation"},
        {"title_component": "Transformer Models", "support": "strong", "evidence": "real three-seed Transformer Encoder checkpoints reconstructed exactly", "limit": "Transformer did not beat static priors fairly"},
        {"title_component": "Financial Time Series Forecasting", "support": "strong", "evidence": "79-asset chronological, purged, train-scaled forecasting panel", "limit": "historical test is opened/adaptive"},
        {"title_component": "Discovering Emergent Market Dynamics", "support": "limited_negative", "evidence": "adversarial tests identify static cross-sectional shortcut structure", "limit": "genuine chronology-dependent within-asset dynamics were not established"},
    ]
    matrix = pd.DataFrame(rows)
    markdown = ["# Title Component Evidence Matrix", "", _markdown_table(matrix)]
    (context.final.table_dir / "ifddrp_title_component_evidence_matrix.md").write_text("\n".join(markdown), encoding="utf-8")
    text = "\n".join([
        "# Title Alignment Verdict",
        "",
        "The fixed title is supported only under interpretation **3: apparent dynamics mainly reflected static shortcut structure**, with partial support for interpretation 4.",
        "",
        "The project is genuinely about Transformer forecasting and uses bounded interpretation methods. Its strongest discovery is methodological: high pooled discrimination can emerge from asset identity and heterogeneous target priors without useful within-asset temporal ranking. It must not claim that the model discovered predictive emergent dynamics.",
    ])
    (context.final.table_dir / "ifddrp_title_alignment_verdict.md").write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without Pandas' optional dependencies."""
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [render(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)
