"""Phase 6 temporal-signal isolation and prior-neutral diagnostics."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset

from market_dynamics.datasets.pooled_window_dataset import build_pooled_window_datasets
from market_dynamics.evaluation.classification_postprocessing import validation_optimal_threshold
from market_dynamics.evaluation.family_generalisation import attach_family_mapping
from market_dynamics.evaluation.market_dynamics import (
    apply_volume_momentum_states,
    breadth_dispersion_results,
    build_family_daily,
    correlation_regime_results,
    fit_volume_momentum_thresholds,
)
from market_dynamics.evaluation.post_freeze import (
    aligned_probability_ensemble,
    binary_probability_metrics,
    fit_calibration_candidates,
)
from market_dynamics.evaluation.prior_neutral import (
    centre_logits_within_group,
    contiguous_positive_events,
    equal_group_weights,
    false_alarm_episodes,
    fit_group_priors,
    grouped_binary_metrics,
    macro_average,
    nonoverlapping_rows,
    remove_prior_logit,
)
from market_dynamics.evaluation.target_comparability import reconstruct_stress_components
from market_dynamics.experiments.run_large_scale_screening import (
    _feature_columns,
    _loaders,
    load_partitioned_panel,
)
from market_dynamics.experiments.run_phase5_diagnosis import _family_mapping
from market_dynamics.experiments.run_walkforward_robustness import _three_walkforward_folds
from market_dynamics.features.engineering import add_features
from market_dynamics.models.deep_learning import (
    AssetAgnosticModel,
    AssetConditionedModel,
    build_deep_model,
)
from market_dynamics.targets.make_targets import add_targets
from market_dynamics.training.losses import build_loss
from market_dynamics.training.sampling import dataset_targets
from market_dynamics.training.train import fit_model, predict_loader
from market_dynamics.utils.torch_utils import resolve_device, set_torch_seed

MACRO_COLUMNS = ["DFF", "DGS2", "DGS10", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2", "DTWEXBGS"]


@dataclass(frozen=True)
class Phase6Context:
    config: dict[str, Any]
    options: dict[str, Any]
    panel: pd.DataFrame
    target_panel: pd.DataFrame
    split: Any
    family_map: dict[str, str]
    run_dir: Path
    table_dir: Path
    device: torch.device


class PerturbedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Apply one fixed temporal perturbation while preserving labels and identity."""

    def __init__(self, base: Dataset[Any], transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, target, source_index, asset_id = self.base[index]
        return self.transform(features), target, source_index, asset_id


def build_context(
    config: dict[str, Any],
    phase6_config: dict[str, Any],
    run_dir: str | Path,
) -> Phase6Context:
    """Load immutable inputs and construct the corrected final fold."""
    options = dict(phase6_config.get("phase6", phase6_config))
    target = str(options["target"])
    frozen_panel = load_partitioned_panel(Path(config["paths"]["processed"]) / "daily_global_panel")
    observed = frozen_panel[frozen_panel["Close"].notna()].copy()
    panel = add_features(observed, config)
    panel = add_targets(panel, {"targets": config["phase2b"]["daily"]["targets"]})
    target_panel = panel.dropna(subset=[target]).copy()
    expected_purge = int(options["corrected_purge"])
    folds = _three_walkforward_folds(
        target_panel,
        target,
        int(options["lookback"]),
        config["phase2c"],
        purge_override=expected_purge,
    )
    split = folds[int(options["fold"]) - 1]
    if int(split.purge) != expected_purge:
        raise RuntimeError(f"Phase 6 expected corrected purge {expected_purge}, got {split.purge}")
    root = Path(config["_meta"]["project_root"])
    phase5 = config.get("phase5", {})
    family_frame = _family_mapping(root, phase5["family_mapping"])
    family_map = family_frame.set_index("ticker")["family"].astype(str).to_dict()
    active = Path(run_dir).resolve()
    for child in ["checkpoints", "predictions", "logs"]:
        (active / child).mkdir(parents=True, exist_ok=True)
    return Phase6Context(
        config=config,
        options=options,
        panel=panel,
        target_panel=target_panel,
        split=split,
        family_map=family_map,
        run_dir=active,
        table_dir=Path(config["paths"]["reports_tables"]),
        device=resolve_device(str(options.get("device", "auto"))),
    )


def run_phase6_training(context: Phase6Context) -> pd.DataFrame:
    """Train the three fixed Phase 6 variants with resumable local artifacts."""
    variants = ["legacy_target_purge18", "corrected_asset_conditioned", "no_explicit_asset_id", "no_macro"]
    rows: list[dict[str, object]] = []
    for variant in variants:
        if not bool(context.options["variants"][variant].get("enabled", False)):
            continue
        for seed in [int(value) for value in context.options["seeds"]]:
            prediction_path = context.run_dir / "predictions" / f"{variant}_seed{seed}.parquet"
            metric_path = context.run_dir / "logs" / f"{variant}_seed{seed}.json"
            if prediction_path.exists() and metric_path.exists():
                rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
                continue
            frame, row = _train_variant(context, variant, seed)
            frame.to_parquet(prediction_path, index=False)
            metric_path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
            rows.append(row)
            pd.DataFrame(rows).to_csv(context.run_dir / "phase6_training_metrics.csv", index=False)
        summary = summarize_variant(context, variant)
        rows.append({"variant": variant, "seed": "ensemble", **summary})
        pd.DataFrame(rows).to_csv(context.run_dir / "phase6_training_metrics.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(context.table_dir / "phase6_model_training_summary.csv", index=False)
    _write_run_manifest(context, metrics)
    return metrics


def run_phase6_data_audit(context: Phase6Context) -> dict[str, pd.DataFrame]:
    """Audit the observed-session repair, target comparability and macro coverage."""
    frozen = load_partitioned_panel(Path(context.config["paths"]["processed"]) / "daily_global_panel")
    target = str(context.options["target"])
    frozen_observed = frozen[frozen["Close"].notna()].copy()
    corrected = context.panel.copy()
    old = frozen_observed.reset_index()[["Date", "Ticker", target]].rename(columns={target: "frozen_target"})
    new = corrected.reset_index()[["Date", "Ticker", target]].rename(columns={target: "corrected_target"})
    comparison = old.merge(new, on=["Date", "Ticker"], how="outer", validate="one_to_one")
    overlap = comparison[["frozen_target", "corrected_target"]].notna().all(axis=1)
    disagreements = int(
        comparison.loc[overlap, "frozen_target"].astype(int).ne(comparison.loc[overlap, "corrected_target"].astype(int)).sum()
    )
    coverage = (
        comparison.groupby("Ticker", observed=True)
        .agg(
            observed_rows=("Date", "size"),
            frozen_labels=("frozen_target", "count"),
            corrected_labels=("corrected_target", "count"),
        )
        .reset_index()
    )
    coverage["restored_labels"] = coverage["corrected_labels"] - coverage["frozen_labels"]
    coverage["family"] = coverage["Ticker"].map(context.family_map).fillna("Unknown")
    coverage.to_csv(context.table_dir / "phase6_data_path_remediation.csv", index=False)

    settings = context.config["phase2b"]["daily"]["targets"]["stress"]
    components = reconstruct_stress_components(
        corrected,
        target_column=target,
        horizon=int(settings["horizon"]),
        large_negative_return=float(settings["large_negative_return"]),
        drawdown_threshold=float(settings["drawdown_threshold"]),
        volatility_spike_multiplier=float(settings["volatility_spike_multiplier"]),
        past_volatility_window=20,
    )
    components["family"] = components["asset_ticker"].map(context.family_map).fillna("Unknown")
    split_dates = {"train": context.split.train_dates, "validation": context.split.val_dates, "test": context.split.test_dates}
    asset_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    for split_name, dates in split_dates.items():
        split = components[components["Date"].isin(dates)].dropna(subset=["stored_target"]).copy()
        for asset, part in split.groupby("asset_ticker", observed=True):
            asset_rows.append(_target_summary(part, split_name, "asset_ticker", str(asset)))
        for family, part in split.groupby("family", observed=True):
            family_rows.append(_target_summary(part, split_name, "family", str(family)))
    asset_table = pd.DataFrame(asset_rows)
    family_table = pd.DataFrame(family_rows)
    asset_table.to_csv(context.table_dir / "phase6_target_prevalence_by_asset.csv", index=False)
    family_table.to_csv(context.table_dir / "phase6_target_prevalence_by_family.csv", index=False)

    alternative = _asset_relative_target_audit(components, context.split)
    alternative.to_csv(context.table_dir / "phase6_target_alternative_results.csv", index=False)
    macro = _macro_missingness(context)
    macro.to_csv(context.table_dir / "phase6_macro_missingness_audit.csv", index=False)
    _write_data_audit_report(context, coverage, asset_table, family_table, alternative, macro, disagreements)
    return {
        "coverage": coverage,
        "asset_target": asset_table,
        "family_target": family_table,
        "alternative": alternative,
        "macro": macro,
    }


def summarize_variant(context: Phase6Context, variant: str) -> dict[str, object]:
    """Fit ensemble calibration on validation only and save selected probabilities."""
    frames = [
        pd.read_parquet(context.run_dir / "predictions" / f"{variant}_seed{seed}.parquet")
        for seed in context.options["seeds"]
    ]
    ensemble = aligned_probability_ensemble(frames, "raw_probability")
    validation = ensemble[ensemble["split"].eq("validation")].copy()
    test = ensemble[ensemble["split"].eq("test")].copy()
    calibration = context.options["calibration"]
    candidates = fit_calibration_candidates(
        validation["y_true"].to_numpy(),
        validation["ensemble_probability"].to_numpy(),
        calibration["methods"],
        threshold_metric=str(calibration["threshold_metric"]),
        threshold_grid_size=int(calibration["threshold_grid_size"]),
        calibration_bins=int(calibration["bins"]),
    )
    selected = candidates[0]
    validation["selected_probability"] = selected.calibrator.predict(validation["ensemble_probability"].to_numpy())
    test["selected_probability"] = selected.calibrator.predict(test["ensemble_probability"].to_numpy())
    output = pd.concat([validation, test], ignore_index=True)
    reverse = _asset_reverse(context)
    output["asset_ticker"] = output["asset_id"].map(reverse)
    output["family"] = output["asset_ticker"].map(context.family_map).fillna("Unknown")
    output.to_parquet(context.run_dir / "predictions" / f"{variant}_ensemble.parquet", index=False)
    metrics = binary_probability_metrics(
        test["y_true"].to_numpy(),
        test["selected_probability"].to_numpy(),
        selected.threshold,
        int(calibration["bins"]),
    )
    return {
        "status": "completed",
        "calibration_method": selected.method,
        "selected_threshold": float(selected.threshold),
        "validation_threshold_f1": float(selected.validation_threshold_f1),
        "train_windows": int(_bundle(context, variant).train.__len__()),
        "validation_windows": int(len(validation)),
        "test_windows": int(len(test)),
        **metrics,
    }


def run_phase6_analysis(context: Phase6Context) -> dict[str, pd.DataFrame]:
    """Create prior decomposition, balanced metrics and event-level summaries."""
    outputs: dict[str, pd.DataFrame] = {}
    model_names = ["legacy_target_purge18", "corrected_asset_conditioned", "no_explicit_asset_id", "no_macro"]
    model_frames = {
        name: pd.read_parquet(context.run_dir / "predictions" / f"{name}_ensemble.parquet")
        for name in model_names
    }
    historical = _historical_ensemble(context)
    if "asset_ticker" not in historical:
        historical["asset_ticker"] = historical["asset_id"].map(_legacy_asset_reverse(context))
    historical["family"] = historical["asset_ticker"].map(context.family_map).fillna("Unknown")
    model_frames = {"historical_frozen": historical, **model_frames}
    decomposition_rows: list[dict[str, object]] = []
    grouped_rows: list[pd.DataFrame] = []
    event_rows: list[dict[str, object]] = []
    equal_asset_rows: list[dict[str, object]] = []
    equal_date_rows: list[dict[str, object]] = []
    equal_family_rows: list[dict[str, object]] = []
    common_date_rows: list[dict[str, object]] = []
    nonoverlap_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []

    for model_name, full in model_frames.items():
        train_labels = (
            _legacy_training_label_frame(context, purge=10)
            if model_name == "historical_frozen"
            else _legacy_training_label_frame(context, purge=18)
            if model_name == "legacy_target_purge18"
            else _training_label_frame(context)
        )
        asset_priors = fit_group_priors(train_labels, "asset_ticker", smoothing=1.0)
        family_priors = fit_group_priors(train_labels, "family", smoothing=1.0)
        validation = full[full["split"].eq("validation")].copy()
        test = full[full["split"].eq("test")].copy()
        score_sets = _score_decomposition(validation, test, asset_priors, family_priors)
        for score_name, (validation_score, test_score) in score_sets.items():
            validation_work = validation.copy()
            test_work = test.copy()
            validation_work["score"] = validation_score
            test_work["score"] = test_score
            threshold, validation_f1 = validation_optimal_threshold(
                validation_work["y_true"].to_numpy(),
                validation_work["score"].to_numpy(),
                metric="f1",
                grid_size=int(context.options["calibration"]["threshold_grid_size"]),
            )
            proper_probability = score_name in {"transformer", "static_asset_prior", "static_family_prior"}
            metrics = _metric_dict(test_work, "score", threshold, proper_probability=proper_probability)
            decomposition_rows.append(
                {
                    "model_variant": model_name,
                    "score": score_name,
                    "split": "test",
                    "threshold": threshold,
                    "validation_f1": validation_f1,
                    **metrics,
                }
            )
            asset_metrics = grouped_binary_metrics(test_work, "score", "asset_ticker", threshold)
            asset_metrics.insert(0, "score", score_name)
            asset_metrics.insert(0, "model_variant", model_name)
            grouped_rows.append(asset_metrics)
            macro = macro_average(asset_metrics, ["f1", "balanced_accuracy", "roc_auc", "pr_auc"])
            eligible = asset_metrics.dropna(subset=["roc_auc"])
            pair_weighted_auc = float(np.average(eligible["roc_auc"], weights=eligible["comparable_pairs"])) if len(eligible) else np.nan
            ranking_rows.append({"model_variant": model_name, "score": score_name, "groups_with_both_classes": int(asset_metrics["roc_auc"].notna().sum()), "pair_weighted_within_asset_roc_auc": pair_weighted_auc, **macro})
            equal_asset_rows.append({"model_variant": model_name, "score": score_name, **_metric_dict(test_work, "score", threshold, equal_group_weights(test_work, "asset_ticker"), proper_probability)})
            equal_date_rows.append({"model_variant": model_name, "score": score_name, **_metric_dict(test_work, "score", threshold, equal_group_weights(test_work, "Date"), proper_probability)})
            equal_family_rows.append({"model_variant": model_name, "score": score_name, **_metric_dict(test_work, "score", threshold, _equal_family_asset_weights(test_work), proper_probability)})
            date_coverage = test_work.groupby("Date", observed=True)["asset_ticker"].nunique()
            common_dates = date_coverage[date_coverage.eq(date_coverage.max())].index
            common = test_work[test_work["Date"].isin(common_dates)]
            common_date_rows.append({"model_variant": model_name, "score": score_name, "common_dates": len(common_dates), "assets_per_date": int(date_coverage.max()), **_metric_dict(common, "score", threshold, proper_probability=proper_probability)})
            for stride in [int(context.options["evaluation"]["nonoverlap_stride"]), 60]:
                for offset in range(stride):
                    sampled = nonoverlapping_rows(test_work, stride=stride, offset=offset)
                    if sampled["y_true"].nunique() < 2:
                        continue
                    nonoverlap_rows.append({"model_variant": model_name, "score": score_name, "stride": stride, "offset": offset, **_metric_dict(sampled, "score", threshold, proper_probability=proper_probability)})
            events = contiguous_positive_events(test_work, "score", threshold)
            asset_years = max(len(test_work) / 365.25, 1e-12)
            predicted = test_work["score"].to_numpy(dtype=float) >= threshold
            labels = test_work["y_true"].to_numpy(dtype=int)
            false_positive_windows = int(np.sum(predicted & (labels == 0)))
            event_rows.append(
                {
                    "model_variant": model_name,
                    "score": score_name,
                    "events": len(events),
                    "onset_detected_events": int(events["onset_detected"].sum()) if len(events) else 0,
                    "onset_event_recall": float(events["onset_detected"].mean()) if len(events) else np.nan,
                    "any_window_event_recall": float(events["any_window_detected"].mean()) if len(events) else np.nan,
                    "false_alarm_episodes": false_alarm_episodes(test_work, "score", threshold),
                    "false_alarm_episodes_per_100_asset_years": 100.0 * false_alarm_episodes(test_work, "score", threshold) / asset_years,
                    "false_positive_windows": false_positive_windows,
                    "false_positive_windows_per_100_asset_years": 100.0 * false_positive_windows / asset_years,
                    "alert_exposure_fraction": float(predicted.mean()),
                }
            )

    outputs["asset_prior"] = pd.DataFrame(decomposition_rows)
    outputs["within_asset"] = pd.DataFrame(ranking_rows)
    outputs["equal_asset"] = pd.DataFrame(equal_asset_rows)
    outputs["equal_date"] = pd.DataFrame(equal_date_rows)
    outputs["equal_family"] = pd.DataFrame(equal_family_rows)
    outputs["common_date"] = pd.DataFrame(common_date_rows)
    outputs["events"] = pd.DataFrame(event_rows)
    outputs["nonoverlap"] = pd.DataFrame(nonoverlap_rows)
    outputs["asset_groups"] = pd.concat(grouped_rows, ignore_index=True)
    _write_analysis_tables(context, outputs)
    return outputs


def run_temporal_destruction(context: Phase6Context, variant: str = "corrected_asset_conditioned") -> pd.DataFrame:
    """Apply three preregistered order perturbations using frozen Phase 6 weights."""
    bundle = _bundle(context, variant)
    original = pd.read_parquet(context.run_dir / "predictions" / f"{variant}_ensemble.parquet")
    validation = original[original["split"].eq("validation")]
    candidates = fit_calibration_candidates(
        validation["y_true"].to_numpy(),
        validation["ensemble_probability"].to_numpy(),
        context.options["calibration"]["methods"],
        threshold_metric="f1",
        threshold_grid_size=int(context.options["calibration"]["threshold_grid_size"]),
        calibration_bins=int(context.options["calibration"]["bins"]),
    )
    selected = candidates[0]
    rows: list[dict[str, object]] = []
    for method in context.options["perturbations"]["methods"]:
        frames: list[pd.DataFrame] = []
        transform = _temporal_transform(str(method), int(context.options["lookback"]), int(context.options["perturbations"]["seed"]))
        dataset = PerturbedDataset(bundle.test, transform)
        loader = DataLoader(dataset, batch_size=int(context.options["training"]["batch_size"]), shuffle=False, num_workers=0)
        for seed in [int(value) for value in context.options["seeds"]]:
            model = _model(context, variant, bundle)
            checkpoint = torch.load(context.run_dir / "checkpoints" / f"{variant}_seed{seed}.pt", map_location=context.device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(context.device)
            y, probability, indices = predict_loader(model, loader, "classification", context.device)
            frames.append(_prediction_frame(bundle.test, indices, y, probability, bundle.asset_to_id, "test"))
        ensemble = aligned_probability_ensemble(frames, "raw_probability")
        ensemble["score"] = selected.calibrator.predict(ensemble["ensemble_probability"].to_numpy())
        reverse = {value: key for key, value in bundle.asset_to_id.items()}
        ensemble["asset_ticker"] = ensemble["asset_id"].map(reverse)
        overall = _metric_dict(ensemble, "score", selected.threshold)
        per_asset = grouped_binary_metrics(ensemble, "score", "asset_ticker", selected.threshold)
        macro = macro_average(per_asset, ["roc_auc", "pr_auc", "balanced_accuracy", "f1"])
        rows.append({"model_variant": variant, "perturbation": method, **overall, **{f"asset_macro_{key}": value for key, value in macro.items()}})
    result = pd.DataFrame(rows)
    result.to_csv(context.table_dir / "phase6_temporal_order_destruction.csv", index=False)
    return result


def run_representation_probes(context: Phase6Context, samples_per_asset: int = 128) -> pd.DataFrame:
    """Probe frozen Phase 6 summaries for identity, family and stress information."""
    rows: list[dict[str, object]] = []
    for variant in ["corrected_asset_conditioned", "no_explicit_asset_id"]:
        bundle = _bundle(context, variant)
        representations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for split_name, dataset in [("train", bundle.train), ("validation", bundle.validation), ("test", bundle.test)]:
            selected = _balanced_dataset_indices(dataset, samples_per_asset)
            loader = DataLoader(Subset(dataset, selected), batch_size=512, shuffle=False, num_workers=0)
            seed_summaries: list[np.ndarray] = []
            raw_summary: np.ndarray | None = None
            labels: np.ndarray | None = None
            assets: np.ndarray | None = None
            for seed in [int(value) for value in context.options["seeds"]]:
                model = _model(context, variant, bundle)
                checkpoint = torch.load(context.run_dir / "checkpoints" / f"{variant}_seed{seed}.pt", map_location=context.device, weights_only=False)
                model.load_state_dict(checkpoint["state_dict"])
                model.to(context.device).eval()
                summary, current_raw, current_labels, current_assets = _encode_loader(model, loader, context.device)
                seed_summaries.append(summary)
                raw_summary = current_raw
                labels = current_labels
                assets = current_assets
            if raw_summary is None or labels is None or assets is None:
                raise RuntimeError("Representation probe produced no rows")
            representations[split_name] = (np.mean(seed_summaries, axis=0), raw_summary, labels, assets)
        for representation_name, position in [("transformer_summary", 0), ("raw_window_mean_last", 1)]:
            train_x, validation_x, test_x = [representations[name][position] for name in ["train", "validation", "test"]]
            train_y = representations["train"][2]
            validation_y = representations["validation"][2]
            test_y = representations["test"][2]
            train_assets = representations["train"][3]
            validation_assets = representations["validation"][3]
            test_assets = representations["test"][3]
            family_by_asset = _family_ids_by_asset(bundle.asset_to_id, context.family_map)
            for task, y_train, y_validation, y_test in [
                ("asset_identity", train_assets, validation_assets, test_assets),
                ("family_identity", family_by_asset[train_assets], family_by_asset[validation_assets], family_by_asset[test_assets]),
                ("stress_state", train_y, validation_y, test_y),
            ]:
                probe = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=20260712),
                )
                probe.fit(train_x, y_train)
                for split_name, x, y, split_assets in [
                    ("validation", validation_x, y_validation, validation_assets),
                    ("test", test_x, y_test, test_assets),
                ]:
                    prediction = probe.predict(x)
                    row: dict[str, object] = {
                        "model_variant": variant,
                        "representation": representation_name,
                        "probe_task": task,
                        "split": split_name,
                        "n_obs": len(y),
                        "n_classes": int(len(np.unique(y_train))),
                        "accuracy": float(accuracy_score(y, prediction)),
                        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
                    }
                    if task == "stress_state":
                        probability = probe.predict_proba(x)[:, 1]
                        row["roc_auc"] = float(roc_auc_score(y, probability))
                        row["pr_auc"] = float(average_precision_score(y, probability))
                        within = _within_group_auc(y, probability, split_assets)
                        row.update(within)
                    rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(context.table_dir / "phase6_representation_probe_results.csv", index=False)
    return result


def run_identity_swap(context: Phase6Context) -> pd.DataFrame:
    """Swap only explicit asset IDs using a fixed cyclic permutation."""
    variant = "corrected_asset_conditioned"
    bundle = _bundle(context, variant)
    original = pd.read_parquet(context.run_dir / "predictions" / f"{variant}_ensemble.parquet")
    validation = original[original["split"].eq("validation")]
    selected = fit_calibration_candidates(
        validation["y_true"].to_numpy(),
        validation["ensemble_probability"].to_numpy(),
        context.options["calibration"]["methods"],
        threshold_metric="f1",
        threshold_grid_size=int(context.options["calibration"]["threshold_grid_size"]),
        calibration_bins=int(context.options["calibration"]["bins"]),
    )[0]
    frames: list[pd.DataFrame] = []
    loader = DataLoader(bundle.test, batch_size=512, shuffle=False, num_workers=0)
    for seed in [int(value) for value in context.options["seeds"]]:
        model = _model(context, variant, bundle)
        checkpoint = torch.load(context.run_dir / "checkpoints" / f"{variant}_seed{seed}.pt", map_location=context.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(context.device).eval()
        y, probability, indices = _predict_with_swapped_ids(model, loader, context.device, len(bundle.asset_to_id))
        frames.append(_prediction_frame(bundle.test, indices, y, probability, bundle.asset_to_id, "test"))
    swapped = aligned_probability_ensemble(frames, "raw_probability")
    swapped["score"] = selected.calibrator.predict(swapped["ensemble_probability"].to_numpy())
    base = original[original["split"].eq("test")][["Date", "source_index", "asset_id", "y_true", "selected_probability"]]
    joined = base.merge(swapped, on=["Date", "source_index", "asset_id", "y_true"], validate="one_to_one")
    reverse = {value: key for key, value in bundle.asset_to_id.items()}
    joined["asset_ticker"] = joined["asset_id"].map(reverse)
    per_asset = grouped_binary_metrics(joined, "score", "asset_ticker", selected.threshold)
    rows = [
        {
            "diagnostic": "cyclic_asset_id_swap",
            "mean_absolute_probability_change": float(np.mean(np.abs(joined["score"] - joined["selected_probability"]))),
            "median_absolute_probability_change": float(np.median(np.abs(joined["score"] - joined["selected_probability"]))),
            **_metric_dict(joined, "score", selected.threshold),
            **{f"asset_macro_{key}": value for key, value in macro_average(per_asset, ["roc_auc", "pr_auc", "balanced_accuracy", "f1"]).items()},
        }
    ]
    result = pd.DataFrame(rows)
    result.to_csv(context.table_dir / "phase6_identity_swap_results.csv", index=False)
    return result


def run_market_dynamics_dependence(context: Phase6Context) -> pd.DataFrame:
    """Reanalyse four fixed market-dynamics associations with date blocks."""
    mapping = pd.DataFrame(
        {
            "ticker": list(context.family_map),
            "family": list(context.family_map.values()),
        }
    )
    universe = pd.read_csv(Path(context.config["_meta"]["project_root"]) / "configs/universes/daily_global_universe.csv")
    mapping = mapping.merge(universe[["ticker", "asset_class"]], on="ticker", how="left", validate="one_to_one")
    working = context.target_panel.reset_index().rename(columns={"Ticker": "asset_ticker"})
    working = attach_family_mapping(working, mapping[["ticker", "asset_class", "family"]])
    working = working[working["Date"].isin(context.split.train_dates.union(context.split.val_dates))].copy()
    working["split"] = np.where(working["Date"].isin(context.split.train_dates), "train", "validation")
    train = working[working["split"].eq("train")]
    thresholds = fit_volume_momentum_thresholds(train, "volume_ma_ratio_20d", "cum_return_20d", 0.25, 0.75)
    stateful = apply_volume_momentum_states(working, thresholds, "volume_ma_ratio_20d", "cum_return_20d")
    family_daily = build_family_daily(stateful, str(context.options["target"]), "cum_return_20d")
    correlation_daily, _ = correlation_regime_results(family_daily, 20, 10)
    breadth_daily, _ = breadth_dispersion_results(stateful, str(context.options["target"]), "cum_return_20d")
    pairs: list[tuple[str, str, pd.DataFrame]] = []
    for split_name, part in family_daily.groupby("split", observed=True):
        by_family = {
            family: group.set_index("Date").sort_index()
            for family, group in part.groupby("family", observed=True)
        }
        for label, source, destination, lag in [
            ("equities_to_equities_lag5", "Equities", "Equities", 5),
            ("equities_to_bonds_lag1", "Equities", "Bonds", 1),
        ]:
            pair = pd.concat(
                [
                    by_family[source]["mean_momentum"].shift(lag).rename("x"),
                    by_family[destination]["future_stress_breadth"].rename("y"),
                ],
                axis=1,
            ).dropna().reset_index()
            pairs.append((label, str(split_name), pair))
    for split_name, part in correlation_daily.groupby("split", observed=True):
        pair = part[["Date", "equity_bond_correlation", "future_stress_breadth"]].dropna().rename(
            columns={"equity_bond_correlation": "x", "future_stress_breadth": "y"}
        )
        pairs.append(("equity_bond_correlation", str(split_name), pair))
    for split_name, part in breadth_daily.groupby("split", observed=True):
        pair = part[["Date", "momentum_dispersion", "future_stress_breadth"]].dropna().rename(
            columns={"momentum_dispersion": "x", "future_stress_breadth": "y"}
        )
        pairs.append(("momentum_dispersion", str(split_name), pair))
    rows: list[dict[str, object]] = []
    for association, split_name, pair in pairs:
        for block in [10, 20, 40, 60]:
            rows.append(
                {
                    "association": association,
                    "split": split_name,
                    "n_dates": len(pair),
                    "block_observations": block,
                    **_date_block_spearman(
                        pair,
                        block,
                        int(context.options["evaluation"]["bootstrap_iterations"]),
                        int(context.options["evaluation"]["bootstrap_seed"]),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(context.table_dir / "phase6_market_dynamics_dependence_results.csv", index=False)
    _write_market_dynamics_reports(context, result)
    return result


def run_fresh_eligibility_audit(context: Phase6Context) -> pd.DataFrame:
    """Exclude fresh rows whose asset embeddings never entered frozen training."""
    path = Path(context.config["paths"]["results_runs"]) / "phase5_fresh_evaluation_20260711" / "predictions" / "phase5_fresh_holdout_predictions.parquet"
    prediction = pd.read_parquet(path)
    represented = set(_legacy_bundle(context).preprocessors)
    strict = prediction[prediction["asset_ticker"].isin(represented)].copy()
    excluded = prediction[~prediction["asset_ticker"].isin(represented)].copy()
    threshold = 0.29
    rows: list[dict[str, object]] = []
    for aggregation, groups in [
        ("overall", [("__all__", strict)]),
        ("family", list(strict.groupby("family", observed=True))),
    ]:
        for group, frame in groups:
            rows.append(
                {
                    "aggregation": aggregation,
                    "group": group,
                    "n_obs": len(frame),
                    "n_assets": int(frame["asset_ticker"].nunique()),
                    "unique_endpoint_dates": int(frame["Date"].nunique()),
                    "excluded_untrained_assets": ";".join(sorted(excluded["asset_ticker"].unique())),
                    "excluded_rows": len(excluded),
                    **binary_probability_metrics(
                        frame["y_true"].to_numpy(),
                        frame["selected_probability"].to_numpy(),
                        threshold,
                        int(context.options["calibration"]["bins"]),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(context.table_dir / "phase6_fresh_eligibility_correction.csv", index=False)
    overall = result[result["aggregation"].eq("overall")].iloc[0]
    lines = [
        "# Phase 6 Fresh Eligibility Correction",
        "",
        "The frozen fresh predictions were not recomputed or adapted. Eligibility was corrected by retaining only assets represented in all three frozen strict-window splits.",
        "",
        f"- Original rows/assets: {len(prediction)}/{prediction['asset_ticker'].nunique()}.",
        f"- Strict rows/assets: {int(overall['n_obs'])}/{int(overall['n_assets'])}.",
        f"- Excluded rows: {len(excluded)} from {excluded['asset_ticker'].nunique()} never-trained embedding IDs.",
        f"- Strict F1={overall['f1']:.4f}, balanced accuracy={overall['balanced_accuracy']:.4f}, ROC-AUC={overall['roc_auc']:.4f}, PR-AUC={overall['pr_auc']:.4f}.",
        "- The archive remains underpowered and cannot confirm or refute generalisation.",
    ]
    (context.table_dir / "phase6_fresh_eligibility_correction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _train_variant(context: Phase6Context, variant: str, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    bundle = _bundle(context, variant)
    training = deepcopy(context.options["training"])
    set_torch_seed(seed, deterministic=bool(training.get("deterministic", True)))
    model = _model(context, variant, bundle).to(context.device)
    train_loader, validation_loader, test_loader = _loaders(bundle.train, bundle.validation, bundle.test, training, seed, sampling_config=None)
    criterion = build_loss("classification", dataset_targets(train_loader.dataset), training, context.device)
    started = perf_counter()
    fitted = fit_model(model, train_loader, validation_loader, criterion, training, context.device)
    runtime = perf_counter() - started
    y_val, p_val, i_val = predict_loader(model, validation_loader, "classification", context.device)
    y_test, p_test, i_test = predict_loader(model, test_loader, "classification", context.device)
    validation = _prediction_frame(bundle.validation, i_val, y_val, p_val, bundle.asset_to_id, "validation")
    test = _prediction_frame(bundle.test, i_test, y_test, p_test, bundle.asset_to_id, "test")
    output = pd.concat([validation, test], ignore_index=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "variant": variant,
        "seed": seed,
        "features": bundle.feature_columns,
        "split": _split_manifest(context.split),
        "training": training,
        "best_epoch": fitted.best_epoch,
        "best_validation_loss": fitted.best_validation_loss,
    }
    torch.save(checkpoint, context.run_dir / "checkpoints" / f"{variant}_seed{seed}.pt")
    return output, {
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "runtime_seconds": runtime,
        "best_epoch": fitted.best_epoch + 1,
        "best_validation_loss": fitted.best_validation_loss,
        "train_windows": len(bundle.train),
        "validation_windows": len(bundle.validation),
        "test_windows": len(bundle.test),
        "feature_count": len(bundle.feature_columns),
    }


def _bundle(context: Phase6Context, variant: str) -> Any:
    if variant == "legacy_target_purge18":
        return _legacy_bundle(context, purge=18)
    features = _feature_columns(context.target_panel, "daily")
    if variant == "no_macro":
        features = [column for column in features if column not in MACRO_COLUMNS]
    return build_pooled_window_datasets(
        context.target_panel,
        "Ticker",
        features,
        str(context.options["target"]),
        context.split,
        int(context.options["lookback"]),
    )


def _model(context: Phase6Context, variant: str, bundle: Any) -> torch.nn.Module:
    model_config = deepcopy(context.options["model"])
    embedding_dim = int(model_config["asset_embedding_dim"])
    if variant == "no_explicit_asset_id":
        base = build_deep_model("transformer_encoder", len(bundle.feature_columns), int(context.options["lookback"]), {"model": model_config})
        return AssetAgnosticModel(base)
    base = build_deep_model(
        "transformer_encoder",
        len(bundle.feature_columns) + embedding_dim,
        int(context.options["lookback"]),
        {"model": model_config},
    )
    return AssetConditionedModel(base, len(bundle.asset_to_id), embedding_dim)


def _prediction_frame(dataset: Any, indices: np.ndarray, y: np.ndarray, probability: np.ndarray, asset_to_id: dict[str, int], split: str) -> pd.DataFrame:
    metadata = dataset.endpoint_metadata().set_index("source_index").loc[indices].reset_index()
    output = metadata[["Date", "source_index", "asset_id"]].copy()
    output["asset_ticker"] = output["asset_id"].map({value: key for key, value in asset_to_id.items()})
    output["y_true"] = y
    output["raw_probability"] = probability
    output["split"] = split
    return output


def _score_decomposition(validation: pd.DataFrame, test: pd.DataFrame, asset_priors: pd.Series, family_priors: pd.Series) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    validation_asset_prior = validation["asset_ticker"].map(asset_priors).to_numpy(dtype=float)
    test_asset_prior = test["asset_ticker"].map(asset_priors).to_numpy(dtype=float)
    validation_family_prior = validation["family"].map(family_priors).to_numpy(dtype=float)
    test_family_prior = test["family"].map(family_priors).to_numpy(dtype=float)
    validation_probability = validation["selected_probability"].to_numpy(dtype=float)
    test_probability = test["selected_probability"].to_numpy(dtype=float)
    validation_centred = centre_logits_within_group(validation, "selected_probability", "asset_ticker", validation)
    test_centred = centre_logits_within_group(test, "selected_probability", "asset_ticker", validation)
    validation_residual = remove_prior_logit(validation_probability, validation["asset_ticker"], asset_priors)
    test_residual = remove_prior_logit(test_probability, test["asset_ticker"], asset_priors)
    validation_family_residual = remove_prior_logit(validation_probability, validation["family"], family_priors)
    test_family_residual = remove_prior_logit(test_probability, test["family"], family_priors)
    return {
        "transformer": (validation_probability, test_probability),
        "static_asset_prior": (validation_asset_prior, test_asset_prior),
        "static_family_prior": (validation_family_prior, test_family_prior),
        "within_asset_centred": (validation_centred, test_centred),
        "asset_prior_residual": (validation_residual, test_residual),
        "family_prior_residual": (validation_family_residual, test_family_residual),
    }


def _training_label_frame(context: Phase6Context) -> pd.DataFrame:
    bundle = _bundle(context, "corrected_asset_conditioned")
    frame = bundle.train.endpoint_metadata().copy()
    frame["y_true"] = dataset_targets(bundle.train)
    frame["asset_ticker"] = frame["asset_id"].map({value: key for key, value in bundle.asset_to_id.items()})
    frame["family"] = frame["asset_ticker"].map(context.family_map).fillna("Unknown")
    return frame


def _legacy_training_label_frame(context: Phase6Context, purge: int = 10) -> pd.DataFrame:
    bundle = _legacy_bundle(context, purge=purge)
    frame = bundle.train.endpoint_metadata().copy()
    frame["y_true"] = dataset_targets(bundle.train)
    frame["asset_ticker"] = frame["asset_id"].map({value: key for key, value in bundle.asset_to_id.items()})
    frame["family"] = frame["asset_ticker"].map(context.family_map).fillna("Unknown")
    return frame


def _legacy_bundle(context: Phase6Context, purge: int = 10) -> Any:
    target = str(context.options["target"])
    panel = load_partitioned_panel(Path(context.config["paths"]["processed"]) / "daily_global_panel")
    target_panel = panel.dropna(subset=[target])
    features = _feature_columns(target_panel, "daily")
    folds = _three_walkforward_folds(
        target_panel,
        target,
        int(context.options["lookback"]),
        context.config["phase2c"],
        purge_override=int(purge),
    )
    split = folds[int(context.options["fold"]) - 1]
    return build_pooled_window_datasets(target_panel, "Ticker", features, target, split, int(context.options["lookback"]))


def _target_summary(frame: pd.DataFrame, split: str, group_type: str, group: str) -> dict[str, object]:
    ordered = frame.sort_values(["asset_ticker", "Date"])
    episodes = 0
    nonoverlap = 0
    durations: list[int] = []
    for _, asset in ordered.groupby("asset_ticker", observed=True):
        y = asset["stored_target"].to_numpy(dtype=int)
        starts = np.flatnonzero((y == 1) & np.r_[True, y[:-1] == 0])
        episodes += len(starts)
        for start in starts:
            stop = start
            while stop + 1 < len(y) and y[stop + 1] == 1:
                stop += 1
            durations.append(stop - start + 1)
        positives = np.flatnonzero(y == 1)
        last = -10_000
        for position in positives:
            if position - last >= 10:
                nonoverlap += 1
                last = int(position)
    stressed = ordered[ordered["stored_target"].eq(1.0)]
    return {
        "split": split,
        "group_type": group_type,
        "group": group,
        "n_assets": int(ordered["asset_ticker"].nunique()),
        "n_obs": len(ordered),
        "positives": int(ordered["stored_target"].sum()),
        "prevalence": float(ordered["stored_target"].mean()),
        "contiguous_label_episodes": int(episodes),
        "greedy_nonoverlap_positive_endpoints": int(nonoverlap),
        "mean_episode_windows": float(np.mean(durations)) if durations else np.nan,
        "median_stressed_future_return": float(stressed["future_return"].median()) if len(stressed) else np.nan,
        "median_stressed_future_drawdown": float(stressed["future_drawdown"].median()) if len(stressed) else np.nan,
        "median_stressed_volatility_ratio": float(stressed["volatility_ratio"].median()) if len(stressed) else np.nan,
        "volatility_only_positive_rate": float(
            (
                stressed["volatility_spike_component"].eq(1.0)
                & stressed["negative_return_component"].eq(0.0)
                & stressed["drawdown_component"].eq(0.0)
            ).mean()
        )
        if len(stressed)
        else np.nan,
    }


def _asset_relative_target_audit(components: pd.DataFrame, split: Any) -> pd.DataFrame:
    """Audit one fixed rarity-equivalent target on train/validation only."""
    working = components.dropna(subset=["future_drawdown"]).copy()
    working["maximum_loss"] = -working["future_drawdown"]
    training = working[working["Date"].isin(split.train_dates)]
    thresholds = training.groupby("asset_ticker", observed=True)["maximum_loss"].quantile(0.90)
    working["alternative_target"] = (
        working["maximum_loss"] >= working["asset_ticker"].map(thresholds)
    ).astype(float)
    rows: list[dict[str, object]] = []
    for split_name, dates in {"train": split.train_dates, "validation": split.val_dates}.items():
        subset = working[working["Date"].isin(dates)]
        for family, part in subset.groupby("family", observed=True):
            rows.append(
                {
                    "target": "asset_relative_10_session_maximum_loss_tail_q90",
                    "split": split_name,
                    "family": family,
                    "n_assets": int(part["asset_ticker"].nunique()),
                    "n_obs": len(part),
                    "positives": int(part["alternative_target"].sum()),
                    "prevalence": float(part["alternative_target"].mean()),
                    "historical_test_scored": False,
                    "fresh_scored": False,
                }
            )
    return pd.DataFrame(rows)


def _macro_missingness(context: Phase6Context) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, dates in {"train": context.split.train_dates, "validation": context.split.val_dates, "test": context.split.test_dates}.items():
        subset = context.target_panel[context.target_panel.index.isin(dates)]
        for column in MACRO_COLUMNS:
            rows.append(
                {
                    "split": split_name,
                    "feature": column,
                    "n_obs": len(subset),
                    "missing": int(subset[column].isna().sum()),
                    "missing_fraction": float(subset[column].isna().mean()),
                    "first_available": subset.loc[subset[column].notna()].index.min() if subset[column].notna().any() else pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def _write_data_audit_report(
    context: Phase6Context,
    coverage: pd.DataFrame,
    asset: pd.DataFrame,
    family: pd.DataFrame,
    alternative: pd.DataFrame,
    macro: pd.DataFrame,
    disagreements: int,
) -> None:
    validation = family[family["split"].eq("validation")]
    alt_validation = alternative[alternative["split"].eq("validation")]
    lines = [
        "# Phase 6 Target Comparability And Data-Path Audit",
        "",
        "## Deterministic Repair",
        "",
        f"- Observed OHLC rows: {int(coverage['observed_rows'].sum()):,}.",
        f"- Frozen stress labels: {int(coverage['frozen_labels'].sum()):,}.",
        f"- Correct observed-session stress labels: {int(coverage['corrected_labels'].sum()):,}.",
        f"- Restored valid labels: {int(coverage['restored_labels'].sum()):,}.",
        f"- Label disagreements on shared non-missing endpoints: {disagreements}.",
        f"- Corrected split: purge {context.split.purge}; train through {context.split.train_dates.max().date()}, validation through {context.split.val_dates.max().date()}, test through {context.split.test_dates.max().date()}.",
        "- Historical processed data and freezes were not overwritten.",
        "",
        "## Original Operational Target",
        "",
        f"- Validation family prevalence ranges from {validation['prevalence'].min():.4f} to {validation['prevalence'].max():.4f}.",
        "- The target remains heterogeneous in rarity and severity and combines adverse returns, minimum future price and volatility spikes.",
        "- Contiguous label runs are reported as label episodes, not independent economic events.",
        "",
        "## Fixed Training-Defined Alternative",
        "",
        "- The only alternative audited is the per-asset training 90th percentile of maximum loss over ten observed sessions.",
        f"- Validation family prevalence ranges from {alt_validation['prevalence'].min():.4f} to {alt_validation['prevalence'].max():.4f}.",
        "- It standardises within-asset rarity, not economic loss. Historical test and fresh labels were not scored.",
        "",
        "## Macro Availability",
        "",
        f"- Maximum training missingness across macro columns is {macro[macro['split'].eq('train')]['missing_fraction'].max():.4f}.",
        "- Current-vintage and release-timing limitations remain. The no-macro model is a sensitivity analysis, not a repaired macro claim.",
    ]
    (context.table_dir / "phase6_target_comparability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric_dict(
    frame: pd.DataFrame,
    probability_column: str,
    threshold: float,
    weights: np.ndarray | None = None,
    proper_probability: bool = True,
) -> dict[str, object]:
    y = frame["y_true"].to_numpy(dtype=int)
    p = frame[probability_column].to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "n_obs": len(frame),
        "positives": int(y.sum()),
        "prevalence": float(np.average(y, weights=weights)),
        "f1": float(f1_score(y, pred, sample_weight=weights, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weights)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=weights)),
        "brier_score": float(brier_score_loss(y, p, sample_weight=weights)) if proper_probability else np.nan,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1.0 - 1e-6), sample_weight=weights, labels=[0, 1])) if proper_probability else np.nan,
        "prediction_positive_rate": float(np.average(pred, weights=weights)),
    }


def _equal_family_asset_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give equal mass to families and then to represented assets within family."""
    family_assets = frame[["family", "asset_ticker"]].drop_duplicates().groupby("family", observed=True)["asset_ticker"].count()
    asset_rows = frame.groupby("asset_ticker", observed=True)["asset_ticker"].transform("size")
    return 1.0 / (frame["family"].map(family_assets).to_numpy(dtype=float) * asset_rows.to_numpy(dtype=float))


def _temporal_transform(method: str, lookback: int, seed: int) -> Callable[[torch.Tensor], torch.Tensor]:
    if method == "reverse":
        return lambda x: torch.flip(x, dims=(0,))
    if method == "circular_shift":
        return lambda x: torch.roll(x, shifts=lookback // 2, dims=0)
    if method == "deterministic_permutation":
        order = torch.from_numpy(np.random.default_rng(seed).permutation(lookback).astype(np.int64))
        return lambda x: x.index_select(0, order)
    raise ValueError(f"Unknown temporal perturbation: {method}")


def _balanced_dataset_indices(dataset: Any, per_asset: int) -> list[int]:
    metadata = dataset.endpoint_metadata().reset_index(names="dataset_index")
    selected: list[int] = []
    for _, part in metadata.groupby("asset_id", observed=True):
        positions = part["dataset_index"].to_numpy(dtype=int)
        if len(positions) > per_asset:
            positions = positions[np.linspace(0, len(positions) - 1, per_asset, dtype=int)]
        selected.extend(positions.tolist())
    return sorted(selected)


@torch.no_grad()
def _encode_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summaries: list[np.ndarray] = []
    raw_summaries: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    assets: list[np.ndarray] = []
    for features, target, _, asset_id in loader:
        features = features.to(device)
        asset_id_device = asset_id.to(device)
        if isinstance(model, AssetConditionedModel):
            encoded_input = model.conditioned_input(features, asset_id_device)
            _, summary, _ = model.base_model.encode(encoded_input)
        elif isinstance(model, AssetAgnosticModel):
            _, summary, _ = model.base_model.encode(features)
        else:
            raise TypeError(f"Unsupported probe model: {type(model).__name__}")
        summaries.append(summary.detach().cpu().numpy())
        raw_summaries.append(torch.cat([features.mean(dim=1), features[:, -1, :]], dim=1).cpu().numpy())
        labels.append(target.numpy())
        assets.append(asset_id.numpy())
    return np.concatenate(summaries), np.concatenate(raw_summaries), np.concatenate(labels).astype(int), np.concatenate(assets).astype(int)


def _family_ids_by_asset(asset_to_id: dict[str, int], family_map: dict[str, str]) -> np.ndarray:
    families = sorted({family_map.get(ticker, "Unknown") for ticker in asset_to_id})
    family_to_id = {family: index for index, family in enumerate(families)}
    output = np.empty(len(asset_to_id), dtype=int)
    for ticker, asset_id in asset_to_id.items():
        output[int(asset_id)] = family_to_id[family_map.get(ticker, "Unknown")]
    return output


def _within_group_auc(y: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    aucs: list[float] = []
    pairs: list[int] = []
    for group in np.unique(groups):
        selected = groups == group
        labels = y[selected]
        if len(np.unique(labels)) < 2:
            continue
        aucs.append(float(roc_auc_score(labels, probability[selected])))
        pairs.append(int(labels.sum() * (len(labels) - labels.sum())))
    return {
        "per_asset_macro_roc_auc": float(np.mean(aucs)) if aucs else np.nan,
        "pair_weighted_within_asset_roc_auc": float(np.average(aucs, weights=pairs)) if aucs else np.nan,
        "eligible_assets": len(aucs),
    }


def _date_block_spearman(pair: pd.DataFrame, block_size: int, iterations: int, seed: int) -> dict[str, float]:
    ordered = pair.sort_values("Date").reset_index(drop=True)
    estimate = float(pd.Series(ordered["x"]).corr(pd.Series(ordered["y"]), method="spearman"))
    rng = np.random.default_rng(seed + block_size)
    draws = np.empty(iterations, dtype=float)
    n = len(ordered)
    for draw in range(iterations):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
        indices = np.concatenate([(np.arange(start, start + block_size) % n) for start in starts])[:n]
        sample = ordered.iloc[indices]
        draws[draw] = (
            float(sample["x"].corr(sample["y"], method="spearman"))
            if sample["x"].nunique() > 1 and sample["y"].nunique() > 1
            else np.nan
        )
    return {
        "spearman_rho": estimate,
        "ci_lower": float(np.nanquantile(draws, 0.025)),
        "ci_upper": float(np.nanquantile(draws, 0.975)),
        "same_sign_fraction": float(np.mean(np.sign(draws) == np.sign(estimate))),
        "interval_excludes_zero": bool(float(np.nanquantile(draws, 0.025)) > 0 or float(np.nanquantile(draws, 0.975)) < 0),
    }


def _write_market_dynamics_reports(context: Phase6Context, result: pd.DataFrame) -> None:
    validation = result[result["split"].eq("validation")]
    lines = [
        "# Phase 6 Market-Dynamics Dependence Audit",
        "",
        "Four previously reported associations were re-estimated on the observed-session-repaired train/validation panel. Moving date blocks of 10, 20, 40 and 60 observations preserve serial dependence approximately; intervals remain sensitivity analyses rather than confirmatory population inference.",
        "",
    ]
    for association, part in validation.groupby("association", observed=True):
        all_splits = result[result["association"].eq(association)]
        stable = bool(all_splits["interval_excludes_zero"].all())
        lines.append(
            f"- `{association}`: validation rho={part['spearman_rho'].iloc[0]:.4f}; every train and validation block interval excludes zero={stable}; minimum validation same-sign fraction={part['same_sign_fraction'].min():.3f}."
        )
    lines.extend(
        [
            "",
            "No result is causal. Common shocks, target composition and post-hoc selection remain. A block interval that excludes zero does not establish economic importance.",
        ]
    )
    (context.table_dir / "phase6_market_dynamics_dependence_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    async_lines = [
        "# Phase 6 Asynchronous-Close Sensitivity",
        "",
        "- The two retained lead-lag tests use US-listed equity and bond ETFs, so quoted-price closes are mostly aligned to the US session. This reduces but does not eliminate nonsynchronous-information risk because international-equity ETFs embed underlying markets that close earlier.",
        "- Crypto and FX are excluded from the two focused lead-lag pairs, but they still affect broad stress and dispersion aggregates through different calendars.",
        "- Equity-to-equity and equity-to-bond lags can still arise from persistence, common macro shocks and overlapping future labels.",
        "- Daily provider timestamps do not identify the exact information cutoff of every underlying market. No causal lead-lag claim is safe.",
    ]
    (context.table_dir / "phase6_async_close_sensitivity.md").write_text("\n".join(async_lines) + "\n", encoding="utf-8")


@torch.no_grad()
def _predict_with_swapped_ids(model: torch.nn.Module, loader: DataLoader, device: torch.device, asset_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for features, target, source_index, asset_id in loader:
        swapped = (asset_id + 1) % asset_count
        logits = model(features.to(device), swapped.to(device))
        labels.append(target.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        indices.append(source_index.numpy())
    return np.concatenate(labels), np.concatenate(probabilities), np.concatenate(indices)


def _asset_reverse(context: Phase6Context) -> dict[int, str]:
    return {index: ticker for index, ticker in enumerate(sorted(context.target_panel["Ticker"].unique()))}


def _legacy_asset_reverse(context: Phase6Context) -> dict[int, str]:
    panel = load_partitioned_panel(Path(context.config["paths"]["processed"]) / "daily_global_panel")
    return {index: ticker for index, ticker in enumerate(sorted(panel["Ticker"].unique()))}


def _historical_ensemble(context: Phase6Context) -> pd.DataFrame:
    run_dir = Path(context.options["historical_reference"]["run_dir"])
    candidate = str(context.options["historical_reference"]["candidate"])
    frames = [
        pd.read_parquet(run_dir / "predictions" / f"full_eligible_{candidate}_seed{seed}.parquet")
        for seed in context.options["seeds"]
    ]
    ensemble = aligned_probability_ensemble(frames, "raw_probability")
    validation = ensemble[ensemble["split"].eq("validation")].copy()
    test = ensemble[ensemble["split"].eq("test")].copy()
    selected = fit_calibration_candidates(
        validation["y_true"].to_numpy(),
        validation["ensemble_probability"].to_numpy(),
        context.options["calibration"]["methods"],
        threshold_metric="f1",
        threshold_grid_size=int(context.options["calibration"]["threshold_grid_size"]),
        calibration_bins=int(context.options["calibration"]["bins"]),
    )[0]
    validation["selected_probability"] = selected.calibrator.predict(validation["ensemble_probability"].to_numpy())
    test["selected_probability"] = selected.calibrator.predict(test["ensemble_probability"].to_numpy())
    frozen = pd.read_parquet(run_dir / str(context.options["historical_reference"]["prediction_file"]))
    frozen_test = frozen[frozen["split"].eq("test")].sort_values(["Date", "source_index", "asset_id"])
    rebuilt_test = test.sort_values(["Date", "source_index", "asset_id"])
    if len(frozen_test) != len(rebuilt_test) or not np.allclose(
        frozen_test["selected_probability"].to_numpy(dtype=float),
        rebuilt_test["selected_probability"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Phase 6 historical ensemble does not reproduce the frozen probabilities")
    return pd.concat([validation, test], ignore_index=True)


def _split_manifest(split: Any) -> dict[str, object]:
    return {
        "purge": int(split.purge),
        "embargo": int(split.embargo),
        "train_start": str(split.train_dates.min()),
        "train_end": str(split.train_dates.max()),
        "validation_start": str(split.val_dates.min()),
        "validation_end": str(split.val_dates.max()),
        "test_start": str(split.test_dates.min()),
        "test_end": str(split.test_dates.max()),
    }


def _write_run_manifest(context: Phase6Context, metrics: pd.DataFrame) -> None:
    manifest = {
        "phase": "6",
        "evidence_status": "posthoc_historical_diagnostic",
        "device": str(context.device),
        "split": _split_manifest(context.split),
        "seeds": context.options["seeds"],
        "metrics_rows": len(metrics),
    }
    (context.run_dir / "phase6_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_analysis_tables(context: Phase6Context, outputs: dict[str, pd.DataFrame]) -> None:
    outputs["asset_prior"].to_csv(context.table_dir / "phase6_asset_prior_metrics.csv", index=False)
    outputs["within_asset"].to_csv(context.table_dir / "phase6_within_asset_ranking_results.csv", index=False)
    outputs["equal_asset"].to_csv(context.table_dir / "phase6_equal_asset_evaluation.csv", index=False)
    outputs["equal_date"].to_csv(context.table_dir / "phase6_equal_date_evaluation.csv", index=False)
    outputs["equal_family"].to_csv(context.table_dir / "phase6_equal_family_evaluation.csv", index=False)
    outputs["common_date"].to_csv(context.table_dir / "phase6_common_date_evaluation.csv", index=False)
    outputs["events"].to_csv(context.table_dir / "phase6_event_level_evaluation.csv", index=False)
    outputs["nonoverlap"].to_csv(context.table_dir / "phase6_nonoverlapping_evaluation.csv", index=False)
    no_id = outputs["asset_prior"][outputs["asset_prior"]["model_variant"].eq("no_explicit_asset_id")]
    no_id.to_csv(context.table_dir / "phase6_no_asset_id_results.csv", index=False)
    no_macro = outputs["asset_prior"][outputs["asset_prior"]["model_variant"].eq("no_macro")]
    no_macro.to_csv(context.table_dir / "phase6_no_macro_sensitivity.csv", index=False)
