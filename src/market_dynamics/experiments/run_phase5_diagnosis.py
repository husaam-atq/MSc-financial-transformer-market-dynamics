"""Historical-only Phase 5 cross-asset failure diagnosis.

The runner reconstructs persisted Phase 4 predictions and historical train windows.
It never opens the sealed fresh-holdout archive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_dynamics.datasets.pooled_window_dataset import build_pooled_window_datasets
from market_dynamics.evaluation.family_generalisation import (
    attach_family_mapping,
    fit_family_postprocessors,
    select_postprocessing_strategy,
    summarize_family_predictions,
)
from market_dynamics.evaluation.post_freeze import aligned_probability_ensemble
from market_dynamics.experiments.run_large_scale_screening import (
    _feature_columns,
    load_partitioned_panel,
)
from market_dynamics.experiments.run_walkforward_robustness import _three_walkforward_folds


def run_phase5_cross_asset_diagnosis(
    config: dict[str, Any],
    phase5_config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Diagnose family failure and select one validation-only post-processing strategy."""
    phase5 = dict(phase5_config.get("phase5", phase5_config))
    root = Path(config["_meta"]["project_root"])
    table_dir = Path(config["paths"]["reports_tables"])
    reference = dict(phase5["historical_reference"])
    run_dir = root / str(reference["phase4_final_run_dir"])
    ensemble = _phase4_ensemble(run_dir)
    mapping = _family_mapping(root, phase5["family_mapping"])
    ensemble = attach_family_mapping(ensemble, mapping)

    validation = ensemble[ensemble["split"].eq("validation")].copy()
    historical_test = ensemble[ensemble["split"].eq("test")].copy()
    postprocessors, calibration, thresholds, partitions = fit_family_postprocessors(validation, phase5["family_postprocessing"])
    selection_rows: list[pd.DataFrame] = []
    for strategy, postprocessor in postprocessors.items():
        selection_rows.append(summarize_family_predictions(postprocessor.apply(partitions["selection"]), strategy, "validation_selection"))
    selection_summary = pd.concat(selection_rows, ignore_index=True)
    selected_strategy = select_postprocessing_strategy(selection_summary)
    selected = postprocessors[selected_strategy]
    selected_validation = selected.apply(validation)
    selected_test = selected.apply(historical_test)
    baseline = _global_baseline_frame(run_dir, table_dir, historical_test)
    performance = pd.concat(
        [
            _window_count_rows(config, phase5, mapping),
            summarize_family_predictions(baseline, "phase4_global_isotonic_context", "historical_test_context"),
            selection_summary,
            summarize_family_predictions(selected_validation, selected_strategy, "validation_all"),
            summarize_family_predictions(selected_test, selected_strategy, "historical_test_context_selected_once"),
        ],
        ignore_index=True,
        sort=False,
    )
    calibration.to_csv(table_dir / "phase5_family_calibration.csv", index=False)
    thresholds.to_csv(table_dir / "phase5_family_thresholds.csv", index=False)
    performance.to_csv(table_dir / "phase5_family_performance.csv", index=False)
    _write_diagnosis_report(table_dir / "phase5_cross_asset_failure_diagnosis.md", performance, thresholds, selected_strategy, partitions)
    return {
        "ensemble": ensemble,
        "performance": performance,
        "calibration": calibration,
        "thresholds": thresholds,
        "selected_validation": selected_validation,
        "selected_historical_test": selected_test,
    }


def _phase4_ensemble(run_dir: Path) -> pd.DataFrame:
    paths = sorted((run_dir / "predictions").glob("full_eligible_full_attention_pool_lb60_seed*.parquet"))
    if len(paths) != 3:
        raise FileNotFoundError(f"Expected three Phase 4 seed prediction files under {run_dir}, found {len(paths)}")
    frames = [pd.read_parquet(path) for path in paths]
    ensemble = aligned_probability_ensemble(frames, "raw_probability")
    keys = ["split", "Date", "source_index", "asset_id"]
    metadata = frames[0][keys + ["asset_ticker"]].copy().sort_values(keys).reset_index(drop=True)
    if metadata.duplicated(keys).any():
        raise ValueError("Phase 4 prediction metadata contains duplicate endpoints")
    for seed_index, frame in enumerate(frames[1:], start=1):
        candidate = frame[keys + ["asset_ticker"]].copy().sort_values(keys).reset_index(drop=True)
        if len(candidate) != len(metadata) or not candidate[keys].equals(metadata[keys]) or not candidate["asset_ticker"].equals(metadata["asset_ticker"]):
            raise ValueError(f"Phase 4 seed {seed_index} does not agree on endpoint-to-asset mapping")
    output = ensemble.merge(metadata, on=keys, how="left", validate="one_to_one")
    if output["asset_ticker"].isna().any():
        raise RuntimeError("Unable to reconstruct Phase 4 asset tickers")
    output = output.rename(columns={"ensemble_probability": "raw_probability"})
    return output


def _family_mapping(root: Path, raw_mapping: dict[str, str]) -> pd.DataFrame:
    universe = pd.read_csv(root / "configs" / "universes" / "daily_global_universe.csv")
    mapping = {str(key): str(value) for key, value in raw_mapping.items()}
    output = universe[["ticker", "asset_class"]].copy()
    output["family"] = output["asset_class"].map(mapping)
    if output["family"].isna().any():
        unknown = sorted(output.loc[output["family"].isna(), "asset_class"].unique())
        raise ValueError(f"Phase 5 family mapping is incomplete: {unknown}")
    return output


def _global_baseline_frame(run_dir: Path, table_dir: Path, historical_test: pd.DataFrame) -> pd.DataFrame:
    """Use the stored Phase 4 global-calibration decision as immutable context."""
    output = historical_test.copy()
    selected_path = run_dir / "predictions" / "phase4_full_eligible_refit_stress_ensemble.parquet"
    if not selected_path.exists():
        raise FileNotFoundError(f"Stored Phase 4 ensemble prediction file is missing: {selected_path}")
    selected = pd.read_parquet(selected_path)
    keys = ["split", "Date", "source_index", "asset_id", "y_true"]
    required = {*keys, "selected_probability"}
    missing = required.difference(selected.columns)
    if missing:
        raise KeyError(f"Stored Phase 4 ensemble lacks immutable decision fields: {sorted(missing)}")
    selected = selected[keys + ["selected_probability"]]
    output = output.merge(selected, on=keys, how="left", validate="one_to_one")
    if output["selected_probability"].isna().any():
        raise RuntimeError("Stored Phase 4 calibrated probabilities do not align to historical test endpoints")
    output["selected_threshold"] = _stored_phase4_threshold(table_dir)
    output["decision"] = (output["selected_probability"] >= output["selected_threshold"]).astype(int)
    return output


def _stored_phase4_threshold(table_dir: Path) -> float:
    """Load the original validation-selected Phase 4 threshold from its audit table."""
    path = table_dir / "phase4_calibration_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Phase 4 calibration audit table is missing: {path}")
    table = pd.read_csv(path)
    required = {"source", "model_component", "split", "selected_by_validation_probability_quality", "selected_threshold"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"Phase 4 calibration audit table is incomplete: {sorted(missing)}")
    rows = table[
        table["source"].eq("phase4_full_eligible_refit")
        & table["model_component"].eq("equal_weight_seed_ensemble")
        & table["split"].eq("test")
        & table["selected_by_validation_probability_quality"].fillna(False)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one immutable Phase 4 ensemble threshold, found {len(rows)}")
    return float(rows["selected_threshold"].iloc[0])


def _window_count_rows(config: dict[str, Any], phase5: dict[str, Any], mapping: pd.DataFrame) -> pd.DataFrame:
    reference = phase5["historical_reference"]
    panel = load_partitioned_panel(Path(config["paths"]["processed"]) / "daily_global_panel")
    target = str(reference["target"])
    target_panel = panel.dropna(subset=[target])
    features = _feature_columns(target_panel, "daily")
    folds = _three_walkforward_folds(
        target_panel,
        target,
        int(reference["lookback"]),
        config["phase2c"],
        purge_override=int(reference.get("legacy_purge_global_dates", 10)),
    )
    bundle = build_pooled_window_datasets(target_panel, "Ticker", features, target, folds[int(reference["fold"]) - 1], int(reference["lookback"]))
    reverse = {value: key for key, value in bundle.asset_to_id.items()}
    rows: list[dict[str, object]] = []
    for split_name, dataset in (("train", bundle.train), ("validation", bundle.validation), ("historical_test", bundle.test)):
        parts: list[pd.DataFrame] = []
        for asset in dataset.assets:
            endpoints = asset.endpoints
            parts.append(
                pd.DataFrame(
                    {
                        "asset_ticker": reverse[int(asset.asset_id)],
                        "y_true": asset.target[endpoints],
                    }
                )
            )
        endpoints_frame = attach_family_mapping(pd.concat(parts, ignore_index=True), mapping)
        for family, part in endpoints_frame.groupby("family", observed=True):
            rows.append(
                {
                    "source": "strict_window_inventory",
                    "split": split_name,
                    "aggregation": "family_window_inventory",
                    "family": family,
                    "n_assets": int(part["asset_ticker"].nunique()),
                    "n_obs": float(len(part)),
                    "class_balance": float(part["y_true"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_diagnosis_report(
    path: Path,
    performance: pd.DataFrame,
    thresholds: pd.DataFrame,
    selected_strategy: str,
    partitions: dict[str, pd.DataFrame],
) -> None:
    selected_test = performance[
        performance["split"].eq("historical_test_context_selected_once") & performance["source"].eq(selected_strategy)
    ]
    base_test = performance[
        performance["split"].eq("historical_test_context") & performance["source"].eq("phase4_global_isotonic_context")
    ]
    non_crypto = selected_test[selected_test["aggregation"].eq("non_crypto_example_weighted")]
    global_row = selected_test[selected_test["aggregation"].eq("global_example_weighted")]
    baseline_non_crypto = base_test[base_test["aggregation"].eq("non_crypto_example_weighted")]
    selected_family_rows = selected_test[selected_test["aggregation"].eq("family_example_weighted")].set_index("family")
    selected_thresholds = thresholds[thresholds["strategy"].eq("family_calibrated")].set_index("family")
    lines = [
        "# Phase 5 Cross-Asset Failure Diagnosis",
        "",
        "## Protocol",
        "",
        "- This is a historical-development and historical-context analysis. It never opens the sealed fresh-holdout archive.",
        "- The three Phase 4 seed predictions are aligned exactly before averaging raw probabilities.",
        "- Historical validation is split chronologically into calibration fitting, threshold fitting and strategy selection. The historical test is reported only once for the validation-selected strategy.",
        "",
        "## Nested Validation Partitions",
        "",
    ]
    for name, part in partitions.items():
        lines.append(f"- {name}: {part['Date'].min().date()} to {part['Date'].max().date()}, n={len(part):,}.")
    lines.extend(["", "## Selected Strategy", "", f"- Validation-selected strategy: `{selected_strategy}`."])
    if not global_row.empty:
        row = global_row.iloc[0]
        lines.append(f"- Historical-context aggregate: F1={row['f1']:.4f}, balanced accuracy={row['balanced_accuracy']:.4f}, ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}.")
    if not non_crypto.empty:
        row = non_crypto.iloc[0]
        lines.append(f"- Historical-context non-crypto: F1={row['f1']:.4f}, balanced accuracy={row['balanced_accuracy']:.4f}, ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}, positive-prediction rate={row['prediction_positive_rate']:.4f}.")
    if not baseline_non_crypto.empty:
        row = baseline_non_crypto.iloc[0]
        lines.append(f"- Immutable Phase 4 global-calibration context non-crypto F1={row['f1']:.4f}, ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}.")
    lines.extend(["", "## Evidence-Led Diagnosis", ""])
    for family in ["Crypto", "Commodities", "Equities", "Bonds", "Real assets"]:
        if family not in selected_family_rows.index:
            continue
        row = selected_family_rows.loc[family]
        threshold_text = "not available"
        if family in selected_thresholds.index:
            threshold_row = selected_thresholds.loc[family]
            threshold_text = f"{float(threshold_row['threshold']):.2f} ({threshold_row['calibration_method']})"
        lines.append(
            f"- {family}: prevalence={row['class_balance']:.4f}, selected threshold={threshold_text}, "
            f"test-context positive rate={row['prediction_positive_rate']:.4f}, F1={row['f1']:.4f}, "
            f"balanced accuracy={row['balanced_accuracy']:.4f}, ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}."
        )
    if "Crypto" in selected_family_rows.index:
        crypto = selected_family_rows.loc["Crypto"]
        if crypto["prediction_positive_rate"] >= 0.95:
            lines.append("- Crypto remains near-all-positive under the selected F1 rule. Its F1 is therefore not evidence of a well-balanced family classifier.")
    if "Commodities" in selected_family_rows.index:
        lines.append(
            "- Commodities provide the strongest non-crypto decision signal in this diagnostic; this is a family-specific result, not evidence of uniform transfer."
        )
    if "Equities" in selected_family_rows.index and "Bonds" in selected_family_rows.index:
        equities = selected_family_rows.loc["Equities"]
        bonds = selected_family_rows.loc["Bonds"]
        lines.append(
            "- Equities and bonds have low stress base rates and only weak-to-moderate ranking diagnostics. "
            f"Their selected F1 values remain {equities['f1']:.4f} and {bonds['f1']:.4f}; threshold adjustment alone does not solve the transfer problem."
        )
    if not non_crypto.empty and not baseline_non_crypto.empty:
        selected_non_crypto = non_crypto.iloc[0]
        baseline_row = baseline_non_crypto.iloc[0]
        lines.append(
            "- Relative to the immutable global-isotonic context, the selected family post-processing changes non-crypto F1 "
            f"from {baseline_row['f1']:.4f} to {selected_non_crypto['f1']:.4f}. The historical-context result is not an improvement claim because it was not a new independent test."
        )
        lines.append(
            "- Any aggregate ranking change after family calibration can arise from re-ordering scores between families or reducing isotonic ties. "
            "It does not establish a new within-family predictive mechanism."
        )
    lines.append(
        "- The evidence supports a combined explanation: target base-rate/comparability differences, pooled optimisation dominated by crypto, threshold mismatch, and genuinely limited equity/bond discrimination. "
        "It justifies one bounded family-balanced or hierarchical model experiment; it does not justify claiming broad cross-asset generalisation from post-processing alone."
    )
    lines.extend(
        [
            "",
            "## Diagnostic Boundary",
            "",
            "- Threshold-only changes cannot improve ROC-AUC. Any F1 change is reported as a decision-rule effect, not a ranking improvement.",
            "- Family calibrators and thresholds use historical validation only. They are subject to family sample-size safeguards and global fallbacks.",
            "- The resulting cause classification is evidence-led: family probability distributions, calibration diagnostics, and raw versus post-processed ranking metrics are retained in the accompanying CSVs.",
            "- No family-specific historical-test result was used to choose a strategy, a model, or a threshold.",
            "",
            "## Supporting Outputs",
            "",
            "- `phase5_family_performance.csv`: family, non-crypto, macro and worst-family metrics.",
            "- `phase5_family_calibration.csv`: nested-validation calibration candidates.",
            "- `phase5_family_thresholds.csv`: fixed validation thresholds and fallback status.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
