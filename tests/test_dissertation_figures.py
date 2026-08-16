import csv
from collections import Counter
from pathlib import Path

import pytest
import yaml

from market_dynamics.reporting.dissertation_figures import (
    CHANCE_AUC,
    FAMILY_ORDER,
    METHODOLOGY_FIGURE_DATA,
    MODEL_WINDOW_ASSET_COUNT,
    MODEL_WINDOW_ENDPOINT_COUNTS,
    MODEL_WINDOW_ENDPOINT_SHA256,
    MODEL_WINDOW_EXCLUSIONS,
    MODEL_WINDOW_FAMILY_COUNTS,
    PREVALENCE_EQUALITY_LINE,
    load_appendix_cross_model_results,
    load_asset_prevalence,
    load_core_results,
    load_family_prevalence,
    load_identity_order_results,
    load_simulation_figure_data,
    load_window_endpoint_prevalence,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
WINDOW_ENDPOINTS = (
    ROOT
    / "src"
    / "market_dynamics"
    / "reporting"
    / "data"
    / "final_model_window_prevalence.csv"
)


def test_methodology_figure_matches_frozen_phase6_configuration() -> None:
    config = yaml.safe_load((ROOT / "configs" / "phase6_config.yaml").read_text(encoding="utf-8"))[
        "phase6"
    ]
    figure = METHODOLOGY_FIGURE_DATA

    assert figure.lookback == config["lookback"] == 60
    assert figure.purge_dates == config["corrected_purge"] == 18
    assert figure.asset_embedding_dim == config["model"]["asset_embedding_dim"] == 12
    assert figure.encoder_layers == config["model"]["num_layers"] == 2
    assert figure.numerical_channels == figure.market_channels + figure.context_channels == 34
    assert figure.conditioned_channels == figure.numerical_channels + figure.asset_embedding_dim == 46
    assert figure.parameters == 272_449
    assert figure.forecast_horizon == 10
    assert figure.embargo_dates == 1
    assert figure.audited_boundary_crossings == 0


def test_core_results_figure_uses_exact_frozen_evidence() -> None:
    results = load_core_results(
        TABLES / "ifddrp_identity_dynamic_information_decomposition.csv",
        TABLES / "prp1_fixed_cross_model_results.csv",
    )

    assert [row.key for row in results] == [
        "static_asset_prior",
        "mlp",
        "transformer",
        "transformer_no_asset_id",
    ]
    assert [row.pooled_auc for row in results] == pytest.approx(
        [0.823905856, 0.796477661, 0.789813558, 0.715476655],
        abs=1e-9,
    )
    assert [row.within_asset_auc for row in results] == pytest.approx(
        [0.5, 0.556966922, 0.491638470, 0.472570086],
        abs=1e-9,
    )
    assert results[-1].label == "Transformer\n(no asset ID)"


def test_simulation_figure_uses_registered_core_slices() -> None:
    data = load_simulation_figure_data(
        TABLES / "prp1_study_a_independent_simulation_results.csv"
    )

    assert data.prior_heterogeneity == (0.0, 0.75, 1.5, 2.5)
    assert data.dynamic_signal == (0.0, 0.5, 1.0, 1.5)
    assert data.static_prior_pooled_auc == pytest.approx(
        [0.499117623, 0.691917133, 0.817072657, 0.899138348],
        abs=1e-9,
    )
    assert data.no_signal_within_asset_auc == pytest.approx(
        [0.496633961, 0.496796273, 0.504159486, 0.503337654],
        abs=1e-9,
    )
    assert data.dynamic_within_asset_auc == pytest.approx(
        [0.496633961, 0.611119410, 0.715264369, 0.787102523],
        abs=1e-9,
    )
    assert data.reversal_auc_loss == pytest.approx(
        [-0.002614808, 0.213771947, 0.425534926, 0.571245572],
        abs=1e-9,
    )


def test_appendix_cross_model_figure_uses_exact_frozen_evidence() -> None:
    results = load_appendix_cross_model_results(
        TABLES / "ifddrp_identity_dynamic_information_decomposition.csv",
        TABLES / "prp1_fixed_cross_model_results.csv",
    )

    assert CHANCE_AUC == 0.5
    assert [row.key for row in results] == [
        "static_asset_prior",
        "mlp",
        "transformer",
        "tcn",
        "lstm",
        "flattened_logistic",
    ]
    assert [row.pooled_auc for row in results] == pytest.approx(
        [0.823905856, 0.796477661, 0.789813558, 0.775881237, 0.692869596, 0.576219972],
        abs=1e-9,
    )
    assert [row.within_asset_auc for row in results] == pytest.approx(
        [0.5, 0.556966922, 0.491638470, 0.547453353, 0.518343737, 0.512171082],
        abs=1e-9,
    )


def test_appendix_identity_order_deltas_use_full_precision_sources() -> None:
    data = load_identity_order_results(
        TABLES / "ifddrp_identity_dynamic_information_decomposition.csv",
        TABLES / "phase6_identity_swap_results.csv",
        TABLES / "phase6_temporal_order_destruction.csv",
    )

    assert data.baseline_auc == pytest.approx(0.7898135576855014, abs=1e-12)
    assert [row.key for row in data.interventions] == [
        "no_asset_id",
        "cyclic_asset_id_swap",
        "reverse_order",
        "permuted_order",
        "circular_shift",
    ]
    assert [row.auc for row in data.interventions] == pytest.approx(
        [0.715476655, 0.682927551, 0.791262639, 0.788585027, 0.790321220],
        abs=1e-9,
    )
    assert [row.auc_change for row in data.interventions] == pytest.approx(
        [-0.074336902, -0.106886007, 0.001449082, -0.001228531, 0.000507662],
        abs=1e-9,
    )
    assert data.interventions[0].label == "No asset ID"


def test_model_window_endpoint_resource_matches_frozen_evaluation_population() -> None:
    rows = load_window_endpoint_prevalence(WINDOW_ENDPOINTS)

    assert len(rows) == 3 * MODEL_WINDOW_ASSET_COUNT == 237
    assets_by_split = {
        split: {row.ticker for row in rows if row.split == split}
        for split in ("train", "validation", "test")
    }
    assert assets_by_split["train"] == assets_by_split["validation"] == assets_by_split["test"]
    assert "UNI-USD" not in assets_by_split["test"]
    assert Counter(row.family for row in rows if row.split == "test") == dict(
        MODEL_WINDOW_FAMILY_COUNTS
    )
    assert {
        split: sum(row.endpoints for row in rows if row.split == split)
        for split in ("train", "validation", "test")
    } == dict(MODEL_WINDOW_ENDPOINT_COUNTS)
    assert all(0.0 <= row.prevalence <= 1.0 for row in rows)

    with (TABLES / "ifddrp_within_asset_pair_audit.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        audit_rows = [
            row for row in csv.DictReader(handle) if row["split"] in {"train", "validation"}
        ]
    expected = {
        (row["split"], int(row["asset_id"])): (
            row["asset_ticker"],
            row["family"],
            int(row["endpoints"]),
            int(row["positives"]),
        )
        for row in audit_rows
    }
    actual = {
        (row.split, row.asset_id): (row.ticker, row.family, row.endpoints, row.positives)
        for row in rows
        if row.split in {"train", "validation"}
    }
    assert actual == expected

    with (TABLES / "prp1_fixed_cross_model_execution_manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        execution_rows = list(csv.DictReader(handle))
    manifest_row = next(
        row
        for row in execution_rows
        if row["model"] == "mlp"
        and row["identity_variant"] == "asset_conditioned"
        and row["seed"] == "7"
    )
    assert manifest_row["endpoint_sha256"] == MODEL_WINDOW_ENDPOINT_SHA256
    assert int(manifest_row["train_windows"]) == dict(MODEL_WINDOW_ENDPOINT_COUNTS)["train"]
    assert int(manifest_row["validation_windows"]) == dict(MODEL_WINDOW_ENDPOINT_COUNTS)[
        "validation"
    ]
    assert int(manifest_row["test_windows"]) == dict(MODEL_WINDOW_ENDPOINT_COUNTS)["test"]


def test_appendix_family_prevalence_uses_model_window_endpoints() -> None:
    data = load_family_prevalence(WINDOW_ENDPOINTS)

    assert tuple(row.family for row in data) == FAMILY_ORDER
    assert len(data) == 6
    assert sum(row.n_assets for row in data) == MODEL_WINDOW_ASSET_COUNT
    assert all(0.0 <= value <= 1.0 for row in data for value in (row.train, row.validation, row.test))
    lookup = {row.family: row for row in data}
    assert lookup["Bonds"].validation == pytest.approx(55 / 2_618, abs=1e-12)
    assert lookup["Crypto"].validation == pytest.approx(2_254 / 4_548, abs=1e-12)
    assert lookup["Crypto"].n_assets == 12


def test_appendix_asset_prevalence_uses_same_endpoint_population_as_a3() -> None:
    endpoint_rows = load_window_endpoint_prevalence(WINDOW_ENDPOINTS)
    data = load_asset_prevalence(WINDOW_ENDPOINTS)
    repeated = load_asset_prevalence(WINDOW_ENDPOINTS)
    expected_tickers = {row.ticker for row in endpoint_rows if row.split == "test"}
    endpoint_lookup = {(row.split, row.ticker): row for row in endpoint_rows}

    assert data == repeated
    assert data.source_track == "final_model_window_endpoints"
    assert data.excluded == MODEL_WINDOW_EXCLUSIONS
    assert [item.ticker for item in data.excluded] == ["UNI-USD"]
    assert "60 sessions" in data.excluded[0].reason
    assert len(data.points) == MODEL_WINDOW_ASSET_COUNT
    assert {row.ticker for row in data.points} == expected_tickers
    assert Counter(row.family for row in data.points) == dict(MODEL_WINDOW_FAMILY_COUNTS)
    assert all(0.0 <= row.train <= 1.0 and 0.0 <= row.test <= 1.0 for row in data.points)
    assert all(
        row.train == endpoint_lookup[("train", row.ticker)].prevalence
        and row.test == endpoint_lookup[("test", row.ticker)].prevalence
        and row.train_n_obs == endpoint_lookup[("train", row.ticker)].endpoints
        and row.test_n_obs == endpoint_lookup[("test", row.ticker)].endpoints
        for row in data.points
    )
    assert PREVALENCE_EQUALITY_LINE == ((0.0, 0.0), (1.0, 1.0))
