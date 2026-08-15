from pathlib import Path

import pytest
import yaml

from market_dynamics.reporting.dissertation_figures import (
    METHODOLOGY_FIGURE_DATA,
    load_core_results,
    load_simulation_figure_data,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"


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
