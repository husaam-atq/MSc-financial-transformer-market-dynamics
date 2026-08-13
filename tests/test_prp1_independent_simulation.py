from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_dynamics.research.independent_shortcut_simulation import (
    IndependentSimulationConfig,
    enumerate_preregistered_design,
    extract_lag_matrix,
    fit_train_only_priors,
    old_to_recent_contrast,
    sequence_order_contrasts,
    simulate_independent_shortcut_panel,
)


def _config(**overrides: object) -> IndependentSimulationConfig:
    base = IndependentSimulationConfig(
        n_assets=8,
        n_families=2,
        sequence_length=8,
        train_periods=90,
        purge_periods=8,
        validation_periods=80,
        seed=73,
        event_onset_probability=0.08,
        min_event_duration=2,
        max_event_duration=5,
        missing_rate=0.12,
    )
    return replace(base, **overrides)


def test_default_protocol_has_exact_period_counts() -> None:
    config = IndependentSimulationConfig()
    panel = simulate_independent_shortcut_panel(config).panel
    period_counts = panel.groupby("split")["origin_time"].nunique().to_dict()

    assert config.sequence_length == 8
    assert period_counts == {"purged": 8, "train": 400, "validation": 200}
    assert len(panel) == (400 + 8 + 200) * config.n_assets


def test_simulation_is_deterministic_and_has_separate_truth() -> None:
    first = simulate_independent_shortcut_panel(_config())
    second = simulate_independent_shortcut_panel(_config())
    changed_seed = simulate_independent_shortcut_panel(_config(seed=74))

    pd.testing.assert_frame_equal(first.panel, second.panel)
    pd.testing.assert_frame_equal(first.truth, second.truth)
    pd.testing.assert_frame_equal(first.event_episodes, second.event_episodes)
    assert not first.panel["target"].equals(changed_seed.panel["target"])
    assert "target_probability" not in first.panel
    assert "target_probability" in first.truth


def test_lagged_features_and_split_do_not_cross_forecast_origin() -> None:
    data = simulate_independent_shortcut_panel(_config(forecast_horizon=3))
    panel = data.panel
    train = panel.loc[panel["split"] == "train"]
    validation = panel.loc[panel["split"] == "validation"]

    assert (panel["sequence_start_time"] <= panel["origin_time"]).all()
    assert (panel["origin_time"] < panel["target_time"]).all()
    assert train["target_time"].max() < validation["origin_time"].min()
    assert panel.groupby("split")["origin_time"].nunique().to_dict() == {
        "purged": 8,
        "train": 90,
        "validation": 80,
    }
    expected_lags = {
        f"observed_dynamic_lag_{lag}" for lag in range(data.config.sequence_length)
    }
    assert expected_lags.issubset(panel.columns)


def test_dynamic_logit_uses_order_sensitive_origin_contrast() -> None:
    config = _config(signal_noise=0.0, missing_rate=0.0)
    data = simulate_independent_shortcut_panel(config)
    latent = extract_lag_matrix(
        data.truth,
        prefix="latent_dynamic_lag_",
        sequence_length=config.sequence_length,
    )
    contrasts = sequence_order_contrasts(latent)

    np.testing.assert_allclose(
        contrasts["ordered"], data.truth["ordered_dynamic_contrast"].to_numpy()
    )
    np.testing.assert_allclose(contrasts["reversed"], -contrasts["ordered"])
    assert np.mean(
        ~np.isclose(contrasts["deterministic_permutation"], contrasts["ordered"])
    ) > 0.9
    np.testing.assert_allclose(
        data.truth["dynamic_logit_component"],
        config.dynamic_signal_strength * contrasts["ordered"],
    )


def test_old_to_recent_contrast_has_registered_direction() -> None:
    increasing = np.arange(8, dtype=np.float64)
    ordered = old_to_recent_contrast(increasing)
    reversed_contrast = old_to_recent_contrast(increasing[::-1])

    assert ordered == pytest.approx(4.0)
    assert reversed_contrast == pytest.approx(-4.0)


def test_missing_lags_are_explicit_and_never_filled() -> None:
    data = simulate_independent_shortcut_panel(_config())
    panel = data.panel
    for lag in range(data.config.sequence_length):
        missing = panel[f"missing_lag_{lag}"]
        assert panel.loc[missing, f"observed_dynamic_lag_{lag}"].isna().all()
        assert panel.loc[missing, f"observed_common_lag_{lag}"].isna().all()
        assert panel.loc[~missing, f"observed_dynamic_lag_{lag}"].notna().all()
    assert panel["sequence_has_missing"].any()
    assert panel["target"].notna().all()


def test_events_have_declared_duration_and_do_not_overlap_per_asset() -> None:
    data = simulate_independent_shortcut_panel(_config())
    episodes = data.event_episodes

    assert not episodes.empty
    assert episodes["duration"].between(2, 5).all()
    for _, group in episodes.groupby("asset_id", sort=False):
        ordered = group.sort_values("start_time")
        assert (
            ordered["start_time"].iloc[1:].to_numpy()
            > ordered["end_time"].iloc[:-1].to_numpy()
        ).all()


def test_persistence_and_dependence_controls_change_latent_structure() -> None:
    low = simulate_independent_shortcut_panel(
        _config(
            train_periods=400,
            validation_periods=200,
            persistence=0.0,
            cross_sectional_dependence=0.0,
        )
    )
    high = simulate_independent_shortcut_panel(
        _config(
            train_periods=400,
            validation_periods=200,
            persistence=0.9,
            cross_sectional_dependence=0.8,
        )
    )

    def structures(truth: pd.DataFrame) -> tuple[float, float]:
        states = truth.pivot(
            index="origin_time", columns="asset_id", values="latent_dynamic_lag_0"
        )
        lag_one = float(states.corrwith(states.shift(1)).mean())
        correlation = states.corr().to_numpy()
        off_diagonal = float(correlation[np.triu_indices_from(correlation, k=1)].mean())
        return lag_one, off_diagonal

    low_persistence, low_dependence = structures(low.truth)
    high_persistence, high_dependence = structures(high.truth)
    assert high_persistence > low_persistence + 0.6
    assert high_dependence > low_dependence + 0.35


def test_static_priors_use_training_targets_only() -> None:
    panel = simulate_independent_shortcut_panel(_config()).panel
    train = panel.loc[panel["split"] == "train"].copy()
    validation = panel.loc[panel["split"] == "validation"].copy()
    fitted = fit_train_only_priors(train, smoothing=1.0)

    changed_validation = validation.copy()
    changed_validation["target"] = 1 - changed_validation["target"]
    predictions = fitted.predict(validation, level="asset")
    changed_predictions = fitted.predict(changed_validation, level="asset")

    np.testing.assert_array_equal(predictions, changed_predictions)
    assert fitted.training_rows == len(train)
    assert np.logical_and(predictions > 0.0, predictions < 1.0).all()
    with pytest.raises(ValueError, match="training rows only"):
        fit_train_only_priors(pd.concat([train, validation.iloc[:1]], ignore_index=True))


def test_preregistered_design_enumerates_every_declared_cell() -> None:
    design = enumerate_preregistered_design(
        _config(),
        {
            "dynamic_signal_strength": [0.0, 0.8],
            "prior_heterogeneity": [0.0, 0.5, 1.0],
            "seed": [7, 42],
        },
    )

    assert len(design) == 12
    observed = {
        (cell.dynamic_signal_strength, cell.prior_heterogeneity, cell.seed) for cell in design
    }
    assert observed == {
        (dynamic, prior, seed)
        for dynamic in (0.0, 0.8)
        for prior in (0.0, 0.5, 1.0)
        for seed in (7, 42)
    }


def test_invalid_sequence_or_inadequate_purge_is_rejected() -> None:
    with pytest.raises(ValueError, match="even integer"):
        _config(sequence_length=7)
    with pytest.raises(ValueError, match="cover the forecast horizon"):
        _config(forecast_horizon=9, purge_periods=8)
    with pytest.raises(ValueError, match="family_prior_heterogeneity"):
        _config(family_prior_heterogeneity=-0.1)


def test_family_and_asset_prior_heterogeneity_are_separate_factors() -> None:
    asset_only = simulate_independent_shortcut_panel(
        _config(prior_heterogeneity=1.0, family_prior_heterogeneity=0.0)
    ).truth
    family_only = simulate_independent_shortcut_panel(
        _config(prior_heterogeneity=0.0, family_prior_heterogeneity=1.0)
    ).truth
    assert asset_only["family_static_logit"].eq(0.0).all()
    assert family_only["asset_static_logit"].eq(0.0).all()
    assert family_only["family_static_logit"].std() > 0.0
