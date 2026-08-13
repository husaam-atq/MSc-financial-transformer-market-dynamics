from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from market_dynamics.evaluation.prior_neutral import (
    centre_logits_within_group,
    contiguous_positive_events,
    fit_group_priors,
    nonoverlapping_rows,
    remove_prior_logit,
)
from market_dynamics.experiments.run_phase6 import _within_group_auc


def test_prior_residual_removes_static_group_logit() -> None:
    frame = pd.DataFrame(
        {
            "asset": ["A"] * 4 + ["B"] * 4,
            "y_true": [0, 0, 0, 1, 0, 1, 1, 1],
        }
    )
    priors = fit_group_priors(frame, "asset", smoothing=0.0)
    scores = frame["asset"].map(priors).to_numpy(dtype=float)

    residual = remove_prior_logit(scores, frame["asset"], priors)

    assert np.allclose(residual, 0.5, atol=1e-5)


def test_centred_logits_remove_group_location_not_order() -> None:
    reference = pd.DataFrame(
        {"asset": ["A", "A", "B", "B"], "probability": [0.1, 0.2, 0.8, 0.9]}
    )
    evaluation = reference.copy()

    centred = centre_logits_within_group(evaluation, "probability", "asset", reference)

    assert centred[1] > centred[0]
    assert centred[3] > centred[2]
    assert abs(float(centred[:2].mean()) - float(centred[2:].mean())) < 0.05


def test_nonoverlap_and_event_grouping_are_outcome_independent() -> None:
    frame = pd.DataFrame(
        {
            "asset_ticker": ["A"] * 6,
            "Date": pd.date_range("2024-01-01", periods=6),
            "y_true": [0, 1, 1, 0, 1, 0],
            "probability": [0.1, 0.7, 0.8, 0.2, 0.4, 0.1],
        }
    )

    sampled = nonoverlapping_rows(frame, stride=2, offset=0)
    events = contiguous_positive_events(frame, "probability", threshold=0.6)

    assert sampled.index.size == 3
    assert len(events) == 2
    assert events["onset_detected"].tolist() == [True, False]


def test_train_only_asset_priors_are_constant_within_asset_but_rank_between_assets() -> None:
    train = pd.DataFrame(
        {
            "asset": ["LOW"] * 10 + ["HIGH"] * 10,
            "family": ["DEFENSIVE"] * 10 + ["RISK"] * 10,
            "y_true": [0] * 9 + [1] + [0] * 2 + [1] * 8,
        }
    )
    evaluation = pd.DataFrame(
        {
            "asset": ["LOW"] * 4 + ["HIGH"] * 4,
            "family": ["DEFENSIVE"] * 4 + ["RISK"] * 4,
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    asset_priors = fit_group_priors(train, "asset", smoothing=0.0)
    family_priors = fit_group_priors(train, "family", smoothing=0.0)
    original_scores = evaluation["asset"].map(asset_priors).to_numpy(dtype=float)
    original_family_scores = evaluation["family"].map(family_priors).to_numpy(dtype=float)

    changed_evaluation = evaluation.copy()
    changed_evaluation["y_true"] = 1 - changed_evaluation["y_true"]
    changed_scores = changed_evaluation["asset"].map(asset_priors).to_numpy(dtype=float)
    changed_family_scores = changed_evaluation["family"].map(family_priors).to_numpy(dtype=float)

    np.testing.assert_array_equal(original_scores, changed_scores)
    np.testing.assert_array_equal(original_family_scores, changed_family_scores)
    assert asset_priors["LOW"] == pytest.approx(0.1)
    assert asset_priors["HIGH"] == pytest.approx(0.8)
    assert family_priors["DEFENSIVE"] == pytest.approx(0.1)
    assert family_priors["RISK"] == pytest.approx(0.8)
    assert evaluation.assign(score=original_scores).groupby("asset")["score"].nunique().eq(1).all()
    assert (
        evaluation.assign(score=original_family_scores)
        .groupby("family")["score"]
        .nunique()
        .eq(1)
        .all()
    )

    within = _within_group_auc(
        evaluation["y_true"].to_numpy(dtype=int),
        original_scores,
        evaluation["asset"].to_numpy(),
    )
    assert within["eligible_assets"] == 2
    assert within["pair_weighted_within_asset_roc_auc"] == pytest.approx(0.5)

    pooled_labels = np.asarray([0, 0, 0, 1, 0, 0, 1, 1], dtype=int)
    pooled_assets = np.asarray(["LOW"] * 4 + ["HIGH"] * 4)
    pooled_scores = pd.Series(pooled_assets).map(asset_priors).to_numpy(dtype=float)
    pooled_families = np.asarray(["DEFENSIVE"] * 4 + ["RISK"] * 4)
    pooled_family_scores = pd.Series(pooled_families).map(family_priors).to_numpy(dtype=float)
    assert roc_auc_score(pooled_labels, pooled_scores) > 0.5
    assert roc_auc_score(pooled_labels, pooled_family_scores) > 0.5
