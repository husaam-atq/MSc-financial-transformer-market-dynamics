from __future__ import annotations

import numpy as np

from market_dynamics.evaluation.bootstrap import bootstrap_metric_difference, moving_block_indices


def test_moving_block_bootstrap_returns_contiguous_index_sample() -> None:
    sample = moving_block_indices(30, 5, np.random.default_rng(7))
    assert sample.shape == (30,)
    assert (sample >= 0).all() and (sample < 30).all()


def test_bootstrap_difference_has_confidence_interval() -> None:
    actual = np.linspace(0.0, 1.0, 50)
    candidate = actual + 0.1
    baseline = actual + 0.3
    result = bootstrap_metric_difference(actual, candidate, baseline, lambda y, p: np.mean(abs(y - p)), iterations=100, block_length=5)
    assert result["ci_lower"] <= result["observed_difference"] <= result["ci_upper"]


def test_bootstrap_two_sided_p_is_bounded_for_degenerate_equal_models() -> None:
    actual = np.linspace(0.0, 1.0, 50)
    predictions = actual + 0.1
    result = bootstrap_metric_difference(
        actual,
        predictions,
        predictions,
        lambda y, p: np.mean(abs(y - p)),
        iterations=100,
        block_length=5,
    )
    assert result["observed_difference"] == 0.0
    assert 0.0 <= result["bootstrap_two_sided_p"] <= 1.0
    assert result["bootstrap_two_sided_p"] == 1.0
