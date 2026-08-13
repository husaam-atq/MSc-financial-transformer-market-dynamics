"""Leakage-safe cross-asset market-structure features and targets.

The rolling geometry functions in this module expect a synchronous, complete
matrix of close-to-close returns. Missing observations are rejected rather than
filled because imputation would change the cross-sectional dependence structure.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.covariance import LedoitWolf


def rolling_market_structure_features(
    returns: pd.DataFrame,
    *,
    window: int = 120,
    change_lag: int = 20,
    families: Mapping[str, str] | pd.Series | None = None,
) -> pd.DataFrame:
    """Compute rolling correlation, spectral, and optional fixed-family features.

    Each row at timestamp ``t`` is estimated only from rows ``t-window+1``
    through ``t``. Ledoit-Wolf covariance shrinkage is converted to a
    correlation matrix before the spectral statistics are calculated.

    When ``families`` is supplied, communities are fixed ex ante by those labels;
    this function does not estimate or optimise community assignments.
    """
    matrix = _validate_returns_matrix(returns, minimum_rows=window)
    if window < 3:
        raise ValueError("window must be at least 3")
    if change_lag < 1:
        raise ValueError("change_lag must be positive")

    family_labels = _align_families(matrix.columns, families)
    base_columns = [
        "normalized_effective_rank",
        "normalized_spectral_entropy",
        "largest_eigenvalue_share",
        "average_off_diagonal_correlation",
        "correlation_turnover",
        "mst_mean_edge_distance",
    ]
    if family_labels is not None:
        base_columns.extend(["fixed_family_weighted_modularity", "cross_family_positive_mixing"])

    result = pd.DataFrame(np.nan, index=matrix.index, columns=base_columns, dtype=float)
    correlation_history: dict[int, np.ndarray] = {}
    values = matrix.to_numpy(dtype=float)
    n_assets = matrix.shape[1]

    for end in range(window - 1, len(matrix)):
        sample = values[end - window + 1 : end + 1]
        covariance = LedoitWolf().fit(sample).covariance_
        correlation = _covariance_to_correlation(covariance)
        eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
        total = float(eigenvalues.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"Invalid correlation spectrum at {matrix.index[end]!s}")

        probabilities = eigenvalues[eigenvalues > 0.0] / total
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        row = matrix.index[end]
        result.at[row, "normalized_effective_rank"] = np.exp(entropy) / n_assets
        result.at[row, "normalized_spectral_entropy"] = entropy / np.log(n_assets)
        result.at[row, "largest_eigenvalue_share"] = eigenvalues[-1] / total
        result.at[row, "average_off_diagonal_correlation"] = _average_off_diagonal(
            correlation
        )
        result.at[row, "mst_mean_edge_distance"] = _mst_mean_edge_distance(correlation)

        comparison_end = end - change_lag
        if comparison_end in correlation_history:
            result.at[row, "correlation_turnover"] = (
                np.linalg.norm(correlation - correlation_history[comparison_end], ord="fro")
                / n_assets
            )
        correlation_history[end] = correlation

        if family_labels is not None:
            result.at[row, "fixed_family_weighted_modularity"] = _fixed_modularity(
                correlation, family_labels
            )
            result.at[row, "cross_family_positive_mixing"] = _cross_family_mixing(
                correlation, family_labels
            )

    for column in base_columns:
        result[f"{column}_change_{change_lag}d"] = result[column].diff(change_lag)
    return result


def fit_downside_thresholds(
    training_returns: pd.DataFrame,
    *,
    quantile: float = 0.10,
) -> pd.Series:
    """Fit per-asset downside thresholds using training rows only."""
    matrix = _validate_returns_matrix(training_returns, minimum_rows=2)
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must lie strictly between 0 and 0.5")
    thresholds = matrix.quantile(quantile, axis=0)
    thresholds.name = f"training_downside_q{quantile:g}"
    return thresholds.astype(float)


def downside_breadth_features(
    returns: pd.DataFrame,
    thresholds: pd.Series | Mapping[str, float],
    *,
    change_lag: int = 20,
) -> pd.DataFrame:
    """Apply fixed downside thresholds to contemporaneous cross-asset returns.

    ``downside_breadth`` is the fraction of assets below their training-fitted
    threshold. ``downside_coexceedance`` is the fraction of distinct asset pairs
    for which both assets exceed their downside thresholds.
    """
    matrix = _validate_returns_matrix(returns, minimum_rows=1)
    if change_lag < 1:
        raise ValueError("change_lag must be positive")
    threshold_series = pd.Series(thresholds, dtype=float).reindex(matrix.columns)
    if threshold_series.isna().any() or not np.isfinite(threshold_series).all():
        missing = threshold_series.index[threshold_series.isna()].tolist()
        raise ValueError(f"Finite downside thresholds are required for every asset: {missing}")

    exceedances = matrix.lt(threshold_series, axis="columns")
    counts = exceedances.sum(axis=1).astype(float)
    n_assets = matrix.shape[1]
    pair_count = n_assets * (n_assets - 1) / 2.0
    result = pd.DataFrame(index=matrix.index)
    result["downside_breadth"] = counts / n_assets
    result["downside_coexceedance"] = counts * (counts - 1.0) / 2.0 / pair_count
    result[f"downside_breadth_change_{change_lag}d"] = result["downside_breadth"].diff(
        change_lag
    )
    result[f"downside_coexceedance_change_{change_lag}d"] = result[
        "downside_coexceedance"
    ].diff(change_lag)
    return result


def equal_family_market_return(
    returns: pd.DataFrame,
    families: Mapping[str, str] | pd.Series,
    *,
    name: str = "equal_family_market_return",
) -> pd.Series:
    """Aggregate asset returns by equally weighting families, then families."""
    matrix = _validate_returns_matrix(returns, minimum_rows=1)
    labels = _align_families(matrix.columns, families)
    if labels is None:  # pragma: no cover - guarded by the required argument
        raise ValueError("families are required")

    family_returns = pd.DataFrame(index=matrix.index)
    for family in pd.unique(labels):
        members = matrix.columns[labels == family]
        family_returns[str(family)] = matrix.loc[:, members].mean(axis=1)
    market_return = family_returns.mean(axis=1)
    market_return.name = name
    return market_return


def future_max_loss_target(
    market_returns: pd.Series,
    *,
    horizon: int = 10,
    returns_are_log: bool = True,
    name: str | None = None,
) -> pd.Series:
    """Calculate maximum origin-to-path loss over ``t+1`` through ``t+horizon``.

    The returned loss is non-negative and expressed as a simple-return fraction.
    The final ``horizon`` rows are ``NaN`` because their complete future path is
    unavailable. No current-row return is included in its own target.
    """
    series = _validate_return_series(market_returns)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if len(series) <= horizon:
        raise ValueError("market_returns must contain more rows than horizon")
    if not returns_are_log and (series <= -1.0).any():
        raise ValueError("Simple returns must be greater than -1")

    values = series.to_numpy(dtype=float)
    target = np.full(len(series), np.nan, dtype=float)
    for origin in range(len(series) - horizon):
        future = values[origin + 1 : origin + horizon + 1]
        if returns_are_log:
            path_returns = np.expm1(np.cumsum(future))
        else:
            path_returns = np.cumprod(1.0 + future) - 1.0
        target[origin] = max(0.0, -float(np.min(path_returns)))

    target_name = name or f"target_future_max_loss_{horizon}d"
    return pd.Series(target, index=series.index, name=target_name)


def fit_upper_tail_threshold(training_target: pd.Series, *, quantile: float = 0.90) -> float:
    """Fit an upper-tail event threshold from a training target only."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0.5 and 1")
    values = pd.to_numeric(training_target, errors="coerce").dropna()
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("training_target must contain finite observations")
    return float(values.quantile(quantile))


def upper_tail_event(target: pd.Series, threshold: float, *, name: str | None = None) -> pd.Series:
    """Create a strict upper-tail event label while preserving unavailable rows."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    numeric = pd.to_numeric(target, errors="coerce")
    event = numeric.gt(threshold).astype(float).where(numeric.notna())
    event.name = name or f"{target.name or 'target'}_event"
    return event


def _validate_returns_matrix(returns: pd.DataFrame, *, minimum_rows: int) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must have a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("returns index must be strictly increasing and unique")
    if returns.shape[1] < 2:
        raise ValueError("returns must contain at least two assets")
    if len(returns) < minimum_rows:
        raise ValueError(f"returns requires at least {minimum_rows} rows")
    if returns.columns.has_duplicates:
        raise ValueError("returns columns must be unique")
    try:
        matrix = returns.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("returns must contain only numeric values") from exc
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("returns must be synchronous, finite, and contain no missing values")
    return matrix


def _validate_return_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("market_returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("market_returns must have a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or returns.index.has_duplicates:
        raise ValueError("market_returns index must be strictly increasing and unique")
    series = pd.to_numeric(returns, errors="coerce").astype(float)
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError("market_returns must be finite and contain no missing values")
    return series


def _align_families(
    columns: pd.Index,
    families: Mapping[str, str] | pd.Series | None,
) -> np.ndarray | None:
    if families is None:
        return None
    labels = pd.Series(families, dtype="object").reindex(columns)
    if labels.isna().any():
        missing = labels.index[labels.isna()].tolist()
        raise ValueError(f"A fixed family label is required for every asset: {missing}")
    if labels.nunique() < 2:
        raise ValueError("At least two fixed families are required")
    return labels.to_numpy(dtype=object)


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    variances = np.diag(covariance)
    if not np.isfinite(variances).all() or np.any(variances <= 0.0):
        raise ValueError("Ledoit-Wolf covariance contains non-positive variances")
    scale = np.sqrt(variances)
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _average_off_diagonal(correlation: np.ndarray) -> float:
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    return float(upper.mean())


def _positive_adjacency(correlation: np.ndarray) -> np.ndarray:
    adjacency = np.maximum(correlation, 0.0).copy()
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _fixed_modularity(correlation: np.ndarray, families: np.ndarray) -> float:
    adjacency = _positive_adjacency(correlation)
    total_weight_twice = float(adjacency.sum())
    if total_weight_twice <= 0.0:
        return np.nan
    strengths = adjacency.sum(axis=1)
    null = np.outer(strengths, strengths) / total_weight_twice
    same_family = families[:, None] == families[None, :]
    return float(np.sum((adjacency - null) * same_family) / total_weight_twice)


def _cross_family_mixing(correlation: np.ndarray, families: np.ndarray) -> float:
    adjacency = _positive_adjacency(correlation)
    upper = np.triu(np.ones_like(adjacency, dtype=bool), k=1)
    weights = adjacency[upper]
    total = float(weights.sum())
    if total <= 0.0:
        return np.nan
    cross_family = (families[:, None] != families[None, :])[upper]
    return float(weights[cross_family].sum() / total)


def _mst_mean_edge_distance(correlation: np.ndarray) -> float:
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - correlation)))
    off_diagonal_zero = (distance == 0.0) & ~np.eye(len(distance), dtype=bool)
    distance[off_diagonal_zero] = np.finfo(float).eps
    tree = minimum_spanning_tree(distance)
    edges = tree.data
    if len(edges) != len(correlation) - 1:
        raise ValueError("Unable to construct a complete minimum spanning tree")
    return float(edges.mean())
