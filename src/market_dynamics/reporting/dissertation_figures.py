"""Load the frozen data used by the final dissertation figures."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodologyFigureData:
    """Frozen methodology values displayed in Figure 1."""

    lookback: int = 60
    numerical_channels: int = 34
    market_channels: int = 27
    context_channels: int = 7
    asset_embedding_dim: int = 12
    conditioned_channels: int = 46
    encoder_layers: int = 2
    parameters: int = 272_449
    forecast_horizon: int = 10
    purge_dates: int = 18
    embargo_dates: int = 1
    audited_boundary_crossings: int = 0


@dataclass(frozen=True)
class CoreResult:
    """One frozen pooled-versus-within-asset comparison row."""

    key: str
    label: str
    pooled_auc: float
    within_asset_auc: float


@dataclass(frozen=True)
class SimulationFigureData:
    """Frozen core simulation series displayed in Figure 3."""

    prior_heterogeneity: tuple[float, ...]
    static_prior_pooled_auc: tuple[float, ...]
    no_signal_within_asset_auc: tuple[float, ...]
    dynamic_signal: tuple[float, ...]
    dynamic_within_asset_auc: tuple[float, ...]
    reversal_auc_loss: tuple[float, ...]


METHODOLOGY_FIGURE_DATA = MethodologyFigureData()


def load_core_results(
    identity_decomposition_path: Path,
    cross_model_results_path: Path,
) -> tuple[CoreResult, ...]:
    """Load Figure 2 values from the final frozen evidence tables."""
    identity_rows = _read_rows(identity_decomposition_path)
    cross_model_rows = _read_rows(cross_model_results_path)

    static_prior = _require_row(
        identity_rows,
        component="training_only_asset_prior",
    )
    transformer = _require_row(
        identity_rows,
        component="conditioned_transformer",
    )
    no_asset_id = _require_row(
        identity_rows,
        component="no_explicit_asset_id_transformer",
    )
    mlp = _require_row(
        cross_model_rows,
        model="mlp",
        identity_variant="asset_conditioned",
        seed="ensemble",
        aggregation="pooled",
        status="completed",
    )

    return (
        CoreResult(
            key="static_asset_prior",
            label="Static asset\nprior",
            pooled_auc=_as_probability(static_prior, "pooled_roc_auc"),
            within_asset_auc=_as_probability(static_prior, "within_asset_roc_auc"),
        ),
        CoreResult(
            key="mlp",
            label="MLP",
            pooled_auc=_as_probability(mlp, "roc_auc"),
            within_asset_auc=_as_probability(mlp, "pair_weighted_within_asset_roc_auc"),
        ),
        CoreResult(
            key="transformer",
            label="Transformer",
            pooled_auc=_as_probability(transformer, "pooled_roc_auc"),
            within_asset_auc=_as_probability(transformer, "within_asset_roc_auc"),
        ),
        CoreResult(
            key="transformer_no_asset_id",
            label="Transformer\n(no asset ID)",
            pooled_auc=_as_probability(no_asset_id, "pooled_roc_auc"),
            within_asset_auc=_as_probability(no_asset_id, "within_asset_roc_auc"),
        ),
    )


def load_simulation_figure_data(path: Path) -> SimulationFigureData:
    """Load the two registered core slices displayed in Figure 3."""
    rows = _read_rows(path)
    no_signal = sorted(
        (
            row
            for row in rows
            if row["scenario"] == "core"
            and _equals(row, "dynamic_signal", 0.0)
            and _equals(row, "persistence", 0.7)
        ),
        key=lambda row: float(row["prior_heterogeneity"]),
    )
    no_heterogeneity = sorted(
        (
            row
            for row in rows
            if row["scenario"] == "core"
            and _equals(row, "prior_heterogeneity", 0.0)
            and _equals(row, "persistence", 0.7)
        ),
        key=lambda row: float(row["dynamic_signal"]),
    )
    if len(no_signal) != 4 or len(no_heterogeneity) != 4:
        raise ValueError("Frozen simulation figure requires four points in each core slice")

    return SimulationFigureData(
        prior_heterogeneity=tuple(float(row["prior_heterogeneity"]) for row in no_signal),
        static_prior_pooled_auc=tuple(
            _as_probability(row, "asset_prior_pooled_roc_auc_mean") for row in no_signal
        ),
        no_signal_within_asset_auc=tuple(
            _as_probability(
                row,
                "pooled_classifier_pair_weighted_within_asset_roc_auc_mean",
            )
            for row in no_signal
        ),
        dynamic_signal=tuple(float(row["dynamic_signal"]) for row in no_heterogeneity),
        dynamic_within_asset_auc=tuple(
            _as_probability(
                row,
                "pooled_classifier_pair_weighted_within_asset_roc_auc_mean",
            )
            for row in no_heterogeneity
        ),
        reversal_auc_loss=tuple(
            _as_auc_difference(row, "reversal_auc_drop_mean") for row in no_heterogeneity
        ),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(column, "")) == expected for column, expected in criteria.items())
    ]
    if len(matches) != 1:
        rendered = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
        raise ValueError(f"Expected one frozen evidence row for {rendered}; found {len(matches)}")
    return matches[0]


def _as_probability(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{column} must be in [0, 1], received {value}")
    return value


def _as_auc_difference(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"{column} must be in [-1, 1], received {value}")
    return value


def _equals(row: dict[str, str], column: str, expected: float) -> bool:
    return abs(float(row[column]) - expected) < 1e-9
