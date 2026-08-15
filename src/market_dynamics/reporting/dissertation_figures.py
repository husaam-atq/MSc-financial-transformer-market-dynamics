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


@dataclass(frozen=True)
class InterventionResult:
    """One identity or temporal-order intervention relative to the Transformer."""

    key: str
    label: str
    category: str
    auc: float
    auc_change: float


@dataclass(frozen=True)
class InterventionFigureData:
    """Frozen baseline and intervention values displayed in Figure A2."""

    baseline_auc: float
    interventions: tuple[InterventionResult, ...]


@dataclass(frozen=True)
class FamilyPrevalence:
    """Corrected target prevalence for one daily-panel asset family."""

    family: str
    n_assets: int
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class AssetPrevalence:
    """Corrected train/test target prevalence for one daily-panel asset."""

    ticker: str
    family: str
    train: float
    test: float
    train_n_obs: int
    test_n_obs: int


@dataclass(frozen=True)
class ExcludedAsset:
    """Asset omitted from Figure A4 because a required public summary is missing."""

    ticker: str
    reason: str


@dataclass(frozen=True)
class AssetPrevalenceFigureData:
    """Daily-panel points and any explicit exclusions displayed in Figure A4."""

    points: tuple[AssetPrevalence, ...]
    excluded: tuple[ExcludedAsset, ...]
    source_track: str = "corrected_daily_phase6"


METHODOLOGY_FIGURE_DATA = MethodologyFigureData()
CHANCE_AUC = 0.5
PREVALENCE_EQUALITY_LINE = ((0.0, 0.0), (1.0, 1.0))
PREVALENCE_SPLITS = ("train", "validation", "test")
FAMILY_ORDER = ("Equities", "Bonds", "Commodities", "FX", "Crypto", "Real assets")


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


def load_appendix_cross_model_results(
    identity_decomposition_path: Path,
    cross_model_results_path: Path,
) -> tuple[CoreResult, ...]:
    """Load Figure A1 values from the canonical final cross-model evidence."""
    identity_rows = _read_rows(identity_decomposition_path)
    cross_model_rows = _read_rows(cross_model_results_path)
    static_prior = _require_row(identity_rows, component="training_only_asset_prior")
    identity_transformer = _require_row(identity_rows, component="conditioned_transformer")

    model_specs = (
        ("mlp", "mlp", "MLP"),
        ("transformer", "transformer_encoder", "Transformer"),
        ("tcn", "tcn", "TCN"),
        ("lstm", "lstm", "LSTM"),
        ("flattened_logistic", "flattened_logistic", "Flattened\nlogistic"),
    )
    learned_results: list[CoreResult] = []
    for key, model, label in model_specs:
        row = _require_row(
            cross_model_rows,
            model=model,
            identity_variant="asset_conditioned",
            seed="ensemble",
            aggregation="pooled",
            status="completed",
        )
        learned_results.append(
            CoreResult(
                key=key,
                label=label,
                pooled_auc=_as_probability(row, "roc_auc"),
                within_asset_auc=_as_probability(
                    row,
                    "pair_weighted_within_asset_roc_auc",
                ),
            )
        )

    transformer = next(result for result in learned_results if result.key == "transformer")
    if not _close(
        transformer.pooled_auc,
        _as_probability(identity_transformer, "pooled_roc_auc"),
    ) or not _close(
        transformer.within_asset_auc,
        _as_probability(identity_transformer, "within_asset_roc_auc"),
    ):
        raise ValueError("Canonical Transformer evidence tables disagree")

    return (
        CoreResult(
            key="static_asset_prior",
            label="Static asset\nprior",
            pooled_auc=_as_probability(static_prior, "pooled_roc_auc"),
            within_asset_auc=_as_probability(static_prior, "within_asset_roc_auc"),
        ),
        *learned_results,
    )


def load_identity_order_results(
    identity_decomposition_path: Path,
    identity_swap_path: Path,
    temporal_order_path: Path,
) -> InterventionFigureData:
    """Load Figure A2 scores and compute full-precision changes from baseline."""
    identity_rows = _read_rows(identity_decomposition_path)
    swap_rows = _read_rows(identity_swap_path)
    order_rows = _read_rows(temporal_order_path)

    baseline = _as_probability(
        _require_row(identity_rows, component="conditioned_transformer"),
        "pooled_roc_auc",
    )
    no_asset_id = _as_probability(
        _require_row(identity_rows, component="no_explicit_asset_id_transformer"),
        "pooled_roc_auc",
    )
    cyclic_swap = _as_probability(
        _require_row(swap_rows, diagnostic="cyclic_asset_id_swap"),
        "roc_auc",
    )
    scores = (
        ("no_asset_id", "No asset ID", "identity", no_asset_id),
        ("cyclic_asset_id_swap", "Cyclic asset-ID swap", "identity", cyclic_swap),
        (
            "reverse_order",
            "Reverse order",
            "order",
            _as_probability(
                _require_row(
                    order_rows,
                    model_variant="corrected_asset_conditioned",
                    perturbation="reverse",
                ),
                "roc_auc",
            ),
        ),
        (
            "permuted_order",
            "Permute order",
            "order",
            _as_probability(
                _require_row(
                    order_rows,
                    model_variant="corrected_asset_conditioned",
                    perturbation="deterministic_permutation",
                ),
                "roc_auc",
            ),
        ),
        (
            "circular_shift",
            "Circular shift",
            "order",
            _as_probability(
                _require_row(
                    order_rows,
                    model_variant="corrected_asset_conditioned",
                    perturbation="circular_shift",
                ),
                "roc_auc",
            ),
        ),
    )
    interventions = tuple(
        InterventionResult(
            key=key,
            label=label,
            category=category,
            auc=auc,
            auc_change=auc - baseline,
        )
        for key, label, category, auc in scores
    )
    return InterventionFigureData(baseline_auc=baseline, interventions=interventions)


def load_family_prevalence(path: Path) -> tuple[FamilyPrevalence, ...]:
    """Load corrected daily-panel family prevalence for Figure A3."""
    rows = [row for row in _read_rows(path) if row.get("group_type") == "family"]
    actual_families = {row["group"] for row in rows}
    actual_splits = {row["split"] for row in rows}
    if actual_families != set(FAMILY_ORDER):
        raise ValueError(f"Unexpected family prevalence groups: {sorted(actual_families)}")
    if actual_splits != set(PREVALENCE_SPLITS):
        raise ValueError(f"Unexpected family prevalence splits: {sorted(actual_splits)}")

    results: list[FamilyPrevalence] = []
    for family in FAMILY_ORDER:
        split_rows = {
            split: _require_row(rows, split=split, group_type="family", group=family)
            for split in PREVALENCE_SPLITS
        }
        asset_counts = {int(row["n_assets"]) for row in split_rows.values()}
        if len(asset_counts) != 1:
            raise ValueError(f"Asset count changes across splits for {family}")
        results.append(
            FamilyPrevalence(
                family=family,
                n_assets=asset_counts.pop(),
                train=_as_probability(split_rows["train"], "prevalence"),
                validation=_as_probability(split_rows["validation"], "prevalence"),
                test=_as_probability(split_rows["test"], "prevalence"),
            )
        )
    return tuple(results)


def load_asset_prevalence(
    asset_prevalence_path: Path,
    daily_family_map_path: Path,
) -> AssetPrevalenceFigureData:
    """Load corrected daily-panel train/test prevalence points for Figure A4."""
    rows = [
        row
        for row in _read_rows(asset_prevalence_path)
        if row.get("group_type") == "asset_ticker"
    ]
    unexpected_splits = {row["split"] for row in rows} - set(PREVALENCE_SPLITS)
    if unexpected_splits:
        raise ValueError(f"Unexpected asset prevalence splits: {sorted(unexpected_splits)}")

    family_rows = _read_rows(daily_family_map_path)
    family_map: dict[str, str] = {}
    for row in family_rows:
        ticker = row["Ticker"]
        if ticker in family_map:
            raise ValueError(f"Duplicate daily-panel family mapping for {ticker}")
        family_map[ticker] = row["family"]

    table_tickers = {row["group"] for row in rows}
    all_tickers = sorted(table_tickers | set(family_map))
    points: list[AssetPrevalence] = []
    excluded: list[ExcludedAsset] = []
    for ticker in all_tickers:
        family = family_map.get(ticker)
        if family is None:
            excluded.append(ExcludedAsset(ticker=ticker, reason="missing daily-panel family map"))
            continue
        if family not in FAMILY_ORDER:
            excluded.append(ExcludedAsset(ticker=ticker, reason=f"unknown family: {family}"))
            continue
        split_rows = [row for row in rows if row["group"] == ticker]
        available = {row["split"] for row in split_rows}
        missing = {"train", "test"} - available
        if missing:
            reason = f"missing required split: {', '.join(sorted(missing))}"
            excluded.append(ExcludedAsset(ticker=ticker, reason=reason))
            continue
        train = _require_row(split_rows, split="train", group_type="asset_ticker", group=ticker)
        test = _require_row(split_rows, split="test", group_type="asset_ticker", group=ticker)
        train_n_obs = int(train["n_obs"])
        test_n_obs = int(test["n_obs"])
        if train_n_obs <= 0 or test_n_obs <= 0:
            excluded.append(ExcludedAsset(ticker=ticker, reason="non-positive split support"))
            continue
        points.append(
            AssetPrevalence(
                ticker=ticker,
                family=family,
                train=_as_probability(train, "prevalence"),
                test=_as_probability(test, "prevalence"),
                train_n_obs=train_n_obs,
                test_n_obs=test_n_obs,
            )
        )

    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    points.sort(key=lambda point: (family_rank[point.family], point.ticker))
    return AssetPrevalenceFigureData(points=tuple(points), excluded=tuple(excluded))


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


def _close(left: float, right: float) -> bool:
    return abs(left - right) < 1e-12
