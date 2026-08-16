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
    """Model-window endpoint prevalence for one daily-panel asset family."""

    family: str
    n_assets: int
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class AssetPrevalence:
    """Model-window endpoint prevalence for one daily-panel asset."""

    ticker: str
    family: str
    train: float
    test: float
    train_n_obs: int
    test_n_obs: int


@dataclass(frozen=True)
class ExcludedAsset:
    """Configured asset excluded from the final model-window population."""

    ticker: str
    reason: str


@dataclass(frozen=True)
class AssetPrevalenceFigureData:
    """Final evaluation points and explicit configured-universe exclusions."""

    points: tuple[AssetPrevalence, ...]
    excluded: tuple[ExcludedAsset, ...]
    source_track: str = "final_model_window_endpoints"


@dataclass(frozen=True)
class WindowEndpointPrevalence:
    """Target-label support for one asset and final model split."""

    split: str
    asset_id: int
    ticker: str
    family: str
    endpoints: int
    positives: int
    prevalence: float


METHODOLOGY_FIGURE_DATA = MethodologyFigureData()
CHANCE_AUC = 0.5
PREVALENCE_EQUALITY_LINE = ((0.0, 0.0), (1.0, 1.0))
PREVALENCE_SPLITS = ("train", "validation", "test")
FAMILY_ORDER = ("Equities", "Bonds", "Commodities", "FX", "Crypto", "Real assets")
MODEL_WINDOW_LOOKBACK = 60
MODEL_WINDOW_ASSET_COUNT = 79
MODEL_WINDOW_ENDPOINT_SHA256 = (
    "47527d25a7a7f1a293c14ee7f6aaa254be3163d14a2c6f2149b687357d9c4a60"
)
MODEL_WINDOW_ENDPOINT_COUNTS = (
    ("train", 245_055),
    ("validation", 20_494),
    ("test", 21_514),
)
MODEL_WINDOW_FAMILY_COUNTS = (
    ("Equities", 39),
    ("Bonds", 11),
    ("Commodities", 8),
    ("FX", 6),
    ("Crypto", 12),
    ("Real assets", 3),
)
MODEL_WINDOW_EXCLUSIONS = (
    ExcludedAsset(
        ticker="UNI-USD",
        reason=(
            "24 valid target-labelled test rows after the ten-session future horizon; "
            "fewer than the 60 sessions required to form a test window"
        ),
    ),
)


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


def load_window_endpoint_prevalence(path: Path) -> tuple[WindowEndpointPrevalence, ...]:
    """Load and validate the exact endpoint-label population used for evaluation."""
    raw_rows = _read_rows(path)
    required_columns = {
        "split",
        "asset_id",
        "asset_ticker",
        "family",
        "endpoints",
        "positives",
        "prevalence",
        "lookback",
        "endpoint_sha256",
    }
    if not raw_rows or set(raw_rows[0]) != required_columns:
        actual_columns = set(raw_rows[0]) if raw_rows else set()
        raise ValueError(f"Unexpected model-window prevalence schema: {sorted(actual_columns)}")

    rows: list[WindowEndpointPrevalence] = []
    for raw in raw_rows:
        split = raw["split"]
        family = raw["family"]
        if split not in PREVALENCE_SPLITS:
            raise ValueError(f"Unexpected model-window split: {split}")
        if family not in FAMILY_ORDER:
            raise ValueError(f"Unexpected model-window family: {family}")
        if int(raw["lookback"]) != MODEL_WINDOW_LOOKBACK:
            raise ValueError(f"Unexpected lookback for {raw['asset_ticker']}: {raw['lookback']}")
        if raw["endpoint_sha256"] != MODEL_WINDOW_ENDPOINT_SHA256:
            raise ValueError(f"Endpoint hash mismatch for {raw['asset_ticker']} ({split})")

        endpoints = int(raw["endpoints"])
        positives = int(raw["positives"])
        prevalence = _as_probability(raw, "prevalence")
        if endpoints <= 0 or not 0 <= positives <= endpoints:
            raise ValueError(f"Invalid endpoint support for {raw['asset_ticker']} ({split})")
        if not _close(prevalence, positives / endpoints):
            raise ValueError(f"Prevalence disagrees with endpoint labels for {raw['asset_ticker']}")
        rows.append(
            WindowEndpointPrevalence(
                split=split,
                asset_id=int(raw["asset_id"]),
                ticker=raw["asset_ticker"],
                family=family,
                endpoints=endpoints,
                positives=positives,
                prevalence=prevalence,
            )
        )

    keys = {(row.split, row.asset_id, row.ticker) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate model-window prevalence rows")

    identities_by_split = {
        split: {
            (row.asset_id, row.ticker, row.family)
            for row in rows
            if row.split == split
        }
        for split in PREVALENCE_SPLITS
    }
    expected_identities = identities_by_split["train"]
    if any(identities != expected_identities for identities in identities_by_split.values()):
        raise ValueError("Model-window asset population changes across splits")
    if len(expected_identities) != MODEL_WINDOW_ASSET_COUNT:
        raise ValueError(f"Expected {MODEL_WINDOW_ASSET_COUNT} model assets; found {len(expected_identities)}")
    if (
        len({asset_id for asset_id, _, _ in expected_identities}) != MODEL_WINDOW_ASSET_COUNT
        or len({ticker for _, ticker, _ in expected_identities}) != MODEL_WINDOW_ASSET_COUNT
    ):
        raise ValueError("Model-window asset IDs and tickers must map one-to-one")
    if any(ticker == "UNI-USD" for _, ticker, _ in expected_identities):
        raise ValueError("UNI-USD must not appear in the final model-window population")

    expected_endpoint_counts = dict(MODEL_WINDOW_ENDPOINT_COUNTS)
    actual_endpoint_counts = {
        split: sum(row.endpoints for row in rows if row.split == split)
        for split in PREVALENCE_SPLITS
    }
    if actual_endpoint_counts != expected_endpoint_counts:
        raise ValueError(f"Unexpected model-window endpoint totals: {actual_endpoint_counts}")

    actual_family_counts = {
        family: sum(identity[2] == family for identity in expected_identities)
        for family in FAMILY_ORDER
    }
    if actual_family_counts != dict(MODEL_WINDOW_FAMILY_COUNTS):
        raise ValueError(f"Unexpected model-window family composition: {actual_family_counts}")

    split_rank = {split: index for index, split in enumerate(PREVALENCE_SPLITS)}
    return tuple(sorted(rows, key=lambda row: (split_rank[row.split], row.asset_id)))


def load_family_prevalence(path: Path) -> tuple[FamilyPrevalence, ...]:
    """Aggregate Figure A3 from final model-window endpoint labels."""
    rows = load_window_endpoint_prevalence(path)
    results: list[FamilyPrevalence] = []
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row.family == family]
        prevalence_by_split = {}
        for split in PREVALENCE_SPLITS:
            split_rows = [row for row in family_rows if row.split == split]
            prevalence_by_split[split] = sum(row.positives for row in split_rows) / sum(
                row.endpoints for row in split_rows
            )
        results.append(
            FamilyPrevalence(
                family=family,
                n_assets=sum(row.split == "train" for row in family_rows),
                train=prevalence_by_split["train"],
                validation=prevalence_by_split["validation"],
                test=prevalence_by_split["test"],
            )
        )
    return tuple(results)


def load_asset_prevalence(path: Path) -> AssetPrevalenceFigureData:
    """Load Figure A4 points from the same final endpoint population as Figure A3."""
    rows = load_window_endpoint_prevalence(path)
    points: list[AssetPrevalence] = []
    for train in (row for row in rows if row.split == "train"):
        test = next(
            row
            for row in rows
            if row.split == "test" and row.asset_id == train.asset_id
        )
        points.append(
            AssetPrevalence(
                ticker=train.ticker,
                family=train.family,
                train=train.prevalence,
                test=test.prevalence,
                train_n_obs=train.endpoints,
                test_n_obs=test.endpoints,
            )
        )

    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    points.sort(key=lambda point: (family_rank[point.family], point.ticker))
    return AssetPrevalenceFigureData(
        points=tuple(points),
        excluded=MODEL_WINDOW_EXCLUSIONS,
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


def _close(left: float, right: float) -> bool:
    return abs(left - right) < 1e-12
