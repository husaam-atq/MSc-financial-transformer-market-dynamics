"""Independent simulation of static shortcuts and ordered temporal signal.

The data-generating process is self-contained. Model-facing lagged observations
are separated from latent truth, and every target is generated from information
available at its forecast origin plus Bernoulli outcome noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PriorLevel = Literal["global", "family", "asset"]


@dataclass(frozen=True)
class IndependentSimulationConfig:
    """Parameters for one fully specified simulation cell."""

    n_assets: int = 12
    n_families: int = 3
    sequence_length: int = 8
    train_periods: int = 400
    purge_periods: int = 8
    validation_periods: int = 200
    forecast_horizon: int = 1
    seed: int = 0
    start: str = "2000-01-01"
    frequency: str = "D"
    base_prevalence: float = 0.15
    prior_heterogeneity: float = 0.8
    family_prior_heterogeneity: float = 0.0
    dynamic_signal_strength: float = 0.8
    persistence: float = 0.8
    common_shock_strength: float = 0.4
    cross_sectional_dependence: float = 0.3
    event_onset_probability: float = 0.03
    event_effect: float = 1.0
    min_event_duration: int = 2
    max_event_duration: int = 6
    signal_noise: float = 0.35
    missing_rate: float = 0.05

    @property
    def n_periods(self) -> int:
        """Total number of model-facing forecast origins."""
        return self.train_periods + self.purge_periods + self.validation_periods

    def __post_init__(self) -> None:
        """Reject parameter combinations that do not define a usable experiment."""
        if self.n_assets < 2:
            raise ValueError("n_assets must be at least 2")
        if not 1 <= self.n_families <= self.n_assets:
            raise ValueError("n_families must be between 1 and n_assets")
        if self.sequence_length < 2 or self.sequence_length % 2:
            raise ValueError("sequence_length must be an even integer of at least 2")
        if self.train_periods < 2:
            raise ValueError("train_periods must be at least 2")
        if self.validation_periods < 1:
            raise ValueError("validation_periods must be positive")
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive")
        if self.purge_periods < self.forecast_horizon:
            raise ValueError("purge_periods must cover the forecast horizon")
        if not 0.0 < self.base_prevalence < 1.0:
            raise ValueError("base_prevalence must be strictly between 0 and 1")
        if self.prior_heterogeneity < 0.0:
            raise ValueError("prior_heterogeneity cannot be negative")
        if self.family_prior_heterogeneity < 0.0:
            raise ValueError("family_prior_heterogeneity cannot be negative")
        if self.dynamic_signal_strength < 0.0:
            raise ValueError("dynamic_signal_strength cannot be negative")
        if not -0.99 < self.persistence < 0.99:
            raise ValueError("persistence must be strictly between -0.99 and 0.99")
        if not 0.0 <= self.cross_sectional_dependence < 1.0:
            raise ValueError("cross_sectional_dependence must be in [0, 1)")
        if not 0.0 <= self.event_onset_probability <= 1.0:
            raise ValueError("event_onset_probability must be in [0, 1]")
        if self.min_event_duration < 1:
            raise ValueError("min_event_duration must be positive")
        if self.max_event_duration < self.min_event_duration:
            raise ValueError("max_event_duration must not be less than min_event_duration")
        if self.signal_noise < 0.0:
            raise ValueError("signal_noise cannot be negative")
        if not 0.0 <= self.missing_rate < 1.0:
            raise ValueError("missing_rate must be in [0, 1)")


@dataclass(frozen=True)
class IndependentSimulationData:
    """Model-facing panel and separately stored data-generating truth."""

    panel: pd.DataFrame
    truth: pd.DataFrame
    event_episodes: pd.DataFrame
    config: IndependentSimulationConfig


@dataclass(frozen=True)
class FittedStaticPriors:
    """Smoothed static priors estimated exclusively from training observations."""

    global_probability: float
    family_probabilities: tuple[tuple[str, float], ...]
    asset_probabilities: tuple[tuple[str, float], ...]
    smoothing: float
    training_rows: int

    def predict(self, frame: pd.DataFrame, *, level: PriorLevel) -> NDArray[np.float64]:
        """Predict with explicit fallback from asset to family to global priors."""
        if level not in {"global", "family", "asset"}:
            raise ValueError(f"Unsupported prior level: {level}")
        required = {"asset_id", "family_id"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Prediction frame is missing columns: {missing}")

        family = dict(self.family_probabilities)
        asset = dict(self.asset_probabilities)
        if level == "global":
            return np.full(len(frame), self.global_probability, dtype=np.float64)
        if level == "family":
            return np.asarray(
                [family.get(str(value), self.global_probability) for value in frame["family_id"]],
                dtype=np.float64,
            )
        return np.asarray(
            [
                asset.get(
                    str(asset_id),
                    family.get(str(family_id), self.global_probability),
                )
                for asset_id, family_id in zip(
                    frame["asset_id"], frame["family_id"], strict=True
                )
            ],
            dtype=np.float64,
        )


def old_to_recent_contrast(sequence: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return recent-half mean minus old-half mean along the last dimension."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] < 2 or values.shape[-1] % 2:
        raise ValueError("sequence must have a positive, even final dimension")
    half = values.shape[-1] // 2
    return values[..., half:].mean(axis=-1) - values[..., :half].mean(axis=-1)


def sequence_order_contrasts(
    sequence: NDArray[np.floating],
) -> dict[str, NDArray[np.float64]]:
    """Evaluate the registered ordered, reversed and deterministic-permutation contrasts."""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] < 2 or values.shape[-1] % 2:
        raise ValueError("sequence must have a positive, even final dimension")
    indices = np.arange(values.shape[-1])
    deterministic_permutation = np.concatenate((indices[::2], indices[1::2]))
    return {
        "ordered": old_to_recent_contrast(values),
        "reversed": old_to_recent_contrast(values[..., ::-1]),
        "deterministic_permutation": old_to_recent_contrast(
            values[..., deterministic_permutation]
        ),
    }


def extract_lag_matrix(
    frame: pd.DataFrame,
    *,
    prefix: str = "observed_dynamic_lag_",
    sequence_length: int = 8,
) -> NDArray[np.float64]:
    """Extract lag columns in oldest-to-most-recent order."""
    columns = [f"{prefix}{lag}" for lag in range(sequence_length - 1, -1, -1)]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Lag frame is missing columns: {missing}")
    return frame[columns].to_numpy(dtype=np.float64)


def simulate_independent_shortcut_panel(
    config: IndependentSimulationConfig,
) -> IndependentSimulationData:
    """Generate one deterministic, ordered, multi-asset forecasting panel.

    Lag columns contain only observations from ``t-sequence_length+1`` through
    the forecast origin ``t``. The dynamic outcome component is the latent
    recent-half mean minus old-half mean over that same origin-available window.
    """
    rng = np.random.default_rng(config.seed)
    history_periods = config.sequence_length - 1
    timeline_periods = history_periods + config.n_periods + config.forecast_horizon
    times = pd.date_range(config.start, periods=timeline_periods, freq=config.frequency, tz="UTC")
    assets = np.asarray([f"asset_{index:03d}" for index in range(config.n_assets)])
    families = np.asarray([f"family_{index:02d}" for index in range(config.n_families)])
    asset_family_index = np.arange(config.n_assets) % config.n_families

    family_draw = _centred_unit_scale(rng.normal(size=config.n_families))
    asset_draw = _centred_unit_scale(rng.normal(size=config.n_assets))
    family_effect = config.family_prior_heterogeneity * family_draw
    asset_effect = config.prior_heterogeneity * asset_draw

    common_state = _ar1_series(rng, timeline_periods, config.persistence)
    latent_state = _asset_states(
        rng,
        total_periods=timeline_periods,
        n_assets=config.n_assets,
        family_index=asset_family_index,
        n_families=config.n_families,
        persistence=config.persistence,
        dependence=config.cross_sectional_dependence,
    )
    events, event_episodes = _event_process(
        rng,
        times=times,
        assets=assets,
        onset_probability=config.event_onset_probability,
        min_duration=config.min_event_duration,
        max_duration=config.max_event_duration,
    )

    origin_index = history_periods + np.arange(config.n_periods)
    sequence_offsets = np.arange(config.sequence_length - 1, -1, -1)
    sequence_index = origin_index[:, None] - sequence_offsets[None, :]
    latent_sequences = np.transpose(latent_state[sequence_index], (0, 2, 1))
    common_sequences = common_state[sequence_index]
    event_sequences = np.transpose(events[sequence_index], (0, 2, 1))
    dynamic_contrast = old_to_recent_contrast(latent_sequences)
    common_contrast = old_to_recent_contrast(common_sequences)

    base_logit = float(np.log(config.base_prevalence / (1.0 - config.base_prevalence)))
    logits = (
        base_logit
        + family_effect[asset_family_index][None, :]
        + asset_effect[None, :]
        + config.dynamic_signal_strength * dynamic_contrast
        + config.common_shock_strength * common_contrast[:, None]
        + config.event_effect * events[origin_index]
    )
    target_probability = _sigmoid(logits)
    target = rng.binomial(1, target_probability).astype(np.int8)

    observed_dynamic = latent_state + rng.normal(
        scale=config.signal_noise, size=latent_state.shape
    )
    observed_common = np.broadcast_to(common_state[:, None], latent_state.shape).copy()
    observed_common += rng.normal(scale=config.signal_noise, size=observed_common.shape)
    feature_missing = rng.random(latent_state.shape) < config.missing_rate
    observed_dynamic[feature_missing] = np.nan
    observed_common[feature_missing] = np.nan
    observed_dynamic_sequences = np.transpose(observed_dynamic[sequence_index], (0, 2, 1))
    observed_common_sequences = np.transpose(observed_common[sequence_index], (0, 2, 1))
    missing_sequences = np.transpose(feature_missing[sequence_index], (0, 2, 1))

    split = _chronological_split_labels(config)
    origin_grid = np.repeat(times[origin_index].to_numpy(), config.n_assets)
    sequence_start_grid = np.repeat(
        times[origin_index - config.sequence_length + 1].to_numpy(), config.n_assets
    )
    target_time_grid = np.repeat(
        times[origin_index + config.forecast_horizon].to_numpy(), config.n_assets
    )
    asset_grid = np.tile(assets, config.n_periods)
    family_grid = np.tile(families[asset_family_index], config.n_periods)
    panel_data: dict[str, object] = {
        "sequence_start_time": sequence_start_grid,
        "origin_time": origin_grid,
        "target_time": target_time_grid,
        "asset_id": asset_grid,
        "family_id": family_grid,
        "split": np.repeat(split, config.n_assets),
        "sequence_has_missing": missing_sequences.any(axis=-1).reshape(-1),
        "target": target.reshape(-1),
    }
    truth_data: dict[str, object] = {
        "sequence_start_time": sequence_start_grid,
        "origin_time": origin_grid,
        "target_time": target_time_grid,
        "asset_id": asset_grid,
        "family_id": family_grid,
        "family_static_logit": np.tile(family_effect[asset_family_index], config.n_periods),
        "asset_static_logit": np.tile(asset_effect, config.n_periods),
        "ordered_dynamic_contrast": dynamic_contrast.reshape(-1),
        "ordered_common_contrast": np.repeat(common_contrast, config.n_assets),
        "dynamic_logit_component": (
            config.dynamic_signal_strength * dynamic_contrast
        ).reshape(-1),
        "target_probability": target_probability.reshape(-1),
    }
    for position, lag in enumerate(range(config.sequence_length - 1, -1, -1)):
        panel_data[f"observed_dynamic_lag_{lag}"] = observed_dynamic_sequences[
            :, :, position
        ].reshape(-1)
        panel_data[f"observed_common_lag_{lag}"] = observed_common_sequences[
            :, :, position
        ].reshape(-1)
        panel_data[f"event_lag_{lag}"] = event_sequences[:, :, position].reshape(-1)
        panel_data[f"missing_lag_{lag}"] = missing_sequences[:, :, position].reshape(-1)
        truth_data[f"latent_dynamic_lag_{lag}"] = latent_sequences[:, :, position].reshape(-1)

    return IndependentSimulationData(
        panel=pd.DataFrame(panel_data),
        truth=pd.DataFrame(truth_data),
        event_episodes=event_episodes,
        config=config,
    )


def fit_train_only_priors(
    training_frame: pd.DataFrame,
    *,
    smoothing: float = 1.0,
) -> FittedStaticPriors:
    """Fit global, family and asset priors after enforcing training-only input."""
    required = {"asset_id", "family_id", "split", "target"}
    missing = sorted(required.difference(training_frame.columns))
    if missing:
        raise ValueError(f"Training frame is missing columns: {missing}")
    if training_frame.empty:
        raise ValueError("Training frame cannot be empty")
    if set(training_frame["split"].astype(str)) != {"train"}:
        raise ValueError("Static priors must be fitted on training rows only")
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    target = pd.to_numeric(training_frame["target"], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("target must contain non-missing binary values")

    global_probability = _smoothed_probability(target, smoothing)
    family_probabilities = tuple(
        (str(key), _smoothed_probability(group["target"], smoothing))
        for key, group in training_frame.groupby("family_id", sort=True, observed=True)
    )
    asset_probabilities = tuple(
        (str(key), _smoothed_probability(group["target"], smoothing))
        for key, group in training_frame.groupby("asset_id", sort=True, observed=True)
    )
    return FittedStaticPriors(
        global_probability=global_probability,
        family_probabilities=family_probabilities,
        asset_probabilities=asset_probabilities,
        smoothing=smoothing,
        training_rows=len(training_frame),
    )


def enumerate_preregistered_design(
    base: IndependentSimulationConfig,
    factor_grid: Mapping[str, Sequence[object]],
) -> tuple[IndependentSimulationConfig, ...]:
    """Return every cell in a declared Cartesian design in stable order."""
    valid_fields = {field.name for field in fields(base)}
    unknown = sorted(set(factor_grid).difference(valid_fields))
    if unknown:
        raise ValueError(f"Unknown simulation factors: {unknown}")
    empty = sorted(name for name, values in factor_grid.items() if not values)
    if empty:
        raise ValueError(f"Simulation factors have no declared levels: {empty}")
    names = tuple(sorted(factor_grid))
    return tuple(
        replace(base, **dict(zip(names, values, strict=True)))
        for values in product(*(factor_grid[name] for name in names))
    )


def _centred_unit_scale(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centred = values - values.mean()
    standard_deviation = centred.std(ddof=0)
    if standard_deviation == 0.0:
        return np.zeros_like(centred)
    return centred / standard_deviation


def _ar1_series(
    rng: np.random.Generator,
    periods: int,
    persistence: float,
) -> NDArray[np.float64]:
    values = np.empty(periods, dtype=np.float64)
    values[0] = rng.normal()
    innovation_scale = np.sqrt(1.0 - persistence**2)
    for index in range(1, periods):
        values[index] = persistence * values[index - 1] + innovation_scale * rng.normal()
    return values


def _asset_states(
    rng: np.random.Generator,
    *,
    total_periods: int,
    n_assets: int,
    family_index: NDArray[np.int64],
    n_families: int,
    persistence: float,
    dependence: float,
) -> NDArray[np.float64]:
    state = np.empty((total_periods, n_assets), dtype=np.float64)
    state[0] = rng.normal(size=n_assets)
    innovation_scale = np.sqrt(1.0 - persistence**2)
    dependence_scale = np.sqrt(dependence)
    independent_scale = np.sqrt(1.0 - dependence)
    for index in range(1, total_periods):
        market_innovation = rng.normal()
        family_innovation = rng.normal(size=n_families)
        shared = (
            market_innovation / np.sqrt(2.0)
            + family_innovation[family_index] / np.sqrt(2.0)
        )
        innovation = dependence_scale * shared + independent_scale * rng.normal(size=n_assets)
        state[index] = persistence * state[index - 1] + innovation_scale * innovation
    return state


def _event_process(
    rng: np.random.Generator,
    *,
    times: pd.DatetimeIndex,
    assets: NDArray[np.str_],
    onset_probability: float,
    min_duration: int,
    max_duration: int,
) -> tuple[NDArray[np.int8], pd.DataFrame]:
    events = np.zeros((len(times), len(assets)), dtype=np.int8)
    episodes: list[dict[str, object]] = []
    for asset_index, asset_id in enumerate(assets):
        index = 0
        episode_id = 0
        while index <= len(times) - min_duration:
            if rng.random() >= onset_probability:
                index += 1
                continue
            available_maximum = min(max_duration, len(times) - index)
            duration = int(rng.integers(min_duration, available_maximum + 1))
            events[index : index + duration, asset_index] = 1
            episodes.append(
                {
                    "asset_id": str(asset_id),
                    "episode_id": episode_id,
                    "start_time": times[index],
                    "end_time": times[index + duration - 1],
                    "duration": duration,
                }
            )
            episode_id += 1
            index += duration + 1
    return events, pd.DataFrame(
        episodes,
        columns=["asset_id", "episode_id", "start_time", "end_time", "duration"],
    )


def _chronological_split_labels(config: IndependentSimulationConfig) -> NDArray[np.str_]:
    labels = np.full(config.n_periods, "validation", dtype="<U10")
    labels[: config.train_periods] = "train"
    labels[config.train_periods : config.train_periods + config.purge_periods] = "purged"
    return labels


def _smoothed_probability(target: pd.Series, smoothing: float) -> float:
    return float((target.sum() + smoothing) / (len(target) + 2.0 * smoothing))


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))
