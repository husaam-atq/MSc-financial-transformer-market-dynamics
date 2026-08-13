"""Pre-test descriptive analyses for Phase 5 emergent market dynamics."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return bounded Benjamini-Hochberg adjusted q-values, preserving NaNs."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    positions = np.flatnonzero(valid)
    ordered = positions[np.argsort(values[positions])]
    count = len(ordered)
    scaled = values[ordered] * count / np.arange(1, count + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted[ordered] = np.clip(scaled, 0.0, 1.0)
    return adjusted


def fit_volume_momentum_thresholds(
    train: pd.DataFrame,
    volume_column: str,
    momentum_column: str,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    """Fit family-specific regime cut-offs on training rows only."""
    required = {"family", volume_column, momentum_column}
    missing = required.difference(train.columns)
    if missing:
        raise KeyError(f"Volume-momentum threshold frame missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for family, part in train.groupby("family", observed=True):
        volume = pd.to_numeric(part[volume_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        momentum = pd.to_numeric(part[momentum_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if volume.empty or momentum.empty:
            continue
        rows.append(
            {
                "family": family,
                "volume_lower": float(volume.quantile(lower_quantile)),
                "volume_upper": float(volume.quantile(upper_quantile)),
                "momentum_lower": float(momentum.quantile(lower_quantile)),
                "momentum_upper": float(momentum.quantile(upper_quantile)),
                "train_n": int(min(len(volume), len(momentum))),
            }
        )
    if not rows:
        raise ValueError("No finite training observations for volume-momentum thresholds")
    return pd.DataFrame(rows)


def apply_volume_momentum_states(
    frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    volume_column: str,
    momentum_column: str,
) -> pd.DataFrame:
    """Apply training-fitted family thresholds to independent split rows."""
    required = {"family", volume_column, momentum_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Volume-momentum state frame missing columns: {sorted(missing)}")
    output = frame.merge(thresholds, on="family", how="left", validate="many_to_one")
    if output["volume_lower"].isna().any():
        unknown = sorted(output.loc[output["volume_lower"].isna(), "family"].dropna().unique())
        raise ValueError(f"No training thresholds for families: {unknown}")
    volume = output[volume_column]
    momentum = output[momentum_column]
    output["volume_momentum_state"] = np.select(
        [
            volume.ge(output["volume_upper"]) & momentum.le(output["momentum_lower"]),
            volume.ge(output["volume_upper"]) & momentum.ge(output["momentum_upper"]),
            volume.le(output["volume_lower"]) & momentum.le(output["momentum_lower"]),
        ],
        ["high_volume_negative_momentum", "high_volume_positive_momentum", "low_volume_negative_momentum"],
        default="other",
    )
    return output


def summarise_volume_momentum_states(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Summarise future stress prevalence by train-derived regime state."""
    required = {"split", "family", "volume_momentum_state", target}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Volume-momentum summary frame missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for (split, family), part in frame.groupby(["split", "family"], observed=True):
        baseline = float(part[target].mean())
        for state, state_part in part.groupby("volume_momentum_state", observed=True):
            rate = float(state_part[target].mean())
            rows.append(
                {
                    "split": split,
                    "family": family,
                    "state": state,
                    "n_obs": int(len(state_part)),
                    "n_assets": int(state_part["asset_ticker"].nunique()),
                    "stress_prevalence": rate,
                    "family_baseline_stress_prevalence": baseline,
                    "risk_difference": rate - baseline,
                    "risk_ratio": rate / baseline if baseline > 0.0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def lead_lag_results(
    family_daily: pd.DataFrame,
    lags: Iterable[int],
    minimum_observations: int,
) -> pd.DataFrame:
    """Calculate pairwise family momentum-to-future-stress associations by split.

    The source series is shifted by its own observed sessions.  No prices or
    returns are filled over absent calendars, so each association is pairwise
    complete rather than a synchronised market-clock causal estimate.
    """
    required = {"split", "Date", "family", "mean_momentum", "future_stress_breadth"}
    missing = required.difference(family_daily.columns)
    if missing:
        raise KeyError(f"Lead-lag frame missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for split, split_part in family_daily.groupby("split", observed=True):
        families = sorted(split_part["family"].dropna().unique())
        by_family = {
            family: part.set_index("Date").sort_index()[["mean_momentum", "future_stress_breadth"]]
            for family, part in split_part.groupby("family", observed=True)
        }
        for source in families:
            for destination in families:
                destination_outcome = by_family[destination]["future_stress_breadth"]
                for lag in [int(value) for value in lags]:
                    source_momentum = by_family[source]["mean_momentum"].shift(lag)
                    pair = pd.concat([source_momentum.rename("source"), destination_outcome.rename("destination")], axis=1).dropna()
                    rho, p_value = _spearman(pair["source"], pair["destination"], minimum_observations)
                    rows.append(
                        {
                            "split": split,
                            "source_family": source,
                            "destination_family": destination,
                            "lag_observed_sessions": lag,
                            "n_obs": int(len(pair)),
                            "spearman_rho": rho,
                            "p_value": p_value,
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value"] = result.groupby("split", observed=True)["p_value"].transform(benjamini_hochberg)
    train = result[result["split"].eq("train")].rename(
        columns={"spearman_rho": "train_rho", "q_value": "train_q"}
    )
    validation = result[result["split"].eq("validation")].rename(
        columns={"spearman_rho": "validation_rho", "q_value": "validation_q"}
    )
    keys = ["source_family", "destination_family", "lag_observed_sessions"]
    replication = train[keys + ["train_rho", "train_q"]].merge(
        validation[keys + ["validation_rho", "validation_q"]], on=keys, how="outer", validate="one_to_one"
    )
    replication["same_sign"] = np.sign(replication["train_rho"]) == np.sign(replication["validation_rho"])
    replication["robust_train_validation"] = (
        replication["same_sign"]
        & replication["train_q"].lt(0.05)
        & replication["validation_q"].lt(0.05)
    )
    return result.merge(
        replication[keys + ["train_rho", "train_q", "validation_rho", "validation_q", "same_sign", "robust_train_validation"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )


def build_family_daily(frame: pd.DataFrame, target: str, momentum_column: str) -> pd.DataFrame:
    """Aggregate active asset observations to a family/day descriptive panel."""
    required = {"Date", "split", "family", "return_close", momentum_column, target}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Family-daily frame missing columns: {sorted(missing)}")
    output = (
        frame.groupby(["split", "Date", "family"], observed=True)
        .agg(
            n_assets=("asset_ticker", "nunique"),
            mean_return=("return_close", "mean"),
            mean_momentum=(momentum_column, "mean"),
            future_stress_breadth=(target, "mean"),
        )
        .reset_index()
    )
    return output.sort_values(["split", "family", "Date"]).reset_index(drop=True)


def correlation_regime_results(
    family_daily: pd.DataFrame,
    rolling_window: int,
    minimum_pair_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise rolling family correlations and their descriptive stress relation."""
    required = {"split", "Date", "family", "mean_return", "future_stress_breadth"}
    missing = required.difference(family_daily.columns)
    if missing:
        raise KeyError(f"Correlation-regime frame missing columns: {sorted(missing)}")
    daily_rows: list[dict[str, object]] = []
    for split, part in family_daily.groupby("split", observed=True):
        returns = part.pivot(index="Date", columns="family", values="mean_return").sort_index()
        outcome = part.groupby("Date", observed=True)["future_stress_breadth"].mean()
        for position, date in enumerate(returns.index):
            window = returns.iloc[max(0, position - int(rolling_window) + 1) : position + 1]
            pairs: list[float] = []
            for left, right in combinations(window.columns, 2):
                paired = window[[left, right]].dropna()
                if len(paired) >= int(minimum_pair_observations):
                    pairs.append(float(paired[left].corr(paired[right])))
            row: dict[str, object] = {
                "split": split,
                "Date": date,
                "future_stress_breadth": float(outcome.get(date, np.nan)),
                "mean_pairwise_correlation": float(np.mean(pairs)) if pairs else np.nan,
                "pair_count": int(len(pairs)),
            }
            for label, left, right in [
                ("equity_bond_correlation", "Equities", "Bonds"),
                ("crypto_equity_correlation", "Crypto", "Equities"),
            ]:
                if left in window and right in window:
                    paired = window[[left, right]].dropna()
                    row[label] = float(paired[left].corr(paired[right])) if len(paired) >= int(minimum_pair_observations) else np.nan
                else:
                    row[label] = np.nan
            daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    summaries: list[dict[str, object]] = []
    metrics = ["mean_pairwise_correlation", "equity_bond_correlation", "crypto_equity_correlation"]
    train_thresholds = {
        metric: float(daily.loc[daily["split"].eq("train"), metric].median())
        for metric in metrics
        if daily.loc[daily["split"].eq("train"), metric].notna().any()
    }
    for split, part in daily.groupby("split", observed=True):
        for metric in metrics:
            pair = part[[metric, "future_stress_breadth"]].dropna()
            rho, p_value = _spearman(pair[metric], pair["future_stress_breadth"], 30)
            threshold = train_thresholds.get(metric, np.nan)
            high = pair[pair[metric] >= threshold]["future_stress_breadth"] if np.isfinite(threshold) else pd.Series(dtype=float)
            low = pair[pair[metric] < threshold]["future_stress_breadth"] if np.isfinite(threshold) else pd.Series(dtype=float)
            summaries.append(
                {
                    "split": split,
                    "metric": metric,
                    "n_obs": int(len(pair)),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "train_median_threshold": threshold,
                    "high_regime_future_stress_breadth": float(high.mean()) if not high.empty else np.nan,
                    "low_regime_future_stress_breadth": float(low.mean()) if not low.empty else np.nan,
                    "high_minus_low_future_stress_breadth": float(high.mean() - low.mean()) if not high.empty and not low.empty else np.nan,
                }
            )
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary["q_value"] = summary.groupby("split", observed=True)["p_value"].transform(benjamini_hochberg)
    return daily, _annotate_train_validation_replication(summary, "metric")


def stress_transition_results(frame: pd.DataFrame, target: str, volume_column: str, momentum_column: str) -> pd.DataFrame:
    """Describe overlapping future-label onset, persistence and recovery states."""
    required = {"split", "Date", "asset_ticker", "family", target, volume_column, momentum_column, "rolling_drawdown_60d"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Stress-transition frame missing columns: {sorted(missing)}")
    rows: list[pd.DataFrame] = []
    for split, split_part in frame.groupby("split", observed=True):
        output = split_part.sort_values(["asset_ticker", "Date"]).copy()
        previous = output.groupby("asset_ticker", observed=True)[target].shift(1)
        current = output[target]
        output["transition_state"] = np.select(
            [current.eq(1.0) & previous.eq(0.0), current.eq(1.0) & previous.eq(1.0), current.eq(0.0) & previous.eq(1.0)],
            ["onset", "persistence", "recovery"],
            default="nonstress_or_initial",
        )
        output["split"] = split
        rows.append(output)
    transitions = pd.concat(rows, ignore_index=True)
    return (
        transitions.groupby(["split", "family", "transition_state"], observed=True)
        .agg(
            n_obs=(target, "size"),
            n_assets=("asset_ticker", "nunique"),
            mean_volume_activity=(volume_column, "mean"),
            mean_momentum=(momentum_column, "mean"),
            mean_drawdown=("rolling_drawdown_60d", "mean"),
        )
        .reset_index()
    )


def breadth_dispersion_results(frame: pd.DataFrame, target: str, momentum_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate active-universe breadth and dispersion without filling calendars."""
    required = {"split", "Date", "family", "asset_ticker", "return_close", momentum_column, "downside_move_indicator", target, "volume_momentum_state"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Breadth-dispersion frame missing columns: {sorted(missing)}")
    daily_rows: list[dict[str, object]] = []
    for (split, date), part in frame.groupby(["split", "Date"], observed=True):
        family_stress = part.groupby("family", observed=True)[target].mean()
        daily_rows.append(
            {
                "split": split,
                "Date": date,
                "n_active_assets": int(part["asset_ticker"].nunique()),
                "negative_momentum_breadth": float(part[momentum_column].lt(0.0).mean()),
                "downside_move_breadth": float(part["downside_move_indicator"].mean()),
                "high_volume_negative_momentum_breadth": float(part["volume_momentum_state"].eq("high_volume_negative_momentum").mean()),
                "return_dispersion": float(part["return_close"].std()),
                "momentum_dispersion": float(part[momentum_column].std()),
                "future_stress_breadth": float(part[target].mean()),
                "family_stress_concentration": float(family_stress.max()) if not family_stress.empty else np.nan,
            }
        )
    daily = pd.DataFrame(daily_rows)
    measures = [
        "negative_momentum_breadth",
        "downside_move_breadth",
        "high_volume_negative_momentum_breadth",
        "return_dispersion",
        "momentum_dispersion",
    ]
    rows: list[dict[str, object]] = []
    for split, part in daily.groupby("split", observed=True):
        for measure in measures:
            pair = part[[measure, "future_stress_breadth"]].dropna()
            rho, p_value = _spearman(pair[measure], pair["future_stress_breadth"], 30)
            rows.append({"split": split, "measure": measure, "n_dates": int(len(pair)), "spearman_rho": rho, "p_value": p_value})
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["q_value"] = summary.groupby("split", observed=True)["p_value"].transform(benjamini_hochberg)
    return daily, _annotate_train_validation_replication(summary, "measure")


def _annotate_train_validation_replication(summary: pd.DataFrame, key_column: str) -> pd.DataFrame:
    """Attach sign and multiplicity-aware replication flags to split summaries."""
    if summary.empty:
        return summary
    train = summary[summary["split"].eq("train")][[key_column, "spearman_rho", "q_value"]].rename(
        columns={"spearman_rho": "train_rho", "q_value": "train_q"}
    )
    validation = summary[summary["split"].eq("validation")][[key_column, "spearman_rho", "q_value"]].rename(
        columns={"spearman_rho": "validation_rho", "q_value": "validation_q"}
    )
    replication = train.merge(validation, on=key_column, how="outer", validate="one_to_one")
    replication["same_sign_train_validation"] = np.sign(replication["train_rho"]) == np.sign(replication["validation_rho"])
    replication["robust_train_validation"] = (
        replication["same_sign_train_validation"]
        & replication["train_q"].lt(0.05)
        & replication["validation_q"].lt(0.05)
    )
    return summary.merge(replication, on=key_column, how="left", validate="many_to_one")


def _spearman(left: pd.Series, right: pd.Series, minimum_observations: int) -> tuple[float, float]:
    pair = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < int(minimum_observations) or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan
    value = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(value.statistic), float(value.pvalue)
