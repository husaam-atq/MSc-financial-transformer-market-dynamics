"""Train/validation-only diagnostics for cross-family stress-target comparability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruct_stress_components(
    panel: pd.DataFrame,
    target_column: str,
    horizon: int,
    large_negative_return: float,
    drawdown_threshold: float,
    volatility_spike_multiplier: float,
    past_volatility_window: int,
) -> pd.DataFrame:
    """Reconstruct every forward-looking stress component separately by asset.

    The calculation intentionally mirrors ``targets.make_targets._future_stress_target``.
    It is used only to audit an already-existing label, never as a model feature.
    """
    required = {"Ticker", "Close", target_column}
    missing = required.difference(panel.columns)
    if missing:
        raise KeyError(f"Target-comparability panel missing columns: {sorted(missing)}")
    rows: list[pd.DataFrame] = []
    for ticker, raw_asset in panel.groupby("Ticker", observed=True, sort=True):
        asset = raw_asset.sort_index()
        close = asset["Close"].astype(float)
        adjusted = asset.get("Adj Close")
        if adjusted is not None and adjusted.notna().any():
            price = adjusted.astype(float).where(adjusted.notna(), close)
        else:
            price = close
        log_return = np.log(price).diff()
        future_return = price.shift(-horizon) / price - 1.0
        future_min_price = price.shift(-1).iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]
        future_drawdown = future_min_price / price - 1.0
        future_squared_sum = log_return.shift(-1).pow(2.0).iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
        future_volatility = np.sqrt(future_squared_sum)
        past_volatility = log_return.rolling(past_volatility_window, min_periods=past_volatility_window).std() * np.sqrt(horizon)
        valid = future_return.notna() & future_volatility.notna()
        negative_return = future_return <= float(large_negative_return)
        drawdown = future_drawdown <= float(drawdown_threshold)
        volatility_spike = future_volatility >= float(volatility_spike_multiplier) * past_volatility
        reconstructed = (negative_return | drawdown | volatility_spike).where(valid, np.nan).astype(float)
        frame = pd.DataFrame(
            {
                "Date": asset.index,
                "asset_ticker": str(ticker),
                "stored_target": asset[target_column].to_numpy(dtype=float),
                "future_return": future_return.to_numpy(dtype=float),
                "future_drawdown": future_drawdown.to_numpy(dtype=float),
                "future_volatility": future_volatility.to_numpy(dtype=float),
                "past_volatility": past_volatility.to_numpy(dtype=float),
                "volatility_ratio": (future_volatility / past_volatility.replace(0.0, np.nan)).to_numpy(dtype=float),
                "negative_return_component": negative_return.where(valid, np.nan).astype(float).to_numpy(),
                "drawdown_component": drawdown.where(valid, np.nan).astype(float).to_numpy(),
                "volatility_spike_component": volatility_spike.where(valid, np.nan).astype(float).to_numpy(),
                "reconstructed_target": reconstructed.to_numpy(dtype=float),
            }
        )
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    valid = output[["stored_target", "reconstructed_target"]].notna().all(axis=1)
    if not valid.any():
        raise ValueError("No valid stored stress targets available for reconstruction")
    disagreement = output.loc[valid, "stored_target"].astype(int).ne(output.loc[valid, "reconstructed_target"].astype(int))
    if disagreement.any():
        raise RuntimeError(f"Stress component reconstruction disagrees with stored labels for {int(disagreement.sum())} rows")
    return output


def summarize_target_comparability(
    components: pd.DataFrame,
    family_mapping: pd.DataFrame,
    split_dates: dict[str, pd.DatetimeIndex],
    minimum_family_observations: int,
) -> pd.DataFrame:
    """Summarise stress prevalence, trigger composition and severity by family/split."""
    required = {
        "Date",
        "asset_ticker",
        "stored_target",
        "future_return",
        "future_drawdown",
        "future_volatility",
        "past_volatility",
        "volatility_ratio",
        "negative_return_component",
        "drawdown_component",
        "volatility_spike_component",
    }
    missing = required.difference(components.columns)
    if missing:
        raise KeyError(f"Stress components missing columns: {sorted(missing)}")
    mapping_required = {"ticker", "asset_class", "family"}
    mapping_missing = mapping_required.difference(family_mapping.columns)
    if mapping_missing:
        raise KeyError(f"Family mapping missing columns: {sorted(mapping_missing)}")
    working = components.merge(
        family_mapping[["ticker", "asset_class", "family"]],
        left_on="asset_ticker",
        right_on="ticker",
        how="left",
        validate="many_to_one",
    ).drop(columns=["ticker"])
    if working["family"].isna().any():
        unknown = sorted(working.loc[working["family"].isna(), "asset_ticker"].unique())
        raise ValueError(f"Target audit has unmapped assets: {unknown}")
    working["Date"] = pd.to_datetime(working["Date"])
    rows: list[dict[str, object]] = []
    for split_name, dates in split_dates.items():
        split = working[working["Date"].isin(pd.DatetimeIndex(dates))].dropna(subset=["stored_target"]).copy()
        if split.empty:
            raise ValueError(f"No target-comparability rows for {split_name}")
        for family, part in split.groupby("family", observed=True):
            rows.append(_family_summary(split_name, str(family), part, minimum_family_observations))
        rows.append(_family_summary(split_name, "__all__", split, minimum_family_observations))
    return pd.DataFrame(rows)


def _family_summary(split_name: str, family: str, frame: pd.DataFrame, minimum_n: int) -> dict[str, object]:
    stress = frame["stored_target"].to_numpy(dtype=float) > 0.5
    negative = frame["negative_return_component"].to_numpy(dtype=float) > 0.5
    drawdown = frame["drawdown_component"].to_numpy(dtype=float) > 0.5
    volatility = frame["volatility_spike_component"].to_numpy(dtype=float) > 0.5
    trigger_count = negative.astype(int) + drawdown.astype(int) + volatility.astype(int)
    stressed = frame.loc[stress]
    return {
        "split": split_name,
        "family": family,
        "n_obs": int(len(frame)),
        "n_assets": int(frame["asset_ticker"].nunique()),
        "meets_minimum_n": bool(len(frame) >= int(minimum_n)),
        "stress_prevalence": float(stress.mean()),
        "negative_return_rate": float(negative.mean()),
        "drawdown_rate": float(drawdown.mean()),
        "volatility_spike_rate": float(volatility.mean()),
        "single_trigger_stress_rate": float(np.mean(trigger_count == 1)),
        "multi_trigger_stress_rate": float(np.mean(trigger_count >= 2)),
        "median_future_return": _median(frame["future_return"]),
        "median_future_drawdown": _median(frame["future_drawdown"]),
        "median_volatility_ratio": _median(frame["volatility_ratio"]),
        "stress_median_future_return": _median(stressed["future_return"]),
        "stress_median_future_drawdown": _median(stressed["future_drawdown"]),
        "stress_median_volatility_ratio": _median(stressed["volatility_ratio"]),
    }


def _median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.median()) if not finite.empty else np.nan
