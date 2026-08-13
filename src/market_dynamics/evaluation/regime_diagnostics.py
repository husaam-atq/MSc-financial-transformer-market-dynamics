"""Error diagnostics by known asset metadata and past-volatility regimes."""

from __future__ import annotations

import pandas as pd


def regression_regime_diagnostics(predictions: pd.DataFrame, volatility_column: str, groups: list[str]) -> pd.DataFrame:
    """Aggregate absolute residuals by pre-forecast volatility regime and metadata."""
    required = {"y_true", "prediction", volatility_column, *groups}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing diagnostic columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["absolute_error"] = (frame["y_true"] - frame["prediction"]).abs()
    frame["volatility_regime"] = pd.qcut(frame[volatility_column], q=3, labels=["low", "medium", "high"], duplicates="drop")
    return frame.groupby([*groups, "volatility_regime"], observed=True).agg(n=("absolute_error", "size"), mae=("absolute_error", "mean")).reset_index()
