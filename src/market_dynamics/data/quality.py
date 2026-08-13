"""Panel quality diagnostics with no implicit data repair."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def panel_quality_summary(frame: pd.DataFrame, identifier: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Measure coverage, invalid values, duplicates and extreme returns per instrument."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Quality reports require a DatetimeIndex")
    if identifier not in frame.columns:
        raise KeyError(f"Missing panel identifier: {identifier}")
    date_name = frame.index.name or "Date"
    prepared = frame.reset_index().rename(columns={date_name: "Date"})
    duplicate_mask = prepared.duplicated([identifier, "Date"], keep=False)
    price_columns = [
        column
        for column in ["Open", "High", "Low", "Close", "Adj Close"]
        if column in prepared and (column != "Adj Close" or prepared[column].notna().any())
    ]
    return_col = "log_return" if "log_return" in prepared else None
    grouped = prepared.groupby(identifier, observed=True)
    rows: list[dict[str, object]] = []
    for asset, subset in grouped:
        missingness = float(subset[price_columns].isna().mean().mean()) if price_columns else np.nan
        non_positive = int((subset[price_columns] <= 0).sum().sum()) if price_columns else 0
        extreme = int((subset[return_col].abs() > 0.30).sum()) if return_col else 0
        rows.append(
            {
                identifier: asset,
                "rows": len(subset),
                "start": subset["Date"].min(),
                "end": subset["Date"].max(),
                "missing_price_fraction": missingness,
                "duplicate_timestamps": int(duplicate_mask.loc[subset.index].sum()),
                "non_positive_prices": non_positive,
                "extreme_return_flags": extreme,
            }
        )
    detail = pd.DataFrame(rows).sort_values(identifier).reset_index(drop=True)
    overview = {
        "rows": int(len(frame)),
        "assets": int(frame[identifier].nunique()),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "duplicate_asset_timestamps": int(duplicate_mask.sum()),
        "non_positive_prices": int((prepared[price_columns] <= 0).sum().sum()) if price_columns else 0,
    }
    return detail, overview


def write_quality_report(
    detail: pd.DataFrame,
    overview: dict[str, object],
    path: str | Path,
    title: str,
    caveats: list[str] | None = None,
) -> None:
    """Write a human-readable quality report alongside the tabular diagnostics."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Overview", ""]
    lines.extend(f"- **{key.replace('_', ' ')}:** {value}" for key, value in overview.items())
    lines.extend(["", "## Per-asset coverage", "", "```csv", detail.to_csv(index=False).rstrip(), "```", "", "## Limitations", ""])
    lines.extend(f"- {caveat}" for caveat in (caveats or ["No additional caveats recorded."]))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_coverage(detail: pd.DataFrame, identifier: str, path: str | Path, title: str) -> Path:
    """Plot actual per-asset start/end coverage without inferring missing dates."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = detail.sort_values("start")
    plt.figure(figsize=(11, max(4, len(ordered) * 0.18)))
    plt.hlines(range(len(ordered)), ordered["start"], ordered["end"], linewidth=2)
    plt.yticks(range(len(ordered)), ordered[identifier])
    plt.xlabel("Date")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(target, dpi=150)
    plt.close()
    return target
