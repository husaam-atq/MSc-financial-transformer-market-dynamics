from __future__ import annotations

import numpy as np
import pandas as pd

from market_dynamics.datasets.pooled_window_dataset import build_pooled_window_datasets
from market_dynamics.splits.temporal import global_chronological_split


def test_pooled_windows_are_asset_local_and_scalers_fit_train_only() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for ticker, offset in [("AAA", 0.0), ("BBB", 100.0)]:
        for index, date in enumerate(dates):
            rows.append({"Date": date, "Ticker": ticker, "feature": offset + index, "target": float(index % 2)})
    panel = pd.DataFrame(rows).set_index("Date")
    split = global_chronological_split(panel, purge=2, embargo=1)
    bundle = build_pooled_window_datasets(panel, "Ticker", ["feature"], "target", split, lookback=10)

    metadata = bundle.test.endpoint_metadata()
    assert set(metadata["Date"]).issubset(set(split.test_dates))
    assert len(bundle.train) > 0 and len(bundle.validation) > 0 and len(bundle.test) > 0
    assert np.isclose(bundle.preprocessors["AAA"].scaler_mean_[0], panel.loc[split.train_dates][panel.loc[split.train_dates, "Ticker"] == "AAA"]["feature"].mean())


def test_skipped_asset_is_not_exposed_as_fitted_or_fresh_eligible() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    complete = pd.DataFrame(
        {"Ticker": "AAA", "feature": np.arange(120.0), "target": np.arange(120) % 2},
        index=dates,
    )
    incomplete = pd.DataFrame(
        {"Ticker": "BBB", "feature": np.arange(72.0), "target": np.arange(72) % 2},
        index=dates[:72],
    )
    panel = pd.concat([complete, incomplete]).sort_index()
    split = global_chronological_split(panel, purge=2, embargo=1)

    bundle = build_pooled_window_datasets(panel, "Ticker", ["feature"], "target", split, lookback=10)

    assert "BBB" in bundle.skipped_assets
    assert "BBB" not in bundle.preprocessors
