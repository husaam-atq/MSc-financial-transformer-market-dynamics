from __future__ import annotations

import pandas as pd
import pytest

from market_dynamics.data.panel_builders import reconcile_ohlcv_frames


def test_ccxt_reconciliation_measures_overlap_without_merging_sources() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", name="Date")
    archive = pd.DataFrame({"Open": [1, 2, 3], "High": [2, 3, 4], "Low": [0.5, 1.5, 2.5], "Close": [1.5, 2.5, 3.5], "Volume": [10, 11, 12]}, index=index)
    ccxt = archive.copy()
    ccxt.loc[index[1], "Close"] = 2.6
    result = reconcile_ohlcv_frames(archive, ccxt)

    assert result["overlap_rows"] == 3
    assert result["max_abs_close_difference"] == pytest.approx(0.1)
