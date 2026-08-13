from __future__ import annotations

import pandas as pd

from market_dynamics.data.universes import final_inclusion_manifest


def test_inclusion_manifest_counts_observed_closes_not_calendar_placeholders() -> None:
    universe = pd.DataFrame({"ticker": ["NEW"], "asset_name": ["New asset"]})
    observations = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "ticker": ["NEW", "NEW", "NEW"],
            "Close": [float("nan"), 10.0, 11.0],
        }
    ).set_index("Date")

    manifest = final_inclusion_manifest(universe, observations, "ticker", min_history_rows=2)

    assert manifest.loc[0, "row_count"] == 2
    assert manifest.loc[0, "actual_start"] == pd.Timestamp("2020-01-02")
    assert manifest.loc[0, "actual_end"] == pd.Timestamp("2020-01-03")
    assert bool(manifest.loc[0, "included"])
