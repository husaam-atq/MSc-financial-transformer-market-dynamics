from __future__ import annotations

from market_dynamics.data.providers.stooq_provider import StooqProvider


def test_manual_stooq_csv_is_standardised(tmp_path) -> None:
    source = tmp_path / "spy.us.csv"
    source.write_text(
        "Date,Open,High,Low,Close,Volume\n2024-01-02,470,472,469,471,100000\n",
        encoding="utf-8",
    )
    frame = StooqProvider(tmp_path).import_manual_csv(source, "SPY")

    assert frame.index.name == "Date"
    assert frame["Ticker"].eq("SPY").all()
    assert frame["Provider"].eq("stooq_manual").all()
