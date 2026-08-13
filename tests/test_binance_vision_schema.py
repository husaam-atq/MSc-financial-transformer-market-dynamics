from __future__ import annotations

from zipfile import ZipFile

from market_dynamics.data.providers.binance_vision_provider import BinanceVisionProvider


def test_binance_vision_archive_is_standardised(tmp_path) -> None:
    archive = tmp_path / "BTCUSDT-1h-2024-01.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "BTCUSDT-1h-2024-01.csv",
            "1704067200000,42000,42100,41900,42050,12,1704070799999,1,2,3,4,0\n",
        )
    frame = BinanceVisionProvider(tmp_path).read_archive(archive, "BTC/USDT")

    assert frame.index.name == "Date"
    assert frame["Ticker"].eq("BTC/USDT").all()
    assert frame["Provider"].eq("binance_vision").all()
    assert frame["Close"].iloc[0] == 42050
