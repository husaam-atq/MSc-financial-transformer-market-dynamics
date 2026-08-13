from __future__ import annotations

from market_dynamics.utils.artifact_paths import artifact_safe_name


def test_exchange_symbols_produce_single_safe_filename_component() -> None:
    assert artifact_safe_name("BTC/USDT") == "BTC_USDT"
    assert "/" not in artifact_safe_name("BTC/USDT")
    assert "\\" not in artifact_safe_name("BTC/USDT")
