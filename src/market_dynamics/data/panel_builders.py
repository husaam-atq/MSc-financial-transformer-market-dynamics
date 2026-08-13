"""Reproducible daily global and hourly crypto panel construction."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_dynamics.data.lineage import make_lineage_record, write_lineage_manifest
from market_dynamics.data.providers.base import ProviderRequest
from market_dynamics.data.providers.binance_vision_provider import BinanceVisionProvider
from market_dynamics.data.providers.ccxt_provider import CCXTOHLCVProvider
from market_dynamics.data.providers.fred_provider import FREDProvider
from market_dynamics.data.providers.stooq_provider import StooqProvider
from market_dynamics.data.providers.yfinance_provider import YFinanceProvider
from market_dynamics.data.quality import panel_quality_summary, plot_coverage, write_quality_report
from market_dynamics.data.universes import (
    final_inclusion_manifest,
    load_crypto_hourly_universe,
    load_daily_universe,
)
from market_dynamics.features.engineering import add_features
from market_dynamics.features.hourly_engineering import add_hourly_features
from market_dynamics.features.macro_alignment import (
    lag_fred_to_market_availability,
    merge_macro_asof,
)
from market_dynamics.targets.hourly_targets import add_hourly_targets
from market_dynamics.targets.make_targets import add_targets

LOGGER = logging.getLogger(__name__)


def build_daily_global_panel(config: dict[str, Any], project_root: str | Path) -> dict[str, object]:
    """Build the Yahoo-led daily panel, with FRED availability controls and reports."""
    root = Path(project_root)
    section = config["phase2b"]["daily"]
    paths = config["paths"]
    universe = load_daily_universe(root / section["universe"])
    raw_root = root / paths["raw"] / "yahoo" / "daily"
    provider = YFinanceProvider()
    all_frames: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    tickers = universe["source_symbol_yahoo"].tolist()
    for start in range(0, len(tickers), int(section["yahoo_batch_size"])):
        batch = tickers[start : start + int(section["yahoo_batch_size"])]
        LOGGER.info("Downloading Yahoo batch %d-%d of %d", start + 1, start + len(batch), len(tickers))
        downloaded, errors = provider.fetch_many(
            batch,
            section["start_date"],
            section.get("end_date"),
            max_retries=int(section["max_retries"]),
            backoff_seconds=float(section["retry_backoff_seconds"]),
        )
        for ticker, frame in downloaded.items():
            asset_path = raw_root / f"ticker={ticker}" / f"retrieved_{_run_stamp()}.parquet"
            _write_immutable_parquet(frame.reset_index(), asset_path)
            lineage.append(
                make_lineage_record(asset_path, "yfinance", "1d", ticker, row_count=len(frame), start=frame.index.min(), end=frame.index.max())
            )
            all_frames.append(frame)
        failures.extend({"ticker": ticker, "source": "yfinance", "reason": error} for ticker, error in errors.items())
    if not all_frames:
        raise RuntimeError("No Yahoo daily data was downloaded; see run log for attempted batches")
    raw_panel = _sort_panel(pd.concat(all_frames, axis=0), "Ticker")
    metadata = universe.set_index("ticker")
    raw_panel["asset_class"] = raw_panel["Ticker"].map(metadata["asset_class"])
    raw_panel["region"] = raw_panel["Ticker"].map(metadata["region"])

    raw_panel = raw_panel[raw_panel["Close"].notna()].copy()
    featured = add_features(raw_panel, config)
    fred_status = _attach_fred_features(featured, section, root, paths, lineage)
    featured = fred_status.pop("frame")
    targetted = add_targets(featured, {"targets": section.get("targets", {})})
    inclusion = final_inclusion_manifest(
        universe,
        targetted.rename(columns={"Ticker": "ticker"}),
        "ticker",
        int(section["min_history_rows"]),
    )
    included_tickers = inclusion.loc[inclusion["included"], "ticker"].tolist()
    final_panel = _sort_panel(targetted[targetted["Ticker"].isin(included_tickers)], "Ticker")
    decision = inclusion.set_index("ticker")
    for column in ["included", "decision_reason", "row_count"]:
        final_panel[column] = final_panel["Ticker"].map(decision[column])

    processed_root = root / paths["processed"] / "daily_global_panel"
    _write_partitioned_panel(final_panel, processed_root, "Ticker")
    inclusion_path = root / paths["manifests"] / "universe" / "daily_global_inclusion.csv"
    inclusion_path.parent.mkdir(parents=True, exist_ok=True)
    inclusion.to_csv(inclusion_path, index=False)
    detail, overview = panel_quality_summary(final_panel, "Ticker")
    quality_csv = root / paths["manifests"] / "quality" / "daily_global_quality.csv"
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(quality_csv, index=False)
    write_quality_report(
        detail,
        overview,
        root / paths["reports_tables"] / "daily_global_data_report.md",
        "Daily Global Panel Data Report",
        caveats=[
            "Yahoo Finance data are provider-supplied and can contain adjustment revisions.",
            "ETF trading calendars differ from crypto calendars; no price forward filling is used.",
            f"FRED status: {fred_status['status']}.",
            f"Yahoo download failures recorded: {len(failures)}.",
        ],
    )
    plot_coverage(detail, "Ticker", root / paths["reports_figures"] / "data_coverage" / "daily_global_coverage.png", "Daily Global Panel Coverage")
    failure_path = root / paths["manifests"] / "source_files" / "daily_yahoo_failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["ticker", "source", "reason"]).to_csv(failure_path, index=False)
    lineage.append(make_lineage_record(inclusion_path, "project", "1d", "daily_global_universe", row_count=len(inclusion)))
    write_lineage_manifest(lineage, root / paths["reports_tables"] / "data_lineage_manifest.csv")
    return {
        "panel": final_panel,
        "inclusion": inclusion,
        "failures": pd.DataFrame(failures),
        "fred_status": fred_status,
        "rows": len(final_panel),
        "assets": final_panel["Ticker"].nunique(),
        "processed_root": processed_root,
    }


def reconcile_daily_providers(config: dict[str, Any], project_root: str | Path) -> pd.DataFrame:
    """Cross-check valid Stooq mappings without merging duplicate provider prices."""
    root = Path(project_root)
    section = config["phase2b"]["daily"]
    universe = load_daily_universe(root / section["universe"])
    candidates = universe[universe["source_symbol_stooq"].notna()].head(int(section["stooq_crosscheck_count"]))
    yahoo_root = root / config["paths"]["processed"] / "daily_global_panel"
    stooq = StooqProvider(root / config["paths"].get("external", "data/external") / "stooq_manual")
    records: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        try:
            yahoo = pd.read_parquet(yahoo_root / f"Ticker={row.ticker}" / "data.parquet").set_index("Date")
            yahoo.index = pd.to_datetime(yahoo.index)
            source = stooq.fetch(ProviderRequest(row.source_symbol_stooq, section["start_date"], section.get("end_date")))
            raw_path = root / config["paths"]["raw"] / "stooq" / "daily" / f"ticker={row.ticker}" / f"retrieved_{_run_stamp()}.parquet"
            _write_immutable_parquet(source.reset_index(), raw_path)
            joined = yahoo[["Close", "Adj Close"]].rename(columns={"Close": "yahoo_close", "Adj Close": "yahoo_adj_close"}).join(
                source[["Close"]].rename(columns={"Close": "stooq_close"}), how="inner"
            ).dropna(subset=["yahoo_close", "stooq_close"])
            if joined.empty:
                raise RuntimeError("No overlapping valid close dates")
            yahoo_return = joined["yahoo_adj_close"].where(joined["yahoo_adj_close"].notna(), joined["yahoo_close"]).pct_change()
            stooq_return = joined["stooq_close"].pct_change()
            records.append({
                "ticker": row.ticker,
                "stooq_symbol": row.source_symbol_stooq,
                "status": "success",
                "overlap_rows": len(joined),
                "overlap_start": joined.index.min(),
                "overlap_end": joined.index.max(),
                "missing_yahoo_close": float(joined["yahoo_close"].isna().mean()),
                "missing_stooq_close": float(joined["stooq_close"].isna().mean()),
                "return_correlation": yahoo_return.corr(stooq_return),
                "median_abs_return_difference": (yahoo_return - stooq_return).abs().median(),
                "adjustment_note": "Yahoo adjusted close is compared with Stooq close; corporate-action conventions may differ.",
            })
        except Exception as exc:
            records.append({"ticker": row.ticker, "stooq_symbol": row.source_symbol_stooq, "status": "failed", "reason": str(exc)})
    report = pd.DataFrame(records)
    target = root / config["paths"]["reports_tables"] / "provider_reconciliation_report.md"
    lines = [
        "# Provider Reconciliation Report",
        "",
        "Yahoo is the training source. Stooq is an independent check only and is not merged into the training panel.",
        "",
        "```csv",
        report.to_csv(index=False).rstrip(),
        "```",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report.to_csv(root / config["paths"]["manifests"] / "quality" / "daily_provider_reconciliation.csv", index=False)
    _plot_provider_reconciliation(report, root / config["paths"]["reports_figures"] / "provider_reconciliation" / "stooq_return_correlation.png")
    return report


def build_crypto_hourly_panel(config: dict[str, Any], project_root: str | Path) -> dict[str, object]:
    """Backfill Binance Vision monthly archives and build a separate hourly panel."""
    root = Path(project_root)
    section = config["phase2b"]["crypto_hourly"]
    paths = config["paths"]
    universe = load_crypto_hourly_universe(root / section["universe"])
    provider = BinanceVisionProvider(root / paths["raw"] / "binance_vision", interval=section["interval"])
    now = pd.Timestamp.utcnow().tz_localize(None).to_period("M").to_timestamp()
    requested_start = pd.Timestamp(section["start_date"]).to_period("M").to_timestamp()
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    for record in universe.itertuples(index=False):
        listing = max(requested_start, pd.Timestamp(record.expected_start_date).to_period("M").to_timestamp())
        monthly: list[pd.DataFrame] = []
        for month in pd.period_range(listing, now, freq="M"):
            month_start = month.to_timestamp()
            try:
                downloaded = provider.download_month(
                    record.binance_symbol,
                    month_start,
                    max_retries=int(section["max_retries"]),
                    backoff_seconds=float(section["retry_backoff_seconds"]),
                )
                if downloaded is None:
                    continue
                frame = provider.read_archive(downloaded.path, record.symbol)
                monthly.append(frame)
                lineage.append(make_lineage_record(downloaded.path, "binance_vision", "1h", record.symbol, row_count=downloaded.rows, start=downloaded.start, end=downloaded.end))
            except Exception as exc:
                failures.append({"symbol": record.symbol, "month": str(month), "source": "binance_vision", "reason": str(exc)})
        if monthly:
            frames.append(_sort_panel(pd.concat(monthly), "Ticker"))
    if not frames:
        raise RuntimeError("No Binance Vision hourly archives were ingested")
    raw_panel = _sort_panel(pd.concat(frames), "Ticker")
    raw_panel["asset_class"] = raw_panel["Ticker"].map(universe.set_index("symbol")["asset_class"])
    featured = add_hourly_features(raw_panel)
    targetted = add_hourly_targets(featured)
    inclusion = final_inclusion_manifest(universe.rename(columns={"symbol": "Ticker"}), targetted, "Ticker", int(section["min_history_rows"]))
    included_symbols = inclusion.loc[inclusion["included"], "Ticker"].tolist()
    final_panel = _sort_panel(targetted[targetted["Ticker"].isin(included_symbols)], "Ticker")
    final_panel.index.name = "Date"
    processed_root = root / paths["processed"] / "crypto_hourly_panel"
    _write_partitioned_panel(final_panel, processed_root, "Ticker")
    inclusion_path = root / paths["manifests"] / "universe" / "crypto_hourly_inclusion.csv"
    inclusion_path.parent.mkdir(parents=True, exist_ok=True)
    inclusion.to_csv(inclusion_path, index=False)
    detail, overview = panel_quality_summary(final_panel, "Ticker")
    quality_csv = root / paths["manifests"] / "quality" / "crypto_hourly_quality.csv"
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(quality_csv, index=False)
    write_quality_report(
        detail,
        overview,
        root / paths["reports_tables"] / "crypto_hourly_data_report.md",
        "Hourly Crypto Panel Data Report",
        caveats=[
            "Monthly archive availability follows actual Binance listing dates; no pre-listing data are imputed.",
            "Hourly crypto trades continuously and is not resampled to ETF business calendars.",
            f"Binance Vision download failures recorded: {len(failures)}.",
        ],
    )
    plot_coverage(detail, "Ticker", root / paths["reports_figures"] / "data_coverage" / "crypto_hourly_coverage.png", "Hourly Crypto Panel Coverage")
    failure_path = root / paths["manifests"] / "source_files" / "crypto_binance_vision_failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["symbol", "month", "source", "reason"]).to_csv(failure_path, index=False)
    lineage.append(make_lineage_record(inclusion_path, "project", "1h", "crypto_hourly_universe", row_count=len(inclusion)))
    write_lineage_manifest(lineage, root / paths["reports_tables"] / "data_lineage_manifest.csv")
    return {"panel": final_panel, "inclusion": inclusion, "failures": pd.DataFrame(failures), "rows": len(final_panel), "assets": final_panel["Ticker"].nunique(), "processed_root": processed_root}


def reconcile_ccxt_recent(config: dict[str, Any], project_root: str | Path) -> pd.DataFrame:
    """Compare recent CCXT spot candles with the immutable Binance Vision panel."""
    root = Path(project_root)
    section = config["phase2b"]["crypto_hourly"]
    universe = load_crypto_hourly_universe(root / section["universe"])
    days = int(section["ccxt_recent_days"])
    start = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    provider = CCXTOHLCVProvider(timeframe="1h", market_type="spot")
    records: list[dict[str, object]] = []
    for row in universe.itertuples(index=False):
        try:
            ccxt_frame = provider.fetch(ProviderRequest(row.ccxt_symbol, start, None))
            cache_path = root / config["paths"]["raw"] / "ccxt" / "binance" / "1h" / f"symbol={row.binance_symbol}" / f"retrieved_{_run_stamp()}.parquet"
            _write_immutable_parquet(ccxt_frame.reset_index(), cache_path)
            archive = pd.read_parquet(
                root
                / config["paths"]["processed"]
                / "crypto_hourly_panel"
                / f"Ticker={_safe_partition_value(row.symbol)}"
                / "data.parquet"
            ).set_index("Date")
            reconciliation = reconcile_ohlcv_frames(archive, ccxt_frame)
            records.append({"symbol": row.symbol, "status": "success", **reconciliation, "retrieval_timestamp_utc": datetime.now(UTC).isoformat()})
        except Exception as exc:
            records.append({"symbol": row.symbol, "status": "failed", "reason": str(exc), "retrieval_timestamp_utc": datetime.now(UTC).isoformat()})
    output = pd.DataFrame(records)
    output.to_csv(root / config["paths"]["manifests"] / "quality" / "ccxt_binance_reconciliation.csv", index=False)
    _append_ccxt_reconciliation_report(output, root / config["paths"]["reports_tables"] / "provider_reconciliation_report.md")
    return output


def _attach_fred_features(frame: pd.DataFrame, section: dict[str, Any], root: Path, paths: dict[str, str], lineage: list[dict[str, object]]) -> dict[str, object]:
    """Attach conservatively lagged FRED columns or persist the credential blocker."""
    try:
        macro = FREDProvider().fetch_series(section["fred_series"], section["start_date"], section.get("end_date"))
        raw_path = root / paths["raw"] / "fred" / f"retrieved_{_run_stamp()}.parquet"
        _write_immutable_parquet(macro.reset_index(), raw_path)
        lineage.append(make_lineage_record(raw_path, "fred", "1d", "macro", row_count=len(macro), start=macro.index.min(), end=macro.index.max()))
        available = lag_fred_to_market_availability(macro, int(section["fred_market_day_lag"]))
        return {"frame": merge_macro_asof(frame, available), "status": "ingested with one-market-day availability lag"}
    except Exception as exc:
        LOGGER.warning("FRED features unavailable: %s", exc)
        return {"frame": frame, "status": f"not ingested: {exc}"}


def reconcile_ohlcv_frames(archive: pd.DataFrame, exchange: pd.DataFrame) -> dict[str, object]:
    """Compare two standard OHLCV frames without combining them into a panel."""
    joined = archive[["Open", "High", "Low", "Close", "Volume"]].join(
        exchange[["Open", "High", "Low", "Close", "Volume"]], how="inner", lsuffix="_archive", rsuffix="_ccxt"
    )
    if joined.empty:
        raise RuntimeError("No overlapping hourly candles")
    close_difference = (joined["Close_archive"] - joined["Close_ccxt"]).abs()
    volume_difference = (joined["Volume_archive"] - joined["Volume_ccxt"]).abs()
    return {
        "overlap_rows": len(joined),
        "overlap_start": joined.index.min(),
        "overlap_end": joined.index.max(),
        "max_abs_close_difference": close_difference.max(),
        "mean_abs_close_difference": close_difference.mean(),
        "mean_abs_volume_difference": volume_difference.mean(),
    }


def _plot_provider_reconciliation(report: pd.DataFrame, path: Path) -> None:
    """Plot only successful independent provider comparisons."""
    successful = report[report["status"] == "success"]
    if successful.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(successful["ticker"], successful["return_correlation"])
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Yahoo/Stooq return correlation")
    plt.title("Independent Daily Provider Reconciliation")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _append_ccxt_reconciliation_report(report: pd.DataFrame, path: Path) -> None:
    """Add the crypto archive/API comparison to the shared provider report."""
    path = Path(path)
    current = path.read_text(encoding="utf-8") if path.exists() else "# Provider Reconciliation Report\n"
    marker = "\n## Binance Vision and CCXT\n"
    if marker in current:
        current = current.split(marker, maxsplit=1)[0].rstrip() + "\n"
    successful = report[report["status"] == "success"]
    lines = [
        current.rstrip(),
        "",
        "## Binance Vision and CCXT",
        "",
        "CCXT is an independent recent spot-API check. Binance Vision remains the historical training source; duplicate candles are not merged.",
        "",
        f"- Successful pairs: {len(successful)} / {len(report)}",
        f"- Maximum absolute close difference: {successful['max_abs_close_difference'].max() if not successful.empty else 'n/a'}",
        f"- Maximum mean absolute volume difference: {successful['mean_abs_volume_difference'].max() if not successful.empty else 'n/a'}",
        "",
        "```csv",
        report.to_csv(index=False).rstrip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _sort_panel(frame: pd.DataFrame, identifier: str) -> pd.DataFrame:
    """Sort each asset chronologically while retaining a Date index."""
    return frame.reset_index().rename(columns={frame.index.name or "index": "Date"}).sort_values([identifier, "Date"]).set_index("Date")


def _write_immutable_parquet(frame: pd.DataFrame, target: Path) -> None:
    """Write an artifact once; never mutate raw cache files in place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        frame.to_parquet(target, index=False)


def _write_partitioned_panel(frame: pd.DataFrame, root: Path, identifier: str) -> None:
    """Store a panel as deterministic ticker partitions for lazy loading."""
    for asset, subset in frame.groupby(identifier, observed=True, sort=True):
        target = root / f"{identifier}={_safe_partition_value(str(asset))}" / "data.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        subset.reset_index().to_parquet(target, index=False)


def _run_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_partition_value(value: str) -> str:
    """Return a Windows-safe partition directory while retaining raw IDs in parquet."""
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")
