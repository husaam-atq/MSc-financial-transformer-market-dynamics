"""Small, append-safe persistence helpers for long-running research stages."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def prepare_run_directory(results_root: str | Path, prefix: str, run_dir: str | Path | None = None) -> Path:
    """Create a new run directory or validate a caller-selected resumable directory."""
    directory = Path(run_dir) if run_dir is not None else Path(results_root) / datetime.now(UTC).strftime(f"{prefix}_%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_metric_snapshot(path: str | Path, rows: list[dict[str, object]] | pd.DataFrame) -> pd.DataFrame:
    """Persist a deduplicated CSV snapshot after a completed experiment unit."""
    destination = Path(path)
    incoming = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if destination.exists():
        existing = pd.read_csv(destination)
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    if not combined.empty:
        key_columns = [
            column
            for column in ["track", "scope", "asset", "target", "task", "lookback", "model", "seed", "fold", "split"]
            if column in combined.columns
        ]
        if key_columns:
            combined = combined.drop_duplicates(subset=key_columns, keep="last")
        combined.to_csv(destination, index=False)
    return combined


def mark_stage_complete(run_dir: str | Path, stage: str, summary: dict[str, object]) -> None:
    """Write an explicit completion marker without conflating partial and final results."""
    payload = {
        "stage": stage,
        "status": "completed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        **summary,
    }
    Path(run_dir, "STAGE_COMPLETE.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
