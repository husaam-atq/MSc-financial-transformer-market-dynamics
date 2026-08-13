"""Deterministic source-file lineage manifests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_lineage_record(
    path: str | Path,
    source: str,
    frequency: str,
    instrument: str,
    retrieval_timestamp: datetime | None = None,
    row_count: int | None = None,
    start: object | None = None,
    end: object | None = None,
) -> dict[str, object]:
    """Create a portable record for a raw or processed source artifact."""
    artifact = Path(path)
    return {
        "path": str(artifact),
        "source": source,
        "frequency": frequency,
        "instrument": instrument,
        "retrieval_timestamp_utc": (retrieval_timestamp or datetime.now(UTC)).isoformat(),
        "bytes": artifact.stat().st_size if artifact.exists() else None,
        "sha256": sha256_file(artifact) if artifact.exists() else None,
        "row_count": row_count,
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
    }


def write_lineage_manifest(records: list[dict[str, object]], path: str | Path) -> pd.DataFrame:
    """Write deterministic source lineage records in a single CSV."""
    target = Path(path)
    existing = pd.read_csv(target) if target.exists() else pd.DataFrame()
    manifest = pd.concat([existing, pd.DataFrame(records)], ignore_index=True)
    if not manifest.empty:
        manifest = manifest.drop_duplicates(subset=["path"], keep="last").sort_values(
            ["source", "frequency", "instrument", "path"]
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(target, index=False)
    return manifest
