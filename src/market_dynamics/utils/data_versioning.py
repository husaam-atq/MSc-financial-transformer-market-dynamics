"""Run metadata and package-version capture for reproducible experiments."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path


def package_versions(packages: list[str]) -> dict[str, str]:
    """Return installed package versions, retaining unavailable packages explicitly."""
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_run_metadata(path: str | Path, config: dict[str, object], packages: list[str]) -> None:
    """Persist immutable run metadata before an experiment begins."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "packages": package_versions(packages),
        "config": config,
    }
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
