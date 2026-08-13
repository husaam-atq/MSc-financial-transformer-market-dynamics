"""Configuration loading for Phase 1 experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward from ``start``."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    return current


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and attach resolved project paths."""
    path = Path(config_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config = deepcopy(config)
    project_root = find_project_root(path.parent)
    config["_meta"] = {
        "config_path": str(path),
        "project_root": str(project_root),
    }
    config["paths"] = _resolve_paths(project_root, config.get("paths", {}))
    return config


def _resolve_paths(project_root: Path, paths: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in paths.items():
        path = Path(value)
        resolved[key] = str(path if path.is_absolute() else project_root / path)
    return resolved


def ensure_project_dirs(config: dict[str, Any]) -> None:
    """Create configured data, result and report directories."""
    for value in config.get("paths", {}).values():
        Path(value).mkdir(parents=True, exist_ok=True)


def get_tickers(config: dict[str, Any]) -> list[str]:
    """Return the configured asset universe in config order."""
    universe = config.get("data", {}).get("asset_universe", {})
    if isinstance(universe, dict):
        return list(universe.keys())
    if isinstance(universe, list):
        return list(universe)
    raise TypeError("data.asset_universe must be a mapping or list of tickers")
