"""Portable filenames for identifiers such as exchange market symbols."""

from __future__ import annotations

import re


def artifact_safe_name(value: object) -> str:
    """Convert a logical identifier into one safe filename component on Windows and POSIX."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "artifact"
