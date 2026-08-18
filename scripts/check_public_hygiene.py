"""Scan tracked public files for secrets, private paths and workflow residue."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 2_000_000

BANNED_PATTERNS = [
    ("private Windows user path", re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+[\\/]", re.IGNORECASE)),
    ("private POSIX home path", re.compile(r"/(?:home|Users)/[^/\s]+/", re.IGNORECASE)),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE)),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "populated secret variable",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*[\"']?"
            r"(?!\s*$|your[_-]|example|placeholder|<|os\.|environ\b|getenv\b|none\b|null\b)"
            r"[A-Za-z0-9._~+/=-]{12,}[\"']?\s*(?:#.*)?$"
        ),
    ),
    ("internal workflow note", re.compile(r"\b(?:task board|internal workflow registry)\b", re.IGNORECASE)),
]

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".local",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "catboost_info",
    "checkpoints",
    "models",
    "predictions",
    "results",
}

EXCLUDED_PREFIXES = (
    "data/raw/",
    "data/interim/",
    "data/external/",
    "data/processed/",
    "reports/figures/",
)

EXCLUDED_NAMES = {
    ".env",
}

EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".feather",
    ".gz",
    ".h5",
    ".hdf5",
    ".joblib",
    ".parquet",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".tar",
    ".zip",
}


def is_tracked_dissertation_manuscript(path: Path) -> bool:
    """Return whether a tracked public-paper file is a complete manuscript artefact."""
    rel = to_posix_relative(path).lower()
    if not rel.startswith("reports/paper/"):
        return False
    suffix = path.suffix.lower()
    name = path.name.lower()
    return suffix in {".pdf", ".docx"} or (
        suffix == ".md" and ("dissertation" in name or "manuscript" in name)
    )


@dataclass(frozen=True)
class Finding:
    path: Path
    line_number: int
    term: str
    line: str


def to_posix_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def git_tracked_files() -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [PROJECT_ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def fallback_files() -> list[Path]:
    return [path for path in PROJECT_ROOT.rglob("*") if path.is_file()]


def is_candidate(path: Path) -> bool:
    rel = to_posix_relative(path)
    parts = set(path.relative_to(PROJECT_ROOT).parts)
    if path.name in EXCLUDED_NAMES:
        return False
    if path == Path(__file__).resolve():
        return False
    if parts & EXCLUDED_DIRS:
        return False
    if rel.startswith(EXCLUDED_PREFIXES):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name.endswith((".pyc", ".pyo")):
        return False
    if path.stat().st_size > MAX_TEXT_BYTES:
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def scan_file(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    for line_number, line in enumerate(text.splitlines(), start=1):
        for description, pattern in BANNED_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(path=path, line_number=line_number, term=description, line=line.strip()))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan tracked text files for private or operational residue.")
    parser.add_argument("--show-files", action="store_true", help="Print scanned files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    files = git_tracked_files()
    if files is None:
        files = fallback_files()
        source = "filesystem fallback"
    else:
        source = "git tracked files"
    candidates = [path for path in files if path.exists() and is_candidate(path)]
    if args.show_files:
        print(f"Scanning source: {source}")
        for path in candidates:
            print(to_posix_relative(path))

    findings: list[Finding] = []
    for path in files:
        if path.exists() and is_tracked_dissertation_manuscript(path):
            findings.append(
                Finding(
                    path=path,
                    line_number=0,
                    term="tracked complete dissertation manuscript",
                    line="Complete dissertation manuscripts are not distributed in the public tree.",
                )
            )
    for path in candidates:
        findings.extend(scan_file(path))

    if findings:
        print("Public hygiene check failed.")
        for finding in findings:
            rel = to_posix_relative(finding.path)
            print(f"{rel}:{finding.line_number}: banned term '{finding.term}' -> {finding.line}")
        return 1

    print(f"Public hygiene check passed. Scanned {len(candidates)} files using {source}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
