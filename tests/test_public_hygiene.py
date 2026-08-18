from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_hygiene.py"
SPEC = importlib.util.spec_from_file_location("check_public_hygiene", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
HYGIENE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HYGIENE
SPEC.loader.exec_module(HYGIENE)


def _matches(line: str) -> set[str]:
    return {description for description, pattern in HYGIENE.BANNED_PATTERNS if pattern.search(line)}


def test_hygiene_detects_private_paths_and_literal_credentials() -> None:
    windows_path = "C:" + "/Users/" + "example/Documents/project"
    posix_path = "/ho" + "me/example/project"
    literal_secret = "API" + "_KEY=" + "abcdefghijklmnop"
    bearer_credential = "Authorisation: " + "Bear" + "er abcdefghijklmnop"

    assert "private Windows user path" in _matches(windows_path)
    assert "private POSIX home path" in _matches(posix_path)
    assert "populated secret variable" in _matches(literal_secret)
    assert "bearer token" in _matches(bearer_credential)


def test_hygiene_allows_placeholders_and_environment_lookups() -> None:
    assert not _matches("FRED_API_KEY=")
    assert not _matches("API_KEY=your_key_here")
    assert not _matches('api_key = os.environ.get("FRED_API_KEY")')
    assert not _matches('secret = getenv("SERVICE_SECRET")')


def test_hygiene_rejects_complete_manuscripts_only_in_the_public_paper_directory() -> None:
    root = HYGIENE.PROJECT_ROOT

    assert HYGIENE.is_tracked_dissertation_manuscript(
        root / "reports" / "paper" / "final_dissertation.pdf"
    )
    assert HYGIENE.is_tracked_dissertation_manuscript(
        root / "reports" / "paper" / "working_manuscript.docx"
    )
    assert HYGIENE.is_tracked_dissertation_manuscript(
        root / "reports" / "paper" / "dissertation_manuscript.md"
    )
    assert not HYGIENE.is_tracked_dissertation_manuscript(
        root / "reports" / "tables" / "ifddrp_final_scientific_audit.md"
    )
