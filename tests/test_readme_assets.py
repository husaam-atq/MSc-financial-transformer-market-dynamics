from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

from market_dynamics.reporting.readme_assets import (
    build_readme_assets,
    load_readme_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
ASSETS = ROOT / "assets" / "readme"
README = ROOT / "README.md"
FINAL_DISSERTATION = ROOT / "reports" / "paper" / "Husaam_Ateeq_Dissertation_Final.pdf"
FINAL_DISSERTATION_SHA256 = "bbc10de84e2d9388658ccc643050589bf09ae21bc6ea26247e24473220ab0800"
EXPECTED_ASSETS = (
    "graphical_abstract.svg",
    "graphical_abstract.png",
    "headline_results.svg",
    "controlled_simulation.svg",
    "social_preview.png",
)
EXPECTED_CHOOSE_PATH_LINKS = (
    ("Examiner: key findings and methodology", "#key-finding"),
    ("Researcher: diagnostics and simulation", "#diagnostic-framework"),
    ("Developer: reproducibility and setup", "#reproducibility"),
    ("Dissertation: final PDF", "reports/paper/Husaam_Ateeq_Dissertation_Final.pdf"),
)
EXPECTED_KEY_LINKS = (
    ("Headline results", "#key-finding"),
    ("Methodology", "#methodology"),
    ("Diagnostic framework", "#diagnostic-framework"),
    ("Controlled simulation", "#controlled-simulation"),
    ("Reproducibility", "#reproducibility"),
    ("Repository structure", "#repository-structure"),
    ("Broader experimental programme", "#broader-experimental-programme"),
    ("Dissertation artefacts", "#dissertation-artefacts"),
    ("Dissertation map", "#dissertation--repository-map"),
    (
        "Frozen release",
        "https://github.com/husaam-atq/"
        "MSc-financial-transformer-market-dynamics/tree/dissertation-final",
    ),
)
EXPECTED_CONTENTS_LINKS = (
    ("Research question", "#research-question"),
    ("Key finding", "#key-finding"),
    ("Research decision path", "#research-decision-path"),
    ("Methodology", "#methodology"),
    ("Leakage-aware evaluation", "#leakage-aware-evaluation"),
    ("Diagnostic framework", "#diagnostic-framework"),
    ("Controlled simulation", "#controlled-simulation"),
    ("Reproducibility", "#reproducibility"),
    ("Platform status", "#platform-status"),
    ("Repository structure", "#repository-structure"),
    ("Data access and reconstruction", "#data-access-and-reconstruction"),
    ("Broader experimental programme", "#broader-experimental-programme"),
    ("Scope and limitations", "#scope-and-limitations"),
    ("Dissertation artefacts", "#dissertation-artefacts"),
    ("Dissertation ↔ repository map", "#dissertation--repository-map"),
    ("Project status", "#project-status"),
    ("Citation", "#citation"),
    ("Licence", "#licence"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", payload[16:24])


def _paragraph_starting_with(text: str, prefix: str) -> str:
    start = text.index(prefix)
    return text[start:].split("\n\n", maxsplit=1)[0]


def _links(text: str) -> tuple[tuple[str, str], ...]:
    return tuple(re.findall(r"\[([^]]+)\]\(([^)]+)\)", text))


def test_readme_evidence_matches_authoritative_frozen_tables() -> None:
    evidence = load_readme_evidence(TABLES)

    assert evidence.transformer_pooled_auc == pytest.approx(0.789813558, abs=1e-9)
    assert evidence.transformer_pr_auc == pytest.approx(0.421056342, abs=1e-9)
    assert evidence.transformer_within_auc == pytest.approx(0.491638470, abs=1e-9)
    assert evidence.transformer_macro_auc == pytest.approx(0.538742925, abs=1e-9)
    assert evidence.static_asset_prior_auc == pytest.approx(0.823905856, abs=1e-9)
    assert evidence.static_family_prior_auc == pytest.approx(0.816548772, abs=1e-9)
    assert evidence.mlp_pooled_auc == pytest.approx(0.796477661, abs=1e-9)
    assert evidence.mlp_within_auc == pytest.approx(0.556966922, abs=1e-9)
    assert evidence.no_asset_id_auc == pytest.approx(0.715476655, abs=1e-9)
    assert evidence.maximum_order_auc_change == pytest.approx(0.001449082, abs=1e-9)
    assert evidence.simulation_runs == 1_040
    assert (evidence.empirical_models, evidence.temporal_gate_passes) == (5, 0)


def test_readme_asset_generation_is_deterministic_and_has_expected_dimensions(
    tmp_path: Path,
) -> None:
    first = build_readme_assets(tmp_path / "first", TABLES)
    second = build_readme_assets(tmp_path / "second", TABLES)

    assert tuple(path.name for path in first) == EXPECTED_ASSETS
    assert tuple(path.name for path in second) == EXPECTED_ASSETS
    assert [_sha256(path) for path in first] == [_sha256(path) for path in second]
    assert _png_dimensions(first[1]) == (1_536, 1_032)
    assert _png_dimensions(first[4]) == (1_280, 640)
    assert first[4].stat().st_size < 1_000_000


def test_tracked_readme_assets_exist_without_local_path_metadata() -> None:
    root_bytes = str(ROOT).encode()

    for name in EXPECTED_ASSETS:
        path = ASSETS / name
        assert path.is_file()
        assert root_bytes not in path.read_bytes()


def test_readme_retains_generated_research_graphics_and_social_preview() -> None:
    text = README.read_text(encoding="utf-8")
    generator = (
        ROOT / "src" / "market_dynamics" / "reporting" / "readme_assets.py"
    ).read_text(encoding="utf-8")
    social_preview_source = generator.split("def _make_social_preview", maxsplit=1)[1].split(
        "def _dark_box", maxsplit=1
    )[0]

    for name in (
        "graphical_abstract.svg",
        "headline_results.svg",
        "controlled_simulation.svg",
    ):
        assert f"(assets/readme/{name})" in text
    assert (ASSETS / "graphical_abstract.png").is_file()
    assert (ASSETS / "social_preview.png").is_file()
    assert "Static heterogeneity vs planted ordered signal" in (
        ASSETS / "graphical_abstract.svg"
    ).read_text(encoding="utf-8")
    assert "np.sin" not in social_preview_source
    assert "attention_weights" in social_preview_source
    for forbidden_metric in ("0.790", "0.824", "0.492", "ROC-AUC", "PR-AUC"):
        assert forbidden_metric not in social_preview_source


def test_readme_navigation_and_contents_match_current_sections() -> None:
    text = README.read_text(encoding="utf-8")
    choose_path = _paragraph_starting_with(text, "**Choose your path:**")
    key_links = _paragraph_starting_with(text, "**Key links:**")
    contents = text.split("## Contents", maxsplit=1)[1].split(
        "## Research question", maxsplit=1
    )[0]

    assert _links(choose_path) == EXPECTED_CHOOSE_PATH_LINKS
    assert _links(key_links) == EXPECTED_KEY_LINKS
    assert _links(contents) == EXPECTED_CONTENTS_LINKS


def test_readme_has_four_expected_mermaid_flowcharts() -> None:
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)\n```", text, flags=re.DOTALL)

    assert len(blocks) == 4
    assert [block.splitlines()[0] for block in blocks] == [
        "flowchart TD",
        "flowchart LR",
        "flowchart LR",
        "flowchart LR",
    ]
    assert "Frozen<br/>conclusion" in blocks[0]
    assert "Transformer within-asset<br/>AUC 0.492" in blocks[0]
    assert "60 sessions x 34<br/>numerical channels" in blocks[1]
    assert "18-date<br/>purge" in blocks[2]
    assert "World A<br/>static heterogeneity" in blocks[3]
    assert "World B<br/>planted ordered signal" in blocks[3]


def test_readme_uses_exactly_the_requested_callout_hierarchy() -> None:
    text = README.read_text(encoding="utf-8")

    assert text.count("> [!IMPORTANT]") == 1
    assert text.count("> [!TIP]") == 1
    assert text.count("> [!NOTE]") == 1
    assert "> [!WARNING]" not in text
    assert "> [!CAUTION]" not in text


def test_readme_contains_no_em_dashes() -> None:
    text = README.read_text(encoding="utf-8")
    generator = (ROOT / "src" / "market_dynamics" / "reporting" / "readme_assets.py").read_text(
        encoding="utf-8"
    )

    assert "\u2014" not in text
    assert "\u2014" not in generator


def test_readme_uses_correct_main_and_secondary_study_framing() -> None:
    text = README.read_text(encoding="utf-8")
    main, secondary = text.split("## Broader experimental programme", maxsplit=1)
    secondary_inline = " ".join(secondary.split())

    assert "306,174 daily OHLCV rows" in main
    assert "79 form valid final model windows" in main
    assert "Transformer pooled ROC-AUC | 0.790" in main
    assert "Static asset prior pooled ROC-AUC | **0.824**" in main
    assert "Transformer pair-weighted within-asset ROC-AUC | 0.492" in main
    assert "approximately 1.52 million" not in main
    assert "1,519,611 observations (approximately 1.52 million)" in secondary_inline
    assert "This is not one unified dataset or one model-training sample." in secondary_inline
    assert "exact numerical reproduction is incomplete" in secondary_inline
    assert "1,213,437 hourly observations" in secondary_inline
    assert "20 pairs" in secondary_inline
    assert "0.6015 ± 0.0049" in secondary_inline
    assert "<details>" in secondary
    expected_note = (
        "> [!NOTE]\n"
        "> The approximately 1.52 million observations are the sum of two distinct "
        "experimental tracks: 306,174 daily observations and 1,213,437 hourly crypto "
        "observations. They are not one unified dataset or one model-training sample."
    )
    assert f"</details>\n\n{expected_note}" in secondary
    assert "1.57 million" not in text
    assert "1.74%" not in text
    assert "46.80%" not in text
    assert "79% accuracy" not in text


def test_readme_relative_links_and_images_resolve() -> None:
    text = README.read_text(encoding="utf-8")
    destinations = re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", text)
    relative = [
        unquote(destination.split("#", maxsplit=1)[0])
        for destination in destinations
        if destination
        and not destination.startswith(("#", "http://", "https://", "mailto:"))
    ]

    assert relative
    missing = [destination for destination in relative if not (ROOT / destination).exists()]
    assert missing == []

    headings = {
        re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")
        for heading in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
    }
    anchors = [destination[1:] for destination in destinations if destination.startswith("#")]
    assert anchors
    assert [anchor for anchor in anchors if anchor not in headings] == []


def test_citation_metadata_is_prepared_for_release_without_fabricated_date() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == (
        "Interpretable Transformer Models for Financial Time Series Forecasting: "
        "Discovering Emergent Market Dynamics"
    )
    assert citation["authors"] == [
        {"family-names": "Ateeq", "given-names": "Muhammad Husaam"}
    ]
    assert citation["version"] == "1.0.0"
    assert "date-released" not in citation
    assert citation["license"] == "BSD-3-Clause"
    assert citation["repository-code"] == (
        "https://github.com/husaam-atq/MSc-financial-transformer-market-dynamics"
    )
    assert "doi" not in citation
    assert "orcid" not in citation["authors"][0]
    assert citation["preferred-citation"] == {
        "type": "thesis",
        "authors": [{"family-names": "Ateeq", "given-names": "Muhammad Husaam"}],
        "title": (
            "Interpretable Transformer Models for Financial Time Series Forecasting: "
            "Discovering Emergent Market Dynamics"
        ),
        "year": 2026,
        "thesis-type": "MSc dissertation",
        "institution": {"name": "Queen Mary University of London"},
    }


def test_final_dissertation_and_scoped_licensing_are_publicly_linked() -> None:
    text = README.read_text(encoding="utf-8")

    assert FINAL_DISSERTATION.is_file()
    assert _sha256(FINAL_DISSERTATION) == FINAL_DISSERTATION_SHA256
    assert not (ROOT / "reports" / "paper" / "draft_dissertation_paper_v4.pdf").exists()
    assert "Muhammad Husaam Ateeq CA" in text
    assert "[Final dissertation PDF](reports/paper/Husaam_Ateeq_Dissertation_Final.pdf)" in text

    software_licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    scope = (ROOT / "LICENSING.md").read_text(encoding="utf-8")
    assert software_licence.startswith("BSD 3-Clause License")
    assert "Copyright (c) 2026, Muhammad Husaam Ateeq" in software_licence
    assert "Creative Commons Attribution 4.0 International" in " ".join(
        scope.split()
    )
    assert not (ROOT / "LICENSE-DOCS.md").exists()
    normalized_scope = " ".join(scope.split())
    assert "submitted dissertation" in normalized_scope
    assert "provider-derived data" in normalized_scope
    assert "[BSD 3-Clause License](LICENSE)" in text
    assert "[CC BY 4.0](LICENSING.md#documentation-and-generated-figures)" in text


def test_data_access_documents_exact_identifiers_and_concise_terms_boundary() -> None:
    text = README.read_text(encoding="utf-8")

    assert "## Data access and reconstruction" in text
    assert "### Data access and provider terms" in text
    assert "### Recorded inputs and reconstruction path" in text
    for series_id in (
        "DFF",
        "DGS2",
        "DGS10",
        "T10Y2Y",
        "VIXCLS",
        "BAMLH0A0HYM2",
        "DTWEXBGS",
    ):
        assert f"https://fred.stlouisfed.org/series/{series_id}" in text
    normalized = " ".join(text.split())
    assert (
        "Raw market and macroeconomic provider data are not redistributed in this "
        "repository."
    ) in normalized
    assert "Users wishing to reproduce the data should obtain the relevant series" in normalized
    assert "Provider availability, licensing and terms may change over time." in normalized
    assert "https://fred.stlouisfed.org/legal/terms/" not in text
    assert "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html" not in text
    assert "Current provider-terms notice" not in text
    assert "machine-learning restrictions" not in text
    assert "source_symbol_yahoo" in text


def test_readme_platform_claims_match_public_execution_paths() -> None:
    text = README.read_text(encoding="utf-8")

    assert "#### Windows (PowerShell)" in text
    assert "#### macOS / Linux (POSIX shell)" in text
    assert "source .venv/bin/activate" in text
    assert "| Ubuntu, x86-64, CPU | Verified in CI |" in text
    assert "| macOS, CPU | Expected but not tested |" in text
    assert "| Apple MPS | Not supported or validated |" in text
