"""Build the evidence-frozen MSc dissertation DOCX and publication figures.

The Markdown source is the editable scholarly record. This script applies the
recorded A4, two-column paper conventions and embeds compact figures generated
only from frozen summary tables or explicitly registered values.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "reports" / "paper" / "draft_dissertation_paper_v4.md"
DEFAULT_OUTPUT = ROOT / "reports" / "paper" / "draft_dissertation_paper_v4.docx"
DEFAULT_ARTIFACTS = ROOT / "results" / "dissertation_build" / "v4"
SIMULATION_TABLE = ROOT / "reports" / "tables" / "prp1_study_a_independent_simulation_results.csv"

TITLE = "Interpretable Transformer Models for Financial Time Series Forecasting: Discovering Emergent Market Dynamics"
AUTHOR = "Husaam Ateeq"

INK = "#17252A"
TEAL = "#2A7F8E"
RED = "#B94A48"
GOLD = "#C38D2E"
BLUE = "#486A9A"
LIGHT = "#E9F0F1"
MID = "#70838A"

FIGURE_CAPTIONS = {
    "methodology": (
        "Figure 1. Forecast construction and corrected chronological evaluation. "
        "Every window ends at the forecast origin; its label uses only the following ten observed sessions."
    ),
    "core_results": (
        "Figure 2. Pooled and pair-weighted within-asset ROC-AUC. The static asset prior is constant "
        "through time within each asset, so its pooled advantage is cross-sectional rather than temporal."
    ),
    "simulation": (
        "Figure 3. Controlled simulation. Static prior heterogeneity raises pooled discrimination without "
        "within-asset skill (top); planted ordered signal raises within-asset AUC and perturbation sensitivity (bottom)."
    ),
}

FIGURE_CAPTIONS_V2 = {
    "methodology": (
        "Figure 1. Forecast design and chronological evaluation. Each 60-session window ends at close t; "
        "only the next ten observed sessions define its label."
    ),
    "simulation": (
        "Figure 3. Controlled simulation. Prior heterogeneity inflates pooled AUC without within-asset skill "
        "(top); planted ordered signal restores within-asset skill and order sensitivity (bottom)."
    ),
}

FIGURE_CAPTIONS_V3 = {
    "methodology": (
        "Figure 1. Forecast design and chronological evaluation. The model receives 60 observed sessions "
        "through close t; only sessions t+1 to t+10 define the adverse-event label."
    ),
    "core_results": (
        "Figure 2. Pooled versus pair-weighted within-asset ROC-AUC. The static asset prior scores 0.824 "
        "pooled but 0.500 within asset because its score never changes through time."
    ),
    "simulation": (
        "Figure 3. Controlled mechanism checks. Static prior heterogeneity inflates pooled AUC without timing "
        "skill (A); planted ordered signal raises within-asset AUC and makes temporal destruction costly (B)."
    ),
}

TABLE_CAPTIONS = {
    1: "Table 1. Direct comparison with the closest methodological work.",
    2: "Table 2. Direct numerical channels supplied at each of the 60 timesteps.",
    3: "Table 3. Cross-model ranking on the corrected held-out test split.",
}

TABLE_CAPTIONS_V3 = {
    **TABLE_CAPTIONS,
    3: "Table 3. Cross-model ranking on the corrected historical test split.",
    4: "Table 4. Bounded recovery tests on the corrected historical test split.",
}

EQUATION_RENDER = {
    "X_{i,t} = [x_{i,t-59}, ..., x_{i,t}], where L = 60.": "X(i,t) = [x(i,t-59), ..., x(i,t)], where L = 60.",
    "R_{i,t}^{(10)} = P_{i,t+10}/P_{i,t} - 1,": "R^(10)(i,t) = P(i,t+10) / P(i,t) - 1,",
    "D_{i,t}^{(10)} = min_{1 <= k <= 10}(P_{i,t+k}/P_{i,t} - 1),": "D^(10)(i,t) = min[1 <= k <= 10] {P(i,t+k) / P(i,t) - 1},",
    "V_{i,t}^{future} = sqrt(sum_{k=1}^{10} r_{i,t+k}^2), and V_{i,t}^{hist} = sqrt(10) sd(r_{i,t-19:t}).": "V^future(i,t) = sqrt[sum(k=1..10) r(i,t+k)^2], and V^hist(i,t) = sqrt(10) sd[r(i,t-19:t)].",
    "y_{i,t} = 1{R_{i,t}^{(10)} <= -0.05 OR D_{i,t}^{(10)} <= -0.07 OR V_{i,t}^{future} >= 2 V_{i,t}^{hist}}.": "y(i,t) = 1{R^(10)(i,t) <= -0.05 OR D^(10)(i,t) <= -0.07 OR V^future(i,t) >= 2 V^hist(i,t)}.",
    "AUC_within = sum_i(n_i^+ n_i^- AUC_i) / sum_i(n_i^+ n_i^-).": "AUC_within = sum_i[n_i(+) n_i(-) AUC_i] / sum_i[n_i(+) n_i(-)].",
}

EQUATION_RENDER_V2 = {
    **EQUATION_RENDER,
    "AUC_within = sum_i(n_i^+ n_i^- AUC_i) / sum_i(n_i^+ n_i^-).": "AUC(within) = sum over i {n(i,+) n(i,-) AUC(i)} / sum over i {n(i,+) n(i,-)}.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--edition", choices=("v1", "v2", "v3", "v4"), help="Figure/layout edition; inferred from source when omitted")
    return parser.parse_args()


def _set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _set_cell_margins(cell, top: int = 50, start: int = 65, bottom: int = 50, end: int = 65) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single" if edge in {"top", "bottom", "insideH"} else "nil")
        element.set(qn("w:sz"), "5")
        element.set(qn("w:color"), "809096")


def _set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def _set_keep_together(paragraph) -> None:
    paragraph.paragraph_format.keep_together = True
    paragraph.paragraph_format.widow_control = True


def _add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])
    run.font.name = "Times New Roman"
    run.font.size = Pt(8)


def _configure_section(section, columns: int) -> None:
    section.page_width = Inches(8.268)
    section.page_height = Inches(11.693)
    section.top_margin = Inches(0.375)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.62)
    section.right_margin = Inches(0.62)
    section.header_distance = Inches(0.18)
    section.footer_distance = Inches(0.35)

    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(columns))
    cols.set(qn("w:space"), "300")
    cols.set(qn("w:equalWidth"), "1")


def _configure_document(doc: Document) -> None:
    _configure_section(doc.sections[0], columns=1)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(10.1)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.04
    normal.paragraph_format.space_after = Pt(2.8)
    normal.paragraph_format.widow_control = True

    for name, size, bold, before, after in (
        ("Heading 1", 11.0, True, 6.0, 2.4),
        ("Heading 2", 10.0, True, 4.0, 1.5),
    ):
        style = styles[name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor.from_string("17252A")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.widow_control = True

    if "Equation Compact" not in styles:
        equation = styles.add_style("Equation Compact", WD_STYLE_TYPE.PARAGRAPH)
    else:
        equation = styles["Equation Compact"]
    equation.font.name = "Cambria Math"
    equation._element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria Math")
    equation.font.size = Pt(9.2)
    equation.font.italic = True
    equation.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.paragraph_format.space_before = Pt(1.2)
    equation.paragraph_format.space_after = Pt(1.8)
    equation.paragraph_format.keep_together = True

    if "Caption Compact" not in styles:
        caption = styles.add_style("Caption Compact", WD_STYLE_TYPE.PARAGRAPH)
    else:
        caption = styles["Caption Compact"]
    caption.font.name = "Times New Roman"
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    caption.font.size = Pt(8.0)
    caption.font.italic = True
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    caption.paragraph_format.space_before = Pt(1.0)
    caption.paragraph_format.space_after = Pt(3.0)
    caption.paragraph_format.keep_together = True

    settings = doc.settings._element
    if settings.find(qn("w:autoHyphenation")) is None:
        hyphenation = OxmlElement("w:autoHyphenation")
        hyphenation.set(qn("w:val"), "true")
        settings.append(hyphenation)

    doc.core_properties.title = TITLE
    doc.core_properties.author = AUTHOR
    doc.core_properties.subject = "MSc Data Science dissertation"
    doc.core_properties.keywords = "financial time series, Transformer, shortcut learning, within-asset evaluation"


def _add_inline_markdown(paragraph, text: str, *, size: float | None = None) -> None:
    pattern = re.compile(r"(\*\*.*?\*\*|`.*?`)")
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor : match.start()])
            if size:
                run.font.size = Pt(size)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        else:
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Courier New"
        if size:
            run.font.size = Pt(size)
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        if size:
            run.font.size = Pt(size)


def _read_simulation_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _make_methodology_figure(path: Path, edition: str) -> None:
    polished = edition in {"v2", "v3", "v4"}
    full_budget = edition in {"v3", "v4"}
    fig_size = (3.38, 3.60) if full_budget else ((3.35, 3.42) if polished else (3.28, 3.35))
    fig, ax = plt.subplots(figsize=fig_size, dpi=280 if full_budget else (260 if polished else 240))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = (
        [
            (0.50, 0.89, "60 observed sessions\nending at close t", TEAL),
            (0.50, 0.71, "34 scaled channels\n27 market + 7 context\n+ 12-D learned asset identity", BLUE),
            (0.50, 0.53, "2-layer Transformer\n272,449 parameters", INK),
            (0.50, 0.35, "Adverse-event probability\nfrom sessions t+1 to t+10", RED),
        ]
        if polished
        else [
            (0.50, 0.89, "60 observed sessions\nthrough close t", TEAL),
            (0.50, 0.71, "34 scaled channels\n+ 12-D asset identity", BLUE),
            (0.50, 0.53, "2-layer Transformer\n272,449 parameters", INK),
            (0.50, 0.35, "Probability of adverse event\nover sessions t+1 ... t+10", RED),
        ]
    )
    for x, y, label, colour in boxes:
        patch = FancyBboxPatch(
            (x - 0.31, y - 0.065),
            0.62,
            0.13,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            facecolor=colour,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.add_patch(patch)
        label_font = 7.9 if full_budget else (7.5 if polished else 7.2)
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=label_font, weight="bold")
    for y1, y2 in ((0.825, 0.775), (0.645, 0.595), (0.465, 0.415)):
        ax.add_patch(FancyArrowPatch((0.5, y1), (0.5, y2), arrowstyle="-|>", mutation_scale=8, color=MID))

    title_font = 8.0 if full_budget else (7.7 if polished else 7.2)
    ax.text(0.02, 0.225, "Corrected chronological evaluation", fontsize=title_font, weight="bold", color=INK)
    segments = [
        (0.03, 0.53, TEAL, "Train\n2010-2023"),
        (0.53, 0.59, GOLD, "gap"),
        (0.59, 0.76, BLUE, "Validation\n2023-2025"),
        (0.76, 0.82, GOLD, "gap"),
        (0.82, 0.98, RED, "Test\n2025-2026"),
    ]
    y0, height = 0.09, 0.095
    for left, right, colour, label in segments:
        ax.add_patch(FancyBboxPatch((left, y0), right - left, height, boxstyle="square,pad=0", facecolor=colour, edgecolor="white", linewidth=0.5))
        segment_font = 6.4 if full_budget else (6.1 if polished else 5.8)
        ax.text((left + right) / 2, y0 + height / 2, label, ha="center", va="center", color="white", fontsize=segment_font, weight="bold")
    footer_font = 6.9 if full_budget else (6.6 if polished else 6.2)
    ax.text(0.5, 0.025, "Each boundary: purge 18 global dates; embargo 1 date", ha="center", fontsize=footer_font, color=INK)

    fig.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_core_results_figure(path: Path, edition: str) -> None:
    polished = edition in {"v2", "v3", "v4"}
    full_budget = edition in {"v3", "v4"}
    labels = ["Static asset\nprior", "MLP", "Transformer", "Transformer\n(no ID)"]
    pooled = [0.823905856, 0.796477661, 0.789813558, 0.715476655]
    within = [0.5, 0.556966922, 0.491638470, 0.472570086]

    fig_size = (3.38, 3.05) if full_budget else ((3.35, 2.68) if polished else (3.28, 2.55))
    fig, ax = plt.subplots(figsize=fig_size, dpi=280 if full_budget else (260 if polished else 240))
    y = list(range(len(labels)))
    offset = 0.18
    ax.barh([value + offset for value in y], pooled, height=0.32, label="Pooled", color=TEAL)
    ax.barh([value - offset for value in y], within, height=0.32, label="Within asset", color=GOLD)
    ax.axvline(0.5, color=INK, linewidth=0.8, linestyle="--")
    ax.text(0.502, -0.72, "chance", fontsize=6.5 if full_budget else 6, color=INK, va="center")
    ax.set_xlim(0.44, 0.85)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    axis_font = 8.0 if full_budget else (7.6 if polished else 7)
    label_font = 8.1 if full_budget else (7.5 if polished else 7)
    ax.set_xlabel("ROC-AUC", fontsize=axis_font)
    ax.set_yticks(y, labels, fontsize=label_font)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=7.5 if full_budget else (7.0 if polished else 6.5))
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5 if full_budget else (7.0 if polished else 6.5), loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2)
    if full_budget:
        ax.get_yticklabels()[0].set_fontweight("bold")
    for row, (pooled_value, within_value) in enumerate(zip(pooled, within, strict=True)):
        value_font = 6.8 if full_budget else (6.4 if polished else 5.8)
        ax.text(pooled_value + 0.005, row + offset, f"{pooled_value:.3f}", va="center", fontsize=value_font, color=INK)
        ax.text(within_value + 0.005, row - offset, f"{within_value:.3f}", va="center", fontsize=value_font, color=INK)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _select_core(rows: Iterable[dict[str, str]], *, dynamic: float, persistence: float) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["scenario"] == "core"
        and abs(_float(row, "dynamic_signal") - dynamic) < 1e-9
        and abs(_float(row, "persistence") - persistence) < 1e-9
    ]
    return sorted(selected, key=lambda row: _float(row, "prior_heterogeneity"))


def _make_simulation_figure(path: Path, edition: str) -> None:
    polished = edition in {"v2", "v3", "v4"}
    full_budget = edition in {"v3", "v4"}
    rows = _read_simulation_rows(SIMULATION_TABLE)
    no_signal = _select_core(rows, dynamic=0.0, persistence=0.7)
    no_heterogeneity = sorted(
        [
            row
            for row in rows
            if row["scenario"] == "core"
            and abs(_float(row, "prior_heterogeneity")) < 1e-9
            and abs(_float(row, "persistence") - 0.7) < 1e-9
        ],
        key=lambda row: _float(row, "dynamic_signal"),
    )

    heterogeneity = [_float(row, "prior_heterogeneity") for row in no_signal]
    prior_auc = [_float(row, "asset_prior_pooled_roc_auc_mean") for row in no_signal]
    no_signal_within = [_float(row, "pooled_classifier_pair_weighted_within_asset_roc_auc_mean") for row in no_signal]

    dynamics = [_float(row, "dynamic_signal") for row in no_heterogeneity]
    dynamic_within = [_float(row, "pooled_classifier_pair_weighted_within_asset_roc_auc_mean") for row in no_heterogeneity]
    reversal = [_float(row, "reversal_auc_drop_mean") for row in no_heterogeneity]

    fig_size = (3.38, 4.85) if full_budget else ((3.35, 4.25) if polished else (3.28, 3.75))
    fig, axes = plt.subplots(2, 1, figsize=fig_size, dpi=280 if full_budget else (260 if polished else 240))
    top, bottom = axes
    top.plot(heterogeneity, prior_auc, marker="o", color=RED, linewidth=1.8 if polished else 1.5, label="Static-prior pooled AUC" if polished else "Static prior: pooled AUC")
    top.plot(heterogeneity, no_signal_within, marker="s", color=TEAL, linewidth=1.8 if polished else 1.5, label="Classifier within-asset AUC" if polished else "Classifier: within AUC")
    top.axhline(0.5, color=INK, linewidth=0.7, linestyle="--")
    top.set_ylim(0.45, 0.94)
    axis_font = 8.4 if full_budget else (8.0 if polished else 7)
    title_font = 9.0 if full_budget else (8.6 if polished else 7.5)
    legend_font = 7.2 if full_budget else (6.8 if polished else 5.8)
    top.set_ylabel("ROC-AUC", fontsize=axis_font)
    top.set_xlabel("Prior heterogeneity", fontsize=axis_font)
    top.set_title("A. No planted temporal signal", fontsize=title_font, loc="left", weight="bold")
    top.legend(frameon=False, fontsize=legend_font, loc="upper left")

    bottom.plot(dynamics, dynamic_within, marker="o", color=TEAL, linewidth=1.8 if polished else 1.5, label="Within-asset AUC")
    bottom.plot(dynamics, reversal, marker="^", color=GOLD, linewidth=1.8 if polished else 1.5, label="AUC loss after reversal" if polished else "AUC drop after reversal")
    bottom.axhline(0.5, color=INK, linewidth=0.7, linestyle="--")
    bottom.set_ylim(-0.03, 0.84)
    bottom.set_ylabel("AUC / AUC loss" if polished else "Metric", fontsize=axis_font)
    bottom.set_xlabel("Planted dynamic signal", fontsize=axis_font)
    bottom.set_title("B. Ordered signal recovery", fontsize=title_font, loc="left", weight="bold")
    bottom.legend(frameon=False, fontsize=legend_font, loc="upper left")

    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.tick_params(labelsize=7.6 if full_budget else (7.2 if polished else 6.2))
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.5, h_pad=0.7)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(artifacts: Path, edition: str) -> dict[str, Path]:
    artifacts.mkdir(parents=True, exist_ok=True)
    paths = {
        "methodology": artifacts / "figure_1_methodology.png",
        "core_results": artifacts / "figure_2_core_results.png",
        "simulation": artifacts / "figure_3_simulation.png",
    }
    _make_methodology_figure(paths["methodology"], edition)
    _make_core_results_figure(paths["core_results"], edition)
    _make_simulation_figure(paths["simulation"], edition)
    return paths


def _split_front_matter(lines: list[str]) -> tuple[str, str, str, str, list[str]]:
    if not lines or not lines[0].startswith("# "):
        raise ValueError("Markdown source must begin with an H1 title")
    title = lines[0][2:].strip()
    if title != TITLE:
        raise ValueError(f"Unexpected fixed title: {title!r}")

    author = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Author:")), "")
    programme = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Programme:")), "")
    try:
        abstract_index = lines.index("## Abstract")
        first_body_index = next(index for index, line in enumerate(lines) if re.match(r"##\s+1\s", line))
    except (ValueError, StopIteration) as exc:
        raise ValueError("Markdown source needs Abstract and numbered body sections") from exc

    abstract_lines = [line for line in lines[abstract_index + 1 : first_body_index] if line.strip()]
    keywords = next((line for line in abstract_lines if line.startswith("Keywords:")), "")
    abstract = " ".join(line for line in abstract_lines if not line.startswith("Keywords:"))
    return title, author, programme, abstract, [keywords, *lines[first_body_index:]]


def _add_title_and_abstract(doc: Document, title: str, author: str, programme: str, abstract: str, keywords: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True
    first_line, second_line = title.split(" Forecasting:", 1)
    run = paragraph.add_run(first_line)
    run.add_break(WD_BREAK.LINE)
    run.add_text("Forecasting:" + second_line)
    run.font.name = "Arial"
    run.font.size = Pt(15.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string("17252A")

    byline = doc.add_paragraph()
    byline.alignment = WD_ALIGN_PARAGRAPH.CENTER
    byline.paragraph_format.space_after = Pt(3)
    byline.paragraph_format.keep_with_next = True
    run = byline.add_run(f"{author} | {programme}")
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)

    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading.paragraph_format.space_before = Pt(2)
    heading.paragraph_format.space_after = Pt(1.5)
    heading.paragraph_format.keep_with_next = True
    run = heading.add_run("ABSTRACT")
    run.font.name = "Arial"
    run.font.size = Pt(9.3)
    run.font.bold = True

    abstract_paragraph = doc.add_paragraph()
    abstract_paragraph.paragraph_format.left_indent = Inches(0.22)
    abstract_paragraph.paragraph_format.right_indent = Inches(0.22)
    abstract_paragraph.paragraph_format.space_after = Pt(2)
    abstract_paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _add_inline_markdown(abstract_paragraph, abstract, size=8.7)
    _set_keep_together(abstract_paragraph)

    keyword_paragraph = doc.add_paragraph()
    keyword_paragraph.paragraph_format.left_indent = Inches(0.22)
    keyword_paragraph.paragraph_format.right_indent = Inches(0.22)
    keyword_paragraph.paragraph_format.space_after = Pt(3)
    label, values = keywords.split(":", 1)
    run = keyword_paragraph.add_run(label + ":")
    run.bold = True
    run.font.size = Pt(8.2)
    run = keyword_paragraph.add_run(values)
    run.font.size = Pt(8.2)


def _add_figure(doc: Document, key: str, figures: dict[str, Path], edition: str) -> None:
    if key not in figures:
        raise KeyError(f"Unknown figure marker: {key}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_together = True
    if edition in {"v2", "v3", "v4"}:
        paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(figures[key]), width=Inches(3.31 if edition in {"v2", "v3", "v4"} else 3.25))
    caption = doc.add_paragraph(style="Caption Compact")
    if edition in {"v3", "v4"}:
        caption_text = FIGURE_CAPTIONS_V3[key]
    elif edition == "v2":
        caption_text = FIGURE_CAPTIONS_V2.get(key, FIGURE_CAPTIONS[key])
    else:
        caption_text = FIGURE_CAPTIONS[key]
    _add_inline_markdown(caption, caption_text)


def _add_markdown_table(doc: Document, rows: list[list[str]], table_number: int, edition: str) -> None:
    title = doc.add_paragraph()
    title.paragraph_format.keep_with_next = True
    title.paragraph_format.space_before = Pt(2.5)
    title.paragraph_format.space_after = Pt(1.2)
    captions = TABLE_CAPTIONS_V3 if edition in {"v3", "v4"} else TABLE_CAPTIONS
    run = title.add_run(captions[table_number])
    run.bold = True
    run.font.size = Pt(7.8)

    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_width = table._tbl.tblPr.find(qn("w:tblW"))
    if tbl_width is None:
        tbl_width = OxmlElement("w:tblW")
        table._tbl.tblPr.append(tbl_width)
    tbl_width.set(qn("w:w"), "4680")
    tbl_width.set(qn("w:type"), "dxa")
    _set_table_borders(table)
    if edition in {"v3", "v4"} and table_number == 4:
        column_widths = [Inches(0.90), Inches(1.08), Inches(1.26)]
    elif table_number == 1:
        column_widths = [Inches(0.72), Inches(1.15), Inches(1.37)]
    elif len(rows[0]) == 2:
        column_widths = [Inches(0.90), Inches(2.34)]
    elif len(rows[0]) == 3:
        column_widths = [Inches(1.48), Inches(0.88), Inches(0.88)]
    else:
        raise ValueError(f"Unsupported table width: {len(rows[0])} columns")
    for row_index, values in enumerate(rows):
        row = table.rows[row_index]
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        row.height = Pt(10)
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if row_index == 0:
            _set_repeat_table_header(row)
        for col_index, value in enumerate(values):
            cell = row.cells[col_index]
            cell.width = column_widths[col_index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            _set_cell_margins(cell)
            if row_index == 0:
                _set_cell_shading(cell, "DCE7E9")
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            prose_table = edition in {"v3", "v4"} and table_number in {1, 4}
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_index == 0 or prose_table else WD_ALIGN_PARAGRAPH.RIGHT
            run = paragraph.add_run(value)
            run.font.name = "Times New Roman"
            run.font.size = Pt(7.5 if edition in {"v3", "v4"} and table_number == 4 else 7.8)
            run.bold = row_index == 0


def _flush_paragraph(doc: Document, buffer: list[str], *, references: bool, edition: str) -> None:
    if not buffer:
        return
    text = " ".join(line.strip() for line in buffer)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if references else WD_ALIGN_PARAGRAPH.JUSTIFY
    if references:
        paragraph.paragraph_format.left_indent = Inches(0.13)
        paragraph.paragraph_format.first_line_indent = Inches(-0.13)
        paragraph.paragraph_format.space_after = Pt(0.5)
        _add_inline_markdown(paragraph, text, size=7.8)
    else:
        _add_inline_markdown(paragraph, text)
    if references or edition == "v1" or text.startswith("The aim is to determine"):
        _set_keep_together(paragraph)
    buffer.clear()


def _add_body(doc: Document, lines: list[str], figures: dict[str, Path], edition: str) -> None:
    body_section = doc.add_section(WD_SECTION.CONTINUOUS)
    _configure_section(body_section, columns=2)
    footer = doc.sections[0].footer
    if not footer.paragraphs:
        footer.add_paragraph()
    _add_page_number(footer.paragraphs[0])

    paragraph_buffer: list[str] = []
    table_buffer: list[list[str]] = []
    table_number = 0
    in_table = False
    in_references = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            if in_table:
                if len(table_buffer) < 2:
                    raise ValueError("Malformed Markdown table")
                table_number += 1
                _add_markdown_table(doc, [table_buffer[0], *table_buffer[2:]], table_number, edition)
                table_buffer.clear()
                in_table = False
            _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)
            continue

        if line.startswith("|"):
            _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)
            values = [cell.strip() for cell in line.strip().strip("|").split("|")]
            table_buffer.append(values)
            in_table = True
            continue

        if in_table:
            table_number += 1
            _add_markdown_table(doc, [table_buffer[0], *table_buffer[2:]], table_number, edition)
            table_buffer.clear()
            in_table = False

        figure_match = re.fullmatch(r"\[\[FIGURE:([a-z_]+)\]\]", line.strip())
        if figure_match:
            _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)
            _add_figure(doc, figure_match.group(1), figures, edition)
            continue

        if line.startswith("## "):
            _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)
            heading = line[3:].strip()
            doc.add_paragraph(heading, style="Heading 1")
            in_references = heading == "References" or heading.endswith(" References")
            continue

        if line.startswith("### "):
            _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)
            doc.add_paragraph(line[4:].strip(), style="Heading 2")
            continue

        if line.startswith("> "):
            _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)
            paragraph = doc.add_paragraph(style="Equation Compact")
            source_equation = line[2:].strip()
            equation_render = EQUATION_RENDER_V2 if edition in {"v2", "v3", "v4"} else EQUATION_RENDER
            paragraph.add_run(equation_render.get(source_equation, source_equation))
            continue

        if line.startswith("- "):
            _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.left_indent = Inches(0.16)
            paragraph.paragraph_format.first_line_indent = Inches(-0.10)
            paragraph.paragraph_format.space_after = Pt(1)
            _add_inline_markdown(paragraph, line[2:].strip())
            continue

        paragraph_buffer.append(line)

    if in_table:
        table_number += 1
        _add_markdown_table(doc, [table_buffer[0], *table_buffer[2:]], table_number, edition)
    _flush_paragraph(doc, paragraph_buffer, references=in_references, edition=edition)


def build(source: Path, output: Path, artifacts: Path, edition: str) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    title, author, programme, abstract, body = _split_front_matter(lines)
    keywords = body.pop(0)
    figures = generate_figures(artifacts, edition)

    doc = Document()
    _configure_document(doc)
    _add_title_and_abstract(doc, title, author, programme, abstract, keywords)
    _add_body(doc, body, figures, edition)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    print(f"Built {output}")
    print(f"Generated {len(figures)} embedded figures under {artifacts}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if args.edition:
        edition = args.edition
    elif source.stem.endswith("_v4"):
        edition = "v4"
    elif source.stem.endswith("_v3"):
        edition = "v3"
    elif source.stem.endswith("_v2"):
        edition = "v2"
    else:
        edition = "v1"
    build(source, args.output.resolve(), args.artifacts.resolve(), edition)


if __name__ == "__main__":
    main()
