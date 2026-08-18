"""Build deterministic public presentation assets from frozen evidence tables."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from market_dynamics.reporting.dissertation_figures import (
    METHODOLOGY_FIGURE_DATA,
    MODEL_WINDOW_ASSET_COUNT,
    load_core_results,
    load_identity_order_results,
    load_simulation_figure_data,
)

INK = "#17252A"
TEAL = "#2A7F8E"
RED = "#B94A48"
GOLD = "#C38D2E"
BLUE = "#486A9A"
LIGHT = "#E9F0F1"
MID = "#70838A"
PALE_BLUE = "#EEF3F8"
PALE_GOLD = "#FAF5E8"
PALE_RED = "#F8ECEB"
WHITE = "#FFFFFF"
NAVY = "#071C26"
DARK_PANEL = "#102D38"
LIGHT_INK = "#F4F8F9"
LIGHT_MID = "#ABC0C7"
BRIGHT_TEAL = "#43B7C3"
BRIGHT_GOLD = "#E0AA3E"
BRIGHT_RED = "#D86661"
BRIGHT_BLUE = "#6E95CE"

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "financial-dynamics-readme-assets-v1",
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "text.color": INK,
    }
)


@dataclass(frozen=True)
class ReadmeEvidence:
    """Frozen values used in the public repository presentation."""

    transformer_pooled_auc: float
    transformer_pr_auc: float
    transformer_within_auc: float
    transformer_macro_auc: float
    static_asset_prior_auc: float
    static_family_prior_auc: float
    mlp_pooled_auc: float
    mlp_within_auc: float
    no_asset_id_auc: float
    maximum_order_auc_change: float
    simulation_runs: int
    empirical_models: int
    temporal_gate_passes: int
    static_prior_auc_increase: float
    no_signal_within_auc: float
    strong_signal_within_auc: float
    strong_signal_reversal_drop: float
    strong_signal_permutation_drop: float


def load_readme_evidence(tables_dir: Path) -> ReadmeEvidence:
    """Load public presentation values from the authoritative frozen tables."""
    core = {
        row.key: row
        for row in load_core_results(
            tables_dir / "ifddrp_identity_dynamic_information_decomposition.csv",
            tables_dir / "prp1_fixed_cross_model_results.csv",
        )
    }
    registry = {
        row["metric_id"]: row
        for row in _read_csv(tables_dir / "ifddrp_authoritative_metric_registry.csv")
    }
    order = load_identity_order_results(
        tables_dir / "ifddrp_identity_dynamic_information_decomposition.csv",
        tables_dir / "phase6_identity_swap_results.csv",
        tables_dir / "phase6_temporal_order_destruction.csv",
    )
    maximum_order_change = max(
        abs(row.auc_change) for row in order.interventions if row.category == "order"
    )

    simulation_rows = _read_csv(
        tables_dir / "prp1_study_a_independent_simulation_results.csv"
    )
    simulation_runs = sum(int(row["replications"]) for row in simulation_rows)
    gate_rows = {
        row["gate"]: row
        for row in _read_csv(
            tables_dir / "prp1_study_a_independent_simulation_gate_inference.csv"
        )
    }
    temporal_rows = _read_csv(
        tables_dir / "prp1_fixed_cross_model_temporal_skill_gates.csv"
    )

    evidence = ReadmeEvidence(
        transformer_pooled_auc=float(registry["M001"]["value"]),
        transformer_pr_auc=float(registry["M002"]["value"]),
        transformer_within_auc=float(registry["M003"]["value"]),
        transformer_macro_auc=float(registry["M004"]["value"]),
        static_asset_prior_auc=float(registry["M009"]["value"]),
        static_family_prior_auc=float(registry["M010"]["value"]),
        mlp_pooled_auc=core["mlp"].pooled_auc,
        mlp_within_auc=core["mlp"].within_asset_auc,
        no_asset_id_auc=core["transformer_no_asset_id"].pooled_auc,
        maximum_order_auc_change=maximum_order_change,
        simulation_runs=simulation_runs,
        empirical_models=len(temporal_rows),
        temporal_gate_passes=sum(
            row["strict_model_gate_pass"].strip().lower() == "true" for row in temporal_rows
        ),
        static_prior_auc_increase=_gate_estimate(gate_rows, "static_prior_inflation"),
        no_signal_within_auc=_gate_estimate(gate_rows, "no_signal_within_near_chance"),
        strong_signal_within_auc=_gate_estimate(gate_rows, "strong_signal_within_recovery"),
        strong_signal_reversal_drop=_gate_estimate(
            gate_rows,
            "strong_signal_reversal_sensitivity",
        ),
        strong_signal_permutation_drop=_gate_estimate(
            gate_rows,
            "strong_signal_permutation_sensitivity",
        ),
    )
    _validate_evidence(evidence)
    return evidence


def build_readme_assets(output_dir: Path, tables_dir: Path) -> tuple[Path, ...]:
    """Generate all tracked README and repository-preview assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = load_readme_evidence(tables_dir)
    simulation = load_simulation_figure_data(
        tables_dir / "prp1_study_a_independent_simulation_results.csv"
    )

    outputs = (
        output_dir / "graphical_abstract.svg",
        output_dir / "graphical_abstract.png",
        output_dir / "headline_results.svg",
        output_dir / "controlled_simulation.svg",
        output_dir / "social_preview.png",
    )
    _make_graphical_abstract(outputs[0], outputs[1], evidence)
    _make_headline_results(outputs[2], evidence)
    _make_controlled_simulation(outputs[3], simulation)
    _make_social_preview(outputs[4])
    return outputs


def _make_graphical_abstract(svg_path: Path, png_path: Path, evidence: ReadmeEvidence) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 8.6), dpi=120)
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.965,
        "When does pooled forecasting measure temporal skill?",
        fontsize=25,
        fontweight="bold",
        color=LIGHT_INK,
        va="top",
    )
    ax.text(
        0.04,
        0.915,
        "One apparent result, three adversarial checks, one controlled mechanism test",
        fontsize=12.8,
        color=LIGHT_MID,
        va="top",
    )

    _dark_box(
        ax,
        0.035,
        0.72,
        0.27,
        0.15,
        "MULTI-ASSET INPUT",
        (
            f"{MODEL_WINDOW_ASSET_COUNT} instruments / 6 families",
            f"{METHODOLOGY_FIGURE_DATA.lookback} sessions / "
            f"{METHODOLOGY_FIGURE_DATA.numerical_channels} channels",
            "+ learned asset identity",
        ),
        accent=BRIGHT_TEAL,
    )
    _dark_box(
        ax,
        0.365,
        0.72,
        0.27,
        0.15,
        "TRANSFORMER",
        (
            f"{METHODOLOGY_FIGURE_DATA.encoder_layers}-layer encoder",
            f"{METHODOLOGY_FIGURE_DATA.parameters:,} parameters",
        ),
        accent=BRIGHT_BLUE,
    )
    _dark_box(
        ax,
        0.695,
        0.72,
        0.27,
        0.15,
        "POOLED RESULT",
        (
            f"ROC-AUC  {evidence.transformer_pooled_auc:.3f}",
            f"PR-AUC  {evidence.transformer_pr_auc:.3f}",
        ),
        accent=BRIGHT_RED,
    )
    _dark_arrow(ax, (0.308, 0.795), (0.36, 0.795))
    _dark_arrow(ax, (0.638, 0.795), (0.69, 0.795))

    ax.text(
        0.035,
        0.655,
        "CHALLENGE / DIAGNOSE",
        fontsize=11.8,
        fontweight="bold",
        color=LIGHT_MID,
        va="center",
    )
    _dark_box(
        ax,
        0.035,
        0.43,
        0.28,
        0.16,
        "STATIC PRIOR",
        (
            f"Pooled AUC  {evidence.static_asset_prior_auc:.3f}",
            "Constant through time",
        ),
        accent=BRIGHT_GOLD,
    )
    _dark_box(
        ax,
        0.36,
        0.43,
        0.28,
        0.16,
        "WITHIN-ASSET TEST",
        (
            f"Transformer AUC  {evidence.transformer_within_auc:.3f}",
            "Approximately chance",
        ),
        accent=BRIGHT_TEAL,
    )
    _dark_box(
        ax,
        0.685,
        0.43,
        0.28,
        0.16,
        "IDENTITY + ORDER",
        (
            f"No-ID pooled AUC  {evidence.no_asset_id_auc:.3f}",
            f"Largest order change  {evidence.maximum_order_auc_change:.4f}",
        ),
        accent=BRIGHT_RED,
    )
    ax.plot((0.83, 0.83), (0.715, 0.63), color=BRIGHT_RED, linewidth=1.5)
    ax.plot((0.175, 0.83), (0.63, 0.63), color=BRIGHT_RED, linewidth=1.5)
    for target_x in (0.175, 0.5, 0.825):
        _dark_arrow(ax, (target_x, 0.63), (target_x, 0.595), colour=BRIGHT_RED)

    ax.plot((0.175, 0.825), (0.365, 0.365), color=LIGHT_MID, linewidth=1.35, alpha=0.9)
    for source_x in (0.175, 0.5, 0.825):
        _dark_arrow(ax, (source_x, 0.43), (source_x, 0.372), colour=LIGHT_MID)

    _dark_box(
        ax,
        0.19,
        0.20,
        0.62,
        0.115,
        "CONTROLLED SIMULATION",
        (
            f"{evidence.simulation_runs:,} registered runs",
            "Static heterogeneity vs planted ordered signal",
        ),
        accent=BRIGHT_BLUE,
    )
    _dark_box(
        ax,
        0.14,
        0.045,
        0.72,
        0.12,
        "CONCLUSION",
        (
            "Strong pooled discrimination did not establish",
            "robust temporal forecasting skill",
        ),
        accent=LIGHT_INK,
    )
    _dark_arrow(ax, (0.5, 0.36), (0.5, 0.32), colour=BRIGHT_BLUE)
    _dark_arrow(ax, (0.5, 0.195), (0.5, 0.155), colour=BRIGHT_BLUE)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save_svg(fig, svg_path, "Graphical abstract of the shortcut-learning diagnostic")
    _save_png(
        fig,
        png_path,
        "Graphical abstract of the shortcut-learning diagnostic",
        dpi=120,
    )
    plt.close(fig)


def _make_headline_results(path: Path, evidence: ReadmeEvidence) -> None:
    labels = ("Static asset prior", "MLP", "Transformer")
    pooled = np.array(
        [
            evidence.static_asset_prior_auc,
            evidence.mlp_pooled_auc,
            evidence.transformer_pooled_auc,
        ]
    )
    within = np.array([0.5, evidence.mlp_within_auc, evidence.transformer_within_auc])
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=120)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.axvline(0.5, color=MID, linewidth=1.2, linestyle="--", zorder=1)
    for index, row_y in enumerate(y):
        ax.plot(
            [within[index], pooled[index]],
            [row_y, row_y],
            color="#C9D4D7",
            linewidth=4,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(pooled, y, s=105, color=BLUE, edgecolor=WHITE, linewidth=0.9, zorder=3)
    ax.scatter(
        within,
        y,
        s=92,
        color=GOLD,
        marker="s",
        edgecolor=WHITE,
        linewidth=0.9,
        zorder=3,
    )
    for value, row_y in zip(pooled, y, strict=True):
        ax.text(value + 0.018, row_y + 0.10, f"{value:.3f}", fontsize=10, color=BLUE)
    for value, row_y in zip(within, y, strict=True):
        ax.text(value + 0.018, row_y - 0.18, f"{value:.3f}", fontsize=10, color=GOLD)

    ax.set_yticks(y, labels, fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.65, 2.65)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("ROC-AUC on a common 0-1 scale", fontsize=11)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#DCE3E5", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    fig.suptitle(
        "Pooled discrimination did not imply within-asset timing",
        fontsize=18,
        fontweight="bold",
        x=0.08,
        ha="left",
        y=0.97,
    )
    ax.set_title(
        "The training-only static asset prior exceeded every learned pooled score.",
        fontsize=11,
        color=MID,
        loc="left",
        pad=18,
    )
    ax.legend(
        handles=(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="Pooled"),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=GOLD,
                label="Within asset",
            ),
        ),
        loc="lower right",
        frameon=False,
        ncol=2,
        fontsize=10,
    )
    ax.text(0.5, -0.54, "chance", ha="center", va="center", fontsize=9, color=MID)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.18, top=0.78)
    _save_svg(fig, path, "Pooled versus within-asset ROC-AUC")
    plt.close(fig)


def _make_controlled_simulation(path: Path, simulation) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=120)
    fig.patch.set_facecolor(WHITE)
    for ax in axes:
        ax.set_facecolor(WHITE)
        ax.axhline(0.5, color=MID, linewidth=1.0, linestyle="--", alpha=0.8)
        ax.grid(color="#DCE3E5", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    left, right = axes
    left.plot(
        simulation.prior_heterogeneity,
        simulation.static_prior_pooled_auc,
        color=RED,
        marker="o",
        linewidth=2.4,
        label="Static prior, pooled AUC",
    )
    left.plot(
        simulation.prior_heterogeneity,
        simulation.no_signal_within_asset_auc,
        color=TEAL,
        marker="s",
        linewidth=2.4,
        label="Classifier, within-asset AUC",
    )
    left.set_title("World A: static heterogeneity", fontsize=13, loc="left", pad=10)
    left.set_xlabel("Prior heterogeneity")
    left.set_ylabel("ROC-AUC")
    left.set_ylim(0.35, 0.95)
    left.legend(frameon=False, fontsize=9, loc="upper left")

    right.plot(
        simulation.dynamic_signal,
        simulation.dynamic_within_asset_auc,
        color=TEAL,
        marker="o",
        linewidth=2.4,
        label="Within-asset AUC",
    )
    right.plot(
        simulation.dynamic_signal,
        simulation.reversal_auc_loss,
        color=GOLD,
        marker="s",
        linewidth=2.4,
        label="AUC loss after reversal",
    )
    right.set_title("World B: planted ordered signal", fontsize=13, loc="left", pad=10)
    right.set_xlabel("Dynamic signal strength")
    right.set_ylabel("AUC or AUC loss")
    right.set_ylim(-0.05, 0.85)
    right.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle(
        "Controlled simulation recovers the mechanism expected in each world",
        fontsize=18,
        fontweight="bold",
        x=0.055,
        ha="left",
        y=0.98,
    )
    fig.text(
        0.055,
        0.02,
        "Stylised mechanism validation; not external market replication or a financial effect-size estimate.",
        fontsize=9.5,
        color=MID,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.78, wspace=0.24)
    _save_svg(fig, path, "Controlled shortcut-learning simulation")
    plt.close(fig)


def _make_social_preview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.60, 0.14),
            0.36,
            0.72,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#0B2530",
            edgecolor="#1B4653",
            linewidth=1.2,
        )
    )

    chart_left, chart_right = 0.625, 0.765
    chart_bottom, chart_top = 0.24, 0.76
    for x in np.linspace(chart_left, chart_right, 6):
        ax.plot(
            (x, x),
            (chart_bottom, chart_top),
            color=LIGHT_MID,
            alpha=0.09,
            linewidth=0.8,
        )
    for y in np.linspace(chart_bottom, chart_top, 5):
        ax.plot(
            (chart_left, chart_right),
            (y, y),
            color=LIGHT_MID,
            alpha=0.09,
            linewidth=0.8,
        )

    time_points = np.linspace(chart_left, chart_right, 10)
    trajectories = (
        (
            np.array([0.39, 0.43, 0.40, 0.47, 0.45, 0.52, 0.49, 0.57, 0.55, 0.62]),
            BRIGHT_TEAL,
            2.0,
        ),
        (
            np.array([0.60, 0.56, 0.63, 0.59, 0.67, 0.64, 0.70, 0.66, 0.72, 0.69]),
            BRIGHT_GOLD,
            2.2,
        ),
        (
            np.array([0.30, 0.34, 0.32, 0.38, 0.36, 0.42, 0.40, 0.45, 0.43, 0.49]),
            BRIGHT_BLUE,
            1.8,
        ),
    )
    for values, colour, linewidth in trajectories:
        ax.plot(
            time_points,
            values,
            color=colour,
            alpha=0.92,
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        ax.scatter(
            (time_points[-1],),
            (values[-1],),
            s=24,
            facecolor=NAVY,
            edgecolor=colour,
            linewidth=1.4,
            zorder=5,
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.718, 0.215),
            0.057,
            0.57,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=BRIGHT_TEAL,
            edgecolor=BRIGHT_TEAL,
            linewidth=1.0,
            alpha=0.08,
        )
    )
    _dark_arrow(ax, (0.775, 0.50), (0.805, 0.50), colour=BRIGHT_TEAL)

    token_colours = (BRIGHT_TEAL, BRIGHT_BLUE, BRIGHT_GOLD, BRIGHT_BLUE, BRIGHT_TEAL)
    token_y = np.linspace(0.30, 0.66, len(token_colours))
    for index, (y, colour) in enumerate(zip(token_y, token_colours, strict=True)):
        width = 0.032 + 0.004 * (index % 2)
        ax.add_patch(
            FancyBboxPatch(
                (0.81, y),
                width,
                0.052,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor=colour,
                edgecolor=colour,
                linewidth=0.8,
                alpha=0.40,
            )
        )

    attention_weights = np.array(
        [
            [0.95, 0.20, 0.35, 0.12, 0.45],
            [0.18, 0.88, 0.26, 0.52, 0.16],
            [0.42, 0.30, 0.92, 0.24, 0.58],
            [0.14, 0.62, 0.28, 0.86, 0.34],
            [0.50, 0.16, 0.56, 0.32, 0.90],
        ]
    )
    matrix_left, matrix_bottom = 0.865, 0.31
    cell_size, cell_gap = 0.012, 0.004
    for row in range(attention_weights.shape[0]):
        source_y = token_y[::-1][row] + 0.026
        target_y = matrix_bottom + (4 - row) * (cell_size + cell_gap) + cell_size / 2
        ax.plot(
            (0.847, matrix_left - 0.006),
            (source_y, target_y),
            color=LIGHT_MID,
            alpha=0.20,
            linewidth=0.7,
        )
        for column in range(attention_weights.shape[1]):
            weight = attention_weights[row, column]
            colour = BRIGHT_GOLD if row == column else BRIGHT_BLUE
            ax.add_patch(
                FancyBboxPatch(
                    (
                        matrix_left + column * (cell_size + cell_gap),
                        matrix_bottom + (4 - row) * (cell_size + cell_gap),
                    ),
                    cell_size,
                    cell_size,
                    boxstyle="round,pad=0.002,rounding_size=0.002",
                    facecolor=colour,
                    edgecolor="none",
                    alpha=0.12 + 0.78 * weight,
                )
            )

    output_x = 0.948
    output_y = (0.40, 0.50, 0.60)
    matrix_right = matrix_left + 4 * (cell_size + cell_gap) + cell_size
    for y, colour in zip(output_y, (BRIGHT_TEAL, BRIGHT_GOLD, BRIGHT_BLUE), strict=True):
        ax.plot(
            (matrix_right + 0.002, output_x - 0.006),
            (0.50, y),
            color=colour,
            alpha=0.42,
            linewidth=1.0,
        )
        ax.scatter(
            (output_x,),
            (y,),
            s=38,
            facecolor=NAVY,
            edgecolor=colour,
            linewidth=1.5,
            zorder=5,
        )

    ax.add_patch(FancyBboxPatch((0.0, 0.0), 0.018, 1.0, boxstyle="square,pad=0", facecolor=BRIGHT_TEAL, edgecolor="none"))
    ax.text(
        0.055,
        0.84,
        "Interpretable Transformer Models\nfor Financial Time Series Forecasting",
        fontsize=25,
        fontweight="bold",
        color=LIGHT_INK,
        va="top",
        linespacing=1.15,
    )
    ax.text(
        0.055,
        0.59,
        "Discovering Emergent Market Dynamics",
        fontsize=16.5,
        fontweight="bold",
        color=BRIGHT_TEAL,
        va="top",
    )
    ax.text(
        0.055,
        0.34,
        "Muhammad Husaam Ateeq CA",
        fontsize=15,
        fontweight="bold",
        color=LIGHT_INK,
        va="top",
    )
    ax.text(
        0.055,
        0.255,
        "MSc Data Science and Artificial Intelligence",
        fontsize=12.5,
        color=LIGHT_MID,
        va="top",
    )
    ax.text(
        0.055,
        0.205,
        "Queen Mary University of London",
        fontsize=12.5,
        color=LIGHT_MID,
        va="top",
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save_png(fig, path, "Repository social preview", dpi=100)
    plt.close(fig)


def _dark_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            facecolor=DARK_PANEL,
            edgecolor=accent,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.012, y + 0.018),
            0.007,
            height - 0.036,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=accent,
            edgecolor="none",
        )
    )
    ax.text(
        x + 0.03,
        y + height - 0.028,
        title,
        fontsize=11.4,
        fontweight="bold",
        color=accent,
        va="top",
    )
    line_y = y + height - 0.067
    for index, line in enumerate(lines):
        ax.text(
            x + 0.03,
            line_y - index * 0.031,
            line,
            fontsize=10.4,
            color=LIGHT_INK,
            va="top",
        )


def _dark_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = BRIGHT_BLUE,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            alpha=0.95,
        )
    )


def _save_svg(fig, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        facecolor=fig.get_facecolor(),
        metadata={"Title": title, "Creator": "Matplotlib", "Date": None},
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _save_png(fig, path: Path, title: str, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        metadata={"Title": title, "Author": "Muhammad Husaam Ateeq"},
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _gate_estimate(rows: dict[str, dict[str, str]], gate: str) -> float:
    return float(rows[gate]["estimate"])


def _validate_evidence(evidence: ReadmeEvidence) -> None:
    probabilities = (
        evidence.transformer_pooled_auc,
        evidence.transformer_pr_auc,
        evidence.transformer_within_auc,
        evidence.transformer_macro_auc,
        evidence.static_asset_prior_auc,
        evidence.static_family_prior_auc,
        evidence.mlp_pooled_auc,
        evidence.mlp_within_auc,
        evidence.no_asset_id_auc,
        evidence.maximum_order_auc_change,
        evidence.static_prior_auc_increase,
        evidence.no_signal_within_auc,
        evidence.strong_signal_within_auc,
        evidence.strong_signal_reversal_drop,
        evidence.strong_signal_permutation_drop,
    )
    if not all(0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("README evidence values must be bounded in [0, 1]")
    if evidence.simulation_runs != 1_040:
        raise ValueError(f"Expected 1,040 registered simulations; found {evidence.simulation_runs}")
    if (evidence.empirical_models, evidence.temporal_gate_passes) != (5, 0):
        raise ValueError("Unexpected empirical temporal-gate summary")
    if evidence.static_asset_prior_auc <= evidence.transformer_pooled_auc:
        raise ValueError("Static asset prior must remain above the Transformer pooled AUC")
