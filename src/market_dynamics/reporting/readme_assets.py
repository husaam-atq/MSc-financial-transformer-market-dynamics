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
    _make_social_preview(outputs[4], evidence)
    return outputs


def _make_graphical_abstract(svg_path: Path, png_path: Path, evidence: ReadmeEvidence) -> None:
    fig, ax = plt.subplots(figsize=(15, 7), dpi=120)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.945,
        "When does pooled forecasting measure temporal skill?",
        fontsize=25,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.04,
        0.895,
        "A multi-asset Transformer examined through static, within-asset, identity and order controls",
        fontsize=12.5,
        color=MID,
        va="top",
    )

    _box(
        ax,
        0.04,
        0.59,
        0.24,
        0.23,
        "DAILY MULTI-ASSET INPUT",
        (
            f"{MODEL_WINDOW_ASSET_COUNT} evaluated instruments / 6 families",
            f"{METHODOLOGY_FIGURE_DATA.lookback} observed sessions x "
            f"{METHODOLOGY_FIGURE_DATA.numerical_channels} numerical channels",
            f"+ {METHODOLOGY_FIGURE_DATA.asset_embedding_dim}-D learned asset embedding",
        ),
        accent=TEAL,
        fill=LIGHT,
    )
    _box(
        ax,
        0.38,
        0.59,
        0.24,
        0.23,
        "TRANSFORMER",
        (
            f"{METHODOLOGY_FIGURE_DATA.encoder_layers}-layer encoder + temporal attention",
            f"{METHODOLOGY_FIGURE_DATA.parameters:,} parameters",
            f"Pooled ROC-AUC  {evidence.transformer_pooled_auc:.3f}",
        ),
        accent=BLUE,
        fill=PALE_BLUE,
    )
    _box(
        ax,
        0.72,
        0.59,
        0.24,
        0.23,
        "POOLED SCORE LOOKS STRONG",
        (
            f"ROC-AUC  {evidence.transformer_pooled_auc:.3f}",
            f"PR-AUC    {evidence.transformer_pr_auc:.3f}",
            "But pooled ranking mixes within- and between-asset pairs",
        ),
        accent=RED,
        fill=PALE_RED,
    )
    _arrow(ax, (0.285, 0.705), (0.372, 0.705), BLUE)
    _arrow(ax, (0.625, 0.705), (0.712, 0.705), BLUE)

    ax.text(
        0.04,
        0.52,
        "ADVERSARIAL CHECKS",
        fontsize=11,
        fontweight="bold",
        color=MID,
        va="center",
    )
    _box(
        ax,
        0.04,
        0.29,
        0.28,
        0.18,
        "STATIC BENCHMARK",
        (
            f"Asset prior pooled AUC  {evidence.static_asset_prior_auc:.3f}",
            "Training labels only; score is constant through time",
        ),
        accent=GOLD,
        fill=PALE_GOLD,
    )
    _box(
        ax,
        0.36,
        0.29,
        0.28,
        0.18,
        "WITHIN-ASSET ESTIMAND",
        (
            f"Transformer within-asset AUC  {evidence.transformer_within_auc:.3f}",
            "Approximately chance ranking within the same instrument",
        ),
        accent=TEAL,
        fill=LIGHT,
    )
    _box(
        ax,
        0.68,
        0.29,
        0.28,
        0.18,
        "IDENTITY + ORDER TESTS",
        (
            f"No asset ID pooled AUC  {evidence.no_asset_id_auc:.3f}",
            f"Largest registered order change  {evidence.maximum_order_auc_change:.4f}",
        ),
        accent=RED,
        fill=PALE_RED,
    )
    _arrow(ax, (0.84, 0.585), (0.84, 0.49), RED)

    _box(
        ax,
        0.04,
        0.07,
        0.43,
        0.14,
        "CONTROLLED SIMULATION",
        (
            f"{evidence.simulation_runs:,} registered runs",
            "Static heterogeneity inflates pooled AUC; planted ordered signal restores sensitivity",
        ),
        accent=BLUE,
        fill=PALE_BLUE,
    )
    _box(
        ax,
        0.55,
        0.07,
        0.41,
        0.14,
        "DIAGNOSTIC CONCLUSION",
        (
            "High pooled discrimination did not establish robust temporal skill.",
            "Evaluate pooled models with within-asset, identity and order controls.",
        ),
        accent=INK,
        fill=WHITE,
    )
    _arrow(ax, (0.475, 0.14), (0.54, 0.14), BLUE)

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


def _make_social_preview(path: Path, evidence: ReadmeEvidence) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0", facecolor=WHITE))
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            0.022,
            1,
            boxstyle="square,pad=0",
            facecolor=TEAL,
            edgecolor=TEAL,
        )
    )
    ax.text(
        0.065,
        0.84,
        "Interpretable Transformer Models",
        fontsize=26,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.065,
        0.765,
        "for Financial Time Series Forecasting",
        fontsize=23,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.065,
        0.65,
        "Separating cross-sectional shortcut learning\nfrom genuine temporal skill",
        fontsize=14,
        color=MID,
        va="top",
        linespacing=1.35,
    )

    metric_rows = (
        ("Static asset prior", evidence.static_asset_prior_auc, GOLD),
        ("Transformer pooled", evidence.transformer_pooled_auc, BLUE),
        ("Transformer within asset", evidence.transformer_within_auc, TEAL),
    )
    for index, (label, value, colour) in enumerate(metric_rows):
        y = 0.75 - index * 0.20
        ax.text(0.60, y + 0.055, label, fontsize=12, color=INK, va="center")
        ax.add_patch(
            FancyBboxPatch(
                (0.60, y - 0.015),
                0.30,
                0.042,
                boxstyle="round,pad=0,rounding_size=0.012",
                facecolor="#E5EBED",
                edgecolor="none",
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.60, y - 0.015),
                0.30 * value,
                0.042,
                boxstyle="round,pad=0,rounding_size=0.012",
                facecolor=colour,
                edgecolor="none",
            )
        )
        ax.text(0.94, y + 0.005, f"{value:.3f}", fontsize=18, fontweight="bold", ha="right")

    ax.add_patch(
        FancyBboxPatch(
            (0.055, 0.105),
            0.89,
            0.13,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=INK,
            edgecolor=INK,
        )
    )
    ax.text(
        0.50,
        0.17,
        "Strong pooled ranking did not establish robust temporal forecasting skill",
        fontsize=14.5,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
    )
    ax.text(
        0.065,
        0.04,
        "MSc Data Science research artefact | within-asset metrics, identity controls, order attacks and simulation",
        fontsize=9.5,
        color=MID,
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _save_png(fig, path, "Repository social preview", dpi=100)
    plt.close(fig)


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    lines: tuple[str, ...],
    *,
    accent: str,
    fill: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.007,
            height,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=accent,
            edgecolor=accent,
            linewidth=0,
        )
    )
    ax.text(x + 0.022, y + height - 0.042, title, fontsize=9.5, fontweight="bold", color=accent)
    line_y = y + height - 0.085
    for index, line in enumerate(lines):
        ax.text(
            x + 0.022,
            line_y - index * 0.039,
            line,
            fontsize=9.0,
            color=INK,
            va="top",
        )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], colour: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=colour,
            shrinkA=0,
            shrinkB=0,
        )
    )


def _save_svg(fig, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        facecolor=WHITE,
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
        facecolor=WHITE,
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
