#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D


# =============================================================================
# 0. PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

# 62 曾经有一次目录仍叫 36，所以自动兼容
CANDIDATE_62 = [
    ROOT / "62_STATIC_3LOOP_ARCHITECTURE_20260930",
    ROOT / "36_STATIC_3LOOP_ARCHITECTURE_20260930",
]

DIR63 = ROOT / "63_STATIC_3LOOP_DEEPDIVE_20260930"
DIR64 = ROOT / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930"

OUT = ROOT / "65_STATIC_3LOOP_PUBLICATION_FIGURE_20261001"

MAIN = OUT / "01_MAIN_FIGURE"
SUPP = OUT / "02_SUPPLEMENTARY_FIGURES"
SOURCE = OUT / "source_data"
AUDIT = OUT / "audit"

for p in [OUT, MAIN, SUPP, SOURCE, AUDIT]:
    p.mkdir(parents=True, exist_ok=True)


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these paths exist:\n"
        + "\n".join(str(x) for x in paths)
    )


DIR62 = first_existing(CANDIDATE_62)


# =============================================================================
# 1. STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# 2. DEFINITIONS
# =============================================================================

LINEAGE_ORDER = [
    "b_cell",
    "t_cell",
    "macrophage_monocyte",
    "fibroblast",
    "endothelial",
]

LINEAGE_DISPLAY = {
    "b_cell": "B cell",
    "t_cell": "T cell",
    "macrophage_monocyte": "Macrophage / monocyte",
    "fibroblast": "Fibroblast",
    "endothelial": "Endothelial",
}

LINEAGE_COLORS = {
    "b_cell": "#4E79A7",
    "t_cell": "#E15759",
    "macrophage_monocyte": "#59A14F",
    "fibroblast": "#F28E2B",
    "endothelial": "#9467BD",
}

LOOP_ORDER = [
    "EBOX_ENGAGEMENT",
    "RORE_ENGAGEMENT",
    "DBOX_ENGAGEMENT",
]

LOOP_LABELS = [
    "E-box",
    "RORE",
    "D-box",
]

ARM_ORDER = [
    "EBOX_POS",
    "EBOX_NEG",
    "RORE_ACT",
    "RORE_REP",
    "DBOX_ACT",
    "DBOX_REP",
]

ARM_SHORT = {
    "EBOX_POS": "E+",
    "EBOX_NEG": "E−",
    "RORE_ACT": "ROR",
    "RORE_REP": "REV",
    "DBOX_ACT": "PAR",
    "DBOX_REP": "NFIL3",
}


# =============================================================================
# 3. HELPERS
# =============================================================================

def savefig(fig, stem: Path):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            stem.with_suffix(f".{ext}"),
            bbox_inches="tight",
            dpi=600 if ext == "png" else None,
        )
    plt.close(fig)


def read_first(paths, **kwargs):
    p = first_existing(paths)
    print("Reading:", p)
    return pd.read_csv(p, **kwargs)


def add_panel_label(ax, label):
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
    )


def heatmap(
    ax,
    arr,
    row_labels,
    col_labels,
    title=None,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    annotate=True,
    fmt=".2f",
    cbar=False,
    cbar_label=None,
):
    im = ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    if title:
        ax.set_title(title, pad=8)

    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(
                        j,
                        i,
                        format(arr[i, j], fmt),
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

    if cbar:
        cb = plt.colorbar(
            im,
            ax=ax,
            fraction=0.04,
            pad=0.025,
        )
        if cbar_label:
            cb.set_label(cbar_label)

    return im


# =============================================================================
# 4. LOAD TABLES
# =============================================================================

def load_tables():

    lineage_summary = read_first([
        DIR62 / "05_LINEAGE" / "STATIC_3LOOP_LINEAGE_SUMMARY.csv",
        DIR62 / "STATIC_3LOOP_LINEAGE_SUMMARY.csv",
    ])

    matched = read_first([
        DIR64 / "04_MATCHED_CONTRASTS" / "MATCHED_CONTRASTS_DONOR_BLOCK_BOOTSTRAP.csv",
        DIR64 / "MATCHED_CONTRASTS_DONOR_BLOCK_BOOTSTRAP.csv",
    ])

    network = read_first([
        DIR64 / "03_DIFFERENTIAL_COUPLING" / "RESIDUALIZED_NETWORK_SUMMARY.csv",
        DIR64 / "RESIDUALIZED_NETWORK_SUMMARY.csv",
    ])

    diff = read_first([
        DIR64 / "03_DIFFERENTIAL_COUPLING" / "DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv",
        DIR64 / "DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv",
    ])

    variance = read_first([
        DIR63 / "03_TISSUE_DONOR_ADJUSTED" / "LINEAGE_TISSUE_VARIANCE_PARTITION.csv",
        DIR63 / "LINEAGE_TISSUE_VARIANCE_PARTITION.csv",
    ])

    six_arm = read_first([
        DIR62 / "05_LINEAGE" / "STATIC_3LOOP_LINEAGE_SUMMARY.csv",
        DIR62 / "STATIC_3LOOP_LINEAGE_SUMMARY.csv",
    ])

    loo = read_first([
        DIR62 / "05_LINEAGE" / "STATIC_3LOOP_LOO_GENE_SENSITIVITY.csv",
        DIR62 / "STATIC_3LOOP_LOO_GENE_SENSITIVITY.csv",
    ])

    dbox = read_first([
        DIR63 / "02_NFIL3_DBOX" / "DBOX_VARIANT_LINEAGE_CENTROIDS.csv",
        DIR63 / "DBOX_VARIANT_LINEAGE_CENTROIDS.csv",
    ])

    return {
        "summary": lineage_summary,
        "matched": matched,
        "network": network,
        "diff": diff,
        "variance": variance,
        "six_arm": six_arm,
        "loo": loo,
        "dbox": dbox,
    }


# =============================================================================
# 5. PANEL A — THREE-LOOP ENGAGEMENT
# =============================================================================

def get_lineage_score_matrix(summary, scores):

    t = (
        summary[
            summary["score"].isin(scores)
        ]
        .pivot(
            index="lineage",
            columns="score",
            values="mean",
        )
        .reindex(
            index=LINEAGE_ORDER,
            columns=scores,
        )
    )

    return t


def panel_A(ax, summary):

    t = get_lineage_score_matrix(
        summary,
        LOOP_ORDER,
    )

    arr = t.to_numpy()

    vmax = np.nanmax(np.abs(arr))

    heatmap(
        ax,
        arr,
        [LINEAGE_DISPLAY[x] for x in LINEAGE_ORDER],
        LOOP_LABELS,
        title="Three-loop engagement",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        annotate=True,
    )

    add_panel_label(ax, "A")

    ax.set_xlabel("")
    ax.set_ylabel("")


# =============================================================================
# 6. PANEL B — SAME-DONOR SAME-TISSUE MATCHED CONTRASTS
# =============================================================================

def build_matched_matrix(matched):

    use = matched[
        matched["score"].isin(LOOP_ORDER)
    ].copy()

    use["comparison"] = (
        use["lineage_1"].map(LINEAGE_DISPLAY)
        + " − "
        + use["lineage_2"].map(LINEAGE_DISPLAY)
    )

    comparison_order = []

    for a, b in combinations_ordered(LINEAGE_ORDER):
        comparison_order.append(
            f"{LINEAGE_DISPLAY[a]} − {LINEAGE_DISPLAY[b]}"
        )

    t = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_mean_delta",
        )
        .reindex(
            index=comparison_order,
            columns=LOOP_ORDER,
        )
    )

    ci_low = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_ci_low",
        )
        .reindex(
            index=comparison_order,
            columns=LOOP_ORDER,
        )
    )

    ci_high = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_ci_high",
        )
        .reindex(
            index=comparison_order,
            columns=LOOP_ORDER,
        )
    )

    return t, ci_low, ci_high


def combinations_ordered(items):
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


def panel_B(ax, matched):

    mat, lo, hi = build_matched_matrix(
        matched
    )

    arr = mat.to_numpy()

    vmax = np.nanmax(
        np.abs(arr)
    )

    heatmap(
        ax,
        arr,
        mat.index.tolist(),
        LOOP_LABELS,
        title="Same-donor, same-tissue lineage contrasts",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        annotate=True,
    )

    # CI 不跨 0：加小圆点
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):

            l = lo.iloc[i, j]
            h = hi.iloc[i, j]

            if (
                np.isfinite(l)
                and np.isfinite(h)
                and (
                    (l > 0)
                    or (h < 0)
                )
            ):
                ax.text(
                    j + 0.34,
                    i - 0.30,
                    "•",
                    fontsize=10,
                    ha="center",
                    va="center",
                )

    ax.text(
        0,
        -0.17,
        "• 95% donor-block bootstrap CI excludes 0",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="left",
        va="top",
    )

    add_panel_label(ax, "B")


# =============================================================================
# 7. PANEL C — CONDENSED NETWORK TOPOLOGY
# =============================================================================

NODE_POS = {
    "EBOX_POS": (0.00, 0.75),
    "EBOX_NEG": (0.00, 0.25),

    "RORE_ACT": (0.50, 0.75),
    "RORE_REP": (0.50, 0.25),

    "DBOX_ACT": (1.00, 0.75),
    "DBOX_REP": (1.00, 0.25),
}


def top_edges_for_lineage(
    lineage,
    network_edges,
    diff,
    abs_rho_min=0.35,
):

    # 用 residualized coupling matrix 信息
    edges = []

    edge_cols = []

    for _, r in diff.iterrows():

        pair = tuple(sorted([
            r["arm_1"],
            r["arm_2"],
        ]))

        if lineage == r["lineage_1"]:
            rho = r["rho_1"]
        elif lineage == r["lineage_2"]:
            rho = r["rho_2"]
        else:
            continue

        if not np.isfinite(rho):
            continue

        if abs(rho) < abs_rho_min:
            continue

        sig = (
            np.isfinite(r.get("p_adj_BH", np.nan))
            and r["p_adj_BH"] < 0.05
        )

        edge_cols.append(
            {
                "a": pair[0],
                "b": pair[1],
                "rho": rho,
                "sig_diff": sig,
            }
        )

    # 去重，同一 edge 保留 |rho| 最大版本
    unique = {}

    for e in edge_cols:
        key = (e["a"], e["b"])

        if (
            key not in unique
            or abs(e["rho"])
            > abs(unique[key]["rho"])
        ):
            unique[key] = e

    return list(unique.values())


def draw_network(ax, lineage, diff):

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.05, 0.95)
    ax.axis("off")

    # nodes
    for node, (x, y) in NODE_POS.items():

        circ = Circle(
            (x, y),
            0.07,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )

        ax.add_patch(circ)

        ax.text(
            x,
            y,
            ARM_SHORT[node],
            ha="center",
            va="center",
            fontsize=6.4,
            zorder=6,
        )

    edges = top_edges_for_lineage(
        lineage,
        None,
        diff,
        abs_rho_min=0.35,
    )

    for e in edges:

        a = e["a"]
        b = e["b"]

        x1, y1 = NODE_POS[a]
        x2, y2 = NODE_POS[b]

        rho = e["rho"]

        width = (
            0.6
            + 2.2 * abs(rho)
        )

        linestyle = (
            "-"
            if rho >= 0
            else "--"
        )

        alpha = (
            0.90
            if e["sig_diff"]
            else 0.35
        )

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-",
            mutation_scale=1,
            linewidth=width,
            linestyle=linestyle,
            color="black",
            alpha=alpha,
            shrinkA=10,
            shrinkB=10,
            zorder=1,
        )

        ax.add_patch(patch)

    ax.text(
        0.5,
        0.97,
        LINEAGE_DISPLAY[lineage],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def panel_C(parent_ax, diff):

    parent_ax.axis("off")

    add_panel_label(
        parent_ax,
        "C",
    )

    parent_ax.set_title(
        "Tissue-residualized six-arm network topology",
        pad=8,
    )

    # 5 lineage mini networks
    positions = [
        (0.00, 0.52, 0.31, 0.40),
        (0.345, 0.52, 0.31, 0.40),
        (0.69, 0.52, 0.31, 0.40),

        (0.17, 0.05, 0.31, 0.40),
        (0.52, 0.05, 0.31, 0.40),
    ]

    for lineage, pos in zip(
        LINEAGE_ORDER,
        positions,
    ):
        ax = parent_ax.inset_axes(pos)

        draw_network(
            ax,
            lineage,
            diff,
        )

    parent_ax.text(
        0.5,
        -0.03,
        "Solid: positive coupling; dashed: negative. "
        "Dark edges participate in BH-significant differential coupling.",
        transform=parent_ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.3,
    )


# =============================================================================
# 8. PANEL D — LINEAGE VS TISSUE VARIANCE
# =============================================================================

VAR_SCORE_ORDER = [
    "EBOX_ARM_BALANCED",
    "RORE_ARM_BALANCED",
    "DBOX_ARM_BALANCED",
    "EBOX_BALANCE",
    "RORE_BALANCE",
    "DBOX_BALANCE",
]

VAR_LABELS = [
    "E-box\nengagement",
    "RORE\nengagement",
    "D-box\nengagement",
    "E-box\nbalance",
    "RORE\nbalance",
    "D-box\nbalance",
]


def panel_D(ax, variance):

    v = (
        variance
        .set_index("score")
        .reindex(VAR_SCORE_ORDER)
    )

    x = np.arange(
        len(v)
    )

    width = 0.36

    ax.bar(
        x - width / 2,
        v[
            "increment_lineage_given_tissue"
        ],
        width,
        label="Lineage | tissue",
    )

    ax.bar(
        x + width / 2,
        v[
            "increment_tissue_given_lineage"
        ],
        width,
        label="Tissue | lineage",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        VAR_LABELS,
    )

    ax.set_ylabel(
        "Incremental explained variance (R²)"
    )

    ax.set_title(
        "Lineage backbone and tissue remodeling"
    )

    ax.axhline(
        0,
        color="black",
        lw=0.6,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
    )

    add_panel_label(
        ax,
        "D",
    )


# =============================================================================
# 9. MAIN COMPOSITE
# =============================================================================

def make_main_figure(tables):

    fig = plt.figure(
        figsize=(14.5, 11.5)
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[
            0.82,
            1.18,
        ],
        height_ratios=[
            0.92,
            1.08,
        ],
        hspace=0.38,
        wspace=0.30,
    )

    axA = fig.add_subplot(
        gs[0, 0]
    )

    axB = fig.add_subplot(
        gs[0, 1]
    )

    axC = fig.add_subplot(
        gs[1, 0]
    )

    axD = fig.add_subplot(
        gs[1, 1]
    )

    panel_A(
        axA,
        tables["summary"],
    )

    panel_B(
        axB,
        tables["matched"],
    )

    panel_C(
        axC,
        tables["diff"],
    )

    panel_D(
        axD,
        tables["variance"],
    )

    fig.suptitle(
        "Lineage-specific static circadian network architecture",
        fontsize=15,
        y=0.995,
    )

    savefig(
        fig,
        MAIN
        / "Figure_static_3loop_network_architecture",
    )


# =============================================================================
# 10. SUPP S1 — 6 ARM + BALANCE
# =============================================================================

def supp_S1(summary):

    six = get_lineage_score_matrix(
        summary,
        ARM_ORDER,
    )

    bal = get_lineage_score_matrix(
        summary,
        [
            "EBOX_BALANCE",
            "RORE_BALANCE",
            "DBOX_BALANCE",
        ],
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.3),
    )

    arr = six.to_numpy()

    vmax = np.nanmax(
        np.abs(arr)
    )

    heatmap(
        axes[0],
        arr,
        [
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],
        [
            "E+",
            "E−",
            "ROR",
            "REV-ERB",
            "PAR-bZIP",
            "NFIL3",
        ],
        title="Six regulatory arms",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        annotate=True,
    )

    arr2 = bal.to_numpy()

    vmax2 = np.nanmax(
        np.abs(arr2)
    )

    heatmap(
        axes[1],
        arr2,
        [
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],
        [
            "E-box",
            "RORE",
            "D-box",
        ],
        title="Activator–repressor balance",
        cmap="RdBu_r",
        vmin=-vmax2,
        vmax=vmax2,
        annotate=True,
    )

    fig.suptitle(
        "Supplementary Fig. S1 — Static six-arm architecture",
        fontsize=12,
    )

    fig.tight_layout()

    savefig(
        fig,
        SUPP
        / "Supplementary_Fig_S1_six_arm_and_balance",
    )


# =============================================================================
# 11. SUPP S2 — SCORE ROBUSTNESS + DBOX
# =============================================================================

def supp_S2(dbox, loo):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
    )

    # D-box decomposition
    d = (
        dbox
        .set_index(
            "priority_lineage"
        )
        .reindex(
            LINEAGE_ORDER
        )
    )

    x = np.arange(
        len(LINEAGE_ORDER)
    )

    width = 0.19

    cols = [
        "DBOX_ACT",
        "DBOX_REP",
        "DBOX_GENE_BALANCED",
        "DBOX_ARM_BALANCED",
    ]

    labels = [
        "PAR-bZIP",
        "NFIL3",
        "Gene-balanced",
        "Arm-balanced",
    ]

    for k, (col, lab) in enumerate(
        zip(cols, labels)
    ):
        axes[0].bar(
            x
            + (
                k - 1.5
            ) * width,
            d[col],
            width,
            label=lab,
        )

    axes[0].axhline(
        0,
        color="black",
        lw=0.6,
    )

    axes[0].set_xticks(x)

    axes[0].set_xticklabels(
        [
            LINEAGE_DISPLAY[i]
            for i in LINEAGE_ORDER
        ],
        rotation=25,
        ha="right",
    )

    axes[0].set_ylabel(
        "Standardized score"
    )

    axes[0].set_title(
        "D-box decomposition"
    )

    axes[0].legend(
        frameon=False,
        fontsize=6.5,
    )

    # LOO robustness
    loops = [
        "EBOX",
        "RORE",
        "DBOX",
    ]

    for loop in loops:

        p = loo[
            loo["loop"] == loop
        ]

        axes[1].scatter(
            p[
                "removed_gene"
            ],
            p[
                "observation_spearman_rho"
            ],
            label=loop,
            s=35,
        )

    axes[1].axhline(
        0.9,
        color="grey",
        linestyle="--",
        lw=0.7,
    )

    axes[1].set_ylabel(
        "Spearman rho vs full score"
    )

    axes[1].set_xlabel(
        "Removed gene"
    )

    axes[1].set_title(
        "Leave-one-gene-out robustness"
    )

    axes[1].tick_params(
        axis="x",
        rotation=50,
    )

    axes[1].legend(
        frameon=False,
    )

    fig.suptitle(
        "Supplementary Fig. S2 — Score robustness",
        fontsize=12,
    )

    fig.tight_layout()

    savefig(
        fig,
        SUPP
        / "Supplementary_Fig_S2_score_robustness",
    )


# =============================================================================
# 12. SUPP S3 — FULL RESIDUALIZED NETWORK SUMMARY
# =============================================================================

def supp_S3(network):

    cols = [
        "EBOX_WITHIN_COUPLING",
        "RORE_WITHIN_COUPLING",
        "DBOX_WITHIN_COUPLING",
        "EBOX_RORE_MEAN_COUPLING",
        "EBOX_DBOX_MEAN_COUPLING",
        "RORE_DBOX_MEAN_COUPLING",
        "NETWORK_MEAN_ABS_COUPLING",
    ]

    labels = [
        "E-box\nwithin",
        "RORE\nwithin",
        "D-box\nwithin",
        "E↔R",
        "E↔D",
        "R↔D",
        "Network\nintegration",
    ]

    t = (
        network
        .set_index("lineage")
        .reindex(LINEAGE_ORDER)
    )

    arr = t[cols].to_numpy()

    vmax = max(
        0.5,
        np.nanmax(
            np.abs(arr)
        ),
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.2)
    )

    heatmap(
        ax,
        arr,
        [
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],
        labels,
        title="Tissue-residualized network summary",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        annotate=True,
        cbar=True,
        cbar_label="Residualized coupling",
    )

    fig.tight_layout()

    savefig(
        fig,
        SUPP
        / "Supplementary_Fig_S3_network_summary",
    )


# =============================================================================
# 13. SUPP S4 — DIFFERENTIAL COUPLING FOREST
# =============================================================================

def supp_S4(diff):

    d = diff.copy()

    d["abs_delta"] = (
        d[
            "delta_rho_1_minus_2"
        ]
        .abs()
    )

    d = (
        d.sort_values(
            [
                "p_adj_BH",
                "abs_delta",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(30)
        .copy()
    )

    labels = (
        d["lineage_1"]
        .map(LINEAGE_DISPLAY)
        + " vs "
        + d["lineage_2"]
        .map(LINEAGE_DISPLAY)
        + " | "
        + d["arm_1"]
        + "–"
        + d["arm_2"]
    )

    y = np.arange(
        len(d)
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(
                6,
                0.28 * len(d)
            ),
        )
    )

    ax.barh(
        y,
        d[
            "delta_rho_1_minus_2"
        ],
    )

    ax.axvline(
        0,
        color="black",
        lw=0.8,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=6.3,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Δ residualized Spearman rho"
    )

    ax.set_title(
        "Supplementary Fig. S4 — Differential coupling"
    )

    fig.tight_layout()

    savefig(
        fig,
        SUPP
        / "Supplementary_Fig_S4_differential_coupling",
    )


# =============================================================================
# 14. SOURCE DATA EXPORT
# =============================================================================

def export_source_data(tables):

    # Main A
    get_lineage_score_matrix(
        tables["summary"],
        LOOP_ORDER,
    ).to_csv(
        SOURCE
        / "MainFig_A_three_loop_engagement.csv"
    )

    # Main B
    mat, lo, hi = build_matched_matrix(
        tables["matched"]
    )

    mat.to_csv(
        SOURCE
        / "MainFig_B_matched_delta.csv"
    )

    lo.to_csv(
        SOURCE
        / "MainFig_B_matched_CI_low.csv"
    )

    hi.to_csv(
        SOURCE
        / "MainFig_B_matched_CI_high.csv"
    )

    tables["network"].to_csv(
        SOURCE
        / "MainFig_C_network_summary.csv",
        index=False,
    )

    tables["diff"].to_csv(
        SOURCE
        / "MainFig_C_differential_coupling.csv",
        index=False,
    )

    tables["variance"].to_csv(
        SOURCE
        / "MainFig_D_variance_partition.csv",
        index=False,
    )


# =============================================================================
# 15. MANIFEST
# =============================================================================

def write_manifest():

    txt = f"""
65 STATIC THREE-LOOP PUBLICATION RENDERING

SOURCE DIRECTORIES
62: {DIR62}
63: {DIR63}
64: {DIR64}

NO statistics were recomputed.
NO source analysis was modified.
NO temporal circadian inference was introduced.

MAIN FIGURE
A. Three-loop engagement
B. Same-donor same-tissue matched lineage contrasts
C. Tissue-residualized six-arm network topology
D. Lineage vs tissue variance decomposition

SUPPLEMENTARY
S1. Six-arm architecture and balance
S2. Score robustness / D-box decomposition / LOO
S3. Residualized network summary
S4. Differential coupling forest
"""

    (
        AUDIT
        / "65_RENDERING_MANIFEST.txt"
    ).write_text(
        txt.strip(),
        encoding="utf-8",
    )


# =============================================================================
# 16. MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("65 — STATIC THREE-LOOP PUBLICATION RENDERING")
    print("=" * 100)

    tables = load_tables()

    print("\nRendering main figure...")

    make_main_figure(
        tables
    )

    print("Rendering Supplementary S1...")

    supp_S1(
        tables["summary"]
    )

    print("Rendering Supplementary S2...")

    supp_S2(
        tables["dbox"],
        tables["loo"],
    )

    print("Rendering Supplementary S3...")

    supp_S3(
        tables["network"]
    )

    print("Rendering Supplementary S4...")

    supp_S4(
        tables["diff"]
    )

    export_source_data(
        tables
    )

    write_manifest()

    print()
    print("=" * 100)
    print("65 PUBLICATION RENDERING COMPLETE")
    print("=" * 100)
    print("OUTPUT:")
    print(OUT)
    print()
    print("MAIN FIGURE:")
    print(
        MAIN
        / "Figure_static_3loop_network_architecture"
    )
    print()
    print("SUPPLEMENTARY:")
    print("  Supplementary_Fig_S1_six_arm_and_balance")
    print("  Supplementary_Fig_S2_score_robustness")
    print("  Supplementary_Fig_S3_network_summary")
    print("  Supplementary_Fig_S4_differential_coupling")
    print()
    print("NO statistics recomputed.")
    print("NO 62–64 files modified.")


if __name__ == "__main__":
    main()