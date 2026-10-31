#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================================================================
# 0. PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

CANDIDATE_62 = [
    ROOT / "62_STATIC_3LOOP_ARCHITECTURE_20260930",
    ROOT / "36_STATIC_3LOOP_ARCHITECTURE_20260930",
]

DIR63 = ROOT / "63_STATIC_3LOOP_DEEPDIVE_20260930"
DIR64 = ROOT / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930"

OUT = ROOT / "66_STATIC_3LOOP_PUBLICATION_PANELS_20261001"

DIR = {
    "main": OUT / "01_MAIN_PANELS",
    "supp": OUT / "02_SUPPLEMENTARY_PANELS",
    "source": OUT / "source_data",
    "audit": OUT / "audit",
}

for p in DIR.values():
    p.mkdir(parents=True, exist_ok=True)


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the candidate paths exists:\n"
        + "\n".join(str(x) for x in paths)
    )


DIR62 = first_existing(CANDIDATE_62)


# =============================================================================
# 1. STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
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
    "macrophage_monocyte": "Macrophage/Monocyte",
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
    "EBOX_ARM_BALANCED",
    "RORE_ARM_BALANCED",
    "DBOX_ARM_BALANCED",
]

LOOP_LABELS = [
    "E-box",
    "RORE",
    "D-box",
]

BALANCE_ORDER = [
    "EBOX_BALANCE",
    "RORE_BALANCE",
    "DBOX_BALANCE",
]

BALANCE_LABELS = [
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

ARM_LABELS = [
    "E-box +",
    "E-box −",
    "ROR",
    "REV-ERB",
    "PAR-bZIP",
    "NFIL3",
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


def add_panel_letter(ax, letter):
    ax.text(
        -0.16,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def minimal_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_narrow_colorbar(ax, im, label):
    cax = inset_axes(
        ax,
        width="4.5%",
        height="55%",
        loc="center right",
        bbox_to_anchor=(0.08, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2)
    cb.set_label(label, fontsize=8)
    return cb


def draw_heatmap_like_fig2(
    ax,
    data,
    row_labels,
    col_labels,
    title,
    colorbar_label,
    vmin=None,
    vmax=None,
    cmap="RdBu_r",
    xtick_rotation=45,
):
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(
        col_labels,
        rotation=xtick_rotation,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    ax.set_title(title, fontweight="bold", pad=8)

    # mimic cleaner fig-2-like appearance
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_narrow_colorbar(ax, im, colorbar_label)

    return im


def ordered_pairs(items):
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


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

    network_summary = read_first([
        DIR64 / "03_DIFFERENTIAL_COUPLING" / "RESIDUALIZED_NETWORK_SUMMARY.csv",
        DIR64 / "RESIDUALIZED_NETWORK_SUMMARY.csv",
    ])

    coupling_edges = read_first([
        DIR64 / "03_DIFFERENTIAL_COUPLING" / "RESIDUALIZED_COUPLING_EDGES.csv",
        DIR64 / "RESIDUALIZED_COUPLING_EDGES.csv",
    ])

    diff = read_first([
        DIR64 / "03_DIFFERENTIAL_COUPLING" / "DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv",
        DIR64 / "DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv",
    ])

    variance = read_first([
        DIR63 / "03_TISSUE_DONOR_ADJUSTED" / "LINEAGE_TISSUE_VARIANCE_PARTITION.csv",
        DIR63 / "LINEAGE_TISSUE_VARIANCE_PARTITION.csv",
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
        "network_summary": network_summary,
        "coupling_edges": coupling_edges,
        "diff": diff,
        "variance": variance,
        "loo": loo,
        "dbox": dbox,
    }


# =============================================================================
# 5. MATRIX HELPERS
# =============================================================================

def make_lineage_matrix(summary, score_order):
    t = (
        summary[
            summary["score"].isin(score_order)
        ]
        .pivot(
            index="lineage",
            columns="score",
            values="mean",
        )
        .reindex(
            index=LINEAGE_ORDER,
            columns=score_order,
        )
    )
    return t


def make_matched_matrix(matched):
    use = matched[
        matched["score"].isin(LOOP_ORDER)
    ].copy()

    use["comparison"] = (
        use["lineage_1"].map(LINEAGE_DISPLAY)
        + " − "
        + use["lineage_2"].map(LINEAGE_DISPLAY)
    )

    comp_order = [
        f"{LINEAGE_DISPLAY[a]} − {LINEAGE_DISPLAY[b]}"
        for a, b in ordered_pairs(LINEAGE_ORDER)
    ]

    mat = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_mean_delta",
        )
        .reindex(
            index=comp_order,
            columns=LOOP_ORDER,
        )
    )

    lo = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_ci_low",
        )
        .reindex(
            index=comp_order,
            columns=LOOP_ORDER,
        )
    )

    hi = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_ci_high",
        )
        .reindex(
            index=comp_order,
            columns=LOOP_ORDER,
        )
    )

    return mat, lo, hi


# =============================================================================
# 6. MAIN PANELS
# =============================================================================

def plot_main_A(summary):
    t = make_lineage_matrix(summary, LOOP_ORDER)
    arr = t.to_numpy()

    vmax = np.nanmax(np.abs(arr))

    fig, ax = plt.subplots(figsize=(4.9, 4.6))

    draw_heatmap_like_fig2(
        ax=ax,
        data=arr,
        row_labels=[LINEAGE_DISPLAY[x] for x in LINEAGE_ORDER],
        col_labels=LOOP_LABELS,
        title="Static three-loop engagement",
        colorbar_label="Mean standardized score",
        vmin=-vmax,
        vmax=vmax,
    )

    add_panel_letter(ax, "A")

    savefig(fig, DIR["main"] / "Fig66A_static_three_loop_engagement")

    t.to_csv(DIR["source"] / "Fig66A_static_three_loop_engagement.csv")


def plot_main_B(matched):
    mat, lo, hi = make_matched_matrix(matched)
    arr = mat.to_numpy()
    vmax = np.nanmax(np.abs(arr))

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    draw_heatmap_like_fig2(
        ax=ax,
        data=arr,
        row_labels=mat.index.tolist(),
        col_labels=LOOP_LABELS,
        title="Same-donor, same-tissue matched lineage contrasts",
        colorbar_label="Matched donor+tissue\nmean lineage difference",
        vmin=-vmax,
        vmax=vmax,
    )

    # CI excludes 0 -> black dot
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            l = lo.iloc[i, j]
            h = hi.iloc[i, j]

            if np.isfinite(l) and np.isfinite(h):
                if (l > 0) or (h < 0):
                    ax.plot(
                        j + 0.38,
                        i - 0.35,
                        "k.",
                        markersize=3.2,
                        clip_on=False,
                    )

    ax.text(
        0.0,
        -0.16,
        "• 95% donor-block bootstrap CI excludes 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )

    add_panel_letter(ax, "B")

    savefig(fig, DIR["main"] / "Fig66B_matched_lineage_contrasts")

    mat.to_csv(DIR["source"] / "Fig66B_matched_lineage_contrasts.csv")
    lo.to_csv(DIR["source"] / "Fig66B_matched_lineage_contrasts_CI_low.csv")
    hi.to_csv(DIR["source"] / "Fig66B_matched_lineage_contrasts_CI_high.csv")


NODE_POS = {
    "EBOX_POS": (0.10, 0.78),
    "EBOX_NEG": (0.10, 0.26),

    "RORE_ACT": (0.52, 0.78),
    "RORE_REP": (0.52, 0.26),

    "DBOX_ACT": (0.90, 0.78),
    "DBOX_REP": (0.90, 0.26),
}


def significant_edge_lookup(diff):
    sig = {}
    d = diff.copy()

    for _, r in d.iterrows():
        if np.isfinite(r.get("p_adj_BH", np.nan)) and r["p_adj_BH"] < 0.05:
            pair = tuple(sorted([r["arm_1"], r["arm_2"]]))

            key1 = (r["lineage_1"], pair)
            key2 = (r["lineage_2"], pair)

            sig[key1] = True
            sig[key2] = True

    return sig


def draw_lineage_network(ax, lineage, edge_df, sig_lookup, abs_rho_min=0.30):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sub = edge_df[
        edge_df["lineage"] == lineage
    ].copy()

    # draw edges first
    for _, r in sub.iterrows():
        a = r["arm_1"]
        b = r["arm_2"]
        rho = r["rho_residualized"]

        if not np.isfinite(rho):
            continue
        if abs(rho) < abs_rho_min:
            continue

        x1, y1 = NODE_POS[a]
        x2, y2 = NODE_POS[b]

        sig = sig_lookup.get((lineage, tuple(sorted([a, b]))), False)

        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-",
            mutation_scale=1,
            linewidth=0.8 + 2.2 * abs(rho),
            linestyle="-" if rho >= 0 else "--",
            color="black",
            alpha=0.95 if sig else 0.28,
            shrinkA=7,
            shrinkB=7,
            zorder=1,
        )
        ax.add_patch(patch)

    # draw nodes
    for node, (x, y) in NODE_POS.items():
        circ = Circle(
            (x, y),
            0.055,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=3,
        )
        ax.add_patch(circ)

        ax.text(
            x,
            y,
            ARM_SHORT[node],
            ha="center",
            va="center",
            fontsize=6.5,
            zorder=4,
        )

    ax.text(
        0.5,
        0.98,
        LINEAGE_DISPLAY[lineage],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def plot_main_C(coupling_edges, diff):
    fig = plt.figure(figsize=(8.2, 5.5))
    parent = fig.add_subplot(111)
    parent.axis("off")
    add_panel_letter(parent, "C")

    parent.set_title(
        "Tissue-residualized six-arm network topology",
        fontweight="bold",
        pad=8,
    )

    sig_lookup = significant_edge_lookup(diff)

    positions = [
        (0.02, 0.53, 0.29, 0.35),
        (0.355, 0.53, 0.29, 0.35),
        (0.69, 0.53, 0.29, 0.35),

        (0.19, 0.08, 0.29, 0.35),
        (0.525, 0.08, 0.29, 0.35),
    ]

    for lineage, pos in zip(LINEAGE_ORDER, positions):
        ax = parent.inset_axes(pos)
        draw_lineage_network(
            ax=ax,
            lineage=lineage,
            edge_df=coupling_edges,
            sig_lookup=sig_lookup,
            abs_rho_min=0.30,
        )

    legend_items = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", alpha=0.85, label="positive coupling"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", alpha=0.85, label="negative coupling"),
        Line2D([0], [0], color="black", lw=2.2, linestyle="-", alpha=0.95, label="BH-significant differential edge"),
    ]

    parent.legend(
        handles=legend_items,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )

    savefig(fig, DIR["main"] / "Fig66C_residualized_six_arm_network_topology")

    coupling_edges.to_csv(
        DIR["source"] / "Fig66C_residualized_coupling_edges.csv",
        index=False,
    )
    diff.to_csv(
        DIR["source"] / "Fig66C_differential_coupling_stats.csv",
        index=False,
    )


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


def plot_main_D(variance):
    v = (
        variance
        .set_index("score")
        .reindex(VAR_SCORE_ORDER)
    )

    x = np.arange(len(v))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    ax.bar(
        x - width / 2,
        v["increment_lineage_given_tissue"],
        width,
        color="#4E79A7",
        label="Lineage | tissue",
    )
    ax.bar(
        x + width / 2,
        v["increment_tissue_given_lineage"],
        width,
        color="#F28E2B",
        label="Tissue | lineage",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(VAR_LABELS)
    ax.set_ylabel("Incremental explained variance (R²)")
    ax.set_title("Lineage backbone and tissue remodeling", fontweight="bold", pad=8)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.axhline(0, color="black", lw=0.7)

    minimal_axes(ax)
    add_panel_letter(ax, "D")

    savefig(fig, DIR["main"] / "Fig66D_lineage_backbone_vs_tissue_remodeling")

    v.to_csv(DIR["source"] / "Fig66D_variance_partition.csv")


# =============================================================================
# 7. SUPPLEMENTARY PANELS
# =============================================================================

def plot_supp_S1(summary):
    t = make_lineage_matrix(summary, ARM_ORDER)
    arr = t.to_numpy()
    vmax = np.nanmax(np.abs(arr))

    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    draw_heatmap_like_fig2(
        ax=ax,
        data=arr,
        row_labels=[LINEAGE_DISPLAY[x] for x in LINEAGE_ORDER],
        col_labels=ARM_LABELS,
        title="Static six-arm architecture",
        colorbar_label="Mean standardized score",
        vmin=-vmax,
        vmax=vmax,
    )

    add_panel_letter(ax, "S1")

    savefig(fig, DIR["supp"] / "Supplementary_Fig_S1_six_arm_architecture")

    t.to_csv(DIR["source"] / "Supplementary_Fig_S1_six_arm_architecture.csv")


def plot_supp_S2(summary):
    t = make_lineage_matrix(summary, BALANCE_ORDER)
    arr = t.to_numpy()
    vmax = np.nanmax(np.abs(arr))

    fig, ax = plt.subplots(figsize=(4.8, 4.2))

    draw_heatmap_like_fig2(
        ax=ax,
        data=arr,
        row_labels=[LINEAGE_DISPLAY[x] for x in LINEAGE_ORDER],
        col_labels=BALANCE_LABELS,
        title="Activator–repressor balance",
        colorbar_label="Balance score",
        vmin=-vmax,
        vmax=vmax,
    )

    add_panel_letter(ax, "S2")

    savefig(fig, DIR["supp"] / "Supplementary_Fig_S2_balance_heatmap")

    t.to_csv(DIR["source"] / "Supplementary_Fig_S2_balance_heatmap.csv")


def plot_supp_S3(dbox):
    d = (
        dbox
        .set_index("priority_lineage")
        .reindex(LINEAGE_ORDER)
    )

    x = np.arange(len(LINEAGE_ORDER))
    width = 0.18

    cols = [
        "DBOX_ACT",
        "DBOX_REP",
        "DBOX_GENE_BALANCED",
        "DBOX_ARM_BALANCED",
    ]

    labels = [
        "PAR-bZIP arm",
        "NFIL3",
        "Gene-balanced D-box",
        "Arm-balanced D-box",
    ]

    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    for i, (col, lab, c) in enumerate(zip(cols, labels, colors)):
        ax.bar(
            x + (i - 1.5) * width,
            d[col],
            width,
            label=lab,
            color=c,
        )

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LINEAGE_DISPLAY[i] for i in LINEAGE_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Mean standardized score")
    ax.set_title("D-box decomposition and NFIL3 sensitivity", fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=7)

    minimal_axes(ax)
    add_panel_letter(ax, "S3")

    savefig(fig, DIR["supp"] / "Supplementary_Fig_S3_dbox_decomposition")

    d.to_csv(DIR["source"] / "Supplementary_Fig_S3_dbox_decomposition.csv")


def plot_supp_S4(loo):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    loop_colors = {
        "EBOX": "#4E79A7",
        "RORE": "#59A14F",
        "DBOX": "#E15759",
    }

    for loop in ["EBOX", "RORE", "DBOX"]:
        p = loo[loo["loop"] == loop].copy()

        ax.scatter(
            p["removed_gene"],
            p["observation_spearman_rho"],
            s=42,
            color=loop_colors[loop],
            alpha=0.9,
            label=loop,
        )

    ax.axhline(0.9, color="grey", lw=0.8, linestyle="--")
    ax.set_ylabel("Spearman rho vs full score")
    ax.set_xlabel("Removed gene")
    ax.set_title("Leave-one-gene-out robustness", fontweight="bold", pad=8)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)

    minimal_axes(ax)
    add_panel_letter(ax, "S4")

    savefig(fig, DIR["supp"] / "Supplementary_Fig_S4_leave_one_gene_out_robustness")

    loo.to_csv(DIR["source"] / "Supplementary_Fig_S4_leave_one_gene_out_robustness.csv", index=False)


NETWORK_SUMMARY_COLS = [
    "EBOX_WITHIN_COUPLING",
    "RORE_WITHIN_COUPLING",
    "DBOX_WITHIN_COUPLING",
    "EBOX_RORE_MEAN_COUPLING",
    "EBOX_DBOX_MEAN_COUPLING",
    "RORE_DBOX_MEAN_COUPLING",
    "NETWORK_MEAN_ABS_COUPLING",
]

NETWORK_SUMMARY_LABELS = [
    "E-box\nwithin",
    "RORE\nwithin",
    "D-box\nwithin",
    "E-box↔RORE",
    "E-box↔D-box",
    "RORE↔D-box",
    "Network\nintegration",
]


def plot_supp_S5(network_summary):
    t = (
        network_summary
        .set_index("lineage")
        .reindex(LINEAGE_ORDER)
    )

    arr = t[NETWORK_SUMMARY_COLS].to_numpy()
    vmax = np.nanmax(np.abs(arr))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    draw_heatmap_like_fig2(
        ax=ax,
        data=arr,
        row_labels=[LINEAGE_DISPLAY[x] for x in LINEAGE_ORDER],
        col_labels=NETWORK_SUMMARY_LABELS,
        title="Residualized network summary",
        colorbar_label="Residualized coupling",
        vmin=-vmax,
        vmax=vmax,
    )

    add_panel_letter(ax, "S5")

    savefig(fig, DIR["supp"] / "Supplementary_Fig_S5_residualized_network_summary")

    t.to_csv(DIR["source"] / "Supplementary_Fig_S5_residualized_network_summary.csv")


def plot_supp_S6(diff):
    d = diff.copy()
    d["abs_delta"] = d["delta_rho_1_minus_2"].abs()

    d = d.sort_values(
        ["p_adj_BH", "abs_delta"],
        ascending=[True, False],
    ).head(30)

    labels = (
        d["lineage_1"].map(LINEAGE_DISPLAY)
        + " vs "
        + d["lineage_2"].map(LINEAGE_DISPLAY)
        + " | "
        + d["arm_1"]
        + "–"
        + d["arm_2"]
    )

    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.8, max(5.5, 0.24 * len(d))))

    colors = ["#4E79A7" if x >= 0 else "#E15759" for x in d["delta_rho_1_minus_2"]]

    ax.barh(
        y,
        d["delta_rho_1_minus_2"],
        color=colors,
    )

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ residualized Spearman rho")
    ax.set_title("Top differential coupling edges", fontweight="bold", pad=8)

    minimal_axes(ax)
    add_panel_letter(ax, "S6")

    savefig(fig, DIR["supp"] / "Supplementary_Fig_S6_differential_coupling")

    d.to_csv(DIR["source"] / "Supplementary_Fig_S6_differential_coupling.csv", index=False)


# =============================================================================
# 8. MANIFEST
# =============================================================================

def write_manifest():
    txt = f"""
66 STATIC THREE-LOOP PUBLICATION PANELS

INPUT DIRECTORIES
62: {DIR62}
63: {DIR63}
64: {DIR64}

OUTPUT
{OUT}

THIS SCRIPT:
- does NOT recompute statistics
- does NOT modify results from 62–64
- only renders publication-style standalone panels

MAIN PANELS
Fig66A_static_three_loop_engagement
Fig66B_matched_lineage_contrasts
Fig66C_residualized_six_arm_network_topology
Fig66D_lineage_backbone_vs_tissue_remodeling

SUPPLEMENTARY PANELS
Supplementary_Fig_S1_six_arm_architecture
Supplementary_Fig_S2_balance_heatmap
Supplementary_Fig_S3_dbox_decomposition
Supplementary_Fig_S4_leave_one_gene_out_robustness
Supplementary_Fig_S5_residualized_network_summary
Supplementary_Fig_S6_differential_coupling
""".strip()

    (DIR["audit"] / "66_RENDERING_MANIFEST.txt").write_text(
        txt,
        encoding="utf-8",
    )


# =============================================================================
# 9. MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("66 — STATIC THREE-LOOP PUBLICATION PANELS")
    print("=" * 100)

    tables = load_tables()

    print("\nRendering main panels...")
    plot_main_A(tables["summary"])
    plot_main_B(tables["matched"])
    plot_main_C(tables["coupling_edges"], tables["diff"])
    plot_main_D(tables["variance"])

    print("Rendering supplementary panels...")
    plot_supp_S1(tables["summary"])
    plot_supp_S2(tables["summary"])
    plot_supp_S3(tables["dbox"])
    plot_supp_S4(tables["loo"])
    plot_supp_S5(tables["network_summary"])
    plot_supp_S6(tables["diff"])

    write_manifest()

    print()
    print("=" * 100)
    print("66 PUBLICATION PANEL RENDERING COMPLETE")
    print("=" * 100)
    print("OUTPUT:", OUT)
    print()
    print("MAIN PANELS:")
    print("  Fig66A_static_three_loop_engagement")
    print("  Fig66B_matched_lineage_contrasts")
    print("  Fig66C_residualized_six_arm_network_topology")
    print("  Fig66D_lineage_backbone_vs_tissue_remodeling")
    print()
    print("SUPPLEMENTARY:")
    print("  Supplementary_Fig_S1_six_arm_architecture")
    print("  Supplementary_Fig_S2_balance_heatmap")
    print("  Supplementary_Fig_S3_dbox_decomposition")
    print("  Supplementary_Fig_S4_leave_one_gene_out_robustness")
    print("  Supplementary_Fig_S5_residualized_network_summary")
    print("  Supplementary_Fig_S6_differential_coupling")
    print()
    print("NO statistics recomputed.")
    print("NO 62–64 files modified.")


if __name__ == "__main__":
    main()