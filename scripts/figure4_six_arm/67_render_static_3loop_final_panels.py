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

OUT = ROOT / "67_STATIC_3LOOP_FINAL_PANELS_20261001"

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
        "None of these candidate paths exists:\n"
        + "\n".join(str(p) for p in paths)
    )


DIR62 = first_existing(CANDIDATE_62)


# =============================================================================
# 1. STYLE — MATCH FIGURE 2
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,

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


# IMPORTANT:
# 62 / 64 的真正 score 名
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

ARM_LABELS = [
    "E-box +",
    "E-box −",
    "ROR",
    "REV-ERB",
    "PAR-bZIP",
    "NFIL3",
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


# 63 variance 文件使用 ARM_BALANCED 命名
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


ARM_SHORT = {
    "EBOX_POS": "E+",
    "EBOX_NEG": "E−",
    "RORE_ACT": "ROR",
    "RORE_REP": "REV",
    "DBOX_ACT": "PAR",
    "DBOX_REP": "NFIL3",
}


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def savefig(fig, stem: Path):

    for ext in ["png", "pdf", "svg"]:

        fig.savefig(
            stem.with_suffix("." + ext),
            bbox_inches="tight",
            pad_inches=0.06,
            dpi=600 if ext == "png" else None,
        )

    plt.close(fig)


def read_first(paths):

    p = first_existing(paths)

    print("Reading:", p)

    return pd.read_csv(p)


def add_panel_letter(ax, letter):

    ax.text(
        -0.13,
        1.08,
        letter,

        transform=ax.transAxes,

        fontsize=18,
        fontweight="bold",

        ha="left",
        va="bottom",
    )


def clean_heatmap_axes(ax):

    # Figure-2-like:
    # labels retained, axes/spines/tick marks removed.

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        width=0,
    )


def add_small_colorbar(
    ax,
    im,
    label,
    width="4%",
    height="55%",
):

    cax = inset_axes(
        ax,

        width=width,
        height=height,

        loc="center right",

        bbox_to_anchor=(
            0.13,
            0,
            1,
            1,
        ),

        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cb = plt.colorbar(
        im,
        cax=cax,
    )

    cb.ax.tick_params(
        labelsize=7,
        length=2,
        width=0.6,
    )

    cb.outline.set_linewidth(
        0.7
    )

    cb.set_label(
        label,
        fontsize=8,
    )

    return cb


def fig2_heatmap(
    ax,
    arr,
    row_labels,
    col_labels,

    title,

    colorbar_label,

    cmap="RdBu_r",

    vmin=None,
    vmax=None,

    rotate_x=45,

    colorbar=True,
):

    im = ax.imshow(
        arr,

        cmap=cmap,

        vmin=vmin,
        vmax=vmax,

        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(
        np.arange(
            len(row_labels)
        )
    )

    ax.set_yticklabels(
        row_labels
    )

    ax.set_xticks(
        np.arange(
            len(col_labels)
        )
    )

    ax.set_xticklabels(
        col_labels,

        rotation=rotate_x,

        ha="right"
        if rotate_x != 0
        else "center",

        rotation_mode="anchor",
    )

    ax.set_title(
        title,

        fontweight="bold",

        pad=8,
    )

    clean_heatmap_axes(ax)

    if colorbar:

        add_small_colorbar(
            ax,
            im,
            colorbar_label,
        )

    return im


# =============================================================================
# 4. LOAD TABLES
# =============================================================================

def load_tables():

    summary = read_first([
        DIR62
        / "05_LINEAGE"
        / "STATIC_3LOOP_LINEAGE_SUMMARY.csv",
    ])

    matched = read_first([
        DIR64
        / "04_MATCHED_CONTRASTS"
        / "MATCHED_CONTRASTS_DONOR_BLOCK_BOOTSTRAP.csv",
    ])

    network_summary = read_first([
        DIR64
        / "03_DIFFERENTIAL_COUPLING"
        / "RESIDUALIZED_NETWORK_SUMMARY.csv",
    ])

    coupling_edges = read_first([
        DIR64
        / "03_DIFFERENTIAL_COUPLING"
        / "RESIDUALIZED_COUPLING_EDGES.csv",
    ])

    diff = read_first([
        DIR64
        / "03_DIFFERENTIAL_COUPLING"
        / "DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv",
    ])

    variance = read_first([
        DIR63
        / "03_TISSUE_DONOR_ADJUSTED"
        / "LINEAGE_TISSUE_VARIANCE_PARTITION.csv",
    ])

    loo = read_first([
        DIR62
        / "05_LINEAGE"
        / "STATIC_3LOOP_LOO_GENE_SENSITIVITY.csv",
    ])

    dbox = read_first([
        DIR63
        / "02_NFIL3_DBOX"
        / "DBOX_VARIANT_LINEAGE_CENTROIDS.csv",
    ])

    tissue_arch = None

    candidate = (
        DIR62
        / "06_TISSUE_STABILITY"
        / "STATIC_3LOOP_LINEAGE_TISSUE_ARCHITECTURE.csv"
    )

    if candidate.exists():

        print(
            "Reading:",
            candidate,
        )

        tissue_arch = pd.read_csv(
            candidate
        )

    return {
        "summary":
            summary,

        "matched":
            matched,

        "network_summary":
            network_summary,

        "coupling_edges":
            coupling_edges,

        "diff":
            diff,

        "variance":
            variance,

        "loo":
            loo,

        "dbox":
            dbox,

        "tissue_arch":
            tissue_arch,
    }


# =============================================================================
# 5. TABLE HELPERS
# =============================================================================

def lineage_score_matrix(
    summary,
    scores,
):

    t = (
        summary[
            summary["score"]
            .isin(scores)
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


def lineage_pair_order():

    return [
        (LINEAGE_ORDER[i],
         LINEAGE_ORDER[j])

        for i in range(
            len(LINEAGE_ORDER)
        )

        for j in range(
            i + 1,
            len(LINEAGE_ORDER)
        )
    ]


def matched_matrices(
    matched,
):

    use = matched[
        matched["score"]
        .isin(LOOP_ORDER)
    ].copy()

    use["comparison"] = (
        use["lineage_1"]
        .map(LINEAGE_DISPLAY)
        + " − "
        + use["lineage_2"]
        .map(LINEAGE_DISPLAY)
    )

    comp_order = [

        (
            LINEAGE_DISPLAY[a]
            + " − "
            + LINEAGE_DISPLAY[b]
        )

        for a, b
        in lineage_pair_order()
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
# 6. MAIN A
# =============================================================================

def plot_main_A(
    summary,
):

    t = lineage_score_matrix(
        summary,
        LOOP_ORDER,
    )

    arr = t.to_numpy(
        dtype=float
    )

    if not np.isfinite(arr).any():

        raise RuntimeError(
            "Main A is empty. "
            "Check EBOX_ENGAGEMENT / RORE_ENGAGEMENT / DBOX_ENGAGEMENT."
        )

    vmax = np.nanmax(
        np.abs(arr)
    )

    fig, ax = plt.subplots(
        figsize=(4.2, 3.6)
    )

    fig2_heatmap(
        ax=ax,

        arr=arr,

        row_labels=[
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],

        col_labels=LOOP_LABELS,

        title=(
            "Static three-loop "
            "engagement"
        ),

        colorbar_label=(
            "Mean standardized "
            "score"
        ),

        cmap="RdBu_r",

        vmin=-vmax,
        vmax=vmax,

        rotate_x=45,
    )

    add_panel_letter(
        ax,
        "A",
    )

    savefig(
        fig,
        DIR["main"]
        / "Fig67A_static_three_loop_engagement",
    )

    t.to_csv(
        DIR["source"]
        / "Fig67A_static_three_loop_engagement.csv"
    )


# =============================================================================
# 7. MAIN B
# =============================================================================

def plot_main_B(
    matched,
):

    mat, lo, hi = (
        matched_matrices(
            matched
        )
    )

    arr = mat.to_numpy(
        dtype=float
    )

    if not np.isfinite(arr).any():

        raise RuntimeError(
            "Main B is empty. "
            "Check EBOX/RORE/DBOX_ENGAGEMENT names in 64 matched table."
        )

    vmax = np.nanmax(
        np.abs(arr)
    )

    fig, ax = plt.subplots(
        figsize=(6.4, 4.8)
    )

    fig2_heatmap(
        ax=ax,

        arr=arr,

        row_labels=
            mat.index.tolist(),

        col_labels=
            LOOP_LABELS,

        title=(
            "Same-donor, same-tissue "
            "matched lineage contrasts"
        ),

        colorbar_label=(
            "Matched lineage "
            "difference"
        ),

        cmap="RdBu_r",

        vmin=-vmax,
        vmax=vmax,

        rotate_x=45,
    )

    # significance dots
    for i in range(
        arr.shape[0]
    ):

        for j in range(
            arr.shape[1]
        ):

            l = lo.iloc[i, j]
            h = hi.iloc[i, j]

            if (
                np.isfinite(l)
                and np.isfinite(h)
                and (
                    l > 0
                    or h < 0
                )
            ):

                ax.plot(
                    j + 0.34,
                    i - 0.32,

                    "k.",

                    markersize=3,
                    clip_on=False,
                )

    ax.text(
        0,
        -0.16,

        "• 95% donor-block bootstrap CI excludes 0",

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=7,
    )

    add_panel_letter(
        ax,
        "B",
    )

    savefig(
        fig,
        DIR["main"]
        / "Fig67B_same_donor_same_tissue_matched_contrasts",
    )

    mat.to_csv(
        DIR["source"]
        / "Fig67B_matched_delta.csv"
    )


# =============================================================================
# 8. MAIN C
#
# IMPORTANT:
# Replace awkward node network with compact Figure-2-style heatmap.
# =============================================================================

def plot_main_C(
    network_summary,
):

    t = (
        network_summary
        .set_index("lineage")
        .reindex(
            LINEAGE_ORDER
        )
    )

    arr = (
        t[
            NETWORK_SUMMARY_COLS
        ]
        .to_numpy(
            dtype=float
        )
    )

    vmax = np.nanmax(
        np.abs(arr)
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 3.9)
    )

    fig2_heatmap(
        ax=ax,

        arr=arr,

        row_labels=[
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],

        col_labels=
            NETWORK_SUMMARY_LABELS,

        title=(
            "Tissue-residualized "
            "network organization"
        ),

        colorbar_label=(
            "Residualized "
            "coupling"
        ),

        cmap="RdBu_r",

        vmin=-vmax,
        vmax=vmax,

        rotate_x=45,
    )

    add_panel_letter(
        ax,
        "C",
    )

    savefig(
        fig,
        DIR["main"]
        / "Fig67C_residualized_network_organization",
    )

    t[
        NETWORK_SUMMARY_COLS
    ].to_csv(
        DIR["source"]
        / "Fig67C_residualized_network_organization.csv"
    )


# =============================================================================
# 9. MAIN D
# =============================================================================

def plot_main_D(
    variance,
):

    v = (
        variance
        .set_index("score")
        .reindex(
            VAR_SCORE_ORDER
        )
    )

    x = np.arange(
        len(v)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(6.1, 4.0)
    )

    ax.bar(
        x - width / 2,

        v[
            "increment_lineage_given_tissue"
        ],

        width,

        color="#4E79A7",

        label="Lineage | tissue",
    )

    ax.bar(
        x + width / 2,

        v[
            "increment_tissue_given_lineage"
        ],

        width,

        color="#F28E2B",

        label="Tissue | lineage",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        VAR_LABELS
    )

    ax.set_ylabel(
        "Incremental explained variance (R²)"
    )

    ax.set_title(
        "Lineage backbone and tissue remodeling",

        fontweight="bold",

        pad=8,
    )

    ax.axhline(
        0,

        color="black",

        linewidth=0.7,
    )

    # Figure-2-like axis:
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        frameon=False,

        ncol=2,

        loc="upper right",
    )

    add_panel_letter(
        ax,
        "D",
    )

    savefig(
        fig,
        DIR["main"]
        / "Fig67D_lineage_backbone_tissue_remodeling",
    )

    v.to_csv(
        DIR["source"]
        / "Fig67D_variance_partition.csv"
    )


# =============================================================================
# 10. SUPPLEMENT S1 — SIX ARM
# =============================================================================

def plot_supp_S1(
    summary,
):

    t = lineage_score_matrix(
        summary,
        ARM_ORDER,
    )

    arr = t.to_numpy(
        dtype=float
    )

    vmax = np.nanmax(
        np.abs(arr)
    )

    fig, ax = plt.subplots(
        figsize=(6.1, 3.8)
    )

    fig2_heatmap(
        ax=ax,

        arr=arr,

        row_labels=[
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],

        col_labels=
            ARM_LABELS,

        title=(
            "Static six-arm "
            "clock architecture"
        ),

        colorbar_label=(
            "Mean standardized "
            "score"
        ),

        cmap="RdBu_r",

        vmin=-vmax,
        vmax=vmax,

        rotate_x=45,
    )

    # no panel letter if supplementary figures are standalone
    # add only if you want:
    # add_panel_letter(ax, "S1")

    savefig(
        fig,
        DIR["supp"]
        / "Supp67_S1_six_arm_architecture",
    )


# =============================================================================
# 11. SUPPLEMENT S2 — BALANCE
# =============================================================================

def plot_supp_S2(
    summary,
):

    t = lineage_score_matrix(
        summary,
        BALANCE_ORDER,
    )

    arr = t.to_numpy(
        dtype=float
    )

    vmax = np.nanmax(
        np.abs(arr)
    )

    fig, ax = plt.subplots(
        figsize=(4.4, 3.7)
    )

    fig2_heatmap(
        ax=ax,

        arr=arr,

        row_labels=[
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],

        col_labels=
            BALANCE_LABELS,

        title=(
            "Activator–repressor "
            "balance"
        ),

        colorbar_label=(
            "Balance score"
        ),

        cmap="RdBu_r",

        vmin=-vmax,
        vmax=vmax,

        rotate_x=45,
    )

    savefig(
        fig,
        DIR["supp"]
        / "Supp67_S2_loop_balance",
    )


# =============================================================================
# 12. SUPPLEMENT S3 — DBOX
# =============================================================================

def plot_supp_S3(
    dbox,
):

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

    width = 0.18

    columns = [
        "DBOX_ACT",
        "DBOX_REP",
        "DBOX_GENE_BALANCED",
        "DBOX_ARM_BALANCED",
    ]

    labels = [
        "PAR-bZIP arm",
        "NFIL3",
        "Gene-balanced",
        "Arm-balanced",
    ]

    colors = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
    ]

    fig, ax = plt.subplots(
        figsize=(6.8, 4.0)
    )

    for i, (
        col,
        label,
        color,
    ) in enumerate(
        zip(
            columns,
            labels,
            colors,
        )
    ):

        ax.bar(
            x
            + (
                i - 1.5
            )
            * width,

            d[col],

            width,

            label=label,

            color=color,
        )

    ax.axhline(
        0,
        color="black",
        linewidth=0.7,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],

        rotation=25,

        ha="right",
    )

    ax.set_ylabel(
        "Mean standardized score"
    )

    ax.set_title(
        "D-box decomposition and NFIL3 sensitivity",

        fontweight="bold",
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        frameon=False,
    )

    savefig(
        fig,
        DIR["supp"]
        / "Supp67_S3_Dbox_decomposition",
    )


# =============================================================================
# 13. SUPPLEMENT S4 — LOO
# =============================================================================

def plot_supp_S4(
    loo,
):

    colors = {
        "EBOX":
            "#4E79A7",

        "RORE":
            "#59A14F",

        "DBOX":
            "#E15759",
    }

    fig, ax = plt.subplots(
        figsize=(7.0, 4.0)
    )

    for loop in [
        "EBOX",
        "RORE",
        "DBOX",
    ]:

        p = loo[
            loo["loop"]
            == loop
        ]

        ax.scatter(
            p["removed_gene"],

            p[
                "observation_spearman_rho"
            ],

            s=36,

            color=colors[loop],

            label=loop,

            zorder=3,
        )

    ax.axhline(
        0.9,

        linestyle="--",

        color="grey",

        linewidth=0.8,
    )

    ax.set_ylabel(
        "Spearman rho vs full score"
    )

    ax.set_xlabel(
        "Removed gene"
    )

    ax.set_title(
        "Leave-one-gene-out robustness",

        fontweight="bold",
    )

    ax.tick_params(
        axis="x",

        rotation=45,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        frameon=False,
    )

    savefig(
        fig,
        DIR["supp"]
        / "Supp67_S4_leave_one_gene_out",
    )


# =============================================================================
# 14. SUPPLEMENT S5 — FULL NETWORK SUMMARY
#
# Same visual rule: NO axis spines / tick marks.
# =============================================================================

def plot_supp_S5(
    network_summary,
):

    t = (
        network_summary
        .set_index(
            "lineage"
        )
        .reindex(
            LINEAGE_ORDER
        )
    )

    arr = (
        t[
            NETWORK_SUMMARY_COLS
        ]
        .to_numpy(
            dtype=float
        )
    )

    vmax = np.nanmax(
        np.abs(arr)
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 3.9)
    )

    fig2_heatmap(
        ax=ax,

        arr=arr,

        row_labels=[
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ],

        col_labels=
            NETWORK_SUMMARY_LABELS,

        title=(
            "Residualized network "
            "summary"
        ),

        colorbar_label=(
            "Residualized "
            "coupling"
        ),

        cmap="RdBu_r",

        vmin=-vmax,
        vmax=vmax,

        rotate_x=45,
    )

    savefig(
        fig,
        DIR["supp"]
        / "Supp67_S5_residualized_network_summary",
    )


# =============================================================================
# 15. SUPPLEMENT S6 — FULL DIFFERENTIAL COUPLING
# =============================================================================

def plot_supp_S6(
    diff,
):

    d = diff.copy()

    d["abs_delta"] = (
        d[
            "delta_rho_1_minus_2"
        ]
        .abs()
    )

    # only BH significant first
    sig = d[
        d["p_adj_BH"]
        < 0.05
    ].copy()

    if len(sig) == 0:

        sig = (
            d.sort_values(
                "abs_delta",
                ascending=False,
            )
            .head(20)
        )

    else:

        sig = (
            sig.sort_values(
                "abs_delta",
                ascending=False,
            )
            .head(25)
        )

    labels = (
        sig["lineage_1"]
        .map(LINEAGE_DISPLAY)

        + " vs "

        + sig["lineage_2"]
        .map(LINEAGE_DISPLAY)

        + " | "

        + sig["arm_1"]

        + "–"

        + sig["arm_2"]
    )

    y = np.arange(
        len(sig)
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,

            max(
                5,
                0.27 * len(sig)
            ),
        )
    )

    colors = np.where(
        sig[
            "delta_rho_1_minus_2"
        ]
        >= 0,

        "#4E79A7",

        "#E15759",
    )

    ax.barh(
        y,

        sig[
            "delta_rho_1_minus_2"
        ],

        color=colors,
    )

    ax.axvline(
        0,

        color="black",

        linewidth=0.7,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels,

        fontsize=6.5,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Δ tissue-residualized Spearman rho"
    )

    ax.set_title(
        "Differential six-arm coupling",

        fontweight="bold",
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    savefig(
        fig,
        DIR["supp"]
        / "Supp67_S6_differential_coupling",
    )


# =============================================================================
# 16. OPTIONAL S7 — DETAILED NETWORKS
#
# Detailed network is now supplementary only.
# =============================================================================

NODE_POS = {
    "EBOX_POS":
        (0.10, 0.78),

    "EBOX_NEG":
        (0.10, 0.24),

    "RORE_ACT":
        (0.50, 0.78),

    "RORE_REP":
        (0.50, 0.24),

    "DBOX_ACT":
        (0.90, 0.78),

    "DBOX_REP":
        (0.90, 0.24),
}


def differential_significant_lookup(
    diff,
):

    lookup = {}

    sig = diff[
        diff["p_adj_BH"]
        < 0.05
    ]

    for _, r in (
        sig.iterrows()
    ):

        edge = tuple(
            sorted(
                [
                    r["arm_1"],
                    r["arm_2"],
                ]
            )
        )

        lookup[
            (
                r["lineage_1"],
                edge,
            )
        ] = True

        lookup[
            (
                r["lineage_2"],
                edge,
            )
        ] = True

    return lookup


def draw_network(
    ax,
    lineage,
    coupling_edges,
    sig_lookup,
):

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.axis(
        "off"
    )

    p = coupling_edges[
        coupling_edges["lineage"]
        == lineage
    ]

    for _, r in (
        p.iterrows()
    ):

        rho = r[
            "rho_residualized"
        ]

        if (
            not np.isfinite(rho)
            or abs(rho) < 0.30
        ):

            continue

        a = r["arm_1"]
        b = r["arm_2"]

        edge = tuple(
            sorted(
                [a, b]
            )
        )

        significant = (
            sig_lookup.get(
                (
                    lineage,
                    edge,
                ),
                False,
            )
        )

        x1, y1 = NODE_POS[a]
        x2, y2 = NODE_POS[b]

        line = FancyArrowPatch(
            (x1, y1),
            (x2, y2),

            arrowstyle="-",

            linewidth=(
                0.8
                + 2
                * abs(rho)
            ),

            linestyle=(
                "-"
                if rho >= 0
                else "--"
            ),

            color="black",

            alpha=(
                0.9
                if significant
                else 0.28
            ),

            shrinkA=8,
            shrinkB=8,

            zorder=1,
        )

        ax.add_patch(
            line
        )

    for node, (
        x,
        y,
    ) in (
        NODE_POS.items()
    ):

        circle = Circle(
            (x, y),

            0.055,

            facecolor="white",

            edgecolor="black",

            linewidth=0.9,

            zorder=3,
        )

        ax.add_patch(
            circle
        )

        ax.text(
            x,
            y,

            ARM_SHORT[node],

            ha="center",
            va="center",

            fontsize=6.5,

            zorder=4,
        )

    ax.set_title(
        LINEAGE_DISPLAY[lineage],

        fontsize=8,

        fontweight="bold",
    )


def plot_supp_S7(
    coupling_edges,
    diff,
):

    fig = plt.figure(
        figsize=(8.2, 5.1)
    )

    parent = fig.add_subplot(
        111
    )

    parent.axis(
        "off"
    )

    parent.set_title(
        "Tissue-residualized six-arm coupling networks",

        fontweight="bold",

        pad=8,
    )

    sig_lookup = (
        differential_significant_lookup(
            diff
        )
    )

    positions = [
        (
            0.02,
            0.54,
            0.29,
            0.34,
        ),

        (
            0.355,
            0.54,
            0.29,
            0.34,
        ),

        (
            0.69,
            0.54,
            0.29,
            0.34,
        ),

        (
            0.19,
            0.10,
            0.29,
            0.34,
        ),

        (
            0.525,
            0.10,
            0.29,
            0.34,
        ),
    ]

    for lineage, pos in zip(
        LINEAGE_ORDER,
        positions,
    ):

        ax = parent.inset_axes(
            pos
        )

        draw_network(
            ax,

            lineage,

            coupling_edges,

            sig_lookup,
        )

    legend = [
        Line2D(
            [0],
            [0],

            color="black",

            linewidth=1.5,

            label="Positive",
        ),

        Line2D(
            [0],
            [0],

            color="black",

            linestyle="--",

            linewidth=1.5,

            label="Negative",
        ),

        Line2D(
            [0],
            [0],

            color="black",

            linewidth=2.5,

            label="Differential edge",
        ),
    ]

    parent.legend(
        handles=legend,

        frameon=False,

        ncol=3,

        loc="lower center",
    )

    savefig(
        fig,
        DIR["supp"]
        / "Supp67_S7_detailed_six_arm_networks",
    )


# =============================================================================
# 17. MAIN
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "67 — STATIC THREE-LOOP FINAL PANEL RENDERING"
    )

    print(
        "=" * 100
    )

    tables = (
        load_tables()
    )

    print(
        "\nRendering MAIN panels..."
    )

    plot_main_A(
        tables["summary"]
    )

    plot_main_B(
        tables["matched"]
    )

    plot_main_C(
        tables[
            "network_summary"
        ]
    )

    plot_main_D(
        tables["variance"]
    )

    print(
        "Rendering SUPPLEMENTARY panels..."
    )

    plot_supp_S1(
        tables["summary"]
    )

    plot_supp_S2(
        tables["summary"]
    )

    plot_supp_S3(
        tables["dbox"]
    )

    plot_supp_S4(
        tables["loo"]
    )

    plot_supp_S5(
        tables[
            "network_summary"
        ]
    )

    plot_supp_S6(
        tables["diff"]
    )

    plot_supp_S7(
        tables[
            "coupling_edges"
        ],
        tables["diff"],
    )

    print()
    print(
        "=" * 100
    )

    print(
        "67 FINAL RENDERING COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "OUTPUT:"
    )

    print(
        OUT
    )

    print()

    print(
        "MAIN:"
    )

    print(
        " Fig67A_static_three_loop_engagement"
    )

    print(
        " Fig67B_same_donor_same_tissue_matched_contrasts"
    )

    print(
        " Fig67C_residualized_network_organization"
    )

    print(
        " Fig67D_lineage_backbone_tissue_remodeling"
    )

    print()

    print(
        "SUPPLEMENTARY:"
    )

    print(
        " Supp67_S1_six_arm_architecture"
    )

    print(
        " Supp67_S2_loop_balance"
    )

    print(
        " Supp67_S3_Dbox_decomposition"
    )

    print(
        " Supp67_S4_leave_one_gene_out"
    )

    print(
        " Supp67_S5_residualized_network_summary"
    )

    print(
        " Supp67_S6_differential_coupling"
    )

    print(
        " Supp67_S7_detailed_six_arm_networks"
    )

    print()
    print(
        "NO statistics recomputed."
    )


if __name__ == "__main__":
    main()