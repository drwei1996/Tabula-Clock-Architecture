#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================================================================
# 0. PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

DIR64 = (
    ROOT
    / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930"
)

OUT = (
    ROOT
    / "68_NETWORK_MATCHED_RERENDER_20261001"
)

FIG = OUT / "figures"
SOURCE = OUT / "source_data"

FIG.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)


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

ARM_SHORT = {
    "EBOX_POS": "E+",
    "EBOX_NEG": "E−",
    "RORE_ACT": "ROR",
    "RORE_REP": "REV",
    "DBOX_ACT": "PAR",
    "DBOX_REP": "NFIL3",
}

NODE_POS = {
    "EBOX_POS": (0.12, 0.76),
    "EBOX_NEG": (0.12, 0.24),

    "RORE_ACT": (0.50, 0.76),
    "RORE_REP": (0.50, 0.24),

    "DBOX_ACT": (0.88, 0.76),
    "DBOX_REP": (0.88, 0.24),
}

NODE_LOOP = {
    "EBOX_POS": "EBOX",
    "EBOX_NEG": "EBOX",

    "RORE_ACT": "RORE",
    "RORE_REP": "RORE",

    "DBOX_ACT": "DBOX",
    "DBOX_REP": "DBOX",
}

LOOP_FACE = {
    "EBOX": "#EEF4FB",
    "RORE": "#FDF1E3",
    "DBOX": "#EEF8EE",
}

LOOP_EDGE = {
    "EBOX": "#CCDDED",
    "RORE": "#EFD7BA",
    "DBOX": "#CEE6CE",
}


# =============================================================================
# 3. HELPERS
# =============================================================================

def savefig(fig, stem: Path):

    for ext in ["png", "pdf", "svg"]:

        fig.savefig(
            stem.with_suffix("." + ext),
            bbox_inches="tight",
            pad_inches=0.08,
            dpi=600 if ext == "png" else None,
        )

    plt.close(fig)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    print("Reading:", path)
    return pd.read_csv(path)


def clean_heatmap_axes(ax):

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        width=0,
    )


def add_colorbar(ax, im, label):

    cax = inset_axes(
        ax,
        width="4%",
        height="55%",
        loc="center right",
        bbox_to_anchor=(0.13, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    cb = plt.colorbar(im, cax=cax)

    cb.ax.tick_params(
        labelsize=7,
        length=2,
        width=0.6,
    )

    cb.outline.set_linewidth(0.7)

    cb.set_label(
        label,
        fontsize=8,
    )


# =============================================================================
# 4. LOAD DATA
# =============================================================================

def load_data():

    matched = read_csv(
        DIR64
        / "04_MATCHED_CONTRASTS"
        / "MATCHED_CONTRASTS_DONOR_BLOCK_BOOTSTRAP.csv"
    )

    coupling_edges = read_csv(
        DIR64
        / "03_DIFFERENTIAL_COUPLING"
        / "RESIDUALIZED_COUPLING_EDGES.csv"
    )

    diff = read_csv(
        DIR64
        / "03_DIFFERENTIAL_COUPLING"
        / "DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv"
    )

    return matched, coupling_edges, diff


# =============================================================================
# 5. MATCHED HEATMAP
# =============================================================================

def lineage_pair_order():

    return [
        (LINEAGE_ORDER[i], LINEAGE_ORDER[j])

        for i in range(len(LINEAGE_ORDER))
        for j in range(i + 1, len(LINEAGE_ORDER))
    ]


def make_matched_matrix(matched):

    use = matched[
        matched["score"].isin(LOOP_ORDER)
    ].copy()

    use["comparison"] = (
        use["lineage_1"].map(LINEAGE_DISPLAY)
        + " − "
        + use["lineage_2"].map(LINEAGE_DISPLAY)
    )

    order = [
        LINEAGE_DISPLAY[a]
        + " − "
        + LINEAGE_DISPLAY[b]

        for a, b in lineage_pair_order()
    ]

    mat = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_mean_delta",
        )
        .reindex(
            index=order,
            columns=LOOP_ORDER,
        )
    )

    return mat


def plot_matched_heatmap(matched):

    mat = make_matched_matrix(
        matched
    )

    arr = mat.to_numpy(
        dtype=float
    )

    vmax = np.nanmax(
        np.abs(arr)
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 4.8)
    )

    im = ax.imshow(
        arr,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_yticks(
        np.arange(
            len(mat.index)
        )
    )

    ax.set_yticklabels(
        mat.index
    )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        LOOP_LABELS,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_title(
        "Same-donor, same-tissue matched lineage contrasts",
        fontweight="bold",
        pad=8,
    )

    clean_heatmap_axes(ax)

    add_colorbar(
        ax,
        im,
        "Matched lineage difference",
    )

    savefig(
        fig,
        FIG
        / "Fig68A_same_donor_same_tissue_matched_contrasts",
    )

    mat.to_csv(
        SOURCE
        / "Fig68A_same_donor_same_tissue_matched_contrasts.csv"
    )


# =============================================================================
# 6. NETWORK HELPERS
# =============================================================================

def differential_significant_lookup(diff):

    lookup = {}

    sig = diff[
        diff["p_adj_BH"] < 0.05
    ].copy()

    for _, r in sig.iterrows():

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


def edge_rad(a, b):

    x1, y1 = NODE_POS[a]
    x2, y2 = NODE_POS[b]

    if abs(y1 - y2) < 1e-6:
        return 0.0

    if abs(x1 - x2) < 1e-6:
        return 0.0

    if (
        (x1 < x2 and y1 > y2)
        or
        (x2 < x1 and y2 > y1)
    ):
        return -0.08

    return 0.08


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

    ax.axis("off")

    # -------------------------------------------------------------------------
    # module backgrounds
    # -------------------------------------------------------------------------

    modules = [
        (
            "EBOX",
            0.025,
            0.08,
            0.19,
            0.82,
            "E-box",
        ),
        (
            "RORE",
            0.405,
            0.08,
            0.19,
            0.82,
            "RORE",
        ),
        (
            "DBOX",
            0.785,
            0.08,
            0.19,
            0.82,
            "D-box",
        ),
    ]

    for (
        key,
        x,
        y,
        width,
        height,
        label,
    ) in modules:

        patch = FancyBboxPatch(
            (x, y),
            width,
            height,

            boxstyle=(
                "round,pad=0.012,"
                "rounding_size=0.025"
            ),

            linewidth=0.7,

            edgecolor=
                LOOP_EDGE[key],

            facecolor=
                LOOP_FACE[key],

            zorder=0,
        )

        ax.add_patch(
            patch
        )

        ax.text(
            x + width / 2,
            0.035,

            label,

            ha="center",
            va="bottom",

            fontsize=6.2,
            color="#555555",
        )

    # -------------------------------------------------------------------------
    # edges
    # -------------------------------------------------------------------------

    p = coupling_edges[
        coupling_edges["lineage"] == lineage
    ].copy()

    p["abs_rho"] = (
        p["rho_residualized"].abs()
    )

    p = p.sort_values(
        "abs_rho"
    )

    for _, r in p.iterrows():

        rho = r[
            "rho_residualized"
        ]

        if not np.isfinite(rho):
            continue

        if abs(rho) < 0.30:
            continue

        a = r["arm_1"]
        b = r["arm_2"]

        edge = tuple(
            sorted(
                [a, b]
            )
        )

        is_diff = sig_lookup.get(
            (
                lineage,
                edge,
            ),
            False,
        )

        x1, y1 = NODE_POS[a]
        x2, y2 = NODE_POS[b]

        if is_diff:

            color = "#111111"

            linewidth = (
                1.3
                + 2.0 * abs(rho)
            )

            linestyle = "-"

            alpha = 0.92

            zorder = 3

        else:

            if rho >= 0:

                color = "#4E79A7"

                linestyle = "-"

            else:

                color = "#E15759"

                linestyle = (
                    0,
                    (4, 2),
                )

            linewidth = (
                0.7
                + 1.5 * abs(rho)
            )

            alpha = 0.42

            zorder = 1

        line = FancyArrowPatch(
            (x1, y1),
            (x2, y2),

            arrowstyle="-",

            linewidth=linewidth,

            linestyle=linestyle,

            color=color,

            alpha=alpha,

            shrinkA=10,
            shrinkB=10,

            connectionstyle=(
                f"arc3,rad="
                f"{edge_rad(a, b)}"
            ),

            zorder=zorder,
        )

        ax.add_patch(
            line
        )

    # -------------------------------------------------------------------------
    # nodes
    # -------------------------------------------------------------------------

    for node, (
        x,
        y,
    ) in NODE_POS.items():

        loop = NODE_LOOP[node]

        outer = Circle(
            (x, y),

            0.058,

            facecolor="white",

            edgecolor="black",

            linewidth=0.9,

            zorder=5,
        )

        ax.add_patch(
            outer
        )

        inner = Circle(
            (x, y),

            0.046,

            facecolor=
                LOOP_FACE[loop],

            edgecolor="none",

            zorder=5,
        )

        ax.add_patch(
            inner
        )

        ax.text(
            x,
            y,

            ARM_SHORT[node],

            ha="center",
            va="center",

            fontsize=6.6,

            zorder=6,
        )

    ax.set_title(
        LINEAGE_DISPLAY[
            lineage
        ],

        fontsize=8.5,

        fontweight="bold",

        pad=3,
    )


# =============================================================================
# 7. NETWORK FIGURE
# =============================================================================

def plot_networks(
    coupling_edges,
    diff,
):

    sig_lookup = (
        differential_significant_lookup(
            diff
        )
    )

    fig = plt.figure(
        figsize=(9.5, 5.4)
    )

    parent = (
        fig.add_subplot(111)
    )

    parent.axis("off")

    parent.set_title(
        "Tissue-residualized six-arm coupling networks",

        fontweight="bold",

        fontsize=12,

        pad=10,
    )

    positions = [
        (
            0.02,
            0.53,
            0.29,
            0.34,
        ),

        (
            0.355,
            0.53,
            0.29,
            0.34,
        ),

        (
            0.69,
            0.53,
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
            ax=ax,

            lineage=lineage,

            coupling_edges=
                coupling_edges,

            sig_lookup=
                sig_lookup,
        )

    legend = [
        Line2D(
            [0],
            [0],

            color="#4E79A7",

            linewidth=2,

            linestyle="-",

            label="Positive coupling",
        ),

        Line2D(
            [0],
            [0],

            color="#E15759",

            linewidth=2,

            linestyle=(
                0,
                (4, 2),
            ),

            label="Negative coupling",
        ),

        Line2D(
            [0],
            [0],

            color="#111111",

            linewidth=3,

            linestyle="-",

            label="Differential edge",
        ),
    ]

    parent.legend(
        handles=legend,

        frameon=False,

        ncol=3,

        loc="lower center",

        bbox_to_anchor=(
            0.5,
            -0.015,
        ),
    )

    savefig(
        fig,
        FIG
        / "Fig68B_six_arm_coupling_networks",
    )

    coupling_edges.to_csv(
        SOURCE
        / "Fig68B_residualized_coupling_edges.csv",

        index=False,
    )

    diff.to_csv(
        SOURCE
        / "Fig68B_differential_coupling_stats.csv",

        index=False,
    )


# =============================================================================
# 8. MAIN
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "68 — RERENDER MATCHED HEATMAP + NETWORK"
    )

    print(
        "=" * 100
    )

    (
        matched,
        coupling_edges,
        diff,
    ) = load_data()

    print(
        "\nRendering matched heatmap..."
    )

    plot_matched_heatmap(
        matched
    )

    print(
        "Rendering six-arm networks..."
    )

    plot_networks(
        coupling_edges,
        diff,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "68 RENDER COMPLETE"
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
        "Fig68A_same_donor_same_tissue_matched_contrasts"
    )

    print(
        "Fig68B_six_arm_coupling_networks"
    )

    print()
    print(
        "No statistics recomputed."
    )


if __name__ == "__main__":
    main()