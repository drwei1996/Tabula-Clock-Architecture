#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
# Global publication style
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


LINEAGE_DISPLAY = {
    "b_cell": "B cell",
    "t_cell": "T cell",
    "macrophage_monocyte": "Macrophage/Monocyte",
    "fibroblast": "Fibroblast",
    "endothelial": "Endothelial",
}

LINEAGE_COLORS = {
    "B cell": "#4E79A7",
    "T cell": "#E15759",
    "Macrophage/Monocyte": "#59A14F",
    "Fibroblast": "#F28E2B",
    "Endothelial": "#9467BD",
}


# ============================================================
# Helpers
# ============================================================

def save(fig, base):
    for ext in [".png", ".pdf", ".svg"]:
        kwargs = dict(
            bbox_inches="tight",
            facecolor="white"
        )
        if ext == ".png":
            kwargs["dpi"] = 600

        fig.savefig(
            base.with_suffix(ext),
            **kwargs
        )

    plt.close(fig)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pretty_feature(x):
    x = str(x)

    if x.startswith("expr_"):
        return x[5:]

    mapping = {
        "core_clock_mean": "Core clock mean",
        "expanded_clock_mean": "Expanded clock mean",
        "positive_limb_mean": "Positive-limb mean",
        "negative_limb_mean": "Negative-limb mean",
        "auxiliary_clock_mean": "Auxiliary clock mean",
        "positive_limb_fraction": "Positive-limb fraction",
        "positive_negative_balance": "Positive–negative balance",
        "core_completeness_at_0.05": "Core completeness",
    }

    return mapping.get(
        x,
        x.replace("_", " ")
    )


def load_feature_table(root):
    path = (
        root
        / "02_A2"
        / "corrected_feature_effects_and_scores.tsv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    d = pd.read_csv(
        path,
        sep="\t",
        low_memory=False
    )

    required = [
        "feature",
        "lineage_stability",
        "tissue_remodeling",
        "cell_identity_unique_adj_r2",
        "tissue_context_unique_adj_r2",
        "donor_unique_adj_r2",
        "cell_identity_tissue_context_interaction_adj_r2",
    ]

    missing = [
        x for x in required
        if x not in d.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    for c in required[1:]:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    d["display_feature"] = (
        d["feature"]
        .map(pretty_feature)
    )

    d["feature_type"] = np.where(
        d["feature"]
        .astype(str)
        .str.startswith("expr_"),
        "Gene expression",
        "Derived clock feature",
    )

    return d


def matrix_from_pairs(pairs, lineage):
    q = pairs[
        pairs["lineage"].astype(str).eq(lineage)
        &
        pairs["tissue_A"].astype(str)
             .ne(pairs["tissue_B"].astype(str))
    ].copy()

    tissues = sorted(
        set(q["tissue_A"].astype(str))
        |
        set(q["tissue_B"].astype(str))
    )

    ix = {
        tissue: i
        for i, tissue in enumerate(tissues)
    }

    M = np.full(
        (len(tissues), len(tissues)),
        np.nan
    )

    np.fill_diagonal(M, 0)

    for _, r in q.iterrows():
        i = ix[str(r["tissue_A"])]
        j = ix[str(r["tissue_B"])]

        v = float(r["distance"])

        M[i, j] = v
        M[j, i] = v

    return tissues, M


# ============================================================
# Panel A
# clean dot-whisker
# ============================================================

def draw_A(root, figdir, srcdir):

    path = root / "02_A2" / "corrected_distance_summary.tsv"
    d = pd.read_csv(path, sep="\t")

    d = d[
        d["summary_type"].astype(str).eq("lineage_within")
    ].copy()

    d["display"] = (
        d["lineage"].astype(str).map(LINEAGE_DISPLAY)
    )

    order = [
        "Fibroblast",
        "Endothelial",
        "Macrophage/Monocyte",
        "T cell",
        "B cell",
    ]

    d["display"] = pd.Categorical(
        d["display"],
        categories=order,
        ordered=True
    )

    d = d.sort_values("display").reset_index(drop=True)

    # 更高一点的画布
    fig, ax = plt.subplots(figsize=(6.2, 4.55))

    y = np.arange(len(d))

    xmin = max(
        0,
        float(d["bootstrap_95_ci_low"].min()) - 0.15
    )

    xmax = (
        float(d["bootstrap_95_ci_high"].max()) + 0.35
    )

    for i, r in d.iterrows():

        name = str(r["display"])
        color = LINEAGE_COLORS[name]

        lo = float(r["bootstrap_95_ci_low"])
        med = float(r["median"])
        hi = float(r["bootstrap_95_ci_high"])

        ax.errorbar(
            med,
            i,
            xerr=np.array([[
                med - lo,
                hi - med
            ]]).T,
            fmt="o",
            ms=8.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=3
        )

        # 数值继续放在点的正上方
        ax.annotate(
            f"{med:.2f}",
            xy=(med, i),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
            clip_on=False,
            zorder=4
        )

    ax.set_yticks(y)
    ax.set_yticklabels(d["display"])

    # 关键：
    # Fibroblast 上方留出明显的 axes 内部空间
    ax.set_ylim(
        len(d) - 0.50,
        -0.85
    )

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_xlabel(
        "Median within-lineage distance across tissues"
    )


    clean_axis(ax)

    # 不允许 ax 自己画 title
    ax.set_title("")

    # 坐标区明显下移
    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        bottom=0.16,
        top=0.73
    )

    # Title 放在 figure 层，与 axes 完全分开
    fig.suptitle(
        "Lineage-specific tissue remodeling",
        x=0.28,
        y=0.975,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    save(
        fig,
        figdir / "FIG2A_lineage_specific_tissue_remodeling_FINAL"
    )

    d.to_csv(
        srcdir / "FIG2A_source.csv",
        index=False
    )

    path = root / "02_A2" / "corrected_distance_summary.tsv"
    d = pd.read_csv(path, sep="\t")

    d = d[d["summary_type"].astype(str).eq("lineage_within")].copy()
    d["display"] = d["lineage"].astype(str).map(LINEAGE_DISPLAY)

    order = ["Fibroblast", "Endothelial", "Macrophage/Monocyte", "T cell", "B cell"]
    d["display"] = pd.Categorical(d["display"], categories=order, ordered=True)
    d = d.sort_values("display").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    y = np.arange(len(d))

    xmin = max(0, float(d["bootstrap_95_ci_low"].min()) - 0.15)
    xmax = float(d["bootstrap_95_ci_high"].max()) + 0.35

    for i, r in d.iterrows():
        name = str(r["display"])
        color = LINEAGE_COLORS[name]

        lo = float(r["bootstrap_95_ci_low"])
        med = float(r["median"])
        hi = float(r["bootstrap_95_ci_high"])

        ax.errorbar(
            med, i,
            xerr=np.array([[med - lo], [hi - med]]),
            fmt="o",
            ms=8.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=3
        )

        # 数值仍然放在点正上方
        # Fibroblast 上方略高一点，但不要改到标题那层
        yoff = 16 if name == "Fibroblast" else 10

        ax.annotate(
            f"{med:.2f}",
            xy=(med, i),
            xytext=(0, yoff),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
            zorder=4,
            clip_on=False
        )

    ax.set_yticks(y)
    ax.set_yticklabels(d["display"])
    ax.set_ylim(len(d) - 0.5, -0.25)   # 顶部稍微留白
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel("Median within-lineage distance across tissues")


    clean_axis(ax)

    # 关键：不要只靠 pad，要用 y 显式抬高 title
    ax.set_title(
        "Lineage-specific tissue remodeling",
        fontweight="bold",
        fontsize=14,
        y=1.07,     # 这里才是关键
        pad=0
    )

    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.18, top=0.84)

    save(fig, figdir / "FIG2A_lineage_specific_tissue_remodeling_CLEAN")

    d.to_csv(srcdir / "FIG2A_source.csv", index=False)

    path = (
        root
        / "02_A2"
        / "corrected_distance_summary.tsv"
    )

    d = pd.read_csv(
        path,
        sep="\t"
    )

    d = d[
        d["summary_type"]
        .astype(str)
        .eq("lineage_within")
    ].copy()

    d["display"] = (
        d["lineage"]
        .astype(str)
        .map(LINEAGE_DISPLAY)
    )

    order = [
        "Fibroblast",
        "Endothelial",
        "Macrophage/Monocyte",
        "T cell",
        "B cell",
    ]

    d["display"] = pd.Categorical(
        d["display"],
        categories=order,
        ordered=True
    )

    d = (
        d.sort_values("display")
         .reset_index(drop=True)
    )

    # 高度略微增加，专门给 title 和第一行数值留空间
    fig, ax = plt.subplots(
        figsize=(5.8, 4.05)
    )

    y = np.arange(len(d))

    xmin = max(
        0,
        float(d["bootstrap_95_ci_low"].min()) - 0.15
    )

    xmax = (
        float(d["bootstrap_95_ci_high"].max())
        * 1.06
    )

    for i, r in d.iterrows():

        name = str(r["display"])
        color = LINEAGE_COLORS[name]

        lo = float(r["bootstrap_95_ci_low"])
        med = float(r["median"])
        hi = float(r["bootstrap_95_ci_high"])

        ax.errorbar(
            med,
            i,
            xerr=np.array([[
                med - lo,
                hi - med
            ]]).T,
            fmt="o",
            ms=7.8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            ecolor="#333333",
            elinewidth=1.15,
            capsize=3.5,
            capthick=1.15,
            zorder=3
        )

        # 数字仍然保持在点正上方
        ax.annotate(
            f"{med:.2f}",
            xy=(med, i),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222222",
            zorder=4
        )

    ax.set_yticks(y)
    ax.set_yticklabels(d["display"])

    # 不再简单 invert_yaxis()
    # 手工设置 ylim，在 Fibroblast 上方多留出空间
    ax.set_ylim(
        len(d) - 0.55,
        -0.55
    )

    ax.set_xlim(
        xmin,
        xmax
    )

    ax.set_xlabel(
        "Median within-lineage distance across tissues"
    )

    clean_axis(ax)

    # 不使用 axes title
    ax.set_title("")

    # 图本体进一步往下压
    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        top=0.78,
        bottom=0.18
    )

    # title 单独放在 figure 上方
    fig.text(
        0.28,
        0.965,
        "Lineage-specific tissue remodeling",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    save(
        fig,
        figdir / "FIG2A_lineage_specific_tissue_remodeling_CLEAN"
    )

    d.to_csv(
        srcdir / "FIG2A_source.csv",
        index=False
    )

    path = (
        root
        / "02_A2"
        / "corrected_distance_summary.tsv"
    )

    d = pd.read_csv(
        path,
        sep="\t"
    )

    d = d[
        d["summary_type"]
        .astype(str)
        .eq("lineage_within")
    ].copy()

    d["display"] = (
        d["lineage"]
        .astype(str)
        .map(LINEAGE_DISPLAY)
    )

    order = [
        "Fibroblast",
        "Endothelial",
        "Macrophage/Monocyte",
        "T cell",
        "B cell",
    ]

    d["display"] = pd.Categorical(
        d["display"],
        categories=order,
        ordered=True
    )

    d = (
        d.sort_values("display")
         .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(5.8, 3.65)
    )

    y = np.arange(len(d))

    xmax = (
        d["bootstrap_95_ci_high"].max()
        * 1.08
    )

    for i, r in d.iterrows():

        name = str(r["display"])
        color = LINEAGE_COLORS[name]

        lo = float(r["bootstrap_95_ci_low"])
        med = float(r["median"])
        hi = float(r["bootstrap_95_ci_high"])

        ax.errorbar(
            med,
            i,
            xerr=np.array([[
                med - lo,
                hi - med
            ]]).T,
            fmt="o",
            ms=7.8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            ecolor="#333333",
            elinewidth=1.15,
            capsize=3.5,
            capthick=1.15,
            zorder=3
        )

        # value above point, not on whisker
        ax.annotate(
            f"{med:.2f}",
            (med, i),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222222"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(d["display"])
    ax.invert_yaxis()

    ax.set_xlim(
        max(0, d["bootstrap_95_ci_low"].min() - 0.15),
        xmax
    )

    ax.set_xlabel(
        "Median within-lineage distance across tissues"
    )

    ax.set_title(
        "Lineage-specific tissue remodeling",
        loc="left",
        fontweight="bold"
    )



    clean_axis(ax)

    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        top=0.82,
        bottom=0.18
    )

    save(
        fig,
        figdir / "FIG2A_lineage_specific_tissue_remodeling_CLEAN"
    )

    d.to_csv(
        srcdir / "FIG2A_source.csv",
        index=False
    )


# ============================================================
# Panel B
# corrected 24-feature heatmap
# ============================================================

def draw_B(features, figdir, srcdir):

    effect_cols = [
        "cell_identity_unique_adj_r2",
        "tissue_context_unique_adj_r2",
        "donor_unique_adj_r2",
        "cell_identity_tissue_context_interaction_adj_r2",
    ]

    effect_labels = [
        "Cell identity",
        "Tissue",
        "Donor",
        "Identity × tissue",
    ]

    d = features.sort_values(
        [
            "lineage_stability",
            "tissue_remodeling"
        ],
        ascending=[
            False,
            False
        ],
        kind="stable"
    ).copy()

    M = d[
        effect_cols
    ].to_numpy(float)

    finite = np.abs(
        M[np.isfinite(M)]
    )

    lim = max(
        0.05,
        float(
            np.nanpercentile(
                finite,
                98
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(5.5, 6.6)
    )

    im = ax.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim
    )

    ax.set_xticks(
        range(4)
    )

    ax.set_xticklabels(
        effect_labels,
        rotation=28,
        ha="right"
    )

    ax.set_yticks(
        range(len(d))
    )

    ax.set_yticklabels(
        d["display_feature"],
        fontsize=6.5
    )

    ax.set_title(
        "Static architecture across clock-associated features",
        loc="left",
        fontweight="bold"
    )

    ax.tick_params(
        length=0
    )

    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.032,
        pad=0.035
    )

    cbar.set_label(
        "Unique / interaction adjusted R²"
    )

    fig.subplots_adjust(
        left=0.31,
        right=0.88,
        top=0.94,
        bottom=0.16
    )

    save(
        fig,
        figdir / "FIG2B_static_architecture_heatmap_CLEAN"
    )

    d.to_csv(
        srcdir / "FIG2B_source.csv",
        index=False
    )


# ============================================================
# Panel C
# proportional-width heatmaps
# ============================================================

def draw_C(root, figdir, srcdir):

    path = (
        root
        / "02_A2"
        / "corrected_between_within_distances.tsv"
    )

    pairs = pd.read_csv(
        path,
        sep="\t"
    )

    fib_labels, fib = matrix_from_pairs(
        pairs,
        "fibroblast"
    )

    b_labels, bmat = matrix_from_pairs(
        pairs,
        "b_cell"
    )

    fib_max = float(np.nanmax(fib))
    bcell_max = float(np.nanmax(bmat))

    if not np.isfinite(fib_max) or not np.isfinite(bcell_max):
        raise RuntimeError(
            f"Non-finite heatmap maximum: Fibroblast={fib_max}, B cell={bcell_max}"
        )

    vmax = max(fib_max, bcell_max)

    # Width proportional to number of tissues,
    # so individual heatmap cells have similar size.
    fig = plt.figure(
        figsize=(10.8, 5.5)
    )

    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[
            len(fib_labels),
            len(b_labels),
            0.8
        ],
        wspace=0.65
    )

    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    ax2 = fig.add_subplot(
        gs[0, 1]
    )

    cax = fig.add_subplot(
        gs[0, 2]
    )

    im = ax1.imshow(
        fib,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )

    ax2.imshow(
        bmat,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )

    for ax, labels, title in [
        (
            ax1,
            fib_labels,
            "Fibroblast"
        ),
        (
            ax2,
            b_labels,
            "B cell"
        ),
    ]:

        ax.set_title(
            title,
            fontsize=10,
            fontweight="bold",
            pad=7
        )

        ax.set_xticks(
            range(len(labels))
        )

        ax.set_xticklabels(
            labels,
            rotation=90,
            fontsize=6.2
        )

        ax.set_yticks(
            range(len(labels))
        )

        ax.set_yticklabels(
            labels,
            fontsize=6.2
        )

        ax.tick_params(
            width=0.7,
            length=3
        )

    cb = fig.colorbar(
        im,
        cax=cax
    )

    cb.set_label(
        "Clock-architecture\nEuclidean distance",
        fontsize=8
    )

    fig.suptitle(
        "Cross-tissue architecture within contrasting lineages",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=11,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        bottom=0.26,
        top=0.88
    )

    save(
        fig,
        figdir / "FIG2C_cross_tissue_contrasting_lineages_CLEAN"
    )

    pd.DataFrame(
        fib,
        index=fib_labels,
        columns=fib_labels
    ).to_csv(
        srcdir / "FIG2C_Fibroblast_matrix.csv"
    )

    pd.DataFrame(
        bmat,
        index=b_labels,
        columns=b_labels
    ).to_csv(
        srcdir / "FIG2C_Bcell_matrix.csv"
    )


# ============================================================
# Panel D
# clean scatter — no automatic label repulsion
# ============================================================

def draw_D(features, figdir, srcdir):

    d = features.copy().reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(8.4, 5.35)
    )

    gene = d[
        d["feature_type"].eq("Gene expression")
    ].copy()

    derived = d[
        d["feature_type"].eq("Derived clock feature")
    ].copy()

    # --------------------------------------------------------
    # Gene-expression points
    # --------------------------------------------------------

    ax.scatter(
        gene["lineage_stability"],
        gene["tissue_remodeling"],
        s=68,
        marker="o",
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.88,
        label="Gene expression",
        zorder=3
    )

    # --------------------------------------------------------
    # Derived-feature points
    # --------------------------------------------------------

    ax.scatter(
        derived["lineage_stability"],
        derived["tissue_remodeling"],
        s=86,
        marker="s",
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.90,
        label="Derived clock feature",
        zorder=3
    )

    # --------------------------------------------------------
    # Clean names
    # --------------------------------------------------------

    def clean_gene_name(x):
        x = str(x)

        if x.startswith("expr_"):
            return x[5:]

        return x

    gene["gene_label"] = (
        gene["feature"]
        .astype(str)
        .map(clean_gene_name)
    )

    # --------------------------------------------------------
    # Manual gene offsets
    #
    # These are deliberately fixed rather than automatically
    # repelled, so the publication figure is deterministic.
    # --------------------------------------------------------

    offsets = {
        "CRY1":  (7, 7, "left"),
        "HLF":   (-7, 7, "right"),
        "NFIL3": (7, 6, "left"),
        "RORA":  (-7, 7, "right"),

        "NPAS2": (7, 7, "left"),
        "PER1":  (-7, 7, "right"),
        "PER2":  (7, -9, "left"),
        "PER3":  (7, 7, "left"),

        "ARNTL": (7, 7, "left"),
        "CLOCK": (-7, -9, "right"),
        "TEF":   (7, -10, "left"),

        "RORC":  (7, -8, "left"),

        "NR1D1": (-7, 7, "right"),
        "NR1D2": (7, 7, "left"),

        "RORB":  (7, 8, "left"),
        "CRY2":  (-7, -7, "right"),
        "DBP":   (-7, -11, "right"),
    }

    for _, r in gene.iterrows():

        lab = str(r["gene_label"])

        dx, dy, ha = offsets.get(
            lab,
            (7, 7, "left")
        )

        ax.annotate(
            lab,
            xy=(
                float(r["lineage_stability"]),
                float(r["tissue_remodeling"])
            ),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=6.8,
            color="#222222",
            zorder=5
        )

    # --------------------------------------------------------
    # Derived features:
    # number them instead of putting long labels in the plot
    # --------------------------------------------------------

    derived = derived.sort_values(
        [
            "tissue_remodeling",
            "lineage_stability"
        ],
        ascending=[
            False,
            True
        ]
    ).reset_index(drop=True)

    derived["derived_id"] = (
        np.arange(len(derived)) + 1
    )

    derived["derived_label"] = (
        derived["display_feature"]
        .astype(str)
    )

    for _, r in derived.iterrows():

        ax.text(
            float(r["lineage_stability"]),
            float(r["tissue_remodeling"]),
            str(int(r["derived_id"])),
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color="white",
            zorder=6
        )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    xmin = float(d["lineage_stability"].min())
    xmax = float(d["lineage_stability"].max())

    ymin = float(d["tissue_remodeling"].min())
    ymax = float(d["tissue_remodeling"].max())

    xr = xmax - xmin
    yr = ymax - ymin

    ax.set_xlim(
        xmin - 0.06 * xr,
        xmax + 0.08 * xr
    )

    ax.set_ylim(
        ymin - 0.12 * yr,
        ymax + 0.13 * yr
    )

    ax.set_xlabel(
        "Tissue-held-out lineage stability"
    )

    ax.set_ylabel(
        "Within-lineage tissue remodeling"
    )

    ax.set_title(
        "Lineage backbone vs tissue remodeling",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=10
    )


    clean_axis(ax)

    # --------------------------------------------------------
    # Main legend
    # --------------------------------------------------------

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0
    )

    # --------------------------------------------------------
    # Derived-feature key
    # --------------------------------------------------------

    key_lines = [
        "Derived features"
    ]

    for _, r in derived.iterrows():
        key_lines.append(
            f"{int(r['derived_id'])}   {r['derived_label']}"
        )

    ax.text(
        1.025,
        0.76,
        "\n".join(key_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        linespacing=1.45,
        color="#222222"
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.11,
        right=0.71,
        bottom=0.15,
        top=0.90
    )

    save(
        fig,
        figdir / "FIG2D_backbone_vs_remodeling_FINAL"
    )

    # source table also contains the derived IDs used in the plot
    out = d.copy()

    id_map = dict(
        zip(
            derived["feature"],
            derived["derived_id"]
        )
    )

    out["derived_id"] = (
        out["feature"]
        .map(id_map)
    )

    out.to_csv(
        srcdir / "FIG2D_source.csv",
        index=False
    )

    d = features.copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    style = {
        "Gene expression": {
            "marker": "o",
            "color": "#4E79A7"
        },
        "Derived clock feature": {
            "marker": "s",
            "color": "#F28E2B"
        },
    }

    # ---------- 画点 ----------
    for typ, q in d.groupby("feature_type", observed=True):
        ax.scatter(
            q["lineage_stability"],
            q["tissue_remodeling"],
            s=64,
            marker=style[typ]["marker"],
            color=style[typ]["color"],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.88,
            label=typ,
            zorder=3
        )

    # ---------- 先构造稳定的标签列 ----------
    def clean_label(row):
        lab = row.get("display_feature", None)
        if pd.isna(lab) or str(lab).strip() == "":
            lab = str(row["feature"])
        lab = str(lab)
        lab = lab.replace("expr_", "")
        lab = lab.replace("_", " ")
        return lab

    d["label_text"] = d.apply(clean_label, axis=1)

    # ---------- 所有点都标注 ----------
    # 针对拥挤点做人工 offset
    manual_offsets = {
        "CRY1": (8, 8, "left"),
        "HLF": (-8, 8, "right"),
        "NFIL3": (8, 4, "left"),
        "RORA": (8, 0, "left"),
        "RORB": (8, 2, "left"),
        "RORC": (8, 0, "left"),
        "ARNTL": (8, 5, "left"),
        "CLOCK": (-8, -2, "right"),
        "TEF": (8, -10, "left"),
        "PER1": (-8, -8, "right"),
        "PER2": (8, 6, "left"),
        "PER3": (8, 6, "left"),
        "CRY2": (-8, -6, "right"),
        "NR1D1": (-8, 6, "right"),
        "NR1D2": (8, 6, "left"),
        "DBP": (-8, -6, "right"),
        "NPAS2": (8, 6, "left"),

        "core clock mean": (8, 6, "left"),
        "expanded clock mean": (-8, 6, "right"),
        "positive limb mean": (8, -8, "left"),
        "negative limb mean": (8, 8, "left"),
        "auxiliary clock mean": (8, 8, "left"),
        "positive limb fraction": (8, -8, "left"),
        "positive negative balance": (8, 8, "left"),
        "core completeness at 0.05": (8, -8, "left"),
    }

    for i, r in d.iterrows():
        x = float(r["lineage_stability"])
        y = float(r["tissue_remodeling"])
        lab = str(r["label_text"])

        if lab in manual_offsets:
            dx, dy, ha = manual_offsets[lab]
        else:
            # fallback：没有手工设定的也必须显示
            if x >= d["lineage_stability"].median():
                dx, ha = 8, "left"
            else:
                dx, ha = -8, "right"
            dy = 7 if i % 2 == 0 else -7

        ax.annotate(
            lab,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=7.1,
            color="#222222",
            zorder=5,
            clip_on=False
        )

    # ---------- 坐标范围 ----------
    xmin = float(d["lineage_stability"].min())
    xmax = float(d["lineage_stability"].max())
    ymin = float(d["tissue_remodeling"].min())
    ymax = float(d["tissue_remodeling"].max())

    xr = xmax - xmin
    yr = ymax - ymin

    ax.set_xlim(xmin - 0.06 * xr, xmax + 0.12 * xr)
    ax.set_ylim(ymin - 0.10 * yr, ymax + 0.15 * yr)

    ax.set_xlabel("Tissue-held-out lineage stability")
    ax.set_ylabel("Within-lineage tissue remodeling")

    # 标题改对
    ax.set_title(
        "Lineage backbone vs tissue remodeling",
        loc="left",
        fontweight="bold",
        fontsize=14,
        pad=10
    )


    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0
    )

    clean_axis(ax)

    fig.subplots_adjust(left=0.13, right=0.80, bottom=0.15, top=0.90)

    save(fig, figdir / "FIG2D_backbone_vs_remodeling_ALL_LABELS")

    d.to_csv(srcdir / "FIG2D_source.csv", index=False)

    d = features.copy().reset_index(drop=True)

    fig, ax = plt.subplots(
        figsize=(7.2, 5.35)
    )

    style = {
        "Gene expression": {
            "marker": "o",
            "color": "#4E79A7"
        },
        "Derived clock feature": {
            "marker": "s",
            "color": "#F28E2B"
        },
    }

    # --------------------------------------------------------
    # Draw points
    # --------------------------------------------------------

    for typ, q in d.groupby(
        "feature_type",
        observed=True
    ):

        ax.scatter(
            q["lineage_stability"],
            q["tissue_remodeling"],
            s=54,
            marker=style[typ]["marker"],
            color=style[typ]["color"],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.88,
            label=typ,
            zorder=3
        )

    # --------------------------------------------------------
    # ALL 24 labels
    # --------------------------------------------------------

    # 对容易重叠的 feature 进行人工微调。
    # 格式：
    # feature: (dx, dy, horizontal_alignment)
    #
    manual_offsets = {

        # genes
        "expr_RORA": (-7, 6, "right"),
        "expr_NFIL3": (7, 6, "left"),
        "expr_CRY1": (7, 7, "left"),
        "expr_HLF": (-7, 6, "right"),
        "expr_RORB": (7, 7, "left"),
        "expr_RORC": (7, -8, "left"),

        "expr_ARNTL": (7, 6, "left"),
        "expr_CLOCK": (-7, -9, "right"),
        "expr_TEF": (7, -10, "left"),

        "expr_NPAS2": (7, 7, "left"),
        "expr_PER1": (-7, 6, "right"),
        "expr_PER2": (7, -8, "left"),
        "expr_PER3": (7, 6, "left"),

        "expr_CRY2": (-7, -8, "right"),
        "expr_NR1D1": (-7, 7, "right"),
        "expr_NR1D2": (7, 6, "left"),
        "expr_DBP": (-7, -8, "right"),

        # derived clock features
        "core_clock_mean": (7, 7, "left"),
        "expanded_clock_mean": (-7, -8, "right"),
        "positive_limb_mean": (7, -9, "left"),
        "negative_limb_mean": (7, 7, "left"),
        "auxiliary_clock_mean": (7, 7, "left"),
        "positive_limb_fraction": (7, 7, "left"),
        "positive_negative_balance": (7, -8, "left"),
        "core_completeness_at_0.05": (7, -8, "left"),
    }

    for i, r in d.iterrows():

        feature = str(
            r["feature"]
        )

        x = float(
            r["lineage_stability"]
        )

        y = float(
            r["tissue_remodeling"]
        )

        label = str(
            r["display_feature"]
        )

        # 有人工 offset 就直接用
        if feature in manual_offsets:

            dx, dy, ha = manual_offsets[
                feature
            ]

        else:

            # fallback：所有没有预定义的 feature 也一定标
            if x > 0.48:
                dx = -7
                ha = "right"
            else:
                dx = 7
                ha = "left"

            dy = (
                7
                if i % 2 == 0
                else -8
            )

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.8,
            ha=ha,
            va="center",
            color="#222222",
            zorder=5
        )

    # --------------------------------------------------------
    # Axis padding
    # --------------------------------------------------------

    xmin = float(
        d["lineage_stability"].min()
    )

    xmax = float(
        d["lineage_stability"].max()
    )

    ymin = float(
        d["tissue_remodeling"].min()
    )

    ymax = float(
        d["tissue_remodeling"].max()
    )

    xrange = xmax - xmin
    yrange = ymax - ymin

    ax.set_xlim(
        xmin - 0.08 * xrange,
        xmax + 0.10 * xrange
    )

    ax.set_ylim(
        ymin - 0.12 * yrange,
        ymax + 0.12 * yrange
    )

    ax.set_xlabel(
        "Tissue-held-out lineage stability"
    )

    ax.set_ylabel(
        "Within-lineage tissue remodeling"
    )

    ax.set_title(
        "Lineage backbone vs tissue remodeling",
        loc="left",
        fontweight="bold",
        pad=10
    )


    # legend 放右侧图外
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.00),
        borderaxespad=0
    )

    clean_axis(ax)

    fig.subplots_adjust(
        left=0.12,
        right=0.78,
        bottom=0.15,
        top=0.91
    )

    save(
        fig,
        figdir / "FIG2D_backbone_vs_remodeling_ALL_LABELS"
    )

    d.to_csv(
        srcdir / "FIG2D_source.csv",
        index=False
    )

    d = features.copy()

    fig, ax = plt.subplots(
        figsize=(6.1, 4.7)
    )

    style = {
        "Gene expression": {
            "marker": "o",
            "color": "#4E79A7"
        },
        "Derived clock feature": {
            "marker": "s",
            "color": "#F28E2B"
        },
    }

    for typ, q in d.groupby(
        "feature_type",
        observed=True
    ):

        ax.scatter(
            q["lineage_stability"],
            q["tissue_remodeling"],
            s=56,
            marker=style[typ]["marker"],
            color=style[typ]["color"],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.85,
            label=typ,
            zorder=3
        )

    # Fixed manuscript labels only.
    # Every point is still shown; only text is reduced for readability.
    label_features = {
        "expr_RORA",
        "expr_NFIL3",
        "expr_CRY1",
        "expr_HLF",
        "expr_RORB",
        "expr_RORC",
        "expr_ARNTL",
        "expr_CLOCK",
        "expr_TEF",
    }

    label_offsets = {
        "expr_RORA": (-7, 7, "right"),
        "expr_NFIL3": (6, 6, "left"),
        "expr_CRY1": (6, 7, "left"),
        "expr_HLF": (-7, 6, "right"),
        "expr_RORB": (6, 6, "left"),
        "expr_RORC": (6, 6, "left"),
        "expr_ARNTL": (6, 6, "left"),
        "expr_CLOCK": (-6, -10, "right"),
        "expr_TEF": (6, -11, "left"),
    }

    for _, r in d.iterrows():

        feature = str(
            r["feature"]
        )

        if feature not in label_features:
            continue

        dx, dy, ha = label_offsets[
            feature
        ]

        ax.annotate(
            r["display_feature"],
            (
                r["lineage_stability"],
                r["tissue_remodeling"]
            ),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.8,
            ha=ha,
            va="center"
        )

    xmin = d["lineage_stability"].min()
    xmax = d["lineage_stability"].max()

    ymin = d["tissue_remodeling"].min()
    ymax = d["tissue_remodeling"].max()

    xpad = (
        xmax - xmin
    ) * 0.07

    ypad = (
        ymax - ymin
    ) * 0.10

    ax.set_xlim(
        xmin - xpad,
        xmax + xpad
    )

    ax.set_ylim(
        ymin - ypad,
        ymax + ypad
    )

    ax.set_xlabel(
        "Tissue-held-out lineage stability"
    )

    ax.set_ylabel(
        "Within-lineage tissue remodeling"
    )

    ax.set_title(
        "Lineage-specific tissue remodeling",
        loc="left",
        fontweight="bold",
        pad=28
    )


    # legend outside, but allocated real space
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.015, 1.00),
        borderaxespad=0
    )

    clean_axis(ax)

    fig.subplots_adjust(
        left=0.14,
        right=0.76,
        bottom=0.17,
        top=0.90
    )

    save(
        fig,
        figdir / "FIG2D_backbone_vs_remodeling_CLEAN"
    )

    d.to_csv(
        srcdir / "FIG2D_source.csv",
        index=False
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).resolve()

    out = Path(
        args.outdir
    ).resolve()

    figdir = (
        out
        / "02_figures"
    )

    srcdir = (
        out
        / "01_source_data"
    )

    figdir.mkdir(
        parents=True,
        exist_ok=True
    )

    srcdir.mkdir(
        parents=True,
        exist_ok=True
    )

    features = load_feature_table(
        root
    )

    print("=" * 90)
    print("REDRAW ALL FOUR FIG2 CANDIDATES — CLEAN VERSION")
    print("=" * 90)

    draw_A(
        root,
        figdir,
        srcdir
    )
    print("[DONE] A")

    draw_B(
        features,
        figdir,
        srcdir
    )
    print("[DONE] B")

    draw_C(
        root,
        figdir,
        srcdir
    )
    print("[DONE] C")

    draw_D(
        features,
        figdir,
        srcdir
    )
    print("[DONE] D")

    print("=" * 90)
    print("COMPLETE")
    print(figdir)
    print("=" * 90)


if __name__ == "__main__":
    main()