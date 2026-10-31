#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
48_run_tissue_responsive_genes_and_pathway_enrichment.py

Question
--------
Within each CURRENT major lineage, which genes and pathways show broad
tissue-associated expression remodeling after accounting for donor?

This analysis is deliberately DIFFERENT from A4.7:
- It does NOT correlate pathway distance with clock-architecture distance.
- It does NOT use the matched clock-specificity null.
- It directly tests a tissue main effect on gene expression within lineage,
  then performs pathway over-representation analysis (ORA) on tissue-responsive genes.

Primary gene model, separately within each major lineage
---------------------------------------------------------
For each gene on donor x tissue x major-lineage aggregates:

    Full:     expression ~ donor + tissue
    Reduced:  expression ~ donor

The tissue term is tested by a nested partial F test.
Because "tissue" is multi-level, this is an OMNIBUS differential-expression
test with no single up/down direction.

Primary tissue-responsive-gene definition
-----------------------------------------
    gene FDR < 0.05
    AND partial omega^2 >= MIN_PARTIAL_OMEGA2

The effect-size gate avoids defining biologically meaningful remodeling only
from P values in a high-dimensional atlas. Sensitivity results are exported for
multiple omega^2 thresholds.

Aggregation / support
---------------------
- Uses frozen corrected metadata and the frozen 2030 x 59410 expression matrix.
- CURRENT five major lineages only.
- Equal-pseudobulk mean for each donor x tissue x major-lineage aggregate,
  consistent with the existing broad-lineage aggregation workflow.
- A lineage x tissue is retained only if represented by >=3 donors.
- Sparse/constant genes are testability-filtered before the omnibus F test.
- BH correction is applied across ALL 59,410 genes per lineage, assigning
  untestable genes P=1 (conservative).

Pathway enrichment
------------------
- Hallmark: complete 50-set screen.
- Reactome: complete 1839-set screen.
- Fisher exact test, one-sided enrichment.
- BH within lineage x collection.
- Reactome redundancy reduction is DISPLAY-ONLY and happens AFTER full testing.

Main figures
------------
Fig3E: Tissue-responsive gene burden across major lineages.
Fig3F: Hallmark enrichment landscape of tissue-responsive genes.
Fig3G: Representative non-redundant Reactome enrichment landscape.

Supplementary
-------------
- Gene-burden sensitivity to omega^2 threshold.
- Complete gene statistics and complete pathway statistics are exported.

IMPORTANT
---------
This is static expression. It does NOT establish rhythmicity, phase, amplitude,
oscillation, causality, or clock-specific regulation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import fisher_exact

import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
FREEZE = ROOT / "00_freeze"

FROZEN_META = FREEZE / "observation_metadata_corrected_v3_FROZEN.tsv"

DEP = (
    PROJECT
    / "results"
    / "phaseA_clock_architecture"
    / "dependencies"
)

NPZ_PATH = DEP / "broad_normalized_mean.npz"
HALLMARK_GMT = DEP / "h.all.v2026.1.Hs.symbols.gmt"
REACTOME_GMT = DEP / "c2.cp.reactome.v2026.1.Hs.symbols.gmt"

OUT = ROOT / "48_TISSUE_RESPONSIVE_DEG_ENRICHMENT_20260921"
FIG = OUT / "figures"
TABLES = OUT / "tables"
GENE_DIR = TABLES / "gene_level"
PATHWAY_DIR = TABLES / "pathway_level"
AUDIT = OUT / "audit"

for d in [OUT, FIG, TABLES, GENE_DIR, PATHWAY_DIR, AUDIT]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================

LINEAGES = [
    "b_cell",
    "t_cell",
    "macrophage_monocyte",
    "fibroblast",
    "endothelial",
]

LINEAGE_LABEL = {
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

EXPECTED_LINEAGE_COUNTS = {
    "b_cell": 153,
    "t_cell": 437,
    "macrophage_monocyte": 280,
    "fibroblast": 148,
    "endothelial": 206,
}

EXPECTED_ROWS = 2030
EXPECTED_GENES = 59410
EXPECTED_HALLMARK = 50
EXPECTED_REACTOME = 1839

SENSITIVITY_OMEGA2 = [0.00, 0.02, 0.05, 0.10]

EPS = 1e-12


# =============================================================================
# STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# UTILITIES
# =============================================================================

def bh(p):
    """Benjamini-Hochberg FDR; preserves NaN as 1."""
    p = np.asarray(p, dtype=float)
    q = np.ones_like(p, dtype=float)

    finite = np.isfinite(p)
    pv = np.clip(p[finite], 0.0, 1.0)

    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    tmp = np.empty_like(adj)
    tmp[order] = adj
    q[finite] = tmp

    return q


def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.12,
            "facecolor": "white",
        }
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def parse_gmt(path, collection):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            z = line.rstrip("\n").split("\t")
            if len(z) < 3:
                continue
            rows.append({
                "collection": collection,
                "pathway": z[0],
                "description": z[1],
                "genes": tuple(dict.fromkeys(g for g in z[2:] if g)),
            })
    return rows


def orthonormal_basis(X):
    """
    SVD basis robust to rank deficiency.
    Returns Q spanning the column space and numerical rank.
    """
    X = np.asarray(X, dtype=float)

    u, s, _ = np.linalg.svd(X, full_matrices=False)

    if len(s) == 0 or s[0] <= 0:
        return np.empty((len(X), 0), dtype=float), 0

    tol = s[0] * max(X.shape) * np.finfo(float).eps
    rank = int(np.sum(s > tol))

    return u[:, :rank], rank


def sse_after_projection(Y, Q):
    """
    SSE for every column of Y after projection onto col(Q),
    without explicitly estimating coefficients.
    """
    y2 = np.sum(Y.astype(np.float64) ** 2, axis=0)

    if Q.shape[1] == 0:
        return y2

    proj = Q.T @ Y
    ss_fit = np.sum(proj.astype(np.float64) ** 2, axis=0)

    return np.maximum(y2 - ss_fit, 0.0)


def corrected_log2_or(a, b, c, d):
    """
    Haldane-Anscombe corrected odds ratio for stable plotting.
    """
    return math.log2(
        ((a + 0.5) * (d + 0.5))
        / ((b + 0.5) * (c + 0.5))
    )


def pathway_jaccard(genes_a, genes_b):
    a = set(genes_a)
    b = set(genes_b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# =============================================================================
# LOAD + ALIGN FROZEN INPUTS
# =============================================================================

def load_inputs():
    required = [FROZEN_META, NPZ_PATH, HALLMARK_GMT, REACTOME_GMT]
    missing = [p for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required input(s):\n" + "\n".join(str(p) for p in missing)
        )

    meta = pd.read_csv(
        FROZEN_META,
        sep="\t",
        low_memory=False,
    )

    required_meta = {
        "group_id",
        "donor_id",
        "tissue",
        "cell_type",
        "priority_lineage_corrected",
        "n_cells",
    }

    miss = required_meta - set(meta.columns)

    if miss:
        raise RuntimeError(
            "Frozen metadata missing columns: "
            + ", ".join(sorted(miss))
        )

    meta["group_id"] = meta["group_id"].astype(str)

    if len(meta) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} frozen metadata rows, found {len(meta)}."
        )

    observed = (
        meta["priority_lineage_corrected"]
        .dropna()
        .value_counts()
        .to_dict()
    )

    if observed != EXPECTED_LINEAGE_COUNTS:
        raise RuntimeError(
            f"Current frozen lineage membership changed.\n"
            f"Expected: {EXPECTED_LINEAGE_COUNTS}\n"
            f"Observed: {observed}"
        )

    dat = np.load(NPZ_PATH, allow_pickle=True)

    ids = dat["group_id"].astype(str)
    genes = dat["genes"].astype(str)
    x = dat["expression"]

    if x.shape != (EXPECTED_ROWS, EXPECTED_GENES):
        raise RuntimeError(
            f"Unexpected expression shape: {x.shape}; "
            f"expected {(EXPECTED_ROWS, EXPECTED_GENES)}."
        )

    if len(np.unique(genes)) != len(genes):
        raise RuntimeError("Expression matrix contains duplicate gene names.")

    if set(ids) != set(meta["group_id"]):
        raise RuntimeError(
            "Frozen metadata group_id and expression-matrix group_id differ."
        )

    meta = meta.set_index("group_id")

    order = pd.Index(ids).get_indexer(meta.index.astype(str))

    if (order < 0).any():
        raise RuntimeError("Failed to align expression matrix to frozen metadata.")

    x = x[order]

    if not np.isfinite(x).all():
        raise RuntimeError("Expression matrix contains non-finite values.")

    return meta, x, genes


# =============================================================================
# BUILD DONOR x TISSUE x MAJOR-LINEAGE AGGREGATES
# =============================================================================

def build_lineage_aggregate(meta, x, lineage, min_tissue_donors):
    """
    Equal-pseudobulk aggregation within donor x tissue x major lineage.
    Then retain lineage x tissue groups represented by >= min_tissue_donors.
    """
    part = meta[
        meta["priority_lineage_corrected"].eq(lineage)
    ].copy()

    if part.empty:
        raise RuntimeError(f"No rows for lineage {lineage}.")

    group_indices = (
        part.reset_index()
        .groupby(
            ["donor_id", "tissue"],
            sort=True,
            observed=True,
        )
        .indices
    )

    # IMPORTANT:
    # groupby.indices above indexes positions within part.reset_index(),
    # not within the global matrix. Build global row-position lookup.
    global_pos = pd.Series(
        np.arange(len(meta), dtype=int),
        index=meta.index,
    )

    agg_y = []
    agg_meta = []

    for (donor, tissue), local_positions in group_indices.items():
        sub = part.reset_index().iloc[local_positions]
        gids = sub["group_id"].astype(str).tolist()
        idx = global_pos.loc[gids].to_numpy(int)

        agg_y.append(
            x[idx].mean(axis=0).astype(np.float32)
        )

        agg_meta.append({
            "donor_id": str(donor),
            "tissue": str(tissue),
            "lineage": lineage,
            "n_pseudobulks": int(len(idx)),
            "n_original_cell_types": int(sub["cell_type"].nunique()),
            "n_cells_total": int(pd.to_numeric(sub["n_cells"]).sum()),
        })

    Y = np.vstack(agg_y)
    am = pd.DataFrame(agg_meta)

    donor_support = (
        am.groupby("tissue", observed=True)["donor_id"]
        .nunique()
    )

    supported_tissues = set(
        donor_support[
            donor_support >= min_tissue_donors
        ].index.astype(str)
    )

    keep = am["tissue"].astype(str).isin(supported_tissues).to_numpy()

    am = am.loc[keep].reset_index(drop=True)
    Y = Y[keep]

    if len(am) == 0:
        raise RuntimeError(
            f"{lineage}: no supported donor×tissue aggregates after filtering."
        )

    return am, Y, donor_support


# =============================================================================
# OMNIBUS TISSUE-EFFECT TEST
# =============================================================================

def tissue_effect_gene_test(
    agg_meta,
    Y,
    genes,
    min_prevalence,
    min_partial_omega2,
):
    """
    Full:    y ~ donor + tissue
    Reduced: y ~ donor

    Classical nested partial F test on the frozen pseudobulk expression scale.
    """
    n = len(agg_meta)

    donor_dummies = pd.get_dummies(
        agg_meta["donor_id"].astype(str),
        drop_first=True,
        dtype=float,
    )

    tissue_dummies = pd.get_dummies(
        agg_meta["tissue"].astype(str),
        drop_first=True,
        dtype=float,
    )

    intercept = np.ones((n, 1), dtype=float)

    Xr = np.column_stack([
        intercept,
        donor_dummies.to_numpy(float),
    ])

    Xf = np.column_stack([
        intercept,
        donor_dummies.to_numpy(float),
        tissue_dummies.to_numpy(float),
    ])

    Qr, rank_r = orthonormal_basis(Xr)
    Qf, rank_f = orthonormal_basis(Xf)

    df1 = int(rank_f - rank_r)
    df2 = int(n - rank_f)

    if df1 <= 0 or df2 <= 0:
        raise RuntimeError(
            f"Invalid tissue-effect degrees of freedom: df1={df1}, df2={df2}."
        )

    prevalence = (Y > 0).mean(axis=0)
    sd = Y.std(axis=0, ddof=1)

    sse_r = sse_after_projection(Y, Qr)
    sse_f = sse_after_projection(Y, Qf)

    delta = np.maximum(sse_r - sse_f, 0.0)

    testable = (
        np.isfinite(sd)
        & (sd > EPS)
        & np.isfinite(sse_r)
        & (sse_r > EPS)
        & (prevalence >= min_prevalence)
    )

    F = np.full(len(genes), np.nan, dtype=float)
    p = np.ones(len(genes), dtype=float)
    partial_r2 = np.zeros(len(genes), dtype=float)
    partial_omega2 = np.zeros(len(genes), dtype=float)

    denominator = sse_f / df2

    ok = (
        testable
        & np.isfinite(denominator)
        & (denominator > EPS)
    )

    F[ok] = (
        (delta[ok] / df1)
        / denominator[ok]
    )

    p[ok] = f_dist.sf(
        F[ok],
        df1,
        df2,
    )

    partial_r2[testable] = np.divide(
        delta[testable],
        sse_r[testable],
        out=np.zeros(np.sum(testable), dtype=float),
        where=sse_r[testable] > EPS,
    )

    # Bias-reduced partial omega-squared from the omnibus F statistic.
    finite_F = ok & np.isfinite(F)

    partial_omega2[finite_F] = np.maximum(
        (
            df1 * (F[finite_F] - 1.0)
        )
        / (
            df1 * F[finite_F]
            + df2
            + 1.0
        ),
        0.0,
    )

    # Conservative multiplicity family = all 59,410 genes.
    q = bh(p)

    responsive = (
        testable
        & (q < 0.05)
        & (partial_omega2 >= min_partial_omega2)
    )

    out = pd.DataFrame({
        "gene": genes,
        "testable": testable,
        "prevalence": prevalence,
        "sd": sd,
        "F_tissue": F,
        "df_tissue": df1,
        "df_residual": df2,
        "P_tissue": p,
        "FDR_tissue": q,
        "partial_R2_tissue": partial_r2,
        "partial_omega2_tissue": partial_omega2,
        "tissue_responsive_primary": responsive,
    })

    design_info = {
        "n_aggregates": int(n),
        "n_donors": int(agg_meta["donor_id"].nunique()),
        "n_tissues": int(agg_meta["tissue"].nunique()),
        "rank_reduced_donor": int(rank_r),
        "rank_full_donor_tissue": int(rank_f),
        "df_tissue": int(df1),
        "df_residual": int(df2),
        "n_testable_genes": int(testable.sum()),
        "n_primary_responsive_genes": int(responsive.sum()),
    }

    return out, design_info


# =============================================================================
# ORA
# =============================================================================

def run_ora(
    gene_stats,
    gene_sets,
    collection,
    min_geneset_overlap,
):
    testable = gene_stats["testable"].to_numpy(bool)
    responsive = gene_stats["tissue_responsive_primary"].to_numpy(bool)
    genes = gene_stats["gene"].astype(str).to_numpy()

    universe_genes = set(genes[testable])
    responsive_genes = set(genes[responsive])

    N = len(universe_genes)
    K = len(responsive_genes)

    rows = []

    for rec in gene_sets:
        members = set(rec["genes"]) & universe_genes
        M = len(members)

        if M < min_geneset_overlap or N <= 0:
            rows.append({
                "collection": collection,
                "pathway": rec["pathway"],
                "description": rec["description"],
                "testable_pathway": False,
                "n_universe": N,
                "n_responsive": K,
                "n_pathway_testable": M,
                "n_overlap_responsive": 0,
                "odds_ratio_raw": np.nan,
                "log2_odds_ratio_corrected": np.nan,
                "enrichment_ratio": np.nan,
                "P_enrichment": 1.0,
                "overlap_genes": "",
            })
            continue

        overlap = sorted(members & responsive_genes)
        a = len(overlap)
        b = M - a
        c = K - a
        d = N - a - b - c

        table = [[a, b], [c, d]]

        try:
            odds_raw, p = fisher_exact(
                table,
                alternative="greater",
            )
        except Exception:
            odds_raw, p = np.nan, 1.0

        baseline = K / N if N else np.nan
        pathway_rate = a / M if M else np.nan

        enrichment_ratio = (
            pathway_rate / baseline
            if baseline and np.isfinite(baseline)
            else np.nan
        )

        rows.append({
            "collection": collection,
            "pathway": rec["pathway"],
            "description": rec["description"],
            "testable_pathway": True,
            "n_universe": N,
            "n_responsive": K,
            "n_pathway_testable": M,
            "n_overlap_responsive": a,
            "odds_ratio_raw": odds_raw,
            "log2_odds_ratio_corrected": corrected_log2_or(a, b, c, d),
            "enrichment_ratio": enrichment_ratio,
            "P_enrichment": p,
            "overlap_genes": ";".join(overlap),
        })

    out = pd.DataFrame(rows)

    # Correct over the COMPLETE collection, with untestable P=1 retained.
    out["FDR_enrichment"] = bh(out["P_enrichment"].to_numpy(float))
    out["enrichment_supported"] = (
        out["testable_pathway"].astype(bool)
        & out["FDR_enrichment"].lt(0.05)
        & out["log2_odds_ratio_corrected"].gt(0)
    )

    return out


# =============================================================================
# DISPLAY SELECTION
# =============================================================================

def select_hallmark_for_display(all_pathways, max_n):
    x = all_pathways[
        all_pathways["collection"].eq("Hallmark")
    ].copy()

    s = (
        x.groupby("pathway", observed=True)
        .agg(
            n_supported=("enrichment_supported", "sum"),
            max_log2OR=("log2_odds_ratio_corrected", "max"),
            min_FDR=("FDR_enrichment", "min"),
        )
        .reset_index()
    )

    hits = s[s["n_supported"] > 0].copy()

    if len(hits):
        chosen = hits.sort_values(
            ["n_supported", "max_log2OR", "min_FDR", "pathway"],
            ascending=[False, False, True, True],
        ).head(max_n)
    else:
        print(
            "WARNING: no Hallmark pathway reached FDR<0.05; "
            "displaying smallest-FDR pathways descriptively."
        )
        chosen = s.sort_values(
            ["min_FDR", "max_log2OR", "pathway"],
            ascending=[True, False, True],
        ).head(min(15, max_n))

    return chosen["pathway"].tolist()


def select_reactome_for_display(
    all_pathways,
    reactome_sets_lookup,
    max_n,
    jaccard_max,
):
    x = all_pathways[
        all_pathways["collection"].eq("Reactome")
    ].copy()

    s = (
        x.groupby("pathway", observed=True)
        .agg(
            n_supported=("enrichment_supported", "sum"),
            max_log2OR=("log2_odds_ratio_corrected", "max"),
            min_FDR=("FDR_enrichment", "min"),
        )
        .reset_index()
    )

    candidates = s[
        s["n_supported"] > 0
    ].sort_values(
        ["n_supported", "max_log2OR", "min_FDR", "pathway"],
        ascending=[False, False, True, True],
    )

    if candidates.empty:
        print(
            "WARNING: no Reactome pathway reached FDR<0.05; "
            "displaying smallest-FDR pathways descriptively."
        )
        candidates = s.sort_values(
            ["min_FDR", "max_log2OR", "pathway"],
            ascending=[True, False, True],
        )

    selected = []

    for r in candidates.itertuples(index=False):
        g = reactome_sets_lookup.get(r.pathway, ())

        redundant = False

        for prev in selected:
            gp = reactome_sets_lookup.get(prev, ())

            if pathway_jaccard(g, gp) > jaccard_max:
                redundant = True
                break

        if not redundant:
            selected.append(r.pathway)

        if len(selected) >= max_n:
            break

    return selected


# =============================================================================
# HEATMAP
# =============================================================================

def plot_pathway_heatmap(
    all_pathways,
    selected_pathways,
    collection,
    stem,
    title,
):
    if not selected_pathways:
        print(f"WARNING: no selected pathways for {collection}; skipping {stem}.")
        return

    sub = all_pathways[
        all_pathways["collection"].eq(collection)
        & all_pathways["pathway"].isin(selected_pathways)
    ].copy()

    order_lookup = {
        p: i
        for i, p in enumerate(selected_pathways)
    }

    sub["_row"] = sub["pathway"].map(order_lookup)

    effect = sub.pivot_table(
        index="pathway",
        columns="lineage",
        values="log2_odds_ratio_corrected",
        aggfunc="first",
    )

    qmat = sub.pivot_table(
        index="pathway",
        columns="lineage",
        values="FDR_enrichment",
        aggfunc="first",
    )

    for lin in LINEAGES:
        if lin not in effect.columns:
            effect[lin] = np.nan
        if lin not in qmat.columns:
            qmat[lin] = np.nan

    effect = effect[LINEAGES].reindex(selected_pathways)
    qmat = qmat[LINEAGES].reindex(selected_pathways)

    mat = effect.to_numpy(float)
    finite = mat[np.isfinite(mat)]

    if len(finite) == 0:
        print(f"WARNING: no finite OR values for {collection}; skipping.")
        return

    vmax = max(
        0.5,
        float(np.nanpercentile(np.abs(finite), 98)),
    )
    vmin = -vmax

    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F2F2F2")

    fig_h = max(3.6, 0.31 * len(selected_pathways) + 1.35)

    fig, ax = plt.subplots(
        figsize=(6.3, fig_h)
    )

    im = ax.imshow(
        np.ma.masked_invalid(mat),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(LINEAGES)))
    ax.set_xticklabels(
        [LINEAGE_LABEL[x] for x in LINEAGES],
        rotation=28,
        ha="right",
    )

    labels = [
        p.replace("HALLMARK_", "").replace("REACTOME_", "").replace("_", " ")
        for p in selected_pathways
    ]

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(
        labels,
        fontsize=6.3 if len(labels) <= 20 else 5.7,
    )

    # Overlay dot only for FDR-supported enrichment.
    q = qmat.to_numpy(float)

    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if np.isfinite(q[i, j]) and q[i, j] < 0.05:
                ax.scatter(
                    j,
                    i,
                    s=9,
                    c="black",
                    marker="o",
                    linewidths=0,
                )

    ax.set_title(
        title,
        loc="left",
        pad=8,
    )

    ax.grid(False)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.025,
    )

    cbar.set_label(
        "log2 enrichment odds ratio",
        fontsize=8,
    )

    fig.text(
        0.995,
        0.01,
        "Black dot: collection-wide FDR < 0.05",
        ha="right",
        va="bottom",
        fontsize=6.5,
    )

    save_figure(fig, stem)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-tissue-donors",
        type=int,
        default=3,
        help="Minimum donors required for a lineage×tissue to enter the omnibus model.",
    )

    parser.add_argument(
        "--min-prevalence",
        type=float,
        default=0.20,
        help="Minimum fraction of retained aggregates with processed expression >0.",
    )

    parser.add_argument(
        "--min-partial-omega2",
        type=float,
        default=0.05,
        help="Primary tissue-effect size threshold for responsive genes.",
    )

    parser.add_argument(
        "--min-pathway-genes",
        type=int,
        default=10,
        help="Minimum testable genes required for ORA of a pathway.",
    )

    parser.add_argument(
        "--max-hallmark-display",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--max-reactome-display",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--reactome-display-jaccard",
        type=float,
        default=0.50,
        help="DISPLAY-ONLY Reactome redundancy cutoff after full testing.",
    )

    args = parser.parse_args()

    print("=" * 108)
    print("TISSUE-RESPONSIVE GENE + PATHWAY ENRICHMENT ANALYSIS")
    print("=" * 108)
    print(
        "Primary gene definition: FDR<0.05 and "
        f"partial omega^2>={args.min_partial_omega2:.3f}"
    )
    print(
        f"Support: lineage×tissue >= {args.min_tissue_donors} donors; "
        f"gene prevalence >= {args.min_prevalence:.2f}"
    )

    meta, x, genes = load_inputs()

    hallmark_sets = parse_gmt(
        HALLMARK_GMT,
        "Hallmark",
    )

    reactome_sets = parse_gmt(
        REACTOME_GMT,
        "Reactome",
    )

    if len(hallmark_sets) != EXPECTED_HALLMARK:
        raise RuntimeError(
            f"Expected {EXPECTED_HALLMARK} Hallmark pathways, found {len(hallmark_sets)}."
        )

    if len(reactome_sets) != EXPECTED_REACTOME:
        raise RuntimeError(
            f"Expected {EXPECTED_REACTOME} Reactome pathways, found {len(reactome_sets)}."
        )

    reactome_lookup = {
        r["pathway"]: r["genes"]
        for r in reactome_sets
    }

    gene_summaries = []
    all_pathways = []
    sensitivity_rows = []
    support_rows = []
    design_rows = []

    for lin in LINEAGES:
        print()
        print("-" * 108)
        print(f"[{lin}] building donor×tissue aggregate expression...")
        print("-" * 108)

        am, Y, donor_support = build_lineage_aggregate(
            meta=meta,
            x=x,
            lineage=lin,
            min_tissue_donors=args.min_tissue_donors,
        )

        am.to_csv(
            TABLES / f"{lin}_donor_tissue_aggregate_metadata.csv",
            index=False,
        )

        for tissue, n_d in donor_support.items():
            support_rows.append({
                "lineage": lin,
                "tissue": str(tissue),
                "n_donors": int(n_d),
                "retained": bool(n_d >= args.min_tissue_donors),
            })

        print(
            f"[{lin}] aggregates={len(am)}, "
            f"donors={am.donor_id.nunique()}, "
            f"tissues={am.tissue.nunique()}"
        )

        stats, design = tissue_effect_gene_test(
            agg_meta=am,
            Y=Y,
            genes=genes,
            min_prevalence=args.min_prevalence,
            min_partial_omega2=args.min_partial_omega2,
        )

        stats.insert(0, "lineage", lin)

        # Full all-gene table compressed; responsive subset uncompressed.
        stats.to_csv(
            GENE_DIR / f"{lin}_all_gene_tissue_effects.csv.gz",
            index=False,
            compression="gzip",
        )

        responsive = stats[
            stats["tissue_responsive_primary"]
        ].copy()

        responsive = responsive.sort_values(
            [
                "partial_omega2_tissue",
                "FDR_tissue",
                "gene",
            ],
            ascending=[False, True, True],
        )

        responsive.to_csv(
            GENE_DIR / f"{lin}_primary_tissue_responsive_genes.csv",
            index=False,
        )

        n_testable = int(stats["testable"].sum())
        n_resp = int(stats["tissue_responsive_primary"].sum())

        gene_summaries.append({
            "lineage": lin,
            "lineage_label": LINEAGE_LABEL[lin],
            "n_aggregates": len(am),
            "n_donors": am["donor_id"].nunique(),
            "n_tissues": am["tissue"].nunique(),
            "n_all_genes": len(stats),
            "n_testable_genes": n_testable,
            "n_primary_tissue_responsive_genes": n_resp,
            "responsive_fraction_testable": (
                n_resp / n_testable if n_testable else np.nan
            ),
            "median_partial_omega2_testable": float(
                stats.loc[
                    stats["testable"],
                    "partial_omega2_tissue",
                ].median()
            ),
        })

        design["lineage"] = lin
        design_rows.append(design)

        # Sensitivity burden for predeclared omega2 gates.
        for thr in SENSITIVITY_OMEGA2:
            n = int(
                (
                    stats["testable"]
                    & stats["FDR_tissue"].lt(0.05)
                    & stats["partial_omega2_tissue"].ge(thr)
                ).sum()
            )

            sensitivity_rows.append({
                "lineage": lin,
                "lineage_label": LINEAGE_LABEL[lin],
                "partial_omega2_threshold": thr,
                "n_responsive": n,
                "n_testable": n_testable,
                "fraction_responsive": (
                    n / n_testable if n_testable else np.nan
                ),
            })

        print(
            f"[{lin}] testable genes={n_testable:,}; "
            f"primary responsive={n_resp:,} "
            f"({100*n_resp/n_testable:.1f}% of testable)"
        )

        print(f"[{lin}] Hallmark ORA...")
        h = run_ora(
            gene_stats=stats,
            gene_sets=hallmark_sets,
            collection="Hallmark",
            min_geneset_overlap=args.min_pathway_genes,
        )
        h.insert(0, "lineage", lin)

        print(f"[{lin}] Reactome ORA...")
        r = run_ora(
            gene_stats=stats,
            gene_sets=reactome_sets,
            collection="Reactome",
            min_geneset_overlap=args.min_pathway_genes,
        )
        r.insert(0, "lineage", lin)

        h.to_csv(
            PATHWAY_DIR / f"{lin}_hallmark_ORA.csv",
            index=False,
        )

        r.to_csv(
            PATHWAY_DIR / f"{lin}_reactome_ORA.csv",
            index=False,
        )

        all_pathways.append(h)
        all_pathways.append(r)

        print(
            f"[{lin}] Hallmark FDR hits={int(h.enrichment_supported.sum())}; "
            f"Reactome FDR hits={int(r.enrichment_supported.sum())}"
        )

    gene_summary = pd.DataFrame(gene_summaries)
    sensitivity = pd.DataFrame(sensitivity_rows)
    support = pd.DataFrame(support_rows)
    design_df = pd.DataFrame(design_rows)
    pathways = pd.concat(all_pathways, ignore_index=True)

    gene_summary.to_csv(
        TABLES / "01_gene_burden_summary.csv",
        index=False,
    )

    sensitivity.to_csv(
        TABLES / "02_gene_burden_effect_threshold_sensitivity.csv",
        index=False,
    )

    support.to_csv(
        TABLES / "03_lineage_tissue_support.csv",
        index=False,
    )

    design_df.to_csv(
        TABLES / "04_gene_model_design_summary.csv",
        index=False,
    )

    pathways.to_csv(
        PATHWAY_DIR / "ALL_LINEAGES_ALL_PATHWAYS_ORA.csv.gz",
        index=False,
        compression="gzip",
    )

    # -------------------------------------------------------------------------
    # Fig 3E
    # -------------------------------------------------------------------------
    e = gene_summary.copy()
    e["_order"] = e["lineage"].map(
        {x: i for i, x in enumerate(LINEAGES)}
    )
    e = e.sort_values("_order")

    fig, ax = plt.subplots(
        figsize=(5.8, 3.0)
    )

    y = np.arange(len(e))
    pct = 100 * e["responsive_fraction_testable"].to_numpy(float)

    ax.barh(
        y,
        pct,
        height=0.68,
        color=[LINEAGE_COLORS[x] for x in e["lineage"]],
        edgecolor="none",
    )

    max_pct = max(1.0, float(np.nanmax(pct)))

    for yi, rr in enumerate(e.itertuples()):
        ax.text(
            100 * rr.responsive_fraction_testable + 0.015 * max_pct,
            yi,
            (
                f"{int(rr.n_primary_tissue_responsive_genes):,}/"
                f"{int(rr.n_testable_genes):,}"
            ),
            va="center",
            ha="left",
            fontsize=7.2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(e["lineage_label"])
    ax.invert_yaxis()

    ax.set_xlabel(
        "Tissue-responsive genes (% of testable genes)"
    )

    ax.set_title(
        "Tissue-associated transcriptional remodeling across major lineages",
        loc="left",
        pad=7,
    )

    ax.set_xlim(
        0,
        min(100, max_pct * 1.20 + 2),
    )

    clean_axis(ax)

    save_figure(
        fig,
        "Fig3E_tissue_responsive_gene_burden",
    )

    # -------------------------------------------------------------------------
    # Fig 3F — Hallmark
    # -------------------------------------------------------------------------
    hallmark_selected = select_hallmark_for_display(
        pathways,
        args.max_hallmark_display,
    )

    pd.DataFrame({
        "display_order": np.arange(1, len(hallmark_selected) + 1),
        "pathway": hallmark_selected,
    }).to_csv(
        TABLES / "Fig3F_selected_hallmark_pathways.csv",
        index=False,
    )

    plot_pathway_heatmap(
        all_pathways=pathways,
        selected_pathways=hallmark_selected,
        collection="Hallmark",
        stem="Fig3F_Hallmark_tissue_remodeling_enrichment",
        title=(
            "Hallmark programs enriched among tissue-responsive genes"
        ),
    )

    # -------------------------------------------------------------------------
    # Fig 3G — Reactome
    # -------------------------------------------------------------------------
    reactome_selected = select_reactome_for_display(
        all_pathways=pathways,
        reactome_sets_lookup=reactome_lookup,
        max_n=args.max_reactome_display,
        jaccard_max=args.reactome_display_jaccard,
    )

    pd.DataFrame({
        "display_order": np.arange(1, len(reactome_selected) + 1),
        "pathway": reactome_selected,
    }).to_csv(
        TABLES / "Fig3G_selected_reactome_pathways.csv",
        index=False,
    )

    plot_pathway_heatmap(
        all_pathways=pathways,
        selected_pathways=reactome_selected,
        collection="Reactome",
        stem="Fig3G_Reactome_tissue_remodeling_enrichment",
        title=(
            "Representative Reactome programs enriched among "
            "tissue-responsive genes"
        ),
    )

    # -------------------------------------------------------------------------
    # Supplementary burden sensitivity
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(5.8, 3.2)
    )

    for lin in LINEAGES:
        z = sensitivity[
            sensitivity["lineage"].eq(lin)
        ].sort_values("partial_omega2_threshold")

        ax.plot(
            z["partial_omega2_threshold"],
            100 * z["fraction_responsive"],
            marker="o",
            linewidth=1.4,
            markersize=4,
            color=LINEAGE_COLORS[lin],
            label=LINEAGE_LABEL[lin],
        )

    ax.axvline(
        args.min_partial_omega2,
        linestyle="--",
        linewidth=0.9,
        color="#666666",
    )

    ax.set_xlabel(
        "Minimum partial omega² tissue effect"
    )

    ax.set_ylabel(
        "FDR-supported genes (% of testable genes)"
    )

    ax.set_title(
        "Sensitivity of tissue-responsive gene burden to effect-size threshold",
        loc="left",
        pad=7,
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    clean_axis(ax)

    save_figure(
        fig,
        "Supp_gene_burden_effect_threshold_sensitivity",
    )

    # -------------------------------------------------------------------------
    # Pathway summary
    # -------------------------------------------------------------------------
    pathway_summary = (
        pathways.groupby(
            ["lineage", "collection"],
            observed=True,
        )
        .agg(
            n_tested=("testable_pathway", "sum"),
            n_FDR_supported=("enrichment_supported", "sum"),
            median_log2OR=("log2_odds_ratio_corrected", "median"),
        )
        .reset_index()
    )

    pathway_summary.to_csv(
        TABLES / "05_pathway_enrichment_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------
    audit = {
        "analysis": (
            "Within-lineage omnibus tissue-responsive gene analysis "
            "followed by pathway over-representation analysis"
        ),
        "static_expression_only": True,
        "clock_distance_used": False,
        "A4_7_correspondence_used": False,
        "matched_specificity_null_used": False,

        "expression_source": str(NPZ_PATH),
        "metadata_source": str(FROZEN_META),
        "hallmark_source": str(HALLMARK_GMT),
        "reactome_source": str(REACTOME_GMT),

        "aggregation": (
            "Equal mean across original cell-type pseudobulks within "
            "donor x tissue x current major lineage"
        ),

        "tissue_support_min_donors": args.min_tissue_donors,

        "gene_model": {
            "full": "expression ~ donor + tissue",
            "reduced": "expression ~ donor",
            "test": "nested classical partial F test",
            "multiplicity": (
                "BH across all 59,410 genes per lineage; "
                "untestable genes assigned P=1"
            ),
            "min_prevalence": args.min_prevalence,
            "primary_FDR": 0.05,
            "primary_min_partial_omega2": args.min_partial_omega2,
        },

        "pathway_ORA": {
            "test": "one-sided Fisher exact over-representation",
            "background": "lineage-specific testable genes",
            "minimum_testable_genes_per_pathway": args.min_pathway_genes,
            "FDR": "BH separately within lineage x collection",
            "hallmark_full_tests": EXPECTED_HALLMARK,
            "reactome_full_tests": EXPECTED_REACTOME,
        },

        "reactome_display_redundancy": {
            "statistics_changed": False,
            "display_only": True,
            "jaccard_max": args.reactome_display_jaccard,
            "max_display_pathways": args.max_reactome_display,
        },

        "limitations": [
            (
                "Tissue is a multi-level factor; gene effects are omnibus "
                "and have no single up/down direction."
            ),
            (
                "The partial F test treats donor x tissue aggregates as "
                "model observations after donor fixed-effect adjustment; "
                "it is not a causal model."
            ),
            (
                "Major-lineage aggregation can still contain cell-state or "
                "fine-subtype composition differences."
            ),
            (
                "Pathway enrichment reflects overlap with tissue-responsive genes; "
                "it does not establish clock-specific regulation."
            ),
        ],
    }

    (AUDIT / "analysis_manifest.json").write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 108)
    print("ANALYSIS COMPLETE")
    print("=" * 108)

    print("\nGENE BURDEN")
    print(
        gene_summary[
            [
                "lineage_label",
                "n_aggregates",
                "n_donors",
                "n_tissues",
                "n_testable_genes",
                "n_primary_tissue_responsive_genes",
                "responsive_fraction_testable",
                "median_partial_omega2_testable",
            ]
        ].to_string(
            index=False,
            formatters={
                "responsive_fraction_testable": lambda z: f"{100*z:.1f}%",
                "median_partial_omega2_testable": lambda z: f"{z:.3f}",
            },
        )
    )

    print("\nPATHWAY ENRICHMENT")
    print(
        pathway_summary.to_string(index=False)
    )

    print("\nMAIN FIGURES")
    print(FIG / "Fig3E_tissue_responsive_gene_burden.png")
    print(FIG / "Fig3F_Hallmark_tissue_remodeling_enrichment.png")
    print(FIG / "Fig3G_Reactome_tissue_remodeling_enrichment.png")

    print("\nSUPPLEMENTARY")
    print(FIG / "Supp_gene_burden_effect_threshold_sensitivity.png")

    print("\nFull gene/pathway tables:")
    print(TABLES)

    print("=" * 108)


if __name__ == "__main__":
    main()
