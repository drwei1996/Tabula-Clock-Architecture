#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
56_run_HCL_AHCA_Fig1_Fig2_replication.py

Purpose
-------
Reproduce the CURRENT Tabula Figure 1 / Figure 2 analysis logic in the two
external datasets (HCL and AHCA) using the already-copied finalized
Expanded17 pseudobulks. This is a NEW analysis, not a redraw of the previous
external-validation summary figures.

Current Tabula analogs reproduced here
--------------------------------------
FIGURE 1-like
  A. Core-clock UMAP
  B. Expanded17 clock UMAP
  C. Global dataset-native cell-type clock-expression landscape
  D-H. Five major-lineage zoom-in clock-expression landscapes

FIGURE 2-like
  A. Within-lineage tissue remodeling vs between-lineage separation
  B. Lineage-specific tissue remodeling with higher-level bootstrap intervals
  C. Cross-tissue architecture within contrasting lineages (Fibroblast vs B cell)
  D. Feature architecture:
       cell identity / tissue / donor / identity×tissue unique adjusted R²
  E. Lineage-specific tissue contribution across Expanded17 genes

Important scientific scope
--------------------------
- Static clock-associated transcriptional architecture only.
- No rhythm / phase / amplitude / oscillation claims.
- External primary feature space is Expanded17.
- Figure 1 / Figure 2 are reproducible locally from the copied pseudobulks.
- Figure 3 pathway reproduction needs full-transcriptome pseudobulks; this
  script writes a readiness report but DOES NOT fake pathway analyses from only
  17 clock genes.
- AHCA is a single-donor atlas: donor contribution is therefore NA by design.

Inputs expected
---------------
ROOT/
  01_HCL/source_data/pseudobulk_standardized.csv
  01_HCL/source_data/tissue_lineage_centroids.csv
  01_HCL/source_data/within_lineage_between_tissue_pairs.csv
  01_HCL/source_data/between_lineage_within_tissue_pairs.csv
  02_AHCA_15ORG/source_data/...
  03_RIGOR_VALIDATION_V1_20260921/FINAL_RIGOR_SUMMARY.csv

Optional:
  project/results/phaseA_clock_architecture/00_audit/prespecified_protocol.json
  Used ONLY to recover exact Core panel membership. If unavailable, the script
  stops rather than guessing Core8.

Output
------
OUT/
  HCL/
    Figure1_like/
    Figure2_like/
    source_data/
  AHCA/
    Figure1_like/
    Figure2_like/
    source_data/
  FIG3_READINESS.csv
  ANALYSIS_MANIFEST.json

Recommended run
---------------
python 56_run_HCL_AHCA_Fig1_Fig2_replication.py ^
  --project "<TABULA_PROJECT_ROOT>" ^
  --root "<TABULA_PROJECT_ROOT>\\corrected_full_analysis_v2\\external validation\\clock_external_validation_ALL_EXCEPT_H5AD_20260921\\clock_external_validation" ^
  --outdir "<TABULA_PROJECT_ROOT>\\HCL_AHCA_Fig123_replication_20260923"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False
    from sklearn.decomposition import PCA


# =============================================================================
# Constants
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

EXPANDED17 = [
    "ARNTL", "CLOCK", "NPAS2",
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "NR1D1", "NR1D2",
    "RORA", "RORB", "RORC",
    "DBP", "TEF", "HLF", "NFIL3",
]

SEED = 20260923


# =============================================================================
# Plot style
# =============================================================================

def choose_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Arial", "Liberation Sans", "DejaVu Sans"]:
        if f in available:
            return f
    return "sans-serif"


mpl.rcParams.update({
    "font.family": choose_font(),
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
})


# =============================================================================
# Helpers
# =============================================================================

def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def save_fig(fig, outdir: Path, stem: str) -> None:
    mkdir(outdir)
    for ext in [".png", ".pdf", ".svg"]:
        kwargs = dict(
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        if ext == ".png":
            kwargs["dpi"] = 600
        fig.savefig(outdir / f"{stem}{ext}", **kwargs)
    plt.close(fig)


def clean_axis(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_csv(df: pd.DataFrame, outdir: Path, name: str) -> None:
    mkdir(outdir)
    df.to_csv(outdir / name, index=False, encoding="utf-8-sig")


def recover_core_genes(project: Path) -> list[str]:
    candidates = [
        project
        / "results"
        / "phaseA_clock_architecture"
        / "00_audit"
        / "prespecified_protocol.json",
        project
        / "corrected_full_analysis_v2"
        / "00_audit"
        / "prespecified_protocol.json",
    ]

    for p in candidates:
        if not p.exists():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        fs = obj.get("feature_sets", {})
        vals = fs.get("CORE_FEATURE_COLS")
        if vals:
            genes = [str(x).removeprefix("expr_") for x in vals]
            if len(genes) >= 2:
                print(f"[SPEC] Core panel recovered from: {p}")
                print(f"[SPEC] Core genes: {genes}")
                return genes

    raise FileNotFoundError(
        "Exact Core panel could not be recovered from prespecified_protocol.json. "
        "The script stops rather than guessing Core8."
    )


def feature_columns(df: pd.DataFrame, genes: list[str]) -> list[str]:
    cols = [f"expr_{g}" for g in genes if f"expr_{g}" in df.columns]
    if len(cols) != len(genes):
        missing = [g for g in genes if f"expr_{g}" not in df.columns]
        raise RuntimeError(f"Missing requested features: {missing}")
    return cols


def fit_embedding(X: np.ndarray) -> np.ndarray:
    if HAVE_UMAP:
        nn = max(3, min(15, len(X) - 1))
        return umap.UMAP(
            n_neighbors=nn,
            min_dist=0.30,
            metric="euclidean",
            random_state=SEED,
            n_components=2,
        ).fit_transform(X)

    return PCA(n_components=2, random_state=SEED).fit_transform(X)


def cluster_rows(M: np.ndarray) -> list[int]:
    if len(M) <= 2:
        return list(range(len(M)))
    try:
        z = linkage(M, method="average", metric="euclidean", optimal_ordering=True)
        return leaves_list(z).astype(int).tolist()
    except Exception:
        return list(range(len(M)))


def mode_or_first(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return "unknown"
    vc = s.value_counts()
    return str(vc.index[0])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    values = values[ok]
    weights = weights[ok]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    c = np.cumsum(weights)
    cut = 0.5 * weights.sum()
    return float(values[np.searchsorted(c, cut, side="left")])


# =============================================================================
# OLS / adjusted R2 architecture
# =============================================================================

def design_matrix(df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    pieces = [np.ones((len(df), 1), dtype=float)]

    for term in terms:
        if term == "celltype_tissue":
            vals = df["cell_type"].astype(str) + "||" + df["tissue"].astype(str)
            d = pd.get_dummies(vals, drop_first=True, dtype=float)
        else:
            vals = df[term].astype(str)
            d = pd.get_dummies(vals, drop_first=True, dtype=float)

        if d.shape[1]:
            pieces.append(d.to_numpy(float))

    return np.column_stack(pieces)


def adjusted_r2(y: np.ndarray, X: np.ndarray) -> float:
    y = np.asarray(y, float)
    X = np.asarray(X, float)

    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y = y[ok]
    X = X[ok]

    n = len(y)
    if n < 4:
        return np.nan

    rank = np.linalg.matrix_rank(X)
    p = rank - 1
    if n - p - 1 <= 0:
        return np.nan

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ssr = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))

    if sst <= 1e-15:
        return np.nan

    r2 = 1.0 - ssr / sst
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def unique_adj_r2(
    df: pd.DataFrame,
    y: np.ndarray,
    full_terms: list[str],
    reduced_terms: list[str],
) -> float:
    rf = adjusted_r2(y, design_matrix(df, full_terms))
    rr = adjusted_r2(y, design_matrix(df, reduced_terms))
    if not (np.isfinite(rf) and np.isfinite(rr)):
        return np.nan
    return float(rf - rr)


def compute_feature_architecture(pb: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    donor_variable = pb["donor_id"].astype(str).nunique() > 1

    additive = ["cell_type", "tissue"] + (["donor_id"] if donor_variable else [])
    interaction = ["cell_type", "tissue", "celltype_tissue"] + (
        ["donor_id"] if donor_variable else []
    )

    rows = []
    for c in feat_cols:
        y = pd.to_numeric(pb[c], errors="coerce").to_numpy(float)

        cell_red = ["tissue"] + (["donor_id"] if donor_variable else [])
        tissue_red = ["cell_type"] + (["donor_id"] if donor_variable else [])

        cell = unique_adj_r2(pb, y, additive, cell_red)
        tissue = unique_adj_r2(pb, y, additive, tissue_red)

        if donor_variable:
            donor = unique_adj_r2(pb, y, additive, ["cell_type", "tissue"])
        else:
            donor = np.nan

        inter = unique_adj_r2(pb, y, interaction, additive)

        rows.append({
            "feature": c.removeprefix("expr_"),
            "cell_identity_unique_adjR2": cell,
            "tissue_unique_adjR2": tissue,
            "donor_unique_adjR2": donor,
            "identity_x_tissue_increment_adjR2": inter,
            "n_rows": len(pb),
            "n_donors": pb["donor_id"].nunique(),
            "n_tissues": pb["tissue"].nunique(),
            "n_cell_types": pb["cell_type"].nunique(),
        })

    return pd.DataFrame(rows)


def compute_lineage_tissue_contribution(
    pb: pd.DataFrame,
    feat_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for lin in LINEAGE_ORDER:
        d = pb[pb["major_lineage"].astype(str).eq(lin)].copy()
        if len(d) < 4 or d["tissue"].nunique() < 2:
            continue

        donor_variable = d["donor_id"].astype(str).nunique() > 1
        full_terms = ["tissue"] + (["donor_id"] if donor_variable else [])
        reduced_terms = ["donor_id"] if donor_variable else []

        for c in feat_cols:
            y = pd.to_numeric(d[c], errors="coerce").to_numpy(float)
            tissue_unique = unique_adj_r2(
                d, y, full_terms, reduced_terms
            )
            rows.append({
                "major_lineage": lin,
                "display": LINEAGE_DISPLAY[lin],
                "feature": c.removeprefix("expr_"),
                "tissue_unique_adjR2": tissue_unique,
                "n_rows": len(d),
                "n_tissues": d["tissue"].nunique(),
                "n_donors": d["donor_id"].nunique(),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Higher-level bootstrap for lineage tissue remodeling
# =============================================================================

def aggregate_donor_tissue_lineage(pb: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    agg = (
        pb.groupby(
            ["donor_id", "tissue", "major_lineage"],
            observed=True,
            as_index=False,
        )[feat_cols]
        .mean()
    )
    return agg


def centroids_with_donor_weights(
    agg: pd.DataFrame,
    feat_cols: list[str],
    donor_weights: dict[str, float],
) -> pd.DataFrame:
    rows = []

    for (tissue, lineage), g in agg.groupby(
        ["tissue", "major_lineage"],
        observed=True,
    ):
        w = np.array(
            [donor_weights.get(str(x), 0.0) for x in g["donor_id"]],
            float,
        )
        if w.sum() <= 0:
            continue
        X = g[feat_cols].to_numpy(float)
        cen = np.average(X, axis=0, weights=w)
        rec = {
            "tissue": str(tissue),
            "major_lineage": str(lineage),
        }
        for j, c in enumerate(feat_cols):
            rec[c] = cen[j]
        rows.append(rec)

    return pd.DataFrame(rows)


def lineage_weighted_distance(
    cent: pd.DataFrame,
    feat_cols: list[str],
    lineage: str,
    tissue_weights: dict[str, float],
) -> float:
    d = cent[cent["major_lineage"].eq(lineage)].reset_index(drop=True)
    if len(d) < 2:
        return np.nan

    X = d[feat_cols].to_numpy(float)
    vals = []
    weights = []

    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            ta = str(d.loc[i, "tissue"])
            tb = str(d.loc[j, "tissue"])
            w = tissue_weights.get(ta, 0.0) * tissue_weights.get(tb, 0.0)
            if w <= 0:
                continue
            vals.append(float(np.linalg.norm(X[i] - X[j])))
            weights.append(w)

    if not vals:
        return np.nan

    return weighted_median(np.asarray(vals), np.asarray(weights))


def bootstrap_lineage_remodeling(
    pb: pd.DataFrame,
    feat_cols: list[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    agg = aggregate_donor_tissue_lineage(pb, feat_cols)
    donors = sorted(agg["donor_id"].astype(str).unique())
    tissues = sorted(agg["tissue"].astype(str).unique())

    rng = np.random.default_rng(seed)

    point_donor = {d: 1.0 for d in donors}
    point_tissue = {t: 1.0 for t in tissues}
    point_cent = centroids_with_donor_weights(
        agg, feat_cols, point_donor
    )

    point = {
        lin: lineage_weighted_distance(
            point_cent, feat_cols, lin, point_tissue
        )
        for lin in LINEAGE_ORDER
    }

    draws = {
        lin: np.full(n_boot, np.nan, float)
        for lin in LINEAGE_ORDER
    }

    for b in range(n_boot):
        dw = rng.dirichlet(np.ones(len(donors)))
        tw = rng.dirichlet(np.ones(len(tissues)))

        donor_weights = {
            donors[i]: float(dw[i])
            for i in range(len(donors))
        }
        tissue_weights = {
            tissues[i]: float(tw[i])
            for i in range(len(tissues))
        }

        cent = centroids_with_donor_weights(
            agg, feat_cols, donor_weights
        )

        for lin in LINEAGE_ORDER:
            draws[lin][b] = lineage_weighted_distance(
                cent, feat_cols, lin, tissue_weights
            )

    rows = []
    for lin in LINEAGE_ORDER:
        x = draws[lin]
        x = x[np.isfinite(x)]
        rows.append({
            "major_lineage": lin,
            "display": LINEAGE_DISPLAY[lin],
            "point_estimate": point[lin],
            "bootstrap_median": float(np.median(x)) if len(x) else np.nan,
            "ci_low": float(np.percentile(x, 2.5)) if len(x) else np.nan,
            "ci_high": float(np.percentile(x, 97.5)) if len(x) else np.nan,
            "n_valid_draws": len(x),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Figure 1
# =============================================================================

def plot_umap_panel(ax, pb, feat_cols, title):
    X = pb[feat_cols].to_numpy(float)
    emb = fit_embedding(X)

    for lin in LINEAGE_ORDER:
        idx = pb["major_lineage"].astype(str).eq(lin).to_numpy()
        if idx.sum() == 0:
            continue
        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=19,
            facecolor=LINEAGE_COLORS[lin],
            edgecolor="white",
            linewidth=0.28,
            alpha=0.92,
            label=LINEAGE_DISPLAY[lin],
        )

    ax.set_xlabel("UMAP1" if HAVE_UMAP else "PC1")
    ax.set_ylabel("UMAP2" if HAVE_UMAP else "PC2")
    ax.set_title(title)
    clean_axis(ax)
    return emb


def build_celltype_heatmap_table(
    pb: pd.DataFrame,
    feat_cols: list[str],
) -> pd.DataFrame:
    expr = (
        pb.groupby("cell_type", observed=True)[feat_cols]
        .mean()
    )

    meta = (
        pb.groupby("cell_type", observed=True)
        .agg(
            major_lineage=("major_lineage", mode_or_first),
            n_rows=("cell_type", "size"),
            n_tissues=("tissue", "nunique"),
            n_donors=("donor_id", "nunique"),
        )
    )

    d = meta.join(expr, how="inner").reset_index()

    ordered = []
    for lin in LINEAGE_ORDER:
        g = d[d["major_lineage"].eq(lin)].copy()
        if g.empty:
            continue
        M = g[feat_cols].to_numpy(float)
        oo = cluster_rows(M)
        ordered.append(g.iloc[oo])

    if not ordered:
        return d
    return pd.concat(ordered, ignore_index=True)


def draw_global_heatmap(ax, d: pd.DataFrame, feat_cols: list[str], title: str):
    M = d[feat_cols].to_numpy(float)
    vmax = min(3.0, max(1.5, float(np.nanpercentile(np.abs(M), 98))))

    im = ax.imshow(
        M,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(
        [c.removeprefix("expr_") for c in feat_cols],
        rotation=60,
        ha="right",
        fontsize=6,
    )
    ax.set_yticks([])
    ax.set_ylabel("Dataset-native cell types")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)

    return im


def draw_lineage_zoom(ax, d: pd.DataFrame, feat_cols: list[str], lin: str):
    g = d[d["major_lineage"].eq(lin)].copy()
    if g.empty:
        ax.axis("off")
        return None

    M = g[feat_cols].to_numpy(float)
    vmax = min(3.0, max(1.5, float(np.nanpercentile(np.abs(M), 98))))

    im = ax.imshow(
        M,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(g["cell_type"].astype(str), fontsize=5.5)
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(
        [c.removeprefix("expr_") for c in feat_cols],
        rotation=60,
        ha="right",
        fontsize=5.2,
    )
    ax.set_title(
        LINEAGE_DISPLAY[lin],
        color=LINEAGE_COLORS[lin],
        fontsize=8.5,
        fontweight="bold",
    )
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def build_figure1(
    dataset: str,
    pb: pd.DataFrame,
    core_cols: list[str],
    expanded_cols: list[str],
    outdir: Path,
    srcdir: Path,
):
    heat = build_celltype_heatmap_table(pb, expanded_cols)
    write_csv(heat, srcdir, "Fig1_celltype_expanded17_heatmap.csv")

    fig = plt.figure(figsize=(12.8, 14.2))
    gs = fig.add_gridspec(
        4, 6,
        height_ratios=[1.0, 1.25, 0.9, 0.9],
        hspace=0.70,
        wspace=0.70,
    )

    ax1 = fig.add_subplot(gs[0, :3])
    emb_core = plot_umap_panel(
        ax1, pb, core_cols,
        f"{dataset}: clock-architecture UMAP — Core panel",
    )

    ax2 = fig.add_subplot(gs[0, 3:])
    emb_exp = plot_umap_panel(
        ax2, pb, expanded_cols,
        f"{dataset}: clock-architecture UMAP — Expanded17",
    )

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=LINEAGE_COLORS[l],
            markeredgecolor="white",
            markersize=6,
            label=LINEAGE_DISPLAY[l],
        )
        for l in LINEAGE_ORDER
    ]
    ax2.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Major lineage",
    )

    emb_out = pb[
        ["dataset", "group_id", "donor_id", "tissue", "cell_type", "major_lineage", "n_cells"]
    ].copy()
    emb_out["core_dim1"] = emb_core[:, 0]
    emb_out["core_dim2"] = emb_core[:, 1]
    emb_out["expanded17_dim1"] = emb_exp[:, 0]
    emb_out["expanded17_dim2"] = emb_exp[:, 1]
    write_csv(emb_out, srcdir, "Fig1_UMAP_coordinates.csv")

    ax3 = fig.add_subplot(gs[1, :])
    im = draw_global_heatmap(
        ax3,
        heat,
        expanded_cols,
        f"{dataset}: global static clock-associated expression landscape",
    )
    cb = fig.colorbar(im, ax=ax3, fraction=0.018, pad=0.02)
    cb.set_label("Mean standardized feature value")

    zoom_axes = [
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[2, 4:6]),
        fig.add_subplot(gs[3, 0:2]),
        fig.add_subplot(gs[3, 2:4]),
    ]

    for ax, lin in zip(zoom_axes, LINEAGE_ORDER):
        draw_lineage_zoom(ax, heat, expanded_cols, lin)

    ax_blank = fig.add_subplot(gs[3, 4:6])
    ax_blank.axis("off")
    ax_blank.text(
        0.03, 0.92,
        "External feature space",
        fontsize=9,
        fontweight="bold",
        transform=ax_blank.transAxes,
    )
    ax_blank.text(
        0.03, 0.80,
        "Primary: Expanded17\nStatic expression only\nNo temporal inference",
        fontsize=8,
        va="top",
        transform=ax_blank.transAxes,
    )

    fig.suptitle(
        f"{dataset} — external reproduction of Figure 1 clock architecture",
        fontsize=13,
        y=0.995,
    )

    save_fig(fig, outdir, f"{dataset}_External_Figure1_like")


# =============================================================================
# Figure 2
# =============================================================================

def get_rigor_row(rigor: pd.DataFrame, dataset: str) -> pd.Series | None:
    if "dataset" not in rigor.columns:
        return None
    d = rigor[rigor["dataset"].astype(str).eq(dataset)]
    return d.iloc[0] if len(d) else None


def draw_fig2A(
    ax,
    within: pd.DataFrame,
    between: pd.DataFrame,
    rigor_row: pd.Series | None,
    dataset: str,
):
    w = float(pd.to_numeric(within["distance"]).median())
    b = float(pd.to_numeric(between["distance"]).median())

    ys = [1.0, 0.0]
    xs = [w, b]

    ax.plot(
        [w, b], [1.0, 0.0],
        color="#C4C4C4",
        lw=1.0,
        zorder=1,
    )
    ax.scatter(
        [w], [1.0],
        s=58,
        color="#2A9D8F",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.scatter(
        [b], [0.0],
        s=58,
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    ax.set_yticks(ys)
    ax.set_yticklabels([
        "Within-lineage\nacross tissues",
        "Between lineages\nwithin tissue",
    ])
    ax.set_xlabel("Clock-architecture distance")
    ax.set_title("Tissue-associated remodeling")

    diff = b - w
    txt = f"Between − within = {diff:.3f}"

    if rigor_row is not None:
        lo = rigor_row.get("hl_difference_ci_low", np.nan)
        hi = rigor_row.get("hl_difference_ci_high", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            txt += f"\n95% CI {float(lo):.3f}–{float(hi):.3f}"

    ax.text(
        0.98, 0.88,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    clean_axis(ax)


def draw_fig2B(ax, boot: pd.DataFrame):
    d = (
        boot.set_index("major_lineage")
        .reindex(LINEAGE_ORDER)
        .reset_index()
    )

    y = np.arange(len(d))[::-1]

    for yi, r in zip(y, d.itertuples(index=False)):
        lin = r.major_lineage
        x = float(r.point_estimate)
        lo = float(r.ci_low)
        hi = float(r.ci_high)
        ax.errorbar(
            x, yi,
            xerr=np.array([[x-lo], [hi-x]]),
            fmt="o",
            color=LINEAGE_COLORS[lin],
            ecolor="#6E6E6E",
            elinewidth=0.9,
            capsize=2.5,
            markersize=5.5,
            zorder=3,
        )
        ax.text(
            x, yi + 0.20,
            f"{x:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [LINEAGE_DISPLAY[l] for l in LINEAGE_ORDER]
    )
    ax.set_xlabel("Median within-lineage distance across tissues")
    ax.set_title("Lineage-specific tissue remodeling")
    clean_axis(ax)


def tissue_distance_matrix(
    cent: pd.DataFrame,
    feat_cols: list[str],
    lineage: str,
    max_tissues: int = 16,
) -> tuple[list[str], np.ndarray]:
    d = cent[cent["major_lineage"].astype(str).eq(lineage)].copy()
    if d.empty:
        return [], np.zeros((0, 0))

    support_col = (
        "n_donors" if "n_donors" in d.columns
        else "n_donor_units" if "n_donor_units" in d.columns
        else None
    )

    if support_col is not None:
        d = d.sort_values(
            [support_col, "tissue"],
            ascending=[False, True],
        ).head(max_tissues)
    else:
        d = d.sort_values("tissue").head(max_tissues)

    labels = d["tissue"].astype(str).tolist()
    X = d[feat_cols].to_numpy(float)
    M = squareform(pdist(X, metric="euclidean"))
    return labels, M


def draw_distance_heatmap(ax, labels, M, title):
    if len(labels) == 0:
        ax.axis("off")
        return None

    im = ax.imshow(
        M,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=75,
        ha="right",
        fontsize=5.5,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(length=0)
    return im


def draw_architecture_heatmap(ax, arch: pd.DataFrame):
    cols = [
        "cell_identity_unique_adjR2",
        "tissue_unique_adjR2",
        "donor_unique_adjR2",
        "identity_x_tissue_increment_adjR2",
    ]
    M = arch.set_index("feature")[cols].to_numpy(float)

    vmax = max(
        0.15,
        float(np.nanpercentile(np.abs(M), 98))
        if np.isfinite(M).any()
        else 0.15,
    )

    masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D9D9D9")

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(4))
    ax.set_xticklabels(
        ["Cell identity", "Tissue", "Donor", "Identity × tissue"],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(arch)))
    ax.set_yticklabels(arch["feature"].astype(str), fontsize=6)
    ax.set_title("Static architecture across Expanded17 features")
    ax.tick_params(length=0)
    return im


def draw_lineage_tissue_heatmap(ax, lt: pd.DataFrame):
    P = (
        lt.pivot(
            index="feature",
            columns="major_lineage",
            values="tissue_unique_adjR2",
        )
        .reindex(index=EXPANDED17, columns=LINEAGE_ORDER)
    )
    M = P.to_numpy(float)

    vmax = max(
        0.15,
        float(np.nanpercentile(np.abs(M), 98))
        if np.isfinite(M).any()
        else 0.15,
    )

    masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D9D9D9")

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(LINEAGE_ORDER)))
    ax.set_xticklabels(
        [LINEAGE_DISPLAY[x] for x in LINEAGE_ORDER],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(P.index)))
    ax.set_yticklabels(P.index, fontsize=6)
    ax.set_title(
        "Lineage-specific tissue contribution across Expanded17"
    )
    ax.tick_params(length=0)
    return im


def build_figure2(
    dataset: str,
    pb: pd.DataFrame,
    feat_cols: list[str],
    root_dataset_dir: Path,
    rigor: pd.DataFrame,
    n_boot: int,
    outdir: Path,
    srcdir: Path,
):
    within = read_csv(
        root_dataset_dir
        / "source_data"
        / "within_lineage_between_tissue_pairs.csv"
    )
    between = read_csv(
        root_dataset_dir
        / "source_data"
        / "between_lineage_within_tissue_pairs.csv"
    )
    cent = read_csv(
        root_dataset_dir
        / "source_data"
        / "tissue_lineage_centroids.csv"
    )

    rigor_row = get_rigor_row(rigor, dataset)

    print(
        f"[{dataset}] higher-level bootstrap for lineage-specific remodeling: "
        f"{n_boot} draws"
    )
    boot = bootstrap_lineage_remodeling(
        pb,
        feat_cols,
        n_boot=n_boot,
        seed=SEED + (1 if dataset == "HCL" else 2),
    )

    arch = compute_feature_architecture(pb, feat_cols)
    lt = compute_lineage_tissue_contribution(pb, feat_cols)

    write_csv(within, srcdir, "Fig2_within_lineage_pairs.csv")
    write_csv(between, srcdir, "Fig2_between_lineage_pairs.csv")
    write_csv(boot, srcdir, "Fig2_lineage_specific_bootstrap.csv")
    write_csv(arch, srcdir, "Fig2_feature_architecture.csv")
    write_csv(lt, srcdir, "Fig2_lineage_tissue_contribution.csv")

    fig = plt.figure(figsize=(12.8, 12.8))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.85, 1.45, 1.55],
        hspace=0.55,
        wspace=0.40,
    )

    axA = fig.add_subplot(gs[0, 0])
    draw_fig2A(axA, within, between, rigor_row, dataset)

    axB = fig.add_subplot(gs[0, 1])
    draw_fig2B(axB, boot)

    sub = gs[1, :].subgridspec(1, 2, wspace=0.40)
    axC1 = fig.add_subplot(sub[0, 0])
    axC2 = fig.add_subplot(sub[0, 1])

    fib_labels, fib_M = tissue_distance_matrix(
        cent, feat_cols, "fibroblast"
    )
    b_labels, b_M = tissue_distance_matrix(
        cent, feat_cols, "b_cell"
    )

    im1 = draw_distance_heatmap(
        axC1,
        fib_labels,
        fib_M,
        "Fibroblast",
    )
    draw_distance_heatmap(
        axC2,
        b_labels,
        b_M,
        "B cell",
    )

    if im1 is not None:
        cb = fig.colorbar(
            im1,
            ax=[axC1, axC2],
            fraction=0.015,
            pad=0.02,
        )
        cb.set_label("Euclidean distance")

    fig.text(
        0.5, 0.655,
        "Cross-tissue architecture within contrasting lineages",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    axD = fig.add_subplot(gs[2, 0])
    imD = draw_architecture_heatmap(axD, arch)
    cbD = fig.colorbar(imD, ax=axD, fraction=0.028, pad=0.03)
    cbD.set_label("Unique / incremental adjusted R²")

    axE = fig.add_subplot(gs[2, 1])
    imE = draw_lineage_tissue_heatmap(axE, lt)
    cbE = fig.colorbar(imE, ax=axE, fraction=0.028, pad=0.03)
    cbE.set_label("Tissue unique adjusted R²")

    if pb["donor_id"].astype(str).nunique() == 1:
        axD.text(
            0.99, 1.02,
            "Donor term = NA (single-donor atlas)",
            transform=axD.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#666666",
        )

    fig.suptitle(
        f"{dataset} — external reproduction of Figure 2 architecture/remodeling",
        fontsize=13,
        y=0.995,
    )

    save_fig(fig, outdir, f"{dataset}_External_Figure2_like")


# =============================================================================
# Figure 3 readiness
# =============================================================================

def detect_fig3_readiness(
    root: Path,
    dataset: str,
) -> dict:
    if dataset == "HCL":
        raw = root / "01_HCL" / "raw"
        h5ads = list(raw.glob("*.h5ad")) if raw.exists() else []

        allgene_candidates = []
        for pattern in [
            "*allgene*pseudobulk*.csv*",
            "*fullgene*pseudobulk*.csv*",
            "*all_gene*pseudobulk*.csv*",
            "*transcriptome*pseudobulk*.csv*",
            "*allgene*pseudobulk*.parquet",
            "*fullgene*pseudobulk*.parquet",
        ]:
            allgene_candidates.extend(root.rglob(pattern))

        return {
            "dataset": "HCL",
            "fig1_ready": True,
            "fig2_ready": True,
            "fig3_full_transcriptome_ready": bool(
                h5ads or allgene_candidates
            ),
            "local_h5ad_count": len(h5ads),
            "local_allgene_pseudobulk_count": len(allgene_candidates),
            "reason": (
                "Ready for pathway analysis"
                if (h5ads or allgene_candidates)
                else
                "Current local HCL pseudobulk contains Expanded17 only. "
                "Fig3 requires HCL H5AD or an H5AD-derived all-gene pseudobulk."
            ),
        }

    raw = root / "02_AHCA_15ORG" / "raw"
    tar_files = list(raw.glob("*.tar")) + list(raw.glob("*.tar.gz"))

    return {
        "dataset": "AHCA",
        "fig1_ready": True,
        "fig2_ready": True,
        "fig3_full_transcriptome_ready": bool(tar_files),
        "local_h5ad_count": 0,
        "local_allgene_pseudobulk_count": 0,
        "reason": (
            "Raw GSE159929 count archive present; full-transcriptome pathway analysis can be run locally."
            if tar_files
            else
            "AHCA raw count archive not found."
        ),
    }


# =============================================================================
# Dataset runner
# =============================================================================

def run_dataset(
    dataset: str,
    root: Path,
    project: Path,
    outroot: Path,
    core_genes: list[str],
    rigor: pd.DataFrame,
    n_boot: int,
):
    print("=" * 108)
    print(f"{dataset}: FIGURE 1 / FIGURE 2 REPLICATION")
    print("=" * 108)

    droot = (
        root / "01_HCL"
        if dataset == "HCL"
        else root / "02_AHCA_15ORG"
    )

    pb_path = droot / "source_data" / "pseudobulk_standardized.csv"
    pb = read_csv(pb_path)

    required_meta = [
        "dataset",
        "group_id",
        "donor_id",
        "tissue",
        "cell_type",
        "major_lineage",
        "n_cells",
    ]
    missing_meta = [c for c in required_meta if c not in pb.columns]
    if missing_meta:
        raise RuntimeError(
            f"{dataset}: missing metadata columns: {missing_meta}"
        )

    pb = pb[
        pb["major_lineage"].astype(str).isin(LINEAGE_ORDER)
    ].copy()

    expanded_cols = feature_columns(pb, EXPANDED17)
    core_cols = feature_columns(pb, core_genes)

    print(
        f"[{dataset}] rows={len(pb)}, donors={pb.donor_id.nunique()}, "
        f"tissues={pb.tissue.nunique()}, cell_types={pb.cell_type.nunique()}"
    )

    out = mkdir(outroot / dataset)
    fig1 = mkdir(out / "Figure1_like")
    fig2 = mkdir(out / "Figure2_like")
    src = mkdir(out / "source_data")

    build_figure1(
        dataset,
        pb,
        core_cols,
        expanded_cols,
        fig1,
        src,
    )

    build_figure2(
        dataset,
        pb,
        expanded_cols,
        droot,
        rigor,
        n_boot,
        fig2,
        src,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--root",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--outdir",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Higher-level Bayesian bootstrap draws for lineage-specific Figure 2B intervals.",
    )

    args = ap.parse_args()

    project = args.project.resolve()
    root = args.root.resolve()
    outroot = mkdir(args.outdir.resolve())

    print("=" * 108)
    print("HCL / AHCA: REPRODUCE CURRENT TABULA FIGURE 1 + FIGURE 2")
    print("=" * 108)
    print(f"PROJECT: {project}")
    print(f"ROOT   : {root}")
    print(f"OUT    : {outroot}")
    print(f"BOOT   : {args.bootstrap}")

    core_genes = recover_core_genes(project)

    rigor_path = (
        root
        / "03_RIGOR_VALIDATION_V1_20260921"
        / "FINAL_RIGOR_SUMMARY.csv"
    )
    rigor = read_csv(rigor_path)

    for dataset in ["HCL", "AHCA"]:
        run_dataset(
            dataset,
            root,
            project,
            outroot,
            core_genes,
            rigor,
            args.bootstrap,
        )

    readiness = pd.DataFrame([
        detect_fig3_readiness(root, "HCL"),
        detect_fig3_readiness(root, "AHCA"),
    ])
    readiness.to_csv(
        outroot / "FIG3_READINESS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "status": "COMPLETE",
        "analysis": "External reproduction of current Tabula Figure 1 and Figure 2",
        "datasets": ["HCL", "AHCA"],
        "feature_space": "Expanded17 primary; exact recovered Core panel for Core UMAP",
        "bootstrap_draws": args.bootstrap,
        "static_expression_only": True,
        "no_temporal_claims": True,
        "figure3_not_faked_from_clock_only_data": True,
        "figure3_readiness": readiness.to_dict("records"),
    }

    (outroot / "ANALYSIS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("COMPLETE")
    print("=" * 108)
    print("Generated:")
    print(outroot / "HCL" / "Figure1_like")
    print(outroot / "HCL" / "Figure2_like")
    print(outroot / "AHCA" / "Figure1_like")
    print(outroot / "AHCA" / "Figure2_like")
    print(outroot / "FIG3_READINESS.csv")
    print()
    print("Figure 3:")
    print(readiness.to_string(index=False))
    print("=" * 108)


if __name__ == "__main__":
    main()
