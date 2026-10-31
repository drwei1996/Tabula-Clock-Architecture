# -*- coding: utf-8 -*-
r"""
74_GSE292219_FIG1_3_MIRROR_VALIDATION.py

Mirror the original Figure 1–3 logic in the time-stamped external validation
dataset GSE292219.

FIG1-like:
  Clock17 lineage architecture using mesor, amplitude, relative amplitude,
  R2, and fitted phase; PCA/MDS + metric heatmaps.

FIG2-like:
  Quantify lineage separation within the same liver context using pairwise
  architecture distances, metric-block contributions, and leave-one-block-out
  sensitivity. This dataset alone cannot validate cross-tissue remodeling.

FIG3-like:
  Identify relatively conserved vs lineage-variable Clock17 components from
  cross-lineage variation in mesor, amplitude, relative amplitude, R2 and phase.
  If the frozen Fig3D pathway source exists locally and contains gene membership,
  validate only those pre-specified pathways. No pathway discovery is performed.

Secondary:
  Six-arm summary only.

Important:
  GSE292219 stores BMAL1 as gene_name="Bmal1"; Clock17 uses "Arntl".
  ARNTL therefore resolves to Bmal1. BMAL2/ARNTL2 is never substituted.
"""

from pathlib import Path
import re, json, warnings
from itertools import combinations

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. PATHS / CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"<TABULA_PROJECT_ROOT>"
)
ANALYSIS_ROOT = PROJECT_ROOT / "corrected_full_analysis_v2"

RAWDIR = (
    ANALYSIS_ROOT
    / "70_TIME_STAMPED_EXTERNAL_VALIDATION"
    / "01_raw"
    / "GSE292219"
)

ANN72_TABLEDIR = (
    ANALYSIS_ROOT
    / "72_GSE292219_ANNOTATION_AUDIT"
    / "01_tables"
)

OUTDIR = ANALYSIS_ROOT / "74_GSE292219_FIG1_3_MIRROR_VALIDATION"
FIG1DIR = OUTDIR / "01_FIG1_LIKE"
FIG2DIR = OUTDIR / "02_FIG2_LIKE"
FIG3DIR = OUTDIR / "03_FIG3_LIKE"
SUPPDIR = OUTDIR / "04_SIX_ARM_SECONDARY"
QCDIR = OUTDIR / "05_QC"
REPORTDIR = OUTDIR / "06_REPORT"

for p in [OUTDIR, FIG1DIR, FIG2DIR, FIG3DIR, SUPPDIR, QCDIR, REPORTDIR]:
    p.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "GSM8853297": {"ZT": 3,  "file": "GSM8853297_ZT3.h5ad"},
    "GSM8853298": {"ZT": 7,  "file": "GSM8853298_ZT7.h5ad"},
    "GSM8853299": {"ZT": 11, "file": "GSM8853299_ZT11.h5ad"},
    "GSM8853300": {"ZT": 15, "file": "GSM8853300_ZT15.h5ad"},
    "GSM8853301": {"ZT": 19, "file": "GSM8853301_ZT19.h5ad"},
    "GSM8853302": {"ZT": 23, "file": "GSM8853302_ZT23.h5ad"},
}

TARGET_LINEAGES = [
    "B cell",
    "T cell",
    "Macrophage/Monocyte",
    "Fibroblast",
    "Endothelial",
]

CLOCK17 = [
    "Arntl", "Clock", "Npas2",
    "Per1", "Per2", "Per3",
    "Cry1", "Cry2",
    "Rora", "Rorb", "Rorc",
    "Nr1d1", "Nr1d2",
    "Dbp", "Tef", "Hlf", "Nfil3",
]

SIX_ARMS = {
    "EBOX_POS": ["Arntl", "Clock", "Npas2"],
    "EBOX_NEG": ["Per1", "Per2", "Per3", "Cry1", "Cry2"],
    "RORE_ACT": ["Rora", "Rorb", "Rorc"],
    "RORE_REP": ["Nr1d1", "Nr1d2"],
    "DBOX_ACT": ["Dbp", "Tef", "Hlf"],
    "DBOX_REP": ["Nfil3"],
}

METRIC_BLOCKS = [
    "mesor",
    "amplitude",
    "relative_amplitude",
    "r2",
    "phase",
]

LINEAGE_COLORS = {
    "B cell": "#4C78A8",
    "T cell": "#E45756",
    "Macrophage/Monocyte": "#59A14F",
    "Fibroblast": "#F28E2B",
    "Endothelial": "#9467BD",
}

SOURCE_DATA_DIR = ANALYSIS_ROOT / "source_data"
FROZEN_SOURCES = {
    "Fig1A": SOURCE_DATA_DIR / "Fig1A_clock_architecture_pca_landscape_source.tsv",
    "Fig1B": SOURCE_DATA_DIR / "Fig1B_clock_architecture_nonlinear_landscape_source.tsv",
    "Fig1C": SOURCE_DATA_DIR / "Fig1C_core_clock_heatmap_source.tsv",
    "Fig2A": SOURCE_DATA_DIR / "Fig2A_formal_variance_partition_source.tsv",
    "Fig3A": SOURCE_DATA_DIR / "Fig3A_conserved_context_stability_source.tsv",
    "Fig3B": SOURCE_DATA_DIR / "Fig3B_lineage_tissue_clock_atlas_source.tsv",
    "Fig3C": SOURCE_DATA_DIR / "Fig3C_same_donor_cross_tissue_difference_map_source.tsv",
    "Fig3D": SOURCE_DATA_DIR / "Fig3D_pathway_context_partial_assoc_source.tsv",
}
FROZEN_FIG3D = FROZEN_SOURCES["Fig3D"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 1. HELPERS
# ============================================================

def log(msg):
    print(msg, flush=True)


def sanitize(x):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")


def savefig(fig, outdir, stem):
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def gene_lookup(names):
    lookup = {str(g).upper(): str(g) for g in names}
    if "ARNTL" not in lookup and "BMAL1" in lookup:
        lookup["ARNTL"] = lookup["BMAL1"]
    return lookup


def set_gene_names(a):
    if "gene_name" not in a.var.columns:
        raise RuntimeError("var['gene_name'] is missing")

    old = a.var_names.astype(str).copy()
    symbols = a.var["gene_name"].astype(str).copy()
    bad = symbols.isna() | symbols.eq("") | symbols.eq("nan")
    symbols.loc[bad] = old[bad]
    a.var_names = pd.Index(symbols)
    a.var_names_make_unique()
    return a


def get_norm10k(a):
    if "norm10k" not in a.layers:
        raise RuntimeError("layer['norm10k'] is missing")
    return a.layers["norm10k"]


def mean_rows(X, idx):
    sub = X[idx, :]
    if sp.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel()
    return np.asarray(sub).mean(axis=0)


def zscore_safe(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros(len(x), dtype=float)
    return (x - mu) / sd


def circular_dispersion_hours(phases_h):
    x = np.asarray(phases_h, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan

    theta = x / 24.0 * 2.0 * np.pi
    c = np.mean(np.cos(theta))
    s = np.mean(np.sin(theta))
    R = np.sqrt(c**2 + s**2)

    circ_sd = np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-8))))
    return float(circ_sd * 24.0 / (2.0 * np.pi))


# ============================================================
# 2. HARMONIC MODEL
# ============================================================

def harmonic_fit(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(t) & np.isfinite(y)
    t = t[ok]
    y = y[ok]

    if len(y) < 4:
        return {
            "mesor": np.nan,
            "amplitude": np.nan,
            "relative_amplitude": np.nan,
            "phase_h": np.nan,
            "r2": np.nan,
            "p_harmonic_descriptive": np.nan,
        }

    w = 2.0 * np.pi / 24.0
    X = np.column_stack([
        np.ones(len(t)),
        np.cos(w * t),
        np.sin(w * t),
    ])

    m = sm.OLS(y, X).fit()

    mesor = float(m.params[0])
    bcos = float(m.params[1])
    bsin = float(m.params[2])

    amp = float(np.sqrt(bcos**2 + bsin**2))
    phase_h = float(
        (np.arctan2(bsin, bcos) % (2.0 * np.pi))
        * 24.0
        / (2.0 * np.pi)
    )

    rel_amp = (
        float(amp / abs(mesor))
        if np.isfinite(mesor) and abs(mesor) > 1e-12
        else np.nan
    )

    try:
        R = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        p = float(np.asarray(m.f_test(R).pvalue).ravel()[0])
    except Exception:
        p = np.nan

    return {
        "mesor": mesor,
        "amplitude": amp,
        "relative_amplitude": rel_amp,
        "phase_h": phase_h,
        "r2": float(m.rsquared),
        "p_harmonic_descriptive": p,
    }


# ============================================================
# 3. LOAD FROZEN ANNOTATIONS
# ============================================================

def load_annotations_72():
    combined = ANN72_TABLEDIR / "GSE292219_all_cells_annotation_audit.csv.gz"

    if combined.exists():
        ann = pd.read_csv(combined)
    else:
        # Robust fallback to six single-sample annotation files.
        files = sorted(ANN72_TABLEDIR.glob("*_cell_annotation_audit.csv.gz"))
        if len(files) != 6:
            raise FileNotFoundError(
                "Could not find combined 72 annotation table and did not find "
                f"exactly six single-sample annotation files in:\n{ANN72_TABLEDIR}"
            )
        ann = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)

    required = {
        "cell_id",
        "sample_id",
        "ZT",
        "high_confidence_lineage",
    }
    missing = required - set(ann.columns)
    if missing:
        raise RuntimeError(
            "Missing columns in frozen 72 annotation table: "
            + ", ".join(sorted(missing))
        )

    ann = ann[
        ann["high_confidence_lineage"].isin(TARGET_LINEAGES)
    ].copy()

    return ann


# ============================================================
# 4. AGGREGATE LINEAGE x ZT EXPRESSION
# ============================================================

def aggregate_expression(ann):
    series_list = []
    meta_rows = []
    gene_sets = []

    for gsm, cfg in SAMPLES.items():
        path = RAWDIR / cfg["file"]
        if not path.exists():
            raise FileNotFoundError(path)

        log("\n" + "=" * 82)
        log(f"[aggregate] {gsm} ZT{cfg['ZT']}")

        a = sc.read_h5ad(path)
        a = set_gene_names(a)
        X = get_norm10k(a)

        genes = a.var_names.astype(str).tolist()
        gene_sets.append(set(genes))

        obs_idx = pd.Series(
            np.arange(a.n_obs, dtype=int),
            index=a.obs_names.astype(str)
        )

        sample_ann = ann[ann["sample_id"] == gsm]

        for lineage in TARGET_LINEAGES:
            ids = (
                sample_ann.loc[
                    sample_ann["high_confidence_lineage"] == lineage,
                    "cell_id"
                ]
                .astype(str)
                .tolist()
            )

            idx = (
                obs_idx.reindex(ids)
                .dropna()
                .astype(int)
                .values
            )

            if len(idx) < 10:
                raise RuntimeError(
                    f"{gsm} ZT{cfg['ZT']} {lineage}: "
                    f"only {len(idx)} high-confidence cells"
                )

            vec = mean_rows(X, idx)

            series_list.append(
                pd.Series(
                    vec,
                    index=genes,
                    dtype=float,
                    name=(gsm, cfg["ZT"], lineage)
                )
            )

            meta_rows.append({
                "sample_id": gsm,
                "ZT": int(cfg["ZT"]),
                "lineage": lineage,
                "n_cells": int(len(idx)),
            })

            log(f"  {lineage}: n={len(idx)}")

        del a
        del X

    common = sorted(set.intersection(*gene_sets))
    log(f"\n[aggregate] common genes={len(common):,}")

    expr = pd.DataFrame(
        [s.reindex(common).values for s in series_list],
        columns=common
    )
    meta = pd.DataFrame(meta_rows)

    pb = pd.concat(
        [meta.reset_index(drop=True), expr.reset_index(drop=True)],
        axis=1
    )

    pb = pb.sort_values(["lineage", "ZT"]).reset_index(drop=True)

    pb.to_pickle(QCDIR / "74_lineage_ZT_mean_norm10k.pkl")
    pb.to_csv(
        QCDIR / "74_lineage_ZT_mean_norm10k.csv.gz",
        index=False,
        compression="gzip"
    )

    return pb


# ============================================================
# 5. CLOCK17 HARMONICS
# ============================================================

def fit_clock17(pb):
    gm = gene_lookup(pb.columns)

    audit = pd.DataFrame([
        {
            "gene": gene,
            "available": gene.upper() in gm,
            "resolved_column": gm.get(gene.upper(), ""),
        }
        for gene in CLOCK17
    ])
    audit.to_csv(QCDIR / "74_Clock17_availability.csv", index=False)

    log("\n[Clock17 availability]")
    log(audit.to_string(index=False))

    rows = []

    for lineage in TARGET_LINEAGES:
        d = pb[pb["lineage"] == lineage].sort_values("ZT")

        for gene in CLOCK17:
            if gene.upper() not in gm:
                continue

            col = gm[gene.upper()]
            fit = harmonic_fit(
                d["ZT"].values,
                d[col].values
            )

            rows.append({
                "lineage": lineage,
                "gene": gene,
                **fit,
            })

    out = pd.DataFrame(rows)
    out.to_csv(QCDIR / "74_Clock17_harmonics.csv", index=False)
    return out, gm


# ============================================================
# 6. FIG1-LIKE ARCHITECTURE MATRIX / LANDSCAPE
# ============================================================

def build_architecture_matrix(clock_fit):
    rows = []

    for lineage in TARGET_LINEAGES:
        d = (
            clock_fit[clock_fit["lineage"] == lineage]
            .set_index("gene")
            .reindex(CLOCK17)
        )

        row = {"lineage": lineage}

        for gene in CLOCK17:
            if gene not in d.index:
                continue

            x = d.loc[gene]

            for metric in [
                "mesor",
                "amplitude",
                "relative_amplitude",
                "r2",
            ]:
                row[f"{gene}__{metric}"] = x[metric]

            phase = float(x["phase_h"])
            theta = phase / 24.0 * 2.0 * np.pi

            row[f"{gene}__phase_cos"] = np.cos(theta)
            row[f"{gene}__phase_sin"] = np.sin(theta)

        rows.append(row)

    mat = pd.DataFrame(rows)

    mat.to_csv(
        FIG1DIR / "74_FIG1A_lineage_clock_architecture_feature_matrix.csv",
        index=False
    )

    return mat


def standardize_architecture(arch):
    X = arch.set_index("lineage").copy()

    # Replace any non-finite values conservatively by feature median.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda s: s.fillna(s.median()), axis=0)

    Xz = pd.DataFrame(
        StandardScaler().fit_transform(X.values),
        index=X.index,
        columns=X.columns
    )

    Xz.to_csv(
        FIG1DIR / "74_FIG1A_standardized_architecture_matrix.csv"
    )

    return Xz


def plot_pca(Xz):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Xz.values)

    out = pd.DataFrame({
        "lineage": Xz.index,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })
    out["PC1_var"] = pca.explained_variance_ratio_[0]
    out["PC2_var"] = pca.explained_variance_ratio_[1]

    out.to_csv(
        FIG1DIR / "74_FIG1A_dynamic_architecture_PCA.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    for _, r in out.iterrows():
        lineage = r["lineage"]
        ax.scatter(
            r["PC1"],
            r["PC2"],
            s=100,
            c=LINEAGE_COLORS[lineage]
        )
        ax.text(
            r["PC1"],
            r["PC2"],
            "  " + lineage,
            va="center",
            fontsize=8
        )

    ax.set_xlabel(
        f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"
    )
    ax.set_title(
        "Dynamic clock architecture landscape"
    )

    savefig(
        fig,
        FIG1DIR,
        "74_FIG1A_DYNAMIC_ARCHITECTURE_PCA"
    )

    return out


def plot_mds(Xz):
    dist = squareform(pdist(Xz.values, metric="euclidean"))

    try:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=20261003,
            n_init=50,
            max_iter=1000,
            normalized_stress="auto"
        )
    except TypeError:
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=20261003,
            n_init=50,
            max_iter=1000
        )

    coords = mds.fit_transform(dist)

    out = pd.DataFrame({
        "lineage": Xz.index,
        "MDS1": coords[:, 0],
        "MDS2": coords[:, 1],
    })

    out.to_csv(
        FIG1DIR / "74_FIG1B_dynamic_architecture_MDS.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    for _, r in out.iterrows():
        lineage = r["lineage"]
        ax.scatter(
            r["MDS1"],
            r["MDS2"],
            s=100,
            c=LINEAGE_COLORS[lineage]
        )
        ax.text(
            r["MDS1"],
            r["MDS2"],
            "  " + lineage,
            va="center",
            fontsize=8
        )

    ax.set_xlabel("MDS1")
    ax.set_ylabel("MDS2")
    ax.set_title("Nonlinear dynamic architecture landscape")

    savefig(
        fig,
        FIG1DIR,
        "74_FIG1B_DYNAMIC_ARCHITECTURE_MDS"
    )

    return out


def metric_matrix(clock_fit, metric):
    return (
        clock_fit.pivot(
            index="gene",
            columns="lineage",
            values=metric
        )
        .reindex(CLOCK17)
        .reindex(columns=TARGET_LINEAGES)
    )


def plot_metric_heatmap(
    clock_fit,
    metric,
    title,
    stem,
    standardize_rows=False,
    vmin=None,
    vmax=None
):
    mat = metric_matrix(clock_fit, metric)

    if standardize_rows:
        plot_mat = pd.DataFrame(
            [zscore_safe(row.values) for _, row in mat.iterrows()],
            index=mat.index,
            columns=mat.columns
        )
    else:
        plot_mat = mat.copy()

    plot_mat.to_csv(FIG1DIR / f"{stem}.csv")

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    im = ax.imshow(
        plot_mat.values,
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks(np.arange(plot_mat.shape[1]))
    ax.set_xticklabels(
        plot_mat.columns,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(np.arange(plot_mat.shape[0]))
    ax.set_yticklabels(plot_mat.index)
    ax.set_title(title)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.02
    )
    cbar.set_label(metric)

    savefig(fig, FIG1DIR, stem)


# ============================================================
# 7. FIG2-LIKE LINEAGE SEPARATION
# ============================================================

def pairwise_lineage_distances(Xz):
    dist = squareform(pdist(Xz.values, metric="euclidean"))

    mat = pd.DataFrame(
        dist,
        index=Xz.index,
        columns=Xz.index
    )

    mat.to_csv(
        FIG2DIR / "74_FIG2A_pairwise_lineage_architecture_distance.csv"
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(mat.values, aspect="auto")

    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(
        mat.columns,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)

    ax.set_title(
        "Pairwise lineage distance in dynamic clock architecture"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.02
    )
    cbar.set_label("Euclidean distance")

    savefig(
        fig,
        FIG2DIR,
        "74_FIG2A_LINEAGE_DISTANCE"
    )

    return mat


def block_columns(X, block):
    if block == "phase":
        return [
            c
            for c in X.columns
            if "__phase_cos" in c or "__phase_sin" in c
        ]

    return [
        c
        for c in X.columns
        if c.endswith("__" + block)
    ]


def block_variance_decomposition(arch):
    X = arch.set_index("lineage").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda s: s.fillna(s.median()), axis=0)

    rows = []

    for block in METRIC_BLOCKS:
        cols = block_columns(X, block)
        if not cols:
            continue

        arr = StandardScaler().fit_transform(
            X[cols].values
        )

        # Mean standardized sum-of-squares per feature:
        # normalizes for the fact that phase has 2x as many columns.
        total_ss = float(np.sum(arr**2))
        ss_per_feature = total_ss / len(cols)

        rows.append({
            "metric_block": block,
            "n_features": len(cols),
            "standardized_SS": total_ss,
            "SS_per_feature": ss_per_feature,
        })

    out = pd.DataFrame(rows)

    out["fraction_of_SS_per_feature_sum"] = (
        out["SS_per_feature"]
        / out["SS_per_feature"].sum()
    )

    out.to_csv(
        FIG2DIR / "74_FIG2B_metric_block_variance_contribution.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(
        out["metric_block"],
        out["fraction_of_SS_per_feature_sum"]
    )
    ax.set_ylabel(
        "Fraction of normalized lineage dispersion"
    )
    ax.set_title(
        "Architecture components contributing to lineage separation"
    )
    ax.tick_params(axis="x", rotation=30)

    savefig(
        fig,
        FIG2DIR,
        "74_FIG2B_ARCHITECTURE_COMPONENT_VARIANCE"
    )

    return out


def leave_one_block_out(arch):
    X = arch.set_index("lineage").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda s: s.fillna(s.median()), axis=0)

    full_z = StandardScaler().fit_transform(X.values)
    full_dist = squareform(
        pdist(full_z, metric="euclidean")
    )

    tri = np.triu_indices(len(X), k=1)
    full_vec = full_dist[tri]

    rows = []

    for block in METRIC_BLOCKS:
        drop_cols = block_columns(X, block)
        keep_cols = [
            c
            for c in X.columns
            if c not in drop_cols
        ]

        z2 = StandardScaler().fit_transform(
            X[keep_cols].values
        )
        d2 = squareform(
            pdist(z2, metric="euclidean")
        )
        vec2 = d2[tri]

        rho = spearmanr(full_vec, vec2).statistic

        rows.append({
            "removed_block": block,
            "n_removed_features": len(drop_cols),
            "distance_pattern_spearman_vs_full": float(rho),
            "mean_pairwise_distance_full": float(np.mean(full_vec)),
            "mean_pairwise_distance_without_block": float(np.mean(vec2)),
            "mean_pairwise_distance_drop": float(
                np.mean(full_vec) - np.mean(vec2)
            ),
        })

    out = pd.DataFrame(rows)
    out.to_csv(
        FIG2DIR / "74_FIG2C_leave_one_metric_block_out.csv",
        index=False
    )
    return out


# ============================================================
# 8. FIG3-LIKE CONSERVED VS VARIABLE COMPONENTS
# ============================================================

def cv_abs(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(np.abs(x))

    if not np.isfinite(mu) or mu <= 1e-12:
        return np.nan

    return float(np.nanstd(x) / mu)


def component_variability(clock_fit):
    rows = []

    for gene in CLOCK17:
        d = (
            clock_fit[clock_fit["gene"] == gene]
            .set_index("lineage")
            .reindex(TARGET_LINEAGES)
        )

        rows.append({
            "gene": gene,
            "mesor_cv": cv_abs(d["mesor"].values),
            "amplitude_cv": cv_abs(d["amplitude"].values),
            "relative_amplitude_cv": cv_abs(
                d["relative_amplitude"].values
            ),
            "r2_sd": float(np.nanstd(d["r2"].values)),
            "phase_dispersion_h": circular_dispersion_hours(
                d["phase_h"].values
            ),
        })

    out = pd.DataFrame(rows)

    var_cols = [
        "mesor_cv",
        "amplitude_cv",
        "relative_amplitude_cv",
        "r2_sd",
        "phase_dispersion_h",
    ]

    Z = np.vstack([
        zscore_safe(out[c].values)
        for c in var_cols
    ])

    out["composite_lineage_variability"] = np.mean(Z, axis=0)

    # Descriptive split only; do not over-interpret as a hard biological class.
    med = np.nanmedian(
        out["composite_lineage_variability"]
    )

    out["architecture_class"] = np.where(
        out["composite_lineage_variability"] >= med,
        "lineage-variable",
        "relatively-conserved"
    )

    out.to_csv(
        FIG3DIR / "74_FIG3A_clock_component_lineage_variability.csv",
        index=False
    )

    return out


def plot_component_variability(var):
    order = var.sort_values(
        "composite_lineage_variability",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.bar(
        np.arange(len(order)),
        order["composite_lineage_variability"]
    )

    ax.axhline(
        np.nanmedian(
            order["composite_lineage_variability"]
        ),
        linestyle="--",
        linewidth=1
    )

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(
        order["gene"],
        rotation=45,
        ha="right"
    )

    ax.set_ylabel(
        "Composite cross-lineage variability"
    )
    ax.set_title(
        "Clock17 components: relatively conserved vs lineage-variable"
    )

    savefig(
        fig,
        FIG3DIR,
        "74_FIG3A_CLOCK_COMPONENT_VARIABILITY"
    )


def plot_component_variability_heatmap(var):
    cols = [
        "mesor_cv",
        "amplitude_cv",
        "relative_amplitude_cv",
        "r2_sd",
        "phase_dispersion_h",
    ]

    mat = (
        var.set_index("gene")[cols]
        .reindex(CLOCK17)
    )

    arr = StandardScaler().fit_transform(mat.values)

    zmat = pd.DataFrame(
        arr,
        index=mat.index,
        columns=mat.columns
    )

    zmat.to_csv(
        FIG3DIR / "74_FIG3B_component_variability_heatmap.csv"
    )

    fig, ax = plt.subplots(figsize=(6.4, 6.5))
    im = ax.imshow(
        zmat.values,
        aspect="auto",
        vmin=-2,
        vmax=2
    )

    ax.set_xticks(np.arange(zmat.shape[1]))
    ax.set_xticklabels(
        [
            "Mesor CV",
            "Amplitude CV",
            "Relative amp CV",
            "R² SD",
            "Phase dispersion",
        ],
        rotation=35,
        ha="right"
    )

    ax.set_yticks(np.arange(zmat.shape[0]))
    ax.set_yticklabels(zmat.index)
    ax.set_title(
        "Sources of lineage-specific clock variation"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.02
    )
    cbar.set_label("Standardized variability")

    savefig(
        fig,
        FIG3DIR,
        "74_FIG3B_COMPONENT_VARIABILITY_HEATMAP"
    )


# ============================================================
# 9. OPTIONAL FROZEN FIG3D PATHWAY VALIDATION
# ============================================================

def infer_pathway_col(df):
    preferred = [
        "pathway",
        "term",
        "gene_set",
        "geneset",
        "name",
        "description",
    ]

    lower = {str(c).lower(): c for c in df.columns}

    for x in preferred:
        if x in lower:
            return lower[x]

    for c in df.columns:
        cl = str(c).lower()
        if "pathway" in cl or "term" in cl:
            return c

    return None


def infer_gene_col(df):
    preferred = [
        "genes",
        "gene_members",
        "members",
        "gene_symbols",
        "leadingedge",
        "leading_edge",
        "core_enrichment",
    ]

    lower = {str(c).lower(): c for c in df.columns}

    for x in preferred:
        if x in lower:
            return lower[x]

    return None


def parse_genes(x):
    if pd.isna(x):
        return []

    return [
        p.strip()
        for p in re.split(r"[;,|/\s]+", str(x))
        if p.strip()
    ]


def load_frozen_fig3d_gene_sets():
    status = {
        "source_exists": FROZEN_FIG3D.exists(),
        "pathway_col": None,
        "gene_col": None,
        "n_pathways": 0,
        "n_pathways_with_gene_membership": 0,
    }

    if not FROZEN_FIG3D.exists():
        with open(
            FIG3DIR / "74_FIG3D_pathway_status.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(status, f, indent=2)
        return {}, status

    df = pd.read_csv(FROZEN_FIG3D, sep="\t")

    pcol = infer_pathway_col(df)
    gcol = infer_gene_col(df)

    status["pathway_col"] = str(pcol) if pcol is not None else None
    status["gene_col"] = str(gcol) if gcol is not None else None

    if pcol is None:
        return {}, status

    names = (
        df[pcol]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    status["n_pathways"] = len(names)

    gene_sets = {}

    if gcol is not None:
        for name, d in df.groupby(pcol, sort=False):
            genes = []
            for x in d[gcol]:
                genes.extend(parse_genes(x))

            genes = list(dict.fromkeys(genes))

            if genes:
                gene_sets[str(name)] = genes

    status["n_pathways_with_gene_membership"] = len(gene_sets)

    with open(
        FIG3DIR / "74_FIG3D_pathway_status.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            status,
            f,
            indent=2,
            ensure_ascii=False
        )

    return gene_sets, status


def frozen_pathway_validation(pb, gene_sets):
    if not gene_sets:
        return pd.DataFrame(), pd.DataFrame()

    gm = gene_lookup(pb.columns)
    coverage_rows = []
    fit_rows = []

    for pathway, genes in gene_sets.items():
        avail = [
            g
            for g in genes
            if g.upper() in gm
        ]

        coverage_rows.append({
            "pathway": pathway,
            "n_requested": len(genes),
            "n_available": len(avail),
            "coverage_fraction": (
                len(avail) / len(genes)
                if genes
                else np.nan
            ),
        })

    coverage = pd.DataFrame(coverage_rows)

    coverage.to_csv(
        FIG3DIR / "74_FIG3D_frozen_pathway_gene_coverage.csv",
        index=False
    )

    usable = (
        coverage.loc[
            coverage["n_available"] >= 5,
            "pathway"
        ]
        .tolist()
    )

    for lineage in TARGET_LINEAGES:
        d = pb[pb["lineage"] == lineage].sort_values("ZT")

        for pathway in usable:
            genes = gene_sets[pathway]
            avail = [
                g
                for g in genes
                if g.upper() in gm
            ]

            gene_z = []

            for gene in avail:
                col = gm[gene.upper()]
                gene_z.append(
                    zscore_safe(
                        d[col].values.astype(float)
                    )
                )

            score = np.mean(
                np.vstack(gene_z),
                axis=0
            )

            fit = harmonic_fit(
                d["ZT"].values,
                score
            )

            fit_rows.append({
                "lineage": lineage,
                "pathway": pathway,
                "n_genes": len(avail),
                **fit,
            })

    fits = pd.DataFrame(fit_rows)

    fits.to_csv(
        FIG3DIR / "74_FIG3D_frozen_pathway_temporal_validation.csv",
        index=False
    )

    return coverage, fits


# ============================================================
# 10. SECONDARY SIX-ARM
# ============================================================

def six_arm_secondary(pb):
    gm = gene_lookup(pb.columns)

    scores = []
    fits = []

    for lineage in TARGET_LINEAGES:
        d = pb[pb["lineage"] == lineage].sort_values("ZT")

        for arm, genes in SIX_ARMS.items():
            avail = [
                g
                for g in genes
                if g.upper() in gm
            ]

            if not avail:
                continue

            gene_z = []

            for gene in avail:
                col = gm[gene.upper()]
                gene_z.append(
                    zscore_safe(
                        d[col].values.astype(float)
                    )
                )

            score = np.mean(
                np.vstack(gene_z),
                axis=0
            )

            for zt, val in zip(d["ZT"].astype(int), score):
                scores.append({
                    "lineage": lineage,
                    "arm": arm,
                    "ZT": int(zt),
                    "score_z": float(val),
                })

            fits.append({
                "lineage": lineage,
                "arm": arm,
                **harmonic_fit(
                    d["ZT"].values,
                    score
                ),
            })

    score_df = pd.DataFrame(scores)
    fit_df = pd.DataFrame(fits)

    score_df.to_csv(
        SUPPDIR / "74_six_arm_scores.csv",
        index=False
    )
    fit_df.to_csv(
        SUPPDIR / "74_six_arm_harmonics.csv",
        index=False
    )

    return score_df, fit_df


# ============================================================
# 11. FROZEN SOURCE AUDIT
# ============================================================

def audit_frozen_sources():
    rows = []

    for name, path in FROZEN_SOURCES.items():
        row = {
            "source": name,
            "path": str(path),
            "exists": path.exists(),
            "n_rows": np.nan,
            "n_cols": np.nan,
        }

        if path.exists():
            try:
                d = pd.read_csv(path, sep="\t")
                row["n_rows"] = d.shape[0]
                row["n_cols"] = d.shape[1]
            except Exception:
                pass

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(
        QCDIR / "74_frozen_source_audit.csv",
        index=False
    )
    return out


# ============================================================
# 12. REPORT
# ============================================================

def write_report(
    pca_out,
    mds_out,
    block_var,
    leave_out,
    variability,
    pathway_status,
    frozen_audit
):
    lines = [
        "74 GSE292219 FIG1-3 MIRROR VALIDATION",
        "=" * 78,
        "",
        "PRIMARY QUESTION",
        "Does the external time-stamped dataset reproduce lineage-specific",
        "circadian architecture when architecture is defined using mesor,",
        "amplitude, relative amplitude, temporal robustness (R2), and phase?",
        "",
        "FIG1-LIKE",
        "---------",
        pca_out.to_string(index=False),
        "",
        "MDS:",
        mds_out.to_string(index=False),
        "",
        "FIG2-LIKE",
        "---------",
        "GSE292219 contains liver only. Cross-tissue remodeling is therefore",
        "not testable here. This module tests whether lineage identity remains",
        "associated with distinct clock architecture within a shared tissue.",
        "",
        "Metric-block contribution:",
        block_var.to_string(index=False),
        "",
        "Leave-one-block-out sensitivity:",
        leave_out.to_string(index=False),
        "",
        "FIG3-LIKE",
        "---------",
        "Clock components are described as relatively conserved or",
        "lineage-variable based on cross-lineage variation in mesor, amplitude,",
        "relative amplitude, R2, and phase dispersion.",
        "",
        variability[
            [
                "gene",
                "mesor_cv",
                "amplitude_cv",
                "relative_amplitude_cv",
                "r2_sd",
                "phase_dispersion_h",
                "composite_lineage_variability",
                "architecture_class",
            ]
        ].to_string(index=False),
        "",
        "Frozen Fig3D pathway status:",
        json.dumps(pathway_status, indent=2, ensure_ascii=False),
        "",
        "IMPORTANT INTERPRETATION",
        "Similar phase of selected core genes does not imply identical lineage",
        "clock architecture. Architecture differences may arise from expression",
        "set-point, rhythmic amplitude, temporal robustness, and selected",
        "relative-phase organization.",
        "",
        "Frozen source audit:",
        frozen_audit.to_string(index=False),
    ]

    (
        REPORTDIR
        / "74_GSE292219_FIG1_3_MIRROR_VALIDATION_REPORT.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():
    log("=" * 96)
    log("74 GSE292219 FIG1-3 MIRROR VALIDATION")
    log(
        "FIG1-like architecture -> FIG2-like lineage separation -> "
        "FIG3-like conserved/variable components"
    )
    log("=" * 96)

    frozen_audit = audit_frozen_sources()

    ann = load_annotations_72()
    pb = aggregate_expression(ann)
    clock_fit, gm = fit_clock17(pb)

    # FIG1-like
    arch = build_architecture_matrix(clock_fit)
    Xz = standardize_architecture(arch)

    pca_out = plot_pca(Xz)
    mds_out = plot_mds(Xz)

    plot_metric_heatmap(
        clock_fit,
        "mesor",
        "Clock17 mesor across lineages",
        "74_FIG1C_CLOCK17_MESOR",
        standardize_rows=True,
        vmin=-2,
        vmax=2
    )

    plot_metric_heatmap(
        clock_fit,
        "amplitude",
        "Clock17 rhythmic amplitude across lineages",
        "74_FIG1C_CLOCK17_AMPLITUDE",
        standardize_rows=True,
        vmin=-2,
        vmax=2
    )

    plot_metric_heatmap(
        clock_fit,
        "relative_amplitude",
        "Clock17 relative amplitude across lineages",
        "74_FIG1C_CLOCK17_RELATIVE_AMPLITUDE",
        standardize_rows=True,
        vmin=-2,
        vmax=2
    )

    plot_metric_heatmap(
        clock_fit,
        "r2",
        "Clock17 temporal robustness across lineages",
        "74_FIG1C_CLOCK17_R2",
        standardize_rows=False,
        vmin=0,
        vmax=1
    )

    plot_metric_heatmap(
        clock_fit,
        "phase_h",
        "Clock17 fitted phase across lineages",
        "74_FIG1C_CLOCK17_PHASE",
        standardize_rows=False,
        vmin=0,
        vmax=24
    )

    # FIG2-like
    dist_mat = pairwise_lineage_distances(Xz)
    block_var = block_variance_decomposition(arch)
    leave_out = leave_one_block_out(arch)

    # FIG3-like
    variability = component_variability(clock_fit)
    plot_component_variability(variability)
    plot_component_variability_heatmap(variability)

    gene_sets, pathway_status = load_frozen_fig3d_gene_sets()
    pathway_coverage, pathway_fit = frozen_pathway_validation(
        pb,
        gene_sets
    )

    # Secondary six-arm
    six_scores, six_fit = six_arm_secondary(pb)

    write_report(
        pca_out,
        mds_out,
        block_var,
        leave_out,
        variability,
        pathway_status,
        frozen_audit
    )

    completion = {
        "pipeline": "74_GSE292219_FIG1_3_MIRROR_VALIDATION",
        "status": "COMPLETE",
        "lineages": TARGET_LINEAGES,
        "clock_genes": CLOCK17,
        "FIG1_like": "multivariate lineage clock architecture",
        "FIG2_like": "lineage separation within shared liver context",
        "FIG3_like": "relatively conserved vs lineage-variable components",
        "cross_tissue_claim": "NOT_TESTABLE_IN_GSE292219_ALONE",
        "pathway_status": pathway_status,
    }

    with open(
        OUTDIR / "74_GSE292219_COMPLETE.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            completion,
            f,
            indent=2,
            ensure_ascii=False
        )

    log("\n" + "=" * 96)
    log("74 GSE292219 FIG1-3 MIRROR VALIDATION COMPLETE")
    log("=" * 96)
    log(f"\nOutput: {OUTDIR}")
    log("\nKey outputs:")
    log("  01_FIG1_LIKE/74_FIG1A_DYNAMIC_ARCHITECTURE_PCA.*")
    log("  01_FIG1_LIKE/74_FIG1B_DYNAMIC_ARCHITECTURE_MDS.*")
    log("  01_FIG1_LIKE/74_FIG1C_CLOCK17_MESOR.*")
    log("  01_FIG1_LIKE/74_FIG1C_CLOCK17_AMPLITUDE.*")
    log("  02_FIG2_LIKE/74_FIG2A_LINEAGE_DISTANCE.*")
    log("  02_FIG2_LIKE/74_FIG2B_ARCHITECTURE_COMPONENT_VARIANCE.*")
    log("  03_FIG3_LIKE/74_FIG3A_CLOCK_COMPONENT_VARIABILITY.*")
    log("  03_FIG3_LIKE/74_FIG3B_COMPONENT_VARIABILITY_HEATMAP.*")
    log("  06_REPORT/74_GSE292219_FIG1_3_MIRROR_VALIDATION_REPORT.txt")


if __name__ == "__main__":
    main()
