#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os

import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from scipy.stats import (
    kruskal,
    mannwhitneyu,
    spearmanr,
)
from sklearn.decomposition import PCA


# =============================================================================
# 0. GLOBAL CONFIG
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

FROZEN_META = (
    ROOT
    / "00_freeze"
    / "observation_metadata_corrected_v3_FROZEN.tsv"
)

NPZ_PATH = (
    PROJECT
    / "results"
    / "phaseA_clock_architecture"
    / "dependencies"
    / "broad_normalized_mean.npz"
)

OUT = ROOT / "36_STATIC_3LOOP_ARCHITECTURE_20260930"

DIR = {
    "audit": OUT / "00_AUDIT",
    "gene": OUT / "01_GENE_LEVEL",
    "arm": OUT / "02_ARM_SCORES",
    "loop": OUT / "03_LOOP_SCORES",
    "relative": OUT / "04_RELATIVE_ARCHITECTURE",
    "lineage": OUT / "05_LINEAGE",
    "tissue": OUT / "06_TISSUE_STABILITY",
    "fig": OUT / "07_FIGURES",
    "source": OUT / "source_data",
}

for p in DIR.values():
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
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
# 1. FROZEN LINEAGES
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

EXPECTED_COUNTS = {
    "b_cell": 153,
    "t_cell": 437,
    "macrophage_monocyte": 280,
    "fibroblast": 148,
    "endothelial": 206,
}


# =============================================================================
# 2. THREE-LOOP DEFINITIONS
# =============================================================================

ARMS = {
    "EBOX_POS": ["ARNTL", "CLOCK", "NPAS2"],
    "EBOX_NEG": ["PER1", "PER2", "PER3", "CRY1", "CRY2"],
    "RORE_ACT": ["RORA", "RORB", "RORC"],
    "RORE_REP": ["NR1D1", "NR1D2"],
    "DBOX_ACT": ["DBP", "TEF", "HLF"],
    "DBOX_REP": ["NFIL3"],
}

LOOPS = {
    "EBOX": {
        "positive": "EBOX_POS",
        "negative": "EBOX_NEG",
    },
    "RORE": {
        "positive": "RORE_ACT",
        "negative": "RORE_REP",
    },
    "DBOX": {
        "positive": "DBOX_ACT",
        "negative": "DBOX_REP",
    },
}

ALL_CLOCK_GENES = [
    g
    for arm in [
        "EBOX_POS",
        "EBOX_NEG",
        "RORE_ACT",
        "RORE_REP",
        "DBOX_ACT",
        "DBOX_REP",
    ]
    for g in ARMS[arm]
]

assert len(ALL_CLOCK_GENES) == 17
assert len(set(ALL_CLOCK_GENES)) == 17


GENE_BLOCKS = [
    ("E-box +", "EBOX_POS", "#5470C6"),
    ("E-box −", "EBOX_NEG", "#91A7E0"),
    ("ROR", "RORE_ACT", "#D98C4A"),
    ("REV-ERB", "RORE_REP", "#EDB47A"),
    ("PAR-bZIP", "DBOX_ACT", "#5FA86A"),
    ("NFIL3", "DBOX_REP", "#9AC5A0"),
]


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


def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)

    sd_safe = sd.copy()
    sd_safe[~np.isfinite(sd_safe)] = 1.0
    sd_safe[sd_safe == 0] = 1.0

    z = (x - mu) / sd_safe

    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite value produced during z-scoring.")

    return z


def bootstrap_mean_ci(
    values,
    n_boot=5000,
    seed=20260930,
    alpha=0.05,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return values[0], values[0]

    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    means = values[idx].mean(axis=1)

    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def rank_biserial_from_u(u, n1, n2):
    """
    Rank-biserial correlation corresponding to Mann-Whitney U.
    Positive means group1 tends to be higher than group2.
    """
    if n1 == 0 or n2 == 0:
        return np.nan

    return (2 * u) / (n1 * n2) - 1


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)

    out = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    pv = p[valid]

    if len(pv) == 0:
        return out

    order = np.argsort(pv)
    ranks = np.arange(1, len(pv) + 1)

    adj = pv[order] * len(pv) / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)

    tmp = np.empty(len(pv))
    tmp[order] = adj

    out[valid] = tmp
    return out


def lineage_label(x):
    return LINEAGE_DISPLAY.get(x, x)


def ternary_xy(w_e, w_r, w_d):
    """
    Triangle:
        RORE
         /\
        /  \
    EBOX----DBOX

    EBOX = (0,0)
    DBOX = (1,0)
    RORE = (0.5,sqrt(3)/2)
    """
    x = w_d + 0.5 * w_r
    y = (math.sqrt(3) / 2) * w_r
    return x, y


# =============================================================================
# 4. LOAD FROZEN DATA
# =============================================================================

def load_data():
    print("=" * 100)
    print("Loading frozen metadata and expression matrix")
    print("=" * 100)

    if not FROZEN_META.exists():
        raise FileNotFoundError(FROZEN_META)

    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)

    meta = pd.read_csv(
        FROZEN_META,
        sep="\t",
        dtype={"group_id": str},
    )

    if "priority_lineage_corrected" in meta.columns:
        meta = meta.rename(
            columns={
                "priority_lineage_corrected": "priority_lineage"
            }
        )

    required_meta = [
        "group_id",
        "donor_id",
        "tissue",
        "priority_lineage",
    ]

    missing = [
        c for c in required_meta
        if c not in meta.columns
    ]

    if missing:
        raise RuntimeError(
            f"Frozen metadata missing columns: {missing}"
        )

    meta["group_id"] = meta["group_id"].astype(str)
    meta = meta.set_index("group_id", drop=False)

    if len(meta) != 2030:
        raise RuntimeError(
            f"Expected 2030 metadata rows, got {len(meta)}"
        )

    priority = meta[
        meta["priority_lineage"].notna()
    ].copy()

    observed_counts = (
        priority["priority_lineage"]
        .value_counts()
        .to_dict()
    )

    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "Frozen lineage counts changed.\n"
            f"Expected: {EXPECTED_COUNTS}\n"
            f"Observed: {observed_counts}"
        )

    if len(priority) != 1224:
        raise RuntimeError(
            f"Expected 1224 priority-lineage rows, got {len(priority)}"
        )

    data = np.load(
        NPZ_PATH,
        allow_pickle=True,
    )

    ids = data["group_id"].astype(str)
    genes = data["genes"].astype(str)
    expr = np.asarray(data["expression"], dtype=float)

    if expr.shape != (2030, 59410):
        raise RuntimeError(
            f"Unexpected expression matrix shape: {expr.shape}"
        )

    if len(set(genes)) != len(genes):
        raise RuntimeError(
            "Gene symbols are not unique."
        )

    if not np.isfinite(expr).all():
        raise RuntimeError(
            "Expression matrix contains non-finite values."
        )

    if set(ids) != set(meta.index):
        raise RuntimeError(
            "Expression and metadata group_id sets differ."
        )

    order = pd.Index(ids).get_indexer(meta.index)

    if (order < 0).any():
        raise RuntimeError(
            "Failed to align expression matrix to metadata."
        )

    expr = expr[order]

    return meta, priority, genes, expr


# =============================================================================
# 5. GENE AUDIT + EXTRACTION
# =============================================================================

def extract_clock_expression(meta, genes, expr):
    gene_index = {
        g: i
        for i, g in enumerate(genes)
    }

    audit = []

    for g in ALL_CLOCK_GENES:
        audit.append({
            "gene": g,
            "available": g in gene_index,
            "matrix_index": (
                gene_index[g]
                if g in gene_index
                else np.nan
            ),
        })

    audit = pd.DataFrame(audit)

    audit.to_csv(
        DIR["audit"] / "STATIC_3LOOP_GENE_AVAILABILITY.csv",
        index=False,
    )

    missing = audit.loc[
        ~audit["available"],
        "gene"
    ].tolist()

    if missing:
        raise RuntimeError(
            f"Required clock genes missing: {missing}"
        )

    idx = [
        gene_index[g]
        for g in ALL_CLOCK_GENES
    ]

    raw = pd.DataFrame(
        expr[:, idx],
        index=meta.index,
        columns=ALL_CLOCK_GENES,
    )

    z = pd.DataFrame(
        safe_zscore(raw.to_numpy()),
        index=raw.index,
        columns=raw.columns,
    )

    raw.to_csv(
        DIR["gene"] / "STATIC_3LOOP_CLOCK17_RAW_EXPRESSION.csv"
    )

    z.to_csv(
        DIR["gene"] / "STATIC_3LOOP_CLOCK17_ZSCORES.csv"
    )

    return raw, z


# =============================================================================
# 6. SCORES
# =============================================================================

def calculate_scores(meta, raw, z):
    df = meta[
        [
            "group_id",
            "donor_id",
            "tissue",
            "priority_lineage",
        ]
        + (
            ["cell_type"]
            if "cell_type" in meta.columns
            else []
        )
    ].copy()

    # -------------------------------------------------------------------------
    # raw + z genes
    # -------------------------------------------------------------------------

    for g in ALL_CLOCK_GENES:
        df[f"RAW_{g}"] = raw[g]
        df[f"Z_{g}"] = z[g]

    # -------------------------------------------------------------------------
    # six arm scores
    # -------------------------------------------------------------------------

    for arm, gs in ARMS.items():
        df[arm] = z[gs].mean(axis=1)

    # -------------------------------------------------------------------------
    # engagement
    # -------------------------------------------------------------------------

    df["EBOX_ENGAGEMENT"] = (
        df["EBOX_POS"]
        + df["EBOX_NEG"]
    ) / 2

    df["RORE_ENGAGEMENT"] = (
        df["RORE_ACT"]
        + df["RORE_REP"]
    ) / 2

    df["DBOX_ENGAGEMENT"] = (
        df["DBOX_ACT"]
        + df["DBOX_REP"]
    ) / 2

    # -------------------------------------------------------------------------
    # balance
    # -------------------------------------------------------------------------

    df["EBOX_BALANCE"] = (
        df["EBOX_POS"]
        - df["EBOX_NEG"]
    )

    df["RORE_BALANCE"] = (
        df["RORE_ACT"]
        - df["RORE_REP"]
    )

    df["DBOX_BALANCE"] = (
        df["DBOX_ACT"]
        - df["DBOX_REP"]
    )

    # -------------------------------------------------------------------------
    # relative architecture
    #
    # Re-z-score engagement across ALL 2030 observations.
    # -------------------------------------------------------------------------

    engagement_cols = [
        "EBOX_ENGAGEMENT",
        "RORE_ENGAGEMENT",
        "DBOX_ENGAGEMENT",
    ]

    engage_z = safe_zscore(
        df[engagement_cols].to_numpy()
    )

    for j, loop in enumerate(
        ["EBOX", "RORE", "DBOX"]
    ):
        df[f"{loop}_ENGAGEMENT_Z"] = engage_z[:, j]

    # numerically stable softmax
    ez = engage_z - engage_z.max(
        axis=1,
        keepdims=True,
    )

    ex = np.exp(ez)

    weights = ex / ex.sum(
        axis=1,
        keepdims=True,
    )

    df["EBOX_RELATIVE_WEIGHT"] = weights[:, 0]
    df["RORE_RELATIVE_WEIGHT"] = weights[:, 1]
    df["DBOX_RELATIVE_WEIGHT"] = weights[:, 2]

    sums = weights.sum(axis=1)

    if not np.allclose(
        sums,
        1.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Relative architecture weights do not sum to 1."
        )

    # -------------------------------------------------------------------------
    # entropy specificity
    # -------------------------------------------------------------------------

    w = np.clip(
        weights,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        w * np.log(w),
        axis=1,
    )

    entropy_norm = entropy / np.log(3)

    df["ARCHITECTURE_ENTROPY"] = entropy_norm
    df["ARCHITECTURE_SPECIFICITY"] = (
        1 - entropy_norm
    )

    loop_names = np.array([
        "EBOX",
        "RORE",
        "DBOX",
    ])

    df["MAX_WEIGHT_LOOP"] = loop_names[
        np.argmax(weights, axis=1)
    ]

    # -------------------------------------------------------------------------
    # output
    # -------------------------------------------------------------------------

    df.to_csv(
        DIR["source"]
        / "STATIC_3LOOP_OBSERVATION_SCORES.csv",
        index=False,
    )

    return df


# =============================================================================
# 7. AGGREGATION
# =============================================================================

SCORE_COLS = [
    "EBOX_POS",
    "EBOX_NEG",
    "RORE_ACT",
    "RORE_REP",
    "DBOX_ACT",
    "DBOX_REP",

    "EBOX_ENGAGEMENT",
    "RORE_ENGAGEMENT",
    "DBOX_ENGAGEMENT",

    "EBOX_BALANCE",
    "RORE_BALANCE",
    "DBOX_BALANCE",

    "EBOX_RELATIVE_WEIGHT",
    "RORE_RELATIVE_WEIGHT",
    "DBOX_RELATIVE_WEIGHT",

    "ARCHITECTURE_SPECIFICITY",
]


def aggregate_scores(obs):
    primary = obs[
        obs["priority_lineage"].isin(
            LINEAGE_ORDER
        )
    ].copy()

    # -------------------------------------------------------------------------
    # donor x lineage
    # -------------------------------------------------------------------------

    dl = (
        primary
        .groupby(
            [
                "donor_id",
                "priority_lineage",
            ],
            observed=True,
            sort=True,
        )[SCORE_COLS]
        .mean()
        .reset_index()
    )

    # IMPORTANT:
    # recompute weights after averaging engagement,
    # rather than assuming average weight == centroid weight.
    #
    # We preserve averaged weights too because they describe
    # mean observation-level composition.
    #
    dl.to_csv(
        DIR["source"]
        / "STATIC_3LOOP_DONOR_LINEAGE_SCORES.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # donor x tissue x lineage
    # -------------------------------------------------------------------------

    dtl = (
        primary
        .groupby(
            [
                "donor_id",
                "tissue",
                "priority_lineage",
            ],
            observed=True,
            sort=True,
        )[SCORE_COLS]
        .mean()
        .reset_index()
    )

    dtl.to_csv(
        DIR["source"]
        / "STATIC_3LOOP_DONOR_TISSUE_LINEAGE_SCORES.csv",
        index=False,
    )

    return primary, dl, dtl


# =============================================================================
# 8. LINEAGE SUMMARY
# =============================================================================

def lineage_summary(dl):
    records = []

    for lineage in LINEAGE_ORDER:
        part = dl[
            dl["priority_lineage"] == lineage
        ]

        donors = part["donor_id"].nunique()

        for score in SCORE_COLS:
            x = (
                part[score]
                .dropna()
                .to_numpy(float)
            )

            if len(x) == 0:
                continue

            lo, hi = bootstrap_mean_ci(
                x,
                n_boot=5000,
                seed=20260930,
            )

            records.append({
                "lineage": lineage,
                "lineage_label": lineage_label(lineage),
                "score": score,
                "n_donors": donors,
                "mean": np.mean(x),
                "median": np.median(x),
                "sd": np.std(x, ddof=1)
                if len(x) > 1 else np.nan,
                "q25": np.quantile(x, 0.25),
                "q75": np.quantile(x, 0.75),
                "bootstrap_mean_ci_low": lo,
                "bootstrap_mean_ci_high": hi,
            })

    out = pd.DataFrame(records)

    out.to_csv(
        DIR["lineage"]
        / "STATIC_3LOOP_LINEAGE_SUMMARY.csv",
        index=False,
    )

    return out


# =============================================================================
# 9. STATISTICS
# =============================================================================

STAT_SCORES = [
    "EBOX_ENGAGEMENT",
    "RORE_ENGAGEMENT",
    "DBOX_ENGAGEMENT",

    "EBOX_BALANCE",
    "RORE_BALANCE",
    "DBOX_BALANCE",

    "EBOX_RELATIVE_WEIGHT",
    "RORE_RELATIVE_WEIGHT",
    "DBOX_RELATIVE_WEIGHT",

    "ARCHITECTURE_SPECIFICITY",
]


def lineage_statistics(dl):
    omnibus = []
    pairwise = []

    for score in STAT_SCORES:
        groups = []

        for lineage in LINEAGE_ORDER:
            x = (
                dl.loc[
                    dl["priority_lineage"] == lineage,
                    score,
                ]
                .dropna()
                .to_numpy(float)
            )

            groups.append(x)

        # ---------------------------------------------------------------------
        # Kruskal-Wallis omnibus
        # ---------------------------------------------------------------------

        if all(len(x) > 0 for x in groups):
            h, p = kruskal(*groups)

            omnibus.append({
                "score": score,
                "test": "Kruskal-Wallis",
                "statistic": h,
                "p_value": p,
                "n_lineages": len(groups),
            })

        # ---------------------------------------------------------------------
        # pairwise Mann-Whitney
        # ---------------------------------------------------------------------

        for l1, l2 in combinations(
            LINEAGE_ORDER,
            2,
        ):
            x1 = (
                dl.loc[
                    dl["priority_lineage"] == l1,
                    score,
                ]
                .dropna()
                .to_numpy(float)
            )

            x2 = (
                dl.loc[
                    dl["priority_lineage"] == l2,
                    score,
                ]
                .dropna()
                .to_numpy(float)
            )

            if len(x1) == 0 or len(x2) == 0:
                continue

            u, p = mannwhitneyu(
                x1,
                x2,
                alternative="two-sided",
            )

            effect = rank_biserial_from_u(
                u,
                len(x1),
                len(x2),
            )

            # bootstrap mean difference
            rng = np.random.default_rng(
                20260930
                + LINEAGE_ORDER.index(l1) * 100
                + LINEAGE_ORDER.index(l2)
            )

            n_boot = 5000

            idx1 = rng.integers(
                0,
                len(x1),
                size=(n_boot, len(x1)),
            )

            idx2 = rng.integers(
                0,
                len(x2),
                size=(n_boot, len(x2)),
            )

            delta = (
                x1[idx1].mean(axis=1)
                - x2[idx2].mean(axis=1)
            )

            pairwise.append({
                "score": score,
                "lineage_1": l1,
                "lineage_1_label": lineage_label(l1),
                "lineage_2": l2,
                "lineage_2_label": lineage_label(l2),
                "n1": len(x1),
                "n2": len(x2),
                "mean_1": np.mean(x1),
                "mean_2": np.mean(x2),
                "mean_difference_1_minus_2":
                    np.mean(x1) - np.mean(x2),
                "mean_diff_boot_ci_low":
                    np.quantile(delta, 0.025),
                "mean_diff_boot_ci_high":
                    np.quantile(delta, 0.975),
                "median_1": np.median(x1),
                "median_2": np.median(x2),
                "mannwhitney_u": u,
                "p_value": p,
                "rank_biserial": effect,
            })

    omni = pd.DataFrame(omnibus)
    pair = pd.DataFrame(pairwise)

    if len(omni):
        omni["p_adj_BH"] = bh_adjust(
            omni["p_value"]
        )

    if len(pair):
        pair["p_adj_BH"] = (
            pair.groupby("score")["p_value"]
            .transform(
                lambda s: bh_adjust(
                    s.to_numpy()
                )
            )
        )

    omni.to_csv(
        DIR["lineage"]
        / "STATIC_3LOOP_LINEAGE_OMNIBUS.csv",
        index=False,
    )

    pair.to_csv(
        DIR["lineage"]
        / "STATIC_3LOOP_LINEAGE_PAIRWISE.csv",
        index=False,
    )

    combined = []

    for _, r in omni.iterrows():
        rec = {
            "analysis_type": "omnibus",
            **r.to_dict(),
        }
        combined.append(rec)

    for _, r in pair.iterrows():
        rec = {
            "analysis_type": "pairwise",
            **r.to_dict(),
        }
        combined.append(rec)

    pd.DataFrame(combined).to_csv(
        DIR["lineage"]
        / "STATIC_3LOOP_LINEAGE_STATISTICS.csv",
        index=False,
    )

    return omni, pair


# =============================================================================
# 10. TISSUE SUPPORT + STABILITY
# =============================================================================

def tissue_stability(dtl):
    support = (
        dtl
        .groupby(
            [
                "priority_lineage",
                "tissue",
            ],
            observed=True,
        )["donor_id"]
        .nunique()
        .rename("n_donors")
        .reset_index()
    )

    support["adequate_support"] = (
        support["n_donors"] >= 3
    )

    support.to_csv(
        DIR["audit"]
        / "STATIC_3LOOP_SUPPORT_AUDIT.csv",
        index=False,
    )

    adequate_keys = set(
        map(
            tuple,
            support.loc[
                support["adequate_support"],
                ["priority_lineage", "tissue"],
            ].to_numpy(),
        )
    )

    keep = dtl.apply(
        lambda r: (
            r["priority_lineage"],
            r["tissue"],
        ) in adequate_keys,
        axis=1,
    )

    adequate = dtl[keep].copy()

    centroid_cols = [
        "EBOX_ENGAGEMENT",
        "RORE_ENGAGEMENT",
        "DBOX_ENGAGEMENT",

        "EBOX_RELATIVE_WEIGHT",
        "RORE_RELATIVE_WEIGHT",
        "DBOX_RELATIVE_WEIGHT",

        "ARCHITECTURE_SPECIFICITY",
    ]

    tissue_arch = (
        adequate
        .groupby(
            [
                "priority_lineage",
                "tissue",
            ],
            observed=True,
        )[centroid_cols]
        .mean()
        .reset_index()
    )

    tissue_arch = tissue_arch.merge(
        support[
            [
                "priority_lineage",
                "tissue",
                "n_donors",
            ]
        ],
        on=[
            "priority_lineage",
            "tissue",
        ],
        how="left",
    )

    tissue_arch.to_csv(
        DIR["tissue"]
        / "STATIC_3LOOP_LINEAGE_TISSUE_ARCHITECTURE.csv",
        index=False,
    )

    records = []

    for lineage in LINEAGE_ORDER:
        part = tissue_arch[
            tissue_arch["priority_lineage"]
            == lineage
        ].copy()

        tissues = part["tissue"].tolist()

        for t1, t2 in combinations(
            tissues,
            2,
        ):
            a = part[
                part["tissue"] == t1
            ].iloc[0]

            b = part[
                part["tissue"] == t2
            ].iloc[0]

            wa = np.array([
                a["EBOX_RELATIVE_WEIGHT"],
                a["RORE_RELATIVE_WEIGHT"],
                a["DBOX_RELATIVE_WEIGHT"],
            ])

            wb = np.array([
                b["EBOX_RELATIVE_WEIGHT"],
                b["RORE_RELATIVE_WEIGHT"],
                b["DBOX_RELATIVE_WEIGHT"],
            ])

            ea = np.array([
                a["EBOX_ENGAGEMENT"],
                a["RORE_ENGAGEMENT"],
                a["DBOX_ENGAGEMENT"],
            ])

            eb = np.array([
                b["EBOX_ENGAGEMENT"],
                b["RORE_ENGAGEMENT"],
                b["DBOX_ENGAGEMENT"],
            ])

            distance = np.linalg.norm(
                wa - wb
            )

            if (
                np.std(ea) > 0
                and np.std(eb) > 0
            ):
                rho, rho_p = spearmanr(
                    ea,
                    eb,
                )
            else:
                rho = np.nan
                rho_p = np.nan

            records.append({
                "lineage": lineage,
                "lineage_label": lineage_label(lineage),
                "tissue_1": t1,
                "tissue_2": t2,
                "weight_euclidean_distance":
                    distance,
                "engagement_profile_spearman":
                    rho,
                "engagement_profile_spearman_p":
                    rho_p,
            })

    stability = pd.DataFrame(records)

    stability.to_csv(
        DIR["tissue"]
        / "STATIC_3LOOP_WITHIN_LINEAGE_TISSUE_STABILITY.csv",
        index=False,
    )

    return support, tissue_arch, stability


# =============================================================================
# 11. LOO GENE SENSITIVITY
# =============================================================================

def loo_sensitivity(meta, z, full_obs):
    """
    Recalculate engagement after leaving each constituent gene out.

    Evaluate:
      - observation-level correlation with full score
      - lineage centroid correlation
      - maximum absolute lineage-centroid shift
    """

    primary_ids = meta.index[
        meta["priority_lineage"].isin(
            LINEAGE_ORDER
        )
    ]

    records = []

    for loop in [
        "EBOX",
        "RORE",
        "DBOX",
    ]:
        pos_arm = LOOPS[loop]["positive"]
        neg_arm = LOOPS[loop]["negative"]

        genes_loop = (
            ARMS[pos_arm]
            + ARMS[neg_arm]
        )

        full_col = (
            f"{loop}_ENGAGEMENT"
        )

        full = full_obs[
            full_col
        ]

        for removed in genes_loop:
            pos_genes = [
                g for g in ARMS[pos_arm]
                if g != removed
            ]

            neg_genes = [
                g for g in ARMS[neg_arm]
                if g != removed
            ]

            if (
                len(pos_genes) == 0
                or len(neg_genes) == 0
            ):
                continue

            pos = z[pos_genes].mean(axis=1)
            neg = z[neg_genes].mean(axis=1)
            loo = (pos + neg) / 2

            rho, p = spearmanr(
                full.loc[loo.index],
                loo,
            )

            tmp = pd.DataFrame({
                "priority_lineage":
                    meta.loc[
                        loo.index,
                        "priority_lineage",
                    ],
                "full":
                    full.loc[loo.index],
                "loo":
                    loo,
            })

            tmp = tmp[
                tmp["priority_lineage"]
                .isin(LINEAGE_ORDER)
            ]

            cent = (
                tmp.groupby(
                    "priority_lineage"
                )[["full", "loo"]]
                .mean()
                .reindex(LINEAGE_ORDER)
            )

            centroid_rho, centroid_p = (
                spearmanr(
                    cent["full"],
                    cent["loo"],
                )
            )

            max_shift = (
                cent["loo"]
                - cent["full"]
            ).abs().max()

            full_order = tuple(
                cent["full"]
                .sort_values(
                    ascending=False
                )
                .index
            )

            loo_order = tuple(
                cent["loo"]
                .sort_values(
                    ascending=False
                )
                .index
            )

            records.append({
                "loop": loop,
                "removed_gene": removed,
                "n_remaining_genes":
                    len(pos_genes)
                    + len(neg_genes),
                "observation_spearman_rho":
                    rho,
                "observation_spearman_p":
                    p,
                "lineage_centroid_spearman_rho":
                    centroid_rho,
                "lineage_centroid_spearman_p":
                    centroid_p,
                "max_abs_lineage_centroid_shift":
                    max_shift,
                "lineage_order_unchanged":
                    full_order == loo_order,
                "full_lineage_order":
                    ";".join(full_order),
                "loo_lineage_order":
                    ";".join(loo_order),
            })

    out = pd.DataFrame(records)

    out.to_csv(
        DIR["lineage"]
        / "STATIC_3LOOP_LOO_GENE_SENSITIVITY.csv",
        index=False,
    )

    return out


# =============================================================================
# 12. FIGURES
# =============================================================================

def lineage_gene_matrix(priority, z):
    ids = priority.index

    tmp = z.loc[ids].copy()
    tmp["priority_lineage"] = (
        priority["priority_lineage"]
    )

    mat = (
        tmp.groupby(
            "priority_lineage",
            observed=True,
        )[ALL_CLOCK_GENES]
        .mean()
        .reindex(LINEAGE_ORDER)
    )

    return mat


def plot_gene_heatmap(priority, z):
    mat = lineage_gene_matrix(
        priority,
        z,
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 3.8)
    )

    vmax = np.nanpercentile(
        np.abs(mat.to_numpy()),
        97.5,
    )

    vmax = max(vmax, 0.5)

    im = ax.imshow(
        mat.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(len(ALL_CLOCK_GENES))
    )

    ax.set_xticklabels(
        ALL_CLOCK_GENES,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(LINEAGE_ORDER))
    )

    ax.set_yticklabels([
        lineage_label(x)
        for x in LINEAGE_ORDER
    ])

    ax.set_title(
        "Static clock-gene architecture organized by feedback-loop structure"
    )

    # arm boundaries and colored top blocks
    cursor = 0

    for label, arm, color in GENE_BLOCKS:
        n = len(ARMS[arm])

        if cursor > 0:
            ax.axvline(
                cursor - 0.5,
                color="black",
                lw=0.7,
            )

        ax.add_patch(
            Rectangle(
                (
                    cursor - 0.5,
                    -1.05,
                ),
                n,
                0.18,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                facecolor=color,
                edgecolor="none",
            )
        )

        ax.text(
            cursor + (n - 1) / 2,
            -1.22,
            label,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=7,
        )

        cursor += n

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Mean atlas-wide gene z-score"
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig_A_loop_ordered_clock17_heatmap",
    )

    mat.to_csv(
        DIR["source"]
        / "Fig_A_loop_ordered_clock17_heatmap_source.csv"
    )


def plot_score_heatmap(
    summary,
    scores,
    labels,
    title,
    stem,
    cmap="RdBu_r",
    center_zero=True,
):
    tab = (
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

    arr = tab.to_numpy()

    fig, ax = plt.subplots(
        figsize=(
            1.3 * len(scores) + 3.2,
            3.5,
        )
    )

    if center_zero:
        vmax = np.nanmax(
            np.abs(arr)
        )
        vmax = max(vmax, 0.2)

        im = ax.imshow(
            arr,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )

    else:
        im = ax.imshow(
            arr,
            cmap=cmap,
            aspect="auto",
        )

    ax.set_xticks(
        np.arange(len(scores))
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(LINEAGE_ORDER))
    )

    ax.set_yticklabels([
        lineage_label(x)
        for x in LINEAGE_ORDER
    ])

    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(
                    j,
                    i,
                    f"{arr[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.04,
        pad=0.03,
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"] / stem,
    )

    tab.to_csv(
        DIR["source"]
        / f"{stem}_source.csv"
    )


def plot_ternary(dl):
    fig, ax = plt.subplots(
        figsize=(6.8, 6.2)
    )

    h = math.sqrt(3) / 2

    # triangle
    ax.plot(
        [0, 1, 0.5, 0],
        [0, 0, h, 0],
        color="black",
        lw=1.0,
    )

    # guide lines
    for frac in [0.25, 0.5, 0.75]:
        # horizontal RORE guide
        y = frac * h
        x_left = frac * 0.5
        x_right = 1 - frac * 0.5

        ax.plot(
            [x_left, x_right],
            [y, y],
            color="#D8D8D8",
            lw=0.5,
            zorder=0,
        )

    # donor-level points
    for lineage in LINEAGE_ORDER:
        p = dl[
            dl["priority_lineage"] == lineage
        ]

        x, y = ternary_xy(
            p["EBOX_RELATIVE_WEIGHT"].to_numpy(),
            p["RORE_RELATIVE_WEIGHT"].to_numpy(),
            p["DBOX_RELATIVE_WEIGHT"].to_numpy(),
        )

        ax.scatter(
            x,
            y,
            s=20,
            alpha=0.35,
            color=LINEAGE_COLORS[lineage],
            edgecolors="none",
        )

        cent = p[
            [
                "EBOX_RELATIVE_WEIGHT",
                "RORE_RELATIVE_WEIGHT",
                "DBOX_RELATIVE_WEIGHT",
            ]
        ].mean()

        cx, cy = ternary_xy(
            cent["EBOX_RELATIVE_WEIGHT"],
            cent["RORE_RELATIVE_WEIGHT"],
            cent["DBOX_RELATIVE_WEIGHT"],
        )

        ax.scatter(
            [cx],
            [cy],
            s=120,
            color=LINEAGE_COLORS[lineage],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

        ax.text(
            cx,
            cy + 0.035,
            lineage_label(lineage),
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

    ax.text(
        -0.03,
        -0.035,
        "E-box / PER–CRY",
        ha="right",
        va="top",
        fontsize=9,
    )

    ax.text(
        1.03,
        -0.035,
        "D-box / PAR-bZIP",
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.text(
        0.5,
        h + 0.04,
        "RORE / REV-ERB–ROR",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xlim(
        -0.15,
        1.15,
    )

    ax.set_ylim(
        -0.10,
        h + 0.12,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        "Relative static three-loop architecture\n"
        "small points: donor × lineage; large points: lineage centroids",
        pad=20,
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig_E_ternary_relative_architecture",
    )


def plot_specificity(dl):
    fig, ax = plt.subplots(
        figsize=(7.5, 4.2)
    )

    rng = np.random.default_rng(
        20260930
    )

    for i, lineage in enumerate(
        LINEAGE_ORDER
    ):
        vals = (
            dl.loc[
                dl["priority_lineage"] == lineage,
                "ARCHITECTURE_SPECIFICITY",
            ]
            .dropna()
            .to_numpy(float)
        )

        jitter = rng.normal(
            0,
            0.055,
            len(vals),
        )

        ax.scatter(
            np.full(len(vals), i)
            + jitter,
            vals,
            s=25,
            alpha=0.55,
            color=LINEAGE_COLORS[lineage],
            edgecolors="none",
        )

        if len(vals):
            ax.plot(
                [i - 0.18, i + 0.18],
                [np.mean(vals)] * 2,
                color="black",
                lw=2,
            )

    ax.set_xticks(
        np.arange(len(LINEAGE_ORDER))
    )

    ax.set_xticklabels(
        [
            lineage_label(x)
            for x in LINEAGE_ORDER
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Architecture specificity\n"
        "1 − normalized entropy"
    )

    ax.set_title(
        "Static three-loop architecture specificity"
    )

    ax.axhline(
        0,
        color="#BBBBBB",
        lw=0.7,
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig_F_architecture_specificity",
    )


def plot_tissue_heatmap(tissue_arch):
    if tissue_arch.empty:
        return

    t = tissue_arch.copy()

    t["row_label"] = (
        t["priority_lineage"]
        .map(LINEAGE_DISPLAY)
        + " | "
        + t["tissue"].astype(str)
        + " (n="
        + t["n_donors"].astype(int).astype(str)
        + ")"
    )

    cols = [
        "EBOX_RELATIVE_WEIGHT",
        "RORE_RELATIVE_WEIGHT",
        "DBOX_RELATIVE_WEIGHT",
    ]

    arr = t[cols].to_numpy()

    fig_h = max(
        5,
        0.22 * len(t) + 2.2,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, fig_h)
    )

    im = ax.imshow(
        arr,
        cmap="viridis",
        vmin=0,
        vmax=max(
            0.6,
            np.nanmax(arr),
        ),
        aspect="auto",
    )

    ax.set_xticks([0, 1, 2])

    ax.set_xticklabels([
        "E-box",
        "RORE",
        "D-box",
    ])

    ax.set_yticks(
        np.arange(len(t))
    )

    ax.set_yticklabels(
        t["row_label"],
        fontsize=6.5,
    )

    ax.set_title(
        "Lineage × tissue relative three-loop architecture\n"
        "adequate support: ≥3 donors"
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
        label="Relative architecture weight",
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig_G_lineage_tissue_architecture_heatmap",
    )


def plot_tissue_stability(stability):
    if stability.empty:
        return

    fig, ax = plt.subplots(
        figsize=(7.5, 4.4)
    )

    rng = np.random.default_rng(
        20260930
    )

    for i, lineage in enumerate(
        LINEAGE_ORDER
    ):
        vals = (
            stability.loc[
                stability["lineage"] == lineage,
                "weight_euclidean_distance",
            ]
            .dropna()
            .to_numpy(float)
        )

        if len(vals) == 0:
            continue

        jitter = rng.normal(
            0,
            0.055,
            len(vals),
        )

        ax.scatter(
            np.full(len(vals), i)
            + jitter,
            vals,
            s=22,
            alpha=0.45,
            color=LINEAGE_COLORS[lineage],
            edgecolors="none",
        )

        ax.plot(
            [i - 0.18, i + 0.18],
            [np.median(vals)] * 2,
            color="black",
            lw=2,
        )

    ax.set_xticks(
        np.arange(len(LINEAGE_ORDER))
    )

    ax.set_xticklabels(
        [
            lineage_label(x)
            for x in LINEAGE_ORDER
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Cross-tissue architecture distance"
    )

    ax.set_title(
        "Within-lineage tissue remodeling of three-loop architecture"
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig_H_within_lineage_tissue_stability",
    )


def plot_pca(dl):
    feature_cols = [
        "EBOX_POS",
        "EBOX_NEG",
        "RORE_ACT",
        "RORE_REP",
        "DBOX_ACT",
        "DBOX_REP",
    ]

    x = dl[feature_cols].to_numpy(float)

    x = safe_zscore(x)

    pca = PCA(
        n_components=2,
        random_state=20260930,
    )

    pcs = pca.fit_transform(x)

    out = dl[
        [
            "donor_id",
            "priority_lineage",
        ]
    ].copy()

    out["PC1"] = pcs[:, 0]
    out["PC2"] = pcs[:, 1]

    out.to_csv(
        DIR["source"]
        / "Fig_I_three_loop_score_PCA_source.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 5.2)
    )

    for lineage in LINEAGE_ORDER:
        mask = (
            out["priority_lineage"]
            == lineage
        )

        ax.scatter(
            out.loc[mask, "PC1"],
            out.loc[mask, "PC2"],
            s=28,
            alpha=0.55,
            color=LINEAGE_COLORS[lineage],
            label=lineage_label(lineage),
            edgecolors="none",
        )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )

    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )

    ax.set_title(
        "Static three-loop score space"
    )

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig_I_three_loop_score_PCA",
    )


# =============================================================================
# 13. AUDIT FILES
# =============================================================================

def write_score_definitions():
    rows = []

    for arm, genes in ARMS.items():
        rows.append({
            "score": arm,
            "type": "arm",
            "definition": "mean atlas-wide gene z-score",
            "genes": ";".join(genes),
        })

    rows.extend([
        {
            "score": "EBOX_ENGAGEMENT",
            "type": "engagement",
            "definition": "(EBOX_POS + EBOX_NEG)/2",
            "genes": ";".join(
                ARMS["EBOX_POS"]
                + ARMS["EBOX_NEG"]
            ),
        },
        {
            "score": "RORE_ENGAGEMENT",
            "type": "engagement",
            "definition": "(RORE_ACT + RORE_REP)/2",
            "genes": ";".join(
                ARMS["RORE_ACT"]
                + ARMS["RORE_REP"]
            ),
        },
        {
            "score": "DBOX_ENGAGEMENT",
            "type": "engagement",
            "definition": "(DBOX_ACT + DBOX_REP)/2",
            "genes": ";".join(
                ARMS["DBOX_ACT"]
                + ARMS["DBOX_REP"]
            ),
        },
        {
            "score": "EBOX_BALANCE",
            "type": "balance",
            "definition": "EBOX_POS - EBOX_NEG",
            "genes": "",
        },
        {
            "score": "RORE_BALANCE",
            "type": "balance",
            "definition": "RORE_ACT - RORE_REP",
            "genes": "",
        },
        {
            "score": "DBOX_BALANCE",
            "type": "balance",
            "definition": "DBOX_ACT - DBOX_REP",
            "genes": "",
        },
    ])

    pd.DataFrame(rows).to_csv(
        DIR["audit"]
        / "STATIC_3LOOP_SCORE_DEFINITIONS.csv",
        index=False,
    )


# =============================================================================
# 14. REPORT
# =============================================================================

def build_report(
    summary,
    omnibus,
    pairwise,
    tissue_arch,
    stability,
    loo,
):
    def mean_score(lineage, score):
        x = summary[
            (summary["lineage"] == lineage)
            & (summary["score"] == score)
        ]

        if x.empty:
            return np.nan

        return float(x.iloc[0]["mean"])

    lines = []

    lines.append(
        "# STATIC THREE-LOOP CIRCADIAN ARCHITECTURE — SCIENTIFIC REPORT"
    )

    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This analysis describes static clock-associated transcriptional "
        "architecture only. It does not infer circadian phase, rhythmicity, "
        "amplitude, period, oscillation, or causal loop dominance."
    )

    lines.append("")
    lines.append("## Lineage-level relative architecture")
    lines.append("")

    for lineage in LINEAGE_ORDER:
        e = mean_score(
            lineage,
            "EBOX_RELATIVE_WEIGHT",
        )

        r = mean_score(
            lineage,
            "RORE_RELATIVE_WEIGHT",
        )

        d = mean_score(
            lineage,
            "DBOX_RELATIVE_WEIGHT",
        )

        spec = mean_score(
            lineage,
            "ARCHITECTURE_SPECIFICITY",
        )

        vals = {
            "E-box": e,
            "RORE": r,
            "D-box": d,
        }

        max_loop = max(
            vals,
            key=vals.get,
        )

        lines.append(
            f"- **{lineage_label(lineage)}**: "
            f"E-box={e:.3f}, RORE={r:.3f}, D-box={d:.3f}; "
            f"largest relative component={max_loop}; "
            f"architecture specificity={spec:.3f}."
        )

    lines.append("")
    lines.append(
        "The largest relative component is descriptive and must not be "
        "interpreted as a causally dominant circadian oscillator loop."
    )

    lines.append("")
    lines.append("## Statistical summary")
    lines.append("")

    for _, r in omnibus.iterrows():
        lines.append(
            f"- {r['score']}: Kruskal-Wallis "
            f"H={r['statistic']:.3f}, "
            f"p={r['p_value']:.3g}, "
            f"BH-adjusted p={r['p_adj_BH']:.3g}."
        )

    lines.append("")
    lines.append("## Leave-one-gene-out robustness")
    lines.append("")

    for loop in ["EBOX", "RORE", "DBOX"]:
        x = loo[
            loo["loop"] == loop
        ]

        if x.empty:
            continue

        lines.append(
            f"- {loop}: minimum observation-level Spearman rho after "
            f"single-gene removal = "
            f"{x['observation_spearman_rho'].min():.3f}; "
            f"{int(x['lineage_order_unchanged'].sum())}/{len(x)} "
            f"single-gene removals preserved the complete lineage ordering."
        )

    lines.append("")
    lines.append("## Tissue stability")
    lines.append("")

    if not stability.empty:
        for lineage in LINEAGE_ORDER:
            x = stability[
                stability["lineage"]
                == lineage
            ]

            if x.empty:
                continue

            lines.append(
                f"- {lineage_label(lineage)}: median pairwise "
                f"cross-tissue architecture distance = "
                f"{x['weight_euclidean_distance'].median():.3f} "
                f"across {len(x)} adequately supported tissue pairs."
            )

    lines.append("")
    lines.append("## Interpretation rules")
    lines.append("")
    lines.append(
        "- Allowed: 'relatively RORE-skewed static clock-associated architecture'."
    )
    lines.append(
        "- Not allowed: 'RORE drives the oscillator' or 'RORE has greater circadian amplitude'."
    )
    lines.append(
        "- All results require later temporal validation before making dynamic circadian claims."
    )

    report = "\n".join(lines)

    (
        OUT
        / "STATIC_3LOOP_SCIENTIFIC_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )


# =============================================================================
# 15. MAIN
# =============================================================================

def main():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    meta, priority, genes, expr = (
        load_data()
    )

    # -------------------------------------------------------------------------
    # audit
    # -------------------------------------------------------------------------

    audit = {
        "n_metadata_rows": len(meta),
        "n_expression_rows": expr.shape[0],
        "n_expression_genes": expr.shape[1],
        "n_priority_lineage_rows": len(priority),
        "expected_lineage_counts": EXPECTED_COUNTS,
        "observed_lineage_counts": (
            priority["priority_lineage"]
            .value_counts()
            .to_dict()
        ),
        "static_expression_only": True,
        "temporal_claims_allowed": False,
        "broad_class_used": False,
    }

    (
        DIR["audit"]
        / "STATIC_3LOOP_INPUT_AUDIT.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_score_definitions()

    # -------------------------------------------------------------------------
    # gene matrices
    # -------------------------------------------------------------------------

    raw, z = extract_clock_expression(
        meta,
        genes,
        expr,
    )

    # -------------------------------------------------------------------------
    # score
    # -------------------------------------------------------------------------

    obs = calculate_scores(
        meta,
        raw,
        z,
    )

    # -------------------------------------------------------------------------
    # aggregation
    # -------------------------------------------------------------------------

    primary, dl, dtl = aggregate_scores(
        obs
    )

    # -------------------------------------------------------------------------
    # summary
    # -------------------------------------------------------------------------

    summary = lineage_summary(
        dl
    )

    # -------------------------------------------------------------------------
    # stats
    # -------------------------------------------------------------------------

    omnibus, pairwise = (
        lineage_statistics(
            dl
        )
    )

    # -------------------------------------------------------------------------
    # tissue
    # -------------------------------------------------------------------------

    support, tissue_arch, stability = (
        tissue_stability(
            dtl
        )
    )

    # -------------------------------------------------------------------------
    # LOO
    # -------------------------------------------------------------------------

    loo = loo_sensitivity(
        meta,
        z,
        obs,
    )

    # -------------------------------------------------------------------------
    # figures
    # -------------------------------------------------------------------------

    print("Generating figures...")

    plot_gene_heatmap(
        priority,
        z,
    )

    plot_score_heatmap(
        summary,
        scores=[
            "EBOX_POS",
            "EBOX_NEG",
            "RORE_ACT",
            "RORE_REP",
            "DBOX_ACT",
            "DBOX_REP",
        ],
        labels=[
            "E-box +",
            "E-box −",
            "ROR",
            "REV-ERB",
            "PAR-bZIP",
            "NFIL3",
        ],
        title="Six-arm static clock architecture",
        stem="Fig_B_six_arm_heatmap",
    )

    plot_score_heatmap(
        summary,
        scores=[
            "EBOX_ENGAGEMENT",
            "RORE_ENGAGEMENT",
            "DBOX_ENGAGEMENT",
        ],
        labels=[
            "E-box",
            "RORE",
            "D-box",
        ],
        title="Static three-loop engagement",
        stem="Fig_C_three_loop_engagement",
    )

    plot_score_heatmap(
        summary,
        scores=[
            "EBOX_BALANCE",
            "RORE_BALANCE",
            "DBOX_BALANCE",
        ],
        labels=[
            "E-box balance",
            "RORE balance",
            "D-box balance",
        ],
        title="Activator–repressor balance",
        stem="Fig_D_loop_balance",
    )

    plot_ternary(
        dl
    )

    plot_specificity(
        dl
    )

    plot_tissue_heatmap(
        tissue_arch
    )

    plot_tissue_stability(
        stability
    )

    plot_pca(
        dl
    )

    # -------------------------------------------------------------------------
    # report
    # -------------------------------------------------------------------------

    build_report(
        summary,
        omnibus,
        pairwise,
        tissue_arch,
        stability,
        loo,
    )

    # -------------------------------------------------------------------------
    # completion flag
    # -------------------------------------------------------------------------

    completion = {
        "STATIC_3LOOP_COMPLETE": True,
        "n_observations": len(meta),
        "n_priority_lineage_observations": len(priority),
        "n_clock_genes": 17,
        "n_major_lineages": 5,
        "broad_class_used": False,
        "circadian_phase_inferred": False,
        "rhythmicity_inferred": False,
        "regulon_inference_used": False,
    }

    (
        OUT
        / "STATIC_3LOOP_COMPLETE.json"
    ).write_text(
        json.dumps(
            completion,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("STATIC 3-LOOP ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"OUTPUT:")
    print(OUT)
    print()
    print("Main outputs:")
    print("  Fig_A_loop_ordered_clock17_heatmap")
    print("  Fig_B_six_arm_heatmap")
    print("  Fig_C_three_loop_engagement")
    print("  Fig_D_loop_balance")
    print("  Fig_E_ternary_relative_architecture")
    print("  Fig_F_architecture_specificity")
    print("  Fig_G_lineage_tissue_architecture_heatmap")
    print("  Fig_H_within_lineage_tissue_stability")
    print("  Fig_I_three_loop_score_PCA")
    print()
    print("STATIC_3LOOP_COMPLETE = TRUE")
    print()


if __name__ == "__main__":
    main()