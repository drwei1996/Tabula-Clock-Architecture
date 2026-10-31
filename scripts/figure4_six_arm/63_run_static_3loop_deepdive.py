#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os

import json
import math
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import (
    spearmanr,
    kruskal,
    mannwhitneyu,
    f_oneway,
)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


# =============================================================================
# 0. PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

META_PATH = (
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

BASE62 = (
    ROOT
    / "62_STATIC_3LOOP_ARCHITECTURE_20260930"
)

OUT = (
    ROOT
    / "63_STATIC_3LOOP_DEEPDIVE_20260930"
)

DIR = {
    "audit": OUT / "00_AUDIT",
    "robust": OUT / "01_SCORE_ROBUSTNESS",
    "nfil3": OUT / "02_NFIL3_DBOX",
    "adjust": OUT / "03_TISSUE_DONOR_ADJUSTED",
    "paired": OUT / "04_WITHIN_DONOR",
    "plasticity": OUT / "05_TISSUE_PLASTICITY",
    "jackknife": OUT / "06_JACKKNIFE",
    "coupling": OUT / "07_LOOP_COUPLING",
    "bootstrap": OUT / "08_BOOTSTRAP_ARCHITECTURE",
    "perm": OUT / "09_PERMUTATION",
    "fig": OUT / "10_FIGURES",
    "source": OUT / "source_data",
    "report": OUT / "11_REPORT",
}

for p in DIR.values():
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# 2. FROZEN BIOLOGICAL DEFINITIONS
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

ARMS = {
    "EBOX_POS": ["ARNTL", "CLOCK", "NPAS2"],
    "EBOX_NEG": ["PER1", "PER2", "PER3", "CRY1", "CRY2"],
    "RORE_ACT": ["RORA", "RORB", "RORC"],
    "RORE_REP": ["NR1D1", "NR1D2"],
    "DBOX_ACT": ["DBP", "TEF", "HLF"],
    "DBOX_REP": ["NFIL3"],
}

LOOP_GENES = {
    "EBOX": (
        ARMS["EBOX_POS"]
        + ARMS["EBOX_NEG"]
    ),
    "RORE": (
        ARMS["RORE_ACT"]
        + ARMS["RORE_REP"]
    ),
    "DBOX": (
        ARMS["DBOX_ACT"]
        + ARMS["DBOX_REP"]
    ),
}

ALL_GENES = (
    LOOP_GENES["EBOX"]
    + LOOP_GENES["RORE"]
    + LOOP_GENES["DBOX"]
)

assert len(ALL_GENES) == 17
assert len(set(ALL_GENES)) == 17


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def savefig(fig, stem: Path):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            stem.with_suffix(f".{ext}"),
            bbox_inches="tight",
            dpi=600 if ext == "png" else None,
        )
    plt.close(fig)


def zscore_cols(x):
    x = np.asarray(x, dtype=float)

    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)

    sd[sd == 0] = 1.0
    sd[~np.isfinite(sd)] = 1.0

    z = (x - mu) / sd

    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite z-score.")

    return z


def softmax3(x):
    x = np.asarray(x, dtype=float)

    x = x - np.max(
        x,
        axis=1,
        keepdims=True,
    )

    e = np.exp(x)

    return e / e.sum(
        axis=1,
        keepdims=True,
    )


def entropy_specificity(weights):
    w = np.clip(
        np.asarray(weights, dtype=float),
        1e-12,
        1,
    )

    h = -np.sum(
        w * np.log(w),
        axis=1,
    )

    return 1 - h / np.log(3)


def bootstrap_ci(
    values,
    n_boot=5000,
    seed=20260930,
):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    ind = rng.integers(
        0,
        len(x),
        size=(n_boot, len(x)),
    )

    means = x[ind].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def rank_biserial(u, n1, n2):
    return 2 * u / (n1 * n2) - 1


def bh_adjust(p):
    p = np.asarray(p, dtype=float)

    result = np.full(
        len(p),
        np.nan,
    )

    valid = np.isfinite(p)

    pv = p[valid]

    if len(pv) == 0:
        return result

    order = np.argsort(pv)
    ranks = np.arange(
        1,
        len(pv) + 1,
    )

    adj = (
        pv[order]
        * len(pv)
        / ranks
    )

    adj = np.minimum.accumulate(
        adj[::-1]
    )[::-1]

    adj = np.minimum(
        adj,
        1,
    )

    tmp = np.empty(
        len(pv)
    )

    tmp[order] = adj

    result[valid] = tmp

    return result


def lineage_label(x):
    return LINEAGE_DISPLAY.get(
        x,
        str(x),
    )


# =============================================================================
# 4. LOAD ORIGINAL FROZEN DATA
# =============================================================================

def load_data():
    meta = pd.read_csv(
        META_PATH,
        sep="\t",
        dtype={"group_id": str},
    )

    if "priority_lineage_corrected" in meta.columns:
        meta = meta.rename(
            columns={
                "priority_lineage_corrected":
                    "priority_lineage"
            }
        )

    meta["group_id"] = (
        meta["group_id"]
        .astype(str)
    )

    meta = meta.set_index(
        "group_id",
        drop=False,
    )

    if len(meta) != 2030:
        raise RuntimeError(
            f"Expected 2030 rows, got {len(meta)}."
        )

    primary = meta[
        meta["priority_lineage"]
        .isin(LINEAGE_ORDER)
    ].copy()

    counts = (
        primary["priority_lineage"]
        .value_counts()
        .to_dict()
    )

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Lineage count mismatch: {counts}"
        )

    data = np.load(
        NPZ_PATH,
        allow_pickle=True,
    )

    ids = (
        data["group_id"]
        .astype(str)
    )

    genes = (
        data["genes"]
        .astype(str)
    )

    expr = np.asarray(
        data["expression"],
        dtype=float,
    )

    if expr.shape != (2030, 59410):
        raise RuntimeError(
            f"Unexpected expression shape {expr.shape}"
        )

    if set(ids) != set(meta.index):
        raise RuntimeError(
            "Expression/metadata IDs differ."
        )

    order = (
        pd.Index(ids)
        .get_indexer(meta.index)
    )

    if (order < 0).any():
        raise RuntimeError(
            "ID alignment failure."
        )

    expr = expr[order]

    gene_index = {
        g: i
        for i, g in enumerate(genes)
    }

    missing = [
        g
        for g in ALL_GENES
        if g not in gene_index
    ]

    if missing:
        raise RuntimeError(
            f"Missing genes: {missing}"
        )

    idx = [
        gene_index[g]
        for g in ALL_GENES
    ]

    raw = pd.DataFrame(
        expr[:, idx],
        index=meta.index,
        columns=ALL_GENES,
    )

    z_global = pd.DataFrame(
        zscore_cols(
            raw.to_numpy()
        ),
        index=meta.index,
        columns=ALL_GENES,
    )

    return (
        meta,
        primary,
        raw,
        z_global,
    )


# =============================================================================
# 5. SCORE DEFINITIONS
# =============================================================================

def calculate_primary_scores(
    meta,
    z,
):
    out = meta[
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

    for arm, genes in ARMS.items():
        out[arm] = (
            z[genes]
            .mean(axis=1)
        )

    # arm-balanced primary
    out["EBOX_ARM_BALANCED"] = (
        out["EBOX_POS"]
        + out["EBOX_NEG"]
    ) / 2

    out["RORE_ARM_BALANCED"] = (
        out["RORE_ACT"]
        + out["RORE_REP"]
    ) / 2

    out["DBOX_ARM_BALANCED"] = (
        out["DBOX_ACT"]
        + out["DBOX_REP"]
    ) / 2

    # gene-balanced sensitivity
    out["EBOX_GENE_BALANCED"] = (
        z[LOOP_GENES["EBOX"]]
        .mean(axis=1)
    )

    out["RORE_GENE_BALANCED"] = (
        z[LOOP_GENES["RORE"]]
        .mean(axis=1)
    )

    out["DBOX_GENE_BALANCED"] = (
        z[LOOP_GENES["DBOX"]]
        .mean(axis=1)
    )

    # balance
    out["EBOX_BALANCE"] = (
        out["EBOX_POS"]
        - out["EBOX_NEG"]
    )

    out["RORE_BALANCE"] = (
        out["RORE_ACT"]
        - out["RORE_REP"]
    )

    out["DBOX_BALANCE"] = (
        out["DBOX_ACT"]
        - out["DBOX_REP"]
    )

    # D-box decomposition
    out["DBOX_ACTIVATOR_ONLY"] = (
        out["DBOX_ACT"]
    )

    out["DBOX_NFIL3_ONLY"] = (
        out["DBOX_REP"]
    )

    out["DBOX_PRIMARY_MINUS_NFIL3"] = (
        out["DBOX_ACT"]
    )

    return out


# =============================================================================
# 6. RECOMPUTE RELATIVE WEIGHTS FOR ANY THREE SCORES
# =============================================================================

def add_relative_weights(
    df,
    cols,
    prefix,
):
    arr = df[cols].to_numpy()

    arr_z = zscore_cols(arr)

    weights = softmax3(
        arr_z
    )

    names = [
        "EBOX",
        "RORE",
        "DBOX",
    ]

    for i, name in enumerate(names):
        df[
            f"{prefix}_{name}_WEIGHT"
        ] = weights[:, i]

    df[
        f"{prefix}_SPECIFICITY"
    ] = entropy_specificity(
        weights
    )

    df[
        f"{prefix}_MAX_LOOP"
    ] = np.array(names)[
        np.argmax(weights, axis=1)
    ]

    return df


# =============================================================================
# 7. ARM-BALANCED VS GENE-BALANCED ROBUSTNESS
# =============================================================================

def score_definition_robustness(
    primary,
):
    p = primary.copy()

    p = add_relative_weights(
        p,
        [
            "EBOX_ARM_BALANCED",
            "RORE_ARM_BALANCED",
            "DBOX_ARM_BALANCED",
        ],
        "ARM",
    )

    p = add_relative_weights(
        p,
        [
            "EBOX_GENE_BALANCED",
            "RORE_GENE_BALANCED",
            "DBOX_GENE_BALANCED",
        ],
        "GENE",
    )

    records = []

    for loop in [
        "EBOX",
        "RORE",
        "DBOX",
    ]:
        x = p[
            f"{loop}_ARM_BALANCED"
        ]

        y = p[
            f"{loop}_GENE_BALANCED"
        ]

        rho, pval = spearmanr(
            x,
            y,
        )

        records.append({
            "loop": loop,
            "level": "observation",
            "spearman_rho": rho,
            "p_value": pval,
        })

    # donor × lineage aggregation
    score_cols = [
        c
        for c in p.columns
        if (
            c.endswith("_BALANCED")
            or "_WEIGHT" in c
            or c.endswith("_SPECIFICITY")
        )
    ]

    dl = (
        p.groupby(
            [
                "donor_id",
                "priority_lineage",
            ],
            observed=True,
        )[score_cols]
        .mean()
        .reset_index()
    )

    cent = (
        dl.groupby(
            "priority_lineage",
            observed=True,
        )[score_cols]
        .mean()
        .reindex(LINEAGE_ORDER)
    )

    for loop in [
        "EBOX",
        "RORE",
        "DBOX",
    ]:
        x = cent[
            f"{loop}_ARM_BALANCED"
        ]

        y = cent[
            f"{loop}_GENE_BALANCED"
        ]

        rho, pval = spearmanr(
            x,
            y,
        )

        records.append({
            "loop": loop,
            "level": "lineage_centroid",
            "spearman_rho": rho,
            "p_value": pval,
        })

    comparison = pd.DataFrame(
        records
    )

    comparison.to_csv(
        DIR["robust"]
        / "ARM_VS_GENE_BALANCED_CORRELATION.csv",
        index=False,
    )

    # centroid table
    centroid_rows = []

    for lineage in LINEAGE_ORDER:
        row = {
            "lineage": lineage,
            "lineage_label":
                lineage_label(lineage),
        }

        for loop in [
            "EBOX",
            "RORE",
            "DBOX",
        ]:
            row[
                f"{loop}_arm"
            ] = cent.loc[
                lineage,
                f"{loop}_ARM_BALANCED",
            ]

            row[
                f"{loop}_gene"
            ] = cent.loc[
                lineage,
                f"{loop}_GENE_BALANCED",
            ]

            row[
                f"{loop}_delta_gene_minus_arm"
            ] = (
                row[f"{loop}_gene"]
                - row[f"{loop}_arm"]
            )

            row[
                f"{loop}_arm_weight"
            ] = cent.loc[
                lineage,
                f"ARM_{loop}_WEIGHT",
            ]

            row[
                f"{loop}_gene_weight"
            ] = cent.loc[
                lineage,
                f"GENE_{loop}_WEIGHT",
            ]

        centroid_rows.append(row)

    centroid = pd.DataFrame(
        centroid_rows
    )

    centroid.to_csv(
        DIR["robust"]
        / "ARM_VS_GENE_BALANCED_LINEAGE_CENTROIDS.csv",
        index=False,
    )

    return p, dl, centroid


# =============================================================================
# 8. D-BOX / NFIL3 DEEP DIVE
# =============================================================================

def dbox_nfil3_analysis(
    primary,
):
    scores = [
        "DBOX_ARM_BALANCED",
        "DBOX_GENE_BALANCED",
        "DBOX_ACT",
        "DBOX_REP",
    ]

    dl = (
        primary.groupby(
            [
                "donor_id",
                "priority_lineage",
            ],
            observed=True,
        )[scores]
        .mean()
        .reset_index()
    )

    records = []

    for lineage in LINEAGE_ORDER:
        part = dl[
            dl["priority_lineage"]
            == lineage
        ]

        for xcol, ycol in [
            (
                "DBOX_ARM_BALANCED",
                "DBOX_REP",
            ),
            (
                "DBOX_ARM_BALANCED",
                "DBOX_ACT",
            ),
            (
                "DBOX_GENE_BALANCED",
                "DBOX_REP",
            ),
            (
                "DBOX_GENE_BALANCED",
                "DBOX_ACT",
            ),
        ]:
            rho, p = spearmanr(
                part[xcol],
                part[ycol],
            )

            records.append({
                "lineage": lineage,
                "x": xcol,
                "y": ycol,
                "rho": rho,
                "p_value": p,
                "n_donors":
                    len(part),
            })

    corr = pd.DataFrame(
        records
    )

    corr.to_csv(
        DIR["nfil3"]
        / "DBOX_NFIL3_CORRELATIONS.csv",
        index=False,
    )

    # lineage centroids
    cent = (
        dl.groupby(
            "priority_lineage",
            observed=True,
        )[scores]
        .mean()
        .reindex(LINEAGE_ORDER)
        .reset_index()
    )

    cent.to_csv(
        DIR["nfil3"]
        / "DBOX_VARIANT_LINEAGE_CENTROIDS.csv",
        index=False,
    )

    # pairwise tests for NFIL3 and activator arm separately
    pairwise = []

    for score in [
        "DBOX_ACT",
        "DBOX_REP",
        "DBOX_GENE_BALANCED",
        "DBOX_ARM_BALANCED",
    ]:
        for a, b in combinations(
            LINEAGE_ORDER,
            2,
        ):
            x = (
                dl.loc[
                    dl["priority_lineage"] == a,
                    score,
                ]
                .dropna()
                .to_numpy()
            )

            y = (
                dl.loc[
                    dl["priority_lineage"] == b,
                    score,
                ]
                .dropna()
                .to_numpy()
            )

            u, p = mannwhitneyu(
                x,
                y,
                alternative="two-sided",
            )

            pairwise.append({
                "score": score,
                "lineage_1": a,
                "lineage_2": b,
                "mean_1": np.mean(x),
                "mean_2": np.mean(y),
                "delta": (
                    np.mean(x)
                    - np.mean(y)
                ),
                "rank_biserial":
                    rank_biserial(
                        u,
                        len(x),
                        len(y),
                    ),
                "p_value": p,
            })

    pairwise = pd.DataFrame(
        pairwise
    )

    pairwise["p_adj_BH"] = (
        pairwise.groupby(
            "score"
        )["p_value"]
        .transform(
            lambda x:
                bh_adjust(
                    x.to_numpy()
                )
        )
    )

    pairwise.to_csv(
        DIR["nfil3"]
        / "DBOX_VARIANT_PAIRWISE_STATISTICS.csv",
        index=False,
    )

    return dl, cent


# =============================================================================
# 9. WITHIN-TISSUE STANDARDIZATION SENSITIVITY
# =============================================================================

def within_tissue_gene_zscores(
    meta,
    raw,
):
    z = pd.DataFrame(
        index=raw.index,
        columns=raw.columns,
        dtype=float,
    )

    for tissue, ids in (
        meta.groupby(
            "tissue",
            observed=True,
        ).groups.items()
    ):
        ids = list(ids)

        arr = (
            raw.loc[ids]
            .to_numpy()
        )

        z.loc[ids] = (
            zscore_cols(arr)
        )

    return z.astype(float)


def tissue_adjusted_sensitivity(
    meta,
    primary_meta,
    raw,
):
    z_tissue = (
        within_tissue_gene_zscores(
            meta,
            raw,
        )
    )

    scored = (
        calculate_primary_scores(
            meta,
            z_tissue,
        )
    )

    scored = scored[
        scored["priority_lineage"]
        .isin(LINEAGE_ORDER)
    ].copy()

    dl = (
        scored.groupby(
            [
                "donor_id",
                "priority_lineage",
            ],
            observed=True,
        )[
            [
                "EBOX_ARM_BALANCED",
                "RORE_ARM_BALANCED",
                "DBOX_ARM_BALANCED",
                "EBOX_BALANCE",
                "RORE_BALANCE",
                "DBOX_BALANCE",
            ]
        ]
        .mean()
        .reset_index()
    )

    cent = (
        dl.groupby(
            "priority_lineage",
            observed=True,
        )
        .mean(
            numeric_only=True
        )
        .reindex(LINEAGE_ORDER)
    )

    cent.to_csv(
        DIR["adjust"]
        / "WITHIN_TISSUE_STANDARDIZED_LINEAGE_CENTROIDS.csv"
    )

    return scored, dl, cent


# =============================================================================
# 10. WITHIN-DONOR PAIRED LINEAGE CONTRASTS
# =============================================================================

def within_donor_paired_analysis(
    donor_lineage,
):
    score_cols = [
        "EBOX_ARM_BALANCED",
        "RORE_ARM_BALANCED",
        "DBOX_ARM_BALANCED",
        "EBOX_BALANCE",
        "RORE_BALANCE",
        "DBOX_BALANCE",
    ]

    records = []

    for score in score_cols:
        for a, b in combinations(
            LINEAGE_ORDER,
            2,
        ):
            aa = (
                donor_lineage[
                    donor_lineage[
                        "priority_lineage"
                    ] == a
                ][
                    [
                        "donor_id",
                        score,
                    ]
                ]
                .rename(
                    columns={
                        score: "score_a"
                    }
                )
            )

            bb = (
                donor_lineage[
                    donor_lineage[
                        "priority_lineage"
                    ] == b
                ][
                    [
                        "donor_id",
                        score,
                    ]
                ]
                .rename(
                    columns={
                        score: "score_b"
                    }
                )
            )

            paired = aa.merge(
                bb,
                on="donor_id",
                how="inner",
            )

            if len(paired) < 3:
                continue

            delta = (
                paired["score_a"]
                - paired["score_b"]
            )

            lo, hi = bootstrap_ci(
                delta,
                n_boot=5000,
                seed=20260930,
            )

            records.append({
                "score": score,
                "lineage_a": a,
                "lineage_b": b,
                "n_paired_donors":
                    len(paired),
                "mean_paired_delta":
                    delta.mean(),
                "median_paired_delta":
                    delta.median(),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "fraction_delta_positive":
                    (delta > 0).mean(),
            })

    out = pd.DataFrame(
        records
    )

    out.to_csv(
        DIR["paired"]
        / "WITHIN_DONOR_LINEAGE_CONTRASTS.csv",
        index=False,
    )

    return out


# =============================================================================
# 11. MIXED MODEL / TISSUE-ADJUSTED LINEAGE EFFECT
# =============================================================================

def mixed_models(
    donor_tissue_lineage,
):
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame()

    scores = [
        "EBOX_ARM_BALANCED",
        "RORE_ARM_BALANCED",
        "DBOX_ARM_BALANCED",
        "EBOX_BALANCE",
        "RORE_BALANCE",
        "DBOX_BALANCE",
    ]

    results = []

    for score in scores:
        data = donor_tissue_lineage[
            [
                score,
                "priority_lineage",
                "tissue",
                "donor_id",
            ]
        ].dropna()

        try:
            model = smf.mixedlm(
                f"{score} ~ C(priority_lineage) + C(tissue)",
                data=data,
                groups=data["donor_id"],
            )

            fit = model.fit(
                reml=False,
                method="lbfgs",
                maxiter=500,
                disp=False,
            )

            for term, coef in fit.params.items():
                results.append({
                    "score": score,
                    "term": term,
                    "coef": coef,
                    "se":
                        fit.bse.get(
                            term,
                            np.nan,
                        ),
                    "p_value":
                        fit.pvalues.get(
                            term,
                            np.nan,
                        ),
                    "converged":
                        bool(
                            getattr(
                                fit,
                                "converged",
                                True,
                            )
                        ),
                    "model":
                        "MixedLM lineage+tissue; donor random intercept",
                })

        except Exception as e:
            results.append({
                "score": score,
                "term": "__MODEL_FAILED__",
                "coef": np.nan,
                "se": np.nan,
                "p_value": np.nan,
                "converged": False,
                "model": str(e),
            })

    out = pd.DataFrame(
        results
    )

    out.to_csv(
        DIR["adjust"]
        / "MIXED_MODEL_LINEAGE_TISSUE_DONOR.csv",
        index=False,
    )

    return out


# =============================================================================
# 12. LINEAGE VS TISSUE EXPLAINED VARIANCE
# =============================================================================

def model_r2(
    df,
    score,
    predictors,
):
    dat = df[
        [score] + predictors
    ].dropna()

    X = pd.get_dummies(
        dat[predictors],
        drop_first=True,
    )

    y = dat[
        score
    ].to_numpy()

    model = LinearRegression()

    model.fit(
        X,
        y,
    )

    return model.score(
        X,
        y,
    )


def variance_partition(
    donor_tissue_lineage,
):
    records = []

    for score in [
        "EBOX_ARM_BALANCED",
        "RORE_ARM_BALANCED",
        "DBOX_ARM_BALANCED",
        "EBOX_BALANCE",
        "RORE_BALANCE",
        "DBOX_BALANCE",
    ]:
        r_lineage = model_r2(
            donor_tissue_lineage,
            score,
            ["priority_lineage"],
        )

        r_tissue = model_r2(
            donor_tissue_lineage,
            score,
            ["tissue"],
        )

        r_both = model_r2(
            donor_tissue_lineage,
            score,
            [
                "priority_lineage",
                "tissue",
            ],
        )

        r_donor = model_r2(
            donor_tissue_lineage,
            score,
            ["donor_id"],
        )

        r_all = model_r2(
            donor_tissue_lineage,
            score,
            [
                "priority_lineage",
                "tissue",
                "donor_id",
            ],
        )

        records.append({
            "score": score,
            "R2_lineage_only":
                r_lineage,
            "R2_tissue_only":
                r_tissue,
            "R2_donor_only":
                r_donor,
            "R2_lineage_plus_tissue":
                r_both,
            "R2_all":
                r_all,
            "increment_lineage_given_tissue":
                r_both - r_tissue,
            "increment_tissue_given_lineage":
                r_both - r_lineage,
        })

    out = pd.DataFrame(
        records
    )

    out.to_csv(
        DIR["adjust"]
        / "LINEAGE_TISSUE_VARIANCE_PARTITION.csv",
        index=False,
    )

    return out


# =============================================================================
# 13. LEAVE-ONE-TISSUE-OUT + LEAVE-ONE-DONOR-OUT
# =============================================================================

def jackknife_robustness(
    dtl,
    dl,
):
    scores = [
        "EBOX_ARM_BALANCED",
        "RORE_ARM_BALANCED",
        "DBOX_ARM_BALANCED",
    ]

    full_cent = (
        dl.groupby(
            "priority_lineage"
        )[scores]
        .mean()
        .reindex(LINEAGE_ORDER)
    )

    tissue_records = []

    tissues = sorted(
        dtl["tissue"]
        .dropna()
        .unique()
    )

    for removed in tissues:
        subset = dtl[
            dtl["tissue"] != removed
        ]

        agg = (
            subset.groupby(
                [
                    "donor_id",
                    "priority_lineage",
                ]
            )[scores]
            .mean()
            .reset_index()
        )

        cent = (
            agg.groupby(
                "priority_lineage"
            )[scores]
            .mean()
            .reindex(LINEAGE_ORDER)
        )

        for lineage in LINEAGE_ORDER:
            for score in scores:
                tissue_records.append({
                    "removed_tissue":
                        removed,
                    "lineage":
                        lineage,
                    "score":
                        score,
                    "full_centroid":
                        full_cent.loc[
                            lineage,
                            score,
                        ],
                    "jackknife_centroid":
                        cent.loc[
                            lineage,
                            score,
                        ],
                    "delta":
                        (
                            cent.loc[
                                lineage,
                                score,
                            ]
                            - full_cent.loc[
                                lineage,
                                score,
                            ]
                        ),
                })

    tissue_out = pd.DataFrame(
        tissue_records
    )

    tissue_out.to_csv(
        DIR["jackknife"]
        / "LEAVE_ONE_TISSUE_OUT.csv",
        index=False,
    )

    # donor jackknife
    donor_records = []

    donors = sorted(
        dl["donor_id"]
        .unique()
    )

    for removed in donors:
        subset = dl[
            dl["donor_id"] != removed
        ]

        cent = (
            subset.groupby(
                "priority_lineage"
            )[scores]
            .mean()
            .reindex(LINEAGE_ORDER)
        )

        for lineage in LINEAGE_ORDER:
            for score in scores:
                if lineage not in cent.index:
                    continue

                donor_records.append({
                    "removed_donor":
                        removed,
                    "lineage":
                        lineage,
                    "score":
                        score,
                    "full_centroid":
                        full_cent.loc[
                            lineage,
                            score,
                        ],
                    "jackknife_centroid":
                        cent.loc[
                            lineage,
                            score,
                        ],
                    "delta":
                        (
                            cent.loc[
                                lineage,
                                score,
                            ]
                            - full_cent.loc[
                                lineage,
                                score,
                            ]
                        ),
                })

    donor_out = pd.DataFrame(
        donor_records
    )

    donor_out.to_csv(
        DIR["jackknife"]
        / "LEAVE_ONE_DONOR_OUT.csv",
        index=False,
    )

    return (
        tissue_out,
        donor_out,
    )


# =============================================================================
# 14. LOOP COUPLING / CORRELATION STRUCTURE
# =============================================================================

def coupling_analysis(
    donor_lineage,
):
    six = [
        "EBOX_POS",
        "EBOX_NEG",
        "RORE_ACT",
        "RORE_REP",
        "DBOX_ACT",
        "DBOX_REP",
    ]

    matrices = {}
    records = []

    for lineage in LINEAGE_ORDER:
        p = donor_lineage[
            donor_lineage[
                "priority_lineage"
            ] == lineage
        ]

        corr = (
            p[six]
            .corr(
                method="spearman"
            )
        )

        matrices[lineage] = corr

        corr.to_csv(
            DIR["coupling"]
            / f"COUPLING_{lineage}.csv"
        )

        for a, b in combinations(
            six,
            2,
        ):
            records.append({
                "lineage":
                    lineage,
                "arm_1":
                    a,
                "arm_2":
                    b,
                "spearman_rho":
                    corr.loc[a, b],
            })

    pd.DataFrame(
        records
    ).to_csv(
        DIR["coupling"]
        / "ALL_LINEAGE_ARM_CORRELATIONS.csv",
        index=False,
    )

    # pairwise matrix distances
    distances = []

    for a, b in combinations(
        LINEAGE_ORDER,
        2,
    ):
        ma = (
            matrices[a]
            .to_numpy()
        )

        mb = (
            matrices[b]
            .to_numpy()
        )

        frob = np.linalg.norm(
            ma - mb,
            ord="fro",
        )

        distances.append({
            "lineage_1":
                a,
            "lineage_2":
                b,
            "frobenius_distance":
                frob,
        })

    pd.DataFrame(
        distances
    ).to_csv(
        DIR["coupling"]
        / "LINEAGE_COUPLING_MATRIX_DISTANCES.csv",
        index=False,
    )

    return matrices


# =============================================================================
# 15. BOOTSTRAP MAX-LOOP PROBABILITY
# =============================================================================

def bootstrap_loop_probabilities(
    donor_lineage,
    n_boot=10000,
):
    records = []

    rng = np.random.default_rng(
        20260930
    )

    for lineage in LINEAGE_ORDER:
        p = donor_lineage[
            donor_lineage[
                "priority_lineage"
            ] == lineage
        ].copy()

        cols = [
            "EBOX_ARM_BALANCED",
            "RORE_ARM_BALANCED",
            "DBOX_ARM_BALANCED",
        ]

        arr = p[
            cols
        ].to_numpy()

        n = len(arr)

        if n < 2:
            continue

        max_counts = np.zeros(
            3,
            dtype=int,
        )

        means = np.zeros(
            (
                n_boot,
                3,
            )
        )

        for b in range(
            n_boot
        ):
            idx = rng.integers(
                0,
                n,
                n,
            )

            m = (
                arr[idx]
                .mean(axis=0)
            )

            means[b] = m

            # z-score across the 3 loop means
            sd = np.std(m)

            if sd == 0:
                z = np.zeros(3)
            else:
                z = (
                    m - np.mean(m)
                ) / sd

            w = softmax3(
                z.reshape(
                    1,
                    3,
                )
            )[0]

            max_counts[
                np.argmax(w)
            ] += 1

        for i, loop in enumerate(
            [
                "EBOX",
                "RORE",
                "DBOX",
            ]
        ):
            records.append({
                "lineage":
                    lineage,
                "loop":
                    loop,
                "probability_max_component":
                    (
                        max_counts[i]
                        / n_boot
                    ),
                "bootstrap_mean_engagement":
                    means[:, i].mean(),
                "bootstrap_ci_low":
                    np.quantile(
                        means[:, i],
                        0.025,
                    ),
                "bootstrap_ci_high":
                    np.quantile(
                        means[:, i],
                        0.975,
                    ),
            })

    out = pd.DataFrame(
        records
    )

    out.to_csv(
        DIR["bootstrap"]
        / "BOOTSTRAP_MAX_LOOP_PROBABILITIES.csv",
        index=False,
    )

    return out


# =============================================================================
# 16. PERMUTATION TEST PRESERVING TISSUE COMPOSITION
# =============================================================================

def lineage_f_stat(
    df,
    score,
):
    groups = []

    for lineage in LINEAGE_ORDER:
        vals = (
            df.loc[
                df[
                    "priority_lineage"
                ] == lineage,
                score,
            ]
            .dropna()
            .to_numpy()
        )

        if len(vals):
            groups.append(
                vals
            )

    if len(groups) < 2:
        return np.nan

    return f_oneway(
        *groups
    ).statistic


def tissue_stratified_permutation(
    dtl,
    n_perm=2000,
):
    scores = [
        "EBOX_ARM_BALANCED",
        "RORE_ARM_BALANCED",
        "DBOX_ARM_BALANCED",
        "EBOX_BALANCE",
        "RORE_BALANCE",
        "DBOX_BALANCE",
    ]

    rng = np.random.default_rng(
        20260930
    )

    results = []

    for score in scores:
        observed = lineage_f_stat(
            dtl,
            score,
        )

        null = []

        for _ in range(
            n_perm
        ):
            perm = dtl.copy()

            shuffled = []

            for tissue, part in (
                perm.groupby(
                    "tissue",
                    sort=False,
                )
            ):
                labels = (
                    part[
                        "priority_lineage"
                    ]
                    .to_numpy()
                    .copy()
                )

                rng.shuffle(
                    labels
                )

                shuffled.append(
                    pd.Series(
                        labels,
                        index=part.index,
                    )
                )

            new_labels = pd.concat(
                shuffled
            ).sort_index()

            perm[
                "priority_lineage"
            ] = new_labels

            stat = lineage_f_stat(
                perm,
                score,
            )

            null.append(
                stat
            )

        null = np.asarray(
            null
        )

        empirical_p = (
            1
            + np.sum(
                null >= observed
            )
        ) / (
            1
            + len(null)
        )

        results.append({
            "score":
                score,
            "observed_F":
                observed,
            "permutation_p":
                empirical_p,
            "null_mean":
                np.nanmean(null),
            "null_q95":
                np.nanquantile(
                    null,
                    0.95,
                ),
            "n_permutations":
                n_perm,
        })

    out = pd.DataFrame(
        results
    )

    out.to_csv(
        DIR["perm"]
        / "TISSUE_STRATIFIED_LINEAGE_PERMUTATION.csv",
        index=False,
    )

    return out


# =============================================================================
# 17. PLOTS
# =============================================================================

def plot_arm_vs_gene(
    centroid,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            12,
            4,
        ),
    )

    for ax, loop in zip(
        axes,
        [
            "EBOX",
            "RORE",
            "DBOX",
        ],
    ):
        for lineage in LINEAGE_ORDER:
            row = centroid[
                centroid["lineage"]
                == lineage
            ].iloc[0]

            ax.scatter(
                row[
                    f"{loop}_arm"
                ],
                row[
                    f"{loop}_gene"
                ],
                s=80,
                color=
                    LINEAGE_COLORS[
                        lineage
                    ],
                edgecolors="black",
                linewidths=0.6,
            )

            ax.text(
                row[
                    f"{loop}_arm"
                ],
                row[
                    f"{loop}_gene"
                ],
                lineage_label(
                    lineage
                ),
                fontsize=7,
                ha="left",
                va="bottom",
            )

        lo = min(
            ax.get_xlim()[0],
            ax.get_ylim()[0],
        )

        hi = max(
            ax.get_xlim()[1],
            ax.get_ylim()[1],
        )

        ax.plot(
            [lo, hi],
            [lo, hi],
            "--",
            color="grey",
            lw=0.8,
        )

        ax.set_xlabel(
            "Arm-balanced"
        )

        ax.set_ylabel(
            "Gene-balanced"
        )

        ax.set_title(loop)

    fig.suptitle(
        "Robustness to score-definition weighting"
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig63A_arm_vs_gene_balanced",
    )


def plot_dbox_decomposition(
    dbox_cent,
):
    order = LINEAGE_ORDER

    x = np.arange(
        len(order)
    )

    width = 0.22

    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    vals = (
        dbox_cent
        .set_index(
            "priority_lineage"
        )
        .reindex(order)
    )

    ax.bar(
        x - 1.5 * width,
        vals[
            "DBOX_ACT"
        ],
        width,
        label="PAR-bZIP arm",
    )

    ax.bar(
        x - 0.5 * width,
        vals[
            "DBOX_REP"
        ],
        width,
        label="NFIL3",
    )

    ax.bar(
        x + 0.5 * width,
        vals[
            "DBOX_GENE_BALANCED"
        ],
        width,
        label="Gene-balanced D-box",
    )

    ax.bar(
        x + 1.5 * width,
        vals[
            "DBOX_ARM_BALANCED"
        ],
        width,
        label="Arm-balanced D-box",
    )

    ax.axhline(
        0,
        color="black",
        lw=0.7,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            lineage_label(i)
            for i in order
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Mean standardized score"
    )

    ax.set_title(
        "D-box decomposition and NFIL3 sensitivity"
    )

    ax.legend(
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig63B_DBOX_NFIL3_decomposition",
    )


def plot_variance_partition(
    vp,
):
    x = np.arange(
        len(vp)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 4.8)
    )

    ax.bar(
        x - width / 2,
        vp[
            "increment_lineage_given_tissue"
        ],
        width,
        label="Incremental lineage R² | tissue",
    )

    ax.bar(
        x + width / 2,
        vp[
            "increment_tissue_given_lineage"
        ],
        width,
        label="Incremental tissue R² | lineage",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        vp["score"],
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "Incremental explained variance"
    )

    ax.set_title(
        "Lineage versus tissue contribution to static clock architecture"
    )

    ax.legend(
        frameon=False,
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig63C_lineage_vs_tissue_variance",
    )


def plot_bootstrap_probabilities(
    boot,
):
    piv = (
        boot.pivot(
            index="lineage",
            columns="loop",
            values="probability_max_component",
        )
        .reindex(
            LINEAGE_ORDER
        )
        .reindex(
            columns=[
                "EBOX",
                "RORE",
                "DBOX",
            ]
        )
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 4.0)
    )

    im = ax.imshow(
        piv.to_numpy(),
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="auto",
    )

    ax.set_xticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        [
            "E-box",
            "RORE",
            "D-box",
        ]
    )

    ax.set_yticks(
        np.arange(
            len(LINEAGE_ORDER)
        )
    )

    ax.set_yticklabels(
        [
            lineage_label(x)
            for x in LINEAGE_ORDER
        ]
    )

    for i in range(
        piv.shape[0]
    ):
        for j in range(
            piv.shape[1]
        ):
            ax.text(
                j,
                i,
                f"{piv.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Bootstrap probability of "
            "largest relative component"
        ),
    )

    ax.set_title(
        "Bootstrap support for relative loop skew"
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig63D_bootstrap_loop_probabilities",
    )


def plot_coupling_matrices(
    matrices,
):
    for lineage, mat in matrices.items():
        fig, ax = plt.subplots(
            figsize=(6, 5.2)
        )

        im = ax.imshow(
            mat.to_numpy(),
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
        )

        ax.set_xticks(
            np.arange(
                len(mat.columns)
            )
        )

        ax.set_xticklabels(
            mat.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            np.arange(
                len(mat.index)
            )
        )

        ax.set_yticklabels(
            mat.index
        )

        for i in range(
            len(mat)
        ):
            for j in range(
                len(mat)
            ):
                ax.text(
                    j,
                    i,
                    f"{mat.iloc[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )

        fig.colorbar(
            im,
            ax=ax,
            label="Spearman rho",
        )

        ax.set_title(
            f"{lineage_label(lineage)}: six-arm coupling"
        )

        fig.tight_layout()

        savefig(
            fig,
            DIR["fig"]
            / f"Fig63E_coupling_{lineage}",
        )


# =============================================================================
# 18. REPORT
# =============================================================================

def build_report(
    centroid,
    dbox_cent,
    variance,
    boot,
    perm,
    tissue_jk,
    donor_jk,
):
    lines = []

    lines.append(
        "# STATIC THREE-LOOP DEEP-DIVE REPORT"
    )

    lines.append("")
    lines.append(
        "This module tests robustness and alternative explanations "
        "for the lineage-specific static circadian loop architecture."
    )

    lines.append("")
    lines.append(
        "No temporal, rhythmic, amplitude, period, or causal dominance "
        "claims are made."
    )

    # -------------------------------------------------------------------------
    # score weighting
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "## 1. Arm-balanced versus gene-balanced scoring"
    )
    lines.append("")

    for _, r in centroid.iterrows():
        lines.append(
            f"- {r['lineage_label']}: "
            f"E-box arm/gene={r['EBOX_arm']:.3f}/{r['EBOX_gene']:.3f}; "
            f"RORE arm/gene={r['RORE_arm']:.3f}/{r['RORE_gene']:.3f}; "
            f"D-box arm/gene={r['DBOX_arm']:.3f}/{r['DBOX_gene']:.3f}."
        )

    # -------------------------------------------------------------------------
    # D-box
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "## 2. D-box / NFIL3 decomposition"
    )
    lines.append("")

    for _, r in dbox_cent.iterrows():
        lines.append(
            f"- {lineage_label(r['priority_lineage'])}: "
            f"PAR-bZIP={r['DBOX_ACT']:.3f}, "
            f"NFIL3={r['DBOX_REP']:.3f}, "
            f"gene-balanced={r['DBOX_GENE_BALANCED']:.3f}, "
            f"arm-balanced={r['DBOX_ARM_BALANCED']:.3f}."
        )

    # -------------------------------------------------------------------------
    # variance
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "## 3. Lineage versus tissue contribution"
    )
    lines.append("")

    for _, r in variance.iterrows():
        lines.append(
            f"- {r['score']}: incremental lineage R² | tissue="
            f"{r['increment_lineage_given_tissue']:.3f}; "
            f"incremental tissue R² | lineage="
            f"{r['increment_tissue_given_lineage']:.3f}."
        )

    # -------------------------------------------------------------------------
    # bootstrap
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "## 4. Bootstrap relative-loop support"
    )
    lines.append("")

    for lineage in LINEAGE_ORDER:
        p = boot[
            boot["lineage"]
            == lineage
        ]

        if p.empty:
            continue

        txt = ", ".join(
            [
                f"{r.loop}={r.probability_max_component:.2f}"
                for r in p.itertuples()
            ]
        )

        lines.append(
            f"- {lineage_label(lineage)}: {txt}"
        )

    # -------------------------------------------------------------------------
    # permutation
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "## 5. Tissue-stratified permutation"
    )
    lines.append("")

    for _, r in perm.iterrows():
        lines.append(
            f"- {r['score']}: observed F={r['observed_F']:.3f}, "
            f"permutation p={r['permutation_p']:.4g}, "
            f"95th-null={r['null_q95']:.3f}."
        )

    # -------------------------------------------------------------------------
    # jackknife
    # -------------------------------------------------------------------------

    lines.append("")
    lines.append(
        "## 6. Tissue/donor jackknife"
    )
    lines.append("")

    if not tissue_jk.empty:
        worst = (
            tissue_jk.assign(
                abs_delta=lambda x:
                    x["delta"].abs()
            )
            .sort_values(
                "abs_delta",
                ascending=False,
            )
            .head(10)
        )

        lines.append(
            "Largest leave-one-tissue-out centroid shifts:"
        )

        for _, r in worst.iterrows():
            lines.append(
                f"- {r['lineage']} / {r['score']} removing "
                f"{r['removed_tissue']}: Δ={r['delta']:.3f}"
            )

    if not donor_jk.empty:
        lines.append("")
        lines.append(
            "Maximum absolute leave-one-donor-out shift by score:"
        )

        maxs = (
            donor_jk.assign(
                abs_delta=lambda x:
                    x["delta"].abs()
            )
            .groupby(
                [
                    "lineage",
                    "score",
                ]
            )["abs_delta"]
            .max()
            .reset_index()
        )

        for _, r in maxs.iterrows():
            lines.append(
                f"- {r['lineage']} / {r['score']}: "
                f"{r['abs_delta']:.3f}"
            )

    report = "\n".join(
        lines
    )

    (
        DIR["report"]
        / "STATIC_3LOOP_DEEPDIVE_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )


# =============================================================================
# 19. MAIN
# =============================================================================

def main():
    warnings.filterwarnings(
        "ignore"
    )

    print("=" * 110)
    print("STATIC THREE-LOOP DEEP DIVE")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # load
    # -------------------------------------------------------------------------

    (
        meta,
        primary_meta,
        raw,
        z_global,
    ) = load_data()

    # -------------------------------------------------------------------------
    # global primary
    # -------------------------------------------------------------------------

    scored_all = (
        calculate_primary_scores(
            meta,
            z_global,
        )
    )

    primary = scored_all[
        scored_all[
            "priority_lineage"
        ].isin(
            LINEAGE_ORDER
        )
    ].copy()

    # -------------------------------------------------------------------------
    # 1. score robustness
    # -------------------------------------------------------------------------

    (
        primary,
        donor_lineage,
        robustness_centroid,
    ) = score_definition_robustness(
        primary
    )

    # save complete enriched score table
    primary.to_csv(
        DIR["source"]
        / "DEEPDIVE_OBSERVATION_SCORES.csv",
        index=False,
    )

    donor_lineage.to_csv(
        DIR["source"]
        / "DEEPDIVE_DONOR_LINEAGE_SCORES.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # add all six arms to donor-lineage table for coupling analysis
    # -------------------------------------------------------------------------

    six_arm_cols = [
        "EBOX_POS",
        "EBOX_NEG",
        "RORE_ACT",
        "RORE_REP",
        "DBOX_ACT",
        "DBOX_REP",
        "EBOX_ARM_BALANCED",
        "RORE_ARM_BALANCED",
        "DBOX_ARM_BALANCED",
        "EBOX_BALANCE",
        "RORE_BALANCE",
        "DBOX_BALANCE",
    ]

    donor_lineage_full = (
        primary.groupby(
            [
                "donor_id",
                "priority_lineage",
            ],
            observed=True,
        )[six_arm_cols]
        .mean()
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # donor × tissue × lineage
    # -------------------------------------------------------------------------

    donor_tissue_lineage = (
        primary.groupby(
            [
                "donor_id",
                "tissue",
                "priority_lineage",
            ],
            observed=True,
        )[six_arm_cols]
        .mean()
        .reset_index()
    )

    donor_tissue_lineage.to_csv(
        DIR["source"]
        / "DEEPDIVE_DONOR_TISSUE_LINEAGE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 2. D-box NFIL3
    # -------------------------------------------------------------------------

    dbox_dl, dbox_cent = (
        dbox_nfil3_analysis(
            primary
        )
    )

    # -------------------------------------------------------------------------
    # 3. within-tissue gene standardization
    # -------------------------------------------------------------------------

    (
        tissue_adjusted_obs,
        tissue_adjusted_dl,
        tissue_adjusted_cent,
    ) = tissue_adjusted_sensitivity(
        meta,
        primary_meta,
        raw,
    )

    # -------------------------------------------------------------------------
    # 4. within-donor paired contrasts
    # -------------------------------------------------------------------------

    paired = (
        within_donor_paired_analysis(
            donor_lineage_full
        )
    )

    # -------------------------------------------------------------------------
    # 5. mixed models
    # -------------------------------------------------------------------------

    mixed = mixed_models(
        donor_tissue_lineage
    )

    # -------------------------------------------------------------------------
    # 6. variance partition
    # -------------------------------------------------------------------------

    variance = (
        variance_partition(
            donor_tissue_lineage
        )
    )

    # -------------------------------------------------------------------------
    # 7. tissue/donor jackknife
    # -------------------------------------------------------------------------

    (
        tissue_jk,
        donor_jk,
    ) = jackknife_robustness(
        donor_tissue_lineage,
        donor_lineage_full,
    )

    # -------------------------------------------------------------------------
    # 8. coupling architecture
    # -------------------------------------------------------------------------

    coupling = (
        coupling_analysis(
            donor_lineage_full
        )
    )

    # -------------------------------------------------------------------------
    # 9. bootstrap max-loop probability
    # -------------------------------------------------------------------------

    boot = (
        bootstrap_loop_probabilities(
            donor_lineage_full,
            n_boot=10000,
        )
    )

    # -------------------------------------------------------------------------
    # 10. tissue-stratified permutation
    # -------------------------------------------------------------------------

    perm = (
        tissue_stratified_permutation(
            donor_tissue_lineage,
            n_perm=2000,
        )
    )

    # -------------------------------------------------------------------------
    # figures
    # -------------------------------------------------------------------------

    plot_arm_vs_gene(
        robustness_centroid
    )

    plot_dbox_decomposition(
        dbox_cent
    )

    plot_variance_partition(
        variance
    )

    plot_bootstrap_probabilities(
        boot
    )

    plot_coupling_matrices(
        coupling
    )

    # -------------------------------------------------------------------------
    # report
    # -------------------------------------------------------------------------

    build_report(
        robustness_centroid,
        dbox_cent,
        variance,
        boot,
        perm,
        tissue_jk,
        donor_jk,
    )

    # -------------------------------------------------------------------------
    # audit
    # -------------------------------------------------------------------------

    audit = {
        "analysis": "63_STATIC_3LOOP_DEEPDIVE",
        "n_total_observations":
            len(meta),
        "n_priority_observations":
            len(primary),
        "lineages":
            LINEAGE_ORDER,
        "n_genes":
            len(ALL_GENES),
        "arm_balanced_primary":
            True,
        "gene_balanced_sensitivity":
            True,
        "NFIL3_specific_analysis":
            True,
        "within_tissue_standardization":
            True,
        "within_donor_pairing":
            True,
        "mixed_model_used":
            STATSMODELS_AVAILABLE,
        "tissue_stratified_permutation":
            True,
        "leave_one_tissue_out":
            True,
        "leave_one_donor_out":
            True,
        "loop_coupling_analysis":
            True,
        "dynamic_claims_allowed":
            False,
    }

    (
        DIR["audit"]
        / "DEEPDIVE_AUDIT.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    completion = {
        "STATIC_3LOOP_DEEPDIVE_COMPLETE":
            True
    }

    (
        OUT
        / "STATIC_3LOOP_DEEPDIVE_COMPLETE.json"
    ).write_text(
        json.dumps(
            completion,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("STATIC 3-LOOP DEEP DIVE COMPLETE")
    print("=" * 110)
    print("OUTPUT:")
    print(OUT)
    print()
    print("Highest-priority outputs:")
    print("  Fig63A_arm_vs_gene_balanced")
    print("  Fig63B_DBOX_NFIL3_decomposition")
    print("  Fig63C_lineage_vs_tissue_variance")
    print("  Fig63D_bootstrap_loop_probabilities")
    print("  Fig63E_coupling_*")
    print()
    print("Key tables:")
    print("  ARM_VS_GENE_BALANCED_LINEAGE_CENTROIDS.csv")
    print("  DBOX_VARIANT_LINEAGE_CENTROIDS.csv")
    print("  WITHIN_DONOR_LINEAGE_CONTRASTS.csv")
    print("  LINEAGE_TISSUE_VARIANCE_PARTITION.csv")
    print("  TISSUE_STRATIFIED_LINEAGE_PERMUTATION.csv")
    print("  LEAVE_ONE_TISSUE_OUT.csv")
    print("  LEAVE_ONE_DONOR_OUT.csv")
    print("  BOOTSTRAP_MAX_LOOP_PROBABILITIES.csv")
    print()
    print("STATIC_3LOOP_DEEPDIVE_COMPLETE = TRUE")


if __name__ == "__main__":
    main()