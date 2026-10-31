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

from scipy.stats import spearmanr


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

OUT = (
    ROOT
    / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930"
)

DIR = {
    "audit": OUT / "00_AUDIT",
    "scores": OUT / "01_SCORES",
    "residual": OUT / "02_TISSUE_RESIDUALIZED",
    "coupling": OUT / "03_DIFFERENTIAL_COUPLING",
    "matched": OUT / "04_MATCHED_CONTRASTS",
    "fig": OUT / "05_FIGURES",
    "source": OUT / "source_data",
    "report": OUT / "06_REPORT",
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
# 2. BIOLOGICAL DEFINITIONS
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

ARM_COLS = [
    "EBOX_POS",
    "EBOX_NEG",
    "RORE_ACT",
    "RORE_REP",
    "DBOX_ACT",
    "DBOX_REP",
]

ENGAGEMENT_COLS = [
    "EBOX_ENGAGEMENT",
    "RORE_ENGAGEMENT",
    "DBOX_ENGAGEMENT",
]

BALANCE_COLS = [
    "EBOX_BALANCE",
    "RORE_BALANCE",
    "DBOX_BALANCE",
]

ALL_CLOCK_GENES = [
    g
    for arm in ARM_COLS
    for g in ARMS[arm]
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


def safe_zscore(x):
    x = np.asarray(x, dtype=float)

    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)

    sd[sd == 0] = 1.0
    sd[~np.isfinite(sd)] = 1.0

    z = (x - mu) / sd

    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite z-score.")

    return z


def bh_adjust(p):
    p = np.asarray(p, dtype=float)

    out = np.full(len(p), np.nan)

    valid = np.isfinite(p)

    pv = p[valid]

    if len(pv) == 0:
        return out

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
        1.0,
    )

    tmp = np.empty(len(pv))
    tmp[order] = adj

    out[valid] = tmp

    return out


def bootstrap_mean_ci(
    x,
    n_boot=10000,
    seed=20260930,
):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return x[0], x[0]

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(n_boot, len(x)),
    )

    vals = x[idx].mean(axis=1)

    return (
        np.quantile(vals, 0.025),
        np.quantile(vals, 0.975),
    )


def bootstrap_spearman(
    x,
    y,
    n_boot=5000,
    seed=20260930,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[ok]
    y = y[ok]

    n = len(x)

    if n < 5:
        return np.nan, np.nan, np.nan

    rho, _ = spearmanr(
        x,
        y,
    )

    rng = np.random.default_rng(seed)

    boots = []

    for _ in range(n_boot):
        idx = rng.integers(
            0,
            n,
            n,
        )

        xb = x[idx]
        yb = y[idx]

        if (
            np.std(xb) == 0
            or np.std(yb) == 0
        ):
            continue

        r, _ = spearmanr(
            xb,
            yb,
        )

        if np.isfinite(r):
            boots.append(r)

    if len(boots) < 100:
        return rho, np.nan, np.nan

    boots = np.asarray(boots)

    return (
        rho,
        np.quantile(boots, 0.025),
        np.quantile(boots, 0.975),
    )


# =============================================================================
# 4. LOAD DATA
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
            f"Expected 2030 rows, got {len(meta)}"
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
            f"Unexpected lineage counts: {counts}"
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
            f"Unexpected shape {expr.shape}"
        )

    if set(ids) != set(meta.index):
        raise RuntimeError(
            "Expression IDs do not match metadata."
        )

    order = (
        pd.Index(ids)
        .get_indexer(meta.index)
    )

    expr = expr[order]

    gene_index = {
        g: i
        for i, g in enumerate(genes)
    }

    missing = [
        g
        for g in ALL_CLOCK_GENES
        if g not in gene_index
    ]

    if missing:
        raise RuntimeError(
            f"Missing clock genes: {missing}"
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
        safe_zscore(
            raw.to_numpy()
        ),
        index=raw.index,
        columns=raw.columns,
    )

    return meta, primary, raw, z


# =============================================================================
# 5. ARM + LOOP SCORES
# =============================================================================

def build_scores(
    meta,
    z,
):
    df = meta[
        [
            "group_id",
            "donor_id",
            "tissue",
            "priority_lineage",
        ]
    ].copy()

    for arm, genes in ARMS.items():
        df[arm] = (
            z[genes]
            .mean(axis=1)
        )

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

    return df


# =============================================================================
# 6. AGGREGATE TO DONOR × TISSUE × LINEAGE
# =============================================================================

def aggregate_dtl(
    scores,
):
    cols = (
        ARM_COLS
        + ENGAGEMENT_COLS
        + BALANCE_COLS
    )

    primary = scores[
        scores["priority_lineage"]
        .isin(LINEAGE_ORDER)
    ].copy()

    dtl = (
        primary
        .groupby(
            [
                "donor_id",
                "tissue",
                "priority_lineage",
            ],
            observed=True,
        )[cols]
        .mean()
        .reset_index()
    )

    dtl.to_csv(
        DIR["source"]
        / "DONOR_TISSUE_LINEAGE_SCORES.csv",
        index=False,
    )

    return dtl


# =============================================================================
# 7. TISSUE RESIDUALIZATION
# =============================================================================

def residualize_by_tissue(
    dtl,
):
    """
    For each score:
        score_resid = score - tissue-specific mean

    This is equivalent to residualizing against tissue fixed effects
    without altering within-tissue lineage contrasts.

    We do this separately per score.
    """

    out = dtl.copy()

    score_cols = (
        ARM_COLS
        + ENGAGEMENT_COLS
        + BALANCE_COLS
    )

    for score in score_cols:
        tissue_mean = (
            out.groupby(
                "tissue",
                observed=True,
            )[score]
            .transform("mean")
        )

        out[
            f"{score}_RESID_TISSUE"
        ] = (
            out[score]
            - tissue_mean
        )

    out.to_csv(
        DIR["residual"]
        / "TISSUE_RESIDUALIZED_DONOR_TISSUE_LINEAGE.csv",
        index=False,
    )

    return out


# =============================================================================
# 8. RESIDUALIZED COUPLING MATRICES
# =============================================================================

def residualized_coupling(
    resid,
):
    records = []
    matrices = {}

    resid_arms = [
        f"{x}_RESID_TISSUE"
        for x in ARM_COLS
    ]

    for lineage in LINEAGE_ORDER:
        p = resid[
            resid["priority_lineage"]
            == lineage
        ].copy()

        corr = (
            p[resid_arms]
            .corr(
                method="spearman"
            )
        )

        corr.index = ARM_COLS
        corr.columns = ARM_COLS

        matrices[lineage] = corr

        corr.to_csv(
            DIR["coupling"]
            / f"RESIDUALIZED_COUPLING_{lineage}.csv"
        )

        for a, b in combinations(
            ARM_COLS,
            2,
        ):
            x = (
                p[
                    f"{a}_RESID_TISSUE"
                ]
                .to_numpy()
            )

            y = (
                p[
                    f"{b}_RESID_TISSUE"
                ]
                .to_numpy()
            )

            rho, lo, hi = bootstrap_spearman(
                x,
                y,
                n_boot=5000,
                seed=(
                    20260930
                    + LINEAGE_ORDER.index(lineage)
                    * 100
                    + ARM_COLS.index(a)
                    * 10
                    + ARM_COLS.index(b)
                ),
            )

            records.append({
                "lineage": lineage,
                "lineage_label":
                    LINEAGE_DISPLAY[lineage],
                "arm_1": a,
                "arm_2": b,
                "n_dtl_observations":
                    len(p),
                "rho_residualized":
                    rho,
                "bootstrap_ci_low":
                    lo,
                "bootstrap_ci_high":
                    hi,
            })

    edges = pd.DataFrame(
        records
    )

    edges.to_csv(
        DIR["coupling"]
        / "RESIDUALIZED_COUPLING_EDGES.csv",
        index=False,
    )

    return matrices, edges


# =============================================================================
# 9. DIFFERENTIAL COUPLING BETWEEN LINEAGES
# =============================================================================

def differential_coupling(
    resid,
    n_perm=5000,
):
    """
    For every arm-pair and every lineage-pair:

    delta_rho =
        rho(lineage A) - rho(lineage B)

    Permutation:
      - preserve tissue
      - shuffle lineage labels within tissue
      - recompute delta rho

    This controls tissue composition in the null.
    """

    rng = np.random.default_rng(
        20260930
    )

    results = []

    arm_pairs = list(
        combinations(
            ARM_COLS,
            2,
        )
    )

    lineage_pairs = list(
        combinations(
            LINEAGE_ORDER,
            2,
        )
    )

    for a, b in arm_pairs:
        ca = f"{a}_RESID_TISSUE"
        cb = f"{b}_RESID_TISSUE"

        for l1, l2 in lineage_pairs:
            p1 = resid[
                resid["priority_lineage"]
                == l1
            ]

            p2 = resid[
                resid["priority_lineage"]
                == l2
            ]

            if len(p1) < 5 or len(p2) < 5:
                continue

            rho1, _ = spearmanr(
                p1[ca],
                p1[cb],
            )

            rho2, _ = spearmanr(
                p2[ca],
                p2[cb],
            )

            observed_delta = (
                rho1 - rho2
            )

            sub = resid[
                resid["priority_lineage"]
                .isin(
                    [l1, l2]
                )
            ][
                [
                    "tissue",
                    "priority_lineage",
                    ca,
                    cb,
                ]
            ].copy()

            null = []

            for _ in range(n_perm):
                perm_labels = (
                    sub["priority_lineage"]
                    .copy()
                )

                for tissue, idx in (
                    sub.groupby(
                        "tissue",
                        observed=True,
                    ).groups.items()
                ):
                    vals = (
                        perm_labels.loc[idx]
                        .to_numpy()
                        .copy()
                    )

                    rng.shuffle(vals)

                    perm_labels.loc[idx] = vals

                s1 = sub[
                    perm_labels == l1
                ]

                s2 = sub[
                    perm_labels == l2
                ]

                if (
                    len(s1) < 5
                    or len(s2) < 5
                ):
                    continue

                r1, _ = spearmanr(
                    s1[ca],
                    s1[cb],
                )

                r2, _ = spearmanr(
                    s2[ca],
                    s2[cb],
                )

                if (
                    np.isfinite(r1)
                    and np.isfinite(r2)
                ):
                    null.append(
                        r1 - r2
                    )

            null = np.asarray(
                null
            )

            if len(null):
                empirical_p = (
                    1
                    + np.sum(
                        np.abs(null)
                        >= abs(observed_delta)
                    )
                ) / (
                    1
                    + len(null)
                )

                null_lo = np.quantile(
                    null,
                    0.025,
                )

                null_hi = np.quantile(
                    null,
                    0.975,
                )

            else:
                empirical_p = np.nan
                null_lo = np.nan
                null_hi = np.nan

            results.append({
                "arm_1": a,
                "arm_2": b,

                "lineage_1": l1,
                "lineage_2": l2,

                "n1": len(p1),
                "n2": len(p2),

                "rho_1": rho1,
                "rho_2": rho2,

                "delta_rho_1_minus_2":
                    observed_delta,

                "permutation_p":
                    empirical_p,

                "null_delta_ci_low":
                    null_lo,
                "null_delta_ci_high":
                    null_hi,

                "n_permutations":
                    len(null),
            })

    out = pd.DataFrame(
        results
    )

    out["p_adj_BH"] = (
        bh_adjust(
            out["permutation_p"]
            .to_numpy()
        )
    )

    out.to_csv(
        DIR["coupling"]
        / "DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv",
        index=False,
    )

    return out


# =============================================================================
# 10. NETWORK SUMMARY METRICS
# =============================================================================

def network_summary(
    matrices,
):
    records = []

    within_edges = {
        "EBOX": (
            "EBOX_POS",
            "EBOX_NEG",
        ),
        "RORE": (
            "RORE_ACT",
            "RORE_REP",
        ),
        "DBOX": (
            "DBOX_ACT",
            "DBOX_REP",
        ),
    }

    cross_groups = {
        "EBOX_RORE": [
            ("EBOX_POS", "RORE_ACT"),
            ("EBOX_POS", "RORE_REP"),
            ("EBOX_NEG", "RORE_ACT"),
            ("EBOX_NEG", "RORE_REP"),
        ],
        "EBOX_DBOX": [
            ("EBOX_POS", "DBOX_ACT"),
            ("EBOX_POS", "DBOX_REP"),
            ("EBOX_NEG", "DBOX_ACT"),
            ("EBOX_NEG", "DBOX_REP"),
        ],
        "RORE_DBOX": [
            ("RORE_ACT", "DBOX_ACT"),
            ("RORE_ACT", "DBOX_REP"),
            ("RORE_REP", "DBOX_ACT"),
            ("RORE_REP", "DBOX_REP"),
        ],
    }

    for lineage in LINEAGE_ORDER:
        m = matrices[lineage]

        rec = {
            "lineage": lineage,
            "lineage_label":
                LINEAGE_DISPLAY[lineage],
        }

        for name, (a, b) in (
            within_edges.items()
        ):
            rec[
                f"{name}_WITHIN_COUPLING"
            ] = m.loc[a, b]

        for name, pairs in (
            cross_groups.items()
        ):
            vals = [
                m.loc[a, b]
                for a, b in pairs
            ]

            rec[
                f"{name}_MEAN_COUPLING"
            ] = np.nanmean(vals)

            rec[
                f"{name}_MEAN_ABS_COUPLING"
            ] = np.nanmean(
                np.abs(vals)
            )

        offdiag = []

        for a, b in combinations(
            ARM_COLS,
            2,
        ):
            offdiag.append(
                m.loc[a, b]
            )

        rec[
            "NETWORK_MEAN_COUPLING"
        ] = np.nanmean(
            offdiag
        )

        rec[
            "NETWORK_MEAN_ABS_COUPLING"
        ] = np.nanmean(
            np.abs(offdiag)
        )

        records.append(rec)

    out = pd.DataFrame(
        records
    )

    out.to_csv(
        DIR["coupling"]
        / "RESIDUALIZED_NETWORK_SUMMARY.csv",
        index=False,
    )

    return out


# =============================================================================
# 11. SAME DONOR + SAME TISSUE MATCHED CONTRASTS
# =============================================================================

def same_donor_same_tissue_contrasts(
    dtl,
):
    """
    Compare lineage A vs lineage B only when both occur in the same:
        donor_id
        tissue

    Each matched donor-tissue pair contributes one paired delta.
    """

    score_cols = (
        ARM_COLS
        + ENGAGEMENT_COLS
        + BALANCE_COLS
    )

    records = []
    raw_records = []

    for score in score_cols:

        for l1, l2 in combinations(
            LINEAGE_ORDER,
            2,
        ):

            a = (
                dtl[
                    dtl["priority_lineage"]
                    == l1
                ][
                    [
                        "donor_id",
                        "tissue",
                        score,
                    ]
                ]
                .rename(
                    columns={
                        score: "score_1"
                    }
                )
            )

            b = (
                dtl[
                    dtl["priority_lineage"]
                    == l2
                ][
                    [
                        "donor_id",
                        "tissue",
                        score,
                    ]
                ]
                .rename(
                    columns={
                        score: "score_2"
                    }
                )
            )

            paired = a.merge(
                b,
                on=[
                    "donor_id",
                    "tissue",
                ],
                how="inner",
            )

            if len(paired) < 3:
                continue

            paired["delta"] = (
                paired["score_1"]
                - paired["score_2"]
            )

            # save raw matches
            for _, r in (
                paired.iterrows()
            ):
                raw_records.append({
                    "score": score,
                    "lineage_1": l1,
                    "lineage_2": l2,
                    "donor_id":
                        r["donor_id"],
                    "tissue":
                        r["tissue"],
                    "score_1":
                        r["score_1"],
                    "score_2":
                        r["score_2"],
                    "delta_1_minus_2":
                        r["delta"],
                })

            delta = (
                paired["delta"]
                .to_numpy()
            )

            lo, hi = bootstrap_mean_ci(
                delta,
                n_boot=10000,
                seed=(
                    20260930
                    + LINEAGE_ORDER.index(l1)
                    * 100
                    + LINEAGE_ORDER.index(l2)
                    * 10
                    + score_cols.index(score)
                ),
            )

            # donor-block sign consistency:
            # first average multiple tissues per donor
            donor_delta = (
                paired.groupby(
                    "donor_id",
                    observed=True,
                )["delta"]
                .mean()
            )

            records.append({
                "score": score,

                "lineage_1": l1,
                "lineage_1_label":
                    LINEAGE_DISPLAY[l1],

                "lineage_2": l2,
                "lineage_2_label":
                    LINEAGE_DISPLAY[l2],

                "n_matched_donor_tissue_pairs":
                    len(paired),

                "n_unique_donors":
                    paired["donor_id"]
                    .nunique(),

                "n_unique_tissues":
                    paired["tissue"]
                    .nunique(),

                "mean_delta_1_minus_2":
                    np.mean(delta),

                "median_delta_1_minus_2":
                    np.median(delta),

                "bootstrap_ci_low":
                    lo,

                "bootstrap_ci_high":
                    hi,

                "fraction_pairwise_delta_positive":
                    np.mean(
                        delta > 0
                    ),

                "donor_mean_fraction_positive":
                    np.mean(
                        donor_delta > 0
                    ),
            })

    summary = pd.DataFrame(
        records
    )

    raw = pd.DataFrame(
        raw_records
    )

    summary.to_csv(
        DIR["matched"]
        / "SAME_DONOR_SAME_TISSUE_MATCHED_CONTRASTS.csv",
        index=False,
    )

    raw.to_csv(
        DIR["matched"]
        / "SAME_DONOR_SAME_TISSUE_MATCHED_RAW_PAIRS.csv",
        index=False,
    )

    return summary, raw


# =============================================================================
# 12. STRICT DONOR-BLOCK BOOTSTRAP FOR MATCHED CONTRASTS
# =============================================================================

def donor_block_bootstrap_matched(
    raw_pairs,
    n_boot=10000,
):
    """
    Bootstrap donor IDs, not individual donor-tissue rows.

    This avoids donors with multiple tissues being over-weighted
    in the uncertainty estimate.
    """

    rng = np.random.default_rng(
        20260930
    )

    records = []

    keys = (
        raw_pairs[
            [
                "score",
                "lineage_1",
                "lineage_2",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    for (
        score,
        l1,
        l2,
    ) in keys:

        p = raw_pairs[
            (raw_pairs["score"] == score)
            & (
                raw_pairs["lineage_1"]
                == l1
            )
            & (
                raw_pairs["lineage_2"]
                == l2
            )
        ]

        donor_means = (
            p.groupby(
                "donor_id",
                observed=True,
            )["delta_1_minus_2"]
            .mean()
        )

        vals = (
            donor_means
            .to_numpy()
        )

        n = len(vals)

        if n < 3:
            continue

        boots = []

        for _ in range(n_boot):
            idx = rng.integers(
                0,
                n,
                n,
            )

            boots.append(
                vals[idx].mean()
            )

        boots = np.asarray(
            boots
        )

        records.append({
            "score": score,
            "lineage_1": l1,
            "lineage_2": l2,

            "n_unique_donors":
                n,

            "donor_block_mean_delta":
                vals.mean(),

            "donor_block_ci_low":
                np.quantile(
                    boots,
                    0.025,
                ),

            "donor_block_ci_high":
                np.quantile(
                    boots,
                    0.975,
                ),

            "fraction_donors_positive":
                np.mean(
                    vals > 0
                ),
        })

    out = pd.DataFrame(
        records
    )

    out.to_csv(
        DIR["matched"]
        / "MATCHED_CONTRASTS_DONOR_BLOCK_BOOTSTRAP.csv",
        index=False,
    )

    return out


# =============================================================================
# 13. PLOTS — RESIDUALIZED COUPLING
# =============================================================================

def plot_residualized_coupling(
    matrices,
):
    for lineage in LINEAGE_ORDER:
        m = matrices[lineage]

        fig, ax = plt.subplots(
            figsize=(6.1, 5.2)
        )

        im = ax.imshow(
            m.to_numpy(),
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            aspect="equal",
        )

        ax.set_xticks(
            np.arange(
                len(ARM_COLS)
            )
        )

        ax.set_xticklabels(
            ARM_COLS,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(
            np.arange(
                len(ARM_COLS)
            )
        )

        ax.set_yticklabels(
            ARM_COLS
        )

        for i in range(
            len(ARM_COLS)
        ):
            for j in range(
                len(ARM_COLS)
            ):
                ax.text(
                    j,
                    i,
                    f"{m.iloc[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        fig.colorbar(
            im,
            ax=ax,
            label="Tissue-residualized Spearman rho",
        )

        ax.set_title(
            f"{LINEAGE_DISPLAY[lineage]}: "
            f"tissue-residualized six-arm coupling"
        )

        fig.tight_layout()

        savefig(
            fig,
            DIR["fig"]
            / f"Fig64A_residualized_coupling_{lineage}",
        )


# =============================================================================
# 14. PLOT — NETWORK SUMMARY
# =============================================================================

def plot_network_summary(
    summary,
):
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
        "E-box↔RORE",
        "E-box↔D-box",
        "RORE↔D-box",
        "Network\nintegration",
    ]

    mat = (
        summary
        .set_index(
            "lineage"
        )
        .reindex(
            LINEAGE_ORDER
        )[cols]
    )

    arr = mat.to_numpy()

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    vmax = max(
        0.5,
        np.nanmax(
            np.abs(arr)
        ),
    )

    im = ax.imshow(
        arr,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(cols)
        )
    )

    ax.set_xticklabels(
        labels,
    )

    ax.set_yticks(
        np.arange(
            len(LINEAGE_ORDER)
        )
    )

    ax.set_yticklabels(
        [
            LINEAGE_DISPLAY[x]
            for x in LINEAGE_ORDER
        ]
    )

    for i in range(
        arr.shape[0]
    ):
        for j in range(
            arr.shape[1]
        ):
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
        label="Residualized coupling",
    )

    ax.set_title(
        "Lineage-specific residualized circadian network topology"
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig64B_network_summary",
    )


# =============================================================================
# 15. PLOT — DIFFERENTIAL COUPLING
# =============================================================================

def plot_top_differential_edges(
    diff,
    top_n=25,
):
    p = diff.copy()

    p["abs_delta"] = (
        p[
            "delta_rho_1_minus_2"
        ]
        .abs()
    )

    p = p.sort_values(
        [
            "p_adj_BH",
            "abs_delta",
        ],
        ascending=[
            True,
            False,
        ],
    )

    p = p.head(
        top_n
    ).copy()

    labels = (
        p["lineage_1"]
        .map(LINEAGE_DISPLAY)
        + " vs "
        + p["lineage_2"]
        .map(LINEAGE_DISPLAY)
        + " | "
        + p["arm_1"]
        + "–"
        + p["arm_2"]
    )

    y = np.arange(
        len(p)
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(
                5,
                0.3 * len(p)
            ),
        )
    )

    ax.barh(
        y,
        p[
            "delta_rho_1_minus_2"
        ],
    )

    ax.axvline(
        0,
        color="black",
        lw=0.8,
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
        "Δ Spearman rho "
        "(lineage 1 − lineage 2)"
    )

    ax.set_title(
        "Top tissue-residualized differential coupling edges"
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig64C_top_differential_coupling",
    )


# =============================================================================
# 16. PLOT — MATCHED CONTRAST HEATMAP
# =============================================================================

def plot_matched_engagement_heatmap(
    matched_block,
):
    use = matched_block[
        matched_block["score"]
        .isin(
            ENGAGEMENT_COLS
        )
    ].copy()

    use[
        "comparison"
    ] = (
        use["lineage_1"]
        .map(LINEAGE_DISPLAY)
        + " − "
        + use["lineage_2"]
        .map(LINEAGE_DISPLAY)
    )

    comps = (
        use["comparison"]
        .drop_duplicates()
        .tolist()
    )

    mat = (
        use.pivot(
            index="comparison",
            columns="score",
            values="donor_block_mean_delta",
        )
        .reindex(
            comps
        )
        .reindex(
            columns=ENGAGEMENT_COLS
        )
    )

    arr = mat.to_numpy()

    vmax = max(
        0.2,
        np.nanmax(
            np.abs(arr)
        ),
    )

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            max(
                5,
                0.45 * len(mat)
            ),
        )
    )

    im = ax.imshow(
        arr,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
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
            len(mat)
        )
    )

    ax.set_yticklabels(
        mat.index,
        fontsize=7,
    )

    for i in range(
        arr.shape[0]
    ):
        for j in range(
            arr.shape[1]
        ):
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
        label=(
            "Matched donor+tissue "
            "mean lineage difference"
        ),
    )

    ax.set_title(
        "Same-donor same-tissue matched three-loop contrasts"
    )

    fig.tight_layout()

    savefig(
        fig,
        DIR["fig"]
        / "Fig64D_matched_engagement_heatmap",
    )


# =============================================================================
# 17. REPORT
# =============================================================================

def build_report(
    network,
    diff,
    matched,
):
    lines = []

    lines.append(
        "# 64 — Tissue-residualized coupling and matched lineage contrasts"
    )

    lines.append("")
    lines.append(
        "This analysis tests whether lineage-specific static circadian "
        "architecture persists after removing tissue-level mean effects "
        "and under same-donor same-tissue matching."
    )

    lines.append("")
    lines.append(
        "No temporal, rhythmic, amplitude, phase, period, or causal "
        "dominance claims are made."
    )

    lines.append("")
    lines.append(
        "## 1. Residualized network summary"
    )

    lines.append("")

    for _, r in network.iterrows():
        lines.append(
            f"- {r['lineage_label']}: "
            f"E-box within={r['EBOX_WITHIN_COUPLING']:.3f}; "
            f"RORE within={r['RORE_WITHIN_COUPLING']:.3f}; "
            f"D-box within={r['DBOX_WITHIN_COUPLING']:.3f}; "
            f"network mean |rho|={r['NETWORK_MEAN_ABS_COUPLING']:.3f}."
        )

    lines.append("")
    lines.append(
        "## 2. Strongest differential coupling edges"
    )

    lines.append("")

    top = (
        diff.sort_values(
            [
                "p_adj_BH",
                "delta_rho_1_minus_2",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(20)
    )

    for _, r in top.iterrows():
        lines.append(
            f"- {r['lineage_1']} vs {r['lineage_2']}, "
            f"{r['arm_1']}–{r['arm_2']}: "
            f"rho={r['rho_1']:.3f} vs {r['rho_2']:.3f}, "
            f"Δrho={r['delta_rho_1_minus_2']:.3f}, "
            f"BH={r['p_adj_BH']:.3g}."
        )

    lines.append("")
    lines.append(
        "## 3. Strongest same-donor same-tissue matched contrasts"
    )

    lines.append("")

    m = matched[
        matched["score"]
        .isin(
            ENGAGEMENT_COLS
        )
    ].copy()

    m["abs_delta"] = (
        m[
            "donor_block_mean_delta"
        ]
        .abs()
    )

    m = m.sort_values(
        "abs_delta",
        ascending=False,
    ).head(20)

    for _, r in m.iterrows():
        lines.append(
            f"- {r['score']}: "
            f"{r['lineage_1']} − {r['lineage_2']} = "
            f"{r['donor_block_mean_delta']:.3f}; "
            f"95% CI "
            f"[{r['donor_block_ci_low']:.3f}, "
            f"{r['donor_block_ci_high']:.3f}]; "
            f"{r['n_unique_donors']} donors."
        )

    report = "\n".join(
        lines
    )

    (
        DIR["report"]
        / "RESIDUALIZED_COUPLING_MATCHED_REPORT.md"
    ).write_text(
        report,
        encoding="utf-8",
    )


# =============================================================================
# 18. MAIN
# =============================================================================

def main():
    warnings.filterwarnings(
        "ignore"
    )

    print("=" * 110)
    print("64 — TISSUE-RESIDUALIZED COUPLING + SAME-DONOR/SAME-TISSUE MATCHING")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # load + scores
    # -------------------------------------------------------------------------

    meta, primary, raw, z = (
        load_data()
    )

    scores = build_scores(
        meta,
        z,
    )

    scores.to_csv(
        DIR["scores"]
        / "STATIC_3LOOP_SCORES.csv",
        index=False,
    )

    dtl = aggregate_dtl(
        scores
    )

    # -------------------------------------------------------------------------
    # A. tissue residualization
    # -------------------------------------------------------------------------

    resid = residualize_by_tissue(
        dtl
    )

    matrices, edges = (
        residualized_coupling(
            resid
        )
    )

    diff = differential_coupling(
        resid,
        n_perm=5000,
    )

    network = network_summary(
        matrices
    )

    # -------------------------------------------------------------------------
    # B. matched donor + tissue contrasts
    # -------------------------------------------------------------------------

    matched_summary, matched_raw = (
        same_donor_same_tissue_contrasts(
            dtl
        )
    )

    matched_block = (
        donor_block_bootstrap_matched(
            matched_raw,
            n_boot=10000,
        )
    )

    # -------------------------------------------------------------------------
    # figures
    # -------------------------------------------------------------------------

    plot_residualized_coupling(
        matrices
    )

    plot_network_summary(
        network
    )

    plot_top_differential_edges(
        diff,
        top_n=25,
    )

    plot_matched_engagement_heatmap(
        matched_block
    )

    # -------------------------------------------------------------------------
    # report
    # -------------------------------------------------------------------------

    build_report(
        network,
        diff,
        matched_block,
    )

    # -------------------------------------------------------------------------
    # audit
    # -------------------------------------------------------------------------

    audit = {
        "analysis":
            "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS",

        "n_total_observations":
            len(meta),

        "n_priority_lineage_observations":
            len(primary),

        "n_donor_tissue_lineage_rows":
            len(dtl),

        "tissue_residualization":
            "subtract tissue-specific mean per score",

        "coupling_metric":
            "Spearman correlation",

        "differential_coupling_permutation":
            "lineage labels shuffled within tissue",

        "matched_contrast_unit":
            "same donor + same tissue + lineage pair",

        "matched_uncertainty":
            "donor-block bootstrap",

        "n_diff_coupling_permutations":
            5000,

        "n_matched_bootstraps":
            10000,

        "dynamic_claims_allowed":
            False,
    }

    (
        DIR["audit"]
        / "ANALYSIS_AUDIT.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        OUT
        / "STATIC_3LOOP_64_COMPLETE.json"
    ).write_text(
        json.dumps(
            {
                "STATIC_3LOOP_64_COMPLETE":
                    True
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("64 ANALYSIS COMPLETE")
    print("=" * 110)
    print("OUTPUT:")
    print(OUT)
    print()
    print("Key outputs:")
    print("  RESIDUALIZED_COUPLING_EDGES.csv")
    print("  DIFFERENTIAL_COUPLING_LINEAGE_COMPARISONS.csv")
    print("  RESIDUALIZED_NETWORK_SUMMARY.csv")
    print("  SAME_DONOR_SAME_TISSUE_MATCHED_CONTRASTS.csv")
    print("  MATCHED_CONTRASTS_DONOR_BLOCK_BOOTSTRAP.csv")
    print()
    print("Key figures:")
    print("  Fig64A_residualized_coupling_*")
    print("  Fig64B_network_summary")
    print("  Fig64C_top_differential_coupling")
    print("  Fig64D_matched_engagement_heatmap")
    print()
    print("STATIC_3LOOP_64_COMPLETE = TRUE")


if __name__ == "__main__":
    main()