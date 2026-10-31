#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
42_run_corrected_24feature_variance_partition.py

Fig.2C
Corrected clock-associated feature variance allocation.

Uses:
1. current corrected frozen metadata
2. current corrected A2 support
3. frozen 24-feature static clock-associated architecture matrix
4. finite-design hierarchical Shapley SS allocation

Components:
- Lineage
- Tissue
- Lineage × tissue
- Donor
- Residual

IMPORTANT:
This is STATIC architecture only.
No rhythmicity / phase / amplitude inference.

The five fractions for every feature sum to 1.
"""

from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STYLE
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


# No gridlines in current figure style.
COLORS = {
    "Lineage": "#4E79A7",
    "Tissue": "#F28E2B",
    "Lineage × tissue": "#B07AA1",
    "Donor": "#59A14F",
}


CANONICAL_GENES = [
    "ARNTL",
    "CLOCK",
    "NPAS2",
    "PER1",
    "PER2",
    "PER3",
    "CRY1",
    "CRY2",
    "NR1D1",
    "NR1D2",
    "RORA",
    "RORB",
    "RORC",
    "DBP",
    "TEF",
    "HLF",
    "NFIL3",
]


# ============================================================
# HELPERS
# ============================================================

def save_fig(fig, base):
    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


def pretty_feature(x):

    s = str(x)

    if s.startswith("expr_"):
        return s[5:]

    mapping = {
        "core_clock_mean":
            "Core clock mean",

        "expanded_clock_mean":
            "Expanded clock mean",

        "positive_limb_mean":
            "Positive-limb mean",

        "negative_limb_mean":
            "Negative-limb mean",

        "auxiliary_clock_mean":
            "Auxiliary clock mean",

        "positive_limb_fraction":
            "Positive-limb fraction",

        "positive_negative_balance":
            "Positive–negative balance",

        "core_completeness_at_0.05":
            "Core completeness (0.05)",
    }

    return mapping.get(
        s,
        s.replace("_", " ")
    )


def orthogonal_basis(design):
    """
    Rank-aware orthonormal basis of a design matrix.
    Equivalent purpose to the historical A5 SVD implementation.
    """

    design = np.asarray(
        design,
        dtype=float
    )

    u, s, _ = np.linalg.svd(
        design,
        full_matrices=False
    )

    if len(s) == 0:
        return np.empty(
            (design.shape[0], 0)
        ), 0

    tol = (
        s[0]
        * max(design.shape)
        * np.finfo(float).eps
    )

    rank = int(
        np.sum(s > tol)
    )

    return u[:, :rank], rank


# ============================================================
# FINITE-DESIGN VARIANCE ALLOCATION
# ============================================================

class Partition:
    """
    Finite-design hierarchical Shapley sums of squares.

    Main effects:
        lineage
        tissue
        donor

    Additive fitted SS is allocated by averaging incremental
    fitted SS across all 6 insertion orders.

    Interaction:
        observed lineage × tissue structure beyond the additive
        lineage + tissue + donor model.

    Residual:
        remaining SS after observed-pair + donor model.

    This is NOT REML and NOT a causal variance-component model.
    """

    def __init__(self, meta):

        self.meta = meta.copy()
        self.n = len(meta)

        # factor order:
        # 0 lineage
        # 1 tissue
        # 2 donor
        factor_arrays = [
            pd.get_dummies(
                meta["priority_lineage"],
                dtype=float
            ).to_numpy(),

            pd.get_dummies(
                meta["tissue"],
                dtype=float
            ).to_numpy(),

            pd.get_dummies(
                meta["donor_id"],
                dtype=float
            ).to_numpy(),
        ]

        # q[bits] = fitted subspace for every additive subset
        self.q = {
            0: np.ones(
                (self.n, 1)
            ) / np.sqrt(self.n)
        }

        self.ranks = {
            0: 1
        }

        for bits in range(1, 8):

            cols = [
                np.ones(self.n)
            ]

            for i in range(3):
                if bits & (1 << i):
                    cols.append(
                        factor_arrays[i]
                    )

            design = np.column_stack(
                cols
            )

            self.q[bits], self.ranks[bits] = (
                orthogonal_basis(design)
            )

        # full model:
        # observed lineage × tissue pairs + donor
        pair = (
            meta["priority_lineage"].astype(str)
            + "|"
            + meta["tissue"].astype(str)
        )

        full_design = np.column_stack([
            np.ones(self.n),

            pd.get_dummies(
                pair,
                dtype=float
            ).to_numpy(),

            factor_arrays[2],
        ])

        self.full, self.full_rank = (
            orthogonal_basis(
                full_design
            )
        )

        self.diagnostics = {
            "n_supported_rows":
                int(self.n),

            "n_lineages":
                int(
                    meta["priority_lineage"]
                    .nunique()
                ),

            "n_tissues":
                int(
                    meta["tissue"]
                    .nunique()
                ),

            "n_donors":
                int(
                    meta["donor_id"]
                    .nunique()
                ),

            "observed_lineage_tissue_pairs":
                int(pair.nunique()),

            "possible_lineage_tissue_pairs":
                int(
                    meta["priority_lineage"].nunique()
                    * meta["tissue"].nunique()
                ),

            "additive_rank":
                int(self.ranks[7]),

            "interaction_full_rank":
                int(self.full_rank),

            "residual_df":
                int(
                    self.n
                    - self.full_rank
                ),
        }

    def fit(self, y):
        """
        y:
            n_samples × n_features

        returns:
            component_ss_per_row
            fractions
            total_variance
        """

        y = np.asarray(
            y,
            dtype=float
        )

        # center each feature
        z = y - y.mean(
            axis=0,
            keepdims=True
        )

        total_ss = (
            z * z
        ).sum(axis=0)

        # fitted SS for every additive subset
        explained = {}

        for bits, q in self.q.items():

            if bits == 0:
                explained[bits] = np.zeros(
                    y.shape[1]
                )
            else:
                explained[bits] = (
                    (q.T @ z) ** 2
                ).sum(axis=0)

        components = np.zeros(
            (
                y.shape[1],
                5
            ),
            dtype=float
        )

        # Average all 6 orders:
        # lineage / tissue / donor
        orders = list(
            itertools.permutations(
                range(3)
            )
        )

        for order in orders:

            previous = 0

            for factor in order:

                nxt = (
                    previous
                    | (1 << factor)
                )

                increment = (
                    explained[nxt]
                    - explained[previous]
                )

                components[:, factor] += (
                    increment
                    / len(orders)
                )

                previous = nxt

        # Full observed lineage×tissue + donor model
        full_explained = (
            (self.full.T @ z) ** 2
        ).sum(axis=0)

        # interaction = full model - additive model
        components[:, 3] = (
            full_explained
            - explained[7]
        )

        # residual
        components[:, 4] = (
            total_ss
            - full_explained
        )

        # only numerical floating-point tolerance
        tol = (
            np.maximum(
                total_ss,
                1.0
            )
            * 1e-10
        )

        if np.any(
            components
            < -tol[:, None]
        ):
            bad = np.where(
                np.any(
                    components
                    < -tol[:, None],
                    axis=1
                )
            )[0]

            raise RuntimeError(
                "Negative variance allocations beyond numerical tolerance "
                f"for feature indices: {bad.tolist()}"
            )

        components = np.maximum(
            components,
            0.0
        )

        valid = (
            total_ss
            > 1e-20
        )

        fractions = np.full_like(
            components,
            np.nan
        )

        denom = components.sum(
            axis=1
        )

        fractions[valid] = (
            components[valid]
            / denom[valid, None]
        )

        np.testing.assert_allclose(
            fractions[valid].sum(
                axis=1
            ),
            1.0,
            atol=1e-12
        )

        return (
            components / self.n,
            fractions,
            total_ss / self.n
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        default=r"<TABULA_PROJECT_ROOT>"
    )

    parser.add_argument(
        "--outdir",
        default=None
    )

    args = parser.parse_args()

    PROJECT = Path(
        args.project
    ).resolve()

    CORR = (
        PROJECT
        / "corrected_full_analysis_v2"
    )

    FREEZE = (
        CORR
        / "00_freeze"
    )

    A2 = (
        CORR
        / "02_A2"
    )

    # Current corrected metadata
    META_PATH = (
        FREEZE
        / "observation_metadata_corrected_v3_FROZEN.tsv"
    )

    # Current frozen 24-feature matrix
    CLOCK24_PATH = (
        PROJECT
        / "figures_revised"
        / "clock_architecture_umap"
        / "source_data"
        / "clock_architecture_UMAP_clock_architecture_standardized_input.tsv"
    )

    # Exact corrected A2 support
    SUPPORT_PATH = (
        A2
        / "corrected_centroid_support.tsv"
    )

    if args.outdir is None:

        OUT = (
            CORR
            / "42_CORRECTED_24FEATURE_VARIANCE_PARTITION_20260921"
        )

    else:

        OUT = Path(
            args.outdir
        ).resolve()

    FIGDIR = OUT / "figures"
    SRCDIR = OUT / "source_data"
    AUDITDIR = OUT / "audit"

    for p in [
        OUT,
        FIGDIR,
        SRCDIR,
        AUDITDIR,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    required_files = [
        META_PATH,
        CLOCK24_PATH,
        SUPPORT_PATH,
    ]

    missing = [
        str(p)
        for p in required_files
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required frozen inputs:\n"
            + "\n".join(missing)
        )

    # ========================================================
    # LOAD CURRENT CORRECTED METADATA
    # ========================================================

    meta = pd.read_csv(
        META_PATH,
        sep="\t",
        low_memory=False
    )

    required_meta = [
        "group_id",
        "donor_id",
        "tissue",
        "priority_lineage_corrected",
    ]

    miss = [
        c
        for c in required_meta
        if c not in meta.columns
    ]

    if miss:
        raise RuntimeError(
            f"Corrected metadata missing: {miss}"
        )

    # IMPORTANT:
    # corrected mapping ONLY.
    # No fallback to old priority_lineage.
    meta["priority_lineage"] = (
        meta[
            "priority_lineage_corrected"
        ]
    )

    meta["group_id"] = (
        meta["group_id"]
        .astype(str)
    )

    meta["donor_id"] = (
        meta["donor_id"]
        .astype(str)
    )

    meta["tissue"] = (
        meta["tissue"]
        .astype(str)
    )

    meta["priority_lineage"] = (
        meta["priority_lineage"]
        .astype("string")
    )

    if meta["group_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate group_id in corrected metadata."
        )

    meta = meta.set_index(
        "group_id"
    )

    # ========================================================
    # LOAD EXACT CORRECTED A2 SUPPORT
    # ========================================================

    support = pd.read_csv(
        SUPPORT_PATH,
        sep="\t",
        low_memory=False
    )

    lineage_col = None

    for c in [
        "lineage",
        "priority_lineage_corrected",
        "priority_lineage",
        "major_lineage",
    ]:
        if c in support.columns:
            lineage_col = c
            break

    if lineage_col is None:
        raise RuntimeError(
            "Could not identify lineage column in "
            "corrected_centroid_support.tsv"
        )

    if "tissue" not in support.columns:
        raise RuntimeError(
            "corrected_centroid_support.tsv lacks tissue."
        )

    # Determine supported rows
    if "adequate_support" in support.columns:

        adequate = (
            support["adequate_support"]
            .astype(str)
            .str.lower()
            .isin([
                "true",
                "1",
                "yes",
                "t"
            ])
        )

        support_use = support[
            adequate
        ].copy()

    elif "n_donors" in support.columns:

        support_use = support[
            pd.to_numeric(
                support["n_donors"],
                errors="coerce"
            ).ge(3)
        ].copy()

    else:

        raise RuntimeError(
            "Support table has neither adequate_support nor n_donors."
        )

    support_keys = set(
        zip(
            support_use[
                lineage_col
            ].astype(str),

            support_use[
                "tissue"
            ].astype(str),
        )
    )

    # Use ONLY corrected five-lineage rows that occur in
    # the exact corrected A2 support.
    keep = np.array([
        (
            pd.notna(r["priority_lineage"])
            and
            (
                str(r["priority_lineage"]),
                str(r["tissue"])
            )
            in support_keys
        )
        for _, r in meta.iterrows()
    ])

    smeta = meta.loc[
        keep
    ].copy()

    if len(smeta) == 0:
        raise RuntimeError(
            "No rows survived corrected A2 support."
        )

    # ========================================================
    # LOAD FROZEN CLOCK FEATURE MATRIX
    # ========================================================

    clock = pd.read_csv(
        CLOCK24_PATH,
        sep="\t",
        float_precision="round_trip"
    )

    if "group_id" not in clock.columns:
        raise RuntimeError(
            "Frozen clock feature matrix lacks group_id."
        )

    feature_cols = [
        c
        for c in clock.columns
        if c != "group_id"
    ]

    if len(feature_cols) != 24:

        print("\nFEATURE COLUMNS FOUND:")
        for i, c in enumerate(
            feature_cols,
            start=1
        ):
            print(
                f"{i:02d}. {c}"
            )

        raise RuntimeError(
            "Expected exactly 24 frozen clock-associated features, "
            f"but found {len(feature_cols)}."
        )

    clock["group_id"] = (
        clock["group_id"]
        .astype(str)
    )

    if clock["group_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate group_id in CLOCK24."
        )

    clock = (
        clock
        .set_index("group_id")
    )

    missing_ids = (
        set(smeta.index)
        - set(clock.index)
    )

    if missing_ids:
        raise RuntimeError(
            "CLOCK24 missing supported metadata IDs: "
            f"{list(missing_ids)[:10]}"
        )

    clock = clock.loc[
        smeta.index,
        feature_cols
    ]

    Y = clock.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        Y
    ).all():
        raise RuntimeError(
            "CLOCK24 contains non-finite values."
        )

    # ========================================================
    # PARTITION
    # ========================================================

    model = Partition(
        smeta
    )

    components, fractions, total_variance = (
        model.fit(
            Y
        )
    )

    component_names_internal = [
        "f_lineage",
        "f_tissue",
        "f_donor",
        "f_lineage_tissue",
        "f_residual",
    ]

    result = pd.DataFrame(
        fractions,
        columns=component_names_internal
    )

    result.insert(
        0,
        "feature",
        feature_cols
    )

    result["display_feature"] = (
        result["feature"]
        .map(pretty_feature)
    )

    result["feature_type"] = np.where(
        result["feature"]
        .astype(str)
        .str.startswith("expr_"),
        "Gene expression",
        "Derived clock feature"
    )

    result["total_supported_subset_variance"] = (
        total_variance
    )

    result["context_fraction"] = (
        result["f_tissue"]
        +
        result[
            "f_lineage_tissue"
        ]
    )

    result["fraction_sum"] = (
        result[
            component_names_internal
        ].sum(axis=1)
    )

    # exact numerical sanity
    np.testing.assert_allclose(
        result[
            "fraction_sum"
        ].to_numpy(),
        1.0,
        atol=1e-12
    )

    # ========================================================
    # ORDER:
    # canonical clock genes first;
    # derived features afterwards in frozen source order.
    # ========================================================

    canonical_rank = {
        g: i
        for i, g in enumerate(
            CANONICAL_GENES
        )
    }

    def row_order(row):

        f = str(
            row["feature"]
        )

        label = str(
            row["display_feature"]
        )

        if f.startswith("expr_"):

            gene = f[5:]

            return (
                0,
                canonical_rank.get(
                    gene,
                    999
                )
            )

        return (
            1,
            feature_cols.index(f)
        )

    result["_order"] = [
        row_order(r)
        for _, r in result.iterrows()
    ]

    result["_group"] = [
        x[0]
        for x in result["_order"]
    ]

    result["_rank"] = [
        x[1]
        for x in result["_order"]
    ]

    result = (
        result
        .sort_values(
            [
                "_group",
                "_rank"
            ],
            kind="stable"
        )
        .drop(
            columns=[
                "_order",
                "_group",
                "_rank"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    mean_summary = pd.Series({
        "Lineage":
            result["f_lineage"].mean(),

        "Tissue":
            result["f_tissue"].mean(),

        "Lineage × tissue":
            result[
                "f_lineage_tissue"
            ].mean(),

        "Donor":
            result["f_donor"].mean(),

        "Residual":
            result["f_residual"].mean(),
    })

    median_summary = pd.Series({
        "Lineage":
            result["f_lineage"].median(),

        "Tissue":
            result["f_tissue"].median(),

        "Lineage × tissue":
            result[
                "f_lineage_tissue"
            ].median(),

        "Donor":
            result["f_donor"].median(),

        "Residual":
            result["f_residual"].median(),
    })

    summary = pd.DataFrame({
        "component":
            mean_summary.index,

        "mean_fraction":
            mean_summary.values,

        "median_fraction":
            median_summary.values,
    })

    # ========================================================
    # SAVE SOURCE DATA
    # ========================================================

    result.to_csv(
        SRCDIR
        / "FIG2C_corrected_24feature_variance_partition.csv",
        index=False
    )

    summary.to_csv(
        SRCDIR
        / "FIG2C_corrected_24feature_variance_summary.csv",
        index=False
    )

    # ========================================================
    # FIGURE
    # ========================================================

    plot_order = [
        (
            "f_lineage",
            "Lineage"
        ),
        (
            "f_tissue",
            "Tissue"
        ),
        (
            "f_lineage_tissue",
            "Lineage × tissue"
        ),
        (
            "f_donor",
            "Donor"
        ),
    ]

    # Tall enough for 24 rows but still manuscript-friendly.
    fig, ax = plt.subplots(
        figsize=(7.0, 7.7)
    )

    y = np.arange(
        len(result)
    )

    left = np.zeros(
        len(result)
    )

    for col, label in plot_order:

        vals = (
            result[col]
            .to_numpy(float)
            * 100.0
        )

        ax.barh(
            y,
            vals,
            left=left,
            height=0.74,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.35,
            label=label
        )

        left += vals

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        result[
            "display_feature"
        ],
        fontsize=6.7
    )

    ax.invert_yaxis()

    ax.set_xlim(
        0,
        100
    )

    ax.set_xlabel(
        "Allocated structured variation (%)"
    )

    ax.set_title(
        "Variance allocation across clock-associated features",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=16
    )

    ax.legend(
        frameon=False,
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0.12, 1.055),
        columnspacing=1.5,
        handlelength=1.4
    )


    # --------------------------------------------------------
    # Dynamic summary line
    # --------------------------------------------------------

    summary_text = (
        "Mean across 24 features:  "
        f"Lineage {100*mean_summary['Lineage']:.1f}%   |   "
        f"Tissue {100*mean_summary['Tissue']:.1f}%   |   "
        f"Lineage × tissue {100*mean_summary['Lineage × tissue']:.1f}%   |   "
        f"Donor {100*mean_summary['Donor']:.1f}%   |   "
        f"Residual {100*mean_summary['Residual']:.1f}%"
    )

    ax.text(
        0.0,
        1.01,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#444444"
    )

    # --------------------------------------------------------
    # Gene / derived-feature separator
    # --------------------------------------------------------

    n_gene_rows = int(
        result[
            "feature_type"
        ]
        .eq(
            "Gene expression"
        )
        .sum()
    )

    if (
        n_gene_rows > 0
        and
        n_gene_rows < len(result)
    ):
        ax.axhline(
            n_gene_rows - 0.5,
            color="#777777",
            linewidth=0.65
        )

    # --------------------------------------------------------
    # No gridlines
    # --------------------------------------------------------

    ax.grid(
        False
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            1.105
        ),
        columnspacing=1.3,
        handlelength=1.4,
    )

    fig.subplots_adjust(
        left=0.28,
        right=0.98,
        bottom=0.09,
        top=0.86
    )

    save_fig(
        fig,
        FIGDIR
        / "FIG2C_corrected_24feature_variance_partition"
    )

    # ========================================================
    # AUDIT
    # ========================================================

    audit = {
        "method": (
            "Finite-design hierarchical Shapley sums of squares. "
            "Lineage, tissue, and donor additive fitted SS are allocated "
            "by averaging incremental SS over all six insertion orders. "
            "Observed lineage×tissue interaction enters after the additive "
            "model. Residual is the remaining SS after the observed-pair "
            "+ donor model."
        ),

        "interpretation": (
            "Allocated observed static feature variation; "
            "not REML variance components and not causal effects."
        ),

        "metadata":
            str(META_PATH),

        "clock24":
            str(CLOCK24_PATH),

        "corrected_A2_support":
            str(SUPPORT_PATH),

        "mapping":
            "priority_lineage_corrected ONLY",

        "n_features":
            int(len(result)),

        "feature_names":
            result[
                "feature"
            ].tolist(),

        "diagnostics":
            model.diagnostics,

        "mean_fraction":
            {
                k: float(v)
                for k, v in
                mean_summary.items()
            },

        "median_fraction":
            {
                k: float(v)
                for k, v in
                median_summary.items()
            },

        "all_feature_fraction_sums_equal_one":
            bool(
                np.allclose(
                    result[
                        "fraction_sum"
                    ],
                    1.0,
                    atol=1e-12
                )
            ),
    }

    with open(
        AUDITDIR
        / "FIG2C_variance_partition_audit.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # TERMINAL
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CORRECTED 24-FEATURE VARIANCE PARTITION COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        f"Supported rows: {len(smeta)}"
    )

    print(
        "Lineages:",
        sorted(
            smeta[
                "priority_lineage"
            ]
            .astype(str)
            .unique()
        )
    )

    print(
        f"Supported lineage×tissue groups: {len(support_keys)}"
    )

    print(
        f"Clock-associated features: {len(result)}"
    )

    print(
        "\nMean fractions across 24 features:"
    )

    for k, v in mean_summary.items():

        print(
            f"  {k:18s}: {100*v:6.2f}%"
        )

    print(
        "\nMedian fractions across 24 features:"
    )

    for k, v in median_summary.items():

        print(
            f"  {k:18s}: {100*v:6.2f}%"
        )

    print(
        "\nFeature-level fractions:"
    )

    show = result[
        [
            "display_feature",
            "f_lineage",
            "f_tissue",
            "f_lineage_tissue",
            "f_donor",
            "f_residual",
        ]
    ].copy()

    for c in [
        "f_lineage",
        "f_tissue",
        "f_lineage_tissue",
        "f_donor",
        "f_residual",
    ]:
        show[c] = (
            100
            * show[c]
        ).round(1)

    print(
        show.to_string(
            index=False
        )
    )

    print(
        "\nFRACTIONS_SUM_TO_ONE:",
        np.allclose(
            result[
                "fraction_sum"
            ],
            1.0,
            atol=1e-12
        )
    )

    print(
        "CORRECTED_MAPPING_ONLY: YES"
    )

    print(
        "OLD_BROAD_MAPPING_USED: NO"
    )

    print(
        f"\nFIGURE:\n{FIGDIR}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()