# -*- coding: utf-8 -*-
r"""
73_GSE292219_time_resolved_validation.py

FULL TIME-RESOLVED EXTERNAL VALIDATION FOR GSE292219
====================================================

This script starts from the frozen high-confidence cell annotations produced by:

    72_GSE292219_annotation_audit.py

It DOES NOT re-cluster cells and DOES NOT re-annotate lineages.

Frozen target lineages
----------------------
- B cell
- T cell
- Macrophage/Monocyte
- Fibroblast
- Endothelial

Primary goals
-------------
1. Build memory-safe lineage x ZT mean normalized expression profiles.
2. Fit descriptive 24-h harmonics for Clock17 genes.
3. Quantify lineage-level temporal clock architecture:
   - amplitude
   - relative amplitude
   - R²
   - phase
   - phase coherence
   - descriptive rhythmic burden
4. Compare temporal clock architecture across the five lineages.
5. Validate the six circadian arms as a SECONDARY analysis.
6. Optionally validate frozen Fig3D pathways, but ONLY if gene membership can
   be recovered from local frozen source data or a local GMT file. No web fetch,
   no post-hoc pathway discovery.
7. Produce publication-oriented tables and separate figures.

Important interpretation
------------------------
Each ZT is represented by one aggregate sample, not independent biological
replicates per ZT. Harmonic P values are therefore DESCRIPTIVE ONLY.

The primary evidence should be:
- amplitude
- phase
- R²
- phase coherence
- temporal architecture differences
- replication with the independent GSE288579 dataset later

Do NOT interpret these six aggregate points as formal replicated rhythm testing.

Time labels
-----------
The deposited sample filenames/titles are:
ZT3, ZT7, ZT11, ZT15, ZT19, ZT23.

These labels are used exactly as deposited. A metadata/protocol conflict noted
elsewhere (ZT4/8/12/16/20/24) is NOT silently reconciled here.

No download code is included.

Gene alias note
---------------
GSE292219 stores BMAL1 as gene_name="Bmal1". The frozen Clock17/six-arm lists
use "Arntl". The workflow resolves ARNTL -> Bmal1 during gene lookup.
BMAL2/ARNTL2 is not substituted.
"""

from pathlib import Path
import re
import json
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc


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

ANN72_ROOT = (
    ANALYSIS_ROOT
    / "72_GSE292219_ANNOTATION_AUDIT"
)

ANN72_TABLEDIR = ANN72_ROOT / "01_tables"

OUTDIR = (
    ANALYSIS_ROOT
    / "73_GSE292219_TIME_RESOLVED_VALIDATION"
)

QCDIR = OUTDIR / "01_qc"
EXPRDIR = OUTDIR / "02_lineage_time_expression"
CLOCKDIR = OUTDIR / "03_clock17"
ARCHDIR = OUTDIR / "04_temporal_architecture"
ARMDIR = OUTDIR / "05_six_arm_secondary"
PATHWAYDIR = OUTDIR / "06_frozen_pathway_validation"
FIGDIR = OUTDIR / "07_figures"
REPORTDIR = OUTDIR / "08_report"

for p in [
    OUTDIR,
    QCDIR,
    EXPRDIR,
    CLOCKDIR,
    ARCHDIR,
    ARMDIR,
    PATHWAYDIR,
    FIGDIR,
    REPORTDIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True
    )

SAMPLES = {
    "GSM8853297": {
        "ZT": 3,
        "file": "GSM8853297_ZT3.h5ad",
    },
    "GSM8853298": {
        "ZT": 7,
        "file": "GSM8853298_ZT7.h5ad",
    },
    "GSM8853299": {
        "ZT": 11,
        "file": "GSM8853299_ZT11.h5ad",
    },
    "GSM8853300": {
        "ZT": 15,
        "file": "GSM8853300_ZT15.h5ad",
    },
    "GSM8853301": {
        "ZT": 19,
        "file": "GSM8853301_ZT19.h5ad",
    },
    "GSM8853302": {
        "ZT": 23,
        "file": "GSM8853302_ZT23.h5ad",
    },
}

TARGET_LINEAGES = [
    "B cell",
    "T cell",
    "Macrophage/Monocyte",
    "Fibroblast",
    "Endothelial",
]

ZT_ORDER = [
    3,
    7,
    11,
    15,
    19,
    23,
]

MIN_CELLS_PER_LINEAGE_ZT = 10

CLOCK17 = [
    "Arntl",
    "Clock",
    "Npas2",
    "Per1",
    "Per2",
    "Per3",
    "Cry1",
    "Cry2",
    "Rora",
    "Rorb",
    "Rorc",
    "Nr1d1",
    "Nr1d2",
    "Dbp",
    "Tef",
    "Hlf",
    "Nfil3",
]

SIX_ARMS = {
    "EBOX_POS": [
        "Arntl",
        "Clock",
        "Npas2",
    ],
    "EBOX_NEG": [
        "Per1",
        "Per2",
        "Per3",
        "Cry1",
        "Cry2",
    ],
    "RORE_ACT": [
        "Rora",
        "Rorb",
        "Rorc",
    ],
    "RORE_REP": [
        "Nr1d1",
        "Nr1d2",
    ],
    "DBOX_ACT": [
        "Dbp",
        "Tef",
        "Hlf",
    ],
    "DBOX_REP": [
        "Nfil3",
    ],
}

LINEAGE_COLORS = {
    "B cell": "#4C78A8",
    "T cell": "#E45756",
    "Macrophage/Monocyte": "#59A14F",
    "Fibroblast": "#F28E2B",
    "Endothelial": "#9467BD",
}

ARM_COLORS = {
    "EBOX_POS": "#4C78A8",
    "EBOX_NEG": "#E45756",
    "RORE_ACT": "#59A14F",
    "RORE_REP": "#F28E2B",
    "DBOX_ACT": "#9467BD",
    "DBOX_REP": "#8C564B",
}

# Descriptive burden thresholds.
# Primary summary uses R² >= 0.50.
R2_THRESHOLDS = [
    0.30,
    0.50,
    0.70,
]

# Fig3 frozen source.
FROZEN_FIG3D = (
    ANALYSIS_ROOT
    / "source_data"
    / "Fig3D_pathway_context_partial_assoc_source.tsv"
)

# Optional static sources are only audited, not forced into the analysis.
FROZEN_FIG1C = (
    ANALYSIS_ROOT
    / "source_data"
    / "Fig1C_core_clock_heatmap_source.tsv"
)

STATIC_SIX_ARM = (
    ANALYSIS_ROOT
    / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930"
    / "source_data"
    / "DONOR_TISSUE_LINEAGE_SCORES.csv"
)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 1. GENERAL HELPERS
# ============================================================

def log(msg):
    print(
        msg,
        flush=True
    )


def sanitize(x):
    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        str(x)
    ).strip("_")


def savefig(
    fig,
    stem
):
    fig.tight_layout()

    fig.savefig(
        FIGDIR / f"{stem}.png",
        dpi=400,
        bbox_inches="tight"
    )

    fig.savefig(
        FIGDIR / f"{stem}.pdf",
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


def gene_lookup(
    names
):
    lookup = {
        str(
            g
        ).upper():
            str(
                g
            )
        for g
        in names
    }

    # --------------------------------------------------------
    # Circadian alias normalization
    # --------------------------------------------------------
    # GSE292219 stores the BMAL1/ARNTL gene as "Bmal1".
    # The frozen Clock17 and six-arm definitions use "Arntl".
    # Therefore resolve ARNTL -> Bmal1 at lookup time.
    if (
        "ARNTL"
        not in lookup
        and
        "BMAL1"
        in lookup
    ):
        lookup[
            "ARNTL"
        ] = lookup[
            "BMAL1"
        ]

    # BMAL2/ARNTL2 is a different paralog.
    # Never substitute BMAL2 for ARNTL/BMAL1.

    return lookup



def set_gene_names(
    a
):
    if (
        "gene_name"
        not in a.var.columns
    ):
        raise RuntimeError(
            "var['gene_name'] is missing"
        )

    old = (
        a.var_names
        .astype(
            str
        )
        .copy()
    )

    symbols = (
        a.var[
            "gene_name"
        ]
        .astype(
            str
        )
        .copy()
    )

    bad = (
        symbols.isna()
        |
        symbols.eq(
            ""
        )
        |
        symbols.eq(
            "nan"
        )
    )

    symbols.loc[
        bad
    ] = old[
        bad
    ]

    a.var_names = pd.Index(
        symbols
    )

    a.var_names_make_unique()

    return a


def get_norm10k_matrix(
    a
):
    if (
        "norm10k"
        not in a.layers
    ):
        raise RuntimeError(
            "layer['norm10k'] is missing"
        )

    return a.layers[
        "norm10k"
    ]


def mean_rows(
    X,
    row_idx
):
    if (
        len(
            row_idx
        )
        ==
        0
    ):
        return None

    sub = X[
        row_idx,
        :
    ]

    if sp.issparse(
        sub
    ):
        return np.asarray(
            sub.mean(
                axis=0
            )
        ).ravel()

    return np.asarray(
        sub
    ).mean(
        axis=0
    )


def zscore_safe(
    x
):
    x = np.asarray(
        x,
        dtype=float
    )

    m = np.nanmean(
        x
    )

    s = np.nanstd(
        x,
        ddof=0
    )

    if (
        not np.isfinite(
            s
        )
        or
        s
        <=
        1e-12
    ):
        return np.zeros(
            len(
                x
            ),
            dtype=float
        )

    return (
        x
        -
        m
    ) / s


def circular_diff_hours(
    a,
    b
):
    d = (
        float(
            a
        )
        -
        float(
            b
        )
        +
        12.0
    ) % 24.0 - 12.0

    return d


def circular_distance_hours(
    a,
    b
):
    return abs(
        circular_diff_hours(
            a,
            b
        )
    )


def circular_resultant(
    phases_h,
    weights=None
):
    phases_h = np.asarray(
        phases_h,
        dtype=float
    )

    ok = np.isfinite(
        phases_h
    )

    if weights is None:
        weights = np.ones(
            len(
                phases_h
            ),
            dtype=float
        )
    else:
        weights = np.asarray(
            weights,
            dtype=float
        )

        ok = (
            ok
            &
            np.isfinite(
                weights
            )
            &
            (
                weights
                >=
                0
            )
        )

    if (
        ok.sum()
        ==
        0
    ):
        return (
            np.nan,
            np.nan
        )

    theta = (
        phases_h[
            ok
        ]
        /
        24.0
        *
        2.0
        *
        np.pi
    )

    w = weights[
        ok
    ]

    if (
        np.sum(
            w
        )
        <=
        0
    ):
        w = np.ones_like(
            theta
        )

    c = np.sum(
        w
        *
        np.cos(
            theta
        )
    ) / np.sum(
        w
    )

    s = np.sum(
        w
        *
        np.sin(
            theta
        )
    ) / np.sum(
        w
    )

    R = float(
        np.sqrt(
            c**2
            +
            s**2
        )
    )

    mean_theta = (
        np.arctan2(
            s,
            c
        )
        %
        (
            2.0
            *
            np.pi
        )
    )

    mean_h = float(
        mean_theta
        *
        24.0
        /
        (
            2.0
            *
            np.pi
        )
    )

    return (
        R,
        mean_h
    )


# ============================================================
# 2. HARMONIC MODEL
# ============================================================

def harmonic_fit(
    t,
    y
):
    t = np.asarray(
        t,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    ok = (
        np.isfinite(
            t
        )
        &
        np.isfinite(
            y
        )
    )

    t = t[
        ok
    ]

    y = y[
        ok
    ]

    if (
        len(
            y
        )
        <
        4
        or
        np.nanstd(
            y
        )
        <=
        1e-12
    ):
        return {
            "mean_expression":
                float(
                    np.nanmean(
                        y
                    )
                )
                if len(
                    y
                )
                else np.nan,

            "amplitude":
                np.nan,

            "relative_amplitude":
                np.nan,

            "phase_h":
                np.nan,

            "r2":
                np.nan,

            "p_harmonic_descriptive":
                np.nan,

            "n_timepoints":
                int(
                    len(
                        y
                    )
                ),
        }

    w = (
        2.0
        *
        np.pi
        /
        24.0
    )

    X = np.column_stack(
        [
            np.ones(
                len(
                    t
                )
            ),
            np.cos(
                w
                *
                t
            ),
            np.sin(
                w
                *
                t
            ),
        ]
    )

    model = sm.OLS(
        y,
        X
    ).fit()

    beta_cos = float(
        model.params[
            1
        ]
    )

    beta_sin = float(
        model.params[
            2
        ]
    )

    amp = float(
        np.sqrt(
            beta_cos**2
            +
            beta_sin**2
        )
    )

    # y = mu + A cos(wt - phi)
    # beta_cos = A cos(phi)
    # beta_sin = A sin(phi)
    phase_h = float(
        (
            np.arctan2(
                beta_sin,
                beta_cos
            )
            %
            (
                2.0
                *
                np.pi
            )
        )
        *
        24.0
        /
        (
            2.0
            *
            np.pi
        )
    )

    mean_y = float(
        np.nanmean(
            y
        )
    )

    if (
        np.isfinite(
            mean_y
        )
        and
        abs(
            mean_y
        )
        >
        1e-12
    ):
        rel_amp = float(
            amp
            /
            abs(
                mean_y
            )
        )
    else:
        rel_amp = np.nan

    try:
        # Explicit matrix restriction:
        # H0: beta_cos = beta_sin = 0
        R = np.array([
            [
                0.0,
                1.0,
                0.0
            ],
            [
                0.0,
                0.0,
                1.0
            ],
        ])

        p = float(
            np.asarray(
                model.f_test(
                    R
                ).pvalue
            )
            .ravel()[
                0
            ]
        )

    except Exception:
        p = np.nan

    return {
        "mean_expression":
            mean_y,

        "amplitude":
            amp,

        "relative_amplitude":
            rel_amp,

        "phase_h":
            phase_h,

        "r2":
            float(
                model.rsquared
            ),

        "p_harmonic_descriptive":
            p,

        "n_timepoints":
            int(
                len(
                    y
                )
            ),
    }


# ============================================================
# 3. LOAD FROZEN 72 ANNOTATIONS
# ============================================================

def load_annotations_72():
    combined = (
        ANN72_TABLEDIR
        /
        "GSE292219_all_cells_annotation_audit.csv.gz"
    )

    if not combined.exists():
        raise FileNotFoundError(
            "Required frozen annotation file is missing:\n"
            f"{combined}\n"
            "Run 72_GSE292219_annotation_audit.py first."
        )

    ann = pd.read_csv(
        combined
    )

    required = {
        "cell_id",
        "sample_id",
        "ZT",
        "high_confidence_lineage",
    }

    missing = (
        required
        -
        set(
            ann.columns
        )
    )

    if missing:
        raise RuntimeError(
            "72 annotation table is missing columns: "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )

    ann = ann[
        ann[
            "high_confidence_lineage"
        ]
        .isin(
            TARGET_LINEAGES
        )
    ].copy()

    counts = (
        ann.groupby(
            [
                "sample_id",
                "ZT",
                "high_confidence_lineage"
            ],
            observed=True
        )
        .size()
        .reset_index(
            name=
                "n_cells"
        )
    )

    counts.to_csv(
        QCDIR
        /
        "73_frozen_high_confidence_cell_counts.csv",
        index=False
    )

    log(
        "\n[frozen annotation counts]"
    )

    log(
        counts.to_string(
            index=False
        )
    )

    return ann


# ============================================================
# 4. MEMORY-SAFE LINEAGE x ZT AGGREGATION
# ============================================================

def aggregate_expression(
    ann
):
    mean_series = []

    metadata_rows = []

    gene_sets = []

    for gsm, cfg in (
        SAMPLES.items()
    ):
        path = (
            RAWDIR
            /
            cfg[
                "file"
            ]
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        log(
            "\n"
            +
            "=" * 84
        )

        log(
            f"[aggregate] {gsm} "
            f"ZT{cfg['ZT']} "
            f"{cfg['file']}"
        )

        a = sc.read_h5ad(
            path
        )

        a = set_gene_names(
            a
        )

        X = get_norm10k_matrix(
            a
        )

        gene_names = (
            a.var_names
            .astype(
                str
            )
            .tolist()
        )

        gene_sets.append(
            set(
                gene_names
            )
        )

        sample_ann = (
            ann[
                ann[
                    "sample_id"
                ]
                ==
                gsm
            ]
            .copy()
        )

        obs_indexer = pd.Series(
            np.arange(
                a.n_obs,
                dtype=int
            ),
            index=
                a.obs_names
                .astype(
                    str
                )
        )

        for lineage in (
            TARGET_LINEAGES
        ):
            ids = (
                sample_ann.loc[
                    sample_ann[
                        "high_confidence_lineage"
                    ]
                    ==
                    lineage,
                    "cell_id"
                ]
                .astype(
                    str
                )
                .tolist()
            )

            idx = (
                obs_indexer.reindex(
                    ids
                )
                .dropna()
                .astype(
                    int
                )
                .values
            )

            n_cells = int(
                len(
                    idx
                )
            )

            if (
                n_cells
                <
                MIN_CELLS_PER_LINEAGE_ZT
            ):
                raise RuntimeError(
                    f"{gsm} ZT{cfg['ZT']} {lineage}: "
                    f"only {n_cells} high-confidence cells; "
                    "frozen five-lineage coverage gate failed."
                )

            vec = mean_rows(
                X,
                idx
            )

            s = pd.Series(
                vec,
                index=
                    gene_names,
                dtype=float,
                name=
                    (
                        gsm,
                        cfg[
                            "ZT"
                        ],
                        lineage
                    )
            )

            mean_series.append(
                s
            )

            metadata_rows.append({
                "sample_id":
                    gsm,

                "ZT":
                    int(
                        cfg[
                            "ZT"
                        ]
                    ),

                "lineage":
                    lineage,

                "n_cells":
                    n_cells,
            })

            log(
                f"  {lineage}: "
                f"n={n_cells}"
            )

        del a
        del X

    common = set.intersection(
        *gene_sets
    )

    common = sorted(
        common
    )

    log(
        f"\n[aggregate] common genes across "
        f"6 H5ADs: {len(common):,}"
    )

    expr_rows = []

    for s in (
        mean_series
    ):
        expr_rows.append(
            s.reindex(
                common
            ).values
        )

    meta = pd.DataFrame(
        metadata_rows
    )

    expr = pd.DataFrame(
        expr_rows,
        columns=
            common
    )

    pb = pd.concat(
        [
            meta.reset_index(
                drop=True
            ),
            expr.reset_index(
                drop=True
            )
        ],
        axis=1
    )

    pb = (
        pb.sort_values(
            [
                "lineage",
                "ZT"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pb.to_pickle(
        EXPRDIR
        /
        "GSE292219_highconf_lineage_ZT_mean_norm10k.pkl"
    )

    pb.to_csv(
        EXPRDIR
        /
        "GSE292219_highconf_lineage_ZT_mean_norm10k.csv.gz",
        index=False,
        compression="gzip"
    )

    meta.to_csv(
        EXPRDIR
        /
        "GSE292219_highconf_lineage_ZT_metadata.csv",
        index=False
    )

    return pb


# ============================================================
# 5. COVERAGE FIGURE
# ============================================================

def plot_cell_counts(
    pb
):
    count_mat = (
        pb.pivot(
            index=
                "lineage",
            columns=
                "ZT",
            values=
                "n_cells"
        )
        .reindex(
            TARGET_LINEAGES
        )
        .reindex(
            columns=
                ZT_ORDER
        )
    )

    count_mat.to_csv(
        QCDIR
        /
        "73_highconf_lineage_ZT_count_matrix.csv"
    )

    fig, ax = plt.subplots(
        figsize=(
            6.5,
            3.4
        )
    )

    im = ax.imshow(
        np.log10(
            count_mat.values
            +
            1
        ),
        aspect="auto"
    )

    ax.set_xticks(
        np.arange(
            len(
                ZT_ORDER
            )
        )
    )

    ax.set_xticklabels(
        [
            f"ZT{x}"
            for x
            in ZT_ORDER
        ]
    )

    ax.set_yticks(
        np.arange(
            len(
                TARGET_LINEAGES
            )
        )
    )

    ax.set_yticklabels(
        TARGET_LINEAGES
    )

    for i in range(
        count_mat.shape[
            0
        ]
    ):
        for j in range(
            count_mat.shape[
                1
            ]
        ):
            ax.text(
                j,
                i,
                str(
                    int(
                        count_mat.iloc[
                            i,
                            j
                        ]
                    )
                ),
                ha="center",
                va="center",
                fontsize=8
            )

    ax.set_title(
        "High-confidence cells per lineage and ZT"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.02
    )

    cbar.set_label(
        "log10(n + 1)"
    )

    savefig(
        fig,
        "73A_HIGHCONF_CELL_COUNTS"
    )


# ============================================================
# 6. CLOCK17 EXTRACTION
# ============================================================

def clock17_available(
    pb
):
    gm = gene_lookup(
        pb.columns
    )

    rows = []

    for g in (
        CLOCK17
    ):
        rows.append({
            "gene":
                g,

            "available":
                g.upper()
                in gm,

            "column":
                gm.get(
                    g.upper(),
                    ""
                ),
        })

    audit = pd.DataFrame(
        rows
    )

    audit.to_csv(
        CLOCKDIR
        /
        "73_Clock17_gene_availability.csv",
        index=False
    )

    missing = (
        audit.loc[
            ~audit[
                "available"
            ],
            "gene"
        ]
        .tolist()
    )

    log(
        "\n[Clock17 availability]"
    )

    log(
        audit.to_string(
            index=False
        )
    )

    if missing:
        log(
            "[Clock17 warning] missing: "
            +
            ", ".join(
                missing
            )
        )

        log(
            "[Clock17 note] Missing genes are skipped; "
            "the analysis continues with the available Clock17 members."
        )

    return gm


# ============================================================
# 7. CLOCK17 GENE-LEVEL HARMONICS
# ============================================================

def fit_clock17(
    pb,
    gm
):
    rows = []

    for lineage in (
        TARGET_LINEAGES
    ):
        d = (
            pb[
                pb[
                    "lineage"
                ]
                ==
                lineage
            ]
            .sort_values(
                "ZT"
            )
        )

        for gene in (
            CLOCK17
        ):
            if (
                gene.upper()
                not in gm
            ):
                continue

            col = gm[
                gene.upper()
            ]

            fit = harmonic_fit(
                d[
                    "ZT"
                ].values,
                d[
                    col
                ].values
            )

            rows.append({
                "lineage":
                    lineage,

                "gene":
                    gene,

                **fit,
            })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        CLOCKDIR
        /
        "73_Clock17_gene_harmonics.csv",
        index=False
    )

    return out


# ============================================================
# 8. CLOCK17 TEMPORAL EXPRESSION HEATMAPS
# ============================================================

def make_clock17_zmatrix(
    pb,
    gm,
    lineage
):
    d = (
        pb[
            pb[
                "lineage"
            ]
            ==
            lineage
        ]
        .sort_values(
            "ZT"
        )
    )

    mat = []

    genes = []

    for gene in (
        CLOCK17
    ):
        if (
            gene.upper()
            not in gm
        ):
            continue

        col = gm[
            gene.upper()
        ]

        v = d[
            col
        ].values.astype(
            float
        )

        mat.append(
            zscore_safe(
                v
            )
        )

        genes.append(
            gene
        )

    z = pd.DataFrame(
        mat,
        index=
            genes,
        columns=
            [
                f"ZT{x}"
                for x
                in d[
                    "ZT"
                ].astype(
                    int
                )
            ]
    )

    z.to_csv(
        CLOCKDIR
        /
        f"73_Clock17_zscore_{sanitize(lineage)}.csv"
    )

    return z


def plot_clock17_heatmaps(
    pb,
    gm
):
    for lineage in (
        TARGET_LINEAGES
    ):
        z = make_clock17_zmatrix(
            pb,
            gm,
            lineage
        )

        fig, ax = plt.subplots(
            figsize=(
                5.2,
                6.4
            )
        )

        im = ax.imshow(
            z.values,
            aspect="auto",
            vmin=-2.0,
            vmax=2.0
        )

        ax.set_xticks(
            np.arange(
                z.shape[
                    1
                ]
            )
        )

        ax.set_xticklabels(
            z.columns,
            rotation=0
        )

        ax.set_yticks(
            np.arange(
                z.shape[
                    0
                ]
            )
        )

        ax.set_yticklabels(
            z.index
        )

        ax.set_title(
            f"{lineage} — Clock17 temporal expression"
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.035,
            pad=0.02
        )

        cbar.set_label(
            "Within-gene Z score across ZT"
        )

        savefig(
            fig,
            "73B_CLOCK17_HEATMAP_"
            +
            sanitize(
                lineage
            )
        )


# ============================================================
# 9. LINEAGE-LEVEL CLOCK ARCHITECTURE
# ============================================================

def summarize_lineage_architecture(
    clock_fit
):
    rows = []

    for lineage in (
        TARGET_LINEAGES
    ):
        d = (
            clock_fit[
                clock_fit[
                    "lineage"
                ]
                ==
                lineage
            ]
            .copy()
        )

        usable = d[
            np.isfinite(
                d[
                    "phase_h"
                ]
            )
        ].copy()

        R_unweighted, mean_phase = (
            circular_resultant(
                usable[
                    "phase_h"
                ].values
            )
        )

        weights = (
            usable[
                "amplitude"
            ]
            .fillna(
                0
            )
            .values
            *
            usable[
                "r2"
            ]
            .fillna(
                0
            )
            .clip(
                lower=0
            )
            .values
        )

        R_weighted, mean_phase_weighted = (
            circular_resultant(
                usable[
                    "phase_h"
                ].values,
                weights=
                    weights
            )
        )

        row = {
            "lineage":
                lineage,

            "n_clock_genes":
                int(
                    len(
                        d
                    )
                ),

            "median_amplitude":
                float(
                    d[
                        "amplitude"
                    ].median()
                ),

            "median_relative_amplitude":
                float(
                    d[
                        "relative_amplitude"
                    ].replace(
                        [
                            np.inf,
                            -np.inf
                        ],
                        np.nan
                    ).median()
                ),

            "median_r2":
                float(
                    d[
                        "r2"
                    ].median()
                ),

            "mean_r2":
                float(
                    d[
                        "r2"
                    ].mean()
                ),

            "phase_coherence_R":
                R_unweighted,

            "circular_mean_phase_h":
                mean_phase,

            "weighted_phase_coherence_R":
                R_weighted,

            "weighted_circular_mean_phase_h":
                mean_phase_weighted,
        }

        for thr in (
            R2_THRESHOLDS
        ):
            row[
                f"burden_R2_ge_{thr:.2f}"
            ] = float(
                np.mean(
                    d[
                        "r2"
                    ].values
                    >=
                    thr
                )
            )

        rows.append(
            row
        )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        ARCHDIR
        /
        "73_lineage_clock17_temporal_architecture.csv",
        index=False
    )

    return out


def plot_architecture_summary(
    arch
):
    metrics = [
        (
            "median_r2",
            "Median Clock17 R²"
        ),
        (
            "median_relative_amplitude",
            "Median relative amplitude"
        ),
        (
            "weighted_phase_coherence_R",
            "Weighted phase coherence R"
        ),
        (
            "burden_R2_ge_0.50",
            "Clock17 burden (R² ≥ 0.50)"
        ),
    ]

    for col, label in (
        metrics
    ):
        if (
            col
            not in arch.columns
        ):
            continue

        fig, ax = plt.subplots(
            figsize=(
                6.0,
                3.7
            )
        )

        x = np.arange(
            len(
                TARGET_LINEAGES
            )
        )

        vals = (
            arch
            .set_index(
                "lineage"
            )
            .reindex(
                TARGET_LINEAGES
            )[
                col
            ]
            .values
        )

        colors = [
            LINEAGE_COLORS[
                l
            ]
            for l
            in TARGET_LINEAGES
        ]

        ax.bar(
            x,
            vals,
            color=
                colors
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            TARGET_LINEAGES,
            rotation=35,
            ha="right"
        )

        ax.set_ylabel(
            label
        )

        ax.set_title(
            label
        )

        savefig(
            fig,
            "73C_ARCHITECTURE_"
            +
            sanitize(
                col
            )
        )


# ============================================================
# 10. CROSS-LINEAGE TEMPORAL ARCHITECTURE
# ============================================================

def cross_lineage_similarity(
    pb,
    gm,
    clock_fit
):
    trajectory_vectors = {}

    for lineage in (
        TARGET_LINEAGES
    ):
        d = (
            pb[
                pb[
                    "lineage"
                ]
                ==
                lineage
            ]
            .sort_values(
                "ZT"
            )
        )

        pieces = []

        for gene in (
            CLOCK17
        ):
            if (
                gene.upper()
                not in gm
            ):
                continue

            col = gm[
                gene.upper()
            ]

            pieces.extend(
                zscore_safe(
                    d[
                        col
                    ].values.astype(
                        float
                    )
                )
            )

        trajectory_vectors[
            lineage
        ] = np.asarray(
            pieces,
            dtype=float
        )

    rows = []

    for l1, l2 in combinations(
        TARGET_LINEAGES,
        2
    ):
        v1 = trajectory_vectors[
            l1
        ]

        v2 = trajectory_vectors[
            l2
        ]

        rho = spearmanr(
            v1,
            v2
        ).statistic

        p1 = (
            clock_fit[
                clock_fit[
                    "lineage"
                ]
                ==
                l1
            ]
            .set_index(
                "gene"
            )
        )

        p2 = (
            clock_fit[
                clock_fit[
                    "lineage"
                ]
                ==
                l2
            ]
            .set_index(
                "gene"
            )
        )

        common = (
            p1.index
            .intersection(
                p2.index
            )
        )

        phase_dists = []

        for gene in (
            common
        ):
            a = p1.loc[
                gene,
                "phase_h"
            ]

            b = p2.loc[
                gene,
                "phase_h"
            ]

            if (
                np.isfinite(
                    a
                )
                and
                np.isfinite(
                    b
                )
            ):
                phase_dists.append(
                    circular_distance_hours(
                        a,
                        b
                    )
                )

        rows.append({
            "lineage_1":
                l1,

            "lineage_2":
                l2,

            "clock17_trajectory_spearman":
                float(
                    rho
                ),

            "median_gene_phase_distance_h":
                float(
                    np.median(
                        phase_dists
                    )
                )
                if phase_dists
                else np.nan,
        })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        ARCHDIR
        /
        "73_cross_lineage_clock_architecture_similarity.csv",
        index=False
    )

    return out


# ============================================================
# 11. SIX-ARM SECONDARY ANALYSIS
# ============================================================

def build_arm_scores(
    pb,
    gm
):
    rows = []

    gene_audit = []

    for arm, genes in (
        SIX_ARMS.items()
    ):
        avail = [
            g
            for g
            in genes
            if g.upper()
            in gm
        ]

        gene_audit.append({
            "arm":
                arm,

            "n_requested":
                len(
                    genes
                ),

            "n_available":
                len(
                    avail
                ),

            "available_genes":
                ";".join(
                    avail
                ),
        })

    pd.DataFrame(
        gene_audit
    ).to_csv(
        ARMDIR
        /
        "73_six_arm_gene_availability.csv",
        index=False
    )

    for lineage in (
        TARGET_LINEAGES
    ):
        d = (
            pb[
                pb[
                    "lineage"
                ]
                ==
                lineage
            ]
            .sort_values(
                "ZT"
            )
            .copy()
        )

        for arm, genes in (
            SIX_ARMS.items()
        ):
            avail = [
                g
                for g
                in genes
                if g.upper()
                in gm
            ]

            if not avail:
                continue

            gene_z = []

            for gene in (
                avail
            ):
                col = gm[
                    gene.upper()
                ]

                gene_z.append(
                    zscore_safe(
                        d[
                            col
                        ].values.astype(
                            float
                        )
                    )
                )

            score = np.mean(
                np.vstack(
                    gene_z
                ),
                axis=0
            )

            for zt, val in zip(
                d[
                    "ZT"
                ].astype(
                    int
                ),
                score
            ):
                rows.append({
                    "lineage":
                        lineage,

                    "ZT":
                        int(
                            zt
                        ),

                    "arm":
                        arm,

                    "score_z":
                        float(
                            val
                        ),

                    "n_genes":
                        len(
                            avail
                        ),
                })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        ARMDIR
        /
        "73_six_arm_scores_by_lineage_ZT.csv",
        index=False
    )

    return out


def fit_arm_harmonics(
    arm_scores
):
    rows = []

    for (
        lineage,
        arm
    ), d in (
        arm_scores.groupby(
            [
                "lineage",
                "arm"
            ],
            observed=True
        )
    ):
        d = d.sort_values(
            "ZT"
        )

        fit = harmonic_fit(
            d[
                "ZT"
            ].values,
            d[
                "score_z"
            ].values
        )

        rows.append({
            "lineage":
                lineage,

            "arm":
                arm,

            **fit,
        })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        ARMDIR
        /
        "73_six_arm_harmonics.csv",
        index=False
    )

    return out


def plot_arm_trajectories(
    arm_scores
):
    for lineage in (
        TARGET_LINEAGES
    ):
        d = arm_scores[
            arm_scores[
                "lineage"
            ]
            ==
            lineage
        ]

        fig, ax = plt.subplots(
            figsize=(
                6.4,
                4.2
            )
        )

        for arm in (
            SIX_ARMS.keys()
        ):
            x = (
                d[
                    d[
                        "arm"
                    ]
                    ==
                    arm
                ]
                .sort_values(
                    "ZT"
                )
            )

            if x.empty:
                continue

            ax.plot(
                x[
                    "ZT"
                ],
                x[
                    "score_z"
                ],
                marker="o",
                linewidth=1.5,
                label=
                    arm,
                color=
                    ARM_COLORS[
                        arm
                    ]
            )

        ax.axhline(
            0,
            linewidth=0.8,
            alpha=0.5
        )

        ax.set_xticks(
            ZT_ORDER
        )

        ax.set_xlabel(
            "Zeitgeber time"
        )

        ax.set_ylabel(
            "Arm score\n(mean gene Z across ZT)"
        )

        ax.set_title(
            f"{lineage} — six-arm temporal architecture"
        )

        ax.legend(
            fontsize=7,
            ncol=2,
            frameon=False
        )

        savefig(
            fig,
            "73D_SIX_ARM_"
            +
            sanitize(
                lineage
            )
        )


def plot_arm_phase_heatmap(
    arm_fit
):
    mat = (
        arm_fit.pivot(
            index=
                "lineage",
            columns=
                "arm",
            values=
                "phase_h"
        )
        .reindex(
            TARGET_LINEAGES
        )
        .reindex(
            columns=
                list(
                    SIX_ARMS.keys()
                )
        )
    )

    mat.to_csv(
        ARMDIR
        /
        "73_six_arm_phase_matrix.csv"
    )

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            3.8
        )
    )

    im = ax.imshow(
        mat.values,
        aspect="auto",
        vmin=0,
        vmax=24
    )

    ax.set_xticks(
        np.arange(
            mat.shape[
                1
            ]
        )
    )

    ax.set_xticklabels(
        mat.columns,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(
        np.arange(
            mat.shape[
                0
            ]
        )
    )

    ax.set_yticklabels(
        mat.index
    )

    ax.set_title(
        "Six-arm fitted phase (secondary analysis)"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.02
    )

    cbar.set_label(
        "Fitted phase (h)"
    )

    savefig(
        fig,
        "73E_SIX_ARM_PHASE_HEATMAP"
    )


# ============================================================
# 12. OPTIONAL FROZEN FIG3D PATHWAY VALIDATION
# ============================================================

def normalize_set_name(
    x
):
    x = str(
        x
    ).upper()

    x = re.sub(
        r"[^A-Z0-9]+",
        "_",
        x
    )

    x = re.sub(
        r"_+",
        "_",
        x
    ).strip(
        "_"
    )

    return x


def infer_pathway_column(
    df
):
    candidates = [
        "pathway",
        "term",
        "gene_set",
        "geneset",
        "pathway_name",
        "description",
        "name",
    ]

    lower = {
        str(
            c
        ).lower():
            c
        for c
        in df.columns
    }

    for cand in (
        candidates
    ):
        if (
            cand
            in lower
        ):
            return lower[
                cand
            ]

    for c in (
        df.columns
    ):
        cl = str(
            c
        ).lower()

        if (
            "pathway"
            in cl
            or
            "term"
            in cl
            or
            "geneset"
            in cl
            or
            "gene_set"
            in cl
        ):
            return c

    return None


def infer_gene_membership_column(
    df
):
    candidates = [
        "genes",
        "gene_members",
        "members",
        "gene_symbols",
        "leadingedge",
        "leading_edge",
        "core_enrichment",
    ]

    lower = {
        str(
            c
        ).lower():
            c
        for c
        in df.columns
    }

    for cand in (
        candidates
    ):
        if (
            cand
            in lower
        ):
            return lower[
                cand
            ]

    return None


def parse_gene_members(
    x
):
    if pd.isna(
        x
    ):
        return []

    s = str(
        x
    )

    parts = re.split(
        r"[;,|/\s]+",
        s
    )

    parts = [
        p.strip()
        for p
        in parts
        if p.strip()
    ]

    return parts


def read_gmt(
    path
):
    sets = {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
            for line in (
                f
            ):
                fields = (
                    line.rstrip(
                        "\n"
                    )
                    .split(
                        "\t"
                    )
                )

                if (
                    len(
                        fields
                    )
                    <
                    3
                ):
                    continue

                name = fields[
                    0
                ]

                genes = [
                    g
                    for g
                    in fields[
                        2:
                    ]
                    if g
                ]

                sets[
                    normalize_set_name(
                        name
                    )
                ] = (
                    name,
                    genes
                )

    except Exception:
        return {}

    return sets


def find_best_local_gmt(
    frozen_names
):
    # No web access. Search local project only.
    gmt_paths = list(
        PROJECT_ROOT.rglob(
            "*.gmt"
        )
    )

    if not gmt_paths:
        return (
            None,
            {},
            0
        )

    target = {
        normalize_set_name(
            x
        )
        for x
        in frozen_names
    }

    best_path = None
    best_sets = {}
    best_match = 0

    for path in (
        gmt_paths
    ):
        sets = read_gmt(
            path
        )

        if not sets:
            continue

        n = len(
            target
            &
            set(
                sets.keys()
            )
        )

        if (
            n
            >
            best_match
        ):
            best_match = n
            best_path = path
            best_sets = sets

    return (
        best_path,
        best_sets,
        best_match
    )


def load_frozen_pathways():
    status = {
        "frozen_source_exists":
            FROZEN_FIG3D.exists(),

        "pathway_column":
            None,

        "gene_membership_source":
            None,

        "n_frozen_pathways":
            0,

        "n_pathways_with_genes":
            0,
    }

    if not FROZEN_FIG3D.exists():
        log(
            "\n[pathway] Frozen Fig3D source not found; "
            "pathway module skipped."
        )

        return (
            {},
            status
        )

    df = pd.read_csv(
        FROZEN_FIG3D,
        sep="\t"
    )

    pathway_col = infer_pathway_column(
        df
    )

    if pathway_col is None:
        log(
            "\n[pathway] Could not infer pathway-name "
            "column in frozen Fig3D source; skipped."
        )

        return (
            {},
            status
        )

    names = (
        df[
            pathway_col
        ]
        .dropna()
        .astype(
            str
        )
        .drop_duplicates()
        .tolist()
    )

    status[
        "pathway_column"
    ] = (
        str(
            pathway_col
        )
    )

    status[
        "n_frozen_pathways"
    ] = (
        len(
            names
        )
    )

    # Prefer genes embedded directly in the frozen source.
    gene_col = infer_gene_membership_column(
        df
    )

    gene_sets = {}

    if (
        gene_col
        is not None
    ):
        for name, d in (
            df.groupby(
                pathway_col,
                sort=False
            )
        ):
            genes = []

            for x in (
                d[
                    gene_col
                ]
            ):
                genes.extend(
                    parse_gene_members(
                        x
                    )
                )

            genes = list(
                dict.fromkeys(
                    genes
                )
            )

            if genes:
                gene_sets[
                    str(
                        name
                    )
                ] = (
                    genes
                )

        status[
            "gene_membership_source"
        ] = (
            f"Fig3D column: {gene_col}"
        )

    else:
        # Otherwise match the frozen pathway names to local GMTs.
        best_path, sets, best_match = (
            find_best_local_gmt(
                names
            )
        )

        if (
            best_path
            is not None
            and
            best_match
            >
            0
        ):
            for name in (
                names
            ):
                key = normalize_set_name(
                    name
                )

                if (
                    key
                    in sets
                ):
                    gene_sets[
                        name
                    ] = sets[
                        key
                    ][
                        1
                    ]

            status[
                "gene_membership_source"
            ] = str(
                best_path
            )

    status[
        "n_pathways_with_genes"
    ] = (
        len(
            gene_sets
        )
    )

    pd.DataFrame({
        "frozen_pathway":
            names
    }).to_csv(
        PATHWAYDIR
        /
        "73_frozen_Fig3D_pathway_order.csv",
        index=False
    )

    with open(
        PATHWAYDIR
        /
        "73_frozen_pathway_status.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            status,
            f,
            indent=2,
            ensure_ascii=False
        )

    log(
        "\n[pathway frozen source]"
    )

    log(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False
        )
    )

    return (
        gene_sets,
        status
    )


def pathway_validation(
    pb,
    gene_sets
):
    if not gene_sets:
        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    gm = gene_lookup(
        pb.columns
    )

    score_rows = []

    pathway_order = list(
        gene_sets.keys()
    )

    audit_rows = []

    for pathway in (
        pathway_order
    ):
        genes = gene_sets[
            pathway
        ]

        avail = [
            g
            for g
            in genes
            if g.upper()
            in gm
        ]

        audit_rows.append({
            "pathway":
                pathway,

            "n_genes_requested":
                len(
                    genes
                ),

            "n_genes_available":
                len(
                    avail
                ),

            "coverage_fraction":
                (
                    len(
                        avail
                    )
                    /
                    len(
                        genes
                    )
                )
                if genes
                else np.nan,
        })

    audit = pd.DataFrame(
        audit_rows
    )

    audit.to_csv(
        PATHWAYDIR
        /
        "73_frozen_pathway_gene_coverage.csv",
        index=False
    )

    # Pre-specified minimum: >= 5 genes available.
    usable_pathways = (
        audit.loc[
            audit[
                "n_genes_available"
            ]
            >=
            5,
            "pathway"
        ]
        .tolist()
    )

    for lineage in (
        TARGET_LINEAGES
    ):
        d = (
            pb[
                pb[
                    "lineage"
                ]
                ==
                lineage
            ]
            .sort_values(
                "ZT"
            )
        )

        for pathway in (
            usable_pathways
        ):
            genes = gene_sets[
                pathway
            ]

            avail = [
                g
                for g
                in genes
                if g.upper()
                in gm
            ]

            gene_z = []

            for gene in (
                avail
            ):
                col = gm[
                    gene.upper()
                ]

                gene_z.append(
                    zscore_safe(
                        d[
                            col
                        ].values.astype(
                            float
                        )
                    )
                )

            score = np.mean(
                np.vstack(
                    gene_z
                ),
                axis=0
            )

            fit = harmonic_fit(
                d[
                    "ZT"
                ].values,
                score
            )

            score_rows.append({
                "lineage":
                    lineage,

                "pathway":
                    pathway,

                "n_genes":
                    len(
                        avail
                    ),

                **fit,
            })

    fit_df = pd.DataFrame(
        score_rows
    )

    fit_df.to_csv(
        PATHWAYDIR
        /
        "73_frozen_Fig3D_pathway_temporal_harmonics.csv",
        index=False
    )

    # Figure uses frozen source order, not validation-performance ranking.
    first_n = [
        p
        for p
        in pathway_order
        if p
        in usable_pathways
    ][
        :20
    ]

    if (
        first_n
        and
        not fit_df.empty
    ):
        mat = (
            fit_df[
                fit_df[
                    "pathway"
                ]
                .isin(
                    first_n
                )
            ]
            .pivot(
                index=
                    "pathway",
                columns=
                    "lineage",
                values=
                    "r2"
            )
            .reindex(
                first_n
            )
            .reindex(
                columns=
                    TARGET_LINEAGES
            )
        )

        mat.to_csv(
            PATHWAYDIR
            /
            "73_frozen_Fig3D_pathway_R2_matrix_first20.csv"
        )

        fig, ax = plt.subplots(
            figsize=(
                7.0,
                max(
                    4.0,
                    0.28
                    *
                    len(
                        mat
                    )
                )
            )
        )

        im = ax.imshow(
            mat.values,
            aspect="auto",
            vmin=0,
            vmax=1
        )

        ax.set_xticks(
            np.arange(
                mat.shape[
                    1
                ]
            )
        )

        ax.set_xticklabels(
            mat.columns,
            rotation=35,
            ha="right"
        )

        ax.set_yticks(
            np.arange(
                mat.shape[
                    0
                ]
            )
        )

        ax.set_yticklabels(
            mat.index,
            fontsize=7
        )

        ax.set_title(
            "Frozen Fig3D pathways — temporal R²"
        )

        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.025,
            pad=0.02
        )

        cbar.set_label(
            "24-h harmonic R²"
        )

        savefig(
            fig,
            "73F_FROZEN_FIG3D_PATHWAY_R2"
        )

    return (
        audit,
        fit_df
    )


# ============================================================
# 13. OPTIONAL STATIC SOURCE AUDIT
# ============================================================

def audit_static_sources():
    rows = []

    for label, path in [
        (
            "Fig1C_core_clock_heatmap_source",
            FROZEN_FIG1C
        ),
        (
            "DONOR_TISSUE_LINEAGE_SCORES",
            STATIC_SIX_ARM
        ),
    ]:
        item = {
            "source":
                label,

            "path":
                str(
                    path
                ),

            "exists":
                path.exists(),

            "n_rows":
                np.nan,

            "n_cols":
                np.nan,

            "columns":
                "",
        }

        if path.exists():
            try:
                if (
                    path.suffix.lower()
                    ==
                    ".tsv"
                ):
                    d = pd.read_csv(
                        path,
                        sep="\t"
                    )
                else:
                    d = pd.read_csv(
                        path
                    )

                item[
                    "n_rows"
                ] = (
                    d.shape[
                        0
                    ]
                )

                item[
                    "n_cols"
                ] = (
                    d.shape[
                        1
                    ]
                )

                item[
                    "columns"
                ] = (
                    ";".join(
                        map(
                            str,
                            d.columns
                        )
                    )
                )

            except Exception as e:
                item[
                    "columns"
                ] = (
                    "READ_ERROR: "
                    +
                    str(
                        e
                    )
                )

        rows.append(
            item
        )

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        ARCHDIR
        /
        "73_static_source_audit.csv",
        index=False
    )

    return out


# ============================================================
# 14. CLOCK17 PHASE / R2 HEATMAPS
# ============================================================

def plot_clock_metric_heatmap(
    clock_fit,
    metric,
    stem,
    title,
    vmin=None,
    vmax=None
):
    mat = (
        clock_fit.pivot(
            index=
                "gene",
            columns=
                "lineage",
            values=
                metric
        )
        .reindex(
            CLOCK17
        )
        .reindex(
            columns=
                TARGET_LINEAGES
        )
    )

    mat.to_csv(
        CLOCKDIR
        /
        f"{stem}.csv"
    )

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            6.4
        )
    )

    im = ax.imshow(
        mat.values,
        aspect="auto",
        vmin=
            vmin,
        vmax=
            vmax
    )

    ax.set_xticks(
        np.arange(
            mat.shape[
                1
            ]
        )
    )

    ax.set_xticklabels(
        mat.columns,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(
        np.arange(
            mat.shape[
                0
            ]
        )
    )

    ax.set_yticklabels(
        mat.index
    )

    ax.set_title(
        title
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.02
    )

    cbar.set_label(
        metric
    )

    savefig(
        fig,
        stem
    )


# ============================================================
# 15. REPORT
# ============================================================

def write_report(
    pb,
    clock_fit,
    arch,
    cross,
    arm_fit,
    pathway_status,
    pathway_fit,
    static_audit
):
    lines = []

    lines.append(
        "73 GSE292219 TIME-RESOLVED VALIDATION"
    )

    lines.append(
        "=" * 78
    )

    lines.append(
        ""
    )

    lines.append(
        "DATA / LABELS"
    )

    lines.append(
        "-----------"
    )

    lines.append(
        "Dataset: GSE292219 mouse liver scRNA-seq."
    )

    lines.append(
        "Time labels used exactly as deposited: ZT3/7/11/15/19/23."
    )

    lines.append(
        "Cell labels frozen from 72_GSE292219_annotation_audit.py."
    )

    lines.append(
        "Only high_confidence_lineage is used downstream."
    )

    lines.append(
        ""
    )

    lines.append(
        "Five target lineages:"
    )

    for lineage in (
        TARGET_LINEAGES
    ):
        d = pb[
            pb[
                "lineage"
            ]
            ==
            lineage
        ]

        lines.append(
            f"- {lineage}: "
            f"{int(d['n_cells'].sum())} total lineage-ZT cells "
            f"(minimum per ZT={int(d['n_cells'].min())})"
        )

    lines.append(
        ""
    )

    lines.append(
        "CLOCK17 TEMPORAL ARCHITECTURE"
    )

    lines.append(
        "-----------------------------"
    )

    lines.append(
        arch.to_string(
            index=False
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "Interpretation note:"
    )

    lines.append(
        "Each ZT is represented by one aggregate sample. Harmonic P values "
        "are descriptive and are not used as the main evidence."
    )

    lines.append(
        ""
    )

    lines.append(
        "CROSS-LINEAGE TEMPORAL SIMILARITY"
    )

    lines.append(
        "---------------------------------"
    )

    lines.append(
        cross.to_string(
            index=False
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "SIX-ARM SECONDARY ANALYSIS"
    )

    lines.append(
        "--------------------------"
    )

    if arm_fit.empty:
        lines.append(
            "No six-arm fits available."
        )
    else:
        lines.append(
            arm_fit[
                [
                    "lineage",
                    "arm",
                    "amplitude",
                    "phase_h",
                    "r2"
                ]
            ]
            .to_string(
                index=False
            )
        )

    lines.append(
        ""
    )

    lines.append(
        "FROZEN FIG3D PATHWAY MODULE"
    )

    lines.append(
        "---------------------------"
    )

    lines.append(
        json.dumps(
            pathway_status,
            indent=2,
            ensure_ascii=False
        )
    )

    lines.append(
        ""
    )

    if (
        pathway_fit is None
        or
        pathway_fit.empty
    ):
        lines.append(
            "Frozen pathway temporal validation was skipped because "
            "gene membership could not be recovered from frozen/local sources."
        )
    else:
        lines.append(
            f"Temporal fits generated for "
            f"{pathway_fit['pathway'].nunique()} frozen pathways."
        )

    lines.append(
        ""
    )

    lines.append(
        "STATIC SOURCE AUDIT"
    )

    lines.append(
        "-------------------"
    )

    lines.append(
        static_audit.to_string(
            index=False
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "SCIENTIFIC SCOPE"
    )

    lines.append(
        "----------------"
    )

    lines.append(
        "This single dataset can validate lineage-specific temporal "
        "organization within liver, but it cannot by itself establish "
        "cross-tissue remodeling."
    )

    lines.append(
        "Final external-validation claims should be based on replication "
        "across GSE292219 and GSE288579, with static-atlas correspondence "
        "added only after the dynamic results are frozen."
    )

    (
        REPORTDIR
        /
        "73_GSE292219_TIME_RESOLVED_VALIDATION_REPORT.txt"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8"
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():
    log(
        "=" * 92
    )

    log(
        "73 GSE292219 TIME-RESOLVED VALIDATION"
    )

    log(
        "Frozen high-confidence annotations -> "
        "Clock17 -> lineage architecture -> six-arm -> frozen Fig3D pathways"
    )

    log(
        "=" * 92
    )

    ann = load_annotations_72()

    pb = aggregate_expression(
        ann
    )

    plot_cell_counts(
        pb
    )

    gm = clock17_available(
        pb
    )

    clock_fit = fit_clock17(
        pb,
        gm
    )

    plot_clock17_heatmaps(
        pb,
        gm
    )

    plot_clock_metric_heatmap(
        clock_fit,
        metric="r2",
        stem="73C_CLOCK17_R2_HEATMAP",
        title="Clock17 24-h harmonic R²",
        vmin=0,
        vmax=1
    )

    plot_clock_metric_heatmap(
        clock_fit,
        metric="phase_h",
        stem="73C_CLOCK17_PHASE_HEATMAP",
        title="Clock17 fitted phase",
        vmin=0,
        vmax=24
    )

    arch = summarize_lineage_architecture(
        clock_fit
    )

    plot_architecture_summary(
        arch
    )

    cross = cross_lineage_similarity(
        pb,
        gm,
        clock_fit
    )

    arm_scores = build_arm_scores(
        pb,
        gm
    )

    arm_fit = fit_arm_harmonics(
        arm_scores
    )

    plot_arm_trajectories(
        arm_scores
    )

    plot_arm_phase_heatmap(
        arm_fit
    )

    gene_sets, pathway_status = (
        load_frozen_pathways()
    )

    pathway_audit, pathway_fit = (
        pathway_validation(
            pb,
            gene_sets
        )
    )

    static_audit = audit_static_sources()

    write_report(
        pb,
        clock_fit,
        arch,
        cross,
        arm_fit,
        pathway_status,
        pathway_fit,
        static_audit
    )

    completion = {
        "pipeline":
            "73_GSE292219_TIME_RESOLVED_VALIDATION",

        "status":
            "COMPLETE",

        "dataset":
            "GSE292219",

        "time_labels":
            ZT_ORDER,

        "annotation_source":
            str(
                ANN72_TABLEDIR
                /
                "GSE292219_all_cells_annotation_audit.csv.gz"
            ),

        "annotation_level":
            "high_confidence_lineage",

        "lineages":
            TARGET_LINEAGES,

        "clock17_n_rows":
            int(
                len(
                    clock_fit
                )
            ),

        "pathway_module":
            pathway_status,

        "interpretation":
            (
                "Descriptive time-resolved validation; "
                "one aggregate sample per ZT."
            ),
    }

    with open(
        OUTDIR
        /
        "73_GSE292219_COMPLETE.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            completion,
            f,
            indent=2,
            ensure_ascii=False
        )

    log(
        "\n"
        +
        "=" * 92
    )

    log(
        "73 GSE292219 TIME-RESOLVED VALIDATION COMPLETE"
    )

    log(
        "=" * 92
    )

    log(
        f"\nOutput: {OUTDIR}"
    )

    log(
        "\nPrimary outputs:"
    )

    log(
        "  03_clock17/73_Clock17_gene_harmonics.csv"
    )

    log(
        "  04_temporal_architecture/73_lineage_clock17_temporal_architecture.csv"
    )

    log(
        "  04_temporal_architecture/73_cross_lineage_clock_architecture_similarity.csv"
    )

    log(
        "  05_six_arm_secondary/73_six_arm_harmonics.csv"
    )

    log(
        "  08_report/73_GSE292219_TIME_RESOLVED_VALIDATION_REPORT.txt"
    )


if __name__ == "__main__":
    main()
