#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
51_redraw_Fig3_EFG_and_matched_tissue_supp.py

PURPOSE
-------
Redraw the matched-competitive tissue-effect results from script 50 without
re-running the statistics.

Outputs
-------
MAIN
1) Fig3E_bridge_clock_vs_tissue_responsiveness
   - x: frozen Fig2B clock-architecture remodeling values
   - y: 90th percentile gene-level tissue effect from script 50
   - descriptive only; NO regression/P value

2) Fig3F_Hallmark_frosted
   - same selected Hallmark pathways and same matched z values as script 50
   - NO significance dots
   - cells with matched-competitive FDR >= 0.05 receive a translucent
     "frosted glass" overlay
   - significant cells remain clear

3) Fig3G_Reactome_frosted
   - same idea for Reactome

SUPPLEMENT
4) Supp_continuous_tissue_effect_summary
   - old script-50 Fig3E preserved as supplementary context

5) Supp_representative_matched_pathways_by_module
   - figure styled after the original Fig3B
   - representative matched-competitive pathways arranged by biological module
   - circle = Hallmark; square = Reactome
   - size = number of lineages with supported matched-competitive enrichment
   - x = median matched z among supporting lineages
   - biological modules are descriptive; frozen V3 assignments are preserved
     where available, otherwise the same keyword framework is extended

IMPORTANT
---------
This script changes ONLY display/organization.
No P values, FDR values, matched z values, or pathway selections from script 50
are recomputed.
"""

from __future__ import annotations
import os

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

SRC50 = ROOT / "50_MATCHED_COMPETITIVE_TISSUE_PATHWAYS_20260922"
SRC_TABLE = SRC50 / "tables"

ALL_RESULTS = SRC_TABLE / "ALL_LINEAGES_matched_competitive_pathways.csv.gz"
E_SUMMARY = SRC_TABLE / "Fig3E_continuous_gene_tissue_effect_summary.csv"
F_SELECTED = SRC_TABLE / "Fig3F_selected_Hallmark.csv"
G_SELECTED = SRC_TABLE / "Fig3G_selected_Reactome.csv"

V3_FREEZE = (
    ROOT
    / "00_freeze"
    / "A4_8_V3_FINAL_ONTOLOGY_20260920"
)
STRICT_ONTOLOGY = V3_FREEZE / "01_V3_PATHWAY_ONTOLOGY.csv"

OUT = ROOT / "51_FIG3_EFG_REDESIGN_20260922"
FIG = OUT / "figures"
TAB = OUT / "source_data"
AUD = OUT / "audit"

for d in [OUT, FIG, TAB, AUD]:
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

# Frozen Fig2B values already used in the manuscript figure.
CLOCK_REMODELING = {
    "b_cell": 1.74,
    "t_cell": 2.31,
    "macrophage_monocyte": 2.36,
    "fibroblast": 3.75,
    "endothelial": 3.05,
}

MODULE_LABELS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE":
        "Cytokine/TNF/NF-κB/innate immune",
    "M02_MAPK_RTK_GROWTH_FACTOR":
        "MAPK/RTK/growth factor",
    "M03_TGFB_SMAD":
        "TGF-β/SMAD",
    "M04_CELL_DEATH_APOPTOSIS":
        "Cell death/apoptosis",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING":
        "Rho/cytoskeleton/adhesion/remodeling",
    "M06_TRAFFICKING_ER_PROTEOSTASIS":
        "Trafficking/ER/proteostasis",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR":
        "Antigen/interferon/immune effector",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION":
        "Transcription/chromatin/differentiation",
    "M09_METABOLIC_OXIDATIVE_STRESS":
        "Metabolic/oxidative stress",
    "M10_CELL_CYCLE_GROWTH":
        "Cell cycle/growth",
    "M11_OTHER_SIGNALING_SPECIALIZED":
        "Other specialized signaling",
    "M99_OTHER_UNCLASSIFIED":
        "Other/unclassified",
}

# IMPORTANT: priority follows frozen V3 module insertion order, not plotting order.
MODULE_PRIORITY = list(MODULE_LABELS.keys())

MODULE_DISPLAY_ORDER = [
    "M02_MAPK_RTK_GROWTH_FACTOR",
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING",
    "M09_METABOLIC_OXIDATIVE_STRESS",
    "M11_OTHER_SIGNALING_SPECIALIZED",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION",
    "M10_CELL_CYCLE_GROWTH",
    "M04_CELL_DEATH_APOPTOSIS",
    "M03_TGFB_SMAD",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR",
    "M99_OTHER_UNCLASSIFIED",
]

MODULE_COLORS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE": "#59A89D",
    "M02_MAPK_RTK_GROWTH_FACTOR": "#4E6A9E",
    "M03_TGFB_SMAD": "#B07AA1",
    "M04_CELL_DEATH_APOPTOSIS": "#D97562",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING": "#76B7B2",
    "M06_TRAFFICKING_ER_PROTEOSTASIS": "#E9968A",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR": "#91C47C",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION": "#8A89A6",
    "M09_METABOLIC_OXIDATIVE_STRESS": "#E0BE63",
    "M10_CELL_CYCLE_GROWTH": "#EDC948",
    "M11_OTHER_SIGNALING_SPECIALIZED": "#86BCB6",
    "M99_OTHER_UNCLASSIFIED": "#A0A4AA",
}

KEYWORDS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE": [
        "TNF", "TNFR", "NFKB", "NF_KB", "NF_KAPPA_B",
        "TOLL", "TLR", "MYD88", "IRAK", "RIPK",
        "INFLAMMATORY", "CYTOKINE", "INTERLEUKIN",
        "IL1", "IL_1", "IL17", "IL_17",
        "IL2_STAT5", "IL_2_STAT5",
        "IL6_JAK_STAT3", "IL_6_JAK_STAT3",
        "INNATE_IMMUNE",
    ],
    "M02_MAPK_RTK_GROWTH_FACTOR": [
        "MAPK", "RAF", "RAS", "RECEPTOR_TYROSINE_KINASE", "RTK",
        "NTRK", "EGFR", "ERBB", "FGFR", "MET",
        "SPRY", "GROWTH_FACTOR", "ERYTHROPOIETIN",
        "PHOSPHOLIPASE_C_GAMMA",
    ],
    "M03_TGFB_SMAD": [
        "TGFB", "TGF_BETA", "TGF_BETA_RECEPTOR", "SMAD",
    ],
    "M04_CELL_DEATH_APOPTOSIS": [
        "APOPTOSIS", "APOPTOTIC", "DEATH_RECEPTOR",
        "BH3", "PUMA", "CASPASE", "NECROSIS", "NECROPTOSIS",
    ],
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING": [
        "RHO", "RAC", "ROCK", "CYTOSKELET", "INTEGRIN",
        "EPHRIN", "SEMAPHORIN", "ADHESION",
        "EPITHELIAL_MESENCHYMAL_TRANSITION",
    ],
    "M06_TRAFFICKING_ER_PROTEOSTASIS": [
        "MEMBRANE_TRAFFICKING", "VESICLE", "GOLGI", "ER_RETROGRADE",
        "RETROGRADE_TRANSPORT", "UBIQUITIN", "PROTEASOME", "AUTOPHAGY",
        "IRE1", "CHAPERONE", "POST_TRANSLATIONAL", "CLATHRIN",
        "UNFOLDED_PROTEIN_RESPONSE", "HEAT_STRESS",
    ],
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR": [
        "ANTIGEN", "MHC", "INTERFERON", "DDX58", "IFIH1",
        "IMMUNE_EFFECTOR", "ADAPTIVE_IMMUNE", "T_CELL_ACTIVATION",
        "NEUTROPHIL", "DEGRANULATION", "PD_L1", "CD274",
    ],
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION": [
        "ESTROGEN", "ANDROGEN", "HDAC", "HISTONE", "CHROMATIN",
        "BETA_CATENIN", "TCF", "RUNX", "TRANSCRIPTIONAL_REGULATION",
        "GENE_EXPRESSION", "DIFFERENTIATION",
    ],
    "M09_METABOLIC_OXIDATIVE_STRESS": [
        "HEME", "HMOX1", "NFE2L2", "KEAP1", "OXIDATIVE_PHOSPHORYLATION",
        "CHOLESTEROL", "FATTY_ACID", "AMINO_ACID", "CYP",
        "METABOLISM", "HYPOXIA", "RESPIRATORY_ELECTRON_TRANSPORT",
    ],
    "M10_CELL_CYCLE_GROWTH": [
        "MITOTIC", "CELL_CYCLE", "MTOR", "PI3K", "P53",
    ],
    "M11_OTHER_SIGNALING_SPECIALIZED": [
        "NOTCH", "HEDGEHOG", "ACTIVATION_OF_SMO", "G_ALPHA",
        "SECOND_MESSENGER", "CHANNEL", "NEURONAL_SYSTEM",
        "PLATELET_ACTIVATION", "GPCR", "HEMOSTASIS", "ATTENUATION_PHASE",
        "CILIUM",
    ],
}

MANUAL_OVERRIDES = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION":
        "M05_RHO_CYTOSKELETON_ADHESION_REMODELING",
    "HALLMARK_IL2_STAT5_SIGNALING":
        "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING":
        "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_MEMBRANE_TRAFFICKING":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_COPI_DEPENDENT_GOLGI_TO_ER_RETROGRADE_TRAFFIC":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_REGULATION_OF_HMOX1_EXPRESSION_AND_ACTIVITY":
        "M09_METABOLIC_OXIDATIVE_STRESS",
    "REACTOME_REGULATION_OF_NFE2L2_GENE_EXPRESSION":
        "M09_METABOLIC_OXIDATIVE_STRESS",
    "REACTOME_ACTIVATION_OF_SMO":
        "M11_OTHER_SIGNALING_SPECIALIZED",
    "REACTOME_NEURONAL_SYSTEM":
        "M11_OTHER_SIGNALING_SPECIALIZED",
}

INTENTIONAL_OTHER = {
    "HALLMARK_UV_RESPONSE_DN",
    "HALLMARK_UV_RESPONSE_UP",
    "REACTOME_DENGUE_VIRUS_GENOME_TRANSLATION_AND_REPLICATION",
    "REACTOME_PROCESSING_OF_CAPPED_INTRON_CONTAINING_PRE_MRNA",
}


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
# HELPERS
# =============================================================================

def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


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

        fig.savefig(
            FIG / f"{stem}.{ext}",
            **kwargs,
        )

    plt.close(fig)


def short_pathway(x):
    return (
        str(x)
        .replace("HALLMARK_", "")
        .replace("REACTOME_", "")
        .replace("_", " ")
    )


def wrap_label(x, width=46):
    return "\n".join(
        textwrap.wrap(
            short_pathway(x),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def assign_module(pathway, frozen_lookup):
    p = str(pathway)

    if p in frozen_lookup:
        return frozen_lookup[p], "frozen_v3"

    up = p.upper()

    if up in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[up], "v3_manual_extended"

    if up in INTENTIONAL_OTHER:
        return "M99_OTHER_UNCLASSIFIED", "v3_intentional_other"

    for module in MODULE_PRIORITY:
        if module == "M99_OTHER_UNCLASSIFIED":
            continue

        if any(
            key in up
            for key in KEYWORDS.get(module, [])
        ):
            return module, "v3_keyword_extended"

    return "M99_OTHER_UNCLASSIFIED", "no_rule_match"


def load_inputs():
    required = [
        ALL_RESULTS,
        E_SUMMARY,
        F_SELECTED,
        G_SELECTED,
    ]

    missing = [
        p
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing script50 output(s):\n"
            + "\n".join(str(p) for p in missing)
        )

    results = pd.read_csv(
        ALL_RESULTS,
        low_memory=False,
    )

    required_cols = {
        "lineage",
        "collection",
        "pathway",
        "matched_z",
        "FDR_matched",
        "supported",
    }

    miss = required_cols - set(results.columns)

    if miss:
        raise RuntimeError(
            "Matched-competitive result table missing columns: "
            + ", ".join(sorted(miss))
        )

    results["supported"] = as_bool(
        results["supported"]
    )

    results["matched_z"] = pd.to_numeric(
        results["matched_z"],
        errors="coerce",
    )

    results["FDR_matched"] = pd.to_numeric(
        results["FDR_matched"],
        errors="coerce",
    )

    e = pd.read_csv(
        E_SUMMARY,
        low_memory=False,
    )

    fsel = pd.read_csv(
        F_SELECTED,
        low_memory=False,
    )

    gsel = pd.read_csv(
        G_SELECTED,
        low_memory=False,
    )

    return results, e, fsel, gsel


# =============================================================================
# FIG 3E — BRIDGE SCATTER
# =============================================================================

def plot_bridge_scatter(e_summary):
    required = {
        "lineage",
        "q90",
    }

    miss = required - set(e_summary.columns)

    if miss:
        raise RuntimeError(
            f"Fig3E summary missing columns: {miss}"
        )

    y_lookup = (
        e_summary
        .set_index("lineage")["q90"]
        .to_dict()
    )

    rows = []

    for lin in LINEAGES:
        if lin not in y_lookup:
            raise RuntimeError(
                f"Missing q90 for lineage {lin}"
            )

        rows.append({
            "lineage": lin,
            "lineage_label": LINEAGE_LABEL[lin],
            "clock_remodeling": CLOCK_REMODELING[lin],
            "q90_tissue_effect": float(y_lookup[lin]),
        })

    dat = pd.DataFrame(rows)

    dat.to_csv(
        TAB / "Fig3E_bridge_scatter_source.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(5.7, 4.0)
    )

    label_offsets = {
        "b_cell": (0.05, 0.006),
        "t_cell": (0.06, -0.010),
        "macrophage_monocyte": (0.08, 0.009),
        "fibroblast": (0.08, 0.006),
        "endothelial": (0.06, 0.009),
    }

    for r in dat.itertuples(index=False):
        ax.scatter(
            r.clock_remodeling,
            r.q90_tissue_effect,
            s=72,
            color=LINEAGE_COLORS[r.lineage],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        dx, dy = label_offsets[r.lineage]

        ax.text(
            r.clock_remodeling + dx,
            r.q90_tissue_effect + dy,
            r.lineage_label,
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel(
        "Clock-architecture tissue remodeling\n"
        "(median within-lineage distance across tissues)"
    )

    ax.set_ylabel(
        "Broader transcriptional tissue responsiveness\n"
        "(90th percentile incremental adjusted R²)"
    )

    ax.set_title(
        "Clock remodeling versus broader tissue responsiveness",
        loc="left",
        pad=7,
    )

    xvals = dat["clock_remodeling"].to_numpy(float)
    yvals = dat["q90_tissue_effect"].to_numpy(float)

    xr = max(xvals) - min(xvals)
    yr = max(yvals) - min(yvals)

    ax.set_xlim(
        min(xvals) - max(0.20, 0.15 * xr),
        max(xvals) + max(0.35, 0.22 * xr),
    )

    ax.set_ylim(
        max(0.0, min(yvals) - max(0.015, 0.20 * yr)),
        max(yvals) + max(0.020, 0.22 * yr),
    )

    clean_axis(ax)

    # Intentionally no regression line and no n=5 correlation P value.
    save_figure(
        fig,
        "Fig3E_bridge_clock_vs_tissue_responsiveness",
    )


# =============================================================================
# SUPP — OLD CONTINUOUS E
# =============================================================================

def plot_old_continuous_e(e_summary):
    required = {
        "lineage",
        "median",
        "q25",
        "q75",
        "q90",
    }

    miss = required - set(e_summary.columns)

    if miss:
        raise RuntimeError(
            f"Fig3E summary missing columns: {miss}"
        )

    s = (
        e_summary
        .set_index("lineage")
        .loc[LINEAGES]
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(5.9, 3.1)
    )

    y = np.arange(
        len(s)
    )

    for yi, r in enumerate(
        s.itertuples(index=False)
    ):
        ax.plot(
            [r.q25, r.q75],
            [yi, yi],
            linewidth=5,
            solid_capstyle="butt",
            color=LINEAGE_COLORS[r.lineage],
        )

        ax.scatter(
            r.median,
            yi,
            s=34,
            color="white",
            edgecolor=LINEAGE_COLORS[r.lineage],
            linewidth=1.2,
            zorder=3,
        )

        ax.scatter(
            r.q90,
            yi,
            s=28,
            color=LINEAGE_COLORS[r.lineage],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            LINEAGE_LABEL[x]
            for x in LINEAGES
        ]
    )
    ax.invert_yaxis()

    ax.set_xlabel(
        "Incremental adjusted R² attributable to tissue"
    )

    ax.set_title(
        "Continuous tissue-effect magnitude after controlling original cell identity",
        loc="left",
        pad=7,
    )

    clean_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markersize=5,
            label="Median",
        ),
        Line2D(
            [0],
            [0],
            linewidth=5,
            color="#999999",
            label="IQR",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=5,
            label="90th percentile",
        ),
    ]

    ax.legend(
        handles=handles,
        frameon=False,
        loc="best",
    )

    save_figure(
        fig,
        "Supp_continuous_tissue_effect_summary",
    )


# =============================================================================
# FIG 3F/G — FROSTED HEATMAP
# =============================================================================

def plot_frosted_heatmap(
    results,
    selected,
    collection,
    stem,
    title,
    z_limit,
    frost_alpha,
):
    order = (
        selected
        .sort_values("display_order")
        ["pathway"]
        .astype(str)
        .tolist()
    )

    z = results[
        results["collection"].eq(collection)
        & results["pathway"].isin(order)
    ].copy()

    effect = z.pivot_table(
        index="pathway",
        columns="lineage",
        values="matched_z",
        aggfunc="first",
    )

    qmat = z.pivot_table(
        index="pathway",
        columns="lineage",
        values="FDR_matched",
        aggfunc="first",
    )

    for lin in LINEAGES:
        if lin not in effect.columns:
            effect[lin] = np.nan

        if lin not in qmat.columns:
            qmat[lin] = np.nan

    effect = (
        effect[LINEAGES]
        .reindex(order)
    )

    qmat = (
        qmat[LINEAGES]
        .reindex(order)
    )

    mat = effect.to_numpy(float)
    q = qmat.to_numpy(float)

    finite = mat[
        np.isfinite(mat)
    ]

    if len(finite) == 0:
        raise RuntimeError(
            f"No finite values for {collection} heatmap."
        )

    if z_limit is None:
        vmax = max(
            2.0,
            float(
                np.nanpercentile(
                    np.abs(finite),
                    98,
                )
            ),
        )
    else:
        vmax = float(z_limit)

    cmap = mpl.colormaps[
        "RdBu_r"
    ].copy()

    cmap.set_bad(
        "#F2F2F2"
    )

    fig_h = max(
        4.1,
        0.30 * len(order) + 1.35,
    )

    fig, ax = plt.subplots(
        figsize=(6.6, fig_h)
    )

    im = ax.imshow(
        np.ma.masked_invalid(mat),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        zorder=1,
    )

    # -------------------------------------------------------------
    # Frosted-glass overlay:
    # non-significant cells get a translucent cool-white veil.
    # No dot markers are used.
    # -------------------------------------------------------------
    nonsig = (
        ~np.isfinite(q)
        | (q >= 0.05)
    )

    # Main translucent white veil.
    for i in range(nonsig.shape[0]):
        for j in range(nonsig.shape[1]):
            if nonsig[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor=(0.97, 0.98, 0.99, frost_alpha),
                        edgecolor=(1.0, 1.0, 1.0, 0.78),
                        linewidth=0.8,
                        zorder=2,
                    )
                )

                # Very subtle cool inner glaze to create a "glass" feel.
                ax.add_patch(
                    Rectangle(
                        (j - 0.46, i - 0.46),
                        0.92,
                        0.92,
                        facecolor=(0.90, 0.94, 0.97, 0.07),
                        edgecolor="none",
                        zorder=2.1,
                    )
                )

    ax.set_xticks(
        np.arange(
            len(LINEAGES)
        )
    )

    ax.set_xticklabels(
        [
            LINEAGE_LABEL[x]
            for x in LINEAGES
        ],
        rotation=28,
        ha="right",
    )

    labels = [
        wrap_label(
            p,
            width=48 if collection == "Reactome" else 42,
        )
        for p in order
    ]

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticklabels(
        labels,
        fontsize=5.8
        if collection == "Reactome"
        else 6.2,
    )

    ax.set_title(
        title,
        loc="left",
        pad=8,
    )

    ax.grid(False)
    ax.tick_params(
        axis="both",
        length=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.025,
    )

    cbar.set_label(
        "Matched competitive enrichment z",
        fontsize=8,
    )

    fig.text(
        0.995,
        0.01,
        "Frosted cells: matched-competitive FDR ≥ 0.05",
        ha="right",
        va="bottom",
        fontsize=6.3,
    )

    # Export exact displayed matrices.
    out_source = effect.copy()
    out_source.columns = [
        f"{x}__matched_z"
        for x in out_source.columns
    ]

    out_q = qmat.copy()
    out_q.columns = [
        f"{x}__FDR"
        for x in out_q.columns
    ]

    pd.concat(
        [out_source, out_q],
        axis=1,
    ).reset_index().to_csv(
        TAB / f"{stem}_source.csv",
        index=False,
    )

    save_figure(
        fig,
        stem,
    )


# =============================================================================
# SUPP — FIG3B-LIKE REPRESENTATIVE PATHWAY PLOT
# =============================================================================

def build_frozen_module_lookup():
    if not STRICT_ONTOLOGY.exists():
        return {}

    x = pd.read_csv(
        STRICT_ONTOLOGY,
        low_memory=False,
    )

    if (
        "pathway" not in x.columns
        or "v3_primary_module_id" not in x.columns
    ):
        return {}

    return {
        str(r.pathway): str(r.v3_primary_module_id)
        for r in x.itertuples(index=False)
        if pd.notna(r.v3_primary_module_id)
    }


def make_supp_module_table(
    results,
    fsel,
    gsel,
    per_module,
    max_total,
):
    frozen_lookup = build_frozen_module_lookup()

    main_selected = set(
        fsel["pathway"].astype(str)
    ) | set(
        gsel["pathway"].astype(str)
    )

    # Summarize only positive supported matched-competitive results.
    supp = results[
        results["supported"]
        & results["matched_z"].gt(0)
    ].copy()

    summary = (
        supp.groupby(
            [
                "collection",
                "pathway",
            ],
            observed=True,
        )
        .agg(
            n_supporting_lineages=(
                "lineage",
                "nunique",
            ),
            median_supported_z=(
                "matched_z",
                "median",
            ),
            max_supported_z=(
                "matched_z",
                "max",
            ),
            supporting_lineages=(
                "lineage",
                lambda x: ";".join(
                    sorted(
                        set(
                            str(v)
                            for v in x
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )

    module_ids = []
    assignment_sources = []

    for p in summary["pathway"].astype(str):
        mid, src = assign_module(
            p,
            frozen_lookup,
        )

        module_ids.append(mid)
        assignment_sources.append(src)

    summary["module_id"] = module_ids
    summary["module_label"] = summary[
        "module_id"
    ].map(
        MODULE_LABELS
    )

    summary["assignment_source"] = assignment_sources
    summary["selected_in_main_heatmaps"] = (
        summary["pathway"]
        .astype(str)
        .isin(main_selected)
    )

    # Keep Other in audit but not in the plotted supplement.
    summary.to_csv(
        AUD / "matched_pathway_module_assignment_all.csv",
        index=False,
    )

    candidate = summary[
        ~summary["module_id"].eq(
            "M99_OTHER_UNCLASSIFIED"
        )
    ].copy()

    mod_order = {
        m: i
        for i, m in enumerate(
            MODULE_DISPLAY_ORDER
        )
    }

    candidate["_module_order"] = (
        candidate["module_id"]
        .map(mod_order)
        .fillna(999)
    )

    # Preference:
    # 1) already shown in main F/G
    # 2) broader lineage support
    # 3) stronger median matched z
    # 4) stronger maximum z
    candidate = candidate.sort_values(
        [
            "_module_order",
            "selected_in_main_heatmaps",
            "n_supporting_lineages",
            "median_supported_z",
            "max_supported_z",
            "pathway",
        ],
        ascending=[
            True,
            False,
            False,
            False,
            False,
            True,
        ],
    )

    selected_parts = []

    for module in MODULE_DISPLAY_ORDER:
        if module == "M99_OTHER_UNCLASSIFIED":
            continue

        z = candidate[
            candidate["module_id"].eq(module)
        ].head(per_module)

        if len(z):
            selected_parts.append(z)

    if not selected_parts:
        raise RuntimeError(
            "No classified supported pathways available for supplementary plot."
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # If > max_total, retain at least one representative per module first,
    # then fill remaining slots by evidence strength.
    if len(selected) > max_total:
        first = (
            selected
            .sort_values(
                [
                    "_module_order",
                    "selected_in_main_heatmaps",
                    "n_supporting_lineages",
                    "median_supported_z",
                ],
                ascending=[
                    True,
                    False,
                    False,
                    False,
                ],
            )
            .groupby(
                "module_id",
                observed=True,
                sort=False,
            )
            .head(1)
        )

        used = set(
            zip(
                first["collection"],
                first["pathway"],
            )
        )

        remaining = selected[
            ~selected.apply(
                lambda r: (
                    r["collection"],
                    r["pathway"],
                )
                in used,
                axis=1,
            )
        ].sort_values(
            [
                "selected_in_main_heatmaps",
                "n_supporting_lineages",
                "median_supported_z",
                "max_supported_z",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )

        need = max(
            0,
            max_total - len(first),
        )

        selected = pd.concat(
            [
                first,
                remaining.head(need),
            ],
            ignore_index=True,
        )

    selected["_module_order"] = (
        selected["module_id"]
        .map(mod_order)
        .fillna(999)
    )

    selected = selected.sort_values(
        [
            "_module_order",
            "median_supported_z",
            "n_supporting_lineages",
            "pathway",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    selected.to_csv(
        TAB / "Supp_representative_matched_pathways_by_module_source.csv",
        index=False,
    )

    return selected


def plot_supp_module_pathways(selected):
    modules_present = [
        m
        for m in MODULE_DISPLAY_ORDER
        if m in set(
            selected["module_id"]
        )
        and m != "M99_OTHER_UNCLASSIFIED"
    ]

    ordered_rows = []
    module_ranges = []
    y_cursor = 0

    for module in modules_present:
        z = selected[
            selected["module_id"].eq(module)
        ].copy()

        z = z.sort_values(
            [
                "median_supported_z",
                "n_supporting_lineages",
                "pathway",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )

        start = y_cursor

        for _, r in z.iterrows():
            ordered_rows.append(r)
            y_cursor += 1

        end = y_cursor - 1

        module_ranges.append(
            (
                module,
                start,
                end,
            )
        )

    p = pd.DataFrame(
        ordered_rows
    ).reset_index(drop=True)

    n = len(p)

    fig_h = max(
        6.4,
        0.43 * n + 1.7,
    )

    fig, ax = plt.subplots(
        figsize=(7.4, fig_h)
    )

    # Reserve some right margin for module labels and legends.
    plt.subplots_adjust(
        left=0.42,
        right=0.77,
        top=0.94,
        bottom=0.14,
    )

    y = np.arange(n)

    xmin = float(
        np.nanmin(
            p["median_supported_z"]
        )
    )
    xmax = float(
        np.nanmax(
            p["median_supported_z"]
        )
    )

    xr = max(
        1.0,
        xmax - xmin,
    )

    x_left = max(
        0.0,
        xmin - 0.18 * xr,
    )

    x_right = xmax + 0.20 * xr

    # Background module bands.
    for module, start, end in module_ranges:
        color = MODULE_COLORS[module]

        ax.axhspan(
            start - 0.48,
            end + 0.48,
            color=color,
            alpha=0.085,
            zorder=0,
        )

        mid = (
            start + end
        ) / 2

        ax.text(
            1.015,
            1 - (
                (mid + 0.5)
                / n
            ),
            MODULE_LABELS[module],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.4,
            color=color,
            clip_on=False,
        )

        if end < n - 1:
            ax.axhline(
                end + 0.5,
                color=color,
                alpha=0.22,
                linewidth=0.7,
                zorder=0.5,
            )

    size_map = {
        1: 38,
        2: 58,
        3: 82,
        4: 108,
        5: 138,
    }

    for yi, r in enumerate(
        p.itertuples(index=False)
    ):
        marker = (
            "o"
            if r.collection == "Hallmark"
            else "s"
        )

        ax.scatter(
            r.median_supported_z,
            yi,
            s=size_map.get(
                int(r.n_supporting_lineages),
                38,
            ),
            marker=marker,
            color=MODULE_COLORS[
                r.module_id
            ],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.96,
            zorder=3,
        )

    labels = []

    for r in p.itertuples(index=False):
        prefix = (
            "Hallmark"
            if r.collection == "Hallmark"
            else "Reactome"
        )

        labels.append(
            textwrap.fill(
                f"{prefix} | {short_pathway(r.pathway)}",
                width=48,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=6.4,
    )
    ax.invert_yaxis()

    ax.set_xlim(
        x_left,
        x_right,
    )

    ax.set_xlabel(
        "Median matched competitive enrichment z\n"
        "(among supporting lineages)"
    )

    ax.set_title(
        "Representative matched-competitive pathways organized by biological module",
        loc="left",
        pad=8,
    )

    clean_axis(ax)

    # Collection legend.
    collection_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=7,
            label="Hallmark",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=7,
            label="Reactome",
        ),
    ]

    leg1 = ax.legend(
        handles=collection_handles,
        title="Collection",
        frameon=False,
        bbox_to_anchor=(1.00, -0.02),
        loc="upper left",
        borderaxespad=0,
    )

    ax.add_artist(leg1)

    # Supporting-lineage size legend.
    size_values = [
        1,
        3,
        5,
    ]

    size_handles = [
        ax.scatter(
            [],
            [],
            s=size_map[k],
            color="#A7A7A7",
            edgecolor="white",
            linewidth=0.7,
            label=f"{k} lineage"
            if k == 1
            else f"{k} lineages",
        )
        for k in size_values
    ]

    ax.legend(
        handles=size_handles,
        title="Supporting lineages",
        frameon=False,
        bbox_to_anchor=(1.30, -0.02),
        loc="upper left",
        borderaxespad=0,
    )

    save_figure(
        fig,
        "Supp_representative_matched_pathways_by_module",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--z-limit",
        type=float,
        default=6.0,
        help=(
            "Display-only symmetric color limit for Fig3F/G. "
            "Underlying matched z values are not modified."
        ),
    )

    parser.add_argument(
        "--frost-alpha",
        type=float,
        default=0.62,
        help="Opacity of the frosted-glass overlay on FDR>=0.05 cells.",
    )

    parser.add_argument(
        "--supp-per-module",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--supp-max-pathways",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.frost_alpha
        <= 1.0
    ):
        raise ValueError(
            "--frost-alpha must be between 0 and 1."
        )

    print("=" * 112)
    print("FIG3 E/F/G REDESIGN + MATCHED-TISSUE SUPPLEMENT")
    print("=" * 112)
    print(
        "No statistics are recomputed. "
        "This script only redraws/reorganizes script50 outputs."
    )

    results, e, fsel, gsel = load_inputs()

    # Main E
    plot_bridge_scatter(
        e
    )

    # Old E retained as supplementary.
    plot_old_continuous_e(
        e
    )

    # Main F/G with frosted nonsignificant cells.
    plot_frosted_heatmap(
        results=results,
        selected=fsel,
        collection="Hallmark",
        stem="Fig3F_Hallmark_frosted",
        title=(
            "Hallmark programs with tissue effects beyond matched gene background"
        ),
        z_limit=args.z_limit,
        frost_alpha=args.frost_alpha,
    )

    plot_frosted_heatmap(
        results=results,
        selected=gsel,
        collection="Reactome",
        stem="Fig3G_Reactome_frosted",
        title=(
            "Representative Reactome programs with tissue effects beyond matched background"
        ),
        z_limit=args.z_limit,
        frost_alpha=args.frost_alpha,
    )

    # Supplement analogous to original Fig3B.
    supp_table = make_supp_module_table(
        results=results,
        fsel=fsel,
        gsel=gsel,
        per_module=args.supp_per_module,
        max_total=args.supp_max_pathways,
    )

    plot_supp_module_pathways(
        supp_table
    )

    audit = {
        "statistics_recomputed": False,
        "script50_source": str(SRC50),
        "Fig3E": {
            "x": "Frozen Fig2B median within-lineage clock distance across tissues",
            "y": "90th percentile of cell-type-adjusted incremental adjusted R2 tissue effect",
            "regression_or_correlation_test": False,
        },
        "Fig3F_G": {
            "values": "matched_z from script50",
            "significance": "FDR_matched < 0.05",
            "significance_dot_removed": True,
            "nonsignificant_display": "translucent cool-white frosted overlay",
            "display_z_limit": args.z_limit,
            "underlying_values_clipped": False,
        },
        "supplementary_pathway_plot": {
            "x": "median matched z among supporting lineages",
            "size": "number of supporting lineages",
            "shape": "Hallmark circle / Reactome square",
            "module_assignment": (
                "Frozen V3 preserved where available; otherwise descriptive "
                "extension using the same manual/keyword framework."
            ),
            "other_unclassified_plotted": False,
        },
    }

    (
        AUD / "redesign_manifest.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("MAIN")
    print(
        FIG
        / "Fig3E_bridge_clock_vs_tissue_responsiveness.png"
    )
    print(
        FIG
        / "Fig3F_Hallmark_frosted.png"
    )
    print(
        FIG
        / "Fig3G_Reactome_frosted.png"
    )

    print()
    print("SUPPLEMENTARY")
    print(
        FIG
        / "Supp_continuous_tissue_effect_summary.png"
    )
    print(
        FIG
        / "Supp_representative_matched_pathways_by_module.png"
    )

    print()
    print(
        "F/G: no significance dots; FDR>=0.05 cells are frosted."
    )
    print(
        "Supplement: representative pathways are individual pathways, "
        "not coherent redundancy families."
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
