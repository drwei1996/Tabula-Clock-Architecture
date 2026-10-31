#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
43_redraw_Fig3D_major_lineage_module_landscape.py

Fig.3D
Biological-module landscape across current five major lineages.

Scientific scope
----------------
- No A4.7 statistics are recomputed.
- Reads frozen A4.7 formal hits + frozen A4.8-v3 ontology.
- Uses only the five major-lineage formal units.
- Broad/Fine terminology is NOT displayed.
- Blank lineage × module positions mean no formal hit.
- Static transcriptional architecture only.
- No rhythm / phase / amplitude interpretation.

Plot
----
x-axis:
    B cell
    T cell
    Macrophage/Monocyte
    Fibroblast
    Endothelial

y-axis:
    final biological modules

Point size:
    number of coherent pathway families

Point color:
    median standardized correspondence beta
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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
# FIVE MAJOR LINEAGES
# =============================================================================

LINEAGE_ORDER = [
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


# =============================================================================
# FINAL BIOLOGICAL MODULES
# =============================================================================

MODULE_ORDER = [
    "M02_MAPK_RTK_GROWTH_FACTOR",
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION",
    "M09_METABOLIC_OXIDATIVE_STRESS",
    "M04_CELL_DEATH_APOPTOSIS",
    "M11_OTHER_SIGNALING_SPECIALIZED",
    "M03_TGFB_SMAD",
    "M10_CELL_CYCLE_GROWTH",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR",
    "M99_OTHER_UNCLASSIFIED",
]

MODULE_LABEL = {
    "M02_MAPK_RTK_GROWTH_FACTOR":
        "MAPK/RTK/growth factor",

    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE":
        "Cytokine/TNF/NF-κB/innate immune",

    "M06_TRAFFICKING_ER_PROTEOSTASIS":
        "Trafficking/ER/proteostasis",

    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING":
        "Rho/cytoskeleton/adhesion/remodeling",

    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION":
        "Transcription/chromatin/differentiation",

    "M09_METABOLIC_OXIDATIVE_STRESS":
        "Metabolic/oxidative stress",

    "M04_CELL_DEATH_APOPTOSIS":
        "Cell death/apoptosis",

    "M11_OTHER_SIGNALING_SPECIALIZED":
        "Other specialized signaling",

    "M03_TGFB_SMAD":
        "TGF-β/SMAD",

    "M10_CELL_CYCLE_GROWTH":
        "Cell cycle/growth",

    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR":
        "Antigen/interferon/immune effector",

    "M99_OTHER_UNCLASSIFIED":
        "Other/unclassified",
}


# =============================================================================
# HELPERS
# =============================================================================

def save_figure(fig, base):

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def detect_major_lineage(row):

    # Prefer explicit lineage column.
    if "lineage" in row.index:

        val = row["lineage"]

        if pd.notna(val):

            val = str(val)

            if val in LINEAGE_ORDER:
                return val

    # Fallback from historical cell_unit_id.
    cell_unit = str(
        row["cell_unit_id"]
    )

    if cell_unit.startswith("broad::"):

        val = cell_unit.split(
            "broad::",
            1
        )[1]

        if val in LINEAGE_ORDER:
            return val

    return None


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=(
            r"<USER_HOME>\Documents\ChatGPT"
            r"\Tabula Sapiens\corrected_full_analysis_v2"
        ),
    )

    parser.add_argument(
        "--outdir",
        default=None,
    )

    args = parser.parse_args()

    ROOT = Path(
        args.root
    ).resolve()

    # -------------------------------------------------------------------------
    # Frozen A4.8-v3 final ontology
    # -------------------------------------------------------------------------

    V3_FREEZE = (
        ROOT
        / "00_freeze"
        / "A4_8_V3_FINAL_ONTOLOGY_20260920"
    )

    V3_WORKING = (
        ROOT
        / "10_A4_8_V3_FINAL_ONTOLOGY"
    )

    BASE = (
        V3_FREEZE
        if V3_FREEZE.exists()
        else V3_WORKING
    )

    FORMAL_PATH = (
        BASE
        / "04_FORMAL_HITS_WITH_V3_ONTOLOGY.csv"
    )

    if not FORMAL_PATH.exists():

        raise FileNotFoundError(
            f"Cannot find frozen formal-hit table:\n{FORMAL_PATH}"
        )

    if args.outdir is None:

        OUT = (
            ROOT
            / "43_FIG3D_MAJOR_LINEAGE_MODULE_LANDSCAPE_20260921"
        )

    else:

        OUT = Path(
            args.outdir
        ).resolve()

    FIG = OUT / "figures"
    SRC = OUT / "source_data"
    AUDIT = OUT / "audit"

    for directory in [
        OUT,
        FIG,
        SRC,
        AUDIT,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================================
    # LOAD FROZEN FORMAL HITS
    # =========================================================================

    formal = pd.read_csv(
        FORMAL_PATH,
        low_memory=False,
    )

    required = {
        "cell_unit_id",
        "v3_primary_module_id",
        "complete_linkage_cluster_id",
        "beta_standardized_pairwise_correspondence",
        "pathway",
    }

    missing = (
        required
        - set(formal.columns)
    )

    if missing:

        raise RuntimeError(
            "Formal table missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # =========================================================================
    # RESOLVE CURRENT MAJOR LINEAGE
    # =========================================================================

    formal[
        "major_lineage"
    ] = formal.apply(
        detect_major_lineage,
        axis=1,
    )

    # Keep only lineage-level/broad-level A4.7 formal units.
    if "level" in formal.columns:

        level_mask = (
            formal["level"]
            .astype(str)
            .str.lower()
            .eq("broad")
        )

    else:

        level_mask = (
            formal["cell_unit_id"]
            .astype(str)
            .str.startswith("broad::")
        )

    major = formal[
        level_mask
        &
        formal[
            "major_lineage"
        ].isin(
            LINEAGE_ORDER
        )
    ].copy()

    if major.empty:

        raise RuntimeError(
            "No five-major-lineage formal hits were found."
        )

    found_lineages = [
        x
        for x in LINEAGE_ORDER
        if x in set(
            major[
                "major_lineage"
            ]
        )
    ]

    missing_lineages = [
        x
        for x in LINEAGE_ORDER
        if x not in set(
            major[
                "major_lineage"
            ]
        )
    ]

    print(
        "\nMajor lineages with formal hits:",
        found_lineages,
    )

    if missing_lineages:

        print(
            "Major lineages with NO formal hits:",
            missing_lineages,
        )

    # =========================================================================
    # MODULE × LINEAGE SUMMARY
    # =========================================================================

    module_lineage = (
        major
        .groupby(
            [
                "v3_primary_module_id",
                "major_lineage",
            ],
            observed=True,
        )
        .agg(
            median_beta=(
                "beta_standardized_pairwise_correspondence",
                "median",
            ),

            n_families=(
                "complete_linkage_cluster_id",
                "nunique",
            ),

            n_formal_hits=(
                "pathway",
                "size",
            ),

            n_unique_pathways=(
                "pathway",
                "nunique",
            ),
        )
        .reset_index()
    )

    module_lineage[
        "lineage_label"
    ] = (
        module_lineage[
            "major_lineage"
        ]
        .map(
            LINEAGE_LABEL
        )
    )

    module_lineage[
        "module_label"
    ] = (
        module_lineage[
            "v3_primary_module_id"
        ]
        .map(
            MODULE_LABEL
        )
    )

    module_lineage.to_csv(
        SRC
        / "FIG3D_major_lineage_module_landscape.csv",
        index=False,
    )

    # =========================================================================
    # DISPLAY ORDERS
    # =========================================================================

    modules_present = set(
        module_lineage[
            "v3_primary_module_id"
        ].astype(str)
    )

    module_order = [
        module
        for module in MODULE_ORDER
        if module in modules_present
    ]

    # Preserve any unexpected module at the end.
    module_order += sorted([
        module
        for module in modules_present
        if module not in module_order
    ])

    lineage_order = (
        LINEAGE_ORDER.copy()
    )

    xmap = {
        lineage: i
        for i, lineage in enumerate(
            lineage_order
        )
    }

    ymap = {
        module: i
        for i, module in enumerate(
            module_order
        )
    }

    # =========================================================================
    # COLOR SCALE
    # =========================================================================

    beta = pd.to_numeric(
        major[
            "beta_standardized_pairwise_correspondence"
        ],
        errors="coerce",
    )

    beta = beta[
        np.isfinite(beta)
    ]

    if beta.empty:

        raise RuntimeError(
            "No finite standardized correspondence beta values."
        )

    bmin = float(
        np.nanpercentile(
            beta,
            2,
        )
    )

    bmax = float(
        np.nanpercentile(
            beta,
            98,
        )
    )

    if (
        not np.isfinite(bmin)
        or
        not np.isfinite(bmax)
        or
        bmin == bmax
    ):

        bmin = float(
            beta.min()
        )

        bmax = float(
            beta.max()
        )

    # =========================================================================
    # FIGURE
    # =========================================================================

    # Wider than previous version:
    # right part is intentionally reserved for colorbar + size legend.
    fig = plt.figure(
        figsize=(9.4, 5.6)
    )

    # Main plotting axis.
    # [left, bottom, width, height]
    ax = fig.add_axes([
        0.34,
        0.17,
        0.36,
        0.74,
    ])

    cmap = (
        mpl.colormaps[
            "viridis"
        ].copy()
    )

    # -------------------------------------------------------------------------
    # Points
    # -------------------------------------------------------------------------

    for _, row in module_lineage.iterrows():

        lineage = str(
            row[
                "major_lineage"
            ]
        )

        module = str(
            row[
                "v3_primary_module_id"
            ]
        )

        if lineage not in xmap:
            continue

        if module not in ymap:
            continue

        n_families = float(
            row[
                "n_families"
            ]
        )

        point_size = (
            26.0
            +
            30.0
            * n_families
        )

        ax.scatter(
            xmap[
                lineage
            ],
            ymap[
                module
            ],

            s=point_size,

            c=[
                float(
                    row[
                        "median_beta"
                    ]
                )
            ],

            cmap=cmap,

            vmin=bmin,
            vmax=bmax,

            edgecolor="white",
            linewidth=0.65,

            zorder=3,
        )

    # =========================================================================
    # AXES
    # =========================================================================

    ax.set_xticks(
        range(
            len(
                lineage_order
            )
        )
    )

    ax.set_xticklabels(
        [
            LINEAGE_LABEL[
                lineage
            ]
            for lineage
            in lineage_order
        ],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_yticks(
        range(
            len(
                module_order
            )
        )
    )

    ax.set_yticklabels(
        [
            MODULE_LABEL.get(
                module,
                module,
            )
            for module
            in module_order
        ]
    )

    ax.invert_yaxis()

    ax.set_xlim(
        -0.20,
        len(
            lineage_order
        )
        - 0.80,
    )

    ax.set_title(
        "Biological-module landscape across major lineages",
        loc="left",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    # No gridlines.
    ax.grid(
        False
    )

    # Keep small visible tick marks only on y-axis.
    ax.tick_params(
        axis="x",
        length=0,
    )

    ax.tick_params(
        axis="y",
        length=3,
        width=0.8,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    # =========================================================================
    # COLORBAR
    #
    # Completely independent axis.
    # This prevents overlap with Supported families legend.
    # =========================================================================

    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(
            vmin=bmin,
            vmax=bmax,
        ),
        cmap=cmap,
    )

    sm.set_array([])

    # Dedicated colorbar axis:
    # leaves a large gap before size legend.
    cax = fig.add_axes([
        0.735,
        0.31,
        0.022,
        0.43,
    ])

    cbar = fig.colorbar(
        sm,
        cax=cax,
    )

    cbar.set_label(
        "Median standardized\ncorrespondence β",
        fontsize=8,
        labelpad=10,
    )

    # Put colorbar label on LEFT.
    # Tick labels remain on right.
    cbar.ax.yaxis.set_label_position(
        "left"
    )

    cbar.ax.tick_params(
        labelsize=7,
        width=0.8,
        length=3,
    )

    # =========================================================================
    # SIZE LEGEND
    #
    # Separate figure-level legend, well to the right of colorbar.
    # =========================================================================

    size_values = [
        1,
        3,
        6,
    ]

    size_handles = []

    for n in size_values:

        size_handles.append(
            plt.scatter(
                [],
                [],
                s=(
                    26
                    +
                    30
                    * n
                ),
                color="#A6ADB3",
                edgecolor="white",
                linewidth=0.65,
            )
        )

    size_labels = [
        "1 family",
        "3 families",
        "6 families",
    ]

    fig.legend(
        size_handles,
        size_labels,

        title="Supported families",

        frameon=False,

        loc="center left",

        # Figure coordinates:
        # safely to the right of colorbar.
        bbox_to_anchor=(
            0.81,
            0.57,
        ),

        borderaxespad=0,

        fontsize=7,
        title_fontsize=7,

        labelspacing=1.00,
        handletextpad=0.9,
    )

    # =========================================================================
    # OPTIONAL NOTE FOR EMPTY MACROPHAGE/MONOCYTE COLUMN
    #
    # Keep it subtle; blank column remains scientifically correct.
    # =========================================================================

   
    # =========================================================================
    # SAVE
    # =========================================================================

    save_figure(
        fig,
        FIG
        / "FIG3D_major_lineage_biological_module_landscape",
    )

    # =========================================================================
    # AUDIT
    # =========================================================================

    audit = {

        "source":
            str(
                FORMAL_PATH
            ),

        "statistics_recomputed":
            False,

        "analysis_level":
            "major-lineage A4.7 formal units only",

        "displayed_lineages": [
            LINEAGE_LABEL[
                x
            ]
            for x
            in LINEAGE_ORDER
        ],

        "lineages_with_formal_hits": [
            LINEAGE_LABEL[
                x
            ]
            for x
            in found_lineages
        ],

        "lineages_without_formal_hits": [
            LINEAGE_LABEL[
                x
            ]
            for x
            in missing_lineages
        ],

        "old_broad_fine_display_labels":
            False,

        "formal_hit_rows_used":
            int(
                len(
                    major
                )
            ),

        "unique_pathways":
            int(
                major[
                    "pathway"
                ]
                .nunique()
            ),

        "unique_coherent_families":
            int(
                major[
                    "complete_linkage_cluster_id"
                ]
                .nunique()
            ),

        "module_lineage_combinations":
            int(
                len(
                    module_lineage
                )
            ),

        "beta_color_min":
            float(
                bmin
            ),

        "beta_color_max":
            float(
                bmax
            ),
    }

    with open(
        AUDIT
        / "FIG3D_audit.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            audit,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # =========================================================================
    # TERMINAL SUMMARY
    #
    # IMPORTANT:
    # sort BEFORE selecting displayed columns.
    # This fixes the previous KeyError.
    # =========================================================================

    print(
        "\n"
        + "=" * 92
    )

    print(
        "FIG3D MAJOR-LINEAGE MODULE LANDSCAPE COMPLETE"
    )

    print(
        "=" * 92
    )

    print(
        f"Formal hit rows used: {len(major)}"
    )

    print(
        f"Unique pathways: {major['pathway'].nunique()}"
    )

    print(
        "Unique coherent families:",
        major[
            "complete_linkage_cluster_id"
        ].nunique(),
    )

    print(
        "\nLineages with formal hits:"
    )

    for lineage in found_lineages:

        print(
            "  ",
            LINEAGE_LABEL[
                lineage
            ],
        )

    if missing_lineages:

        print(
            "\nLineages with NO formal hits:"
        )

        for lineage in missing_lineages:

            print(
                "  ",
                LINEAGE_LABEL[
                    lineage
                ],
            )

    print(
        "\nModule × lineage summary:"
    )

    print(
        module_lineage
        .sort_values(
            [
                "major_lineage",
                "v3_primary_module_id",
            ]
        )[
            [
                "lineage_label",
                "module_label",
                "median_beta",
                "n_families",
                "n_formal_hits",
                "n_unique_pathways",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nOLD BROAD/FINE DISPLAY LABELS: NO"
    )

    print(
        "A4.7 STATISTICS RECOMPUTED: NO"
    )

    print(
        "BACKGROUND GRIDLINES: NO"
    )

    print(
        "\nFigure:"
    )

    print(
        FIG
        / "FIG3D_major_lineage_biological_module_landscape.png"
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUT
    )

    print(
        "=" * 92
    )


if __name__ == "__main__":
    main()