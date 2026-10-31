#!/usr/bin/env python
from __future__ import annotations

"""
34_freeze_and_build_tabula_style_additions.py

Purpose
-------
1) FREEZE all current article-facing figure outputs without modifying them.
2) Build a completely separate set of Tabula-style candidate additions inspired by:
   - global atlas heatmap + lineage zoom-ins
   - same-coordinate lineage/tissue/donor UMAP comparisons
   - feature-level backbone/context architecture
   - tissue × lineage remodeling landscape
   - pathway-family ontology landscape
   - within-donor A4.7 design schematic
3) Keep all previous figures untouched and verify this with SHA256 before/after.

IMPORTANT
---------
This script does NOT:
- overwrite any existing final figure;
- alter A1/A2/A4.7/A4.8 statistics;
- refit the article-clean UMAP;
- change pathway ontology or complete-linkage families;
- change source data;
- make temporal circadian claims.

Tabula Sapiens is static-expression data here. No phase/rhythm/amplitude/
oscillation interpretation is allowed.

Outputs
-------
A) Immutable-style archive copy:
   corrected_full_analysis_v2/
   00_ARTICLE_FIGURE_FREEZE_V1_20260920/

B) New candidate additions only:
   corrected_full_analysis_v2/
   34_TABULA_STYLE_CANDIDATE_ADDITIONS_20260920/

The frozen V1 and the new candidate additions are deliberately separate.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    SCIPY_CLUSTER = True
except Exception:
    linkage = None
    leaves_list = None
    SCIPY_CLUSTER = False


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

# Current article-facing outputs that must never be modified by this script.
FINAL_V1 = ROOT / "FINAL_MANUSCRIPT_FIGURES"
CLEAN16 = ROOT / "16_ARTICLE_CLEAN_ARCHITECTURE_20260920"
PANELS14 = ROOT / "14_PUBLICATION_PANELS_20260920"
REDRAW22 = ROOT / "22_SELECTED_SUPPLEMENT_REDRAW_20260920"

PROTECTED_ROOTS = [
    FINAL_V1,
    CLEAN16,
    PANELS14,
    REDRAW22,
]

# Frozen data / statistics.
FROZEN_META = ROOT / "00_freeze" / "observation_metadata_corrected_v3_FROZEN.tsv"
STD24 = (
    PROJECT
    / "figures_revised"
    / "clock_architecture_umap"
    / "source_data"
    / "clock_architecture_UMAP_clock_architecture_standardized_input.tsv"
)

A2 = ROOT / "02_A2"
A2_FEATURE = A2 / "corrected_feature_effects_and_scores.tsv"
A2_BYLINEAGE = A2 / "corrected_lineage_feature_tissue_effects.tsv"
A2_SUPPORT = A2 / "corrected_centroid_support.tsv"

CLEAN_SRC = CLEAN16 / "source_data"
UMAP_LINEAGE_SRC = CLEAN_SRC / "U_clock_architecture_by_lineage.csv"

V3 = ROOT / "00_freeze" / "A4_8_V3_FINAL_ONTOLOGY_20260920"
V3_ONTOLOGY = V3 / "01_V3_PATHWAY_ONTOLOGY.csv"
V3_FAMILY = V3 / "03_V3_REDUNDANCY_FAMILY_ONTOLOGY.csv"
V3_FORMAL = V3 / "04_FORMAL_HITS_WITH_V3_ONTOLOGY.csv"
V3_MODULE = V3 / "05_V3_MODULE_SUMMARY.csv"

# Freeze copy of all current article-facing figure roots.
FREEZE = ROOT / "00_ARTICLE_FIGURE_FREEZE_V1_20260920"
FREEZE_MANIFEST = FREEZE / "00_MANIFEST"

# New work only.
OUT = ROOT / "34_TABULA_STYLE_CANDIDATE_ADDITIONS_20260920"
DIR = {
    "global": OUT / "01_GLOBAL_CLOCK_FEATURE_LANDSCAPE",
    "umap": OUT / "02_SAME_COORDINATE_UMAP_CONTEXT",
    "feature": OUT / "03_FEATURE_ARCHITECTURE_SPECTRUM",
    "remodel": OUT / "04_TISSUE_LINEAGE_REMODELING",
    "ontology": OUT / "05_PATHWAY_ONTOLOGY_LANDSCAPE",
    "design": OUT / "06_A47_DESIGN_SCHEMATIC",
    "source": OUT / "source_data",
    "audit": OUT / "audit",
}

FIG_EXTS = [".png", ".pdf", ".svg"]


# =============================================================================
# PUBLICATION STYLE
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
    "savefig.facecolor": "white",
})

COL = {
    "blue": "#4E79A7",
    "red": "#E15759",
    "green": "#59A14F",
    "orange": "#F28E2B",
    "purple": "#9467BD",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "gray": "#9EA4AA",
    "lightgray": "#E8EAED",
    "dark": "#30343B",
    "white": "#FFFFFF",
}

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
    "macrophage_monocyte": "Macrophage/monocyte",
    "fibroblast": "Fibroblast",
    "endothelial": "Endothelial",
    "other": "Other",
}
LINEAGE_COLORS = {
    "b_cell": "#4E79A7",
    "t_cell": "#E15759",
    "macrophage_monocyte": "#59A14F",
    "fibroblast": "#F28E2B",
    "endothelial": "#9467BD",
    "other": "#B9B9B9",
}

MODULE_SHORT = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE": "Cytokine/TNF/NF-κB/innate immune",
    "M02_MAPK_RTK_GROWTH_FACTOR": "MAPK/RTK/growth factor",
    "M03_TGFB_SMAD": "TGF-β/SMAD",
    "M04_CELL_DEATH_APOPTOSIS": "Cell death/apoptosis",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING": "Rho/cytoskeleton/adhesion/remodeling",
    "M06_TRAFFICKING_ER_PROTEOSTASIS": "Trafficking/ER/proteostasis",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR": "Antigen/interferon/immune effector",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION": "Transcription/chromatin/differentiation",
    "M09_METABOLIC_OXIDATIVE_STRESS": "Metabolic/oxidative stress",
    "M10_CELL_CYCLE_GROWTH": "Cell cycle/growth",
    "M11_OTHER_SIGNALING_SPECIALIZED": "Other specialized signaling",
    "M99_OTHER_UNCLASSIFIED": "Other/unclassified",
}

MODULE_COLORS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE": "#2A9D8F",
    "M02_MAPK_RTK_GROWTH_FACTOR": "#355C9A",
    "M03_TGFB_SMAD": "#B07AA1",
    "M04_CELL_DEATH_APOPTOSIS": "#E76F51",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING": "#76B7B2",
    "M06_TRAFFICKING_ER_PROTEOSTASIS": "#F28E7F",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR": "#8CD17D",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION": "#8A89A6",
    "M09_METABOLIC_OXIDATIVE_STRESS": "#E9C46A",
    "M10_CELL_CYCLE_GROWTH": "#EDC948",
    "M11_OTHER_SIGNALING_SPECIALIZED": "#86BCB6",
    "M99_OTHER_UNCLASSIFIED": "#9EA4AA",
}

CELL_ORDER = [
    "broad::b_cell",
    "broad::endothelial",
    "broad::fibroblast",
    "broad::macrophage_monocyte",
    "broad::t_cell",
    "fine::b_cell::B cell",
    "fine::b_cell::plasma cell",
    "fine::endothelial::endothelial cell",
    "fine::macrophage_monocyte::classical monocyte",
    "fine::t_cell::T cell",
]
CELL_LABELS = {
    "broad::b_cell": "Broad B",
    "broad::endothelial": "Broad endothelial",
    "broad::fibroblast": "Broad fibroblast",
    "broad::macrophage_monocyte": "Broad macrophage/monocyte",
    "broad::t_cell": "Broad T",
    "fine::b_cell::B cell": "Fine B",
    "fine::b_cell::plasma cell": "Plasma",
    "fine::endothelial::endothelial cell": "Fine endothelial",
    "fine::macrophage_monocyte::classical monocyte": "Classical monocyte",
    "fine::t_cell::T cell": "Fine generic T",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def snapshot_tree(roots: list[Path]) -> dict[str, str]:
    out = {}
    for root in roots:
        if not root.exists():
            continue
        root_key = root.name
        for p in sorted(root.rglob("*")):
            if p.is_file():
                key = f"{root_key}/{p.relative_to(root).as_posix()}"
                out[key] = sha256(p)
    return out


def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, directory: Path, stem: str):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in FIG_EXTS:
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.12,
            "facecolor": "white",
        }
        if ext == ".png":
            kwargs["dpi"] = 600
        fig.savefig(directory / f"{stem}{ext}", **kwargs)
    plt.close(fig)


def write_csv(df: pd.DataFrame, name: str):
    DIR["source"].mkdir(parents=True, exist_ok=True)
    df.to_csv(DIR["source"] / name, index=False, encoding="utf-8-sig")


def pretty_feature(x: str) -> str:
    s = str(x)
    s = re.sub(r"^expr_", "", s)
    s = re.sub(r"^module_", "", s)
    s = s.replace("_", " ")
    return s


def pretty_pathway(x: str) -> str:
    s = str(x)
    s = s.replace("HALLMARK_", "").replace("REACTOME_", "")
    return s.replace("_", " ")


def pretty_cell(x: str) -> str:
    return CELL_LABELS.get(str(x), str(x).replace("::", " / "))


def mode_string(s: pd.Series, fallback="Unknown") -> str:
    x = s.dropna().astype(str)
    x = x[~x.str.lower().isin({"", "nan", "none", "na"})]
    if x.empty:
        return fallback
    vc = x.value_counts()
    return str(vc.index[0])


def categorical_palette(categories: list[str], cmap_name="tab20") -> dict[str, str]:
    cats = list(dict.fromkeys([str(x) for x in categories]))
    if not cats:
        return {}
    if len(cats) <= 20:
        cmap = mpl.colormaps[cmap_name]
        return {
            c: mpl.colors.to_hex(cmap(i / max(1, len(cats) - 1)))
            for i, c in enumerate(cats)
        }
    # More than 20 categories: evenly sample a continuous categorical-like palette.
    cmap = mpl.colormaps["turbo"]
    return {
        c: mpl.colors.to_hex(cmap(i / max(1, len(cats) - 1)))
        for i, c in enumerate(cats)
    }


def cluster_rows(matrix: np.ndarray, labels: list[str]) -> list[int]:
    if len(labels) <= 2 or not SCIPY_CLUSTER:
        return list(range(len(labels)))
    try:
        z = linkage(matrix, method="average", metric="euclidean", optimal_ordering=True)
        return leaves_list(z).astype(int).tolist()
    except Exception:
        return list(range(len(labels)))


def textwrap_labels(values, width=36):
    return [
        "\n".join(
            textwrap.wrap(
                str(x),
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        for x in values
    ]


# =============================================================================
# FREEZE CURRENT FIGURES
# =============================================================================

def copy_freeze_root(src: Path, dest_parent: Path):
    dst = dest_parent / src.name
    if dst.exists():
        return
    shutil.copytree(src, dst)


def verify_freeze_snapshot():
    """
    Freeze is write-once:
    - If absent, copy all protected roots.
    - If present, never refresh it; verify against the current source roots.
    """
    missing = [p for p in PROTECTED_ROOTS if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Protected article-facing roots are missing:\n"
            + "\n".join(str(x) for x in missing)
        )

    before = snapshot_tree(PROTECTED_ROOTS)

    if not FREEZE.exists():
        FREEZE.mkdir(parents=True)
        for src in PROTECTED_ROOTS:
            copy_freeze_root(src, FREEZE)

        FREEZE_MANIFEST.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"path": k, "sha256": v} for k, v in sorted(before.items())]
        ).to_csv(
            FREEZE_MANIFEST / "ORIGINAL_ARTICLE_FIGURE_HASHES.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (FREEZE / "DO_NOT_MODIFY.txt").write_text(
            "This is the frozen V1 article-facing figure snapshot created on 2026-09-20.\n"
            "Do not replace files in this directory. New candidate figures belong in\n"
            "34_TABULA_STYLE_CANDIDATE_ADDITIONS_20260920.\n",
            encoding="utf-8",
        )
    else:
        manifest = FREEZE_MANIFEST / "ORIGINAL_ARTICLE_FIGURE_HASHES.csv"
        if not manifest.exists():
            raise RuntimeError(
                f"Freeze directory exists without hash manifest:\n{FREEZE}"
            )
        old = pd.read_csv(manifest)
        expected = dict(zip(old["path"].astype(str), old["sha256"].astype(str)))
        if before != expected:
            extra = sorted(set(before) - set(expected))
            missing_keys = sorted(set(expected) - set(before))
            changed = sorted(
                k for k in set(before) & set(expected)
                if before[k] != expected[k]
            )
            msg = [
                "Current article-facing files no longer match the V1 freeze.",
                "The frozen copy was NOT modified.",
                f"Changed: {len(changed)}; Added: {len(extra)}; Missing: {len(missing_keys)}",
            ]
            if changed:
                msg += ["Changed examples:"] + changed[:10]
            if extra:
                msg += ["Added examples:"] + extra[:10]
            if missing_keys:
                msg += ["Missing examples:"] + missing_keys[:10]
            raise RuntimeError("\n".join(msg))

    return before


# =============================================================================
# LOAD FROZEN INPUTS
# =============================================================================

def load_inputs():
    for p in [
        FROZEN_META,
        STD24,
        A2_FEATURE,
        A2_BYLINEAGE,
        A2_SUPPORT,
        UMAP_LINEAGE_SRC,
        V3_ONTOLOGY,
        V3_FORMAL,
        V3_MODULE,
    ]:
        require(p)

    meta = pd.read_csv(FROZEN_META, sep="\t", low_memory=False)
    std = pd.read_csv(STD24, sep="\t", low_memory=False)
    umap = pd.read_csv(UMAP_LINEAGE_SRC, low_memory=False)

    if "group_id" not in meta.columns or "group_id" not in std.columns:
        raise ValueError("Frozen metadata/standardized input must contain group_id.")
    if meta["group_id"].duplicated().any() or std["group_id"].duplicated().any():
        raise ValueError("group_id must be unique in frozen metadata and standardized input.")

    # Add any article-clean metadata columns that are not in the frozen metadata.
    if "group_id" in umap.columns:
        extra_cols = [
            c for c in [
                "group_id",
                "cell_type",
                "broad_cell_class",
                "tissue",
                "donor_id",
                "major_lineage",
                "priority_lineage_corrected",
                "n_cells",
            ]
            if c in umap.columns
        ]
        if extra_cols:
            u = umap[extra_cols].drop_duplicates("group_id")
            add = [c for c in u.columns if c != "group_id" and c not in meta.columns]
            if add:
                meta = meta.merge(u[["group_id"] + add], on="group_id", how="left", validate="one_to_one")

    # Normalize lineage field without altering the frozen source.
    if "priority_lineage_corrected" in meta.columns:
        meta["major_lineage_current"] = meta["priority_lineage_corrected"]
    elif "major_lineage" in meta.columns:
        meta["major_lineage_current"] = meta["major_lineage"]
    elif "priority_lineage" in meta.columns:
        meta["major_lineage_current"] = meta["priority_lineage"]
    else:
        raise ValueError("No current major-lineage column found in frozen/article-clean metadata.")

    # Required descriptive metadata.
    for col in ["cell_type", "tissue", "donor_id"]:
        if col not in meta.columns:
            raise ValueError(f"Required metadata column missing: {col}")

    joined = meta.merge(std, on="group_id", how="inner", validate="one_to_one")
    if len(joined) != len(std):
        raise RuntimeError(
            f"Metadata/standardized matrix mismatch: metadata-joined={len(joined)}, standardized={len(std)}"
        )

    feature_cols = [c for c in std.columns if c != "group_id"]
    if len(feature_cols) != 24:
        raise RuntimeError(
            f"Expected frozen 24-feature clock architecture; found {len(feature_cols)} features."
        )

    X = joined[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Frozen standardized 24-feature matrix contains non-finite values.")

    a2f = pd.read_csv(A2_FEATURE, sep="\t", low_memory=False)
    a2l = pd.read_csv(A2_BYLINEAGE, sep="\t", low_memory=False)
    support = pd.read_csv(A2_SUPPORT, sep="\t", low_memory=False)

    ontology = pd.read_csv(V3_ONTOLOGY, low_memory=False)
    formal = pd.read_csv(V3_FORMAL, low_memory=False)
    module_summary = pd.read_csv(V3_MODULE, low_memory=False)

    if len(formal) != 127:
        raise RuntimeError(f"Expected 127 frozen formal-hit rows; found {len(formal)}")
    if ontology[["collection", "pathway"]].drop_duplicates().shape[0] != 98:
        raise RuntimeError("Expected 98 frozen formal pathways.")
    if ontology["complete_linkage_cluster_id"].nunique() != 69:
        raise RuntimeError("Expected 69 frozen complete-linkage families.")

    return joined, feature_cols, umap, a2f, a2l, support, ontology, formal, module_summary


# =============================================================================
# C01 — GLOBAL CLOCK-FEATURE LANDSCAPE + LINEAGE ZOOM-INS
# =============================================================================

def build_global_feature_landscape(joined: pd.DataFrame, feature_cols: list[str]):
    outdir = DIR["global"]
    outdir.mkdir(parents=True, exist_ok=True)

    # Equal weight to each donor × tissue × original-cell-type pseudobulk.
    g = joined.groupby("cell_type", observed=True)

    means = g[feature_cols].mean()
    support = g.agg(
        n_pseudobulks=("group_id", "size"),
        n_donors=("donor_id", "nunique"),
        n_tissues=("tissue", "nunique"),
    )

    annotations = []
    for ct, part in g:
        current = part["major_lineage_current"].dropna().astype(str)
        current = current[current.isin(LINEAGE_ORDER)]
        lineage = mode_string(current, "other") if len(current) else "other"

        if "broad_cell_class" in part.columns:
            broad = mode_string(part["broad_cell_class"], lineage)
        else:
            broad = LINEAGE_DISPLAY.get(lineage, "Other")

        annotations.append({
            "cell_type": str(ct),
            "broad_cell_class": broad,
            "major_lineage": lineage,
        })

    ann = pd.DataFrame(annotations).set_index("cell_type")
    tab = means.join(support).join(ann)

    # Stable atlas-like ordering: broad class -> current major lineage -> cell type.
    broad_order = (
        tab.groupby("broad_cell_class")
        .size()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    broad_rank = {x: i for i, x in enumerate(broad_order)}
    lineage_rank = {x: i for i, x in enumerate(LINEAGE_ORDER + ["other"])}

    tab["_broad_rank"] = tab["broad_cell_class"].map(broad_rank).fillna(999)
    tab["_lineage_rank"] = tab["major_lineage"].map(lineage_rank).fillna(999)
    tab = tab.sort_values(["_broad_rank", "_lineage_rank"], kind="stable")
    tab = tab.sort_index(kind="stable") if False else tab

    # Within each broad class, preserve alphabetical cell-type order.
    ordered_parts = []
    for bc in broad_order:
        d = tab[tab["broad_cell_class"].astype(str).eq(str(bc))].copy()
        d = d.sort_index(key=lambda x: x.astype(str).str.lower())
        ordered_parts.append(d)
    tab = pd.concat(ordered_parts, axis=0) if ordered_parts else tab

    source = tab.reset_index().rename(columns={"index": "cell_type"})
    source["row_order"] = np.arange(len(source))
    write_csv(source, "C01_global_clock_feature_landscape.csv")

    # Broad-class side colors.
    broad_categories = list(dict.fromkeys(source["broad_cell_class"].astype(str)))
    broad_colors = categorical_palette(broad_categories)
    broad_color_table = pd.DataFrame({
        "broad_cell_class": broad_categories,
        "color": [broad_colors[x] for x in broad_categories],
    })
    write_csv(broad_color_table, "C01_broad_cell_class_palette.csv")

    M = source[feature_cols].to_numpy(float)
    vmax = max(1.5, float(np.nanpercentile(np.abs(M), 98)))
    vmax = min(vmax, 3.0)

    # Global overview.
    fig = plt.figure(figsize=(11.8, 12.6))
    ax_bar = fig.add_axes([0.055, 0.12, 0.018, 0.80])
    ax = fig.add_axes([0.085, 0.12, 0.62, 0.80])
    cax = fig.add_axes([0.72, 0.70, 0.014, 0.20])
    lax = fig.add_axes([0.76, 0.11, 0.23, 0.80])
    lax.axis("off")

    im = ax.imshow(
        M,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(
        [pretty_feature(x) for x in feature_cols],
        rotation=65,
        ha="right",
        fontsize=6.2,
    )
    ax.set_xlabel("Frozen 24-feature clock-associated architecture")
    ax.set_ylabel("Original cell types")
    ax.set_title(
        "Global static clock-associated feature landscape",
        pad=10,
    )
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Side annotation.
    rgba = np.array([
        mpl.colors.to_rgba(broad_colors[str(x)])
        for x in source["broad_cell_class"].astype(str)
    ]).reshape(len(source), 1, 4)
    ax_bar.imshow(rgba, aspect="auto", interpolation="nearest")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for s in ax_bar.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mean standardized feature value")

    handles = [
        Patch(facecolor=broad_colors[x], edgecolor="none", label=x)
        for x in broad_categories
    ]
    lax.legend(
        handles=handles,
        title="Broad cell class",
        frameon=False,
        loc="upper left",
        ncol=1 if len(handles) <= 18 else 2,
        fontsize=6.0,
        title_fontsize=7.0,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.45,
        borderaxespad=0,
    )

    save_fig(fig, outdir, "C01A_global_clock_feature_landscape")

    # Five prespecified major-lineage zoom-ins.
    zoom_rows = []
    zoom_sources = {}
    for lin in LINEAGE_ORDER:
        cts = (
            source.loc[source["major_lineage"].astype(str).eq(lin), "cell_type"]
            .astype(str)
            .tolist()
        )
        if not cts:
            continue

        d = means.loc[[x for x in cts if x in means.index]].copy()
        labels = d.index.astype(str).tolist()

        order = cluster_rows(d.to_numpy(float), labels)
        d = d.iloc[order]
        labels = [labels[i] for i in order]

        zsrc = d.reset_index().rename(columns={"index": "cell_type"})
        zsrc.insert(0, "major_lineage", lin)
        zsrc["row_order"] = np.arange(len(zsrc))
        write_csv(
            zsrc,
            f"C01_zoom_{lin}.csv",
        )
        zoom_sources[lin] = (d, labels)

        fig_h = max(3.0, min(12.0, 1.4 + 0.30 * len(labels)))
        fig, ax = plt.subplots(figsize=(9.2, fig_h))
        im = ax.imshow(
            d.to_numpy(float),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(
            [pretty_feature(x) for x in feature_cols],
            rotation=65,
            ha="right",
            fontsize=6,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.3)
        ax.set_title(
            f"{LINEAGE_DISPLAY[lin]} — original cell-type zoom",
            pad=9,
        )
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Mean standardized feature value")
        fig.subplots_adjust(
            left=min(0.46, max(0.24, 0.18 + 0.006 * max(map(len, labels)))),
            right=0.93,
            bottom=0.25,
            top=0.90,
        )
        save_fig(fig, outdir, f"C01B_zoom_{lin}")

        zoom_rows.append({
            "major_lineage": lin,
            "n_original_cell_types": len(labels),
            "ordering": "average-linkage visualization" if SCIPY_CLUSTER else "alphabetical/fallback",
        })

    write_csv(pd.DataFrame(zoom_rows), "C01_zoom_manifest.csv")

    # Compact five-lineage overview using mean cell-type profiles.
    available = [x for x in LINEAGE_ORDER if x in zoom_sources]
    if available:
        fig, axes = plt.subplots(
            len(available),
            1,
            figsize=(10.5, max(6.0, 2.1 * len(available))),
            squeeze=False,
        )
        for ax, lin in zip(axes[:, 0], available):
            d, labels = zoom_sources[lin]
            im = ax.imshow(
                d.to_numpy(float),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=5.0)
            ax.set_xticks(range(len(feature_cols)))
            ax.set_xticklabels(
                [pretty_feature(x) for x in feature_cols]
                if lin == available[-1] else [],
                rotation=65,
                ha="right",
                fontsize=5.5,
            )
            ax.set_title(LINEAGE_DISPLAY[lin], loc="left", fontsize=9, pad=3)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)

        fig.suptitle(
            "Prespecified major-lineage zoom-ins of the global clock-associated landscape",
            y=0.995,
            fontsize=11,
        )
        fig.subplots_adjust(
            left=0.30,
            right=0.88,
            bottom=0.12,
            top=0.96,
            hspace=0.42,
        )
        cax = fig.add_axes([0.91, 0.20, 0.014, 0.55])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Mean standardized feature value")
        save_fig(fig, outdir, "C01C_five_lineage_zoom_overview")


# =============================================================================
# C02 — SAME-COORDINATE LINEAGE / TISSUE / DONOR UMAP
# =============================================================================

def build_same_coordinate_umaps(umap: pd.DataFrame):
    outdir = DIR["umap"]
    outdir.mkdir(parents=True, exist_ok=True)

    req = {"UMAP1", "UMAP2", "tissue", "donor_id"}
    if not req.issubset(umap.columns):
        raise ValueError(f"Article-clean UMAP source missing columns: {sorted(req - set(umap.columns))}")

    d = umap.copy()

    lineage_col = None
    for c in ["major_lineage", "priority_lineage_corrected", "priority_lineage", "major_lineage_current"]:
        if c in d.columns:
            lineage_col = c
            break
    if lineage_col is None:
        raise ValueError("Article-clean UMAP source has no lineage field.")

    d["plot_lineage"] = d[lineage_col].astype(object)
    d["plot_lineage"] = d["plot_lineage"].where(
        d["plot_lineage"].astype(str).isin(LINEAGE_ORDER),
        "other",
    )

    # Save one source table proving all panels share exactly the same coordinates.
    source_cols = [
        c for c in [
            "group_id", "UMAP1", "UMAP2", "plot_lineage",
            "tissue", "donor_id", "cell_type"
        ]
        if c in d.columns
    ]
    write_csv(d[source_cols], "C02_same_coordinate_UMAP_source.csv")

    # Full tissue palette is exported even though a huge legend is not forced into the panel.
    tissues = sorted(d["tissue"].astype(str).unique())
    tissue_palette = categorical_palette(tissues)
    write_csv(
        pd.DataFrame({
            "tissue": tissues,
            "color": [tissue_palette[x] for x in tissues],
            "n_points": [int((d["tissue"].astype(str) == x).sum()) for x in tissues],
        }),
        "C02_tissue_palette.csv",
    )

    # Lineage vs tissue on the same frozen UMAP coordinates.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1))
    for lin in ["other"] + LINEAGE_ORDER:
        q = d[d["plot_lineage"].eq(lin)]
        if q.empty:
            continue
        axes[0].scatter(
            q["UMAP1"], q["UMAP2"],
            s=8.5,
            color=LINEAGE_COLORS.get(lin, COL["gray"]),
            alpha=0.20 if lin == "other" else 0.78,
            edgecolors="none",
            rasterized=True,
        )
    axes[0].set_title("Major lineage")
    handles = [
        Line2D([], [], marker="o", ls="", markersize=4.5,
               color=LINEAGE_COLORS[x], label=LINEAGE_DISPLAY[x])
        for x in ["other"] + LINEAGE_ORDER
        if (d["plot_lineage"] == x).any()
    ]
    axes[0].legend(handles=handles, frameon=False, loc="best", fontsize=6.5)

    # All tissues, same coordinates.
    for tissue in tissues:
        q = d[d["tissue"].astype(str).eq(tissue)]
        axes[1].scatter(
            q["UMAP1"], q["UMAP2"],
            s=7.5,
            color=tissue_palette[tissue],
            alpha=0.65,
            edgecolors="none",
            rasterized=True,
        )
    axes[1].set_title(f"Tissue context ({len(tissues)} tissues)")

    xlim = (
        min(float(d["UMAP1"].min()), float(d["UMAP1"].min())),
        max(float(d["UMAP1"].max()), float(d["UMAP1"].max())),
    )
    ylim = (
        min(float(d["UMAP2"].min()), float(d["UMAP2"].min())),
        max(float(d["UMAP2"].max()), float(d["UMAP2"].max())),
    )

    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        clean_axis(ax)

    fig.suptitle(
        "Same frozen static clock-architecture UMAP coordinates",
        y=0.98,
        fontsize=11,
    )
    fig.text(
        0.74, 0.015,
        "Full tissue-color mapping is exported in source_data; no UMAP refit.",
        ha="center", va="bottom", fontsize=6.6,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.88, wspace=0.20)
    save_fig(fig, outdir, "C02A_same_coordinates_lineage_vs_tissue")

    # Deterministic top-20 tissue labeled version, all others gray.
    top_tissues = (
        d["tissue"].astype(str).value_counts()
        .sort_values(ascending=False)
        .head(20)
        .index.tolist()
    )
    top_palette = categorical_palette(top_tissues)
    fig = plt.figure(figsize=(8.3, 5.2))
    ax = fig.add_axes([0.10, 0.14, 0.62, 0.76])
    lax = fig.add_axes([0.75, 0.13, 0.24, 0.76])
    lax.axis("off")

    rest = d[~d["tissue"].astype(str).isin(top_tissues)]
    ax.scatter(
        rest["UMAP1"], rest["UMAP2"],
        s=7, color="#D5D5D5", alpha=0.30,
        edgecolors="none", rasterized=True,
    )
    for t in top_tissues:
        q = d[d["tissue"].astype(str).eq(t)]
        ax.scatter(
            q["UMAP1"], q["UMAP2"],
            s=8.5, color=top_palette[t], alpha=0.75,
            edgecolors="none", rasterized=True,
        )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Same UMAP — 20 most represented tissues highlighted", pad=9)
    clean_axis(ax)

    handles = [
        Line2D([], [], marker="o", ls="", markersize=4,
               color=top_palette[t], label=f"{t} (n={(d['tissue'].astype(str)==t).sum()})")
        for t in top_tissues
    ]
    lax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        fontsize=6.0,
        ncol=1,
        borderaxespad=0,
        labelspacing=0.45,
    )
    save_fig(fig, outdir, "C02B_top20_tissues_labeled")

    # Donor coloring — manageable 24-donor atlas context view.
    donors = sorted(d["donor_id"].astype(str).unique())
    donor_palette = categorical_palette(donors)
    write_csv(
        pd.DataFrame({
            "donor_id": donors,
            "color": [donor_palette[x] for x in donors],
            "n_points": [int((d["donor_id"].astype(str) == x).sum()) for x in donors],
        }),
        "C02_donor_palette.csv",
    )

    fig = plt.figure(figsize=(8.8, 5.2))
    ax = fig.add_axes([0.09, 0.13, 0.62, 0.77])
    lax = fig.add_axes([0.74, 0.12, 0.25, 0.78])
    lax.axis("off")
    for donor in donors:
        q = d[d["donor_id"].astype(str).eq(donor)]
        ax.scatter(
            q["UMAP1"], q["UMAP2"],
            s=8, color=donor_palette[donor],
            alpha=0.70, edgecolors="none", rasterized=True,
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Same frozen UMAP — donor context", pad=9)
    clean_axis(ax)
    handles = [
        Line2D([], [], marker="o", ls="", markersize=4,
               color=donor_palette[x], label=x)
        for x in donors
    ]
    lax.legend(
        handles=handles,
        title="Donor",
        frameon=False,
        loc="upper left",
        fontsize=6.0,
        title_fontsize=7.0,
        ncol=2 if len(handles) > 14 else 1,
        borderaxespad=0,
    )
    save_fig(fig, outdir, "C02C_same_coordinates_by_donor")


# =============================================================================
# C03 — CURRENT CORRECTED 24-FEATURE ARCHITECTURE SPECTRUM
# =============================================================================

def build_feature_architecture_spectrum(a2f: pd.DataFrame, a2l: pd.DataFrame):
    outdir = DIR["feature"]
    outdir.mkdir(parents=True, exist_ok=True)

    if "feature" not in a2f.columns:
        raise ValueError("A2 corrected feature table lacks 'feature'.")

    # This uses only already-computed corrected A2 quantities.
    required = {
        "lineage_stability",
        "tissue_remodeling",
        "cell_identity_unique_adj_r2",
        "tissue_context_unique_adj_r2",
        "donor_unique_adj_r2",
        "cell_identity_tissue_context_interaction_adj_r2",
    }
    miss = required - set(a2f.columns)
    if miss:
        raise ValueError(f"A2 corrected feature table missing: {sorted(miss)}")

    d = a2f.copy()
    d["display_feature"] = d["feature"].map(pretty_feature)
    d["feature_type"] = np.where(
        d["feature"].astype(str).str.startswith("expr_"),
        "Gene expression",
        "Derived clock feature",
    )
    write_csv(d, "C03_corrected_A2_feature_architecture_source.csv")

    # A: backbone vs remodeling.
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for typ, part in d.groupby("feature_type", observed=True):
        marker = "o" if typ == "Gene expression" else "s"
        color = COL["blue"] if typ == "Gene expression" else COL["orange"]
        ax.scatter(
            part["lineage_stability"],
            part["tissue_remodeling"],
            s=46,
            marker=marker,
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidth=0.45,
            label=typ,
            zorder=3,
        )

    # Label all 24 — small atlas panel; deterministic alternating offsets reduce overlap.
    for i, r in d.reset_index(drop=True).iterrows():
        dx = 4 if i % 2 == 0 else -4
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            r["display_feature"],
            (r["lineage_stability"], r["tissue_remodeling"]),
            xytext=(dx, 3 + (i % 3) * 2),
            textcoords="offset points",
            fontsize=6.0,
            ha=ha,
            va="bottom",
        )

    ax.set_xlabel("Tissue-held-out lineage stability")
    ax.set_ylabel("Within-lineage tissue remodeling")
    ax.set_title("Clock-associated feature spectrum: lineage backbone vs tissue remodeling")
    clean_axis(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_fig(fig, outdir, "C03A_backbone_vs_tissue_remodeling")

    # B: current fixed-effect architecture, no new model fitting.
    effect_cols = [
        "cell_identity_unique_adj_r2",
        "tissue_context_unique_adj_r2",
        "donor_unique_adj_r2",
        "cell_identity_tissue_context_interaction_adj_r2",
    ]
    effect_labels = [
        "Cell identity",
        "Tissue",
        "Donor",
        "Cell identity × tissue",
    ]

    # Order by existing lineage stability, then tissue remodeling.
    dd = d.sort_values(
        ["lineage_stability", "tissue_remodeling"],
        ascending=[False, False],
        kind="stable",
    )
    M = dd[effect_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    lim = max(0.05, float(np.nanpercentile(np.abs(M), 98)))

    fig, ax = plt.subplots(figsize=(6.7, 8.4))
    im = ax.imshow(
        M,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(effect_labels)))
    ax.set_xticklabels(effect_labels, rotation=28, ha="right")
    ax.set_yticks(range(len(dd)))
    ax.set_yticklabels(dd["display_feature"], fontsize=6.5)
    ax.set_title("Feature-level static architecture from corrected A2 fixed-effect scores")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
    cbar.set_label("Unique / interaction adjusted R²")
    fig.subplots_adjust(left=0.29, right=0.90, bottom=0.18, top=0.92)
    save_fig(fig, outdir, "C03B_feature_fixed_effect_architecture")

    # C: lineage-specific tissue remodeling by feature.
    if {"lineage", "feature", "tissue_unique_adj_r2"}.issubset(a2l.columns):
        piv = a2l.pivot_table(
            index="feature",
            columns="lineage",
            values="tissue_unique_adj_r2",
            aggfunc="first",
        )
        ordered_features = dd["feature"].astype(str).tolist()
        cols = [x for x in LINEAGE_ORDER if x in piv.columns]
        piv = piv.reindex(index=ordered_features, columns=cols)

        write_csv(
            piv.reset_index(),
            "C03_lineage_specific_tissue_effect_matrix.csv",
        )

        vals = piv.to_numpy(float)
        lim2 = max(0.05, float(np.nanpercentile(np.abs(vals), 98)))
        fig, ax = plt.subplots(figsize=(6.8, 8.2))
        im = ax.imshow(
            np.ma.masked_invalid(vals),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-lim2,
            vmax=lim2,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(
            [LINEAGE_DISPLAY.get(x, x) for x in cols],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([pretty_feature(x) for x in piv.index], fontsize=6.5)
        ax.set_title("Lineage-specific tissue contribution across clock-associated features")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
        cbar.set_label("Tissue unique adjusted R²")
        fig.subplots_adjust(left=0.29, right=0.90, bottom=0.18, top=0.92)
        save_fig(fig, outdir, "C03C_lineage_specific_tissue_effects")


# =============================================================================
# C04 — WHERE DOES STATIC REMODELING OCCUR?
# =============================================================================

def supported_lineage_tissue_keys(support: pd.DataFrame) -> set[tuple[str, str]]:
    lin_col = "lineage" if "lineage" in support.columns else None
    if lin_col is None:
        for c in ["major_lineage", "priority_lineage"]:
            if c in support.columns:
                lin_col = c
                break
    if lin_col is None or "tissue" not in support.columns:
        raise ValueError("Corrected A2 support table lacks lineage/tissue columns.")

    s = support.copy()
    if "adequate_support" in s.columns:
        keep = s["adequate_support"].astype(bool)
    elif "n_donors" in s.columns:
        keep = pd.to_numeric(s["n_donors"], errors="coerce").ge(3)
    else:
        # The corrected A2 table itself is already a support table;
        # if no explicit flag/count exists, retain listed rows.
        keep = np.ones(len(s), dtype=bool)

    return set(
        zip(
            s.loc[keep, lin_col].astype(str),
            s.loc[keep, "tissue"].astype(str),
        )
    )


def build_tissue_lineage_remodeling(
    joined: pd.DataFrame,
    feature_cols: list[str],
    support: pd.DataFrame,
):
    outdir = DIR["remodel"]
    outdir.mkdir(parents=True, exist_ok=True)

    d = joined[
        joined["major_lineage_current"].astype(str).isin(LINEAGE_ORDER)
    ].copy()
    d["major_lineage_current"] = d["major_lineage_current"].astype(str)

    supported_keys = supported_lineage_tissue_keys(support)

    # donor × tissue × major-lineage aggregate: equal mean across original
    # pseudobulk rows. This is descriptive and uses frozen standardized values.
    agg = (
        d.groupby(
            ["donor_id", "tissue", "major_lineage_current"],
            observed=True,
        )[feature_cols]
        .mean()
        .reset_index()
    )

    # Leave-one-donor lineage reference, computed from frozen pseudobulk vectors.
    # This is descriptive re-expression of the already frozen phenotype concept.
    rows = []
    for r in agg.itertuples(index=False):
        donor = str(r.donor_id)
        tissue = str(r.tissue)
        lin = str(r.major_lineage_current)

        if (lin, tissue) not in supported_keys:
            continue

        ref = d[
            d["major_lineage_current"].astype(str).eq(lin)
            & ~d["donor_id"].astype(str).eq(donor)
        ]
        if ref.empty:
            continue

        ref_vec = ref[feature_cols].mean(axis=0).to_numpy(float)
        vec = np.array([getattr(r, c) for c in feature_cols], dtype=float)
        dist = float(np.linalg.norm(vec - ref_vec))

        rows.append({
            "donor_id": donor,
            "tissue": tissue,
            "major_lineage": lin,
            "clock_architecture_distance_to_leave_one_donor_lineage_reference": dist,
        })

    phen = pd.DataFrame(rows)
    if phen.empty:
        raise RuntimeError("No supported tissue × lineage remodeling phenotypes could be constructed.")

    summary = (
        phen.groupby(["tissue", "major_lineage"], observed=True)
        .agg(
            median_remodeling=(
                "clock_architecture_distance_to_leave_one_donor_lineage_reference",
                "median",
            ),
            q25=(
                "clock_architecture_distance_to_leave_one_donor_lineage_reference",
                lambda x: float(np.quantile(x, 0.25)),
            ),
            q75=(
                "clock_architecture_distance_to_leave_one_donor_lineage_reference",
                lambda x: float(np.quantile(x, 0.75)),
            ),
            n_donors=("donor_id", "nunique"),
        )
        .reset_index()
    )

    write_csv(phen, "C04_remodeling_aggregate_level_source.csv")
    write_csv(summary, "C04_tissue_lineage_remodeling_summary.csv")

    # Order tissues by coverage first, then overall median remodeling.
    order = (
        summary.groupby("tissue")
        .agg(
            n_supported_lineages=("major_lineage", "nunique"),
            overall_median=("median_remodeling", "median"),
        )
        .sort_values(
            ["n_supported_lineages", "overall_median"],
            ascending=[False, False],
        )
        .index.astype(str)
        .tolist()
    )
    cols = LINEAGE_ORDER
    H = summary.pivot(
        index="tissue",
        columns="major_lineage",
        values="median_remodeling",
    ).reindex(index=order, columns=cols)
    N = summary.pivot(
        index="tissue",
        columns="major_lineage",
        values="n_donors",
    ).reindex(index=order, columns=cols)

    vals = H.to_numpy(float)
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))

    fig_h = max(7.0, min(18.0, 2.0 + 0.26 * len(H)))
    fig, ax = plt.subplots(figsize=(7.1, fig_h))
    im = ax.imshow(
        np.ma.masked_invalid(vals),
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LINEAGE_DISPLAY[x] for x in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(H)))
    ax.set_yticklabels(H.index.astype(str), fontsize=6.2)
    ax.set_title("Where does static clock-associated remodeling occur?")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # donor support shown as small text where feasible
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if np.isfinite(vals[i, j]) and pd.notna(N.iloc[i, j]):
                ax.text(
                    j, i,
                    str(int(N.iloc[i, j])),
                    ha="center", va="center",
                    fontsize=4.6,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.023, pad=0.02)
    cbar.set_label("Median 24D distance to leave-one-donor lineage reference")
    fig.text(
        0.5, 0.01,
        "Numbers in cells = contributing donors; blank = outside corrected A2 support.",
        ha="center", fontsize=6.5,
    )
    fig.subplots_adjust(left=0.32, right=0.91, bottom=0.08, top=0.94)
    save_fig(fig, outdir, "C04A_tissue_lineage_remodeling_heatmap")

    # Dot-map variant: same complete data, size = donor support.
    fig_h = max(7.0, min(18.0, 2.0 + 0.26 * len(H)))
    fig, ax = plt.subplots(figsize=(7.3, fig_h))
    for i, tissue in enumerate(H.index):
        for j, lin in enumerate(cols):
            v = H.loc[tissue, lin]
            n = N.loc[tissue, lin]
            if not np.isfinite(v) or pd.isna(n):
                continue
            size = 12 + 11 * float(n)
            ax.scatter(
                j, i,
                s=size,
                c=[float(v)],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                edgecolors="white",
                linewidth=0.45,
            )
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(H) - 0.4, -0.6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LINEAGE_DISPLAY[x] for x in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(H)))
    ax.set_yticklabels(H.index.astype(str), fontsize=6.2)
    ax.set_title("Tissue × lineage remodeling landscape")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap="viridis",
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.023, pad=0.02)
    cbar.set_label("Median remodeling distance")
    fig.subplots_adjust(left=0.32, right=0.91, bottom=0.08, top=0.94)
    save_fig(fig, outdir, "C04B_tissue_lineage_remodeling_dotmap")


# =============================================================================
# C05 — PATHWAY-FAMILY ONTOLOGY LANDSCAPE
# =============================================================================

def module_order(module_summary: pd.DataFrame) -> list[str]:
    d = module_summary.copy()
    if "v3_primary_module_id" not in d.columns:
        return sorted(
            [x for x in d.iloc[:, 0].astype(str).unique()]
        )
    d["_other"] = d["v3_primary_module_id"].astype(str).eq("M99_OTHER_UNCLASSIFIED")
    sort_cols = ["_other"]
    ascending = [True]
    for c in ["n_v2_redundancy_families", "n_formal_hit_rows"]:
        if c in d.columns:
            sort_cols.append(c)
            ascending.append(False)
    d = d.sort_values(sort_cols, ascending=ascending)
    return d["v3_primary_module_id"].astype(str).tolist()


def derive_family_meta(ontology: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in ontology.groupby("complete_linkage_cluster_id", sort=True):
        rep_path = None
        rep_coll = None
        if "representative_pathway" in g.columns:
            vals = g["representative_pathway"].dropna().astype(str)
            if len(vals):
                rep_path = vals.iloc[0]
        if "representative_collection" in g.columns:
            vals = g["representative_collection"].dropna().astype(str)
            if len(vals):
                rep_coll = vals.iloc[0]

        z = pd.DataFrame()
        if rep_path is not None:
            z = g[g["pathway"].astype(str).eq(rep_path)]
            if rep_coll is not None and "collection" in g.columns:
                z2 = z[z["collection"].astype(str).eq(rep_coll)]
                if not z2.empty:
                    z = z2
        if z.empty:
            z = g.iloc[[0]]
            rep_path = str(z.iloc[0]["pathway"])
            rep_coll = str(z.iloc[0]["collection"])

        mod = (
            str(z.iloc[0]["v3_primary_module_id"])
            if "v3_primary_module_id" in z.columns
            else "M99_OTHER_UNCLASSIFIED"
        )
        label = (
            str(z.iloc[0]["v3_primary_module_label"])
            if "v3_primary_module_label" in z.columns
            else MODULE_SHORT.get(mod, mod)
        )

        rows.append({
            "complete_linkage_cluster_id": str(cid),
            "representative_collection": rep_coll,
            "representative_pathway": rep_path,
            "display_module_id": mod,
            "display_module_label": label,
        })
    return pd.DataFrame(rows)


def build_pathway_ontology(
    ontology: pd.DataFrame,
    formal: pd.DataFrame,
    module_summary: pd.DataFrame,
):
    outdir = DIR["ontology"]
    outdir.mkdir(parents=True, exist_ok=True)

    fam_meta = derive_family_meta(ontology)
    mods = module_order(module_summary)
    mod_rank = {m: i for i, m in enumerate(mods)}

    beta_col = "beta_standardized_pairwise_correspondence"
    if beta_col not in formal.columns:
        raise ValueError(f"Frozen formal table lacks {beta_col}")

    formal = formal.copy()
    formal[beta_col] = pd.to_numeric(formal[beta_col], errors="coerce")

    stats = (
        formal.groupby("complete_linkage_cluster_id", observed=True)
        .agg(
            median_beta=(beta_col, "median"),
            max_beta=(beta_col, "max"),
            n_formal_hits=("pathway", "size"),
            n_formal_cell_units=("cell_unit_id", "nunique"),
            n_unique_pathways=("pathway", "nunique"),
        )
        .reset_index()
    )

    collections = (
        formal.groupby("complete_linkage_cluster_id", observed=True)["collection"]
        .agg(lambda x: ";".join(sorted(set(map(str, x)))))
        .rename("collections_present")
        .reset_index()
    )

    fam = (
        fam_meta
        .merge(stats, on="complete_linkage_cluster_id", how="left")
        .merge(collections, on="complete_linkage_cluster_id", how="left")
    )
    fam["n_formal_hits"] = fam["n_formal_hits"].fillna(0).astype(int)
    fam["n_formal_cell_units"] = fam["n_formal_cell_units"].fillna(0).astype(int)
    fam["n_unique_pathways"] = fam["n_unique_pathways"].fillna(0).astype(int)
    fam["_module_rank"] = fam["display_module_id"].map(mod_rank).fillna(999)
    fam = fam.sort_values(
        ["_module_rank", "n_formal_cell_units", "median_beta", "complete_linkage_cluster_id"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    write_csv(fam.drop(columns="_module_rank"), "C05_family_ontology_summary.csv")

    # Deterministic display subset:
    # up to 2 formal families per non-M99 module, ranked by
    # (cell-unit coverage, median beta, family ID).
    selected_parts = []
    for m in mods:
        if m == "M99_OTHER_UNCLASSIFIED":
            continue
        q = fam[
            fam["display_module_id"].astype(str).eq(m)
            & fam["n_formal_hits"].gt(0)
        ].copy()
        if q.empty:
            continue
        q = q.sort_values(
            ["n_formal_cell_units", "median_beta", "complete_linkage_cluster_id"],
            ascending=[False, False, True],
        ).head(2)
        selected_parts.append(q)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else fam.iloc[0:0].copy()
    write_csv(selected.drop(columns="_module_rank", errors="ignore"), "C05_selected_family_ontology.csv")

    def family_label(r):
        return (
            f"{r['complete_linkage_cluster_id']} | "
            f"{pretty_pathway(r['representative_pathway'])}"
        )

    def marker_for(collections):
        s = set(str(collections).split(";"))
        if {"Hallmark", "Reactome"}.issubset(s):
            return "D"
        if "Hallmark" in s:
            return "o"
        return "s"

    def plot_bubble(df: pd.DataFrame, stem: str, title: str, label_fs=6.0):
        if df.empty:
            return
        dd = df.copy().reset_index(drop=True)
        # Keep module blocks together and strongest/most covered families at top in each block.
        dd["_module_rank"] = dd["display_module_id"].map(mod_rank).fillna(999)
        dd = dd.sort_values(
            ["_module_rank", "n_formal_cell_units", "median_beta", "complete_linkage_cluster_id"],
            ascending=[True, False, False, True],
            kind="stable",
        ).reset_index(drop=True)

        h = max(6.5, min(23.0, 2.0 + 0.30 * len(dd)))
        fig, ax = plt.subplots(figsize=(9.4, h))
        y = np.arange(len(dd))

        # module background bands
        start = 0
        for m, grp in dd.groupby("display_module_id", sort=False):
            idx = grp.index.to_numpy()
            lo = idx.min() - 0.5
            hi = idx.max() + 0.5
            color = MODULE_COLORS.get(str(m), COL["lightgray"])
            ax.axhspan(lo, hi, color=color, alpha=0.09, zorder=0)
            ax.text(
                1.005,
                (lo + hi) / 2,
                MODULE_SHORT.get(str(m), str(m)),
                transform=ax.get_yaxis_transform(),
                ha="left", va="center",
                fontsize=6.2,
                color=color,
            )

        for i, r in dd.iterrows():
            ax.scatter(
                float(r["median_beta"]),
                i,
                s=24 + 26 * float(r["n_formal_cell_units"]),
                marker=marker_for(r["collections_present"]),
                color=MODULE_COLORS.get(str(r["display_module_id"]), COL["gray"]),
                edgecolors="white",
                linewidth=0.55,
                alpha=0.90,
                zorder=3,
            )

        labels = [family_label(r) for _, r in dd.iterrows()]
        ax.set_yticks(y)
        ax.set_yticklabels(textwrap_labels(labels, width=50), fontsize=label_fs)
        ax.set_ylim(len(dd) - 0.4, -0.6)
        ax.set_xlabel("Median standardized pairwise correspondence β")
        ax.set_title(title, pad=10)
        clean_axis(ax)

        shape_handles = [
            Line2D([], [], marker="o", ls="", color=COL["dark"], label="Hallmark"),
            Line2D([], [], marker="s", ls="", color=COL["dark"], label="Reactome"),
            Line2D([], [], marker="D", ls="", color=COL["dark"], label="Both collections"),
        ]
        size_handles = [
            plt.scatter([], [], s=24 + 26*n, color=COL["gray"], edgecolor="white",
                        label=f"{n} cell unit" if n == 1 else f"{n} cell units")
            for n in [1, 3, 6]
        ]
        leg1 = ax.legend(
            handles=shape_handles,
            frameon=False,
            loc="lower right",
            title="Collection representation",
            fontsize=6.2,
            title_fontsize=6.5,
        )
        ax.add_artist(leg1)
        ax.legend(
            handles=size_handles,
            frameon=False,
            loc="upper right",
            title="Formal cell-unit coverage",
            fontsize=6.2,
            title_fontsize=6.5,
        )

        fig.subplots_adjust(
            left=0.46 if len(dd) <= 30 else 0.49,
            right=0.79,
            bottom=0.10,
            top=0.94,
        )
        save_fig(fig, outdir, stem)

    plot_bubble(
        selected,
        "C05A_selected_pathway_family_ontology",
        "Representative coherent pathway families organized by final biological module",
        label_fs=6.5,
    )
    plot_bubble(
        fam[fam["n_formal_hits"].gt(0)],
        "C05B_all_formal_family_ontology",
        "All formal coherent pathway families organized by final biological module",
        label_fs=5.1,
    )

    # Selected family × formal cell-unit heatmap.
    selected_ids = selected["complete_linkage_cluster_id"].astype(str).tolist()
    if selected_ids:
        q = formal[
            formal["complete_linkage_cluster_id"].astype(str).isin(selected_ids)
        ].copy()
        cells = [c for c in CELL_ORDER if c in set(q["cell_unit_id"].astype(str))]
        cells += [
            c for c in q["cell_unit_id"].astype(str).unique()
            if c not in cells
        ]

        H = q.pivot_table(
            index="complete_linkage_cluster_id",
            columns="cell_unit_id",
            values=beta_col,
            aggfunc="median",
        ).reindex(index=selected_ids, columns=cells)

        selmeta = selected.set_index("complete_linkage_cluster_id")
        labels = [
            f"{cid} | {pretty_pathway(selmeta.loc[cid, 'representative_pathway'])}"
            for cid in H.index
        ]

        write_csv(
            H.reset_index(),
            "C05_selected_family_cellunit_matrix.csv",
        )

        vals = H.to_numpy(float)
        bmin = float(np.nanpercentile(formal[beta_col], 2))
        bmax = float(np.nanpercentile(formal[beta_col], 98))
        cmap = mpl.colormaps["viridis"].copy()
        cmap.set_bad("white")

        fig_h = max(6.0, 2.0 + 0.34 * len(H))
        fig, ax = plt.subplots(figsize=(9.0, fig_h))
        im = ax.imshow(
            np.ma.masked_invalid(vals),
            aspect="auto",
            cmap=cmap,
            vmin=bmin,
            vmax=bmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(
            [pretty_cell(x) for x in cells],
            rotation=45,
            ha="right",
            fontsize=6.2,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(textwrap_labels(labels, width=45), fontsize=5.8)
        ax.set_title("Representative coherent families across formal cell units")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("Median standardized correspondence β")
        fig.subplots_adjust(left=0.43, right=0.91, bottom=0.20, top=0.92)
        save_fig(fig, outdir, "C05C_selected_family_cellunit_heatmap")


# =============================================================================
# C06 — A4.7 WITHIN-DONOR / SAME-IDENTITY DESIGN SCHEMATIC
# =============================================================================

def rounded_box(ax, xy, width, height, text, fc, ec=COL["dark"], fs=8):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + width/2,
        y + height/2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        wrap=True,
    )
    return patch


def arrow(ax, start, end):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.0,
        color=COL["dark"],
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


def build_a47_design_schematic():
    outdir = DIR["design"]
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02, 0.96,
        "Within-donor, same-cell-identity pathway–clock correspondence design",
        ha="left", va="top", fontsize=12, weight="bold",
    )
    ax.text(
        0.02, 0.91,
        "Static expression architecture only — no temporal phase/rhythm inference",
        ha="left", va="top", fontsize=7.5, color=COL["gray"],
    )

    # Left: matched biological unit.
    rounded_box(
        ax, (0.04, 0.67), 0.19, 0.13,
        "Same donor\n+\nSame cell identity",
        "#F4F4F4", fs=9,
    )

    rounded_box(
        ax, (0.30, 0.75), 0.17, 0.11,
        "Tissue A",
        "#E9F2FB", ec=COL["blue"], fs=9,
    )
    rounded_box(
        ax, (0.30, 0.57), 0.17, 0.11,
        "Tissue B",
        "#E9F2FB", ec=COL["blue"], fs=9,
    )
    arrow(ax, (0.23, 0.735), (0.30, 0.805))
    arrow(ax, (0.23, 0.735), (0.30, 0.625))

    # Clock vectors.
    rounded_box(
        ax, (0.54, 0.75), 0.19, 0.11,
        "Frozen 24D clock-associated\nstate vector A",
        "#EFF7EE", ec=COL["green"], fs=8,
    )
    rounded_box(
        ax, (0.54, 0.57), 0.19, 0.11,
        "Frozen 24D clock-associated\nstate vector B",
        "#EFF7EE", ec=COL["green"], fs=8,
    )
    arrow(ax, (0.47, 0.805), (0.54, 0.805))
    arrow(ax, (0.47, 0.625), (0.54, 0.625))

    rounded_box(
        ax, (0.79, 0.66), 0.17, 0.13,
        "Δclock\nEuclidean distance\nin frozen 24D space",
        "#E5F3E3", ec=COL["green"], fs=8,
    )
    arrow(ax, (0.73, 0.805), (0.79, 0.735))
    arrow(ax, (0.73, 0.625), (0.79, 0.715))

    # Pathway side.
    rounded_box(
        ax, (0.30, 0.30), 0.17, 0.11,
        "Pathway score A",
        "#FFF4E7", ec=COL["orange"], fs=8.5,
    )
    rounded_box(
        ax, (0.30, 0.14), 0.17, 0.11,
        "Pathway score B",
        "#FFF4E7", ec=COL["orange"], fs=8.5,
    )
    arrow(ax, (0.385, 0.57), (0.385, 0.41))
    arrow(ax, (0.385, 0.30), (0.385, 0.25))

    rounded_box(
        ax, (0.54, 0.20), 0.19, 0.13,
        "Δpathway\nabsolute tissue difference\n(clock genes excluded)",
        "#FDEBD7", ec=COL["orange"], fs=8,
    )
    arrow(ax, (0.47, 0.355), (0.54, 0.285))
    arrow(ax, (0.47, 0.195), (0.54, 0.255))

    rounded_box(
        ax, (0.79, 0.20), 0.17, 0.13,
        "Within-donor\ncorrespondence β\nΔpathway ↔ Δclock",
        "#F3EAF8", ec=COL["purple"], fs=8,
    )
    arrow(ax, (0.73, 0.265), (0.79, 0.265))
    arrow(ax, (0.875, 0.66), (0.875, 0.33))

    ax.text(
        0.04, 0.06,
        "Primary null: within each donor/cell unit, permute tissue labels of the 24D clock states; "
        "reconstruct pairwise clock distances while pathway-distance geometry remains fixed.",
        ha="left", va="bottom", fontsize=7.2,
    )

    save_fig(fig, outdir, "C06_A47_within_donor_same_identity_design")

    # Source/definition table for manuscript traceability.
    write_csv(
        pd.DataFrame([
            {
                "element": "analysis_unit",
                "definition": "unordered tissue pair within the same donor and same cell unit",
            },
            {
                "element": "delta_clock",
                "definition": "Euclidean distance between frozen 24D static clock-associated state vectors",
            },
            {
                "element": "delta_pathway",
                "definition": "absolute difference in pathway score between the two tissues",
            },
            {
                "element": "primary_effect",
                "definition": "positive standardized weighted within-donor pairwise correspondence beta",
            },
            {
                "element": "primary_null",
                "definition": "within-donor/cell-unit tissue-label permutation of the 24D clock states",
            },
        ]),
        "C06_A47_design_definitions.csv",
    )


# =============================================================================
# MANIFEST / README
# =============================================================================

CANDIDATE_FIGURES = [
    {
        "candidate": "C01A",
        "stem": "C01A_global_clock_feature_landscape",
        "concept": "Tabula-style global atlas overview",
        "recommended_role": "Main Figure 1 candidate",
        "status": "DESCRIPTIVE_NEW_RENDERING",
        "claim": "Shows static clock-associated feature architecture across original cell types.",
    },
    {
        "candidate": "C01B/C01C",
        "stem": "C01B_zoom_* / C01C_five_lineage_zoom_overview",
        "concept": "Prespecified lineage zoom-ins",
        "recommended_role": "Main inset or Supplement",
        "status": "DESCRIPTIVE_NEW_RENDERING",
        "claim": "Shows broad/fine cell-type structure without independent-replication claims.",
    },
    {
        "candidate": "C02A",
        "stem": "C02A_same_coordinates_lineage_vs_tissue",
        "concept": "Same-coordinate cell-identity versus tissue display",
        "recommended_role": "Main Figure 1 candidate",
        "status": "RENDERING_OF_FROZEN_COORDINATES",
        "claim": "Visual comparison only; quantitative backbone/remodeling claims still rely on A1/A2.",
    },
    {
        "candidate": "C02B/C02C",
        "stem": "C02B_top20_tissues_labeled / C02C_same_coordinates_by_donor",
        "concept": "Tissue/donor context sensitivity displays",
        "recommended_role": "Supplement",
        "status": "RENDERING_OF_FROZEN_COORDINATES",
        "claim": "Context visualization only.",
    },
    {
        "candidate": "C03A",
        "stem": "C03A_backbone_vs_tissue_remodeling",
        "concept": "24-feature backbone–remodeling spectrum",
        "recommended_role": "Main Figure 1 candidate",
        "status": "RENDERING_OF_EXISTING_CORRECTED_A2_SCORES",
        "claim": "Feature heterogeneity in static lineage stability and tissue remodeling.",
    },
    {
        "candidate": "C03B/C03C",
        "stem": "C03B_feature_fixed_effect_architecture / C03C_lineage_specific_tissue_effects",
        "concept": "Feature-level architecture decomposition",
        "recommended_role": "Main or Supplement",
        "status": "RENDERING_OF_EXISTING_CORRECTED_A2_SCORES",
        "claim": "Uses current corrected fixed-effect scores; not causal variance allocation.",
    },
    {
        "candidate": "C04A/C04B",
        "stem": "C04*_tissue_lineage_remodeling_*",
        "concept": "Where static remodeling occurs",
        "recommended_role": "Main Figure 1 or Supplement",
        "status": "NEW_DESCRIPTIVE_SUMMARY_OF_FROZEN_24D_SPACE",
        "claim": "Descriptive tissue × lineage remodeling magnitude only; no new hypothesis test.",
    },
    {
        "candidate": "C05A",
        "stem": "C05A_selected_pathway_family_ontology",
        "concept": "Tabula-style biological-process ontology map",
        "recommended_role": "Main Figure 2 candidate replacing/augmenting P08",
        "status": "POST_INFERENCE_DESCRIPTIVE_SYNTHESIS",
        "claim": "Deterministic display of existing formal families/modules; no new inferential test.",
    },
    {
        "candidate": "C05B",
        "stem": "C05B_all_formal_family_ontology",
        "concept": "Full coherent-family ontology landscape",
        "recommended_role": "Supplement",
        "status": "POST_INFERENCE_DESCRIPTIVE_SYNTHESIS",
        "claim": "Complete family-level display.",
    },
    {
        "candidate": "C05C",
        "stem": "C05C_selected_family_cellunit_heatmap",
        "concept": "Representative coherent families × formal cell units",
        "recommended_role": "Main Figure 2 candidate",
        "status": "POST_INFERENCE_DESCRIPTIVE_SYNTHESIS",
        "claim": "Shows where representative formal family correspondences occur.",
    },
    {
        "candidate": "C06",
        "stem": "C06_A47_within_donor_same_identity_design",
        "concept": "A4.7 study-design schematic",
        "recommended_role": "Main Figure 2 candidate",
        "status": "METHOD_SCHEMATIC",
        "claim": "Explains the within-donor/same-identity contrast and permutation null.",
    },
]


def write_readme():
    df = pd.DataFrame(CANDIDATE_FIGURES)
    df.to_csv(
        DIR["audit"] / "CANDIDATE_FIGURE_MAP.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# Tabula-style candidate additions",
        "",
        "## Critical status",
        "",
        "- Previous article-facing figures were frozen first and were not modified.",
        "- These are **candidate additions**, not replacements.",
        "- No A1/A2/A4.7/A4.8 formal statistics were rerun or changed.",
        "- The article-clean UMAP coordinates were reused exactly.",
        "- C04 is a new descriptive summary of the already frozen 24D space; it is not a new inferential test.",
        "- C05 uses the frozen 127 formal-hit rows, 98 formal pathways, 69 complete-linkage families, and final v3 ontology.",
        "- Tabula Sapiens is static-expression data; no phase/rhythm/amplitude/oscillation claim.",
        "- Broad/fine agreement is not independent replication.",
        "- Hallmark/Reactome convergence is not independent experimental replication.",
        "",
        "## Candidate set",
        "",
    ]
    for r in CANDIDATE_FIGURES:
        lines += [
            f"### {r['candidate']} — {r['concept']}",
            f"- File stem: `{r['stem']}`",
            f"- Suggested role: {r['recommended_role']}",
            f"- Status: {r['status']}",
            f"- Safe claim: {r['claim']}",
            "",
        ]

    (DIR["audit"] / "README_CANDIDATE_ADDITIONS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace-candidates",
        action="store_true",
        help="Delete and rebuild ONLY the 34_TABULA_STYLE_CANDIDATE_ADDITIONS_20260920 folder. Frozen V1 is never replaced.",
    )
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Create/verify the V1 figure freeze and exit without building new candidate additions.",
    )
    args = parser.parse_args()

    print("=" * 116)
    print("STEP 1 — FREEZE / VERIFY ALL CURRENT ARTICLE-FACING FIGURES")
    print("=" * 116)

    protected_before = verify_freeze_snapshot()
    print("Frozen/verified protected file count:", len(protected_before))
    print("Freeze:", FREEZE)

    if args.freeze_only:
        print("FREEZE-ONLY COMPLETE")
        return

    if OUT.exists():
        if not args.replace_candidates:
            raise RuntimeError(
                f"\nCandidate output already exists:\n{OUT}\n\n"
                "Nothing was overwritten. To intentionally rebuild ONLY the candidate additions, "
                "rerun with --replace-candidates."
            )
        shutil.rmtree(OUT)

    for d in DIR.values():
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 116)
    print("STEP 2 — LOAD FROZEN DATA / EXISTING CURRENT STATISTICS")
    print("=" * 116)

    (
        joined,
        feature_cols,
        umap,
        a2f,
        a2l,
        support,
        ontology,
        formal,
        module_summary,
    ) = load_inputs()

    print("Frozen pseudobulk observations:", len(joined))
    print("Frozen clock-associated features:", len(feature_cols))
    print("Original cell types:", joined["cell_type"].nunique())
    print("Tissues:", joined["tissue"].nunique())
    print("Donors:", joined["donor_id"].nunique())
    print("Formal hit rows:", len(formal))
    print("Unique formal pathways:", formal[["collection", "pathway"]].drop_duplicates().shape[0])
    print("Complete-linkage families:", ontology["complete_linkage_cluster_id"].nunique())

    # Input manifest.
    input_manifest = pd.DataFrame([
        {"input": "frozen_metadata", "path": rel(FROZEN_META), "sha256": sha256(FROZEN_META)},
        {"input": "frozen_24D_standardized_matrix", "path": rel(STD24), "sha256": sha256(STD24)},
        {"input": "corrected_A2_feature_scores", "path": rel(A2_FEATURE), "sha256": sha256(A2_FEATURE)},
        {"input": "corrected_A2_lineage_feature_scores", "path": rel(A2_BYLINEAGE), "sha256": sha256(A2_BYLINEAGE)},
        {"input": "corrected_A2_support", "path": rel(A2_SUPPORT), "sha256": sha256(A2_SUPPORT)},
        {"input": "article_clean_UMAP_source", "path": rel(UMAP_LINEAGE_SRC), "sha256": sha256(UMAP_LINEAGE_SRC)},
        {"input": "v3_pathway_ontology", "path": rel(V3_ONTOLOGY), "sha256": sha256(V3_ONTOLOGY)},
        {"input": "v3_formal_hits", "path": rel(V3_FORMAL), "sha256": sha256(V3_FORMAL)},
        {"input": "v3_module_summary", "path": rel(V3_MODULE), "sha256": sha256(V3_MODULE)},
    ])
    input_manifest.to_csv(
        DIR["audit"] / "INPUT_HASHES.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 116)
    print("STEP 3 — BUILD TABULA-STYLE CANDIDATE ADDITIONS")
    print("=" * 116)

    print("[C01] Global clock-feature atlas + five lineage zoom-ins")
    build_global_feature_landscape(joined, feature_cols)

    print("[C02] Same-coordinate lineage/tissue/donor UMAP")
    build_same_coordinate_umaps(umap)

    print("[C03] Corrected A2 feature architecture spectrum")
    build_feature_architecture_spectrum(a2f, a2l)

    print("[C04] Tissue × lineage remodeling landscape")
    build_tissue_lineage_remodeling(joined, feature_cols, support)

    print("[C05] Pathway-family ontology landscape")
    build_pathway_ontology(ontology, formal, module_summary)

    print("[C06] A4.7 within-donor/same-identity design schematic")
    build_a47_design_schematic()

    write_readme()

    # Candidate output hashes.
    candidate_files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            candidate_files.append({
                "path": str(p.relative_to(OUT)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256(p),
            })
    pd.DataFrame(candidate_files).to_csv(
        DIR["audit"] / "CANDIDATE_OUTPUT_HASHES.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Absolute guardrail: protected figures/data must be byte-identical.
    protected_after = snapshot_tree(PROTECTED_ROOTS)
    if protected_after != protected_before:
        changed = [
            k for k in sorted(set(protected_before) | set(protected_after))
            if protected_before.get(k) != protected_after.get(k)
        ]
        raise RuntimeError(
            "A protected article-facing file changed during candidate generation:\n"
            + "\n".join(changed[:50])
        )

    run_manifest = {
        "status": "COMPLETE",
        "freeze": str(FREEZE),
        "candidate_output": str(OUT),
        "protected_files_verified_unchanged": len(protected_before),
        "old_figures_modified": False,
        "old_statistics_modified": False,
        "A1_A2_A47_A48_formal_statistics_recomputed": False,
        "article_clean_UMAP_refit": False,
        "ontology_modified": False,
        "complete_linkage_families_modified": False,
        "new_descriptive_C04_constructed": True,
        "static_expression_only": True,
        "broad_fine_independent_replication_claim": False,
        "hallmark_reactome_independent_replication_claim": False,
        "candidate_figures": [x["candidate"] for x in CANDIDATE_FIGURES],
    }
    (DIR["audit"] / "RUN_MANIFEST.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 116)
    print("TABULA-STYLE CANDIDATE ADDITIONS COMPLETE")
    print("=" * 116)
    print("FROZEN V1:", FREEZE)
    print("NEW CANDIDATES:", OUT)
    print()
    print("Generated candidate families:")
    print("  C01  Global 24-feature × original-cell-type atlas + 5 lineage zoom-ins")
    print("  C02  Same-coordinate lineage vs tissue UMAP + labeled tissue/donor views")
    print("  C03  24-feature backbone/remodeling + fixed-effect architecture")
    print("  C04  Tissue × lineage remodeling heatmap/dotmap")
    print("  C05  Selected + full pathway-family ontology maps + family×cell-unit heatmap")
    print("  C06  A4.7 within-donor/same-cell-identity design schematic")
    print()
    print("Protected previous article-facing files changed: NO")
    print("Formal statistics recomputed:                  NO")
    print("Article-clean UMAP refit:                      NO")
    print("V3 ontology/families modified:                 NO")
    print("-" * 116)
    print("OPEN FIRST:")
    print(" ", DIR["audit"] / "README_CANDIDATE_ADDITIONS.md")
    print(" ", DIR["audit"] / "CANDIDATE_FIGURE_MAP.csv")
    print(" ", DIR["global"] / "C01A_global_clock_feature_landscape.png")
    print(" ", DIR["umap"] / "C02A_same_coordinates_lineage_vs_tissue.png")
    print(" ", DIR["feature"] / "C03A_backbone_vs_tissue_remodeling.png")
    print(" ", DIR["remodel"] / "C04A_tissue_lineage_remodeling_heatmap.png")
    print(" ", DIR["ontology"] / "C05A_selected_pathway_family_ontology.png")
    print(" ", DIR["design"] / "C06_A47_within_donor_same_identity_design.png")
    print("=" * 116)


if __name__ == "__main__":
    main()
