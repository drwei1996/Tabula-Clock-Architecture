#!/usr/bin/env python
from __future__ import annotations
import os

"""
Rebuild article-facing architecture panels WITHOUT any visible
"Legacy" / "Corrected" terminology.

Principles
----------
1) Old Legacy-vs-Corrected comparison figures are NOT remade for the article.
   They are audit/provenance only and are listed as EXCLUDE_FROM_ARTICLE.
2) Current frozen priority-lineage membership is still used internally, but
   article-facing labels say only "Major lineage".
3) UMAP coordinates are NOT refit. Existing frozen revised UMAP coordinates are
   reused and joined to the frozen current metadata by group_id.
4) PCA is recomputed deterministically from the frozen standardized 24-feature
   static clock-architecture matrix.
5) No inferential statistics are recomputed.
6) Tabula Sapiens is static-expression data; no temporal claims are introduced.
"""

from pathlib import Path
import json
import re
import shutil

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

FREEZE = ROOT / "00_freeze"
FROZEN_META = FREEZE / "observation_metadata_corrected_v3_FROZEN.tsv"

REVISED_SRC = (
    PROJECT
    / "figures_revised"
    / "clock_architecture_umap"
    / "source_data"
)

STANDARDIZED = (
    REVISED_SRC
    / "clock_architecture_UMAP_clock_architecture_standardized_input.tsv"
)

OLD_SUITE = ROOT / "12_COMPLETE_PUBLICATION_FIGURE_SUITE_20260920"
OLD_LEGACY_REVISED = OLD_SUITE / "03_legacy_revised_architecture"
OLD_PANELS = ROOT / "14_PUBLICATION_PANELS_20260920"

OUT = ROOT / "16_ARTICLE_CLEAN_ARCHITECTURE_20260920"
OUT_PCA = OUT / "01_PCA"
OUT_UMAP = OUT / "02_UMAP"
OUT_SRC = OUT / "source_data"
OUT_AUDIT = OUT / "audit"

for d in [OUT_PCA, OUT_UMAP, OUT_SRC, OUT_AUDIT]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STYLE
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

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

FEATURE_LABELS = {
    "clock_architecture": "Original clock-feature matrix",
    "core_clock": "Core clock",
    "expanded_clock": "Expanded clock",
    "clock_modules": "Clock modules",
}

manifest = []

# =============================================================================
# HELPERS
# =============================================================================

def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_figure(fig, stem: str, outdir: Path, source_df=None, note=""):
    outdir.mkdir(parents=True, exist_ok=True)

    for ext in [".png", ".pdf", ".svg"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.10}
        if ext == ".png":
            kwargs["dpi"] = 600
        fig.savefig(outdir / f"{stem}{ext}", **kwargs)

    plt.close(fig)

    if source_df is not None:
        source_df.to_csv(OUT_SRC / f"{stem}.csv", index=False)

    manifest.append({
        "figure": stem,
        "status": "ARTICLE_CLEAN",
        "visible_legacy_corrected_terms": "NO",
        "note": note,
        "output_directory": str(outdir),
    })

def categorical_colors(values):
    cats = [str(x) for x in pd.Series(values).dropna().astype(str).unique()]
    cats = sorted(cats)
    cmap = mpl.colormaps["tab20"]
    colors = {
        cat: mpl.colors.to_hex(cmap(i % 20))
        for i, cat in enumerate(cats)
    }
    return cats, colors

def merge_current_major_lineage(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Use current frozen membership internally.
    Never expose 'corrected' terminology in the article-facing display.
    """
    if "group_id" not in df.columns:
        raise ValueError("UMAP/PCA source must contain group_id.")

    cols = ["group_id", "priority_lineage_corrected", "tissue", "donor_id", "cell_type"]
    cols = [c for c in cols if c in meta.columns]

    right = meta[cols].copy()
    right["group_id"] = right["group_id"].astype(str)

    left = df.copy()
    left["group_id"] = left["group_id"].astype(str)

    # Remove stale display metadata if already present in old coordinate table.
    for c in ["priority_lineage_corrected", "priority_lineage", "tissue", "donor_id", "cell_type"]:
        if c in left.columns and c != "group_id":
            left = left.drop(columns=c)

    out = left.merge(right, on="group_id", how="left", validate="one_to_one")
    out["major_lineage"] = out["priority_lineage_corrected"].fillna("other").astype(str)
    return out

def plot_lineage_embedding(df, x, y, title, stem):
    fig = plt.figure(figsize=(5.6, 4.3))
    ax = fig.add_axes([0.12, 0.14, 0.66, 0.75])
    lax = fig.add_axes([0.80, 0.14, 0.18, 0.73])
    lax.axis("off")

    order = ["other"] + LINEAGE_ORDER
    for lin in order:
        d = df[df["major_lineage"].eq(lin)]
        if d.empty:
            continue
        ax.scatter(
            d[x], d[y],
            s=10,
            color=LINEAGE_COLORS.get(lin, "#B9B9B9"),
            alpha=0.25 if lin == "other" else 0.78,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title, pad=10)
    clean_axis(ax)

    handles = [
        Line2D(
            [], [], marker="o", ls="",
            markersize=4.5,
            color=LINEAGE_COLORS[lin],
            label=LINEAGE_DISPLAY[lin],
        )
        for lin in order
        if (df["major_lineage"] == lin).any()
    ]

    lax.legend(
        handles=handles,
        title="Major lineage",
        frameon=False,
        loc="center left",
        borderaxespad=0,
        title_fontsize=7.5,
        fontsize=7,
    )

    save_figure(
        fig, stem, OUT_UMAP if x.startswith("UMAP") else OUT_PCA,
        df,
        note="Current frozen lineage membership used internally; article-facing label is 'Major lineage'."
    )

def plot_context_embedding(df, x, y, color_col, title, stem):
    if color_col not in df.columns:
        return

    cats, colors = categorical_colors(df[color_col])

    fig = plt.figure(figsize=(6.4, 4.5))
    ax = fig.add_axes([0.10, 0.14, 0.66, 0.75])
    lax = fig.add_axes([0.78, 0.08, 0.21, 0.82])
    lax.axis("off")

    for cat in cats:
        d = df[df[color_col].astype(str).eq(cat)]
        ax.scatter(
            d[x], d[y],
            s=8,
            color=colors[cat],
            alpha=0.68,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title, pad=10)
    clean_axis(ax)

    # Large category legends are context/QC displays, so keep them outside.
    handles = [
        Line2D([], [], marker="o", ls="", markersize=3.8, color=colors[c], label=c)
        for c in cats
    ]

    ncol = 2 if len(cats) > 14 else 1
    legend_title = {
        "tissue": "Tissue",
        "donor_id": "Donor",
        "cell_type": "Original cell type",
    }.get(color_col, color_col)

    lax.legend(
        handles=handles,
        title=legend_title,
        frameon=False,
        loc="center left",
        ncol=ncol,
        borderaxespad=0,
        fontsize=5.6 if len(cats) > 20 else 6.5,
        title_fontsize=7,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    save_figure(
        fig, stem, OUT_UMAP, df,
        note=f"Frozen UMAP coordinates displayed by {legend_title.lower()}."
    )

# =============================================================================
# INPUT CHECKS
# =============================================================================

if not FROZEN_META.exists():
    raise FileNotFoundError(FROZEN_META)
if not STANDARDIZED.exists():
    raise FileNotFoundError(STANDARDIZED)
if not REVISED_SRC.exists():
    raise FileNotFoundError(REVISED_SRC)

meta = pd.read_csv(FROZEN_META, sep="\t", low_memory=False)
if "group_id" not in meta.columns:
    raise ValueError("Frozen metadata is missing group_id.")

# =============================================================================
# 1) CLEAN PCA
# =============================================================================

std = pd.read_csv(STANDARDIZED, sep="\t", low_memory=False)
feature_cols = [c for c in std.columns if c != "group_id"]
X = std[feature_cols].to_numpy(dtype=float)

if not np.isfinite(X).all():
    raise ValueError("Frozen standardized matrix contains non-finite values.")

pca = PCA(n_components=2)
coord = pca.fit_transform(X)

pca_df = pd.DataFrame({
    "group_id": std["group_id"].astype(str),
    "PC1": coord[:, 0],
    "PC2": coord[:, 1],
})

pca_df = merge_current_major_lineage(pca_df, meta)

fig = plt.figure(figsize=(5.6, 4.3))
ax = fig.add_axes([0.12, 0.14, 0.66, 0.75])
lax = fig.add_axes([0.80, 0.14, 0.18, 0.73])
lax.axis("off")

for lin in ["other"] + LINEAGE_ORDER:
    d = pca_df[pca_df["major_lineage"].eq(lin)]
    if d.empty:
        continue
    ax.scatter(
        d["PC1"], d["PC2"],
        s=10,
        color=LINEAGE_COLORS[lin],
        alpha=0.25 if lin == "other" else 0.78,
        edgecolors="none",
        rasterized=True,
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
ax.set_title("Static clock-architecture PCA", pad=10)
clean_axis(ax)

handles = [
    Line2D([], [], marker="o", ls="", markersize=4.5,
           color=LINEAGE_COLORS[lin], label=LINEAGE_DISPLAY[lin])
    for lin in ["other"] + LINEAGE_ORDER
    if (pca_df["major_lineage"] == lin).any()
]
lax.legend(
    handles=handles,
    title="Major lineage",
    frameon=False,
    loc="center left",
    borderaxespad=0,
    title_fontsize=7.5,
    fontsize=7,
)

save_figure(
    fig,
    "A01_clock_architecture_PCA_by_lineage",
    OUT_PCA,
    pca_df,
    note="PCA recomputed from frozen standardized static clock-architecture input."
)

(OUT_AUDIT / "PCA_metadata.json").write_text(
    json.dumps({
        "input": str(STANDARDIZED),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "statistics_recomputed": False,
        "display_only": True,
    }, indent=2),
    encoding="utf-8"
)

# =============================================================================
# 2) CLEAN UMAPS — REUSE FROZEN COORDINATES
# =============================================================================

coord_files = sorted(REVISED_SRC.glob("clock_architecture_UMAP_*_coordinates.tsv"))
if not coord_files:
    raise RuntimeError(f"No UMAP coordinate files found in {REVISED_SRC}")

for coord_path in coord_files:
    m = re.match(r"clock_architecture_UMAP_(.+)_coordinates$", coord_path.stem)
    if not m:
        continue

    set_name = m.group(1)
    label = FEATURE_LABELS.get(set_name, set_name.replace("_", " "))

    d = pd.read_csv(coord_path, sep="\t", low_memory=False)
    if not {"group_id", "UMAP1", "UMAP2"}.issubset(d.columns):
        raise ValueError(f"Missing required columns in {coord_path}")

    d = merge_current_major_lineage(d, meta)

    # Article-facing lineage panel
    plot_lineage_embedding(
        d,
        "UMAP1", "UMAP2",
        f"Clock-architecture UMAP — {label}",
        f"U_{set_name}_by_lineage",
    )

    # Context displays
    plot_context_embedding(
        d,
        "UMAP1", "UMAP2",
        "tissue",
        f"Clock-architecture UMAP — {label} — tissue",
        f"U_{set_name}_by_tissue",
    )

    plot_context_embedding(
        d,
        "UMAP1", "UMAP2",
        "donor_id",
        f"Clock-architecture UMAP — {label} — donor",
        f"U_{set_name}_by_donor",
    )

    if set_name == "clock_architecture" and "cell_type" in d.columns:
        plot_context_embedding(
            d,
            "UMAP1", "UMAP2",
            "cell_type",
            "Clock-architecture UMAP — original cell-type annotations",
            "U_clock_architecture_by_original_cell_type",
        )

# =============================================================================
# 3) ARTICLE EXCLUSION AUDIT
# =============================================================================

exclude_rows = []

# Explicit mapping-comparison figures should never enter the article.
known_exclusions = [
    OLD_SUITE / "02_current_corrected_supplementary" / "C00_lineage_mapping_correction.png",
    OLD_PANELS / "supplementary_panels" / "S01_lineage_mapping_correction.png",
]

for p in known_exclusions:
    exclude_rows.append({
        "path": str(p),
        "reason": "Legacy-vs-current mapping audit only; no biological result.",
        "article_use": "EXCLUDE",
    })

# Inventory any image whose filename itself still contains legacy/corrected.
scan_roots = [OLD_SUITE, ROOT / "13_MANUSCRIPT_FIGURES_20260920", OLD_PANELS]
for base in scan_roots:
    if not base.exists():
        continue
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".png", ".pdf", ".svg"}:
            continue
        low = p.name.lower()
        if "legacy" in low or "corrected" in low:
            exclude_rows.append({
                "path": str(p),
                "reason": "Old article-facing filename contains Legacy/Corrected terminology; use clean rebuild instead where applicable.",
                "article_use": "EXCLUDE_OLD_VERSION",
            })

exclude_df = pd.DataFrame(exclude_rows).drop_duplicates()
exclude_df.to_csv(OUT_AUDIT / "EXCLUDE_FROM_ARTICLE.csv", index=False)

# Clean manifest
manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv(OUT / "ARTICLE_CLEAN_MANIFEST.csv", index=False)

# Guardrail: article-facing figure names must not contain the terms.
bad = []
for p in OUT.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".png", ".pdf", ".svg"}:
        low = p.name.lower()
        if "legacy" in low or "corrected" in low:
            bad.append(str(p))

if bad:
    raise RuntimeError(
        "Article-clean output still contains forbidden filename terminology:\n"
        + "\n".join(bad)
    )

summary = {
    "article_clean_output": str(OUT),
    "pca_panels": len(list(OUT_PCA.glob("*.png"))),
    "umap_panels": len(list(OUT_UMAP.glob("*.png"))),
    "old_versions_flagged_for_exclusion": int(len(exclude_df)),
    "statistics_recomputed": False,
    "umap_refit": False,
    "pca_recomputed_from_frozen_standardized_input": True,
    "visible_legacy_corrected_terms_in_new_figures": False,
}

(OUT_AUDIT / "run_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("=" * 110)
print("ARTICLE-CLEAN ARCHITECTURE REBUILD COMPLETE")
print("=" * 110)
print("OUTPUT:", OUT)
print("PCA panels:", summary["pca_panels"])
print("UMAP panels:", summary["umap_panels"])
print("UMAP refit: NO — frozen coordinates reused")
print("Legacy/Corrected visible terminology in new figures: NO")
print("Mapping-comparison figures: EXCLUDED FROM ARTICLE")
print("EXCLUSION AUDIT:", OUT_AUDIT / "EXCLUDE_FROM_ARTICLE.csv")
print("=" * 110)
