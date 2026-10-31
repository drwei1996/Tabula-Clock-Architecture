#!/usr/bin/env python
from __future__ import annotations
import os

from pathlib import Path
import hashlib
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    SCIPY = True
except Exception:
    SCIPY = False

# ============================================================================
# PATHS
# ============================================================================
PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

FROZEN_META = ROOT / "00_freeze" / "observation_metadata_corrected_v3_FROZEN.tsv"
STD24 = (
    PROJECT
    / "figures_revised"
    / "clock_architecture_umap"
    / "source_data"
    / "clock_architecture_UMAP_clock_architecture_standardized_input.tsv"
)

# The frozen corrected metadata intentionally contains the analysis mapping but
# does not necessarily retain all original annotation/display columns.
# Recover original cell_type / broad_cell_class from the exact UMAP coordinate
# table generated from the same eligible pseudobulk observations.
ORIGINAL_DISPLAY_META = (
    PROJECT
    / "figures_revised"
    / "clock_architecture_umap"
    / "source_data"
    / "clock_architecture_UMAP_clock_architecture_coordinates.tsv"
)

OUT = ROOT / "34_TABULA_STYLE_CANDIDATE_ADDITIONS_20260920" / "01_GLOBAL_CLOCK_FEATURE_LANDSCAPE_V3"
SRC = OUT / "source_data"
AUDIT = OUT / "audit"

# ============================================================================
# STYLE
# ============================================================================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
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

LINEAGE_ORDER = [
    "b_cell",
    "t_cell",
    "macrophage_monocyte",
    "fibroblast",
    "endothelial",
    "other",
]
LINEAGE_DISPLAY = {
    "b_cell": "B/plasma",
    "t_cell": "T/lymphoid",
    "macrophage_monocyte": "Myeloid",
    "fibroblast": "Stromal",
    "endothelial": "Endothelial",
    "other": "Other",
}
LINEAGE_COLORS = {
    "b_cell": "#4E79A7",
    "t_cell": "#E15759",
    "macrophage_monocyte": "#59A14F",
    "fibroblast": "#F28E2B",
    "endothelial": "#9467BD",
    "other": "#9EA4AA",
}

# a calmer, atlas-like palette for broad classes (top bar)
BROAD_CLASS_COLOR_HINTS = {
    "b/plasma": "#C9D6EB",
    "lymphoid": "#F3C4C6",
    "myeloid/immune": "#D5E9CF",
    "stromal": "#F9D4B5",
    "endothelial": "#D9CCEF",
    "epithelial": "#F1E1B8",
    "muscle": "#D6E8E8",
    "endocrine": "#F7D4DF",
    "neuronal/glial": "#D1D1F0",
    "germ line": "#E8C9B0",
    "other": "#D9D9D9",
}

FIG_EXTS = [".png", ".pdf", ".svg"]

# ============================================================================
# HELPERS
# ============================================================================
def require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_fig(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in FIG_EXTS:
        kwargs = dict(bbox_inches="tight", pad_inches=0.08, facecolor="white")
        if ext == ".png":
            kwargs["dpi"] = 600
        fig.savefig(OUT / f"{stem}{ext}", **kwargs)
    plt.close(fig)


def write_csv(df: pd.DataFrame, name: str):
    SRC.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRC / name, index=False, encoding="utf-8-sig")


def mode_string(s: pd.Series, fallback="other") -> str:
    x = s.dropna().astype(str)
    x = x[~x.str.lower().isin({"", "nan", "none", "na"})]
    if x.empty:
        return fallback
    return str(x.value_counts().index[0])


def pretty_feature(x: str) -> str:
    x = str(x)
    x = x.replace("expr_", "")
    x = x.replace("module_", "")
    x = x.replace("_", " ")
    return x


def short_feature(x: str) -> str:
    x = str(x)
    x = x.replace("expr_", "")
    x = x.replace("module_", "")
    # Keep common clock modules compact
    compact = {
        "positive_limb_mean": "positive limb",
        "negative_limb_mean": "negative limb",
        "auxiliary_clock_mean": "aux clock",
        "core_clock_mean": "core clock",
        "expanded_clock_mean": "expanded clock",
        "positive_negative_balance": "pos/neg balance",
        "positive_inhibition": "pos inhibition",
    }
    return compact.get(x, x)


def cluster_order(matrix: np.ndarray) -> list[int]:
    n = matrix.shape[0]
    if n <= 2 or not SCIPY:
        return list(range(n))
    try:
        z = linkage(matrix, method="average", metric="euclidean", optimal_ordering=True)
        return leaves_list(z).astype(int).tolist()
    except Exception:
        return list(range(n))


def make_sequential_blue_cmap():
    # calm atlas-like white -> light blue -> deep blue
    colors = ["#FFFFFF", "#EAF2FA", "#C9DDF0", "#94B7DA", "#5D89BC", "#2E5E9E"]
    return mpl.colors.LinearSegmentedColormap.from_list("tabula_blue", colors)


def broad_color(name: str) -> str:
    key = str(name).strip().lower()
    return BROAD_CLASS_COLOR_HINTS.get(key, "#D9D9D9")


def wrapped(vals, width=22):
    return ["\n".join(textwrap.wrap(str(v), width=width, break_long_words=False, break_on_hyphens=False)) for v in vals]

# ============================================================================
# LOAD + PREP
# ============================================================================
def load_data():
    require(FROZEN_META)
    require(STD24)
    require(ORIGINAL_DISPLAY_META)

    meta = pd.read_csv(FROZEN_META, sep="\t", low_memory=False)
    std = pd.read_csv(STD24, sep="\t", low_memory=False)
    original_meta = pd.read_csv(ORIGINAL_DISPLAY_META, sep="\t", low_memory=False)

    for name, df in [
        ("frozen metadata", meta),
        ("standardized input", std),
        ("original display metadata", original_meta),
    ]:
        if "group_id" not in df.columns:
            raise ValueError(f"{name} must contain group_id.")
        if df["group_id"].duplicated().any():
            raise ValueError(f"group_id must be unique in {name}.")

    # The exact historical UMAP coordinate table contains the original
    # donor/tissue/cell_type/broad_cell_class metadata for the same pseudobulks.
    needed_original = [
        "group_id",
        "donor_id",
        "tissue",
        "cell_type",
        "broad_cell_class",
        "n_cells",
    ]
    missing_original = [c for c in needed_original if c not in original_meta.columns]
    if missing_original:
        raise ValueError(
            "Original UMAP coordinate metadata is missing required display columns: "
            + ", ".join(missing_original)
        )

    # First align the 24D standardized matrix with the original annotation
    # metadata from the exact same historical eligible-observation table.
    base = std.merge(
        original_meta[needed_original],
        on="group_id",
        how="inner",
        validate="one_to_one",
    )
    if len(base) != len(std):
        raise RuntimeError(
            f"Standardized/original-metadata mismatch: standardized={len(std)}, merged={len(base)}"
        )

    # Then add only the CURRENT corrected lineage mapping from the frozen
    # corrected metadata. We deliberately do not use broad_cell_class as the
    # formal lineage definition.
    lineage_col = None
    for c in ["priority_lineage_corrected", "major_lineage", "priority_lineage"]:
        if c in meta.columns:
            lineage_col = c
            break
    if lineage_col is None:
        raise ValueError(
            "Frozen corrected metadata contains no current major-lineage mapping column."
        )

    corr = meta[["group_id", lineage_col]].copy()
    corr = corr.rename(columns={lineage_col: "current_major_lineage"})
    joined = base.merge(
        corr,
        on="group_id",
        how="left",
        validate="one_to_one",
    )

    if len(joined) != len(std):
        raise RuntimeError(
            f"Final metadata join mismatch: standardized={len(std)}, joined={len(joined)}"
        )

    features = [c for c in std.columns if c != "group_id"]
    if len(features) != 24:
        raise RuntimeError(
            f"Expected 24 frozen clock-associated features, found {len(features)}"
        )

    joined["plot_lineage"] = joined["current_major_lineage"].astype(object)
    joined["plot_lineage"] = joined["plot_lineage"].where(
        joined["plot_lineage"].astype(str).isin(LINEAGE_ORDER[:-1]),
        "other",
    )
    joined["broad_cell_class_clean"] = (
        joined["broad_cell_class"].astype(str).str.strip()
    )

    # Audit that the exact original annotations were recovered rather than
    # fabricated from the corrected lineage map.
    if joined["cell_type"].isna().any():
        raise RuntimeError("Recovered original cell_type contains missing values.")
    if joined["broad_cell_class"].isna().any():
        raise RuntimeError("Recovered original broad_cell_class contains missing values.")

    return joined, features


def build_celltype_matrix(joined: pd.DataFrame, features: list[str]):
    g = joined.groupby("cell_type", observed=True)
    means = g[features].mean()
    support = g.agg(
        n_pseudobulks=("group_id", "size"),
        n_donors=("donor_id", "nunique"),
        n_tissues=("tissue", "nunique"),
    )
    ann_rows = []
    for ct, part in g:
        ann_rows.append({
            "cell_type": str(ct),
            "broad_cell_class": mode_string(part["broad_cell_class_clean"], "other"),
            "major_lineage": mode_string(part["plot_lineage"], "other"),
        })
    ann = pd.DataFrame(ann_rows).set_index("cell_type")
    tab = means.join(support).join(ann)

    # primary block order by broad class abundance; within each broad class use local clustering on 24D profiles
    broad_order = (
        tab.groupby("broad_cell_class")
        .size()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )

    ordered_parts = []
    for bc in broad_order:
        d = tab[tab["broad_cell_class"].astype(str).eq(bc)].copy()
        # sub-order by current major lineage first, then clustered within lineage if enough items
        blocks = []
        present_lineages = [x for x in LINEAGE_ORDER if x in set(d["major_lineage"].astype(str))]
        for lin in present_lineages:
            q = d[d["major_lineage"].astype(str).eq(lin)].copy()
            if q.empty:
                continue
            order = cluster_order(q[features].to_numpy(float))
            q = q.iloc[order]
            blocks.append(q)
        # any missing/other lineages last
        remainder = d[~d.index.isin(pd.concat(blocks).index if blocks else [])].copy()
        if not remainder.empty:
            order = cluster_order(remainder[features].to_numpy(float))
            remainder = remainder.iloc[order]
            blocks.append(remainder)
        ordered_parts.append(pd.concat(blocks) if blocks else d)

    ordered = pd.concat(ordered_parts)
    ordered = ordered.reset_index().rename(columns={"index": "cell_type"})
    ordered["column_order"] = np.arange(len(ordered))

    # blue-style min-max scaled within each feature for display only
    disp = ordered[["cell_type", "broad_cell_class", "major_lineage", "column_order", "n_pseudobulks", "n_donors", "n_tissues"]].copy()
    for f in features:
        v = ordered[f].astype(float).to_numpy()
        lo, hi = np.nanmin(v), np.nanmax(v)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            disp[f] = (v - lo) / (hi - lo)
        else:
            disp[f] = 0.5

    write_csv(ordered, "C01v3_celltype_feature_matrix_zscore_means.csv")
    write_csv(disp, "C01v3_celltype_feature_matrix_display_scaled.csv")

    broad_palette = pd.DataFrame({
        "broad_cell_class": broad_order,
        "color": [broad_color(x) for x in broad_order],
    })
    lineage_palette = pd.DataFrame({
        "major_lineage": list(LINEAGE_DISPLAY.keys()),
        "color": [LINEAGE_COLORS[x] for x in LINEAGE_DISPLAY.keys()],
        "display": [LINEAGE_DISPLAY[x] for x in LINEAGE_DISPLAY.keys()],
    })
    write_csv(broad_palette, "C01v3_broad_class_palette.csv")
    write_csv(lineage_palette, "C01v3_lineage_palette.csv")
    return ordered, disp, features

# ============================================================================
# DRAWING
# ============================================================================
def draw_global(ordered: pd.DataFrame, disp: pd.DataFrame, features: list[str], sequential=False):
    mat = disp[features].to_numpy(float).T if sequential else ordered[features].to_numpy(float).T
    n_feat, n_cell = mat.shape
    broad_classes = ordered["broad_cell_class"].astype(str).tolist()
    lineages = ordered["major_lineage"].astype(str).tolist()

    broad_rgba = np.array([mpl.colors.to_rgba(broad_color(x)) for x in broad_classes]).reshape(1, n_cell, 4)
    lineage_rgba = np.array([mpl.colors.to_rgba(LINEAGE_COLORS.get(x, "#BDBDBD")) for x in lineages]).reshape(1, n_cell, 4)

    fig = plt.figure(figsize=(16.0, 6.8))
    fig.suptitle("Global cell-type landscape of static clock-associated expression architecture", y=0.975, fontsize=12)
    left = 0.085
    width = 0.72
    ax_top1 = fig.add_axes([left, 0.905, width, 0.026])
    ax_top2 = fig.add_axes([left, 0.878, width, 0.026])
    ax = fig.add_axes([left, 0.145, width, 0.733])
    cax = fig.add_axes([0.823, 0.60, 0.015, 0.28])
    lax = fig.add_axes([0.855, 0.14, 0.13, 0.76])
    lax.axis("off")

    cmap = make_sequential_blue_cmap() if sequential else mpl.colormaps["RdBu_r"]
    if sequential:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Relative mean expression\n(within-feature display scaling)")
    else:
        vmax = min(3.0, max(1.5, float(np.nanpercentile(np.abs(mat), 98))))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Mean standardized feature value")

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([short_feature(x) for x in features], fontsize=7.2)
    ax.set_xticks([])
    ax.set_xlabel("Original cell types (ordered within broad-class blocks)", labelpad=8)
    ax.set_ylabel("Frozen 24 clock-associated features")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    counts = ordered.groupby("broad_cell_class").size().tolist()
    cum = np.cumsum(counts)
    for x in cum[:-1]:
        ax.axvline(x - 0.5, color="#E0E0E0", lw=0.9)
        ax_top1.axvline(x - 0.5, color="white", lw=1.1)
        ax_top2.axvline(x - 0.5, color="white", lw=1.1)

    ax_top1.imshow(broad_rgba, aspect="auto", interpolation="nearest")
    ax_top2.imshow(lineage_rgba, aspect="auto", interpolation="nearest")
    for a in [ax_top1, ax_top2]:
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    ax_top1.text(-0.012, 0.5, "Broad\nclass", transform=ax_top1.transAxes, ha="right", va="center", fontsize=6.7)
    ax_top2.text(-0.012, 0.5, "Major\nlineage", transform=ax_top2.transAxes, ha="right", va="center", fontsize=6.7)

    broad_unique = list(dict.fromkeys(broad_classes))
    handles1 = [Patch(facecolor=broad_color(x), edgecolor="none", label=x) for x in broad_unique[:12]]
    handles2 = [Patch(facecolor=LINEAGE_COLORS[x], edgecolor="none", label=LINEAGE_DISPLAY[x]) for x in LINEAGE_DISPLAY]
    leg1 = lax.legend(handles=handles1, title="Broad cell class", frameon=False,
                      loc="upper left", fontsize=6.0, title_fontsize=7.0,
                      borderaxespad=0, labelspacing=0.40, handlelength=1.0)
    lax.add_artist(leg1)
    lax.legend(handles=handles2, title="Current major lineage", frameon=False,
               loc="lower left", fontsize=6.0, title_fontsize=7.0,
               borderaxespad=0, labelspacing=0.45, handlelength=1.0)

    stem = "C01v3_A_global_atlas_sequential_blue" if sequential else "C01v3_A_global_atlas_zscore"
    save_fig(fig, stem)


def lineage_zoom_tables(ordered: pd.DataFrame, disp: pd.DataFrame, features: list[str]):
    chosen = ["b_cell", "t_cell", "macrophage_monocyte", "fibroblast", "endothelial"]
    out = {}
    for lin in chosen:
        q1 = ordered[ordered["major_lineage"].astype(str).eq(lin)].copy()
        q2 = disp[disp["major_lineage"].astype(str).eq(lin)].copy()
        if q1.empty:
            continue
        # preserve the global order already determined
        out[lin] = (q1, q2)
        write_csv(q1, f"C01v3_zoom_{lin}_zscore_means.csv")
        write_csv(q2, f"C01v3_zoom_{lin}_display_scaled.csv")
    return out


def draw_zoom_composite(zoom_map: dict, features: list[str]):
    panels = [
        ("B", "b_cell"),
        ("C", "t_cell"),
        ("D", "macrophage_monocyte"),
        ("E", "fibroblast"),
        ("F", "endothelial"),
    ]
    fig = plt.figure(figsize=(14.5, 10.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], hspace=0.62, wspace=0.55)
    cmap = mpl.colormaps["RdBu_r"]

    positions = {0: gs[0, 0], 1: gs[0, 1], 2: gs[1, 0], 3: gs[1, 1], 4: gs[2, 0]}
    last_im = None
    for idx, (letter, lin) in enumerate(panels):
        if lin not in zoom_map:
            continue
        zmean, zdisp = zoom_map[lin]
        ax = fig.add_subplot(positions[idx])
        mat = zmean[features].to_numpy(float)
        vmax = min(3.0, max(1.5, float(np.nanpercentile(np.abs(mat), 98))))
        last_im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(zmean)))
        ax.set_yticklabels(wrapped(zmean["cell_type"].astype(str).tolist(), 28), fontsize=5.8)
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", pad=2, length=0)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([short_feature(x) for x in features], rotation=60, ha="right", fontsize=5.8)
        ax.tick_params(axis="x", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{letter}  {LINEAGE_DISPLAY[lin]}", loc="left", fontsize=10, pad=6, fontweight="bold")

    ax_blank = fig.add_subplot(gs[2, 1])
    ax_blank.axis("off")
    if last_im is not None:
        cax = fig.add_axes([0.925, 0.17, 0.012, 0.22])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("Mean standardized\nfeature value")
    fig.suptitle("Prespecified major-lineage zoom-ins of the global clock-associated landscape", y=0.985, fontsize=12)
    fig.text(0.90, 0.10, "Same z-score color scale logic as the global panel", ha="center", fontsize=6.5)
    save_fig(fig, "C01v3_BF_five_lineage_zoom_composite")


def draw_full_composite(ordered: pd.DataFrame, disp: pd.DataFrame, features: list[str], zoom_map: dict):
    fig = plt.figure(figsize=(16.8, 13.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.08], hspace=0.24)

    sub = outer[0].subgridspec(3, 2, height_ratios=[0.04, 0.04, 0.92], width_ratios=[0.80, 0.20], hspace=0.0, wspace=0.05)
    ax_top1 = fig.add_subplot(sub[0, 0])
    ax_top2 = fig.add_subplot(sub[1, 0])
    ax = fig.add_subplot(sub[2, 0])
    lax = fig.add_subplot(sub[:, 1])
    lax.axis("off")

    mat = ordered[features].to_numpy(float).T
    broad_classes = ordered["broad_cell_class"].astype(str).tolist()
    lineages = ordered["major_lineage"].astype(str).tolist()
    broad_rgba = np.array([mpl.colors.to_rgba(broad_color(x)) for x in broad_classes]).reshape(1, len(broad_classes), 4)
    lineage_rgba = np.array([mpl.colors.to_rgba(LINEAGE_COLORS.get(x, "#BDBDBD")) for x in lineages]).reshape(1, len(lineages), 4)

    vmax = min(3.0, max(1.5, float(np.nanpercentile(np.abs(mat), 98))))
    im = ax.imshow(mat, aspect="auto", cmap=mpl.colormaps["RdBu_r"], vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([short_feature(x) for x in features], fontsize=6.8)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Original cell types")
    ax.set_ylabel("Clock-associated features")
    ax.set_title("A", loc="left", fontsize=14, fontweight="bold", pad=2)

    counts = ordered.groupby("broad_cell_class").size().tolist()
    cum = np.cumsum(counts)
    for x in cum[:-1]:
        ax.axvline(x - 0.5, color="#E4E4E4", lw=0.8)
        ax_top1.axvline(x - 0.5, color="white", lw=1.0)
        ax_top2.axvline(x - 0.5, color="white", lw=1.0)
    ax_top1.imshow(broad_rgba, aspect="auto", interpolation="nearest")
    ax_top2.imshow(lineage_rgba, aspect="auto", interpolation="nearest")
    for a in [ax_top1, ax_top2]:
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    ax_top1.text(-0.01, 0.5, "Broad\nclass", transform=ax_top1.transAxes, ha="right", va="center", fontsize=6.2)
    ax_top2.text(-0.01, 0.5, "Major\nlineage", transform=ax_top2.transAxes, ha="right", va="center", fontsize=6.2)

    broad_unique = list(dict.fromkeys(broad_classes))
    handles1 = [Patch(facecolor=broad_color(x), edgecolor="none", label=x) for x in broad_unique[:12]]
    handles2 = [Patch(facecolor=LINEAGE_COLORS[x], edgecolor="none", label=LINEAGE_DISPLAY[x]) for x in LINEAGE_DISPLAY]
    leg1 = lax.legend(handles=handles1, title="Broad cell class", frameon=False,
                      loc="upper left", fontsize=6.0, title_fontsize=7.0,
                      borderaxespad=0, labelspacing=0.38, handlelength=1.0)
    lax.add_artist(leg1)
    lax.legend(handles=handles2, title="Major lineage", frameon=False,
               loc="lower left", fontsize=6.0, title_fontsize=7.0,
               borderaxespad=0, labelspacing=0.40, handlelength=1.0)
    cax = fig.add_axes([0.685, 0.807, 0.10, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Mean standardized feature value", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    lower = outer[1].subgridspec(2, 3, wspace=0.58, hspace=0.52)
    panel_map = [("B", "b_cell", lower[0,0]), ("C", "t_cell", lower[0,1]), ("D", "macrophage_monocyte", lower[0,2]),
                 ("E", "fibroblast", lower[1,0]), ("F", "endothelial", lower[1,1])]
    for letter, lin, slot in panel_map:
        if lin not in zoom_map:
            continue
        zmean, _ = zoom_map[lin]
        axz = fig.add_subplot(slot)
        matz = zmean[features].to_numpy(float)
        zvmax = min(3.0, max(1.5, float(np.nanpercentile(np.abs(matz), 98))))
        axz.imshow(matz, aspect="auto", cmap=mpl.colormaps["RdBu_r"], vmin=-zvmax, vmax=zvmax, interpolation="nearest")
        axz.set_xticks(range(len(features)))
        axz.set_xticklabels([short_feature(x) for x in features], rotation=60, ha="right", fontsize=5.6)
        axz.set_yticks(range(len(zmean)))
        axz.set_yticklabels(wrapped(zmean["cell_type"].astype(str).tolist(), 24), fontsize=5.5)
        axz.yaxis.tick_right()
        axz.tick_params(length=0, axis="y", pad=2)
        for s in axz.spines.values():
            s.set_visible(False)
        axz.set_title(f"{letter}  {LINEAGE_DISPLAY[lin]}", loc="left", fontsize=10.5, pad=4, fontweight="bold")

    blank = fig.add_subplot(lower[1,2])
    blank.axis("off")
    blank.text(0.03, 0.86,
               "Definitions:\n"
               "Broad cell class: original high-level annotation family\n"
               "(e.g. epithelial, stromal, endothelial, immune-like groups).\n\n"
               "Major lineage: current corrected analysis lineage used in\n"
               "the architecture framework (B/plasma, T/lymphoid, Myeloid,\n"
               "Stromal, Endothelial, Other).\n\n"
               "B–F use the same red-blue z-score color logic as panel A.",
               ha="left", va="top", fontsize=7.2)

    fig.suptitle("Static clock-associated expression architecture across original cell types", y=0.985, fontsize=12.5)
    save_fig(fig, "C01v3_FULL_composite")


def write_manifest(joined: pd.DataFrame, features: list[str], ordered: pd.DataFrame):
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "COMPLETE",
        "input_metadata": str(FROZEN_META),
        "input_standardized_matrix": str(STD24),
        "input_original_display_metadata": str(ORIGINAL_DISPLAY_META),
        "cell_type_broad_class_provenance": "historical UMAP coordinate table from the same eligible pseudobulks",
        "major_lineage_provenance": "current corrected frozen metadata",
        "n_frozen_pseudobulks": int(len(joined)),
        "n_original_cell_types": int(joined["cell_type"].nunique()),
        "n_features": int(len(features)),
        "old_figures_modified": False,
        "statistics_recomputed": False,
        "display_scaling_for_blue_panels": "min-max within each feature across original cell types",
        "global_ordering": "broad class blocks -> current major lineage -> local hierarchical display ordering",
        "outputs": [
            "C01v3_A_global_atlas_sequential_blue",
            "C01v3_A_global_atlas_zscore",
            "C01v3_BF_five_lineage_zoom_composite",
            "C01v3_FULL_composite"
        ],
    }
    (AUDIT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([
        {"file": str(FROZEN_META), "sha256": sha256(FROZEN_META)},
        {"file": str(STD24), "sha256": sha256(STD24)},
        {"file": str(ORIGINAL_DISPLAY_META), "sha256": sha256(ORIGINAL_DISPLAY_META)},
    ]).to_csv(AUDIT / "INPUT_HASHES.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "feature": features,
        "display_label": [short_feature(x) for x in features],
    }).to_csv(AUDIT / "FEATURE_LABELS.csv", index=False, encoding="utf-8-sig")
    # compact block summary
    ordered.groupby(["broad_cell_class", "major_lineage"], observed=True).size().reset_index(name="n_cell_types").to_csv(
        AUDIT / "ORDERED_BLOCK_SUMMARY.csv", index=False, encoding="utf-8-sig"
    )


def main():
    joined, features = load_data()
    ordered, disp, features = build_celltype_matrix(joined, features)
    zoom_map = lineage_zoom_tables(ordered, disp, features)
    draw_global(ordered, disp, features, sequential=False)
    draw_global(ordered, disp, features, sequential=True)
    draw_zoom_composite(zoom_map, features)
    draw_full_composite(ordered, disp, features, zoom_map)
    write_manifest(joined, features, ordered)

    print("=" * 112)
    print("C01 TABULA-STYLE V3 REDRAW COMPLETE")
    print("=" * 112)
    print("OUTPUT:", OUT)
    print()
    print("Generated:")
    print("  C01v3_A_global_atlas_sequential_blue")
    print("  C01v3_A_global_atlas_zscore")
    print("  C01v3_BF_five_lineage_zoom_composite")
    print("  C01v3_FULL_composite")
    print()
    print("Old figures modified: NO")
    print("Statistics recomputed: NO")
    print("OPEN FIRST:")
    print(" ", OUT / "C01v3_FULL_composite.png")
    print(" ", OUT / "C01v3_A_global_atlas_sequential_blue.png")
    print("=" * 112)


if __name__ == "__main__":
    main()
