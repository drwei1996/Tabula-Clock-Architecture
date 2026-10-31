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
# Recover original cell_type / donor / tissue metadata from the exact UMAP coordinate
# table generated from the same eligible pseudobulk observations.
# Broad-cell-class annotations are intentionally NOT used in V10.
ORIGINAL_DISPLAY_META = (
    PROJECT
    / "figures_revised"
    / "clock_architecture_umap"
    / "source_data"
    / "clock_architecture_UMAP_clock_architecture_coordinates.tsv"
)

OUT = ROOT / "34_TABULA_STYLE_CANDIDATE_ADDITIONS_20260920" / "01_GLOBAL_CLOCK_FEATURE_LANDSCAPE_V10_LINEAGE_ONLY"
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
    "b_cell": "B cell",
    "t_cell": "T cell",
    "macrophage_monocyte": "Macrophage / monocyte",
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
    "other": "#9EA4AA",
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



def wrapped(vals, width=22):
    return ["\n".join(textwrap.wrap(str(v), width=width, break_long_words=False, break_on_hyphens=False)) for v in vals]

# ============================================================================
# LOAD + PREP
# ============================================================================

def load_data():
    require(FROZEN_META)
    require(STD24)
    require(ORIGINAL_DISPLAY_META)

    meta = pd.read_csv(FROZEN_META, sep="	", low_memory=False)
    std = pd.read_csv(STD24, sep="	", low_memory=False)
    original_meta = pd.read_csv(ORIGINAL_DISPLAY_META, sep="	", low_memory=False)

    for name, df in [
        ("frozen metadata", meta),
        ("standardized input", std),
        ("original display metadata", original_meta),
    ]:
        if "group_id" not in df.columns:
            raise ValueError(f"{name} must contain group_id.")
        if df["group_id"].duplicated().any():
            raise ValueError(f"group_id must be unique in {name}.")

    # V10 intentionally does NOT use either the project-derived or official
    # broad_cell_class annotation. Only original cell_type metadata plus the
    # current corrected Major lineage mapping are used.
    needed_original = [
        "group_id",
        "donor_id",
        "tissue",
        "cell_type",
        "n_cells",
    ]
    missing_original = [c for c in needed_original if c not in original_meta.columns]
    if missing_original:
        raise ValueError(
            "Original UMAP coordinate metadata is missing required display columns: "
            + ", ".join(missing_original)
        )

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

    if joined["cell_type"].isna().any():
        raise RuntimeError("Recovered original cell_type contains missing values.")

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
            "major_lineage": mode_string(part["plot_lineage"], "other"),
        })

    ann = pd.DataFrame(ann_rows).set_index("cell_type")
    tab = means.join(support).join(ann)

    # V10 ordering:
    #   Major lineage blocks only -> local hierarchical ordering within lineage.
    # Broad cell class plays no role in ordering, annotation, legend, or output.
    ordered_parts = []
    for lin in LINEAGE_ORDER:
        d = tab[tab["major_lineage"].astype(str).eq(lin)].copy()
        if d.empty:
            continue

        order = cluster_order(d[features].to_numpy(float))
        d = d.iloc[order]
        ordered_parts.append(d)

    if not ordered_parts:
        raise RuntimeError("No cell types available after Major-lineage ordering.")

    ordered = pd.concat(ordered_parts)
    ordered = ordered.reset_index().rename(columns={"index": "cell_type"})
    ordered["column_order"] = np.arange(len(ordered))

    # Sequential-blue version is display-only min-max scaling within each feature.
    disp = ordered[
        [
            "cell_type",
            "major_lineage",
            "column_order",
            "n_pseudobulks",
            "n_donors",
            "n_tissues",
        ]
    ].copy()

    for f in features:
        v = ordered[f].astype(float).to_numpy()
        lo, hi = np.nanmin(v), np.nanmax(v)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            disp[f] = (v - lo) / (hi - lo)
        else:
            disp[f] = 0.5

    write_csv(ordered, "C01v10_celltype_feature_matrix_zscore_means.csv")
    write_csv(disp, "C01v10_celltype_feature_matrix_display_scaled.csv")

    lineage_palette = pd.DataFrame({
        "major_lineage": list(LINEAGE_DISPLAY.keys()),
        "color": [LINEAGE_COLORS[x] for x in LINEAGE_DISPLAY.keys()],
        "display": [LINEAGE_DISPLAY[x] for x in LINEAGE_DISPLAY.keys()],
    })
    write_csv(lineage_palette, "C01v10_lineage_palette.csv")

    return ordered, disp, features


# ============================================================================
# DRAWING
# ============================================================================

def draw_global(
    ordered: pd.DataFrame,
    disp: pd.DataFrame,
    features: list[str],
    sequential=False,
):
    mat = (
        disp[features].to_numpy(float).T
        if sequential
        else ordered[features].to_numpy(float).T
    )
    n_feat, n_cell = mat.shape

    lineages = ordered["major_lineage"].astype(str).tolist()
    lineage_rgba = np.array(
        [mpl.colors.to_rgba(LINEAGE_COLORS.get(x, "#BDBDBD")) for x in lineages]
    ).reshape(1, n_cell, 4)

    fig = plt.figure(figsize=(15.8, 6.6))
    fig.suptitle(
        "Global cell-type landscape of static clock-associated expression architecture",
        y=0.978,
        fontsize=12,
    )

    left = 0.082
    width = 0.748
    bar_h = 0.031
    lineage_y = 0.895
    overlap = 0.0035

    # One annotation strip only: Major lineage.
    ax_top = fig.add_axes([left, lineage_y, width, bar_h])

    heatmap_bottom = 0.145
    heatmap_top = lineage_y + overlap
    ax = fig.add_axes([
        left,
        heatmap_bottom,
        width,
        heatmap_top - heatmap_bottom,
    ])

    # Right side: colorbar + one Major-lineage legend only.
    cax = fig.add_axes([0.855, 0.865, 0.125, 0.018])
    lineage_lax = fig.add_axes([0.855, 0.535, 0.14, 0.245])
    lineage_lax.axis("off")

    cmap = make_sequential_blue_cmap() if sequential else mpl.colormaps["RdBu_r"]

    if sequential:
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(
            "Relative mean expression (within-feature display scaling)",
            fontsize=6.6,
            labelpad=2,
        )
    else:
        vmax = min(
            3.0,
            max(1.5, float(np.nanpercentile(np.abs(mat), 98))),
        )
        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(
            "Mean standardized feature value",
            fontsize=6.6,
            labelpad=2,
        )

    cb.ax.tick_params(labelsize=6, length=2, pad=1)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(
        [short_feature(x) for x in features],
        fontsize=7.2,
    )
    ax.set_xticks([])
    ax.set_xlabel(
        "Original cell types (ordered within Major-lineage blocks)",
        labelpad=8,
    )
    ax.set_ylabel("Frozen 24 clock-associated features")
    ax.tick_params(length=0)

    for s in ax.spines.values():
        s.set_visible(False)

    ax_top.imshow(
        lineage_rgba,
        aspect="auto",
        interpolation="nearest",
    )
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    ax_top.text(
        -0.012,
        0.5,
        "Major\nlineage",
        transform=ax_top.transAxes,
        ha="right",
        va="center",
        fontsize=6.7,
    )

    handles = [
        Patch(
            facecolor=LINEAGE_COLORS[x],
            edgecolor="none",
            label=LINEAGE_DISPLAY[x],
        )
        for x in LINEAGE_ORDER
        if x in set(lineages)
    ]

    lineage_lax.legend(
        handles=handles,
        title="Major lineage",
        frameon=False,
        loc="upper left",
        fontsize=6.2,
        title_fontsize=7.3,
        borderaxespad=0,
        labelspacing=0.44,
        handlelength=1.0,
        handleheight=0.88,
        handletextpad=0.45,
        ncol=1,
    )

    stem = (
        "C01v10_A_global_atlas_sequential_blue"
        if sequential
        else "C01v10_A_global_atlas_zscore"
    )
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
        write_csv(q1, f"C01v10_zoom_{lin}_zscore_means.csv")
        write_csv(q2, f"C01v10_zoom_{lin}_display_scaled.csv")
    return out


def draw_zoom_composite(zoom_map: dict, features: list[str]):
    panels = [
        ("B", "b_cell"),
        ("C", "t_cell"),
        ("D", "macrophage_monocyte"),
        ("E", "fibroblast"),
        ("F", "endothelial"),
    ]

    fig = plt.figure(figsize=(14.8, 10.0))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1],
        hspace=0.62,
        wspace=0.55,
    )

    cmap = mpl.colormaps["RdBu_r"]

    # One common z-score scale for B–F.
    all_values = []
    for _, lin in panels:
        if lin in zoom_map:
            zmean, _ = zoom_map[lin]
            all_values.append(
                zmean[features].to_numpy(float).ravel()
            )

    if all_values:
        vv = np.concatenate(all_values)
        shared_vmax = min(
            3.0,
            max(1.5, float(np.nanpercentile(np.abs(vv), 98))),
        )
    else:
        shared_vmax = 2.5

    positions = {
        0: gs[0, 0],
        1: gs[0, 1],
        2: gs[1, 0],
        3: gs[1, 1],
        4: gs[2, 0],
    }

    last_im = None

    for idx, (letter, lin) in enumerate(panels):
        if lin not in zoom_map:
            continue

        zmean, _ = zoom_map[lin]
        ax = fig.add_subplot(positions[idx])
        mat = zmean[features].to_numpy(float)

        last_im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=-shared_vmax,
            vmax=shared_vmax,
            interpolation="nearest",
        )

        ax.set_yticks(range(len(zmean)))
        ax.set_yticklabels(
            wrapped(
                zmean["cell_type"].astype(str).tolist(),
                28,
            ),
            fontsize=5.8,
        )

        ax.yaxis.tick_right()
        ax.tick_params(
            axis="y",
            pad=2,
            length=0,
        )

        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(
            [short_feature(x) for x in features],
            rotation=60,
            ha="right",
            fontsize=5.8,
        )
        ax.tick_params(axis="x", length=0)

        for s in ax.spines.values():
            s.set_visible(False)

        ax.set_title(
            f"{letter}  {LINEAGE_DISPLAY[lin]}",
            loc="left",
            fontsize=10,
            pad=6,
            fontweight="bold",
        )

    ax_blank = fig.add_subplot(gs[2, 1])
    ax_blank.axis("off")

    if last_im is not None:
        cax = fig.add_axes([0.905, 0.16, 0.075, 0.014])
        cb = fig.colorbar(
            last_im,
            cax=cax,
            orientation="horizontal",
        )
        cb.set_label(
            "Mean standardized feature value",
            fontsize=6.6,
            labelpad=2,
        )
        cb.ax.tick_params(
            labelsize=6,
            length=2,
            pad=1,
        )

    fig.suptitle(
        "Prespecified major-lineage zoom-ins of the global clock-associated landscape",
        y=0.985,
        fontsize=12,
    )

    fig.text(
        0.90,
        0.105,
        "B–F use the same red–white–blue scale.",
        ha="center",
        fontsize=6.5,
    )

    save_fig(
        fig,
        "C01v10_BF_five_lineage_zoom_composite",
    )



def draw_full_composite(
    ordered: pd.DataFrame,
    disp: pd.DataFrame,
    features: list[str],
    zoom_map: dict,
):
    fig = plt.figure(figsize=(16.2, 12.9))

    # -------------------------------------------------------------------------
    # PANEL A — one Major-lineage strip only.
    # -------------------------------------------------------------------------
    left = 0.075
    width = 0.715
    bar_h = 0.021
    lineage_y = 0.908
    overlap = 0.0025

    ax_top = fig.add_axes([
        left,
        lineage_y,
        width,
        bar_h,
    ])

    heatmap_bottom = 0.575
    heatmap_top = lineage_y + overlap

    ax = fig.add_axes([
        left,
        heatmap_bottom,
        width,
        heatmap_top - heatmap_bottom,
    ])

    cax = fig.add_axes([0.815, 0.888, 0.15, 0.014])
    lineage_lax = fig.add_axes([0.815, 0.685, 0.17, 0.15])
    lineage_lax.axis("off")

    mat = ordered[features].to_numpy(float).T
    lineages = ordered["major_lineage"].astype(str).tolist()

    lineage_rgba = np.array(
        [mpl.colors.to_rgba(LINEAGE_COLORS.get(x, "#BDBDBD")) for x in lineages]
    ).reshape(1, len(lineages), 4)

    global_vmax = min(
        3.0,
        max(
            1.5,
            float(np.nanpercentile(np.abs(mat), 98)),
        ),
    )

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=mpl.colormaps["RdBu_r"],
        vmin=-global_vmax,
        vmax=global_vmax,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(
        [short_feature(x) for x in features],
        fontsize=6.8,
    )
    ax.set_xticks([])
    ax.tick_params(length=0)

    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xlabel("Original cell types")
    ax.set_ylabel("Clock-associated features")
    ax.set_title(
        "A",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=2,
    )

    ax_top.imshow(
        lineage_rgba,
        aspect="auto",
        interpolation="nearest",
    )
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)

    ax_top.text(
        -0.010,
        0.5,
        "Major\nlineage",
        transform=ax_top.transAxes,
        ha="right",
        va="center",
        fontsize=6.2,
    )

    cb = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
    )
    cb.set_label(
        "Mean standardized feature value",
        fontsize=6.5,
        labelpad=2,
    )
    cb.ax.tick_params(
        labelsize=5.8,
        length=2,
        pad=1,
    )

    handles = [
        Patch(
            facecolor=LINEAGE_COLORS[x],
            edgecolor="none",
            label=LINEAGE_DISPLAY[x],
        )
        for x in LINEAGE_ORDER
        if x in set(lineages)
    ]

    lineage_lax.legend(
        handles=handles,
        title="Major lineage",
        frameon=False,
        loc="upper left",
        fontsize=5.8,
        title_fontsize=6.9,
        borderaxespad=0,
        labelspacing=0.34,
        handlelength=0.95,
        handleheight=0.85,
        handletextpad=0.42,
        ncol=1,
    )

    # -------------------------------------------------------------------------
    # B–F — same exact red/white/blue numeric scale as panel A.
    # -------------------------------------------------------------------------
    lower = fig.add_gridspec(
        2,
        3,
        left=0.065,
        right=0.955,
        bottom=0.065,
        top=0.485,
        wspace=0.56,
        hspace=0.55,
    )

    panel_map = [
        ("B", "b_cell", lower[0, 0]),
        ("C", "t_cell", lower[0, 1]),
        ("D", "macrophage_monocyte", lower[0, 2]),
        ("E", "fibroblast", lower[1, 0]),
        ("F", "endothelial", lower[1, 1]),
    ]

    for letter, lin, slot in panel_map:
        if lin not in zoom_map:
            continue

        zmean, _ = zoom_map[lin]
        axz = fig.add_subplot(slot)
        matz = zmean[features].to_numpy(float)

        axz.imshow(
            matz,
            aspect="auto",
            cmap=mpl.colormaps["RdBu_r"],
            vmin=-global_vmax,
            vmax=global_vmax,
            interpolation="nearest",
        )

        axz.set_xticks(range(len(features)))
        axz.set_xticklabels(
            [short_feature(x) for x in features],
            rotation=60,
            ha="right",
            fontsize=5.6,
        )

        axz.set_yticks(range(len(zmean)))
        axz.set_yticklabels(
            wrapped(
                zmean["cell_type"].astype(str).tolist(),
                24,
            ),
            fontsize=5.5,
        )

        axz.yaxis.tick_right()
        axz.tick_params(
            length=0,
            axis="y",
            pad=2,
        )

        for s in axz.spines.values():
            s.set_visible(False)

        axz.set_title(
            f"{letter}  {LINEAGE_DISPLAY[lin]}",
            loc="left",
            fontsize=10.5,
            pad=4,
            fontweight="bold",
        )

    blank = fig.add_subplot(lower[1, 2])
    blank.axis("off")
    blank.text(
        0.03,
        0.86,
        "Major lineage = prespecified analysis grouping derived from original cell-type labels.\n\n"
        "B–F use the same red–white–blue z-score scale as A.",
        ha="left",
        va="top",
        fontsize=7.2,
    )

    fig.suptitle(
        "Static clock-associated expression architecture across original cell types",
        y=0.982,
        fontsize=12.5,
    )

    save_fig(
        fig,
        "C01v10_FULL_composite",
    )



def write_manifest(joined: pd.DataFrame, features: list[str], ordered: pd.DataFrame):
    AUDIT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "COMPLETE",
        "version": "C01v10_lineage_only",
        "input_metadata": str(FROZEN_META),
        "input_standardized_matrix": str(STD24),
        "input_original_display_metadata": str(ORIGINAL_DISPLAY_META),
        "broad_cell_class_used": False,
        "major_lineage_provenance": "current corrected frozen metadata",
        "n_frozen_pseudobulks": int(len(joined)),
        "n_original_cell_types": int(joined["cell_type"].nunique()),
        "n_features": int(len(features)),
        "old_figures_modified": False,
        "statistics_recomputed": False,
        "display_scaling_for_blue_panels": (
            "min-max within each feature across original cell types"
        ),
        "global_ordering": (
            "Major lineage blocks only -> local hierarchical display ordering "
            "within each lineage"
        ),
        "outputs": [
            "C01v10_A_global_atlas_sequential_blue",
            "C01v10_A_global_atlas_zscore",
            "C01v10_BF_five_lineage_zoom_composite",
            "C01v10_FULL_composite",
        ],
    }

    (AUDIT / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"file": str(FROZEN_META), "sha256": sha256(FROZEN_META)},
        {"file": str(STD24), "sha256": sha256(STD24)},
        {"file": str(ORIGINAL_DISPLAY_META), "sha256": sha256(ORIGINAL_DISPLAY_META)},
    ]).to_csv(
        AUDIT / "INPUT_HASHES.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        "feature": features,
        "display_label": [short_feature(x) for x in features],
    }).to_csv(
        AUDIT / "FEATURE_LABELS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ordered.groupby(
        ["major_lineage"],
        observed=True,
    ).size().reset_index(
        name="n_cell_types"
    ).to_csv(
        AUDIT / "ORDERED_LINEAGE_BLOCK_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
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
    print("C01 TABULA-STYLE V10 LINEAGE-ONLY REDRAW COMPLETE")
    print("=" * 112)
    print("OUTPUT:", OUT)
    print()
    print("Broad cell class used: NO")
    print("Major lineage annotation only: YES")
    print("Generated:")
    print("  C01v10_A_global_atlas_sequential_blue")
    print("  C01v10_A_global_atlas_zscore")
    print("  C01v10_BF_five_lineage_zoom_composite")
    print("  C01v10_FULL_composite")
    print()
    print("Old figures modified: NO")
    print("Statistics recomputed: NO")
    print("OPEN FIRST:")
    print(" ", OUT / "C01v10_FULL_composite.png")
    print(" ", OUT / "C01v10_A_global_atlas_zscore.png")
    print("=" * 112)


if __name__ == "__main__":
    main()
