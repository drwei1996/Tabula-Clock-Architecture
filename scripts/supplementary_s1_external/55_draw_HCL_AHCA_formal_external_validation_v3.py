#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Formal plotting script v3 for finalized HCL / AHCA external-validation figures.

What this script does
---------------------
1) Reads the finalized external-validation root directory.
2) Prefers manuscript-facing source tables if present.
3) Falls back to frozen final outputs when manuscript package is missing.
4) Generates:
   - Main 2x2 figure (Fig1-like / Fig2-like external validation summary)
   - Supplementary robustness figures
   - Supplementary HCL / AHCA pseudobulk UMAPs
5) Writes a small README + copies the exact plot-ready source tables used.

Important scope
---------------
This script only draws the parts that are actually supported by the finalized
external-validation chain:
  * major-lineage architecture backbone
  * lineage-specific tissue remodeling
  * concordance with Tabula lineage remodeling
  * higher-level resampling / leave-one-out / composition control
  * pseudobulk UMAP visualization
It does NOT invent pathway-level external validation.

Usage
-----
python 55_draw_HCL_AHCA_formal_external_validation_v3.py \
  --root "<TABULA_PROJECT_ROOT>\\corrected_full_analysis_v2\\external validation\\clock_external_validation_ALL_EXCEPT_H5AD_20260921\\clock_external_validation" \
  --outdir "<TABULA_PROJECT_ROOT>\\HCL_AHCA_formal_external_validation_20260922"
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from itertools import permutations
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import umap  # type: ignore
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    from sklearn.decomposition import PCA
    HAVE_PCA = True
except Exception:
    HAVE_PCA = False

# -----------------------------------------------------------------------------
# Frozen display settings
# -----------------------------------------------------------------------------
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

DATASET_MARKERS = {
    "Tabula": "o",
    "HCL": "s",
    "AHCA": "^",
}

TABULA_REFERENCE = {
    "b_cell": 1.8593,
    "t_cell": 2.3052,
    "macrophage_monocyte": 2.3576,
    "fibroblast": 3.7516,
    "endothelial": 3.0475,
}

EXPANDED17 = [
    "ARNTL", "CLOCK", "NPAS2",
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "NR1D1", "NR1D2",
    "RORA", "RORB", "RORC",
    "DBP", "TEF", "HLF", "NFIL3",
]


def choose_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ["Arial", "Liberation Sans", "DejaVu Sans"]:
        if candidate in available:
            return candidate
    return "sans-serif"


mpl.rcParams.update({
    "font.family": choose_font(),
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig: plt.Figure, outdir: Path, stem: str) -> None:
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(outdir / f"{stem}{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14, 1.06, letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def find_first_existing(candidates: Iterable[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate files exist:\n" + "\n".join(str(x) for x in candidates))


def find_by_basename(root: Path, basename: str) -> Path | None:
    """Robust recursive fallback for nested external-validation archives."""
    hits = [p for p in root.rglob(basename) if p.is_file()]
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.parts), len(str(p)), str(p)))
    return hits[0]


def find_first_existing_or_recursive(root: Path, candidates: Iterable[Path], basenames: Iterable[str]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    for name in basenames:
        q = find_by_basename(root, name)
        if q is not None:
            print(f"[PATH FALLBACK] {name} -> {q}")
            return q
    raise FileNotFoundError(
        "Could not locate any requested file.\nCandidates:\n"
        + "\n".join(str(x) for x in candidates)
        + "\nRecursive basenames:\n"
        + "\n".join(basenames)
    )


def exact_spearman_permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    obs = abs(float(spearmanr(x, y).statistic))
    n = len(y)
    if n <= 8:
        allp = list(permutations(range(n)))
        ge = 0
        for perm in allp:
            yp = y[list(perm)]
            r = abs(float(spearmanr(x, yp).statistic))
            ge += int(r >= obs - 1e-15)
        return ge / len(allp)
    rng = np.random.default_rng(20260922)
    B = 20000
    ge = 0
    for _ in range(B):
        yp = rng.permutation(y)
        r = abs(float(spearmanr(x, yp).statistic))
        ge += int(r >= obs - 1e-15)
    return (ge + 1) / (B + 1)


def pretty_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def locate_manuscript_source_dir(root: Path) -> Path | None:
    candidates = [
        root / "05_MANUSCRIPT_EXTERNAL_VALIDATION_20260921" / "01_SOURCE_DATA",
        root / "05_MANUSCRIPT_EXTERNAL_VALIDATION_20260921" / "source_data",
        root / "MANUSCRIPT_EXTERNAL_VALIDATION" / "01_SOURCE_DATA",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_final_summary(root: Path, msrc: Path | None) -> pd.DataFrame:
    candidates = []
    if msrc is not None:
        candidates.append(msrc / "EXT_source_01_dataset_summary.csv")
    candidates += [
        root / "FINAL_SUMMARY.csv",
        root / "04_FINAL_SUMMARY" / "FINAL_SUMMARY.csv",
    ]
    path = find_first_existing_or_recursive(
        root, candidates, ["EXT_source_01_dataset_summary.csv", "FINAL_SUMMARY.csv"]
    )
    print(f"[SOURCE] dataset summary: {path}")
    df = read_csv_any(path)
    rename = {
        "v4_backbone_difference": "backbone_difference",
        "composition_matched_within_median": "composition_matched_within_median",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df


def reconstruct_lineage_long(root: Path) -> pd.DataFrame:
    hcl = read_csv_any(root / "01_HCL" / "source_data" / "lineage_tissue_remodeling.csv")
    ahca = read_csv_any(root / "02_AHCA_15ORG" / "source_data" / "lineage_tissue_remodeling.csv")
    rows = []
    for lin in LINEAGE_ORDER:
        rows.append({
            "dataset": "Tabula",
            "major_lineage": lin,
            "display": LINEAGE_DISPLAY[lin],
            "distance": TABULA_REFERENCE[lin],
            "feature_space": "Frozen Tabula 24D reference",
        })
    for ds, df in [("HCL", hcl), ("AHCA", ahca)]:
        col = "median_tissue_remodeling_distance" if "median_tissue_remodeling_distance" in df.columns else "median_tissue_distance"
        for _, r in df.iterrows():
            rows.append({
                "dataset": ds,
                "major_lineage": r["major_lineage"],
                "display": LINEAGE_DISPLAY.get(r["major_lineage"], str(r["major_lineage"])),
                "distance": float(r[col]),
                "feature_space": "Expanded17 external-validation space",
            })
    return pd.DataFrame(rows)


def load_lineage_long(root: Path, msrc: Path | None) -> pd.DataFrame:
    if msrc is not None and (msrc / "EXT_source_02_lineage_remodeling_three_atlases.csv").exists():
        p = msrc / "EXT_source_02_lineage_remodeling_three_atlases.csv"
        print(f"[SOURCE] lineage remodeling: {p}")
        return read_csv_any(p)
    return reconstruct_lineage_long(root)


def load_rank_concordance(root: Path, msrc: Path | None, lineage_long: pd.DataFrame) -> pd.DataFrame:
    if msrc is not None and (msrc / "EXT_source_03_rank_concordance.csv").exists():
        p = msrc / "EXT_source_03_rank_concordance.csv"
        print(f"[SOURCE] rank concordance: {p}")
        return read_csv_any(p)

    rank_rows = []
    for extds in ["HCL", "AHCA"]:
        tab = lineage_long[lineage_long["dataset"].eq("Tabula")].set_index("major_lineage")["distance"]
        ext = lineage_long[lineage_long["dataset"].eq(extds)].set_index("major_lineage")["distance"]
        xx = np.array([float(tab[l]) for l in LINEAGE_ORDER if l in tab.index and l in ext.index], float)
        yy = np.array([float(ext[l]) for l in LINEAGE_ORDER if l in tab.index and l in ext.index], float)
        rank_rows.append({
            "comparison": f"Tabula vs {extds}",
            "n_lineages": len(xx),
            "spearman_rho": float(spearmanr(xx, yy).statistic),
            "exact_permutation_p": exact_spearman_permutation_p(xx, yy),
        })
    return pd.DataFrame(rank_rows)


def load_bootstrap_summary(
    root: Path,
    msrc: Path | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Detailed bootstrap tables were not included in every local export.
    Return (None, None) rather than aborting; the plotting layer then uses
    FINAL_RIGOR_SUMMARY.csv for the primary HCL donor+tissue and AHCA tissue CIs.
    """
    h_candidates = []
    a_candidates = []

    if msrc is not None:
        h_candidates.append(msrc / "EXT_source_05_HCL_bootstrap_summary.csv")
        a_candidates.append(msrc / "EXT_source_06_AHCA_bootstrap_summary.csv")

    rigor = root / "03_RIGOR_VALIDATION_V1_20260921"
    h_candidates += [
        rigor / "01_HIGHER_LEVEL_RESAMPLING" / "HCL" / "HCL_higher_level_bootstrap_summary.csv",
        rigor / "01_HIGHER_LEVEL_RESAMPLING" / "HCL_higher_level_bootstrap_summary.csv",
    ]
    a_candidates += [
        rigor / "01_HIGHER_LEVEL_RESAMPLING" / "AHCA" / "AHCA_tissue_bootstrap_summary.csv",
        rigor / "01_HIGHER_LEVEL_RESAMPLING" / "AHCA_tissue_bootstrap_summary.csv",
    ]

    hp = next((p for p in h_candidates if p.exists()), None)
    ap = next((p for p in a_candidates if p.exists()), None)

    if hp is None:
        hp = find_by_basename(root, "HCL_higher_level_bootstrap_summary.csv")
    if ap is None:
        ap = find_by_basename(root, "AHCA_tissue_bootstrap_summary.csv")

    if hp is None:
        print("[WARN] Detailed HCL bootstrap summary not present in local export.")
    else:
        print(f"[SOURCE] HCL detailed bootstrap:  {hp}")

    if ap is None:
        print("[WARN] Detailed AHCA bootstrap summary not present in local export.")
    else:
        print(f"[SOURCE] AHCA detailed bootstrap: {ap}")

    return (
        read_csv_any(hp) if hp is not None else None,
        read_csv_any(ap) if ap is not None else None,
    )


def load_rigor_summary(root: Path, msrc: Path | None) -> pd.DataFrame | None:
    candidates = []
    if msrc is not None:
        candidates.append(msrc / "EXT_source_04_higher_level_rigor_summary.csv")
    candidates += [
        root / "03_RIGOR_VALIDATION_V1_20260921" / "FINAL_RIGOR_SUMMARY.csv",
    ]

    p = next((x for x in candidates if x.exists()), None)
    if p is None:
        p = find_by_basename(root, "FINAL_RIGOR_SUMMARY.csv")

    if p is None:
        print("[WARN] FINAL_RIGOR_SUMMARY.csv not found; rigor-CI panel will be skipped.")
        return None

    print(f"[SOURCE] final rigor summary: {p}")
    return read_csv_any(p)


def load_leave_one_out(
    root: Path,
    msrc: Path | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Leave-one-out draw tables are optional in the local export.
    Missing files should never prevent the finalized main external figure.
    """
    names = [
        ("HCL donor LOO", "EXT_source_09_HCL_leave_one_donor_out.csv", "HCL_leave_one_donor_out.csv"),
        ("HCL tissue LOO", "EXT_source_10_HCL_leave_one_tissue_out.csv", "HCL_leave_one_tissue_out.csv"),
        ("AHCA tissue LOO", "EXT_source_11_AHCA_leave_one_tissue_out.csv", "AHCA_leave_one_tissue_out.csv"),
    ]

    out = []
    for label, manuscript_name, raw_name in names:
        p = None
        if msrc is not None:
            q = msrc / manuscript_name
            if q.exists():
                p = q
        if p is None:
            p = find_by_basename(root, raw_name)

        if p is None:
            print(f"[WARN] {label} table not present in local export.")
            out.append(None)
        else:
            print(f"[SOURCE] {label}: {p}")
            out.append(read_csv_any(p))

    return out[0], out[1], out[2]


def load_composition_long(root: Path, msrc: Path | None) -> pd.DataFrame | None:
    if msrc is not None and (msrc / "EXT_source_18_composition_matched_plot_data.csv").exists():
        p = msrc / "EXT_source_18_composition_matched_plot_data.csv"
        print(f"[SOURCE] composition matched: {p}")
        return read_csv_any(p)

    rigor = root / "03_RIGOR_VALIDATION_V1_20260921"
    h_candidates = [
        rigor / "02_COMPOSITION_CONTROLLED" / "HCL" / "source_data" / "HCL_composition_matched_lineage_summary.csv",
    ]
    a_candidates = [
        rigor / "02_COMPOSITION_CONTROLLED" / "AHCA" / "source_data" / "AHCA_composition_matched_lineage_summary.csv",
    ]

    try:
        hsum = find_first_existing_or_recursive(
            root, h_candidates, ["HCL_composition_matched_lineage_summary.csv"]
        )
        asum = find_first_existing_or_recursive(
            root, a_candidates, ["AHCA_composition_matched_lineage_summary.csv"]
        )
    except FileNotFoundError:
        print("[WARN] Composition-matched lineage summaries not found; composition panel will be skipped.")
        return None

    print(f"[SOURCE] HCL composition:  {hsum}")
    print(f"[SOURCE] AHCA composition: {asum}")
    h = read_csv_any(hsum)
    a = read_csv_any(asum)
    rows = []
    for lin in LINEAGE_ORDER:
        rows.append({
            "dataset": "Tabula",
            "major_lineage": lin,
            "display": LINEAGE_DISPLAY[lin],
            "distance": TABULA_REFERENCE[lin],
        })
    for ds, df in [("HCL", h), ("AHCA", a)]:
        col = "median_composition_matched_tissue_distance"
        for _, r in df.iterrows():
            rows.append({
                "dataset": ds,
                "major_lineage": r["major_lineage"],
                "display": LINEAGE_DISPLAY.get(r["major_lineage"], str(r["major_lineage"])),
                "distance": float(r[col]),
            })
    return pd.DataFrame(rows)


def load_pseudobulk(root: Path, dataset: str) -> tuple[pd.DataFrame, list[str]]:
    if dataset == "HCL":
        path = root / "01_HCL" / "source_data" / "pseudobulk_standardized.csv"
    elif dataset == "AHCA":
        path = root / "02_AHCA_15ORG" / "source_data" / "pseudobulk_standardized.csv"
    else:
        raise ValueError(dataset)

    df = read_csv_any(path)

    # Prefer exact Expanded17 z-scored columns if present.
    feat_pref = [c for c in [f"expr_{g}" for g in EXPANDED17] if c in df.columns]
    if len(feat_pref) >= 12:
        return df, feat_pref

    feat_alt = [c for c in EXPANDED17 if c in df.columns]
    if len(feat_alt) >= 12:
        return df, feat_alt

    meta_like = {
        "dataset", "group_id", "donor_id", "tissue", "cell_type", "major_lineage", "n_cells",
        "sample", "donor", "lineage", "broad_class", "cellclass", "cell_class",
    }
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in meta_like]
    # Drop obvious metadata-ish numeric columns.
    num = [c for c in num if c.lower() not in {"n_cells", "ncell", "cell_count"}]
    if len(num) < 2:
        raise RuntimeError(f"{dataset}: cannot determine feature columns for pseudobulk UMAP.")
    return df, num


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------
def plot_panel_A_backbone(ax: plt.Axes, final_summary: pd.DataFrame) -> None:
    d = final_summary.copy()
    d = d.set_index("dataset").loc[[x for x in ["HCL", "AHCA"] if x in d["dataset"].unique()]].reset_index()
    x = np.arange(len(d), dtype=float)

    for i, r in d.iterrows():
        y1 = float(r["within_lineage_median"])
        y2 = float(r["between_lineage_median"])
        ax.plot([x[i], x[i]], [y1, y2], color="#C2C2C2", lw=1.25, zorder=1)
        ax.scatter(x[i], y1, s=44, facecolor="white", edgecolor="#595959", linewidth=1.0, zorder=3)
        ax.scatter(x[i], y2, s=48, facecolor="#4D4D4D", edgecolor="#4D4D4D", linewidth=1.0, zorder=3)
        ax.text(x[i] + 0.09, y1, "Within lineage", va="center", ha="left", fontsize=7, color="#555555")
        ax.text(x[i] + 0.09, y2, "Between lineages", va="center", ha="left", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(d["dataset"].tolist())
    ax.set_ylabel("Median Euclidean distance")
    ax.set_title("External major-lineage architecture")
    ax.set_xlim(-0.35, len(d) - 0.25)
    clean_axis(ax)


def plot_panel_B_lineage_remodeling(ax: plt.Axes, long_df: pd.DataFrame) -> None:
    x = np.arange(len(LINEAGE_ORDER), dtype=float)
    offsets = {"Tabula": -0.22, "HCL": 0.0, "AHCA": 0.22}

    for ds in ["Tabula", "HCL", "AHCA"]:
        p = long_df[long_df["dataset"].eq(ds)].set_index("major_lineage")
        for i, lin in enumerate(LINEAGE_ORDER):
            if lin not in p.index:
                continue
            ax.scatter(
                x[i] + offsets[ds],
                float(p.loc[lin, "distance"]),
                s=52,
                marker=DATASET_MARKERS[ds],
                facecolor=LINEAGE_COLORS[lin],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([LINEAGE_DISPLAY[l] for l in LINEAGE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Median tissue-remodeling distance")
    ax.set_title("Lineage-specific tissue remodeling")
    clean_axis(ax)

    handles = [
        Line2D([0], [0], marker=DATASET_MARKERS[ds], linestyle="none",
               markerfacecolor="#666666", markeredgecolor="white",
               markersize=6.7, label=ds)
        for ds in ["Tabula", "HCL", "AHCA"]
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper left")


def plot_scatter_concordance(ax: plt.Axes, long_df: pd.DataFrame, external: str, rank_df: pd.DataFrame | None = None) -> None:
    tab = long_df[long_df["dataset"].eq("Tabula")].set_index("major_lineage")["distance"]
    ext = long_df[long_df["dataset"].eq(external)].set_index("major_lineage")["distance"]
    rows = [(lin, float(tab[lin]), float(ext[lin])) for lin in LINEAGE_ORDER if lin in tab.index and lin in ext.index]

    x = np.array([r[1] for r in rows], float)
    y = np.array([r[2] for r in rows], float)

    if rank_df is not None and not rank_df.empty:
        match = rank_df[rank_df["comparison"].eq(f"Tabula vs {external}")]
        if len(match):
            rho = float(match.iloc[0]["spearman_rho"])
            p = float(match.iloc[0]["exact_permutation_p"])
        else:
            rho = float(spearmanr(x, y).statistic)
            p = exact_spearman_permutation_p(x, y)
    else:
        rho = float(spearmanr(x, y).statistic)
        p = exact_spearman_permutation_p(x, y)

    for lin, xv, yv in rows:
        ax.scatter(xv, yv, s=55, facecolor=LINEAGE_COLORS[lin], edgecolor="white", linewidth=0.7, zorder=3)
        ax.annotate(LINEAGE_DISPLAY[lin], (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=7)

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], ls="--", lw=0.9, color="#B0B0B0", zorder=0)
    ax.set_xlim(lo-pad, hi+pad)
    ax.set_ylim(lo-pad, hi+pad)
    ax.set_xlabel("Tabula tissue-remodeling distance")
    ax.set_ylabel(f"{external} tissue-remodeling distance")
    ax.set_title(f"Tabula vs {external}")
    ax.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}\nExact P = {pretty_p(p)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.4)
    clean_axis(ax)


def plot_primary_rigor_from_summary(ax: plt.Axes, rigor_df: pd.DataFrame) -> None:
    """
    Plot the prespecified higher-level primary CI for each external atlas using
    FINAL_RIGOR_SUMMARY.csv. This is the robust fallback when detailed bootstrap
    draw/summary tables were intentionally not included in the local export.
    """
    d = rigor_df.copy()
    d = d[d["dataset"].isin(["HCL", "AHCA"])].copy()

    rows = []
    for ds in ["HCL", "AHCA"]:
        p = d[d["dataset"].eq(ds)]
        if p.empty:
            continue
        r = p.iloc[0]

        est_col = "v4_backbone_difference" if "v4_backbone_difference" in p.columns else "backbone_difference"
        if est_col not in p.columns:
            continue

        rows.append({
            "dataset": ds,
            "estimate": float(r[est_col]),
            "low": float(r["hl_difference_ci_low"]),
            "high": float(r["hl_difference_ci_high"]),
            "method": str(r.get("higher_level_primary_method", "")),
        })

    if not rows:
        ax.text(0.5, 0.5, "Higher-level rigor summary unavailable",
                transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    dd = pd.DataFrame(rows)
    y = np.arange(len(dd))
    err = np.vstack([
        dd["estimate"].to_numpy(float) - dd["low"].to_numpy(float),
        dd["high"].to_numpy(float) - dd["estimate"].to_numpy(float),
    ])
    err = np.maximum(err, 0)

    ax.errorbar(
        dd["estimate"], y,
        xerr=err,
        fmt="o",
        color="#4D4D4D",
        ecolor="#777777",
        elinewidth=1.2,
        capsize=3,
        markersize=5.5,
        zorder=3,
    )
    ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(dd["dataset"].tolist())
    ax.set_xlabel("Between-lineage − within-lineage median distance")
    ax.set_title("Higher-level resampling")
    ax.invert_yaxis()
    clean_axis(ax)


def plot_resampling(fig: plt.Figure, ax: plt.Axes, hcl_boot: pd.DataFrame, ahca_boot: pd.DataFrame) -> None:
    items = []
    h = hcl_boot.set_index("method")
    for method, label in [
        ("donor_only", "HCL donor"),
        ("tissue_only", "HCL tissue"),
        ("donor_tissue_two_way", "HCL donor+tissue"),
    ]:
        if method in h.index:
            r = h.loc[method]
            items.append({
                "label": label,
                "estimate": float(r["difference_median_draw"]),
                "low": float(r["difference_ci_low"]),
                "high": float(r["difference_ci_high"]),
            })

    if len(ahca_boot):
        r = ahca_boot.iloc[0]
        items.append({
            "label": "AHCA tissue",
            "estimate": float(r["difference_median_draw"]),
            "low": float(r["difference_ci_low"]),
            "high": float(r["difference_ci_high"]),
        })

    d = pd.DataFrame(items)
    y = np.arange(len(d))
    xerr = np.vstack([d["estimate"].to_numpy() - d["low"].to_numpy(), d["high"].to_numpy() - d["estimate"].to_numpy()])
    xerr = np.maximum(xerr, 0)
    ax.errorbar(d["estimate"], y, xerr=xerr, fmt="o", color="#555555", ecolor="#777777", elinewidth=1.0, capsize=2.7, markersize=5.0)
    ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Between-lineage − within-lineage median distance")
    ax.set_title("Higher-level resampling")
    ax.invert_yaxis()
    clean_axis(ax)


def plot_leave_one_out(ax: plt.Axes, hcl_donor: pd.DataFrame, hcl_tissue: pd.DataFrame, ahca_tissue: pd.DataFrame, final_summary: pd.DataFrame) -> None:
    datasets = [
        ("HCL donor LOO", hcl_donor, "HCL"),
        ("HCL tissue LOO", hcl_tissue, "HCL"),
        ("AHCA tissue LOO", ahca_tissue, "AHCA"),
    ]
    rng = np.random.default_rng(20260922)

    for i, (label, df, ds) in enumerate(datasets):
        vals = df["difference"].to_numpy(float)
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(vals, np.full(len(vals), i, dtype=float) + jitter, s=14, facecolor="white", edgecolor="#666666", linewidth=0.55, alpha=0.9)
        med = np.median(vals)
        ax.plot([med, med], [i-0.22, i+0.22], color="#222222", lw=1.4)
        base = float(final_summary.set_index("dataset").loc[ds, "backbone_difference"])
        ax.scatter(base, i, s=40, marker="D", facecolor="#222222", edgecolor="white", linewidth=0.55, zorder=4)

    ax.axvline(0, color="#B0B0B0", lw=0.9, ls="--")
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels([x[0] for x in datasets])
    ax.set_xlabel("Backbone difference after one-cluster deletion")
    ax.set_title("Leave-one-cluster-out robustness")
    clean_axis(ax)


def plot_composition_matched(ax: plt.Axes, comp_long: pd.DataFrame) -> None:
    x = np.arange(len(LINEAGE_ORDER), dtype=float)
    offsets = {"Tabula": -0.22, "HCL": 0.0, "AHCA": 0.22}

    for ds in ["Tabula", "HCL", "AHCA"]:
        p = comp_long[comp_long["dataset"].eq(ds)].set_index("major_lineage")
        for i, lin in enumerate(LINEAGE_ORDER):
            if lin not in p.index:
                continue
            y = float(p.loc[lin, "distance"])
            if not np.isfinite(y):
                continue
            ax.scatter(x[i] + offsets[ds], y, s=52, marker=DATASET_MARKERS[ds], facecolor=LINEAGE_COLORS[lin], edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([LINEAGE_DISPLAY[l] for l in LINEAGE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Composition-matched tissue-remodeling distance")
    ax.set_title("Composition-matched sensitivity")
    clean_axis(ax)
    handles = [
        Line2D([0], [0], marker=DATASET_MARKERS[ds], linestyle="none", markerfacecolor="#666666", markeredgecolor="white", markersize=6.7, label=ds)
        for ds in ["Tabula", "HCL", "AHCA"]
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper left")


def compute_2d_embedding(X: np.ndarray) -> np.ndarray:
    if HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, max(3, len(X)-1)), min_dist=0.35, metric="euclidean", random_state=20260922)
        return reducer.fit_transform(X)
    if HAVE_PCA:
        return PCA(n_components=2, random_state=20260922).fit_transform(X)
    raise RuntimeError("Neither umap-learn nor scikit-learn PCA is available.")


def plot_umap(ax: plt.Axes, pb: pd.DataFrame, feat_cols: list[str], dataset_name: str) -> None:
    X = pb[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keep = np.isfinite(X).all(axis=1)
    dat = pb.loc[keep].copy().reset_index(drop=True)
    X = X[keep]
    emb = compute_2d_embedding(X)

    for lin in LINEAGE_ORDER:
        idx = dat["major_lineage"].astype(str).eq(lin).to_numpy()
        if idx.sum() == 0:
            continue
        ax.scatter(emb[idx, 0], emb[idx, 1], s=18, facecolor=LINEAGE_COLORS[lin], edgecolor="white", linewidth=0.35, alpha=0.95, label=LINEAGE_DISPLAY[lin])

    ax.set_xlabel("UMAP1" if HAVE_UMAP else "PC1")
    ax.set_ylabel("UMAP2" if HAVE_UMAP else "PC2")
    ax.set_title(f"{dataset_name} pseudobulk {'UMAP' if HAVE_UMAP else 'PCA'}")
    clean_axis(ax)
    ax.legend(frameon=False, loc="best", fontsize=7)


def draw_main_2x2(final_summary: pd.DataFrame, lineage_long: pd.DataFrame, rank_df: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8))
    axA, axB, axC, axD = axes.flatten()

    plot_panel_A_backbone(axA, final_summary)
    add_panel_letter(axA, "A")

    plot_panel_B_lineage_remodeling(axB, lineage_long)
    add_panel_letter(axB, "B")

    plot_scatter_concordance(axC, lineage_long, "HCL", rank_df)
    add_panel_letter(axC, "C")

    plot_scatter_concordance(axD, lineage_long, "AHCA", rank_df)
    add_panel_letter(axD, "D")

    fig.suptitle("External validation of major-lineage clock-associated architecture", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, outdir, "EV_Fig1_2x2_main")


def draw_supplementary(
    final_summary: pd.DataFrame,
    rigor_summary: pd.DataFrame | None,
    hcl_boot: pd.DataFrame | None,
    ahca_boot: pd.DataFrame | None,
    hcl_lodo: pd.DataFrame | None,
    hcl_loto: pd.DataFrame | None,
    ahca_loto: pd.DataFrame | None,
    comp_long: pd.DataFrame | None,
    outdir: Path,
) -> None:

    # Primary finalized higher-level CI. Prefer FINAL_RIGOR_SUMMARY because
    # it is guaranteed in the user's local archive.
    if rigor_summary is not None:
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        plot_primary_rigor_from_summary(ax, rigor_summary)
        fig.tight_layout()
        save_fig(fig, outdir, "EV_Supp1_primary_higher_level_resampling")

    # Richer four-row resampling panel only when detailed tables are available.
    if hcl_boot is not None and ahca_boot is not None:
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        plot_resampling(fig, ax, hcl_boot, ahca_boot)
        fig.tight_layout()
        save_fig(fig, outdir, "EV_Supp1b_detailed_higher_level_resampling")

    # LOO only when all three tables are actually available.
    if hcl_lodo is not None and hcl_loto is not None and ahca_loto is not None:
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        plot_leave_one_out(ax, hcl_lodo, hcl_loto, ahca_loto, final_summary)
        fig.tight_layout()
        save_fig(fig, outdir, "EV_Supp2_leave_one_cluster_out")

    # Composition-matched lineage panel only if plot-ready lineage table exists.
    if comp_long is not None:
        fig, ax = plt.subplots(figsize=(7.3, 4.0))
        plot_composition_matched(ax, comp_long)
        fig.tight_layout()
        save_fig(fig, outdir, "EV_Supp3_composition_matched_sensitivity")


def draw_umaps(root: Path, outdir: Path) -> None:
    for ds in ["HCL", "AHCA"]:
        pb, feat_cols = load_pseudobulk(root, ds)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        plot_umap(ax, pb, feat_cols, ds)
        fig.tight_layout()
        save_fig(fig, outdir, f"EV_Supp_{ds}_pseudobulk_UMAP")


# -----------------------------------------------------------------------------
# Output bookkeeping
# -----------------------------------------------------------------------------
def write_readme(outdir: Path, root: Path, msrc: Path | None, used: dict[str, str]) -> None:
    text = f"""# HCL / AHCA formal external-validation figures

Generated from root:
`{root}`

Manuscript package source dir:
`{msrc if msrc is not None else 'NOT FOUND (fallback to frozen raw outputs)'}`

## Figures generated
- `figures_main/EV_Fig1_2x2_main.*`
- `figures_supp/EV_Supp1_higher_level_resampling.*`
- `figures_supp/EV_Supp2_leave_one_cluster_out.*`
- `figures_supp/EV_Supp3_composition_matched_sensitivity.*` (if composition-control tables were available)
- `figures_supp/EV_Supp_HCL_pseudobulk_UMAP.*`
- `figures_supp/EV_Supp_AHCA_pseudobulk_UMAP.*`

## Exact tables used
"""
    for k, v in used.items():
        text += f"- **{k}**: `{v}`\n"
    text += "\n## Scope\nThis formal plotting script supports finalized Fig1/Fig2-like external validation only. It does not create pathway-level Fig3-like external validation because those final source tables are not part of the frozen external-validation package.\n"
    (outdir / "README.md").write_text(text, encoding="utf-8")


def copy_used_tables(outdir: Path, tables: dict[str, pd.DataFrame]) -> None:
    sdir = mkdir(outdir / "source_data_used")
    for name, df in tables.items():
        df.to_csv(sdir / f"{name}.csv", index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Draw formal HCL/AHCA external-validation figures from frozen outputs.")
    ap.add_argument("--root", required=True, help="external-validation root, e.g. ...\\clock_external_validation")
    ap.add_argument("--outdir", required=True, help="output directory")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = mkdir(Path(args.outdir).expanduser().resolve())
    fig_main = mkdir(outdir / "figures_main")
    fig_supp = mkdir(outdir / "figures_supp")

    print("=" * 110)
    print("DRAW FORMAL HCL / AHCA EXTERNAL-VALIDATION FIGURES")
    print("=" * 110)
    print(f"ROOT: {root}")
    print(f"OUT : {outdir}")

    msrc = locate_manuscript_source_dir(root)
    print(f"MANUSCRIPT SOURCE: {msrc if msrc is not None else 'NOT FOUND -> using raw frozen outputs'}")

    used_paths: dict[str, str] = {}

    final_summary = load_final_summary(root, msrc)
    lineage_long = load_lineage_long(root, msrc)
    rank_df = load_rank_concordance(root, msrc, lineage_long)
    hcl_boot, ahca_boot = load_bootstrap_summary(root, msrc)
    rigor_summary = load_rigor_summary(root, msrc)
    hcl_lodo, hcl_loto, ahca_loto = load_leave_one_out(root, msrc)
    comp_long = load_composition_long(root, msrc)

    # Bookkeeping labels for README
    used_paths["dataset_summary"] = "EXT_source_01_dataset_summary.csv or FINAL_SUMMARY.csv"
    used_paths["lineage_remodeling"] = "EXT_source_02_lineage_remodeling_three_atlases.csv or reconstructed from HCL/AHCA lineage_tissue_remodeling.csv"
    used_paths["rank_concordance"] = "EXT_source_03_rank_concordance.csv or recomputed from lineage distances"
    used_paths["bootstrap_summary"] = "EXT_source_05/06_*bootstrap_summary.csv or raw rigor summaries"
    used_paths["leave_one_out"] = "EXT_source_09/10/11_* or raw rigor leave-one-out tables"
    used_paths["composition_matched"] = "EXT_source_18_composition_matched_plot_data.csv or raw composition summaries"
    used_paths["pseudobulk_umap"] = "01_HCL/source_data/pseudobulk_standardized.csv and 02_AHCA_15ORG/source_data/pseudobulk_standardized.csv"

    source_tables = {
        "dataset_summary": final_summary,
        "lineage_remodeling_three_atlases": lineage_long,
        "rank_concordance": rank_df,
    }
    if rigor_summary is not None:
        source_tables["final_rigor_summary"] = rigor_summary
    if hcl_boot is not None:
        source_tables["HCL_bootstrap_summary"] = hcl_boot
    if ahca_boot is not None:
        source_tables["AHCA_bootstrap_summary"] = ahca_boot
    if hcl_lodo is not None:
        source_tables["HCL_leave_one_donor_out"] = hcl_lodo
    if hcl_loto is not None:
        source_tables["HCL_leave_one_tissue_out"] = hcl_loto
    if ahca_loto is not None:
        source_tables["AHCA_leave_one_tissue_out"] = ahca_loto
    if comp_long is not None:
        source_tables["composition_matched_plot_data"] = comp_long

    copy_used_tables(outdir, source_tables)

    draw_main_2x2(final_summary, lineage_long, rank_df, fig_main)
    draw_supplementary(
        final_summary,
        rigor_summary,
        hcl_boot,
        ahca_boot,
        hcl_lodo,
        hcl_loto,
        ahca_loto,
        comp_long,
        fig_supp,
    )
    try:
        draw_umaps(root, fig_supp)
    except Exception as e:
        print(f"[WARN] Pseudobulk UMAP/PCA panels skipped: {type(e).__name__}: {e}")

    write_readme(outdir, root, msrc, used_paths)

    print("\n[DONE] Main figure:")
    print(fig_main / "EV_Fig1_2x2_main.png")
    print("[DONE] Supplementary figures:")
    for p in sorted(fig_supp.glob("*.png")):
        print(p)
    print("=" * 110)


if __name__ == "__main__":
    main()
