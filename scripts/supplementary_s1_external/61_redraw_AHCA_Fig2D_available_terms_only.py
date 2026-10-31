#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
61_redraw_AHCA_Fig2D_available_terms_only.py

Redraw the AHCA Figure-2-like feature-architecture heatmap from the FROZEN
source table produced by script 56.

Key change
----------
Terms that contain no finite values in AHCA are REMOVED from the heatmap
instead of being drawn as grey/blank columns.

This is a plotting-only script:
- NO model refitting
- NO bootstrap
- NO recalculation of adjusted R2
- reads existing Fig2_feature_architecture.csv
- writes one standalone panel + filtered source table + audit table

Expected AHCA result:
    Cell identity | Tissue
while unavailable single-donor terms such as Donor and Identity × tissue are
omitted automatically if they are entirely non-finite.

Outputs
-------
<outdir>/
    AHCA_Fig2D_feature_architecture_available_terms_only.png
    AHCA_Fig2D_feature_architecture_available_terms_only.pdf
    AHCA_Fig2D_feature_architecture_available_terms_only.svg
    AHCA_Fig2D_feature_architecture_available_terms_only_source.csv
    AHCA_Fig2D_removed_terms_audit.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

EXPANDED17 = [
    "ARNTL", "CLOCK", "NPAS2",
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "NR1D1", "NR1D2",
    "RORA", "RORB", "RORC",
    "DBP", "TEF", "HLF", "NFIL3",
]

TERM_SPECS = [
    ("cell_identity_unique_adjR2", "Cell identity"),
    ("tissue_unique_adjR2", "Tissue"),
    ("donor_unique_adjR2", "Donor"),
    ("identity_x_tissue_increment_adjR2", "Identity × tissue"),
]

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
})


# =============================================================================
# Helpers
# =============================================================================

def find_source(project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    preferred = (
        project
        / "HCL_AHCA_Fig123_replication_20260923"
        / "AHCA"
        / "source_data"
        / "Fig2_feature_architecture.csv"
    )
    if preferred.exists():
        return preferred

    hits = sorted(
        project.glob("**/AHCA/source_data/Fig2_feature_architecture.csv")
    )
    if len(hits) == 1:
        return hits[0]

    hits2 = [
        p for p in project.glob("**/Fig2_feature_architecture.csv")
        if "AHCA" in str(p).upper()
    ]
    hits2 = sorted(set(hits2))

    if len(hits2) == 1:
        return hits2[0]

    if len(hits2) > 1:
        msg = "\n".join(str(x) for x in hits2[:30])
        raise RuntimeError(
            "Multiple AHCA Fig2_feature_architecture.csv candidates found. "
            "Please pass --source explicitly:\n" + msg
        )

    raise FileNotFoundError(
        "Could not find AHCA Fig2_feature_architecture.csv below:\n"
        f"{project}\n"
        "Run script 56 first or pass --source explicitly."
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.12,
            "facecolor": "white",
        }
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(outdir / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project",
        required=True,
        type=Path,
        help="Tabula Sapiens project root.",
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional explicit Fig2_feature_architecture.csv.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory.",
    )
    ap.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=(
            "Optional symmetric color limit. If omitted, use max(0.15, "
            "98th percentile of absolute finite retained values), matching "
            "the original script logic."
        ),
    )
    args = ap.parse_args()

    project = args.project.resolve()
    source = find_source(project, args.source)

    if args.outdir is None:
        outdir = (
            project
            / "HCL_AHCA_Fig123_replication_20260923"
            / "AHCA"
            / "Figure2_like_single_panels_redraw"
        )
    else:
        outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("AHCA FIG2D REDRAW — AVAILABLE TERMS ONLY")
    print("=" * 100)
    print(f"SOURCE : {source}")
    print(f"OUTDIR : {outdir}")

    arch = pd.read_csv(source, low_memory=False)

    if "feature" not in arch.columns:
        raise RuntimeError(
            f"'feature' column missing from source table. Columns:\n"
            f"{list(arch.columns)}"
        )

    # Preserve Expanded17 order exactly.
    arch["feature"] = arch["feature"].astype(str)
    missing_features = [g for g in EXPANDED17 if g not in set(arch["feature"])]
    if missing_features:
        raise RuntimeError(
            "Source table is missing Expanded17 features: "
            + ", ".join(missing_features)
        )

    arch = (
        arch.set_index("feature")
        .reindex(EXPANDED17)
        .reset_index()
    )

    available = []
    removed = []

    for col, label in TERM_SPECS:
        if col not in arch.columns:
            removed.append({
                "term_column": col,
                "display_label": label,
                "reason": "column_missing",
                "n_finite": 0,
                "n_total": len(arch),
            })
            continue

        vals = pd.to_numeric(arch[col], errors="coerce").to_numpy(float)
        n_finite = int(np.isfinite(vals).sum())

        if n_finite == 0:
            removed.append({
                "term_column": col,
                "display_label": label,
                "reason": "all_nonfinite",
                "n_finite": 0,
                "n_total": len(arch),
            })
        else:
            available.append((col, label))
            removed.append({
                "term_column": col,
                "display_label": label,
                "reason": "retained",
                "n_finite": n_finite,
                "n_total": len(arch),
            })

    if not available:
        raise RuntimeError(
            "No estimable architecture terms remain after removing "
            "all-missing/all-nonfinite columns."
        )

    keep_cols = [x[0] for x in available]
    keep_labels = [x[1] for x in available]

    M = (
        arch[keep_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    if args.vmax is None:
        finite_abs = np.abs(M[np.isfinite(M)])
        vmax = max(
            0.15,
            float(np.nanpercentile(finite_abs, 98))
            if finite_abs.size
            else 0.15,
        )
    else:
        vmax = float(args.vmax)

    # Dynamic width: removing unavailable terms should actually shrink the panel.
    n_terms = len(available)
    fig_w = 4.7 + 1.15 * n_terms
    fig_h = 7.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D9D9D9")

    im = ax.imshow(
        np.ma.masked_invalid(M),
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(n_terms))
    ax.set_xticklabels(
        keep_labels,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_yticks(np.arange(len(EXPANDED17)))
    ax.set_yticklabels(EXPANDED17)

    ax.set_title(
        "Static architecture across Expanded17 features",
        pad=12,
    )

    ax.tick_params(axis="both", length=0)

    # Keep a clean manuscript-style frame.
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.045,
    )
    cbar.set_label("Unique / incremental adjusted R²")
    cbar.ax.tick_params(length=3)

    fig.subplots_adjust(
        left=0.18,
        right=0.86,
        bottom=0.15,
        top=0.93,
    )

    stem = "AHCA_Fig2D_feature_architecture_available_terms_only"
    save_figure(fig, outdir, stem)

    # Source-data table: only terms actually shown.
    out_source = arch[["feature"] + keep_cols].copy()
    out_source.to_csv(
        outdir / f"{stem}_source.csv",
        index=False,
        encoding="utf-8-sig",
    )

    audit = pd.DataFrame(removed)
    audit.to_csv(
        outdir / "AHCA_Fig2D_removed_terms_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("[RETAINED TERMS]")
    for col, label in available:
        print(f"  + {label} ({col})")

    print("[REMOVED TERMS]")
    dropped = [x for x in removed if x["reason"] != "retained"]
    if dropped:
        for x in dropped:
            print(
                f"  - {x['display_label']} ({x['term_column']}): "
                f"{x['reason']}"
            )
    else:
        print("  NONE")

    print(f"[COLOR] symmetric range = ±{vmax:.4f}")
    print()
    print("[DONE]")
    for ext in ("png", "pdf", "svg"):
        print(outdir / f"{stem}.{ext}")
    print(outdir / f"{stem}_source.csv")
    print(outdir / "AHCA_Fig2D_removed_terms_audit.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()
