"""
58_run_HCL_AHCA_Fig1_Fig2_single_panels.py

Same computation as the 56-script, but outputs one panel per file instead of
pre-assembling multi-panel figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# Loader
# =============================================================================

def load_base_module(base_script: Path):
    if not base_script.exists():
        raise FileNotFoundError(f"Base script not found: {base_script}")

    spec = importlib.util.spec_from_file_location("tabula_external_fig12_base", base_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script: {base_script}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Generic plotting helpers
# =============================================================================

def finish_and_save(m, fig, outdir: Path, stem: str):
    fig.tight_layout()
    m.save_fig(fig, outdir, stem)


def safe_height_by_rows(n_rows: int, base=1.2, per_row=0.23, min_h=2.7, max_h=10.5):
    h = base + per_row * max(1, int(n_rows))
    return float(max(min_h, min(max_h, h)))


# =============================================================================
# Figure 1 single panels
# =============================================================================

def save_fig1_panels(m, dataset: str, pb: pd.DataFrame, core_cols: list[str], expanded_cols: list[str], figdir: Path, srcdir: Path):
    panels = m.mkdir(figdir / "panels")
    src = m.mkdir(srcdir)

    heat = m.build_celltype_heatmap_table(pb, expanded_cols)
    m.write_csv(heat, src, "Fig1_celltype_expanded17_heatmap.csv")

    # Fig1A
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    emb_core = m.plot_umap_panel(ax, pb, core_cols, f"{dataset}: clock-architecture UMAP — Core panel")
    finish_and_save(m, fig, panels, f"{dataset}_Fig1A_core_clock_umap")

    # Fig1B
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    emb_exp = m.plot_umap_panel(ax, pb, expanded_cols, f"{dataset}: clock-architecture UMAP — Expanded17")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=m.LINEAGE_COLORS[l],
               markeredgecolor="white", markersize=6, label=m.LINEAGE_DISPLAY[l])
        for l in m.LINEAGE_ORDER
    ]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Major lineage")
    finish_and_save(m, fig, panels, f"{dataset}_Fig1B_expanded17_umap")

    emb_out = pb[["dataset", "group_id", "donor_id", "tissue", "cell_type", "major_lineage", "n_cells"]].copy()
    emb_out["core_dim1"] = emb_core[:, 0]
    emb_out["core_dim2"] = emb_core[:, 1]
    emb_out["expanded17_dim1"] = emb_exp[:, 0]
    emb_out["expanded17_dim2"] = emb_exp[:, 1]
    m.write_csv(emb_out, src, "Fig1_UMAP_coordinates.csv")

    # Fig1C
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    im = m.draw_global_heatmap(ax, heat, expanded_cols, f"{dataset}: global static clock-associated expression landscape")
    cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
    cb.set_label("Mean standardized feature value")
    finish_and_save(m, fig, panels, f"{dataset}_Fig1C_global_clock_landscape")

    # Fig1D-H lineage zooms
    letter_map = {
        "b_cell": "D",
        "t_cell": "E",
        "macrophage_monocyte": "F",
        "fibroblast": "G",
        "endothelial": "H",
    }
    name_map = {
        "b_cell": "bcell",
        "t_cell": "tcell",
        "macrophage_monocyte": "macrophage_monocyte",
        "fibroblast": "fibroblast",
        "endothelial": "endothelial",
    }

    for lin in m.LINEAGE_ORDER:
        g = heat[heat["major_lineage"].eq(lin)].copy()
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.0, safe_height_by_rows(len(g), base=1.1, per_row=0.28, min_h=2.8, max_h=9.0)))
        im = m.draw_lineage_zoom(ax, heat, expanded_cols, lin)
        cb = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02)
        cb.set_label("Mean standardized feature value")
        finish_and_save(m, fig, panels, f"{dataset}_Fig1{letter_map[lin]}_{name_map[lin]}_zoom")

    manifest = pd.DataFrame([
        [f"{dataset}_Fig1A_core_clock_umap", "Fig1A", "Core-clock UMAP", "Figure1_like/panels"],
        [f"{dataset}_Fig1B_expanded17_umap", "Fig1B", "Expanded17 UMAP", "Figure1_like/panels"],
        [f"{dataset}_Fig1C_global_clock_landscape", "Fig1C", "Global dataset-native cell-type landscape", "Figure1_like/panels"],
        [f"{dataset}_Fig1D_bcell_zoom", "Fig1D", "B cell zoom-in landscape", "Figure1_like/panels"],
        [f"{dataset}_Fig1E_tcell_zoom", "Fig1E", "T cell zoom-in landscape", "Figure1_like/panels"],
        [f"{dataset}_Fig1F_macrophage_monocyte_zoom", "Fig1F", "Macrophage/Monocyte zoom-in landscape", "Figure1_like/panels"],
        [f"{dataset}_Fig1G_fibroblast_zoom", "Fig1G", "Fibroblast zoom-in landscape", "Figure1_like/panels"],
        [f"{dataset}_Fig1H_endothelial_zoom", "Fig1H", "Endothelial zoom-in landscape", "Figure1_like/panels"],
    ], columns=["file_stem", "panel", "description", "relative_dir"])
    m.write_csv(manifest, figdir, "PANEL_MANIFEST_Figure1.csv")


# =============================================================================
# Figure 2 single panels
# =============================================================================

def save_fig2_panels(m, dataset: str, pb: pd.DataFrame, feat_cols: list[str], root_dataset_dir: Path, rigor: pd.DataFrame, n_boot: int, figdir: Path, srcdir: Path):
    panels = m.mkdir(figdir / "panels")
    src = m.mkdir(srcdir)

    within = m.read_csv(root_dataset_dir / "source_data" / "within_lineage_between_tissue_pairs.csv")
    between = m.read_csv(root_dataset_dir / "source_data" / "between_lineage_within_tissue_pairs.csv")
    cent = m.read_csv(root_dataset_dir / "source_data" / "tissue_lineage_centroids.csv")
    rigor_row = m.get_rigor_row(rigor, dataset)

    print(f"[{dataset}] higher-level bootstrap for lineage-specific remodeling: {n_boot} draws")
    boot = m.bootstrap_lineage_remodeling(pb, feat_cols, n_boot=n_boot, seed=m.SEED + (1 if dataset == "HCL" else 2))
    arch = m.compute_feature_architecture(pb, feat_cols)
    lt = m.compute_lineage_tissue_contribution(pb, feat_cols)

    m.write_csv(within, src, "Fig2_within_lineage_pairs.csv")
    m.write_csv(between, src, "Fig2_between_lineage_pairs.csv")
    m.write_csv(boot, src, "Fig2_lineage_specific_bootstrap.csv")
    m.write_csv(arch, src, "Fig2_feature_architecture.csv")
    m.write_csv(lt, src, "Fig2_lineage_tissue_contribution.csv")

    # Fig2A
    fig, ax = plt.subplots(figsize=(5.3, 4.2))
    m.draw_fig2A(ax, within, between, rigor_row, dataset)
    finish_and_save(m, fig, panels, f"{dataset}_Fig2A_tissue_associated_remodeling")

    # Fig2B
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    m.draw_fig2B(ax, boot)
    finish_and_save(m, fig, panels, f"{dataset}_Fig2B_lineage_specific_remodeling")

    # Fig2C1 / C2
    fib_labels, fib_M = m.tissue_distance_matrix(cent, feat_cols, "fibroblast")
    b_labels, b_M = m.tissue_distance_matrix(cent, feat_cols, "b_cell")

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = m.draw_distance_heatmap(ax, fib_labels, fib_M, "Fibroblast")
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cb.set_label("Euclidean distance")
    finish_and_save(m, fig, panels, f"{dataset}_Fig2C1_fibroblast_cross_tissue_architecture")

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = m.draw_distance_heatmap(ax, b_labels, b_M, "B cell")
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
        cb.set_label("Euclidean distance")
    finish_and_save(m, fig, panels, f"{dataset}_Fig2C2_bcell_cross_tissue_architecture")

    # Fig2D
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = m.draw_architecture_heatmap(ax, arch)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.03)
    cb.set_label("Unique / incremental adjusted R²")
    if pb["donor_id"].astype(str).nunique() == 1:
        ax.text(0.99, 1.02, "Donor term = NA (single-donor atlas)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color="#666666")
    finish_and_save(m, fig, panels, f"{dataset}_Fig2D_feature_architecture")

    # Fig2E
    fig, ax = plt.subplots(figsize=(6.9, 5.8))
    im = m.draw_lineage_tissue_heatmap(ax, lt)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.03)
    cb.set_label("Tissue unique adjusted R²")
    finish_and_save(m, fig, panels, f"{dataset}_Fig2E_lineage_specific_tissue_contribution")

    manifest = pd.DataFrame([
        [f"{dataset}_Fig2A_tissue_associated_remodeling", "Fig2A", "Within-lineage vs between-lineage distance", "Figure2_like/panels"],
        [f"{dataset}_Fig2B_lineage_specific_remodeling", "Fig2B", "Lineage-specific tissue remodeling with bootstrap CI", "Figure2_like/panels"],
        [f"{dataset}_Fig2C1_fibroblast_cross_tissue_architecture", "Fig2C1", "Fibroblast cross-tissue architecture", "Figure2_like/panels"],
        [f"{dataset}_Fig2C2_bcell_cross_tissue_architecture", "Fig2C2", "B cell cross-tissue architecture", "Figure2_like/panels"],
        [f"{dataset}_Fig2D_feature_architecture", "Fig2D", "Feature architecture", "Figure2_like/panels"],
        [f"{dataset}_Fig2E_lineage_specific_tissue_contribution", "Fig2E", "Lineage-specific tissue contribution across Expanded17", "Figure2_like/panels"],
    ], columns=["file_stem", "panel", "description", "relative_dir"])
    m.write_csv(manifest, figdir, "PANEL_MANIFEST_Figure2.csv")


# =============================================================================
# Dataset runner
# =============================================================================

def run_dataset(m, dataset: str, root: Path, project: Path, outroot: Path, core_genes: list[str], rigor: pd.DataFrame, n_boot: int):
    print("=" * 108)
    print(f"{dataset}: FIGURE 1 / FIGURE 2 REPLICATION (SINGLE PANELS)")
    print("=" * 108)

    droot = root / "01_HCL" if dataset == "HCL" else root / "02_AHCA_15ORG"
    pb_path = droot / "source_data" / "pseudobulk_standardized.csv"
    pb = m.read_csv(pb_path)

    required_meta = ["dataset", "group_id", "donor_id", "tissue", "cell_type", "major_lineage", "n_cells"]
    missing_meta = [c for c in required_meta if c not in pb.columns]
    if missing_meta:
        raise RuntimeError(f"{dataset}: missing metadata columns: {missing_meta}")

    pb = pb[pb["major_lineage"].astype(str).isin(m.LINEAGE_ORDER)].copy()
    expanded_cols = m.feature_columns(pb, m.EXPANDED17)
    core_cols = m.feature_columns(pb, core_genes)

    print(f"[{dataset}] rows={len(pb)}, donors={pb.donor_id.nunique()}, tissues={pb.tissue.nunique()}, cell_types={pb.cell_type.nunique()}")

    out = m.mkdir(outroot / dataset)
    fig1 = m.mkdir(out / "Figure1_like")
    fig2 = m.mkdir(out / "Figure2_like")

    save_fig1_panels(m, dataset, pb, core_cols, expanded_cols, fig1, fig1 / "source_data")
    save_fig2_panels(m, dataset, pb, expanded_cols, droot, rigor, n_boot, fig2, fig2 / "source_data")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument(
        "--base-script",
        type=Path,
        default=None,
        help="Path to 56_run_HCL_AHCA_Fig1_Fig2_replication.py; defaults to the sibling file in the same scripts directory.",
    )
    args = ap.parse_args()

    project = args.project.resolve()
    root = args.root.resolve()
    outroot = args.outdir.resolve()
    outroot.mkdir(parents=True, exist_ok=True)

    base_script = args.base_script
    if base_script is None:
        base_script = Path(__file__).resolve().with_name("56_run_HCL_AHCA_Fig1_Fig2_replication.py")
    else:
        base_script = base_script.resolve()

    m = load_base_module(base_script)

    print("=" * 108)
    print("HCL / AHCA: REPRODUCE CURRENT TABULA FIGURE 1 + FIGURE 2 (SINGLE PANEL OUTPUT)")
    print("=" * 108)
    print(f"PROJECT    : {project}")
    print(f"ROOT       : {root}")
    print(f"OUT        : {outroot}")
    print(f"BOOT       : {args.bootstrap}")
    print(f"BASE SCRIPT: {base_script}")

    core_genes = m.recover_core_genes(project)
    rigor_path = root / "03_RIGOR_VALIDATION_V1_20260921" / "FINAL_RIGOR_SUMMARY.csv"
    rigor = m.read_csv(rigor_path)

    for dataset in ["HCL", "AHCA"]:
        run_dataset(m, dataset, root, project, outroot, core_genes, rigor, args.bootstrap)

    readiness = pd.DataFrame([
        m.detect_fig3_readiness(root, "HCL"),
        m.detect_fig3_readiness(root, "AHCA"),
    ])
    readiness.to_csv(outroot / "FIG3_READINESS.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "status": "COMPLETE",
        "analysis": "External reproduction of current Tabula Figure 1 and Figure 2 with single-panel outputs",
        "datasets": ["HCL", "AHCA"],
        "feature_space": "Expanded17 primary; exact recovered Core panel for Core UMAP",
        "bootstrap_draws": args.bootstrap,
        "static_expression_only": True,
        "no_temporal_claims": True,
        "assembled_composite_figures": False,
        "single_panel_outputs": True,
        "figure3_readiness": readiness.to_dict("records"),
    }
    (outroot / "ANALYSIS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 108)
    print("COMPLETE")
    print("=" * 108)
    print("Generated panel directories:")
    print(outroot / "HCL" / "Figure1_like" / "panels")
    print(outroot / "HCL" / "Figure2_like" / "panels")
    print(outroot / "AHCA" / "Figure1_like" / "panels")
    print(outroot / "AHCA" / "Figure2_like" / "panels")
    print(outroot / "FIG3_READINESS.csv")
    print("=" * 108)


if __name__ == "__main__":
    main()