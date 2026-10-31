#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
54_freeze_HCL_AHCA_final_chain.py

Purpose
-------
Freeze the FINAL HCL/AHCA external-validation chain that was identified by the
completed Tabula master workflow manifest.

This script DOES NOT rerun biological/statistical analyses.

It:
1) Locates the frozen V4 baseline external-validation workflow.
2) Locates the final rigor and composition-audit scripts.
3) Locates the manuscript-ready package and its source data / documentation.
4) Inventories HCL/AHCA result files and existing figures.
5) Copies small final scripts/reports/tables into one compact freeze folder.
   Large raw objects (h5ad, npz, parquet, pickle, etc.) are referenced but NOT copied.
6) Writes a figure-reconstruction matrix for external Fig1/Fig2/Fig3-style panels.

Run from:
    <TABULA_PROJECT_ROOT>

Output:
    HCL_AHCA_FINAL_FREEZE_20260922
"""

from __future__ import annotations
import os

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

SMALL_COPY_SUFFIXES = {
    ".py", ".r", ".R", ".csv", ".tsv", ".txt", ".md", ".json",
}

DO_NOT_COPY_SUFFIXES = {
    ".h5ad", ".h5", ".hdf5", ".npz", ".npy", ".pkl", ".pickle",
    ".parquet", ".loom", ".mtx", ".gz",
}


def sha256(path: Path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_rel(path: Path, project: Path):
    try:
        return path.relative_to(project)
    except Exception:
        return path


def list_files(root: Path):
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
    )


def candidate_existing(*paths):
    return [p for p in paths if p.exists()]


def choose_first(*paths):
    xs = candidate_existing(*paths)
    return xs[0] if xs else None


def copy_small_file(src: Path, dst_root: Path, project: Path):
    rel = safe_rel(src, project)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def inventory_records(files, project, category):
    rows = []
    for p in files:
        try:
            size = p.stat().st_size
        except Exception:
            size = None

        rows.append({
            "category": category,
            "path": str(p),
            "relative_path": str(safe_rel(p, project)),
            "filename": p.name,
            "suffix": p.suffix.lower(),
            "size_bytes": size,
            "exists": p.exists(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
    )

    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--copy-max-mb",
        type=float,
        default=20.0,
        help="Copy only small text/table/code files <= this size.",
    )

    args = ap.parse_args()

    project = args.project.resolve()

    if not project.exists():
        raise FileNotFoundError(project)

    outdir = (
        args.outdir.resolve()
        if args.outdir is not None
        else project / "HCL_AHCA_FINAL_FREEZE_20260922"
    )

    copied_root = outdir / "frozen_files"
    outdir.mkdir(parents=True, exist_ok=True)
    copied_root.mkdir(parents=True, exist_ok=True)

    ext_root = (
        project
        / "external validation"
        / "clock_external_validation_ALL_EXCEPT_H5AD_20260921"
        / "clock_external_validation"
    )

    v4_freeze = (
        ext_root
        / "03_RIGOR_VALIDATION_V1_20260921"
        / "00_FREEZE_V4"
    )

    rigor_root = (
        ext_root
        / "03_RIGOR_VALIDATION_V1_20260921"
    )

    comp_root = (
        ext_root
        / "04_COMPOSITION_AUDIT_STRICT_V1_20260921"
    )

    manuscript_root = (
        project
        / "external validation"
        / "MANUSCRIPT_READY_EXTERNAL_VALIDATION_20260921"
        / "05_MANUSCRIPT_EXTERNAL_VALIDATION_20260921"
    )

    fontfix_root = (
        project
        / "external validation"
        / "MANUSCRIPT_READY_EXTERNAL_VALIDATION_FONTFIX_20260921"
        / "05_MANUSCRIPT_EXTERNAL_VALIDATION_20260921"
    )

    # ------------------------------------------------------------------
    # Exact high-priority final scripts
    # ------------------------------------------------------------------
    final_scripts = [
        choose_first(
            v4_freeze / "run_external_validation_all_v4.py",
            project / "corrected_full_analysis_v2" / "scripts" / "run_external_validation_all_v4.py",
        ),
        choose_first(
            rigor_root / "run_external_validation_rigor_v1_1.py",
            project / "corrected_full_analysis_v2" / "scripts" / "run_external_validation_rigor_v1_1.py",
        ),
        choose_first(
            comp_root / "run_composition_audit_strict_v1_1.py",
            project / "corrected_full_analysis_v2" / "scripts" / "run_composition_audit_strict_v1_1.py",
        ),
        choose_first(
            manuscript_root / "05_DOCUMENTATION" / "build_external_validation_manuscript_package_v1.py",
            project / "corrected_full_analysis_v2" / "scripts" / "build_external_validation_manuscript_package_v1_1.py",
            project / "corrected_full_analysis_v2" / "scripts" / "build_external_validation_manuscript_package_v1.py",
        ),
    ]

    final_scripts = [p for p in final_scripts if p is not None]

    # ------------------------------------------------------------------
    # Final HCL/AHCA summaries / reports
    # ------------------------------------------------------------------
    high_priority_files = []

    for p in [
        v4_freeze / "01_HCL" / "VALIDATION_SUMMARY.json",
        v4_freeze / "02_AHCA_15ORG" / "VALIDATION_SUMMARY.json",
        v4_freeze / "FINAL_REPORT.md",
        v4_freeze / "PROTOCOL_FROZEN_BEFORE_SENSITIVITY.json",
        rigor_root / "FINAL_RIGOR_REPORT.md",
        manuscript_root / "05_DOCUMENTATION" / "RIGOR_FINAL_REPORT.md",
        manuscript_root / "05_DOCUMENTATION" / "V4_FINAL_REPORT.md",
        manuscript_root / "README_MANUSCRIPT_PACKAGE.md",
        manuscript_root / "01_SOURCE_DATA" / "EXT_source_02_lineage_remodeling_three_atlases.csv",
    ]:
        if p.exists():
            high_priority_files.append(p)

    # ------------------------------------------------------------------
    # Inventory full relevant roots
    # ------------------------------------------------------------------
    roots = {
        "V4_FREEZE": v4_freeze,
        "RIGOR": rigor_root,
        "COMPOSITION_AUDIT": comp_root,
        "MANUSCRIPT_READY": manuscript_root,
        "MANUSCRIPT_FONTFIX": fontfix_root,
    }

    inventory = []

    for category, root in roots.items():
        for row in inventory_records(
            list_files(root),
            project,
            category,
        ):
            inventory.append(row)

    inv = pd.DataFrame(inventory)

    inv.to_csv(
        outdir / "01_full_external_validation_inventory.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Classify likely source tables and figures
    # ------------------------------------------------------------------
    if len(inv):
        low = inv["relative_path"].astype(str).str.lower()

        source_mask = (
            low.str.contains("source_data")
            | low.str.contains("summary")
            | low.str.contains("report")
            | low.str.contains("result")
        )

        fig_mask = (
            inv["suffix"].isin(
                [".png", ".pdf", ".svg", ".tif", ".tiff"]
            )
            | low.str.contains("figure")
            | low.str.contains("fig")
        )

        likely_source = inv[source_mask].copy()
        likely_fig = inv[fig_mask].copy()
    else:
        likely_source = pd.DataFrame()
        likely_fig = pd.DataFrame()

    likely_source.to_csv(
        outdir / "02_likely_source_tables_and_reports.csv",
        index=False,
    )

    likely_fig.to_csv(
        outdir / "03_existing_external_validation_figures.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Copy small final files
    # ------------------------------------------------------------------
    max_bytes = int(
        args.copy_max_mb
        * 1024
        * 1024
    )

    copy_candidates = []

    copy_candidates.extend(
        final_scripts
    )

    copy_candidates.extend(
        high_priority_files
    )

    if len(likely_source):
        for p in likely_source["path"].astype(str):
            q = Path(p)
            if (
                q.exists()
                and q.suffix.lower() in SMALL_COPY_SUFFIXES
                and q.stat().st_size <= max_bytes
            ):
                copy_candidates.append(q)

    unique_candidates = []
    seen = set()

    for p in copy_candidates:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            unique_candidates.append(p)

    copy_log = []

    for src in unique_candidates:
        if not src.exists():
            continue

        if (
            src.suffix.lower() in DO_NOT_COPY_SUFFIXES
            or src.stat().st_size > max_bytes
        ):
            copy_log.append({
                "source": str(src),
                "copied": False,
                "reason": "large_or_binary_reference_only",
                "destination": "",
                "sha256": "",
            })
            continue

        dst = copy_small_file(
            src,
            copied_root,
            project,
        )

        copy_log.append({
            "source": str(src),
            "copied": True,
            "reason": "small_final_file",
            "destination": str(dst),
            "sha256": sha256(src),
        })

    pd.DataFrame(
        copy_log
    ).to_csv(
        outdir / "04_freeze_copy_log.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figure reconstruction matrix
    # ------------------------------------------------------------------
    figure_plan = pd.DataFrame([
        {
            "external_panel_group": "EV-Fig1-like",
            "target_question": "Does the major-lineage clock-architecture backbone reproduce externally?",
            "HCL": "Yes",
            "AHCA": "Yes",
            "primary_source": "Frozen V4 distance/validation outputs",
            "preferred_outputs": "within-vs-between lineage summary; bootstrap/CI; external atlas overview",
            "status": "reconstruct from frozen final outputs",
        },
        {
            "external_panel_group": "EV-Fig2-like",
            "target_question": "Does lineage-specific tissue remodeling reproduce externally?",
            "HCL": "Yes, cautious",
            "AHCA": "Yes",
            "primary_source": "Frozen V4 lineage-remodeling source tables + rigor results",
            "preferred_outputs": "five-lineage tissue-remodeling plot; composition-controlled sensitivity",
            "status": "reconstruct from frozen final outputs",
        },
        {
            "external_panel_group": "EV-Fig3-like",
            "target_question": "How far can the functional/pathway architecture be reproduced in the external atlas?",
            "HCL": "Only if compatible pathway/source outputs exist",
            "AHCA": "Only if compatible pathway/source outputs exist",
            "primary_source": "Final manuscript-ready source tables / existing external scripts",
            "preferred_outputs": "use same visual language as Tabula Fig3, but do not invent unavailable analyses",
            "status": "audit available source tables before plotting",
        },
    ])

    figure_plan.to_csv(
        outdir / "05_external_Fig1_Fig2_Fig3_reconstruction_plan.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final chain table
    # ------------------------------------------------------------------
    chain_rows = []

    def add_chain(stage, p, role):
        chain_rows.append({
            "stage": stage,
            "role": role,
            "path": str(p) if p is not None else "",
            "exists": bool(
                p is not None
                and p.exists()
            ),
        })

    add_chain(
        "V4_BASELINE",
        choose_first(
            v4_freeze / "run_external_validation_all_v4.py",
            project / "corrected_full_analysis_v2" / "scripts" / "run_external_validation_all_v4.py",
        ),
        "Primary external-validation analysis",
    )

    add_chain(
        "RIGOR",
        choose_first(
            rigor_root / "run_external_validation_rigor_v1_1.py",
            project / "corrected_full_analysis_v2" / "scripts" / "run_external_validation_rigor_v1_1.py",
        ),
        "Bootstrap / leave-one-out / fine-identity rigor",
    )

    add_chain(
        "COMPOSITION",
        choose_first(
            comp_root / "run_composition_audit_strict_v1_1.py",
            project / "corrected_full_analysis_v2" / "scripts" / "run_composition_audit_strict_v1_1.py",
        ),
        "Strict composition-control sensitivity",
    )

    add_chain(
        "MANUSCRIPT_PACKAGE",
        manuscript_root,
        "Final source-data/report packaging",
    )

    add_chain(
        "FONTFIX_PACKAGE",
        fontfix_root,
        "Optional display-only font-fixed manuscript package",
    )

    pd.DataFrame(
        chain_rows
    ).to_csv(
        outdir / "06_final_validation_chain.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    lines = []

    lines.append("# HCL / AHCA Final External-Validation Freeze")
    lines.append("")
    lines.append("No biology/statistics were rerun.")
    lines.append("")
    lines.append("## Final chain")
    lines.append("")

    for r in chain_rows:
        lines.append(
            f"- **{r['stage']}** — {r['role']} — "
            f"{'FOUND' if r['exists'] else 'MISSING'} — `{r['path']}`"
        )

    lines.append("")
    lines.append("## Figure strategy")
    lines.append("")
    lines.append(
        "External figures should mirror the QUESTION and visual language of Tabula Figures 1–3, "
        "but only where the frozen HCL/AHCA outputs support the same estimand."
    )
    lines.append("")
    lines.append(
        "- EV-Fig1-like: external major-lineage clock architecture."
    )
    lines.append(
        "- EV-Fig2-like: external lineage-specific tissue remodeling + rigor/composition sensitivity."
    )
    lines.append(
        "- EV-Fig3-like: only reconstruct pathway/function panels if the final frozen external outputs actually contain compatible pathway-level evidence."
    )
    lines.append("")
    lines.append("## Open next")
    lines.append("")
    lines.append("1. `06_final_validation_chain.csv`")
    lines.append("2. `02_likely_source_tables_and_reports.csv`")
    lines.append("3. `03_existing_external_validation_figures.csv`")
    lines.append("4. `05_external_Fig1_Fig2_Fig3_reconstruction_plan.csv`")
    lines.append("")

    (
        outdir
        / "HCL_AHCA_FINAL_FREEZE_SUMMARY.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 108)
    print("HCL / AHCA FINAL EXTERNAL-VALIDATION FREEZE")
    print("=" * 108)
    print(f"Project     : {project}")
    print(f"Output      : {outdir}")
    print(f"Inventory   : {len(inv):,} files")
    print(f"Likely data : {len(likely_source):,}")
    print(f"Figures     : {len(likely_fig):,}")
    print(f"Copied      : {sum(bool(x['copied']) for x in copy_log):,} small final files")
    print()
    print("OPEN FIRST:")
    print(outdir / "HCL_AHCA_FINAL_FREEZE_SUMMARY.md")
    print(outdir / "06_final_validation_chain.csv")
    print(outdir / "02_likely_source_tables_and_reports.csv")
    print(outdir / "03_existing_external_validation_figures.csv")
    print("=" * 108)


if __name__ == "__main__":
    main()
