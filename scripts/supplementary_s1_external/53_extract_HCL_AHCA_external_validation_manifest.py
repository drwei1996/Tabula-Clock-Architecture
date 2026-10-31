#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
53_extract_HCL_AHCA_external_validation_manifest.py

Purpose
-------
Extract an HCL/AHCA-specific submanifest from the already-built Tabula master
workflow manifest. This does NOT rerun any biology/statistics.

It searches the manifest tables for external-validation evidence related to:
- HCL / Human Cell Landscape
- AHCA / Adult Human Cell Atlas
- external validation / replication / validation
- known HCL/AHCA result terms used in this project

Outputs
-------
external_validation_manifest/
    HCL_AHCA_MASTER_SUMMARY.md
    01_relevant_scripts.csv
    02_relevant_workflow_rows.csv
    03_relevant_feature_evidence.csv
    04_relevant_pathway_evidence.csv
    05_relevant_stage_evidence.csv
    06_relevant_panel_gene_definitions.csv
    07_relevant_figure_files.csv
    08_all_keyword_hits_across_manifest.csv

The script is deliberately schema-tolerant: it searches all text columns rather
than assuming exact column names.
"""

from __future__ import annotations
import os

import argparse
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


DEFAULT_PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_MANIFEST = DEFAULT_PROJECT / "tabula_master_workflow_manifest"

KEYWORD_GROUPS = {
    "HCL": [
        r"\bHCL\b",
        r"Human Cell Landscape",
        r"human[_\s-]*cell[_\s-]*landscape",
    ],
    "AHCA": [
        r"\bAHCA\b",
        r"Adult Human Cell Atlas",
        r"adult[_\s-]*human[_\s-]*cell[_\s-]*atlas",
    ],
    "EXTERNAL_VALIDATION": [
        r"external[_\s-]*validation",
        r"external[_\s-]*replication",
        r"\breplication\b",
        r"\bvalidation\b",
        r"independent[_\s-]*atlas",
    ],
    "KNOWN_EXTERNAL_TERMS": [
        r"composition[_\s-]*controlled",
        r"comp[_\s-]*controlled",
        r"within[_\s-]*lineage",
        r"between[_\s-]*lineage",
        r"crossed[_\s-]*bootstrap",
        r"bayesian[_\s-]*bootstrap",
        r"tissue[_\s-]*bootstrap",
        r"priority[_\s-]*lineage",
        r"five[_\s-]*lineage",
        r"major[_\s-]*lineage",
    ],
}

PREFERRED_FILES = {
    "01_script_master_manifest.csv": "01_relevant_scripts.csv",
    "08_workflow_matrix.csv": "02_relevant_workflow_rows.csv",
    "02_feature_evidence.csv": "03_relevant_feature_evidence.csv",
    "03_pathway_evidence.csv": "04_relevant_pathway_evidence.csv",
    "04_stage_evidence.csv": "05_relevant_stage_evidence.csv",
    "05_panel_gene_definitions.csv": "06_relevant_panel_gene_definitions.csv",
    "07_final_figure_files.csv": "07_relevant_figure_files.csv",
}


def compile_patterns():
    compiled = {}
    for group, pats in KEYWORD_GROUPS.items():
        compiled[group] = [
            re.compile(p, flags=re.IGNORECASE)
            for p in pats
        ]
    return compiled


PATTERNS = compile_patterns()


def row_text(df: pd.DataFrame) -> pd.Series:
    text_cols = []
    for col in df.columns:
        s = df[col].astype(str).fillna("")
        text_cols.append(s)

    if not text_cols:
        return pd.Series([""] * len(df), index=df.index)

    joined = text_cols[0]
    for s in text_cols[1:]:
        joined = joined + " | " + s

    return joined


def classify_text(text: str):
    hits = []
    groups = []

    for group, pats in PATTERNS.items():
        local = []
        for pat in pats:
            if pat.search(text):
                local.append(pat.pattern)

        if local:
            groups.append(group)
            hits.extend(local)

    return groups, hits


def filter_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    joined = row_text(df)

    keep = []
    group_col = []
    hit_col = []

    for txt in joined.astype(str):
        groups, hits = classify_text(txt)

        # For HCL/AHCA submanifest, require HCL or AHCA OR explicit external-validation
        # wording. "KNOWN_EXTERNAL_TERMS" alone is too broad.
        selected = (
            "HCL" in groups
            or "AHCA" in groups
            or "EXTERNAL_VALIDATION" in groups
        )

        keep.append(selected)
        group_col.append(";".join(groups))
        hit_col.append(";".join(hits))

    out = df.loc[keep].copy()

    if len(out):
        out.insert(0, "_manifest_source", source_name)
        out.insert(1, "_keyword_groups", [g for g, k in zip(group_col, keep) if k])
        out.insert(2, "_keyword_hits", [h for h, k in zip(hit_col, keep) if k])

    return out


def locate_manifest_files(manifest_dir: Path):
    csvs = sorted(manifest_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in manifest directory:\n{manifest_dir}"
        )
    return csvs


def find_preferred_file(manifest_dir: Path, exact_name: str):
    p = manifest_dir / exact_name
    if p.exists():
        return p

    # Schema/name tolerance: try distinctive substrings.
    stem = Path(exact_name).stem.lower()
    tokens = [
        t for t in re.split(r"[_\W]+", stem)
        if len(t) >= 4 and not t.isdigit()
    ]

    candidates = []
    for q in manifest_dir.glob("*.csv"):
        qname = q.stem.lower()
        score = sum(t in qname for t in tokens)
        if score:
            candidates.append((score, q))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1].name))
    return candidates[0][1]


def summarize_file_rows(df: pd.DataFrame):
    if df is None or df.empty:
        return 0, {}

    groups = defaultdict(int)

    if "_keyword_groups" in df.columns:
        for cell in df["_keyword_groups"].fillna("").astype(str):
            for g in cell.split(";"):
                if g:
                    groups[g] += 1

    return len(df), dict(groups)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to tabula_master_workflow_manifest.",
    )

    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Default: <project>/HCL_AHCA_external_validation_manifest",
    )

    args = ap.parse_args()

    manifest_dir = args.manifest.resolve()

    if not manifest_dir.exists():
        raise FileNotFoundError(
            f"Manifest directory does not exist:\n{manifest_dir}"
        )

    outdir = (
        args.outdir.resolve()
        if args.outdir is not None
        else manifest_dir.parent / "HCL_AHCA_external_validation_manifest"
    )

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("EXTRACT HCL / AHCA EXTERNAL-VALIDATION SUBMANIFEST")
    print("=" * 108)
    print(f"MANIFEST: {manifest_dir}")
    print(f"OUT     : {outdir}")
    print()

    outputs = {}
    all_hits = []

    # ------------------------------------------------------------------
    # Preferred high-value manifest tables
    # ------------------------------------------------------------------
    for expected_name, output_name in PREFERRED_FILES.items():
        src = find_preferred_file(
            manifest_dir,
            expected_name,
        )

        if src is None:
            print(f"[SKIP] Could not locate manifest table similar to {expected_name}")
            continue

        print(f"[READ] {src.name}")

        try:
            df = pd.read_csv(
                src,
                low_memory=False,
            )
        except Exception as e:
            print(f"[WARN] Failed to read {src}: {e}")
            continue

        hit = filter_df(
            df,
            src.name,
        )

        hit.to_csv(
            outdir / output_name,
            index=False,
        )

        outputs[output_name] = hit

        if len(hit):
            all_hits.append(hit)

        n, groups = summarize_file_rows(hit)

        print(
            f"       relevant rows={n:,}"
            + (
                f" | groups={groups}"
                if groups
                else ""
            )
        )

    # ------------------------------------------------------------------
    # Search every other manifest CSV too
    # ------------------------------------------------------------------
    known_sources = set(
        p.name
        for p in [
            find_preferred_file(manifest_dir, x)
            for x in PREFERRED_FILES
        ]
        if p is not None
    )

    extra_hits = []

    for src in locate_manifest_files(
        manifest_dir
    ):
        if src.name in known_sources:
            continue

        try:
            df = pd.read_csv(
                src,
                low_memory=False,
            )
        except Exception:
            continue

        hit = filter_df(
            df,
            src.name,
        )

        if len(hit):
            extra_hits.append(hit)
            all_hits.append(hit)

    if extra_hits:
        pd.concat(
            extra_hits,
            ignore_index=True,
            sort=False,
        ).to_csv(
            outdir / "08_extra_manifest_keyword_hits.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Unified all-hit table
    # ------------------------------------------------------------------
    if all_hits:
        all_df = pd.concat(
            all_hits,
            ignore_index=True,
            sort=False,
        )
    else:
        all_df = pd.DataFrame()

    all_df.to_csv(
        outdir / "09_all_keyword_hits_across_manifest.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # A compact path/script-oriented table
    # ------------------------------------------------------------------
    if len(all_df):
        path_cols = [
            c for c in all_df.columns
            if any(
                k in c.lower()
                for k in [
                    "path",
                    "file",
                    "script",
                    "figure",
                    "output",
                    "input",
                    "dir",
                ]
            )
        ]

        keep_cols = [
            "_manifest_source",
            "_keyword_groups",
            "_keyword_hits",
        ] + path_cols

        keep_cols = [
            c for c in dict.fromkeys(keep_cols)
            if c in all_df.columns
        ]

        path_view = (
            all_df[keep_cols]
            .drop_duplicates()
            .copy()
        )

        path_view.to_csv(
            outdir / "10_external_validation_paths_and_scripts.csv",
            index=False,
        )
    else:
        path_view = pd.DataFrame()

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    lines = []

    lines.append("# HCL / AHCA External Validation Manifest")
    lines.append("")
    lines.append(
        "This submanifest was extracted from the completed Tabula master workflow manifest."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "- HCL / Human Cell Landscape"
    )
    lines.append(
        "- AHCA / Adult Human Cell Atlas"
    )
    lines.append(
        "- explicit external-validation / replication evidence"
    )
    lines.append("")
    lines.append(
        "**No biological/statistical analysis was rerun by this extraction script.**"
    )
    lines.append("")

    lines.append("## Relevant rows by manifest table")
    lines.append("")

    for name, df in outputs.items():
        n, groups = summarize_file_rows(df)

        group_text = (
            ", ".join(
                f"{k}={v}"
                for k, v in groups.items()
            )
            if groups
            else "none"
        )

        lines.append(
            f"- `{name}`: **{n:,}** relevant rows ({group_text})"
        )

    lines.append("")
    lines.append(
        f"- Unified keyword-hit rows: **{len(all_df):,}**"
    )
    lines.append(
        f"- Compact path/script rows: **{len(path_view):,}**"
    )
    lines.append("")

    lines.append("## Open next")
    lines.append("")
    lines.append(
        "1. `10_external_validation_paths_and_scripts.csv` — fastest way to identify the exact HCL/AHCA scripts and result directories."
    )
    lines.append(
        "2. `01_relevant_scripts.csv` — scripts/notebooks containing HCL/AHCA/external-validation evidence."
    )
    lines.append(
        "3. `02_relevant_workflow_rows.csv` — place each script/result in the workflow."
    )
    lines.append(
        "4. `07_relevant_figure_files.csv` — existing external-validation figures, if the master manifest indexed them."
    )
    lines.append("")

    lines.append("## Intended next step")
    lines.append("")
    lines.append(
        "Use this submanifest to identify the final HCL and AHCA validation chain, then freeze the exact scripts, source tables, estimands, and figures before any further reruns."
    )
    lines.append("")

    summary_path = (
        outdir
        / "HCL_AHCA_MASTER_SUMMARY.md"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("DONE")
    print("=" * 108)
    print(f"Unified relevant rows : {len(all_df):,}")
    print(f"Path/script rows       : {len(path_view):,}")
    print()
    print("OPEN FIRST:")
    print(summary_path)
    print(outdir / "10_external_validation_paths_and_scripts.csv")
    print(outdir / "01_relevant_scripts.csv")
    print(outdir / "02_relevant_workflow_rows.csv")
    print("=" * 108)


if __name__ == "__main__":
    main()
