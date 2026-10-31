#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
60_search_AHCA_Fig3_inputs.py

Inventory a Tabula Sapiens project folder and identify the exact files needed
to build the FINAL AHCA Figure 3 A-F replication script.

This script DOES NOT rerun any biology/statistics.
It only:
  1) inventories files recursively;
  2) identifies likely AHCA all-gene inputs;
  3) identifies frozen Tabula Fig3 reference tables;
  4) inspects CSV/TSV/CSV.GZ headers;
  5) searches source-code/text files for Fig3-relevant keywords;
  6) writes a compact dependency/readiness report.

Outputs:
  00_file_inventory.csv
  01_fig3_candidate_files.csv
  02_tabular_schema_summary.csv
  03_keyword_hits.csv
  04_dependency_status.csv
  05_priority_files_to_share.csv
  SEARCH_SUMMARY.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
import pandas as pd


TEXT_EXTS = {
    ".py", ".r", ".txt", ".md", ".json", ".yaml", ".yml",
    ".csv", ".tsv", ".gmt", ".log", ".ps1",
}

SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".idea", ".vscode",
    "node_modules", ".conda", "site-packages",
}

ROLE_PATTERNS = {
    "AHCA_ALLGENE_MATRIX": [
        r"ahca.*allgene.*pseudobulk.*matrix",
        r"ahca.*all[_-]?gene.*matrix",
        r"allgene.*ahca.*matrix",
    ],
    "AHCA_ALLGENE_METADATA": [
        r"ahca.*allgene.*pseudobulk.*metadata",
        r"ahca.*all[_-]?gene.*metadata",
        r"allgene.*ahca.*metadata",
    ],
    "AHCA_ALLGENE_GENES": [
        r"ahca.*allgene.*genes",
        r"ahca.*all[_-]?gene.*genes",
    ],
    "AHCA_EXPORT_AUDIT": [
        r"ahca.*allgene.*audit",
        r"ahca.*export.*audit",
    ],
    "AHCA_RAW_COUNTS": [
        r"gse159929",
        r"_counts\.csv",
        r"_counts\.csv\.gz",
        r"ahca.*count",
    ],
    "AHCA_ANNOTATION": [
        r"annotation.*ahca",
        r"ahca.*annotation",
        r"alltissues.*meta",
    ],
    "TABULA_V3_ONTOLOGY": [
        r"01_v3_pathway_ontology",
        r"01_final_pathway_ontology",
        r"v3.*pathway.*ontology",
        r"final.*pathway.*ontology",
    ],
    "TABULA_FORMAL_HITS": [
        r"03_formal_hits_with_final_ontology",
        r"formal.*hits.*final.*ontology",
        r"01_formal_hits_combined",
        r"formal.*hits.*combined",
    ],
    "TABULA_COMPLETE_LINKAGE": [
        r"02_complete_linkage_clusters",
        r"complete.*linkage.*cluster",
        r"redundancy.*cluster",
    ],
    "TABULA_MODULE_SUMMARY": [
        r"module.*summary",
        r"biological.*module.*composition",
        r"module.*composition",
    ],
    "TABULA_REPRESENTATIVE_FAMILIES": [
        r"selected.*family.*ontology",
        r"representative.*family",
        r"representative.*coherent.*pathway",
        r"family.*ontology.*summary",
    ],
    "TABULA_FIG3_SOURCE": [
        r"fig3[a-z0-9_ -]*source",
        r"fig3.*selected",
        r"fig3.*bridge",
        r"fig3.*hallmark",
        r"fig3.*reactome",
    ],
    "HALLMARK_GMT": [
        r"hallmark.*\.gmt$",
        r"h\.all\..*\.gmt$",
    ],
    "REACTOME_GMT": [
        r"reactome.*\.gmt$",
        r"m2\.cp\.reactome.*\.gmt$",
    ],
    "A47_SCRIPT": [
        r"08_run_corrected_a4_7",
        r"a4_7.*pathway",
    ],
    "A48_SCRIPT": [
        r"a4_8",
        r"v3.*ontology",
    ],
    "MATCHED_COMPETITIVE_SCRIPT": [
        r"50_run_matched_competitive",
    ],
    "FIG3_REDRAW_SCRIPT": [
        r"51_redraw_fig3",
        r"52_update_ontology",
        r"fig3.*redraw",
    ],
}

KEYWORDS = [
    "specificity_supported",
    "specificity_z_vs_null",
    "specificity_empirical_P",
    "specificity_FDR_collection",
    "beta_standardized_pairwise_correspondence",
    "pairwise_correspondence_FDR_collection",
    "supported_positive_pairwise_correspondence",
    "complete_linkage_cluster_id",
    "representative_pathway",
    "representative_collection",
    "v3_primary_module_id",
    "v3_primary_module_label",
    "primary_module_id",
    "primary_module_label",
    "genes_scored",
    "cell_unit_id",
    "Fig3F_selected_Hallmark",
    "Fig3G_selected_Reactome",
    "matched_z",
    "FDR_matched",
    "delta_adjR2_tissue",
    "AHCA_allgene_pseudobulk_matrix",
    "AHCA_allgene_pseudobulk_metadata",
]

PRIORITY_COLUMNS = [
    "collection",
    "pathway",
    "genes_scored",
    "complete_linkage_cluster_id",
    "representative_pathway",
    "representative_collection",
    "specificity_supported",
    "specificity_z_vs_null",
    "specificity_empirical_P",
    "specificity_FDR_collection",
    "beta_standardized_pairwise_correspondence",
    "pairwise_correspondence_FDR_collection",
    "supported_positive_pairwise_correspondence",
    "v3_primary_module_id",
    "v3_primary_module_label",
    "primary_module_id",
    "primary_module_label",
    "cell_unit_id",
    "lineage",
    "fine_subtype",
    "matched_z",
    "FDR_matched",
    "supported",
    "delta_adjR2_tissue",
    "group_id",
    "donor_id",
    "tissue",
    "cell_type",
    "major_lineage",
    "n_cells",
]


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.1f} {u}"
        x /= 1024
    return f"{n} B"


def is_skipped(path: Path) -> bool:
    return any(part.lower() in SKIP_DIR_NAMES for part in path.parts)


def classify_roles(path: Path) -> list[str]:
    s = str(path).lower().replace("\\", "/")
    roles = []
    for role, patterns in ROLE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, s, flags=re.I):
                roles.append(role)
                break
    return roles


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if is_skipped(p):
            continue
        yield p


def suffix_key(path: Path) -> str:
    n = path.name.lower()
    if n.endswith(".csv.gz"):
        return ".csv.gz"
    if n.endswith(".tar.gz"):
        return ".tar.gz"
    return path.suffix.lower()


def safe_read_header(path: Path) -> tuple[list[str], str | None]:
    n = path.name.lower()
    try:
        if n.endswith(".csv.gz"):
            df = pd.read_csv(path, nrows=0, compression="gzip")
        elif n.endswith(".csv"):
            df = pd.read_csv(path, nrows=0)
        elif n.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", nrows=0)
        elif n.endswith(".txt"):
            try:
                df = pd.read_csv(path, sep="\t", nrows=0)
                if len(df.columns) <= 1:
                    df = pd.read_csv(path, nrows=0)
            except Exception:
                df = pd.read_csv(path, nrows=0)
        else:
            return [], "unsupported"
        return [str(c) for c in df.columns], None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def search_text_file(path: Path, keywords: list[str], max_hits_per_file: int = 50):
    hits = []
    kw_lower = [(k, k.lower()) for k in keywords]

    try:
        size = path.stat().st_size
    except Exception:
        size = 0

    if size > 50 * 1024 * 1024:
        return hits

    opener = gzip.open if path.name.lower().endswith(".gz") else open

    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                low = line.lower()
                for original, lowkw in kw_lower:
                    if lowkw in low:
                        hits.append({
                            "path": str(path),
                            "line": lineno,
                            "keyword": original,
                            "text": line.strip()[:500],
                        })
                        if len(hits) >= max_hits_per_file:
                            return hits
    except Exception:
        return hits

    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    root = args.root.resolve()
    outdir = mkdir(args.outdir.resolve())

    if not root.exists():
        raise FileNotFoundError(root)

    print("=" * 112)
    print("SEARCH TABULA / AHCA FIG3 INPUTS")
    print("=" * 112)
    print(f"ROOT: {root}")
    print(f"OUT : {outdir}")
    print()

    inventory_rows = []
    candidate_rows = []
    schema_rows = []
    keyword_rows = []

    files = list(iter_files(root))
    print(f"[SCAN] files found: {len(files):,}")

    for i, p in enumerate(files, start=1):
        try:
            size = int(p.stat().st_size)
        except Exception:
            size = -1

        roles = classify_roles(p)
        rel = norm_rel(p, root)

        inventory_rows.append({
            "path": str(p),
            "relative_path": rel,
            "name": p.name,
            "suffix": suffix_key(p),
            "size_bytes": size,
            "size_human": human_size(size) if size >= 0 else "",
            "roles": " | ".join(roles),
        })

        if roles:
            candidate_rows.append({
                "path": str(p),
                "relative_path": rel,
                "name": p.name,
                "size_bytes": size,
                "size_human": human_size(size) if size >= 0 else "",
                "roles": " | ".join(roles),
                "priority_score": len(roles),
            })

        nlow = p.name.lower()
        inspect_schema = (
            bool(roles)
            or "fig3" in nlow
            or "a4_7" in nlow
            or "a4_8" in nlow
            or "ontology" in nlow
            or "formal" in nlow
            or "pathway" in nlow
            or "allgene" in nlow
            or "pseudobulk" in nlow
        )

        if inspect_schema and (
            nlow.endswith(".csv")
            or nlow.endswith(".csv.gz")
            or nlow.endswith(".tsv")
            or nlow.endswith(".txt")
        ):
            cols, err = safe_read_header(p)
            priority_present = [c for c in PRIORITY_COLUMNS if c in cols]
            schema_rows.append({
                "path": str(p),
                "relative_path": rel,
                "name": p.name,
                "n_columns": len(cols),
                "columns": " | ".join(cols),
                "priority_columns_present": " | ".join(priority_present),
                "n_priority_columns_present": len(priority_present),
                "read_error": err or "",
            })

        if p.suffix.lower() in TEXT_EXTS or p.name.lower().endswith(".csv.gz"):
            if (
                p.suffix.lower() in {".py", ".r", ".txt", ".md", ".json", ".yaml", ".yml", ".gmt", ".ps1"}
                or bool(roles)
            ):
                hits = search_text_file(p, KEYWORDS)
                for h in hits:
                    h["relative_path"] = rel
                    keyword_rows.append(h)

        if i % 500 == 0:
            print(f"[SCAN] processed {i:,}/{len(files):,}")

    inv = pd.DataFrame(inventory_rows)
    cand = pd.DataFrame(candidate_rows)
    schemas = pd.DataFrame(schema_rows)
    khits = pd.DataFrame(keyword_rows)

    if len(cand):
        cand = cand.sort_values(
            ["priority_score", "roles", "relative_path"],
            ascending=[False, True, True],
        )

    if len(schemas):
        schemas = schemas.sort_values(
            ["n_priority_columns_present", "relative_path"],
            ascending=[False, True],
        )

    inv.to_csv(outdir / "00_file_inventory.csv", index=False, encoding="utf-8-sig")
    cand.to_csv(outdir / "01_fig3_candidate_files.csv", index=False, encoding="utf-8-sig")
    schemas.to_csv(outdir / "02_tabular_schema_summary.csv", index=False, encoding="utf-8-sig")
    khits.to_csv(outdir / "03_keyword_hits.csv", index=False, encoding="utf-8-sig")

    role_to_paths = {}
    for role in ROLE_PATTERNS:
        if len(cand):
            sub = cand[cand["roles"].astype(str).str.contains(role, regex=False)]
            role_to_paths[role] = sub["path"].tolist()
        else:
            role_to_paths[role] = []

    dependencies = [
        ("AHCA full-transcriptome matrix", "AHCA_ALLGENE_MATRIX", True),
        ("AHCA pseudobulk metadata", "AHCA_ALLGENE_METADATA", True),
        ("AHCA gene list", "AHCA_ALLGENE_GENES", False),
        ("Frozen Tabula V3 pathway ontology", "TABULA_V3_ONTOLOGY", True),
        ("Frozen Tabula formal hits", "TABULA_FORMAL_HITS", True),
        ("Frozen complete-linkage family mapping", "TABULA_COMPLETE_LINKAGE", True),
        ("Frozen module summary / composition", "TABULA_MODULE_SUMMARY", False),
        ("Representative family/source table", "TABULA_REPRESENTATIVE_FAMILIES", False),
        ("Hallmark GMT", "HALLMARK_GMT", True),
        ("Reactome GMT", "REACTOME_GMT", True),
        ("Original A4.7 pathway-statistics script", "A47_SCRIPT", True),
        ("Original A4.8/final ontology script", "A48_SCRIPT", True),
        ("Matched-competitive Script50", "MATCHED_COMPETITIVE_SCRIPT", True),
        ("Final Fig3 redraw scripts", "FIG3_REDRAW_SCRIPT", False),
    ]

    dep_rows = []
    for label, role, required in dependencies:
        paths = role_to_paths.get(role, [])
        dep_rows.append({
            "dependency": label,
            "role": role,
            "required_for_final_script": required,
            "found": len(paths) > 0,
            "n_candidates": len(paths),
            "best_candidate": paths[0] if paths else "",
        })

    deps = pd.DataFrame(dep_rows)
    deps.to_csv(outdir / "04_dependency_status.csv", index=False, encoding="utf-8-sig")

    priority_role_order = [
        "AHCA_ALLGENE_MATRIX",
        "AHCA_ALLGENE_METADATA",
        "AHCA_ALLGENE_GENES",
        "AHCA_EXPORT_AUDIT",
        "TABULA_V3_ONTOLOGY",
        "TABULA_FORMAL_HITS",
        "TABULA_COMPLETE_LINKAGE",
        "TABULA_MODULE_SUMMARY",
        "TABULA_REPRESENTATIVE_FAMILIES",
        "TABULA_FIG3_SOURCE",
        "HALLMARK_GMT",
        "REACTOME_GMT",
        "A47_SCRIPT",
        "A48_SCRIPT",
        "MATCHED_COMPETITIVE_SCRIPT",
        "FIG3_REDRAW_SCRIPT",
    ]

    pr_rows = []
    seen = set()

    for role in priority_role_order:
        paths = role_to_paths.get(role, [])
        for rank, path in enumerate(paths[:8], start=1):
            key = (role, path)
            if key in seen:
                continue
            seen.add(key)

            sch = schemas[schemas["path"].eq(path)] if len(schemas) else pd.DataFrame()
            pr_rows.append({
                "role": role,
                "rank_within_role": rank,
                "path": path,
                "relative_path": norm_rel(Path(path), root),
                "columns_if_tabular": sch.iloc[0]["columns"] if len(sch) else "",
                "priority_columns_present": sch.iloc[0]["priority_columns_present"] if len(sch) else "",
            })

    priority = pd.DataFrame(pr_rows)
    priority.to_csv(outdir / "05_priority_files_to_share.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# AHCA Figure 3 input search summary")
    lines.append("")
    lines.append(f"- Root: `{root}`")
    lines.append(f"- Files inventoried: **{len(inv):,}**")
    lines.append(f"- Fig3-relevant candidates: **{len(cand):,}**")
    lines.append(f"- Tabular schemas inspected: **{len(schemas):,}**")
    lines.append(f"- Keyword-hit rows: **{len(khits):,}**")
    lines.append("")
    lines.append("## Dependency status")
    lines.append("")
    lines.append("| Dependency | Required | Found | Candidates | Best candidate |")
    lines.append("|---|---:|---:|---:|---|")

    for r in deps.itertuples(index=False):
        best = str(r.best_candidate).replace("|", "\\|")
        lines.append(
            f"| {r.dependency} | {'YES' if r.required_for_final_script else 'No'} "
            f"| {'YES' if r.found else 'NO'} | {r.n_candidates} | `{best}` |"
        )

    lines.append("")
    lines.append("## Most informative candidate tables")
    lines.append("")

    if len(schemas):
        top = schemas.head(30)
        for r in top.itertuples(index=False):
            lines.append(
                f"- `{r.relative_path}`  \n"
                f"  columns={r.n_columns}; priority columns: {r.priority_columns_present or 'none'}"
            )
    else:
        lines.append("- No relevant tabular schema was readable.")

    lines.append("")
    lines.append("## What to send back")
    lines.append("")
    lines.append("Upload these first:")
    lines.append("1. `SEARCH_SUMMARY.md`")
    lines.append("2. `05_priority_files_to_share.csv`")
    lines.append("3. `02_tabular_schema_summary.csv`")
    lines.append("4. `03_keyword_hits.csv`")
    lines.append("")
    lines.append("If needed, also upload `01_fig3_candidate_files.csv`.")
    lines.append("")
    lines.append("Do NOT upload the AHCA all-gene NPZ unless specifically requested.")

    (outdir / "SEARCH_SUMMARY.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 112)
    print("SEARCH COMPLETE")
    print("=" * 112)
    print(outdir / "SEARCH_SUMMARY.md")
    print(outdir / "05_priority_files_to_share.csv")
    print(outdir / "02_tabular_schema_summary.csv")
    print(outdir / "03_keyword_hits.csv")
    print(outdir / "01_fig3_candidate_files.csv")
    print()
    print("Required dependencies missing:")
    miss = deps[deps["required_for_final_script"] & ~deps["found"]]
    if len(miss):
        for x in miss["dependency"]:
            print(f"  - {x}")
    else:
        print("  NONE")
    print("=" * 112)


if __name__ == "__main__":
    main()
