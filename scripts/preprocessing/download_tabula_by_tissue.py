from pathlib import Path
import argparse
import gc
import json
import re
import shutil
import sys
import traceback

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import cellxgene_census


# ============================================================
# USER CONFIG
# ============================================================

DATASET_ID = "53d208b0-2cfd-4366-9866-c3c6114081bc"

# Windows:
# <TABULA_PROJECT_ROOT>\katv2
#
# WSL sees it as:
PROJECT_ROOT = Path(
    "<TABULA_PROJECT_ROOT>/katv2"
)

OUT_DIR = PROJECT_ROOT / "03_TABULA_INPUT" / "by_tissue"
AUDIT_DIR = PROJECT_ROOT / "03_TABULA_INPUT" / "download_audit"

# 推荐先按高层级组织拆分。
# 如果以后想严格按原始 tissue 字段拆，只需要改成 "tissue"
GROUP_FIELD = "tissue_general"

VERSION_FILE = AUDIT_DIR / "CENSUS_VERSION.txt"
SUMMARY_FILE = AUDIT_DIR / "TABULA_TISSUE_DOWNLOAD_SUMMARY.csv"
LOG_FILE = AUDIT_DIR / "DOWNLOAD_LOG.txt"


OBS_COLUMNS = [
    "soma_joinid",
    "dataset_id",
    "donor_id",
    "tissue",
    "tissue_general",
    "cell_type",
    "cell_type_ontology_term_id",
    "assay",
    "sex",
    "suspension_type",
    "is_primary_data",
]

VAR_COLUMNS = [
    "soma_joinid",
    "feature_id",
    "feature_name",
]


# ============================================================
# HELPERS
# ============================================================

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def safe_filename(x):
    x = str(x).strip()
    x = re.sub(r'[<>:"/\\|?*]+', "_", x)
    x = re.sub(r"\s+", "_", x)
    return x[:150]


def filter_literal(x):
    """
    Quote string for SOMA value_filter.
    Most tissue names contain no apostrophe, but protect them anyway.
    """
    x = str(x).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{x}'"


def resolve_census_version():
    """
    On first run:
        resolve current stable Census release to exact build date.
    Later runs:
        always reuse same exact build.
    """
    if VERSION_FILE.exists():
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
        if version:
            return version

    desc = cellxgene_census.get_census_version_description("stable")
    version = desc["release_build"]

    VERSION_FILE.write_text(version + "\n", encoding="utf-8")
    return version


def raw_count_check(X):
    """
    Small audit only. Census 'raw' may be stored as floating dtype
    but values should behave as integer counts.
    """
    if sp.issparse(X):
        vals = X.data[: min(len(X.data), 100000)]
    else:
        arr = np.asarray(X)
        vals = arr.flat[: min(arr.size, 100000)]

    if len(vals) == 0:
        return True, 0.0

    vals = np.asarray(vals, dtype=float)
    err = float(np.max(np.abs(vals - np.round(vals))))

    return err < 1e-6, err


def verify_existing(path, expected_n):
    """
    Fast backed-mode verification of an already completed H5AD.
    """
    try:
        x = ad.read_h5ad(path, backed="r")
        n = x.n_obs
        m = x.n_vars
        x.file.close()

        return n == expected_n, n, m
    except Exception:
        return False, None, None


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list Tabula tissue groups; do not download."
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Download only exact tissue_general names shown by --list."
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Disk audit
    # --------------------------------------------------------

    usage = shutil.disk_usage(PROJECT_ROOT)

    log("=" * 72)
    log("Tabula Sapiens tissue downloader")
    log("=" * 72)

    log(
        f"Disk free: {usage.free / (1024**3):.1f} GB"
    )

    if usage.free < 80 * (1024**3):
        log(
            "WARNING: less than 80 GB free. "
            "Consider freeing disk space before downloading all tissues."
        )

    # --------------------------------------------------------
    # Freeze Census version
    # --------------------------------------------------------

    census_version = resolve_census_version()

    log(f"Census frozen release: {census_version}")
    log(f"Dataset ID: {DATASET_ID}")
    log(f"Grouping field: {GROUP_FIELD}")

    # --------------------------------------------------------
    # Metadata first
    # --------------------------------------------------------

    log("")
    log("Reading Tabula metadata only...")

    with cellxgene_census.open_soma(
        census_version=census_version
    ) as census:

        obs = cellxgene_census.get_obs(
            census,
            organism="Homo sapiens",
            value_filter=f"dataset_id == '{DATASET_ID}'",
            column_names=OBS_COLUMNS,
        )

    if len(obs) == 0:
        raise RuntimeError(
            f"No cells found for dataset {DATASET_ID} "
            f"in Census {census_version}"
        )

    log(f"Dataset cells found: {len(obs):,}")

    # --------------------------------------------------------
    # Tissue summary
    # --------------------------------------------------------

    summary = (
    obs.groupby(
        GROUP_FIELD,
        dropna=False,
        observed=True
    )
    .size()
    .reset_index(name="n_cells")
    .query("n_cells > 0")
    .sort_values("n_cells", ascending=True)
    .reset_index(drop=True)
    )

    summary["status"] = "NOT_DOWNLOADED"
    summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print(summary.to_string(index=False))
    print()

    if args.list:
        log("LIST mode complete. No expression data downloaded.")
        return

    # --------------------------------------------------------
    # Optional selection
    # --------------------------------------------------------

    groups = summary[GROUP_FIELD].astype(str).tolist()

    if args.only is not None:
        requested = set(args.only)

        missing = requested - set(groups)

        if missing:
            print("\nThese requested groups were not found:")
            for x in sorted(missing):
                print("  ", repr(x))

            raise SystemExit(
                "Use --list first and copy the exact tissue name."
            )

        groups = [
            x for x in groups
            if x in requested
        ]

    # --------------------------------------------------------
    # Download tissue by tissue
    # --------------------------------------------------------

    results = []

    with cellxgene_census.open_soma(
        census_version=census_version
    ) as census:

        for idx, group in enumerate(groups, 1):

            expected_n = int(
                summary.loc[
                    summary[GROUP_FIELD].astype(str) == group,
                    "n_cells"
                ].iloc[0]
            )

            stem = safe_filename(group)

            final_path = OUT_DIR / f"TabulaSapiens__{stem}.h5ad"
            tmp_path = OUT_DIR / f"TabulaSapiens__{stem}.partial.h5ad"
            done_path = OUT_DIR / f"TabulaSapiens__{stem}.done.json"

            log("")
            log("=" * 72)
            log(
                f"[{idx}/{len(groups)}] {group} "
                f"({expected_n:,} cells)"
            )
            log("=" * 72)

            # ------------------------------------------------
            # Resume: skip completed tissue
            # ------------------------------------------------

            if final_path.exists():

                ok, n_obs, n_vars = verify_existing(
                    final_path,
                    expected_n
                )

                if ok:
                    log(
                        f"SKIP: already complete "
                        f"({n_obs:,} cells × {n_vars:,} genes)"
                    )

                    results.append({
                        GROUP_FIELD: group,
                        "n_cells": n_obs,
                        "n_genes": n_vars,
                        "status": "COMPLETE_EXISTING",
                        "file": str(final_path),
                    })

                    continue

                else:
                    log(
                        "Existing H5AD failed verification. "
                        "Renaming as suspicious."
                    )

                    bad = final_path.with_suffix(".invalid.h5ad")
                    final_path.replace(bad)

            # Remove only unfinished current-tissue temp file
            if tmp_path.exists():
                log(
                    "Found interrupted partial file; "
                    "removing partial tissue download."
                )
                tmp_path.unlink()

            # ------------------------------------------------
            # Query exact dataset + tissue group
            # ------------------------------------------------

            value_filter = (
                f"dataset_id == '{DATASET_ID}' "
                f"and {GROUP_FIELD} == {filter_literal(group)}"
            )

            try:

                log("Downloading raw count matrix from Census...")

                adata = cellxgene_census.get_anndata(
                    census=census,
                    organism="Homo sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter=value_filter,
                    obs_column_names=OBS_COLUMNS,
                    var_column_names=VAR_COLUMNS,
                )

                log(
                    f"Received: "
                    f"{adata.n_obs:,} cells × "
                    f"{adata.n_vars:,} genes"
                )

                if adata.n_obs != expected_n:
                    raise RuntimeError(
                        f"Cell-count mismatch: "
                        f"expected {expected_n}, "
                        f"received {adata.n_obs}"
                    )

                # Keep sparse CSR for downstream KATv2
                if sp.issparse(adata.X):
                    adata.X = adata.X.tocsr()

                # ------------------------------------------------
                # Raw-count integrity
                # ------------------------------------------------

                raw_ok, integer_error = raw_count_check(adata.X)

                if not raw_ok:
                    raise RuntimeError(
                        "X_name='raw' failed integer-count audit. "
                        f"max fractional error={integer_error}"
                    )

                log(
                    "Raw-count audit: PASS "
                    f"(max fractional error={integer_error:.3g})"
                )

                # Stable Census-specific unique cell identifier
                adata.obs["census_cell_id"] = (
                    adata.obs.index.astype(str)
                )

                # Record provenance
                adata.uns["KATv2_Tabula_download"] = {
                    "dataset_id": DATASET_ID,
                    "census_version": census_version,
                    "group_field": GROUP_FIELD,
                    "group_value": group,
                    "matrix": "X['raw'] from CELLxGENE Census",
                }

                # ------------------------------------------------
                # Write to temporary file first
                # ------------------------------------------------

                log("Writing H5AD...")

                adata.write_h5ad(
                    tmp_path,
                    compression="gzip",
                )

                # ------------------------------------------------
                # Verify temporary output before rename
                # ------------------------------------------------

                chk = ad.read_h5ad(tmp_path, backed="r")

                written_n_obs = chk.n_obs
                written_n_vars = chk.n_vars

                chk.file.close()

                if written_n_obs != adata.n_obs:
                    raise RuntimeError(
                        "Written H5AD cell count mismatch."
                    )

                # Atomic-ish same-directory rename
                tmp_path.replace(final_path)

                metadata = {
                    "status": "COMPLETE",
                    "dataset_id": DATASET_ID,
                    "census_version": census_version,
                    GROUP_FIELD: group,
                    "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars),
                    "raw_count_audit": True,
                    "file": str(final_path),
                }

                done_path.write_text(
                    json.dumps(
                        metadata,
                        indent=2,
                        ensure_ascii=False
                    ),
                    encoding="utf-8",
                )

                log(
                    f"DONE: {final_path.name}"
                )

                results.append(metadata)

                del adata
                gc.collect()

            except KeyboardInterrupt:

                log(
                    "Interrupted by user. "
                    "Completed tissues are preserved."
                )

                raise

            except Exception as e:

                log(f"FAILED: {group}")
                log(str(e))
                log(traceback.format_exc())

                results.append({
                    GROUP_FIELD: group,
                    "n_cells": expected_n,
                    "status": "FAILED",
                    "error": str(e),
                })

                gc.collect()

                # Continue with the next tissue rather than losing
                # the entire overnight download.
                continue

    # --------------------------------------------------------
    # Final manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(results)

    manifest.to_csv(
        AUDIT_DIR / "DOWNLOAD_MANIFEST.csv",
        index=False,
    )

    log("")
    log("=" * 72)
    log("DOWNLOAD PASS COMPLETE")
    log("=" * 72)

    if len(manifest):
        print(
            manifest[
                [
                    GROUP_FIELD,
                    "n_cells",
                    "status",
                ]
            ].to_string(index=False)
        )

    log(
        f"Files saved under:\n{OUT_DIR}"
    )


if __name__ == "__main__":
    main()