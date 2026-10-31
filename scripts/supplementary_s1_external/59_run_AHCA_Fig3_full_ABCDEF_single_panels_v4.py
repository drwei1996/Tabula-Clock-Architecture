#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
59_run_AHCA_Fig3_full_ABCDEF_single_panels_v3.py

AHCA-only external validation of the CURRENT Tabula Figure 3, with one panel
per output file.

Scientific structure
--------------------
A-C  : supportive/context replication
       A  Clock remodeling versus broader tissue responsiveness
       B  Representative Reactome programs with tissue effects beyond matched background
       C  Hallmark programs with tissue effects beyond matched gene background

D-F  : confirmatory replication of the FROZEN Tabula pathway architecture
       D  Tabula-defined pathways in AHCA: correspondence beta vs matched-gene specificity
       E  Biological-module composition of AHCA-confirmed frozen families
       F  Representative frozen coherent pathway families organized by final biological module

Important guardrail
-------------------
AHCA is a SINGLE-DONOR atlas. Therefore D-F do NOT recreate the original
multi-donor "formal hit" inference used in Tabula. Instead they are a
pre-specified single-donor confirmatory sensitivity:

  - only the 98 frozen Tabula strict pathways are tested;
  - the original pathway score definition is retained:
        global gene z-score -> pathway mean, Expanded17 excluded;
  - within each major lineage, tissue-to-tissue pathway differences are related
    to Expanded17 clock-architecture distances;
  - a within-donor tissue-label permutation provides a directional
    correspondence P value;
  - a 4D expression/tissue-structure matched gene-set null provides pathway
    specificity;
  - BH correction is performed within lineage x collection over the frozen
    pathway family;
  - "confirmatory_supported" requires positive beta, correspondence FDR<0.05,
    specificity FDR<0.05 and specificity z>0.

These are external single-donor confirmatory results, not independent
multi-donor formal hits. No rhythmicity, phase, amplitude, oscillation,
causality, or temporal regulation is inferred.

For A-C, a lineage whose cell-type-adjusted tissue model has <=2 residual degrees of freedom is explicitly marked non-estimable and left NA rather than forcing a different model.


Additional v4 QC
----------------
- Untestable pathways retain NaN P/FDR values and are excluded from denominators.
- All-NaN pathway-score columns are handled without RuntimeWarning.
- Permutation calculations are restricted to pathways with finite observed beta
  and finite non-zero pairwise-distance denominator.
- Final n_tested values therefore mean statistically testable pathways, not
  merely the 98 pre-specified frozen targets.

The script caches all expensive intermediate tables. Re-running without
--force will reuse them and only redraw figures.

Required local inputs
---------------------
1) AHCA all-gene pseudobulk export from script 57A:
   AHCA_allgene_pseudobulk_matrix.npz
   AHCA_allgene_pseudobulk_metadata.csv
   AHCA_allgene_genes.txt

2) Existing AHCA standardized Expanded17 pseudobulk:
   external validation/.../02_AHCA_15ORG/source_data/pseudobulk_standardized.csv

3) Tabula frozen V3 ontology:
   01_V3_PATHWAY_ONTOLOGY.csv
   (98 unique frozen pathways, 69 complete-linkage families)

4) Hallmark / Reactome GMTs already used by the Tabula analysis.

If the all-gene export is missing, pass --prepare-allgene to invoke the sibling
57A_prepare_AHCA_allgene_pseudobulk.py automatically.

Example
-------
$py = "<PYTHON>"

& $py ".\\corrected_full_analysis_v2\\scripts\\59_run_AHCA_Fig3_full_ABCDEF_single_panels_v4.py" `
  --project "<TABULA_PROJECT_ROOT>" `
  --external-root ".\\corrected_full_analysis_v2\\external validation\\clock_external_validation_ALL_EXCEPT_H5AD_20260921\\clock_external_validation" `
  --outdir ".\\AHCA_Fig3_full_replication_20260923" `
  --prepare-allgene `
  --permutations 5000 `
  --specificity-draws 5000
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# =============================================================================
# CONSTANTS
# =============================================================================

SEED_DEFAULT = 20260923
EPS = 1e-12

LINEAGES = [
    "b_cell",
    "t_cell",
    "macrophage_monocyte",
    "fibroblast",
    "endothelial",
]

LINEAGE_LABEL = {
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

CLOCK17 = {
    "ARNTL", "CLOCK", "NPAS2",
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "NR1D1", "NR1D2",
    "RORA", "RORB", "RORC",
    "DBP", "TEF", "HLF", "NFIL3",
}

CLOCK17_ORDER = [
    "ARNTL", "CLOCK", "NPAS2",
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "NR1D1", "NR1D2",
    "RORA", "RORB", "RORC",
    "DBP", "TEF", "HLF", "NFIL3",
]

MODULE_PALETTE = [
    "#4E79A7",  # growth
    "#2A9D8F",  # cytokine
    "#E9967A",  # trafficking/ER
    "#76B7B2",  # Rho/cytoskeleton
    "#E3BA22",  # metabolic
    "#86BCB6",  # other specialized
    "#8A89A6",  # transcription
    "#EDC948",  # cell cycle
    "#E76F51",  # death
    "#AF7AA1",  # TGF
    "#8CD17D",  # antigen/interferon
    "#9D9D9D",  # other
]

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
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
# GENERIC HELPERS
# =============================================================================

def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str = "") -> None:
    print(msg, flush=True)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    mkdir(outdir)
    for ext in ["png", "pdf", "svg"]:
        kw = {
            "bbox_inches": "tight",
            "pad_inches": 0.12,
            "facecolor": "white",
        }
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(outdir / f"{stem}.{ext}", **kw)
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bh(p) -> np.ndarray:
    """
    Benjamini-Hochberg adjustment while preserving NaN for hypotheses that
    were not statistically testable. This is important for the AHCA external
    analysis because incomplete gene coverage can make a subset of pathways
    non-testable; those pathways must not silently become FDR=1 "tested"
    hypotheses.
    """
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = np.clip(p[ok], 0, 1)
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    q[ok] = tmp
    return q


def quantile_bins(v, n) -> np.ndarray:
    r = pd.Series(np.asarray(v, float)).rank(method="average", pct=True).to_numpy()
    return np.clip(np.floor(r * n).astype(np.int16), 0, n - 1)


def stable_seed(seed: int, text: str) -> int:
    h = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


def short_pathway(x: str) -> str:
    return (
        str(x)
        .replace("HALLMARK_", "")
        .replace("REACTOME_", "")
        .replace("_", " ")
    )


def wrap_label(s: str, width: int = 42) -> str:
    words = str(s).split()
    lines, cur = [], []
    n = 0
    for w in words:
        extra = len(w) + (1 if cur else 0)
        if cur and n + extra > width:
            lines.append(" ".join(cur))
            cur = [w]
            n = len(w)
        else:
            cur.append(w)
            n += extra
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


def parse_gmt(path: Path, collection: str) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            z = line.rstrip("\n").split("\t")
            if len(z) >= 3:
                out.append({
                    "collection": collection,
                    "pathway": z[0],
                    "description": z[1],
                    "genes": tuple(dict.fromkeys(g for g in z[2:] if g)),
                })
    return out


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def rglob_one(root: Path, name: str, prefer_terms: tuple[str, ...] = ()) -> Path | None:
    hits = [p for p in root.rglob(name) if p.is_file()]
    if not hits:
        return None
    hits.sort(
        key=lambda p: (
            -sum(t.lower() in str(p).lower() for t in prefer_terms),
            len(p.parts),
            len(str(p)),
            str(p),
        )
    )
    return hits[0]


# =============================================================================
# INPUT DISCOVERY
# =============================================================================

def discover_dependencies(project: Path) -> tuple[Path, Path]:
    dep = project / "results" / "phaseA_clock_architecture" / "dependencies"
    hallmark = dep / "h.all.v2026.1.Hs.symbols.gmt"
    reactome = dep / "c2.cp.reactome.v2026.1.Hs.symbols.gmt"
    if not hallmark.exists():
        hallmark = rglob_one(project, "h.all.v2026.1.Hs.symbols.gmt")
    if not reactome.exists():
        reactome = rglob_one(project, "c2.cp.reactome.v2026.1.Hs.symbols.gmt")
    if hallmark is None or reactome is None:
        raise FileNotFoundError("Could not locate Hallmark/Reactome GMT files.")
    return hallmark, reactome


def discover_ontology(project: Path) -> Path:
    candidates = [
        project / "corrected_full_analysis_v2" / "00_freeze"
        / "A4_8_V3_FINAL_ONTOLOGY_20260920" / "01_V3_PATHWAY_ONTOLOGY.csv",
        project / "corrected_full_analysis_v2"
        / "10_A4_8_V3_FINAL_ONTOLOGY" / "01_V3_PATHWAY_ONTOLOGY.csv",
    ]
    p = first_existing(candidates)
    if p is None:
        p = rglob_one(
            project / "corrected_full_analysis_v2",
            "01_V3_PATHWAY_ONTOLOGY.csv",
            ("FINAL", "V3", "FREEZE"),
        )
    if p is None:
        raise FileNotFoundError("Could not locate frozen 01_V3_PATHWAY_ONTOLOGY.csv")
    return p


def discover_ahca_clock_pb(external_root: Path) -> Path:
    p = external_root / "02_AHCA_15ORG" / "source_data" / "pseudobulk_standardized.csv"
    if not p.exists():
        p2 = rglob_one(external_root, "pseudobulk_standardized.csv", ("AHCA",))
        if p2 is None:
            raise FileNotFoundError("Could not locate AHCA pseudobulk_standardized.csv")
        return p2
    return p


def find_allgene_dir(project: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.resolve()
        if (p / "AHCA_allgene_pseudobulk_matrix.npz").exists():
            return p
        return None

    preferred = [
        project / "HCL_AHCA_Fig123_replication_20260923" / "AHCA" / "allgene",
        project / "AHCA_Fig3_full_replication_20260923" / "allgene",
    ]
    for p in preferred:
        if (p / "AHCA_allgene_pseudobulk_matrix.npz").exists():
            return p

    hits = list(project.rglob("AHCA_allgene_pseudobulk_matrix.npz"))
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.parts), len(str(p))))
    return hits[0].parent


def prepare_allgene_if_requested(
    project: Path,
    external_root: Path,
    allgene_dir: Path,
    requested: bool,
) -> None:
    if (allgene_dir / "AHCA_allgene_pseudobulk_matrix.npz").exists():
        return
    if not requested:
        raise FileNotFoundError(
            "AHCA all-gene pseudobulk is missing. Run script 57A first or rerun "
            "this script with --prepare-allgene."
        )

    script57 = (
        project / "corrected_full_analysis_v2" / "scripts"
        / "57A_prepare_AHCA_allgene_pseudobulk.py"
    )
    if not script57.exists():
        raise FileNotFoundError(
            f"--prepare-allgene requested but sibling script 57A is missing: {script57}"
        )

    mkdir(allgene_dir)
    cmd = [
        sys.executable,
        str(script57),
        "--project", str(project),
        "--root", str(external_root),
        "--outdir", str(allgene_dir),
    ]
    log("[PREP] Running AHCA all-gene export via script 57A...")
    subprocess.check_call(cmd)


# =============================================================================
# LOAD AHCA DATA
# =============================================================================

def load_allgene(
    allgene_dir: Path,
    audit_dir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the AHCA all-gene pseudobulk export.

    Important AHCA-specific handling
    --------------------------------
    The 57A exporter reused the original V4 per-tissue aggregation. Different
    GSE159929 tissue count files do not necessarily contain an identical gene
    universe. When the per-tissue pseudobulk tables are concatenated, genes
    absent from a tissue remain NaN in pseudobulks from that tissue.

    For full-transcriptome Fig3 analyses we do NOT convert these missing values
    to biological zero and we do NOT median-impute them, because either choice
    could create artificial tissue effects. Instead we use a conservative
    complete-measurement universe: retain only genes with finite values in every
    eligible AHCA pseudobulk. This is equivalent to restricting the analysis to
    genes represented across all retained tissue/cell-type pseudobulks.
    """
    pX = allgene_dir / "AHCA_allgene_pseudobulk_matrix.npz"
    pm = allgene_dir / "AHCA_allgene_pseudobulk_metadata.csv"
    pg = allgene_dir / "AHCA_allgene_genes.txt"

    for p in [pX, pm, pg]:
        if not p.exists():
            raise FileNotFoundError(p)

    d = np.load(pX, allow_pickle=False)
    if "X" not in d:
        raise RuntimeError(f"{pX} has no key 'X'")
    X = np.asarray(d["X"], dtype=np.float32)
    meta = pd.read_csv(pm, low_memory=False)
    genes = np.array(
        [x.strip() for x in pg.read_text(encoding="utf-8").splitlines() if x.strip()],
        dtype=str,
    )

    if X.shape != (len(meta), len(genes)):
        raise RuntimeError(
            f"AHCA all-gene shape mismatch: X={X.shape}, meta={len(meta)}, genes={len(genes)}"
        )

    req = {
        "dataset", "group_id", "donor_id", "tissue",
        "cell_type", "major_lineage", "n_cells",
    }
    miss = req - set(meta.columns)
    if miss:
        raise RuntimeError(f"AHCA all-gene metadata missing columns: {sorted(miss)}")

    for c in ["group_id", "donor_id", "tissue", "cell_type", "major_lineage"]:
        meta[c] = meta[c].astype(str)
    meta["n_cells"] = pd.to_numeric(meta["n_cells"], errors="coerce")

    finite = np.isfinite(X)
    n_bad = int((~finite).sum())
    gene_finite_fraction = finite.mean(axis=0)
    row_finite_fraction = finite.mean(axis=1)
    keep_gene = gene_finite_fraction >= (1.0 - 1e-12)

    if audit_dir is not None:
        mkdir(audit_dir)
        pd.DataFrame({
            "gene": genes,
            "finite_fraction": gene_finite_fraction,
            "n_nonfinite_pseudobulks": (~finite).sum(axis=0).astype(int),
            "kept_complete_measurement_universe": keep_gene,
        }).to_csv(
            audit_dir / "AHCA_allgene_nonfinite_gene_coverage.csv",
            index=False,
            encoding="utf-8-sig",
        )

        q = meta[[
            "group_id", "tissue", "cell_type", "major_lineage", "n_cells"
        ]].copy()
        q["finite_fraction_across_exported_genes"] = row_finite_fraction
        q["n_nonfinite_genes"] = (~finite).sum(axis=1).astype(int)
        q.to_csv(
            audit_dir / "AHCA_allgene_nonfinite_pseudobulk_coverage.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if n_bad:
        n_bad_genes = int((~keep_gene).sum())
        n_keep = int(keep_gene.sum())
        min_cov = float(np.min(gene_finite_fraction))
        med_cov = float(np.median(gene_finite_fraction))
        log(
            f"[ALLGENE QC] non-finite entries detected: {n_bad:,}; "
            f"affected genes={n_bad_genes:,}/{len(genes):,}."
        )
        log(
            f"[ALLGENE QC] gene finite coverage: median={med_cov:.4f}, "
            f"minimum={min_cov:.4f}."
        )
        log(
            f"[ALLGENE QC] conservative fix: retaining {n_keep:,}/{len(genes):,} "
            "genes measured in every eligible AHCA pseudobulk; no zero-filling "
            "and no imputation."
        )

        if n_keep < 5000:
            raise RuntimeError(
                "After restricting AHCA to genes finite in every pseudobulk, "
                f"only {n_keep:,} genes remain. This is unexpectedly small; "
                "inspect audit/AHCA_allgene_nonfinite_gene_coverage.csv before continuing."
            )

        X = X[:, keep_gene]
        genes = genes[keep_gene]

    if not np.isfinite(X).all():
        raise RuntimeError(
            "AHCA all-gene matrix still contains non-finite values after "
            "complete-measurement filtering."
        )

    log(
        f"[ALLGENE QC] analysis matrix: pseudobulks={X.shape[0]:,}, "
        f"genes={X.shape[1]:,}, all_finite=True"
    )

    return meta.reset_index(drop=True), X, genes


def load_clock_pb(clock_path: Path, all_meta: pd.DataFrame) -> pd.DataFrame:
    clock = pd.read_csv(clock_path, low_memory=False)
    req = {
        "group_id", "donor_id", "tissue", "cell_type",
        "major_lineage", "n_cells",
    } | {f"expr_{g}" for g in CLOCK17_ORDER}
    miss = req - set(clock.columns)
    if miss:
        raise RuntimeError(f"AHCA clock pseudobulk missing columns: {sorted(miss)}")

    clock["group_id"] = clock["group_id"].astype(str)
    if set(clock["group_id"]) != set(all_meta["group_id"]):
        raise RuntimeError(
            "AHCA all-gene and Expanded17 pseudobulk group membership differ."
        )

    clock = clock.set_index("group_id").loc[all_meta["group_id"]].reset_index()
    return clock


# =============================================================================
# A-C: SINGLE-DONOR ADAPTATION OF SCRIPTS 49/50
# =============================================================================

def basis(X: np.ndarray) -> tuple[np.ndarray, int]:
    X = np.asarray(X, float)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    if len(s) == 0 or s[0] <= 0:
        return np.empty((len(X), 0)), 0
    tol = s[0] * max(X.shape) * np.finfo(float).eps
    r = int((s > tol).sum())
    return u[:, :r], r


def sse_matrix(Y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    y2 = np.sum(Y.astype(float) ** 2, axis=0)
    if Q.shape[1] == 0:
        return y2
    z = Q.T @ Y
    return np.maximum(y2 - np.sum(z.astype(float) ** 2, axis=0), 0)


def support_filter_single_donor(meta: pd.DataFrame, lineage: str) -> pd.DataFrame:
    """
    AHCA adaptation of script49 support.

    Because AHCA contains only one donor, donor-count support cannot be used.
    Instead, retain recurrent original cell types (>=2 tissues) and tissue
    contexts represented by >=2 retained cell types, with one iterative
    re-check. This is composition support for a descriptive single-donor
    cell-type-adjusted tissue effect, not multi-donor inference.
    """
    p = meta[meta["major_lineage"].eq(lineage)].copy()

    for _ in range(3):
        if p.empty:
            break

        ct = (
            p.groupby("cell_type", observed=True)
            .agg(
                n_rows=("cell_type", "size"),
                n_tissues=("tissue", "nunique"),
            )
            .reset_index()
        )
        keep_ct = set(ct.loc[(ct.n_rows >= 2) & (ct.n_tissues >= 2), "cell_type"])
        p2 = p[p.cell_type.isin(keep_ct)].copy()

        ts = (
            p2.groupby("tissue", observed=True)
            .agg(
                n_rows=("tissue", "size"),
                n_celltypes=("cell_type", "nunique"),
            )
            .reset_index()
        )
        keep_ts = set(ts.loc[(ts.n_rows >= 2) & (ts.n_celltypes >= 2), "tissue"])
        p2 = p2[p2.tissue.isin(keep_ts)].copy()

        if set(p2.index) == set(p.index):
            p = p2
            break
        p = p2

    if p.empty or p.tissue.nunique() < 3 or p.cell_type.nunique() < 2:
        raise RuntimeError(
            f"{lineage}: insufficient recurrent cell-type/tissue support after "
            "single-donor composition filtering."
        )

    return p.sort_values(["tissue", "cell_type"]).copy()


def design_single_donor(p: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(p)
    cols = [np.ones((n, 1), float)]

    ct = pd.get_dummies(p["cell_type"], drop_first=True, dtype=float).to_numpy()
    if ct.shape[1]:
        cols.append(ct)

    z = np.log1p(p["n_cells"].to_numpy(float))
    z = (z - np.nanmean(z))[:, None]
    cols.append(z)

    red = np.column_stack(cols)

    tis = pd.get_dummies(p["tissue"], drop_first=True, dtype=float).to_numpy()
    full = np.column_stack(cols + ([tis] if tis.shape[1] else []))
    return red, full


def continuous_gene_tissue_effect(
    Y: np.ndarray,
    genes: np.ndarray,
    p: pd.DataFrame,
    min_prevalence: float,
) -> tuple[pd.DataFrame, dict]:
    Xr, Xf = design_single_donor(p)
    Qr, rr = basis(Xr)
    Qf, rf = basis(Xf)

    n = len(p)
    df2 = n - rf
    if rf <= rr:
        raise RuntimeError("Tissue adds no estimable design rank after cell-type adjustment.")
    if df2 <= 2:
        raise RuntimeError(
            f"Insufficient residual degrees of freedom after cell-type+tissue adjustment: {df2}"
        )

    sr = sse_matrix(Y, Qr)
    sf = sse_matrix(Y, Qf)
    mu = Y.astype(float).mean(0)
    st = np.sum((Y.astype(float) - mu[None, :]) ** 2, axis=0)
    prev = (Y > 0).mean(0)
    sd = Y.std(0, ddof=1)

    test = (
        np.isfinite(sd)
        & (sd > EPS)
        & (prev >= min_prevalence)
        & np.isfinite(st)
        & (st > EPS)
        & np.isfinite(sr)
        & (sr > EPS)
    )

    ar = np.full(len(genes), np.nan)
    af = np.full(len(genes), np.nan)
    ar[test] = 1 - (sr[test] / (n - rr)) / (st[test] / (n - 1))
    af[test] = 1 - (sf[test] / (n - rf)) / (st[test] / (n - 1))
    delta = af - ar

    out = pd.DataFrame({
        "gene": genes,
        "testable": test,
        "prevalence": prev,
        "adjR2_reduced": ar,
        "adjR2_full": af,
        "delta_adjR2_tissue": delta,
    })

    info = {
        "n_rows": n,
        "n_tissues": int(p.tissue.nunique()),
        "n_cell_types": int(p.cell_type.nunique()),
        "rank_reduced": rr,
        "rank_full": rf,
        "df_residual": int(df2),
        "n_testable": int(test.sum()),
    }
    return out, info


def global_gene_characteristics(
    X: np.ndarray,
    genes: np.ndarray,
    nbins: int,
) -> pd.DataFrame:
    mean = X.mean(0).astype(float)
    sd = X.std(0, ddof=1).astype(float)
    prev = (X > 0).mean(0).astype(float)
    variable = np.isfinite(mean) & np.isfinite(sd) & (sd > EPS) & np.isfinite(prev)

    b1 = quantile_bins(mean, nbins)
    b2 = quantile_bins(sd, nbins)
    b3 = quantile_bins(prev, nbins)
    code = ((b1.astype(int) * nbins + b2.astype(int)) * nbins + b3.astype(int))

    return pd.DataFrame({
        "gene": genes.astype(str),
        "global_mean": mean,
        "global_sd": sd,
        "global_prevalence": prev,
        "match_code": code,
        "global_variable": variable,
        "is_clock": [g in CLOCK17 for g in genes],
    })


def lineage_competitive_table(effect: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    x = effect.copy()
    x["gene"] = x["gene"].astype(str)
    x["testable"] = x["testable"].astype(bool)
    x["delta_adjR2_tissue"] = pd.to_numeric(x["delta_adjR2_tissue"], errors="coerce")

    x = x.merge(G, on="gene", how="left", validate="one_to_one").reset_index(drop=True)
    x["universe"] = (
        x.testable
        & x.global_variable.fillna(False)
        & ~x.is_clock.fillna(False)
        & np.isfinite(x.delta_adjR2_tissue)
    )

    idx = x.universe.to_numpy(bool)
    vals = x.loc[idx, "delta_adjR2_tissue"].to_numpy(float)
    if len(vals) < 50:
        raise RuntimeError("Too few testable non-clock genes for matched-competitive analysis.")

    ranks = pd.Series(vals).rank(method="average").to_numpy(float)
    u = (ranks - 0.5) / len(ranks)

    x["rank_pct"] = np.nan
    x["score"] = np.nan
    x.loc[idx, "rank_pct"] = u
    x.loc[idx, "score"] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    return x


def code_coord(code: int, n: int) -> np.ndarray:
    return np.array([code // (n * n), (code % (n * n)) // n, code % n], int)


def get_pool(
    code: int,
    target_mask: np.ndarray,
    pools: dict,
    coords: np.ndarray,
    needed: int,
    min_extra: int,
) -> tuple[np.ndarray, float]:
    exact = pools.get(int(code), np.array([], int))
    exact = exact[~target_mask[exact]]
    if len(exact) >= needed + min_extra:
        return exact, 0

    target_coord = coords[int(code)]
    dist = np.abs(coords - target_coord[None, :]).sum(1)
    gather = []
    maxd = 0
    for c in np.argsort(dist, kind="stable"):
        p = pools.get(int(c))
        if p is None:
            continue
        p = p[~target_mask[p]]
        if len(p):
            gather.append(p)
            maxd = int(dist[c])
        if sum(len(a) for a in gather) >= needed + min_extra:
            break

    return (
        (np.unique(np.concatenate(gather)), maxd)
        if gather
        else (np.array([], int), np.nan)
    )


def competitive_enrichment(
    tab: pd.DataFrame,
    sets: list[dict],
    collection: str,
    nbins: int,
    min_genes: int,
    min_extra: int,
) -> pd.DataFrame:
    U = tab[tab.universe].copy()
    # index is kept aligned to tab after reset_index in lineage_competitive_table
    gene2idx = {g: int(i) for i, g in zip(U.index, U.gene)}
    pools = {int(c): part.index.to_numpy(int) for c, part in U.groupby("match_code")}
    coords = np.vstack([code_coord(i, nbins) for i in range(nbins**3)])

    scores = tab.score.to_numpy(float)
    codes = tab.match_code.to_numpy()
    rows = []

    for rec in sets:
        tg = [g for g in rec["genes"] if g not in CLOCK17 and g in gene2idx]
        tidx = np.array([gene2idx[g] for g in tg], int)

        if len(tidx) < min_genes:
            rows.append({
                "collection": collection,
                "pathway": rec["pathway"],
                "description": rec["description"],
                "testable_pathway": False,
                "n_genes": len(tidx),
                "observed_mean_score": np.nan,
                "observed_mean_rank_pct": np.nan,
                "null_mean": np.nan,
                "null_sd": np.nan,
                "matched_z": np.nan,
                "P_matched": 1.0,
                "max_match_bin_distance": np.nan,
                "genes_scored": ";".join(tg),
            })
            continue

        mask = np.zeros(len(tab), bool)
        mask[tidx] = True

        uc, cnt = np.unique(codes[tidx], return_counts=True)
        obs = float(np.mean(scores[tidx]))
        obs_pct = float(np.mean(tab.loc[tidx, "rank_pct"].to_numpy(float)))

        sm = 0.0
        sv = 0.0
        maxdist = 0
        valid = True

        for code, n_take in zip(uc, cnt):
            pool, dist = get_pool(
                int(code), mask, pools, coords, int(n_take), min_extra
            )
            vals = scores[pool]
            vals = vals[np.isfinite(vals)]
            N = len(vals)
            n = int(n_take)

            if N < n or N == 0:
                valid = False
                break

            mu = float(vals.mean())
            var = float(vals.var(ddof=0))
            sm += n * mu
            if N > 1:
                sv += n * var * (N - n) / (N - 1)
            if np.isfinite(dist):
                maxdist = max(maxdist, int(dist))

        m = len(tidx)
        if valid:
            null_mean = sm / m
            null_sd = math.sqrt(max(sv / (m * m), 0))
            z = (obs - null_mean) / null_sd if null_sd > EPS else np.nan
            p = float(norm.sf(z)) if np.isfinite(z) else 1.0
        else:
            null_mean = null_sd = z = np.nan
            p = 1.0

        rows.append({
            "collection": collection,
            "pathway": rec["pathway"],
            "description": rec["description"],
            "testable_pathway": valid,
            "n_genes": m,
            "observed_mean_score": obs,
            "observed_mean_rank_pct": obs_pct,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "matched_z": z,
            "P_matched": p,
            "max_match_bin_distance": maxdist if valid else np.nan,
            "genes_scored": ";".join(tg),
        })

    out = pd.DataFrame(rows)
    out["FDR_matched"] = bh(out.P_matched.to_numpy(float))
    out["supported"] = (
        out.testable_pathway
        & out.FDR_matched.lt(0.05)
        & out.matched_z.gt(0)
    )
    return out


def select_hallmark(R: pd.DataFrame, maxn: int) -> list[str]:
    x = R[R.collection.eq("Hallmark")]
    s = (
        x.groupby("pathway")
        .agg(
            n_supported=("supported", "sum"),
            max_z=("matched_z", "max"),
            min_fdr=("FDR_matched", "min"),
        )
        .reset_index()
    )
    h = s[s.n_supported.gt(0)].copy()
    if h.empty:
        h = s.copy()
    return (
        h.sort_values(
            ["n_supported", "max_z", "min_fdr", "pathway"],
            ascending=[False, False, True, True],
        )
        .head(maxn)
        .pathway.tolist()
    )


def select_reactome(
    R: pd.DataFrame,
    lookup: dict[str, tuple],
    maxn: int,
    jmax: float,
) -> list[str]:
    x = R[R.collection.eq("Reactome")]
    s = (
        x.groupby("pathway")
        .agg(
            n_supported=("supported", "sum"),
            max_z=("matched_z", "max"),
            min_fdr=("FDR_matched", "min"),
        )
        .reset_index()
    )
    c = s[s.n_supported.gt(0)].copy()
    if c.empty:
        c = s.copy()
    c = c.sort_values(
        ["n_supported", "max_z", "min_fdr", "pathway"],
        ascending=[False, False, True, True],
    )

    out = []
    for r in c.itertuples():
        if all(jaccard(lookup.get(r.pathway, ()), lookup.get(p, ())) <= jmax for p in out):
            out.append(r.pathway)
        if len(out) >= maxn:
            break
    return out


def compute_clock_remodeling(clock_pb: pd.DataFrame) -> pd.DataFrame:
    feat = [f"expr_{g}" for g in CLOCK17_ORDER]
    rows = []

    for lin in LINEAGES:
        d = clock_pb[clock_pb.major_lineage.astype(str).eq(lin)].copy()
        if d.empty:
            continue
        cent = d.groupby("tissue", observed=True)[feat].mean()
        if len(cent) < 2:
            continue
        X = cent.to_numpy(float)
        dist = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dist.append(float(np.linalg.norm(X[i] - X[j])))
        rows.append({
            "lineage": lin,
            "lineage_label": LINEAGE_LABEL[lin],
            "clock_remodeling": float(np.median(dist)),
            "n_tissues": len(cent),
            "n_pairs": len(dist),
        })

    return pd.DataFrame(rows)


def plot_A_bridge(
    clock_remod: pd.DataFrame,
    effect_summary: pd.DataFrame,
    figdir: Path,
    srcdir: Path,
) -> None:
    d = clock_remod.merge(
        effect_summary[["lineage", "q90_tissue_effect", "n_testable_genes"]],
        on="lineage",
        how="inner",
        validate="one_to_one",
    )
    d.to_csv(srcdir / "AHCA_Fig3A_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.7, 4.0))
    offsets = {
        "b_cell": (0.04, 0.004),
        "t_cell": (0.04, -0.006),
        "macrophage_monocyte": (0.05, 0.004),
        "fibroblast": (0.05, 0.004),
        "endothelial": (0.05, 0.004),
    }

    for r in d.itertuples(index=False):
        ax.scatter(
            r.clock_remodeling,
            r.q90_tissue_effect,
            s=70,
            color=LINEAGE_COLORS[r.lineage],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        dx, dy = offsets[r.lineage]
        ax.text(
            r.clock_remodeling + dx,
            r.q90_tissue_effect + dy,
            r.lineage_label,
            fontsize=8,
            va="center",
        )

    ax.set_xlabel(
        "Clock-architecture tissue remodeling\n"
        "(median within-lineage distance across tissues)"
    )
    ax.set_ylabel(
        "Broader transcriptional tissue responsiveness\n"
        "(90th percentile incremental adjusted R²)"
    )
    ax.set_title(
        "Clock remodeling versus broader tissue responsiveness",
        loc="left",
        pad=7,
    )
    clean_axis(ax)
    save_figure(fig, figdir, "AHCA_Fig3A_clock_vs_broader_tissue_responsiveness")


def plot_matched_heatmap(
    R: pd.DataFrame,
    selected: list[str],
    collection: str,
    figdir: Path,
    srcdir: Path,
    stem: str,
    title: str,
    frost_alpha: float,
    z_limit: float,
) -> None:
    z = R[R.collection.eq(collection) & R.pathway.isin(selected)].copy()
    M = z.pivot_table(index="pathway", columns="lineage", values="matched_z", aggfunc="first")
    Q = z.pivot_table(index="pathway", columns="lineage", values="FDR_matched", aggfunc="first")

    for lin in LINEAGES:
        if lin not in M:
            M[lin] = np.nan
        if lin not in Q:
            Q[lin] = np.nan

    M = M[LINEAGES].reindex(selected)
    Q = Q[LINEAGES].reindex(selected)

    src = z.copy()
    src.to_csv(srcdir / f"{stem}_source_long.csv", index=False)
    M.reset_index().to_csv(srcdir / f"{stem}_source_matrix_z.csv", index=False)
    Q.reset_index().to_csv(srcdir / f"{stem}_source_matrix_FDR.csv", index=False)

    A = M.to_numpy(float)
    if not np.isfinite(A).any():
        log(f"[WARN] {stem}: no finite values; panel skipped.")
        return

    vmax = float(z_limit)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F2F2F2")

    fig, ax = plt.subplots(
        figsize=(6.5, max(3.8, 0.31 * len(selected) + 1.35))
    )
    im = ax.imshow(
        np.ma.masked_invalid(A),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(range(5))
    ax.set_xticklabels(
        [LINEAGE_LABEL[x] for x in LINEAGES],
        rotation=28,
        ha="right",
    )
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(
        [short_pathway(p) for p in selected],
        fontsize=6 if len(selected) <= 20 else 5.5,
    )

    q = Q.to_numpy(float)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            nonsig = (
                (not np.isfinite(A[i, j]))
                or (not np.isfinite(q[i, j]))
                or q[i, j] >= 0.05
                or A[i, j] <= 0
            )
            if nonsig and np.isfinite(A[i, j]):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor="#D9DDE2",
                        linewidth=0.35,
                        alpha=frost_alpha,
                        zorder=2,
                    )
                )

    ax.set_title(title, loc="left", pad=8)
    ax.grid(False)
    ax.tick_params(axis="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("Matched competitive enrichment z", fontsize=8)

    fig.text(
        0.995,
        0.01,
        "Clear cell: FDR < 0.05 and z > 0   |   Frosted: otherwise",
        ha="right",
        fontsize=6.4,
    )
    save_figure(fig, figdir, stem)


# =============================================================================
# D-F: FROZEN TABULA CONFIRMATORY REPLICATION
# =============================================================================

def load_frozen_ontology(path: Path) -> pd.DataFrame:
    o = pd.read_csv(path, low_memory=False)

    req = {
        "collection",
        "pathway",
        "complete_linkage_cluster_id",
        "representative_collection",
        "representative_pathway",
        "v3_primary_module_id",
        "v3_primary_module_label",
    }
    miss = req - set(o.columns)
    if miss:
        raise RuntimeError(f"Frozen ontology missing columns: {sorted(miss)}")

    if len(o[["collection", "pathway"]].drop_duplicates()) != 98:
        raise RuntimeError(
            "Frozen ontology is not the expected 98-pathway final V3 table."
        )
    if o["complete_linkage_cluster_id"].nunique() != 69:
        raise RuntimeError(
            "Frozen ontology is not the expected 69-family final V3 table."
        )

    return o


def module_metadata(ontology: pd.DataFrame) -> tuple[list[str], dict[str, str], dict[str, str]]:
    x = (
        ontology[["v3_primary_module_id", "v3_primary_module_label"]]
        .drop_duplicates()
        .sort_values("v3_primary_module_id")
    )
    module_ids = x["v3_primary_module_id"].astype(str).tolist()
    labels = dict(
        zip(
            x["v3_primary_module_id"].astype(str),
            x["v3_primary_module_label"].astype(str),
        )
    )
    colors = {
        m: MODULE_PALETTE[i % len(MODULE_PALETTE)]
        for i, m in enumerate(module_ids)
    }
    return module_ids, labels, colors


def build_pathway_scores(
    X: np.ndarray,
    genes: np.ndarray,
    hallmark_sets: list[dict],
    reactome_sets: list[dict],
    nbins: int,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    gene_index = {g: i for i, g in enumerate(genes)}
    global_mean = X.mean(axis=0)
    global_sd = X.std(axis=0)
    variable = np.isfinite(global_sd) & (global_sd > EPS)

    raw = hallmark_sets + reactome_sets
    defs_rows = []

    for rec in raw:
        members = set(rec["genes"])
        available = sorted(
            g
            for g in members - CLOCK17
            if g in gene_index and variable[gene_index[g]]
        )
        tested = len(available) >= 5

        defs_rows.append({
            "collection": rec["collection"],
            "pathway": rec["pathway"],
            "n_original": len(members),
            "n_clock_excluded": len(members & CLOCK17),
            "n_scored": len(available),
            "coverage": len(available) / len(members) if members else np.nan,
            "status": "TESTED" if tested else "INSUFFICIENT_COVERAGE",
            "genes_scored": ";".join(available),
        })

    defs = pd.DataFrame(defs_rows)

    clock_mask = np.array([g in CLOCK17 for g in genes], dtype=bool)
    bg_mask = variable & (~clock_mask)
    bg_ids = np.flatnonzero(bg_mask)
    bg_genes = genes[bg_ids].astype(str)

    bg_mean = global_mean[bg_ids].astype(float)
    bg_sd = global_sd[bg_ids].astype(float)
    bg_prevalence = (X[:, bg_ids] > 0).mean(axis=0).astype(np.float32)
    bg_z = ((X[:, bg_ids] - bg_mean[None, :]) / bg_sd[None, :]).astype(np.float32)
    bg_index = {g: i for i, g in enumerate(bg_genes)}

    base_bins = np.column_stack([
        quantile_bins(bg_mean, nbins),
        quantile_bins(bg_sd, nbins),
        quantile_bins(bg_prevalence, nbins),
    ]).astype(np.int16)

    scores = np.zeros((X.shape[0], len(defs)), dtype=np.float32)
    for j, r in enumerate(defs.itertuples()):
        if r.status != "TESTED":
            scores[:, j] = np.nan
            continue
        idx = np.array(
            [bg_index[g] for g in str(r.genes_scored).split(";") if g],
            dtype=int,
        )
        scores[:, j] = bg_z[:, idx].mean(axis=1).astype(np.float32)

    null_background = {
        "genes": bg_genes,
        "gene_index": bg_index,
        "mean": bg_mean,
        "sd": bg_sd,
        "prevalence": bg_prevalence,
        "z": bg_z,
        "base_bins": base_bins,
        "n_bins": int(nbins),
    }
    return defs, scores, null_background


def aggregate_broad_units(
    meta: pd.DataFrame,
    pathway_scores: np.ndarray,
    clock_pb: pd.DataFrame,
    gene_z: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    clock_cols = [f"expr_{g}" for g in CLOCK17_ORDER]
    records, Prows, Crows, Grows = [], [], [], []

    for (lin, donor, tissue), part in meta.groupby(
        ["major_lineage", "donor_id", "tissue"],
        sort=True,
        observed=True,
    ):
        if lin not in LINEAGES:
            continue
        pos = part.index.to_numpy(int)

        records.append({
            "lineage": str(lin),
            "donor_id": str(donor),
            "tissue": str(tissue),
            "n_pseudobulks": len(part),
            "n_cells": int(part["n_cells"].sum()),
        })

        _ps = pathway_scores[pos].astype(float)
        _finite = np.isfinite(_ps)
        _den = _finite.sum(axis=0)
        _num = np.where(_finite, _ps, 0.0).sum(axis=0)
        _mean = np.full(_ps.shape[1], np.nan, dtype=float)
        _ok = _den > 0
        _mean[_ok] = _num[_ok] / _den[_ok]
        Prows.append(_mean)
        Crows.append(
            clock_pb.iloc[pos][clock_cols].to_numpy(float).mean(axis=0)
        )
        Grows.append(gene_z[pos].mean(axis=0).astype(np.float32))

    return (
        pd.DataFrame(records),
        np.vstack(Prows),
        np.vstack(Crows),
        np.vstack(Grows).astype(np.float32),
    )


def weighted_sd_zero_mean_vector(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w * x**2) / np.sum(w)))


def weighted_sd_zero_mean_matrix(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(w[:, None] * X.astype(float)**2, axis=0) / np.sum(w))


def prepare_pairwise_single_donor(
    unit_meta: pd.DataFrame,
    P: np.ndarray,
    C: np.ndarray,
) -> dict:
    m = unit_meta.copy().reset_index().rename(columns={"index": "source_unit_row"})
    if m["donor_id"].nunique() != 1:
        raise RuntimeError("AHCA confirmatory analysis expects exactly one donor.")
    if m["tissue"].nunique() < 3:
        raise RuntimeError("At least 3 tissues are required.")

    if m["tissue"].duplicated().any():
        raise RuntimeError("Duplicate tissue rows remain after broad aggregation.")

    n = len(m)
    pairs = list(itertools.combinations(range(n), 2))
    a = np.array([p[0] for p in pairs], dtype=int)
    b = np.array([p[1] for p in pairs], dtype=int)

    Cdist_mat = np.linalg.norm(
        C[:, None, :] - C[None, :, :],
        axis=2,
    )
    Yraw = Cdist_mat[a, b].astype(float)
    Pdist = np.abs(P[a] - P[b]).astype(np.float32)

    Ywd = Yraw - Yraw.mean()
    _finite = np.isfinite(Pdist)
    _den = _finite.sum(axis=0)
    _num = np.where(_finite, Pdist, 0.0).sum(axis=0)
    _colmean = np.full(Pdist.shape[1], np.nan, dtype=float)
    _ok = _den > 0
    _colmean[_ok] = _num[_ok] / _den[_ok]
    Xwd = Pdist - _colmean[None, :]

    w = np.full(len(pairs), 1.0 / len(pairs), dtype=float)

    ysd = weighted_sd_zero_mean_vector(Ywd, w)
    if not np.isfinite(ysd) or ysd <= EPS:
        raise RuntimeError("Clock pairwise distance has zero variation.")

    xsd = weighted_sd_zero_mean_matrix(Xwd, w)
    usable = np.isfinite(xsd) & (xsd > EPS)

    Yz = Ywd / ysd
    Xz = np.zeros_like(Xwd, dtype=np.float32)
    Xz[:, usable] = (Xwd[:, usable] / xsd[usable][None, :]).astype(np.float32)

    denom = np.sum(w[:, None] * Xz.astype(float)**2, axis=0)
    usable = usable & np.isfinite(denom) & (denom > EPS)

    WX = Xz.astype(float) * w[:, None]
    beta = np.full(Pdist.shape[1], np.nan)
    beta[usable] = (WX[:, usable].T @ Yz) / denom[usable]

    pair_meta = pd.DataFrame({
        "tissue_A": m.loc[a, "tissue"].astype(str).to_numpy(),
        "tissue_B": m.loc[b, "tissue"].astype(str).to_numpy(),
        "unit_row_A": a,
        "unit_row_B": b,
        "weight": w,
        "clock_distance": Yraw,
        "clock_distance_centered": Ywd,
    })

    return {
        "unit_meta": m,
        "pair_meta": pair_meta,
        "Praw": P,
        "Craw": C,
        "Cdist_mat": Cdist_mat,
        "a": a,
        "b": b,
        "Pdist": Pdist,
        "Xwd": Xwd,
        "Xz": Xz,
        "xsd": xsd,
        "Yraw": Yraw,
        "Ywd": Ywd,
        "Yz": Yz,
        "ysd": ysd,
        "w": w,
        "denom": denom,
        "usable": usable,
        "WX": WX,
        "beta": beta,
    }


def permutation_pvalues_single_donor(
    prepared: dict,
    pathway_idx: np.ndarray,
    n_perm: int,
    seed: int,
    batch_size: int = 200,
) -> np.ndarray:
    obs = prepared["beta"][pathway_idx]
    denom = prepared["denom"][pathway_idx]
    valid_path = (
        np.isfinite(obs)
        & np.isfinite(denom)
        & (denom > EPS)
    )

    p = np.full(len(pathway_idx), np.nan, dtype=float)
    if not valid_path.any():
        return p

    valid_idx = np.flatnonzero(valid_path)
    counts = np.zeros(len(valid_idx), dtype=np.int64)

    Cmat = prepared["Cdist_mat"]
    a = prepared["a"]
    b = prepared["b"]
    w = prepared["w"]
    WX = prepared["WX"][:, pathway_idx][:, valid_idx]
    denom_v = denom[valid_idx]
    obs_v = obs[valid_idx]
    n_tissue = Cmat.shape[0]

    rng = np.random.default_rng(seed)
    done = 0

    while done < n_perm:
        B = min(batch_size, n_perm - done)
        perms = np.vstack([rng.permutation(n_tissue) for _ in range(B)])

        # B x n_pairs
        Y = Cmat[perms[:, a], perms[:, b]].astype(float)
        Y = Y - Y.mean(axis=1, keepdims=True)

        sd = np.sqrt(
            np.sum(w[None, :] * Y**2, axis=1) / np.sum(w)
        )
        good = np.isfinite(sd) & (sd > EPS)

        if good.any():
            Yz = Y[good] / sd[good, None]
            bp = (Yz @ WX) / denom_v[None, :]
            counts += np.sum(
                bp >= obs_v[None, :] - 1e-15,
                axis=0,
            ).astype(np.int64)

        done += B

    p[valid_idx] = (1.0 + counts) / (n_perm + 1.0)
    return p


def build_unit_null_matcher(
    unit_meta: pd.DataFrame,
    unit_gene_z: np.ndarray,
    null_background: dict,
) -> dict:
    G = unit_gene_z.astype(np.float32, copy=False)
    n_gene = G.shape[1]

    # one donor: equal-donor tissue structure reduces to within-donor RMS
    Xc = G.astype(float) - G.astype(float).mean(axis=0, keepdims=True)
    tissue_contrast = np.sqrt(np.mean(Xc**2, axis=0))

    n_bins = int(null_background["n_bins"])
    contrast_bin = quantile_bins(tissue_contrast, n_bins)

    bins4 = np.column_stack([
        null_background["base_bins"],
        contrast_bin,
    ]).astype(np.int16)

    codes = (
        (((bins4[:, 0] * n_bins + bins4[:, 1]) * n_bins + bins4[:, 2])
         * n_bins + bins4[:, 3])
    ).astype(np.int32)

    n_codes = n_bins**4
    coords = np.zeros((n_codes, 4), dtype=np.int16)

    for code in range(n_codes):
        q = code
        for k in range(3, -1, -1):
            coords[code, k] = q % n_bins
            q //= n_bins

    pools = {
        int(code): np.flatnonzero(codes == code)
        for code in np.unique(codes)
    }

    return {
        "G": G,
        "tissue_contrast": tissue_contrast,
        "bins4": bins4,
        "codes": codes,
        "coords": coords,
        "pools": pools,
        "n_bins": n_bins,
    }


def prepare_pathway_match_plan(target_idx: np.ndarray, matcher: dict) -> list[dict]:
    target_idx = np.asarray(target_idx, dtype=int)
    target_mask = np.zeros(len(matcher["codes"]), dtype=bool)
    target_mask[target_idx] = True

    target_codes, counts = np.unique(
        matcher["codes"][target_idx],
        return_counts=True,
    )

    exact_available = []
    for code, count in zip(target_codes, counts):
        pool = matcher["pools"].get(int(code), np.array([], dtype=int))
        n_avail = int((~target_mask[pool]).sum())
        exact_available.append((n_avail, int(code), int(count)))
    exact_available.sort(key=lambda z: z[0])

    plans = []
    all_codes = np.arange(len(matcher["coords"]), dtype=int)

    for _, code, count in exact_available:
        coord = matcher["coords"][code]
        distances = np.abs(matcher["coords"] - coord[None, :]).sum(axis=1)
        order = all_codes[np.argsort(distances, kind="stable")]
        groups = []

        for c in order:
            pool = matcher["pools"].get(int(c))
            if pool is None or len(pool) == 0:
                continue
            pool = pool[~target_mask[pool]]
            if len(pool) == 0:
                continue
            groups.append((int(distances[c]), pool))

        plans.append({
            "target_code": code,
            "count": count,
            "groups": groups,
        })

    return plans


def sample_one_matched_set(
    plans: list[dict],
    pathway_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    exact_ok = True
    for plan in plans:
        if (
            len(plan["groups"]) == 0
            or plan["groups"][0][0] != 0
            or len(plan["groups"][0][1]) < int(plan["count"])
        ):
            exact_ok = False
            break

    if exact_ok:
        chunks = []
        for plan in plans:
            need = int(plan["count"])
            pool = plan["groups"][0][1]
            chosen = rng.choice(pool, size=need, replace=False)
            chunks.append(np.atleast_1d(chosen).astype(int))
        out = np.concatenate(chunks)
        return out, 0.0

    selected = set()
    out = []
    total_distance = 0.0

    for plan in plans:
        need = int(plan["count"])
        for distance, pool in plan["groups"]:
            if need <= 0:
                break
            if selected:
                available = np.array(
                    [g for g in pool if int(g) not in selected],
                    dtype=int,
                )
            else:
                available = pool

            if len(available) == 0:
                continue

            take = min(need, len(available))
            chosen = rng.choice(available, size=take, replace=False)
            chosen = np.atleast_1d(chosen).astype(int)
            out.extend(chosen.tolist())
            selected.update(int(g) for g in chosen)
            total_distance += float(distance) * take
            need -= take

        if need > 0:
            raise RuntimeError(
                "Matched-null background exhausted before constructing full control set."
            )

    if len(out) != pathway_size:
        raise RuntimeError("Matched-null gene-set size changed unexpectedly.")

    return np.asarray(out, dtype=int), total_distance / max(1, pathway_size)


def sample_matched_set_batch(
    plans: list[dict],
    pathway_size: int,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    exact_ok = True
    for plan in plans:
        if (
            len(plan["groups"]) == 0
            or plan["groups"][0][0] != 0
            or len(plan["groups"][0][1]) < int(plan["count"])
        ):
            exact_ok = False
            break

    if exact_ok:
        chunks = []
        for plan in plans:
            need = int(plan["count"])
            pool = plan["groups"][0][1]

            if need == 1:
                chosen = rng.choice(pool, size=(n_draws, 1), replace=True)
            else:
                key = rng.random((n_draws, len(pool)))
                take = np.argpartition(key, kth=need - 1, axis=1)[:, :need]
                chosen = pool[take]
            chunks.append(chosen.astype(int))

        sets = np.concatenate(chunks, axis=1)
        return sets, np.zeros(n_draws, dtype=float)

    sets, distances = [], []
    for _ in range(n_draws):
        gs, md = sample_one_matched_set(plans, pathway_size, rng)
        sets.append(gs)
        distances.append(md)
    return np.vstack(sets), np.asarray(distances, dtype=float)


def matched_null_beta_batch(
    prepared: dict,
    G: np.ndarray,
    gene_sets: np.ndarray,
) -> np.ndarray:
    gene_sets = np.asarray(gene_sets, dtype=int)

    # unit rows x draws x pathway_size -> unit rows x draws
    score = G[:, gene_sets].mean(axis=2)

    a = prepared["a"]
    b = prepared["b"]

    # pairs x draws
    X = np.abs(score[a] - score[b]).astype(float)
    X -= X.mean(axis=0, keepdims=True)

    w = prepared["w"]
    sd = np.sqrt(
        np.sum(w[:, None] * X**2, axis=0) / np.sum(w)
    )

    valid = np.isfinite(sd) & (sd > EPS)
    beta = np.full(X.shape[1], np.nan)

    if valid.any():
        Z = X[:, valid] / sd[valid][None, :]
        denom = np.sum(w[:, None] * Z**2, axis=0)
        beta[valid] = np.sum(
            w[:, None] * Z * prepared["Yz"][:, None],
            axis=0,
        ) / denom

    return beta


def pathway_specificity_null(
    prepared: dict,
    matcher: dict,
    target_idx: np.ndarray,
    obs_beta: float,
    draws: int,
    batch: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    target_idx = np.asarray(target_idx, dtype=int)

    plans = prepare_pathway_match_plan(target_idx, matcher)
    null_beta = []
    match_distance = []

    generated = 0
    while generated < draws:
        B = min(batch, draws - generated)
        sets, distances = sample_matched_set_batch(
            plans,
            len(target_idx),
            B,
            rng,
        )
        beta = matched_null_beta_batch(
            prepared,
            matcher["G"],
            sets,
        )
        null_beta.append(beta)
        match_distance.extend(np.asarray(distances, dtype=float).tolist())
        generated += B

    nb = np.concatenate(null_beta)
    nb = nb[np.isfinite(nb)]

    if len(nb) == 0:
        return {
            "specificity_null_valid_draws": 0,
            "specificity_empirical_P": np.nan,
            "specificity_null_beta_mean": np.nan,
            "specificity_null_beta_sd": np.nan,
            "specificity_z_vs_null": np.nan,
            "specificity_match_mean_bin_distance": np.nan,
            "specificity_match_max_bin_distance": np.nan,
        }

    p = (1.0 + np.sum(nb >= obs_beta - 1e-15)) / (len(nb) + 1.0)
    mu = float(np.mean(nb))
    sd = float(np.std(nb))

    return {
        "specificity_null_valid_draws": int(len(nb)),
        "specificity_empirical_P": float(p),
        "specificity_null_beta_mean": mu,
        "specificity_null_beta_sd": sd,
        "specificity_z_vs_null": float((obs_beta - mu) / sd) if sd > EPS else np.nan,
        "specificity_match_mean_bin_distance": float(np.mean(match_distance)),
        "specificity_match_max_bin_distance": float(np.max(match_distance)),
    }


def run_confirmatory_frozen_pathways(
    ontology: pd.DataFrame,
    defs: pd.DataFrame,
    pathway_scores: np.ndarray,
    null_background: dict,
    unit_meta: pd.DataFrame,
    broad_P: np.ndarray,
    broad_C: np.ndarray,
    broad_G: np.ndarray,
    permutations: int,
    specificity_draws: int,
    specificity_batch: int,
    seed: int,
) -> pd.DataFrame:
    defs_key = {
        (str(r.collection), str(r.pathway)): i
        for i, r in enumerate(defs.itertuples())
    }

    frozen_keys = list(
        ontology[["collection", "pathway"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    missing = [k for k in frozen_keys if k not in defs_key]
    if missing:
        raise RuntimeError(
            f"{len(missing)} frozen pathways are missing from current GMT definitions. "
            f"Examples: {missing[:5]}"
        )

    frozen_idx = np.array([defs_key[k] for k in frozen_keys], dtype=int)
    frozen_defs = defs.iloc[frozen_idx].copy().reset_index(drop=True)

    rows = []

    for li, lin in enumerate(LINEAGES, start=1):
        idx = np.flatnonzero(unit_meta["lineage"].astype(str).eq(lin).to_numpy())
        if len(idx) < 3:
            log(f"[DEF] {lin}: <3 tissues, skipped.")
            continue

        um = unit_meta.iloc[idx].reset_index(drop=True)
        P = broad_P[idx][:, frozen_idx]
        C = broad_C[idx]
        G = broad_G[idx]

        # only pathways with usable score coverage
        prep = prepare_pairwise_single_donor(um, P, C)
        obs_beta = prep["beta"]

        log(
            f"[DEF] {li}/5 {lin}: tissues={len(um)}, pairs={len(prep['pair_meta'])}, "
            f"frozen pathways={len(frozen_idx)}"
        )

        perm_p = permutation_pvalues_single_donor(
            prep,
            np.arange(len(frozen_idx), dtype=int),
            permutations,
            stable_seed(seed, f"perm::{lin}"),
        )

        matcher = build_unit_null_matcher(um, G, null_background)
        bg_index = null_background["gene_index"]

        unit_rows = []
        for j, r in frozen_defs.iterrows():
            beta = float(obs_beta[j]) if np.isfinite(obs_beta[j]) else np.nan
            genes_scored = [
                g for g in str(r["genes_scored"]).split(";")
                if g and g in bg_index
            ]

            pathway_testable = (
                str(r.get("status", "")) == "TESTED"
                and int(r["n_scored"]) >= 5
                and np.isfinite(beta)
                and np.isfinite(perm_p[j])
            )

            rec = {
                "lineage": lin,
                "lineage_label": LINEAGE_LABEL[lin],
                "collection": str(r["collection"]),
                "pathway": str(r["pathway"]),
                "n_scored": int(r["n_scored"]),
                "pathway_testable": bool(pathway_testable),
                "beta_standardized_pairwise_correspondence": beta,
                "pairwise_correspondence_one_sided_P": (
                    float(perm_p[j]) if np.isfinite(perm_p[j]) else np.nan
                ),
                "n_tissues": len(um),
                "n_pairs": len(prep["pair_meta"]),
            }

            if (not pathway_testable) or len(genes_scored) < 5:
                rec.update({
                    "specificity_null_valid_draws": 0,
                    "specificity_empirical_P": np.nan,
                    "specificity_null_beta_mean": np.nan,
                    "specificity_null_beta_sd": np.nan,
                    "specificity_z_vs_null": np.nan,
                    "specificity_match_mean_bin_distance": np.nan,
                    "specificity_match_max_bin_distance": np.nan,
                })
            else:
                target_idx = np.array([bg_index[g] for g in genes_scored], dtype=int)
                sp = pathway_specificity_null(
                    prep,
                    matcher,
                    target_idx,
                    beta,
                    specificity_draws,
                    specificity_batch,
                    stable_seed(seed, f"spec::{lin}::{r['collection']}::{r['pathway']}"),
                )
                rec.update(sp)

            unit_rows.append(rec)

            if (j + 1) % 20 == 0 or j + 1 == len(frozen_defs):
                log(
                    f"      specificity {j+1:3d}/{len(frozen_defs)} "
                    f"[{lin}]"
                )

        u = pd.DataFrame(unit_rows)

        # Prespecified frozen family: BH only across the 98 frozen pathways,
        # separately by collection, within this lineage.
        u["pairwise_correspondence_FDR_frozen_collection"] = np.nan
        u["specificity_FDR_frozen_collection"] = np.nan

        for coll, ids in u.groupby("collection").groups.items():
            ids = np.asarray(list(ids), dtype=int)
            u.loc[ids, "pairwise_correspondence_FDR_frozen_collection"] = bh(
                u.loc[ids, "pairwise_correspondence_one_sided_P"].to_numpy(float)
            )
            u.loc[ids, "specificity_FDR_frozen_collection"] = bh(
                u.loc[ids, "specificity_empirical_P"].to_numpy(float)
            )

        u["directionally_concordant"] = (
            u["pathway_testable"].astype(bool)
            & u["beta_standardized_pairwise_correspondence"].gt(0)
            & u["specificity_z_vs_null"].gt(0)
        )

        u["confirmatory_supported"] = (
            u["pathway_testable"].astype(bool)
            & u["directionally_concordant"]
            & u["pairwise_correspondence_FDR_frozen_collection"].lt(0.05)
            & u["specificity_FDR_frozen_collection"].lt(0.05)
        )

        rows.append(u)

    if not rows:
        raise RuntimeError("No AHCA lineage had sufficient tissue support for confirmatory analysis.")

    return pd.concat(rows, ignore_index=True)


def family_display_meta(ontology: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in ontology.groupby("complete_linkage_cluster_id", sort=True):
        rep_coll = str(g["representative_collection"].iloc[0])
        rep_path = str(g["representative_pathway"].iloc[0])

        z = g[
            g["collection"].astype(str).eq(rep_coll)
            & g["pathway"].astype(str).eq(rep_path)
        ]
        if z.empty:
            z = g[g["pathway"].astype(str).eq(rep_path)]
        if z.empty:
            z = g.iloc[[0]]

        r = z.iloc[0]
        rows.append({
            "complete_linkage_cluster_id": str(cid),
            "representative_collection": rep_coll,
            "representative_pathway": rep_path,
            "v3_primary_module_id": str(r["v3_primary_module_id"]),
            "v3_primary_module_label": str(r["v3_primary_module_label"]),
        })

    return pd.DataFrame(rows)


def build_family_replication(
    confirm: pd.DataFrame,
    ontology: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fam = family_display_meta(ontology)

    # One pre-specified representative pathway per frozen family.
    fl = confirm.merge(
        fam,
        left_on=["collection", "pathway"],
        right_on=["representative_collection", "representative_pathway"],
        how="inner",
        validate="many_to_one",
    )

    if fl["complete_linkage_cluster_id"].nunique() > 69:
        raise RuntimeError("Family replication unexpectedly exceeds 69 frozen families.")

    summary = (
        fl.groupby(
            [
                "complete_linkage_cluster_id",
                "representative_collection",
                "representative_pathway",
                "v3_primary_module_id",
                "v3_primary_module_label",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_lineages_present=("lineage", "nunique"),
            n_lineages_tested=("pathway_testable", "sum"),
            n_directional_lineages=("directionally_concordant", "sum"),
            n_confirmatory_lineages=("confirmatory_supported", "sum"),
            median_beta=("beta_standardized_pairwise_correspondence", "median"),
            median_specificity_z=("specificity_z_vs_null", "median"),
            min_correspondence_FDR=("pairwise_correspondence_FDR_frozen_collection", "min"),
            min_specificity_FDR=("specificity_FDR_frozen_collection", "min"),
        )
        .reset_index()
    )

    summary["family_confirmatory_supported"] = summary["n_confirmatory_lineages"].gt(0)
    summary["family_directional_support"] = summary["n_directional_lineages"].gt(0)
    return fl, summary


def plot_D_confirmatory_scatter(
    confirm: pd.DataFrame,
    ontology: pd.DataFrame,
    module_ids: list[str],
    module_labels: dict[str, str],
    module_colors: dict[str, str],
    figdir: Path,
    srcdir: Path,
) -> None:
    o = ontology[
        ["collection", "pathway", "v3_primary_module_id", "v3_primary_module_label"]
    ].drop_duplicates()

    d = confirm.merge(o, on=["collection", "pathway"], how="left", validate="many_to_one")
    d.to_csv(srcdir / "AHCA_Fig3D_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.3, 4.8))

    # Every pre-specified point is shown; unsupported points are muted rather
    # than deleted. This prevents "winner-only" external validation.
    for mid in module_ids:
        sub = d[d.v3_primary_module_id.astype(str).eq(mid)]
        if sub.empty:
            continue

        supported = sub.confirmatory_supported.astype(bool)
        ax.scatter(
            sub.loc[~supported, "beta_standardized_pairwise_correspondence"],
            sub.loc[~supported, "specificity_z_vs_null"],
            s=18,
            color=module_colors[mid],
            alpha=0.20,
            edgecolor="none",
        )
        ax.scatter(
            sub.loc[supported, "beta_standardized_pairwise_correspondence"],
            sub.loc[supported, "specificity_z_vs_null"],
            s=24,
            color=module_colors[mid],
            alpha=0.95,
            edgecolor="white",
            linewidth=0.4,
            label=module_labels[mid],
        )

    ax.axvline(0, color="#B8B8B8", lw=0.8)
    ax.axhline(0, color="#B8B8B8", lw=0.8)
    ax.set_xlabel("Standardized correspondence β")
    ax.set_ylabel("Specificity z vs matched null")
    ax.set_title(
        "Tabula-defined pathways in AHCA: effect size and specificity",
        loc="left",
        pad=7,
    )
    clean_axis(ax)

    # show all module legend entries that occur, independent of support
    handles = []
    for mid in module_ids:
        if mid not in set(d.v3_primary_module_id.astype(str)):
            continue
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markerfacecolor=module_colors[mid],
                markeredgecolor="none",
                markersize=5,
                label=module_labels[mid],
            )
        )
    ax.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6.2,
        title="Frozen Tabula module",
        title_fontsize=6.5,
    )

    fig.text(
        0.995,
        0.01,
        "Opaque: correspondence FDR < 0.05 + specificity FDR < 0.05 + β,z > 0; muted: tested but not confirmed",
        ha="right",
        fontsize=5.8,
    )
    save_figure(fig, figdir, "AHCA_Fig3D_frozen_pathway_confirmatory_scatter")


def plot_E_module_composition(
    family_summary: pd.DataFrame,
    module_ids: list[str],
    module_labels: dict[str, str],
    module_colors: dict[str, str],
    figdir: Path,
    srcdir: Path,
) -> None:
    rows = []
    for mid in module_ids:
        g = family_summary[
            family_summary.v3_primary_module_id.astype(str).eq(mid)
        ]
        if g.empty:
            continue
        rows.append({
            "module_id": mid,
            "module_label": module_labels[mid],
            "n_frozen_families_tested": int(len(g)),
            "n_confirmed_families": int(g.family_confirmatory_supported.sum()),
            "n_directional_families": int(g.family_directional_support.sum()),
            "confirmed_fraction": float(g.family_confirmatory_supported.mean()),
            "median_beta": float(g.median_beta.median()),
        })

    d = pd.DataFrame(rows)
    d.to_csv(srcdir / "AHCA_Fig3E_source.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    y = np.arange(len(d))
    vals = d["n_confirmed_families"].to_numpy(int)

    ax.barh(
        y,
        vals,
        color=[module_colors[m] for m in d.module_id],
        height=0.66,
        edgecolor="none",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(d.module_label)
    ax.invert_yaxis()
    ax.set_xlabel("AHCA-confirmed frozen coherent pathway families")
    ax.set_title("Biological-module composition", loc="left", pad=7)
    clean_axis(ax)

    xmax = max(1, int(vals.max()) if len(vals) else 1)
    for yi, r in enumerate(d.itertuples(index=False)):
        ax.text(
            r.n_confirmed_families + 0.08 * max(1, xmax / 8),
            yi,
            f"{r.n_confirmed_families}/{r.n_frozen_families_tested}",
            va="center",
            fontsize=7,
        )

    ax.set_xlim(0, max(1.5, xmax * 1.25 + 0.5))
    fig.text(
        0.995,
        0.01,
        "Label = confirmed / frozen families tested",
        ha="right",
        fontsize=6.0,
    )
    save_figure(fig, figdir, "AHCA_Fig3E_biological_module_composition")


def plot_F_representative_families(
    family_summary: pd.DataFrame,
    module_ids: list[str],
    module_labels: dict[str, str],
    module_colors: dict[str, str],
    figdir: Path,
    srcdir: Path,
    per_module: int,
    max_total: int,
) -> None:
    # Primary selection is confirmatory support. If a module has no confirmed
    # family, it contributes no row; there is no rescue by directional-only data.
    selected = []

    for mid in module_ids:
        g = family_summary[
            family_summary.v3_primary_module_id.astype(str).eq(mid)
            & family_summary.family_confirmatory_supported.astype(bool)
        ].copy()
        if g.empty:
            continue

        g = g.sort_values(
            ["n_confirmatory_lineages", "median_beta", "median_specificity_z"],
            ascending=[False, False, False],
        )
        selected.append(g.head(per_module))

    if selected:
        d = pd.concat(selected, ignore_index=True)
    else:
        d = family_summary.iloc[0:0].copy()

    if len(d) > max_total:
        d = d.sort_values(
            ["n_confirmatory_lineages", "median_beta", "median_specificity_z"],
            ascending=[False, False, False],
        ).head(max_total)

    d["module_rank"] = d.v3_primary_module_id.astype(str).map(
        {m: i for i, m in enumerate(module_ids)}
    )
    d = d.sort_values(
        ["module_rank", "median_beta"],
        ascending=[True, False],
    ).reset_index(drop=True)

    d.to_csv(srcdir / "AHCA_Fig3F_source.csv", index=False)

    if d.empty:
        fig, ax = plt.subplots(figsize=(6.0, 2.2))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "No frozen coherent family met the prespecified AHCA confirmatory criterion.",
            ha="center", va="center",
        )
        save_figure(fig, figdir, "AHCA_Fig3F_representative_frozen_families")
        return

    fig, ax = plt.subplots(
        figsize=(6.8, max(5.2, 0.42 * len(d) + 1.5))
    )

    y = np.arange(len(d))[::-1]

    # module background bands
    for mid in module_ids:
        idx = np.flatnonzero(d.v3_primary_module_id.astype(str).eq(mid).to_numpy())
        if len(idx) == 0:
            continue
        yy = y[idx]
        ax.axhspan(
            yy.min() - 0.5,
            yy.max() + 0.5,
            color=module_colors[mid],
            alpha=0.10,
            zorder=0,
        )

    size_map = {1: 55, 2: 80, 3: 110, 4: 145, 5: 180}

    for yi, r in zip(y, d.itertuples(index=False)):
        marker = "o" if str(r.representative_collection) == "Hallmark" else "s"
        ax.scatter(
            r.median_beta,
            yi,
            s=size_map.get(int(r.n_confirmatory_lineages), 55),
            marker=marker,
            color=module_colors[str(r.v3_primary_module_id)],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    labels = [
        f"{r.complete_linkage_cluster_id} | {short_pathway(r.representative_pathway)}"
        for r in d.itertuples(index=False)
    ]

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.1)
    ax.set_xlabel("Median standardized pairwise correspondence β")
    ax.set_title(
        "Representative frozen coherent pathway families organized by final biological module",
        loc="left",
        pad=7,
    )
    clean_axis(ax)

    # Collection legend
    coll_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=6,
            label="Hallmark",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=6,
            label="Reactome",
        ),
    ]
    leg1 = ax.legend(
        handles=coll_handles,
        title="Collection",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.32),
        borderaxespad=0,
    )
    ax.add_artist(leg1)

    size_handles = [
        ax.scatter(
            [], [],
            s=size_map[k],
            color="#A7A7A7",
            edgecolor="white",
            linewidth=0.7,
            label=f"{k} lineage" if k == 1 else f"{k} lineages",
        )
        for k in [1, 3, 5]
    ]
    ax.legend(
        handles=size_handles,
        title="Confirmed lineages",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.18, 0.32),
        borderaxespad=0,
    )

    save_figure(fig, figdir, "AHCA_Fig3F_representative_frozen_families")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--external-root", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)

    ap.add_argument("--allgene-dir", type=Path, default=None)
    ap.add_argument("--prepare-allgene", action="store_true")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--match-bins", type=int, default=4)
    ap.add_argument("--min-prevalence", type=float, default=0.20)
    ap.add_argument("--min-pathway-genes", type=int, default=10)
    ap.add_argument("--min-extra-match-genes", type=int, default=20)
    ap.add_argument("--max-hallmark-display", type=int, default=20)
    ap.add_argument("--max-reactome-display", type=int, default=20)
    ap.add_argument("--reactome-display-jaccard", type=float, default=0.50)

    ap.add_argument("--permutations", type=int, default=5000)
    ap.add_argument("--specificity-draws", type=int, default=5000)
    ap.add_argument("--specificity-batch", type=int, default=100)

    ap.add_argument("--frost-alpha", type=float, default=0.40)
    ap.add_argument("--z-limit", type=float, default=6.0)
    ap.add_argument("--families-per-module", type=int, default=2)
    ap.add_argument("--max-family-display", type=int, default=22)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)

    args = ap.parse_args()

    project = args.project.resolve()
    external_root = args.external_root.resolve()
    outdir = mkdir(args.outdir.resolve())
    figdir = mkdir(outdir / "figures")
    tabdir = mkdir(outdir / "tables")
    srcdir = mkdir(outdir / "source_data")
    auddir = mkdir(outdir / "audit")
    abc_gene_dir = mkdir(tabdir / "ABC_gene_effects")

    log("=" * 112)
    log("AHCA FIGURE 3 — FULL A-F EXTERNAL REPLICATION (SINGLE PANELS)")
    log("=" * 112)
    log(f"PROJECT       : {project}")
    log(f"EXTERNAL ROOT : {external_root}")
    log(f"OUT           : {outdir}")
    log(
        f"PERMUTATIONS  : {args.permutations} | "
        f"SPECIFICITY DRAWS: {args.specificity_draws}"
    )

    hallmark_gmt, reactome_gmt = discover_dependencies(project)
    ontology_path = discover_ontology(project)
    clock_path = discover_ahca_clock_pb(external_root)

    allgene_dir = find_allgene_dir(project, args.allgene_dir)
    if allgene_dir is None:
        allgene_dir = (
            args.allgene_dir.resolve()
            if args.allgene_dir is not None
            else project / "HCL_AHCA_Fig123_replication_20260923" / "AHCA" / "allgene"
        )

    prepare_allgene_if_requested(
        project,
        external_root,
        allgene_dir,
        args.prepare_allgene,
    )

    log(f"[SOURCE] all-gene : {allgene_dir}")
    log(f"[SOURCE] clock    : {clock_path}")
    log(f"[SOURCE] ontology : {ontology_path}")
    log(f"[SOURCE] Hallmark : {hallmark_gmt}")
    log(f"[SOURCE] Reactome : {reactome_gmt}")

    meta, X, genes = load_allgene(allgene_dir, auddir)
    clock_pb = load_clock_pb(clock_path, meta)
    ontology = load_frozen_ontology(ontology_path)

    H = parse_gmt(hallmark_gmt, "Hallmark")
    R = parse_gmt(reactome_gmt, "Reactome")
    if len(H) != 50:
        raise RuntimeError(f"Expected 50 Hallmark pathways, found {len(H)}")
    if len(R) != 1839:
        raise RuntimeError(f"Expected 1839 Reactome pathways, found {len(R)}")

    # -------------------------------------------------------------------------
    # A-C: broader tissue-context branch
    # -------------------------------------------------------------------------
    abc_cache = tabdir / "ABC_ALL_LINEAGES_matched_competitive.csv.gz"
    abc_summary_cache = tabdir / "ABC_continuous_gene_tissue_effect_summary.csv"

    if abc_cache.exists() and abc_summary_cache.exists() and not args.force:
        log("[ABC] Reusing cached matched-competitive results.")
        ABC = pd.read_csv(abc_cache, low_memory=False)
        effect_summary = pd.read_csv(abc_summary_cache, low_memory=False)
    else:
        log("[ABC] Computing single-donor cell-type-adjusted continuous tissue effects...")
        G = global_gene_characteristics(X, genes, args.match_bins)

        comp_tabs = {}
        all_comp = []
        effect_rows = []
        abc_support_rows = []

        pos = pd.Series(np.arange(len(meta)), index=meta.index)

        for lin in LINEAGES:
            log("-" * 112)
            log(f"[ABC] {lin}: composition-filtered tissue effect")

            try:
                p = support_filter_single_donor(meta, lin)
            except RuntimeError as e:
                log(f"[WARN] {e}")
                continue

            idx = p.index.to_numpy(int)
            Y = X[idx].astype(np.float32, copy=False)

            # Diagnose estimability BEFORE fitting. In a single-donor atlas,
            # original cell type and tissue can be nearly nested. In that case,
            # a cell-type + tissue model may have essentially no residual df.
            # We do NOT relax the model ad hoc for one lineage because that would
            # mix estimands across lineages. Instead, A-C mark that lineage as
            # non-estimable and continue; D-F use a separate pre-specified
            # tissue-pair confirmatory design and can still include that lineage.
            Xr0, Xf0 = design_single_donor(p)
            _, rr0 = basis(Xr0)
            _, rf0 = basis(Xf0)
            df0 = int(len(p) - rf0)
            log(
                f"[ABC] {lin}: rows={len(p)}, tissues={p.tissue.nunique()}, "
                f"cell_types={p.cell_type.nunique()}, rank_reduced={rr0}, "
                f"rank_full={rf0}, residual_df={df0}"
            )

            try:
                effect, info = continuous_gene_tissue_effect(
                    Y,
                    genes,
                    p,
                    args.min_prevalence,
                )
            except RuntimeError as e:
                reason = str(e)
                log(
                    f"[ABC][SKIP] {lin}: {reason} "
                    "This lineage is left NA in supportive panels A-C; "
                    "no lower-dimensional fallback is substituted."
                )
                abc_support_rows.append({
                    "lineage": lin,
                    "lineage_label": LINEAGE_LABEL[lin],
                    "status": "NOT_ESTIMABLE_CELLTYPE_ADJUSTED_TISSUE_EFFECT",
                    "reason": reason,
                    "n_rows": int(len(p)),
                    "n_tissues": int(p.tissue.nunique()),
                    "n_cell_types": int(p.cell_type.nunique()),
                    "rank_reduced": int(rr0),
                    "rank_full": int(rf0),
                    "df_residual": int(df0),
                })
                continue

            abc_support_rows.append({
                "lineage": lin,
                "lineage_label": LINEAGE_LABEL[lin],
                "status": "ESTIMABLE",
                "reason": "",
                **info,
            })

            effect.insert(0, "lineage", lin)
            effect.to_csv(
                abc_gene_dir / f"{lin}_celltype_adjusted_continuous_tissue_effect.csv.gz",
                index=False,
                compression="gzip",
            )

            tab = lineage_competitive_table(effect.drop(columns=["lineage"]), G)
            comp_tabs[lin] = tab

            vals = tab.loc[tab.universe, "delta_adjR2_tissue"].to_numpy(float)
            effect_rows.append({
                "lineage": lin,
                "lineage_label": LINEAGE_LABEL[lin],
                "median": float(np.nanmedian(vals)),
                "q25": float(np.nanquantile(vals, 0.25)),
                "q75": float(np.nanquantile(vals, 0.75)),
                "q90_tissue_effect": float(np.nanquantile(vals, 0.90)),
                "n_testable_genes": int(len(vals)),
                **info,
            })

            h = competitive_enrichment(
                tab,
                H,
                "Hallmark",
                args.match_bins,
                args.min_pathway_genes,
                args.min_extra_match_genes,
            )
            h.insert(0, "lineage", lin)

            r = competitive_enrichment(
                tab,
                R,
                "Reactome",
                args.match_bins,
                args.min_pathway_genes,
                args.min_extra_match_genes,
            )
            r.insert(0, "lineage", lin)

            all_comp.extend([h, r])

            log(
                f"[ABC] {lin}: Hallmark {int(h.supported.sum())}/{int(h.testable_pathway.sum())}; "
                f"Reactome {int(r.supported.sum())}/{int(r.testable_pathway.sum())}"
            )

        if not all_comp:
            raise RuntimeError("ABC analysis produced no lineage-level pathway results.")

        ABC = pd.concat(all_comp, ignore_index=True)
        ABC.to_csv(abc_cache, index=False, compression="gzip")
        effect_summary = pd.DataFrame(effect_rows)
        effect_summary.to_csv(abc_summary_cache, index=False)
        pd.DataFrame(abc_support_rows).to_csv(
            auddir / "ABC_lineage_estimability_audit.csv",
            index=False,
        )

        skipped_abc = [
            r["lineage"] for r in abc_support_rows
            if r.get("status") != "ESTIMABLE"
        ]
        if skipped_abc:
            log(
                "[ABC] Non-estimable cell-type-adjusted lineages (shown as NA/blank in A-C): "
                + ", ".join(skipped_abc)
            )

    clock_remod = compute_clock_remodeling(clock_pb)
    plot_A_bridge(clock_remod, effect_summary, figdir, srcdir)

    Rlookup = {
        rec["pathway"]: tuple(g for g in rec["genes"] if g not in CLOCK17)
        for rec in R
    }
    selected_reactome = select_reactome(
        ABC,
        Rlookup,
        args.max_reactome_display,
        args.reactome_display_jaccard,
    )
    selected_hallmark = select_hallmark(
        ABC,
        args.max_hallmark_display,
    )

    pd.DataFrame({
        "display_order": range(1, len(selected_reactome) + 1),
        "pathway": selected_reactome,
    }).to_csv(srcdir / "AHCA_Fig3B_selected_Reactome.csv", index=False)

    pd.DataFrame({
        "display_order": range(1, len(selected_hallmark) + 1),
        "pathway": selected_hallmark,
    }).to_csv(srcdir / "AHCA_Fig3C_selected_Hallmark.csv", index=False)

    plot_matched_heatmap(
        ABC,
        selected_reactome,
        "Reactome",
        figdir,
        srcdir,
        "AHCA_Fig3B_Reactome_matched_competitive",
        "Representative Reactome programs with tissue effects beyond matched background",
        args.frost_alpha,
        args.z_limit,
    )

    plot_matched_heatmap(
        ABC,
        selected_hallmark,
        "Hallmark",
        figdir,
        srcdir,
        "AHCA_Fig3C_Hallmark_matched_competitive",
        "Hallmark programs with tissue effects beyond matched gene background",
        args.frost_alpha,
        args.z_limit,
    )

    # -------------------------------------------------------------------------
    # D-F: frozen Tabula confirmatory pathway architecture
    # -------------------------------------------------------------------------
    def_cache = tabdir / "DEF_frozen_pathway_confirmatory_by_lineage.csv.gz"

    if def_cache.exists() and not args.force:
        log("[DEF] Reusing cached frozen-pathway confirmatory results.")
        confirm = pd.read_csv(def_cache, low_memory=False)
    else:
        log("[DEF] Building pathway scores and matched-gene background...")
        defs, pathway_scores, null_background = build_pathway_scores(
            X,
            genes,
            H,
            R,
            args.match_bins,
        )

        # Align clock rows to all-gene metadata row order.
        clock_pb = clock_pb.reset_index(drop=True)

        unit_meta, broad_P, broad_C, broad_G = aggregate_broad_units(
            meta,
            pathway_scores,
            clock_pb,
            null_background["z"],
        )

        confirm = run_confirmatory_frozen_pathways(
            ontology,
            defs,
            pathway_scores,
            null_background,
            unit_meta,
            broad_P,
            broad_C,
            broad_G,
            args.permutations,
            args.specificity_draws,
            args.specificity_batch,
            args.seed,
        )

        confirm.to_csv(def_cache, index=False, compression="gzip")

    fam_lineage, fam_summary = build_family_replication(confirm, ontology)
    fam_lineage.to_csv(tabdir / "DEF_frozen_family_by_lineage.csv", index=False)
    fam_summary.to_csv(tabdir / "DEF_frozen_family_summary.csv", index=False)

    module_ids, module_labels, module_colors = module_metadata(ontology)

    plot_D_confirmatory_scatter(
        confirm,
        ontology,
        module_ids,
        module_labels,
        module_colors,
        figdir,
        srcdir,
    )

    plot_E_module_composition(
        fam_summary,
        module_ids,
        module_labels,
        module_colors,
        figdir,
        srcdir,
    )

    plot_F_representative_families(
        fam_summary,
        module_ids,
        module_labels,
        module_colors,
        figdir,
        srcdir,
        args.families_per_module,
        args.max_family_display,
    )

    # -------------------------------------------------------------------------
    # Final audit / summary
    # -------------------------------------------------------------------------
    _confirm_for_summary = confirm.copy()
    _confirm_for_summary["beta_if_testable"] = _confirm_for_summary[
        "beta_standardized_pairwise_correspondence"
    ].where(_confirm_for_summary["pathway_testable"].astype(bool))
    _confirm_for_summary["specificity_if_testable"] = _confirm_for_summary[
        "specificity_z_vs_null"
    ].where(_confirm_for_summary["pathway_testable"].astype(bool))

    hit_summary = (
        _confirm_for_summary.groupby(["lineage", "collection"])
        .agg(
            n_prespecified=("pathway", "size"),
            n_tested=("pathway_testable", "sum"),
            n_directional=("directionally_concordant", "sum"),
            n_confirmatory=("confirmatory_supported", "sum"),
            median_beta=("beta_if_testable", "median"),
            median_specificity_z=("specificity_if_testable", "median"),
        )
        .reset_index()
    )
    hit_summary.to_csv(tabdir / "DEF_confirmatory_hit_summary.csv", index=False)

    manifest = {
        "status": "COMPLETE",
        "dataset": "AHCA",
        "single_donor": True,
        "figure3_panels": {
            "A": "Clock remodeling versus broader tissue responsiveness",
            "B": "Reactome matched-competitive tissue programs",
            "C": "Hallmark matched-competitive tissue programs",
            "D": "Frozen Tabula pathways: correspondence beta vs matched-gene specificity",
            "E": "Biological-module composition of confirmed frozen families",
            "F": "Representative confirmed frozen coherent families",
        },
        "ABC_role": "supportive/context replication",
        "AHCA_allgene_missingness_handling": (
            "Conservative complete-measurement gene universe: genes with any "
            "non-finite value across eligible AHCA pseudobulks were excluded; "
            "no zero-filling and no imputation were used for full-transcriptome Fig3 analyses."
        ),
        "ABC_model": (
            "single-donor adaptation of cell-type-adjusted continuous tissue effect: "
            "reduced = original_cell_type + log1p(n_cells); "
            "full adds tissue; matched competitive enrichment uses mean/SD/prevalence bins"
        ),
        "DEF_role": "pre-specified confirmatory external sensitivity",
        "DEF_frozen_reference": str(ontology_path),
        "DEF_frozen_counts_expected": {
            "unique_pathways": 98,
            "coherent_families": 69,
        },
        "DEF_clock_endpoint": "Expanded17 Euclidean tissue-pair distance within lineage",
        "DEF_pathway_endpoint": "absolute tissue-pair difference of mean global-z pathway score",
        "DEF_correspondence_test": (
            f"{args.permutations} within-donor tissue-label permutations; "
            "BH within lineage x collection over the 98 frozen pathways"
        ),
        "DEF_specificity_test": (
            f"{args.specificity_draws} matched gene-set draws/pathway; "
            "4D matching = global mean + SD + prevalence + lineage tissue-structure; "
            "BH within lineage x collection over the 98 frozen pathways"
        ),
        "DEF_confirmatory_rule": (
            "beta>0 AND specificity_z>0 AND correspondence_FDR<0.05 "
            "AND specificity_FDR<0.05"
        ),
        "important_limitation": (
            "AHCA contains one donor. DEF results are single-donor confirmatory "
            "sensitivities of frozen Tabula discoveries and are not independent "
            "multi-donor formal hits."
        ),
        "static_expression_only": True,
        "no_rhythmicity_phase_amplitude_oscillation_claim": True,
        "no_causal_claim": True,
        "inputs": {
            "allgene_dir": str(allgene_dir),
            "clock_pseudobulk": str(clock_path),
            "ontology": str(ontology_path),
            "hallmark_gmt": str(hallmark_gmt),
            "reactome_gmt": str(reactome_gmt),
        },
        "parameters": vars(args),
    }

    # Path objects are not JSON serializable.
    manifest["parameters"] = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in manifest["parameters"].items()
    }

    (auddir / "AHCA_Fig3_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log()
    log("=" * 112)
    log("ANALYSIS COMPLETE")
    log("=" * 112)
    log("FIGURES")
    for p in sorted(figdir.glob("AHCA_Fig3*.png")):
        log(str(p))
    log()
    log("CONFIRMATORY SUMMARY")
    log(hit_summary.to_string(index=False))
    log()
    log(
        f"Confirmed frozen families: "
        f"{int(fam_summary.family_confirmatory_supported.sum())}/"
        f"{len(fam_summary)} tested representative families"
    )
    log("=" * 112)


if __name__ == "__main__":
    main()
