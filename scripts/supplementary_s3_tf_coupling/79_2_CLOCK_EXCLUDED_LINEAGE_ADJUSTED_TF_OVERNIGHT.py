#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_OVERNIGHT.py

79.2 — final computational TF gate before experiments

Primary questions
-----------------
Q1. After removing all 17 core-clock genes from every DoRothEA regulon,
    do TF-activity ↔ six-arm associations remain?

Q2. After additionally removing the additive donor + tissue + lineage main effects,
    do TF ↔ six-arm associations remain?

Interpretation
--------------
- "clock-excluded matched association" asks whether the old cross-lineage signal
  survives removal of mathematical clock-gene leakage.
- "lineage-adjusted association" asks whether the TF-arm relationship contains
  information beyond simple lineage identity.

This analysis is observational. PASS means "good mechanistic candidate for
perturbation", not causal proof.

Designed for unattended overnight execution with checkpoints/resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
FREEZE = ROOT / "00_freeze"

OUT = ROOT / "79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_20261004"
AUDIT = OUT / "00_AUDIT"
CACHE = OUT / "01_CACHE"
TABLES = OUT / "02_TABLES"
FIGURES = OUT / "03_FIGURES"
REPORT = OUT / "04_REPORT"

for d in [OUT, AUDIT, CACHE, TABLES, FIGURES, REPORT]:
    d.mkdir(parents=True, exist_ok=True)

LOG = OUT / "79_2_console.log"

META = FREEZE / "observation_metadata_corrected_v3_FROZEN.tsv"
NPZ = (
    PROJECT
    / "results"
    / "phaseA_clock_architecture"
    / "dependencies"
    / "broad_normalized_mean.npz"
)

NETWORK = (
    ROOT
    / "79_FIX2_STRICT_INTERACTION_AND_TF_20261004"
    / "00_INPUTS"
    / "79_FIX2_DOROTHEA_HS_ABC.csv"
)

PREV_PRIORITY = (
    ROOT
    / "79_FIX2_STRICT_INTERACTION_AND_TF_20261004"
    / "02_TF_DISCOVERY"
    / "79_FIX2_TF_CANDIDATE_PRIORITY.csv"
)

PREV_ACTIVITY_ASSOC = (
    ROOT
    / "79_FIX2_STRICT_INTERACTION_AND_TF_20261004"
    / "02_TF_DISCOVERY"
    / "79_FIX2_TF_activity_arm_association.csv"
)

SIXARM_CANDIDATES = [
    ROOT
    / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930"
    / "source_data"
    / "DONOR_TISSUE_LINEAGE_SCORES.csv",

    ROOT
    / "78_LINEAGE_INTRINSIC_3LOOP_VALIDATION"
    / "01_TABLES"
    / "Tabula_DONOR_TISSUE_LINEAGE_SIXARM.csv",

    ROOT
    / "78_LINEAGE_INTRINSIC_3LOOP_VALIDATION"
    / "source_data"
    / "Tabula_DONOR_TISSUE_LINEAGE_SIXARM.csv",
]

LINEAGES = [
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

ARMS = [
    "EBOX_POS",
    "EBOX_NEG",
    "RORE_ACT",
    "RORE_REP",
    "DBOX_ACT",
    "DBOX_REP",
]

ARM_GENES = {
    "EBOX_POS": ["ARNTL", "CLOCK", "NPAS2"],
    "EBOX_NEG": ["PER1", "PER2", "PER3", "CRY1", "CRY2"],
    "RORE_ACT": ["RORA", "RORB", "RORC"],
    "RORE_REP": ["NR1D1", "NR1D2"],
    "DBOX_ACT": ["DBP", "TEF", "HLF"],
    "DBOX_REP": ["NFIL3"],
}

CLOCK17 = sorted({g for v in ARM_GENES.values() for g in v})

FOCUS_TFS = ["SPI1", "ERG", "PAX5", "LEF1"]

MIN_REGULON = 10
N_PERM = 10000
RNG_SEED = 20261004
TOP_N_ROBUST = 60
EPS = 1e-12

# Pre-frozen interpretation thresholds.
LEAKAGE_ROBUST_RETENTION = 0.75
LEAKAGE_SENSITIVE_RETENTION = 0.50
ADJUSTED_MIN_ABS_R = 0.15
ADJUSTED_FDR = 0.05
LINEAGE_SIGN_FRACTION = 0.60

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# =============================================================================
# UTILITIES
# =============================================================================

def log(msg=""):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def savefig(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kwargs = dict(bbox_inches="tight", pad_inches=0.08, facecolor="white")
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIGURES / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def bh(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    if len(x) == 0:
        return q
    order = np.argsort(x)
    xx = x[order]
    qq = xx * len(xx) / np.arange(1, len(xx) + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    qq = np.clip(qq, 0, 1)
    back = np.empty(len(x))
    back[order] = qq
    q[ok] = back
    return q


def ensure_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return x


def zscore_columns(M):
    M = ensure_2d(M)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= EPS)] = 1.0
    Z = (M - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z


def normalize_lineage(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in LINEAGES:
        return s
    low = s.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
    if low in LINEAGES:
        return low
    if "macrophage" in low or "monocyte" in low or "myeloid" in low:
        return "macrophage_monocyte"
    if low in {"b_cell", "b_cells", "bcell", "b"}:
        return "b_cell"
    if low in {"t_cell", "t_cells", "tcell", "t"}:
        return "t_cell"
    if "fibro" in low:
        return "fibroblast"
    if "endothel" in low:
        return "endothelial"
    return None


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def categorical_design(meta, cols):
    parts = [np.ones((len(meta), 1), dtype=float)]
    names = ["Intercept"]
    for c in cols:
        d = pd.get_dummies(
            meta[c].astype(str),
            prefix=c,
            drop_first=True,
            dtype=float,
        )
        if d.shape[1]:
            parts.append(d.to_numpy(float))
            names.extend(d.columns.tolist())
    return np.column_stack(parts), names


def residualize(M, X):
    M = ensure_2d(M)
    X = ensure_2d(X)
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    R = M - X @ beta
    return R


def correlation_matrix(A, B):
    A = zscore_columns(A)
    B = zscore_columns(B)
    if len(A) != len(B):
        raise RuntimeError("correlation_matrix row mismatch")
    return (A.T @ B) / max(len(A) - 1, 1)


def center_within_blocks(M, block_arrays):
    M = ensure_2d(M).copy()
    for idx in block_arrays:
        if len(idx) == 0:
            continue
        M[idx] -= M[idx].mean(axis=0, keepdims=True)
    return M


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_table(df, max_rows=30):
    if df is None or df.empty:
        return "_No rows._"
    d = df.head(max_rows).copy()

    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.5g}"
        return str(v).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, d.columns)) + " |",
        "| " + " | ".join(["---"] * len(d.columns)) + " |",
    ]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in d.columns) + " |")
    return "\n".join(lines)


# =============================================================================
# INPUT AUDIT
# =============================================================================

def load_network():
    if not NETWORK.exists():
        raise FileNotFoundError(
            f"DoRothEA network missing. Expected:\n{NETWORK}\n"
            "79 FIX2.1 must have completed first."
        )

    net = pd.read_csv(NETWORK)
    cmap = {c.lower(): c for c in net.columns}
    source = cmap.get("source") or cmap.get("tf")
    target = cmap.get("target")
    mor = cmap.get("mor") or cmap.get("weight")
    conf = cmap.get("confidence")

    if source is None or target is None or mor is None:
        raise RuntimeError(
            f"Unexpected DoRothEA columns: {net.columns.tolist()}"
        )

    out = pd.DataFrame({
        "source": net[source].astype(str).str.upper(),
        "target": net[target].astype(str).str.upper(),
        "mor": pd.to_numeric(net[mor], errors="coerce"),
        "confidence": (
            net[conf].astype(str)
            if conf is not None
            else "NA"
        ),
    })

    out = (
        out
        .dropna(subset=["source", "target", "mor"])
        .drop_duplicates(["source", "target"])
        .reset_index(drop=True)
    )

    return out


def load_sixarm():
    p = first_existing(SIXARM_CANDIDATES)

    if p is None:
        # Last-resort recursive search under 64/78 only.
        hits = []
        for base in [
            ROOT / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930",
            ROOT / "78_LINEAGE_INTRINSIC_3LOOP_VALIDATION",
        ]:
            if base.exists():
                hits.extend(base.rglob("*DONOR*TISSUE*LINEAGE*SCORE*.csv"))
                hits.extend(base.rglob("*SIXARM*.csv"))

        # De-duplicate.
        uniq = []
        seen = set()
        for h in hits:
            if h not in seen:
                seen.add(h)
                uniq.append(h)

        for h in uniq:
            try:
                x = pd.read_csv(h, nrows=5)
                cols = set(x.columns)
                if set(ARMS).issubset(cols):
                    p = h
                    break
            except Exception:
                pass

    if p is None:
        raise FileNotFoundError(
            "Could not locate donor×tissue×lineage six-arm table."
        )

    d = pd.read_csv(p)

    lineage_col = None
    for c in [
        "lineage",
        "priority_lineage",
        "priority_lineage_corrected",
    ]:
        if c in d.columns:
            lineage_col = c
            break

    if lineage_col is None:
        raise RuntimeError(
            f"Six-arm file lacks lineage column: {p}\n"
            f"columns={d.columns.tolist()}"
        )

    required = {"donor_id", "tissue", lineage_col, *ARMS}
    miss = required - set(d.columns)
    if miss:
        raise RuntimeError(
            f"Six-arm file missing columns {sorted(miss)}: {p}"
        )

    d = d.copy()
    d["lineage"] = d[lineage_col].map(normalize_lineage)
    d = d[d["lineage"].isin(LINEAGES)].copy()

    # If multiple rows map to the same major-lineage aggregate, average six-arm values.
    d["donor_id"] = d["donor_id"].astype(str)
    d["tissue"] = d["tissue"].astype(str)

    d = (
        d
        .groupby(["donor_id", "tissue", "lineage"], as_index=False)[ARMS]
        .mean()
    )

    return d, p


def load_and_aggregate_expression(net, six):
    if not META.exists():
        raise FileNotFoundError(META)
    if not NPZ.exists():
        raise FileNotFoundError(NPZ)

    meta = pd.read_csv(META, sep="\t")

    if "priority_lineage_corrected" in meta.columns:
        meta["lineage"] = meta["priority_lineage_corrected"].map(normalize_lineage)
    elif "priority_lineage" in meta.columns:
        meta["lineage"] = meta["priority_lineage"].map(normalize_lineage)
    else:
        raise RuntimeError("Frozen metadata lacks priority lineage.")

    required_meta = {"group_id", "donor_id", "tissue", "lineage"}
    miss = required_meta - set(meta.columns)
    if miss:
        raise RuntimeError(f"Frozen metadata missing {sorted(miss)}")

    meta["group_id"] = meta["group_id"].astype(str)
    meta["donor_id"] = meta["donor_id"].astype(str)
    meta["tissue"] = meta["tissue"].astype(str)

    data = np.load(NPZ, allow_pickle=True)
    ids = data["group_id"].astype(str)
    genes = data["genes"].astype(str)
    X = data["expression"]

    if len(set(genes)) != len(genes):
        raise RuntimeError("Expression gene symbols are not unique.")
    if set(ids) != set(meta["group_id"]):
        raise RuntimeError("Frozen expression IDs differ from frozen metadata.")

    order = pd.Index(ids).get_indexer(meta["group_id"])
    if (order < 0).any():
        raise RuntimeError("Could not align frozen matrix to metadata.")
    X = X[order]

    keep = meta["lineage"].isin(LINEAGES).to_numpy()
    meta5 = meta.loc[
        keep,
        ["group_id", "donor_id", "tissue", "lineage"]
    ].reset_index(drop=True)

    original_rows = np.where(keep)[0]

    gene_upper = np.char.upper(genes.astype(str))
    gene_map = {}
    for i, g in enumerate(gene_upper):
        gene_map.setdefault(str(g), i)

    # Need every network target + source that exists.
    wanted = sorted(
        {
            g
            for g in set(net["source"]).union(net["target"])
            if g in gene_map
        }
    )
    if not wanted:
        raise RuntimeError("No DoRothEA genes found in expression matrix.")

    sel_idx = np.array([gene_map[g] for g in wanted], dtype=int)

    local = meta5.copy()
    local["_local_row"] = np.arange(len(local), dtype=int)

    agg_rows = []
    agg_values = []

    for key, g in local.groupby(
        ["donor_id", "tissue", "lineage"],
        sort=False
    ):
        local_idx = g["_local_row"].to_numpy(dtype=int)
        source_idx = original_rows[local_idx]

        # Critical: use two-step indexing to preserve 2D shape.
        arr = np.asarray(X[source_idx][:, sel_idx], dtype=np.float64)
        agg_values.append(arr.mean(axis=0))
        agg_rows.append(key)

    agg_meta = pd.DataFrame(
        agg_rows,
        columns=["donor_id", "tissue", "lineage"]
    )
    agg_X = np.vstack(agg_values)
    agg_genes = np.array(wanted, dtype=object)

    # Exact alignment to six-arm phenotype.
    agg_meta["key"] = (
        agg_meta["donor_id"].astype(str) + "||" +
        agg_meta["tissue"].astype(str) + "||" +
        agg_meta["lineage"].astype(str)
    )
    six = six.copy()
    six["key"] = (
        six["donor_id"].astype(str) + "||" +
        six["tissue"].astype(str) + "||" +
        six["lineage"].astype(str)
    )

    if agg_meta["key"].duplicated().any():
        raise RuntimeError("Duplicate aggregated expression key.")
    if six["key"].duplicated().any():
        raise RuntimeError("Duplicate six-arm key.")

    miss_six = sorted(set(six["key"]) - set(agg_meta["key"]))
    extra_expr = sorted(set(agg_meta["key"]) - set(six["key"]))

    audit = {
        "sixarm_rows": len(six),
        "expression_aggregate_rows": len(agg_meta),
        "missing_sixarm_keys": len(miss_six),
        "extra_expression_keys": len(extra_expr),
        "selected_expression_genes": len(agg_genes),
    }
    write_json(AUDIT / "79_2_ALIGNMENT_AUDIT.json", audit)

    if miss_six or extra_expr:
        pd.DataFrame({"missing_sixarm_key": miss_six}).to_csv(
            AUDIT / "79_2_missing_sixarm_keys.csv", index=False
        )
        pd.DataFrame({"extra_expression_key": extra_expr}).to_csv(
            AUDIT / "79_2_extra_expression_keys.csv", index=False
        )
        raise RuntimeError(
            f"Exact alignment failed: missing_six={len(miss_six)}, "
            f"extra_expression={len(extra_expr)}"
        )

    pos = pd.Index(agg_meta["key"]).get_indexer(six["key"])
    if (pos < 0).any():
        raise RuntimeError("Unexpected six-arm alignment failure.")

    agg_meta = agg_meta.iloc[pos].reset_index(drop=True)
    agg_X = agg_X[pos]
    six = six.reset_index(drop=True)

    return agg_meta, agg_X, agg_genes, six


# =============================================================================
# TF ACTIVITY
# =============================================================================

def compute_ulm_style_activity(
    agg_X,
    agg_genes,
    network,
    exclude_clock_targets,
):
    """
    ULM-style TF activity.

    Target genes are z-scored across donor×tissue×lineage observations.
    For each sample and TF, target expression is regressed across signed
    DoRothEA MoR weights; slope t-statistic is used as activity.

    This matches the mathematical logic used in 79 FIX2, while allowing
    an explicit CLOCK17-excluded regulon.
    """
    gm = {str(g).upper(): i for i, g in enumerate(agg_genes)}

    Z = np.asarray(agg_X, dtype=float)
    Z = zscore_columns(Z)

    activity_cols = []
    tf_names = []
    audit_rows = []

    for tf, d0 in network.groupby("source", sort=True):
        tf = str(tf).upper()

        # Source TF itself cannot be one of the Clock17 outcome genes.
        if tf in CLOCK17:
            continue

        d = d0.copy()
        n_original = len(d)

        original_clock_targets = sorted(
            set(d.loc[d["target"].isin(CLOCK17), "target"])
        )

        if exclude_clock_targets:
            d = d[~d["target"].isin(CLOCK17)].copy()

        d = d[d["target"].isin(gm)].copy()
        d = d[np.isfinite(pd.to_numeric(d["mor"], errors="coerce"))].copy()

        n_used = len(d)
        if n_used < MIN_REGULON:
            audit_rows.append({
                "TF": tf,
                "exclude_clock_targets": exclude_clock_targets,
                "n_edges_original": n_original,
                "n_clock_targets_removed": len(original_clock_targets),
                "clock_targets_removed": ",".join(original_clock_targets),
                "n_targets_used": n_used,
                "status": "DROPPED_MIN_REGULON",
            })
            continue

        idx = np.array([gm[g] for g in d["target"]], dtype=int)
        w = d["mor"].to_numpy(dtype=float)

        wc = w - w.mean()
        den = float(np.dot(wc, wc))

        if den <= EPS:
            audit_rows.append({
                "TF": tf,
                "exclude_clock_targets": exclude_clock_targets,
                "n_edges_original": n_original,
                "n_clock_targets_removed": len(original_clock_targets),
                "clock_targets_removed": ",".join(original_clock_targets),
                "n_targets_used": n_used,
                "status": "DROPPED_ZERO_WEIGHT_VARIANCE",
            })
            continue

        Y = Z[:, idx]
        yc = Y - Y.mean(axis=1, keepdims=True)

        beta = (yc @ wc) / den
        fitted = beta[:, None] * wc[None, :]
        resid = yc - fitted

        df = max(n_used - 2, 1)
        s2 = np.sum(resid * resid, axis=1) / df
        se = np.sqrt(np.maximum(s2 / den, EPS))
        tscore = beta / se

        activity_cols.append(tscore)
        tf_names.append(tf)

        audit_rows.append({
            "TF": tf,
            "exclude_clock_targets": exclude_clock_targets,
            "n_edges_original": n_original,
            "n_clock_targets_removed": len(original_clock_targets),
            "clock_targets_removed": ",".join(original_clock_targets),
            "n_targets_used": n_used,
            "status": "USED",
        })

    if not activity_cols:
        raise RuntimeError(
            f"No TF activities computed. exclude_clock_targets={exclude_clock_targets}"
        )

    A = np.column_stack(activity_cols)
    audit = pd.DataFrame(audit_rows)

    return A, tf_names, audit


def tf_expression_matrix(agg_X, agg_genes, network):
    gm = {str(g).upper(): i for i, g in enumerate(agg_genes)}
    tfs = sorted(
        {
            tf
            for tf in network["source"].astype(str).str.upper()
            if tf not in CLOCK17 and tf in gm
        }
    )
    idx = [gm[tf] for tf in tfs]
    X = np.asarray(agg_X[:, idx], dtype=float)
    return X, tfs


# =============================================================================
# ASSOCIATION 1: DONOR×TISSUE MATCHED (OLD LOGIC, CLOCK-EXCLUDED)
# =============================================================================

def donor_tissue_multilineage_blocks(meta):
    key = (
        meta["donor_id"].astype(str)
        + "||"
        + meta["tissue"].astype(str)
    )

    blocks = []
    for _, idx in key.groupby(key, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        if len(idx) < 2:
            continue
        if meta.loc[idx, "lineage"].nunique() < 2:
            continue
        blocks.append(idx)

    return blocks


def matched_association(P, pnames, meta, six, label, n_perm=N_PERM):
    blocks = donor_tissue_multilineage_blocks(meta)
    if not blocks:
        raise RuntimeError("No donor×tissue multilineage blocks.")

    use = np.sort(np.concatenate(blocks))
    remap = {old: new for new, old in enumerate(use)}
    blocks2 = [
        np.array([remap[i] for i in b], dtype=int)
        for b in blocks
    ]

    Pc = center_within_blocks(P[use], blocks2)
    Yc = center_within_blocks(
        six.loc[use, ARMS].to_numpy(dtype=float),
        blocks2,
    )

    Pz = zscore_columns(Pc)
    Yz = zscore_columns(Yc)

    R = correlation_matrix(Pz, Yz)

    # Permute TF rows within exact donor×tissue block.
    rng = np.random.default_rng(
        RNG_SEED + {
            "full_activity": 10,
            "clock_excluded_activity": 20,
            "expression": 30,
        }.get(label, 100)
    )

    exceed = np.zeros(R.shape, dtype=np.int64)
    exceed_max = np.zeros(R.shape, dtype=np.int64)
    obs_abs = np.abs(R)

    for b in range(n_perm):
        Pperm = Pz.copy()
        for idx in blocks2:
            Pperm[idx] = Pperm[rng.permutation(idx)]

        Rp = np.abs(
            (Pperm.T @ Yz)
            / max(len(Pperm) - 1, 1)
        )

        exceed += (Rp >= obs_abs - 1e-12)
        mx = float(np.max(Rp))
        exceed_max += (mx >= obs_abs - 1e-12)

        if (b + 1) % 500 == 0:
            log(
                f"matched permutation [{label}] "
                f"{b+1}/{n_perm}"
            )

    rawp = (exceed + 1) / (n_perm + 1)
    maxp = (exceed_max + 1) / (n_perm + 1)

    rows = []
    for j, tf in enumerate(pnames):
        for a, arm in enumerate(ARMS):
            rows.append({
                "predictor_type": label,
                "TF": tf,
                "arm": arm,
                "matched_r": float(R[j, a]),
                "matched_R2": float(R[j, a] ** 2),
                "n_rows": len(use),
                "n_blocks": len(blocks2),
                "perm_P": float(rawp[j, a]),
                "maxT_FWER": float(maxp[j, a]),
            })

    out = pd.DataFrame(rows)
    out["perm_FDR"] = bh(out["perm_P"].to_numpy(dtype=float))

    return out


# =============================================================================
# ASSOCIATION 2: REMOVE DONOR + TISSUE + LINEAGE
# =============================================================================

def lineage_adjusted_association(
    P,
    pnames,
    meta,
    six,
    label,
    n_perm=N_PERM,
):
    """
    Frisch-Waugh-Lovell:
      residual(TF ~ donor+tissue+lineage)
      residual(arm ~ donor+tissue+lineage)
      correlate residuals.

    Inference:
      donor-cluster sign-flip permutation.
    All rows from a donor receive the same random +/- sign, preserving
    within-donor dependence and the observed tissue/lineage structure.
    """
    X, confound_names = categorical_design(
        meta,
        ["donor_id", "tissue", "lineage"]
    )

    Pr = residualize(P, X)
    Yr = residualize(
        six[ARMS].to_numpy(dtype=float),
        X,
    )

    Pz = zscore_columns(Pr)
    Yz = zscore_columns(Yr)
    R = correlation_matrix(Pz, Yz)

    donor_codes, donor_levels = pd.factorize(
        meta["donor_id"].astype(str),
        sort=True
    )
    n_donors = len(donor_levels)

    rng = np.random.default_rng(
        RNG_SEED + {
            "clock_excluded_activity": 200,
            "expression": 300,
        }.get(label, 400)
    )

    exceed = np.zeros(R.shape, dtype=np.int64)
    exceed_max = np.zeros(R.shape, dtype=np.int64)
    obs_abs = np.abs(R)

    for b in range(n_perm):
        donor_sign = rng.choice(
            np.array([-1.0, 1.0]),
            size=n_donors,
            replace=True,
        )
        row_sign = donor_sign[donor_codes][:, None]

        Pperm = Pz * row_sign

        Rp = np.abs(
            (Pperm.T @ Yz)
            / max(len(Pperm) - 1, 1)
        )

        exceed += (Rp >= obs_abs - 1e-12)
        mx = float(np.max(Rp))
        exceed_max += (mx >= obs_abs - 1e-12)

        if (b + 1) % 500 == 0:
            log(
                f"lineage-adjusted donor sign-flip [{label}] "
                f"{b+1}/{n_perm}"
            )

    rawp = (exceed + 1) / (n_perm + 1)
    maxp = (exceed_max + 1) / (n_perm + 1)

    rows = []
    for j, tf in enumerate(pnames):
        for a, arm in enumerate(ARMS):
            rows.append({
                "predictor_type": label,
                "TF": tf,
                "arm": arm,
                "adjusted_r": float(R[j, a]),
                "adjusted_R2": float(R[j, a] ** 2),
                "n_rows": len(meta),
                "n_donors": n_donors,
                "n_tissues": meta["tissue"].nunique(),
                "perm_P": float(rawp[j, a]),
                "maxT_FWER": float(maxp[j, a]),
            })

    out = pd.DataFrame(rows)
    out["perm_FDR"] = bh(out["perm_P"].to_numpy(dtype=float))

    # Cache residual matrices for diagnostics.
    np.savez_compressed(
        CACHE / f"79_2_{label}_lineage_adjusted_residuals.npz",
        predictor_residual=Pr.astype(np.float32),
        arm_residual=Yr.astype(np.float32),
        predictor_names=np.array(pnames, dtype=object),
        arm_names=np.array(ARMS, dtype=object),
    )

    return out


# =============================================================================
# WITHIN-LINEAGE ROBUSTNESS
# =============================================================================

def within_lineage_associations(P, pnames, meta, six, label):
    rows = []

    for lin in LINEAGES:
        idx = np.where(meta["lineage"].eq(lin).to_numpy())[0]
        if len(idx) < 20:
            continue

        m = meta.iloc[idx].reset_index(drop=True)
        X, _ = categorical_design(
            m,
            ["donor_id", "tissue"]
        )

        Pr = residualize(P[idx], X)
        Yr = residualize(
            six.iloc[idx][ARMS].to_numpy(dtype=float),
            X,
        )

        R = correlation_matrix(Pr, Yr)

        for j, tf in enumerate(pnames):
            for a, arm in enumerate(ARMS):
                rows.append({
                    "predictor_type": label,
                    "lineage": lin,
                    "TF": tf,
                    "arm": arm,
                    "r": float(R[j, a]),
                    "n_rows": len(idx),
                    "n_donors": m["donor_id"].nunique(),
                    "n_tissues": m["tissue"].nunique(),
                })

    return pd.DataFrame(rows)


# =============================================================================
# LEAVE-ONE-BLOCK ROBUSTNESS FOR TOP CANDIDATES
# =============================================================================

def leave_one_out_r(
    P,
    pnames,
    meta,
    six,
    candidate_pairs,
    mode,
):
    name_to_j = {n: j for j, n in enumerate(pnames)}
    rows = []

    blocks = sorted(meta[mode].astype(str).unique())

    for tf, arm in candidate_pairs:
        if tf not in name_to_j:
            continue
        j = name_to_j[tf]

        vals = []

        for held in blocks:
            keep = meta[mode].astype(str).ne(held).to_numpy()

            if keep.sum() < 50:
                continue

            m = meta.loc[keep].reset_index(drop=True)
            X, _ = categorical_design(
                m,
                ["donor_id", "tissue", "lineage"]
            )

            Pr = residualize(
                P[keep][:, [j]],
                X,
            )
            Yr = residualize(
                six.loc[keep, [arm]].to_numpy(dtype=float),
                X,
            )

            if (
                np.std(Pr[:, 0]) <= EPS
                or np.std(Yr[:, 0]) <= EPS
            ):
                continue

            r = float(
                np.corrcoef(
                    Pr[:, 0],
                    Yr[:, 0],
                )[0, 1]
            )

            if np.isfinite(r):
                vals.append((held, r))

        rows.append({
            "TF": tf,
            "arm": arm,
            "deletion_type": mode,
            "n_deletions": len(vals),
            "median_r": (
                float(np.median([x[1] for x in vals]))
                if vals else np.nan
            ),
            "min_r": (
                float(np.min([x[1] for x in vals]))
                if vals else np.nan
            ),
            "max_r": (
                float(np.max([x[1] for x in vals]))
                if vals else np.nan
            ),
            "detail": ";".join(
                f"{x[0]}:{x[1]:.4f}"
                for x in vals
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# DIRECT CLOCK TARGETS
# =============================================================================

def direct_clock_targets(network):
    rows = []

    for tf, d in network.groupby("source", sort=True):
        if tf in CLOCK17:
            continue

        dc = d[d["target"].isin(CLOCK17)]
        targets = sorted(set(dc["target"]))

        arms = sorted(
            {
                arm
                for arm, genes in ARM_GENES.items()
                if any(t in genes for t in targets)
            }
        )

        rows.append({
            "TF": tf,
            "n_clock_targets": len(targets),
            "clock_targets": ",".join(targets),
            "clock_target_arms": ",".join(arms),
        })

    return pd.DataFrame(rows)


# =============================================================================
# CANDIDATE RANKING
# =============================================================================

def best_per_tf(df, r_col, prefix):
    if df.empty:
        return pd.DataFrame(columns=["TF"])

    d = df.copy()
    d["_abs_r"] = d[r_col].abs()

    d = (
        d
        .sort_values(
            ["maxT_FWER", "perm_FDR", "_abs_r"],
            ascending=[True, True, False]
        )
        .groupby("TF", as_index=False)
        .first()
    )

    keep = [
        "TF",
        "arm",
        r_col,
        "perm_P",
        "perm_FDR",
        "maxT_FWER",
    ]

    d = d[keep].rename(
        columns={
            c: f"{prefix}_{c}"
            for c in keep
            if c != "TF"
        }
    )

    return d


def same_arm_lookup(df, tf, arm, rcol):
    x = df[(df["TF"] == tf) & (df["arm"] == arm)]
    if x.empty:
        return np.nan
    return float(x.iloc[0][rcol])


def build_final_ranking(
    full_match,
    excl_match,
    excl_adj,
    expr_adj,
    within_lin,
    direct,
    prev_priority,
):
    excl_best = best_per_tf(
        excl_match,
        "matched_r",
        "excl_match"
    )

    adj_best = best_per_tf(
        excl_adj,
        "adjusted_r",
        "adj"
    )

    expr_best = best_per_tf(
        expr_adj,
        "adjusted_r",
        "expr_adj"
    )

    all_tfs = sorted(
        set(excl_best["TF"])
        .union(adj_best["TF"])
        .union(expr_best["TF"])
    )

    out = pd.DataFrame({"TF": all_tfs})

    for d in [excl_best, adj_best, expr_best, direct]:
        if d is not None and not d.empty:
            out = out.merge(d, on="TF", how="left")

    # Compare full versus clock-excluded activity on the same arm chosen by
    # clock-excluded matched analysis.
    full_same = []
    excl_same = []
    retention = []
    leakage_status = []

    for r in out.itertuples():
        arm = getattr(r, "excl_match_arm", np.nan)

        if pd.isna(arm):
            full_same.append(np.nan)
            excl_same.append(np.nan)
            retention.append(np.nan)
            leakage_status.append("NO_MATCHED_ARM")
            continue

        rf = same_arm_lookup(
            full_match,
            r.TF,
            arm,
            "matched_r"
        )
        re = same_arm_lookup(
            excl_match,
            r.TF,
            arm,
            "matched_r"
        )

        full_same.append(rf)
        excl_same.append(re)

        if not np.isfinite(rf) or abs(rf) <= EPS:
            ret = np.nan
        else:
            ret = abs(re) / abs(rf)

        retention.append(ret)

        same_sign = (
            np.isfinite(rf)
            and np.isfinite(re)
            and np.sign(rf) == np.sign(re)
        )

        if (
            same_sign
            and np.isfinite(ret)
            and ret >= LEAKAGE_ROBUST_RETENTION
        ):
            leakage_status.append("ROBUST_TO_CLOCK_EXCLUSION")
        elif (
            (not same_sign)
            or (
                np.isfinite(ret)
                and ret < LEAKAGE_SENSITIVE_RETENTION
            )
        ):
            leakage_status.append("CLOCK_LEAKAGE_SENSITIVE")
        else:
            leakage_status.append("INTERMEDIATE")

    out["full_activity_same_arm_r"] = full_same
    out["clock_excluded_same_arm_r"] = excl_same
    out["clock_exclusion_retention"] = retention
    out["clock_leakage_status"] = leakage_status

    # Cross-lineage sign robustness on the ADJUSTED best arm.
    sign_rows = []

    for r in out.itertuples():
        arm = getattr(r, "adj_arm", np.nan)

        if pd.isna(arm):
            sign_rows.append(
                (np.nan, np.nan, np.nan, "")
            )
            continue

        d = within_lin[
            (within_lin["TF"] == r.TF)
            & (within_lin["arm"] == arm)
        ].copy()

        if d.empty:
            sign_rows.append(
                (np.nan, np.nan, np.nan, "")
            )
            continue

        global_r = getattr(r, "adj_adjusted_r")
        sign = np.sign(global_r)

        vals = d["r"].to_numpy(dtype=float)
        same = (
            np.mean(np.sign(vals) == sign)
            if sign != 0
            else np.nan
        )

        sign_rows.append((
            len(vals),
            same,
            float(np.median(vals)),
            ";".join(
                f"{rr.lineage}:{rr.r:.4f}"
                for rr in d.itertuples()
            ),
        ))

    out["n_lineages_evaluable"] = [x[0] for x in sign_rows]
    out["adjusted_same_sign_lineage_fraction"] = [x[1] for x in sign_rows]
    out["adjusted_median_within_lineage_r"] = [x[2] for x in sign_rows]
    out["within_lineage_r_detail"] = [x[3] for x in sign_rows]

    # Experimental gate.
    persists = (
        out["adj_adjusted_r"].abs().ge(ADJUSTED_MIN_ABS_R)
        & out["adj_perm_FDR"].lt(ADJUSTED_FDR)
    )
    out["persists_after_lineage_adjustment"] = persists

    lineage_robust = (
        out["adjusted_same_sign_lineage_fraction"]
        .fillna(0)
        .ge(LINEAGE_SIGN_FRACTION)
    )
    out["cross_lineage_direction_robust"] = lineage_robust

    leakage_robust = out["clock_leakage_status"].eq(
        "ROBUST_TO_CLOCK_EXCLUSION"
    )

    matched_supported = (
        out["excl_match_maxT_FWER"].fillna(1).lt(0.05)
        | out["excl_match_perm_FDR"].fillna(1).lt(0.05)
    )
    out["clock_excluded_matched_supported"] = matched_supported

    # Expression support: no need to be significant for activity candidate,
    # but same-sign adjusted expression is useful.
    expr_same = (
        np.sign(out["expr_adj_adjusted_r"].fillna(0))
        == np.sign(out["adj_adjusted_r"].fillna(0))
    )
    out["activity_expression_adjusted_sign_agree"] = expr_same

    # Score is descriptive; tier rules below are the gate.
    score = np.zeros(len(out), dtype=float)
    score += 2 * matched_supported.astype(int)
    score += 3 * persists.astype(int)
    score += 2 * leakage_robust.astype(int)
    score += 1 * lineage_robust.astype(int)
    score += 1 * expr_same.astype(int)
    score += 1 * out["n_clock_targets"].fillna(0).gt(0).astype(int)
    out["mechanism_evidence_score"] = score

    tier = []

    for r in out.itertuples():
        if (
            r.clock_leakage_status == "ROBUST_TO_CLOCK_EXCLUSION"
            and bool(r.persists_after_lineage_adjustment)
            and bool(r.cross_lineage_direction_robust)
        ):
            tier.append("TIER1_EXPERIMENT_READY")

        elif (
            r.clock_leakage_status != "CLOCK_LEAKAGE_SENSITIVE"
            and bool(r.persists_after_lineage_adjustment)
        ):
            tier.append("TIER2_CONTEXT_SPECIFIC")

        elif (
            bool(r.clock_excluded_matched_supported)
            and not bool(r.persists_after_lineage_adjustment)
        ):
            tier.append("LINEAGE_MARKER_LIKELY")

        elif r.clock_leakage_status == "CLOCK_LEAKAGE_SENSITIVE":
            tier.append("LEAKAGE_SENSITIVE")

        else:
            tier.append("LOW")

    out["mechanism_tier"] = tier

    # Add previous 79 FIX2 information for continuity only.
    if prev_priority is not None and not prev_priority.empty:
        keep = [
            c for c in [
                "TF",
                "act_arm",
                "act_matched_r",
                "act_maxT_FWER",
                "act_lineage_R2",
                "act_lineage_FDR",
                "n_clock_targets",
                "clock_targets",
                "evidence_score",
                "priority",
            ]
            if c in prev_priority.columns
        ]
        pp = prev_priority[keep].copy()

        pp = pp.rename(
            columns={
                c: f"fix2_{c}"
                for c in pp.columns
                if c != "TF"
            }
        )

        out = out.merge(
            pp,
            on="TF",
            how="left"
        )

    focus_order = {
        tf: i
        for i, tf in enumerate(FOCUS_TFS)
    }
    out["focus_candidate"] = out["TF"].isin(FOCUS_TFS)

    tier_order = {
        "TIER1_EXPERIMENT_READY": 0,
        "TIER2_CONTEXT_SPECIFIC": 1,
        "LINEAGE_MARKER_LIKELY": 2,
        "LEAKAGE_SENSITIVE": 3,
        "LOW": 4,
    }
    out["_tier_order"] = out["mechanism_tier"].map(tier_order).fillna(9)

    out = out.sort_values(
        [
            "_tier_order",
            "mechanism_evidence_score",
            "adj_perm_FDR",
            "adj_adjusted_r",
        ],
        ascending=[
            True,
            False,
            True,
            False,
        ],
    ).drop(columns="_tier_order")

    return out


# =============================================================================
# FIGURES
# =============================================================================

def make_figures(
    full_match,
    excl_match,
    excl_adj,
    ranking,
    within_lin,
):
    # F1: full vs clock-excluded matched activity.
    m = full_match[
        ["TF", "arm", "matched_r"]
    ].rename(
        columns={"matched_r": "full_r"}
    ).merge(
        excl_match[
            ["TF", "arm", "matched_r"]
        ].rename(
            columns={"matched_r": "excluded_r"}
        ),
        on=["TF", "arm"],
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(
        m["full_r"],
        m["excluded_r"],
        s=12,
        alpha=0.35,
    )
    lim = max(
        0.25,
        float(np.nanmax(np.abs(
            m[["full_r", "excluded_r"]].to_numpy()
        )))
    )
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        lw=0.8,
        linestyle="--",
    )
    ax.axhline(0, lw=0.6)
    ax.axvline(0, lw=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("TF activity ↔ arm r\n(full DoRothEA regulon)")
    ax.set_ylabel("TF activity ↔ arm r\n(Clock17-excluded regulon)")
    ax.set_title("79.2A. Clock-gene leakage sensitivity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "79_2_F1_CLOCK_EXCLUSION_SCATTER")

    # F2: matched cross-lineage versus after lineage adjustment.
    d = excl_match[
        ["TF", "arm", "matched_r"]
    ].merge(
        excl_adj[
            ["TF", "arm", "adjusted_r"]
        ],
        on=["TF", "arm"],
        how="inner",
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(
        d["matched_r"],
        d["adjusted_r"],
        s=12,
        alpha=0.35,
    )
    ax.axhline(0, lw=0.6)
    ax.axvline(0, lw=0.6)
    ax.set_xlabel("Clock17-excluded matched r\n(lineage retained)")
    ax.set_ylabel("Clock17-excluded adjusted r\n(donor+tissue+lineage removed)")
    ax.set_title("79.2B. How much TF–arm signal survives lineage adjustment?")

    for tf in FOCUS_TFS:
        x = d[d["TF"] == tf].copy()
        if x.empty:
            continue
        k = x["adjusted_r"].abs().idxmax()
        rr = x.loc[k]
        ax.text(
            rr["matched_r"],
            rr["adjusted_r"],
            tf,
            fontsize=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "79_2_F2_LINEAGE_ADJUSTMENT_SCATTER")

    # F3: top final candidates.
    top = ranking.head(25).copy()
    fig, ax = plt.subplots(
        figsize=(7.2, max(5.0, 0.27 * len(top) + 2))
    )
    y = np.arange(len(top))
    ax.barh(
        y,
        top["mechanism_evidence_score"].to_numpy(float),
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top["TF"])
    ax.invert_yaxis()
    ax.set_xlabel("Mechanism evidence score")
    ax.set_title("79.2C. Final computational TF ranking")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "79_2_F3_FINAL_TF_RANKING")

    # F4: adjusted association heatmap for top 20.
    top_tfs = ranking["TF"].head(20).tolist()
    h = excl_adj[excl_adj["TF"].isin(top_tfs)].copy()
    mat = h.pivot(
        index="TF",
        columns="arm",
        values="adjusted_r"
    ).reindex(
        index=[x for x in top_tfs if x in set(h["TF"])],
        columns=ARMS,
    )

    if len(mat):
        fig, ax = plt.subplots(
            figsize=(7.4, max(5.0, 0.28 * len(mat) + 2))
        )
        im = ax.imshow(
            mat.to_numpy(float),
            aspect="auto",
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
        )
        ax.set_xticks(np.arange(len(ARMS)))
        ax.set_xticklabels(
            ARMS,
            rotation=45,
            ha="right"
        )
        ax.set_yticks(np.arange(len(mat)))
        ax.set_yticklabels(mat.index)
        ax.set_title(
            "79.2D. TF–arm associations after donor+tissue+lineage adjustment"
        )
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            label="partial residual r",
        )
        savefig(fig, "79_2_F4_ADJUSTED_TF_ARM_HEATMAP")

    # F5: lineage-stratified robustness for focus + top Tier1.
    show_tfs = []
    for tf in FOCUS_TFS:
        if tf in set(ranking["TF"]):
            show_tfs.append(tf)
    for tf in ranking.loc[
        ranking["mechanism_tier"].eq("TIER1_EXPERIMENT_READY"),
        "TF"
    ].head(8):
        if tf not in show_tfs:
            show_tfs.append(tf)

    rows = []
    for tf in show_tfs:
        rr = ranking[ranking["TF"] == tf].iloc[0]
        arm = rr["adj_arm"]
        dlin = within_lin[
            (within_lin["TF"] == tf)
            & (within_lin["arm"] == arm)
        ]
        for r in dlin.itertuples():
            rows.append({
                "TF": tf,
                "arm": arm,
                "lineage": r.lineage,
                "r": r.r,
            })

    hh = pd.DataFrame(rows)

    if len(hh):
        mat = hh.pivot(
            index="TF",
            columns="lineage",
            values="r"
        ).reindex(
            index=show_tfs,
            columns=LINEAGES,
        )

        fig, ax = plt.subplots(
            figsize=(7.4, max(4.4, 0.38 * len(mat) + 2))
        )
        im = ax.imshow(
            mat.to_numpy(float),
            aspect="auto",
            vmin=-1,
            vmax=1,
            cmap="RdBu_r",
        )
        ax.set_xticks(np.arange(len(LINEAGES)))
        ax.set_xticklabels(
            [LINEAGE_DISPLAY[x] for x in LINEAGES],
            rotation=35,
            ha="right"
        )
        ax.set_yticks(np.arange(len(mat)))
        ax.set_yticklabels(mat.index)
        ax.set_title(
            "79.2E. Within-lineage residual TF–arm association"
        )
        fig.colorbar(
            im,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            label="r",
        )
        savefig(fig, "79_2_F5_WITHIN_LINEAGE_ROBUSTNESS")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--permutations",
        type=int,
        default=N_PERM,
        help="Permutation/sign-flip iterations; default 10000."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoints and recompute."
    )
    args = parser.parse_args()

    n_perm = int(args.permutations)

    log("=" * 100)
    log("79.2 START — CLOCK17-EXCLUDED + LINEAGE-ADJUSTED TF GATE")
    log(f"permutations/sign-flips = {n_perm}")
    log("=" * 100)

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------
    required = [META, NPZ, NETWORK]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n" + "\n".join(missing)
        )

    six, six_path = load_sixarm()
    net = load_network()

    audit = {
        "project": str(PROJECT),
        "root": str(ROOT),
        "sixarm_file": str(six_path),
        "metadata": str(META),
        "expression_npz": str(NPZ),
        "dorothea_network": str(NETWORK),
        "n_dorothea_edges": len(net),
        "n_dorothea_tfs": int(net["source"].nunique()),
        "clock17": CLOCK17,
        "n_permutations": n_perm,
        "rng_seed": RNG_SEED,
        "focus_tfs": FOCUS_TFS,
        "thresholds": {
            "LEAKAGE_ROBUST_RETENTION": LEAKAGE_ROBUST_RETENTION,
            "LEAKAGE_SENSITIVE_RETENTION": LEAKAGE_SENSITIVE_RETENTION,
            "ADJUSTED_MIN_ABS_R": ADJUSTED_MIN_ABS_R,
            "ADJUSTED_FDR": ADJUSTED_FDR,
            "LINEAGE_SIGN_FRACTION": LINEAGE_SIGN_FRACTION,
        },
        "input_hashes": {
            "metadata_sha256": sha256(META),
            "npz_sha256": sha256(NPZ),
            "network_sha256": sha256(NETWORK),
            "sixarm_sha256": sha256(six_path),
        }
    }
    write_json(AUDIT / "79_2_PROTOCOL_AND_INPUT_AUDIT.json", audit)

    # -------------------------------------------------------------------------
    # Aggregate frozen transcriptome exactly to six-arm observations.
    # -------------------------------------------------------------------------
    agg_cache = CACHE / "79_2_aggregated_expression.npz"
    meta_cache = CACHE / "79_2_aggregated_metadata.csv"
    six_cache = CACHE / "79_2_aligned_sixarm.csv"

    if (
        agg_cache.exists()
        and meta_cache.exists()
        and six_cache.exists()
        and not args.force
    ):
        log("Loading cached aligned expression/six-arm data.")
        z = np.load(agg_cache, allow_pickle=True)
        agg_X = z["X"]
        agg_genes = z["genes"].astype(object)
        meta = pd.read_csv(meta_cache)
        six = pd.read_csv(six_cache)
    else:
        log("Aggregating frozen expression to donor×tissue×lineage.")
        meta, agg_X, agg_genes, six = load_and_aggregate_expression(
            net,
            six,
        )

        np.savez_compressed(
            agg_cache,
            X=agg_X.astype(np.float32),
            genes=np.array(agg_genes, dtype=object),
        )
        meta.to_csv(meta_cache, index=False)
        six.to_csv(six_cache, index=False)

    log(
        f"Aligned observations={len(meta)}, "
        f"donors={meta['donor_id'].nunique()}, "
        f"tissues={meta['tissue'].nunique()}"
    )

    # -------------------------------------------------------------------------
    # TF expression.
    # -------------------------------------------------------------------------
    Pexpr, expr_names = tf_expression_matrix(
        agg_X,
        agg_genes,
        net,
    )
    log(f"TF expression predictors={len(expr_names)}")

    # -------------------------------------------------------------------------
    # Full and Clock17-excluded activity.
    # -------------------------------------------------------------------------
    act_full_cache = CACHE / "79_2_activity_full.npz"
    act_excl_cache = CACHE / "79_2_activity_clock17_excluded.npz"

    if (
        act_full_cache.exists()
        and act_excl_cache.exists()
        and not args.force
    ):
        log("Loading cached TF activity matrices.")
        z = np.load(act_full_cache, allow_pickle=True)
        Afull = z["activity"]
        full_names = z["names"].astype(str).tolist()

        z = np.load(act_excl_cache, allow_pickle=True)
        Aexcl = z["activity"]
        excl_names = z["names"].astype(str).tolist()

        audit_full = pd.read_csv(
            TABLES / "79_2_REGULON_AUDIT_FULL.csv"
        )
        audit_excl = pd.read_csv(
            TABLES / "79_2_REGULON_AUDIT_CLOCK17_EXCLUDED.csv"
        )
    else:
        log("Computing full-regulon TF activities.")
        Afull, full_names, audit_full = compute_ulm_style_activity(
            agg_X,
            agg_genes,
            net,
            exclude_clock_targets=False,
        )
        audit_full.to_csv(
            TABLES / "79_2_REGULON_AUDIT_FULL.csv",
            index=False,
        )
        np.savez_compressed(
            act_full_cache,
            activity=Afull.astype(np.float32),
            names=np.array(full_names, dtype=object),
        )

        log("Computing CLOCK17-excluded TF activities.")
        Aexcl, excl_names, audit_excl = compute_ulm_style_activity(
            agg_X,
            agg_genes,
            net,
            exclude_clock_targets=True,
        )
        audit_excl.to_csv(
            TABLES / "79_2_REGULON_AUDIT_CLOCK17_EXCLUDED.csv",
            index=False,
        )
        np.savez_compressed(
            act_excl_cache,
            activity=Aexcl.astype(np.float32),
            names=np.array(excl_names, dtype=object),
        )

    log(
        f"Full-regulon activity TFs={len(full_names)}; "
        f"Clock17-excluded activity TFs={len(excl_names)}"
    )

    # -------------------------------------------------------------------------
    # Matched association — full activity diagnostic.
    # -------------------------------------------------------------------------
    p_full_match = TABLES / "79_2_ACTIVITY_FULL_MATCHED_ASSOCIATION.csv"

    if p_full_match.exists() and not args.force:
        full_match = pd.read_csv(p_full_match)
        log("Loaded checkpoint: full activity matched association.")
    else:
        log("Running full-regulon matched association diagnostic.")
        full_match = matched_association(
            Afull,
            full_names,
            meta,
            six,
            "full_activity",
            n_perm=n_perm,
        )
        full_match.to_csv(
            p_full_match,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Matched association — Clock17 excluded PRIMARY leakage test.
    # -------------------------------------------------------------------------
    p_excl_match = (
        TABLES
        / "79_2_ACTIVITY_CLOCK17_EXCLUDED_MATCHED_ASSOCIATION.csv"
    )

    if p_excl_match.exists() and not args.force:
        excl_match = pd.read_csv(p_excl_match)
        log("Loaded checkpoint: Clock17-excluded matched association.")
    else:
        log("Running Clock17-excluded matched association.")
        excl_match = matched_association(
            Aexcl,
            excl_names,
            meta,
            six,
            "clock_excluded_activity",
            n_perm=n_perm,
        )
        excl_match.to_csv(
            p_excl_match,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Lineage-adjusted association — activity PRIMARY confounding test.
    # -------------------------------------------------------------------------
    p_excl_adj = (
        TABLES
        / "79_2_ACTIVITY_CLOCK17_EXCLUDED_LINEAGE_ADJUSTED_ASSOCIATION.csv"
    )

    if p_excl_adj.exists() and not args.force:
        excl_adj = pd.read_csv(p_excl_adj)
        log("Loaded checkpoint: lineage-adjusted activity association.")
    else:
        log("Running donor+tissue+lineage-adjusted activity association.")
        excl_adj = lineage_adjusted_association(
            Aexcl,
            excl_names,
            meta,
            six,
            "clock_excluded_activity",
            n_perm=n_perm,
        )
        excl_adj.to_csv(
            p_excl_adj,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Lineage-adjusted expression support.
    # -------------------------------------------------------------------------
    p_expr_adj = (
        TABLES
        / "79_2_TF_EXPRESSION_LINEAGE_ADJUSTED_ASSOCIATION.csv"
    )

    if p_expr_adj.exists() and not args.force:
        expr_adj = pd.read_csv(p_expr_adj)
        log("Loaded checkpoint: lineage-adjusted TF-expression association.")
    else:
        log("Running donor+tissue+lineage-adjusted TF-expression association.")
        expr_adj = lineage_adjusted_association(
            Pexpr,
            expr_names,
            meta,
            six,
            "expression",
            n_perm=n_perm,
        )
        expr_adj.to_csv(
            p_expr_adj,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Within-lineage robustness.
    # -------------------------------------------------------------------------
    p_within = (
        TABLES
        / "79_2_ACTIVITY_CLOCK17_EXCLUDED_WITHIN_LINEAGE_ASSOCIATION.csv"
    )

    if p_within.exists() and not args.force:
        within_lin = pd.read_csv(p_within)
    else:
        log("Computing lineage-stratified residual associations.")
        within_lin = within_lineage_associations(
            Aexcl,
            excl_names,
            meta,
            six,
            "clock_excluded_activity",
        )
        within_lin.to_csv(
            p_within,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Direct clock targets = external biological support only.
    # -------------------------------------------------------------------------
    direct = direct_clock_targets(net)
    direct.to_csv(
        TABLES / "79_2_DOROTHEA_DIRECT_CLOCK_TARGETS_SUPPORT_ONLY.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Previous FIX2 results for continuity, not required.
    # -------------------------------------------------------------------------
    prev_priority = (
        pd.read_csv(PREV_PRIORITY)
        if PREV_PRIORITY.exists()
        else pd.DataFrame()
    )

    # -------------------------------------------------------------------------
    # Final ranking.
    # -------------------------------------------------------------------------
    ranking = build_final_ranking(
        full_match,
        excl_match,
        excl_adj,
        expr_adj,
        within_lin,
        direct,
        prev_priority,
    )

    ranking.to_csv(
        TABLES / "79_2_FINAL_TF_MECHANISM_PRIORITY.csv",
        index=False,
    )

    # Explicit focus table.
    focus = ranking[
        ranking["TF"].isin(FOCUS_TFS)
    ].copy()

    focus["focus_order"] = focus["TF"].map(
        {tf: i for i, tf in enumerate(FOCUS_TFS)}
    )
    focus = focus.sort_values("focus_order").drop(
        columns="focus_order"
    )

    focus.to_csv(
        TABLES / "79_2_FOCUS_SPI1_ERG_PAX5_LEF1.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Leave-one-tissue / donor robustness for top candidate pairs.
    # -------------------------------------------------------------------------
    pair_rows = ranking.head(TOP_N_ROBUST)[
        ["TF", "adj_arm"]
    ].dropna()

    candidate_pairs = [
        (r.TF, r.adj_arm)
        for r in pair_rows.itertuples()
    ]

    log(
        f"Running leave-one-tissue/donor robustness for "
        f"{len(candidate_pairs)} top candidate pairs."
    )

    loo_tissue = leave_one_out_r(
        Aexcl,
        excl_names,
        meta,
        six,
        candidate_pairs,
        "tissue",
    )
    loo_donor = leave_one_out_r(
        Aexcl,
        excl_names,
        meta,
        six,
        candidate_pairs,
        "donor_id",
    )

    loo = pd.concat(
        [loo_tissue, loo_donor],
        ignore_index=True,
    )
    loo.to_csv(
        TABLES / "79_2_TOP_TF_LEAVE_ONE_BLOCK_ROBUSTNESS.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Two primary question summaries.
    # -------------------------------------------------------------------------
    leakage_summary = []

    for tf in sorted(set(excl_names).intersection(full_names)):
        # Use CLOCK-EXCLUDED best matched arm as the non-leaky target.
        d = ranking[ranking["TF"] == tf]
        if d.empty:
            continue

        r = d.iloc[0]
        leakage_summary.append({
            "TF": tf,
            "best_clock_excluded_arm": r["excl_match_arm"],
            "full_same_arm_r": r["full_activity_same_arm_r"],
            "clock_excluded_same_arm_r": r["clock_excluded_same_arm_r"],
            "retention": r["clock_exclusion_retention"],
            "status": r["clock_leakage_status"],
        })

    leakage_summary = pd.DataFrame(leakage_summary)
    leakage_summary.to_csv(
        TABLES / "79_2_Q1_CLOCK17_EXCLUSION_SUMMARY.csv",
        index=False,
    )

    conf_summary = ranking[
        [
            "TF",
            "excl_match_arm",
            "excl_match_matched_r",
            "adj_arm",
            "adj_adjusted_r",
            "adj_perm_P",
            "adj_perm_FDR",
            "adj_maxT_FWER",
            "persists_after_lineage_adjustment",
            "adjusted_same_sign_lineage_fraction",
            "mechanism_tier",
        ]
    ].copy()

    conf_summary.to_csv(
        TABLES / "79_2_Q2_LINEAGE_ADJUSTMENT_SUMMARY.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figures.
    # -------------------------------------------------------------------------
    make_figures(
        full_match,
        excl_match,
        excl_adj,
        ranking,
        within_lin,
    )

    # -------------------------------------------------------------------------
    # Final gate.
    # -------------------------------------------------------------------------
    tier1 = ranking[
        ranking["mechanism_tier"].eq(
            "TIER1_EXPERIMENT_READY"
        )
    ].copy()

    tier2 = ranking[
        ranking["mechanism_tier"].eq(
            "TIER2_CONTEXT_SPECIFIC"
        )
    ].copy()

    focus_tier1 = tier1[
        tier1["TF"].isin(FOCUS_TFS)
    ].copy()

    if len(focus_tier1) >= 1:
        gate = "PASS_FOCUS_TF"
        start_experiment = True
    elif len(tier1) >= 1:
        gate = "PASS_OTHER_TF"
        start_experiment = True
    elif len(tier2) >= 1:
        gate = "PARTIAL_CONTEXT_SPECIFIC"
        start_experiment = False
    else:
        gate = "NO_TF_PASSES_BOTH_GATES"
        start_experiment = False

    verdict = {
        "status": "COMPLETE",
        "TF_MECHANISM_GATE": gate,
        "START_TF_PERTURBATION_EXPERIMENT": start_experiment,
        "n_TIER1_EXPERIMENT_READY": int(len(tier1)),
        "n_TIER2_CONTEXT_SPECIFIC": int(len(tier2)),
        "focus_TIER1": focus_tier1["TF"].tolist(),
        "top_TIER1": tier1["TF"].head(10).tolist(),
        "n_permutations": n_perm,
        "clock17_excluded_from_activity": True,
        "lineage_main_effect_removed_in_primary_adjusted_test": True,
    }
    write_json(
        OUT / "79_2_COMPLETE.json",
        verdict,
    )

    # -------------------------------------------------------------------------
    # Report.
    # -------------------------------------------------------------------------
    report_lines = [
        "# 79.2 — Clock17-excluded and lineage-adjusted TF mechanism gate",
        "",
        "## Why this analysis was run",
        "",
        "79 FIX2 identified strong TF–six-arm associations, but two alternative "
        "explanations remained: (1) direct mathematical leakage because DoRothEA "
        "activity can include the same Clock17 genes used to build the six-arm "
        "outcome; and (2) lineage identity can jointly drive both TF state and "
        "clock architecture. 79.2 tests both problems directly.",
        "",
        "## Question 1 — Clock17 leakage",
        "",
        "Every one of the 17 core-clock genes was removed from every DoRothEA "
        "regulon before TF activity was recomputed. Direct TF→Clock17 edges are "
        "retained only as external biological-support annotations and are not "
        "used to calculate the Clock17-excluded activity.",
        "",
        "Top leakage audit:",
        "",
        markdown_table(
            leakage_summary.sort_values(
                "retention",
                ascending=False
            ),
            max_rows=25,
        ),
        "",
        "## Question 2 — lineage confounding",
        "",
        "Both Clock17-excluded TF activity and each six-arm score were residualized "
        "against additive donor, tissue and lineage fixed effects. Association "
        "was then tested on residuals. P values use donor-cluster sign-flip "
        f"inference with {n_perm} iterations; BH FDR and maxT family-wise P are reported.",
        "",
        "Top final candidates:",
        "",
        markdown_table(
            ranking[
                [
                    "TF",
                    "excl_match_arm",
                    "excl_match_matched_r",
                    "clock_exclusion_retention",
                    "clock_leakage_status",
                    "adj_arm",
                    "adj_adjusted_r",
                    "adj_perm_FDR",
                    "adj_maxT_FWER",
                    "adjusted_same_sign_lineage_fraction",
                    "n_clock_targets",
                    "clock_targets",
                    "mechanism_tier",
                ]
            ],
            max_rows=30,
        ),
        "",
        "## Prespecified focus candidates",
        "",
        markdown_table(
            focus[
                [
                    "TF",
                    "excl_match_arm",
                    "excl_match_matched_r",
                    "clock_exclusion_retention",
                    "clock_leakage_status",
                    "adj_arm",
                    "adj_adjusted_r",
                    "adj_perm_FDR",
                    "adjusted_same_sign_lineage_fraction",
                    "n_clock_targets",
                    "clock_targets",
                    "mechanism_tier",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## Final gate",
        "",
        f"- TF_MECHANISM_GATE = **{gate}**",
        f"- START_TF_PERTURBATION_EXPERIMENT = **{str(start_experiment).upper()}**",
        f"- Tier 1 count = {len(tier1)}",
        f"- Focus Tier 1 = {', '.join(focus_tier1['TF'].tolist()) if len(focus_tier1) else 'NONE'}",
        "",
        "## Interpretation boundary",
        "",
        "A Tier-1 result means the association is robust to direct Clock17 leakage, "
        "persists after donor+tissue+lineage adjustment, and shows cross-lineage "
        "directional robustness. It is therefore a strong perturbation candidate. "
        "It still does not establish causality without experimental perturbation "
        "and, ideally, orthogonal chromatin/occupancy evidence.",
    ]

    (REPORT / "79_2_FINAL_REPORT.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final console.
    # -------------------------------------------------------------------------
    log("=" * 100)
    log("79.2 COMPLETE")
    log(f"TF_MECHANISM_GATE = {gate}")
    log(
        "START_TF_PERTURBATION_EXPERIMENT = "
        f"{str(start_experiment).upper()}"
    )
    log(f"N_TIER1 = {len(tier1)}")
    log(
        "FOCUS_TIER1 = "
        + (
            ",".join(focus_tier1["TF"].tolist())
            if len(focus_tier1)
            else "NONE"
        )
    )
    log(
        "TOP_TIER1 = "
        + (
            ",".join(tier1["TF"].head(10).tolist())
            if len(tier1)
            else "NONE"
        )
    )
    log(
        f"REPORT = {REPORT / '79_2_FINAL_REPORT.md'}"
    )
    log(
        "PRIORITY = "
        f"{TABLES / '79_2_FINAL_TF_MECHANISM_PRIORITY.csv'}"
    )
    log("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()

        (OUT / "79_2_FAILED.txt").write_text(
            tb,
            encoding="utf-8",
        )

        write_json(
            OUT / "79_2_FAILED.json",
            {
                "status": "FAILED",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "traceback": tb,
            },
        )

        print(tb, flush=True)
        raise
