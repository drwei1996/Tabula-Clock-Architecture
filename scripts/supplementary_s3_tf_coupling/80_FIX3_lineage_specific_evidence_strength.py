#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
80_FIX3_lineage_specific_evidence_strength.py

Purpose
-------
Reframe Analysis 80 around the actual biological hypothesis:

    Different lineages are associated with different candidate TF programs
    that may help establish distinct core-clock architectures.

This script does NOT search for one universal TF.

It integrates, per lineage:
1) 79.2 within-lineage TF–clock-arm residual association
2) ChIP-Atlas binding evidence to the candidate's predicted clock arm
3) KnockTF perturbation evidence (kept as GLOBAL external support unless
   lineage/context-matched metadata are available; never silently assigned
   to a lineage)

Outputs:
- lineage-specific candidate tables
- evidence strength grades
- per-lineage top candidates
- publication-oriented heatmaps
- explicit separation of:
    lineage evidence
    binding evidence
    global perturbational evidence

No wet-lab requirement is introduced.
"""

from __future__ import annotations
import os

import argparse
import json
import math
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

SRC79 = ROOT / "79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_20261004"
FIX2 = ROOT / "80_FIX2_EXTERNAL_COMPUTATIONAL_TF_VALIDATION_20261004"

INTEGRATED_FIX2 = (
    FIX2 / "04_INTEGRATED" / "80_FIX2_FINAL_COMPUTATIONAL_REGULATORY_CANDIDATES.csv"
)
CHIP_SUMMARY = (
    FIX2 / "02_BINDING" / "80_FIX2_CHIPATLAS_CLOCK17_BINDING_SUMMARY.csv"
)
KNOCK_SUMMARY = (
    FIX2 / "03_KNOCKTF" / "80_FIX2_KNOCKTF_CANDIDATE_SUMMARY.csv"
)
KNOCK_PROFILE = (
    FIX2 / "03_KNOCKTF" / "80_FIX2_KNOCKTF_PROFILE_LEVEL.csv"
)

OUT = ROOT / "80_FIX3_LINEAGE_SPECIFIC_EVIDENCE_STRENGTH_20261005"
AUDIT = OUT / "00_AUDIT"
TABLE = OUT / "01_TABLES"
FIG = OUT / "02_FIGURES"
REPORT = OUT / "03_REPORT"

for d in [OUT, AUDIT, TABLE, FIG, REPORT]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BIOLOGY / FROZEN DEFINITIONS
# =============================================================================

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

ARM_GENES = {
    "EBOX_POS": ["ARNTL", "CLOCK", "NPAS2"],
    "EBOX_NEG": ["PER1", "PER2", "PER3", "CRY1", "CRY2"],
    "RORE_ACT": ["RORA", "RORB", "RORC"],
    "RORE_REP": ["NR1D1", "NR1D2"],
    "DBOX_ACT": ["DBP", "TEF", "HLF"],
    "DBOX_REP": ["NFIL3"],
}
ARMS = list(ARM_GENES)


# =============================================================================
# HELPERS
# =============================================================================

def safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def normalize_lineage(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = re.sub(r"[\s/\-]+", "_", s)
    if s in LINEAGES:
        return s
    if "macrophage" in s or "monocyte" in s or "myeloid" in s:
        return "macrophage_monocyte"
    if "fibro" in s:
        return "fibroblast"
    if "endothel" in s:
        return "endothelial"
    if "b_cell" in s or s in {"b", "bcell", "b_cells"}:
        return "b_cell"
    if "t_cell" in s or s in {"t", "tcell", "t_cells"}:
        return "t_cell"
    return None


def savefig(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kw = dict(bbox_inches="tight", pad_inches=0.08, facecolor="white")
        if ext == "png":
            kw["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kw)
    plt.close(fig)


def md_table(df, n=40):
    if df is None or df.empty:
        return "_No rows._"
    x = df.head(n)
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4g}"
        return str(v).replace("|", r"\|").replace("\n", " ")
    out = [
        "| " + " | ".join(map(str, x.columns)) + " |",
        "| " + " | ".join(["---"] * len(x.columns)) + " |",
    ]
    for _, r in x.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in x.columns) + " |")
    return "\n".join(out)


# =============================================================================
# DISCOVER 79.2 WITHIN-LINEAGE SOURCE
# =============================================================================

def score_within_lineage_candidate(path: Path) -> tuple[int, dict]:
    d = safe_csv(path)
    if d.empty:
        return -1, {}
    cols = {c.lower(): c for c in d.columns}

    tf_col = next((cols[k] for k in ["tf", "source", "factor"] if k in cols), None)
    lin_col = next((cols[k] for k in ["lineage", "cell_lineage", "priority_lineage"] if k in cols), None)

    r_col = None
    for c in d.columns:
        cl = c.lower()
        if cl in {"r", "corr", "correlation", "adjusted_r", "residual_r"} or cl.endswith("_r"):
            r_col = c
            break

    arm_col = next((c for c in d.columns if c.lower() in {"arm", "clock_arm", "predicted_arm"}), None)

    score = 0
    if tf_col: score += 10
    if lin_col: score += 10
    if r_col: score += 10
    if arm_col: score += 3
    if tf_col and lin_col:
        lins = d[lin_col].map(normalize_lineage)
        score += int(lins.notna().sum() > 0) * 10
        score += min(10, int(lins.nunique(dropna=True)) * 2)

    # Filename hints.
    name = path.name.lower()
    if "within" in name and "lineage" in name:
        score += 15
    if "residual" in name:
        score += 5
    if "79_2" in name or "79.2" in name:
        score += 3

    return score, {
        "tf_col": tf_col,
        "lineage_col": lin_col,
        "r_col": r_col,
        "arm_col": arm_col,
        "n_rows": len(d),
    }


def discover_within_lineage_file():
    candidates = []
    for p in SRC79.rglob("*.csv"):
        sc, meta = score_within_lineage_candidate(p)
        if sc >= 0:
            candidates.append((sc, p, meta))

    if not candidates:
        raise FileNotFoundError("No CSV files found under 79.2 directory.")

    candidates.sort(key=lambda x: (x[0], x[2].get("n_rows", 0)), reverse=True)

    audit_rows = []
    for sc, p, meta in candidates:
        audit_rows.append({
            "score": sc,
            "path": str(p),
            **meta,
        })
    pd.DataFrame(audit_rows).to_csv(
        AUDIT / "80_FIX3_WITHIN_LINEAGE_SOURCE_DISCOVERY.csv",
        index=False
    )

    best_sc, best_path, best_meta = candidates[0]
    if best_sc < 35:
        raise RuntimeError(
            "Could not confidently identify a 79.2 within-lineage TF association table. "
            "See 00_AUDIT/80_FIX3_WITHIN_LINEAGE_SOURCE_DISCOVERY.csv"
        )

    return best_path, best_meta


def load_within_lineage():
    path, meta = discover_within_lineage_file()
    d = pd.read_csv(path)

    tf_col = meta["tf_col"]
    lin_col = meta["lineage_col"]
    r_col = meta["r_col"]
    arm_col = meta["arm_col"]

    x = pd.DataFrame({
        "TF": d[tf_col].astype(str).str.upper(),
        "lineage": d[lin_col].map(normalize_lineage),
        "within_lineage_r": pd.to_numeric(d[r_col], errors="coerce"),
    })
    if arm_col:
        x["arm"] = d[arm_col].astype(str)
    else:
        x["arm"] = np.nan

    # Keep only the five lineages.
    x = x[x["lineage"].isin(LINEAGES) & x["within_lineage_r"].notna()].copy()

    # If duplicated TF-lineage rows exist (e.g., multiple arms), keep the row with
    # the largest absolute association, preserving its sign.
    x["_abs"] = x["within_lineage_r"].abs()
    x = (
        x.sort_values("_abs", ascending=False)
         .drop_duplicates(["TF", "lineage"])
         .drop(columns="_abs")
         .reset_index(drop=True)
    )

    x.to_csv(TABLE / "80_FIX3_WITHIN_LINEAGE_ASSOCIATIONS.csv", index=False)

    return x, path


# =============================================================================
# EVIDENCE GRADES
# =============================================================================

def binding_grade_for_tf(tf, predicted_arm, chip):
    """
    B0: not evaluable
    B1: evaluable, no predicted-arm target
    B2: >=1 predicted-arm target within 10 kb
    B3: >=1 predicted-arm target within 5 kb
    B4: >=1 predicted-arm target within 1 kb OR >=2 predicted-arm targets within 5 kb
    """
    if chip.empty or "TF" not in chip.columns:
        return "B0", 0, ""

    c = chip[chip["TF"].astype(str).str.upper().eq(tf)].copy()
    if c.empty:
        return "B0", 0, ""

    if "status" in c.columns:
        c_ok = c[c["status"].astype(str).str.upper().eq("OK")].copy()
        if c_ok.empty:
            return "B0", 0, ""
        c = c_ok

    def get_hits(dist):
        z = c[pd.to_numeric(c.get("distance_kb", np.nan), errors="coerce").eq(dist)].copy()
        if z.empty:
            return 0, []
        n = int(pd.to_numeric(z.get("n_predicted_arm_bound", 0), errors="coerce").fillna(0).max())
        tg = []
        if "predicted_arm_targets" in z.columns:
            tg = sorted({
                g for s in z["predicted_arm_targets"].fillna("").astype(str)
                for g in s.split(",") if g
            })
        return n, tg

    n1, g1 = get_hits(1)
    n5, g5 = get_hits(5)
    n10, g10 = get_hits(10)

    if n1 >= 1 or n5 >= 2:
        return "B4", max(n1, n5), ",".join(sorted(set(g1 + g5)))
    if n5 >= 1:
        return "B3", n5, ",".join(g5)
    if n10 >= 1:
        return "B2", n10, ",".join(g10)
    return "B1", 0, ""


def perturb_grade(row):
    """
    P0: not evaluable / no profile
    P1: n=1 and direction-consistent
    P2: n>=2 and >=60% direction-consistent
    P3: n>=3 and >=75% direction-consistent
    P4: n>=3, >=75% direction-consistent AND binomial p<=0.05

    Note:
    This is GLOBAL public perturbation support unless a lineage-matched
    perturbation context is explicitly available. It is NOT silently
    assigned as lineage-specific causal evidence.
    """
    n = row.get("n_profiles", np.nan)
    frac = row.get("direction_match_fraction", np.nan)
    p = row.get("binom_P", np.nan)
    ev = row.get("perturbation_evaluable", False)

    try:
        n = int(n)
    except Exception:
        n = 0
    try:
        frac = float(frac)
    except Exception:
        frac = np.nan
    try:
        p = float(p)
    except Exception:
        p = np.nan

    if not bool(ev) or n <= 0:
        return "P0"

    if n >= 3 and np.isfinite(frac) and frac >= 0.75 and np.isfinite(p) and p <= 0.05:
        return "P4"
    if n >= 3 and np.isfinite(frac) and frac >= 0.75:
        return "P3"
    if n >= 2 and np.isfinite(frac) and frac >= 0.60:
        return "P2"
    if n == 1 and np.isfinite(frac) and frac >= 1.0:
        return "P1"
    return "P0"


def lineage_assoc_grade(r):
    """
    L0: |r| < 0.10
    L1: 0.10 <= |r| < 0.20
    L2: 0.20 <= |r| < 0.30
    L3: 0.30 <= |r| < 0.40
    L4: |r| >= 0.40

    This is a descriptive strength grade, not a p-value.
    """
    a = abs(float(r))
    if a >= 0.40: return "L4"
    if a >= 0.30: return "L3"
    if a >= 0.20: return "L2"
    if a >= 0.10: return "L1"
    return "L0"


def grade_num(g):
    try:
        return int(str(g)[1:])
    except Exception:
        return 0


# =============================================================================
# INTEGRATION
# =============================================================================

def build_lineage_table():
    fix2 = pd.read_csv(INTEGRATED_FIX2)
    fix2["TF"] = fix2["TF"].astype(str).str.upper()

    chip = safe_csv(CHIP_SUMMARY)
    knock = safe_csv(KNOCK_SUMMARY)
    within, within_source = load_within_lineage()

    if not knock.empty:
        knock["TF"] = knock["TF"].astype(str).str.upper()

    # Merge the 79.2 within-lineage associations with FIX2 candidate metadata.
    cols_keep = [
        c for c in [
            "TF",
            "tabula_predicted_arm",
            "tabula_r",
            "mechanism_evidence_score",
            "computational_candidate_tier",
            "binding_status",
            "perturbation_status",
            "n_profiles",
            "direction_match_fraction",
            "binom_P",
            "perturbation_evaluable",
        ]
        if c in fix2.columns
    ]

    x = within.merge(fix2[cols_keep], on="TF", how="inner")

    if not knock.empty:
        nk_cols = [
            c for c in [
                "TF",
                "n_profiles",
                "direction_match_fraction",
                "binom_P",
                "perturbation_evaluable",
                "perturbation_status",
            ]
            if c in knock.columns
        ]
        # Prefer dedicated KnockTF summary columns if present.
        nk = knock[nk_cols].drop_duplicates("TF")
        for c in nk_cols:
            if c == "TF":
                continue
            if c in x.columns:
                x = x.drop(columns=c)
        x = x.merge(nk, on="TF", how="left")

    # Prefer 79.2 arm when available; otherwise use frozen FIX2 predicted arm.
    x["predicted_arm"] = x["arm"]
    bad = ~x["predicted_arm"].isin(ARMS)
    x.loc[bad, "predicted_arm"] = x.loc[bad, "tabula_predicted_arm"]

    # Grades.
    bind_grades = []
    bind_n = []
    bind_targets = []
    for _, r in x.iterrows():
        g, n, tg = binding_grade_for_tf(r["TF"], r["predicted_arm"], chip)
        bind_grades.append(g)
        bind_n.append(n)
        bind_targets.append(tg)

    x["lineage_assoc_grade"] = x["within_lineage_r"].map(lineage_assoc_grade)
    x["binding_grade"] = bind_grades
    x["binding_n_targets"] = bind_n
    x["binding_targets"] = bind_targets
    x["perturb_grade_global"] = x.apply(perturb_grade, axis=1)

    x["L_score"] = x["lineage_assoc_grade"].map(grade_num)
    x["B_score"] = x["binding_grade"].map(grade_num)
    x["P_score"] = x["perturb_grade_global"].map(grade_num)

    # Primary ranking is lineage-specific evidence first, then binding, then global perturbation.
    # DO NOT let a global perturbation profile outweigh lineage-specific association.
    x["lineage_priority_score"] = (
        3 * x["L_score"]
        + 2 * x["B_score"]
        + 1 * x["P_score"]
    )

    # Classify within each lineage.
    tiers = []
    for _, r in x.iterrows():
        L = int(r["L_score"])
        B = int(r["B_score"])
        P = int(r["P_score"])

        if L >= 3 and B >= 3 and P >= 2:
            t = "TIER_A_LINEAGE_TF_STRONG_MULTI_EVIDENCE"
        elif L >= 3 and B >= 2:
            t = "TIER_B_LINEAGE_TF_BINDING_SUPPORTED"
        elif L >= 2 and (B >= 2 or P >= 1):
            t = "TIER_C_LINEAGE_TF_CONVERGENT_SUPPORT"
        elif L >= 1:
            t = "TIER_D_LINEAGE_ASSOCIATION_ONLY"
        else:
            t = "TIER_E_WEAK_LINEAGE_ASSOCIATION"
        tiers.append(t)

    x["lineage_specific_tier"] = tiers

    # Rank separately WITHIN EACH LINEAGE.
    x["rank_within_lineage"] = (
        x.groupby("lineage")["lineage_priority_score"]
         .rank(method="first", ascending=False)
         .astype(int)
    )

    x = x.sort_values(
        ["lineage", "rank_within_lineage"]
    ).reset_index(drop=True)

    # Critical interpretive flag.
    x["perturbation_scope"] = "GLOBAL_PUBLIC_PERTURBATION_SUPPORT"
    x["do_not_interpret_as_universal_TF"] = True

    x.to_csv(TABLE / "80_FIX3_LINEAGE_SPECIFIC_EVIDENCE_STRENGTH.csv", index=False)

    # Top candidates per lineage.
    top = x[x["rank_within_lineage"] <= 8].copy()
    top.to_csv(TABLE / "80_FIX3_TOP8_PER_LINEAGE.csv", index=False)

    # A compact top-3 summary.
    top3 = x[x["rank_within_lineage"] <= 3].copy()
    top3.to_csv(TABLE / "80_FIX3_TOP3_PER_LINEAGE.csv", index=False)

    write_audit = {
        "within_lineage_source": str(within_source),
        "n_lineage_candidate_rows": len(x),
        "n_unique_TFs": int(x["TF"].nunique()),
        "lineages": sorted(x["lineage"].unique().tolist()),
        "ranking_principle": "rank separately within lineage; lineage evidence weighted above binding and global perturbation",
        "score_formula": "3*L + 2*B + 1*P",
        "perturbation_scope": "global public perturbation support unless context-matched evidence is explicitly available",
        "universal_tf_search": False,
    }
    (AUDIT / "80_FIX3_ANALYSIS_AUDIT.json").write_text(
        json.dumps(write_audit, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return x, top3


# =============================================================================
# FIGURES
# =============================================================================

def make_figures(x):
    # F1: lineage x TF association heatmap for union of top 5 per lineage.
    top_union = (
        x[x["rank_within_lineage"] <= 5]["TF"]
        .drop_duplicates()
        .tolist()
    )
    piv = x[x["TF"].isin(top_union)].pivot_table(
        index="TF", columns="lineage", values="within_lineage_r", aggfunc="first"
    )
    piv = piv.reindex(columns=LINEAGES)

    # Sort by strongest lineage-specific signal.
    piv["_mx"] = piv.abs().max(axis=1)
    piv = piv.sort_values("_mx", ascending=False).drop(columns="_mx")

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.34 * len(piv) + 1.8)))
    im = ax.imshow(piv.to_numpy(float), aspect="auto", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(LINEAGES)))
    ax.set_xticklabels([LINEAGE_LABEL[l] for l in LINEAGES], rotation=25, ha="right")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    ax.set_title("80 FIX3A. Lineage-specific residual TF–clock-arm associations")
    fig.colorbar(im, ax=ax, fraction=.028, pad=.02, label="within-lineage residual r")
    savefig(fig, "80_FIX3_F1_LINEAGE_TF_HEATMAP")

    # F2: evidence grades for top 3 per lineage.
    top3 = x[x["rank_within_lineage"] <= 3].copy()
    top3["label"] = top3["lineage"].map(LINEAGE_LABEL) + " | " + top3["TF"]

    M = top3[["L_score", "B_score", "P_score"]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.5, max(5.0, 0.34 * len(top3) + 2)))
    im = ax.imshow(M, aspect="auto", vmin=0, vmax=4)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Lineage association", "Binding", "Global perturbation"])
    ax.set_yticks(range(len(top3)))
    ax.set_yticklabels(top3["label"])
    ax.set_title("80 FIX3B. Evidence strength of top TFs within each lineage")
    fig.colorbar(im, ax=ax, fraction=.028, pad=.02, label="evidence grade (0–4)")
    savefig(fig, "80_FIX3_F2_TOP_LINEAGE_EVIDENCE")

    # F3-F7: one figure per lineage, top 8.
    for lin in LINEAGES:
        z = x[(x["lineage"] == lin) & (x["rank_within_lineage"] <= 8)].copy()
        if z.empty:
            continue
        z = z.sort_values("rank_within_lineage", ascending=True)
        labels = z["TF"].tolist()
        y = np.arange(len(z))

        fig, ax = plt.subplots(figsize=(7.6, max(4.6, 0.48 * len(z) + 1.6)))
        ax.scatter(z["within_lineage_r"], y, s=70)
        ax.axvline(0, lw=.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Within-lineage residual TF–clock-arm r")
        ax.set_title(f"80 FIX3. {LINEAGE_LABEL[lin]} candidate TF architecture")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for yi, (_, rr) in enumerate(z.iterrows()):
            txt = f"{rr['binding_grade']} / {rr['perturb_grade_global']}"
            xloc = rr["within_lineage_r"]
            ax.annotate(
                txt,
                xy=(xloc, yi),
                xytext=(8 if xloc >= 0 else -8, 0),
                textcoords="offset points",
                va="center",
                ha="left" if xloc >= 0 else "right",
                fontsize=8,
            )

        savefig(fig, f"80_FIX3_LINEAGE_{lin}")

    # F8: top TF identity is different across lineages (ranked heatmap).
    rank_piv = x.pivot_table(
        index="TF", columns="lineage", values="rank_within_lineage", aggfunc="first"
    ).reindex(columns=LINEAGES)

    union = x[x["rank_within_lineage"] <= 5]["TF"].drop_duplicates()
    rank_piv = rank_piv.loc[rank_piv.index.intersection(union)].copy()
    rank_plot = rank_piv.copy()
    rank_plot = 6 - rank_plot.clip(upper=6)  # higher = better rank
    rank_plot = rank_plot.fillna(0)

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.34 * len(rank_plot) + 1.8)))
    im = ax.imshow(rank_plot.to_numpy(float), aspect="auto", vmin=0, vmax=5)
    ax.set_xticks(range(len(LINEAGES)))
    ax.set_xticklabels([LINEAGE_LABEL[l] for l in LINEAGES], rotation=25, ha="right")
    ax.set_yticks(range(len(rank_plot)))
    ax.set_yticklabels(rank_plot.index)
    ax.set_title("80 FIX3C. Different lineages prioritize different candidate TFs")
    fig.colorbar(im, ax=ax, fraction=.028, pad=.02, label="within-lineage rank strength")
    savefig(fig, "80_FIX3_F8_DIFFERENT_TFS_BY_LINEAGE")


# =============================================================================
# REPORT
# =============================================================================

def write_report(x, top3):
    lines = [
        "# Analysis 80 FIX3 — lineage-specific TF evidence strength",
        "",
        "## Biological question",
        "",
        "This analysis explicitly does **not** search for one universal TF.",
        "",
        "The tested model is:",
        "",
        "`cell lineage -> lineage-associated TF program -> lineage-specific core-clock architecture`",
        "",
        "Each lineage is ranked independently.",
        "",
        "## Evidence components",
        "",
        "- **L0-L4**: within-lineage residual TF–clock-arm association strength.",
        "- **B0-B4**: public TF-binding evidence at predicted clock-arm genes.",
        "- **P0-P4**: public TF-loss perturbation consistency.",
        "",
        "Perturbation evidence is retained as **global external support** unless the public profile "
        "is explicitly lineage/context matched. It is not silently treated as lineage-specific evidence.",
        "",
        "## Ranking rule",
        "",
        "`lineage_priority_score = 3*L + 2*B + 1*P`",
        "",
        "This intentionally weights lineage-specific association more strongly than global external perturbation.",
        "",
        "## Top three candidates per lineage",
        "",
        md_table(
            top3[[
                "lineage", "rank_within_lineage", "TF",
                "predicted_arm", "within_lineage_r",
                "lineage_assoc_grade", "binding_grade",
                "perturb_grade_global", "lineage_specific_tier"
            ]],
            30
        ),
        "",
        "## Interpretation",
        "",
        "The intended conclusion is not that one TF controls the clock in every lineage. "
        "Instead, different lineages can converge on the same core circadian network through "
        "different regulatory TF programs. A TF can therefore be highly prioritized in one "
        "lineage and weak or absent in another without constituting a contradiction.",
        "",
        "## Claim boundary",
        "",
        "These remain computationally supported candidate regulatory programs. "
        "Binding and perturbation resources strengthen the interpretation but do not establish experimental causality.",
    ]
    (REPORT / "80_FIX3_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    verdict = {
        "status": "COMPLETE",
        "universal_TF_search": False,
        "main_hypothesis": "different lineages are associated with different candidate TF programs linked to distinct core-clock architectures",
        "n_lineages": int(x["lineage"].nunique()),
        "n_unique_TFs": int(x["TF"].nunique()),
        "top3_per_lineage": {
            lin: top3[top3["lineage"] == lin]["TF"].tolist()
            for lin in LINEAGES
        },
        "wet_lab_required": False,
    }
    (OUT / "80_FIX3_COMPLETE.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not INTEGRATED_FIX2.exists():
        raise FileNotFoundError(INTEGRATED_FIX2)

    x, top3 = build_lineage_table()
    make_figures(x)
    write_report(x, top3)

    print("=" * 92)
    print("80 FIX3 LINEAGE-SPECIFIC EVIDENCE STRENGTH COMPLETE")
    print("=" * 92)
    print("UNIVERSAL_TF_SEARCH = FALSE")
    print("LINEAGE_SPECIFIC_RANKING = TRUE")
    print("LINEAGES =", ", ".join(LINEAGES))
    print("N_UNIQUE_TFS =", x["TF"].nunique())
    print()
    for lin in LINEAGES:
        z = top3[top3["lineage"] == lin].sort_values("rank_within_lineage")
        vals = [f"{r.TF}({r.lineage_assoc_grade}/{r.binding_grade}/{r.perturb_grade_global})"
                for _, r in z.iterrows()]
        print(f"{LINEAGE_LABEL[lin]} TOP3 =", ", ".join(vals))
    print()
    print("PERTURBATION_SCOPE = GLOBAL_EXTERNAL_SUPPORT_UNLESS_CONTEXT_MATCHED")
    print("WET_LAB_REQUIRED = FALSE")
    print("REPORT =", REPORT / "80_FIX3_FINAL_REPORT.md")
    print("FINAL_TABLE =", TABLE / "80_FIX3_LINEAGE_SPECIFIC_EVIDENCE_STRENGTH.csv")
    print("=" * 92)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        (OUT / "80_FIX3_FAILED.txt").write_text(tb, encoding="utf-8")
        print(tb)
        raise
