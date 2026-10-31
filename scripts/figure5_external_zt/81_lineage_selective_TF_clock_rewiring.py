#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
81_lineage_selective_TF_clock_rewiring.py

Primary question
----------------
Which TF–clock-arm relationships are truly lineage-selective, rather than merely
strong in one lineage?

This analysis deliberately fixes the clock arm for each TF to the frozen
Analysis-79/80 prediction and compares the SAME TF–SAME ARM relationship across
B cell, T cell, macrophage/monocyte, fibroblast and endothelial lineages.

Core outputs
------------
1) Same-arm TF×lineage residual-r matrix.
2) Per-TF heterogeneity across lineages.
3) Per-lineage specificity contrasts versus the other four lineages.
4) Robust sign-switch detection.
5) Classification:
      SIGN_SWITCHING
      LINEAGE_SELECTIVE
      PAN_LINEAGE_POSITIVE
      PAN_LINEAGE_NEGATIVE
      SHARED_SIGN_HETEROGENEOUS
      WEAK_OR_UNSTABLE
6) Binding/perturbation grades are carried only as external annotations.
   They do NOT determine lineage specificity.

Formal inference
----------------
If per-lineage sample size n is available in the 79.2 source:
- Fisher-r-to-z heterogeneity test across lineages.
- Fisher-z contrast of one lineage vs inverse-variance weighted mean of the
  other lineages.
- Pairwise Fisher-z tests for robust opposite-sign pairs.
- BH correction.

If n is unavailable:
- descriptive rewiring metrics are still produced;
- inferential q-values are NA;
- the report explicitly says formal selectivity inference was not evaluable.

No wet-lab work is introduced.
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
from scipy import stats

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
FIX2_FINAL = (
    FIX2 / "04_INTEGRATED" /
    "80_FIX2_FINAL_COMPUTATIONAL_REGULATORY_CANDIDATES.csv"
)

FIX3 = ROOT / "80_FIX3_LINEAGE_SPECIFIC_EVIDENCE_STRENGTH_20261005"
FIX3_TABLE = (
    FIX3 / "01_TABLES" /
    "80_FIX3_LINEAGE_SPECIFIC_EVIDENCE_STRENGTH.csv"
)

OUT = ROOT / "81_LINEAGE_SELECTIVE_TF_CLOCK_REWIRING_20261005"
AUDIT = OUT / "00_AUDIT"
TABLE = OUT / "01_TABLES"
FIG = OUT / "02_FIGURES"
REPORT = OUT / "03_REPORT"

for d in [OUT, AUDIT, TABLE, FIG, REPORT]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FROZEN BIOLOGICAL DEFINITIONS
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

# Pre-registered descriptive thresholds.
MIN_ABS_R = 0.20
SIGN_SWITCH_MIN_ABS_R = 0.20
LINEAGE_SIGNED_SPEC_DELTA = 0.25
LINEAGE_MAG_SPEC_DELTA = 0.15
PAN_MIN_ABS_R = 0.20
PAN_MIN_SAME_SIGN = 4
PAN_MAX_RANGE = 0.35

# Inferential gates.
HET_Q_GATE = 0.05
CONTRAST_Q_GATE = 0.05
PAIRWISE_Q_GATE = 0.05

EPS = 1e-7


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


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    if len(x) == 0:
        return out
    order = np.argsort(x)
    xx = x[order]
    qq = xx * len(xx) / np.arange(1, len(xx) + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    qq = np.minimum(qq, 1.0)
    back = np.empty(len(xx))
    back[order] = qq
    out[ok] = back
    return out


def fisher_z(r):
    return np.arctanh(np.clip(float(r), -1 + EPS, 1 - EPS))


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
    d = df.head(n)
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4g}"
        return str(v).replace("|", r"\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(map(str, d.columns)) + " |",
        "| " + " | ".join(["---"] * len(d.columns)) + " |",
    ]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in d.columns) + " |")
    return "\n".join(lines)


# =============================================================================
# FROZEN TF -> ARM
# =============================================================================

def load_frozen_tf_arm():
    if not FIX2_FINAL.exists():
        raise FileNotFoundError(FIX2_FINAL)

    d = pd.read_csv(FIX2_FINAL)
    if "TF" not in d.columns or "tabula_predicted_arm" not in d.columns:
        raise RuntimeError(
            "FIX2 final table must contain TF and tabula_predicted_arm."
        )

    x = d[["TF", "tabula_predicted_arm"]].copy()
    x["TF"] = x["TF"].astype(str).str.upper()
    x = x.drop_duplicates("TF")
    x = x.rename(columns={"tabula_predicted_arm": "frozen_arm"})
    return x


# =============================================================================
# DISCOVER THE 79.2 SAME-ARM WITHIN-LINEAGE ASSOCIATION TABLE
# =============================================================================

def candidate_columns(d):
    cm = {str(c).lower(): c for c in d.columns}

    tf_col = next(
        (cm[k] for k in ["tf", "source", "factor"] if k in cm),
        None
    )
    lin_col = next(
        (cm[k] for k in ["lineage", "cell_lineage", "priority_lineage"] if k in cm),
        None
    )
    arm_col = next(
        (cm[k] for k in ["arm", "clock_arm", "predicted_arm", "best_arm"] if k in cm),
        None
    )

    r_col = None
    preferred_r = [
        "adjusted_r", "residual_r", "within_lineage_r",
        "r", "corr", "correlation"
    ]
    for k in preferred_r:
        if k in cm:
            r_col = cm[k]
            break
    if r_col is None:
        for c in d.columns:
            cl = str(c).lower()
            if cl.endswith("_r") and not cl.endswith("fdr"):
                r_col = c
                break

    # sample-size column if available
    n_col = None
    for k in [
        "n_rows", "n", "n_obs", "n_observations",
        "n_samples", "n_blocks", "n_pairs"
    ]:
        if k in cm:
            n_col = cm[k]
            break

    return tf_col, lin_col, arm_col, r_col, n_col


def score_source(path: Path):
    d = safe_csv(path)
    if d.empty:
        return -1, {}

    tf_col, lin_col, arm_col, r_col, n_col = candidate_columns(d)

    score = 0
    if tf_col: score += 20
    if lin_col: score += 20
    if arm_col: score += 20
    if r_col: score += 20
    if n_col: score += 6

    if lin_col:
        lins = d[lin_col].map(normalize_lineage)
        score += min(10, int(lins.nunique(dropna=True)) * 2)

    name = path.name.lower()
    if "within" in name: score += 6
    if "lineage" in name: score += 6
    if "residual" in name: score += 4
    if "arm" in name: score += 3

    return score, {
        "tf_col": tf_col,
        "lineage_col": lin_col,
        "arm_col": arm_col,
        "r_col": r_col,
        "n_col": n_col,
        "n_rows": len(d),
    }


def discover_source():
    rows = []
    for p in SRC79.rglob("*.csv"):
        sc, meta = score_source(p)
        if sc >= 0:
            rows.append((sc, p, meta))

    if not rows:
        raise FileNotFoundError("No CSVs under 79.2 directory.")

    rows.sort(key=lambda x: (x[0], x[2].get("n_rows", 0)), reverse=True)

    audit = []
    for sc, p, meta in rows:
        audit.append({"score": sc, "path": str(p), **meta})

    pd.DataFrame(audit).to_csv(
        AUDIT / "81_WITHIN_LINEAGE_SOURCE_DISCOVERY.csv",
        index=False
    )

    best_sc, best_path, best_meta = rows[0]

    # Same-arm analysis REQUIRES an arm column.
    if (
        best_sc < 80
        or best_meta["tf_col"] is None
        or best_meta["lineage_col"] is None
        or best_meta["arm_col"] is None
        or best_meta["r_col"] is None
    ):
        raise RuntimeError(
            "Could not confidently identify a 79.2 table containing "
            "TF + lineage + arm + residual association. "
            "See 00_AUDIT/81_WITHIN_LINEAGE_SOURCE_DISCOVERY.csv"
        )

    return best_path, best_meta


def load_same_arm_matrix(frozen):
    path, meta = discover_source()
    d = pd.read_csv(path)

    tf = meta["tf_col"]
    lin = meta["lineage_col"]
    arm = meta["arm_col"]
    rc = meta["r_col"]
    nc = meta["n_col"]

    x = pd.DataFrame({
        "TF": d[tf].astype(str).str.upper(),
        "lineage": d[lin].map(normalize_lineage),
        "arm": d[arm].astype(str),
        "r": pd.to_numeric(d[rc], errors="coerce"),
    })
    if nc is not None:
        x["n"] = pd.to_numeric(d[nc], errors="coerce")
    else:
        x["n"] = np.nan

    x = x[
        x["lineage"].isin(LINEAGES)
        & x["r"].notna()
    ].copy()

    x = x.merge(frozen, on="TF", how="inner")

    # Critical: compare ONLY the frozen same arm across all lineages.
    x = x[x["arm"].eq(x["frozen_arm"])].copy()

    # If duplicate rows remain for TF-lineage-arm, do NOT cherry-pick by |r|.
    # Prefer exact duplicate collapse by arithmetic mean; audit the duplicates.
    dup = (
        x.groupby(["TF", "lineage", "frozen_arm"])
         .size()
         .reset_index(name="n_duplicate_rows")
    )
    dup = dup[dup["n_duplicate_rows"] > 1]
    dup.to_csv(AUDIT / "81_DUPLICATE_SAME_ARM_ROWS.csv", index=False)

    agg = (
        x.groupby(["TF", "lineage", "frozen_arm"], as_index=False)
         .agg(
             r=("r", "mean"),
             n=("n", "max"),
             n_source_rows=("r", "size"),
         )
    )

    agg.to_csv(TABLE / "81_SAME_ARM_LINEAGE_ASSOCIATIONS.csv", index=False)

    source_audit = {
        "source_path": str(path),
        "source_columns": meta,
        "same_arm_rule": "Only frozen TF predicted arm is compared across lineages",
        "n_rows_after_same_arm_filter": len(agg),
        "n_TFs": int(agg["TF"].nunique()),
        "n_lineages": int(agg["lineage"].nunique()),
        "n_values_with_n": int(agg["n"].notna().sum()),
        "formal_inference_possible": bool(
            (agg["n"].notna() & (agg["n"] > 3)).sum() >= 10
        ),
    }
    (AUDIT / "81_SOURCE_AUDIT.json").write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return agg, source_audit


# =============================================================================
# DESCRIPTIVE REWIRING METRICS
# =============================================================================

def descriptive_metrics(df):
    tf_rows = []
    link_rows = []

    for tf, g in df.groupby("TF"):
        arm = g["frozen_arm"].iloc[0]
        rmap = {
            row["lineage"]: float(row["r"])
            for _, row in g.iterrows()
        }

        vals = np.array([rmap.get(l, np.nan) for l in LINEAGES], dtype=float)
        ok = np.isfinite(vals)
        vv = vals[ok]

        if len(vv) == 0:
            continue

        r_min = float(np.min(vv))
        r_max = float(np.max(vv))
        r_range = r_max - r_min
        r_sd = float(np.std(vv, ddof=1)) if len(vv) > 1 else np.nan
        median_r = float(np.median(vv))
        median_abs_r = float(np.median(np.abs(vv)))

        pos_n = int(np.sum(vv >= MIN_ABS_R))
        neg_n = int(np.sum(vv <= -MIN_ABS_R))
        robust_sign_switch = bool(pos_n >= 1 and neg_n >= 1)

        strongest_pos_lineage = None
        strongest_neg_lineage = None
        if np.any(vv > 0):
            idx = np.nanargmax(vals)
            strongest_pos_lineage = LINEAGES[idx]
        if np.any(vv < 0):
            idx = np.nanargmin(vals)
            strongest_neg_lineage = LINEAGES[idx]

        tf_rows.append({
            "TF": tf,
            "frozen_arm": arm,
            "n_lineages_observed": int(ok.sum()),
            "r_min": r_min,
            "r_max": r_max,
            "r_range": r_range,
            "r_sd": r_sd,
            "median_r": median_r,
            "median_abs_r": median_abs_r,
            "n_positive_ge_0.20": pos_n,
            "n_negative_le_-0.20": neg_n,
            "robust_sign_switch_descriptive": robust_sign_switch,
            "strongest_positive_lineage": strongest_pos_lineage,
            "strongest_negative_lineage": strongest_neg_lineage,
        })

        for lin in LINEAGES:
            if lin not in rmap:
                continue
            rL = rmap[lin]
            others = [rmap[x] for x in LINEAGES if x != lin and x in rmap]
            if not others:
                continue

            signed_spec = rL - float(np.median(others))
            mag_spec = abs(rL) - float(np.median(np.abs(others)))

            link_rows.append({
                "TF": tf,
                "frozen_arm": arm,
                "lineage": lin,
                "r_lineage": rL,
                "median_r_other_lineages": float(np.median(others)),
                "median_abs_r_other_lineages": float(np.median(np.abs(others))),
                "signed_specificity": signed_spec,
                "magnitude_specificity": mag_spec,
                "direction_relative_to_others": (
                    "MORE_POSITIVE" if signed_spec > 0
                    else "MORE_NEGATIVE" if signed_spec < 0
                    else "SAME"
                ),
            })

    return pd.DataFrame(tf_rows), pd.DataFrame(link_rows)


# =============================================================================
# FORMAL FISHER-Z INFERENCE IF n IS AVAILABLE
# =============================================================================

def heterogeneity_tests(df):
    rows = []

    for tf, g in df.groupby("TF"):
        zvals = []
        weights = []

        for _, r in g.iterrows():
            n = r["n"]
            rr = r["r"]
            if np.isfinite(n) and n > 3 and np.isfinite(rr):
                zvals.append(fisher_z(rr))
                weights.append(n - 3)

        if len(zvals) < 2:
            rows.append({
                "TF": tf,
                "heterogeneity_Q": np.nan,
                "heterogeneity_df": np.nan,
                "heterogeneity_P": np.nan,
                "inference_n_lineages": len(zvals),
            })
            continue

        zvals = np.asarray(zvals)
        weights = np.asarray(weights, dtype=float)
        zbar = np.sum(weights * zvals) / np.sum(weights)
        Q = float(np.sum(weights * (zvals - zbar) ** 2))
        dfree = len(zvals) - 1
        p = float(stats.chi2.sf(Q, dfree))

        rows.append({
            "TF": tf,
            "heterogeneity_Q": Q,
            "heterogeneity_df": dfree,
            "heterogeneity_P": p,
            "inference_n_lineages": len(zvals),
        })

    out = pd.DataFrame(rows)
    out["heterogeneity_FDR"] = bh(out["heterogeneity_P"].to_numpy(float))
    return out


def lineage_vs_others_tests(df):
    rows = []

    for tf, g in df.groupby("TF"):
        for lin in LINEAGES:
            one = g[g["lineage"] == lin]
            others = g[g["lineage"] != lin]

            if len(one) != 1:
                continue

            rr = float(one.iloc[0]["r"])
            nn = one.iloc[0]["n"]

            valid_others = others[
                others["n"].notna()
                & (others["n"] > 3)
                & others["r"].notna()
            ].copy()

            if (
                not np.isfinite(nn)
                or nn <= 3
                or len(valid_others) < 2
            ):
                rows.append({
                    "TF": tf,
                    "lineage": lin,
                    "contrast_z": np.nan,
                    "contrast_P": np.nan,
                    "z_lineage": np.nan,
                    "z_other_weighted": np.nan,
                    "contrast_delta_z": np.nan,
                })
                continue

            z1 = fisher_z(rr)
            v1 = 1.0 / (float(nn) - 3.0)

            zo = np.array([fisher_z(x) for x in valid_others["r"]], dtype=float)
            wo = valid_others["n"].to_numpy(float) - 3.0
            z_other = float(np.sum(wo * zo) / np.sum(wo))
            v_other = 1.0 / float(np.sum(wo))

            se = math.sqrt(v1 + v_other)
            zstat = (z1 - z_other) / se
            p = 2 * stats.norm.sf(abs(zstat))

            rows.append({
                "TF": tf,
                "lineage": lin,
                "contrast_z": zstat,
                "contrast_P": p,
                "z_lineage": z1,
                "z_other_weighted": z_other,
                "contrast_delta_z": z1 - z_other,
            })

    out = pd.DataFrame(rows)
    out["contrast_FDR"] = bh(out["contrast_P"].to_numpy(float))
    return out


def pairwise_tests(df):
    rows = []

    for tf, g in df.groupby("TF"):
        gg = g.set_index("lineage")
        for i, l1 in enumerate(LINEAGES):
            for l2 in LINEAGES[i+1:]:
                if l1 not in gg.index or l2 not in gg.index:
                    continue

                r1 = float(gg.loc[l1, "r"])
                r2 = float(gg.loc[l2, "r"])
                n1 = gg.loc[l1, "n"]
                n2 = gg.loc[l2, "n"]

                opposite_robust = bool(
                    np.sign(r1) != np.sign(r2)
                    and abs(r1) >= SIGN_SWITCH_MIN_ABS_R
                    and abs(r2) >= SIGN_SWITCH_MIN_ABS_R
                )

                if (
                    np.isfinite(n1) and n1 > 3
                    and np.isfinite(n2) and n2 > 3
                ):
                    z1 = fisher_z(r1)
                    z2 = fisher_z(r2)
                    se = math.sqrt(
                        1.0 / (float(n1) - 3.0)
                        + 1.0 / (float(n2) - 3.0)
                    )
                    zstat = (z1 - z2) / se
                    p = 2 * stats.norm.sf(abs(zstat))
                else:
                    zstat = np.nan
                    p = np.nan

                rows.append({
                    "TF": tf,
                    "lineage_1": l1,
                    "lineage_2": l2,
                    "r1": r1,
                    "r2": r2,
                    "delta_r": r1 - r2,
                    "opposite_robust_signs": opposite_robust,
                    "pair_z": zstat,
                    "pair_P": p,
                })

    out = pd.DataFrame(rows)
    out["pair_FDR"] = bh(out["pair_P"].to_numpy(float))
    return out


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify(tf_metrics, link_metrics, het, contrasts, pairs):
    x = tf_metrics.merge(het, on="TF", how="left")

    # inferential flags
    robust_pair = {}
    for tf, g in pairs.groupby("TF"):
        robust_pair[tf] = bool(
            (
                g["opposite_robust_signs"]
                & g["pair_FDR"].le(PAIRWISE_Q_GATE)
            ).fillna(False).any()
        )

    selective_link = {}
    for tf, g in link_metrics.groupby("TF"):
        c = contrasts[contrasts["TF"] == tf].copy()
        gg = g.merge(c[["TF", "lineage", "contrast_FDR"]], on=["TF", "lineage"], how="left")

        flag = (
            (
                gg["signed_specificity"].abs().ge(LINEAGE_SIGNED_SPEC_DELTA)
                | gg["magnitude_specificity"].ge(LINEAGE_MAG_SPEC_DELTA)
            )
            & gg["contrast_FDR"].le(CONTRAST_Q_GATE)
        ).fillna(False)

        selective_link[tf] = bool(flag.any())

    classes = []
    inference_mode = []

    for _, r in x.iterrows():
        tf = r["TF"]
        vals = link_metrics[link_metrics["TF"] == tf]["r_lineage"].to_numpy(float)
        n_obs = int(r["n_lineages_observed"])

        formal = bool(np.isfinite(r.get("heterogeneity_FDR", np.nan)))
        inference_mode.append("FORMAL_FISHER_Z" if formal else "DESCRIPTIVE_ONLY")

        # 1) strongest result: statistically supported robust sign switch
        if formal and robust_pair.get(tf, False):
            cls = "SIGN_SWITCHING"

        # 2) heterogeneous and a lineage is specifically different from others
        elif (
            formal
            and r["heterogeneity_FDR"] <= HET_Q_GATE
            and selective_link.get(tf, False)
        ):
            cls = "LINEAGE_SELECTIVE"

        # 3) descriptive fallback sign switch if formal n missing
        elif (
            not formal
            and bool(r["robust_sign_switch_descriptive"])
        ):
            cls = "SIGN_SWITCHING_DESCRIPTIVE"

        # 4) pan-lineage same-sign programs
        elif (
            n_obs >= 4
            and np.sum(vals >= PAN_MIN_ABS_R) >= PAN_MIN_SAME_SIGN
            and r["r_range"] <= PAN_MAX_RANGE
        ):
            cls = "PAN_LINEAGE_POSITIVE"

        elif (
            n_obs >= 4
            and np.sum(vals <= -PAN_MIN_ABS_R) >= PAN_MIN_SAME_SIGN
            and r["r_range"] <= PAN_MAX_RANGE
        ):
            cls = "PAN_LINEAGE_NEGATIVE"

        # 5) same sign but heterogeneous strength
        elif (
            n_obs >= 4
            and (
                np.sum(vals > 0) >= PAN_MIN_SAME_SIGN
                or np.sum(vals < 0) >= PAN_MIN_SAME_SIGN
            )
            and (
                (formal and r["heterogeneity_FDR"] <= HET_Q_GATE)
                or (not formal and r["r_range"] > PAN_MAX_RANGE)
            )
        ):
            cls = "SHARED_SIGN_HETEROGENEOUS"

        else:
            cls = "WEAK_OR_UNSTABLE"

        classes.append(cls)

    x["rewiring_class"] = classes
    x["inference_mode"] = inference_mode

    order = {
        "SIGN_SWITCHING": 1,
        "LINEAGE_SELECTIVE": 2,
        "SIGN_SWITCHING_DESCRIPTIVE": 3,
        "SHARED_SIGN_HETEROGENEOUS": 4,
        "PAN_LINEAGE_POSITIVE": 5,
        "PAN_LINEAGE_NEGATIVE": 6,
        "WEAK_OR_UNSTABLE": 7,
    }
    x["_order"] = x["rewiring_class"].map(order).fillna(99)
    x = x.sort_values(
        ["_order", "r_range", "r_sd"],
        ascending=[True, False, False]
    ).drop(columns="_order")

    return x


# =============================================================================
# EXTERNAL EVIDENCE ANNOTATIONS FROM FIX3
# =============================================================================

def add_external_annotations(link_metrics):
    f3 = safe_csv(FIX3_TABLE)
    if f3.empty:
        link_metrics["binding_grade"] = np.nan
        link_metrics["perturb_grade_global"] = np.nan
        return link_metrics

    keep = [
        c for c in [
            "TF", "lineage",
            "binding_grade",
            "perturb_grade_global",
            "binding_targets",
            "lineage_specific_tier",
        ]
        if c in f3.columns
    ]
    z = f3[keep].copy()
    z["TF"] = z["TF"].astype(str).str.upper()
    z["lineage"] = z["lineage"].map(normalize_lineage)
    z = z.drop_duplicates(["TF", "lineage"])

    return link_metrics.merge(z, on=["TF", "lineage"], how="left")


# =============================================================================
# FIGURES
# =============================================================================

def make_figures(same_arm, tf_final, link_final):
    # F1: same-arm r heatmap sorted by rewiring class/range.
    order = tf_final["TF"].tolist()
    piv = same_arm.pivot_table(
        index="TF", columns="lineage", values="r", aggfunc="first"
    ).reindex(index=order, columns=LINEAGES)

    fig, ax = plt.subplots(figsize=(8.6, max(5.2, 0.32 * len(piv) + 1.8)))
    im = ax.imshow(piv.to_numpy(float), aspect="auto", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(len(LINEAGES)))
    ax.set_xticklabels([LINEAGE_LABEL[l] for l in LINEAGES], rotation=25, ha="right")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index)
    ax.set_title("81A. Same-arm TF–clock coupling across lineages")
    fig.colorbar(im, ax=ax, fraction=.028, pad=.02, label="within-lineage residual r")
    savefig(fig, "81_F1_SAME_ARM_REWIRING_HEATMAP")

    # F2: specificity heatmap
    sp = link_final.pivot_table(
        index="TF", columns="lineage", values="signed_specificity", aggfunc="first"
    ).reindex(index=order, columns=LINEAGES)

    fig, ax = plt.subplots(figsize=(8.6, max(5.2, 0.32 * len(sp) + 1.8)))
    im = ax.imshow(sp.to_numpy(float), aspect="auto", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(len(LINEAGES)))
    ax.set_xticklabels([LINEAGE_LABEL[l] for l in LINEAGES], rotation=25, ha="right")
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels(sp.index)
    ax.set_title("81B. Lineage specificity of TF–clock coupling")
    fig.colorbar(
        im, ax=ax, fraction=.028, pad=.02,
        label="r(lineage) − median r(other lineages)"
    )
    savefig(fig, "81_F2_LINEAGE_SPECIFICITY_HEATMAP")

    # F3: range vs heterogeneity
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    q = pd.to_numeric(tf_final["heterogeneity_FDR"], errors="coerce")
    y = -np.log10(np.clip(q.to_numpy(float), 1e-12, 1.0))
    x = tf_final["r_range"].to_numpy(float)
    ax.scatter(x, y, s=50, alpha=.75)
    ax.axhline(-np.log10(HET_Q_GATE), lw=.7, linestyle="--")
    ax.set_xlabel("Across-lineage r range")
    ax.set_ylabel("-log10 heterogeneity FDR")
    ax.set_title("81C. Statistical heterogeneity of TF–clock coupling")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for _, rr in tf_final.head(15).iterrows():
        if np.isfinite(rr.get("heterogeneity_FDR", np.nan)):
            ax.annotate(
                rr["TF"],
                (rr["r_range"], -np.log10(max(rr["heterogeneity_FDR"], 1e-12))),
                xytext=(4, 4), textcoords="offset points", fontsize=8
            )
    savefig(fig, "81_F3_HETEROGENEITY")

    # F4: class count
    cc = tf_final["rewiring_class"].value_counts()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(np.arange(len(cc)), cc.to_numpy())
    ax.set_xticks(np.arange(len(cc)))
    ax.set_xticklabels(cc.index, rotation=30, ha="right")
    ax.set_ylabel("Number of TFs")
    ax.set_title("81D. TF–clock rewiring classes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "81_F4_REWIRING_CLASSES")

    # F5-F9: top lineage-specific contrasts
    for lin in LINEAGES:
        z = link_final[link_final["lineage"] == lin].copy()
        z["_abs_spec"] = z["signed_specificity"].abs()
        z = z.sort_values("_abs_spec", ascending=False).head(10)
        if z.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.6, max(4.8, .42 * len(z) + 1.6)))
        y = np.arange(len(z))
        ax.scatter(z["signed_specificity"], y, s=58)
        ax.axvline(0, lw=.7)
        ax.set_yticks(y)
        ax.set_yticklabels(z["TF"])
        ax.invert_yaxis()
        ax.set_xlabel("r(lineage) − median r(other lineages)")
        ax.set_title(f"81. {LINEAGE_LABEL[lin]}-selective TF–clock coupling")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        savefig(fig, f"81_LINEAGE_{lin}_SPECIFICITY")


# =============================================================================
# REPORT
# =============================================================================

def write_report(source_audit, tf_final, link_final, pairs):
    formal_n = int((tf_final["inference_mode"] == "FORMAL_FISHER_Z").sum())
    sign_switch = tf_final[
        tf_final["rewiring_class"].isin(
            ["SIGN_SWITCHING", "SIGN_SWITCHING_DESCRIPTIVE"]
        )
    ]
    selective = tf_final[tf_final["rewiring_class"] == "LINEAGE_SELECTIVE"]
    pan_pos = tf_final[tf_final["rewiring_class"] == "PAN_LINEAGE_POSITIVE"]
    pan_neg = tf_final[tf_final["rewiring_class"] == "PAN_LINEAGE_NEGATIVE"]

    cols = [
        "TF", "frozen_arm", "rewiring_class",
        "r_min", "r_max", "r_range", "r_sd",
        "heterogeneity_FDR", "inference_mode"
    ]

    top_links = link_final.copy()
    top_links["_abs_spec"] = top_links["signed_specificity"].abs()
    top_links = top_links.sort_values("_abs_spec", ascending=False)

    lines = [
        "# Analysis 81 — lineage-selective TF–clock rewiring",
        "",
        "## Central question",
        "",
        "This analysis asks whether the **same TF–same clock-arm relationship** changes "
        "across lineages. It does not search for one universal TF, and it does not "
        "compare whichever arm happens to be strongest in each lineage.",
        "",
        "The clock arm for each TF is frozen from Analysis 79/80 before lineage comparison.",
        "",
        "## Source audit",
        "",
        "```json",
        json.dumps(source_audit, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Rewiring metrics",
        "",
        "- `r_range = max(r_lineage) - min(r_lineage)`",
        "- `signed_specificity = r_lineage - median(r_other lineages)`",
        "- `magnitude_specificity = |r_lineage| - median(|r_other lineages|)`",
        "- robust sign-switch requires opposite signs with |r| >= 0.20 on both sides",
        "",
        "If per-lineage n was available, Fisher-r-to-z heterogeneity and lineage-vs-others "
        "contrasts were tested with BH correction.",
        "",
        f"Formal Fisher-z inference available for {formal_n}/{len(tf_final)} TFs.",
        "",
        "## TF-level rewiring classes",
        "",
        md_table(tf_final[cols], 50),
        "",
        "## Strongest lineage-specific links",
        "",
        md_table(
            top_links[
                [
                    "TF", "lineage", "frozen_arm",
                    "r_lineage", "median_r_other_lineages",
                    "signed_specificity", "magnitude_specificity",
                    "contrast_FDR",
                    "binding_grade", "perturb_grade_global",
                ]
            ],
            40
        ),
        "",
        "## Interpretation",
        "",
        f"- Sign-switching TFs: {', '.join(sign_switch['TF'].tolist()) if len(sign_switch) else 'none'}",
        f"- Lineage-selective TFs: {', '.join(selective['TF'].tolist()) if len(selective) else 'none'}",
        f"- Pan-lineage positive programs: {', '.join(pan_pos['TF'].tolist()) if len(pan_pos) else 'none'}",
        f"- Pan-lineage negative programs: {', '.join(pan_neg['TF'].tolist()) if len(pan_neg) else 'none'}",
        "",
        "The intended biological model is therefore:",
        "",
        "`shared TF regulatory backbone + lineage-selective rewiring of TF–clock coupling -> distinct clock architectures`",
        "",
        "Binding and perturbation grades are external annotations only; they are not used "
        "to manufacture lineage specificity.",
        "",
        "## Claim boundary",
        "",
        "These are computational regulatory associations. A significant TF×lineage-like "
        "contrast supports lineage-selective rewiring, but it is not experimental proof of causality.",
    ]

    (REPORT / "81_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    verdict = {
        "status": "COMPLETE",
        "same_arm_comparison": True,
        "universal_TF_search": False,
        "n_TFs": int(len(tf_final)),
        "n_formal_inference_TFs": formal_n,
        "sign_switching_TFs": sign_switch["TF"].tolist(),
        "lineage_selective_TFs": selective["TF"].tolist(),
        "pan_lineage_positive_TFs": pan_pos["TF"].tolist(),
        "pan_lineage_negative_TFs": pan_neg["TF"].tolist(),
        "main_model": "shared regulatory backbone plus lineage-selective TF-clock rewiring",
        "wet_lab_required": False,
    }
    (OUT / "81_COMPLETE.json").write_text(
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

    frozen = load_frozen_tf_arm()
    same_arm, source_audit = load_same_arm_matrix(frozen)

    tf_desc, link_desc = descriptive_metrics(same_arm)

    het = heterogeneity_tests(same_arm)
    contrasts = lineage_vs_others_tests(same_arm)
    pairs = pairwise_tests(same_arm)

    tf_final = classify(tf_desc, link_desc, het, contrasts, pairs)

    link_final = link_desc.merge(
        contrasts[["TF", "lineage", "contrast_z", "contrast_P", "contrast_FDR"]],
        on=["TF", "lineage"], how="left"
    )
    link_final = add_external_annotations(link_final)

    # Append TF class to every lineage link.
    link_final = link_final.merge(
        tf_final[["TF", "rewiring_class", "heterogeneity_FDR", "inference_mode"]],
        on="TF", how="left"
    )

    # Save all statistical source data.
    tf_final.to_csv(TABLE / "81_TF_LEVEL_REWIRING_CLASSIFICATION.csv", index=False)
    link_final.to_csv(TABLE / "81_LINEAGE_SPECIFIC_LINKS.csv", index=False)
    het.to_csv(TABLE / "81_HETEROGENEITY_TESTS.csv", index=False)
    contrasts.to_csv(TABLE / "81_LINEAGE_VS_OTHERS_CONTRASTS.csv", index=False)
    pairs.to_csv(TABLE / "81_PAIRWISE_LINEAGE_CONTRASTS.csv", index=False)

    # Per-lineage top candidates by |specificity|, not by raw |r|.
    tops = []
    for lin in LINEAGES:
        z = link_final[link_final["lineage"] == lin].copy()
        z["_abs_spec"] = z["signed_specificity"].abs()
        z = z.sort_values(
            ["_abs_spec", "magnitude_specificity"],
            ascending=[False, False]
        ).head(10)
        z["rank_within_lineage_by_specificity"] = np.arange(1, len(z) + 1)
        tops.append(z.drop(columns="_abs_spec"))
    topdf = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()
    topdf.to_csv(TABLE / "81_TOP10_SELECTIVE_LINKS_PER_LINEAGE.csv", index=False)

    make_figures(same_arm, tf_final, link_final)
    write_report(source_audit, tf_final, link_final, pairs)

    print("=" * 100)
    print("81 LINEAGE-SELECTIVE TF-CLOCK REWIRING COMPLETE")
    print("=" * 100)
    print("SAME_TF_SAME_ARM_COMPARISON = TRUE")
    print("UNIVERSAL_TF_SEARCH = FALSE")
    print("N_TFS =", len(tf_final))
    print("FORMAL_FISHER_Z_TFS =", int((tf_final["inference_mode"] == "FORMAL_FISHER_Z").sum()))
    print()
    for cls in [
        "SIGN_SWITCHING",
        "LINEAGE_SELECTIVE",
        "SIGN_SWITCHING_DESCRIPTIVE",
        "SHARED_SIGN_HETEROGENEOUS",
        "PAN_LINEAGE_POSITIVE",
        "PAN_LINEAGE_NEGATIVE",
        "WEAK_OR_UNSTABLE",
    ]:
        vals = tf_final.loc[tf_final["rewiring_class"] == cls, "TF"].tolist()
        print(f"{cls} =", ", ".join(vals) if vals else "NONE")
    print()
    print("MAIN_MODEL = SHARED_BACKBONE_PLUS_LINEAGE_SELECTIVE_REWIRING")
    print("WET_LAB_REQUIRED = FALSE")
    print("REPORT =", REPORT / "81_FINAL_REPORT.md")
    print("TF_TABLE =", TABLE / "81_TF_LEVEL_REWIRING_CLASSIFICATION.csv")
    print("LINK_TABLE =", TABLE / "81_LINEAGE_SPECIFIC_LINKS.csv")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        (OUT / "81_FAILED.txt").write_text(tb, encoding="utf-8")
        print(tb)
        raise
