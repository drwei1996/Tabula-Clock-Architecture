# -*- coding: utf-8 -*-
"""
78_LINEAGE_INTRINSIC_3LOOP_VALIDATION.py

Purpose
-------
Strictly test the biological premise:

    Major cell lineages carry a stable core-clock 3-loop architecture
    across tissues, rather than the observed structure being explained only
    by tissue, donor, or random sampling phase.

This is a GATE analysis. It intentionally does NOT search for TF drivers yet.
Only if this gate is supported should a subsequent TF/regulon/motif analysis
be interpreted mechanistically.

Inputs already present in the project
-------------------------------------
1) Tabula frozen donor×tissue×lineage six-arm scores
2) HCL standardized Expanded17 pseudobulks
3) AHCA standardized Expanded17 pseudobulks
4) GSE292219 time-stamped six-arm scores from analysis 76

No internet, no new packages, no downloading.

Primary tests
-------------
A. Additive variance decomposition:
   six-arm architecture ~ lineage + tissue + donor
   unique multivariate partial R2 for lineage/tissue/donor

B. Tissue-held-out lineage portability:
   build tissue×lineage centroids
   train lineage prototypes on all other tissues
   predict held-out tissue by nearest six-arm prototype
   compare balanced accuracy to within-tissue lineage-label permutation null

C. Cross-tissue same-lineage stability:
   median cosine similarity among tissue centroids of the same lineage
   compare to within-tissue label-permutation null

D. Cross-atlas lineage architecture replication:
   tissue-center six-arm scores, construct 5-lineage × 6-arm prototypes
   compare Tabula vs HCL and Tabula vs AHCA by flattened Pearson correlation
   exact lineage-label permutation test (<= 5! = 120 permutations)

E. Time-stamped validation:
   GSE292219 six-arm scores are residualized within each ZT.
   Test whether lineage identity remains predictable across held-out ZT.
   This directly asks whether shared phase can coexist with stable lineage architecture.

Interpretation
--------------
STRONG_SUPPORT requires:
- Tabula tissue-held-out lineage portability above permutation null;
- at least one independent human atlas supports portability;
- at least one external prototype concordance is significant;
- GSE292219 retains lineage structure after explicit ZT residualization.

The script also reports the stricter statement "Lineage > Tissue", but does NOT
require it for the more defensible claim that lineage architecture is independent
of tissue.

Output
------
corrected_full_analysis_v2/
78_LINEAGE_INTRINSIC_3LOOP_VALIDATION/
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. PATHS / CONSTANTS
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

OUT = ROOT / "78_LINEAGE_INTRINSIC_3LOOP_VALIDATION"
TABLE = OUT / "01_TABLES"
FIG = OUT / "02_FIGURES"
REPORT = OUT / "03_REPORT"
AUDIT = OUT / "00_AUDIT"

for p in [OUT, TABLE, FIG, REPORT, AUDIT]:
    p.mkdir(parents=True, exist_ok=True)

TABULA = (
    ROOT
    / "64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930"
    / "source_data"
    / "DONOR_TISSUE_LINEAGE_SCORES.csv"
)

EXT_ROOT = (
    ROOT
    / "external validation"
    / "clock_external_validation_ALL_EXCEPT_H5AD_20260921"
    / "clock_external_validation"
)

HCL = EXT_ROOT / "01_HCL" / "source_data" / "pseudobulk_standardized.csv"
AHCA = EXT_ROOT / "02_AHCA_15ORG" / "source_data" / "pseudobulk_standardized.csv"

GSE292 = (
    ROOT
    / "76_GSE292219_CLOCK_TIME_VS_LINEAGE_STORY"
    / "01_SOURCE_DATA"
    / "76_LOOP_ARM_AND_BALANCE_SCORES.csv"
)

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

DISPLAY_TO_KEY = {v: k for k, v in LINEAGE_DISPLAY.items()}

LINEAGE_COLORS = {
    "b_cell": "#4E79A7",
    "t_cell": "#E15759",
    "macrophage_monocyte": "#59A14F",
    "fibroblast": "#F28E2B",
    "endothelial": "#9467BD",
}

CLOCK17 = [
    "ARNTL", "CLOCK", "NPAS2",
    "PER1", "PER2", "PER3",
    "CRY1", "CRY2",
    "NR1D1", "NR1D2",
    "RORA", "RORB", "RORC",
    "DBP", "TEF", "HLF", "NFIL3",
]

ARM_GENES = {
    "EBOX_POS": ["ARNTL", "CLOCK", "NPAS2"],
    "EBOX_NEG": ["PER1", "PER2", "PER3", "CRY1", "CRY2"],
    "RORE_ACT": ["RORA", "RORB", "RORC"],
    "RORE_REP": ["NR1D1", "NR1D2"],
    "DBOX_ACT": ["DBP", "TEF", "HLF"],
    "DBOX_REP": ["NFIL3"],
}

ARMS = list(ARM_GENES)

N_PERM = 5000
SEED = 20261004
EPS = 1e-12

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
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
})


# =============================================================================
# 1. HELPERS
# =============================================================================

def log(msg=""):
    s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s, flush=True)
    with (OUT / "78_console.log").open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def savefig(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kwargs = dict(bbox_inches="tight", pad_inches=0.08, facecolor="white")
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def write_csv(df, name):
    df.to_csv(TABLE / name, index=False, encoding="utf-8-sig")


def write_json(obj, name):
    (AUDIT / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    if len(x) == 0:
        return q
    order = np.argsort(x)
    ranked = x[order]
    vals = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    vals = np.clip(vals, 0, 1)
    back = np.empty(len(x))
    back[order] = vals
    q[ok] = back
    return q


def zscore_columns(df, cols):
    out = df.copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce").astype(float)
        sd = v.std(ddof=0)
        out[c] = (v - v.mean()) / sd if np.isfinite(sd) and sd > 0 else 0.0
    return out


def normalize_lineage(x):
    s = str(x).strip()
    if s in LINEAGES:
        return s
    if s in DISPLAY_TO_KEY:
        return DISPLAY_TO_KEY[s]
    low = s.lower().replace("/", "_").replace(" ", "_")
    if low in LINEAGES:
        return low
    if "macrophage" in low or "monocyte" in low:
        return "macrophage_monocyte"
    if low in {"b_cell", "b_cells"}:
        return "b_cell"
    if low in {"t_cell", "t_cells"}:
        return "t_cell"
    if "fibro" in low:
        return "fibroblast"
    if "endothel" in low:
        return "endothelial"
    return None


def orthobasis(X):
    X = np.asarray(X, float)
    if X.size == 0:
        return np.empty((len(X), 0)), 0
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    if len(s) == 0 or s[0] <= 0:
        return np.empty((len(X), 0)), 0
    tol = s[0] * max(X.shape) * np.finfo(float).eps
    rank = int(np.sum(s > tol))
    return u[:, :rank], rank


def design_matrix(df, terms):
    mats = [np.ones((len(df), 1), float)]
    for term in terms:
        if term == "lineage_x_tissue":
            s = df["lineage"].astype(str) + "||" + df["tissue"].astype(str)
        else:
            s = df[term].astype(str)
        d = pd.get_dummies(s, drop_first=False, dtype=float)
        mats.append(d.to_numpy(float))
    return np.column_stack(mats)


def pooled_sse(Y, Q):
    Y = np.asarray(Y, float)
    y2 = float(np.sum(Y ** 2))
    if Q.shape[1] == 0:
        return y2
    proj = Q.T @ Y
    return max(y2 - float(np.sum(proj ** 2)), 0.0)


def adjusted_r2_multivariate(Y, X):
    Q, rank = orthobasis(X)
    sse = pooled_sse(Y, Q)
    Y = np.asarray(Y, float)
    centered = Y - Y.mean(axis=0, keepdims=True)
    sst = float(np.sum(centered ** 2))
    n_scalar = Y.size
    # rank applies to observations and acts on every response column.
    p_eff = rank * Y.shape[1]
    if sst <= 0 or n_scalar - p_eff - 1 <= 0:
        return np.nan, sse, rank
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n_scalar - 1) / (n_scalar - p_eff - 1)
    return float(adj), sse, rank


def unique_partial_r2(Y, full_terms, reduced_terms, meta):
    Xf = design_matrix(meta, full_terms)
    Xr = design_matrix(meta, reduced_terms)
    Qf, _ = orthobasis(Xf)
    Qr, _ = orthobasis(Xr)
    sf = pooled_sse(Y, Qf)
    sr = pooled_sse(Y, Qr)
    if sr <= 0:
        return np.nan
    return float(max(0.0, (sr - sf) / sr))


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den <= 0:
        return np.nan
    return float(np.dot(a, b) / den)


def perm_p(obs, null, greater=True):
    x = np.asarray(null, float)
    x = x[np.isfinite(x)]
    if not np.isfinite(obs) or len(x) == 0:
        return np.nan
    if greater:
        return float((1 + np.sum(x >= obs - 1e-12)) / (len(x) + 1))
    return float((1 + np.sum(np.abs(x) >= abs(obs) - 1e-12)) / (len(x) + 1))


def df_to_markdown(df):
    d = df.copy()
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.5g}"
        return str(v).replace("|", r"\|").replace("\n", " ")
    headers = [fmt(c) for c in d.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in d.columns) + " |")
    return "\n".join(lines)


# =============================================================================
# 2. LOAD DATASETS IN THE SAME SIX-ARM SPACE
# =============================================================================

def aggregate_donor_tissue_lineage(df, donor_col, tissue_col, lineage_col, arms):
    x = df.copy()
    x["lineage"] = x[lineage_col].map(normalize_lineage)
    x = x[x["lineage"].isin(LINEAGES)].copy()
    x["donor_id"] = x[donor_col].astype(str)
    x["tissue"] = x[tissue_col].astype(str)
    for c in arms:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    out = (
        x.groupby(["donor_id", "tissue", "lineage"], as_index=False, observed=True)[arms]
        .mean()
    )
    return out


def load_tabula():
    if not TABULA.exists():
        raise FileNotFoundError(TABULA)
    x = pd.read_csv(TABULA, low_memory=False)
    need = {"donor_id", "tissue", "priority_lineage", *ARMS}
    miss = need - set(x.columns)
    if miss:
        raise RuntimeError(f"Tabula missing columns: {sorted(miss)}")
    return aggregate_donor_tissue_lineage(
        x, "donor_id", "tissue", "priority_lineage", ARMS
    )


def derive_arms_from_expanded17(path, dataset):
    if not path.exists():
        raise FileNotFoundError(path)
    x = pd.read_csv(path, low_memory=False)

    need_meta = {"donor_id", "tissue", "major_lineage"}
    miss = need_meta - set(x.columns)
    if miss:
        raise RuntimeError(f"{dataset} missing metadata columns: {sorted(miss)}")

    for arm, genes in ARM_GENES.items():
        cols = [f"expr_{g}" for g in genes if f"expr_{g}" in x.columns]
        if not cols:
            raise RuntimeError(f"{dataset}: no represented genes for {arm}")
        x[arm] = x[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    return aggregate_donor_tissue_lineage(
        x, "donor_id", "tissue", "major_lineage", ARMS
    )


def load_gse292():
    if not GSE292.exists():
        raise FileNotFoundError(GSE292)
    x = pd.read_csv(GSE292, low_memory=False)
    need = {"lineage", "ZT", *ARMS}
    miss = need - set(x.columns)
    if miss:
        raise RuntimeError(f"GSE292219 six-arm table missing: {sorted(miss)}")

    y = x[["lineage", "ZT", *ARMS]].copy()
    y["lineage"] = y["lineage"].map(normalize_lineage)
    y = y[y["lineage"].isin(LINEAGES)].copy()
    y["ZT"] = pd.to_numeric(y["ZT"], errors="coerce")
    for c in ARMS:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    return y.dropna(subset=["ZT"])


# =============================================================================
# 3. DATASET-LEVEL ADDITIVE EFFECTS
# =============================================================================

def effect_decomposition(dataset, agg):
    d = zscore_columns(agg.dropna(subset=ARMS).copy(), ARMS)
    Y = d[ARMS].to_numpy(float)

    terms_all = ["lineage", "tissue"]
    donor_variable = d["donor_id"].nunique() > 1
    if donor_variable:
        terms_all.append("donor_id")

    lineage_red = [x for x in terms_all if x != "lineage"]
    tissue_red = [x for x in terms_all if x != "tissue"]
    donor_red = [x for x in terms_all if x != "donor_id"]

    lin = unique_partial_r2(Y, terms_all, lineage_red, d)
    tis = unique_partial_r2(Y, terms_all, tissue_red, d)
    don = unique_partial_r2(Y, terms_all, donor_red, d) if donor_variable else np.nan

    add_X = design_matrix(d, terms_all)
    full_X = design_matrix(d, terms_all + ["lineage_x_tissue"])
    Qa, _ = orthobasis(add_X)
    Qf, _ = orthobasis(full_X)
    s_add = pooled_sse(Y, Qa)
    s_full = pooled_sse(Y, Qf)
    inter = max(0.0, (s_add - s_full) / s_add) if s_add > 0 else np.nan

    return {
        "dataset": dataset,
        "n_rows": len(d),
        "n_donors": d["donor_id"].nunique(),
        "n_tissues": d["tissue"].nunique(),
        "n_lineages": d["lineage"].nunique(),
        "lineage_unique_partial_R2": lin,
        "tissue_unique_partial_R2": tis,
        "donor_unique_partial_R2": don,
        "lineage_x_tissue_increment_partial_R2": inter,
        "lineage_minus_tissue": lin - tis if np.isfinite(lin) and np.isfinite(tis) else np.nan,
        "strict_lineage_gt_tissue": bool(np.isfinite(lin) and np.isfinite(tis) and lin > tis),
    }


# =============================================================================
# 4. TISSUE-HELD-OUT PORTABILITY
# =============================================================================

def tissue_lineage_centroids(agg):
    d = zscore_columns(agg, ARMS)
    c = (
        d.groupby(["tissue", "lineage"], as_index=False, observed=True)[ARMS]
        .mean()
    )
    return c


def loto_predict(cent):
    records = []
    tissues = sorted(cent["tissue"].astype(str).unique())

    for tissue in tissues:
        train = cent[cent["tissue"].astype(str).ne(tissue)].copy()
        test = cent[cent["tissue"].astype(str).eq(tissue)].copy()

        proto = train.groupby("lineage", observed=True)[ARMS].mean()
        if len(proto) < 2:
            continue

        for _, r in test.iterrows():
            true = r["lineage"]
            if true not in proto.index:
                continue
            x = r[ARMS].to_numpy(float)
            dist = {
                lin: float(np.linalg.norm(x - proto.loc[lin, ARMS].to_numpy(float)))
                for lin in proto.index
            }
            pred = min(dist, key=dist.get)
            order = sorted(dist, key=dist.get)
            rank = order.index(true) + 1
            records.append({
                "tissue": tissue,
                "true": true,
                "pred": pred,
                "true_rank": rank,
                "true_distance": dist[true],
                "nearest_other_distance": min(
                    [v for k, v in dist.items() if k != true],
                    default=np.nan
                ),
            })
    return pd.DataFrame(records)


def loto_score(cent):
    pred = loto_predict(cent)
    if pred.empty or pred["true"].nunique() < 2:
        return np.nan, np.nan, pred
    bal = float(balanced_accuracy_score(pred["true"], pred["pred"]))
    mean_rank = float(pred["true_rank"].mean())
    return bal, mean_rank, pred


def permute_labels_within_tissue(cent, rng):
    p = cent.copy()
    labels = []
    for _, g in p.groupby("tissue", sort=False):
        idx = g.index.to_numpy()
        shuf = g["lineage"].to_numpy(copy=True)
        rng.shuffle(shuf)
        labels.extend(zip(idx, shuf))
    for idx, lab in labels:
        p.at[idx, "lineage"] = lab
    return p


def portability_permutation(dataset, cent, n_perm=N_PERM):
    obs_bal, obs_rank, pred = loto_score(cent)
    rng = np.random.default_rng(SEED + abs(hash(dataset)) % 100000)

    null_bal = np.full(n_perm, np.nan)
    null_rank = np.full(n_perm, np.nan)
    for b in range(n_perm):
        p = permute_labels_within_tissue(cent, rng)
        null_bal[b], null_rank[b], _ = loto_score(p)

    result = {
        "dataset": dataset,
        "n_centroids": len(cent),
        "n_tissues": cent["tissue"].nunique(),
        "n_lineages": cent["lineage"].nunique(),
        "n_test_predictions": len(pred),
        "loto_balanced_accuracy": obs_bal,
        "loto_mean_true_rank": obs_rank,
        "null_balanced_accuracy_mean": float(np.nanmean(null_bal)),
        "null_balanced_accuracy_q95": float(np.nanquantile(null_bal, .95)),
        "perm_p_accuracy_upper": perm_p(obs_bal, null_bal, greater=True),
        "null_mean_rank_mean": float(np.nanmean(null_rank)),
        "null_mean_rank_q05": float(np.nanquantile(null_rank, .05)),
        "perm_p_rank_lower": float(
            (1 + np.sum(null_rank <= obs_rank + 1e-12))
            / (np.sum(np.isfinite(null_rank)) + 1)
        ) if np.isfinite(obs_rank) else np.nan,
    }

    null_df = pd.DataFrame({
        "dataset": dataset,
        "draw": np.arange(1, n_perm + 1),
        "balanced_accuracy": null_bal,
        "mean_true_rank": null_rank,
    })
    return result, pred, null_df


# =============================================================================
# 5. SAME-LINEAGE CROSS-TISSUE STABILITY
# =============================================================================

def stability_score(cent):
    vals = []
    rows = []
    for lin, d in cent.groupby("lineage", observed=True):
        if len(d) < 2:
            continue
        X = d[ARMS].to_numpy(float)
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                c = cosine(X[i], X[j])
                if np.isfinite(c):
                    vals.append(c)
                    rows.append({
                        "lineage": lin,
                        "tissue_a": d.iloc[i]["tissue"],
                        "tissue_b": d.iloc[j]["tissue"],
                        "cosine_similarity": c,
                    })
    return (float(np.median(vals)) if vals else np.nan), pd.DataFrame(rows)


def stability_permutation(dataset, cent, n_perm=N_PERM):
    obs, pair = stability_score(cent)
    rng = np.random.default_rng(SEED + 777 + abs(hash(dataset)) % 100000)
    null = np.full(n_perm, np.nan)
    for b in range(n_perm):
        p = permute_labels_within_tissue(cent, rng)
        null[b], _ = stability_score(p)

    res = {
        "dataset": dataset,
        "median_same_lineage_cross_tissue_cosine": obs,
        "null_mean": float(np.nanmean(null)),
        "null_q95": float(np.nanquantile(null, .95)),
        "perm_p_upper": perm_p(obs, null, greater=True),
        "n_pairs": len(pair),
    }
    return res, pair, pd.DataFrame({
        "dataset": dataset,
        "draw": np.arange(1, n_perm + 1),
        "median_cosine": null,
    })


# =============================================================================
# 6. CROSS-ATLAS PROTOTYPE CONCORDANCE
# =============================================================================

def residual_lineage_prototype(cent):
    """
    Remove tissue mean within each tissue, then average residual architecture
    by lineage. Finally z-score each arm across lineages.
    """
    d = cent.copy()
    for c in ARMS:
        d[c] = d[c] - d.groupby("tissue")[c].transform("mean")

    p = (
        d.groupby("lineage", observed=True)[ARMS]
        .mean()
        .reindex(LINEAGES)
    )
    for c in ARMS:
        v = p[c].to_numpy(float)
        sd = np.nanstd(v)
        p[c] = (v - np.nanmean(v)) / sd if np.isfinite(sd) and sd > 0 else 0.0
    return p


def matrix_corr(A, B):
    common = [x for x in LINEAGES if x in A.index and x in B.index]
    a = A.loc[common, ARMS].to_numpy(float).ravel()
    b = B.loc[common, ARMS].to_numpy(float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    return float(pearsonr(a[ok], b[ok]).statistic)


def exact_lineage_permutation(A, B):
    common = [x for x in LINEAGES if x in A.index and x in B.index]
    obs = matrix_corr(A.loc[common], B.loc[common])
    vals = []
    for perm in itertools.permutations(common):
        Bp = B.loc[list(perm), ARMS].copy()
        Bp.index = common
        vals.append(matrix_corr(A.loc[common], Bp))
    vals = np.asarray(vals, float)
    p = (np.sum(vals >= obs - 1e-12)) / len(vals)
    return obs, float(p), vals


# =============================================================================
# 7. TIME-STAMPED GSE292219: LINEAGE AFTER ZT REMOVAL
# =============================================================================

def gse_residualize_zt(gse):
    d = zscore_columns(gse.copy(), ARMS)
    for c in ARMS:
        d[c] = d[c] - d.groupby("ZT")[c].transform("mean")
    return d


def zt_loto_score(d):
    rows = []
    for zt in sorted(d["ZT"].unique()):
        train = d[d["ZT"].ne(zt)]
        test = d[d["ZT"].eq(zt)]
        proto = train.groupby("lineage", observed=True)[ARMS].mean()
        if len(proto) < 2:
            continue
        for _, r in test.iterrows():
            true = r["lineage"]
            if true not in proto.index:
                continue
            x = r[ARMS].to_numpy(float)
            dist = {
                lin: float(np.linalg.norm(x - proto.loc[lin, ARMS].to_numpy(float)))
                for lin in proto.index
            }
            pred = min(dist, key=dist.get)
            rank = sorted(dist, key=dist.get).index(true) + 1
            rows.append({"ZT": zt, "true": true, "pred": pred, "true_rank": rank})
    pred = pd.DataFrame(rows)
    if pred.empty:
        return np.nan, np.nan, pred
    return (
        float(balanced_accuracy_score(pred["true"], pred["pred"])),
        float(pred["true_rank"].mean()),
        pred,
    )


def gse_permutation(d, n_perm=N_PERM):
    obs_bal, obs_rank, pred = zt_loto_score(d)
    rng = np.random.default_rng(SEED + 4242)
    null_bal = np.full(n_perm, np.nan)
    null_rank = np.full(n_perm, np.nan)

    for b in range(n_perm):
        p = d.copy()
        for zt, g in p.groupby("ZT"):
            idx = g.index.to_numpy()
            lab = g["lineage"].to_numpy(copy=True)
            rng.shuffle(lab)
            p.loc[idx, "lineage"] = lab
        null_bal[b], null_rank[b], _ = zt_loto_score(p)

    return {
        "dataset": "GSE292219",
        "zt_residual_loto_balanced_accuracy": obs_bal,
        "zt_residual_loto_mean_true_rank": obs_rank,
        "null_accuracy_mean": float(np.nanmean(null_bal)),
        "null_accuracy_q95": float(np.nanquantile(null_bal, .95)),
        "perm_p_accuracy_upper": perm_p(obs_bal, null_bal, True),
        "null_rank_mean": float(np.nanmean(null_rank)),
        "perm_p_rank_lower": float(
            (1 + np.sum(null_rank <= obs_rank + 1e-12))
            / (np.sum(np.isfinite(null_rank)) + 1)
        ),
        "n_predictions": len(pred),
    }, pred, pd.DataFrame({
        "draw": np.arange(1, n_perm + 1),
        "balanced_accuracy": null_bal,
        "mean_true_rank": null_rank,
    })


def gse_effect_decomp(gse):
    d = zscore_columns(gse.copy(), ARMS)
    d = d.rename(columns={"ZT": "tissue"})
    d["donor_id"] = "mouse_timecourse"
    Y = d[ARMS].to_numpy(float)

    full = ["lineage", "tissue"]
    lin = unique_partial_r2(Y, full, ["tissue"], d)
    tim = unique_partial_r2(Y, full, ["lineage"], d)

    Xa = design_matrix(d, full)
    Xi = design_matrix(d, full + ["lineage_x_tissue"])
    Qa, _ = orthobasis(Xa)
    Qi, _ = orthobasis(Xi)
    sa = pooled_sse(Y, Qa)
    si = pooled_sse(Y, Qi)
    inter = max(0, (sa - si) / sa) if sa > 0 else np.nan

    return {
        "dataset": "GSE292219",
        "lineage_unique_partial_R2_after_ZT": lin,
        "ZT_unique_partial_R2_after_lineage": tim,
        "lineage_x_ZT_increment_partial_R2": inter,
    }


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    log("=" * 100)
    log("78 LINEAGE-INTRINSIC 3-LOOP VALIDATION")
    log("=" * 100)

    inputs = {
        "TABULA": str(TABULA),
        "HCL": str(HCL),
        "AHCA": str(AHCA),
        "GSE292219": str(GSE292),
    }
    missing = [k for k, p in inputs.items() if not Path(p).exists()]
    write_json(
        {k: {"path": v, "exists": Path(v).exists()} for k, v in inputs.items()},
        "78_INPUT_AUDIT.json"
    )
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    log("Loading Tabula / HCL / AHCA...")
    datasets = {
        "Tabula": load_tabula(),
        "HCL": derive_arms_from_expanded17(HCL, "HCL"),
        "AHCA": derive_arms_from_expanded17(AHCA, "AHCA"),
    }

    for name, d in datasets.items():
        log(
            f"{name}: rows={len(d)}, donors={d.donor_id.nunique()}, "
            f"tissues={d.tissue.nunique()}, lineages={d.lineage.nunique()}"
        )
        write_csv(d, f"{name}_DONOR_TISSUE_LINEAGE_SIXARM.csv")

    # -------------------------------------------------------------------------
    # A. Effect decomposition
    # -------------------------------------------------------------------------
    log("A. Multivariate unique-effect decomposition...")
    effects = pd.DataFrame([
        effect_decomposition(name, d)
        for name, d in datasets.items()
    ])
    write_csv(effects, "78_A_MULTIVARIATE_EFFECT_DECOMPOSITION.csv")

    # -------------------------------------------------------------------------
    # B/C. Cross-tissue portability + stability
    # -------------------------------------------------------------------------
    log("B/C. Tissue-held-out portability + same-lineage stability...")
    port_rows = []
    stab_rows = []
    centroids = {}
    proto = {}

    for name, d in datasets.items():
        c = tissue_lineage_centroids(d)
        centroids[name] = c
        write_csv(c, f"{name}_TISSUE_LINEAGE_CENTROIDS.csv")

        pr, pred, nullp = portability_permutation(name, c)
        port_rows.append(pr)
        write_csv(pred, f"{name}_LOTO_PREDICTIONS.csv")
        nullp.to_csv(
            TABLE / f"{name}_LOTO_NULL.csv.gz",
            index=False,
            compression="gzip"
        )

        sr, pair, nulls = stability_permutation(name, c)
        stab_rows.append(sr)
        write_csv(pair, f"{name}_SAME_LINEAGE_CROSS_TISSUE_PAIRS.csv")
        nulls.to_csv(
            TABLE / f"{name}_STABILITY_NULL.csv.gz",
            index=False,
            compression="gzip"
        )

        proto[name] = residual_lineage_prototype(c)
        proto[name].reset_index().to_csv(
            TABLE / f"{name}_TISSUE_RESIDUAL_LINEAGE_PROTOTYPE.csv",
            index=False
        )

    portability = pd.DataFrame(port_rows)
    stability = pd.DataFrame(stab_rows)
    write_csv(portability, "78_B_TISSUE_HELD_OUT_LINEAGE_PORTABILITY.csv")
    write_csv(stability, "78_C_CROSS_TISSUE_STABILITY.csv")

    # -------------------------------------------------------------------------
    # D. Cross-atlas prototype concordance
    # -------------------------------------------------------------------------
    log("D. Cross-atlas prototype concordance...")
    cross_rows = []
    for ext in ["HCL", "AHCA"]:
        r, p, null = exact_lineage_permutation(proto["Tabula"], proto[ext])
        cross_rows.append({
            "comparison": f"Tabula_vs_{ext}",
            "prototype_matrix_pearson_r": r,
            "exact_lineage_permutation_p_upper": p,
            "n_exact_permutations": len(null),
        })
        pd.DataFrame({
            "comparison": f"Tabula_vs_{ext}",
            "null_r": null
        }).to_csv(
            TABLE / f"Tabula_vs_{ext}_PROTOTYPE_EXACT_NULL.csv",
            index=False
        )
    cross = pd.DataFrame(cross_rows)
    write_csv(cross, "78_D_CROSS_ATLAS_PROTOTYPE_CONCORDANCE.csv")

    # -------------------------------------------------------------------------
    # E. Time-stamped GSE292219
    # -------------------------------------------------------------------------
    log("E. GSE292219 time-stamped validation...")
    gse = load_gse292()
    gres = gse_residualize_zt(gse)
    write_csv(gres, "GSE292219_ZT_RESIDUAL_SIXARM.csv")

    gperm, gpred, gnull = gse_permutation(gres)
    write_csv(pd.DataFrame([gperm]), "78_E_GSE292219_ZT_RESIDUAL_PORTABILITY.csv")
    write_csv(gpred, "GSE292219_ZT_LOTO_PREDICTIONS.csv")
    gnull.to_csv(
        TABLE / "GSE292219_ZT_LOTO_NULL.csv.gz",
        index=False,
        compression="gzip"
    )

    geff = gse_effect_decomp(gse)
    write_csv(pd.DataFrame([geff]), "78_E_GSE292219_EFFECT_DECOMPOSITION.csv")

    # Compare time-residualized mouse-liver lineage prototype with Tabula.
    gp = (
        gres.groupby("lineage", observed=True)[ARMS]
        .mean()
        .reindex(LINEAGES)
    )
    for c in ARMS:
        v = gp[c].to_numpy(float)
        sd = np.nanstd(v)
        gp[c] = (v - np.nanmean(v)) / sd if np.isfinite(sd) and sd > 0 else 0.0
    gp.reset_index().to_csv(
        TABLE / "GSE292219_ZT_RESIDUAL_LINEAGE_PROTOTYPE.csv",
        index=False
    )
    g_r, g_p, g_null = exact_lineage_permutation(proto["Tabula"], gp)
    g_cross = pd.DataFrame([{
        "comparison": "Tabula_vs_GSE292219_ZT_residual",
        "prototype_matrix_pearson_r": g_r,
        "exact_lineage_permutation_p_upper": g_p,
        "n_exact_permutations": len(g_null),
    }])
    write_csv(g_cross, "78_E_TABULA_VS_GSE292219_PROTOTYPE.csv")

    # -------------------------------------------------------------------------
    # F. Decision gate
    # -------------------------------------------------------------------------
    log("F. Decision gate...")

    tab_port = portability[portability.dataset.eq("Tabula")].iloc[0]
    hcl_port = portability[portability.dataset.eq("HCL")].iloc[0]
    ahca_port = portability[portability.dataset.eq("AHCA")].iloc[0]

    tab_pass = bool(tab_port.perm_p_accuracy_upper < 0.05)
    hcl_pass = bool(hcl_port.perm_p_accuracy_upper < 0.05)
    ahca_pass = bool(ahca_port.perm_p_accuracy_upper < 0.05)
    ext_port_pass = hcl_pass or ahca_pass

    cross_passes = [
        bool(r.exact_lineage_permutation_p_upper < 0.05 and r.prototype_matrix_pearson_r > 0)
        for _, r in cross.iterrows()
    ]
    ext_proto_pass = any(cross_passes)

    gse_pass = bool(gperm["perm_p_accuracy_upper"] < 0.05)

    if tab_pass and ext_port_pass and ext_proto_pass and gse_pass:
        verdict = "STRONG_SUPPORT"
    elif tab_pass and (ext_port_pass or ext_proto_pass) and gse_pass:
        verdict = "MODERATE_SUPPORT"
    elif tab_pass and (ext_port_pass or gse_pass):
        verdict = "PARTIAL_SUPPORT"
    else:
        verdict = "NOT_SUPPORTED"

    strict_lineage_gt_tissue_count = int(effects["strict_lineage_gt_tissue"].sum())
    strict_claim = (
        "SUPPORTED_IN_MAJORITY_OF_STATIC_ATLASES"
        if strict_lineage_gt_tissue_count >= 2
        else "NOT_SUPPORTED_AS_GENERAL_RULE"
    )

    gate = pd.DataFrame([
        {
            "criterion": "Tabula tissue-held-out lineage portability",
            "pass": tab_pass,
            "value": (
                f"balanced_acc={tab_port.loto_balanced_accuracy:.3f}; "
                f"permP={tab_port.perm_p_accuracy_upper:.4g}"
            ),
        },
        {
            "criterion": "Independent human atlas portability (HCL or AHCA)",
            "pass": ext_port_pass,
            "value": (
                f"HCL P={hcl_port.perm_p_accuracy_upper:.4g}; "
                f"AHCA P={ahca_port.perm_p_accuracy_upper:.4g}"
            ),
        },
        {
            "criterion": "Cross-atlas lineage prototype concordance",
            "pass": ext_proto_pass,
            "value": "; ".join(
                f"{r.comparison}: r={r.prototype_matrix_pearson_r:.3f}, "
                f"P={r.exact_lineage_permutation_p_upper:.4g}"
                for _, r in cross.iterrows()
            ),
        },
        {
            "criterion": "Time-stamped lineage structure after explicit ZT removal",
            "pass": gse_pass,
            "value": (
                f"balanced_acc={gperm['zt_residual_loto_balanced_accuracy']:.3f}; "
                f"permP={gperm['perm_p_accuracy_upper']:.4g}"
            ),
        },
    ])
    write_csv(gate, "78_F_DECISION_GATE.csv")

    summary = {
        "primary_verdict": verdict,
        "strict_lineage_gt_tissue_claim": strict_claim,
        "n_static_atlases_lineage_unique_R2_gt_tissue_unique_R2": strict_lineage_gt_tissue_count,
        "tabula_portability_pass": tab_pass,
        "external_portability_pass": ext_port_pass,
        "external_prototype_concordance_pass": ext_proto_pass,
        "gse292219_zt_residual_lineage_pass": gse_pass,
    }
    (OUT / "78_COMPLETE.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------
    log("Drawing figures...")

    # F1 effects
    d = effects.set_index("dataset").loc[["Tabula", "HCL", "AHCA"]]
    x = np.arange(len(d))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    ax.bar(x - w, d["lineage_unique_partial_R2"], width=w, label="Lineage")
    ax.bar(x, d["tissue_unique_partial_R2"], width=w, label="Tissue")
    ax.bar(x + w, d["lineage_x_tissue_increment_partial_R2"], width=w, label="Lineage × Tissue")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index)
    ax.set_ylabel("Unique / incremental multivariate partial R²")
    ax.set_title("F1. Lineage contribution persists after tissue adjustment")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "78_F1_LINEAGE_TISSUE_EFFECTS")

    # F2 portability
    d = portability.set_index("dataset").loc[["Tabula", "HCL", "AHCA"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    y = d["loto_balanced_accuracy"].to_numpy(float)
    null = d["null_balanced_accuracy_mean"].to_numpy(float)
    q95 = d["null_balanced_accuracy_q95"].to_numpy(float)
    x = np.arange(len(d))
    ax.bar(x, y, label="Observed")
    ax.scatter(x, null, marker="o", s=45, label="Permutation mean")
    ax.scatter(x, q95, marker="_", s=180, label="Permutation 95th percentile")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, min(1.02, y[i] + .035), f"P={r.perm_p_accuracy_upper:.3g}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(d.index)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Tissue-held-out balanced accuracy")
    ax.set_title("F2. Can lineage architecture transfer to unseen tissues?")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "78_F2_TISSUE_HELD_OUT_PORTABILITY")

    # F3 prototype matrices
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), constrained_layout=True)
    allv = np.concatenate([proto[k].to_numpy(float).ravel() for k in ["Tabula", "HCL", "AHCA"]])
    vmax = max(abs(np.nanpercentile(allv, 2)), abs(np.nanpercentile(allv, 98)))
    for ax, name in zip(axes, ["Tabula", "HCL", "AHCA"]):
        M = proto[name].loc[LINEAGES, ARMS].to_numpy(float)
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(ARMS)))
        ax.set_xticklabels(ARMS, rotation=55, ha="right")
        ax.set_yticks(np.arange(len(LINEAGES)))
        ax.set_yticklabels([LINEAGE_DISPLAY[x] for x in LINEAGES])
        ax.set_title(name)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.02,
                 label="Tissue-residual lineage prototype (z)")
    fig.suptitle("F3. Cross-atlas lineage-specific six-arm architecture")
    savefig(fig, "78_F3_CROSS_ATLAS_LINEAGE_PROTOTYPES")

    # F4 time-stamped
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), constrained_layout=True)
    ge = geff
    vals = [
        ge["lineage_unique_partial_R2_after_ZT"],
        ge["ZT_unique_partial_R2_after_lineage"],
        ge["lineage_x_ZT_increment_partial_R2"],
    ]
    axes[0].bar(["Lineage|ZT", "ZT|Lineage", "L×ZT"], vals)
    axes[0].set_ylabel("Multivariate partial R²")
    axes[0].set_title("GSE292219 effect decomposition")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    obs = gperm["zt_residual_loto_balanced_accuracy"]
    nullm = gperm["null_accuracy_mean"]
    q95 = gperm["null_accuracy_q95"]
    axes[1].bar([0], [obs], width=.55)
    axes[1].scatter([0], [nullm], s=50, label="Permutation mean")
    axes[1].scatter([0], [q95], s=180, marker="_", label="Permutation 95th pct")
    axes[1].text(0, min(1.02, obs + .05),
                 f"P={gperm['perm_p_accuracy_upper']:.3g}",
                 ha="center")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["ZT-residual\nlineage prediction"])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Lineage survives explicit time removal")
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.suptitle("F4. Shared circadian phase does not imply identical lineage architecture")
    savefig(fig, "78_F4_TIME_STAMPED_VALIDATION")

    # F5 gate
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    y = np.arange(len(gate))
    vals = gate["pass"].astype(int).to_numpy()
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(gate["criterion"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Gate pass")
    ax.set_title(f"F5. Biological architecture gate: {verdict}")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "78_F5_DECISION_GATE")

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    lines = []
    lines += [
        "# 78 Lineage-intrinsic 3-loop architecture validation",
        "",
        f"## Primary verdict: **{verdict}**",
        "",
        "This analysis asks whether the five major lineages carry a stable six-arm "
        "(E-box / RORE / D-box) clock architecture across tissues.",
        "",
        "## A. Static human-atlas multivariate effects",
        "",
        df_to_markdown(effects),
        "",
        "Interpretation rule: the broad claim does **not** require Lineage > Tissue. "
        "The key requirement is a non-zero lineage contribution after accounting for "
        "tissue/donor plus portability to unseen tissues. The stricter statement "
        f"`Lineage > Tissue` is: **{strict_claim}**.",
        "",
        "## B. Tissue-held-out lineage portability",
        "",
        df_to_markdown(portability),
        "",
        "## C. Same-lineage cross-tissue stability",
        "",
        df_to_markdown(stability),
        "",
        "## D. Cross-atlas residual lineage prototypes",
        "",
        df_to_markdown(cross),
        "",
        "## E. Time-stamped GSE292219",
        "",
        df_to_markdown(pd.DataFrame([geff])),
        "",
        df_to_markdown(pd.DataFrame([gperm])),
        "",
        df_to_markdown(g_cross),
        "",
        "## F. Pre-specified decision gate",
        "",
        df_to_markdown(gate),
        "",
        "## Biological interpretation boundary",
        "",
        "- If STRONG/MODERATE support is obtained, it is defensible to proceed to "
        "lineage-TF/regulon/motif analysis.",
        "- If only PARTIAL support is obtained, TF analysis may be exploratory but "
        "should not be framed as a mechanism of a conserved lineage architecture.",
        "- If NOT_SUPPORTED, stop the lineage-intrinsic mechanism branch rather than "
        "searching TFs post hoc.",
        "",
        "No rhythmicity claim is made from Tabula/HCL/AHCA. The time-dependent claim "
        "comes only from GSE292219.",
    ]

    (REPORT / "78_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log(f"78 COMPLETE — {verdict}")
    log(f"Report: {REPORT / '78_FINAL_REPORT.md'}")
    log(f"Summary: {OUT / '78_COMPLETE.json'}")
    log("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        fail = {
            "status": "FAILED",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (OUT / "78_FAILED.json").write_text(
            json.dumps(fail, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(traceback.format_exc(), flush=True)
        raise
