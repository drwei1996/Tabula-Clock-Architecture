#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tabula/SCP259 biology-extension overnight pipeline
--------------------------------------------------
Purpose:
1) Regenerate modified panels that need refreshing.
2) Add the biology-oriented supplementary panels proposed in the figure review.
3) Save every panel in both text and no-text versions.
4) Use real local CSV outputs already present in /mnt/data.

Author: ChatGPT (generated for user)
Date: 2026-10-20
"""

from __future__ import annotations
import os
import math
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT = SCRIPT_DIR.parent


def configure_paths():
    parser = argparse.ArgumentParser(description="Tabula + SCP259 biology panels (local WSL, real bundled CSV)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_PROJECT,
                        help="katv2 project directory (default: parent of extracted script folder)")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "INPUT_DATA",
                        help="Source tables, exact exported user-provided data included in ZIP")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Output directory (default: <data-root>/TABULA_SCP259_BIOLOGY_EXTENSION_V2)")
    args = parser.parse_args()
    project = args.data_root.expanduser().resolve()
    base = args.input_dir.expanduser().resolve()
    out = (args.outdir.expanduser().resolve() if args.outdir else
           project / "TABULA_SCP259_BIOLOGY_EXTENSION_V2")
    if not base.is_dir():
        raise FileNotFoundError(f"Input data folder not found: {base}")
    for p in (out, out / "PANELS", out / "TABLES", out / "LOGS"):
        p.mkdir(parents=True, exist_ok=True)
    return project, base, out


PROJECT_ROOT, BASE, OUT = configure_paths()
PANELS = OUT / "PANELS"
TABLES = OUT / "TABLES"
LOGS = OUT / "LOGS"

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

LINEAGE_ORDER = ["B cell", "T cell", "Macrophage/Monocyte", "Fibroblast", "Endothelial"]
LINEAGE_COLORS = {
    "B cell": "#5B8CC0",
    "T cell": "#D97A84",
    "Macrophage/Monocyte": "#E0A64B",
    "Fibroblast": "#79B8A6",
    "Endothelial": "#8F6BCB",
}
COND_ORDER = ["Healthy", "UC_noninflamed", "UC_inflamed"]
COND_COLORS = {
    "Healthy": "#5F89A2",
    "UC_noninflamed": "#D4A343",
    "UC_inflamed": "#CC6B73",
}
ARM_ORDER = ["EBOX_POS", "EBOX_NEG", "RORE_ACT", "RORE_REP", "DBOX_ACT", "DBOX_REP"]
ARM_LABELS = {
    "EBOX_POS": "E-box +",
    "EBOX_NEG": "E-box -",
    "RORE_ACT": "RORE +",
    "RORE_REP": "RORE -",
    "DBOX_ACT": "D-box +",
    "DBOX_REP": "D-box -",
}
ARM_SHAPE_COLS = [f"SHAPE_{a}" for a in ARM_ORDER]
CLOCK17_ORDER = ["ARNTL", "CLOCK", "NPAS2", "PER1", "PER2", "PER3", "CRY1", "CRY2", "RORA", "RORB", "RORC", "NR1D1", "NR1D2", "DBP", "TEF", "HLF", "NFIL3"]
ARM_TO_GENES = {
    "EBOX_POS": ["ARNTL", "CLOCK", "NPAS2"],
    "EBOX_NEG": ["PER1", "PER2", "PER3", "CRY1", "CRY2"],
    "RORE_ACT": ["RORA", "RORB", "RORC"],
    "RORE_REP": ["NR1D1", "NR1D2"],
    "DBOX_ACT": ["DBP", "TEF", "HLF"],
    "DBOX_REP": ["NFIL3"],
}

CURATED_PATHWAYS = {
    "B cell": [
        "PATH_HALLMARK_G2M_CHECKPOINT",
        "PATH_HALLMARK_E2F_TARGETS",
        "PATH_HALLMARK_MITOTIC_SPINDLE",
        "PATH_HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "PATH_HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "PATH_HALLMARK_INTERFERON_ALPHA_RESPONSE",
    ],
    "T cell": [
        "PATH_HALLMARK_IL2_STAT5_SIGNALING",
        "PATH_HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "PATH_HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "PATH_HALLMARK_G2M_CHECKPOINT",
        "PATH_HALLMARK_E2F_TARGETS",
        "PATH_HALLMARK_HYPOXIA",
    ],
    "Macrophage/Monocyte": [
        "PATH_HALLMARK_INFLAMMATORY_RESPONSE",
        "PATH_HALLMARK_COMPLEMENT",
        "PATH_HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "PATH_HALLMARK_HYPOXIA",
        "PATH_HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
        "PATH_HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    ],
    "Fibroblast": [
        "PATH_HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "PATH_HALLMARK_TGF_BETA_SIGNALING",
        "PATH_HALLMARK_PI3K_AKT_MTOR_SIGNALING",
        "PATH_HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
        "PATH_HALLMARK_MTORC1_SIGNALING",
        "PATH_HALLMARK_MYOGENESIS",
    ],
    "Endothelial": [
        "PATH_HALLMARK_ANGIOGENESIS",
        "PATH_HALLMARK_COAGULATION",
        "PATH_HALLMARK_HYPOXIA",
        "PATH_HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "PATH_HALLMARK_INFLAMMATORY_RESPONSE",
        "PATH_HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    ],
}
PATHWAY_DISPLAY = {
    k: k.replace("PATH_HALLMARK_", "").replace("_", " ") for v in CURATED_PATHWAYS.values() for k in v
}

SELECTED_UC_COMPONENTS = [
    ("B cell", "EBOX_POS"),
    ("Fibroblast", "EBOX_POS"),
    ("Fibroblast", "DBOX_ACT"),
    ("Endothelial", "EBOX_POS"),
]

B_CELL_TF_CANDIDATES = ["TF_REL", "TF_STAT5B", "TF_PAX5", "TF_ETS1", "TF_BCL6", "TF_NFKB2", "TF_RUNX3", "TF_SMAD3", "TF_STAT5A", "TF_BACH2", "TF_STAT3", "TF_IRF8", "TF_EBF1", "TF_PRDM1", "TF_NR4A1"]

RNG = np.random.default_rng(20261020)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.Series(x)
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean()) / sd


def ensure_columns(df: pd.DataFrame, cols: List[str], name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")


def bootstrap_ci_diff(a, b, paired=False, n_boot=4000, seed=20261020):
    rng = np.random.default_rng(seed)
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if paired:
        n = min(len(a), len(b))
        d = a[:n] - b[:n]
        if n == 0:
            return np.nan, np.nan, np.nan
        boot = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[i] = d[idx].mean()
        est = d.mean()
    else:
        if len(a) == 0 or len(b) == 0:
            return np.nan, np.nan, np.nan
        boot = np.empty(n_boot)
        for i in range(n_boot):
            ia = rng.integers(0, len(a), size=len(a))
            ib = rng.integers(0, len(b), size=len(b))
            boot[i] = a[ia].mean() - b[ib].mean()
        est = a.mean() - b.mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return est, lo, hi


def mannwhitney_p(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue


def paired_wilcoxon_p(a, b):
    dd = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(dd) < 3:
        return np.nan
    try:
        return stats.wilcoxon(dd["a"], dd["b"], alternative="two-sided", zero_method="wilcox").pvalue
    except Exception:
        return np.nan


def fdr_bh(pvals: pd.Series) -> np.ndarray:
    x = np.asarray(pvals, dtype=float)
    out = np.full(len(x), np.nan)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        return out
    out[mask] = multipletests(x[mask], method="fdr_bh")[1]
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, covars: List[str]) -> Tuple[float, float, int]:
    dd = df[[x, y] + covars].dropna().copy()
    if len(dd) < 6:
        return np.nan, np.nan, len(dd)
    X = dd[[x] + covars].rank(method="average")
    Y = dd[[y] + covars].rank(method="average")
    xr = X.iloc[:, 0].to_numpy(dtype=float)
    yr = Y.iloc[:, 0].to_numpy(dtype=float)
    C = X.iloc[:, 1:].to_numpy(dtype=float)
    if C.shape[1] == 0:
        rho, p = stats.spearmanr(xr, yr)
        return float(rho), float(p), len(dd)
    C = np.column_stack([np.ones(len(C)), C])
    bx, *_ = np.linalg.lstsq(C, xr, rcond=None)
    by, *_ = np.linalg.lstsq(C, yr, rcond=None)
    rx = xr - C @ bx
    ry = yr - C @ by
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, len(dd)
    r = np.corrcoef(rx, ry)[0, 1]
    dfree = len(dd) - C.shape[1] - 1
    if dfree <= 0 or abs(r) >= 1:
        p = np.nan
    else:
        t = r * math.sqrt(dfree / max(1e-12, 1 - r**2))
        p = 2 * stats.t.sf(abs(t), dfree)
    return float(r), float(p), len(dd)


def show_sig(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def clean_pathway_name(x: str) -> str:
    return x.replace("PATH_HALLMARK_", "").replace("HALLMARK_", "").replace("_", " ")


def save_figure(fig: plt.Figure, stem: str):
    fig.savefig(PANELS / f"{stem}.png")
    fig.savefig(PANELS / f"{stem}.pdf")
    plt.close(fig)


def add_condition_legend(ax, show_text=True):
    if not show_text:
        return
    handles = [Line2D([0], [0], color=COND_COLORS[c], lw=2, marker='o', markersize=6, label=c.replace('_',' ')) for c in COND_ORDER]
    ax.legend(handles=handles, frameon=False, loc="best")


def finalize_axes(ax, show_text=True, title=None, xlabel=None, ylabel=None):
    if show_text:
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    else:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
def load_data():
    d = {}
    d["tabula_profile"] = pd.read_csv(BASE / "01_FROZEN_TABULA_STAGE78_PROFILE.csv")
    d["scp_healthy_profile"] = pd.read_csv(BASE / "03_SCP259_HEALTHY_PROFILE.csv")
    d["donor_scores"] = pd.read_csv(BASE / "02_DONOR_CONDITION_LINEAGE_SCORES.csv")
    d["all_tests"] = pd.read_csv(BASE / "04_ALL_TESTS_WITH_FDR.csv")
    d["shape_tests"] = pd.read_csv(BASE / "03_SIX_ARM_AND_GENE_TESTS.csv")
    d["tf_assoc_existing"] = pd.read_csv(BASE / "SCP259_FIGURE_EXTENSION_REAL_DATA/04_TF_AND_PATHWAY_ARM_SHAPE_ASSOCIATIONS.csv")
    d["tf_path_assoc_existing"] = pd.read_csv(BASE / "SCP259_FIGURE_EXTENSION_REAL_DATA/05_BCELL_TF_PATHWAY_ASSOCIATIONS.csv")
    return d

# -----------------------------------------------------------------------------
# Derived tables
# -----------------------------------------------------------------------------
def derive_normal_comparison(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    a = data["tabula_profile"].copy()
    a["atlas"] = "Tabula all tissues"
    b = data["scp_healthy_profile"].copy()
    b["atlas"] = "SCP259 Healthy colon"
    long = pd.concat([a, b], ignore_index=True)
    long = long.melt(id_vars=["lineage", "atlas"], value_vars=ARM_ORDER, var_name="arm", value_name="score")
    long["lineage"] = pd.Categorical(long["lineage"], categories=LINEAGE_ORDER, ordered=True)
    long["arm"] = pd.Categorical(long["arm"], categories=ARM_ORDER, ordered=True)
    return long


def derive_healthy_gene_architecture(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["donor_scores"].copy()
    h = df[df["condition"] == "Healthy"].copy()
    gene_cols = [f"GENE_{g}" for g in CLOCK17_ORDER]
    ensure_columns(h, gene_cols, "donor_scores")
    out = h.groupby("lineage")[gene_cols].mean().reindex(LINEAGE_ORDER)
    out.columns = [c.replace("GENE_", "") for c in out.columns]
    out = out.loc[:, CLOCK17_ORDER]
    return out


def derive_uc_gene_component_effects(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["donor_scores"].copy()
    rows = []
    for lineage, arm in SELECTED_UC_COMPONENTS:
        genes = ARM_TO_GENES[arm]
        sub = df[df["lineage"] == lineage].copy()
        ref = sub[sub["condition"] == "Healthy"]
        dis = sub[sub["condition"] == "UC_inflamed"]
        for g in genes:
            col = f"GENE_{g}"
            est, lo, hi = bootstrap_ci_diff(dis[col], ref[col], paired=False)
            p = mannwhitney_p(dis[col], ref[col])
            rows.append({
                "lineage": lineage,
                "arm": arm,
                "gene": g,
                "delta": est,
                "ci_low": lo,
                "ci_high": hi,
                "p": p,
                "n_ref": ref[col].notna().sum(),
                "n_dis": dis[col].notna().sum(),
            })
    out = pd.DataFrame(rows)
    out["q_bh"] = fdr_bh(out["p"])
    return out


def derive_uc_pathway_effects(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["donor_scores"].copy()
    rows = []
    for lineage in LINEAGE_ORDER:
        sub = df[df["lineage"] == lineage].copy()
        ref = sub[sub["condition"] == "Healthy"]
        dis = sub[sub["condition"] == "UC_inflamed"]
        for pw in CURATED_PATHWAYS[lineage]:
            est, lo, hi = bootstrap_ci_diff(dis[pw], ref[pw], paired=False)
            p = mannwhitney_p(dis[pw], ref[pw])
            rows.append({
                "lineage": lineage,
                "pathway": pw,
                "delta": est,
                "ci_low": lo,
                "ci_high": hi,
                "p": p,
                "n_ref": ref[pw].notna().sum(),
                "n_dis": dis[pw].notna().sum(),
            })
    out = pd.DataFrame(rows)
    out["q_bh_within_lineage"] = out.groupby("lineage")["p"].transform(fdr_bh)
    return out


def derive_arm_pathway_coupling(data: Dict[str, pd.DataFrame], condition: str) -> pd.DataFrame:
    df = data["donor_scores"].copy()
    out = []
    for lineage in LINEAGE_ORDER:
        sub = df[(df["condition"] == condition) & (df["lineage"] == lineage)].copy()
        if len(sub) < 6:
            continue
        if "n_cells" not in sub.columns:
            sub["n_cells"] = 1.0
        sub["log_n_cells"] = np.log1p(sub["n_cells"])
        for arm in ARM_SHAPE_COLS:
            for pw in CURATED_PATHWAYS[lineage]:
                rho, p, n = partial_spearman(sub, arm, pw, ["ARM_GLOBAL", "log_n_cells"])
                out.append({
                    "condition": condition,
                    "lineage": lineage,
                    "arm": arm,
                    "pathway": pw,
                    "rho": rho,
                    "p": p,
                    "n": n,
                })
    out = pd.DataFrame(out)
    if not out.empty:
        out["q_bh"] = out.groupby(["condition", "lineage"])["p"].transform(fdr_bh)
    return out


def derive_uc_bcell_tf_bridge(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = data["donor_scores"].copy()
    b = df[(df["lineage"] == "B cell") & (df["condition"].isin(["Healthy", "UC_inflamed"]))].copy()
    b["log_n_cells"] = np.log1p(b["n_cells"])
    # expression difference
    rows1 = []
    for tf in B_CELL_TF_CANDIDATES:
        est, lo, hi = bootstrap_ci_diff(
            b.loc[b["condition"] == "UC_inflamed", tf],
            b.loc[b["condition"] == "Healthy", tf],
            paired=False,
        )
        p = mannwhitney_p(
            b.loc[b["condition"] == "UC_inflamed", tf],
            b.loc[b["condition"] == "Healthy", tf],
        )
        rows1.append({"TF": tf, "delta": est, "ci_low": lo, "ci_high": hi, "p": p})
    tf_diff = pd.DataFrame(rows1)
    tf_diff["q_bh"] = fdr_bh(tf_diff["p"])
    tf_diff["TF_clean"] = tf_diff["TF"].str.replace("TF_", "", regex=False)
    # within inflamed partial correlations to B-cell EBOX_POS shape
    infl = df[(df["lineage"] == "B cell") & (df["condition"] == "UC_inflamed")].copy()
    infl["log_n_cells"] = np.log1p(infl["n_cells"])
    rows2 = []
    for tf in B_CELL_TF_CANDIDATES:
        rho, p, n = partial_spearman(infl, "SHAPE_EBOX_POS", tf, ["ARM_GLOBAL", "log_n_cells"])
        rows2.append({"TF": tf, "rho": rho, "p": p, "n": n})
    tf_cor = pd.DataFrame(rows2)
    tf_cor["q_bh"] = fdr_bh(tf_cor["p"])
    tf_cor["TF_clean"] = tf_cor["TF"].str.replace("TF_", "", regex=False)
    tf_cor = tf_cor.sort_values("rho", ascending=False)
    return tf_diff, tf_cor


def derive_uc_paired_shape(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["donor_scores"].copy()
    uc = df[df["condition"].isin(["UC_noninflamed", "UC_inflamed"])].copy()
    rows = []
    for lineage in LINEAGE_ORDER:
        sub = uc[uc["lineage"] == lineage].copy()
        common = sorted(set(sub.loc[sub["condition"] == "UC_noninflamed", "Subject"]) & set(sub.loc[sub["condition"] == "UC_inflamed", "Subject"]))
        if len(common) < 3:
            continue
        for arm in ARM_SHAPE_COLS:
            piv = sub[sub["Subject"].isin(common)].pivot_table(index="Subject", columns="condition", values=arm, aggfunc="first")
            piv = piv[["UC_inflamed", "UC_noninflamed"]].dropna()
            est, lo, hi = bootstrap_ci_diff(piv["UC_inflamed"], piv["UC_noninflamed"], paired=True)
            p = paired_wilcoxon_p(piv["UC_inflamed"], piv["UC_noninflamed"])
            rows.append({
                "lineage": lineage,
                "arm": arm,
                "delta": est,
                "ci_low": lo,
                "ci_high": hi,
                "p": p,
                "n_pairs": len(piv),
            })
    out = pd.DataFrame(rows)
    out["q_bh"] = out.groupby("lineage")["p"].transform(fdr_bh)
    return out


def derive_uc_clock17_heatmap(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = data["donor_scores"].copy()
    gene_cols = [f"GENE_{g}" for g in CLOCK17_ORDER]
    zdf = []
    for lineage in LINEAGE_ORDER:
        sub = df[df["lineage"] == lineage].copy()
        h = sub[sub["condition"] == "Healthy"][gene_cols].mean()
        sd = sub[sub["condition"] == "Healthy"][gene_cols].std(ddof=0).replace(0, np.nan)
        for cond in COND_ORDER:
            m = sub[sub["condition"] == cond][gene_cols].mean()
            z = (m - h) / sd
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
            zz = pd.DataFrame({"lineage": lineage, "condition": cond, "gene": CLOCK17_ORDER, "z": [z.get(f"GENE_{g}", np.nan) for g in CLOCK17_ORDER]})
            zdf.append(zz)
    out = pd.concat(zdf, ignore_index=True)
    return out

# -----------------------------------------------------------------------------
# Panel functions
# -----------------------------------------------------------------------------
def panel_01_normal_two_atlas_heatmap(normal_long: pd.DataFrame, show_text=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, atlas in zip(axes, ["Tabula all tissues", "SCP259 Healthy colon"]):
        sub = normal_long[normal_long["atlas"] == atlas]
        mat = sub.pivot(index="lineage", columns="arm", values="score").loc[LINEAGE_ORDER, ARM_ORDER]
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax.set_xticks(np.arange(len(ARM_ORDER)))
        ax.set_yticks(np.arange(len(LINEAGE_ORDER)))
        ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER] if show_text else [])
        ax.set_yticklabels(LINEAGE_ORDER if show_text else [])
        if show_text:
            ax.set_title(atlas)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for spine in ax.spines.values():
            spine.set_visible(False)
    if show_text:
        fig.suptitle("P01 | Five-lineage normal transcriptional architecture in two atlases")
        cbar = fig.colorbar(im, ax=axes, shrink=0.85)
        cbar.set_label("Within-study normal-reference z; mean per arm")
    save_figure(fig, f"P01_normal_two_atlas_heatmap_{'text' if show_text else 'notext'}")


def panel_02_uc_shape_effects(data: Dict[str, pd.DataFrame], show_text=True):
    tt = data["shape_tests"].copy()
    tt = tt[(tt["family"] == "SHAPE") & (tt["contrast"] == "UC_inflamed__vs__Healthy")].copy()
    tt["lineage"] = pd.Categorical(tt["lineage"], LINEAGE_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True, constrained_layout=True)
    for ax, lineage in zip(axes, LINEAGE_ORDER):
        sub = tt[tt["lineage"] == lineage].set_index("feature").reindex([f"SHAPE_{a}" for a in ARM_ORDER]).reset_index()
        y = np.arange(len(sub))
        ax.axvline(0, color="#B0B8C0", lw=1)
        ax.errorbar(sub["delta"], y,
                    xerr=[sub["delta"] - sub["ci_low"], sub["ci_high"] - sub["delta"]],
                    fmt='o', color=LINEAGE_COLORS[lineage], ecolor=LINEAGE_COLORS[lineage], elinewidth=2, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABELS[s.replace("SHAPE_","")] for s in sub["feature"]] if show_text and ax is axes[0] else ([] if not show_text else []))
        if show_text:
            ax.set_title(lineage, color=LINEAGE_COLORS[lineage])
        ax.invert_yaxis()
        for _, r in sub.iterrows():
            if show_text and pd.notna(r["q_bh"]):
                sig = show_sig(r["q_bh"])
                if sig:
                    ax.text(r["ci_high"] + 0.03*np.nanmax(np.abs(sub["ci_high"])), y[sub.index[sub["feature"] == r["feature"]][0]], sig, va="center", color=LINEAGE_COLORS[lineage], fontsize=12)
    if show_text:
        fig.suptitle("P02 | UC inflamed vs Healthy: relative six-arm shape contrasts")
        axes[2].set_xlabel("Patient-level mean difference (shape arm score)")
        axes[0].set_ylabel("")
    save_figure(fig, f"P02_uc_shape_effects_{'text' if show_text else 'notext'}")


def panel_03_uc_bcell_donors(data: Dict[str, pd.DataFrame], show_text=True):
    df = data["donor_scores"].copy()
    b = df[df["lineage"] == "B cell"].copy()
    shape_tests = data["shape_tests"].copy()
    qmap = shape_tests[(shape_tests["family"] == "SHAPE") & (shape_tests["lineage"] == "B cell") & (shape_tests["contrast"] == "UC_inflamed__vs__Healthy")].set_index("feature")["q_bh"].to_dict()
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    for ax, arm in zip(axes, ARM_ORDER):
        col = f"SHAPE_{arm}"
        groups = [b[b["condition"] == c][col].dropna().values for c in COND_ORDER]
        for i, (cond, vals) in enumerate(zip(COND_ORDER, groups)):
            x = np.full(len(vals), i) + np.linspace(-0.13, 0.13, len(vals))
            ax.scatter(x, vals, s=35, color=COND_COLORS[cond], alpha=0.8)
            if len(vals) > 0:
                ax.hlines(np.mean(vals), i-0.22, i+0.22, color=COND_COLORS[cond], lw=2.5)
        ax.axhline(0, color="#D0D0D0", lw=1)
        if show_text:
            q = qmap.get(col, np.nan)
            ax.set_title(f"{ARM_LABELS[arm]} | q={q:.3g}" if pd.notna(q) else ARM_LABELS[arm])
            ax.set_xticks(range(3)); ax.set_xticklabels([c.replace('_',' ') for c in COND_ORDER], rotation=20, ha='right')
        else:
            ax.set_xticks(range(3)); ax.set_xticklabels([])
            ax.set_title("")
        if not show_text:
            ax.set_yticklabels([])
    if show_text:
        fig.suptitle("P03 | B-cell donor-level relative six-arm scores")
    save_figure(fig, f"P03_uc_bcell_donors_{'text' if show_text else 'notext'}")


def panel_04_uc_distance(data: Dict[str, pd.DataFrame], show_text=True):
    df = data["donor_scores"].copy()
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for li, lineage in enumerate(LINEAGE_ORDER):
        sub = df[df["lineage"] == lineage].copy()
        for ci, cond in enumerate(COND_ORDER):
            vals = sub[sub["condition"] == cond]["SIX_ARM_DISTANCE"].dropna().values
            x0 = li * 4 + ci
            x = np.full(len(vals), x0) + np.linspace(-0.18, 0.18, len(vals))
            ax.scatter(x, vals, s=40, color=COND_COLORS[cond], alpha=0.75)
            if len(vals) > 0:
                ax.hlines(np.mean(vals), x0-0.28, x0+0.28, color=COND_COLORS[cond], lw=2.5)
    centers = [i*4 + 1 for i in range(len(LINEAGE_ORDER))]
    ax.set_xticks(centers)
    ax.set_xticklabels(LINEAGE_ORDER if show_text else [])
    if show_text:
        ax.set_ylabel("Six-arm RMS distance from healthy centroid")
        ax.set_title("P04 | Healthy-centroid departures across the five lineages")
        add_condition_legend(ax, show_text=True)
    else:
        ax.set_yticklabels([])
    save_figure(fig, f"P04_uc_distance_{'text' if show_text else 'notext'}")


def panel_05_healthy_gene_architecture(gene_arch: pd.DataFrame, show_text=True):
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    mat = gene_arch.loc[LINEAGE_ORDER, CLOCK17_ORDER].values
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
    ax.set_xticks(np.arange(len(CLOCK17_ORDER)))
    ax.set_yticks(np.arange(len(LINEAGE_ORDER)))
    ax.set_xticklabels(CLOCK17_ORDER if show_text else [], rotation=60, ha='right')
    ax.set_yticklabels(LINEAGE_ORDER if show_text else [])
    if show_text:
        ax.set_title("P05 | Healthy SCP259 donor means: gene-level clock architecture")
        # arm boundary rectangles
        idx = 0
        for arm in ARM_ORDER:
            ng = len(ARM_TO_GENES[arm])
            ax.add_patch(Rectangle((idx-0.5, -0.5), ng, len(LINEAGE_ORDER), fill=False, lw=1.6, edgecolor="#666666"))
            ax.text(idx + (ng-1)/2, -1.0, ARM_LABELS[arm], ha="center", va="bottom", fontsize=10)
            idx += ng
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Healthy-reference gene z")
    save_figure(fig, f"P05_healthy_gene_architecture_{'text' if show_text else 'notext'}")


def panel_06_arm_pathway_coupling(coup: pd.DataFrame, condition: str, show_text=True):
    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=False, constrained_layout=True)
    for ax, lineage in zip(axes, LINEAGE_ORDER):
        sub = coup[coup["lineage"] == lineage].copy()
        pws = CURATED_PATHWAYS[lineage]
        mat = sub.pivot(index="pathway", columns="arm", values="rho").reindex(index=pws, columns=ARM_SHAPE_COLS)
        qmat = sub.pivot(index="pathway", columns="arm", values="q_bh").reindex(index=pws, columns=ARM_SHAPE_COLS)
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-0.85, vmax=0.85)
        ax.set_xticks(np.arange(len(ARM_SHAPE_COLS)))
        ax.set_yticks(np.arange(len(pws)))
        ax.set_xticklabels([ARM_LABELS[a.replace("SHAPE_","")] for a in ARM_SHAPE_COLS] if show_text else [], rotation=50, ha='right')
        ax.set_yticklabels([clean_pathway_name(x) for x in pws] if show_text else [])
        if show_text:
            ax.set_title(lineage, color=LINEAGE_COLORS[lineage])
            for i in range(len(pws)):
                for j in range(len(ARM_SHAPE_COLS)):
                    sig = show_sig(qmat.iloc[i, j])
                    if sig:
                        ax.text(j, i, sig, ha="center", va="center", color="black", fontsize=9)
    if show_text:
        fig.suptitle(f"P06 | {condition.replace('_',' ')}: arm–pathway partial Spearman coupling")
        cbar = fig.colorbar(im, ax=axes, shrink=0.9)
        cbar.set_label("Partial Spearman rho\n(control ARM_GLOBAL + log n_cells)")
    save_figure(fig, f"P06_arm_pathway_coupling_{condition}_{'text' if show_text else 'notext'}")


def panel_07_uc_gene_decomposition(uc_gene: pd.DataFrame, show_text=True):
    fig, axes = plt.subplots(1, len(SELECTED_UC_COMPONENTS), figsize=(16, 5), sharey=False, constrained_layout=True)
    for ax, (lineage, arm) in zip(axes, SELECTED_UC_COMPONENTS):
        sub = uc_gene[(uc_gene["lineage"] == lineage) & (uc_gene["arm"] == arm)].copy()
        sub["gene"] = pd.Categorical(sub["gene"], ARM_TO_GENES[arm], ordered=True)
        sub = sub.sort_values("gene")
        y = np.arange(len(sub))
        ax.axvline(0, color="#B0B8C0", lw=1)
        ax.errorbar(sub["delta"], y,
                    xerr=[sub["delta"] - sub["ci_low"], sub["ci_high"] - sub["delta"]],
                    fmt='o', color=LINEAGE_COLORS[lineage], ecolor=LINEAGE_COLORS[lineage], capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["gene"].tolist() if show_text else [])
        ax.invert_yaxis()
        if show_text:
            ax.set_title(f"{lineage}\n{ARM_LABELS[arm]}")
            for yi, (_, r) in enumerate(sub.iterrows()):
                sig = show_sig(r["q_bh"])
                if sig:
                    ax.text(r["ci_high"] + 0.03, yi, sig, va="center", color=LINEAGE_COLORS[lineage])
    if show_text:
        fig.suptitle("P07 | UC inflamed vs Healthy: gene-level decomposition of selected arm changes")
        axes[1].set_xlabel("Disease − Healthy gene-score difference")
    save_figure(fig, f"P07_uc_gene_decomposition_{'text' if show_text else 'notext'}")


def panel_08_uc_pathway_effects(uc_path: pd.DataFrame, show_text=True):
    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=False, constrained_layout=True)
    for ax, lineage in zip(axes, LINEAGE_ORDER):
        sub = uc_path[uc_path["lineage"] == lineage].copy()
        sub["pathway"] = pd.Categorical(sub["pathway"], CURATED_PATHWAYS[lineage], ordered=True)
        sub = sub.sort_values("pathway")
        y = np.arange(len(sub))
        ax.axvline(0, color="#B0B8C0", lw=1)
        ax.errorbar(sub["delta"], y,
                    xerr=[sub["delta"] - sub["ci_low"], sub["ci_high"] - sub["delta"]],
                    fmt='o', color=LINEAGE_COLORS[lineage], ecolor=LINEAGE_COLORS[lineage], capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([clean_pathway_name(x) for x in sub["pathway"]] if show_text else [])
        ax.invert_yaxis()
        if show_text:
            ax.set_title(lineage, color=LINEAGE_COLORS[lineage])
            for yi, (_, r) in enumerate(sub.iterrows()):
                sig = show_sig(r["q_bh_within_lineage"])
                if sig:
                    ax.text(r["ci_high"] + 0.04, yi, sig, va="center", color=LINEAGE_COLORS[lineage], fontsize=11)
    if show_text:
        fig.suptitle("P08 | UC inflamed vs Healthy: curated pathway effects across five lineages")
        axes[2].set_xlabel("Disease − Healthy pathway score")
    save_figure(fig, f"P08_uc_pathway_effects_{'text' if show_text else 'notext'}")


def panel_09_uc_bcell_tf_bridge(tf_diff: pd.DataFrame, tf_cor: pd.DataFrame, show_text=True):
    # keep top 12 by absolute correlation for rhs and top 12 by q/delta for lhs
    left = tf_diff.sort_values(["q_bh", "delta"], ascending=[True, False]).head(15).copy()
    right = tf_cor.sort_values("rho", ascending=False).head(15).copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(left))
    ax.axvline(0, color="#B0B8C0", lw=1)
    ax.errorbar(left["delta"], y,
                xerr=[left["delta"] - left["ci_low"], left["ci_high"] - left["delta"]],
                fmt='o', color="#8CA2B5", ecolor="#8CA2B5", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(left["TF_clean"] if show_text else [])
    ax.invert_yaxis()
    if show_text:
        ax.set_title("TF expression difference\nUC inflamed − Healthy")
        for yi, (_, r) in enumerate(left.iterrows()):
            sig = show_sig(r["q_bh"])
            if sig:
                ax.text(r["ci_high"] + 0.03, yi, sig, va="center")
    ax = axes[1]
    y = np.arange(len(right))
    colors = ["#9BAFC1" if (pd.isna(q) or q >= 0.05) else "#5B8CC0" for q in right["q_bh"]]
    ax.barh(y, right["rho"], color=colors)
    ax.axvline(0, color="#B0B8C0", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(right["TF_clean"] if show_text else [])
    ax.invert_yaxis()
    if show_text:
        ax.set_title("Within UC inflamed: TF vs B-cell E-box+ SHAPE\npartial rank correlations")
        ax.set_xlabel("Condition on arm global + cell-count proxy")
    if show_text:
        fig.suptitle("P09 | B-cell TF candidates: expression vs architecture association")
    save_figure(fig, f"P09_uc_bcell_tf_bridge_{'text' if show_text else 'notext'}")


def panel_10_uc_paired_shape(paired_shape: pd.DataFrame, show_text=True):
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True, constrained_layout=True)
    for ax, lineage in zip(axes, LINEAGE_ORDER):
        sub = paired_shape[paired_shape["lineage"] == lineage].copy()
        sub["arm"] = pd.Categorical(sub["arm"], ARM_SHAPE_COLS, ordered=True)
        sub = sub.sort_values("arm")
        y = np.arange(len(sub))
        ax.axvline(0, color="#B0B8C0", lw=1)
        ax.errorbar(sub["delta"], y,
                    xerr=[sub["delta"] - sub["ci_low"], sub["ci_high"] - sub["delta"]],
                    fmt='o', color=LINEAGE_COLORS[lineage], ecolor=LINEAGE_COLORS[lineage], capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABELS[a.replace("SHAPE_","")] for a in sub["arm"]] if show_text and ax is axes[0] else ([] if not show_text else []))
        ax.invert_yaxis()
        if show_text:
            ax.set_title(lineage, color=LINEAGE_COLORS[lineage])
            for yi, (_, r) in enumerate(sub.iterrows()):
                sig = show_sig(r["q_bh"])
                if sig:
                    ax.text(r["ci_high"] + 0.03, yi, sig, va="center")
    if show_text:
        fig.suptitle("P10 | Matched UC donors: inflamed vs noninflamed relative six-arm changes")
        axes[2].set_xlabel("Within-donor mean difference with paired bootstrap CI")
    save_figure(fig, f"P10_uc_paired_shape_{'text' if show_text else 'notext'}")


def panel_11_uc_clock17_3states(heat: pd.DataFrame, show_text=True):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True, constrained_layout=True)
    for ax, cond in zip(axes, COND_ORDER):
        sub = heat[heat["condition"] == cond].copy()
        mat = sub.pivot(index="lineage", columns="gene", values="z").reindex(index=LINEAGE_ORDER, columns=CLOCK17_ORDER)
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
        ax.set_xticks(np.arange(len(CLOCK17_ORDER)))
        ax.set_yticks(np.arange(len(LINEAGE_ORDER)))
        ax.set_xticklabels(CLOCK17_ORDER if show_text else [], rotation=60, ha='right')
        ax.set_yticklabels(LINEAGE_ORDER if show_text else [])
        if show_text:
            ax.set_title(cond.replace("_"," "))
    if show_text:
        fig.suptitle("P11 | CLOCK17 architecture across Healthy / noninflamed / inflamed UC")
        cbar = fig.colorbar(im, ax=axes, shrink=0.88)
        cbar.set_label("Healthy-referenced gene z")
    save_figure(fig, f"P11_uc_clock17_3states_{'text' if show_text else 'notext'}")


def panel_12_summary_contacts(show_text=True):
    # simple placeholder multi-panel contact image listing produced outputs
    files = sorted([p.name for p in PANELS.glob("*.png") if ("text" in p.stem and "summary" not in p.stem)])
    fig, ax = plt.subplots(figsize=(12, max(4, len(files) * 0.22)), constrained_layout=True)
    ax.axis("off")
    if show_text:
        ax.set_title("Produced panel files")
        for i, name in enumerate(files):
            ax.text(0.01, 0.98 - i*(0.95/max(len(files), 1)), name, va='top', family='monospace', fontsize=9)
    save_figure(fig, f"P12_output_manifest_{'text' if show_text else 'notext'}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    required = [
        BASE / "01_FROZEN_TABULA_STAGE78_PROFILE.csv",
        BASE / "03_SCP259_HEALTHY_PROFILE.csv",
        BASE / "02_DONOR_CONDITION_LINEAGE_SCORES.csv",
        BASE / "04_ALL_TESTS_WITH_FDR.csv",
        BASE / "03_SIX_ARM_AND_GENE_TESTS.csv",
        BASE / "SCP259_FIGURE_EXTENSION_REAL_DATA/04_TF_AND_PATHWAY_ARM_SHAPE_ASSOCIATIONS.csv",
        BASE / "SCP259_FIGURE_EXTENSION_REAL_DATA/05_BCELL_TF_PATHWAY_ASSOCIATIONS.csv",
    ]
    missing = [str(f) for f in required if not f.is_file()]
    if missing:
        raise FileNotFoundError("Required real source tables missing: " + "; ".join(missing))
    provenance = []
    for src in required:
        provenance.append({"relative_path": src.relative_to(BASE).as_posix(),
                           "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                           "file_size_bytes": src.stat().st_size})
    pd.DataFrame(provenance).to_csv(LOGS / "INPUT_FILES_SHA256.csv", index=False)
    print("[INPUT] source tables loaded from", BASE, flush=True)
    print("[OUTPUT] all figure panels will be saved to", PANELS, flush=True)
    data = load_data()
    normal_long = derive_normal_comparison(data)
    healthy_gene_arch = derive_healthy_gene_architecture(data)
    uc_gene = derive_uc_gene_component_effects(data)
    uc_path = derive_uc_pathway_effects(data)
    coup_h = derive_arm_pathway_coupling(data, "Healthy")
    coup_i = derive_arm_pathway_coupling(data, "UC_inflamed")
    tf_diff, tf_cor = derive_uc_bcell_tf_bridge(data)
    paired_shape = derive_uc_paired_shape(data)
    heat_uc = derive_uc_clock17_heatmap(data)

    # save tables
    normal_long.to_csv(TABLES / "T01_normal_two_atlas_long.csv", index=False)
    healthy_gene_arch.to_csv(TABLES / "T02_healthy_gene_architecture.csv")
    uc_gene.to_csv(TABLES / "T03_uc_gene_component_effects.csv", index=False)
    uc_path.to_csv(TABLES / "T04_uc_pathway_effects.csv", index=False)
    coup_h.to_csv(TABLES / "T05_healthy_arm_pathway_coupling.csv", index=False)
    coup_i.to_csv(TABLES / "T06_ucinflamed_arm_pathway_coupling.csv", index=False)
    tf_diff.to_csv(TABLES / "T07_uc_bcell_tf_expression_diffs.csv", index=False)
    tf_cor.to_csv(TABLES / "T08_uc_bcell_tf_shape_correlations.csv", index=False)
    paired_shape.to_csv(TABLES / "T09_uc_paired_shape_effects.csv", index=False)
    heat_uc.to_csv(TABLES / "T10_uc_clock17_heatmap_matrix_long.csv", index=False)

    # panels, text + notext
    for show_text in [True, False]:
        panel_01_normal_two_atlas_heatmap(normal_long, show_text=show_text)
        panel_02_uc_shape_effects(data, show_text=show_text)
        panel_03_uc_bcell_donors(data, show_text=show_text)
        panel_04_uc_distance(data, show_text=show_text)
        panel_05_healthy_gene_architecture(healthy_gene_arch, show_text=show_text)
        panel_06_arm_pathway_coupling(coup_h, "Healthy", show_text=show_text)
        panel_06_arm_pathway_coupling(coup_i, "UC_inflamed", show_text=show_text)
        panel_07_uc_gene_decomposition(uc_gene, show_text=show_text)
        panel_08_uc_pathway_effects(uc_path, show_text=show_text)
        panel_09_uc_bcell_tf_bridge(tf_diff, tf_cor, show_text=show_text)
        panel_10_uc_paired_shape(paired_shape, show_text=show_text)
        panel_11_uc_clock17_3states(heat_uc, show_text=show_text)

    manifest = {
        "project_root": str(PROJECT_ROOT),
        "input_root": str(BASE),
        "output_root": str(OUT),
        "panels_dir": str(PANELS),
        "tables_dir": str(TABLES),
        "generated_panels": sorted([p.name for p in PANELS.glob("*.png")]),
        "generated_tables": sorted([p.name for p in TABLES.glob("*.csv")]),
        "note": "Figures computed from real archived donor-level summaries; only frozen Tabula Stage78 profile used for cross-atlas main comparison, full Tabula data not rederived here; no TF causality claimed.",
    }
    with open(LOGS / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[OK] Biology-extension overnight pipeline finished.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
