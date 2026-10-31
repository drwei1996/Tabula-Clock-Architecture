# -*- coding: utf-8 -*-
"""
76_GSE292219_CLOCK_TIME_VS_LINEAGE_STORY.py

Directly reads the existing six GSE292219 H5AD files plus the frozen 74G
annotation. No pooled frozen74G h5ad is required.

Analyses:
1. Raw clock micro-pseudobulk UMAP by ZT and lineage
2. Time-residualized micro-pseudobulk UMAP by lineage
3. Raw trajectories
4. Lineage-centered trajectories
5. Cross-lineage phase dispersion
6. Lineage / Time / Lineage×Time variance decomposition on 30 lineage×ZT states
7. Six-arm and three-loop balance heatmaps
8. Fig1-like global heatmap with explicit time
"""

import os
from pathlib import Path
import json, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
RAWDIR = ROOT / "70_TIME_STAMPED_EXTERNAL_VALIDATION" / "01_raw" / "GSE292219"
ANN74G = (
    ROOT
    / "74G_GSE292219_POOLED_TWO_STAGE_MARKER_REFINEMENT_AND_VALIDATION"
    / "01_MARKER_BASED_UNIFIED_ANNOTATION"
    / "74G_CELL_LEVEL_FINAL_ANNOTATIONS.csv.gz"
)
OUT = ROOT / "76_GSE292219_CLOCK_TIME_VS_LINEAGE_STORY"
AUD = OUT / "00_AUDIT"
SRC = OUT / "01_SOURCE_DATA"
FIG = OUT / "02_FIGURES"
for p in [OUT, AUD, SRC, FIG]:
    p.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "GSM8853297": ("GSM8853297_ZT3.h5ad", 3),
    "GSM8853298": ("GSM8853298_ZT7.h5ad", 7),
    "GSM8853299": ("GSM8853299_ZT11.h5ad", 11),
    "GSM8853300": ("GSM8853300_ZT15.h5ad", 15),
    "GSM8853301": ("GSM8853301_ZT19.h5ad", 19),
    "GSM8853302": ("GSM8853302_ZT23.h5ad", 23),
}
ZT_ORDER = [3, 7, 11, 15, 19, 23]
LINEAGE_ORDER = ["B cell", "T cell", "Macrophage/Monocyte", "Fibroblast", "Endothelial"]

LINEAGE_COLORS = {
    "B cell": "#4E79A7",
    "T cell": "#E15759",
    "Macrophage/Monocyte": "#59A14F",
    "Fibroblast": "#F28E2B",
    "Endothelial": "#9467BD",
}
ZT_COLORS = {
    3: "#4E79A7",
    7: "#76B7B2",
    11: "#59A14F",
    15: "#EDC948",
    19: "#F28E2B",
    23: "#B07AA1",
}

SEED = 20260923
MICRO_N = 10
UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.30

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
})

def find_protocol():
    candidates = [
        PROJECT / "results" / "phaseA_clock_architecture" / "00_audit" / "prespecified_protocol.json",
        ROOT / "00_audit" / "prespecified_protocol.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in PROJECT.rglob("prespecified_protocol.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
            fs = obj.get("feature_sets", {})
            if fs.get("CORE_FEATURE_COLS") and fs.get("EXPANDED_FEATURE_COLS"):
                return p
        except Exception:
            pass
    raise FileNotFoundError("Could not recover frozen Core/Expanded panels.")

def recover_panels():
    p = find_protocol()
    obj = json.loads(p.read_text(encoding="utf-8-sig"))
    fs = obj["feature_sets"]
    core = [str(x).removeprefix("expr_") for x in fs["CORE_FEATURE_COLS"]]
    expanded = [str(x).removeprefix("expr_") for x in fs["EXPANDED_FEATURE_COLS"]]
    return p, core, expanded

def savefig(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kwargs = dict(bbox_inches="tight", pad_inches=0.08, facecolor="white")
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kwargs)
    plt.close(fig)

def dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)

def set_gene_names(a):
    if "gene_name" not in a.var.columns:
        raise RuntimeError("var['gene_name'] missing.")
    names = a.var["gene_name"].astype(str).copy()
    bad = names.isna() | names.eq("") | names.eq("nan")
    names.loc[bad] = a.var_names.astype(str)[bad]
    a.var_names = pd.Index(names)
    a.var_names_make_unique()
    return a

def resolve_genes(var_names, genes):
    lookup = {str(x).upper(): str(x) for x in var_names}
    if "ARNTL" not in lookup and "BMAL1" in lookup:
        lookup["ARNTL"] = lookup["BMAL1"]
    resolved, missing = [], []
    for g in genes:
        if g.upper() in lookup:
            resolved.append((g, lookup[g.upper()]))
        else:
            missing.append(g)
    if missing:
        raise RuntimeError("Missing genes: " + ", ".join(missing))
    return resolved

def load_cells(union_genes):
    if not ANN74G.exists():
        raise FileNotFoundError(ANN74G)

    ann = pd.read_csv(ANN74G)
    required = {"cell_id", "sample_id", "ZT", "final_annotation_74G"}
    if not required.issubset(ann.columns):
        raise RuntimeError(f"74G annotation missing {required - set(ann.columns)}")

    ann["cell_id"] = ann["cell_id"].astype(str)
    ann["sample_id"] = ann["sample_id"].astype(str)
    ann = ann[ann["final_annotation_74G"].isin(LINEAGE_ORDER)].copy()

    Xs, metas = [], []
    resolution = None

    for gsm, (fname, zt) in SAMPLES.items():
        path = RAWDIR / fname
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[LOAD] {gsm} ZT{zt}: {path}", flush=True)

        a = set_gene_names(sc.read_h5ad(path))
        resolved = resolve_genes(a.var_names, union_genes)
        if resolution is None:
            resolution = resolved
        idx_genes = [a.var_names.get_loc(dep) for _, dep in resolved]

        if "norm10k" not in a.layers:
            raise RuntimeError(f"{fname}: layer['norm10k'] missing")

        obs_map = pd.Series(np.arange(a.n_obs), index=a.obs_names.astype(str))
        dd = ann[ann["sample_id"].eq(gsm)].copy()
        idx = obs_map.reindex(dd["cell_id"].astype(str))
        keep = idx.notna()
        dd = dd.loc[keep.values].reset_index(drop=True)
        idx = idx.loc[keep].astype(int).values

        X = dense(a.layers["norm10k"][idx, :][:, idx_genes]).astype(np.float32)
        Xs.append(X)

        mm = dd[["cell_id", "sample_id", "ZT", "final_annotation_74G"]].copy()
        mm = mm.rename(columns={"final_annotation_74G": "lineage"})
        mm["ZT"] = int(zt)
        metas.append(mm)

        del a, X

    Xall = np.vstack(Xs)
    meta = pd.concat(metas, ignore_index=True)

    if len(meta) != Xall.shape[0]:
        raise RuntimeError("metadata/expression mismatch")

    return meta, Xall, resolution

def make_states(meta, X, genes):
    rows, mats = [], []
    for lin in LINEAGE_ORDER:
        for zt in ZT_ORDER:
            mask = (meta["lineage"].values == lin) & (meta["ZT"].values == zt)
            if mask.sum() == 0:
                continue
            mats.append(X[mask].mean(axis=0))
            rows.append({"lineage": lin, "ZT": zt, "n_cells": int(mask.sum())})
    states = pd.DataFrame(rows)
    M = np.vstack(mats)
    expr = pd.DataFrame(M, columns=genes)
    return pd.concat([states, expr], axis=1)

def make_micro_pb(meta, X, n_per=10, seed=SEED):
    rng = np.random.default_rng(seed)
    rows, mats = [], []
    gid = 0

    for lin in LINEAGE_ORDER:
        for zt in ZT_ORDER:
            ids = np.where((meta["lineage"].values == lin) & (meta["ZT"].values == zt))[0]
            if len(ids) == 0:
                continue
            ids = ids.copy()
            rng.shuffle(ids)

            n_full = len(ids) // n_per
            for k in range(n_full):
                take = ids[k*n_per:(k+1)*n_per]
                gid += 1
                mats.append(X[take].mean(axis=0))
                rows.append({
                    "micro_id": f"MPB{gid:04d}",
                    "lineage": lin,
                    "ZT": zt,
                    "n_cells": len(take),
                })

    if not mats:
        raise RuntimeError("No micro-pseudobulks were created.")
    return pd.DataFrame(rows), np.vstack(mats)

def standardize_log1p(M):
    L = np.log1p(np.maximum(M, 0))
    Z = StandardScaler().fit_transform(L)
    return L, Z

def residualize_time(md, Z):
    R = Z.copy()
    for zt in ZT_ORDER:
        mask = md["ZT"].values == zt
        R[mask] -= Z[mask].mean(axis=0, keepdims=True)
    return R

def fit_umap(Z):
    return umap.UMAP(
        n_neighbors=min(UMAP_N_NEIGHBORS, len(Z)-1),
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=SEED,
        n_components=2,
    ).fit_transform(Z)

def plot_umap(md, emb, color_by, stem, title):
    fig, ax = plt.subplots(figsize=(7.1, 6.1))
    if color_by == "lineage":
        for lin in LINEAGE_ORDER:
            m = md["lineage"].eq(lin).values
            ax.scatter(emb[m,0], emb[m,1], s=20, alpha=0.78, linewidths=0,
                       c=LINEAGE_COLORS[lin], label=lin)
        ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left", title="Lineage")
    else:
        for zt in ZT_ORDER:
            m = md["ZT"].eq(zt).values
            ax.scatter(emb[m,0], emb[m,1], s=20, alpha=0.78, linewidths=0,
                       c=ZT_COLORS[zt], label=f"ZT{zt}")
        ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left", title="Time")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, stem)

def plot_trajectories(states, genes):
    n = len(genes)
    ncol = 4
    nrow = math.ceil(n / ncol)
    handles = [Line2D([],[],color=LINEAGE_COLORS[x],marker="o",label=x) for x in LINEAGE_ORDER]

    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.7*nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for j, g in enumerate(genes):
        ax = axes[j]
        for lin in LINEAGE_ORDER:
            d = states[states["lineage"].eq(lin)].sort_values("ZT")
            ax.plot(d["ZT"], d[g], "-o", lw=1.5, ms=3.3,
                    color=LINEAGE_COLORS[lin], alpha=0.95)
        ax.set_title(g)
        ax.set_xticks(ZT_ORDER)
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center")
    fig.suptitle("A. Raw clock-gene trajectories across lineages", y=.995, fontsize=13)
    fig.subplots_adjust(bottom=.08, hspace=.45, wspace=.25)
    savefig(fig, "76_A_RAW_TRAJECTORIES")

    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.7*nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for j, g in enumerate(genes):
        ax = axes[j]
        for lin in LINEAGE_ORDER:
            d = states[states["lineage"].eq(lin)].sort_values("ZT").copy()
            y = d[g].to_numpy(float)
            y = y - y.mean()
            ax.plot(d["ZT"], y, "-o", lw=1.5, ms=3.3,
                    color=LINEAGE_COLORS[lin], alpha=0.95)
        ax.axhline(0, lw=.6, c="#888888")
        ax.set_title(g)
        ax.set_xticks(ZT_ORDER)
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend(handles=handles, ncol=5, frameon=False, loc="lower center")
    fig.suptitle("B. Lineage-centered trajectories reveal shared temporal waveform",
                 y=.995, fontsize=13)
    fig.subplots_adjust(bottom=.08, hspace=.45, wspace=.25)
    savefig(fig, "76_B_LINEAGE_CENTERED_TRAJECTORIES")

def cosinor_peak(y, t):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    w = 2*np.pi/24
    X = np.column_stack([np.ones(len(t)), np.cos(w*t), np.sin(w*t)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    grid = np.linspace(0, 24, 1441, endpoint=False)
    pred = b[0] + b[1]*np.cos(w*grid) + b[2]*np.sin(w*grid)
    peak = float(grid[np.argmax(pred)])
    amp = float((pred.max()-pred.min())/2)
    return peak, amp

def circ_dist(a, b):
    d = abs(a-b) % 24
    return min(d, 24-d)

def circular_dispersion_hours(vals):
    a = np.array(vals)*2*np.pi/24
    R = np.abs(np.mean(np.exp(1j*a)))
    if R <= 1e-12:
        return 12.0
    return float(np.sqrt(-2*np.log(R))*24/(2*np.pi))

def phase_analysis(states, genes):
    rows, detailed = [], []
    for g in genes:
        phases = []
        for lin in LINEAGE_ORDER:
            d = states[states["lineage"].eq(lin)].sort_values("ZT")
            peak, amp = cosinor_peak(d[g].values, d["ZT"].values)
            phases.append(peak)
            detailed.append({"gene":g, "lineage":lin, "peak_h":peak, "amplitude":amp})
        rows.append({
            "gene": g,
            "cross_lineage_phase_dispersion_h": circular_dispersion_hours(phases),
            "phase_range_circular_h": max(circ_dist(a,b) for a in phases for b in phases),
        })
    phase = pd.DataFrame(rows).sort_values("cross_lineage_phase_dispersion_h")
    detail = pd.DataFrame(detailed)
    phase.to_csv(SRC/"76_PHASE_DISPERSION.csv", index=False)
    detail.to_csv(SRC/"76_PHASE_BY_LINEAGE.csv", index=False)

    fig, ax = plt.subplots(figsize=(10,4.6))
    ax.bar(np.arange(len(phase)), phase["cross_lineage_phase_dispersion_h"], color="#4E79A7")
    ax.set_xticks(np.arange(len(phase)))
    ax.set_xticklabels(phase["gene"], rotation=60, ha="right")
    ax.set_ylabel("Cross-lineage phase dispersion (h)")
    ax.set_title("C. Circadian phase is broadly conserved across lineages")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "76_C_PHASE_DISPERSION")

def variance_partition(states, genes):
    rows = []
    for g in genes:
        tab = states.pivot(index="lineage", columns="ZT", values=g).loc[LINEAGE_ORDER, ZT_ORDER]
        Y = tab.to_numpy(float)
        grand = Y.mean()
        rowmean = Y.mean(axis=1)
        colmean = Y.mean(axis=0)

        ss_total = np.sum((Y-grand)**2)
        ss_lin = len(ZT_ORDER)*np.sum((rowmean-grand)**2)
        ss_time = len(LINEAGE_ORDER)*np.sum((colmean-grand)**2)
        interaction = Y - rowmean[:,None] - colmean[None,:] + grand
        ss_int = np.sum(interaction**2)

        denom = ss_total if ss_total > 0 else np.nan
        rows.append({
            "gene": g,
            "lineage_fraction": ss_lin/denom,
            "time_fraction": ss_time/denom,
            "interaction_fraction": ss_int/denom,
            "ss_total": ss_total,
        })

    vd = pd.DataFrame(rows)
    vd.to_csv(SRC/"76_VARIANCE_DECOMPOSITION.csv", index=False)

    fig, ax = plt.subplots(figsize=(11,5.0))
    x = np.arange(len(vd))
    b = np.zeros(len(vd))
    cols = [
        ("lineage_fraction", "#4E79A7", "Lineage"),
        ("time_fraction", "#59A14F", "Time"),
        ("interaction_fraction", "#F28E2B", "Lineage × Time"),
    ]
    for c,color,label in cols:
        ax.bar(x, vd[c], bottom=b, color=color, width=.82, label=label)
        b += vd[c].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(vd["gene"], rotation=60, ha="right")
    ax.set_ylim(0,1)
    ax.set_ylabel("Fraction of state-level variance")
    ax.set_title("D. Lineage, time and non-additive Lineage × Time contributions")
    ax.legend(frameon=False, ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "76_D_VARIANCE_DECOMPOSITION_PER_GENE")

    means = vd[["lineage_fraction","time_fraction","interaction_fraction"]].mean()
    fig, ax = plt.subplots(figsize=(4.5,5))
    bottom = 0
    for c,color,label in cols:
        ax.bar([0], [means[c]], bottom=bottom, color=color, width=.6, label=label)
        bottom += means[c]
    ax.set_ylim(0,1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Expanded clock"])
    ax.set_ylabel("Mean fraction of state-level variance")
    ax.set_title("D. Mean variance architecture")
    ax.legend(frameon=False, bbox_to_anchor=(1.02,1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, "76_D_VARIANCE_DECOMPOSITION_SUMMARY")

ARMS = {
    "EBOX_POS": ["ARNTL","CLOCK","NPAS2"],
    "EBOX_NEG": ["PER1","PER2","PER3","CRY1","CRY2"],
    "RORE_ACT": ["RORA","RORB","RORC"],
    "RORE_REP": ["NR1D1","NR1D2"],
    "DBOX_ACT": ["DBP","TEF","HLF"],
    "DBOX_REP": ["NFIL3"],
}

def gene_z_across_30(states, genes):
    Z = StandardScaler().fit_transform(states[genes].to_numpy(float))
    return pd.DataFrame(Z, columns=genes)

def compute_arm_scores(states, expanded):
    Z = gene_z_across_30(states, expanded)
    base = states[["lineage","ZT","n_cells"]].reset_index(drop=True).copy()
    for arm, genes in ARMS.items():
        use = [g for g in genes if g in Z.columns]
        base[arm] = Z[use].mean(axis=1)
    base["EBOX_BALANCE"] = base["EBOX_POS"] - base["EBOX_NEG"]
    base["RORE_BALANCE"] = base["RORE_ACT"] - base["RORE_REP"]
    base["DBOX_BALANCE"] = base["DBOX_ACT"] - base["DBOX_REP"]
    base.to_csv(SRC/"76_LOOP_ARM_AND_BALANCE_SCORES.csv", index=False)
    return base

def plot_loop_heatmap(scores):
    cols = ["EBOX_POS","EBOX_NEG","RORE_ACT","RORE_REP","DBOX_ACT","DBOX_REP",
            "EBOX_BALANCE","RORE_BALANCE","DBOX_BALANCE"]
    M = scores[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.6,9.2))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=55, ha="right")
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels([f"{r.lineage} | ZT{int(r.ZT)}" for _,r in scores.iterrows()], fontsize=6.3)
    for k in range(1,len(LINEAGE_ORDER)):
        ax.axhline(k*len(ZT_ORDER)-.5, c="white", lw=2)
    ax.set_title("E. Shared time axis, lineage-specific loop utilization")
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.025)
    cb.set_label("Standardized loop-arm / balance score")
    savefig(fig, "76_E_LOOP_BALANCE_HEATMAP")

def plot_global_heatmap(states, genes):
    Z = gene_z_across_30(states, genes).to_numpy(float)
    fig, ax = plt.subplots(figsize=(10.5,9.0))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right")
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([f"{r.lineage} | ZT{int(r.ZT)}" for _,r in states.iterrows()], fontsize=6.2)
    for k in range(1,len(LINEAGE_ORDER)):
        ax.axhline(k*len(ZT_ORDER)-.5, c="white", lw=2.2)
    ax.set_title("Fig1-like global Expanded-clock landscape with explicit time")
    ax.set_xlabel("Expanded clock genes")
    ax.set_ylabel("Lineage × Zeitgeber time")
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.025)
    cb.set_label("Gene-wise standardized expression")
    savefig(fig, "76_FIG1_LIKE_GLOBAL_HEATMAP_WITH_TIME")

def main():
    print("="*100)
    print("76 — GSE292219 SHARED TIME / LINEAGE ARCHITECTURE")
    print("="*100)

    protocol, core, expanded = recover_panels()
    union = []
    for g in core + expanded:
        if g not in union:
            union.append(g)

    meta, Xall, resolution = load_cells(union)
    pos = {g:i for i,g in enumerate(union)}
    core_idx = [pos[g] for g in core]
    exp_idx = [pos[g] for g in expanded]

    counts = meta.groupby(["lineage","ZT"], as_index=False).size().rename(columns={"size":"n_cells"})
    counts.to_csv(AUD/"76_CELL_COUNTS.csv", index=False)

    pd.DataFrame([{"canonical_gene":a,"deposited_gene":b} for a,b in resolution]).to_csv(
        AUD/"76_GENE_RESOLUTION.csv", index=False
    )

    states = make_states(meta, Xall[:,exp_idx], expanded)
    states.to_csv(SRC/"76_LINEAGE_ZT_30_STATES_EXPANDED.csv", index=False)

    core_states = make_states(meta, Xall[:,core_idx], core)
    core_states.to_csv(SRC/"76_LINEAGE_ZT_30_STATES_CORE.csv", index=False)

    for name, idx in [("CORE",core_idx),("EXPANDED",exp_idx)]:
        md, M = make_micro_pb(meta, Xall[:,idx], n_per=MICRO_N)
        _, Z = standardize_log1p(M)
        R = residualize_time(md, Z)

        raw_emb = fit_umap(Z)
        res_emb = fit_umap(R)

        out_raw = md.copy()
        out_raw["UMAP1"] = raw_emb[:,0]
        out_raw["UMAP2"] = raw_emb[:,1]
        out_raw.to_csv(SRC/f"76_{name}_MICROPB_RAW_UMAP.csv", index=False)

        out_res = md.copy()
        out_res["UMAP1"] = res_emb[:,0]
        out_res["UMAP2"] = res_emb[:,1]
        out_res.to_csv(SRC/f"76_{name}_MICROPB_TIME_RESID_UMAP.csv", index=False)

        plot_umap(md, raw_emb, "ZT",
                  f"76_{name}_RAW_MICROPB_UMAP_BY_ZT",
                  f"{name.title()} clock — raw micro-pseudobulk UMAP colored by ZT")

        plot_umap(md, raw_emb, "lineage",
                  f"76_{name}_RAW_MICROPB_UMAP_BY_LINEAGE",
                  f"{name.title()} clock — raw micro-pseudobulk UMAP colored by lineage")

        plot_umap(md, res_emb, "lineage",
                  f"76_{name}_TIME_RESID_MICROPB_UMAP_BY_LINEAGE",
                  f"{name.title()} clock — time-residualized UMAP colored by lineage")

        plot_umap(md, res_emb, "ZT",
                  f"76_{name}_TIME_RESID_MICROPB_UMAP_BY_ZT",
                  f"{name.title()} clock — time-residualized UMAP colored by ZT")

    plot_trajectories(states, expanded)
    phase_analysis(states, expanded)
    variance_partition(states, expanded)
    scores = compute_arm_scores(states, expanded)
    plot_loop_heatmap(scores)
    plot_global_heatmap(states, expanded)

    audit = {
        "status":"COMPLETE",
        "protocol":str(protocol),
        "raw_dir":str(RAWDIR),
        "annotation":str(ANN74G),
        "n_priority_cells":int(len(meta)),
        "micro_pseudobulk_cells_per_point":MICRO_N,
        "micro_pseudobulk_inference_warning":
            "Micro-pseudobulk points are visualization only and are not independent biological replicates.",
        "primary_dynamic_unit":"lineage x ZT mean; 5 x 6 = 30 states",
        "core_genes":core,
        "expanded_genes":expanded,
        "variance_decomposition":
            "balanced descriptive two-way SS on 30 lineage x ZT means: lineage + time + non-additive interaction; no p-value claims",
        "main_interpretation":
            "Test whether shared temporal organization coexists with stable lineage-specific clock-expression and loop architecture."
    }
    (OUT/"76_COMPLETE.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")

    print("\n76 COMPLETE")
    print("Output:", OUT)
    print("\nInspect first:")
    print("  76_EXPANDED_RAW_MICROPB_UMAP_BY_ZT.png")
    print("  76_EXPANDED_TIME_RESID_MICROPB_UMAP_BY_LINEAGE.png")
    print("  76_A_RAW_TRAJECTORIES.png")
    print("  76_B_LINEAGE_CENTERED_TRAJECTORIES.png")
    print("  76_C_PHASE_DISPERSION.png")
    print("  76_D_VARIANCE_DECOMPOSITION_PER_GENE.png")
    print("  76_E_LOOP_BALANCE_HEATMAP.png")
    print("  76_FIG1_LIKE_GLOBAL_HEATMAP_WITH_TIME.png")

if __name__ == "__main__":
    main()
