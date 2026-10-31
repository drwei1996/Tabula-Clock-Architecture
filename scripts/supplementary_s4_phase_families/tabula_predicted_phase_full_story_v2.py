#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TABULA predicted-phase full biological story
--------------------------------------------

Goal
----
Use frozen B0 inferred phase in Tabula Sapiens to mirror and extend the
GSE292219-style biological analyses, while keeping the same visual style.

This script:
1) reuses the existing extraction + core figure logic from
   tabula_predicted_phase_GSE292219_mirror_v1.py
2) adds:
   - inferred-phase expanded-clock expression landscape (UMAP-like panel)
   - clock17 variance decomposition:
       shared phase / lineage baseline / lineage×phase
   - relative six-arm architecture (row-centered across six arms)
   - arm relative phase offsets
   - tissue-context phase heterogeneity
   - descriptive phase-dependent TF-expression / clock-arm coupling

Important
---------
- This is still inferred phase, not true timestamp.
- These analyses are biological interpretation / organization analyses,
  not algorithm benchmarking.
- Six-arm scores are expression composites, not biochemical loop activity.

Run
---
python -u tabula_predicted_phase_full_story_v2.py \
  --project-root "$PWD" \
  --scope all \
  --resume \
  --hard \
  --n-permutations 499
"""

from __future__ import annotations
import argparse
import json
import math
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from sklearn.decomposition import PCA

# optional UMAP
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ----------------------------- constants -----------------------------

LINEAGES = ['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
COLORS = {
    'B cell':'#4E79A7',
    'T cell':'#E15759',
    'Macrophage/Monocyte':'#59A14F',
    'Fibroblast':'#F28E2B',
    'Endothelial':'#9467BD'
}
LINEAGE_MARKERS = {
    'B cell':'o',
    'T cell':'s',
    'Macrophage/Monocyte':'^',
    'Fibroblast':'D',
    'Endothelial':'v'
}

# inferred phase bins: 2,6,10,14,18,22
PHASE_MARKERS = {2:'o', 6:'s', 10:'^', 14:'D', 18:'P', 22:'X'}

CLOCK6 = ['ARNTL','PER2','CRY1','NR1D1','DBP','NFIL3']
W = 2*np.pi/24

FONT = {
    'font.family':'DejaVu Sans',
    'font.size':9,
    'axes.titlesize':10,
    'axes.labelsize':9,
    'pdf.fonttype':42,
    'ps.fonttype':42,
    'svg.fonttype':'none',
    'figure.dpi':120,
    'savefig.dpi':300,
    'axes.spines.top':False,
    'axes.spines.right':False
}
plt.rcParams.update(FONT)


# ----------------------------- helpers -----------------------------

def log(*args):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), *args, flush=True)

def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(path.with_suffix('.' + ext), bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)

def bh(p):
    a = np.asarray(p, float)
    out = np.full(len(a), np.nan)
    valid = np.where(np.isfinite(a))[0]
    if len(valid):
        order = valid[np.argsort(a[valid])]
        f = a[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.clip(np.minimum.accumulate(f[::-1])[::-1], 0, 1)
    return out

def harmonic(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 4 or np.unique(t[ok]).size < 4:
        return dict(amp=np.nan, phase=np.nan, r2=np.nan, mesor=np.nan)
    A = np.column_stack([np.ones(ok.sum()), np.cos(W*t[ok]), np.sin(W*t[ok])])
    beta = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    yh = A @ beta
    sst = np.sum((y[ok] - y[ok].mean())**2)
    r2 = float(1 - np.sum((y[ok] - yh)**2)/sst) if sst > 1e-10 else np.nan
    phase = float((np.arctan2(beta[2], beta[1]) % (2*np.pi)) / W)
    amp = float(np.hypot(beta[1], beta[2]))
    return dict(amp=amp, phase=phase, r2=r2, mesor=float(beta[0]))

def onehot(series):
    return pd.get_dummies(series.astype(str), drop_first=True)

def fit_sse(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return float(np.sum(resid * resid))

def cyc_delta_h(phase, ref):
    """signed circular difference in hours, range [-12, 12)."""
    x = (phase - ref + 12) % 24 - 12
    return x

def load_base_module(root: Path):
    p = root / 'tabula_predicted_phase_GSE292219_mirror_v1.py'
    if not p.exists():
        raise FileNotFoundError(f'Missing required base script: {p}')
    spec = importlib.util.spec_from_file_location('tabula_pred_phase_base', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------- extra figure / analysis functions ----------------------

def make_phase_landscape(wide, base, out):
    """
    Expanded-clock / clock17 expression landscape with inferred phase.
    Rows = lineage × phase-bin mean profiles
    Cols = CLOCK17 genes
    """
    genes = [g for g in base.CLOCK17 if g in wide.columns]
    need = ['lineage', 'phase_bin', 'phase_center_h']
    df = wide[need + genes].copy()
    land = df.groupby(['lineage','phase_bin','phase_center_h'], as_index=False)[genes].mean()

    mat = land[genes].to_numpy(float)
    col_sd = np.nanstd(mat, axis=0)
    keep = col_sd > 1e-8
    mat = mat[:, keep]
    genes_used = [g for g, k in zip(genes, keep) if k]

    # column standardize
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    X = (mat - mu) / np.where(sd > 1e-8, sd, 1.0)

    method = 'PCA'
    if HAS_UMAP and X.shape[0] >= 10 and X.shape[1] >= 3:
        reducer = umap.UMAP(
            n_neighbors=min(6, max(3, X.shape[0]-1)),
            min_dist=0.35,
            metric='euclidean',
            random_state=20261012
        )
        emb = reducer.fit_transform(X)
        method = 'UMAP'
        xlab, ylab = 'UMAP1', 'UMAP2'
    else:
        emb = PCA(n_components=2, random_state=20261012).fit_transform(X)
        xlab, ylab = 'PC1', 'PC2'

    land['dim1'] = emb[:, 0]
    land['dim2'] = emb[:, 1]
    land['method'] = method
    land['n_genes'] = len(genes_used)
    land.to_csv(out/'Fig5U_PHASE_LANDSCAPE_embedding.csv', index=False)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))

    # chronological line per lineage
    for li in LINEAGES:
        z = land[land.lineage.eq(li)].sort_values('phase_center_h')
        if z.empty:
            continue

        ax.plot(z.dim1, z.dim2, color=COLORS[li], lw=1.6, alpha=0.6)

        # arrows between neighboring inferred bins
        for i in range(len(z)-1):
            p1 = (z.iloc[i].dim1, z.iloc[i].dim2)
            p2 = (z.iloc[i+1].dim1, z.iloc[i+1].dim2)
            arr = FancyArrowPatch(
                p1, p2, arrowstyle='->', mutation_scale=10,
                lw=1.1, color=COLORS[li], alpha=0.45
            )
            ax.add_patch(arr)

        # points: color=lineage, marker=phase-bin
        for r in z.itertuples(index=False):
            mk = PHASE_MARKERS.get(int(r.phase_center_h), 'o')
            ax.scatter(
                r.dim1, r.dim2, s=95, marker=mk,
                color=COLORS[li], edgecolors='white', linewidth=0.7, zorder=5
            )
            ax.text(r.dim1+0.03, r.dim2+0.03, f"P{int(r.phase_center_h)}",
                    color=COLORS[li], fontsize=7)

    ax.set_title('Expanded-clock expression landscape with inferred phase')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    lineage_handles = [
        Line2D([0],[0], marker='o', linestyle='None', color=COLORS[l],
               markersize=8, label=l)
        for l in LINEAGES
    ]
    phase_handles = [
        Line2D([0],[0], marker=PHASE_MARKERS[h], linestyle='None', color='#666666',
               markersize=7, label=f'P{h}')
        for h in [2,6,10,14,18,22]
    ]

    leg1 = ax.legend(handles=lineage_handles, title='Lineage',
                     loc='upper left', bbox_to_anchor=(1.02, 1.00),
                     frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=phase_handles, title='Inferred phase',
              loc='upper left', bbox_to_anchor=(1.02, 0.45),
              frameon=False)

    fig.text(
        0.01, 0.01,
        'Lines connect phase-balanced inferred B0 bins within each lineage; descriptive overlays, not independent pseudotime inference.',
        fontsize=8
    )
    fig.tight_layout(rect=[0, 0.03, 0.86, 1])
    save(fig, out/'Fig5U_expanded_clock_expression_landscape')

    return land


def variance_decomposition(wide, base, out):
    """
    For each CLOCK17 gene:
      y ~ block
      + shared phase (cos,sin)
      + lineage baseline
      + lineage × phase
    """
    genes = [g for g in base.CLOCK17 if g in wide.columns]
    meta = wide[['block','lineage','phase_center_h']].copy()

    block_oh = onehot(meta['block'])
    line_oh = onehot(meta['lineage'])
    phase_df = pd.DataFrame({
        'cos': np.cos(W * meta['phase_center_h'].to_numpy(float)),
        'sin': np.sin(W * meta['phase_center_h'].to_numpy(float))
    })

    inter_cols = {}
    for c in line_oh.columns:
        inter_cols[f'{c}:cos'] = line_oh[c].to_numpy(float) * phase_df['cos'].to_numpy(float)
        inter_cols[f'{c}:sin'] = line_oh[c].to_numpy(float) * phase_df['sin'].to_numpy(float)
    inter_df = pd.DataFrame(inter_cols)

    X0 = np.column_stack([np.ones(len(meta)), block_oh.to_numpy(float)])
    X1 = np.column_stack([X0, phase_df.to_numpy(float)])
    X2 = np.column_stack([X1, line_oh.to_numpy(float)])
    X3 = np.column_stack([X2, inter_df.to_numpy(float)])

    rows = []
    for g in genes:
        y = wide[g].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 30:
            continue
        yy = y[ok]
        sst = float(np.sum((yy - yy.mean())**2))
        if sst <= 1e-10:
            continue

        sse0 = fit_sse(yy, X0[ok])
        sse1 = fit_sse(yy, X1[ok])
        sse2 = fit_sse(yy, X2[ok])
        sse3 = fit_sse(yy, X3[ok])

        rows.append({
            'gene': g,
            'n': int(ok.sum()),
            'R2_block_only': 1 - sse0/sst,
            'delta_R2_shared_phase': max(0.0, (sse0 - sse1)/sst),
            'delta_R2_lineage_baseline': max(0.0, (sse1 - sse2)/sst),
            'delta_R2_lineage_phase_interaction': max(0.0, (sse2 - sse3)/sst),
            'R2_full': 1 - sse3/sst
        })

    vd = pd.DataFrame(rows).sort_values('gene')
    vd.to_csv(out/'Fig6A_CLOCK17_variance_decomposition.csv', index=False)

    # plot: all CLOCK17 genes heatmap
    show_cols = [
        'delta_R2_shared_phase',
        'delta_R2_lineage_baseline',
        'delta_R2_lineage_phase_interaction'
    ]
    mat = vd.set_index('gene')[show_cols]
    fig, ax = plt.subplots(figsize=(5.8, max(4.2, 0.28*len(mat)+1.5)))
    im = ax.imshow(mat.values, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(show_cols)))
    ax.set_xticklabels(['Shared\nphase', 'Lineage\nbaseline', 'Lineage×phase'], rotation=0)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7)
    ax.set_title('Fig6A  CLOCK17 variance decomposition\n(block-adjusted)')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label='Incremental R²')
    fig.tight_layout()
    save(fig, out/'Fig6A_CLOCK17_variance_decomposition_heatmap')

    # plot: clock6 stacked bars
    sub = vd[vd.gene.isin(CLOCK6)].copy()
    if not sub.empty:
        sub = sub.set_index('gene').reindex(CLOCK6)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        bottom = np.zeros(len(sub))
        stack_info = [
            ('delta_R2_shared_phase', '#8FBBD9', 'Shared phase'),
            ('delta_R2_lineage_baseline', '#E3B36A', 'Lineage baseline'),
            ('delta_R2_lineage_phase_interaction', '#C97B84', 'Lineage×phase')
        ]
        x = np.arange(len(sub))
        for col, color, label in stack_info:
            v = sub[col].fillna(0).to_numpy(float)
            ax.bar(x, v, bottom=bottom, color=color, width=0.72, label=label)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels(['Bmal1','Per2','Cry1','Nr1d1','Dbp','Nfil3'])
        ax.set_ylabel('Incremental R²')
        ax.set_title('Fig6A  CLOCK6 variance decomposition')
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, out/'Fig6A_CLOCK6_variance_decomposition_bar')

    return vd


def make_relative_six_arm(wide, base, out):
    """
    Relative six-arm architecture:
      subtract mean across the 6 arms within each donor×lineage×phase unit
    """
    arms = list(base.ARMS.keys())
    need = ['tissue','assay','donor','lineage','phase_bin','phase_center_h','block']
    rel = wide[need + arms].copy()

    arm_mat = rel[arms].to_numpy(float)
    rel_mean = np.nanmean(arm_mat, axis=1, keepdims=True)
    rel[arms] = arm_mat - rel_mean
    rel.to_csv(out/'Fig6B_relative_six_arm_unit_level.csv.gz', index=False, compression='gzip')

    # profile
    long = rel.melt(
        id_vars=need,
        value_vars=arms,
        var_name='arm',
        value_name='rel_z'
    ).dropna(subset=['rel_z'])

    prof = long.groupby(
        ['lineage','phase_bin','phase_center_h','arm'], as_index=False
    ).agg(
        mean_rel_z=('rel_z','mean'),
        sd_rel_z=('rel_z','std'),
        n_blocks=('block','nunique')
    )
    prof.to_csv(out/'Fig6B_relative_six_arm_profiles.csv', index=False)

    # six panels trajectories
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.7), sharex=True)
    for j, arm in enumerate(arms):
        ax = axes.flat[j]
        for li in LINEAGES:
            s = prof[(prof.lineage.eq(li)) & (prof.arm.eq(arm))].sort_values('phase_center_h')
            if s.empty:
                continue
            ax.plot(
                s.phase_center_h, s.mean_rel_z,
                color=COLORS[li], marker=LINEAGE_MARKERS[li],
                lw=1.5, markersize=4.2
            )
        ax.axhline(0, color='#bbbbbb', lw=0.7)
        ax.set_xlim(0, 24)
        ax.set_xticks([2,6,10,14,18,22])
        ax.set_title(arm.replace('_',' '))
        if j >= 3:
            ax.set_xlabel('Inferred phase (h)')
        if j % 3 == 0:
            ax.set_ylabel('Relative arm z')
    fig.legend(
        handles=[Line2D([0],[0], color=COLORS[l], marker=LINEAGE_MARKERS[l], label=l) for l in LINEAGES],
        ncol=5, loc='upper center', bbox_to_anchor=(0.5,1.03), frameon=False
    )
    fig.suptitle('Fig6B  Relative six-arm architecture by inferred phase and lineage', y=1.08, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.98])
    save(fig, out/'Fig6B_relative_six_arm_phase_profiles')

    # mean heatmap
    hm = prof.groupby(['lineage','arm'])['mean_rel_z'].mean().unstack('arm').reindex(index=LINEAGES, columns=arms)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    maxv = np.nanmax(np.abs(hm.values))
    im = ax.imshow(hm.values, cmap='RdBu_r', vmin=-max(0.25, maxv), vmax=max(0.25, maxv), aspect='auto')
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=35, ha='right')
    ax.set_yticks(range(len(LINEAGES)))
    ax.set_yticklabels(LINEAGES)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.iloc[i,j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
    ax.set_title('Fig6B  Relative six-arm lineage architecture\n(phase-balanced mean)')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, label='Mean relative z')
    fig.tight_layout()
    save(fig, out/'Fig6B_relative_six_arm_heatmap')

    return rel, prof, hm


def arm_relative_phase(rel_prof, out):
    """
    Harmonic phase of each relative arm; show phase offset vs EBOX_POS within lineage.
    """
    arms = ['EBOX_POS','EBOX_NEG','RORE_ACT','RORE_REP','DBOX_ACT','DBOX_REP']
    rows = []
    for li in LINEAGES:
        ref = rel_prof[(rel_prof.lineage.eq(li)) & (rel_prof.arm.eq('EBOX_POS'))].sort_values('phase_center_h')
        ref_fit = harmonic(ref.phase_center_h, ref.mean_rel_z) if len(ref) >= 4 else dict(phase=np.nan, amp=np.nan, r2=np.nan)
        ref_phase = ref_fit['phase']
        for arm in arms:
            s = rel_prof[(rel_prof.lineage.eq(li)) & (rel_prof.arm.eq(arm))].sort_values('phase_center_h')
            h = harmonic(s.phase_center_h, s.mean_rel_z) if len(s) >= 4 else dict(phase=np.nan, amp=np.nan, r2=np.nan)
            rows.append({
                'lineage': li,
                'arm': arm,
                'amp': h['amp'],
                'phase_h': h['phase'],
                'r2': h['r2'],
                'delta_vs_EBOX_POS_h': cyc_delta_h(h['phase'], ref_phase) if np.isfinite(h['phase']) and np.isfinite(ref_phase) else np.nan
            })
    dd = pd.DataFrame(rows)
    dd.to_csv(out/'Fig6C_relative_six_arm_harmonic_phase.csv', index=False)

    mat = dd.pivot(index='lineage', columns='arm', values='delta_vs_EBOX_POS_h').reindex(index=LINEAGES, columns=arms)
    fig, ax = plt.subplots(figsize=(7.3, 4.0))
    im = ax.imshow(mat.values, cmap='coolwarm', vmin=-6, vmax=6, aspect='auto')
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=35, ha='right')
    ax.set_yticks(range(len(LINEAGES)))
    ax.set_yticklabels(LINEAGES)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iloc[i,j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7)
    ax.set_title('Fig6C  Relative arm phase offsets\n(hours vs EBOX_POS within lineage)')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03, label='Δ phase (h)')
    fig.tight_layout()
    save(fig, out/'Fig6C_relative_six_arm_phase_offset_heatmap')

    return dd


def tissue_context_phase_heterogeneity(wide, base, out):
    """
    For each lineage & phase bin:
      compute cross-tissue SD of six-arm scores
      then average across arms
    """
    arms = list(base.ARMS.keys())
    df = wide[['tissue','lineage','phase_bin','phase_center_h'] + arms].copy()

    # tissue means first
    tiss = df.groupby(['tissue','lineage','phase_bin','phase_center_h'], as_index=False)[arms].mean()

    rows = []
    for li in LINEAGES:
        for ph in sorted(tiss.phase_center_h.unique()):
            s = tiss[(tiss.lineage.eq(li)) & (tiss.phase_center_h.eq(ph))]
            if len(s) < 2:
                continue
            for arm in arms:
                vals = s[arm].to_numpy(float)
                rows.append({
                    'lineage': li,
                    'phase_center_h': ph,
                    'arm': arm,
                    'cross_tissue_sd': float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() >= 2 else np.nan
                })
    het = pd.DataFrame(rows)
    het.to_csv(out/'FigS6_tissue_context_phase_heterogeneity_long.csv', index=False)

    summ = het.groupby(['lineage','phase_center_h'], as_index=False)['cross_tissue_sd'].mean()
    summ.to_csv(out/'FigS6_tissue_context_phase_heterogeneity_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for li in LINEAGES:
        s = summ[summ.lineage.eq(li)].sort_values('phase_center_h')
        if s.empty:
            continue
        ax.plot(
            s.phase_center_h, s.cross_tissue_sd,
            color=COLORS[li], marker=LINEAGE_MARKERS[li], lw=1.6, markersize=4.5,
            label=li
        )
    ax.set_xticks([2,6,10,14,18,22])
    ax.set_xlabel('Inferred phase (h)')
    ax.set_ylabel('Mean cross-tissue SD across six arms')
    ax.set_title('FigS6  Tissue-context phase heterogeneity of six-arm architecture')
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, out/'FigS6_tissue_context_phase_heterogeneity')

    return het, summ


def dynamic_tf_coupling_by_phase(wide, tf_all, base, out):
    """
    Descriptive phase-by-phase TF-expression / arm correlation.
    For each TF, choose the same dominant arm across lineages
    (based on global median |partial r| from the phase-adjusted overall analysis),
    then show correlation in each inferred phase bin separately.
    """
    if tf_all is None or tf_all.empty:
        return pd.DataFrame()

    chosen = {}
    for tf in base.TF_MAIN:
        z = tf_all[tf_all.TF.eq(tf)]
        if z.empty:
            continue
        arm = z.groupby('arm')['partial_r'].median().abs().idxmax()
        chosen[tf] = arm

    rows = []
    for tf, arm in chosen.items():
        if tf not in wide.columns or arm not in wide.columns:
            continue
        for li in LINEAGES:
            d = wide[wide.lineage.eq(li)][['phase_bin','phase_center_h',tf,arm]].dropna().copy()
            if d.empty:
                continue
            for ph in sorted(d.phase_center_h.unique()):
                s = d[d.phase_center_h.eq(ph)]
                if len(s) < 5:
                    continue
                x = s[tf].to_numpy(float)
                y = s[arm].to_numpy(float)
                if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    continue
                r = float(np.corrcoef(x, y)[0,1])
                rows.append({
                    'TF': tf,
                    'arm': arm,
                    'lineage': li,
                    'phase_center_h': ph,
                    'phase_bin_corr': r,
                    'n_units': int(len(s))
                })

    dd = pd.DataFrame(rows)
    dd.to_csv(out/'FigS7_phase_dependent_TF_clock_coupling.csv', index=False)

    if dd.empty:
        return dd

    # one panel per lineage
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 5.2), sharey=True)
    row_order = [f'{tf} [{arm}]' for tf, arm in chosen.items()]

    for ax, li in zip(axes, LINEAGES):
        s = dd[dd.lineage.eq(li)].copy()
        if s.empty:
            ax.set_axis_off()
            continue
        s['rowname'] = s['TF'] + ' [' + s['arm'] + ']'
        mat = s.pivot(index='rowname', columns='phase_center_h', values='phase_bin_corr').reindex(index=row_order, columns=[2,6,10,14,18,22])
        im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-0.8, vmax=0.8, aspect='auto')
        ax.set_title(li, color=COLORS[li], fontsize=10)
        ax.set_xticks(range(6))
        ax.set_xticklabels([2,6,10,14,18,22])
        if ax is axes[0]:
            ax.set_yticks(range(len(mat)))
            ax.set_yticklabels(mat.index, fontsize=8)
        else:
            ax.set_yticks(range(len(mat)))
            ax.set_yticklabels([])
        ax.set_xlabel('Phase (h)')

    axes[0].set_ylabel('TF [selected arm]')
    fig.suptitle('FigS7  Phase-dependent TF-expression / clock-arm coupling (descriptive)', y=1.02, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Pearson r')
    fig.tight_layout()
    save(fig, out/'FigS7_phase_dependent_TF_clock_coupling_heatmap')

    return dd


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--scope', choices=['all','pilot'], default='all')
    parser.add_argument('--min-detect', type=int, default=2)
    parser.add_argument('--min-R', dest='min_R', type=float, default=0.3)
    parser.add_argument('--min-bin-weight', type=float, default=8)
    parser.add_argument('--min-bin-effective', type=float, default=8)
    parser.add_argument('--h5ad-block', type=int, default=2500)
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-extract', action='store_true')
    parser.add_argument('--n-permutations', type=int, default=99)
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = (args.output_dir or root/'TABULA_PREDICTED_PHASE_FULL_STORY_V2_20261012').resolve()
    out.mkdir(parents=True, exist_ok=True)

    # provenance
    with open(out/'COMMAND.json', 'w', encoding='utf-8') as f:
        json.dump({k:(str(v) if isinstance(v, Path) else v) for k,v in vars(args).items()}, f, indent=2, ensure_ascii=False)

    log('LOAD base module')
    base = load_base_module(root)

    # extraction / aggregation using your existing pipeline
    log('EXTRACT / AGGREGATE')
    d = base.extract(args, root, out)
    if d.empty:
        raise RuntimeError('Empty extracted long table')

    base.phase_coverage_audit(d, None, out)

    wide = base.make_wide(d, 'soft')
    wide.to_csv(out/'01_DONOR_LINEAGE_PSEUDOTIME_WIDE.csv.gz', index=False, compression='gzip')
    log('WIDE units:', len(wide))

    pr, liv = base.profiles(wide)
    pr.to_csv(out/'02_PANATLAS_LINEAGE_PSEUDOTIME_PROFILES.csv', index=False)
    liv.to_csv(out/'02_LIVER_LINEAGE_PSEUDOTIME_PROFILES.csv', index=False)

    # ---------------- original core analyses / figures ----------------
    log('FIG 5A original trajectory panel')
    ff, ps = base.trajectory_figure(pr, out)
    ff.to_csv(out/'Fig5A_CLOCK6_HARMONIC_FITS.csv', index=False)
    ps.to_csv(out/'Fig5A_CROSS_LINEAGE_PHASE_DISPERSION.csv', index=False)

    log('FIG 5D original six-arm panels')
    base.six_arm_fig(pr, out, 'Fig5D_predicted_phase_six_arms')
    base.arm_heatmap(pr, out)
    if not liv.empty:
        base.six_arm_fig(liv, out, 'Fig5D_LIVER_predicted_phase_six_arms')

    if args.hard:
        log('HARD sensitivity extraction')
        hw = base.make_wide(d, 'hard')
        hw.to_csv(out/'01_HARD_DONOR_LINEAGE_PSEUDOTIME_WIDE.csv.gz', index=False, compression='gzip')
        hp, hliv = base.profiles(hw)
        hp.to_csv(out/'02_HARD_PANATLAS_LINEAGE_PSEUDOTIME_PROFILES.csv', index=False)
        base.six_arm_fig(hp, out, 'FigS5_HARD_phase_six_arms')

    log('FIG 5B/C lineage information after phase regression')
    scores = base.loo_tissue(wide, out, n_perm=args.n_permutations)
    base.holdout_phase_bin(wide, out)

    log('FIG 5E TF-expression / arm coupling')
    tf_all = base.tf_partial_coupling(wide, out)

    # ---------------- new analyses requested ----------------
    log('NEW: inferred-phase landscape panel')
    landscape = make_phase_landscape(wide, base, out)

    log('NEW: variance decomposition')
    vd = variance_decomposition(wide, base, out)

    log('NEW: relative six-arm architecture')
    rel, rel_prof, rel_hm = make_relative_six_arm(wide, base, out)

    log('NEW: arm relative phase offsets')
    arm_phase = arm_relative_phase(rel_prof, out)

    log('NEW: tissue-context phase heterogeneity')
    tissue_het, tissue_het_summ = tissue_context_phase_heterogeneity(wide, base, out)

    log('NEW: descriptive dynamic TF coupling')
    dyn_tf = dynamic_tf_coupling_by_phase(wide, tf_all, base, out)

    # ---------------- summary / status ----------------
    status = {
        'status': 'COMPLETE',
        'source': 'frozen_B0_predicted_phase',
        'cohorts_expected': 60 if args.scope == 'all' else 'pilot',
        'aggregated_units': int(len(wide)),
        'lineages': int(wide.lineage.nunique()),
        'n_tissues': int(wide.tissue.nunique()),
        'n_donor_blocks': int(wide.block.nunique()),
        'clock17_variance_partition_genes': int(len(vd)),
        'relative_six_arm_units': int(len(rel)),
        'dynamic_tf_phase_records': int(len(dyn_tf)) if dyn_tf is not None else 0,
        'umap_available': bool(HAS_UMAP),
        'warnings': [
            'This is inferred phase, NOT timestamped sampling time',
            'Clock-gene pseudotime trajectories are not independent of B0 training genes',
            'Leave-one-inferred-bin and phase-balanced bins are not equivalent to leave-one-true-ZT',
            'TF-expression correlations are observational, not TF activity or causal validation',
            'Six-arm scores reflect expression composites, not biochemical loop activity'
        ]
    }
    with open(out/'STATUS.json', 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    report = [
        '# Tabula predicted-phase full biological story',
        '',
        '## What this run produced',
        '- Core mirror analyses of GSE292219-style panels on Tabula inferred phase',
        '- Added expanded-clock expression landscape panel',
        '- Added block-adjusted CLOCK17 variance decomposition',
        '- Added relative six-arm architecture analysis',
        '- Added relative arm phase offset analysis',
        '- Added tissue-context phase heterogeneity analysis',
        '- Added descriptive phase-dependent TF–clock coupling',
        '',
        '## Main interpretation boundary',
        '- Shared temporal program and lineage-specific architecture can coexist.',
        '- Inferred-phase results are not equivalent to real timestamped circadian validation.',
        '- Six-arm metrics are organizational / statistical, not biochemical loop strength measurements.',
        '',
        '## Status JSON',
        '```json',
        json.dumps(status, indent=2, ensure_ascii=False),
        '```'
    ]
    (out/'README_FULL_STORY.md').write_text('\n'.join(report), encoding='utf-8')

    log('COMPLETE ->', out)


if __name__ == '__main__':
    main()