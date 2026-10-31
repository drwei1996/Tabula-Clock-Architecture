#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Fig6D and Fig6E from donor-matched baseline / phase-interaction results.

Input:
    katv2/**/05_DONOR_PERM_BASELINE_AND_PHASE_INTERACTION.csv

Outputs:
    katv2/FIG6_DE_PANELS/Fig6D_donor_matched_baseline_effects.{png,pdf,svg}
    katv2/FIG6_DE_PANELS/Fig6E_phase_dependent_modulation.{png,pdf,svg}

Design principles:
- Keep style consistent with previous polished figures.
- Heatmaps contain NO numeric annotations.
- No text overlaps.
- D: baseline lineage differences (phase-balanced, donor-matched).
- E: phase-dependent modulation, shown as
     E-left  = interaction amplitude heatmap
     E-right = peak phase of significant interactions only
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# Global style
# -----------------------------
LINEAGES = ['B cell', 'T cell', 'Macrophage/Monocyte', 'Fibroblast', 'Endothelial']
SHORT = {
    'B cell': 'B',
    'T cell': 'T',
    'Macrophage/Monocyte': 'Mac/Mono',
    'Fibroblast': 'Fibro',
    'Endothelial': 'Endo'
}
ARMS = ['EBOX_POS', 'EBOX_NEG', 'RORE_ACT', 'RORE_REP', 'DBOX_ACT', 'DBOX_REP']
ARM_DISPLAY = {
    'EBOX_POS': 'E-box (+)',
    'EBOX_NEG': 'E-box (−)',
    'RORE_ACT': 'RORE (+)',
    'RORE_REP': 'RORE (−)',
    'DBOX_ACT': 'D-box (+)',
    'DBOX_REP': 'D-box (−)'
}
PAIR_ORDER = [(a, b) for a, b in combinations(LINEAGES, 2)]
PAIR_LABELS = [f"{SHORT[a]} vs {SHORT[b]}" for a, b in PAIR_ORDER]
FMTS = ('png', 'pdf', 'svg')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10.5,
    'axes.titleweight': 'medium',
    'axes.labelsize': 9.3,
    'xtick.labelsize': 8.6,
    'ytick.labelsize': 8.4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 450,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
})


def save(fig, outdir: Path, stem: str, formats=FMTS):
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(outdir / f'{stem}.{fmt}', dpi=450 if fmt == 'png' else None,
                    bbox_inches='tight', pad_inches=0.16, facecolor='white')
    plt.close(fig)


def pair_to_label(a: str, b: str) -> str:
    return f"{SHORT[a]} vs {SHORT[b]}"


def prep_df(path: Path, assignment: str = 'soft_adjusted') -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {
        'lineage_A', 'lineage_B', 'arm', 'assignment',
        'phase_balanced_lineage_difference_z', 'baseline_BH_q_within_assignment',
        'interaction_24h_amplitude_z', 'interaction_peak_phase_h',
        'interaction_BH_q_within_assignment'
    }
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f'Missing columns: {sorted(miss)}')
    df = df[df['assignment'] == assignment].copy()
    df['pair'] = [pair_to_label(a, b) for a, b in zip(df['lineage_A'], df['lineage_B'])]
    df['pair'] = pd.Categorical(df['pair'], categories=PAIR_LABELS, ordered=True)
    df['arm'] = pd.Categorical(df['arm'], categories=ARMS, ordered=True)
    df = df.sort_values(['pair', 'arm']).reset_index(drop=True)
    return df


def draw_heatmap(ax, mat, *, cmap, vmin=None, vmax=None, center=None,
                 xlabels=None, ylabels=None, cbar_ax=None, cbar_label=''):
    data = mat.to_numpy(dtype=float)
    if center is not None:
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm,
                       aspect='auto', interpolation='nearest')
    else:
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(xlabels if xlabels is not None else list(mat.columns),
                       rotation=28, ha='right')
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(ylabels if ylabels is not None else list(mat.index))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar_ax is not None:
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.set_label(cbar_label, labelpad=9)
    return im


def add_sig_overlay(ax, qmat: pd.DataFrame, strong_thr=0.05, weak_thr=0.10):
    """Overlay significance markers.
    q<=0.05 : filled black dot
    0.05<q<=0.10 : hollow black circle
    """
    for i in range(qmat.shape[0]):
        for j in range(qmat.shape[1]):
            q = qmat.iat[i, j]
            if pd.isna(q):
                continue
            if q <= strong_thr:
                ax.scatter(j, i, s=18, c='black', marker='o', linewidths=0, zorder=3)
            elif q <= weak_thr:
                ax.scatter(j, i, s=22, facecolors='none', edgecolors='black',
                           marker='o', linewidths=0.8, zorder=3)


def plot_fig6d(df: pd.DataFrame, outdir: Path):
    effect = df.pivot(index='pair', columns='arm', values='phase_balanced_lineage_difference_z')
    qmat = df.pivot(index='pair', columns='arm', values='baseline_BH_q_within_assignment')
    effect = effect.reindex(index=PAIR_LABELS, columns=ARMS)
    qmat = qmat.reindex(index=PAIR_LABELS, columns=ARMS)
    bound = max(0.85, np.nanmax(np.abs(effect.to_numpy(dtype=float))))

    fig = plt.figure(figsize=(7.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04],
                          left=0.23, right=0.94, top=0.83, bottom=0.22, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    draw_heatmap(
        ax, effect,
        cmap='RdBu_r', vmin=-bound, vmax=bound, center=0,
        xlabels=[ARM_DISPLAY[a] for a in ARMS],
        ylabels=PAIR_LABELS,
        cbar_ax=cax,
        cbar_label='Phase-balanced lineage difference (z)'
    )
    add_sig_overlay(ax, qmat)
    fig.suptitle('D  Donor-matched lineage main effects', x=0.48, y=0.965,
                 fontsize=12, fontweight='medium')
    ax.set_title('Filled dot: q ≤ 0.05   |   Hollow dot: 0.05 < q ≤ 0.10',
                 fontsize=8.4, pad=10, color='#444444')
    save(fig, outdir, 'Fig6D_donor_matched_baseline_effects')


def plot_fig6e(df: pd.DataFrame, outdir: Path):
    amp = df.pivot(index='pair', columns='arm', values='interaction_24h_amplitude_z')
    qmat = df.pivot(index='pair', columns='arm', values='interaction_BH_q_within_assignment')
    peak = df.pivot(index='pair', columns='arm', values='interaction_peak_phase_h')
    amp = amp.reindex(index=PAIR_LABELS, columns=ARMS)
    qmat = qmat.reindex(index=PAIR_LABELS, columns=ARMS)
    peak = peak.reindex(index=PAIR_LABELS, columns=ARMS)

    # Only show peak phase for reasonably supported interactions.
    peak_masked = peak.copy()
    peak_masked[qmat > 0.10] = np.nan

    vmax_amp = max(0.12, float(np.nanmax(amp.to_numpy(dtype=float))))

    fig = plt.figure(figsize=(12.0, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.035, 1, 0.035],
                          left=0.15, right=0.96, top=0.83, bottom=0.22, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    cax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cax2 = fig.add_subplot(gs[0, 3])

    draw_heatmap(
        ax1, amp,
        cmap='YlOrRd', vmin=0, vmax=vmax_amp,
        xlabels=[ARM_DISPLAY[a] for a in ARMS],
        ylabels=PAIR_LABELS,
        cbar_ax=cax1,
        cbar_label='Interaction amplitude (z)'
    )
    add_sig_overlay(ax1, qmat)
    ax1.set_title('Interaction amplitude', pad=10)

    draw_heatmap(
        ax2, peak_masked,
        cmap='twilight_shifted', vmin=0, vmax=24,
        xlabels=[ARM_DISPLAY[a] for a in ARMS],
        ylabels=[''] * len(PAIR_LABELS),
        cbar_ax=cax2,
        cbar_label='Peak phase (h)'
    )
    # Put subtle x markers on cells with q<=0.1 so the user sees where the shown peak phases are supported.
    for i in range(qmat.shape[0]):
        for j in range(qmat.shape[1]):
            q = qmat.iat[i, j]
            if pd.notna(q) and q <= 0.10:
                ax2.scatter(j, i, s=18, c='black', marker='x', linewidths=0.7, zorder=3)
    ax2.set_title('Peak phase (only q ≤ 0.10 shown)', pad=10)

    fig.suptitle('E  Phase-dependent lineage modulation', x=0.50, y=0.965,
                 fontsize=12, fontweight='medium')
    save(fig, outdir, 'Fig6E_phase_dependent_modulation')


def resolve_input(project_root: Path, requested: Path | None) -> Path:
    """Locate the EXISTING 05_* statistical table without re-running any analysis."""
    if requested is not None:
        p = requested.expanduser()
        if not p.is_absolute():
            p = project_root / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--input-csv not found: {p}")
        return p

    filename = '05_DONOR_PERM_BASELINE_AND_PHASE_INTERACTION.csv'
    direct = project_root / filename
    if direct.is_file():
        return direct.resolve()
    preferred = project_root / 'TABULA_SIX_ARM_PHASE_MATCHED_OVERNIGHT_20261010' / filename
    if preferred.is_file():
        return preferred.resolve()

    hits = sorted(p.resolve() for p in project_root.rglob(filename)
                  if p.is_file() and 'FIG6_DE_PANELS' not in p.parts)
    if not hits:
        raise FileNotFoundError(
            f'Could not find {filename} under {project_root}. '
            'Pass the exact location with --input-csv PATH. '
            'This script only plots existing statistical outputs.'
        )
    if len(hits) > 1:
        pretty = '\n'.join('  - ' + str(p) for p in hits)
        raise RuntimeError(
            'Multiple versions of donor-permutation results were found; '
            'to avoid silently mixing analysis versions, specify --input-csv:\n' + pretty
        )
    return hits[0]


def main():
    parser = argparse.ArgumentParser(
        description='Render Fig6D and Fig6E from existing Tabula donor-matched results. '
                    'No extraction, fitting, or re-testing is performed.'
    )
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                        help='Your local katv2 directory; defaults to current directory.')
    parser.add_argument('--input-csv', type=Path, default=None,
                        help='Optional exact input CSV; path may be relative to katv2.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Optional output directory; defaults to katv2/FIG6_DE_PANELS.')
    parser.add_argument('--assignment', type=str, default='soft_adjusted',
                        choices=['soft_adjusted', 'soft', 'hard_adjusted', 'hard'])
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Project directory not found: {root}')
    input_csv = resolve_input(root, args.input_csv)
    outdir = args.output_dir if args.output_dir is not None else root / 'FIG6_DE_PANELS'
    if not outdir.is_absolute():
        outdir = root / outdir
    outdir = outdir.expanduser().resolve()

    print('[PROJECT]', root, flush=True)
    print('[INPUT]', input_csv, flush=True)
    print('[ASSIGNMENT]', args.assignment, flush=True)
    print('[OUTPUT]', outdir, flush=True)

    df = prep_df(input_csv, assignment=args.assignment)
    expected = len(PAIR_LABELS) * len(ARMS)
    if len(df) != expected or df[['pair', 'arm']].duplicated().any():
        raise ValueError(
            f'The selected assignment should have {expected} unique lineage pairs × arms, '
            f'but has {len(df)} rows; check the upstream statistical results.'
        )

    plot_fig6d(df, outdir)
    print('[PLOT] Figure 6D saved (PNG/PDF/SVG)', flush=True)
    plot_fig6e(df, outdir)
    print('[PLOT] Figure 6E saved (PNG/PDF/SVG)', flush=True)

    status = {
        'status': 'COMPLETE',
        'project_root': str(root),
        'input_csv': str(input_csv),
        'output_dir': str(outdir),
        'assignment': args.assignment,
        'n_lineage_pairs': len(PAIR_LABELS),
        'n_arms': len(ARMS),
        'n_tests': len(df),
        'baseline_q_le_0.05': int((df['baseline_BH_q_within_assignment'] <= 0.05).sum()),
        'interaction_q_le_0.05': int((df['interaction_BH_q_within_assignment'] <= 0.05).sum()),
        'notes': [
            'These are previously computed donor-matched statistics; drawing does not rerun any tests.',
            'Phase is frozen B0 inferred phase, not true ZT sampling time.',
            'D cells are lineage_A minus lineage_B; dot marks baseline BH q <= 0.05.',
            'E interaction peak phases are masked at BH q > 0.10.',
            'Six-arm scores describe expression organization, not experimentally verified biochemical loop activity.'
        ]
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'FIG6_DE_PLOT_STATUS.json').write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[COMPLETE]', outdir, flush=True)


if __name__ == '__main__':
    main()
