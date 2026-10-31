#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 6A — schematic of nested pathway models (M0 / M1 / M2).

Publication-style, standalone, VECTOR-FIRST figure. No data files required.

Model specification faithfully follows tabula_fig6_Hallmark_Reactome_M012_v1.py:
- M0: donor × tissue × assay × lineage block intercepts, optionally
      a within-block log1p(n_eff) proxy; block intercepts are eliminated
      by within-block centering in the actual fit.
- M1: add common cos/sin harmonics of cohort-specific frozen B0 inferred phase.
- M2: add lineage-specific cos/sin deviations for lineages 2–5 (8 df).
- The drawn curves are SCHEMATIC, not fitted data. B0 phase is NOT true ZT.

Usage:
  python plot_fig6_panel_A_model_framework.py
  python plot_fig6_panel_A_model_framework.py --outdir ./MY_FIGURES --dpi 600

Outputs: FIG6_A_M0_M1_M2_model_framework.{svg,pdf,png}

Only dependencies: numpy, matplotlib.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COLORS = {
    'B cell': '#4E79A7',
    'T cell': '#E15759',
    'Macrophage/Monocyte': '#59A14F',
    'Fibroblast': '#F28E2B',
    'Endothelial': '#9467BD',
}
LINEAGES = list(COLORS)
INK = '#192B3A'
MUTED = '#607080'
BORDER = '#DDE5EC'
BG = '#FFFFFF'
ACCENTS = ['#607080', '#167D8D', '#C27B42']


def label(ax, x, y, s, size=10, color=INK, weight='normal', **kwargs):
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=size,
                   color=color, weight=weight, **kwargs)


def rounded(ax, x, y, w, h, face, edge=BORDER, radius=0.014, lw=0.95):
    q = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={radius}',
                       facecolor=face, edgecolor=edge, linewidth=lw,
                       transform=ax.transAxes, clip_on=False)
    ax.add_patch(q)
    return q


def miniature(fig, card_x, card_y, card_w, card_h, mode):
    # A separate axes in figure-relative coordinates keeps SVG curves vectorized.
    # No real data used: purely conceptual models.
    sub = fig.add_axes([card_x + 0.043 * card_w, card_y + 0.290 * card_h,
                        0.914 * card_w, 0.300 * card_h])
    t = np.linspace(0, 24, 241)
    theta = 2 * np.pi * t / 24
    offsets = np.array([0.36, 0.18, 0.0, -0.18, -0.36])
    amplitudes = np.array([0.34, 0.34, 0.34, 0.34, 0.34])
    # Synthetic phases only for the explanatory drawing.
    shifts = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    if mode == 'M2':
        amplitudes = np.array([0.26, 0.39, 0.48, 0.27, 0.37])
        shifts = np.array([1.1, -0.95, 2.7, -2.4, 0.2])
    for i, lin in enumerate(LINEAGES):
        baseline = offsets[i]
        if mode == 'M0':
            y = np.full_like(t, baseline)
        else:
            y = baseline + amplitudes[i] * np.cos(theta - shifts[i])
        sub.plot(t, y, lw=1.72 if mode != 'M0' else 1.75,
                 color=COLORS[lin], alpha=.97, solid_capstyle='round')
    sub.axhline(0, color='#E2E8ED', lw=.75, zorder=0)
    sub.set_xlim(0, 24)
    sub.set_ylim(-0.99, .99)
    sub.set_xticks([0, 6, 12, 18, 24])
    sub.tick_params(axis='x', labelsize=7.6, colors=MUTED, length=0, pad=5)
    sub.set_yticks([])
    for side in ('top', 'right', 'left'):
        sub.spines[side].set_visible(False)
    sub.spines['bottom'].set_color('#CAD4DE')
    sub.spines['bottom'].set_linewidth(.85)
    sub.text(.5, -0.30, 'B0 inferred phase (h)', ha='center', va='center',
             transform=sub.transAxes, color=MUTED, fontsize=8.1)


def build(outdir: Path, dpi: int = 450, show_legend: bool = True):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'pdf.fonttype': 42,           # editable vector text in PDF
        'ps.fonttype': 42,
        'svg.fonttype': 'none',       # preserve editable text in SVG
        'axes.unicode_minus': False,
        'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(15.6, 6.15), facecolor=BG)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)

    # Main label and information hierarchy.
    rounded(canvas, .032, .885, .041, .080, '#243C50', edge='#243C50', radius=.011)
    label(canvas, .0525, .925, 'A', 18, 'white', weight='bold', ha='center', va='center')
    label(canvas, .090, .945, 'Decomposing pathway expression by inferred clock state',
          17.2, INK, weight='bold', va='center')
    label(canvas, .091, .897,
          'Nested, donor-balanced models at the donor × tissue × assay × lineage × inferred-phase-bin level',
          10.2, MUTED, va='center')

    card_y, card_h, card_w = .217, .611, .287
    xs = [.035, .3565, .678]
    blocks = [
        ('M0', 'Identity baseline',
         r'$\alpha_{\mathrm{donor\times tissue\times assay\times lineage}}$',
         'Controls block-specific expression offsets',
         'Baseline only; no phase terms.'),
        ('M1', 'Shared phase association',
         r'$M0\; +\; \beta_c\cos\theta\; +\; \beta_s\sin\theta$',
         'One common harmonic across lineages',
         r'Test $M1\;\mathrm{vs}\;M0$ (2 additional terms).'),
        ('M2', 'Lineage-modulated association',
         r'$M1+\sum_{\ell=2}^{5}\mathbf{1}_{L=\ell}(\delta_{\ell c}\cos\theta+\delta_{\ell s}\sin\theta)$',
         'Lineage-specific amplitude and/or phase',
         r'Test $M2\;\mathrm{vs}\;M1$ (8 additional terms).'),
    ]
    for i, (mode, name, formula, note, test) in enumerate(blocks):
        x = xs[i]
        rounded(canvas, x, card_y, card_w, card_h, 'white', edge=BORDER, radius=.015, lw=1.1)
        rounded(canvas, x + .017, card_y + card_h - .067, .066, .044,
                ACCENTS[i], edge=ACCENTS[i], radius=.010)
        label(canvas, x + .050, card_y + card_h - .0445, mode,
              12.2, 'white', weight='bold', ha='center', va='center')
        label(canvas, x + .019, card_y + card_h - .105, name,
              12.5, INK, weight='bold', va='center')
        formula_size = 11.4 if i < 2 else 9.1
        label(canvas, x + .018, card_y + card_h - .179, formula,
              formula_size, INK, va='center')
        label(canvas, x + .019, card_y + .080, note,
              9.0, MUTED, va='center')
        label(canvas, x + .019, card_y + .042, test,
              8.9, ACCENTS[i], va='center', weight='medium')
        miniature(fig, x, card_y, card_w, card_h, mode)

    # Cross-model nesting arrows, drawn outside cards.
    for idx in (0, 1):
        x1 = xs[idx] + card_w + .005
        x2 = xs[idx+1] - .005
        ar = FancyArrowPatch((x1, .538), (x2, .538),
                             arrowstyle='-|>', mutation_scale=12,
                             color='#8A9BA9', lw=1.15,
                             transform=canvas.transAxes,
                             clip_on=False)
        canvas.add_patch(ar)

    # One compact legend that is shared across the three conceptual charts.
    if show_legend:
        legend_y = .162
        label(canvas, .040, legend_y, 'Lineages', 9.3, INK,
              weight='bold', va='center')
        legend_spec = [
            ('B cell', .124), ('T cell', .241),
            ('Macrophage/Monocyte', .354), ('Fibroblast', .592),
            ('Endothelial', .737),
        ]
        for ln, x in legend_spec:
            canvas.plot([x, x+.024], [legend_y, legend_y], lw=3.1,
                        color=COLORS[ln], transform=canvas.transAxes,
                        solid_capstyle='round', clip_on=False)
            label(canvas, x+.031, legend_y, ln, 9.15, INK, va='center')

    canvas.plot([.035, .965], [.122, .122], color='#E3E9EF',
                lw=.9, transform=canvas.transAxes)
    label(canvas, .040, .087,
          r'$\theta=2\pi\,t_{\mathrm{B0}}/24$; B0 is cohort-specifically inferred phase, NOT observed circadian time.',
          9.2, INK, va='center')
    label(canvas, .040, .047,
          'Conceptual curves, not fitted data. M0 block means are removed by within-block centering; '
          'log1p(effective cell count) is an optional nuisance term.',
          8.5, MUTED, va='center')

    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / 'FIG6_A_M0_M1_M2_model_framework'
    paths = []
    for ext in ('svg', 'pdf', 'png'):
        path = stem.with_suffix('.' + ext)
        fig.savefig(path, dpi=dpi, facecolor='white')
        paths.append(path)
        print(f'SAVED {path}')
    plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser(description='Draw Figure 6 Panel A (standalone schematic).')
    p.add_argument('--outdir', default='FIG6_FINAL_PANELS', help='Output directory')
    p.add_argument('--dpi', type=int, default=450, help='Raster PNG DPI')
    args = p.parse_args()
    build(Path(args.outdir).expanduser().resolve(), dpi=args.dpi)


if __name__ == '__main__':
    main()
