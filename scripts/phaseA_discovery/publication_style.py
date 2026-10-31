"""Shared publication style; no data processing or statistical calculations."""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
dependency_dir = Path(os.environ.get('TABULA_PLOT_DEPS', ROOT / '.plotdeps'))
if dependency_dir.is_dir():
    sys.path.insert(0, str(dependency_dir))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LINEAGE_ORDER = ['b_cell', 't_cell', 'macrophage_monocyte', 'fibroblast', 'endothelial']
CORE_CLOCK = ['ARNTL', 'CLOCK', 'NPAS2', 'PER1', 'PER2', 'PER3', 'CRY1', 'CRY2']
DISPLAY_LABELS = {
    'b_cell': 'B cell', 't_cell': 'T cell', 'macrophage_monocyte': 'Macrophage/monocyte',
    'fibroblast': 'Fibroblast', 'endothelial': 'Endothelial', 'other': 'Other',
    'core_clock': 'Core clock', 'expanded_clock': 'Expanded clock', 'clock_modules': 'Clock modules',
    'cell_identity_unique_adj_r2': 'Cell identity', 'tissue_context_unique_adj_r2': 'Tissue context',
    'donor_unique_adj_r2': 'Donor', 'cell_identity_tissue_context_interaction_adj_r2': 'Cell × tissue',
    'leave_one_donor_out': 'Leave one donor out', 'log1p_cpm_core': 'Log1p CPM · core clock',
    'core_clock_mean': 'Core clock mean', 'positive_limb_mean': 'Positive limb mean',
    'negative_limb_mean': 'Negative limb mean', 'positive_negative_balance': 'Positive–negative balance',
    'expanded_clock_mean': 'Expanded clock mean',
    'GO_ENDOTHELIAL_CELL_PROGRAM_REPRESENTED': 'Endothelial cell\nprogram (GO)',
    'HALLMARK_FATTY_ACID_METABOLISM_REPRESENTED': 'Fatty acid\nmetabolism\n(Hallmark)',
    'HALLMARK_HYPOXIA_REPRESENTED': 'Hypoxia\n(Hallmark)',
    'HALLMARK_INTERFERON_GAMMA_RESPONSE_REPRESENTED': 'Interferon-γ\nresponse\n(Hallmark)',
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB_REPRESENTED': 'TNFα signaling\nvia NF-κB\n(Hallmark)',
    'REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION_REPRESENTED': 'Extracellular matrix\norganization\n(Reactome)',
}
CELLTYPE_COLORS = {
    'B cell': '#4E79A7', 'T cell': '#59A14F', 'Macrophage/monocyte': '#E15759',
    'Fibroblast': '#9467BD', 'Endothelial': '#F28E2B', 'Other': '#D3D3D3',
}
LINEAGE_COLORS = {k: CELLTYPE_COLORS[DISPLAY_LABELS[k]] for k in LINEAGE_ORDER + ['other']}
COMPONENT_COLORS = dict(zip(
    ['cell_identity_unique_adj_r2', 'tissue_context_unique_adj_r2', 'donor_unique_adj_r2',
     'cell_identity_tissue_context_interaction_adj_r2'],
    ['#4E79A7', '#F28E2B', '#59A14F', '#9467BD']))
FEATURE_COLORS = {'core_clock': '#4E79A7', 'expanded_clock': '#E15759', 'clock_modules': '#59A14F'}
DPI = 400

def apply_publication_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.titlepad': 15, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 9, 'legend.title_fontsize': 10,
        'figure.titlesize': 14, 'axes.linewidth': .7, 'lines.linewidth': 1.4,
        'axes.edgecolor': '#555555', 'text.color': '#222222', 'axes.labelcolor': '#333333',
        'xtick.color': '#444444', 'ytick.color': '#444444', 'axes.grid': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'figure.dpi': 110, 'savefig.dpi': DPI, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'legend.frameon': False,
    })

def clean_axis(ax, heatmap=False):
    ax.grid(False)
    ax.set_facecolor('white')
    for name, spine in ax.spines.items():
        spine.set_visible(not heatmap and name in ('left', 'bottom'))
    ax.tick_params(length=0 if heatmap else 3, pad=5)

def place_external_legend(legend_ax, handles, title=None, ncol=1, loc='center left'):
    """Use a reserved GridSpec cell, never an overlay on a data axes."""
    legend_ax.set_axis_off()
    legend = legend_ax.legend(handles=handles, title=title, ncol=ncol, loc=loc,
                              borderaxespad=0, handletextpad=.8, labelspacing=.9)
    return legend

def heatmap_layout(title, rows, columns=8, width=12, row_height=.30, left=.35, bottom=1.5):
    height = max(4.8, rows * row_height + bottom + .9)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(1, 2, left=left, right=.90, top=1-.8/height,
                         bottom=bottom/height, width_ratios=[25, 1], wspace=.12)
    ax = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1])
    fig.suptitle(title, y=1-.20/height, fontsize=12)
    return fig, ax, cax

def save_figure(fig, stem, output, dpi=DPI):
    """Save one canonical figure per format and a small preview for visual QA."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    preview = output / '_preview'; preview.mkdir(exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Tight export includes external labels and legends; record any text outside
    # the nominal canvas so layout problems can be reviewed instead of hidden.
    outside = []
    # Matplotlib creates extra ticks beyond the view limits. Those text objects
    # are not rendered and must not be mistaken for clipped labels.
    not_drawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lower-1e-9 <= tick.get_loc() <= upper+1e-9:
                    not_drawn.update([tick.label1, tick.label2])
    for artist in fig.findobj(matplotlib.text.Text):
        if artist in not_drawn or not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width and box.height and not fig.bbox.fully_contains(box.x0, box.y0):
            outside.append(artist.get_text())
        elif box.width and box.height and not fig.bbox.fully_contains(box.x1, box.y1):
            outside.append(artist.get_text())
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{stem}.{extension}', dpi=dpi, bbox_inches='tight', pad_inches=.16)
    fig.savefig(preview / f'{stem}.png', dpi=110, bbox_inches='tight', pad_inches=.16)
    legend_overlaps = []
    for legend in fig.findobj(matplotlib.legend.Legend):
        bounds = legend.get_window_extent(renderer)
        for ax in fig.axes:
            if ax.axison and (ax.collections or ax.lines or ax.containers) and bounds.overlaps(ax.bbox):
                legend_overlaps.append(ax.get_title() or ax.get_xlabel() or 'data panel')
    assert not legend_overlaps, f'Legend overlaps a data panel: {legend_overlaps}'
    plt.close(fig)
    return {'stem': stem, 'text_outside_nominal_canvas': sorted(set(outside)),
            'legends_overlapping_data_panels': legend_overlaps}
