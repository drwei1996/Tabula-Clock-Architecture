import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DARK = '#20334D'
GREY = '#7A8797'
LIGHT_GREY = '#DCE3EA'
GRID = '#E9EEF3'
M1_COLOR = '#2F7E98'
M2_COLOR = '#B44E8D'
B_COLOR = '#3B82C4'
M_COLOR = '#E79A2D'
LINEAGE_COLORS = {
    'B cell': '#3B82C4',
    'T cell': '#D8576B',
    'Macrophage/Monocyte': '#E79A2D',
    'Fibroblast': '#45B39D',
    'Endothelial': '#8E63BF',
}
LINEAGE_ORDER = ['B cell', 'T cell', 'Macrophage/Monocyte', 'Fibroblast', 'Endothelial']

# Frozen Fig.3 biological-module palette from
# corrected_full_analysis_v2/scripts/34_freeze_and_build_tabula_style_additions.py
# These are the ORIGINAL Fig.3 hex codes, keyed by frozen module ID.
FIG3_MODULE_COLORS = {
    'M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE': '#2A9D8F',
    'M02_MAPK_RTK_GROWTH_FACTOR': '#355C9A',
    'M03_TGFB_SMAD': '#B07AA1',
    'M04_CELL_DEATH_APOPTOSIS': '#E76F51',
    'M05_RHO_CYTOSKELETON_ADHESION_REMODELING': '#76B7B2',
    'M06_TRAFFICKING_ER_PROTEOSTASIS': '#F28E7F',
    'M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR': '#8CD17D',
    'M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION': '#8A89A6',
    'M09_METABOLIC_OXIDATIVE_STRESS': '#E9C46A',
    'M10_CELL_CYCLE_GROWTH': '#EDC948',
    'M11_OTHER_SIGNALING_SPECIALIZED': '#86BCB6',
    'M99_OTHER_UNCLASSIFIED': '#9EA4AA',
}

def lighten_hex(hex_color, white_fraction=0.5):
    """Mix an ORIGINAL Fig.3 hex color with white without changing its hue."""
    from matplotlib.colors import to_rgb
    rgb = np.asarray(to_rgb(hex_color), dtype=float)
    return tuple(rgb + (1.0-rgb)*white_fraction)
MAIN_SHARED = [
    'HALLMARK_OXIDATIVE_PHOSPHORYLATION',
    'HALLMARK_MTORC1_SIGNALING',
    'HALLMARK_UNFOLDED_PROTEIN_RESPONSE',
    'REACTOME_IRE1ALPHA_ACTIVATES_CHAPERONES',
]
MAIN_LINEAGE = [
    'HALLMARK_INFLAMMATORY_RESPONSE',
    'HALLMARK_TNFA_SIGNALING_VIA_NFKB',
    'HALLMARK_IL6_JAK_STAT3_SIGNALING',
]
REP_TISSUES = [
    ('blood', "10x 3' v3"),
    ('bone marrow', "10x 3' v3"),
    ('lung', "10x 3' v3"),
    ('colon', "10x 3' v3"),
]
REP_PATHWAY = 'HALLMARK_INFLAMMATORY_RESPONSE'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def nice_name(pathway: str) -> str:
    x = pathway.replace('HALLMARK_', '').replace('REACTOME_', '')
    x = x.replace('_', ' ').title()
    rep = {
        'Tnfa': 'TNFα', 'Nfkb': 'NFκB', 'Il6': 'IL6', 'Jak': 'JAK', 'Stat3': 'STAT3',
        'Mtorc1': 'mTORC1', 'Ire1Alpha': 'IRE1α',
        'Oxidative Phosphorylation': 'Oxidative phosphorylation',
        'Unfolded Protein Response': 'Unfolded protein response',
        'Activates Chaperones': 'activates chaperones',
    }
    for a, b in rep.items():
        x = x.replace(a, b)
    return x


def setup_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#98A6B5')
    ax.spines['bottom'].set_color('#98A6B5')
    ax.tick_params(colors=DARK)
    ax.grid(True, axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_panel_header(fig, letter, title, with_text=True, x_letter=0.02, x_title=0.10, y=0.98):
    if with_text:
        fig.text(x_letter, y, letter, fontsize=16, fontweight='bold', color=DARK, va='top', ha='left')
        fig.text(x_title, y, title, fontsize=11, fontweight='bold', color=DARK, va='top', ha='left')


def savefig(fig, outbase):
    fig.savefig(outbase + '.png', dpi=450, bbox_inches='tight')
    fig.savefig(outbase + '.pdf', bbox_inches='tight')
    fig.savefig(outbase + '.svg', bbox_inches='tight')
    plt.close(fig)


def curve_from_beta(beta_cos, beta_sin, x):
    return beta_cos * np.cos(2*np.pi*x/24.0) + beta_sin * np.sin(2*np.pi*x/24.0)


def load_tables(base):
    """Read frozen results from their ORIGINAL project subfolders.

    The three inputs intentionally come from two different previous analyses.
    A copy placed in the project root also works, for testing or portability.
    This lookup never silently substitutes a similarly named older output.
    """
    from pathlib import Path
    root = Path(base).expanduser().resolve()
    candidates = {
        'integrated': [
            root / 'FIG3_LAYER23_FAMILY_RECOVERY_BMAC_STABILITY_V2' / '02_UPDATED_69_FAMILY_INTEGRATED_RESULTS.csv',
            root / '02_UPDATED_69_FAMILY_INTEGRATED_RESULTS.csv',
        ],
        'profiles': [
            root / 'FIG3_FIG6_LAYER2_LAYER3_BRIDGE_W999' / '02_LAYER3_M2_LINEAGE_CONDITIONAL_PROFILES.csv',
            root / '02_LAYER3_M2_LINEAGE_CONDITIONAL_PROFILES.csv',
        ],
        'within_tissue': [
            root / 'FIG3_LAYER23_FAMILY_RECOVERY_BMAC_STABILITY_V2' / '03_WITHIN_TISSUE_ASSAY.csv',
            root / '03_WITHIN_TISSUE_ASSAY.csv',
        ],
    }
    out = {}
    for key, paths in candidates.items():
        selected = next((path for path in paths if path.is_file()), None)
        if selected is None:
            tried = '\n'.join('  - ' + str(path) for path in paths)
            raise FileNotFoundError(
                f'Missing required input [{key}].\nLooked in:\n{tried}\n'
                'Please check that both analysis folders are under --data-root.'
            )
        out[key] = pd.read_csv(selected)
        print(f'[INPUT] {key}: {selected} ({len(out[key])} rows)', flush=True)
    return out


def plot_A(df, outdir, with_text):
    """Fig.6A: original Fig.3 module colors; M1/M2 are NOT mutually exclusive."""
    d = df.loc[df['status'].eq('FIT')].copy()
    d['M1_sig'] = d['BHq_recomputed_M1_available_families'].lt(0.05)
    d['M2_sig'] = d['BHq_recomputed_M2_available_families'].lt(0.05)

    # group by frozen ID, not by free-text label (prevents wrong color remapping)
    summ = (d.groupby(['module_id','module_label'], as_index=False)
            .agg(n=('family_id','nunique'), M1=('M1_sig','sum'), M2=('M2_sig','sum'))
            .sort_values(['M2','M1','n'],ascending=False).reset_index(drop=True))
    unknown = sorted(set(summ['module_id']) - set(FIG3_MODULE_COLORS))
    if unknown:
        raise ValueError(f'Fig.3 colors missing for module IDs: {unknown}')
    cols = [FIG3_MODULE_COLORS[m] for m in summ['module_id']]
    light_cols = [lighten_hex(c, 0.82) for c in cols]
    m1_cols = [lighten_hex(c, 0.34) for c in cols]
    y = np.arange(len(summ))

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    fig.subplots_adjust(top=0.85, bottom=0.17, right=0.92)
    setup_ax(ax)

    # 3 tracks per biological module: total, shared, interaction.
    # Original hue is conserved for all tracks, as in Fig.3.
    ax.barh(y, summ['n'], color=light_cols, edgecolor='none', height=0.76, zorder=1)
    ax.barh(y - 0.18, summ['M1'], color=m1_cols, edgecolor='none', height=0.27, zorder=3)
    ax.barh(y + 0.18, summ['M2'], color=cols, edgecolor='none', height=0.27, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(summ['module_label'] if with_text else [])
    ax.invert_yaxis()
    ax.set_xlim(0, max(summ['n'])+1.7)
    ax.set_xticks(np.arange(0, int(max(summ['n']))+2, 2))

    if with_text:
        ax.set_xlabel('Number of family-representative pathways')
        fig_panel_header(fig, 'A',
            'Shared and lineage-dependent phase associations across functional modules',True)
        # The right-hand summary uses explicit per-module denominators.
        for yi, (_, r) in zip(y, summ.iterrows()):
            ax.text(float(r['n'])+0.12, yi,
                    f"{int(r['M1'])}/{int(r['M2'])}/{int(r['n'])}",
                    ha='left', va='center', fontsize=7, color=GREY)
        # Use a single demonstration hue in the legend; actual row hue = Fig.3 module.
        sample = FIG3_MODULE_COLORS['M02_MAPK_RTK_GROWTH_FACTOR']
        legends = [
            Rectangle((0,0),1,1,facecolor=lighten_hex(sample,0.82),edgecolor='none'),
            Rectangle((0,0),1,1,facecolor=lighten_hex(sample,0.34),edgecolor='none'),
            Rectangle((0,0),1,1,facecolor=sample,edgecolor='none'),
        ]
        fig.legend(legends,
                   ['All fitted', 'Shared association (M1)', 'Lineage interaction (M2)'],
                   loc='lower center', bbox_to_anchor=(0.62,0.012),
                   frameon=False, ncol=3, fontsize=8, columnspacing=1.2)
        fig.text(0.60,0.058,'Right labels: M1 / M2 / total; M1 and M2 may overlap',
                 fontsize=7, ha='center', color=GREY)
        # Category names are identified by the same frozen Fig.3 color.
        from matplotlib.colors import to_rgb
        for tick, c in zip(ax.get_yticklabels(), cols):
            rgb=np.asarray(to_rgb(c))
            tick.set_color(tuple(0.70*rgb+0.30*np.array([0.16,0.18,0.20])))
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticklabels([])

    savefig(fig, os.path.join(outdir,
         f'FIG6_A_module_landscape_{"text" if with_text else "notext"}'))


def plot_B(df, outdir, with_text):
    d = df[df['pathway'].isin(MAIN_SHARED)].copy()
    order = [p for p in MAIN_SHARED if p in set(d['pathway'])]
    x = np.linspace(0, 24, 241)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharex=True)
    fig.subplots_adjust(top=0.83, bottom=0.12, hspace=0.28, wspace=0.20)
    if with_text:
        fig_panel_header(fig, 'B', 'Shared phase-associated metabolic and proteostasis programs', True)
    axes = axes.flatten()
    for i, pw in enumerate(order):
        ax = axes[i]
        setup_ax(ax)
        r = d[d['pathway'] == pw].iloc[0]
        y = curve_from_beta(r['beta_common_cos'], r['beta_common_sin'], x)
        amp = np.sqrt(r['beta_common_cos']**2 + r['beta_common_sin']**2)
        ax.plot(x, y, color=M1_COLOR, linewidth=2.2)
        ax.axhline(0, color=LIGHT_GREY, linewidth=1.0)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        if with_text:
            ax.set_title(nice_name(pw), fontsize=9, color=DARK, pad=10)
            txt = f"ΔR²={r['deltaR2_M1_vs_M0']:.3f}\nq={r['BHq_recomputed_M1_available_families']:.3g}\nΔRMSE={r['mean_deltaRMSE_M1']:.4f}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha='left', va='top', fontsize=6.5, color=GREY)
            if i % 2 == 0:
                ax.set_ylabel('Centered pathway score')
            if i >= 2:
                ax.set_xlabel('Inferred B0 phase (h)')
        else:
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticklabels([])
            if i < 2:
                ax.set_xticklabels([])
    savefig(fig, os.path.join(outdir, f'FIG6_B_shared_programs_{"text" if with_text else "notext"}'))


def plot_C(prof, outdir, with_text):
    d = prof[prof['pathway'].isin(MAIN_LINEAGE)].copy()
    d['lineage'] = d['lineage'].replace({'Monocyte/Macrophage': 'Macrophage/Monocyte'})
    x = np.linspace(0, 24, 241)
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.9), sharex=True)
    fig.subplots_adjust(top=0.93, bottom=0.12, hspace=0.20)
    if with_text:
        fig_panel_header(fig, 'C', 'Inflammatory programs show lineage-specific phase associations', True)
    for i, pw in enumerate(MAIN_LINEAGE):
        ax = axes[i]
        setup_ax(ax)
        sub = d[d['pathway'] == pw].copy()
        for lin in LINEAGE_ORDER:
            ss = sub[sub['lineage'] == lin]
            if ss.empty:
                continue
            r = ss.iloc[0]
            y = curve_from_beta(r['beta_cos'], r['beta_sin'], x)
            lw = 2.4 if lin in ['B cell', 'Macrophage/Monocyte'] else 1.6
            alpha = 1.0 if lin in ['B cell', 'Macrophage/Monocyte'] else 0.85
            ax.plot(x, y, color=LINEAGE_COLORS[lin], linewidth=lw, alpha=alpha, label=lin)
        ax.axhline(0, color=LIGHT_GREY, linewidth=1.0)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        if with_text:
            ax.set_title(nice_name(pw), fontsize=9, color=DARK, pad=8)
            ax.set_ylabel('Centered pathway score')
            if i == 2:
                ax.set_xlabel('Inferred B0 phase (h)')
            sb = sub[sub['lineage']=='B cell']
            sm = sub[sub['lineage']=='Macrophage/Monocyte']
            if not sb.empty and not sm.empty:
                gap = abs(float(sb.iloc[0]['peak_B0_h']) - float(sm.iloc[0]['peak_B0_h']))
                gap = min(gap, 24-gap)
                ax.text(0.98, 0.92, f'B–Macro gap ≈ {gap:.1f} h', transform=ax.transAxes, ha='right', va='top', fontsize=7, color=GREY)
        else:
            ax.set_title('')
            ax.set_ylabel('')
            ax.set_yticklabels([])
            if i < 2:
                ax.set_xticklabels([])
    if with_text:
        handles, labels = axes[-1].get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        axes[-1].legend(uniq.values(), uniq.keys(), ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.36))
    savefig(fig, os.path.join(outdir, f'FIG6_C_lineage_inflammatory_{"text" if with_text else "notext"}'))


def plot_D(wt, outdir, with_text):
    d = wt[wt['pathway'].isin(MAIN_LINEAGE) & (wt['status']=='FIT') & (wt['n_independent_donors']>=2)].copy()
    d['row_label'] = d['tissue'] + ' | ' + d['assay']
    rows = d[['row_label','tissue','assay','n_independent_donors']].drop_duplicates().copy()
    pv = d.pivot_table(index='row_label', columns='pathway', values='phase_gap_h', aggfunc='first')
    rows['mean_gap'] = rows['row_label'].map(pv.mean(axis=1))
    rows = rows.sort_values(['mean_gap','tissue','assay'])
    row_order = rows['row_label'].tolist()
    mat = d.pivot_table(index='row_label', columns='pathway', values='phase_gap_h', aggfunc='first').reindex(row_order)
    pathways = [p for p in MAIN_LINEAGE if p in mat.columns]
    mat = mat[pathways]
    fig, ax = plt.subplots(figsize=(5.8, max(4.8, 0.28*len(row_order)+1.8)))
    fig.subplots_adjust(top=0.93)
    if with_text:
        fig_panel_header(fig, 'D', 'B cell–Macrophage phase relationships vary across tissue contexts', True)
    cmap = plt.cm.magma_r.copy()
    cmap.set_bad('#F4F6F8')
    im = ax.imshow(mat.values, aspect='auto', cmap=cmap, vmin=0, vmax=12)
    if with_text:
        ylabels = []
        for rlab in row_order:
            rr = rows[rows['row_label']==rlab].iloc[0]
            ylabels.append(f"{rr['tissue']} | {rr['assay']} (n={int(rr['n_independent_donors'])})")
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_xticks(np.arange(len(pathways)))
        ax.set_xticklabels([nice_name(x).replace(' Signaling','') for x in pathways], rotation=25, ha='right')
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('Absolute phase gap (h)')
    else:
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(len(pathways)))
        ax.set_xticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(pathways), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_order), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    savefig(fig, os.path.join(outdir, f'FIG6_D_tissue_context_heatmap_{"text" if with_text else "notext"}'))


def plot_E(wt, outdir, with_text):
    d = wt[(wt['pathway'] == REP_PATHWAY) & (wt['status']=='FIT')].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.90, bottom=0.14, hspace=0.22, wspace=0.20)
    if with_text:
        fig_panel_header(fig, 'E', 'Representative tissue-specific B cell vs Macrophage phase profiles', True)
    axes = axes.flatten()
    x = np.linspace(0, 24, 241)
    for i, (tissue, assay) in enumerate(REP_TISSUES):
        ax = axes[i]
        setup_ax(ax)
        sub = d[(d['tissue'] == tissue) & (d['assay'] == assay)]
        if sub.empty:
            ax.axis('off')
            continue
        r = sub.iloc[0]
        yb = curve_from_beta(r['beta_B_cos'], r['beta_B_sin'], x)
        ym = curve_from_beta(r['beta_M_cos'], r['beta_M_sin'], x)
        ax.plot(x, yb, color=B_COLOR, linewidth=2.4, label='B cell')
        ax.plot(x, ym, color=M_COLOR, linewidth=2.4, label='Macrophage/Monocyte')
        ax.axhline(0, color=LIGHT_GREY, linewidth=1.0)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        if with_text:
            ax.set_title(f"{tissue.title()} | {assay}\nΔphase={r['phase_gap_h']:.1f} h, n={int(r['n_independent_donors'])}", fontsize=8.5, color=DARK, pad=8)
            if i % 2 == 0:
                ax.set_ylabel('Centered pathway score')
            if i >= 2:
                ax.set_xlabel('Inferred B0 phase (h)')
        else:
            ax.set_title('')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_yticklabels([])
            if i < 2:
                ax.set_xticklabels([])
    if with_text:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.02))
    savefig(fig, os.path.join(outdir, f'FIG6_E_representative_tissues_{"text" if with_text else "notext"}'))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='.')
    ap.add_argument('--outdir', default='FIG6_PANELS_ABCDE_V5_2')
    args = ap.parse_args()
    ensure_dir(args.outdir)
    tabs = load_tables(args.data_root)
    for with_text in [True, False]:
        plot_A(tabs['integrated'], args.outdir, with_text)
        plot_B(tabs['integrated'], args.outdir, with_text)
        plot_C(tabs['profiles'], args.outdir, with_text)
        plot_D(tabs['within_tissue'], args.outdir, with_text)
        plot_E(tabs['within_tissue'], args.outdir, with_text)
    print('DONE', os.path.abspath(args.outdir))

if __name__ == '__main__':
    main()
