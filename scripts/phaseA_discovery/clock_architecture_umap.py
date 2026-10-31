"""UMAP of existing donor x tissue x cell-type CLOCK features, not single cells.

The original feature definitions, row filter, lineage mapping and matrix
preparation are extracted from the inspected V2 runner, without importing its
analysis module. Fit one UMAP per feature set; all colorings reuse its coordinates.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tarfile
import textwrap

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault('TABULA_PLOT_DEPS', str(PROJECT / '.umapdeps'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(PROJECT / '.numba_cache'))
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from publication_style import (
    ROOT, np, plt, CELLTYPE_COLORS, LINEAGE_COLORS, LINEAGE_ORDER, DISPLAY_LABELS,
    apply_publication_style, clean_axis, place_external_legend, save_figure,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex

V1 = ROOT.parent / 'TabulaCircadian_deliverables'
V2 = ROOT.parent / 'TabulaCircadian_deliverables_v2'
OUT = ROOT / 'figures_revised' / 'clock_architecture_umap'
FEATURE_PATH = V1 / 'source_data/pseudobulk__clock_architecture_features.tsv'
RUNNER = V2 / 'scripts/revision_v2_runner.py'
SET_LABELS = {'clock_architecture': 'Original clock-feature matrix', 'core_clock': 'Core clock',
              'expanded_clock': 'Expanded clock', 'clock_modules': 'Clock modules'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    roots=[V1,V2]
    return {str(p):sha(p) for root in roots for p in sorted(root.rglob('*')) if p.is_file()}

def original_spec():
    constants={
        'SEED','POSITIVE_LIMB','NEGATIVE_LIMB','AUXILIARY_CLOCK','CORE_CLOCK','CLOCK_GENES',
        'CORE_FEATURE_COLS','EXPANDED_FEATURE_COLS','MODULE_COLS','CLOCK_FEATURE_COLS',
        'MIN_CELLS_PER_GROUP','LINEAGE_ORDER',
    }
    funcs={'lineage_label','add_lineage_columns','get_feature_matrix'}
    tree=ast.parse(RUNNER.read_text(encoding='utf-8')); nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):
            nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name in funcs:
            nodes.append(node)
    env={'np':np,'pd':pd,'StandardScaler':StandardScaler}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(RUNNER),'exec'),env)
    assert constants.issubset(env) and funcs.issubset(env)
    return env

def inspect_inputs():
    files=[p for root in (V1,V2) for p in root.rglob('*') if p.is_file()]
    suffixes={'.h5ad','.h5seurat','.rds','.rdata','.loom','.h5'}
    objects=[str(p) for p in files if p.suffix.lower() in suffixes or '.zarr' in str(p).lower()]
    headers={}
    embeddings=[]
    for p in files:
        if p.suffix.lower() in ('.tsv','.csv'):
            header=p.open(encoding='utf-8').readline().strip().split('\t' if p.suffix=='.tsv' else ',')
            headers[str(p)]=header
            if any('umap' in col.lower() for col in header):
                embeddings.append(str(p))
    archive=ROOT.parent/'TabulaCircadian_deliverables_v2.tar.gz'
    with tarfile.open(archive) as tar:
        members=tar.getnames()
    environment=pd.read_csv(V2/'statistics/package_environment.tsv',sep='\t').set_index('package').version.to_dict()
    report={
        'roots_inspected':[str(V1),str(V2),str(ROOT)],
        'original_file_count':len(files),'single_cell_objects':objects,
        'precomputed_umap_tables':embeddings,'tabular_headers_inspected':len(headers),
        'archive_members':len(members),
        'archive_single_cell_or_umap_candidates':[n for n in members if 'umap' in n.lower() or Path(n).suffix.lower() in suffixes],
        'original_environment':environment,
        'single_cell_transcriptomic_umap_status':'Deferred by user; single-cell objects likely remain on server.',
        'fallback_evidence':{
            'environment_file':str(V2/'statistics/package_environment.tsv'),
            'umap_learn':environment['umap-learn'],
            'runner_lines':'1689–1701: import umap and fit inside try; except Exception invokes TSNE',
            'conclusion':'umap-learn was not installed in the original recorded environment; the explicit exception fallback produced t-SNE.',
            'limitation':'The exact exception text/traceback was written to remote REVISION_STATUS.md, which is absent from the local package.'},
        'files':[str(p) for p in files],
    }
    return report

def validate_observations(features, env):
    valid=env['add_lineage_columns'](features[features.n_cells>=env['MIN_CELLS_PER_GROUP']].copy())
    if not valid.group_id.is_unique:
        raise ValueError('Duplicate pseudobulk group IDs.')
    pca=pd.read_csv(V2/'source_data/Fig1A_clock_architecture_pca_landscape_source.tsv',sep='\t')
    tsne=pd.read_csv(V2/'source_data/Fig1B_clock_architecture_nonlinear_landscape_source.tsv',sep='\t')
    columns=['group_id','donor_id','tissue','cell_type','broad_cell_class','n_cells','core_completeness_at_0.05']
    for reference in (pca,tsne):
        pd.testing.assert_frame_equal(valid[columns].reset_index(drop=True),reference[columns],check_exact=True)
        pd.testing.assert_series_equal(valid.priority_lineage.astype('string').reset_index(drop=True),
                                       reference.priority_lineage.astype('string'),check_names=False)
    expected_cols=pd.read_csv(V2/'source_data/input_clock_architecture_feature_columns.tsv',sep='\t',nrows=0).columns.tolist()
    assert features.columns.tolist()==expected_cols
    # Independent V1 export contains the same input features, confirming source provenance.
    exported=pd.read_csv(V1/'source_data/fig1__fig1_input_clock_architecture_features.tsv',sep='\t')
    # V1 re-exported decimal floats: tolerate only CSV round-trip precision.
    pd.testing.assert_frame_equal(features,exported,check_exact=False,rtol=1e-13,atol=1e-15)
    clock_cols=env['CLOCK_FEATURE_COLS']
    export_delta=float(np.nanmax(np.abs(features[clock_cols].to_numpy()-exported[clock_cols].to_numpy())))
    # Compare already-saved V2 module scores for every priority-lineage row.
    modules=pd.read_csv(V2/'source_data/pathway_module_scores_source.tsv',sep='\t').set_index('group_id')
    cols=[c for c in env['MODULE_COLS'] if c in modules]
    left=valid.set_index('group_id').loc[modules.index,cols]
    np.testing.assert_allclose(left,modules[cols],rtol=1e-13,atol=1e-15,equal_nan=True)
    return valid,{'rows_before_filter':len(features),'rows_after_original_filter':len(valid),
                  'min_cells_per_group':env['MIN_CELLS_PER_GROUP'],
                  'pca_and_tsne_group_order_metadata_and_lineages_identical':True,
                  'v2_input_column_schema_identical':True,'independent_v1_feature_export_matches_to_csv_precision':True,
                  'v1_export_max_abs_clock_feature_difference':export_delta,
                  'v2_saved_module_values_match':True,
                  'categories':valid[['cell_type','tissue','donor_id']].nunique().to_dict()}

def category_palette(labels):
    labels=sorted(labels)
    # Interleave established qualitative palettes for large tissue/cell-type legends.
    colors=[]
    for i in range(20):
        for cmap in ('tab20','tab20b','tab20c'):
            colors.append(to_hex(plt.get_cmap(cmap)(i)))
    if len(labels)>len(colors):
        import colorsys
        for i in range(len(labels)-len(colors)):
            colors.append(to_hex(colorsys.hsv_to_rgb((i*.61803398875)%1,.52,.65 if i%2 else .88)))
    return dict(zip(labels,colors))

def color_spec(data, color_by, palettes):
    if color_by=='cell_identity':
        labels=data.priority_lineage.fillna('other').astype(str)
        palette=LINEAGE_COLORS
        display={k:DISPLAY_LABELS[k] for k in palette}
        title='Cell identity (major lineage)'
        order=['other']+LINEAGE_ORDER
    else:
        column={'tissue_context':'tissue','donor':'donor_id','cell_type':'cell_type'}[color_by]
        labels=data[column].astype(str); palette=palettes[column]
        display={k:k for k in palette}; order=list(palette)
        title={'tissue_context':'Tissue context','donor':'Donor','cell_type':'Cell identity (original cell-type annotations)'}[color_by]
    return labels,palette,display,order,title

def plot_coloring(data,set_name,color_by,palettes):
    labels,palette,display,order,color_title=color_spec(data,color_by,palettes)
    many=color_by in ('tissue_context','cell_type')
    if many:
        ncol=3 if color_by=='tissue_context' else 4
        rows=int(np.ceil(len(order)/ncol))
        height=max(8.7,rows*.31+1.8)
        fig=plt.figure(figsize=(19 if ncol==3 else 25,height))
        gs=fig.add_gridspec(1,2,left=.045,right=.985,bottom=.65/height,top=1-1.15/height,
                            width_ratios=[7,10.5 if ncol==3 else 16.5],wspace=.06)
        ax=fig.add_subplot(gs[0]); lax=fig.add_subplot(gs[1])
    else:
        fig=plt.figure(figsize=(11,6.4))
        gs=fig.add_gridspec(1,2,left=.08,right=.98,bottom=.12,top=.84,width_ratios=[3.5,1.5],wspace=.10)
        ax=fig.add_subplot(gs[0]); lax=fig.add_subplot(gs[1]); ncol=2 if color_by=='donor' else 1
    for category in order:
        subset=data[labels.eq(category)]
        ax.scatter(subset.UMAP1,subset.UMAP2,s=12 if many else 15,c=palette[category],
                   alpha=.4 if category=='other' else .83,edgecolors='none',
                   zorder=1 if category=='other' else 3,gid='group_ids:'+','.join(subset.group_id))
    ax.set(xlabel='UMAP 1',ylabel='UMAP 2'); clean_axis(ax)
    # All colorings share coordinate limits; scatter aspect preserves geometry.
    delta=np.ptp(data[['UMAP1','UMAP2']].to_numpy(),axis=0)
    ax.set_xlim(data.UMAP1.min()-.05*delta[0],data.UMAP1.max()+.05*delta[0])
    ax.set_ylim(data.UMAP2.min()-.05*delta[1],data.UMAP2.max()+.05*delta[1]); ax.set_aspect('equal',adjustable='box')
    handles=[Line2D([],[],marker='o',ls='',markersize=5,color=palette[k],
                    label=textwrap.fill(display[k],45) if color_by=='cell_type' else display[k]) for k in order]
    legend=place_external_legend(lax,handles,title=color_title,ncol=ncol,loc='center')
    if many:
        for t in legend.get_texts(): t.set_fontsize(8)
    fig.suptitle('Clock-architecture UMAP',y=1-.22/height if many else .975,fontsize=14)
    fig.text(.5,1-.55/height if many else .925,f'{SET_LABELS[set_name]} · {color_title}',ha='center',fontsize=11)
    fig.text(.5,1-.84/height if many else .88,f'{len(data):,} donor × tissue × cell-type observations; clock features only',ha='center',fontsize=9,color='#666666')
    return fig

def verify_plotted_coordinates(fig,data):
    expected=data.set_index('group_id')[['UMAP1','UMAP2']]; observed=[]
    for collection in fig.axes[0].collections:
        suffix=collection.get_gid().split(':',1)[1]
        ids=suffix.split(',') if suffix else []
        np.testing.assert_array_equal(collection.get_offsets().data,expected.loc[ids].to_numpy())
        observed.extend(ids)
    assert len(observed)==len(data) and set(observed)==set(data.group_id)

def fit_or_load(valid,matrix,columns,set_name,env,output,recompute=False):
    params=dict(n_neighbors=20,min_dist=.25,metric='euclidean',random_state=env['SEED'],n_jobs=1)
    coordinate_file=output/'source_data'/f'clock_architecture_UMAP_{set_name}_coordinates.tsv'
    meta_file=coordinate_file.with_suffix('.json')
    fingerprint=hashlib.sha256(matrix.to_numpy(dtype=np.float64).tobytes()+
                               '\n'.join(valid.group_id).encode()+'\n'.join(columns).encode()).hexdigest()
    signature={'standardized_matrix_sha256':fingerprint,'feature_columns':columns,
               'parameters':params,'umap_version':importlib.metadata.version('umap-learn')}
    if coordinate_file.exists() and meta_file.exists() and not recompute:
        previous=json.loads(meta_file.read_text(encoding='utf-8'))
        if previous['signature']!=signature or previous['coordinate_sha256']!=sha(coordinate_file):
            raise ValueError('UMAP cache differs from the input, parameters, or version; review before --recompute.')
        data=pd.read_csv(coordinate_file,sep='\t',float_precision='round_trip')
        assert data.group_id.tolist()==valid.group_id.tolist()
        print(f'Reusing saved UMAP: {set_name}',flush=True)
        return data,previous
    print(f'Fitting UMAP: {set_name}, matrix {matrix.shape}',flush=True)
    import umap
    reducer=umap.UMAP(**params)
    coords=reducer.fit_transform(matrix.to_numpy())
    assert coords.shape==(len(valid),2) and np.isfinite(coords).all()
    data=valid[['group_id','donor_id','tissue','cell_type','broad_cell_class','priority_lineage','n_cells']].copy()
    data['UMAP1']=coords[:,0]; data['UMAP2']=coords[:,1]
    coordinate_file.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(coordinate_file,sep='\t',index=False,float_format='%.17g')
    standardized=matrix.copy(); standardized.insert(0,'group_id',valid.group_id.to_numpy())
    standardized.to_csv(output/'source_data'/f'clock_architecture_UMAP_{set_name}_standardized_input.tsv',sep='\t',index=False,float_format='%.17g')
    metadata={'signature':signature,'coordinate_sha256':sha(coordinate_file),
              'n_observations':len(valid),'n_features':len(columns),'actual_parameters':reducer.get_params(),
              'input_file':str(FEATURE_PATH),'input_sha256':sha(FEATURE_PATH),
              'interpretation':'Clock-feature pseudobulk architecture; not a single-cell transcriptomic embedding.'}
    meta_file.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    # Use the saved values for every display, including the first run.
    return pd.read_csv(coordinate_file,sep='\t',float_precision='round_trip'),metadata

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sets',nargs='+',choices=list(SET_LABELS),default=list(SET_LABELS))
    parser.add_argument('--recompute',action='store_true')
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    before=snapshot(); inspection=inspect_inputs(); env=original_spec()
    features=pd.read_csv(FEATURE_PATH,sep='\t')
    valid,checks=validate_observations(features,env)
    inspection['input_verification']=checks
    (OUT/'local_data_inspection.json').write_text(json.dumps(inspection,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)
    if args.inspect_only:
        return
    apply_publication_style()
    definitions={'clock_architecture':env['CLOCK_FEATURE_COLS'],'core_clock':env['CORE_FEATURE_COLS'],
                 'expanded_clock':env['EXPANDED_FEATURE_COLS'],'clock_modules':env['MODULE_COLS']}
    palettes={col:category_palette(valid[col].unique()) for col in ['tissue','donor_id','cell_type']}
    # Exact major cell-type names keep their canonical colors even in the detailed panel.
    aliases={'B cell':'B cell','T cell':'T cell','fibroblast':'Fibroblast','endothelial cell':'Endothelial',
             'macrophage':'Macrophage/monocyte','monocyte':'Macrophage/monocyte'}
    for name,label in aliases.items():
        if name in palettes['cell_type']:palettes['cell_type'][name]=CELLTYPE_COLORS[label]
    (OUT/'category_palettes.json').write_text(json.dumps(palettes,indent=2),encoding='utf-8')
    results=[]
    for set_name in args.sets:
        requested=definitions[set_name]
        assert set(requested).issubset(valid.columns),'Missing existing feature definitions.'
        matrix,columns=env['get_feature_matrix'](valid,requested)
        if not np.isfinite(matrix.to_numpy()).all() or len(columns)<2:
            raise ValueError('Insufficient finite features; no alternative analysis is substituted.')
        data,metadata=fit_or_load(valid,matrix,columns,set_name,env,OUT,args.recompute)
        colors=['cell_identity','tissue_context','donor']
        if set_name=='clock_architecture':colors.append('cell_type')
        figures=[]
        for color_by in colors:
            fig=plot_coloring(data,set_name,color_by,palettes)
            verify_plotted_coordinates(fig,data)
            stem=f'Fig_clock_architecture_UMAP_{set_name}_by_{color_by}'
            result=save_figure(fig,stem,OUT); figures.append(result)
            print(f'Saved {stem}',flush=True)
        results.append({'feature_set':set_name,'metadata':metadata,'figures':figures})
    after=snapshot(); assert before==after,'An original source file changed.'
    audit={'input_verification':checks,'results':results,'original_files_unchanged':len(before),
           'original_sha256':before,'versions':{p:importlib.metadata.version(p) for p in
             ['numpy','pandas','matplotlib','seaborn','scipy','scikit-learn','umap-learn','numba','pynndescent']},
           'tsne_preserved':str(ROOT/'figures_revised/Fig_clock_tSNE.png'),
           'single_cell_transcriptomic_umap':'Deferred by user; no local single-cell object/embedding found.'}
    (OUT/'umap_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f'Complete. {len(before)} original files unchanged.',flush=True)

if __name__=='__main__':
    main()
