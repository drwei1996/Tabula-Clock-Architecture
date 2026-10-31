"""New descriptive centroid/distance summaries from frozen clock results.

No PCA/UMAP fitting, no feature redefinition, no edits to existing statistics.
Uncertainty uses donor/tissue block multiplier bootstrap, not independent pairs.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re
import sys

if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from clock_architecture_umap import (
    ROOT, V2, FEATURE_PATH, original_spec, validate_observations, snapshot, sha,
    np, pd, plt, LINEAGE_ORDER, DISPLAY_LABELS, LINEAGE_COLORS,
    apply_publication_style, clean_axis, place_external_legend, save_figure,
)
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits

OUTPUT = ROOT/'figures_revised'/'clock_centroid_distances'
STANDARDIZED = ROOT/'figures_revised/clock_architecture_umap/source_data/clock_architecture_UMAP_clock_architecture_standardized_input.tsv'
PCA_SOURCE = V2/'source_data/Fig1A_clock_architecture_pca_landscape_source.tsv'
WITHIN = 'Within lineage / between tissues'
BETWEEN = 'Between lineages / within tissue'
BOOTSTRAP_SEED = 20260913
DISTANCE_COLORS = ['#687F92','#B18D73']

def load_frozen_inputs():
    env=original_spec()
    raw=pd.read_csv(FEATURE_PATH,sep='\t')
    valid,verification=validate_observations(raw,env)
    saved=pd.read_csv(STANDARDIZED,sep='\t',float_precision='round_trip').set_index('group_id')
    prepared,columns=env['get_feature_matrix'](valid,env['CLOCK_FEATURE_COLS'])
    assert saved.columns.tolist()==columns
    assert saved.index.tolist()==valid.group_id.tolist()
    np.testing.assert_array_equal(saved.to_numpy(),prepared.to_numpy())
    pca=pd.read_csv(PCA_SOURCE,sep='\t',float_precision='round_trip').set_index('group_id')
    metadata=valid.set_index('group_id')[['donor_id','tissue','cell_type','broad_cell_class','n_cells','priority_lineage']].copy()
    metadata[['PC1','PC2']]=pca.loc[metadata.index,['PC1','PC2']]
    # Existing analysis minimum, read directly from the inspected runner.
    text=(V2/'scripts/revision_v2_runner.py').read_text(encoding='utf-8')
    threshold=int(re.search(r'^MIN_DONORS_ADEQUATE = (\d+)',text,re.M).group(1))
    assert threshold==3
    verification.update(n_features=len(columns),standardized_input_exactly_matches_original_preparation=True,
                        minimum_donors_adequate=threshold,pca_refitted=False)
    return metadata,saved,verification,threshold

def centroid_tables(meta,features,threshold):
    primary=meta[meta.priority_lineage.notna()].copy()
    primary['lineage']=primary.priority_lineage.astype(str)
    combined=primary.join(features)
    cols=features.columns.tolist()
    overall=combined.groupby('lineage',observed=True)[cols+['PC1','PC2']].mean()
    records=[]; vectors=[]
    for (lineage,tissue),part in combined.groupby(['lineage','tissue'],sort=True,observed=True):
        centroid=part[cols].mean().to_numpy()
        donor_count=part.donor_id.nunique()
        records.append(dict(lineage=lineage,lineage_label=DISPLAY_LABELS[lineage],tissue=tissue,
                            n_donors=int(donor_count),n_observations=len(part),n_cell_types=int(part.cell_type.nunique()),
                            centroid_PC1=float(part.PC1.mean()),centroid_PC2=float(part.PC2.mean()),
                            distance_to_lineage_centroid=float(np.linalg.norm(centroid-overall.loc[lineage,cols].to_numpy())),
                            adequate_support=bool(donor_count>=threshold),
                            support='adequate' if donor_count>=threshold else 'moderate' if donor_count==2 else 'low',
                            donor_ids=';'.join(sorted(part.donor_id.unique())),
                            centroid_definition='Equal-weight mean of existing donor × tissue × cell-type pseudobulks'))
        vectors.append(centroid)
    table=pd.DataFrame(records)
    matrix=np.vstack(vectors)
    # Verify support is exactly the existing result, not an ad hoc new cutoff.
    reference=pd.read_csv(V2/'statistics/priority_lineage_tissue_coverage.tsv',sep='\t')
    check=table.merge(reference,left_on=['lineage','tissue'],right_on=['priority_lineage','tissue'],suffixes=('','_original'),validate='one_to_one')
    assert len(check)==len(reference)==len(table)
    np.testing.assert_array_equal(check.n_donors,check.n_donors_original)
    np.testing.assert_array_equal(check.n_observations,check.n_groups)
    assert check.support.eq(check.support_original).all()
    overall_rows=[]; inclusion_rows=[]
    for lin in LINEAGE_ORDER:
        allrows=combined[combined.lineage.eq(lin)]
        tab=table[table.lineage.eq(lin)]
        eligible=tab[tab.adequate_support]
        included=allrows[allrows.tissue.isin(eligible.tissue)]
        overall_rows.append(dict(lineage=lin,lineage_label=DISPLAY_LABELS[lin],n_tissues=len(tab),
                                 n_donors=int(allrows.donor_id.nunique()),n_observations=len(allrows),
                                 centroid_PC1=overall.loc[lin,'PC1'],centroid_PC2=overall.loc[lin,'PC2'],
                                 **dict(zip(cols,overall.loc[lin,cols]))))
        inclusion_rows.append(dict(lineage=lin,lineage_label=DISPLAY_LABELS[lin],
                                   figure_A_tissues=len(tab),figure_A_donors=int(allrows.donor_id.nunique()),
                                   figure_A_observations=len(allrows),
                                   figure_B_tissues=len(eligible),figure_B_donors=int(included.donor_id.nunique()),
                                   figure_B_observations=len(included),excluded_low_support_tissues=len(tab)-len(eligible)))
    feature_centroids=table[['lineage','tissue']].join(pd.DataFrame(matrix,columns=cols))
    return table,matrix,pd.DataFrame(overall_rows),pd.DataFrame(inclusion_rows),feature_centroids

def pair_tables(centroids,matrix):
    supported=centroids[centroids.adequate_support]
    records=[]
    for lin in LINEAGE_ORDER:
        indices=supported.index[supported.lineage.eq(lin)].tolist()
        for a,b in itertools.combinations(indices,2):
            left,right=centroids.loc[a],centroids.loc[b]
            records.append(dict(comparison_type=WITHIN,lineage=lin,tissue_A=left.tissue,tissue_B=right.tissue,
                                tissue='',lineage_A=lin,lineage_B=lin,distance=float(np.linalg.norm(matrix[a]-matrix[b])),
                                n_donors_A=int(left.n_donors),n_donors_B=int(right.n_donors),
                                n_shared_donors=len(set(left.donor_ids.split(';'))&set(right.donor_ids.split(';'))),
                                centroid_index_A=a,centroid_index_B=b))
    for tissue,part in supported.groupby('tissue',sort=True):
        for a,b in itertools.combinations(part.index,2):
            left,right=centroids.loc[a],centroids.loc[b]
            records.append(dict(comparison_type=BETWEEN,lineage='',tissue_A=tissue,tissue_B=tissue,
                                tissue=tissue,lineage_A=left.lineage,lineage_B=right.lineage,
                                distance=float(np.linalg.norm(matrix[a]-matrix[b])),
                                n_donors_A=int(left.n_donors),n_donors_B=int(right.n_donors),
                                n_shared_donors=len(set(left.donor_ids.split(';'))&set(right.donor_ids.split(';'))),
                                centroid_index_A=a,centroid_index_B=b))
    return pd.DataFrame(records)

def weighted_median(values,weights):
    """Midpoint convention equals np.median for uniform weights, including even n."""
    order=np.argsort(values,kind='stable'); values=values[order]; weights=weights[order]
    cumulative=np.cumsum(weights); halfway=cumulative[-1]/2
    i=np.searchsorted(cumulative,halfway,side='left')
    if i+1<len(values) and np.isclose(cumulative[i],halfway,rtol=1e-13,atol=0):
        return float((values[i]+values[i+1])/2)
    return float(values[i])

def bootstrap(meta,features,centroids,pairs,iterations,seed):
    """Positive crossed donor/tissue multiplier bootstrap, fixed lineage strata.

    A single Exp(1) weight is drawn for each donor and reused for ALL that
    donor's observations. Each lineage-tissue centroid is recomputed from
    donor-weighted pseudobulks. A single Exp(1) tissue weight is shared by all
    lineages and edges involving that tissue. Within edges use u_A*u_B;
    tissue-matched between edges use u_t, not u_t**2. No distance gets an
    independently sampled weight. Positive weights preserve the support mask.

    Percentile intervals are approximate dependence-aware sensitivity intervals;
    no exact permutation test or distribution-free calibration is claimed.
    """
    if iterations<5000:raise ValueError('At least 5000 bootstrap iterations required.')
    donors=sorted(meta[meta.priority_lineage.notna()].donor_id.unique())
    donor_index={d:i for i,d in enumerate(donors)}
    tissues=sorted(centroids[centroids.adequate_support].tissue.unique())
    tissue_index={t:i for i,t in enumerate(tissues)}
    group_index={(r.lineage,r.tissue):i for i,r in centroids.iterrows()}
    counts=np.zeros((len(centroids),len(donors)))
    sums=np.zeros((len(centroids),len(donors),features.shape[1]))
    for group_id,row in meta[meta.priority_lineage.notna()].iterrows():
        g=group_index[(str(row.priority_lineage),row.tissue)]; d=donor_index[row.donor_id]
        counts[g,d]+=1; sums[g,d]+=features.loc[group_id].to_numpy()
    # All-ones weights must recover the unperturbed centroids exactly.
    point=sums.sum(axis=1)/counts.sum(axis=1)[:,None]
    a=pairs.centroid_index_A.to_numpy(); b=pairs.centroid_index_B.to_numpy()
    np.testing.assert_allclose(np.linalg.norm(point[a]-point[b],axis=1),pairs.distance,rtol=1e-13,atol=1e-13)
    ta=np.array([tissue_index[t] for t in pairs.tissue_A]); tb=np.array([tissue_index[t] for t in pairs.tissue_B])
    within=pairs.comparison_type.eq(WITHIN).to_numpy(); between=~within
    lineage_masks={lin:within & pairs.lineage.eq(lin).to_numpy() for lin in LINEAGE_ORDER}
    rng=np.random.default_rng(seed); records=[]
    for iteration in range(iterations):
        donor_weight=rng.exponential(1,size=len(donors))
        tissue_weight=rng.exponential(1,size=len(tissues))
        denominator=counts@donor_weight
        group_means=np.einsum('gdf,d->gf',sums,donor_weight,optimize=False)/denominator[:,None]
        distances=np.linalg.norm(group_means[a]-group_means[b],axis=1)
        weights=tissue_weight[ta].copy()
        weights[within]*=tissue_weight[tb[within]]
        mw=weighted_median(distances[within],weights[within]); mb=weighted_median(distances[between],weights[between])
        rec={'iteration':iteration+1,'median_within':mw,'median_between':mb,'delta':mb-mw,
             'donor_only_delta':float(np.median(distances[between])-np.median(distances[within]))}
        rec.update({f'within_{lin}':weighted_median(distances[mask],weights[mask]) for lin,mask in lineage_masks.items()})
        records.append(rec)
        if (iteration+1)%1000==0:print(f'Bootstrap {iteration+1}/{iterations}',flush=True)
    result=pd.DataFrame(records)
    assert np.isfinite(result.to_numpy()).all()
    return result,{'method':'Crossed donor/tissue exponential multiplier bootstrap',
                   'iterations':iterations,'seed':seed,'n_donor_blocks':len(donors),'n_tissue_blocks':len(tissues),
                   'lineages_treated_as_fixed_strata':LINEAGE_ORDER,'support_mask_fixed':True,
                   'independent_pair_resampling':False,'permutation_p_value':None,
                   'interval':'2.5th–97.5th percentiles; approximate conditional uncertainty, not an exact test'}

def summarize(pairs,boot,inclusion):
    rows=[]
    for typ,col in [(WITHIN,'median_within'),(BETWEEN,'median_between')]:
        values=pairs.loc[pairs.comparison_type.eq(typ),'distance']
        rows.append(dict(summary_type='overall',comparison_type=typ,lineage='',n_comparisons=len(values),
                         median=float(values.median()),q25=float(values.quantile(.25)),q75=float(values.quantile(.75)),
                         bootstrap_95_ci_low=float(boot[col].quantile(.025)),bootstrap_95_ci_high=float(boot[col].quantile(.975))))
    delta=rows[1]['median']-rows[0]['median']
    rows.append(dict(summary_type='effect',comparison_type='Delta: between − within',lineage='',n_comparisons=len(pairs),
                     median=delta,q25=np.nan,q75=np.nan,bootstrap_95_ci_low=float(boot.delta.quantile(.025)),
                     bootstrap_95_ci_high=float(boot.delta.quantile(.975))))
    for lin in LINEAGE_ORDER:
        values=pairs.loc[pairs.comparison_type.eq(WITHIN)&pairs.lineage.eq(lin),'distance']
        rows.append(dict(summary_type='lineage_within',comparison_type=WITHIN,lineage=lin,n_comparisons=len(values),
                         median=float(values.median()),q25=float(values.quantile(.25)),q75=float(values.quantile(.75)),
                         bootstrap_95_ci_low=float(boot[f'within_{lin}'].quantile(.025)),
                         bootstrap_95_ci_high=float(boot[f'within_{lin}'].quantile(.975))))
    table=pd.DataFrame(rows)
    return table

def axis_labels():
    text=(V2/'figures_main/Fig1A_clock_architecture_pca_landscape.svg').read_text(encoding='utf-8')
    return [re.search(rf'<!-- (PC{i} \([^<]+\)) -->',text).group(1) for i in (1,2)]

def figure_A(table,overall):
    fig=plt.figure(figsize=(11.8,8))
    gs=fig.add_gridspec(2,1,left=.085,right=.71,top=.86,bottom=.045,height_ratios=[8,1],hspace=.35)
    ax=fig.add_subplot(gs[0]); lax=fig.add_subplot(gs[1])
    for lin in LINEAGE_ORDER:
        part=table[table.lineage.eq(lin)]; center=overall.set_index('lineage').loc[lin]; color=LINEAGE_COLORS[lin]
        for _,r in part.iterrows():
            ax.plot([center.centroid_PC1,r.centroid_PC1],[center.centroid_PC2,r.centroid_PC2],
                    color=color,lw=.65,alpha=.35 if r.adequate_support else .12,zorder=1)
        for adequate in (False,True):
            group=part[part.adequate_support.eq(adequate)]
            ax.scatter(group.centroid_PC1,group.centroid_PC2,s=24 if adequate else 18,
                       facecolors=color if adequate else 'none',edgecolors=color,linewidth=.55,
                       alpha=.70 if adequate else .32,zorder=3 if adequate else 2)
        ax.scatter(center.centroid_PC1,center.centroid_PC2,s=175,color=color,edgecolors='#222222',linewidth=.8,zorder=5)
    # Dedicated side labels: no tissue-label selection and no hidden tissue points.
    centers=overall.sort_values('centroid_PC2')
    for position,(_,r) in zip(np.linspace(.15,.85,5),centers.iterrows()):
        ax.annotate(DISPLAY_LABELS[r.lineage],xy=(r.centroid_PC1,r.centroid_PC2),
                    xytext=(1.035,position),textcoords='axes fraction',fontsize=10,weight='bold',
                    color=LINEAGE_COLORS[r.lineage],va='center',annotation_clip=False,
                    arrowprops={'arrowstyle':'-','color':LINEAGE_COLORS[r.lineage],'lw':.8})
    labels=axis_labels(); ax.set(xlabel=labels[0],ylabel=labels[1]); clean_axis(ax)
    fig.suptitle('Cell identity and tissue-specific clock-architecture displacement',y=.98,fontsize=13)
    fig.text(.5,.93,'Centroids of existing pseudobulk observations · projection uses stored PCA scores',ha='center',fontsize=10,color='#666666')
    handles=[Line2D([],[],marker='o',ls='',color='#666666',markersize=10,markeredgecolor='black',label='Overall lineage centroid'),
             Line2D([],[],marker='o',ls='',color='#888888',markersize=5,label='Tissue centroid: ≥3 donors'),
             Line2D([],[],marker='o',ls='',color='#999999',markerfacecolor='none',markersize=5,label='Tissue centroid: <3 donors')]
    place_external_legend(lax,handles,ncol=3,loc='center')
    return fig

def figure_B(pairs,summary,iterations=5000):
    fig=plt.figure(figsize=(8.3,7))
    ax=fig.add_axes([.12,.23,.82,.57]); clean_axis(ax)
    rng=np.random.default_rng(BOOTSTRAP_SEED+1)
    for i,(typ,color) in enumerate(zip([WITHIN,BETWEEN],DISTANCE_COLORS)):
        values=pairs.loc[pairs.comparison_type.eq(typ),'distance'].to_numpy()
        parts=ax.violinplot([values],positions=[i],widths=.78,showextrema=False)
        for body in parts['bodies']:body.set_facecolor(color);body.set_edgecolor('none');body.set_alpha(.16)
        ax.scatter(i+rng.uniform(-.22,.22,len(values)),values,s=8,color=color,alpha=.27,edgecolors='none',zorder=2,gid=typ)
        ax.boxplot([values],positions=[i],widths=.21,showfliers=False,patch_artist=True,
                   boxprops={'facecolor':'white','edgecolor':'#333333','linewidth':1},
                   medianprops={'color':'#111111','linewidth':1.6},zorder=4)
        ax.text(i,1.015,f'n = {len(values):,} comparisons',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
    ax.set_xticks([0,1],['Within lineage\nbetween tissues','Between lineages\nwithin tissue'])
    ax.set(xlim=(-.6,1.6),ylabel='Clock-architecture distance'); ax.set_ylim(bottom=0)
    fig.suptitle('Between-lineage versus within-lineage tissue distance',y=.97,fontsize=13)
    effect=summary[summary.summary_type.eq('effect')].iloc[0]
    fig.text(.5,.89,f'Δ (between − within) = {effect["median"]:.3f}   |   95% bootstrap interval [{effect.bootstrap_95_ci_low:.3f}, {effect.bootstrap_95_ci_high:.3f}]',ha='center',fontsize=10)
    fig.text(.5,.065,f'Euclidean distance in the original 24-feature standardized space\nAll adequate-support comparisons shown; points are dependent pairs.\n{iterations:,} donor/tissue block-weight bootstrap draws; no permutation P value.',
             ha='center',fontsize=9,color='#555555',linespacing=1.65)
    return fig

def figure_supp(pairs):
    fig=plt.figure(figsize=(10.3,6.6)); ax=fig.add_axes([.25,.19,.67,.65]); clean_axis(ax)
    rng=np.random.default_rng(BOOTSTRAP_SEED+2)
    for i,lin in enumerate(LINEAGE_ORDER):
        values=pairs.loc[pairs.comparison_type.eq(WITHIN)&pairs.lineage.eq(lin),'distance'].to_numpy()
        color=LINEAGE_COLORS[lin]
        parts=ax.violinplot([values],positions=[i],orientation='horizontal',widths=.75,showextrema=False)
        for body in parts['bodies']:body.set_facecolor(color);body.set_edgecolor('none');body.set_alpha(.13)
        ax.scatter(values,i+rng.uniform(-.20,.20,len(values)),s=9,color=color,alpha=.35,edgecolors='none',zorder=2,gid=lin)
        ax.boxplot([values],positions=[i],orientation='horizontal',widths=.19,showfliers=False,patch_artist=True,
                   boxprops={'facecolor':'white','edgecolor':color,'linewidth':1.1},medianprops={'color':'#222222','linewidth':1.5},zorder=4)
        ax.text(1.015,i,f'n = {len(values)}',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
    ax.set_yticks(range(5),[DISPLAY_LABELS[l] for l in LINEAGE_ORDER]); ax.invert_yaxis()
    for tick,lin in zip(ax.get_yticklabels(),LINEAGE_ORDER):tick.set_color(LINEAGE_COLORS[lin])
    ax.set(xlabel='Within-lineage cross-tissue clock-architecture distance',xlim=(0,None))
    fig.suptitle('Within-lineage tissue variation in clock architecture',y=.97,fontsize=13)
    fig.text(.5,.91,'All tissue pairs with ≥3 donors in each lineage–tissue group',ha='center',fontsize=10,color='#666666')
    fig.text(.5,.05,'Distances use all 24 standardized clock features. Counts refer to dependent tissue pairs.',ha='center',fontsize=9,color='#555555')
    return fig

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=BOOTSTRAP_SEED)
    args=parser.parse_args(); OUTPUT.mkdir(parents=True,exist_ok=True)
    before=snapshot(); meta,features,verification,threshold=load_frozen_inputs()
    centroids,matrix,overall,inclusion,feature_centroids=centroid_tables(meta,features,threshold)
    pairs=pair_tables(centroids,matrix)
    print(inclusion.to_string(index=False),flush=True)
    print(pairs.groupby('comparison_type').distance.agg(['size','median']).to_string(),flush=True)
    with threadpool_limits(limits=1):
        boot,method=bootstrap(meta,features,centroids,pairs,args.iterations,args.seed)
    summary=summarize(pairs,boot,inclusion)
    tables={'clock_centroid_tissue_displacement':centroids,'clock_lineage_overall_centroids':overall,
            'clock_lineage_tissue_feature_centroids':feature_centroids,'clock_lineage_inclusion':inclusion,
            'clock_between_within_distances':pairs,'clock_distance_summary':summary,'clock_distance_bootstrap_draws':boot}
    for stem,data in tables.items():data.to_csv(OUTPUT/f'{stem}.csv',index=False,float_format='%.17g')
    apply_publication_style()
    figures=[]
    for stem,fig in [('fig_clock_centroid_tissue_displacement',figure_A(centroids,overall)),
                     ('fig_between_vs_within_clock_distance',figure_B(pairs,summary,args.iterations)),
                     ('fig_within_tissue_distance_by_lineage_supp',figure_supp(pairs))]:
        figures.append(save_figure(fig,stem,OUTPUT,dpi=300))
    assert before==snapshot(),'An original input file was modified.'
    audit={'inputs':verification,'bootstrap':method,'figures':figures,'original_files_unchanged':len(before),
           'original_sha256':before,'feature_source':str(FEATURE_PATH),'standardized_source':str(STANDARDIZED),
           'standardized_source_sha256':sha(STANDARDIZED),'pca_source_sha256':sha(PCA_SOURCE),
           'figure_A_all_lineage_tissue_groups':len(centroids),'figure_B_adequate_groups':int(centroids.adequate_support.sum()),
           'figure_B_n_distances':len(pairs),'n_features':features.shape[1],
           'donor_only_delta_95_interval':boot.donor_only_delta.quantile([.025,.975]).tolist(),
           'other_handling':'Other is heterogeneous and is not treated as a sixth biological lineage.',
           'pca_retention_sensitivity':'Not run: no pre-existing PC-retention criterion was identified.',
           'method_reference':'https://doi.org/10.1214/12-AOAS547',
           'calibration_limit':'Product-weight bootstrap applied to a nonlinear median-distance contrast is an approximate sensitivity analysis, not an exact calibrated test.'}
    (OUTPUT/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
    print(f'Completed; {len(before)} original files unchanged.',flush=True)

if __name__=='__main__':
    main()
