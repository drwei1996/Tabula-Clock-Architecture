"""Focused read-only audit of stored A1 random panels; no panel metrics rerun."""
from phaseA_unattended import *
from scipy.linalg import svd
from scipy.stats import hypergeom,spearmanr
DEST=OUT/'01_matched_random_controls/null_model_audit'
MARKERS={
 't_cell':['CD3D','CD3E','CD3G','TRAC','TRBC1','TRBC2','CD247','LCK','IL7R'],
 'b_cell':['MS4A1','CD79A','CD79B','CD19','CD22','BANK1','CD74'],
 'macrophage_monocyte':['LYZ','LST1','TYROBP','FCER1G','CSF1R','C1QA','C1QB','C1QC','FCGR3A','CTSS','S100A8','S100A9'],
 'endothelial':['PECAM1','VWF','CDH5','KDR','FLT1','EMCN','RAMP2','ESAM'],
 'fibroblast':['COL1A1','COL1A2','COL3A1','DCN','LUM','PDGFRA','COL6A1','COL6A2']}

def basis(meta,cols):
    d=np.column_stack([np.ones(len(meta))]+[pd.get_dummies(meta[c],dtype=float).to_numpy() for c in cols]);u,s,_=svd(d,full_matrices=False)
    rank=int((s>s[0]*max(d.shape)*np.finfo(float).eps).sum());return u[:,:rank],rank

def gene_effects(y,labels,q0,q1):
    centered=y-y.mean(0);sst=(centered**2).sum(0);between=np.zeros(y.shape[1])
    for label in np.unique(labels):
        v=y[labels==label];between+=len(v)*(v.mean(0)-y.mean(0))**2
    eta=np.divide(between,sst,out=np.full(y.shape[1],np.nan),where=sst>1e-20)
    # Work on centered data to avoid subtractive cancellation of a large expression mean.
    sse0=sst-((q0.T@centered)**2).sum(0);sse1=sst-((q1.T@centered)**2).sum(0)
    partial=np.divide(sse0-sse1,sse0,out=np.full(y.shape[1],np.nan),where=sse0>1e-20)
    return np.clip(eta,0,1),np.clip(partial,0,1)

def run():
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'figures').mkdir(exist_ok=True)
    before={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and DEST not in p.parents and p.name not in ['A1_NULL_MODEL_AUDIT.md','null_model_audit_run.log']}
    jwrite(DEST/'protected_file_hashes_before.json',before)
    meta,x,genes=load_data();stats=pd.read_csv(A1/'gene_matching_statistics.csv').set_index('gene');pr=protocol()
    core=[c[5:] for c in pr['feature_sets']['CORE_FEATURE_COLS']];expanded=[c[5:] for c in pr['feature_sets']['EXPANDED_FEATURE_COLS']]
    sufficient=(x.mean(0)>0)&(x.var(0)>1e-12);eligible=sufficient&~np.isin(genes,expanded)
    assert np.array_equal(eligible,stats.loc[genes,'eligible_nonclock'].to_numpy())
    cm=meta[meta.priority_lineage.notna()];idx=meta.index.get_indexer(cm.index);labels=cm.priority_lineage.to_numpy()
    q0,r0=basis(cm,['donor_id','tissue']);q1,r1=basis(cm,['donor_id','tissue','priority_lineage'])
    rows=[]
    for start in range(0,len(genes),512):
        end=min(start+512,len(genes));block=x[:,start:end];y=block[idx]
        eta,partial=gene_effects(y,labels,q0,q1)
        lineage_means=np.array([y[labels==l].mean(0) for l in LINEAGE_ORDER])
        for j,g in enumerate(genes[start:end]):
            rows.append({'gene':g,'sufficient_expression':bool(sufficient[start+j]),'eligible_background':bool(eligible[start+j]),'mean_expression':stats.loc[g,'mean'],'expression_variance':stats.loc[g,'variance'],
              'pseudobulk_nonzero_fraction_all2030':float(np.mean(block[:,j]>0)), 'pseudobulk_nonzero_fraction_lineages1302':float(np.mean(y[:,j]>0)),
              'lineage_eta_squared':eta[j] if sufficient[start+j] else np.nan,'lineage_partial_R2_given_donor_tissue':partial[j] if sufficient[start+j] else np.nan,
              'highest_mean_lineage':LINEAGE_ORDER[int(np.argmax(lineage_means[:,j]))],**{f'mean_{l}':lineage_means[k,j] for k,l in enumerate(LINEAGE_ORDER)}})
    effects=pd.DataFrame(rows).set_index('gene')
    old=pd.read_csv(ROOT.parent/'TabulaCircadian_deliverables/source_data/pseudobulk__clock_architecture_features.tsv',sep='\t').set_index('group_id').loc[meta.index]
    true_detect={c[7:]:float(old[c].mean()) for c in old.columns if c.startswith('detect_')}
    effects['mean_single_cell_detection_fraction_where_available']=[true_detect.get(g,np.nan) for g in effects.index]
    for col in ['lineage_eta_squared','lineage_partial_R2_given_donor_tissue']:
        background=np.sort(effects.loc[effects.eligible_background,col].dropna().to_numpy())
        effects[col+'_eligible_percentile']=100*np.searchsorted(background,effects[col],side='right')/len(background)
        effects.loc[effects[col].isna(),col+'_eligible_percentile']=np.nan
    cwrite(DEST/'all_gene_lineage_statistics.csv',effects.reset_index())
    actual=pd.concat([pd.read_csv(A1/f'{variant}_{slug}_genes.csv') for variant,_,_,_ in VARIANTS for slug in ['core','expanded']],ignore_index=True)
    cwrite(DEST/'all_actual_matched_gene_draws.csv',actual)
    # Size controls are separate; real-clock8 subsets are never labeled nonclock null panels.
    size=pd.read_csv(A1/'size_control_genes.csv');cwrite(DEST/'size_control_actual_genes.csv',size)
    frequencies=actual.groupby(['variant','panel','matched_gene']).agg(selection_count=('iteration','size'),n_panels_selected=('iteration','nunique'),n_target_genes=('clock_gene','nunique')).reset_index().rename(columns={'matched_gene':'gene'})
    frequencies['fraction_of_1000_panels']=frequencies.n_panels_selected/1000
    cwrite(DEST/'selection_frequencies_by_variant_panel.csv',frequencies.merge(effects.reset_index(),on='gene'))
    total=actual.groupby('matched_gene').size().rename('selection_count').reset_index().rename(columns={'matched_gene':'gene'})
    cwrite(DEST/'selection_frequencies_all_matched_panels.csv',total.merge(effects.reset_index(),on='gene'))
    substitute=actual.groupby(['variant','panel','clock_gene','matched_gene']).size().rename('selection_count').reset_index()
    substitute['selection_fraction_per_target']=substitute.selection_count/1000
    substitute=substitute.merge(effects.reset_index(),left_on='matched_gene',right_on='gene',validate='many_to_one').drop(columns='gene')
    substitute=substitute.sort_values(['variant','panel','clock_gene','selection_count','matched_gene'],ascending=[True,True,True,False,True])
    cwrite(DEST/'all_substitute_frequencies_and_statistics.csv',substitute)
    cwrite(DEST/'top20_substitutes_per_clock_gene_variant_panel.csv',substitute.groupby(['variant','panel','clock_gene'],sort=False).head(20))
    pools=pd.read_csv(A1/'candidate_pools.csv');poolstats=pools.merge(effects.reset_index(),left_on='matched_gene',right_on='gene',validate='many_to_one').drop(columns='gene')
    cwrite(DEST/'all_candidate_pools_with_lineage_statistics.csv',poolstats)
    # Independently regenerate candidate ranks and stored gene assignments only, not classifier/silhouette evaluations.
    univ=np.flatnonzero(eligible);lookup={g:i for i,g in enumerate(genes)};z=stats.loc[genes,['standardized_log_mean','standardized_log_variance']].to_numpy()
    for variant,seed,k,dim in VARIANTS:
        recovered={g:univ[np.argsort(((z[univ,:dim]-z[lookup[g],:dim])**2).sum(1),kind='stable')[:k]] for g in expanded}
        for g in expanded:
            saved=pools[pools.variant.eq(variant)&pools.clock_gene.eq(g)].sort_values('rank')
            assert saved.matched_gene.tolist()==genes[recovered[g]].tolist()
        rng=np.random.default_rng(seed)
        for panel,targets in [('Core clock',core),('Expanded clock',expanded)]:
            generated=[]
            for iteration in range(1000):
                selected=[]
                for g in targets:
                    pool=recovered[g];j=int(rng.choice(pool[~np.isin(pool,selected)]));selected.append(j);generated.append(genes[j])
            saved=actual[actual.variant.eq(variant)&actual.panel.eq(panel)]
            assert saved.matched_gene.tolist()==generated
    markerrows=[]
    for lin,markerlist in MARKERS.items():
        for g in markerlist:
            if g not in effects.index:continue
            freq=frequencies[frequencies.gene.eq(g)]
            if len(freq):
                for r in freq.itertuples():markerrows.append({'canonical_lineage_annotation':lin,'gene':g,'variant':r.variant,'panel':r.panel,'selection_count':r.selection_count,'fraction_of_panels':r.fraction_of_1000_panels,**effects.loc[g].to_dict()})
    markers=pd.DataFrame(markerrows);cwrite(DEST/'canonical_lineage_markers_actually_selected.csv',markers)
    cwrite(DEST/'canonical_marker_reference_list.csv',pd.DataFrame([{'lineage':l,'gene':g,'annotation_note':'Illustrative conventional lineage-associated marker list; not exhaustive or lineage-exclusive; expression peaks reported independently.'} for l,gs in MARKERS.items() for g in gs]))
    summaries=[];enrichment=[];candidate_comparison=[]
    for metric in ['lineage_eta_squared','lineage_partial_R2_given_donor_tissue']:
        bg=effects.loc[effects.eligible_background,metric].dropna();cut=bg.quantile(.9);high=set(bg[bg>=cut].index)
        sets=[('Core clock','clock','all',effects.loc[core,metric]),('Expanded clock','clock','all',effects.loc[expanded,metric]),('Eligible background','background','all',bg)]
        for variant,_,_,_ in VARIANTS:
            for panel in SETS:
                part=actual[actual.variant.eq(variant)&actual.panel.eq(panel)];unique=part.matched_gene.unique()
                sets.extend([('Selected unique',variant,panel,effects.loc[unique,metric]),('Selected frequency weighted',variant,panel,effects.loc[part.matched_gene,metric])])
                n=len(unique);k=len(set(unique)&high);enrichment.append({'metric':metric,'variant':variant,'panel':panel,'eligible_N':len(bg),'eligible_top_decile_n':len(high),'top_decile_cutoff':cut,'unique_selected_n':n,'top_decile_selected_n':k,'top_decile_fraction':k/n,'fold_enrichment_vs_uniform_background':(k/n)/(len(high)/len(bg)),'uniform_gene_sampling_hypergeom_P':hypergeom.sf(k-1,len(bg),len(high),n)})
        for name,variant,panel,values in sets:
            v=values.dropna().to_numpy();summaries.append({'metric':metric,'distribution':name,'variant':variant,'panel':panel,'n':len(v),'median':np.median(v),'q25':np.quantile(v,.25),'q75':np.quantile(v,.75),'mean':np.mean(v),'fraction_above_background_top_decile':np.mean(v>=cut)})
        for (variant,g),part in poolstats.groupby(['variant','clock_gene']):
            v=part[metric].dropna().to_numpy();value=effects.loc[g,metric]
            candidate_comparison.append({'metric':metric,'variant':variant,'clock_gene':g,'clock_effect':value,'candidate_median':np.median(v),'candidate_mean':np.mean(v),'candidate_q25':np.quantile(v,.25),'candidate_q75':np.quantile(v,.75),'clock_percentile_in_candidates':100*(np.sum(v<value)+.5*np.sum(v==value))/len(v),'median_candidate_minus_clock':np.median(v)-value,'pool_size':len(v)})
    summaries=pd.DataFrame(summaries);enrichment=pd.DataFrame(enrichment);comparison=pd.DataFrame(candidate_comparison)
    cwrite(DEST/'lineage_information_distribution_summary.csv',summaries);cwrite(DEST/'lineage_informative_gene_enrichment.csv',enrichment);cwrite(DEST/'per_clock_candidate_lineage_comparison.csv',comparison)
    # Descriptive technical association and stratified benchmark, not a redefined null.
    bg=effects[effects.eligible_background].copy();bg['mean_bin']=pd.qcut(np.log10(bg.mean_expression),10,labels=False,duplicates='drop');bg['variance_bin']=pd.qcut(np.log10(bg.expression_variance),10,labels=False,duplicates='drop')
    bin_means=bg.groupby(['mean_bin','variance_bin']).lineage_eta_squared.mean();bg['technical_bin_expected_eta']=[bin_means.loc[(r.mean_bin,r.variance_bin)] for r in bg.itertuples()]
    bin_count=bg.groupby(['mean_bin','variance_bin']).size().rename('gene_count').reset_index();cwrite(DEST/'posthoc_technical_benchmark_bin_sizes.csv',bin_count)
    context=[]
    for col in ['mean_expression','expression_variance','pseudobulk_nonzero_fraction_all2030']:
        paired=bg[[col,'lineage_eta_squared']].dropna();rho,p=spearmanr(paired[col],paired.lineage_eta_squared);context.append({'predictor':col,'Spearman_rho_with_eta':rho,'n':len(paired),'note':'Descriptive across correlated genes; no independent-gene biological P claim.'})
    cwrite(DEST/'technical_properties_lineage_correlation.csv',pd.DataFrame(context))
    technical=[]
    for variant,_,_,_ in VARIANTS:
        for panel in SETS:
            selected=actual[actual.variant.eq(variant)&actual.panel.eq(panel)].matched_gene.unique();part=bg.loc[selected]
            technical.append({'variant':variant,'panel':panel,'unique_selected_mean_eta':part.lineage_eta_squared.mean(),'logmean_logvariance_decile_bin_expected_mean_eta':part.technical_bin_expected_eta.mean(),'difference':(part.lineage_eta_squared-part.technical_bin_expected_eta).mean(),'n_unique_genes':len(part)})
    technical=pd.DataFrame(technical);cwrite(DEST/'posthoc_technical_stratified_benchmark.csv',technical)
    plots(effects,actual,frequencies,comparison,core,expanded)
    render_report(effects,actual,pools,markers,summaries,enrichment,comparison,technical,context,r0,r1,core,expanded)
    assert all(Path(p).exists() and sha(Path(p))==digest for p,digest in before.items())
    # Independent numerical sanity check on a small gene subset using explicit residual OLS.
    sample=np.array([lookup[g] for g in expanded[:3]]);y=x[idx][:,sample];eta,partial=gene_effects(y,labels,q0,q1)
    ss0=((y-q0@(q0.T@y))**2).sum(0);ss1=((y-q1@(q1.T@y))**2).sum(0)
    np.testing.assert_allclose(partial,1-ss1/ss0,atol=1e-10)
    for j in range(3):
        total=((y[:,j]-y[:,j].mean())**2).sum();ss=sum((labels==l).sum()*(y[labels==l,j].mean()-y[:,j].mean())**2 for l in LINEAGE_ORDER)
        np.testing.assert_allclose(eta[j],ss/total,atol=1e-12)
    jwrite(DEST/'audit_validation.json',{'stored_candidate_pools_exactly_reproduced':True,'all100000_matched_gene_draws_exactly_reproduced':len(actual)==100000,'n_observations_for_lineage_effects':len(cm),'n_genes':len(effects),'eligible_genes':int(eligible.sum()),'model_ranks_donor_tissue_and_plus_lineage':[r0,r1],'eta_and_partialR2_independent_checks':'PASS','all_preexisting_phaseA_files_unchanged':len(before),'gate_unchanged':True,'A2_unchanged':True,'panel_evaluations_rerun':0,'visual_review':'pending'})
    print(summaries[(summaries.metric.eq('lineage_eta_squared'))&(summaries.variant.isin(['primary','background','clock']))].to_string(index=False))
    print(enrichment[enrichment.variant.eq('primary')].to_string(index=False));print('AUDIT COMPLETE',flush=True)

def plots(effects,actual,freq,comparison,core,expanded):
    apply_publication_style();destination=DEST/'figures';fig_sources=DEST/'figure_source_data';fig_sources.mkdir(exist_ok=True)
    def save(fig,name,source):cwrite(fig_sources/(name+'.csv'),source);save_figure(fig,name,destination,dpi=300)
    colors={'Core clock':'#4E79A7','Expanded clock':'#E15759','Eligible background':'#777777','Matched unique':'#9467BD','Matched frequency weighted':'#59A14F'}
    fig,axes=plt.subplots(1,2,figsize=(12,5.7));fig.subplots_adjust(left=.08,right=.97,bottom=.25,top=.80,wspace=.25)
    plotrows=[]
    for ax,panel,clockgenes in zip(axes,SETS,[core,expanded]):
        selected=actual[actual.variant.eq('primary')&actual.panel.eq(panel)].matched_gene
        for name,ids,color in [('Eligible background',effects.index[effects.eligible_background],colors['Eligible background']),(panel,clockgenes,colors[panel]),('Matched unique',selected.unique(),colors['Matched unique']),('Matched frequency weighted',selected,colors['Matched frequency weighted'])]:
            v=np.sort(effects.loc[ids,'lineage_eta_squared'].dropna().to_numpy());ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,label=name,lw=1.5)
            plotrows.extend({'panel':panel,'distribution':name,'eta':a,'ECDF':b} for a,b in zip(v,np.arange(1,len(v)+1)/len(v)))
        ax.set(xlabel='Major-lineage η²',ylabel='Cumulative fraction',title=panel,xlim=(0,1));clean_axis(ax)
    handles={}
    for ax in axes:
        hh,ll=ax.get_legend_handles_labels();handles.update(dict(zip(ll,hh)))
    fig.legend(handles.values(),handles.keys(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,fontsize=8,frameon=False)
    fig.suptitle('Lineage information in clock genes and matched backgrounds',y=.97,fontsize=13)
    fig.text(.5,.135,'Same 1,302 five-lineage pseudobulks · primary matching only · frequency weighting is descriptive',ha='center',fontsize=9)
    save(fig,'Audit1_lineage_information_distributions',pd.DataFrame(plotrows))
    counts=actual[actual.variant.eq('primary')].groupby('matched_gene').size().sort_values(ascending=False);frequent=counts.head(12).index
    fig,ax=plt.subplots(figsize=(10,6.5));fig.subplots_adjust(left=.10,right=.97,bottom=.21,top=.84);bg=effects[effects.eligible_background]
    ax.scatter(np.log10(bg.expression_variance),bg.lineage_eta_squared,s=3,c='#B8BEC5',alpha=.3,rasterized=True,label='Eligible background')
    for ids,c,label,marker in [(frequent,'#9467BD','12 most frequent matched genes','o'),(expanded,'#E15759','Expanded clock','D'),(core,'#4E79A7','Core clock','D')]:
        part=effects.loc[ids];ax.scatter(np.log10(part.expression_variance),part.lineage_eta_squared,s=38,c=c,marker=marker,edgecolor='white',linewidth=.5,label=label)
    for g in list(counts.head(4).index)+['ARNTL','NFIL3']:
        r=effects.loc[g];ax.annotate(g,(np.log10(r.expression_variance),r.lineage_eta_squared),xytext=(5,6),textcoords='offset points',fontsize=8)
    clean_axis(ax);ax.set(xlabel='log10 expression variance across all 2,030 observations',ylabel='Major-lineage η² in the 1,302 lineage observations');fig.suptitle('Expression variance and lineage information',y=.96)
    fig.legend(loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,fontsize=9,frameon=False)
    save(fig,'Audit2_variance_vs_lineage_information',effects.reset_index())
    top=counts.head(20);fig,ax=plt.subplots(figsize=(10,7));fig.subplots_adjust(left=.15,right=.97,bottom=.16,top=.84)
    ax.barh(range(len(top)),top.to_numpy(),color='#9467BD',alpha=.8);ax.set_yticks(range(len(top)),top.index);ax.invert_yaxis();clean_axis(ax)
    ax.set_xlabel('Selections across the 2,000 primary Core + Expanded panels');fig.suptitle('Most frequently selected matched nonclock genes',y=.96)
    fig.text(.5,.055,'25,000 primary gene slots · at most once per panel · reuse across panels is allowed',ha='center',fontsize=9)
    save(fig,'Audit3_most_frequent_substitutes',top.rename('selection_count').reset_index())
    pool=pd.read_csv(DEST/'all_candidate_pools_with_lineage_statistics.csv');fig,axes=plt.subplots(1,2,figsize=(12,8));fig.subplots_adjust(left=.10,right=.97,bottom=.16,top=.84,wspace=.26)
    for ax,metric,title in zip(axes,['lineage_eta_squared','lineage_partial_R2_given_donor_tissue'],['Marginal lineage η²','Partial R² given donor + tissue']):
        for i,g in enumerate(expanded):
            values=pool[pool.variant.eq('primary')&pool.clock_gene.eq(g)][metric]
            ax.boxplot(values,positions=[i],orientation='horizontal',widths=.55,showfliers=False,patch_artist=True,boxprops={'facecolor':'#E2D9E9','edgecolor':'#9467BD'},medianprops={'color':'#666666'},whiskerprops={'color':'#bbbbbb'},capprops={'color':'#bbbbbb'})
            ax.scatter(effects.loc[g,metric],i,color='#4E79A7' if g in core else '#E15759',marker='D',s=30,zorder=4)
        ax.set_yticks(range(len(expanded)),expanded);ax.invert_yaxis();ax.set(xlabel=title,xlim=(0,1));clean_axis(ax)
    fig.suptitle('Each clock gene versus its 100 matched candidates',y=.97,fontsize=13)
    fig.text(.5,.055,'Boxes: candidate median / interquartile range / 1.5×IQR whiskers · diamonds: clock genes\nCore genes blue; auxiliary genes red. Candidate pools are shared across Core/Expanded where applicable.',ha='center',fontsize=9)
    save(fig,'Audit4_per_clock_candidate_lineage_effects',comparison[comparison.variant.eq('primary')])

def render_report(effects,actual,pools,markers,summaries,enrichment,comparison,technical,context,r0,r1,core,expanded):
    primary_comparison=comparison[comparison.variant.eq('primary')&comparison.metric.eq('lineage_eta_squared')]
    decision='D) INCONCLUSIVE'
    primary_counts=actual[actual.variant.eq('primary')].groupby('matched_gene').size().sort_values(ascending=False)
    report=f'''# Focused A1 matched-null-model audit

## Scope and algorithm recovered
Read-only audit of saved panels. No classifier, silhouette, distance-ratio panel evaluation was rerun; no A2 recomputation or gate change. Original code: scripts/phaseA_resume.py:a1 and scripts/phaseA_unattended.py:matched_controls. Broad normalized-mean matrix2030×59410, original five-lineage subset1302 observations. Original per-gene mean/variance are computed over all2030 observations, not only the five lineages.

Eligible nonclock universe: {int(effects.eligible_background.sum())} genes, from {int(effects.sufficient_expression.sum())} genes with mean>0 and population variance>1e-12. Exclude ALL17 expanded-clock genes for both Core8 and Expanded17; upstream broad export had already excluded duplicate gene symbols. No marker, lineage-effect, pathway, high-variance upper-bound or detection-based exclusion. Eligibility is identical to the frozen resume code. All values finite. CTGF is absent/zero in original selected data.

Primary: log10(mean),log10(population variance); each variable centered/scaled over eligible nonclock genes. For each clock gene independently, Euclidean distance in this2D matching space; take nearest100 genes using stable argsort. No bins, explicit caliper or tolerance. Candidate pools adapt to local gene density. No panel-level mean/variance balancing. Targets traversed in frozen panel order; choose uniformly from each target's candidate pool after removing genes already selected for that panel. Shared pool competition can alter marginal frequencies. Cross-panel and cross-target reuse allowed; within-panel duplicates prohibited. Pools for the same Core gene are the same in Core and Expanded, but target order/competition and RNG differ. Same RNG seeded20260913 proceeds through all1000 Core panels then1000 Expanded panels. Independent seed20260914; tighter sensitivity nearest25 in2D; expression-only nearest100 using standardized logmean alone. Each sensitivity preserves order and no-replacement-within-panel rule. All candidate rankings and100,000 target/substitute assignments across8000 matched panels were independently regenerated exactly, without evaluating panels.

Detection was NOT a matching variable. Broad dependency lacks cell-level detection for all genes. We report pseudobulk nonzero prevalence (fraction of normalized pseudobulk means>0), explicitly distinct from single-cell detection. Mean single-cell detection is additionally supplied only where an original detect_GENE column exists; otherwise NA. No detection values fabricated. Two separate1000-panel size checks exported: random8 subsets from matched expanded panels, and real8-of17 subsets; neither is pooled into the primary matched-null frequencies.

## Per-gene lineage statistics
For every sufficient-expression broad gene, on the SAME1302 five-lineage observations: η²=Σ_l n_l(mean_l−grandmean)² / Σ_i(y_i−grandmean)². Equal pseudobulk weight, unbalanced lineage sizes retained. Additional classifier-independent partial R²=(SSE_donor+tissue−SSE_donor+tissue+lineage)/SSE_donor+tissue, using rank-aware orthogonal projections (ranks{r0}/{r1}). This is an in-sample association/effect size, not cross-validated classification and not a causal decomposition. No individual-observation independence P values. Genes are correlated; frequency weighting does not create extra biological replicates. Complete five-lineage-zero-variance genes have undefined lineage effects recorded as NA.

{markdown_table(summaries[summaries.variant.isin(['primary','clock','background'])])}

## Enrichment relative to the unrestricted eligible universe
{markdown_table(enrichment[enrichment.variant.eq('primary')])}
Top-decile thresholds fixed from all eligible genes. Hypergeometric P is a finite-population UNIFORM-gene-sampling reference only, not a calibrated test that technical matching is invalid: matched genes are intentionally not uniform. Unique selected genes are the sample units; no repeated selection slots treated as independent genes. Frequency-weighted distributions are reported separately.

## Expression/variance coupling and conditional benchmark
{markdown_table(pd.DataFrame(context))}
{markdown_table(technical)}
The post-hoc descriptive benchmark uses10 marginal quantile bins each for logmean/logvariance, compares selected-gene mean η² with the mean η² of all eligible genes in the same bins, and weights bins by selected genes. This is an audit only, not a change to matching or a new matched-null test. Technical-bin residuals are not causal estimates. Total expression variance includes between-lineage biology; conditioning on variance can legitimately remove that component when asking for information beyond equal variance, but can also control away a biological mechanism when asking for total clock-associated lineage information. These are different estimands.

## Per-clock comparisons
{markdown_table(comparison[comparison.variant.eq('primary')])}
Candidate pools are evaluated unweighted; actual substitute top20 lists preserve separate Core/Expanded and variant frequencies. Files: all_candidate_pools_with_lineage_statistics.csv; top20_substitutes_per_clock_gene_variant_panel.csv; all_substitute_frequencies_and_statistics.csv. Each includes mean, variance, pseudobulk prevalence, true cell detection where available, η², conditional partial R² and ranks/frequencies as applicable. Clock genes themselves are in all_gene_lineage_statistics.csv.

## Canonical markers
The fixed illustrative list spans conventional T/B/myeloid/endothelial/fibroblast-associated markers. It is not exhaustive or lineage-exclusive (e.g.CD74,LYZ and stromal matrix genes can cross categories). Dataset highest-mean lineage is supplied independently; annotations are not used to exclude any gene. Full reference list and every matched-marker occurrence/frequency are exported. Selected-marker rows:
{markdown_table(markers[['canonical_lineage_annotation','gene','variant','panel','selection_count','fraction_of_panels','lineage_eta_squared','highest_mean_lineage']] if len(markers) else pd.DataFrame({'result':['No listed canonical marker selected']}))}

## Outputs and integrity
Audit directory:01_matched_random_controls/null_model_audit. Actual primary/seed/tighter/expression-only assignments and frequency tables are exported separately from size controls. Four figures have PNG300dpi,PDF,SVG and source CSV. All pre-existing Phase A files, including gate and A2, are hash-checked unchanged.

## Decision and quantitative interpretation

**{decision}**

The implementation is faithful to the declared conditional null: no lineage statistic entered candidate selection, and all recorded candidate ranks and actual draws reproduce exactly. However, technical matching DOES indirectly select lineage-informative genes. Primary unique-selected median η² is0.2496/Core and0.2405/Expanded, versus0.0153 for the evaluable background; corresponding top-decile enrichments are4.59× and4.41×. Conditional partial-R² top-decile enrichments remain4.89×/4.55×, so the pattern is not explained solely by unadjusted donor/tissue main effects.

This enrichment does not on its own establish inappropriate overmatching. Mean expression and variance correlate strongly with η² (Spearman0.795/0.764). Expression-only matching already yields mean η²0.2706/Core and0.2442/Expanded; adding variance increases these to0.2808/0.2752. A coarse logmean/logvariance-stratified descriptive expectation is0.2898/0.2561, leaving selected-minus-expected differences−0.0090/+0.0191. These post-hoc differences are not formal conditional tests, but show why comparison with uniform background is insufficient to diagnose a flawed null.

Clock mean η² is0.2826/Core and0.2809/Expanded, close to selected-null means0.2808/0.2752; clock medians0.2887/0.2716 are modestly above selected medians. Across17 clock genes, {int(primary_comparison.clock_effect.gt(primary_comparison.candidate_median).sum())} are above their own candidate median; candidate pools are not uniformly more lineage-informative than the genes they replace. ARNTL, CLOCK and TEF are at the88th,92nd and92nd candidate percentiles, whereas PER2 and DBP are at the18th. This argues against a simple uniformly stronger-lineage-control artifact, but does not identify the correct biological estimand.

Canonical-marker domination is not supported by the illustrative marker list: primary Core contains none of the listed markers; primary Expanded includes CTSS in6/1000 panels (6/25000 total primary gene slots). The expression-only sensitivity includes additional myeloid/endothelial markers (full table above). The most frequent primary gene, {primary_counts.index[0]}, occupies {int(primary_counts.iloc[0])}/25000 gene slots ({primary_counts.iloc[0]/25000:.3%}); no single repeatedly selected gene dominates the primary pool. The marker list is explicitly incomplete; absence from it does not mean a gene lacks lineage information.

Thus the audit confirms substantial implicit lineage conditioning, not an explicit lineage-matching bug. Total variance contains biological between-lineage variation as well as technical/statistical variation. These observations alone cannot establish whether conditioning on it appropriately defines the intended equal-abundance/equal-variance comparison or removes part of the biological phenomenon of interest. The broad gene-level eligibility rule is weak, and many very-low-expression background genes have low η²; uniform-background enrichment therefore exaggerates what would be surprising under abundance matching.

The defensible A1 wording is: **the predefined clock panels did not show additional lineage information relative to genes matched on abundance and total variance under the saved procedure.** This does not show absence of clock-associated lineage architecture, prove statistical equivalence, or establish that matching is biologically neutral. No marker exclusion, rematching, panel rerun, gate change or Phase B was performed. This audit alone does not justify replacing the primary analysis with a more favorable null. Distinguishing appropriate conditioning from biological overmatching requires an explicitly defined estimand and any future outcome-independent sensitivity specification, not removing markers after observing these results.

Eligible background count is58035;141 genes have no evaluable marginal lineage variation in the1302-observation subset (57894 evaluable), and142 lack evaluable conditional residual variation (57893 evaluable). These are excluded only from undefined effect-distribution calculations, not retroactively from the saved matching universe. Pseudobulk prevalence is not a substitute for missing cell-level detection.
'''
    (REP/'A1_NULL_MODEL_AUDIT.md').write_text(report,encoding='utf-8')
    jwrite(DEST/'audit_decision.json',{'decision':decision,'reason':'Implicit lineage conditioning is quantified, but appropriate conditional control versus biological overmatching is not identifiable from these diagnostics alone.','no_explicit_lineage_matching':True,'gate_changed':False,'A2_changed':False,'no_panel_reevaluation':True})

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
