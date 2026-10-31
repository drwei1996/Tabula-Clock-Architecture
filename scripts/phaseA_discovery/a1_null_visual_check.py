"""Outcome-blind panel sampling; BA-ranked visual audit, reusing frozen A1 metrics."""
from phaseA_unattended import *
import clock_architecture_umap as original
from matplotlib.lines import Line2D

DEST=OUT/'01_matched_random_controls/null_model_audit/visual_sanity_check'
SEED_NAIVE=20260917

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    report=REP/'A1_NULL_MODEL_AUDIT.md'
    protected=DEST/'protected_hashes.json'
    if not protected.exists():
        jwrite(protected,{str(p):sha(p) for base in [OUT,ROOT/'figures_revised'] for p in base.rglob('*') if p.is_file() and DEST not in p.parents and p!=report})
    meta,x,genes=load_data(); index={g:i for i,g in enumerate(genes)}
    clock=[g[5:] for g in protocol()['feature_sets']['EXPANDED_FEATURE_COLS']]
    stats=pd.read_csv(A1/'gene_matching_statistics.csv'); universe=stats.loc[stats.eligible_nonclock,'gene'].to_numpy()
    spec={'seed':SEED_NAIVE,'n_naive':1000,'n_matched_reused':1000,'genes_per_panel':len(clock),'universe_n':len(universe),'sampling':'Uniform eligible nonclock genes without replacement within panels; reuse across panels allowed. Same frozen A1 eligibility. No matching.','primary_metric':'balanced_accuracy','representative_quantiles':[.10,.50,.90,.99],'quantile_method':'numpy linear quantile; nearest actual balanced accuracy; lowest iteration breaks ties','UMAP':'Exact original preprocessing and parameters; all 2030 observations, cached original Expanded coordinates; eight random-panel embeddings only. Formal A1 metrics use frozen 1302 five-lineage observations and donor-disjoint folds.','detection':'Not matched in original A1; no detection-matched null is invented.'}
    if (DEST/'prespecified_visual_protocol.json').exists():assert json.loads((DEST/'prespecified_visual_protocol.json').read_text())==spec
    else:jwrite(DEST/'prespecified_visual_protocol.json',spec)
    rng=np.random.default_rng(SEED_NAIVE); selections=[rng.choice(universe,len(clock),replace=False).tolist() for _ in range(1000)]
    cwrite(DEST/'all_naive_panel_genes.csv',pd.DataFrame([{'iteration':i+1,'position':j+1,'gene':g} for i,p in enumerate(selections) for j,g in enumerate(p)]))
    ev=Evaluator(meta,x)
    observed=pd.read_csv(OUT/'01_matched_random_controls/observed_metrics.csv').set_index('panel').loc['Expanded clock']
    check=ev.evaluate([index[g] for g in clock])
    for m in METRICS:np.testing.assert_allclose(check[m],observed[m],rtol=1e-9,atol=1e-10)
    naive=checkpoint_metrics(DEST/'naive_metrics.csv',ev,[[index[g] for g in p] for p in selections],{'panel':'Expanded-size naive random','variant':'naive','seed':SEED_NAIVE})
    matched=pd.read_csv(A1/'primary_expanded_checkpoint.csv'); assert len(matched)==1000
    draws=pd.read_csv(A1/'primary_expanded_genes.csv')
    matched_lists={int(i):d.matched_gene.tolist() for i,d in draws.groupby('iteration',sort=False)}
    cwrite(DEST/'matched_metrics_reused.csv',matched)
    summary=[]; representatives=[]; display=[('Expanded clock',clock,observed.to_dict(),0,'clock',0)]
    for label,df,lists in [('Naive random',naive,{i+1:p for i,p in enumerate(selections)}),('Matched random',matched,matched_lists)]:
        for m in METRICS:
            v=df[m].to_numpy(); obs=float(observed[m]);summary.append({'null':label,'metric':m,'clock':obs,'n':len(v),'mean':v.mean(),'median':np.median(v),'q025':np.quantile(v,.025),'q975':np.quantile(v,.975),'clock_percentile':100*((v<obs).sum()+.5*(v==obs).sum())/len(v),'upper_tail_P':(1+(v>=obs).sum())/(len(v)+1)})
        for q in spec['representative_quantiles']:
            target=np.quantile(df.balanced_accuracy,q); chosen=df.assign(delta=(df.balanced_accuracy-target).abs()).sort_values(['delta','iteration'],kind='stable').iloc[0]
            i=int(chosen.iteration); p=lists[i]; assert len(p)==17 and len(set(p))==17
            row={'class':label,'quantile':q,'target_balanced_accuracy':target,'iteration':i,**{m:float(chosen[m]) for m in METRICS}}
            representatives.append(row); display.append((f'{label} · P{int(q*100)}',p,row,i,label,q))
    summary=pd.DataFrame(summary); summary['BH_8_comparisons']=bh(summary.upper_tail_P.to_numpy())
    cwrite(DEST/'formal_null_comparison.csv',summary);cwrite(DEST/'representative_panel_selection.csv',pd.DataFrame(representatives))
    cwrite(DEST/'displayed_panel_genes.csv',pd.DataFrame([{'display':title,'class':label,'quantile':q,'iteration':i,'position':j+1,'gene':g} for title,p,row,i,label,q in display for j,g in enumerate(p)]))
    env=original.original_spec(); refdir=original.OUT/'source_data'
    reference=pd.read_csv(refdir/'clock_architecture_UMAP_expanded_clock_coordinates.tsv',sep='\t',float_precision='round_trip')
    assert reference.group_id.tolist()==meta.index.tolist()
    saved=json.loads((refdir/'clock_architecture_UMAP_expanded_clock_coordinates.json').read_text())
    assert saved['signature']['umap_version']==importlib.metadata.version('umap-learn')
    params=saved['signature']['parameters']; import umap
    valid=meta.reset_index(); embeddings=[]
    for k,(title,p,row,i,label,q) in enumerate(display):
        cols=['expr_'+g for g in p]; raw=pd.DataFrame(x[:,[index[g] for g in p]],columns=cols)
        mat,present=env['get_feature_matrix'](raw,cols); assert present==cols and mat.shape==(2030,17)
        if k==0:
            saved_input=pd.read_csv(refdir/'clock_architecture_UMAP_expanded_clock_standardized_input.tsv',sep='\t').set_index('group_id')
            np.testing.assert_allclose(mat.to_numpy(),saved_input.loc[meta.index,cols],rtol=1e-10,atol=1e-10)
            data=reference.copy()
        else:
            path=DEST/f'embedding_{k}.csv'
            if path.exists():data=pd.read_csv(path,float_precision='round_trip')
            else:
                print('UMAP',title,'iteration',i,flush=True)
                reducer=umap.UMAP(**params)
                coords=reducer.fit_transform(mat.to_numpy())
                assert json.loads(json.dumps(reducer.get_params()))==saved['actual_parameters']
                data=valid[['group_id','priority_lineage']].copy(); data[['UMAP1','UMAP2']]=coords
                cwrite(path,data)
        assert data.group_id.tolist()==meta.index.tolist();data['display']=title;embeddings.append(data)
    combined=pd.concat(embeddings,ignore_index=True);cwrite(DEST/'displayed_embedding_coordinates.csv',combined)
    jwrite(DEST/'umap_parameters.json',saved['actual_parameters'])
    apply_publication_style();fig=plt.figure(figsize=(16,11));gs=fig.add_gridspec(3,4,left=.06,right=.985,bottom=.10,top=.86,hspace=.48,wspace=.30)
    positions=[(0,0)]+[(1,j) for j in range(4)]+[(2,j) for j in range(4)]
    spans=[np.ptp(d[['UMAP1','UMAP2']].to_numpy(),axis=0).max() for d in embeddings];span=max(spans)*1.08
    for (title,p,row,i,label,q),data,pos in zip(display,embeddings,positions):
        ax=fig.add_subplot(gs[pos]); labels=data.priority_lineage.fillna('other')
        xy=data[['UMAP1','UMAP2']].to_numpy();center=(xy.max(0)+xy.min(0))/2;xy=xy-center
        for lineage in ['other']+LINEAGE_ORDER:
            use=labels.eq(lineage);ax.scatter(xy[use,0],xy[use,1],s=3.5,c=LINEAGE_COLORS[lineage],alpha=.4 if lineage=='other' else .83,edgecolors='none',rasterized=True)
        ax.set(xlim=(-span/2,span/2),ylim=(-span/2,span/2),xlabel='UMAP 1',ylabel='UMAP 2');ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([]);clean_axis(ax)
        ax.set_title(f'{title}\nBA = {row["balanced_accuracy"]:.3f}'+(f' · panel {i}' if i else ''),fontsize=10,pad=9)
    note=fig.add_subplot(gs[0,1:]);note.axis('off')
    note.text(.02,.94,'17 genes per panel · identical UMAP pipeline',fontsize=13,weight='bold',va='top')
    note.text(.02,.75,'Random panels selected by held-out balanced accuracy (BA), never by appearance.\nP10 / P50 / P90 / P99: nearest actual panel to each BA quantile.\nAll 2,030 observations embedded; formal metrics use 1,302 major-lineage observations.\nMatched null controls expression and variance; detection was not matched.\nShared scale and aspect; independent embeddings centered without rotation.\nUMAP separation is descriptive, not a statistical test.',fontsize=10,va='top',linespacing=1.8)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=LINEAGE_COLORS[l],label=DISPLAY_LABELS[l],markersize=5) for l in LINEAGE_ORDER+['other']],loc='lower center',bbox_to_anchor=(.5,.018),ncol=6,frameon=False)
    fig.suptitle('Expanded clock versus random-gene UMAP landscapes',y=.975,fontsize=17)
    fig.text(.5,.932,'Donor × tissue × cell-type pseudobulks · same observations, scaling, Euclidean metric and UMAP settings',ha='center',fontsize=11)
    save_figure(fig,'A1_random_panel_UMAP_comparison',DEST/'figures',dpi=300);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,3.8));fig.subplots_adjust(left=.06,right=.99,top=.76,bottom=.24,wspace=.35)
    for ax,m in zip(axes,METRICS):
        for label,df,color in [('Naive',naive,'#777777'),('Matched',matched,'#4E79A7')]:
            v=np.sort(df[m]);ax.plot(v,np.arange(1,len(v)+1)/len(v),color=color,label=label)
        ax.axvline(observed[m],color='#E15759',ls='--',label='Expanded clock');ax.set(xlabel=m.replace('_',' '),ylabel='Cumulative fraction');clean_axis(ax)
    fig.suptitle('High-dimensional A1 metrics: 1,000 panels per null',y=.97);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3)
    save_figure(fig,'A1_random_panel_metric_distributions',DEST/'figures',dpi=300);plt.close(fig)
    marker='\n## Visual sanity-check experiment'
    base=report.read_text(encoding='utf-8').split(marker)[0]
    addition=f'''{marker}

### Design and reproducibility
Expanded clock, 1,000 uniformly sampled naive panels, and the 1,000 existing primary matched panels all contain exactly 17 distinct genes. Naive seed: {SEED_NAIVE}; the same 58,035 eligible nonclock genes are used, with no expression/variance/detection matching. Sampling is uniform without replacement within panels; reuse across panels is allowed. Original A1 matched panels control log expression mean and log total variance, **not detection**. The original 1,000 matched metric evaluations are reused unchanged; only 1,000 new naive panels are evaluated. One Expanded metric recalculation verifies the pipeline numerically.

The original extracted get_feature_matrix function is reused: numeric conversion, median imputation if needed, removal of zero-variance features, then StandardScaler over all 2,030 observations. All selected panels retain 17 finite features. UMAP uses the original installed version and complete parameter set (20 neighbors, min_dist 0.25, Euclidean metric, seed 20260913, n_jobs 1, spectral initialization, other original defaults). Original Expanded coordinates are reused; the standardized input and observation order are checked against the saved original. Every embedding includes all 2,030 pseudobulks, including Other. These are not single-cell transcriptomic UMAPs. The colors come from the same CELLTYPE_COLORS-derived palette.

The exact existing Evaluator supplies donor-disjoint five-fold Ridge balanced accuracy and macro-F1 (train-only scaling), five-lineage silhouette (global original scaling), and median between/within lineage-tissue centroid Euclidean distance ratio with original support rules. No metric is calculated in UMAP space. All formal lineage metrics retain the original 1,302 observations and folds.

### Objective display selection
Balanced accuracy was fixed as the primary ranking metric before evaluating naive panels. Linear sample quantiles at 10%, 50%, 90%, and 99% select the nearest actual evaluated panel; lowest iteration resolves ties. Only these eight random UMAPs are computed. No visual search, seed sweep or appearance-based replacement is used. Independent embeddings have arbitrary origins/orientations; figures use translation-only centering and identical square axis spans/aspect, without rotation or rescaling. Apparent 2D cluster distances are not comparable inferential quantities.

{markdown_table(pd.DataFrame(representatives))}

### Quantitative comparison
Empirical upper-tail P = (1 + number of null metrics >= observed) / 1001; the smallest resolvable P is 0.000999. Clock percentile uses midranks. BH across these eight comparisons is supplied as a descriptive multiplicity adjustment; this added audit does not change the frozen gate.

{markdown_table(summary)}

### Answers to the biological sanity-check questions
1. **Can arbitrary random genes recover lineage structure?** Yes, in the restricted sense of uniformly sampled eligible expressed-gene panels showing predictive lineage information: naive median held-out BA is 0.628 (five-class chance reference 0.20), and the objectively selected P99 panel has BA 0.823. This is not a label-permutation significance test. Typical naive geometric separation is weak: median high-dimensional silhouette is -0.019. Arbitrary genes do not all yield equivalent structure, and above-chance prediction need not imply clean UMAP clusters.
2. **Can matched random genes recover lineage structure?** Yes. Matched median BA is 0.885, nearly the clock value 0.888; the P10 matched panel already has BA 0.845, and its median silhouette is 0.109 versus clock 0.093. Thus strong lineage information is readily recovered without clock genes when matching expression and variance. Detection matching was not performed.
3. **Is the clock UMAP unusual?** No statistical claim is made about UMAP appearance. In the original high-dimensional metrics, clock exceeds all 1,000 naive panels in BA (empirical P=0.000999), with macro-F1 P=0.001998 and silhouette P=0.01698; the distance ratio is not unusual at P=0.09491. These first three survive the descriptive eight-comparison BH adjustment. Against matched panels, clock is ordinary on all four metrics (P=0.455–0.676; BA percentile 53.8). This supports unusual predictive information relative to uniform gene sampling, but no demonstrated added information relative to the expression/variance-matched null. It does not establish statistical equivalence.
4. **Does variance/detection matching enrich lineage-informative genes?** Combined expression/variance matching substantially enriches them relative to uniform background (Expanded top-decile eta-squared enrichment 4.41-fold; selected mean eta-squared 0.275 versus background 0.085). Attribution specifically to variance is more limited: the existing expression-only sensitivity already gives selected mean eta-squared 0.244, with variance matching adding about 0.031 (13% relative increase). Detection contributes no designed matching effect because it was not used; true cell-level detection is unavailable for most broad genes. These diagnostics do not isolate a causal variance effect.

The visual sanity check strengthens the distinction between the two null hypotheses, not the case for choosing whichever null gives a favorable result. It leaves the prior audit decision **D) INCONCLUSIVE** about appropriate conditioning versus biological overmatching unchanged. The current Phase A gate, frozen A2 outputs and Phase B status remain unchanged.

### Files
`01_matched_random_controls/null_model_audit/visual_sanity_check/` contains all naive gene draws, reused matched metrics, formal comparisons, the selection manifest, exact ordered displayed gene lists, embedding coordinates and parameters. Figures: `figures/A1_random_panel_UMAP_comparison` and `figures/A1_random_panel_metric_distributions` (PNG/PDF/SVG). The preceding audit's per-gene enrichment analysis remains unchanged.
'''
    report.write_text(base+addition,encoding='utf-8')
    before=json.loads(protected.read_text());changed=[p for p,h in before.items() if sha(Path(p))!=h];assert not changed,changed
    jwrite(DEST/'validation.json',{'protected_files_unchanged':len(before),'matched_evaluations_reused':1000,'naive_evaluations':1000,'random_embeddings':8,'clock_embedding_reused':True,'original_clock_metrics_reproduced':True,'original_clock_standardized_input_verified':True,'A2_and_gate_unchanged':True,'visual_QA':'pending'})
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
