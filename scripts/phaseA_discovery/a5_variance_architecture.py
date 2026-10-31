"""A5 static-expression variance architecture; all writes confined to a new module."""
from phaseA_unattended import *
from scipy.linalg import svd
from scipy.stats import spearmanr, rankdata
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

DEST=OUT/'05_variance_architecture'
FRAC=['f_lineage','f_tissue','f_donor','f_lineage_tissue','f_residual']
MET=FRAC+['context_fraction']
COLORS=['#4E79A7','#F28E2B','#59A14F','#9467BD','#B9B9B9']
SEED5=20260918

def write(name,df):cwrite(DEST/name,df)
def read(name):return pd.read_csv(DEST/name)
def support(meta,minimum=3):
    cm=meta[meta.priority_lineage.notna()].copy()
    n=cm.groupby(['priority_lineage','tissue']).donor_id.transform('nunique')
    return cm[n>=minimum]

def orthogonal(d):
    u,s,_=svd(d,full_matrices=False,check_finite=False)
    r=int((s>s[0]*max(d.shape)*np.finfo(float).eps).sum())
    return u[:,:r],r

class Partition:
    """Average main-effect SS increments over all 6 orders; interaction enters last."""
    def __init__(self,meta):
        arrays=[pd.get_dummies(meta[c],dtype=float).to_numpy() for c in ['priority_lineage','tissue','donor_id']]
        self.q={0:np.ones((len(meta),1))/np.sqrt(len(meta))};self.ranks={0:1}
        for bits in range(1,8):
            self.q[bits],self.ranks[bits]=orthogonal(np.column_stack([np.ones(len(meta))]+[arrays[i] for i in range(3) if bits&(1<<i)]))
        pair=meta.priority_lineage.astype(str)+'|'+meta.tissue.astype(str)
        self.full,self.fullrank=orthogonal(np.column_stack([np.ones(len(meta)),pd.get_dummies(pair,dtype=float),arrays[2]]))
        self.n=len(meta)
        self.diagnostics={'n':len(meta),'n_lineages':meta.priority_lineage.nunique(),'n_tissues':meta.tissue.nunique(),'n_donors':meta.donor_id.nunique(),'observed_pairs':pair.nunique(),'possible_pairs':meta.priority_lineage.nunique()*meta.tissue.nunique(),'subset_ranks':self.ranks,'additive_rank':self.ranks[7],'interaction_rank':self.fullrank,'residual_df':len(meta)-self.fullrank}
    def fit(self,y,alternative=False):
        z=y-y.mean(0);ss=(z*z).sum(0)
        explained={b:((q.T@z)**2).sum(0) for b,q in self.q.items()};explained[0]=np.zeros(y.shape[1])
        components=np.zeros((y.shape[1],5))
        orders=[(2,0,1),(2,1,0)] if alternative else list(itertools.permutations(range(3)))
        for order in orders:
            previous=0
            for factor in order:
                nxt=previous|(1<<factor);components[:,factor]+=(explained[nxt]-explained[previous])/len(orders);previous=nxt
        full=((self.full.T@z)**2).sum(0)
        components[:,3]=full-explained[7];components[:,4]=ss-full
        tol=np.maximum(ss,1)*1e-10
        assert np.all(components>=-tol[:,None])
        components=np.maximum(components,0)
        valid=ss>1e-20
        fractions=np.full_like(components,np.nan);fractions[valid]=components[valid]/components[valid].sum(1)[:,None]
        np.testing.assert_allclose(fractions[valid].sum(1),1,atol=1e-12)
        return components/self.n,fractions,ss/self.n

def marginal(y,labels):
    z=y-y.mean(0);ss=(z*z).sum(0);between=np.zeros(y.shape[1])
    for label in pd.unique(labels):
        v=y[np.asarray(labels)==label];between+=len(v)*(v.mean(0)-y.mean(0))**2
    return np.divide(between,ss,out=np.full(len(ss),np.nan),where=ss>1e-20)

def percentile(pool,value):
    pool=np.asarray(pool);pool=pool[np.isfinite(pool)]
    return 100*((pool<value).sum()+.5*(pool==value).sum())/len(pool) if len(pool) and np.isfinite(value) else np.nan

def empirical(pool,value,two=True):
    pool=np.asarray(pool);pool=pool[np.isfinite(pool)]
    if not len(pool) or not np.isfinite(value):return np.nan
    hi=(1+(pool>=value).sum())/(len(pool)+1);lo=(1+(pool<=value).sum())/(len(pool)+1)
    return min(1,2*min(hi,lo)) if two else hi

def freeze():
    for folder in ['00_audit','statistics','source_data','figures','reports','cache']:(DEST/folder).mkdir(parents=True,exist_ok=True)
    p=DEST/'00_audit/protected_hashes.json'
    if not p.exists():jwrite(p,{str(f):sha(f) for f in OUT.rglob('*') if f.is_file() and DEST not in f.parents})
    spec={'seed':SEED5,'primary_method':'Finite-design hierarchical Shapley sums of squares: average lineage/tissue/donor increments across all 6 orders; interaction after additive effects; residual after observed-pair + donor model. All genes use identical fixed-effect estimand. Not REML variance parameters.','primary_support':'>=3 distinct donors per observed lineage x tissue, preserving original pseudobulk rows; missing combinations not filled','primary_panel_metrics':MET,'panel_tests':'Two-sided equal-tail empirical tests for 5 median fractions; upper-tail for median context_fraction; BH across 12 tests (2 panels x 6 metrics).','gene_tests':'Two-sided 17 x 5 candidate-pool tests; BH across 85. Context reported descriptively.','compositional':'Hellinger square-root transform (zeros allowed); PCA background fit; panel-centroid distances from null centroid, leave-one-panel-out for null distances. Raw-fraction Euclidean sensitivity. Secondary BH across 4 tests.','robustness':'All 18 donor and 29 tissue deletions; primary support fixed for deletions; support >=2 and >=4 reselected; donor-first then average lineage/tissue order alternative; all 4 saved matching variants; remove clock targets above background expression/variance P95 or P99 or pseudobulk prevalence below 0.5/0.8, along with same slots in matched panels; no primary panel redefinition.','A2':'Spearman across 17 clock genes, 3 frozen A2 scores x 4 A5 fractions, BH12; 2000 paired-gene bootstrap intervals (descriptive correlated genes); donor/tissue deletion correlation ranges. Same-data methodological corroboration, not independent validation.','negative_control':'Spearman and rank-linear conditional associations; standardized logmean/logvariance + pseudobulk nonzero prevalence; incremental R2 and residual rank correlation for variance; no causal or independent-gene P claims. True single-cell detection not broadly available.','classification':'Choose only after all robustness checks. Robust panel specificity requires primary BH<.05 and >=90% deletion direction consistency, >=80% deletion BH<.05, and direction retained in support/model sensitivities. Otherwise component-specific supported if gene BH findings robust in direction, or C if no matched specificity; D if unavailable/unstable.'}
    if (DEST/'00_audit/prespecified_protocol.json').exists():assert json.loads((DEST/'00_audit/prespecified_protocol.json').read_text())==spec
    else:jwrite(DEST/'00_audit/prespecified_protocol.json',spec)

def landscape():
    freeze();meta,x,genes=load_data();core=[g[5:] for g in protocol()['feature_sets']['CORE_FEATURE_COLS']];clock=[g[5:] for g in protocol()['feature_sets']['EXPANDED_FEATURE_COLS']]
    sm=support(meta);cm=meta[meta.priority_lineage.notna()];si=meta.index.get_indexer(sm.index);ci=meta.index.get_indexer(cm.index)
    model=Partition(sm);jwrite(DEST/'00_audit/design_diagnostics.json',model.diagnostics)
    stats=pd.read_csv(A1/'gene_matching_statistics.csv').set_index('gene').loc[genes]
    enough=(stats['mean']>0)&(stats.variance>1e-12); selected=set(pd.read_csv(A1/'primary_expanded_genes.csv').matched_gene)
    universe=pd.DataFrame({'gene':genes,'sufficient':enough.to_numpy(),'eligible_background':stats.eligible_nonclock.to_numpy(),'exclusion_reason':np.where(~enough,'mean<=0 or variance<=1e-12',np.where(np.isin(genes,clock),'expanded_clock_excluded_from_null','eligible_nonclock'))})
    write('00_audit/exact_gene_universe.csv',universe);write('00_audit/primary_observations.csv',sm.reset_index());write('00_audit/matched_candidate_membership.csv',pd.read_csv(A1/'candidate_pools.csv'))
    support_table=cm.groupby(['priority_lineage','tissue']).agg(n_donors=('donor_id','nunique'),n_observations=('donor_id','size')).reset_index();write('00_audit/lineage_tissue_support.csv',support_table)
    audit=f'''# A5 input audit
Broad matrix: `{DEP/'broad_normalized_mean.npz'}`; SHA256 `{sha(DEP/'broad_normalized_mean.npz')}`.
Metadata: `{OUT/'00_audit/observation_metadata.csv'}`; SHA256 `{sha(OUT/'00_audit/observation_metadata.csv')}`.
Validated matrix: {x.shape}; all-observation landscape uses 2030 rows; marginal major-lineage analysis uses {len(cm)} rows; primary partition uses {len(sm)} original rows in observed lineage/tissue groups with >=3 donors. Equal pseudobulk weights; no cell-count weighting or broad matrix regeneration.
Core: {', '.join(core)}.
Expanded: {', '.join(clock)}.
A1 universe: mean>0 and population variance>1e-12; exclude all 17 clock genes. Primary candidate pool: nearest100 in standardized logmean/logvariance, gene-specific. Uniform draws without duplicate genes per panel; reuse across panels. Detection was not matched. Frozen panels and all variant candidate membership copied read-only.

## Design/model choice, before outcome analysis
{json.dumps(model.diagnostics,indent=2)}
The unbalanced partially crossed design has absent lineage/tissue combinations. Main-effect categorical columns are redundant, handled by rank-aware SVD; the pair model spans only observed combinations, not invented cells. Interaction is the additional observed-pair structure beyond additive lineage+tissue+donor. Its support can overlap donor incidence; only identifiable fitted subspaces are allocated.
The primary estimand is observed finite-design expression variation, not exchangeable random-category population variance. Five named lineages and selected tissues are not treated as draws from a population of interchangeable categories; fine cell types within donor/tissue remain dependent. We therefore choose a single transparent fixed-effect hierarchical SS fallback for every gene, rather than report unstable/assumption-dependent random-lineage variance with five groups. REML was not attempted; no claim of numerical REML failure is made. No fixed/random quantities are mixed. All factors are fixed. Convergence is not applicable to SVD; rank, residual degrees of freedom and zero-variance failures are explicitly exported. This is not an exact random-effect variance-component estimate.
Main lineage, tissue, donor contributions are nonnegative Shapley allocations of additive fitted SS (average all six insertion orders). Shared/confounded additive variation is allocated symmetrically, not uniquely attributed causally. Interaction enters last under marginality; residual is full-model SSE. Fractions divide by supported-subset total SS and sum to one. Absolute components divide SS by n_supported. `total_variance` separately describes all2030 rows; `decomposition_total_variance` is the denominator for component sums. No adjusted-R2 truncation or per-gene model switching.
Global detection is not available. `pseudobulk_nonzero_fraction` is prevalence of positive pseudobulk means, not single-cell detection. Existing true detection summaries are reused only for genes where available.
'''
    (DEST/'00_audit/A5_INPUT_AUDIT.md').write_text(audit,encoding='utf-8')
    if (DEST/'statistics/gene_variance_components.csv').exists():return
    prior=pd.read_csv(OUT/'01_matched_random_controls/null_model_audit/all_gene_lineage_statistics.csv').set_index('gene')
    rows=[];failure=[]
    for start in range(0,len(genes),512):
        ids=np.arange(start,min(start+512,len(genes)));ids=ids[enough.to_numpy()[ids]]
        if not len(ids):continue
        y=x[:,ids];components,f,v=model.fit(y[si]);tissue=marginal(y,meta.tissue);donor=marginal(y,meta.donor_id)
        for j,idx in enumerate(ids):
            g=genes[idx];r={'gene':g,'mean_expression':stats.iloc[idx]['mean'],'total_variance':stats.iloc[idx].variance,'CV':np.nan,'SD_over_mean_processed_scale':np.sqrt(stats.iloc[idx].variance)/stats.iloc[idx]['mean'],'CV_interpretation':'CV withheld: processed source X ratio-scale interpretation not established; SD/mean supplied only as arithmetic descriptive ratio','pseudobulk_nonzero_fraction':float(np.mean(y[:,j]>0)),'single_cell_detection_where_available':prior.loc[g,'mean_single_cell_detection_fraction_where_available'],'lineage_eta_squared':prior.loc[g,'lineage_eta_squared'],'tissue_eta_squared':tissue[j],'donor_eta_squared':donor[j],'decomposition_total_variance':v[j],'is_core':g in core,'is_expanded':g in clock,'eligible_background':bool(stats.iloc[idx].eligible_nonclock),'selected_primary_expanded':g in selected}
            r.update(dict(zip([c[2:]+'_component' for c in FRAC],components[j])));r.update(dict(zip(FRAC,f[j])));r['context_fraction']=f[j,1]+f[j,3];rows.append(r)
            if not np.isfinite(f[j]).all():failure.append({'gene':g,'reason':'No supported-subset variation; fractions undefined','convergence':'not_applicable_fixed_SVD'})
        if start%8192==0:print('A5 landscape',start,flush=True)
    table=pd.DataFrame(rows);bg=table[table.eligible_background]
    for col in ['mean_expression','total_variance','lineage_eta_squared','tissue_eta_squared','donor_eta_squared']+MET:
        values=np.sort(bg[col].dropna());v=table[col].to_numpy();p=(np.searchsorted(values,v,'left')+np.searchsorted(values,v,'right'))*.5/len(values)*100;p[~np.isfinite(v)]=np.nan;table[col+'_background_percentile']=p
    write('statistics/gene_variance_components.csv',table);write('statistics/model_failures.csv',pd.DataFrame(failure,columns=['gene','reason','convergence']))
    write('statistics/clock_variance_components.csv',table[table.is_expanded]);print('A5 decomposed',len(table),flush=True)

def panels():
    result={}
    for variant,_,_,_ in VARIANTS:
        for panel in ['core','expanded']:
            d=pd.read_csv(A1/f'{variant}_{panel}_genes.csv');result[(variant,panel)]=[p.matched_gene.tolist() for _,p in d.groupby('iteration',sort=True)]
    return result

def panel_tests(table,draws,clocksets):
    rows=[];nullrows=[]
    for (variant,panel),selections in draws.items():
        real=clocksets[panel]
        ids=table.index.get_indexer(np.asarray(selections).ravel()).reshape(len(selections),-1)
        assert (ids>=0).all()
        medians=np.nanmedian(table[MET].to_numpy()[ids],axis=1)
        for m in MET:
            observed=float(table.loc[real,m].median());null=medians[:,MET.index(m)]
            rows.append({'variant':variant,'panel':panel,'metric':m,'observed':observed,'null_median':np.median(null),'delta':observed-np.median(null),'percentile':percentile(null,observed),'P':empirical(null,observed,m!='context_fraction'),'n_null':len(null)})
            nullrows.extend({'variant':variant,'panel':panel,'metric':m,'iteration':i+1,'value':v} for i,v in enumerate(null))
    result=pd.DataFrame(rows);result['BH12']=result.groupby('variant').P.transform(lambda v:bh(v.to_numpy()))
    return result,pd.DataFrame(nullrows)

def inference():
    table=read('statistics/gene_variance_components.csv').set_index('gene');clocksets={p:[g[5:] for g in protocol()['feature_sets'][key]] for p,key in [('core','CORE_FEATURE_COLS'),('expanded','EXPANDED_FEATURE_COLS')]}
    draws=panels(); tests,null=panel_tests(table,draws,clocksets);write('statistics/panel_matched_tests.csv',tests);write('statistics/panel_null_distributions.csv',null)
    pools=pd.read_csv(A1/'candidate_pools.csv');rows=[]
    for (variant,g),part in pools.groupby(['variant','clock_gene'],sort=False):
        for m in FRAC:
            v=table.loc[part.matched_gene,m].dropna().to_numpy();obs=table.loc[g,m]
            rows.append({'variant':variant,'gene':g,'metric':m,'observed':obs,'candidate_median':np.median(v),'n_candidates':len(v),'delta':obs-np.median(v),'percentile':percentile(v,obs),'P_two_sided':empirical(v,obs,True)})
    gtests=pd.DataFrame(rows);gtests['BH85']=gtests.groupby('variant').P_two_sided.transform(lambda v:bh(v.to_numpy()));write('statistics/gene_candidate_tests.csv',gtests)
    # Hellinger is an ordinary Euclidean metric on square-root compositions; zeros valid.
    good=table[table[FRAC].notna().all(axis=1)];bg=good[good.eligible_background];rng=np.random.default_rng(SEED5)
    sample=sorted(rng.choice(bg.index,min(5000,len(bg)),replace=False));selected=sorted(set(pools[pools.variant.eq('primary')].matched_gene));projection=[];ctests=[]
    for method in ['hellinger','raw_fraction']:
        transform=lambda a:np.sqrt(a) if method=='hellinger' else a
        values=transform(good[FRAC].to_numpy());pca=PCA(n_components=2).fit(transform(bg[FRAC].to_numpy()));coords=pca.transform(values)
        proj=pd.DataFrame(coords,index=good.index,columns=['PC1','PC2']);proj['gene']=good.index;proj['method']=method;proj['is_clock']=good.is_expanded;proj['sample_background']=good.index.isin(sample);proj['is_matched_candidate']=good.index.isin(selected);projection.append(proj.reset_index(drop=True))
        jwrite(DEST/f'statistics/{method}_PCA.json',{'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),'loadings':pca.components_.tolist(),'columns':FRAC,'background_fit_n':len(bg)})
        for panel in ['core','expanded']:
            cloud=np.array([transform(table.loc[s,FRAC].to_numpy()).mean(0) for s in draws[('primary',panel)]]);center=cloud.mean(0);obsvec=transform(table.loc[clocksets[panel],FRAC].to_numpy()).mean(0)
            obs=np.linalg.norm(obsvec-center);dist=np.linalg.norm(cloud-(cloud.sum(0)-cloud)/(len(cloud)-1),axis=1)
            ctests.append({'panel':panel,'space':method,'centroid_distance':obs,'null_median_distance':np.median(dist),'P':empirical(dist,obs,False),'percentile':percentile(dist,obs)})
    ct=pd.DataFrame(ctests);ct['BH4']=bh(ct.P.to_numpy());write('statistics/composition_space_tests.csv',ct);write('source_data/composition_projection.csv',pd.concat(projection,ignore_index=True))
    return table,clocksets,draws

def a2_link(table):
    a2=pd.read_csv(OUT/'statistics/feature_effects_and_scores.csv');a2['gene']=a2.feature.str.replace('expr_','',regex=False)
    joined=a2.set_index('gene').join(table[MET],how='inner');write('statistics/A2_A5_gene_join.csv',joined.reset_index())
    rng=np.random.default_rng(SEED5);rows=[]
    for old in ['lineage_stability','tissue_remodeling','cell_identity_tissue_context_interaction_adj_r2']:
        for new in ['f_lineage','f_tissue','f_lineage_tissue','context_fraction']:
            x=joined[old].to_numpy();y=joined[new].to_numpy();r,p=spearmanr(x,y);boots=[]
            for _ in range(2000):
                idx=rng.integers(0,len(x),len(x));boots.append(spearmanr(x[idx],y[idx]).statistic)
            rows.append({'A2':old,'A5':new,'n':len(x),'rho':r,'nominal_P':p,'bootstrap_low':np.nanquantile(boots,.025),'bootstrap_high':np.nanquantile(boots,.975),'uncertainty_note':'Descriptive paired-gene bootstrap; genes correlated and same data, not independent validation.'})
    df=pd.DataFrame(rows);df['nominal_BH12']=bh(df.nominal_P.to_numpy());write('statistics/A2_A5_correlations.csv',df)

def negative_control(table):
    bg=table[table.eligible_background].copy();cols=['mean_expression','total_variance','pseudobulk_nonzero_fraction','lineage_eta_squared','tissue_eta_squared']+MET
    pairs=[]
    for a,b in itertools.combinations(cols,2):
        d=bg[[a,b]].dropna();pairs.append({'x':a,'y':b,'n':len(d),'Spearman_rho':spearmanr(d[a],d[b]).statistic})
    write('statistics/negative_control_pairwise.csv',pd.DataFrame(pairs));rows=[]
    for outcome in ['lineage_eta_squared','tissue_eta_squared']+MET:
        d=bg[['mean_expression','total_variance','pseudobulk_nonzero_fraction',outcome]].dropna();y=rankdata(d[outcome]);y=(y-y.mean())/y.std()
        predictors=np.column_stack([rankdata(d[c]) for c in ['mean_expression','total_variance','pseudobulk_nonzero_fraction']]);predictors=StandardScaler().fit_transform(predictors)
        full=np.column_stack([np.ones(len(d)),predictors]);reduced=full[:,[0,1,3]]
        beta=np.linalg.lstsq(full,y,rcond=None)[0];res=y-full@beta;res0=y-reduced@np.linalg.lstsq(reduced,y,rcond=None)[0];varres=predictors[:,1]-reduced@np.linalg.lstsq(reduced,predictors[:,1],rcond=None)[0]
        rows.append({'outcome':outcome,'n':len(d),'standardized_rank_beta_logmean':beta[1],'standardized_rank_beta_logvariance':beta[2],'standardized_rank_beta_prevalence':beta[3],'full_R2':1-np.mean(res**2),'variance_incremental_R2':np.mean(res0**2)-np.mean(res**2),'variance_partial_rank_correlation':np.corrcoef(varres,res0)[0,1],'design_condition_number':np.linalg.cond(full)})
    write('statistics/negative_control_multivariable.csv',pd.DataFrame(rows))
    # Expression regime association with variance, descriptive conditional residuals.
    d=bg[['mean_expression','total_variance','pseudobulk_nonzero_fraction']];X=np.column_stack([np.ones(len(d)),np.log10(d.mean_expression),np.log10(d.mean_expression)**2,d.pseudobulk_nonzero_fraction]);y=np.log10(d.total_variance);b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b
    clock=table[table.is_expanded];cx=np.column_stack([np.ones(len(clock)),np.log10(clock.mean_expression),np.log10(clock.mean_expression)**2,clock.pseudobulk_nonzero_fraction]);cr=np.log10(clock.total_variance)-cx@b
    write('statistics/clock_expression_adjusted_variance.csv',pd.DataFrame({'gene':clock.index,'logvariance_residual':cr,'residual_background_percentile':[percentile(res,v) for v in cr]}))
    jwrite(DEST/'statistics/expression_variance_model.json',{'formula':'log10 variance ~ 1 + log10 mean + log10 mean squared + pseudobulk prevalence','R2':float(1-np.var(res)/np.var(y)),'n':len(d),'coefficients':b.tolist(),'causal':False})
    candidate=set(pd.read_csv(A1/'candidate_pools.csv').query('variant == "primary"').matched_gene);local=bg.loc[bg.index.isin(candidate)].copy();local['dominant_fraction']=local[FRAC].idxmax(axis=1)
    write('statistics/clock_like_gene_dominance.csv',local.reset_index());write('statistics/clock_like_dominance_summary.csv',local.groupby('dominant_fraction')[['mean_expression','total_variance','pseudobulk_nonzero_fraction']+MET].median().reset_index())
    rows=[]
    for outcome in ['lineage_eta_squared','tissue_eta_squared']+MET:
        d=local[['mean_expression','total_variance','pseudobulk_nonzero_fraction',outcome]].dropna();y=rankdata(d[outcome]);y=(y-y.mean())/y.std()
        predictors=StandardScaler().fit_transform(np.column_stack([rankdata(d[c]) for c in ['mean_expression','total_variance','pseudobulk_nonzero_fraction']]))
        full=np.column_stack([np.ones(len(d)),predictors]);reduced=full[:,[0,1,3]];b=np.linalg.lstsq(full,y,rcond=None)[0];res=y-full@b;res0=y-reduced@np.linalg.lstsq(reduced,y,rcond=None)[0];vr=predictors[:,1]-reduced@np.linalg.lstsq(reduced,predictors[:,1],rcond=None)[0]
        rows.append({'outcome':outcome,'n':len(d),'rank_beta_mean':b[1],'rank_beta_variance':b[2],'rank_beta_prevalence':b[3],'full_R2':1-np.mean(res**2),'variance_incremental_R2':np.mean(res0**2)-np.mean(res**2),'variance_partial_rank_correlation':np.corrcoef(vr,res0)[0,1]})
    write('statistics/clock_like_multivariable.csv',pd.DataFrame(rows))

def robustness(table,clocksets,draws):
    meta,x,genes=load_data();sm=support(meta);pools=pd.read_csv(A1/'candidate_pools.csv');needed=sorted(set(pools[pools.variant.eq('primary')].matched_gene)|set(clocksets['expanded']));index=pd.Index(genes).get_indexer(needed)
    primary={k:v for k,v in draws.items() if k[0]=='primary'};base=read('statistics/panel_matched_tests.csv').query('variant == "primary"').set_index(['panel','metric'])
    scenarios=[('delete_donor',str(v),sm[sm.donor_id.ne(v)],False) for v in sorted(sm.donor_id.unique())]+[('delete_tissue',str(v),sm[sm.tissue.ne(v)],False) for v in sorted(sm.tissue.unique())]+[('minimum_support','2',support(meta,2),False),('minimum_support','4',support(meta,4),False),('alternative_model','donor_first',sm,True)]
    alltests=[];gene_changes=[];corrs=[];diag=[]
    a2=read('statistics/A2_A5_gene_join.csv').set_index('gene')
    for kind,name,sub,alt in scenarios:
        safe=hashlib.sha256((kind+name).encode()).hexdigest()[:12];path=DEST/f'cache/robust_{safe}.csv'
        if path.exists():f=pd.read_csv(path).set_index('gene')
        else:
            model=Partition(sub);ii=meta.index.get_indexer(sub.index);_,fraction,_=model.fit(x[np.ix_(ii,index)],alternative=alt);f=pd.DataFrame(fraction,index=needed,columns=FRAC);f['context_fraction']=f.f_tissue+f.f_lineage_tissue;f.index.name='gene';cwrite(path,f.reset_index());jwrite(path.with_suffix('.json'),model.diagnostics)
        diag.append({'kind':kind,'removed_or_rule':name,'n':len(sub),'n_donors':sub.donor_id.nunique(),'n_tissues':sub.tissue.nunique(),'undefined_gene_fractions':int(f[FRAC].isna().any(axis=1).sum())})
        assert f[FRAC].notna().all().all(),'Undefined matched gene fraction requires explicit handling'
        tests,_=panel_tests(f,primary,clocksets);tests['scenario']=kind;tests['removed_or_rule']=name;alltests.append(tests)
        for g in clocksets['expanded']:
            candidates=pools[(pools.variant.eq('primary'))&pools.clock_gene.eq(g)].matched_gene
            for m in MET:gene_changes.append({'scenario':kind,'removed_or_rule':name,'gene':g,'metric':m,'value':f.loc[g,m],'delta_vs_candidate_median':f.loc[g,m]-f.loc[candidates,m].median()})
        for old in ['lineage_stability','tissue_remodeling','cell_identity_tissue_context_interaction_adj_r2']:
            for new in ['f_lineage','f_tissue','f_lineage_tissue','context_fraction']:corrs.append({'scenario':kind,'removed_or_rule':name,'A2':old,'A5':new,'rho':spearmanr(a2.loc[clocksets['expanded'],old],f.loc[clocksets['expanded'],new]).statistic})
        print('A5 robustness',kind,name,flush=True)
    # Outcome-independent expression/variance/prevalence target-slot deletion sensitivities.
    bg=table[table.eligible_background]
    for col,cut,op in [('mean_expression',.95,'quantile'),('mean_expression',.99,'quantile'),('total_variance',.95,'quantile'),('total_variance',.99,'quantile'),('pseudobulk_nonzero_fraction',.5,'minimum'),('pseudobulk_nonzero_fraction',.8,'minimum')]:
        threshold=bg[col].quantile(cut) if op=='quantile' else cut;sets={};filtered={};removed={}
        for panel,real in clocksets.items():
            keep=[j for j,g in enumerate(real) if (table.loc[g,col]<=threshold if op=='quantile' else table.loc[g,col]>=threshold)];sets[panel]=[real[j] for j in keep];removed[panel]=[g for j,g in enumerate(real) if j not in keep]
            filtered[('primary',panel)]=[[s[j] for j in keep] for s in primary[('primary',panel)]]
        if min(map(len,sets.values()))==0:continue
        tests,_=panel_tests(table,filtered,sets);tests['scenario']='target_slot_sensitivity';tests['removed_or_rule']=f'{col}_{op}_{cut}';tests['retained_clock_n']=tests.panel.map({p:len(v) for p,v in sets.items()});tests['removed_clock_genes']=tests.panel.map({p:','.join(v) for p,v in removed.items()});alltests.append(tests)
    results=pd.concat(alltests,ignore_index=True);write('statistics/robustness_panel_tests.csv',results);write('statistics/robustness_gene_effects.csv',pd.DataFrame(gene_changes));write('statistics/robustness_A2_correlations.csv',pd.DataFrame(corrs));write('statistics/robustness_design_diagnostics.csv',pd.DataFrame(diag))
    rows=[]
    for (panel,m),part in results.groupby(['panel','metric']):
        point=base.loc[(panel,m)];deletion=part[part.scenario.str.startswith('delete_')]
        rows.append({'panel':panel,'metric':m,'primary_delta':point.delta,'primary_BH':point.BH12,'n_deletions':len(deletion),'deletion_direction_consistency':float(np.mean(np.sign(deletion.delta)==np.sign(point.delta))),'deletion_BH_significant_fraction':float(np.mean(deletion.BH12<.05)),'deletion_delta_min':deletion.delta.min(),'deletion_delta_max':deletion.delta.max(),'all_sensitivity_delta_min':part.delta.min(),'all_sensitivity_delta_max':part.delta.max(),'worst_removal':deletion.iloc[np.argmax(np.abs(deletion.delta-point.delta))].removed_or_rule})
    write('statistics/robustness_summary.csv',pd.DataFrame(rows))

def run():
    landscape();table,sets,draws=inference();a2_link(table);negative_control(table);robustness(table,sets,draws)
    print('A5 NUMERICAL ANALYSES COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
