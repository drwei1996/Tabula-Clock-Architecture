"""Resume only Phase A1/A3; preserve completed A2 and frozen upstream inputs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import json,sys,hashlib,time,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.umapdeps'))
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplconfig'))
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist,squareform
from scipy.stats import t, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,silhouette_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/phaseA_clock_architecture'
DEP=OUT/'dependencies'
SEED=20260913
METRICS=['balanced_accuracy','macro_f1','silhouette','distance_ratio']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): (OUT/p).write_text(json.dumps(x,indent=2,default=str),encoding='utf-8')
def csv(p,x): x.to_csv(OUT/p,index=False)
def bh(p):
    p=np.asarray(p); order=np.argsort(p); q=np.empty(len(p)); q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)
def frozen():
    paths=[p for p in (OUT/'02_backbone_remodeling').rglob('*') if p.is_file()]
    paths += [OUT/p for p in ['statistics/feature_effects_and_scores.csv','statistics/distance_summary.csv','statistics/distance_driver_sensitivity.csv','source_data/FigA3_distances.csv','source_data/FigA4_lineage_plasticity.csv','source_data/FigA5_feature_scores.csv','03_unbiased_pathways/clock_remodeling_phenotype_original_observations.csv','03_unbiased_pathways/clock_remodeling_phenotype_donor_tissue_lineage.csv']]
    return {str(p):sha(p) for p in paths}

def load():
    data=np.load(DEP/'broad_normalized_mean.npz'); meta=pd.read_csv(OUT/'00_audit/observation_metadata.csv').set_index('group_id')
    order=pd.Index(data['group_id']).get_indexer(meta.index); assert (order>=0).all() and len(order)==len(data['group_id'])
    x=data['expression'][order]; genes=data['genes']; dm=pd.read_csv(DEP/'observation_metadata.tsv',sep='\t').set_index('group_id').loc[meta.index]
    for c in ['donor_id','tissue','cell_type','n_cells']: assert np.array_equal(meta[c],dm[c])
    assert np.isfinite(x).all()
    original_path=ROOT.parent/'TabulaCircadian_deliverables/source_data/pseudobulk__clock_architecture_features.tsv'
    original=pd.read_csv(original_path,sep='\t').set_index('group_id').loc[meta.index]
    errors={}
    for col in original.columns:
        if col.startswith('expr_'):
            where=np.flatnonzero(genes==col[5:])
            if len(where):
                np.testing.assert_allclose(x[:,where[0]],original[col],rtol=1e-10,atol=1e-10)
                errors[col]=float(np.abs(x[:,where[0]]-original[col]).max())
            else:assert col=='expr_CTGF' and original[col].eq(0).all()
    write('00_audit/local_dependency_verification.json',{'original_feature_sha256':sha(original_path),'measured_gene_overlap_verified':len(errors),'max_absolute_error':max(errors.values()),'metadata_exact_match':True})
    return meta,x,genes

def a1(meta,x,genes,protocol):
    path=OUT/'01_matched_random_controls'; rng=np.random.default_rng(SEED)
    means=x.mean(0); variances=x.var(0); valid=(means>0)&(variances>1e-12)
    expanded=[v[5:] for v in protocol['feature_sets']['EXPANDED_FEATURE_COLS']]
    universe=np.flatnonzero(valid&~np.isin(genes,expanded))
    # Matching uses unlabeled marginal abundance/variance; no outcome-dependent selection.
    stats=np.column_stack([np.log10(np.maximum(means,1e-300)),np.log10(np.maximum(variances,1e-300))])
    stats=(stats-stats[universe].mean(0))/stats[universe].std(0)
    pools={g:universe[np.argsort(((stats[universe]-stats[np.flatnonzero(genes==g)[0]])**2).sum(1),kind='stable')[:100]] for g in expanded}
    csv('01_matched_random_controls/gene_universe_and_matching_statistics.csv',pd.DataFrame({'gene':genes,'mean':means,'variance':variances,'eligible_nonclock':np.isin(np.arange(len(genes)),universe)}))
    csv('01_matched_random_controls/matching_candidate_pools.csv',pd.DataFrame([{'clock_gene':g,'candidate':genes[j],'rank':r+1,'distance':float(np.linalg.norm(stats[j]-stats[np.flatnonzero(genes==g)[0]]))} for g,ids in pools.items() for r,j in enumerate(ids)]))
    cm=meta[meta.priority_lineage.notna()]; ii=meta.index.get_indexer(cm.index); labels=cm.priority_lineage.to_numpy()
    split=pd.read_csv(path/'classification_split_manifest.csv').set_index('group_id').loc[cm.index]
    folds=[(np.flatnonzero(split.test_fold.to_numpy()!=f),np.flatnonzero(split.test_fold.to_numpy()==f)) for f in sorted(split.test_fold.unique())]
    for tr,te in folds: assert not set(cm.iloc[tr].donor_id)&set(cm.iloc[te].donor_id)
    groups=[]; keys=[]
    for key,part in cm.groupby(['priority_lineage','tissue'],sort=True):
        if part.donor_id.nunique()>=3: keys.append(key);groups.append(meta.index.get_indexer(part.index))
    assert len(groups)==97
    w=[]; b=[]
    for a,c in itertools.combinations(range(len(keys)),2):
        if keys[a][0]==keys[c][0]:w.append((a,c))
        elif keys[a][1]==keys[c][1]:b.append((a,c))
    w=np.array(w);b=np.array(b);assert len(w)==928 and len(b)==139
    def metrics(ids):
        values=x[:,ids]; z=StandardScaler().fit_transform(values); xx=values[ii]; pred=np.empty(len(ii),dtype=object)
        for tr,te in folds:
            scale=StandardScaler(); train=scale.fit_transform(xx[tr]); test=scale.transform(xx[te])
            model=RidgeClassifier(alpha=1,class_weight='balanced').fit(train,labels[tr]);pred[te]=model.predict(test)
        cent=np.array([z[g].mean(0) for g in groups]);
        return dict(balanced_accuracy=balanced_accuracy_score(labels,pred),macro_f1=f1_score(labels,pred,average='macro'),silhouette=silhouette_score(z[ii],labels),distance_ratio=np.median(np.linalg.norm(cent[b[:,0]]-cent[b[:,1]],axis=1))/np.median(np.linalg.norm(cent[w[:,0]]-cent[w[:,1]],axis=1)))
    null=[]; panels=[]; observed=[]; diagnostics=[]
    old=pd.read_csv(path/'observed_metrics.csv').set_index('panel')
    for panel,key in [('Core clock','CORE_FEATURE_COLS'),('Expanded clock','EXPANDED_FEATURE_COLS')]:
        targets=[v[5:] for v in protocol['feature_sets'][key]]; ids=[np.flatnonzero(genes==g)[0] for g in targets]
        obs=metrics(ids)
        for m in METRICS: np.testing.assert_allclose(obs[m],old.loc[panel,m],rtol=1e-9,atol=1e-10)
        observed.append({'panel':panel,'n_observations':len(ii),'n_genes':len(ids),**obs,'n_null_panels':1000,'null_status':'COMPLETE'})
        checkpoint=path/('null_core_checkpoint.csv' if panel=='Core clock' else 'null_expanded_checkpoint.csv')
        rows=[]
        for iteration in range(1000):
            selected=[]
            for g in targets:
                pool=pools[g]; j=int(rng.choice(pool[~np.isin(pool,selected)])); selected.append(j)
                panels.append({'panel':panel,'iteration':iteration+1,'clock_gene':g,'matched_gene':genes[j]})
            rows.append({'panel':panel,'iteration':iteration+1,**metrics(selected)})
            delta=stats[selected]-stats[ids]
            diagnostics.append({'panel':panel,'iteration':iteration+1,'mean_abs_standardized_log_mean_difference':np.abs(delta[:,0]).mean(),'mean_abs_standardized_log_variance_difference':np.abs(delta[:,1]).mean(),'max_matching_distance':np.linalg.norm(delta,axis=1).max()})
            if (iteration+1)%100==0:
                pd.DataFrame(rows).to_csv(checkpoint,index=False); print(panel,iteration+1,flush=True)
        null.extend(rows)
    null=pd.DataFrame(null);obs=pd.DataFrame(observed);summary=[]
    for _,r in obs.iterrows():
        for metric in METRICS:
            v=null.loc[null.panel.eq(r.panel),metric].to_numpy(); value=r[metric]
            summary.append({'panel':r.panel,'metric':metric,'observed':value,'null_mean':v.mean(),'null_median':np.median(v),'null_sd':v.std(ddof=1),'null_q025':np.quantile(v,.025),'null_q975':np.quantile(v,.975),'empirical_percentile':100*(np.sum(v<value)+.5*np.sum(v==value))/len(v),'empirical_P':(1+np.sum(v>=value))/(len(v)+1),'Z':(value-v.mean())/v.std(ddof=1),'n_null_panels':len(v)})
    summary=pd.DataFrame(summary); summary['FDR_8_metrics']=bh(summary.empirical_P)
    for name,df in [('observed_metrics',obs),('empirical_null_distributions',null),('empirical_null_summary',summary),('matched_panel_genes',pd.DataFrame(panels)),('matching_diagnostics',pd.DataFrame(diagnostics))]:csv(f'01_matched_random_controls/{name}.csv',df)
    return summary

def associations(scores,meta,y,extra_tissue=False):
    """FWL slopes, conservative crossed-cluster sandwich and small-cluster t reference."""
    n=len(y); yy=(y-y.mean())/y.std(); sd=scores.std(0); xx=(scores-scores.mean(0))/np.where(sd>0,sd,1)
    cols=[np.ones((n,1)),pd.get_dummies(meta.donor_id,dtype=float).to_numpy(),np.log1p(meta.n_cells.to_numpy())[:,None]]
    if extra_tissue: cols.append(pd.get_dummies(meta.tissue,dtype=float).to_numpy())
    design=np.column_stack(cols); u,s,_=np.linalg.svd(design,full_matrices=False); rank=np.sum(s>s[0]*max(design.shape)*np.finfo(float).eps); q=u[:,:rank]
    rx=xx-q@(q.T@xx); ry=yy-q@(q.T@yy); denom=(rx*rx).sum(0); usable=(denom>1e-10)&(sd>1e-12)
    beta=np.divide(rx.T@ry,denom,out=np.full(len(sd),np.nan),where=usable)
    residual=ry[:,None]-rx*beta; influence=rx*residual
    def meat(keys):
        codes,unique=pd.factorize(keys); count=len(unique)
        block=np.zeros((count,len(sd)));np.add.at(block,codes,influence)
        return (block*block).sum(0)*count/(count-1)*(n-1)/(n-rank-1),count
    vd,nd=meat(meta.donor_id);vt,nt=meat(meta.tissue);vi,_=meat(meta.donor_id.astype(str)+'|'+meta.tissue.astype(str))
    variance=np.maximum.reduce([vd,vt,vd+vt-vi])/denom**2
    se=np.sqrt(variance);df=min(nd,nt)-1; stat=beta/se;p=2*t.sf(np.abs(stat),df); p[~usable|~np.isfinite(p)]=1
    return beta,se,p,df,rank,rx,ry

def a3(meta,x,genes,protocol):
    clock={v[5:] for v in protocol['feature_sets']['EXPANDED_FEATURE_COLS']}; index={g:i for i,g in enumerate(genes)}
    z=StandardScaler().fit_transform(x); variable=x.var(0)>1e-12
    definitions=[]; scorecols=[]; provenance=[]
    for collection,filename in [('Hallmark','h.all.v2026.1.Hs.symbols.gmt'),('Reactome','c2.cp.reactome.v2026.1.Hs.symbols.gmt')]:
        file=DEP/filename; lines=file.read_text().splitlines(); provenance.append({'collection':collection,'version':'MSigDB 2026.1.Hs','reactome_release':95 if collection=='Reactome' else None,'url':'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/'+filename,'sha256':sha(file),'n_sets':len(lines)})
        for line in lines:
            name,url,*members=line.split('\t'); members=set(members);members={'ARNTL' if v=='BMAL1' else v for v in members}
            available=sorted(g for g in members-clock if g in index and variable[index[g]])
            ids=[index[g] for g in available];eligible=len(ids)>=5
            definitions.append({'collection':collection,'pathway':name,'n_genes_original':len(members),'n_clock_excluded':len(members&clock),'n_genes_scored':len(ids),'coverage_fraction':len(ids)/len(members),'status':'TESTED' if eligible else 'INSUFFICIENT_COVERAGE','genes_scored':';'.join(available)})
            scorecols.append(z[:,ids].mean(1) if eligible else np.zeros(len(meta)))
    definitions=pd.DataFrame(definitions); scores=np.column_stack(scorecols)
    csv('03_unbiased_pathways/pathway_coverage.csv',definitions);write('dependencies/pathway_provenance.json',provenance)
    phenotype=pd.read_csv(OUT/'03_unbiased_pathways/clock_remodeling_phenotype_donor_tissue_lineage.csv')
    keycols=['priority_lineage','donor_id','tissue']; agg=meta.reset_index().groupby(keycols,sort=False).indices
    aggregate_scores=np.array([scores[agg[(r.lineage,r.donor_id,r.tissue)]].mean(0) for r in phenotype.itertuples()])
    np.savez_compressed(OUT/'03_unbiased_pathways/pathway_scores.npz',scores=aggregate_scores,pathways=definitions.pathway.to_numpy(str),lineage=phenotype.lineage.to_numpy(str),donor_id=phenotype.donor_id.to_numpy(str),tissue=phenotype.tissue.to_numpy(str))
    rows=[]
    for lineage,part in phenotype.groupby('lineage'):
        sc=aggregate_scores[part.index];y=part.distance.to_numpy(); beta,se,p,df,rank,rx,ry=associations(sc,part,y)
        sb,sse,sp,*_=associations(sc,part,y,True)
        signs=[]
        for donor in part.donor_id.unique():
            keep=part.donor_id.ne(donor).to_numpy();bb,*_=associations(sc[keep],part.loc[keep],y[keep]);signs.append(np.sign(bb)==np.sign(beta))
        signfrac=np.mean(signs,0)
        for j,r in definitions.iterrows():
            rows.append({**r.to_dict(),'lineage':lineage,'n_observations':len(part),'n_donors':part.donor_id.nunique(),'n_tissues':part.tissue.nunique(),'beta_standardized':beta[j],'se_crossed_cluster':se[j],'ci_low':beta[j]-t.ppf(.975,df)*se[j],'ci_high':beta[j]+t.ppf(.975,df)*se[j],'P':p[j],'cluster_df':df,'donor_deletion_sign_fraction':signfrac[j],'tissue_adjusted_beta':sb[j],'tissue_adjusted_P':sp[j]})
    result=pd.DataFrame(rows);result['FDR']=np.nan;result['tissue_adjusted_FDR']=np.nan
    for collection in result.collection.unique():
        mask=result.collection.eq(collection)
        # Full collection x five lineages is the multiplicity family; ineligible sets carry P=1.
        result.loc[mask,'FDR']=bh(result.loc[mask,'P'].fillna(1));result.loc[mask,'tissue_adjusted_FDR']=bh(result.loc[mask,'tissue_adjusted_P'].fillna(1))
    result['rank_within_collection']=result.groupby('collection').P.rank(method='min').astype(int)
    csv('03_unbiased_pathways/all_pathway_lineage_associations.csv',result)
    for collection in ['Hallmark','Reactome']:csv(f'03_unbiased_pathways/{collection.lower()}_screen.csv',result[result.collection.eq(collection)])
    names={'Hypoxia':'HALLMARK_HYPOXIA','Fatty acid metabolism':'HALLMARK_FATTY_ACID_METABOLISM','IFN-gamma response':'HALLMARK_INTERFERON_GAMMA_RESPONSE','TNF-alpha/NF-kB':'HALLMARK_TNFA_SIGNALING_VIA_NFKB','ECM':'REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION'}
    hypotheses=[]
    for label,name in names.items():
        subset=result[result.pathway.eq(name)].sort_values('P');assert len(subset)==5
        best=subset.iloc[0];sig=subset[subset.FDR.lt(.05)]
        hypotheses.append({'hypothesis':label,'pathway':name,'status':'YES' if len(sig) else 'NO','best_lineage':best.lineage,'beta':best.beta_standardized,'P':best.P,'FDR':best.FDR,'rank':best.rank_within_collection,'n_significant_lineages':len(sig),'significant_lineages':';'.join(sig.lineage),'interpretation':'Emergence in complete screen, not causal independence or independent replication'})
    hypotheses=pd.DataFrame(hypotheses);csv('03_unbiased_pathways/hypothesized_pathway_status.csv',hypotheses)
    return result,hypotheses,definitions

def plots(summary,pathways):
    null=pd.read_csv(OUT/'01_matched_random_controls/empirical_null_distributions.csv')
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for i,panel in enumerate(['Core clock','Expanded clock']):
        for j,metric in enumerate(METRICS):
            ax=axes[i,j];ax.hist(null.loc[null.panel.eq(panel),metric],bins=30,color='#879caf');r=summary[summary.panel.eq(panel)&summary.metric.eq(metric)].iloc[0];ax.axvline(r.observed,color='#b23b3b',lw=2);ax.set_title(f'{panel}\n{metric}\nP = {r.empirical_P:.4g}',fontsize=9)
    fig.tight_layout();save(fig,'FigA1_matched_random_controls')
    predictions=pd.read_csv(OUT/'source_data/FigA2_predictions.csv');lineages=['b_cell','t_cell','macrophage_monocyte','fibroblast','endothelial']
    from sklearn.metrics import confusion_matrix
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,panel in zip(axes,['Core clock','Expanded clock']):
        part=predictions[predictions.panel.eq(panel)];counts=confusion_matrix(part.observed,part.predicted,labels=lineages);rate=counts/counts.sum(1)[:,None];ax.imshow(rate,vmin=0,vmax=1,cmap='Blues')
        for i,j in itertools.product(range(5),repeat=2):ax.text(j,i,f'{rate[i,j]:.0%}\n({counts[i,j]})',ha='center',va='center',fontsize=8,color='white' if rate[i,j]>.55 else 'black')
        ax.set_xticks(range(5),lineages,rotation=35,ha='right',fontsize=8);ax.set_yticks(range(5),lineages,fontsize=8);ax.set_title(panel);ax.set_xlabel('Predicted lineage');ax.set_ylabel('Observed lineage')
    fig.suptitle('Original donor-held-out predictions; matched-control comparison in FigA1');fig.tight_layout();save(fig,'FigA2_clock_lineage_classifier_confusion')
    top=pathways[pathways.collection.eq('Hallmark')].sort_values('P').head(20).sort_values('beta_standardized')
    fig,ax=plt.subplots(figsize=(12,9));ax.errorbar(top.beta_standardized,np.arange(len(top)),xerr=[top.beta_standardized-top.ci_low,top.ci_high-top.beta_standardized],fmt='o',color='#416785');ax.set_yticks(np.arange(len(top)),[r.pathway.replace('HALLMARK_','')+' | '+r.lineage for r in top.itertuples()],fontsize=8);ax.axvline(0,color='grey',lw=.6);ax.set_xlabel('Standardized association with clock remodeling (approximate 95% interval)');ax.set_title('Complete Hallmark screen: 20 smallest P values');fig.tight_layout();save(fig,'FigA6_unbiased_hallmark_screen')
    targets=['HALLMARK_HYPOXIA','HALLMARK_FATTY_ACID_METABOLISM','HALLMARK_INTERFERON_GAMMA_RESPONSE','HALLMARK_TNFA_SIGNALING_VIA_NFKB','REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION']
    ss=pathways[pathways.pathway.isin(targets)];pivot=ss.pivot(index='pathway',columns='lineage',values='beta_standardized').loc[targets];q=ss.pivot(index='pathway',columns='lineage',values='FDR').loc[targets,pivot.columns]
    fig,ax=plt.subplots(figsize=(11,5));im=ax.imshow(pivot,cmap='RdBu_r',vmin=-max(abs(pivot.to_numpy()).max(),.1),vmax=max(abs(pivot.to_numpy()).max(),.1));ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=25,ha='right');ax.set_yticks(range(5),['Hypoxia','Fatty acid metabolism','IFN-gamma','TNF/NF-kB','ECM']);
    for i,j in itertools.product(range(5),range(len(pivot.columns))):ax.text(j,i,f'{pivot.iloc[i,j]:.2f}'+('*' if q.iloc[i,j]<.05 else ''),ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Standardized beta');ax.set_title('Prespecified programs within the complete screens\n* collection-wide FDR < 0.05');fig.tight_layout();save(fig,'FigA7_pathway_remodeling_associations')
def save(fig,name):
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'{name}.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    from PIL import Image
    preview=OUT/'figures/_preview';preview.mkdir(exist_ok=True)
    with Image.open(OUT/'figures'/f'{name}.png') as im:
        im.thumbnail((1600,1000));im.save(preview/f'{name}.png')

def report(summary,pathways,hypotheses,definitions):
    primary=summary[summary.panel.eq('Expanded clock')];wins=primary[primary.empirical_P.lt(.05)&primary.Z.gt(0)]
    specificity='balanced_accuracy' in set(wins.metric) and bool({'silhouette','distance_ratio'}&set(wins.metric))
    assoc=pathways.FDR.lt(.05).any()
    conclusion='STRONGLY SUPPORTED' if specificity and assoc else 'PARTIALLY SUPPORTED'
    decision='GO FOR INDEPENDENT REPLICATION' if specificity and assoc else 'PAUSE BEFORE REPLICATION'
    def show(df):return '```csv\n'+df.to_csv(index=False,float_format='%.6g')+'```\n'
    oldreport=OUT/'reports/PHASE_A_GATE_REPORT.md'; archived=OUT/'00_audit/PHASE_A_GATE_REPORT_before_resume.md'
    if not archived.exists():archived.write_bytes(oldreport.read_bytes())
    dependency=json.loads((DEP/'dependency_audit.json').read_text())
    counts=pathways.groupby('collection').agg(tests=('P','size'),significant_associations=('FDR',lambda v:int((v<.05).sum()))).reset_index()
    text=f'''# PHASE A GATE REPORT

Status: A1 and A3 COMPLETE. A2 retained without rerunning. {conclusion}. {decision}. Phase B not started.

## Dependencies and validation

Broad matrix: {dependency['shape'][0]} observations x {dependency['shape'][1]} genes, exactly the frozen donor x tissue x original-cell-type observations (24 donors, 74 tissues, 161 cell types; n_cells >=20). Mean of existing processed CELLxGENE X, on the same source-normalized scale as the clock features; not counts or CPM and no additional log transform. All {dependency['overlapping_original_genes_verified']} measured overlapping genes reproduced (original absent zero-filled symbols: {dependency['original_absent_symbols_exported_as_zero']}), maximum absolute error {dependency['all_69_original_gene_max_error']:.3g}. Upstream X normalization algorithm is not established by these exports. Remote generation used existing processed H5AD and wrote only new Phase A files. No original remote result was modified.

No pre-existing GMT files were found. Complete MSigDB 2026.1.Hs Hallmark and Reactome (Reactome release 95) GMT files were downloaded from Broad Institute. Version, URL and SHA256 are in dependencies/pathway_provenance.json. Reactome here means the complete MSigDB Reactome subcollection (its upstream redundancy filtering is retained).

## A1 matched null controls

1,000 panels for core8 and 1,000 for expanded17; sample without replacement within panels from each clock gene's 100 nearest non-clock genes in standardized log10(mean), log10(variance) space. All expanded clock genes excluded from the control universe. Matching uses 2,030 observations without lineage labels. Matching diagnostics and all panel memberships saved. No outcome-based tuning. Empirical one-sided P=(1+null >= observed)/1001; descriptive conditional gene-panel reference, not a randomized biological experiment. Eight-metric BH FDR also saved.

Original five donor-disjoint folds, train-only standardization, balanced Ridge alpha=1; pooled held-out balanced accuracy and macro F1. Silhouette uses the frozen all-observation scaling convention. Distance ratio uses the same 97 adequately supported lineage/tissue centroids, 928 within-lineage and 139 between-lineage pairs. Original observed metrics numerically reproduced.

{show(summary)}
## A2 retained findings

Original 24-feature median between-lineage distance 3.9895 vs within-lineage 2.5436; delta 1.4459, approximate crossed donor/tissue 95% interval 0.80523–1.6951. Fibroblast median tissue distance 3.7516, endothelial 3.0475, macrophage/monocyte 2.3576, T cell 2.3052, B cell 1.8593. A2 data and bootstrap files unchanged; full prior report archived in 00_audit/PHASE_A_GATE_REPORT_before_resume.md. These are static associations, not rhythmicity, amplitude or phase.

## A3 complete screens

Pathway score = equal-weight mean of per-gene z scores across original 2,030 observations; all 17 expanded clock genes excluded. All sets inventoried; >=5 measured nonconstant non-clock genes required. No set selected based on anticipated biology. Equal-pseudobulk aggregation into the existing 417 donor/tissue/lineage phenotype observations, from 936 supported original observations. Existing donor-excluded clock-remodeling distances are unchanged.

Within each lineage, standardized remodeling distance is regressed on standardized pathway score with donor fixed effects and log1p(n_cells). Approximate sandwich variance is max(donor-cluster, tissue-cluster, donor+tissue-intersection cluster variance); t reference df=min(donors,tissues)-1. BH correction covers all pathway x five-lineage tests separately for Hallmark primary and Reactome secondary, with untestable sets assigned P=1. Signed standardized beta, approximate interval, P/FDR, coverage, donor-deletion sign fraction, and donor+tissue-adjusted sensitivity saved. Small cluster counts and shared phenotype references make inference approximate. Fine-cell-type mixture and cross-sectional confounding remain unresolved. A score association does not establish causal remodeling.

{show(counts)}
Prespecified programs emerge independently of targeted pathway selection if collection-wide FDR<0.05 in any lineage. This does not mean independence from other correlated pathways or independent-cohort replication.

{show(hypotheses)}
Top Hallmark associations:
{show(pathways[pathways.collection.eq('Hallmark')].sort_values('P')[['pathway','lineage','beta_standardized','P','FDR','donor_deletion_sign_fraction']].head(10))}
## Gate interpretation

{conclusion}. Static lineage-associated clock organization and tissue variation remain supported internally. Expanded-clock superiority to matched panels is {'supported on multiple metrics' if specificity else 'not supported by the required classification-plus-multiple-metric criterion'}. Generic cell-identity transcription remains a plausible explanation unless clock specificity clears the matched-control gate. Pathway findings add observational context and cannot override a failed specificity gate.

{decision}. Gate requires expanded-clock one-sided P<0.05 with positive excess for balanced accuracy and at least one geometric metric (silhouette or distance ratio; macro F1 supports classification but is not a separate geometric criterion), plus an FDR-supported unbiased pathway association. This operational completion rule was written before computing the resumed results; it implements the prior report's multiple-metric superiority requirement. Core is secondary and cannot replace a failed expanded primary panel. No Phase B analysis performed.

Reproduce only resumed components with scripts/phaseA_resume.py (A2 is never rerun). See 00_audit/resume_protocol.json, resume_validation.json, dependencies/dependency_audit.json and input hashes. Numerical results derive from saved scripts.
'''
    oldreport.write_text(text,encoding='utf-8'); (ROOT/'reports').mkdir(exist_ok=True);(ROOT/'reports/PHASE_A_GATE_REPORT.md').write_text(text,encoding='utf-8')
    write('04_integrated_model/resumed_gate_decision.json',{'conclusion':conclusion,'decision':decision,'expanded_metrics_above_null':wins.metric.tolist(),'significant_pathway_associations':int(pathways.FDR.lt(.05).sum()),'remote_generated':True,'phase_B':'NOT STARTED'})
    evidence=pd.DataFrame([{'component':'Lineage-associated clock architecture','status':'SUPPORTED DESCRIPTIVELY','evidence':'A2 retained: distance delta 1.4459; interval 0.80523 to 1.6951'}, {'component':'Tissue-associated remodeling','status':'SUPPORTED DESCRIPTIVELY','evidence':'A2 retained; lineage-dependent static tissue distances'}, {'component':'Clock specificity','status':'SUPPORTED' if specificity else 'NOT SUPPORTED BY MATCHED CONTROLS','evidence':f'Expanded clock significant positive metrics: {len(wins)}/4'}, {'component':'Pathway context','status':'ASSOCIATED' if assoc else 'NO FDR-SUPPORTED ASSOCIATION','evidence':f'{int(pathways.FDR.lt(.05).sum())} collection-wide FDR-significant lineage/pathway associations'}])
    csv('04_integrated_model/evidence_summary.csv',evidence);csv('source_data/FigA8_integrated_evidence.csv',evidence)
    fig,axes=plt.subplots(2,2,figsize=(12,6))
    import textwrap
    for ax,r in zip(axes.flat,evidence.itertuples()):
        ax.axis('off');ax.text(0,.9,r.component,weight='bold',fontsize=12);ax.text(0,.65,r.status,fontsize=10);ax.text(0,.4,textwrap.fill(r.evidence,52),fontsize=10,va='top')
    fig.suptitle(conclusion+' | '+decision,fontsize=12);fig.tight_layout();save(fig,'FigA8_integrated_clock_architecture_model')
    csv('source_data/figure_manifest.csv',pd.DataFrame([{'figure':f'FigA{i}','status':'GENERATED','reason':'A2 retained unchanged' if i in [3,4,5] else 'A1 observed confusion retained; current null results in FigA1' if i==2 else 'Resumed A1/A3 and integrated gate'} for i in range(1,9)]))
    csv('source_data/FigA1_empirical_null_summary.csv',summary)
    csv('source_data/FigA6_hallmark_associations.csv',pathways[pathways.collection.eq('Hallmark')])
    csv('source_data/FigA7_prespecified_programs.csv',pathways[pathways.pathway.isin(hypotheses.pathway)])
    csv('statistics/blocked_analyses.csv',pd.DataFrame(columns=['analysis','status','reason','n_completed']))
    return conclusion,decision

def main():
    before=frozen(); meta,x,genes=load();protocol=json.loads((OUT/'00_audit/prespecified_protocol.json').read_text())
    write('00_audit/resume_protocol.json',{'seed':SEED,'A2':'READ ONLY, never rerun','matching':'100 nearest in standardized log10 mean and variance; exclude expanded17; no within-panel duplicates; 1000 panels each','pathway_score':'mean standardized genes excluding expanded17; >=5 variable covered genes','association':'standardized distance ~ standardized pathway + donor FE + log1p(n_cells); conservative donor/tissue clustered SE, small-cluster t; BH all pathways x five lineages per collection','sensitivity':'donor+tissue FE and leave-one-donor sign consistency','gate':'expanded primary BA P<.05 and silhouette or distance ratio positive P<.05 plus >=1 pathway association FDR<.05','timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    if '--pathways-only' not in sys.argv: summary=a1(meta,x,genes,protocol)
    else: summary=pd.read_csv(OUT/'01_matched_random_controls/empirical_null_summary.csv')
    pathways,hypotheses,definitions=a3(meta,x,genes,protocol);plots(summary,pathways);decision=report(summary,pathways,hypotheses,definitions)
    assert before==frozen()
    write('00_audit/resume_validation.json',{'A2_hashes_unchanged':before,'metadata_exact_match':True,'original_observed_metrics_reproduced':True,'null_panels_per_panel':1000,'pathway_sets':definitions.groupby('collection').size().to_dict(),'dependency_sha256':{p.name:sha(p) for p in DEP.iterdir() if p.is_file()},'script_sha256':sha(Path(__file__)),'phase_B':'NOT STARTED'})
    print(summary.to_string(index=False));print(hypotheses.to_string(index=False));print(decision,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
