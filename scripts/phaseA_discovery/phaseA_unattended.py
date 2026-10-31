"""Local-only completion of frozen Phase A; resumable A1/A3 without rerunning A2."""
from __future__ import annotations
import argparse, json, sys, os, time, hashlib, itertools, importlib.metadata, shutil
from pathlib import Path
from phaseA_resume import ROOT, OUT, DEP, METRICS, associations, bh, frozen, np, pd, t, StandardScaler, RidgeClassifier, balanced_accuracy_score, f1_score, silhouette_score, threadpool_limits
from publication_style import LINEAGE_ORDER, LINEAGE_COLORS, DISPLAY_LABELS, FEATURE_COLORS, apply_publication_style, clean_axis, save_figure, plt
from scipy.stats import false_discovery_control
from PIL import Image

AUD=OUT/'00_audit/unattended'; A1=OUT/'01_matched_random_controls/unattended'; A3=OUT/'03_unbiased_pathways/unattended'; ST=OUT/'statistics/unattended'; SRC=OUT/'source_data/unattended'; REP=OUT/'reports'; FIG=OUT/'figures'
SEED=20260913
SETS={'Core clock':'CORE_FEATURE_COLS','Expanded clock':'EXPANDED_FEATURE_COLS'}
VARIANTS=[('primary',SEED,100,2),('second_seed',SEED+1,100,2),('tighter_matching',SEED,25,2),('expression_only',SEED,100,1)]
HYPOTHESES={'Hypoxia':'HALLMARK_HYPOXIA','Fatty acid metabolism':'HALLMARK_FATTY_ACID_METABOLISM','IFNG':'HALLMARK_INTERFERON_GAMMA_RESPONSE','TNF-NFkB':'HALLMARK_TNFA_SIGNALING_VIA_NFKB','ECM':'REACTOME_EXTRACELLULAR_MATRIX_ORGANIZATION'}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def jwrite(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8');os.replace(temp,path)

def cwrite(path,df):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
    df.to_csv(temp,index=False,float_format='%.17g');os.replace(temp,path)

def protocol():return json.loads((OUT/'00_audit/prespecified_protocol.json').read_text(encoding='utf-8-sig'))

def load_data():
    data=np.load(DEP/'broad_normalized_mean.npz',allow_pickle=False)
    meta=pd.read_csv(OUT/'00_audit/observation_metadata.csv').set_index('group_id')
    assert not meta.index.duplicated().any()
    ids=data['group_id']; genes=data['genes']; x=data['expression']
    assert x.shape==(2030,59410) and len(ids)==2030 and len(genes)==59410
    assert len(set(ids))==2030 and len(set(genes))==59410
    order=pd.Index(ids).get_indexer(meta.index);assert (order>=0).all()
    if not np.array_equal(order,np.arange(len(order))):x=x[order]
    assert np.isfinite(x).all()
    return meta,x,genes

def parse_gmt(path):
    result=[];names=set()
    for line in path.read_text(encoding='utf-8').splitlines():
        name,url,*members=line.split('\t');assert name and members and name not in names;names.add(name)
        result.append((name,url,sorted({'ARNTL' if g=='BMAL1' else g for g in members})))
    return result

def preflight():
    for p in [AUD,A1,A3,ST,SRC,FIG]:p.mkdir(parents=True,exist_ok=True)
    snapshot=AUD/'initial_frozen_hashes.json'
    if not snapshot.exists():
        # Entire pre-existing result tree plus source figures and scripts: immutable except designated report updates.
        paths=[p for base in [OUT,ROOT/'figures_revised'] for p in base.rglob('*') if p.is_file() and 'unattended' not in p.parts]
        paths+=[p for p in (ROOT/'scripts').glob('*.py') if 'unattended' not in p.name]
        originals={str(p):sha(p) for p in paths}
        jwrite(snapshot,originals)
        backup=AUD/'previous_reports';backup.mkdir(exist_ok=True)
        for p in REP.glob('*.md'):shutil.copy2(p,backup/p.name)
    meta,x,genes=load_data(); dm=pd.read_csv(DEP/'observation_metadata.tsv',sep='\t').set_index('group_id');gm=pd.read_csv(DEP/'gene_metadata.tsv',sep='\t')
    assert len(dm)==2030 and not dm.index.duplicated().any() and len(gm)==59410 and not gm.gene.duplicated().any()
    assert np.array_equal(gm.gene,genes)
    for col in ['donor_id','tissue','cell_type','n_cells']:assert np.array_equal(dm.loc[meta.index,col],meta[col])
    original=pd.read_csv(ROOT.parent/'TabulaCircadian_deliverables/source_data/pseudobulk__clock_architecture_features.tsv',sep='\t').set_index('group_id').loc[meta.index]
    index={g:i for i,g in enumerate(genes)};errors=[]
    for col in original:
        if not col.startswith('expr_'):continue
        g=col[5:]
        if g not in index:assert g=='CTGF' and original[col].eq(0).all();continue
        errors.append(float(np.abs(x[:,index[g]]-original[col].to_numpy()).max()))
    assert len(errors)==68 and max(errors)<1e-10
    collections={p.name:len(parse_gmt(p)) for p in DEP.glob('*.gmt')};assert collections['h.all.v2026.1.Hs.symbols.gmt']==50
    dep_hash={p.name:sha(p) for p in DEP.iterdir() if p.is_file()}
    jwrite(AUD/'dependency_integrity.json',{'shape':x.shape,'finite':True,'unique_observations':len(meta),'unique_genes':len(genes),'original_gene_overlap':68,'max_discrepancy':max(errors),'absent_zero_gene':'CTGF','gmt_sets':collections,'dependency_sha256':dep_hash,'A2_frozen':frozen()})
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','threadpoolctl']}
    jwrite(AUD/'environment.json',{'executable':sys.executable,'python':sys.version,'versions':versions,'threads':1,'packages_installed':False,'remote_access':False})
    spec={'frozen_panel_definitions':protocol()['feature_sets'],'seed':SEED,'variants':VARIANTS,'panels_per_variant_per_clock_panel':1000,
      'matching':'Frozen resume method: mean>0, variance>1e-12; exclude ALL expanded17 genes; standardized log10 mean/variance over nonclock eligible universe; stable nearest100 pool per clock gene; uniform without within-panel duplicates. No outcome-based choices.',
      'matching_QC_rule':'Flag primary mean absolute standardized log-mean OR log-variance mismatch >0.10, or pooled95th absolute mismatch >0.25. Run nearest25 tighter sensitivity regardless, never replace primary.',
      'detection':'Not consistently supplied in broad dependency; not matched and not reconstructed.',
      'CV':'Exact frozen donor-disjoint5-fold manifest; fixed RidgeClassifier alpha1,class_weight balanced; train-only StandardScaler; all values finite. Original2030 global StandardScaler for silhouette and distances.',
      'deletion':'Delete each clock gene; delete each of29 sufficiently supported tissues or each24donors from all observations. Reuse surviving original folds; reapply original>=3donor support for distance centroids; scaling fitted to remaining eligible observations. Donor labels never overlap folds.',
      'size_control':'For each primary random expanded17 panel select8 genes uniformly using seed+31 and paired17-vs8 metrics; same eligible universe, not rematched to core targets. Also1000 uniform8-of17 real expanded subsets. Composition and size effects are not fully separable.',
      'pathway_scoring':'Frozen resume: mean across globally standardized genes; exclude expanded17; variance>1e-12; at least5 retained genes; no coverage-ratio exclusion. BMAL1->ARNTL mapping. Score original pseudobulks then average same donor/tissue/lineage as frozen phenotype.',
      'association':'Reuse phaseA_resume.associations: standardized distance ~ standardized pathway + donor fixed effects + log1p(n_cells); FWL OLS; max of donor, tissue, and donor+tissue-intersection cluster sandwich variance; t df=min(D,T)-1; fixed reference phenotype. Secondary donor+tissue FE.',
      'FDR':'BH across ALL collection pathway x five-lineage tests including ineligible P=1; Hallmark primary, Reactome secondary. No cross-collection pooled correction.',
      'pathway_sensitivity':'All tested pathways: each donor/tissue deletion; report beta range, max change, sign consistency and driving block. Descriptive influence diagnostics, not post-selection confirmatory inference.',
      'prioritization':'FDR<0.05 first, then absolute standardized effect descending, then P/name; unsupported top entries explicitly marked. Reactome visualization only greedily removes Jaccard>0.5 overlap after complete testing.',
      'hypothesis_status':'SUPPORTED if >=1 FDR<0.05 association with donor and tissue sign-consistency>=0.8 and coverage>=0.2 and>=10genes; PARTIAL if any FDR<0.05 but no such robust association; otherwise NOT SUPPORTED (not proof of absence).',
      'gate':'Original user gate retained: at least one justified panel upper-tailP<.05 on at least2 of BA/silhouette/ratio; require second-seed confirmation, frozen reproducibleA2, no single-driver reversal. Pathways qualify program claims separately. Primary8-test BH also reported; GO requires corresponding BH significance for at least2 primary metrics.',
      'no_temporal_or_causal_claims':True}
    if (AUD/'completion_protocol.json').exists():assert json.loads((AUD/'completion_protocol.json').read_text())==json.loads(json.dumps(spec,default=str))
    else:jwrite(AUD/'completion_protocol.json',spec)
    print('Preflight PASS:',x.shape,'GMT',collections,'max discrepancy',max(errors),flush=True)

class Evaluator:
    def __init__(self,meta,x,mask=None):
        self.meta=meta if mask is None else meta.loc[mask]
        self.x=x if mask is None else x[np.asarray(mask)]
        cm=self.meta[self.meta.priority_lineage.notna()];self.ii=self.meta.index.get_indexer(cm.index);self.labels=cm.priority_lineage.to_numpy()
        split=pd.read_csv(OUT/'01_matched_random_controls/classification_split_manifest.csv').set_index('group_id').loc[cm.index]
        self.folds=[]
        for f in sorted(split.test_fold.unique()):
            tr=np.flatnonzero(split.test_fold.to_numpy()!=f);te=np.flatnonzero(split.test_fold.to_numpy()==f)
            assert not set(cm.iloc[tr].donor_id)&set(cm.iloc[te].donor_id)
            assert set(cm.iloc[tr].priority_lineage)==set(LINEAGE_ORDER)
            self.folds.append((tr,te))
        keys=[];groups=[]
        for key,part in cm.groupby(['priority_lineage','tissue'],sort=True):
            if part.donor_id.nunique()>=3:keys.append(key);groups.append(self.meta.index.get_indexer(part.index))
        self.groups=groups;self.w=[];self.b=[]
        for a,b in itertools.combinations(range(len(keys)),2):
            if keys[a][0]==keys[b][0]:self.w.append((a,b))
            elif keys[a][1]==keys[b][1]:self.b.append((a,b))
        self.w=np.array(self.w);self.b=np.array(self.b)

    def evaluate(self,ids):
        values=self.x[:,ids];z=StandardScaler().fit_transform(values);xx=values[self.ii];pred=np.empty(len(xx),object)
        for tr,te in self.folds:
            scaler=StandardScaler();train=scaler.fit_transform(xx[tr]);test=scaler.transform(xx[te])
            pred[te]=RidgeClassifier(alpha=1,class_weight='balanced').fit(train,self.labels[tr]).predict(test)
        cent=np.array([z[g].mean(0) for g in self.groups]);w=self.w;b=self.b
        return dict(balanced_accuracy=balanced_accuracy_score(self.labels,pred),macro_f1=f1_score(self.labels,pred,average='macro'),silhouette=silhouette_score(z[self.ii],self.labels),distance_ratio=float(np.median(np.linalg.norm(cent[b[:,0]]-cent[b[:,1]],axis=1))/np.median(np.linalg.norm(cent[w[:,0]]-cent[w[:,1]],axis=1))))

def null_summary(observed,null):
    rows=[]
    for panel in SETS:
        for variant,part in null[null.panel.eq(panel)].groupby('variant',sort=False):
            for m in METRICS:
                v=part[m].to_numpy();value=float(observed.loc[panel,m]);sd=v.std(ddof=1)
                rows.append({'panel':panel,'variant':variant,'metric':m,'observed':value,'n_null':len(v),'null_mean':v.mean(),'null_median':np.median(v),'null_sd':sd,'null_q025':np.quantile(v,.025),'null_q975':np.quantile(v,.975),'percentile':100*(np.sum(v<value)+.5*np.sum(v==value))/len(v),'P':(1+np.sum(v>=value))/(len(v)+1),'Z':(value-v.mean())/sd if sd else np.nan})
    result=pd.DataFrame(rows);result['BH_8_metrics']=result.groupby('variant',sort=False).P.transform(lambda v:bh(v.to_numpy()))
    return result

def checkpoint_metrics(path,ev,selections,extra):
    if path.exists():
        old=pd.read_csv(path);assert old.iteration.tolist()==list(range(1,len(old)+1));rows=old.to_dict('records')
    else:rows=[]
    for i in range(len(rows),len(selections)):
        rows.append({**extra,'iteration':i+1,**ev.evaluate(selections[i])})
        if (i+1)%100==0 or i==len(selections)-1:
            cwrite(path,pd.DataFrame(rows));print(path.stem,i+1,'/',len(selections),flush=True)
    return pd.DataFrame(rows)

def matched_controls():
    meta,x,genes=load_data();ev=Evaluator(meta,x);pr=protocol();index={g:i for i,g in enumerate(genes)}
    means=x.mean(0);variances=x.var(0);clock=[s[5:] for s in pr['feature_sets']['EXPANDED_FEATURE_COLS']]
    eligible=(means>0)&(variances>1e-12)&~np.isin(genes,clock);universe=np.flatnonzero(eligible)
    log=np.column_stack([np.log10(np.maximum(means,1e-300)),np.log10(np.maximum(variances,1e-300))]);stats=(log-log[eligible].mean(0))/log[eligible].std(0)
    cwrite(A1/'gene_matching_statistics.csv',pd.DataFrame({'gene':genes,'mean':means,'variance':variances,'standardized_log_mean':stats[:,0],'standardized_log_variance':stats[:,1],'eligible_nonclock':eligible}))
    observed=pd.read_csv(OUT/'01_matched_random_controls/observed_metrics.csv').set_index('panel')
    for panel,key in SETS.items():
        real=ev.evaluate([index[g[5:]] for g in pr['feature_sets'][key]])
        for m in METRICS:np.testing.assert_allclose(real[m],observed.loc[panel,m],rtol=1e-9,atol=1e-10)
    allnull=[];qc=[];poolsrows=[]
    for variant,seed,k,dim in VARIANTS:
        pools={g:universe[np.argsort(((stats[universe,:dim]-stats[index[g],:dim])**2).sum(1),kind='stable')[:k]] for g in clock}
        poolsrows.extend({'variant':variant,'clock_gene':g,'matched_gene':genes[j],'rank':rank+1,'distance':float(np.linalg.norm(stats[j,:dim]-stats[index[g],:dim]))} for g,ids in pools.items() for rank,j in enumerate(ids))
        rng=np.random.default_rng(seed)
        for panel,key in SETS.items():
            targets=[v[5:] for v in pr['feature_sets'][key]];ids=[index[g] for g in targets];selected=[];panelrows=[]
            for iteration in range(1000):
                choice=[]
                for g in targets:
                    pool=pools[g];j=int(rng.choice(pool[~np.isin(pool,choice)]));choice.append(j)
                    panelrows.append({'panel':panel,'variant':variant,'iteration':iteration+1,'clock_gene':g,'matched_gene':genes[j]})
                selected.append(choice);delta=stats[choice]-stats[ids]
                qc.append({'panel':panel,'variant':variant,'iteration':iteration+1,'mean_abs_logmean_z_difference':np.abs(delta[:,0]).mean(),'mean_abs_logvariance_z_difference':np.abs(delta[:,1]).mean(),'max_distance':np.linalg.norm(delta,axis=1).max(),'panel_logmean_z':stats[choice,0].mean(),'panel_logvariance_z':stats[choice,1].mean(),'target_logmean_z':stats[ids,0].mean(),'target_logvariance_z':stats[ids,1].mean()})
            slug='core' if panel=='Core clock' else 'expanded';p=A1/f'{variant}_{slug}_genes.csv'
            if p.exists():assert pd.read_csv(p).equals(pd.DataFrame(panelrows))
            else:cwrite(p,pd.DataFrame(panelrows))
            allnull.append(checkpoint_metrics(A1/f'{variant}_{slug}_checkpoint.csv',ev,selected,{'panel':panel,'variant':variant,'seed':seed}))
    null=pd.concat(allnull,ignore_index=True);cwrite(A1/'empirical_null_distributions.csv',null);cwrite(A1/'empirical_null_summary.csv',null_summary(observed,null));cwrite(A1/'matching_quality.csv',pd.DataFrame(qc));cwrite(A1/'candidate_pools.csv',pd.DataFrame(poolsrows))
    # Size-only paired sensitivity: keep same expanded random gene universe.
    panels=pd.read_csv(A1/'primary_expanded_genes.csv');rng=np.random.default_rng(SEED+31)
    selections=[];sizegenes=[];real=[index[v[5:]] for v in pr['feature_sets']['EXPANDED_FEATURE_COLS']];real_selections=[]
    for iteration,part in panels.groupby('iteration',sort=True):
        ids=[index[g] for g in part.matched_gene];s=rng.choice(ids,8,replace=False);selections.append(s)
        sizegenes.extend({'iteration':iteration,'kind':'random_expanded_pool_8','gene':genes[j]} for j in s)
        s=rng.choice(real,8,replace=False);real_selections.append(s);sizegenes.extend({'iteration':iteration,'kind':'real_expanded_8','gene':genes[j]} for j in s)
    cwrite(A1/'size_control_genes.csv',pd.DataFrame(sizegenes))
    checkpoint_metrics(A1/'size_random8_checkpoint.csv',ev,selections,{'panel':'Random expanded pool 8','variant':'size_control'})
    checkpoint_metrics(A1/'size_real8_checkpoint.csv',ev,real_selections,{'panel':'Real expanded subset 8','variant':'size_control'})
    print('A1 matched controls complete.',flush=True)

def a1_robustness():
    meta,x,genes=load_data();index={g:i for i,g in enumerate(genes)};ev=Evaluator(meta,x);pr=protocol();rows=[]
    for panel,key in SETS.items():
        targets=[v[5:] for v in pr['feature_sets'][key]]
        for omitted in targets:rows.append({'panel':panel,'deletion':'gene','identifier':omitted,'n_features':len(targets)-1,**ev.evaluate([index[g] for g in targets if g!=omitted])})
    cwrite(A1/'leave_one_gene_out.csv',pd.DataFrame(rows))
    support=meta[meta.priority_lineage.notna()].groupby(['priority_lineage','tissue']).donor_id.nunique();tissues=sorted(set(support[support.ge(3)].index.get_level_values(1)))
    path=A1/'donor_tissue_deletions.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else [];done={(r['deletion'],r['identifier'],r['panel']) for r in rows}
    for kind,ids,col in [('tissue',tissues,'tissue'),('donor',sorted(meta.donor_id.unique()),'donor_id')]:
        for omitted in ids:
            if all((kind,omitted,p) in done for p in SETS):continue
            mask=meta[col].ne(omitted);part=Evaluator(meta,x,mask)
            for panel,key in SETS.items():
                if (kind,omitted,panel) in done:continue
                rows.append({'panel':panel,'deletion':kind,'identifier':omitted,'n_observations':len(part.ii),'n_supported_groups':len(part.groups),'n_within_pairs':len(part.w),'n_between_pairs':len(part.b),**part.evaluate([index[g[5:]] for g in pr['feature_sets'][key]])})
            cwrite(path,pd.DataFrame(rows))
    print('A1 gene/donor/tissue deletions complete.',flush=True)

def delete_effects(scores,meta,y,column,point):
    blocks=sorted(meta[column].unique());effects=[]
    for block in blocks:
        keep=meta[column].ne(block).to_numpy();effects.append(associations(scores[keep],meta.loc[keep],y[keep])[0])
    effects=np.array(effects);change=np.abs(effects-point);finite=np.where(np.isfinite(change),change,-np.inf);worst=np.argmax(finite,axis=0)
    return effects,np.array(blocks)[worst]

def pathways():
    meta,x,genes=load_data();pr=protocol();clock={v[5:] for v in pr['feature_sets']['EXPANDED_FEATURE_COLS']};index={g:i for i,g in enumerate(genes)}
    mean=x.mean(0);sd=x.std(0);variable=sd**2>1e-12
    phenotype=pd.read_csv(OUT/'03_unbiased_pathways/clock_remodeling_phenotype_donor_tissue_lineage.csv')
    groups=meta.reset_index().groupby(['priority_lineage','donor_id','tissue'],sort=False).indices
    agg=[groups[(r.lineage,r.donor_id,r.tissue)] for r in phenotype.itertuples()]
    # Linear averaging commutes with gene standardization; reduces memory while preserving exact scoring definition.
    broad=np.array([x[g].mean(0) for g in agg]);z=(broad-mean)/np.where(sd>0,sd,1)
    del x,broad
    allresults=[];defs=[]
    for collection,filename in [('Hallmark','h.all.v2026.1.Hs.symbols.gmt'),('Reactome','c2.cp.reactome.v2026.1.Hs.symbols.gmt')]:
        definitions=[];columns=[]
        for name,url,members in parse_gmt(DEP/filename):
            available=sorted(g for g in set(members)-clock if g in index and variable[index[g]])
            ids=[index[g] for g in available];eligible=len(ids)>=5
            definitions.append({'collection':collection,'pathway':name,'n_original':len(members),'n_clock_excluded':len(set(members)&clock),'n_scored':len(ids),'coverage':len(ids)/len(members),'status':'TESTED' if eligible else 'INSUFFICIENT_COVERAGE','small_or_poorly_covered':len(ids)<10 or len(ids)/len(members)<.2,'genes_scored':';'.join(available),'source':url})
            columns.append(z[:,ids].mean(1) if eligible else np.zeros(len(phenotype)))
        definitions=pd.DataFrame(definitions);scores=np.column_stack(columns);defs.append(definitions)
        cwrite(A3/f'{collection.lower()}_coverage.csv',definitions)
        scorepath=A3/f'{collection.lower()}_scores.npz'
        if not scorepath.exists():np.savez_compressed(scorepath,scores=scores,pathways=definitions.pathway.to_numpy(str),lineage=phenotype.lineage.to_numpy(str),donor_id=phenotype.donor_id.to_numpy(str),tissue=phenotype.tissue.to_numpy(str))
        results=[]
        for lin in LINEAGE_ORDER:
            path=A3/f'{collection.lower()}_{lin}_checkpoint.csv'
            if path.exists():results.append(pd.read_csv(path));continue
            part=phenotype[phenotype.lineage.eq(lin)];sc=scores[part.index];y=part.distance.to_numpy()
            beta,se,p,df,rank,*_=associations(sc,part,y);sb,sse,sp,*_=associations(sc,part,y,True)
            dd,dworst=delete_effects(sc,part,y,'donor_id',beta);td,tworst=delete_effects(sc,part,y,'tissue',beta)
            np.savez_compressed(A3/f'{collection.lower()}_{lin}_deletion_effects.npz',donor_beta=dd,tissue_beta=td,donors=np.array(sorted(part.donor_id.unique()),str),tissues=np.array(sorted(part.tissue.unique()),str),pathways=definitions.pathway.to_numpy(str))
            frame=definitions.drop(columns=['genes_scored','source']).copy();frame['lineage']=lin;frame['n_observations']=len(part);frame['n_donors']=part.donor_id.nunique();frame['n_tissues']=part.tissue.nunique();frame['nuisance_rank']=rank;frame['cluster_df']=df
            frame['beta']=beta;frame['SE']=se;frame['ci_low']=beta-t.ppf(.975,df)*se;frame['ci_high']=beta+t.ppf(.975,df)*se;frame['P']=p;frame['tissue_adjusted_beta']=sb;frame['tissue_adjusted_P']=sp
            for name,arr,worst in [('donor',dd,dworst),('tissue',td,tworst)]:
                frame[name+'_sign_fraction']=np.mean(np.sign(arr)==np.sign(beta),axis=0)
                frame[name+'_beta_min']=np.min(arr,axis=0);frame[name+'_beta_max']=np.max(arr,axis=0);frame[name+'_max_abs_change']=np.max(np.abs(arr-beta),axis=0);frame[name+'_worst_block']=worst
                frame[name+'_any_sign_reversal']=np.any(np.sign(arr)!=np.sign(beta),axis=0)
            cwrite(path,frame);results.append(frame);print(collection,lin,'checkpoint',len(frame),flush=True)
        result=pd.concat(results,ignore_index=True);result['FDR']=bh(result.P.fillna(1).to_numpy());result['tissue_adjusted_FDR']=bh(result.tissue_adjusted_P.fillna(1).to_numpy())
        result['abs_beta']=result.beta.abs();result['significant']=result.FDR.lt(.05)
        result=result.sort_values(['lineage','significant','abs_beta','P','pathway'],ascending=[True,False,False,True,True]);result['effect_priority_rank']=result.groupby('lineage').cumcount()+1
        result['P_rank_within_lineage']=result.groupby('lineage').P.rank(method='min');result['P_rank_collection']=result.P.rank(method='min')
        cwrite(A3/f'{collection.lower()}_screen.csv',result);allresults.append(result)
    result=pd.concat(allresults,ignore_index=True);cwrite(A3/'all_pathway_associations.csv',result);definitions=pd.concat(defs,ignore_index=True);cwrite(A3/'pathway_coverage.csv',definitions)
    cwrite(A3/'top_pathways_per_lineage.csv',result.groupby(['collection','lineage'],sort=False).head(10))
    consistency=[]
    for (collection,path),part in result.groupby(['collection','pathway'],sort=True):
        v=part.beta.to_numpy();finite=np.isfinite(v);sig=part[part.FDR.lt(.05)];signs=np.sign(sig.beta)
        kind='opposite significant effects' if (signs>0).any() and (signs<0).any() else 'same-sign multi-lineage' if len(sig)>=2 else 'one significant lineage' if len(sig)==1 else 'no FDR-supported lineage'
        consistency.append({'collection':collection,'pathway':path,'n_significant_lineages':len(sig),'significant_lineages':';'.join(sig.lineage),'positive_lineages':int((v>0).sum()),'negative_lineages':int((v<0).sum()),'beta_min':np.min(v[finite]) if finite.any() else np.nan,'beta_max':np.max(v[finite]) if finite.any() else np.nan,'classification':kind,'min_FDR':part.FDR.min(),'max_abs_beta':part.beta.abs().max()})
    cwrite(A3/'cross_lineage_consistency.csv',pd.DataFrame(consistency))
    hypotheses=[]
    for label,name in HYPOTHESES.items():
        part=result[result.pathway.eq(name)].copy();assert len(part)==5
        robust=part.FDR.lt(.05)&part.donor_sign_fraction.ge(.8)&part.tissue_sign_fraction.ge(.8)&part.coverage.ge(.2)&part.n_scored.ge(10)
        status='SUPPORTED' if robust.any() else 'PARTIAL' if part.FDR.lt(.05).any() else 'NOT SUPPORTED'
        part['hypothesis']=label;part['hypothesis_status']=status;hypotheses.append(part)
    cwrite(A3/'hypothesized_programs.csv',pd.concat(hypotheses,ignore_index=True))
    print('Complete Hallmark/Reactome screens and deletion sensitivities finished.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','null','robustness','pathways','finish']);args=parser.parse_args()
    with threadpool_limits(limits=1):
        {'preflight':preflight,'null':matched_controls,'robustness':a1_robustness,'pathways':pathways,'finish':finish}[args.stage]()

def load_results():
    return (pd.read_csv(A1/'empirical_null_summary.csv'),pd.read_csv(A1/'empirical_null_distributions.csv'),pd.read_csv(A3/'all_pathway_associations.csv'),pd.read_csv(A3/'pathway_coverage.csv'))

def choose_pathways(results,definitions,collection,n=24,nonredundant=False):
    part=results[results.collection.eq(collection)].copy()
    ranking=part.sort_values(['significant','abs_beta','P','pathway'],ascending=[False,False,True,True]).drop_duplicates('pathway')
    sets={r.pathway:set(str(r.genes_scored).split(';')) for r in definitions.itertuples()}
    selected=[];records=[]
    for r in ranking.itertuples():
        overlap=max((len(sets[r.pathway]&sets[p])/max(1,len(sets[r.pathway]|sets[p])) for p in selected),default=0.)
        include=len(selected)<n and (not nonredundant or overlap<=.5)
        records.append({'pathway':r.pathway,'selected':include,'largest_Jaccard_to_already_selected':overlap,'reason':'included' if include else 'overlap>0.5' if nonredundant and overlap>.5 else 'display limit','any_FDR_supported':bool(r.significant)})
        if include:selected.append(r.pathway)
    cwrite(SRC/f'{collection.lower()}_visualization_selection.csv',pd.DataFrame(records))
    return selected

def render_figures(summary,null,results,definitions):
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch
    import textwrap
    apply_publication_style();audits=[]
    cmap=LinearSegmentedColormap.from_list('clock_signed',['#4E79A7','#FFFFFF','#E15759'])
    colors={'Core clock':FEATURE_COLORS['core_clock'],'Expanded clock':FEATURE_COLORS['expanded_clock']}
    short={'balanced_accuracy':'Balanced accuracy','macro_f1':'Macro F1','silhouette':'Lineage silhouette','distance_ratio':'Between / within distance ratio'}
    def save(fig,stem,source):
        for ax in fig.axes:ax.grid(False)
        cwrite(SRC/(stem+'.csv'),source);audits.append(save_figure(fig,stem,FIG,dpi=300))
    # Split metric groups rather than crowd eight plots.
    for suffix,metrics in [('classification',['balanced_accuracy','macro_f1']),('geometry',['silhouette','distance_ratio'])]:
        fig,axes=plt.subplots(2,2,figsize=(10,7.5));fig.subplots_adjust(left=.10,right=.96,bottom=.13,top=.84,hspace=.65,wspace=.26)
        for j,m in enumerate(metrics):
            vals=null[null.variant.eq('primary')][m];real=summary[summary.variant.eq('primary')&summary.metric.eq(m)].observed
            lo=min(vals.min(),real.min());hi=max(vals.max(),real.max());pad=(hi-lo)*.06
            for i,panel in enumerate(SETS):
                ax=axes[i,j];v=null[null.variant.eq('primary')&null.panel.eq(panel)][m];r=summary[summary.variant.eq('primary')&summary.panel.eq(panel)&summary.metric.eq(m)].iloc[0]
                ax.hist(v,bins=np.linspace(lo-pad,hi+pad,32),color='#C5CDD4',edgecolor='white',linewidth=.3)
                ax.axvline(r.observed,color=colors[panel],lw=2);clean_axis(ax);ax.set(xlim=(lo-pad,hi+pad),xlabel=short[m],ylabel='Random panels')
                ax.set_title(f'{panel} · observed {r.observed:.3f}\nupper-tail P = {r.P:.4f}',fontsize=11)
        fig.suptitle('Clock panels versus expression–variance matched genes',y=.98,fontsize=13)
        fig.text(.5,.035,'1,000 controls per panel · colored line: observed clock panel · common horizontal scales',ha='center',fontsize=9)
        save(fig,'FigA9_'+suffix,summary[summary.variant.eq('primary')&summary.metric.isin(metrics)])
    qc=pd.read_csv(A1/'matching_quality.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,5.7));fig.subplots_adjust(left=.09,right=.97,bottom=.20,top=.78,wspace=.3)
    for j,(col,target,title) in enumerate([('panel_logmean_z','target_logmean_z','Mean expression'),('panel_logvariance_z','target_logvariance_z','Expression variance')]):
        ax=axes[j];clean_axis(ax)
        for i,panel in enumerate(SETS):
            for k,variant in enumerate(['primary','tighter_matching']):
                p=qc[qc.panel.eq(panel)&qc.variant.eq(variant)];pos=i*3+k
                ax.boxplot(p[col],positions=[pos],widths=.6,showfliers=False,patch_artist=True,boxprops={'facecolor':colors[panel],'alpha':.35},medianprops={'color':'#222222'})
                ax.scatter(pos,p[target].iloc[0],marker='D',c=colors[panel],s=36,zorder=3)
        ax.set_xticks([0,1,3,4],['Core\nnearest100','Core\nnearest25','Expanded\nnearest100','Expanded\nnearest25']);ax.set_title(title);ax.set_ylabel('Panel mean of standardized log10 statistic')
    fig.suptitle('Matched-panel technical quality',y=.97);fig.text(.5,.06,'Box: matched panel distribution · diamond: clock target · nearest25 is prespecified sensitivity',ha='center',fontsize=9)
    save(fig,'FigA10_matching_quality',qc)
    loo=pd.read_csv(A1/'leave_one_gene_out.csv');obs=pd.read_csv(OUT/'01_matched_random_controls/observed_metrics.csv').set_index('panel')
    for panel in SETS:
        part=loo[loo.panel.eq(panel)];fig,axes=plt.subplots(1,2,figsize=(10.5,7));fig.subplots_adjust(left=.13,right=.97,bottom=.15,top=.82,wspace=.30)
        for ax,m in zip(axes,['balanced_accuracy','distance_ratio']):
            ax.scatter(part[m],range(len(part)),color=colors[panel],s=30);ax.axvline(obs.loc[panel,m],color='#555555',ls='--',lw=1)
            ax.set_yticks(range(len(part)),part.identifier);ax.invert_yaxis();ax.set_xlabel(short[m]);clean_axis(ax)
        fig.suptitle(f'{panel}: leave one clock gene out',y=.97);fig.text(.5,.06,'Each row removes the named gene · dashed line: complete panel · frozen donor folds',ha='center',fontsize=9)
        save(fig,'FigA11_'+('core' if panel=='Core clock' else 'expanded'),part)
    deletion=pd.read_csv(A1/'donor_tissue_deletions.csv')
    for suffix,metrics in [('classification',['balanced_accuracy','macro_f1']),('geometry',['silhouette','distance_ratio'])]:
        fig,axes=plt.subplots(1,2,figsize=(10.5,5.8));fig.subplots_adjust(left=.1,right=.97,bottom=.20,top=.80,wspace=.30)
        rng=np.random.default_rng(SEED)
        for ax,m in zip(axes,metrics):
            for i,panel in enumerate(SETS):
                for k,kind in enumerate(['donor','tissue']):
                    v=deletion[deletion.panel.eq(panel)&deletion.deletion.eq(kind)][m];pos=i*3+k
                    ax.scatter(pos+rng.uniform(-.17,.17,len(v)),v,color=colors[panel],s=16,alpha=.7)
                    ax.plot([pos-.28,pos+.28],[obs.loc[panel,m]]*2,color='#333333',lw=1.2)
            ax.set_xticks([0,1,3,4],['Core\ndonor','Core\ntissue','Expanded\ndonor','Expanded\ntissue']);ax.set_ylabel(short[m]);clean_axis(ax)
        fig.suptitle('Sensitivity to removing one donor or one supported tissue',y=.97,fontsize=13)
        fig.text(.5,.055,'Points: deletion results · short bars: complete-panel metrics · 24 donors / 29 tissues',ha='center',fontsize=9)
        save(fig,'FigA12_'+suffix,deletion)

    def heatmap(names,collection,stem,title):
        part=results[results.collection.eq(collection)&results.pathway.isin(names)].copy()
        beta=part.pivot(index='pathway',columns='lineage',values='beta').reindex(index=names,columns=LINEAGE_ORDER)
        fdr=part.pivot(index='pathway',columns='lineage',values='FDR').reindex(index=names,columns=LINEAGE_ORDER)
        height=max(6,len(names)*.26+2.6);fig=plt.figure(figsize=(12,height));ax=fig.add_axes([.52,1.6/height,.37,(height-2.7)/height]);cax=fig.add_axes([.92,1.6/height,.017,(height-2.7)/height])
        limit=max(.05,float(np.nanmax(np.abs(beta.to_numpy()))));im=ax.imshow(beta,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
        for i,j in itertools.product(range(len(names)),range(5)):
            if fdr.iloc[i,j]<.05:ax.text(j,i,'●',ha='center',va='center',fontsize=5,color='#222222')
        labels=[p.replace(collection.upper()+'_','').replace('_',' ').lower().capitalize() for p in names]
        ax.set_yticks(range(len(names)),labels,fontsize=8);ax.set_xticks(range(5),['B cell','T cell','Macrophage/\nmonocyte','Fibroblast','Endothelial'],rotation=35,ha='right',fontsize=9)
        for tick,lin in zip(ax.get_xticklabels(),LINEAGE_ORDER):tick.set_color(LINEAGE_COLORS[lin])
        clean_axis(ax,True);fig.colorbar(im,cax=cax,label='Standardized association');fig.suptitle(title,y=1-.18/height,fontsize=13)
        fig.text(.72,1-.67/height,'● collection-wide BH FDR < 0.05',ha='center',fontsize=9)
        fig.text(.5,.025,'Frozen remodeling phenotype · donor-adjusted associations · conservative crossed-cluster uncertainty',ha='center',fontsize=9)
        save(fig,stem,part)
    hallnames=sorted(definitions[definitions.collection.eq('Hallmark')].pathway)
    heatmap(hallnames,'Hallmark','FigA13_complete_Hallmark_landscape','Complete Hallmark screen · all 50 pathways')
    reactnames=choose_pathways(results,definitions,'Reactome',24,True)
    # Order selected Reactome pathways by effect-profile similarity, only after complete testing.
    from scipy.cluster.hierarchy import linkage,leaves_list
    prof=results[results.pathway.isin(reactnames)].pivot(index='pathway',columns='lineage',values='beta').reindex(reactnames).fillna(0)
    reactnames=[reactnames[i] for i in leaves_list(linkage(prof.to_numpy(),method='average'))]
    heatmap(reactnames,'Reactome','FigA14_Reactome_landscape','Complete Reactome screen · prioritized nonredundant view')
    consistent=pd.read_csv(A3/'cross_lineage_consistency.csv')
    selected=[]
    for collection in ['Hallmark','Reactome']:
        part=consistent[consistent.collection.eq(collection)].sort_values(['n_significant_lineages','max_abs_beta','min_FDR'],ascending=[False,False,True]).head(6)
        selected.extend(part.pathway)
    part=results[results.pathway.isin(selected)];fig=plt.figure(figsize=(12,8));ax=fig.add_axes([.48,.20,.46,.67]);clean_axis(ax)
    for i,path in enumerate(selected):
        pp=part[part.pathway.eq(path)].set_index('lineage')
        for j,lin in enumerate(LINEAGE_ORDER):
            r=pp.loc[lin];pos=i+(j-2)*.13
            ax.plot([r.ci_low,r.ci_high],[pos,pos],color=LINEAGE_COLORS[lin],alpha=.45,lw=.7)
            ax.scatter(r.beta,pos,s=22,edgecolor=LINEAGE_COLORS[lin],facecolor=LINEAGE_COLORS[lin] if r.FDR<.05 else 'white',zorder=3)
    ax.axvline(0,color='#999999',lw=.7);ax.set_yticks(range(len(selected)),[p.replace('HALLMARK_','H: ').replace('REACTOME_','R: ').replace('_',' ').lower() for p in selected],fontsize=8);ax.invert_yaxis();ax.set_xlabel('Standardized effect and approximate 95% interval')
    fig.suptitle('Cross-lineage pathway consistency and heterogeneity',y=.97,fontsize=13)
    handles=[Line2D([],[],marker='o',ls='',color=LINEAGE_COLORS[l],label=DISPLAY_LABELS[l]) for l in LINEAGE_ORDER]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.07),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.035,'Filled markers: collection-wide FDR < 0.05 · descriptive consistency; no independent-lineage meta-test',ha='center',fontsize=9)
    save(fig,'FigA15_pathway_consistency',part)
    # Full-screen effect-vs-significance overview, not hypothesis-selected.
    for collection in ['Hallmark','Reactome']:
        fig,ax=plt.subplots(figsize=(8,6));fig.subplots_adjust(left=.13,right=.97,bottom=.23,top=.86)
        part=results[results.collection.eq(collection)]
        for lin in LINEAGE_ORDER:
            p=part[part.lineage.eq(lin)];ax.scatter(p.beta,-np.log10(np.maximum(p.FDR,1e-300)),s=12,color=LINEAGE_COLORS[lin],alpha=.55,label=DISPLAY_LABELS[lin])
        ax.axhline(-np.log10(.05),ls='--',lw=.8,color='#888888');ax.axvline(0,color='#bbbbbb',lw=.6);clean_axis(ax)
        ax.set(xlabel='Standardized pathway–remodeling association',ylabel='−log10(collection-wide BH FDR)',title=f'{collection}: complete effect–significance overview')
        fig.legend(loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=9)
        save(fig,f'FigS_{collection}_effect_significance',part)
    # Selected pathway details are chosen by the predeclared full-screen priority rule.
    phenotype=pd.read_csv(OUT/'03_unbiased_pathways/clock_remodeling_phenotype_donor_tissue_lineage.csv')
    for collection in ['Hallmark','Reactome']:
        top=results[results.collection.eq(collection)].sort_values(['significant','abs_beta','P'],ascending=[False,False,True]).iloc[0]
        scores=np.load(A3/f'{collection.lower()}_scores.npz');j=int(np.flatnonzero(scores['pathways']==top.pathway)[0]);part=phenotype[phenotype.lineage.eq(top.lineage)]
        sc=scores['scores'][part.index,j:j+1];b,se,p,df,rank,rx,ry=associations(sc,part,part.distance.to_numpy())
        fig,ax=plt.subplots(figsize=(8.5,6));fig.subplots_adjust(left=.12,right=.96,bottom=.22,top=.73)
        ax.scatter(rx[:,0],ry,s=23,color=LINEAGE_COLORS[top.lineage],alpha=.65);xx=np.array([rx[:,0].min(),rx[:,0].max()]);ax.plot(xx,b[0]*xx,color='#333333',lw=1)
        clean_axis(ax);ax.set(xlabel='Pathway score residual (donor and cell-count adjusted)',ylabel='Frozen remodeling-distance residual')
        fig.suptitle(textwrap.fill(top.pathway.replace('_',' '),66),y=.97,fontsize=12)
        fig.text(.5,.79,f'{DISPLAY_LABELS[top.lineage]} · β = {top.beta:.3f} · FDR = {top.FDR:.3g}',ha='center',fontsize=10)
        fig.text(.5,.065,'Chosen after the complete screen; points are dependent donor × tissue aggregates.\nLine is descriptive; uncertainty uses donor/tissue clusters, not independent points.',ha='center',fontsize=9)
        source=part.copy();source['pathway']=top.pathway;source['adjusted_score']=rx[:,0];source['adjusted_distance']=ry
        save(fig,f'FigS_{collection}_top_association_detail',source)
    evidence,decision,classification,alternative=integrated_evidence(summary,results)
    for stem,abstract in [('FigA16_integrated_evidence_map',False)]:
        fig=plt.figure(figsize=(12,7));ax=fig.add_axes([.03,.16,.94,.70]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
        cards=[('Clock-specific lineage information',evidence.iloc[0].finding,evidence.iloc[0].status),('Lineage-associated backbone',evidence.iloc[1].finding,evidence.iloc[1].status),('Tissue-associated remodeling',evidence.iloc[2].finding,evidence.iloc[2].status),('Associated context programs',evidence.iloc[3].finding,evidence.iloc[3].status)]
        for i,(title,body,status) in enumerate(cards):
            col=i%2;row=i//2;left=.015+col*.51;bottom=.54-row*.52
            patch=FancyBboxPatch((left,bottom),.46,.42,boxstyle='round,pad=0.012,rounding_size=0.02',facecolor='#F5F7F9',edgecolor='#C5CDD4',lw=1);ax.add_patch(patch)
            ax.text(left+.02,bottom+.35,title,fontsize=12,weight='bold',va='top')
            ax.text(left+.02,bottom+.265,status,fontsize=9,color='#4E79A7',va='top')
            ax.text(left+.02,bottom+.19,textwrap.fill(body,52),fontsize=10,va='top',linespacing=1.5)
        fig.suptitle('Human clock transcriptional architecture' if abstract else 'Phase A integrated evidence map',y=.97,fontsize=15)
        fig.text(.5,.89,'Static multi-tissue pseudobulk associations · no causal arrows',ha='center',fontsize=10,color='#555555')
        fig.text(.5,.065,classification+' · '+decision,ha='center',fontsize=11)
        fig.text(.5,.025,'No inference about temporal rhythms, phase or amplitude; independent replication not performed.',ha='center',fontsize=9)
        save(fig,stem,evidence)
    # A distinct graphical abstract: analytical units and noncausal association structure.
    fig=plt.figure(figsize=(12,7.2));ax=fig.add_axes([.03,.15,.94,.73]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.5,.96,'2,030 donor × tissue × cell-type pseudobulks',ha='center',fontsize=13,weight='bold')
    ax.text(.5,.885,'Predefined clock genes and modules · static transcriptional architecture',ha='center',fontsize=10)
    for i,lin in enumerate(LINEAGE_ORDER):
        pos=.10+i*.20;ax.scatter(pos,.74,s=380,color=LINEAGE_COLORS[lin],edgecolor='white',linewidth=1.5)
        label=DISPLAY_LABELS[lin].replace('Macrophage/monocyte','Macrophage/\nmonocyte');ax.text(pos,.64,label,ha='center',va='top',fontsize=10,color=LINEAGE_COLORS[lin])
    for pos in [.15,.5,.85]:ax.plot([pos,pos],[.53,.46],color='#AAB3BD',lw=1.2)
    frozen_distance=pd.read_csv(OUT/'statistics/distance_summary.csv');delta=frozen_distance[frozen_distance.summary_type.eq('effect')].iloc[0]
    hits=results[results.FDR.lt(.05)].groupby('collection').size().to_dict()
    boxes=[(.015,'Lineage organization',f'Between > within distance\nΔ = {delta["median"]:.3f} [{delta.bootstrap_95_ci_low:.3f}, {delta.bootstrap_95_ci_high:.3f}]'),(.35,'Tissue-associated variation','Different lineage dispersions\nFrozen full 24-feature analysis'),(.685,'Context associations',f'{hits.get("Hallmark",0)} Hallmark / {hits.get("Reactome",0)} Reactome hits\nSome strongest hits are tissue-sensitive')]
    for left,title,body in boxes:
        ax.add_patch(FancyBboxPatch((left,.23),.30,.24,boxstyle='round,pad=0.01',facecolor='#F4F6F8',edgecolor='#C7CFD7'))
        ax.text(left+.15,.425,title,ha='center',va='top',fontsize=11,weight='bold');ax.text(left+.15,.34,body,ha='center',va='top',fontsize=9,linespacing=1.6)
    ax.text(.5,.12,'Matched random genes explain comparable lineage information',ha='center',fontsize=12,weight='bold',color='#555555')
    ax.text(.5,.045,'Clock-specific enrichment is not established; generic cell identity remains an alternative.',ha='center',fontsize=10)
    fig.suptitle('Human clock transcriptional architecture',y=.97,fontsize=16)
    fig.text(.5,.067,classification+' · '+decision,ha='center',fontsize=11)
    fig.text(.5,.026,'Connecting lines denote associations only · no temporal or causal inference · no external replication',ha='center',fontsize=9)
    save(fig,'FigA_graphical_abstract',evidence)
    jwrite(AUD/'figure_validation.json',{'figures':audits,'visual_review':'pending','dpi':300})
    return evidence,decision,classification,alternative

def integrated_evidence(summary,results):
    primary=summary[summary.variant.eq('primary')];second=summary[summary.variant.eq('second_seed')]
    qualified=[]
    for panel in SETS:
        p=primary[primary.panel.eq(panel)&primary.metric.isin(['balanced_accuracy','silhouette','distance_ratio'])].set_index('metric')
        s=second[second.panel.eq(panel)].set_index('metric');good=(p.P.lt(.05)&p.BH_8_metrics.lt(.05)&p.observed.gt(p.null_mean)&s.loc[p.index,'P'].lt(.05))
        if good.sum()>=2:qualified.append(panel)
    # Driver checks are descriptive geometric checks, not a new matched-null test for each deletion.
    deletion=pd.read_csv(A1/'donor_tissue_deletions.csv');gene=pd.read_csv(A1/'leave_one_gene_out.csv')
    robust=[panel for panel in qualified if deletion[deletion.panel.eq(panel)].distance_ratio.gt(1).all() and deletion[deletion.panel.eq(panel)].silhouette.gt(0).all() and gene[gene.panel.eq(panel)].distance_ratio.gt(1).all()]
    distance=pd.read_csv(OUT/'statistics/distance_summary.csv');delta=distance[distance.summary_type.eq('effect')].iloc[0]
    a2good=delta.bootstrap_95_ci_low>0
    decision='GO FOR INDEPENDENT REPLICATION' if robust and a2good else 'PAUSE BEFORE REPLICATION'
    # Generic explanation is only partially weakened by matched results: controls do not isolate clock mechanism.
    alternative='PARTIALLY WEAKENED' if qualified else 'NOT WEAKENED'
    classification='STRONGLY SUPPORTED' if robust and a2good and results.FDR.lt(.05).any() else 'PARTIALLY SUPPORTED' if a2good else 'NOT SUPPORTED'
    sig=results[results.FDR.lt(.05)];counts=sig.groupby('collection').size().to_dict()
    text='; '.join(qualified)+' exceeds matched expectation on multiple metrics.' if qualified else 'No panel meets the prespecified multiple-metric matched-control criterion.'
    scores=pd.read_csv(OUT/'statistics/feature_effects_and_scores.csv');best=scores.nlargest(2,'lineage_stability').feature.str.replace('expr_','').tolist();rem=scores.nlargest(2,'tissue_remodeling').feature.str.replace('expr_','').tolist()
    rows=[{'layer':1,'component':'Clock specificity','status':'SUPPORTED RELATIVELY' if qualified else 'UNRESOLVED / NOT ESTABLISHED','finding':text},
      {'layer':2,'component':'Conserved lineage backbone','status':'FROZEN INTERNAL EVIDENCE','finding':f'{", ".join(best)} have high cross-tissue stability. Between–within distance Δ={delta["median"]:.3f}.'},
      {'layer':2,'component':'Tissue remodeling','status':'FROZEN INTERNAL EVIDENCE','finding':f'{", ".join(rem)} show high tissue remodeling; fibroblast point dispersion exceeds B cell.'},
      {'layer':3,'component':'Biological context pathways','status':'ASSOCIATIONS; NOT CAUSATION','finding':f'{counts.get("Hallmark",0)} Hallmark and {counts.get("Reactome",0)} Reactome pathway–lineage tests pass collection-wide FDR<0.05.'}]
    evidence=pd.DataFrame(rows);cwrite(ST/'integrated_evidence.csv',evidence)
    jwrite(ST/'gate_decision.json',{'decision':decision,'classification':classification,'generic_alternative':alternative,'qualified_panels':qualified,'driver_robust_qualified_panels':robust,'A2_positive_interval':bool(a2good),'pathway_FDR_supported_counts':counts,'gate_rule_source':'00_audit/unattended/completion_protocol.json'})
    return evidence,decision,classification,alternative

def markdown_table(df):
    # Keep reports dependency-free and Markdown-readable.
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.5g}' if np.isfinite(v) else 'N/A'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def write_reports(summary,null,results,definitions,evidence,decision,classification,alternative):
    primary=summary[summary.variant.eq('primary')];hyp=pd.read_csv(A3/'hypothesized_programs.csv');consistency=pd.read_csv(A3/'cross_lineage_consistency.csv');gene=pd.read_csv(A1/'leave_one_gene_out.csv');deletion=pd.read_csv(A1/'donor_tissue_deletions.csv')
    features=pd.read_csv(OUT/'statistics/feature_effects_and_scores.csv');distance=pd.read_csv(OUT/'statistics/distance_summary.csv');plasticity=pd.read_csv(OUT/'source_data/FigA4_lineage_plasticity.csv')
    cols=['panel','metric','observed','null_mean','null_median','null_sd','percentile','P','Z','BH_8_metrics']
    scorecols=['feature','lineage_stability','lineage_stability_ci_low','lineage_stability_ci_high'];remcols=['feature','tissue_remodeling','tissue_remodeling_ci_low','tissue_remodeling_ci_high']
    top=results.groupby(['collection','lineage'],sort=False).head(3)
    topcols=['lineage','pathway','beta','FDR','effect_priority_rank','donor_sign_fraction','tissue_sign_fraction','coverage']
    ranges=deletion.groupby(['panel','deletion'])[METRICS].agg(['min','max']);ranges.columns=['_'.join(c) for c in ranges.columns];ranges=ranges.reset_index()
    genemin=gene.groupby('panel')[METRICS].agg(['min','max']);genemin.columns=['_'.join(c) for c in genemin.columns];genemin=genemin.reset_index()
    null8=pd.read_csv(A1/'size_random8_checkpoint.csv');real8=pd.read_csv(A1/'size_real8_checkpoint.csv');null17=null[null.variant.eq('primary')&null.panel.eq('Expanded clock')].sort_values('iteration')
    size=[]
    for metric in METRICS:
        gain=null17[metric].to_numpy()-null8.sort_values('iteration')[metric].to_numpy()
        size.append({'metric':metric,'random17_mean':null17[metric].mean(),'paired_random8_mean':null8[metric].mean(),'mean_17_minus_8':gain.mean(),'gain_q025':np.quantile(gain,.025),'gain_q975':np.quantile(gain,.975),'real_expanded8_median':real8[metric].median(),'real_expanded8_q025':real8[metric].quantile(.025),'real_expanded8_q975':real8[metric].quantile(.975)})
    size=pd.DataFrame(size);cwrite(A1/'gene_set_size_summary.csv',size)
    qc=pd.read_csv(A1/'matching_quality.csv');qcsummary=qc.groupby(['panel','variant'])[['mean_abs_logmean_z_difference','mean_abs_logvariance_z_difference','max_distance']].agg(['mean','median','max']);qcsummary.columns=['_'.join(c) for c in qcsummary.columns];qcsummary=qcsummary.reset_index();cwrite(A1/'matching_quality_summary.csv',qcsummary)
    # Per-slot mismatch distributions independently reconstruct from exported matching records.
    stats=pd.read_csv(A1/'gene_matching_statistics.csv').set_index('gene');quality=[]
    for variant,_,_,_ in VARIANTS:
        for panel,slug in [('Core clock','core'),('Expanded clock','expanded')]:
            p=pd.read_csv(A1/f'{variant}_{slug}_genes.csv');a=stats.loc[p.matched_gene,['standardized_log_mean','standardized_log_variance']].to_numpy();b=stats.loc[p.clock_gene,['standardized_log_mean','standardized_log_variance']].to_numpy();delta=np.abs(a-b)
            quality.append({'panel':panel,'variant':variant,'mean_abs_mean_z':delta[:,0].mean(),'mean_abs_variance_z':delta[:,1].mean(),'q95_abs_mean_z':np.quantile(delta[:,0],.95),'q95_abs_variance_z':np.quantile(delta[:,1],.95),'poor_match_flag':bool((delta.mean(0)>.1).any() or (np.quantile(delta,.95,axis=0)>.25).any())})
    quality=pd.DataFrame(quality);cwrite(A1/'matching_quality_threshold_audit.csv',quality)
    drivers=results[results.FDR.lt(.05)].copy()
    drivers['tissue_max_change_relative_to_full']=drivers.tissue_max_abs_change/drivers.beta.abs()
    drivers['donor_max_change_relative_to_full']=drivers.donor_max_abs_change/drivers.beta.abs()
    cwrite(A3/'FDR_supported_pathway_driver_audit.csv',drivers)
    drivercols=['collection','pathway','lineage','beta','FDR','n_scored','coverage','donor_beta_min','donor_beta_max','donor_worst_block','tissue_beta_min','tissue_beta_max','tissue_worst_block','tissue_any_sign_reversal']
    strongest=drivers.sort_values('abs_beta',ascending=False).groupby('collection').head(5)
    ecm=results[results.collection.eq('Reactome')&results.pathway.str.contains('EXTRACELLULAR|COLLAGEN|LAMININ|FIBRONECTIN|ELASTIN|PROTEOGLYCAN|INTEGRIN')]
    cwrite(A3/'postscreen_ECM_keyword_annotation.csv',ecm)
    secondary_ifng=results[results.collection.eq('Reactome')&results.pathway.str.contains('IFNG|INTERFERON_GAMMA')]
    cwrite(A3/'postscreen_secondary_IFNG_annotation.csv',secondary_ifng)
    influences=[]
    observed=pd.read_csv(OUT/'01_matched_random_controls/observed_metrics.csv').set_index('panel')
    for frame in [gene,deletion]:
        for (panel,kind),part in frame.groupby(['panel','deletion']):
            for metric in METRICS:
                r=part.loc[part[metric].idxmin()];influences.append({'panel':panel,'deletion':kind,'metric':metric,'worst_removed':r.identifier,'full':observed.loc[panel,metric],'minimum':r[metric],'drop':observed.loc[panel,metric]-r[metric]})
    influences=pd.DataFrame(influences);cwrite(A1/'worst_deletion_influence.csv',influences)
    robustness_report=f'''# Pathway and lineage-information robustness

All deletion analyses preserve the already defined outcomes; pathway reference centroids are frozen. No new P values from deleting selected adverse observations are substituted for primary results.

## Clock-panel influence
{markdown_table(influences[influences.metric.eq('balanced_accuracy')])}
Deletion preserves substantial absolute lineage information. Multiple genes contribute; these checks do not establish clock-specific enrichment, which fails the matched-null gate. Detailed four-metric ranges and every deletion are available in A1 outputs.

## Strongest FDR-supported pathway effects
{markdown_table(strongest[drivercols])}
Sign stability is not equivalent to magnitude stability. B-cell Hallmark coagulation drops from0.884 to0.101 after lung deletion, despite retaining its sign. The strongest B-cell Reactome extrinsic fibrin-clot and alternative-complement effects reverse direction when lung is removed. These specific associations are strongly tissue-sensitive; their full-data FDR values must not be presented as broadly robust context mechanisms. Five T-cell Hallmark hits retain their negative sign under every individual donor/tissue deletion. Full effect ranges, worst blocks and poor-coverage flags are saved for all FDR-supported tests. Reactome clot-initiation and alternative-complement sets contain only5 scored genes and are flagged as small.

## ECM and secondary IFNG annotation after complete testing
The post-screen ECM keyword annotation covers{ecm.pathway.nunique()} Reactome sets ({len(ecm)} lineage tests), with{int(ecm.FDR.lt(.05).sum())} FDR-supported associations. This is an explicit name-based annotation, not an exhaustive ontology mapping and not a new statistical family. The exact extracellular-matrix-organization pathway also lacks FDR support.

The frozen primary IFN-gamma-response hypothesis is the Hallmark set and is NOT SUPPORTED. Different Reactome IFNG-related pathways can nevertheless show secondary associations:
{markdown_table(secondary_ifng[secondary_ifng.FDR.lt(.05)][['pathway','lineage','beta','FDR','donor_sign_fraction','tissue_sign_fraction']])}
This distinction is retained: no pathway definition was changed to rescue the Hallmark hypothesis, and secondary IFNG regulation is not relabeled as confirmation of the same response program.
'''
    (REP/'PHASE_A_ROBUSTNESS_REPORT.md').write_text(robustness_report,encoding='utf-8')
    gate=f'''# Phase A final gate report — local completion

Decision: **{decision}**. Overall model: **{classification}**. Generic-cell-identity alternative: **{alternative}**.

The previous blocked report is archived in00_audit/unattended/previous_reports. This report supersedes its missing-dependency status. Frozen A2 was neither rerun nor edited. Broad matrix2030×59410, exact metadata,68 measurable genes reproduced within8.88e-16; CTGF absent/zero. Complete MSigDB2026.1.Hs:50 Hallmark and1839 Reactome sets. No remote connection, package installation, raw ingestion, new embedding, or external replication.

## 1–3. Core and Expanded primary matched-null results
{markdown_table(primary[cols])}
Each empirical null contains1000 unique-within-panel matched draws (draws may repeat across iterations). Upper-tail P=(1+count(null>=observed))/1001; percentile uses midrank ties. Primary BH covers8 metrics; macro F1 is reported but gate metrics are balanced accuracy, silhouette and ratio. Standardization for CV is training-only; geometry uses original eligible-observation scale. No donor enters both training and test in a fold.

## 4. Multiple-metric clock specificity
{markdown_table(evidence[evidence.layer.eq(1)])}
Core8 and Expanded17 have separate exact-size matched controls. A high absolute classifier score alone is not clock specificity. Negative or nonsignificant upper-tail results are retained.

## 5. Independent seed and matching sensitivity
{markdown_table(summary[summary.variant.ne('primary')][['variant','panel','metric','observed','null_mean','percentile','P']])}
All4 variants ran1000 panels for each panel. Primary nearest100 expression+variance matching is unchanged; seed20260914 tests Monte Carlo reproducibility. Nearest25 and expression-only sensitivity were fixed before outcomes. Detection was unavailable consistently and was not reconstructed.

Matching diagnostics, with predeclared poor-match threshold:
{markdown_table(quality)}
Poor matching is not repaired by changing the primary result. Both tighter and primary results are reported, including any disagreement. QC variables are standardized log10 mean/variance over eligible nonclock genes.

Panel-size control (1000 paired random17→8 and1000 real8-of17 subsets):
{markdown_table(size)}
This assesses size within the expanded matching universe; composition and size are not fully separable. Random17-vs8 percentile intervals are descriptive Monte Carlo distributions, not biological confidence intervals.

## 6. Single-clock-gene sensitivity
{markdown_table(genemin)}
All25 one-gene deletions saved with four metrics. No deletion-specific matched-null inference is claimed. A fall in score is influence, not proof that one gene causes lineage separation. Named worst deletions are listed in the machine-readable source tables.

## 7–8. Tissue and donor drivers
{markdown_table(ranges)}
Observed panels were evaluated after removing each29 supported tissues and24 donors, preserving surviving frozen folds and reapplying the original≥3-donor centroid support. These are actual refitted classifier/geometry sensitivity results, not new1000-panel nulls; no deletion-specific P values. Fixed-remodeling-reference pathway sensitivities are separate below.

## 9. Frozen A2 distance confirmation
{markdown_table(distance[distance.summary_type.ne('lineage_within')])}
Full24-feature space,928 within and139 same-tissue between comparisons;5000 shared donor/tissue multiplier draws. Percentile interval is approximate for a nonlinear distance contrast, not an exact calibrated test. The broad dependency does not change A2 preprocessing or scores.

## 10. Conserved-backbone features
{markdown_table(features.nlargest(5,'lineage_stability')[scorecols])}

## 11. Tissue-remodeled features
{markdown_table(features.nlargest(5,'tissue_remodeling')[remcols])}
{markdown_table(plasticity)}
These frozen continuous scores are descriptive, with approximate donor-cluster jackknife intervals; no newly optimized binary categories.

## 12. Unbiased Hallmark results by lineage
{markdown_table(top[top.collection.eq('Hallmark')][topcols])}

## 13. Complete secondary Reactome results by lineage
{markdown_table(top[top.collection.eq('Reactome')][topcols])}
Top rows prioritize FDR-supported results, then absolute effect. Entries with FDR≥0.05 are not supported hits. All sets were retained in the multiplicity family, including untestable P=1. Reactome redundancy reduction occurs only after statistical testing for visualization.

## 14. Prespecified biological programs
{markdown_table(hyp[['hypothesis','hypothesis_status','lineage','P_rank_within_lineage','effect_priority_rank','beta','FDR','donor_sign_fraction','tissue_sign_fraction','coverage']])}
Ranks are unbiased within the full collection for each lineage. P-rank is provided for comparison; interpretation prioritizes effect/FDR rather than P alone. SUPPORTED requires≥1 FDR<0.05 lineage with≥80% sign retention under both donor/tissue deletions,≥10 scored genes and≥20% original coverage; PARTIAL means an FDR hit without those robustness/coverage criteria; NOT SUPPORTED means no FDR hit, not proof of absence. Exact ECM organization membership was inspected only after the complete Reactome screen.

## 15–16. Alternative explanation
**{alternative}**. Generic cell-identity transcription remains the main alternative; matched mean/variance controls address technical abundance/variance, not functional annotation, gene–gene correlation, fine-subtype mixture, donor/tissue imbalance or residual cell-number effects. If a panel exceeds matched controls, that establishes relative enrichment under this null, not a clock-specific causal mechanism. If it does not, absolute separation remains compatible with generic lineage information.

## 17–18. Overall model and strongest claim
**{classification}**. Predefined clock-associated expression features contain donor-held-out lineage information and exhibit static tissue-associated architectural variation in these Tabula Sapiens pseudobulks. Matched-null results above qualify specificity, and complete pathway screens identify associations at the stated collection-wide FDR. Associations and internal validation do not establish regulation or temporal dynamics.

## 19. Unsupported / limitations
No time-stamped evidence for rhythmicity, phase, amplitude or oscillation; no causal tissue regulation, external replication or CircadianKAT integration. Module overlap is retained from the original analysis. Pathway effects use donor fixed effects plus log1p(n_cells), with conservative crossed-cluster sandwich and df=min(donors,tissues)-1; small cluster counts and shared phenotype references make intervals/P approximate. Pathway genes exclude all17 clock genes, removing direct overlap but not correlation. Gene scores are standardized over the fixed atlas, then averaged to the frozen donor×tissue×lineage observations. The frozen phenotype is not redefined under deletion. Donor+tissue fixed-effect sensitivities and all donor/tissue deletion effect ranges are saved. Sign stability alone does not establish statistical replication; cross-lineage consistency is descriptive because lineages share donors/tissues, so no independent-study meta-analysis is asserted. Very small/poorly covered pathways are flagged, not hidden. Full primary and sensitivity results are retained even when unfavorable.

## 20. Phase B gate
**{decision}**. Gate requires a justified panel to exceed controls on≥2 primary lineage-information metrics, with8-test BH support and independent-seed agreement, plus frozen A2 evidence and no simple geometric driver reversal. No external dataset was accessed. Replication specifications are created only if GO.

## Reproducibility and files
New results are in01_matched_random_controls/unattended,03_unbiased_pathways/unattended,statistics/unattended,source_data/unattended; figures FigA9–FigA16 and supplementary displays. Dependency hashes, frozen protocol and initial/final integrity checks are in00_audit/unattended. Existing A2 files and original result files remain unchanged except this designated report. Run regenerate_phaseA_unattended.ps1 to resume checkpoints.
'''
    gate+='\n## Explicit driver verdicts and secondary-program distinction\n'+markdown_table(influences[influences.metric.eq('balanced_accuracy')])+'\nAbsolute clock-panel classification persists across deletions; no single deletion establishes or removes a matched-null enrichment because none was demonstrated. B-cell coagulation and the strongest Reactome clot/complement associations are strongly lung-sensitive (including Reactome sign reversals). Their large full-data effects should not be called context-general mechanisms. T-cell Hallmark hits retain negative direction across all individual block deletions. See PHASE_A_ROBUSTNESS_REPORT.md for effect ranges, small-set flags and exact worst blocks.\n\nAll four predefined Hallmark programs and exact Reactome ECM organization lack FDR support. This does not mean every related pathway is null: Reactome REGULATION_OF_IFNG_SIGNALING has a secondary T-cell association, distinct from the failed Hallmark IFN-gamma-response hypothesis. Definitions and multiplicity were not changed.\n'
    (REP/'PHASE_A_GATE_REPORT.md').write_text(gate,encoding='utf-8')
    # Statistical audit with numerical multiplicity and matching checks.
    audit=f'''# Phase A statistical/data-integrity audit

- Matrix2030×59410; metadata2030 unique IDs;59410 unique genes; all finite. Gene metadata identical to matrix order;68 original measurable genes agree to8.8818e-16; CTGF missing and zero originally. Dependency hashes and versions saved. No new package installation.
- Frozen donor split manifest reused without donor overlap. Observed metrics reproduce original export within1e-9 relative /1e-10 absolute tolerance. All primary and sensitivity panels have8 or17 distinct genes; every expanded-clock gene excluded. Primary and second-seed each2000 panels; tighter and expression-only each2000; paired random-size and real-size analyses each1000 additional panels. Checkpoints every100 evaluations.
- Matching statistics and candidate pools exported. Quality thresholds fixed before results: mean absolute z mismatch>0.10 or95th absolute mismatch>0.25; no change of primary method. Diagnostics:
{markdown_table(quality)}
- Empirical P recalculated from saved null draws; BH independently checked using scipy.stats.false_discovery_control. Upper-tail P minimum1/1001. The8 metric BH family is supplemental to raw empirical tests and conservatively required for GO.
- Hallmark family250 pathway×lineage tests; Reactome family9195 tests. Full collections retained; ineligible scores P=1. All pathway memberships explicitly exclude17 clock genes after alias harmonization. No biological-program preselection. Retained coverage and poor-coverage flags saved.
- Association implementation reused unchanged from phaseA_resume.py. Independent full-design OLS, donor/tissue/intersection sandwich, score-scale invariance and BH regression tests run. Approximate inference under small clusters/shared references; no exact independent-sample claim.
- Pathway driver diagnostics include leave-one-donor and leave-one-tissue effect ranges, sign retention, largest-change block, and tissue-FE sensitivity for every pathway. A significance hit is not declared causal or externally replicated. Full cross-lineage effect profiles saved; qualitative consistency, no invalid independent-lineage meta-P.
- A2 and original result hashes are compared with the pre-run baseline; only designated reports may change. See final_integrity.json for exact status. Final figures have CSV source data, PNG≥300dpi,SVG,PDF. Programmatic checks and visual review are recorded separately.

No outcome-based thresholds or filtered pathway rescue. Gene-set-size analysis distinguishes size effects descriptively; it cannot fully separate gene composition from count. Existing report/data files outside newly named outputs remain historical and are not silently relabeled.
'''
    (REP/'PHASE_A_STATISTICAL_AUDIT.md').write_text(audit,encoding='utf-8')
    # Results draft uses numeric findings, without assuming desired enrichment.
    finding=[]
    for panel in SETS:
        part=primary[primary.panel.eq(panel)].set_index('metric')
        finding.append(f'{panel} yielded balanced accuracy {part.loc["balanced_accuracy","observed"]:.3f}, macro F1 {part.loc["macro_f1","observed"]:.3f}, silhouette {part.loc["silhouette","observed"]:.3f}, and distance ratio {part.loc["distance_ratio","observed"]:.3f}. Corresponding matched-null upper-tail P values were '+', '.join(f'{part.loc[m,"P"]:.4g}' for m in METRICS)+'.')
    counts=results.groupby('collection').FDR.apply(lambda s:int(s.lt(.05).sum()))
    d=distance[distance.summary_type.eq('effect')].iloc[0]
    narrative='# Phase A Results draft\n\nWe analyzed2030 eligible donor×tissue×cell-type pseudobulks using frozen clock-feature definitions. Five-lineage classification included1302 observations and donor-disjoint cross-validation.\n\n'+'\n\n'.join(finding)+f'\n\nEach panel was compared with1000 exact-size nonclock panels matched on expression mean and variance. The independent-seed, tighter-match and expression-only results are reported without replacing the primary null. The generic-cell-identity alternative was classified as {alternative.lower()}.\n\nIn the frozen24-feature analysis, between-lineage distances exceeded within-lineage cross-tissue distances by {d["median"]:.3f} (approximate95% donor/tissue-bootstrap interval {d.bootstrap_95_ci_low:.3f}–{d.bootstrap_95_ci_high:.3f}). NFIL3 and RORA had high tissue-held-out lineage-stability scores, whereas CRY1 and HLF had high within-lineage tissue effects. Fibroblast had higher point cross-tissue dispersion than B cell, with overlapping uncertainty across some lineages.\n\nComplete screens evaluated50 Hallmark and1839 Reactome gene sets with clock genes excluded from pathway scores. {counts.get("Hallmark",0)} Hallmark and {counts.get("Reactome",0)} Reactome pathway–lineage associations passed collection-wide FDR<0.05. Top effects and donor/tissue sensitivity are shown below.\n\n'+markdown_table(top[['collection']+topcols])+f'\nThe integrated static association model was {classification.lower()}; the prespecified recommendation was {decision.lower()}. These internal observational analyses do not establish temporal clock behavior or causal regulation.\n'
    narrative+='\nThe strongest positive B-cell coagulation/clot associations were sensitive to lung: Hallmark coagulation attenuated substantially, and selected Reactome clot/complement effects reversed sign on lung deletion. Several highest-effect Reactome sets contained only5 genes. In contrast, five negative T-cell Hallmark associations retained their directions across every individual donor and tissue deletion. The predefined Hypoxia, fatty-acid metabolism, IFN-gamma-response, TNF/NF-kB and ECM-organization hypotheses did not attain collection-wide FDR support. A distinct Reactome IFNG-regulation association was observed as secondary evidence and was not substituted for the Hallmark hypothesis.\n'
    (REP/'PHASE_A_RESULTS_DRAFT.md').write_text(narrative,encoding='utf-8')
    claims=pd.DataFrame([
      ['Clock panels contain lineage information','Frozen donor-group CV metrics','A1 observed','Internal predictive evidence','Generic transcriptional lineage information','Predictive lineage information in these clock-associated genes','Clock-specific mechanism established'],
      ['Clock-specific enrichment under matched null','Primary1000 panels per set + second seed','A1 matched controls','See numeric empirical P/BH',alternative,'Relative to expression/variance-matched controls, panel-specific evidence follows reported tests','All clock features outperform arbitrary genes'],
      ['Lineage organization and tissue variation',f'Delta{d["median"]:.3f}, interval[{d.bootstrap_95_ci_low:.3f},{d.bootstrap_95_ci_high:.3f}]','Frozen A2','Approximate dependency-aware evidence','Fine-subtype composition and sampling imbalance','Static architecture varies across lineages and tissues','Tissues shift circadian phase'],
      ['Context pathways associate with remodeling',f'{counts.get("Hallmark",0)} Hallmark/{counts.get("Reactome",0)} Reactome FDR-supported tests','Complete A3','Cluster-aware internal associations','Shared cell state, cell composition, donor/tissue confounding','Association at stated FDR and coverage, with sensitivity limitations','Pathway causally remodels the clock'],
      ['Cross-lineage consistency','Effect profiles, directions and FDR by lineage','A3 consistency','Descriptive','Shared donors/tissues','Concordant or heterogeneous associations across sampled lineages','Independent replication across lineages']
    ],columns=['Claim','Evidence','Analysis','Strength','Alternative explanation','Supported wording','Prohibited overclaim'])
    (REP/'PHASE_A_CLAIMS_TABLE.md').write_text('# Phase A claims table\n\n'+markdown_table(claims),encoding='utf-8')
    readiness=f'''# Manuscript evidence readiness

Current assessment: **{classification}**; **{decision}**. No journal acceptance prediction.

Present: validated broad pseudobulk provenance, exact frozen observation compatibility, donor-disjoint classification,1000-panel primary matched controls per clock panel, independent-seed/alternative-match/size/deletion checks, frozen full-dimensional dependence-aware distance analysis, complete versioned Hallmark/Reactome screens, clock-overlap exclusion, collection-wide FDR, pathway coverage and donor/tissue sensitivity, source tables, reproducibility checkpoints and figure QA.

Still missing: independent human multi-tissue replication; time-stamped experimental evidence for any temporal clock claim; experimental causal evidence; stronger resolution of fine-subtype mixture and donor/tissue confounding; validation of interval calibration with few crossed clusters/shared reference phenotypes. Matched-null failure or mixed metrics cannot be reframed as clock specificity. Effect-size prioritization and robustness flags must accompany pathway hit counts; pathway redundancy is not independent evidence.

Do not start Phase B unless the numerical gate is GO. If GO, freeze hypotheses and success/failure criteria before accessing an external dataset. No replication or CircadianKAT integration was run.
'''
    (REP/'PHASE_A_NC_READINESS.md').write_text(readiness,encoding='utf-8')
    legends='''# Phase A figure legends

Existing FigA2: Frozen donor-held-out Core/Expanded confusion matrices; row-normalized percentages and counts, fixed ridge classifier, five donor groups. FigA3: Frozen24-feature within-/between-lineage distances;928 and139 dependent pairs,5000 shared donor/tissue multiplier draws. FigA4: Frozen lineage-specific median tissue distances and approximate bootstrap intervals; support≥3donors. FigA5: Frozen tissue-held-out stability versus within-lineage tissue unique adjusted R²; donor-jackknife intervals, signed values. FigA8: Historical pre-broad-data evidence summary; superseded by FigA16 and retained unchanged. Historical FigA1/A6/A7 were blocked; the completed corresponding analyses are now FigA9/A13/A14/A15.

FigA9 (classification/geometry): Gray distributions of1000 expression+variance-matched nonclock panels of identical size to Core8/Expanded17; colored observed values and empirical upper-tail P. Separate metrics share horizontal scales across panels; no embedding endpoints.

FigA10: Distribution of panel-mean standardized log-expression/log-variance for nearest100 primary and nearest25 prespecified tighter matching. Diamonds mark clock targets. Box plots show quartiles/median and1.5IQR whiskers; all per-gene/panel diagnostics saved.

FigA11 (Core/Expanded): Balanced accuracy and high-dimensional distance ratio after removing each named clock gene. Dashed reference indicates full-panel observed metric; source tables also provide F1 and silhouette. No deletion-specific matched-null significance is implied.

FigA12 (classification/geometry): Four observed metrics after removing each24donors or29 supported tissues. Frozen donor-fold assignments retained for remaining observations; centroid groups reapply≥3donors. Short horizontal bars show full-panel references; points are sensitivities, not independent replicates.

FigA13: All50 Hallmark pathways, five lineages. Color denotes standardized donor-/cell-count-adjusted remodeling association; dots indicate BH FDR<0.05 across250 tests. Clock genes excluded; frozen phenotype unchanged. Approximate donor/tissue-cluster uncertainty uses small-cluster t reference.

FigA14: Up to24 prioritized Reactome pathways from the complete1839-set/9195-test screen. Selection prioritizes FDR-supported absolute effects, then removes Jaccard>0.5 overlaps only for display; ordering clusters effect profiles. Color/dots as FigA13; complete statistics retained.

FigA15: Twelve pathways chosen after complete screens by number of FDR-supported lineages, effect magnitude and FDR. Lineage-colored estimates/approximate95% intervals; filled symbols FDR<0.05. Descriptive concordance/heterogeneity; no independent-lineage meta-test.

FigA16: Evidence map separates matched-null specificity, frozen lineage backbone, tissue variation and context-program associations. No causal arrows. Decision follows the frozen gate criteria.

Graphical abstract: Conceptual summary of the same evidence layers and limits; static associations only, not temporal clock activity.

Supplementary complete-screen effect–significance views: Standardized effects versus−log10 collection-wide FDR, lineage colors, horizontal0.05 threshold. All pathway tests shown where effects estimable.

Supplementary selected-association details: One effect-prioritized pathway–lineage pair per complete collection, selected only after screening. Residualized pathway score and fixed remodeling phenotype after donor/cell-count adjustment. Points are donor×tissue aggregates with dependent references; line shows FWL slope, not independent-point inference. Selection is exploratory and preserved in source data.

All new figures: white background, no grid, shared lineage/Core/Expanded palette,300dpi PNG plus SVG/PDF, exact source tables in source_data/unattended. See figure_validation.json for programmatic and visual checks.
'''
    (REP/'PHASE_A_FIGURE_LEGENDS.md').write_text(legends,encoding='utf-8')
    if decision=='GO FOR INDEPENDENT REPLICATION':
        frozen_spec='# Phase B frozen replication specification — NOT RUN\n\n'+json.dumps(json.loads((AUD/'completion_protocol.json').read_text()),indent=2)+'\n\nSuccess: same justified panel shows upper-tail matched-nullP<0.05 with8-testBH support on≥2 of BA/silhouette/ratio, independent-seed agreement, between-minus-within interval excluding0 under the same support rules, and no single donor/tissue geometry reversal. Core/Expanded definitions, lineage mapping from original revision_v2_runner.py, n_cells≥20 and groupdonors≥3 frozen. Train-only scaling for CV; atlas-wide gene scaling for geometry/pathways. No threshold tuning on external results. Failure: insufficient support, specificity criterion not met, inconsistent architecture or a dominant driver. Permitted sensitivities are precisely the predeclared variants/deletions; they cannot replace a failed primary endpoint. Pathway replication tests selected Phase A pathway–lineage associations with multiplicity correction over all frozen selected pairs, while reporting coverage and sign agreement. No new gene/pathway definitions based on replication outcomes.\n'
        (REP/'PHASE_B_FROZEN_REPLICATION_SPEC.md').write_text(frozen_spec,encoding='utf-8')
        requirements='# Independent atlas requirements — no data accessed\n\nHuman multi-tissue sc/snRNA-seq with independently sampled donors, donor IDs, tissue annotations, cell labels allowing frozen major-lineage mapping, per-cell count/normalized-expression provenance and broad gene coverage. Preserve independence from Tabula Sapiens donors. Enough≥20-cell pseudobulks and≥3donors per lineage×tissue group across multiple tissues;≥5donor groups for grouped CV. Clock genes and pathways must have coverage reported before results. Record sex/age/assay/batch where available and separate single-nucleus assay limitations. No temporal claims without collection-time labels. Do not choose datasets based on favorable clock separation.\n'
        (REP/'PHASE_B_DATASET_REQUIREMENTS.md').write_text(requirements,encoding='utf-8')
    (OUT/'README_UNATTENDED_COMPLETION.md').write_text('# Latest Phase A outputs\n\nUse reports/PHASE_A_GATE_REPORT.md and new unattended subdirectories for current A1/A3 results. Earlier blocked tables/reports were retained as historical snapshots; frozen A2 is unchanged. New figures are FigA9–FigA16 plus graphical abstract and supplementary plots. See reports/PHASE_A_FIGURE_LEGENDS.md. No external replication run.\n',encoding='utf-8')

def final_integrity():
    baseline=json.loads((AUD/'initial_frozen_hashes.json').read_text(encoding='utf-8-sig'));allowed={str(REP/'PHASE_A_GATE_REPORT.md')};changes=[];missing=[]
    for name,digest in baseline.items():
        p=Path(name)
        if not p.exists():missing.append(name)
        elif sha(p)!=digest and name not in allowed:changes.append(name)
    assert not changes and not missing,(changes,missing)
    assert frozen()==json.loads((AUD/'dependency_integrity.json').read_text())['A2_frozen']
    figures=[]
    for p in FIG.glob('*.png'):
        if str(p) in baseline:continue
        with Image.open(p) as im:assert min(im.info.get('dpi',(0,0)))>=299
        for suffix in ['.svg','.pdf']:assert p.with_suffix(suffix).exists()
        assert (SRC/(p.stem+'.csv')).exists();figures.append(p.name)
    result={'baseline_files':len(baseline),'allowed_designated_report_updates':sorted(allowed),'unexpected_changed_files':changes,'missing_files':missing,'A2_hashes_unchanged':True,'figures_with_sources_and_all_formats':figures,'source_script_sha256':sha(Path(__file__)),'no_remote_or_replication':True}
    jwrite(AUD/'final_integrity.json',result);return result

def finish():
    summary,null,results,definitions=load_results()
    evidence,decision,classification,alternative=render_figures(summary,null,results,definitions)
    write_reports(summary,null,results,definitions,evidence,decision,classification,alternative)
    final_integrity();print(decision,classification,alternative,flush=True)

if __name__=='__main__':main()
