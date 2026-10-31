"""Isolated Phase A validation; never reconstruct or overwrite the original analysis."""
from __future__ import annotations
import json, sys, itertools, importlib.metadata, shutil
from pathlib import Path
from clock_centroid_distances import *
from clock_architecture_umap import V1
from scipy.linalg import lstsq
from scipy.stats import t as student_t
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score, confusion_matrix
from PIL import Image

OUT=ROOT/'results/phaseA_clock_architecture'
OLD=ROOT/'figures_revised/clock_centroid_distances'
SEED=20260913
EFFECTS=['cell_identity_unique_adj_r2','tissue_context_unique_adj_r2','donor_unique_adj_r2','cell_identity_tissue_context_interaction_adj_r2']
SUBS=['00_audit','01_matched_random_controls','02_backbone_remodeling','03_unbiased_pathways','04_integrated_model','figures','source_data','statistics','reports']

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def table(name,data):
    data.to_csv(OUT/name,index=False,float_format='%.17g')

def adjusted_r2(meta,y,terms):
    pieces=[np.ones((len(meta),1))]
    for term in terms:
        value=(meta.cell_type.astype(str)+'||'+meta.tissue.astype(str)) if term=='celltype_tissue' else meta[term].astype(str)
        pieces.append(pd.get_dummies(value,dtype=float).to_numpy())
    x=np.column_stack(pieces)
    beta,_,rank,_=lstsq(x,y,cond=max(x.shape)*np.finfo(float).eps,lapack_driver='gelsd')
    sse=((y-x@beta)**2).sum(axis=0); sst=((y-y.mean(axis=0))**2).sum(axis=0)
    return 1-sse/sst*(len(meta)-1)/(len(meta)-rank),int(rank)

def effects(meta,y):
    main,rank=adjusted_r2(meta,y,['cell_type','tissue','donor_id'])
    reduced=[adjusted_r2(meta,y,terms)[0] for terms in [['tissue','donor_id'],['cell_type','donor_id'],['cell_type','tissue']]]
    # Cell x tissue indicators already span both cell and tissue main effects.
    full,fullrank=adjusted_r2(meta,y,['celltype_tissue','donor_id'])
    return np.column_stack([main-r for r in reduced]+[full-main]),(rank,fullrank)

def stability(meta,y):
    """Leave-one-tissue-out lineage predictor relative to a training grand mean."""
    sse=np.zeros(y.shape[1]); baseline=sse.copy()
    labels=meta.priority_lineage.to_numpy()
    for tissue in sorted(meta.tissue.unique()):
        test=meta.tissue.eq(tissue).to_numpy(); train=~test
        grand=y[train].mean(axis=0)
        for lin in LINEAGE_ORDER:
            target=test&(labels==lin); reference=train&(labels==lin)
            if target.any():
                if not reference.any():raise ValueError('No cross-tissue lineage reference')
                sse+=((y[target]-y[reference].mean(axis=0))**2).sum(axis=0)
                baseline+=((y[target]-grand)**2).sum(axis=0)
    return 1-sse/baseline

def remodeling(meta,y):
    result=[]
    for lin in LINEAGE_ORDER:
        mask=meta.priority_lineage.eq(lin).to_numpy(); part=meta.loc[mask]
        full,_=adjusted_r2(part,y[mask],['tissue','donor_id'])
        reduced,_=adjusted_r2(part,y[mask],['donor_id'])
        result.append(full-reduced)
    return np.mean(result,axis=0),np.array(result)

def jackknife_interval(point,draws):
    """Approximate donor-cluster delete-one jackknife t interval, signed/unclipped."""
    draws=np.asarray(draws); n=len(draws)
    se=np.sqrt((n-1)/n*((draws-draws.mean(axis=0))**2).sum(axis=0))
    return point-student_t.ppf(.975,n-1)*se,point+student_t.ppf(.975,n-1)*se,se

def distance_delta(meta,x,centroids,pairs):
    means=meta.join(x).groupby(['priority_lineage','tissue'],observed=True)[list(x.columns)].mean()
    vectors=np.array([means.loc[(r.lineage,r.tissue)].to_numpy() for _,r in centroids.iterrows() if r.adequate_support])
    supported=centroids[centroids.adequate_support]
    mapping={old:new for new,old in enumerate(supported.index)}
    a=pairs.centroid_index_A.map(mapping).to_numpy(); b=pairs.centroid_index_B.map(mapping).to_numpy()
    d=np.linalg.norm(vectors[a]-vectors[b],axis=1); w=pairs.comparison_type.eq(WITHIN).to_numpy()
    return float(np.median(d[~w])-np.median(d[w]))

def run():
    for sub in SUBS:(OUT/sub).mkdir(parents=True,exist_ok=True)
    before=snapshot(); existing={str(p):sha(p) for p in (ROOT/'figures_revised').rglob('*') if p.is_file()}
    meta,x,verification,threshold=load_frozen_inputs(); env=original_spec()
    raw=pd.read_csv(FEATURE_PATH,sep='\t'); valid=raw[raw.n_cells.ge(20)].set_index('group_id').loc[meta.index]
    feature_cols=list(dict.fromkeys(env['CLOCK_FEATURE_COLS']+env['MODULE_COLS']))
    prepared,retained=env['get_feature_matrix'](valid.copy(),feature_cols)
    assert retained==feature_cols
    y=prepared.to_numpy(float)
    centroids,matrix,overall,inclusion,fc=centroid_tables(meta,x,threshold); pairs=pair_tables(centroids,matrix)
    supported=centroids[centroids.adequate_support]
    keys=set(zip(supported.lineage,supported.tissue))
    keep=np.array([(r.priority_lineage,r.tissue) in keys for _,r in meta.iterrows()])
    sm=meta.loc[keep]; sy=y[keep]
    primary=meta.priority_lineage.notna(); cm=meta.loc[primary]
    protocol={'seed':SEED,'minimum_cells':20,'formal_group_minimum_donors':3,'classification':'5-fold GroupKFold by donor; RidgeClassifier alpha=1 class_weight=balanced; train-only median imputation and StandardScaler; no tuning; all five priority lineages passing n_cells>=20',
      'primary_confusion_panel':'Expanded clock chosen a priori for full clock-gene coverage; core also shown',
      'feature_sets':{k:env[k] for k in ['CORE_FEATURE_COLS','EXPANDED_FEATURE_COLS','CLOCK_FEATURE_COLS','MODULE_COLS']},
      'highdim_metrics':'Silhouette on original standardization across all 2030 eligible pseudobulks, evaluated on 1302 major-lineage observations. Ratio uses adequate lineage-tissue centroids and medians of tissue-matched between / within-lineage tissue distances.',
      'random_controls':{'status':'BLOCKED','requested_panels':1000,'matching':'expression mean + variance; optional detection; no selected-gene substitute'},
      'feature_model':'Existing unbalanced fixed-effects adjusted R2 differences: cell+tissue+donor vs each reduced model; interaction full cell*tissue+donor vs additive. All 2030 observations. Signed values retained.',
      'lineage_stability':'1 - pooled squared prediction error of leave-one-tissue-out training-lineage means / error of training grand mean. Supported 936 observations, five major lineages; can be negative.',
      'tissue_remodeling':'Macro-average across five lineages of adjusted R2(tissue+donor) - adjusted R2(donor), on supported observations; signed.',
      'uncertainty':'Delete-one donor cluster jackknife, t_(D-1) approximate 95% intervals:24 donor blocks for full model,18 present blocks for supported-lineage scores. Fixed original preprocessing/support. Not calibrated crossed-design inference; tissue holdouts separately reported.',
      'categories':'No binary feature categories assigned; continuous scores and deletion sensitivity only; no tuned cutoffs.',
      'bootstrap':'Reuse verified original 5000 crossed Exp(1) donor/tissue multiplier draws, seed20260913; original24 features; fixed adequate-support mask; approximate percentile CI, not independent pairs.',
      'phenotype':'For each supported donor x tissue x major-lineage aggregate, Euclidean distance in original24 standardized features from equal-pseudobulk lineage reference excluding its donor; also retain original cell-type-level phenotype.',
      'pathways':{'status':'BLOCKED','GMT':None,'screened_pathways':0,'planned_overlap_rule':'Exclude all 17 expanded-clock genes from pathway scores; report coverage; complete Hallmark primary and Reactome secondary'},
      'sensitivity':'Drop each donor and re-estimate means retaining original supported groups (minimum becomes2); drop each tissue and its pairs. No feature rescaling.'}
    write_json(OUT/'00_audit/prespecified_protocol.json',protocol)
    longpath=V1/'source_data/pseudobulk__clock_context_pseudobulk_long.tsv'
    long=pd.read_csv(longpath,sep='\t')
    inventory=[]
    for base in [ROOT,V1,V2,ROOT.parent/'TabulaCircadian']:
        if not base.exists():continue
        for p in base.rglob('*'):
            if not p.is_file() or any(v.startswith('.') or v=='__pycache__' for v in p.relative_to(base).parts):continue
            if OUT in p.parents:continue
            inventory.append({'path':str(p),'bytes':p.stat().st_size,'suffix':p.suffix})
    table('00_audit/local_file_inventory.csv',pd.DataFrame(inventory))
    genes=[c[5:] for c in raw.columns if c.startswith('expr_')]
    table('00_audit/available_gene_universe.csv',pd.DataFrame({'gene':genes,'preselected':True}))
    table('00_audit/observation_metadata.csv',valid[[c for c in valid.columns if c in ['donor_id','tissue','cell_type','broad_cell_class','n_cells','assays_present','sexes_present','development_stages','datasets_present','dataset_titles']]].join(meta.priority_lineage).reset_index())
    inputs=[FEATURE_PATH,STANDARDIZED,longpath,V2/'scripts/revision_v2_runner.py',V2/'statistics/formal_variance_partition_per_feature.tsv',V2/'statistics/priority_lineage_tissue_coverage.tsv',OLD/'clock_distance_bootstrap_draws.csv',OLD/'clock_distance_summary.csv',PCA_SOURCE]
    write_json(OUT/'00_audit/input_manifest.json',[{'path':str(p),'sha256':sha(p)} for p in inputs])
    audit=f'''# Phase A preflight audit

Input feature table: `{FEATURE_PATH}`: {raw.shape[0]} rows × {raw.shape[1]} columns, {len(genes)} preselected expression genes. Long table: `{longpath}`: {long.shape[0]} rows × {long.shape[1]} columns, {long.gene.nunique()} unique genes, {long.group_id.nunique()} groups. These are selected clock/context genes, not a genome-wide candidate universe.

Frozen pre-UMAP matrix: `{STANDARDIZED}`: {len(x)} observations × {x.shape[1]} features plus group_id. Exact numeric equality to original preprocessing verified. Original features: expanded17 expression genes + seven derived module/balance features. Core8; expanded17; standalone modules8 (includes core completeness at0.05). Union25 distinct features. No embedding coordinates used for inference.

Original normalized_mean originates from source X (raw_sum from raw.X); the upstream normalization algorithm is not reconstructible from these exports and is not redefined here. Original preparation median-imputes each feature, removes zero variance, then StandardScaler across all eligible observations. No additional transformation. CV alone fits imputation/scaling on training donors.

Verified n_cells≥20: {len(meta)} observations, {meta.donor_id.nunique()} donors, {meta.tissue.nunique()} tissues, {meta.cell_type.nunique()} original cell types. Five-lineage subset {len(cm)} observations. Formal ≥3-donor lineage×tissue subset {len(sm)} observations, {len(supported)} groups, {sm.tissue.nunique()} tissues, {sm.donor_id.nunique()} donors. Metadata and gene lists saved separately. Equal pseudobulk weights, not cell-number weights; multiple fine cell types from one donor remain dependent.

Recursive inventory covers workspace and delivered input directories, excluding dependency/cache internals. Requested legacy directory `TabulaCircadian/_deliverables/_v2` does not exist; actual V2 is `{V2}`. No local broad-expression matrix, single-cell object, or complete GMT found. Existing six represented pathway sets are preselected and ineligible for an unbiased screen. Prior archive inspection also found only the delivered files.

BLOCKED: A1 1000 matched random panels, their percentiles/P/Z and FigA1; A3 complete Hallmark/Reactome scoring, FDR/consistency and FigA6/A7. Completed locally: observed A1 metrics/confusions; A2 full-space distances, scores, uncertainty and driver checks; independent remodeling phenotype; descriptive A4. No remote access/download, original pipeline rerun, or Phase B.
'''
    (OUT/'00_audit/phaseA_preflight_audit.md').write_text(audit,encoding='utf-8')
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib']}
    write_json(OUT/'00_audit/software_versions.json',{'python':sys.version,**versions})
    print(f'Audit: {len(genes)} selected genes; broad controls/pathways blocked. Running valid components.',flush=True)
    # Donor-disjoint classification; no hyperparameter selection.
    splits=list(GroupKFold(n_splits=5).split(cm,cm.priority_lineage,cm.donor_id)); manifest=[]
    for fold,(tr,te) in enumerate(splits,1):
        assert not set(cm.iloc[tr].donor_id)&set(cm.iloc[te].donor_id)
        for pos in te:manifest.append({'group_id':cm.index[pos],'donor_id':cm.iloc[pos].donor_id,'test_fold':fold,'lineage':cm.iloc[pos].priority_lineage})
    table('01_matched_random_controls/classification_split_manifest.csv',pd.DataFrame(manifest))
    metrics=[]; predictions=[]; foldmetrics=[]; confusions={}
    for panel,key in [('Core clock','CORE_FEATURE_COLS'),('Expanded clock','EXPANDED_FEATURE_COLS')]:
        cols=env[key]; xx=valid.loc[cm.index,cols].to_numpy(); labels=cm.priority_lineage.to_numpy(); pred=np.empty(len(cm),dtype=object)
        for fold,(tr,te) in enumerate(splits,1):
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeClassifier(alpha=1,class_weight='balanced'))
            model.fit(xx[tr],labels[tr]); pred[te]=model.predict(xx[te])
            foldmetrics.append({'panel':panel,'fold':fold,'n_test':len(te),'balanced_accuracy':balanced_accuracy_score(labels[te],pred[te]),'macro_f1':f1_score(labels[te],pred[te],labels=LINEAGE_ORDER,average='macro',zero_division=0)})
        z=env['get_feature_matrix'](valid.copy().reset_index(),cols)[0]; z.index=valid.index
        ct,mat,*_=centroid_tables(meta,z,threshold); pp=pair_tables(ct,mat)
        ratio=pp.loc[pp.comparison_type.eq(BETWEEN),'distance'].median()/pp.loc[pp.comparison_type.eq(WITHIN),'distance'].median()
        metrics.append({'panel':panel,'n_observations':len(cm),'n_genes':len(cols),'balanced_accuracy':balanced_accuracy_score(labels,pred),'macro_f1':f1_score(labels,pred,average='macro'),'silhouette':silhouette_score(z.loc[cm.index],labels),'distance_ratio':ratio,'n_null_panels':0,'null_status':'BLOCKED: broad pseudobulk missing','null_mean':np.nan,'null_median':np.nan,'null_sd':np.nan,'empirical_percentile':np.nan,'empirical_P':np.nan,'Z':np.nan})
        predictions.extend({'group_id':g,'panel':panel,'observed':a,'predicted':b} for g,a,b in zip(cm.index,labels,pred))
        confusions[panel]=confusion_matrix(labels,pred,labels=LINEAGE_ORDER)
    metrics=pd.DataFrame(metrics); table('01_matched_random_controls/observed_metrics.csv',metrics)
    table('01_matched_random_controls/classification_fold_metrics.csv',pd.DataFrame(foldmetrics))
    table('source_data/FigA2_predictions.csv',pd.DataFrame(predictions))
    # Same formal model, rechecked against original saved results.
    point,ranks=effects(meta,y)
    reference=pd.read_csv(V2/'statistics/formal_variance_partition_per_feature.tsv',sep='\t').drop_duplicates('feature').set_index('feature')
    np.testing.assert_allclose(point,reference.loc[feature_cols,EFFECTS].to_numpy(),atol=2e-10,rtol=2e-10)
    stability_point=stability(sm,sy); remodel_point,bylineage=remodeling(sm,sy)
    draws=[]; score_draws=[]; donors=sorted(meta.donor_id.unique())
    cache=OUT/'02_backbone_remodeling/donor_deletion_cache.npz'
    if cache.exists():
        cached=np.load(cache); draws=cached['effects']; score_draws=cached['scores']
        assert cached['features'].tolist()==feature_cols and cached['donors'].tolist()==donors
    else:
        for i,donor in enumerate(donors):
            mask=meta.donor_id.ne(donor).to_numpy(); ds=sm.donor_id.ne(donor).to_numpy()
            draws.append(effects(meta.loc[mask],y[mask])[0])
            score_draws.append(np.column_stack([stability(sm.loc[ds],sy[ds]),remodeling(sm.loc[ds],sy[ds])[0]]))
            print(f'Donor deletion {i+1}/{len(donors)}',flush=True)
        draws=np.array(draws); score_draws=np.array(score_draws)
        np.savez_compressed(cache,effects=draws,scores=score_draws,features=feature_cols,donors=donors)
    lo,hi,se=jackknife_interval(point,draws)
    score_draws=score_draws[np.array([d in set(sm.donor_id) for d in donors])]
    feature_table=pd.DataFrame({'feature':feature_cols,'n_observations':len(meta),'n_donors':meta.donor_id.nunique(),'n_tissues':meta.tissue.nunique(),'main_rank':ranks[0],'interaction_rank':ranks[1]})
    feature_table['n_original_nonmissing']=valid[feature_cols].notna().sum().to_numpy()
    feature_table['score_n_observations']=len(sm)
    feature_table['score_n_donors']=sm.donor_id.nunique()
    feature_table['score_n_tissues']=sm.tissue.nunique()
    for j,name in enumerate(EFFECTS):
        feature_table[name]=point[:,j]; feature_table[name+'_ci_low']=lo[:,j]; feature_table[name+'_ci_high']=hi[:,j]
    for j,(name,value) in enumerate([('lineage_stability',stability_point),('tissue_remodeling',remodel_point)]):
        low,high,_=jackknife_interval(value,score_draws[:,:,j]); feature_table[name]=value; feature_table[name+'_ci_low']=low; feature_table[name+'_ci_high']=high
        feature_table[name+'_deletion_min']=score_draws[:,:,j].min(axis=0);feature_table[name+'_deletion_max']=score_draws[:,:,j].max(axis=0)
    table('statistics/feature_effects_and_scores.csv',feature_table); table('source_data/FigA5_feature_scores.csv',feature_table)
    table('02_backbone_remodeling/lineage_feature_tissue_effects.csv',pd.DataFrame([{'lineage':lin,'feature':f,'tissue_unique_adj_r2':bylineage[i,j]} for i,lin in enumerate(LINEAGE_ORDER) for j,f in enumerate(feature_cols)]))
    # Reuse independently verified full-feature bootstrap; check exact point distances.
    oldpairs=pd.read_csv(OLD/'clock_between_within_distances.csv'); np.testing.assert_allclose(pairs.distance,oldpairs.distance,atol=1e-12)
    boot=pd.read_csv(OLD/'clock_distance_bootstrap_draws.csv'); assert len(boot)==5000
    summary=pd.read_csv(OLD/'clock_distance_summary.csv')
    for col,kind in [('median_within',WITHIN),('median_between',BETWEEN),('delta','Delta: between − within')]:
        rr=summary[summary.comparison_type.eq(kind)].iloc[0]
        np.testing.assert_allclose(boot[col].quantile([.025,.975]),[rr.bootstrap_95_ci_low,rr.bootstrap_95_ci_high])
    table('source_data/FigA3_distances.csv',pairs); table('statistics/distance_summary.csv',summary)
    shutil.copy2(OLD/'clock_distance_bootstrap_draws.csv',OUT/'02_backbone_remodeling/clock_distance_bootstrap_draws.csv')
    plasticity=[]; displacement=[]
    for lin in LINEAGE_ORDER:
        ct=supported[supported.lineage.eq(lin)]; vectors=matrix[ct.index]; center=vectors.mean(axis=0); d=np.linalg.norm(vectors-center,axis=1)
        rr=summary[summary.lineage.eq(lin)].iloc[0]; inc=inclusion[inclusion.lineage.eq(lin)].iloc[0]
        plasticity.append({'lineage':lin,'supported_tissues':len(ct),'donors':int(inc.figure_B_donors),'observations':int(inc.figure_B_observations),'median_cross_tissue_distance':rr['median'],'ci_low':rr.bootstrap_95_ci_low,'ci_high':rr.bootstrap_95_ci_high,'median_centroid_displacement':float(np.median(d)),'rms_tissue_dispersion':float(np.sqrt(np.mean(d*d)))})
        displacement.extend({'lineage':lin,'tissue':t,'distance_to_equal_tissue_lineage_reference':v} for t,v in zip(ct.tissue,d))
    plasticity=pd.DataFrame(plasticity); table('source_data/FigA4_lineage_plasticity.csv',plasticity); table('02_backbone_remodeling/tissue_centroid_displacement.csv',pd.DataFrame(displacement))
    sensitivity=[]
    for donor in donors:
        mask=meta.donor_id.ne(donor); dd=distance_delta(meta.loc[mask],x.loc[mask],centroids,pairs)
        sensitivity.append({'deletion':'donor','identifier':donor,'delta':dd,'n_pairs':len(pairs)})
    for tissue in sorted(supported.tissue.unique()):
        pp=pairs[pairs.tissue_A.ne(tissue)&pairs.tissue_B.ne(tissue)]
        dd=pp.loc[pp.comparison_type.eq(BETWEEN),'distance'].median()-pp.loc[pp.comparison_type.eq(WITHIN),'distance'].median()
        sensitivity.append({'deletion':'tissue','identifier':tissue,'delta':dd,'n_pairs':len(pp)})
    sensitivity=pd.DataFrame(sensitivity);table('statistics/distance_driver_sensitivity.csv',sensitivity)
    # Pathway-independent phenotype at both original and requested aggregated level.
    phenotype=[]; aggregated=[]
    joined=sm.join(x)
    for lin in LINEAGE_ORDER:
        part=joined[joined.priority_lineage.eq(lin)]
        for donor,dp in part.groupby('donor_id'):
            reference=part[part.donor_id.ne(donor)][list(x.columns)].mean().to_numpy()
            for gid,r in dp.iterrows():phenotype.append({'group_id':gid,'lineage':lin,'donor_id':donor,'tissue':r.tissue,'cell_type':r.cell_type,'n_cells':r.n_cells,'distance':np.linalg.norm(r[list(x.columns)].to_numpy(float)-reference)})
            for tissue,tp in dp.groupby('tissue'):
                aggregated.append({'lineage':lin,'donor_id':donor,'tissue':tissue,'n_pseudobulks':len(tp),'n_cells':int(tp.n_cells.sum()),'distance':np.linalg.norm(tp[list(x.columns)].mean().to_numpy()-reference)})
    table('03_unbiased_pathways/clock_remodeling_phenotype_original_observations.csv',pd.DataFrame(phenotype));table('03_unbiased_pathways/clock_remodeling_phenotype_donor_tissue_lineage.csv',pd.DataFrame(aggregated))
    blocked=pd.DataFrame([{'analysis':name,'status':'BLOCKED','reason':'Broad gene-level pseudobulk missing'+('; complete versioned GMT missing' if 'pathway' in name.lower() else ''),'n_completed':0} for name in ['1000 matched random panels per clock panel','Complete Hallmark pathway screen','Complete Reactome pathway screen']])
    table('statistics/blocked_analyses.csv',blocked)
    hypotheses=pd.DataFrame({'pathway':['Hypoxia','Fatty acid metabolism','IFN-gamma response','TNF-alpha/NF-kB','ECM organization'],'status':'NOT TESTED','rank':np.nan,'effect':np.nan,'FDR':np.nan})
    table('03_unbiased_pathways/hypothesized_pathway_status.csv',hypotheses)
    print('Valid analyses complete; rendering five figures.',flush=True)
    figures=render(confusions,pairs,summary,plasticity,feature_table,metrics)
    for path in (OUT/'figures').glob('*.png'):
        with Image.open(path) as im:
            assert min(im.info.get('dpi',(0,0)))>=299
            assert im.width>1000 and im.height>1000
    report(metrics,summary,plasticity,feature_table,sensitivity,inputs,len(genes),len(meta),len(sm),len(aggregated))
    assert before==snapshot(); assert existing=={str(p):sha(p) for p in (ROOT/'figures_revised').rglob('*') if p.is_file()}
    write_json(OUT/'00_audit/run_validation.json',{'input_verification':verification,'figure_checks':figures,'old_source_files_unchanged':len(before),'existing_figure_files_unchanged':len(existing),'formal_effects_reproduced_at_tolerance':2e-10,'bootstrap_draws':len(boot),'new_png_programmatic_checks':'PASS','visual_review':'pending','phase_B':'NOT STARTED'})
    print(metrics.to_string(index=False)); print(plasticity.to_string(index=False));print('Phase A local components finished.',flush=True)

def render(confusions,pairs,summary,plasticity,features,metrics):
    apply_publication_style(); audits=[]
    def save(fig,stem):audits.append(save_figure(fig,stem,OUT/'figures',dpi=300))
    fig,axes=plt.subplots(1,2,figsize=(12,6.8));fig.subplots_adjust(left=.18,right=.96,bottom=.28,top=.81,wspace=.25)
    labels=['B cell','T cell','Macrophage/\nmonocyte','Fibroblast','Endothelial']
    for i,(name,counts) in enumerate(confusions.items()):
        ax=axes[i]; norm=counts/counts.sum(axis=1,keepdims=True); ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for a,b in itertools.product(range(5),repeat=2):ax.text(b,a,f'{norm[a,b]:.0%}\n({counts[a,b]})',ha='center',va='center',fontsize=8,color='white' if norm[a,b]>.55 else '#222222')
        ax.set_xticks(range(5),labels,rotation=40,ha='right'); ax.set_yticks(range(5),labels if i==0 else ['']*5);ax.set_title(name);clean_axis(ax,True)
        for tick,lin in zip(ax.get_xticklabels(),LINEAGE_ORDER):tick.set_color(LINEAGE_COLORS[lin])
        if i==0:
            for tick,lin in zip(ax.get_yticklabels(),LINEAGE_ORDER):tick.set_color(LINEAGE_COLORS[lin])
            ax.set_ylabel('Observed lineage')
        ax.set_xlabel('Predicted lineage')
    fig.suptitle('Clock-gene lineage classification · held-out donors',y=.96)
    fig.text(.5,.08,'Five donor-disjoint folds · row percentages and observation counts\nExpanded clock is the prespecified primary panel; matched-random comparison unavailable.',ha='center',fontsize=9)
    save(fig,'FigA2_clock_lineage_classifier_confusion')
    save(figure_B(pairs,summary),'FigA3_clock_architecture_between_vs_within_distance')
    fig=plt.figure(figsize=(10,6));ax=fig.add_axes([.26,.19,.64,.62]);clean_axis(ax)
    for i,r in plasticity.iterrows():
        ax.plot([r.ci_low,r.ci_high],[i,i],color=LINEAGE_COLORS[r.lineage],lw=2)
        ax.scatter(r.median_cross_tissue_distance,i,c=LINEAGE_COLORS[r.lineage],s=55,zorder=3)
        ax.text(.98,i+.20,f'{r.supported_tissues} tissues · {r.donors} donors',transform=ax.get_yaxis_transform(),ha='right',fontsize=8,color='#555555')
    ax.set_yticks(range(5),[DISPLAY_LABELS[l] for l in LINEAGE_ORDER]);ax.set(xlabel='Median cross-tissue distance (approximate 95% interval)',xlim=(0,7),ylim=(4.6,-.5))
    for tick,lin in zip(ax.get_yticklabels(),LINEAGE_ORDER):tick.set_color(LINEAGE_COLORS[lin])
    fig.suptitle('Lineage-specific tissue plasticity of clock architecture',y=.96,fontsize=13)
    fig.text(.5,.06,'Original 24-feature space · ≥3 donors per group · 5,000 shared donor/tissue bootstrap draws',ha='center',fontsize=9)
    save(fig,'FigA4_lineage_specific_tissue_plasticity')
    fig=plt.figure(figsize=(9.5,7));ax=fig.add_axes([.12,.22,.82,.60]);clean_axis(ax)
    # Limited labels: top three stability and top three remodeling scores, selected by declared rule.
    selected=set(features.nlargest(3,'lineage_stability').feature)|set(features.nlargest(3,'tissue_remodeling').feature)
    core=set(original_spec()['CORE_FEATURE_COLS'])
    for _,r in features.iterrows():
        color='#4E79A7' if r.feature in core else '#E15759' if r.feature.startswith('expr_') else '#59A14F'
        ax.errorbar(r.lineage_stability,r.tissue_remodeling,xerr=[[r.lineage_stability-r.lineage_stability_ci_low],[r.lineage_stability_ci_high-r.lineage_stability]],yerr=[[r.tissue_remodeling-r.tissue_remodeling_ci_low],[r.tissue_remodeling_ci_high-r.tissue_remodeling]],fmt='o',color=color,alpha=.65,ms=6,elinewidth=.6)
        if r.feature in selected:ax.annotate(r.feature.replace('expr_','').replace('_',' '),(r.lineage_stability,r.tissue_remodeling),xytext=(7,7),textcoords='offset points',fontsize=8)
    ax.axhline(0,color='#bbbbbb',lw=.6);ax.axvline(0,color='#bbbbbb',lw=.6)
    ax.set(xlabel='Lineage stability: tissue-held-out prediction improvement',ylabel='Tissue remodeling: within-lineage unique adjusted R²')
    fig.suptitle('Conserved lineage information and tissue remodeling',y=.96,fontsize=13)
    fig.text(.5,.89,'Continuous scores · signed effects · approximate donor-jackknife 95% intervals',ha='center',fontsize=9)
    handles=[Line2D([],[],marker='o',ls='',color=c,label=t) for c,t in [('#4E79A7','Core clock genes'),('#E15759','Auxiliary clock genes'),('#59A14F','Clock modules')]]
    lax=fig.add_axes([.14,.085,.76,.07]);place_external_legend(lax,handles,ncol=3,loc='center')
    fig.text(.5,.035,'No binary cutoffs. High stability and remodeling may coexist; these scores have different estimands.',ha='center',fontsize=9)
    save(fig,'FigA5_conserved_backbone_vs_context_remodeling')
    effect=summary[summary.summary_type.eq('effect')].iloc[0]
    model=pd.DataFrame([{'component':'Lineage-associated architecture','status':'SUPPORTED DESCRIPTIVELY','evidence':f'Between-minus-within distance {effect["median"]:.3f}; interval {effect.bootstrap_95_ci_low:.3f}–{effect.bootstrap_95_ci_high:.3f}'},
      {'component':'Tissue-associated remodeling','status':'SUPPORTED DESCRIPTIVELY','evidence':'Lineage-dependent cross-tissue distances and continuous feature effects'},
      {'component':'Clock-specific enrichment','status':'UNRESOLVED','evidence':'Matched-random controls blocked: broad pseudobulk missing'},
      {'component':'Associated biological pathways','status':'NOT TESTED','evidence':'Complete pathway screen blocked: broad matrix and GMT missing'}])
    table('source_data/FigA8_integrated_evidence.csv',model);table('04_integrated_model/evidence_summary.csv',model)
    fig=plt.figure(figsize=(10,6));fig.suptitle('Clock transcriptional architecture · internal evidence',y=.95,fontsize=14)
    for i,r in model.iterrows():
        col=i%2;row=i//2;ax=fig.add_axes([.07+.48*col,.55-.37*row,.40,.27]);ax.axis('off')
        ax.text(0,.95,r.component,fontsize=12,weight='bold',va='top');ax.text(0,.68,r.status,fontsize=10,color='#4E79A7' if i<2 else '#777777',va='top')
        import textwrap
        ax.text(0,.43,textwrap.fill(r.evidence,48),fontsize=10,va='top',linespacing=1.5)
    fig.text(.5,.075,'PARTIALLY SUPPORTED · PAUSE BEFORE REPLICATION\nStatic associations; no causal or time-dependent inference.',ha='center',fontsize=11,linespacing=1.6)
    save(fig,'FigA8_integrated_clock_architecture_model')
    table('source_data/figure_manifest.csv',pd.DataFrame([{'figure':f'FigA{i}','status':'BLOCKED' if i in [1,6,7] else 'GENERATED','reason':'Missing broad pseudobulk / complete GMT where applicable' if i in [1,6,7] else ''} for i in range(1,9)]))
    return audits

def report(metrics,summary,plasticity,features,sensitivity,inputs,ngenes,n,ns,nagg):
    def md(df):return df.to_markdown(index=False,floatfmt='.4f')
    # to_markdown dependency may not be present: keep a plain fenced CSV representation instead.
    def show(df):return '```csv\n'+df.to_csv(index=False,float_format='%.5g',lineterminator='\n')+'```'
    strongest=features.nlargest(5,'lineage_stability')[['feature','lineage_stability','lineage_stability_ci_low','lineage_stability_ci_high']]
    remodel=features.nlargest(5,'tissue_remodeling')[['feature','tissue_remodeling','tissue_remodeling_ci_low','tissue_remodeling_ci_high']]
    text=f'''# PHASE A GATE REPORT

Status: LOCAL COMPONENTS COMPLETE; A1 matched null and A3 pathway screen BLOCKED. Model PARTIALLY SUPPORTED. Decision PAUSE BEFORE REPLICATION. Phase B not started.

1. Observations: {n} donor×tissue×original-cell-type pseudobulks after original ≥20-cell filter; 1302 five-lineage observations for donor-disjoint classification; {ns} observations in 97 ≥3-donor lineage×tissue groups for formal major-lineage stability/remodeling. Five lineages, 29 supported tissues. Full feature-effect model uses all2030, 161 original cell types, 74 tissues, 24 donors. No single cells analyzed.

2. Core8 genes; expanded17 genes; original24 features (17 expression +7 modules); module-only8; union25 modeled features. Definitions and exact input hashes in audit files.

3–4. Core and expanded observed metrics:
{show(metrics)}
For BOTH panels, balanced-accuracy, silhouette and distance-ratio null percentile/P = N/A; null panels0/1000; no random mean/SD/Z. Only {ngenes} preselected genes available. Cannot claim enrichment above matched random expectation. Classification uses five donor-disjoint folds, fixed balanced ridge classifier, train-only preprocessing, no tuning. Pooled out-of-fold metrics are descriptive and not cell-independent uncertainty estimates. Expanded panel primary choice was specified before computing results, not selected for highest test accuracy.

5. Highest tissue-held-out lineage stability scores (continuous, not binary conserved classes):
{show(strongest)}
6. Highest macro-average within-lineage tissue unique adjusted R² scores:
{show(remodel)}
All25 features' cell identity/tissue/donor/interaction adjusted R² contributions and approximate donor-jackknife intervals are in statistics/feature_effects_and_scores.csv. Original unbalanced model reproduced within2e-10. Fine-cell-type effects and five-lineage scores are distinct estimands. Negative adjusted effects/intervals retained. Main/interaction ranks257/971. Original modules overlap and do not constitute independent evidence. Existing donor/tissue imbalance prevents causal separation of all effects.

7. Full24-feature Euclidean distances and 5000 crossed donor/tissue multiplier intervals:
{show(summary[summary.summary_type.ne('lineage_within')])}
928 within-lineage cross-tissue and139 between-lineage same-tissue pairs; pairs dependent. No independent-pair P test. Bootstrap retains original support mask and shares donor/tissue weights; percentile intervals approximate for nonlinear median contrast, not calibrated exact confidence guarantees. Standardization remains fixed.

8. Lineage-specific plasticity:
{show(plasticity)}
Centroid displacement and RMS dispersion use an equal-tissue reference across each lineage's supported tissue centroids. This differs explicitly from the previous all-observation lineage reference. Median pair-distance intervals reuse original dependence-aware bootstrap; no interval claimed for displacement/RMS.

9. Fibroblast has the largest point median cross-tissue distance; B cell the smallest. Intervals overlap; no formal ranked-lineage difference test or amplitude interpretation. Driver sensitivity:
{show(sensitivity.groupby('deletion').delta.agg(['count','min','max']).reset_index())}
Every donor/tissue deletion's delta and pair count saved. Donor deletion fixes original adequate-group mask (remaining support≥2); tissue deletion removes all incident pairs. These checks test a single-driver explanation for distance, not clock specificity. They do not prove full transferability across cohorts.

10. Complete Hallmark screen: NOT RUN, zero pathways tested; no top-ranked pathways or FDR available. Reactome likewise. Broad expression matrix and complete versioned GMT absent; six preselected represented sets are not a substitute.

11. Hypoxia: NOT TESTED; fatty acid metabolism: NOT TESTED; IFN-gamma: NOT TESTED; TNF/NF-kB: NOT TESTED; ECM: NOT TESTED. Rank/effect/FDR all N/A. YES/PARTIAL/NO would falsely imply a completed test, so no negative biological conclusion is drawn.

12. PARTIALLY SUPPORTED: static lineage-associated clock-feature organization and heterogeneous tissue-associated variation have quantitative internal evidence. Clock-specific enrichment and unbiased context-program associations remain unresolved.

13. Strongest alternative: generic cell-identity transcriptional information explains clock-gene separation; unbalanced donor/tissue sampling, fine-subtype composition and correlated module features may contribute. n_cells support and donor-disjoint validation mitigate some issues but do not resolve this alternative.

14. Matched random controls neither weaken nor strengthen that alternative: analysis blocked, not a failed or successful test.

15. Strongest claim: In the available Tabula Sapiens pseudobulks, predefined clock-associated expression features contain donor-held-out lineage information, and supported tissue centroids show lineage-dependent static architecture variation. This is an internal observational association.

16. Unsupported: clock specificity relative to matched genes; any complete pathway ranking/association; causal tissue regulation; temporal rhythmicity, amplitude or phase; independent replication. No collection-time labels. A pathway-independent remodeling phenotype is saved for {ns} original observations and {nagg} donor×tissue×lineage aggregates; reference excludes each target donor, but remains a descriptive phenotype with dependent references.

## Gate
PAUSE BEFORE REPLICATION: first required criterion (significant superiority to matched random panels on multiple metrics) cannot be evaluated. Do not start Phase B.

## Exact inputs
'''+ '\n'.join('- `'+str(p)+'`' for p in inputs)+'''

See 00_audit/prespecified_protocol.json for formulas, filtering, bootstrap and split choices; software_versions.json and input_manifest.json for versions/hashes. All saved figures have source tables; FigA1, FigA6, FigA7 remain explicitly blocked rather than generated from unsuitable data.
'''
    (OUT/'reports/PHASE_A_GATE_REPORT.md').write_text(text,encoding='utf-8')
    missing='''# Minimum missing dependencies

Supply an existing broad, post-QC gene-level donor × tissue × original-cell-type pseudobulk matrix from the same completed server analysis. It must align by group_id with these3592 exported groups (at minimum all2030 eligible groups), use the identical original normalized expression definition, include all retained candidate genes (not a clock/context subset), and include unique gene IDs/symbols, normalization provenance and original QC/exclusion rules. Include gene detection/prevalence when reliably available; it is optional for the primary mean+variance matching but needed to reproduce any detection QC. Metadata must retain donor_id, tissue, cell_type and n_cells. Preserve original aggregation and thresholds.

Also supply the complete locally used human MSigDB Hallmark GMT, and Reactome GMT for secondary analyses, with version/source/license provenance. No selected represented-gene lists. Complete collections must compete before prioritization; remove expanded-clock overlap from pathway scoring and document retained coverage.

Minimum transfer example (ONLY after identifying an existing server-wide export; placeholders must be replaced, command not executed):
```powershell
scp <server>:<existing-broad-pseudobulk-export> <local-input-directory>
scp <server>:<existing-versioned-hallmark.gmt> <local-input-directory>
scp <server>:<existing-versioned-reactome.gmt> <local-input-directory>
```
There is no verified existing path for the broad export. Do not pretend the selected `clock_context_pseudobulk_long.tsv` is sufficient. If only the original server AnnData exists, minimum additional server computation is a batched export of normalized X gene sums/means (plus detection when available) for the EXACT existing group_id membership; do not refit embeddings or alter normalization. Its known provenance path is `/autodl-fs/data/TabulaCircadian/raw/cellxgene_h5ad/TS_All_Cells__53d208b0-2cfd-4366-9866-c3c6114081bc.h5ad`, but it has not been accessed/verified in this Phase A. A safe exact exporter requires the original membership/QC mapping and normalization provenance; these cannot be inferred from selected genes alone.

Once supplied, implement/run the blocked1000 matched panels and complete pathway screen against frozen protocol, then reevaluate the gate. The present local script deliberately does not implement a fake null/pathway substitute and will not auto-unblock merely upon adding unrelated files.
'''
    (OUT/'reports/MISSING_DATA_REQUIREMENTS.md').write_text(missing,encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):run()

