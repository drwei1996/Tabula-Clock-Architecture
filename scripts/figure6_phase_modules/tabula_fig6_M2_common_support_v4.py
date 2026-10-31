#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 6 V4: newly designed fixed covariance-aware null panels + fail-closed
common-support evaluation of the M2-M1 increment.

SOURCE: frozen B0 Tabula, V2.1 target/null scores and V3 conditional test.
OBSERVATIONAL and POST-SELECTION; output-derived within-block variance used for
screening makes empirical ranks exploratory, NOT calibrated frequentist p-values.
No B0 refit. Never edits prior outputs. Each candidate gene set is fixed for all
cohorts before pathway-score-based screening.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, traceback, time
from datetime import datetime
from pathlib import Path
from argparse import Namespace
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

VERSION='FIG6_M2_COMMON_SUPPORT_V4_20261013'
KEY=['tissue','assay','donor','lineage','phase_bin','phase_center_h']

def log(*x):print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def req(b,msg):
 if not b:raise RuntimeError(msg)
def jread(p):return json.loads(Path(p).read_text(encoding='utf8'))
def jwrite(p,data):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8');os.replace(t,p)
def put(df,p):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=Path(str(p)+'.tmp')
 df.to_csv(tmp,index=False,compression='gzip' if p.name.endswith('.gz') else None);os.replace(tmp,p)
def sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def align_keys(frame):
 v=frame[KEY].copy()
 for c in ['tissue','assay','donor','lineage']:v[c]=v[c].astype(str)
 v['phase_bin']=v.phase_bin.astype(int)
 v['phase_center_h']=v.phase_center_h.astype(float).round(6)
 return pd.MultiIndex.from_frame(v)

def load(root,args):
 path=root/'tabula_fig6_M2_conditional_null_cv_v3.py'
 req(path.exists(),'Missing tabula_fig6_M2_conditional_null_cv_v3.py in project root')
 import importlib.util
 spec=importlib.util.spec_from_file_location('fig6_common_support_v3',path)
 v3=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v3;spec.loader.exec_module(v3)
 v2=root/args.source
 v3dir=root/args.v3
 req((v3dir/'02_CONDITIONAL_M2_RESULTS.csv').exists(),'V3 conditional M2 file missing')
 env=v3.load_env(root,v2)
 old=pd.read_csv(v3dir/'02_CONDITIONAL_M2_RESULTS.csv')
 req(len(old)==36 and set(old['mode'])=={'baseline','adjusted'},'Unexpected V3 conditional results')
 req(len(env['design']['targets'])==18 and len(env['nulls'])==3582,'Expected same 18 targets / 3582 fixed controls')
 return v3,env,old

def corr_matrix_for_gene_panels(args,env,v3,root,out):
 """Reuse the V3 strictly aligned H5AD sampling; use one identical correlation
 estimator for TARGET, old controls AND new controls. This is descriptive RNA
 covariance, not phase-free independent data.
 """
 args3=Namespace(sample_cells=args.sample_cells,h5ad_block=args.h5ad_block,seed=args.seed,
                 min_pair_cells=80,pairs_per_set=48,pair_chunk=64)
 X,genes,coverage=v3.sample_for_covariance(args3,env,root,out)
 req(len(genes)>=1000 and X.shape[0]>=300,'Insufficient cells/genes for common covariance screen')
 rng=np.random.default_rng(args.seed)
 if len(X)>args.corr_cells:X=X[np.sort(rng.choice(len(X),args.corr_cells,replace=False))].copy()
 else:X=X.copy()
 log('COV sample',X.shape,'estimator: cohort-lineage centered, pooled Pearson')
 idx={g:i for i,g in enumerate(genes)}
 Z=np.asarray(X,dtype=np.float32);del X
 finite=np.isfinite(Z)
 ct=finite.sum(axis=0)
 # Missing values are 0 after column centering; retain only well-supported genes.
 Z=np.where(finite,Z,0.).astype(np.float32);del finite
 av=Z.sum(axis=0,dtype=np.float64)/np.maximum(ct,1)
 Z-=av.astype(np.float32)[None,:]*(Z!=0.)
 dd=np.sqrt((Z*Z).sum(axis=0,dtype=np.float64))
 good=(ct>=min(80,max(25,int(.3*len(Z)))))&(dd>1e-7)
 Z[:,good]/=dd[good].astype(np.float32)[None,:]
 Z[:,~good]=0.
 # Gram-matrix calculation is bounded: 7000 genes ~196 MB of float32.
 C=(Z.T@Z).astype(np.float32);del Z;gc.collect()
 np.fill_diagonal(C,1.)
 req(np.isfinite(C).all(),'Nonfinite pooled gene correlations')
 return C,idx,good

def pair_metrics(genes,C,idx,good,maximum_pairs=0):
 ids=np.array([idx[g] for g in genes if g in idx and good[idx[g]]],dtype=int)
 if len(ids)<5:return (np.nan,np.nan,int(len(ids)),0)
 block=C[np.ix_(ids,ids)]
 r=block[np.triu_indices(len(ids),k=1)];r=r[np.isfinite(r)]
 if maximum_pairs and len(r)>maximum_pairs:
  # This is not used for common-support inference: use all pairs by default.
  r=r[:maximum_pairs]
 return (float(np.median(np.abs(r))),float(np.sqrt(np.mean(r*r))),int(len(ids)),int(len(r)))

def propose(args,env,v3,root,out):
 genefile=out/'01_NEW_FIXED_NULL_GENE_SETS.json'
 if args.resume and genefile.exists() and (out/'01_COMMON_CORRELATION_AUDIT.csv').exists():
  log('DESIGN CACHE',genefile);return
 req(not genefile.exists() and not (out/'01_COMMON_CORRELATION_AUDIT.csv').exists(),
     'Partial design exists; use a fresh --output-dir (never mix signatures).')
 C,idx,good=corr_matrix_for_gene_panels(args,env,v3,root,out)
 design=env['design'];pool=[g for g in design['pool'] if g in idx and good[idx[g]]]
 prof=pd.read_csv(root/args.source/'01_GLOBAL_GENE_MATCHING_FEATURES.csv.gz').set_index('gene')
 features=['mean_logCP10k','sd_logCP10k','detect_rate','lineage_spread']
 pool=[g for g in pool if g in prof.index and np.isfinite(prof.loc[g,features].to_numpy(float)).all()]
 req(len(pool)>=500,f'Too few background genes after covariance QC: {len(pool)}')
 pp=prof.loc[pool,features].to_numpy(float)
 scale=np.maximum(np.quantile(pp,.75,axis=0)-np.quantile(pp,.25,axis=0),.025)
 pp=(pp-np.median(pp,axis=0))/scale
 tree=cKDTree(pp)
 poolids=np.array([idx[g] for g in pool])
 rng=np.random.default_rng(args.seed+667)
 nulls={};audit=[];target_lists={}
 for ti,(target,genes) in enumerate(design['targets'].items()):
  tg=[g for g in genes if g in idx and good[idx[g]] and g in prof.index]
  req(len(tg)>=10,f'{target}: <10 covariance-eligible genes')
  target_lists[target]=tg
  tt=(prof.loc[tg,features].to_numpy(float)-np.median(prof.loc[pool,features],axis=0))/scale
  _,nn=tree.query(tt,k=min(args.neighbors,len(pool)))
  nn=np.atleast_2d(nn)
  if nn.shape[0]!=len(tg):nn=nn.reshape(len(tg),-1)
  targetids=np.array([idx[g] for g in tg]);tcor=C[np.ix_(targetids,targetids)]
  # Randomized order; multi-objective greedy selection using MATCHED gene-level
  # expression features AND signed correlation to previously assigned targets.
  for rep in range(args.new_per_target):
   used=set();select=[-1]*len(tg)
   order=rng.permutation(len(tg))
   for pos,k in enumerate(order):
    possibilities=np.asarray([int(z) for z in nn[k] if int(z) not in used],dtype=int)
    if not len(possibilities):
     _,expanded=tree.query(tt[k],k=min(len(pool),max(args.neighbors*4,len(tg)+30)))
     possibilities=np.asarray([int(z) for z in np.atleast_1d(expanded) if int(z) not in used],dtype=int)
    req(len(possibilities)>0,'Cannot allocate distinct null genes')
    gene_dist=np.linalg.norm(pp[possibilities]-tt[k],axis=1)
    if pos:
     prior=order[:pos];assigned=np.array([poolids[select[j]] for j in prior],dtype=int)
     actual=tcor[k,prior]
     # Signed correlation matching; avoids exclusively matching strong positive
     # correlations when the target includes anticorrelated genes.
     deviation=np.sqrt(np.mean((C[np.ix_(poolids[possibilities],assigned)]-actual[None,:])**2,axis=1))
     objective=gene_dist+.65*deviation/max(args.corr_penalty_scale,1e-4)
    else:objective=gene_dist
    top=np.argsort(objective)[:min(args.selection_top,len(objective))]
    w=np.exp(-(objective[top]-objective[top].min())/max(args.selection_temperature,1e-4))
    w/=w.sum();pick=int(rng.choice(possibilities[top],p=w))
    select[k]=pick;used.add(pick)
   nm=f'V4NULL__{ti:02d}__{rep:03d}'
   picked=[pool[p] for p in select]
   req(len(set(picked))==len(tg),'Duplicate genes within newly designed null')
   nulls[nm]=picked
   audit.append(dict(pathway=target,null=nm,n_target_profile_genes=len(tg),n_null_genes=len(picked)))
  log('DESIGN',ti+1,'/18',target,'new fixed nulls',args.new_per_target)
 # Same identical correlation estimator for TARGET/OLD/NEW gene sets.
 pairs={**design['targets'],**env['nulls'],**nulls}
 records=[]
 for k,(nm,genes) in enumerate(pairs.items(),1):
  ma,rms,n,pr=pair_metrics(genes,C,idx,good)
  records.append(dict(pathway=nm,median_abs_gene_corr=ma,rms_gene_corr=rms,
                      n_cov_eligible_genes=n,n_gene_pairs=pr,n_original_genes=len(genes),
                      kind='Target' if nm in design['targets'] else 'OldNull' if nm in env['nulls'] else 'NewNull'))
  if k%750==0:log('PANEL COV AUDIT',k,'/',len(pairs))
 corr=pd.DataFrame(records)
 req(corr[corr.kind=='Target'][['median_abs_gene_corr','rms_gene_corr']].notna().all().all(),
     'Target covariance estimation incomplete')
 put(corr,out/'01_COMMON_CORRELATION_AUDIT.csv')
 put(pd.DataFrame(audit),out/'01_NEW_NULL_MAPPING.csv')
 jwrite(genefile,dict(version=VERSION,seed=args.seed,sets=nulls,
                      targets_covariance_genes=target_lists,
                      design_parameters=dict(new_per_target=args.new_per_target,
                      neighbors=args.neighbors,corr_cells=args.corr_cells,correlation_penalty=args.corr_penalty_scale,
                      sample_cells=args.sample_cells,seed=args.seed)))
 del C;gc.collect()
 log('DESIGN COMPLETE',len(nulls),'additional cohort-fixed candidate gene sets')

def score(args,env,v3,root,out):
 dest=out/'02_NEW_ATLAS_UNIT_METADATA.csv.gz'
 if args.resume and dest.exists() and (out/'02_NEW_SCORES_WIDE.npz').exists() and (out/'02_TARGET_SCORE_REPRODUCTION.csv').exists():
  log('SCORE CACHE',dest);return
 new=jread(out/'01_NEW_FIXED_NULL_GENE_SETS.json')['sets']
 o=env['o'];v2=env['v2'];phase=env['phase'];orig=env['orig'];helper=env['helper']
 sets={**env['design']['targets'],**env['design']['state_markers'],**new}
 union=sorted(set().union(*map(set,sets.values())))
 manifest,oldmeta,_=v2.get_manifest(root,root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1',o)
 req(len(manifest)==env['design']['n_informative_cohorts'], 'Cohort manifest changed')
 sig=sha(dict(sets=sets,version=VERSION,R=.3,detect=2,bin=8,bin_eff=8,source=jread(root/args.source/'03_SCORE_SIGNATURE.json').get('signature')))
 oldsig=out/'02_EXTRACTION_SIGNATURE.json'
 if oldsig.exists():req(jread(oldsig)['signature']==sig,'Cache signature mismatch; choose fresh directory')
 else:jwrite(oldsig,dict(signature=sig,n_cohorts=len(manifest)))
 op=Namespace(resume=args.resume,min_R=.3,min_detect=2,min_coverage=.45,
              min_bin_weight=8,min_bin_effective=8,min_cells=50,h5ad_block=args.h5ad_block,
              extra_background_genes=0,matched_nulls=0,seed=args.seed)
 cache=out/'02_CACHE_PER_COHORT';cache.mkdir(exist_ok=True)
 aud=[]
 for j,(_,row) in enumerate(manifest.iterrows(),1):
  item=o.extract_cohort_fast(root,orig,phase,helper,row,sets,union,[],op,sig,cache)
  req(item['status']=='SUCCESS',f'V4 score failed cohort {row.tissue_general} {row.assay}: {item}')
  aud.append({k:v for k,v in item.items() if k!='null_audit'})
  put(pd.DataFrame(aud),out/'02_EXTRACTION_AUDIT.csv')
  log('SCORED',j,'/',len(manifest),str(row.tissue_general),str(row.assay))
 m,A,names=o.assemble_cached(cache,sets,[],out,sig)
 aa=align_keys(m);bb=align_keys(env['meta'])
 req(aa.is_unique and bb.is_unique and aa.difference(bb).empty and bb.difference(aa).empty,
     'V4 aggregation unit keys differ from frozen V2 (fail closed)')
 ix=bb.get_indexer(aa)
 checks=[]
 for target in env['design']['targets']:
  req(target in names and target in env['index'],f'Missing target: {target}')
  x=A[:,names.index(target)].astype(float)
  y=env['arr'][ix,env['index'][target]].astype(float)
  miss=int((np.isfinite(x)!=np.isfinite(y)).sum())
  v=np.isfinite(x)&np.isfinite(y)
  diff=float(np.max(np.abs(x[v]-y[v]))) if v.any() else float('inf')
  checks.append(dict(pathway=target,unmatched_rows=miss,max_abs_diff=diff,n_rows=int(v.sum())))
 req(all(z['unmatched_rows']==0 and z['max_abs_diff']<=1e-4 for z in checks),
     'V4 target scores failed frozen V2 reproduction: no inferential outputs allowed')
 put(pd.DataFrame(checks),out/'02_TARGET_SCORE_REPRODUCTION.csv')
 # Avoid overwriting original V2 matrix; V4 source assembles into V4 directory.
 put(m,dest)
 tmp=out/'02_NEW_SCORES_WIDE.npz.tmp'
 with open(tmp,'wb') as file:np.savez_compressed(file,scores=A,pathway=np.array(names))
 os.replace(tmp,out/'02_NEW_SCORES_WIDE.npz')
 log('SCORES COMPLETE',len(m),'frozen units',len(new),'new fixed sets')

def bounded_var(meta,x):
 z=np.asarray(x,float);ok=np.isfinite(z)
 if ok.sum()<32:return np.nan
 d=meta.loc[ok,['tissue','assay','donor','lineage','n_eff']].copy()
 vals=z[ok];key=d[['tissue','assay','donor','lineage']].astype(str).agg('|'.join,axis=1)
 mu=pd.Series(vals).groupby(key.to_numpy()).transform('mean').to_numpy(float)
 w=np.minimum(np.maximum(d.n_eff.to_numpy(float),1),100)
 return float(np.average((vals-mu)**2,weights=w))

def support_mask(t,refs,args):
 """NO nearest-K fallback. Absolute and multiplicative *joint* calipers,
 preceded by target-within-range checks (positivity). All dimensions must pass.
 """
 m=np.ones(len(refs),dtype=bool);reasons=[]
 # Marginal range checks are for continuous shape/variance metrics ONLY.
 # Gene count can differ slightly (e.g. 199 vs 200) and is handled by its
 # explicit fixed caliper; exact extreme-value range there would be spurious.
 dims=['median_abs_gene_corr','rms_gene_corr','within_block_variance']
 for c in dims:
  val=float(t[c]);vv=refs[c].to_numpy(float)
  ok=np.isfinite(vv)&np.isfinite(val)
  if not ok.any():reasons.append('INVALID_'+c);return m&False,reasons
  if val<np.nanmin(vv[ok])-1e-10 or val>np.nanmax(vv[ok])+1e-10:reasons.append('OUTSIDE_RANGE_'+c)
  m &= ok
 c='n_original_genes';m &= np.abs(refs[c].to_numpy(float)-float(t[c])) <= max(2,args.gene_tolerance*float(t[c]))
 c='n_finite_units';m &= np.abs(refs[c].to_numpy(float)-float(t[c])) <= args.unit_tolerance
 for c,atol,rtol in [('median_abs_gene_corr',args.median_corr_atol,args.corr_relative_tol),
                     ('rms_gene_corr',args.rms_corr_atol,args.corr_relative_tol)]:
  x=refs[c].to_numpy(float);v=float(t[c]);m &= np.abs(x-v) <= np.maximum(atol,rtol*max(abs(v),1e-4))
 x=refs.within_block_variance.to_numpy(float);v=float(t.within_block_variance)
 m &= (x>0)&(v>0)&(np.abs(np.log(np.maximum(x,1e-15)/max(v,1e-15)))<=np.log(args.variance_ratio))
 return m,reasons

def paired_fits(env,v3,source,oldmeta,oldarr,oldnames,target,null,mode,fields):
 """Exact same donor/lineage/phase-bin rows for target and null;
 ONLY M2-M1 increment tested. Do not compare unpaired R2.
 """
 arr=source['arr'];meta=source['meta'];idx=source['idx'];o=env['o']
 req(target in idx and null in idx, 'Score missing for '+target+' '+null)
 take=np.isfinite(arr[:,idx[target]]) & np.isfinite(arr[:,idx[null]])
 for f in fields:take &= np.isfinite(meta[f].to_numpy(float))
 if take.sum()<100:return None
 cols=KEY+['n_eff'];d=meta.loc[take,cols].copy()
 if mode=='adjusted':
  for j,f in enumerate(fields):d['state_'+str(j)]=meta.loc[take,f].to_numpy(float)
 d['score']=arr[take,idx[target]].astype(float)
 a=o.build_design(d)
 d['score']=arr[take,idx[null]].astype(float)
 b=o.build_design(d)
 if a is None or b is None:return None
 fa=o.fit_models(a);fb=o.fit_models(b)
 if fa is None or fb is None:return None
 return dict(target_delta=float(fa['deltaR2_M2_vs_M1']),null_delta=float(fb['deltaR2_M2_vs_M1']),n_paired_units=len(a['y']))

def evaluate(args,env,v3,old,root,out):
 out_file=out/'03_COMMON_SUPPORT_M2_RESULTS.csv'
 if args.resume and out_file.exists() and (out/'03_SUPPORT_CANDIDATE_DETAILS.csv.gz').exists():
  log('EVALUATION CACHE',out_file);return
 oldcorr=pd.read_csv(out/'01_COMMON_CORRELATION_AUDIT.csv')
 # V4 recalculates original and new correlation features with SAME sample+estimator.
 matrix=np.load(out/'02_NEW_SCORES_WIDE.npz',allow_pickle=False)
 vnames=matrix['pathway'].astype(str).tolist();vmat=matrix['scores'];meta=pd.read_csv(out/'02_NEW_ATLAS_UNIT_METADATA.csv.gz')
 req(len(meta)==len(vmat)==len(env['meta']), 'V4 source length mismatch')
 a=align_keys(meta);b=align_keys(env['meta']);idxold=b.get_indexer(a)
 req((idxold>=0).all(),'Missing V2 atlas keys in V4')
 req(len(set(vnames))==len(vnames),'Duplicated V4 pathway columns')
 # Add original V2 fixed-null data to new V4 matrices; same ordering by keys.
 add=[nm for nm in env['nulls'] if nm not in vnames]
 oldscores=env['arr'][idxold,:][:,[env['index'][nm] for nm in add]]
 vmat=np.concatenate([vmat,oldscores],axis=1)
 names=vnames+add;idx={nm:i for i,nm in enumerate(names)}
 # Add target-disjoint RNA state proxy scores from frozen V2 (if absent).
 for name in env['design']['state_markers']:
  if name not in idx:
   idx[name]=len(names);names.append(name)
   vmat=np.column_stack((vmat,env['arr'][idxold,env['index'][name]]))
 for name in env['design']['state_markers']:
  meta['state_proxy_'+name]=vmat[:,idx[name]]
 source=dict(arr=vmat,meta=meta,idx=idx)
 vv=[]
 for nm in oldcorr.pathway:
  req(nm in idx,'Missing scored module '+nm)
  x=vmat[:,idx[nm]].astype(float)
  vv.append((bounded_var(meta,x),int(np.isfinite(x).sum())))
 oldcorr['within_block_variance']=[q[0] for q in vv]
 oldcorr['n_finite_units']=[q[1] for q in vv]
 oldcorr['source_stage']='V4 same-covariance-estimator + scores'
 put(oldcorr,out/'03_ALL_MODULE_JOINT_AUDIT.csv')
 records=[];details=[]
 targets=list(env['design']['targets'])
 for ti,target in enumerate(targets):
  masks=oldcorr[oldcorr.pathway==target]
  req(len(masks)==1,'Missing audited target '+target)
  t=masks.iloc[0]
  chosen=oldcorr[((oldcorr.kind=='OldNull')&(oldcorr.pathway.str.startswith(f'FIXNULL__{ti:02d}__')))|
                 ((oldcorr.kind=='NewNull')&(oldcorr.pathway.str.startswith(f'V4NULL__{ti:02d}__')))].copy()
  req(len(chosen)==199+args.new_per_target, 'Wrong target-specific null collection')
  support,violations=support_mask(t,chosen,args)
  chosen['joint_common_support']=support
  chosen['target']=target
  for c in ['median_abs_gene_corr','rms_gene_corr','within_block_variance','n_original_genes']:
   chosen['target_'+c]=t[c]
  details.append(chosen)
  eligible=chosen.loc[support].copy()
  for mode in ['baseline','adjusted']:
   row=old[(old.pathway==target)&(old['mode']==mode)]
   req(len(row)==1,'Missing prior conditional M2 row: '+target+' '+mode)
   entry=row.iloc[0]
   fields=str(env['fix'][(env['fix'].pathway==target)&(env['fix']['mode']=='disjoint_RNA_proxies')].iloc[0].state_fields).split(';')
   fields=['state_proxy_'+s.removeprefix('state_proxy_') for s in fields]
   req(len(fields)==2 and all(f in meta.columns for f in fields),'Missing target-specific RNA proxies')
   d=dict(pathway=target,mode=mode,old_v3_conditional_wild_p=float(entry.conditional_M1null_wild_p),
          old_v3_wild_BHq=float(entry.BHq_conditional_M1null_wild_p),
          n_old_null=199,n_new_null=args.new_per_target,n_total_candidates=len(chosen),
          n_joint_supported=len(eligible),outside_marginal_support=';'.join(violations),
          target_within_block_variance=float(t.within_block_variance),
          target_median_abs_gene_corr=float(t.median_abs_gene_corr),target_rms_gene_corr=float(t.rms_gene_corr))
   # Support failure is UNTESTABLE, NOT a nonsignificant empirical result.
   if violations or len(eligible)<args.min_supported:
    d.update(status='NO_COMMON_SUPPORT',empirical_rank_p=np.nan,
             comment='Target out of marginal null range OR fewer than min_support jointly comparable controls')
   else:
    pairs=[]
    for nm in eligible.pathway:
     q=paired_fits(env,v3,source,meta,vmat,names,target,nm,mode,fields)
     if q is not None:pairs.append(dict(null=nm,**q))
    if len(pairs)<args.min_supported:
     d.update(status='INSUFFICIENT_PAIRED_FITS',n_paired=len(pairs),empirical_rank_p=np.nan)
    else:
     q=pd.DataFrame(pairs)
     # Different paired support could alter the target delta; record range.
     spread=float(q.target_delta.max()-q.target_delta.min())
     # Conservative safety: if paired model's target score shifts appreciably,
     # do not compare unpaired empirical ranks.
     if spread>args.max_target_delta_spread:
      d.update(status='INCONSISTENT_PAIRED_UNIT_SUPPORT',n_paired=len(q),
               target_delta_spread=spread,empirical_rank_p=np.nan)
     else:
      p=(1+int((q.null_delta>=q.target_delta-1e-12).sum()))/(1+len(q))
      d.update(status='SUPPORTED_EXPLORATORY',n_paired=len(q),
       empirical_rank_p=float(p),target_delta_spread=spread,
       observed_target_delta_median=float(q.target_delta.median()),
       supported_null_delta_median=float(q.null_delta.median()),
       supported_null_delta_q95=float(q.null_delta.quantile(.95)))
      q['pathway']=target;q['mode']=mode
      if 'paired_all' not in locals():paired_all=[]
      paired_all.append(q)
   records.append(d)
  log('SUPPORT',ti+1,'/18',target,'eligible',len(eligible),'violations',violations)
 result=pd.DataFrame(records)
 # For family-wise 18-panel bookkeeping, missing tests get an implicit p=1;
 # unsupported results remain NA and are never called significant.
 for mode in ['baseline','adjusted']:
  ii=result['mode']==mode
  p=result.loc[ii,'empirical_rank_p'].fillna(1).to_numpy(float)
  q=env['o'].bh(pd.Series(p))
  result.loc[ii,'BHq_exploratory_18_test_panel']=np.where(result.loc[ii,'empirical_rank_p'].notna(),np.asarray(q,float),np.nan)
 put(result,out_file)
 put(pd.concat(details,ignore_index=True),out/'03_SUPPORT_CANDIDATE_DETAILS.csv.gz')
 if 'paired_all' in locals():put(pd.concat(paired_all,ignore_index=True),out/'03_SUPPORTED_PAIRED_DELTAS.csv.gz')
 status_counts=result[result['mode']=='adjusted'].status.value_counts().to_dict()
 report=['# Tabula Figure 6 | V4 strict common-support null calibration','',
 '**Frozen B0 phase is inferred, NOT sampling time. The same 18 pathways were selected from the same atlas.**','',
 '## Joint support protocol','',
 f'- Old 199 + newly designed {args.new_per_target} cohort-fixed null sets per target. New candidates use target-matched gene-level expression and approximate signed covariance from sampled RNA; memberships fixed before cohort-wide scoring.',
 f'- Shared cohort-lineage residualized Pearson correlation estimator for all targets, old controls and new controls; {args.corr_cells} sampled cells maximum.',
 f'- Exact/near gene count, absolute Pearson correlation and RMS correlation calipers, same observed unit coverage, and multiplicative within-block variance caliper of {args.variance_ratio}.',
 f'- ALL nuisance dimensions must be inside marginal control support; at least {args.min_supported} controls must satisfy all joint calipers. No nearest-N fallback.',
 '- Any candidate with insufficient support receives NA P and state NO_COMMON_SUPPORT, NOT a non-significant P or a manufactured q.','',
 '## Critical limitation','',
 'Within-block variance and gene covariance are learned from the same RNA outcomes used in M2; this is POST-HOC exploratory sensitivity. Even strong matches DO NOT establish a formally calibrated null or valid confirmatory familywise P values.',
 'Target and null gene membership may overlap in biological co-regulation through shared latent cell state. This is not proof of clock causality.','',
 f'- 18-panel adjusted states: {status_counts}',
 f'- Supported adjusted: {int(((result["mode"]=="adjusted")&(result.status=="SUPPORTED_EXPLORATORY")).sum())}/18','',
 '## Files','',
 '- 01_NEW_FIXED_NULL_GENE_SETS.json : fixed new cohort-independent null memberships',
 '- 01_COMMON_CORRELATION_AUDIT.csv : consistent pooled coexpression estimator for all modules',
 '- 02_TARGET_SCORE_REPRODUCTION.csv : frozen target score fail-closed check',
 '- 03_ALL_MODULE_JOINT_AUDIT.csv : nuisance features/within-block variance',
 '- 03_SUPPORT_CANDIDATE_DETAILS.csv.gz : per-null joint caliper decisions',
 '- 03_COMMON_SUPPORT_M2_RESULTS.csv : conditional support and exploratory ranks',
 '- 03_SUPPORTED_PAIRED_DELTAS.csv.gz : paired M2 effect details where estimable','',
 'This is an internal sensitivity analysis, not independent external validation.']
 (out/'INTERPRETATION_REPORT.md').write_text('\n'.join(report),encoding='utf8')
 log('EVALUATE COMPLETE',status_counts)

def preflight(args,env,root,out):
 """Using existing V3 nuisance audit only, show which targets had ANY
 reasonable support among the original 199 sets; NO p from nearest-neighbors.
 Values will be reestimated uniformly using V4 covariance in the final analysis.
 """
 a=pd.read_csv(root/args.v3/'01_MODULE_COVARIANCE_AND_VAR_AUDIT.csv')
 a=a.rename(columns={'n_genes':'n_original_genes'})
 report=[]
 for i,target in enumerate(env['design']['targets']):
  one=a[a.pathway==target]
  req(len(one)==1, 'V3 audit missing '+target)
  null=a[(a.pathway.str.startswith(f'FIXNULL__{i:02d}__')) & (a.kind=='FixedNull')]
  req(len(null)==199, 'V3 missing original controls for '+target)
  mask,reasons=support_mask(one.iloc[0],null,args)
  report.append(dict(pathway=target,n_existing_fixed_controls=len(null),
    n_joint_support_existing=int(mask.sum()),outside_marginal_support=';'.join(reasons),
    status='SUFFICIENT_OLD_SUPPORT_PREVIEW' if mask.sum()>=args.min_supported and not reasons else 'NEEDS_NEW_NULLS',
    note='V3 covariance estimator; will be RECOMPUTED identically for all modules in V4'))
 table=pd.DataFrame(report);put(table,out/'00_OLD_NULL_COMMON_SUPPORT_PREFLIGHT.csv')
 log('PREFLIGHT old 199 support at least',args.min_supported,':',int((table.status=='SUFFICIENT_OLD_SUPPORT_PREVIEW').sum()),'/18')


def selftest():
 rng=np.random.default_rng(21)
 dat=pd.DataFrame(dict(n_original_genes=[100]*40,median_abs_gene_corr=rng.normal(.035,.004,40),
  rms_gene_corr=rng.normal(.07,.008,40),within_block_variance=np.exp(rng.normal(-8,.20,40)),n_finite_units=[2444]*40))
 ref=dat.median(numeric_only=True)
 cfg=Namespace(gene_tolerance=.1,unit_tolerance=0,median_corr_atol=.012,rms_corr_atol=.02,
               corr_relative_tol=.40,variance_ratio=1.6)
 m,reason=support_mask(ref,dat,cfg)
 req(m.sum()>20 and not reason,'Selftest: expected successful common support')
 far=ref.copy();far['within_block_variance']=2*dat.within_block_variance.max()
 m,reason=support_mask(far,dat,cfg)
 req((not m.any()) and 'OUTSIDE_RANGE_within_block_variance' in reason,
     'Selftest: extrapolated variance must not be called supported')
 weak=dat.iloc[:4].copy();m,reason=support_mask(ref,weak,cfg)
 req(len(m)==4,'Selftest shape mismatch')
 print('SELFTEST_PASS: joint calipers + out-of-range refusal + no forced p')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project-root',default='.')
 p.add_argument('--source',default='TABULA_FIG6_STRICT_VALIDATION_V2_1')
 p.add_argument('--v3',default='TABULA_FIG6_M2_CONDITIONAL_NULL_CV_V3')
 p.add_argument('--output-dir',default='TABULA_FIG6_M2_COMMON_SUPPORT_V4')
 p.add_argument('--stage',choices=['all','preflight','design','score','evaluate','selftest'],default='all')
 p.add_argument('--resume',action='store_true')
 p.add_argument('--new-per-target',type=int,default=200)
 p.add_argument('--sample-cells',type=int,default=240)
 p.add_argument('--corr-cells',type=int,default=1800)
 p.add_argument('--h5ad-block',type=int,default=192)
 p.add_argument('--neighbors',type=int,default=60)
 p.add_argument('--selection-top',type=int,default=6)
 p.add_argument('--selection-temperature',type=float,default=.25)
 p.add_argument('--corr-penalty-scale',type=float,default=.07)
 p.add_argument('--seed',type=int,default=20261013)
 p.add_argument('--min-supported',type=int,default=30)
 p.add_argument('--gene-tolerance',type=float,default=.10)
 p.add_argument('--unit-tolerance',type=float,default=0)
 p.add_argument('--median-corr-atol',type=float,default=.012)
 p.add_argument('--rms-corr-atol',type=float,default=.020)
 p.add_argument('--corr-relative-tol',type=float,default=.4)
 p.add_argument('--variance-ratio',type=float,default=1.6)
 p.add_argument('--max-target-delta-spread',type=float,default=.01)
 args=p.parse_args()
 if args.stage=='selftest':selftest();return
 req(args.new_per_target>=30 and args.corr_cells>=300 and 1<args.variance_ratio<=2.5 and args.min_supported>=20,
     'Unsafe parameters; use reasonable overlap/precision')
 root=Path(args.project_root).expanduser().resolve();out=root/args.output_dir
 req(root.exists(),'Project root does not exist');out.mkdir(exist_ok=True,parents=True)
 v3,env,old=load(root,args)
 params=dict(version=VERSION,source_score_sig=jread(root/args.source/'03_SCORE_SIGNATURE.json').get('signature'),
  v3_config=jread(root/args.v3/'00_V3_RUN_CONFIG.json'),
  params={k:v for k,v in vars(args).items() if k not in ('project_root','output_dir','stage','resume','source','v3')})
 stamp=out/'00_CONFIG.json'
 if stamp.exists():req(jread(stamp)==params,'Existing V4 output config differs; choose a new --output-dir')
 else:
  req(not any(out.glob('01_*')) and not any(out.glob('02_*')) and not any(out.glob('03_*')),
      'Partial prior V4 files without configuration. Use a fresh output directory.')
  jwrite(stamp,params)
 seq=['design','score','evaluate'] if args.stage=='all' else [args.stage]
 try:
  for st in seq:
   log('START',st);jwrite(out/'00_STATUS.json',dict(state='RUNNING',step=st,version=VERSION))
   if st=='preflight':preflight(args,env,root,out)
   elif st=='design':propose(args,env,v3,root,out)
   elif st=='score':score(args,env,v3,root,out)
   elif st=='evaluate':evaluate(args,env,v3,old,root,out)
  done=(out/'03_COMMON_SUPPORT_M2_RESULTS.csv').exists()
  jwrite(out/'00_STATUS.json',dict(state='COMPLETE' if done else 'PARTIAL_STAGE_COMPLETE',
   stages=seq,version=VERSION,finished=datetime.now().isoformat()))
  log('V4',('COMPLETE' if done else 'PARTIAL_STAGE_COMPLETE'),out)
 except Exception:
  jwrite(out/'00_STATUS.json',dict(state='FAILED',version=VERSION,traceback=traceback.format_exc()[-6000:]))
  raise
if __name__=='__main__':main()
