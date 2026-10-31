#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabula Figure 6 V3: covariance/coverage audit + conditional M2 null + held-tissue CV.

NO B0 REFITTING. NO original files overwritten. Requires completed
TABULA_FIG6_STRICT_VALIDATION_V2_1 and original M0/M1/M2 folders.

Two distinct nulls:
 * Fixed-gene matched-null empirical ranks, reweighted/trimmed by covariance,
   coverage, gene count and within-block pathway variance (EXPLORATORY only).
 * Donor-cluster wild-sign bootstrap against the restricted M1 model, targeting
   ΔR²(M2-M1), NOT M2-vs-M0 omnibus phase association (conditional test).

Optional independent table requires real replicated timepoints; if absent,
report NOT_RUN rather than implying independent validation. B0 is inferred
clock state, never observed circadian time.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, traceback, math, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

TARGET_KEY=['tissue','assay','donor','lineage','phase_bin','phase_center_h']
BLOCK=['tissue','assay','donor','lineage']
VERSION='fig6-m2-v3.0'

def log(*s):print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),*s,flush=True)
def check(cond,msg):
 if not cond:raise RuntimeError(msg)
def put(df,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp')
 df.to_csv(tmp,index=False,compression='gzip' if p.suffix=='.gz' else None)
 os.replace(tmp,p)
def jout(p,content):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(content,indent=2,ensure_ascii=False,default=str),encoding='utf8');os.replace(tmp,p)
def jin(p):return json.loads(Path(p).read_text(encoding='utf8'))
def import_file(path,tag):
 spec=importlib.util.spec_from_file_location(tag,path)
 check(spec is not None and spec.loader is not None,f'Cannot import {path}')
 mod=importlib.util.module_from_spec(spec);sys.modules[tag]=mod;spec.loader.exec_module(mod);return mod

def load_env(root,source):
 check((root/'tabula_fig6_Hallmark_Reactome_M012_v1.py').is_file(),'Missing original M012 source script')
 original=import_file(root/'tabula_fig6_Hallmark_Reactome_M012_v1.py','fig6_v3_reference')
 v2file=root/'tabula_fig6_strict_validation_v2_1_1.py'
 check(v2file.is_file(),'Missing tabula_fig6_strict_validation_v2_1_1.py; copy the previous V2.1.1 script to the project root')
 v2=import_file(v2file,'fig6_v3_previous')
 _, phase, original_genes, helper =v2.load_modules(root)
 check(jin(root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1'/'01_HALLMARK_SOURCE_REPRODUCTION_AUDIT.json')['status']=='PASS','Old Hallmark reproduction not PASS')
 check(jin(source/'00_STATUS.json').get('state')=='COMPLETE','V2.1 must be finalized: 00_STATUS.json != COMPLETE')
 meta=pd.read_csv(source/'01_ATLAS_UNIT_METADATA.csv.gz')
 p=np.load(source/'01_ALL_PATHWAY_SCORES_WIDE.npz',allow_pickle=False)
 arr=p['scores'];names=list(p['pathway'].astype(str));check(len(arr)==len(meta),'Score matrix length mismatch')
 design=jin(source/'00_DESIGN.json');nullsets=jin(source/'02_FIXED_NULL_GENE_SETS.json')['sets']
 fix=pd.read_csv(source/'04_FIXED_NULL_AND_STATE_RESULTS.csv')
 previous_pairs=pd.read_csv(source/'04_FIXED_MATCHED_NULL_PAIRED_DISTRIBUTION.csv.gz')
 checks=pd.read_csv(source/'03_FROZEN_TARGET_SCORE_REPRODUCTION.csv')
 check((checks.unmatched_rows.eq(0)&checks.max_abs_diff.lt(1e-4)).all(),'Original target score reproduction failed')
 check(len(design['targets'])==18 and len(nullsets)==3582,'V2 target/null-set counts not expected 18/3582')
 check(not meta.duplicated(TARGET_KEY).any(),'Duplicated atlas unit keys')
 for tag,genes in design['state_markers'].items():
  if tag in names:meta['state_proxy_'+tag]=arr[:,names.index(tag)]
 return dict(o=original,v2=v2,phase=phase,orig=original_genes,helper=helper,
             meta=meta,arr=arr,names=names,index={g:i for i,g in enumerate(names)},design=design,nulls=nullsets,
             fix=fix,prev=previous_pairs)

def by_target(env):
 for t in env['design']['targets']:
  row=env['fix'][(env['fix'].pathway==t)&(env['fix']['mode']=='disjoint_RNA_proxies')]
  check(len(row)==1,f'Missing adjusted V2 row {t}')
  fields=str(row.iloc[0].state_fields).split(';')
  check(len(fields)==2 and all(f in env['meta'] for f in fields),f'Invalid V2 state fields: {t} {fields}')
  ix=env['index'][t];take=np.isfinite(env['arr'][:,ix])
  for f in fields:take&=np.isfinite(env['meta'][f].to_numpy(float))
  yield t,fields,take

def model_design(env,t,mode,sel,other=None):
 meta=env['meta'];arr=env['arr'];idx=env['index'];o=env['o']
 name=t if other is None else other
 ix=sel & np.isfinite(arr[:,idx[name]])
 if ix.sum()<100:return None
 columns=TARGET_KEY + (['n_eff'] if 'n_eff' in meta else [])
 d=meta.loc[ix,columns].copy()
 if mode=='adjusted':
  fields=next(f for tt,f,_ in by_target(env) if tt==t)
  for k,g in enumerate(fields):d['state_'+str(k)]=meta.loc[ix,g].to_numpy(float)
 d['score']=arr[ix,idx[name]].astype(float)
 return o.build_design(d)

def paired_design(env,t,n,mode,mask,fields):
 idx=env['index'];arr=env['arr']
 pair=mask &np.isfinite(arr[:,idx[t]]) &np.isfinite(arr[:,idx[n]])
 des0=model_design(env,t,mode,pair)
 des1=model_design(env,t,mode,pair,n)
 if des0 is None or des1 is None:return None,None
 if len(des0['y'])!=len(des1['y']) or not des0['z'][TARGET_KEY].reset_index(drop=True).equals(des1['z'][TARGET_KEY].reset_index(drop=True)):
  raise RuntimeError('Paired M2 null units misaligned: '+str((t,n)))
 return des0,des1

def block_variance(meta,values):
 mask=np.isfinite(values)
 if mask.sum()<16:return np.nan
 d=meta.loc[mask,BLOCK].copy();d['value']=values[mask]
 if 'n_eff' in meta:d['n_eff']=meta.loc[mask,'n_eff'].to_numpy(float)
 key=d[BLOCK].astype(str).agg('||'.join,axis=1)
 means=d.groupby(key).value.transform('mean')
 res=(d.value-means).to_numpy(float)
 w=np.maximum(d.n_eff.to_numpy(float),1.0) if 'n_eff' in d else np.ones(len(d))
 return float(np.average(res**2,weights=np.minimum(w,100.)))

def module_pairs(genes,n_pairs,seed):
 G=len(genes)
 if G<2:return []
 alln=G*(G-1)//2; rng=np.random.default_rng(seed)
 if alln<=n_pairs:return [(genes[i],genes[j]) for i in range(G) for j in range(i+1,G)]
 picked=set()
 while len(picked)<n_pairs:
  i,j=rng.integers(G,size=2)
  if i==j:continue
  picked.add((int(min(i,j)),int(max(i,j))))
 return [(genes[i],genes[j]) for i,j in sorted(picked)]

def sample_for_covariance(args,env,root,out):
 """Sample individual cells under frozen B0 QC. Residualize expression within
 cohort×lineage, then audit gene-pair correlations across pooled residual cells.
 Samples/correlations are descriptive nuisance diagnostics, NOT external evidence.
 """
 import h5py
 design=env['design'];sets={**design['targets'],**env['nulls']}
 allgenes=sorted(set().union(*(set(gs) for gs in sets.values())))
 gene_idx={g:i for i,g in enumerate(allgenes)}
 phase=env['phase'];orig=env['orig'];helper=env['helper']
 oldroot=root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1'
 manifest,_,_=env['v2'].get_manifest(root,oldroot,env['o'])
 allX=[];coverage=[];counts=[]
 for k,(_,row) in enumerate(manifest.iterrows(),1):
  frozen,source=env['v2'].pathsource(row,root,phase)
  pred=pd.read_csv(frozen/'CELL_PHASE.csv.gz',usecols=['cell_id','R_B0','total_raw_counts'])
  _,post=phase.posterior(frozen/'PHASE_POSTERIOR.npz',len(pred))
  R=np.abs(post@np.exp(2j*np.pi*np.arange(48)/48))
  check(np.max(np.abs(R-pred.R_B0.to_numpy(float)))<1e-6,'Frozen R mismatch')
  with h5py.File(source,'r') as h:
   joined,_,_,_=helper.exact_join(pred,h,str(row.assay))
   dcol=next((c for c in phase.DONOR_FIELDS if phase.read_col(h['obs'],c) is not None),None)
   lcol=next((c for c in phase.CELLTYPE_FIELDS if phase.read_col(h['obs'],c) is not None),None)
   check(dcol and lcol,'Missing donor/lineage labels '+str((row.tissue_general,row.assay)))
   donors=np.asarray(phase.read_col(h['obs'],dcol))[joined]
   line=np.asarray([phase.standard_lineage(s) for s in phase.read_col(h['obs'],lcol)[joined]])
   ok=(R>=.3)&np.isin(line,env['o'].LINEAGES)&np.asarray([str(x).lower().strip() not in ('','nan','na','unknown','none') for x in donors])
   candidates=np.flatnonzero(ok)
   rng=np.random.default_rng(args.seed+sum(str(row.tissue_general).encode())+sum(str(row.assay).encode()))
   chosen=[]
   for label in env['o'].LINEAGES:
    v=candidates[line[candidates]==label]
    if len(v):chosen.extend(rng.choice(v,size=min(len(v),max(1,args.sample_cells//5)),replace=False))
   chosen=np.asarray(sorted(set(chosen)),int)
   mapped,_=orig.make_target_indices(phase,h['var'],sorted(set(allgenes)|set(phase.CORE_DETECTION)))
   for name,gs in sets.items():
    nmap=sum(g in mapped for g in gs)
    coverage.append(dict(pathway=name,cohort=str(row.tissue_general)+'||'+str(row.assay),
      n_original_genes=len(gs),n_mapped=nmap,coverage=nmap/len(gs)))
   if not len(chosen):continue
   goodgenes=[g for g in allgenes if g in mapped]
   core=[mapped[g] for g in phase.CORE_DETECTION if g in mapped]
   check(len(core)==len(phase.CORE_DETECTION),'CORE detection mapping incomplete')
   cols=[mapped[g] for g in goodgenes]+core
   raw,total=phase.read_counts(h['X'],np.asarray(joined)[chosen],cols,args.h5ad_block)
   check(np.allclose(total,pred.total_raw_counts.to_numpy(float)[chosen],atol=1e-3,rtol=1e-5),
     'Raw frozen cell totals mismatch')
   passing=(raw[:,len(goodgenes):]>0).sum(axis=1)>=2
   chosen=chosen[passing];raw=raw[passing,:len(goodgenes)];total=total[passing]
   if not len(chosen):continue
   x=np.log1p(1e4*raw/np.maximum(total[:,None],1)).astype(np.float32)
   # Lineage means removed inside EACH cohort; no pooled cross-cohort identity mean.
   for label in env['o'].LINEAGES:
    ix=np.flatnonzero(line[chosen]==label)
    if len(ix)>=3:x[ix]-=np.mean(x[ix],axis=0,keepdims=True)
    else:x[ix]=np.nan
   m=np.full((len(x),len(allgenes)),np.nan,dtype=np.float32)
   m[:,[gene_idx[g] for g in goodgenes]]=x
   keep=np.isfinite(m).any(axis=1)
   if keep.any():allX.append(m[keep])
   counts.append(dict(tissue=row.tissue_general,assay=row.assay,n_cells=len(chosen),n_used_corr=int(keep.sum()),n_genes_mapped=len(goodgenes)))
  if k%5==0:log('COVARIANCE SAMPLE',k,'/',len(manifest),'sampled cells',sum(len(q) for q in allX))
 check(allX,'No usable cells for gene covariance calibration')
 X=np.concatenate(allX,axis=0)
 # Guard memory: gene-level residual matrix is temporary and not exported.
 coverage=pd.DataFrame(coverage)
 put(coverage,out/'01_COHORT_GENE_COVERAGE.csv.gz')
 put(pd.DataFrame(counts),out/'01_COVARIANCE_SAMPLING_AUDIT.csv')
 check(X.shape[0]>=300,'Too few cells for pairwise covariance audit')
 return X,allgenes,coverage

def pairwise_correlations(X,genelist,sets,args):
 """Paired-observation Pearson r on within cohort×lineage centered RNA.
 All 199 null memberships are fixed; no selection on the M2 outcome here.
 """
 gidx={g:i for i,g in enumerate(genelist)}
 allpairs=[];nameindices={};offset=0
 for s,genes in sets.items():
  have=[g for g in genes if g in gidx]
  pairs=module_pairs(have,args.pairs_per_set,args.seed+sum(s.encode()))
  nameindices[s]=(offset,offset+len(pairs))
  allpairs.extend((gidx[a],gidx[b]) for a,b in pairs)
  offset+=len(pairs)
 check(allpairs,'No covariance gene pairs')
 xy=np.array(allpairs,dtype=np.int32)
 outcorr=np.full(len(xy),np.nan,dtype=float)
 for start in range(0,len(xy),args.pair_chunk):
  sel=xy[start:start+args.pair_chunk]
  a=X[:,sel[:,0]];b=X[:,sel[:,1]]
  valid=np.isfinite(a)&np.isfinite(b)
  na=valid.sum(axis=0)
  aa=np.where(valid,a,0.);bb=np.where(valid,b,0.)
  sums1=aa.sum(axis=0,dtype=np.float64);sums2=bb.sum(axis=0,dtype=np.float64)
  cov=(aa*bb).sum(axis=0,dtype=np.float64)-sums1*sums2/np.maximum(na,1)
  va=(aa*aa).sum(axis=0,dtype=np.float64)-sums1*sums1/np.maximum(na,1)
  vb=(bb*bb).sum(axis=0,dtype=np.float64)-sums2*sums2/np.maximum(na,1)
  validcols=(na>=args.min_pair_cells)&(va>1e-10)&(vb>1e-10)
  vals=np.full(len(sel),np.nan)
  vals[validcols]=cov[validcols]/np.sqrt(va[validcols]*vb[validcols])
  outcorr[start:start+len(sel)]=np.clip(vals,-1,1)
  if start%(args.pair_chunk*100)==0:log('CORRELATION PAIRS',start,'/',len(xy))
 records=[]
 for s,(a,b) in nameindices.items():
  r=outcorr[a:b];r=r[np.isfinite(r)]
  records.append(dict(pathway=s,n_requested_pairs=b-a,n_valid_pairs=len(r),
    median_abs_gene_corr=float(np.median(abs(r))) if len(r) else np.nan,
    median_signed_gene_corr=float(np.median(r)) if len(r) else np.nan,
    rms_gene_corr=float(np.sqrt(np.mean(r*r))) if len(r) else np.nan))
 return pd.DataFrame(records)

def do_audit(args,env,root,out):
 covfile=out/'01_MODULE_COVARIANCE_AND_VAR_AUDIT.csv'
 if args.resume and covfile.exists():
  log('AUDIT CACHE existing',covfile);return
 sets={**env['design']['targets'],**env['nulls']}
 X,genes,cov=sample_for_covariance(args,env,root,out)
 correlations=pairwise_correlations(X,genes,sets,args)
 del X
 group=cov.groupby('pathway',as_index=False).agg(
  min_cohort_coverage=('coverage','min'),median_cohort_coverage=('coverage','median'),
  fraction_cohorts_ge_045=('coverage',lambda s:float(np.mean(s>=.45))),
  n_cohorts_mapped=('n_mapped',lambda s:int(np.sum(s>0))),
  n_cohorts=('cohort','nunique'),n_genes=('n_original_genes','first'))
 res=group.merge(correlations,on='pathway',validate='one_to_one')
 ar=env['arr'];idx=env['index'];meta=env['meta']
 res['within_block_variance']=[block_variance(meta,ar[:,idx[t]].astype(float)) for t in res.pathway]
 res['n_finite_units']=[int(np.isfinite(ar[:,idx[t]]).sum()) for t in res.pathway]
 res['kind']=['Target' if t in env['design']['targets'] else 'FixedNull' for t in res.pathway]
 put(res,covfile)
 jout(out/'01_AUDIT_METHOD.json',dict(state='AUDIT_COMPLETE',n_cells_from_sample=int(pd.read_csv(out/'01_COVARIANCE_SAMPLING_AUDIT.csv').n_used_corr.sum()),
  n_fixed_nulls=len(env['nulls']),pairs_per_set=args.pairs_per_set,
  source='Original H5AD; frozen B0 R>=0.3; min CORE detection 2; within cohort-lineage residualized log1p CP10k',
  limits=['Unmeasured gene-gene covariance outside sampled pairs','Sparse small-cohort gene pairs','Pairwise correlation matching is exploratory and does not create independent data','Within-block variance matching uses outcome-derived scores and is only sensitivity']))
 log('AUDIT COMPLETE',len(res),'modules')

def conditional_wild(args,o,des,seed):
 """M1-null donor-cluster wild Rademacher, test SSE(X1)-SSE(X2).
 M1 is refitted for each surrogate; clusters are biological donors.
 This directly tests conditional lineage×phase additions, NOT M2total.
 """
 rng=np.random.default_rng(seed)
 y,w=des['y'],des['w'];X1,X2=des['X1'],des['X2'];dc=des['donor_codes']
 n_donors=des['n_donors']
 b1,pred1,res1,sse1=o.weighted_lstsq(X1,y,w)
 sse2=o.weighted_lstsq(X2,y,w)[3]
 denominator=np.dot(y*w,y)
 observed=(sse1-sse2)/denominator
 null=np.zeros(args.wild)
 for i in range(args.wild):
  signs=rng.choice([-1.,1.],size=n_donors)[dc]
  yp=pred1+res1*signs
  a=o.weighted_lstsq(X1,yp,w)[3]
  b=o.weighted_lstsq(X2,yp,w)[3]
  null[i]=(a-b)/denominator
 p=(1+np.sum(null>=observed-1e-12))/(args.wild+1)
 return dict(observed_M2_deltaR2=float(observed),conditional_M1null_wild_p=float(p),
   conditional_null95=float(np.quantile(null,.95)),conditional_null_median=float(np.median(null)),
   n_donors=int(n_donors),n_units=len(y))

def candidate_calibration(audit,table,target,mode,args):
 """Covariance/coverage/variance-matched exploratory empirical null. Distances
 only use nuisance audit metrics, not conditional M2 fit, p, or phases.
 """
 feature=['n_genes','median_abs_gene_corr','median_cohort_coverage','fraction_cohorts_ge_045','within_block_variance','n_finite_units']
 lookup=audit.set_index('pathway')
 if target not in lookup.index:return pd.DataFrame(),None
 observed=lookup.loc[target,feature].astype(float).to_numpy()
 v=table[table.pathway==target].copy()
 v=v[v['null'].isin(lookup.index)].copy()
 if v.empty:return pd.DataFrame(),None
 cols=lookup.loc[v['null'],feature].astype(float).to_numpy()
 finite=np.isfinite(observed).all()&np.isfinite(cols).all(axis=1)
 v=v.loc[finite].copy();cols=cols[finite]
 if len(v)<args.min_calibrated:return pd.DataFrame(),None
 # Robust null-distribution scaling. These are exploratory conditioning features.
 scale=np.maximum(np.nanquantile(cols,.75,axis=0)-np.nanquantile(cols,.25,axis=0),[2,.015,.05,.05,.005,25.])
 dist=np.sqrt(np.mean(((cols-observed)/scale)**2,axis=1))
 v['match_distance']=dist
 v=v.sort_values('match_distance').head(min(args.calibrated_k,len(v))).copy()
 if len(v)<args.min_calibrated:return pd.DataFrame(),None
 p=(1+int((v['null_delta_M2']>=v['target_delta_M2']-1e-12).sum()))/(len(v)+1)
 # Paired response selection can alter target fit; inspect deviation.
 rec=dict(pathway=target,mode=mode,n_calibrated_nulls=len(v),
  calibrated_empirical_p_M2=float(p),
  calibrated_null_median_M2=float(v.null_delta_M2.median()),
  calibrated_target_median_M2=float(v.target_delta_M2.median()),
  calibration_median_distance=float(v.match_distance.median()),
  calibration_max_distance=float(v.match_distance.max()),
  interpretation='POST-HOC outcome-derived variance matching; explorative, not formal pathway-specific p')
 return v,rec

def do_conditional(args,env,out):
 outfile=out/'02_CONDITIONAL_M2_RESULTS.csv'
 if args.resume and outfile.exists():log('CONDITIONAL CACHE existing',outfile);return
 auditfile=out/'01_MODULE_COVARIANCE_AND_VAR_AUDIT.csv'
 check(auditfile.exists(),'First run --stage audit')
 audit=pd.read_csv(auditfile)
 o=env['o'];records=[];comparison=[];recal=[]
 basepairs=env['prev']
 for i,(target,fields,take) in enumerate(by_target(env),1):
  for mode in ('baseline','adjusted'):
   design=model_design(env,target,mode,take)
   check(design is not None and o.fit_models(design) is not None,f'Insufficient M2 target {target} {mode}')
   wild=conditional_wild(args,o,design,args.seed+sum(target.encode())+(0 if mode=='baseline' else 100000))
   rec=dict(pathway=target,mode=mode,**wild)
   nullnames=sorted(n for n in env['nulls'] if n.startswith(f'FIXNULL__{i-1:02d}__'))
   if mode=='baseline':
    # V2 precomputed paired target/null fits are EXACTLY for baseline_common_rows.
    src=basepairs[basepairs.pathway==target]
    pair=src[['null','null_M2','target_M2_paired']].copy().rename(columns={'null_M2':'null_delta_M2','target_M2_paired':'target_delta_M2'})
   else:
    parts=[]
    for null in nullnames:
     des0,desn=paired_design(env,target,null,mode,take,fields)
     if des0 is None or desn is None:continue
     f0=o.fit_models(des0);fn=o.fit_models(desn)
     if f0 is None or fn is None:continue
     parts.append(dict(pathway=target,null=null,n_units=len(des0['y']),
       target_delta_M2=f0['deltaR2_M2_vs_M1'],null_delta_M2=fn['deltaR2_M2_vs_M1']))
    pair=pd.DataFrame(parts)
   check(not pair.empty and len(pair)>=100,f'Too few fixed null models for {target} {mode}')
   pair['pathway']=target;pair['mode']=mode
   raw_p=(1+int((pair.null_delta_M2>=pair.target_delta_M2-1e-12).sum()))/(len(pair)+1)
   rec.update(n_valid_nulls=len(pair),raw_matched_empirical_p_M2=raw_p)
   calibrated,detail=candidate_calibration(audit,pair,target,mode,args)
   if detail:rec.update({k:v for k,v in detail.items() if k not in ('pathway','mode')})
   else:rec.update(n_calibrated_nulls=0,calibrated_empirical_p_M2=np.nan)
   if not calibrated.empty:
    calibrated['pathway']=target;calibrated['mode']=mode
    recal.append(calibrated)
   records.append(rec)
   if mode=='baseline':comparison.append(pair)
   log('M2 CONDITIONAL',i,'/18',mode,target,'wild p',rec['conditional_M1null_wild_p'],
       'cal p',rec['calibrated_empirical_p_M2'])
   put(pd.DataFrame(records),out/'02_CONDITIONAL_M2_PARTIAL.csv')
 check(len(records)==36,'Expected 18 x 2 M2 tests')
 a=pd.DataFrame(records)
 for mode,loc in a.groupby('mode').groups.items():
  for p in ('conditional_M1null_wild_p','raw_matched_empirical_p_M2','calibrated_empirical_p_M2'):
   a.loc[loc,'BHq_'+p]=o.bh(a.loc[loc,p])
 put(a,outfile)
 if recal:put(pd.concat(recal,ignore_index=True),out/'02_COVARIANCE_MATCHED_NULL_SELECTION.csv.gz')
 log('CONDITIONAL COMPLETE',len(records),'target/mode tests')

def cv_fold(o,train,test,mode):
 a=o.build_design(train);b=o.build_design(test)
 if a is None or b is None:return None
 if a['n_donors']<8 or a['n_lineages']<4 or b['n_lineages']<2:return None
 if not set(a['donors']).isdisjoint(set(b['donors'])):raise RuntimeError('Train/test donor leakage')
 if any(a[k].shape[1]!=b[k].shape[1] for k in ('X1','X2')):return None
 if np.linalg.matrix_rank(a['X2'])!=a['X2'].shape[1]:return None
 out=[]
 for m in ('X1','X2'):
  A=a[m]*np.sqrt(a['w'])[:,None];z=a['y']*np.sqrt(a['w'])
  beta=np.linalg.lstsq(A,z,rcond=None)[0]
  yhat=b[m]@beta
  out.append(float(np.sqrt(np.average((b['y']-yhat)**2,weights=b['w']))))
 return dict(RMSE_M1=out[0],RMSE_M2=out[1],deltaRMSE_M2_vs_M1=out[0]-out[1],
   n_train_units=len(a['y']),n_test_units=len(b['y']),
   n_train_donors=a['n_donors'],n_test_donors=b['n_donors'])

def do_cv(args,env,out):
 outfile=out/'03_HELD_TISSUE_DONOR_DISJOINT_M2_CV.csv'
 if args.resume and outfile.exists() and (out/'03_M2_CV_SUMMARY.csv').exists():
  log('CV CACHE existing',outfile);return
 rows=[];meta=env['meta'];arr=env['arr'];idx=env['index'];o=env['o']
 for i,(target,fields,take) in enumerate(by_target(env),1):
  for mode in ('baseline','adjusted'):
   use=take.copy()
   cols=TARGET_KEY+(['n_eff'] if 'n_eff' in meta else [])
   d=meta.loc[use,cols].copy();d['score']=arr[use,idx[target]].astype(float)
   if mode=='adjusted':
    for j,g in enumerate(fields):d['state_'+str(j)]=meta.loc[use,g].to_numpy(float)
   for tissue in sorted(d.tissue.dropna().unique()):
    test=d[d.tissue==tissue].copy()
    held=set(test.donor.astype(str))
    train=d[(d.tissue!=tissue)&(~d.donor.astype(str).isin(held))].copy()
    if len(train)<100 or len(test)<10:continue
    result=cv_fold(o,train,test,mode)
    if result:rows.append(dict(pathway=target,mode=mode,held_tissue=tissue,**result))
  log('M2 HELD CV',i,'/18',target,'folds',len(rows))
  if rows:put(pd.DataFrame(rows),out/'03_HELD_CV_PARTIAL.csv')
 check(rows,'No CV folds feasible')
 cv=pd.DataFrame(rows);put(cv,outfile)
 summaries=[]
 for (t,mode),g in cv.groupby(['pathway','mode']):
  vals=g.deltaRMSE_M2_vs_M1.to_numpy(float)
  # Tissues share training cohorts. This CI is descriptive; not an independence claim.
  rng=np.random.default_rng(args.seed+sum(t.encode())+(1 if mode=='adjusted' else 0))
  boot=vals[rng.integers(len(vals),size=(args.cv_boot,len(vals)))].mean(axis=1)
  summaries.append(dict(pathway=t,mode=mode,n_held_tissues=g.held_tissue.nunique(),
   mean_delta_RMSE=float(vals.mean()),median_delta_RMSE=float(np.median(vals)),
   fraction_held_tissues_better=float((vals>0).mean()),
   descriptive_bootstrap_lo=float(np.quantile(boot,.025)),
   descriptive_bootstrap_hi=float(np.quantile(boot,.975)),
   caution='Overlapping training sets across folds; CI descriptive; centered held-block outcome not absolute expression'))
 put(pd.DataFrame(summaries),out/'03_M2_CV_SUMMARY.csv')
 log('CV COMPLETE',len(cv),'folds')

def do_external(args,env,out):
 outfile=out/'04_EXTERNAL_REPLICATION_STATUS.json'
 if not args.external_table:
  jout(outfile,dict(state='NOT_RUN_NO_INDEPENDENT_DATA',verified=False,
    required='Independently sampled, replicated real timepoints; do not use 6 ZT single-source mouse data as biological replicates',
    input_schema='sample_id,donor,tissue,assay,lineage,time_h,pathway,score'))
  return
 p=Path(args.external_table)
 check(p.exists(),f'External data missing: {p}')
 tab=pd.read_csv(p)
 fields=['sample_id','donor','tissue','lineage','time_h','pathway','score']
 check(set(fields).issubset(tab),'External sample table missing: '+str(set(fields)-set(tab)))
 if 'assay' not in tab:tab['assay']='unknown'
 check(not tab.duplicated(['sample_id','lineage','pathway']).any(),'External table has duplicate sample×lineage×pathway')
 check(tab.sample_id.notna().all() and tab.donor.notna().all(),'Sample/donor IDs missing')
 check(np.isfinite(tab.time_h.to_numpy(float)).all() and np.isfinite(tab.score.to_numpy(float)).all(),'Nonfinite external data')
 tab.time_h=tab.time_h.astype(float)%24
 # A single sample at each timepoint cannot validate circadian rhythm.
 level=tab[['sample_id','donor','time_h']].drop_duplicates()
 rep=level.groupby('time_h').sample_id.nunique()
 donors_per_time=level.groupby('time_h').donor.nunique()
 if len(rep)<4 or (donors_per_time<3).any():
  jout(outfile,dict(state='INSUFFICIENT_REPLICATED_TIMEPOINTS',verified=False,
     n_unique_timepoints=len(rep),min_donors_at_time=int(donors_per_time.min()) if len(donors_per_time) else 0,
     note='Do not treat different cell lineages from the same individual/timepoint as biological time replicates'))
  return
 results=[];o=env['o']
 for pathway in env['design']['targets']:
  x=tab[tab.pathway==pathway].copy()
  if x.empty:continue
  x['phase_center_h']=x.time_h
  x['phase_bin']=np.floor((x.time_h%24)/4).astype(int)
  if x.lineage.nunique()<5 or x.donor.nunique()<8:
   results.append(dict(pathway=pathway,state='INSUFFICIENT_DONORS_OR_LINEAGES'));continue
  # Build_design requires >=4 bins PER donor×lineage block: this ensures
  # genuine within-block temporal leverage and no cross-sectional pseudotime claim.
  use=x[TARGET_KEY+['score']].copy()
  dd=o.build_design(use)
  fit=o.fit_models(dd) if dd is not None else None
  if fit is None:
   results.append(dict(pathway=pathway,state='NOT_IDENTIFIABLE_WITHIN_DONOR_LINEAGE',n_donors=x.donor.nunique()))
   continue
  stat=conditional_wild(args,o,dd,args.seed+sum(pathway.encode()))
  results.append(dict(pathway=pathway,state='REPLICATED_TIME_ASSOCIATION_TEST',
   n_sample_sources=x.sample_id.nunique(),n_donors=x.donor.nunique(),**stat))
 if not results:results=[dict(pathway='',state='NO_TARGET_PATHWAYS_IN_EXTERNAL_TABLE')]
 data=pd.DataFrame(results)
 ok=data.state=='REPLICATED_TIME_ASSOCIATION_TEST'
 if ok.any():data.loc[ok,'BHq_external_conditional_M2']=o.bh(data.loc[ok,'conditional_M1null_wild_p'])
 put(data,out/'04_EXTERNAL_REPLICATION_RESULTS.csv')
 jout(outfile,dict(state='FINISHED_WITH_LIMITATIONS',verified=bool(ok.any()),
  n_eligible_pathways=int(ok.sum()),n_submitted_pathways=tab.pathway.nunique(),
  caution='True time-series association only, not causal effect. Sample and donor IDs represent biological replication; need independent assay evidence.'))

def report(env,out):
 need=[out/'01_MODULE_COVARIANCE_AND_VAR_AUDIT.csv',out/'02_CONDITIONAL_M2_RESULTS.csv',out/'03_M2_CV_SUMMARY.csv',out/'04_EXTERNAL_REPLICATION_STATUS.json']
 if not all(p.exists() for p in need):return
 d=pd.read_csv(out/'02_CONDITIONAL_M2_RESULTS.csv');cv=pd.read_csv(out/'03_M2_CV_SUMMARY.csv');external=jin(out/'04_EXTERNAL_REPLICATION_STATUS.json')
 merged=d.merge(cv,on=['pathway','mode'],how='left',validate='one_to_one')
 merged['candidate_support_label']=np.select([
  (merged.BHq_conditional_M1null_wild_p<=.05)&(merged.BHq_calibrated_empirical_p_M2<=.05)&(merged.mean_delta_RMSE>0),
  (merged.BHq_conditional_M1null_wild_p<=.05)&(merged.mean_delta_RMSE>0),
 ],['INTERNAL_MULTIPLE_CONTROLS_PASS_EXPLORATORY','CONDITIONAL_AND_CV_DIRECTION_ONLY'],default='NOT_FULLY_SUPPORTED')
 put(merged,out/'05_INTEGRATED_M2_EVIDENCE.csv')
 lines=['# Figure 6 V3 | M2 conditional and calibrated null audit','',
 '**Phases are frozen cohort-specific B0 inferred phase, not observed ZT.**','',
 '- Uses the same 18 post-selected Tabula pathways; all tests are within-data sensitivities.','- The 199 matched null gene sets per target are fixed across cohorts.','- Empirical null calibration matches estimated gene correlations, per-cohort coverage, gene counts and within-block variance. Matching within-block variance is outcome-dependent; corrected P values are **exploratory**, not a formal confirmatory test.','- The M2 donor wild test is under the **M1 restricted null**, so tests incremental lineage×phase interactions (not M2-vs-M0 omnibus).','- Held-tissue and donor-disjoint CV evaluates centered within-block shape, not unobserved absolute score; shared train donors across folds limit independent fold inference.','- No independent real-ZT validation unless sufficiently replicated real-time cohort supplied.','',
 '## Completed metrics','',
 f'- Modules audited: {len(pd.read_csv(out/"01_MODULE_COVARIANCE_AND_VAR_AUDIT.csv"))}.',
 f'- Conditional target/mode results: {len(d)}.',f'- CV target/mode summaries: {len(cv)}.',
 f'- Independent external data: {external["state"]}.','',
 '## Summary by mode','']
 for mode,g in merged.groupby('mode'):
  lines.extend([f'### {mode}',
  f'- Conditional M2 wild BH q<=0.05: {int((g.BHq_conditional_M1null_wild_p<=.05).sum())}/{len(g)}',
  f'- Calibrated matched gene-set BH q<=0.05: {int((g.BHq_calibrated_empirical_p_M2<=.05).sum())}/{len(g)}',
  f'- CV mean M2 improvement >0: {int((g.mean_delta_RMSE>0).sum())}/{len(g)}',''])
 lines.extend(['## Main outputs','',
 '01_COHORT_GENE_COVERAGE.csv.gz — all target/null sets per original cohort',
 '01_MODULE_COVARIANCE_AND_VAR_AUDIT.csv — fixed set sizes, correlation and block variance',
 '02_CONDITIONAL_M2_RESULTS.csv — donor M1-null wild P and matched-null recalibration',
 '02_COVARIANCE_MATCHED_NULL_SELECTION.csv.gz — detailed chosen fixed controls',
 '03_M2_CV_SUMMARY.csv — truly held-tissue/donor-disjoint descriptive prediction',
 '04_EXTERNAL_REPLICATION_STATUS.json — external validation status, never infer it silently',
 '05_INTEGRATED_M2_EVIDENCE.csv — evidence per target and model condition',''])
 (out/'INTERPRETATION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')

def figures(out):
 """Simple, separate publication-readable summaries; no p-value becomes causal."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 dat=out/'05_INTEGRATED_M2_EVIDENCE.csv'
 if not dat.exists():return
 a=pd.read_csv(dat)
 d=out/'FIGURES';d.mkdir(exist_ok=True)
 names=sorted(a.pathway.unique())
 lab=[n.replace('HALLMARK_','').replace('REACTOME_','R: ').replace('_',' ')[:49] for n in names]
 def save(fig,name):
  for ext in ('png','pdf','svg'):
   fig.savefig(d/(name+'.'+ext),bbox_inches='tight',dpi=320 if ext=='png' else None)
  plt.close(fig)
 for mode in ['baseline','adjusted']:
  m=a[a['mode']==mode].set_index('pathway').reindex(names)
  if m.empty:continue
  fig,ax=plt.subplots(figsize=(9.6,7.4));y=np.arange(len(m))
  def neglog(x):return -np.log10(np.maximum(pd.to_numeric(x,errors='coerce').to_numpy(float),1e-6))
  ax.scatter(neglog(m.BHq_conditional_M1null_wild_p),y-.13,label='Donor wild, M1 conditional null',s=28)
  ax.scatter(neglog(m.BHq_calibrated_empirical_p_M2),y+.13,label='Covariance-matched null (exploratory)',s=24,marker='s')
  ax.axvline(-np.log10(.05),ls='--',c='0.6',lw=1)
  ax.set_yticks(y);ax.set_yticklabels(lab,fontsize=7.3)
  ax.set_xlabel('-log10(selected-panel BH q), descriptive comparison')
  ax.set_title('M2 conditional evidence | '+mode)
  ax.legend(loc='lower right',fontsize=8)
  save(fig,'A_M2_Conditional_'+mode)
  fig,ax=plt.subplots(figsize=(9.6,7.4))
  ax.scatter(m.observed_M2_deltaR2,m.mean_delta_RMSE,s=35)
  for i,name in enumerate(names):
   if np.isfinite(m.mean_delta_RMSE.iloc[i]):ax.annotate(str(i+1), (m.observed_M2_deltaR2.iloc[i],m.mean_delta_RMSE.iloc[i]),fontsize=7)
  ax.axhline(0,ls='--',c='0.6',lw=1)
  ax.set_xlabel('Within-block M2-M1 training ΔR²')
  ax.set_ylabel('Tissue/donor-disjoint mean ΔRMSE (positive = M2 improves)')
  ax.set_title('M2 in-sample fit vs cross-tissue prediction | '+mode)
  save(fig,'B_M2_CV_'+mode)


def do_selftest():
 # Synthetic M1-null and M2-positive clustered data, no Tabula files required.
 root=Path(__file__).resolve().parent/'reference'/'tabula_fig6_Hallmark_Reactome_M012_v1.py'
 if not root.exists():root=Path(__file__).resolve().parent/'tabula_fig6_Hallmark_Reactome_M012_v1.py'
 if not root.exists():print('SELFTEST_SKIP source-dependent checks; syntax only');return
 o=import_file(root,'selftest_m012')
 rng=np.random.default_rng(909);z=[]
 for d in range(12):
  for j,line in enumerate(o.LINEAGES):
   for b in range(6):
    h=b*4+2; c=np.cos(2*np.pi*h/24)
    z.append(dict(tissue='T'+str(d%6),assay='A',donor='D'+str(d),lineage=line,phase_bin=b,phase_center_h=h,
                  n_eff=30,score=(.2*c if j==0 else -.2*c)+rng.normal(0,.10)))
 data=o.build_design(pd.DataFrame(z));f=o.fit_models(data)
 check(data is not None and f is not None,'Synthetic design failed')
 cfg=type('Args',(),dict(wild=59))()
 p=conditional_wild(cfg,o,data,5)
 check(np.isfinite(p['conditional_M1null_wild_p']) and 0<=p['conditional_M1null_wild_p']<=1,'Conditional wild invalid')
 ok=cv_fold(o,pd.DataFrame(z)[:240].copy(),pd.DataFrame(z)[240:].copy(),'baseline')
 check(ok is not None and np.isfinite(ok['deltaRMSE_M2_vs_M1']),'Synthetic donor-disjoint CV failed')
 q=module_pairs(list('ABCDEF'),8,2)
 check(len(q)==8 and all(a!=b for a,b in q),'Gene-pair selection failed')
 print('SELFTEST_PASS: nested M1/M2, donor conditional wild, pair-sampler')

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 ap.add_argument('--project-root',default='.')
 ap.add_argument('--source-dir',default='TABULA_FIG6_STRICT_VALIDATION_V2_1')
 ap.add_argument('--output-dir',default='TABULA_FIG6_M2_CONDITIONAL_NULL_CV_V3')
 ap.add_argument('--stage',choices=['all','audit','conditional','cv','external','report','selftest'],default='all')
 ap.add_argument('--resume',action='store_true')
 ap.add_argument('--sample-cells',type=int,default=240)
 ap.add_argument('--h5ad-block',type=int,default=192)
 ap.add_argument('--pairs-per-set',type=int,default=48)
 ap.add_argument('--min-pair-cells',type=int,default=80)
 ap.add_argument('--pair-chunk',type=int,default=64)
 ap.add_argument('--wild',type=int,default=999)
 ap.add_argument('--calibrated-k',type=int,default=60)
 ap.add_argument('--min-calibrated',type=int,default=30)
 ap.add_argument('--cv-boot',type=int,default=500)
 ap.add_argument('--seed',type=int,default=20261013)
 ap.add_argument('--external-table',default=None)
 a=ap.parse_args()
 if a.stage=='selftest':do_selftest();return
 check(a.wild>=99 and a.pairs_per_set>=16 and a.calibrated_k>=a.min_calibrated>=20,'Unsafe test precision/config')
 root=Path(a.project_root).expanduser().resolve();source=root/a.source_dir;out=root/a.output_dir
 check(source.exists(),f'No V2.1 data folder: {source}')
 out.mkdir(parents=True,exist_ok=True)
 env=load_env(root,source)
 configfile=out/'00_V3_RUN_CONFIG.json'
 config=dict(source_signature=jin(source/'03_SCORE_SIGNATURE.json').get('signature'),
  sample_cells=a.sample_cells,pairs_per_set=a.pairs_per_set,min_pair_cells=a.min_pair_cells,
  seed=a.seed,wild=a.wild,calibrated_k=a.calibrated_k,min_calibrated=a.min_calibrated,
  cv_boot=a.cv_boot)
 if configfile.exists():
  check(jin(configfile)==config,'V3 cached stages use different parameters or a different V2 score signature. Use a fresh --output-dir.')
 else:
  check(not any(out.glob('01_*')) and not any(out.glob('02_*')),
   'V3 output files exist without a config. Choose a fresh --output-dir.')
  jout(configfile,config)
 stages=['audit','conditional','cv','external'] if a.stage=='all' else [a.stage]
 status=out/'00_STATUS.json';jout(status,dict(state='RUNNING',version=VERSION,stages=stages,source=str(source),started=datetime.now().isoformat()))
 try:
  for s in stages:
   if s=='audit':do_audit(a,env,root,out)
   elif s=='conditional':do_conditional(a,env,out)
   elif s=='cv':do_cv(a,env,out)
   elif s=='external':do_external(a,env,out)
   elif s=='report':pass
  report(env,out)
  if a.stage in ('all','report'):
   try:figures(out)
   except Exception as ex:log('FIGURE_WARNING',str(ex))
  completed=all((out/f).exists() for f in ['01_MODULE_COVARIANCE_AND_VAR_AUDIT.csv','02_CONDITIONAL_M2_RESULTS.csv','03_M2_CV_SUMMARY.csv','04_EXTERNAL_REPLICATION_STATUS.json'])
  jout(status,dict(state='COMPLETE' if completed else 'PARTIAL_STAGE_COMPLETE',version=VERSION,stages=stages,
   sample_cells_per_cohort=a.sample_cells,donor_M1null_wild_iterations=a.wild,
   covariance_matched_controls=a.calibrated_k,
   independent_external_provided=bool(a.external_table),finished=datetime.now().isoformat()))
  log('V3 STAGE COMPLETE',','.join(stages),'all_outputs',completed)
 except Exception:
  jout(status,dict(state='FAILED',version=VERSION,stages=stages,traceback=traceback.format_exc()[-4000:]))
  raise

if __name__=='__main__':main()
