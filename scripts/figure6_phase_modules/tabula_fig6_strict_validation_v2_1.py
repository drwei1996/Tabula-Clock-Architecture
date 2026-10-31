#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen Tabula B0 Figure 6 validation v2.

Three NON-equivalent sensitivity controls:
 (i) cohort-invariant, genome/expression-matched NEGATIVE GENE SETS;
(ii) donor-coherent and within-block phase-rotation controls for phase association,
     plus conditional M1-null donor wild signs for M2;
(iii) gene-disjoint cell-state marker proxies and (only if explicitly provided)
      independently measured orthogonal state covariates.

Independent verification, NOT independent discovery replication: candidates came
from the original Tabula atlas. Outputs never overwrite prior results.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, traceback, time
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import h5py

TARGETS = [
 'HALLMARK_OXIDATIVE_PHOSPHORYLATION',
 'HALLMARK_MTORC1_SIGNALING',
 'HALLMARK_UNFOLDED_PROTEIN_RESPONSE',
 'HALLMARK_GLYCOLYSIS',
 'HALLMARK_INFLAMMATORY_RESPONSE',
 'HALLMARK_TNFA_SIGNALING_VIA_NFKB',
 'HALLMARK_IL6_JAK_STAT3_SIGNALING',
 'HALLMARK_INTERFERON_GAMMA_RESPONSE',
 'HALLMARK_TGF_BETA_SIGNALING',
 'HALLMARK_HYPOXIA',
 'REACTOME_RESPIRATORY_ELECTRON_TRANSPORT',
 'REACTOME_CITRIC_ACID_CYCLE_TCA_CYCLE',
 'REACTOME_TNFR1_INDUCED_NF_KAPPA_B_SIGNALING_PATHWAY',
 'REACTOME_SIGNALING_BY_CSF1_M_CSF_IN_MYELOID_CELLS',
 'REACTOME_THE_NLRP3_INFLAMMASOME',
 'REACTOME_IL_6_TYPE_CYTOKINE_RECEPTOR_LIGAND_INTERACTIONS',
 'REACTOME_CELLULAR_RESPONSE_TO_HYPOXIA',
 'REACTOME_MTORC1_MEDIATED_SIGNALLING',
]
# Pre-declared gene panels; remove overlapping target/CLOCK/TF genes GLOBALLY,
# not dynamically in response to outcomes. Remain transcript-derived proxies!
STATE_MARKERS = {
 'CELL_CYCLE': 'MKI67 TOP2A UBE2C CENPF CCNB1 CCNB2 CDC20 PCNA MCM2 MCM4 MCM6 BIRC5 RRM2 TK1 CENPA HMGB2',
 'IMMEDIATE_EARLY': 'FOS FOSB JUN JUNB EGR1 EGR2 DUSP1 NR4A1 ATF3 ZFP36 IER2 IER3 BTG2',
 'HYPOXIA': 'CA9 EGLN3 BNIP3 PDK1 NDRG1 SLC2A1 LDHA VEGFA HILPDA ADM',
 'PROTEOSTASIS': 'DDIT3 ATF4 HSPA5 HSP90B1 DNAJB9 XBP1 PPP1R15A HERPUD1 HYOU1 MANF',
 'IFN_RESPONSE': 'ISG15 IFIT1 IFIT2 IFIT3 MX1 OAS1 OAS2 RSAD2 IFI6 IRF7',
}
KEY=['tissue','assay','donor','lineage','phase_bin','phase_center_h']
BLOCK=['tissue','assay','donor','lineage']
VERSION='v2.1.0'

def log(*v): print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),*v,flush=True)
def jwrite(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(o,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def jread(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def cwrite(p,df):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(p,index=False,compression='gzip' if p.suffix=='.gz' else None)
def assertok(test,msg):
 if not test: raise RuntimeError(msg)
def crc(x):return hashlib.sha256(x.encode()).hexdigest()
def stats(x):
 a=np.asarray(x,float);return [float(np.nanmedian(a)),float(np.nanmean(a))]

def load_modules(root):
 sys.path.insert(0,str(root))
 src=root/'tabula_fig6_Hallmark_Reactome_M012_v1.py'
 assertok(src.exists(),f'Missing original unchanged reference script: {src}')
 import importlib.util
 spec=importlib.util.spec_from_file_location('fig6_orig_v2_reference',src)
 o=importlib.util.module_from_spec(spec);sys.modules[spec.name]=o;spec.loader.exec_module(o)
 p=o.mod_import(root/'tabula_phase_resolved_six_arm_overnight.py','fig6_v2_phase_reference')
 f=o.mod_import(root/'tabula_final_figure_clock_outputs_v1.py','fig6_v2_output_reference')
 helper,_=p.loader_helper(root)
 return o,p,f,helper

def get_manifest(root,oldout,o):
 m=pd.read_csv(root/'08_PHASE_ATLAS'/'TABULA_B0_RUN_MANIFEST.csv')
 m=m[m.status.eq('COMPLETE')].copy()
 oldaudit=pd.read_csv(oldout/'00_PER_COHORT_EXTRACTION_AUDIT.csv')
 # No post-hoc rescue of cohorts with too few eligible cells; match original
 # contributing units exactly. All 60 remain audited in historical report.
 succ=set((str(x.tissue),str(x.assay)) for x in oldaudit.itertuples() if x.status=='SUCCESS')
 cohort=m[m.apply(lambda x:(str(x.tissue_general),str(x.assay)) in succ,axis=1)].copy()
 assertok(len(cohort)>=30 and len(cohort)<=60,'Expected informative frozen cohorts >=30, inspect old audit.')
 oldmeta=pd.read_csv(oldout/'01_ATLAS_UNIT_METADATA.csv.gz')
 assertok(len(oldmeta)>1000,'Original frozen atlas score rows missing')
 return cohort,oldmeta,oldaudit

def pathsource(row,root,phase):
 frozen=phase.path_reloc(row['output_dir'],root)
 qc=jread(frozen/'QC_SUMMARY.json')
 return frozen,phase.path_reloc(qc.get('source_h5ad',qc.get('source_file','')),root)

def collect_universe(args,root,cohort,phase,original,full_gene_list):
 import collections
 common=None;per=[]
 for i,(_,r) in enumerate(cohort.iterrows(),1):
  _,source=pathsource(r,root,phase)
  assertok(source.exists(),f'Missing {source}')
  with h5py.File(source,'r') as h:
   # EXACT one mapping per gene (no forced duplicate feature summation).
   mapped,field=original.make_target_indices(phase,h['var'],full_gene_list)
  u=set(mapped)
  common=u if common is None else common.intersection(u)
  per.append(dict(tissue=r['tissue_general'],assay=r['assay'],n_mapped=len(u)))
  if i%10==0:log('MAP',i,'/',len(cohort),'common',len(common))
 return sorted(common),pd.DataFrame(per)

def prepare(args,root,out,o,phase,orig,cohort,oldmeta):
 hall=o.resolve_gmt(root,args.hallmark_gmt,'Hallmark');react=o.resolve_gmt(root,args.reactome_gmt,'Reactome')
 h,_=o.gmt_read(hall,orig,10,300);r,_=o.gmt_read(react,orig,10,300)
 library={**h,**r}
 missing=[g for g in TARGETS if g not in library]
 assertok(not missing,f'Missing named target pathways in GMT: {missing}')
 targets={g:library[g] for g in TARGETS}
 # Target-specific gene-disjoint covariates. The old version excluded the
 # UNION of all 18 broad pathways, eliminating almost every state marker.
 # For each target, overlap is checked against THAT target, frozen CLOCK17 and
 # TF_MAIN. State proxies and all fixed-null genes remain disjoint by construction:
 # the global matched-null pool excludes the entire ORIGINAL marker inventory.
 clock_and_tf=set(orig.CLOCK17)|set(orig.TF_MAIN)
 all_marker_genes=set().union(*(set(x.split()) for x in STATE_MARKERS.values()))
 markers={};target_to_panels={};records=[]
 for i,(target,target_list) in enumerate(targets.items()):
  excluded=set(target_list)|clock_and_tf
  usable=[]
  for tag,raw in STATE_MARKERS.items():
   original=raw.split();keep=[g for g in original if g not in excluded]
   key=f'STATEPROXY__{i:02d}__{tag}'
   retained=(len(keep)>=5)
   records.append(dict(pathway=target,panel=tag,proxy_module=key,
    n_requested=len(original),n_retained=len(keep),accepted=retained,
    removed=';'.join(g for g in original if g in excluded),
    retained_genes=';'.join(keep)))
   if retained:
    markers[key]=keep
    usable.append(key)
  assertok(len(usable)>=2,
   f'{target}: <2 state marker panels with >=5 target-disjoint genes; audit STATE_MARKERS')
  target_to_panels[target]=usable
 state_union=set().union(*(set(x) for x in markers.values()))
 target_genes=set().union(*map(set,targets.values()))
 # Source universe from the first successful cohort's feature field; union with
 # target genes so partial-coverage genes still get profiling features.
 _,source=pathsource(cohort.iloc[0],root,phase)
 with h5py.File(source,'r') as f:
  field,present,ranking=phase.pick_symbols(f['var'])
  candidates=set(g for g,inds in present.items() if len(inds)==1)
  syms=phase.read_index(f['var']) if field=='var_names' else phase.read_col(f['var'],field)
  c=Counter(orig.canon(x) for x in syms)
  candidates |=set(k for k,v in c.items() if v==1)
 full_gene_list=sorted(candidates|target_genes|state_union)
 common,mapping_audit=collect_universe(args,root,cohort,phase,orig,full_gene_list)
 pool=sorted(set(common)-target_genes-all_marker_genes-clock_and_tf)
 assertok(len(pool)>=400,f'Matched background pool too small ({len(pool)}), review intersection rule')
 if len(pool)>args.max_pool:
  rng=np.random.default_rng(args.seed)
  pool=sorted(rng.choice(pool,args.max_pool,replace=False).tolist())
 jwrite(out/'00_DESIGN.json',dict(version=VERSION,targets=targets,state_markers=markers,
  target_state_mapping=target_to_panels,excluded_marker_gene_inventory=sorted(all_marker_genes),pool=pool,
  n_informative_cohorts=len(cohort),n_old_units=len(oldmeta),hallmark_gmt=str(hall),reactome_gmt=str(react),
  hallmark_sha256=o.hash_file(hall),reactome_sha256=o.hash_file(react),seed=args.seed,
  old_hallmark_audit=jread(root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1'/'01_HALLMARK_SOURCE_REPRODUCTION_AUDIT.json'),
  scientific_scope='REUSE ORIGINAL CONTRIBUTING COHORTS; FROZEN B0; PRESELECTED SAME-ATLAS TARGETS'))
 cwrite(out/'00_STATE_PROXY_GENE_DISJOINT_AUDIT.csv',pd.DataFrame(records))
 cwrite(out/'00_GENE_MAPPING_AUDIT.csv',mapping_audit)
 log('DESIGN COMPLETE: targets',len(targets),'pool',len(pool),'target-specific state modules',len(markers),'cohorts',len(cohort))

def profile(args,root,out,o,phase,orig,helper,cohort):
 design=jread(out/'00_DESIGN.json')
 allgenes=sorted(set(design['pool'])|set().union(*map(set,design['targets'].values())))
 n=len(allgenes);index={g:i for i,g in enumerate(allgenes)}
 s1=np.zeros(n);s2=np.zeros(n);spre=np.zeros(n);sline=np.zeros(n);availability=np.zeros(n)
 for i,(_,r) in enumerate(cohort.iterrows(),1):
  frozen,source=pathsource(r,root,phase)
  pred=pd.read_csv(frozen/'CELL_PHASE.csv.gz',usecols=['cell_id','theta_B0_h','R_B0','total_raw_counts'])
  _,P=phase.posterior(frozen/'PHASE_POSTERIOR.npz',len(pred))
  R=np.abs(P@np.exp(2j*np.pi*np.arange(48)/48))
  assertok(np.max(np.abs(R-pred.R_B0.to_numpy(float)))<1e-6,'B0 posterior mismatch during profile')
  with h5py.File(source,'r') as h:
   joined,_,_,_=helper.exact_join(pred,h,str(r.assay))
   dc=next((x for x in phase.DONOR_FIELDS if phase.read_col(h['obs'],x) is not None),None)
   lc=next((x for x in phase.CELLTYPE_FIELDS if phase.read_col(h['obs'],x) is not None),None)
   assertok(dc is not None and lc is not None,'No donor/lineage metadata')
   donor=np.asarray(phase.read_col(h['obs'],dc))[joined]
   lin=np.array([phase.standard_lineage(x) for x in phase.read_col(h['obs'],lc)[joined]],dtype=object)
   possible=(R>=args.min_R)&np.isin(lin,o.LINEAGES)&np.array([str(x).lower().strip() not in ('','nan','none','na','unknown') for x in donor])
   ids=np.flatnonzero(possible)
   rng=np.random.default_rng(args.seed+sum(str(r.tissue_general).encode())+sum(str(r.assay).encode()))
   # Balanced lineage-stratified sampling avoids domination by abundant cell types.
   strata=[ids[lin[ids]==l] for l in o.LINEAGES]
   sampled=[];per=max(1,args.profile_cells//5)
   for a in strata:
    if len(a):sampled.extend(rng.choice(a,min(len(a),per),replace=False).tolist())
   remaining=args.profile_cells-len(sampled)
   if remaining>0:
    other=np.setdiff1d(ids,np.asarray(sampled,int),assume_unique=False)
    if len(other):sampled.extend(rng.choice(other,min(remaining,len(other)),replace=False).tolist())
   sampled=np.asarray(sorted(set(sampled)),int)
   if not len(sampled):continue
   mapping,_=orig.make_target_indices(phase,h['var'],allgenes)
   available=[g for g in allgenes if g in mapping]
   core=[mapping[g] for g in phase.CORE_DETECTION if g in mapping]
   assertok(len(core)==len(phase.CORE_DETECTION),'Core mapping incomplete')
   cols=[mapping[g] for g in available]+core
   raw,total=phase.read_counts(h['X'],np.asarray(joined)[sampled],cols,args.h5ad_block)
   assertok(np.allclose(total,pred.total_raw_counts.to_numpy(float)[sampled],rtol=1e-5,atol=1e-3),'Profile raw counts mismatch')
   keep=(raw[:,len(available):]>0).sum(axis=1)>=args.min_detect
   raw=raw[keep,:len(available)];total=total[keep];ll=lin[sampled][keep]
   if not len(total):continue
   x=np.log1p(1e4*raw/np.maximum(total[:,None],1))
   means=x.mean(axis=0);sd=x.std(axis=0);detect=(raw>0).mean(axis=0)
   lv=np.array([x[ll==label].mean(axis=0) for label in o.LINEAGES if np.sum(ll==label)>=2])
   spread=lv.std(axis=0) if len(lv)>=2 else np.zeros(len(available))
   ix=np.array([index[g] for g in available]);s1[ix]+=means;s2[ix]+=sd;spre[ix]+=detect;sline[ix]+=spread;availability[ix]+=1
  if i%5==0:log('PROFILE',i,'/',len(cohort),'profiled genes',int(np.sum(availability>0)))
 assertok(np.sum(availability>0)>=len(design['pool']),'Profile incomplete for reference pool')
 denom=np.maximum(availability,1)
 p=pd.DataFrame({'gene':allgenes,'mean_logCP10k':s1/denom,'sd_logCP10k':s2/denom,
  'detect_rate':spre/denom,'lineage_spread':sline/denom,
  'n_cohorts_profiled':availability.astype(int)})
 cwrite(out/'01_GLOBAL_GENE_MATCHING_FEATURES.csv.gz',p)
 log('PROFILE COMPLETE',len(p),'genes')

def make_nulls(args,out):
 design=jread(out/'00_DESIGN.json')
 feat=pd.read_csv(out/'01_GLOBAL_GENE_MATCHING_FEATURES.csv.gz').set_index('gene')
 pool=[g for g in design['pool'] if g in feat.index]
 cols=['mean_logCP10k','sd_logCP10k','detect_rate','lineage_spread']
 vals=feat[cols].astype(float)
 centers=vals.loc[pool].median();scale=(vals.loc[pool].quantile(.75)-vals.loc[pool].quantile(.25)).clip(lower=.025)
 Z=(vals-centers)/scale
 pool_arr=np.array(pool,dtype=object)
 pv=Z.loc[pool,cols].to_numpy(float)
 assertok(np.isfinite(pv).all(),'Nonfinite pool matching features')
 rng=np.random.default_rng(args.seed)
 sets={};audit=[];used=None
 for target,tgenes in design['targets'].items():
  # Target genes absent from all informative H5ADs cannot be matched and should
  # not be present in actual module scoring. Keep fixed membership denominator.
  matchable=[g for g in tgenes if g in Z.index and feat.loc[g,'n_cohorts_profiled']>0]
  assertok(len(matchable)>=10,f'Target {target} has <10 mapped genes to profile')
  assertok(len(pool)>=len(matchable),f'Insufficient pool for {target}')
  tvec=Z.loc[matchable,cols].to_numpy(float)
  # Need sampling variation without selecting outcome-fitting statistics.
  # Restrict to 50 nearest neighbours; fixed mapping across ALL cohorts.
  dist=np.sum((tvec[:,None,:]-pv[None,:,:])**2,axis=2)
  nn=np.argsort(dist,axis=1)[:,:min(60,len(pool))]
  pool_index={g:i for i,g in enumerate(pool)}
  for rep in range(args.null_sets):
   order=rng.permutation(len(matchable));chosen=[None]*len(matchable);used=set()
   for k in order:
    candidates=nn[k];idx=None
    for q in rng.permutation(candidates[:min(len(candidates),40)]):
     if int(q) not in used:idx=int(q);break
    if idx is None:
     for q in np.argsort(dist[k]):
      if int(q) not in used:idx=int(q);break
    if idx is None:raise RuntimeError('Could not assign unique matched gene')
    used.add(idx);chosen[k]=str(pool_arr[idx])
   nm=f"FIXNULL__{TARGETS.index(target):02d}__{rep:03d}"
   sets[nm]=chosen
   audit.append(dict(pathway=target,null=nm,n_target_matchable=len(matchable),
    null_genes=len(set(chosen)),median_scaled_sq_distance=float(np.median([dist[j,pool_index[g]] for j,g in enumerate(chosen)])),
    genes=';'.join(chosen)))
 # Save once: permanent selection independent of phase, pathway fits or p-values.
 jwrite(out/'02_FIXED_NULL_GENE_SETS.json',dict(sets=sets,seed=args.seed,n_null=args.null_sets,
  targets=TARGETS,matching_features=cols,version=VERSION))
 cwrite(out/'02_FIXED_NULL_MATCH_AUDIT.csv.gz',pd.DataFrame(audit))
 log('FIXED SETS COMPLETE',len(sets),'never reselected by cohort')

def scoring(args,root,out,o,phase,orig,helper,cohort,oldmeta):
 d=jread(out/'00_DESIGN.json');nul=jread(out/'02_FIXED_NULL_GENE_SETS.json')
 sets={**d['targets'],**d['state_markers'],**nul['sets']}
 union=sorted(set().union(*map(set,sets.values())))
 config=dict(targets=d['targets'],markers=d['state_markers'],nulls=nul['sets'],min_R=args.min_R,
   min_detect=args.min_detect,min_coverage=args.min_coverage,min_bin_weight=args.min_bin_weight,
   min_bin_effective=args.min_bin_effective,seed=args.seed,version=VERSION)
 signature=crc(json.dumps(config,sort_keys=True))
 cache=out/'CACHE_PER_COHORT';cache.mkdir(parents=True,exist_ok=True)
 assertok(args.min_R==.3 and args.min_detect==2 and args.min_bin_weight==8 and args.min_bin_effective==8,
  'Current defaults frozen to reproduce source cohort selection. Use baseline values.')
 from argparse import Namespace
 opt=Namespace(resume=args.resume,min_R=args.min_R,min_detect=args.min_detect,
   min_coverage=args.min_coverage,min_bin_weight=args.min_bin_weight,
   min_bin_effective=args.min_bin_effective,min_cells=50,h5ad_block=args.h5ad_block,
   extra_background_genes=0,matched_nulls=0,seed=args.seed)
 audits=[]
 for i,(_,row) in enumerate(cohort.iterrows(),1):
  log('SCORE',i,'/',len(cohort),row.tissue_general,row.assay)
  a=o.extract_cohort_fast(root,orig,phase,helper,row,sets,union,[],opt,signature,cache)
  assertok(a['status']=='SUCCESS',f'Reference informative cohort failed unexpectedly: {a}')
  audits.append({k:v for k,v in a.items() if k!='null_audit'})
  cwrite(out/'03_COHORT_EXTRACTION_AUDIT.csv',pd.DataFrame(audits))
 meta,arr,names=o.assemble_cached(cache,sets,[],out,signature)
 assertok(len(meta)==len(oldmeta),f'Unit count mismatch {len(meta)} != {len(oldmeta)}')
 for t in (meta,oldmeta):
  for k in KEY:t[k]=t[k].astype(str)
  assertok(not t.duplicated(KEY).any(),'Duplicate original/v2 analytical unit keys')
 a=meta.set_index(KEY).index;b=oldmeta.set_index(KEY).index
 assertok(len(a.difference(b))==0 and len(b.difference(a))==0,'New vs frozen atlas analytical unit keys do not match')
 prev=np.load(root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1'/'01_ALL_PATHWAY_SCORES_WIDE.npz',allow_pickle=False)
 prevnames=prev['pathway'].astype(str).tolist();prevmat=prev['scores']
 ixold=oldmeta.set_index(KEY).index.get_indexer(a)
 checks=[]
 for t in d['targets']:
  assertok(t in names and t in prevnames,f'Target missing from old/new matrix {t}')
  new=arr[:,names.index(t)].astype(float);old=prevmat[ixold,prevnames.index(t)].astype(float)
  both=np.isfinite(new)&np.isfinite(old)
  miss=int((np.isfinite(new)!=np.isfinite(old)).sum())
  err=float(np.max(abs(new[both]-old[both]))) if both.any() else float('inf')
  checks.append(dict(pathway=t,matched_rows=int(both.sum()),unmatched_rows=miss,max_abs_diff=err))
 assertok(all(q['unmatched_rows']==0 and q['max_abs_diff']<1e-4 for q in checks),
  'FAIL CLOSED: old-v2 target scores differ or missingness changed')
 cwrite(out/'03_FROZEN_TARGET_SCORE_REPRODUCTION.csv',pd.DataFrame(checks))
 jwrite(out/'03_SCORE_SIGNATURE.json',dict(signature=signature,state='EXTRACTION_COMPLETE',
  n_units=len(meta),n_genes=len(union),n_null_sets=len(nul['sets']),n_cohorts=len(cohort)))
 log('SCORES COMPLETE; all selected original target scores reproduced')

def external_state(meta,table):
 if not table:return meta,[]
 path=Path(table);assertok(path.is_file(),f'Orthogonal external state table missing: {path}')
 ext=pd.read_csv(path)
 need=set(KEY);assertok(need.issubset(ext.columns),'Orthogonal state table must have exactly six KEY columns')
 c=[x for x in ext.columns if x.startswith('state_orthogonal_')]
 assertok(c,'External table needs one or more columns named state_orthogonal_*')
 assertok(not ext.duplicated(KEY).any(),'External orthogonal state table has duplicate unit keys')
 for k in KEY:meta[k]=meta[k].astype(str);ext[k]=ext[k].astype(str)
 merged=meta.merge(ext[KEY+c],how='left',on=KEY,validate='one_to_one')
 assertok(merged[c].notna().all().all(),'External orthogonal state table has missing unit values; no silent subset')
 assertok(np.isfinite(merged[c].to_numpy(float)).all(),'External orthogonal state data nonfinite')
 return merged,c

def phase_design_with_shift(o,design,shift_by,scope):
 """Rebuild X1/X2 by rotating *predictor* phase; response/weights/blocks fixed.
 Avoids illicit independent pseudo-replicates and preserves within-block shapes.
 """
 z=design['z'];codes=design['block_codes'];N=len(z)
 if scope=='donor':
  groups=pd.factorize(z.donor.astype(str),sort=True)[0]
 else:groups=codes
 shifts=shift_by[groups]
 t=(z.phase_center_h.to_numpy(float)+4*shifts)%24
 phi=2*np.pi*t/24
 co=np.cos(phi);si=np.sin(phi)
 size=np.bincount(codes)
 def centre(a):
  avg=np.bincount(codes,weights=a,minlength=len(size))/size
  return a-avg[codes]
 co=centre(co);si=centre(si)
 X0=design['X0'];X1=np.column_stack([X0,co,si]);ln=z.lineage.astype(str).to_numpy()
 X2=np.column_stack([X1]+[((ln==ll).astype(float)[:,None]*np.column_stack([co,si])) for ll in o.LINEAGES[1:]])
 return X0,X1,X2

def fit_delta(o,des,model='M1',Xs=None):
 y=des['y'];w=des['w'];den=float(np.dot(y*w,y))
 X0,X1,X2=(des['X0'],des['X1'],des['X2']) if Xs is None else Xs
 s0=o.weighted_lstsq(X0,y,w)[3]
 s1=o.weighted_lstsq(X1,y,w)[3]
 s2=o.weighted_lstsq(X2,y,w)[3]
 if model=='M1':return (s0-s1)/den
 if model=='M2':return (s1-s2)/den
 return (s0-s2)/den

def phase_rotations(args,o,des,seed):
 rng=np.random.default_rng(seed)
 observed1=fit_delta(o,des,'M1');observed_omni=fit_delta(o,des,'OMNI')
 ret=dict(observed_delta_M1=observed1,observed_delta_M2=fit_delta(o,des,'M2'),
  observed_delta_M2total=observed_omni)
 for scope in ('donor','block'):
  n=des['n_donors'] if scope=='donor' else des['nblocks']
  cnt1=cnt2=0;vals1=[];vals2=[]
  for b in range(args.rotations):
   shifts=rng.integers(0,6,size=n)
   xs=phase_design_with_shift(o,des,shifts,scope)
   a=fit_delta(o,des,'M1',xs);c=fit_delta(o,des,'OMNI',xs)
   vals1.append(a);vals2.append(c)
   cnt1+=a>=observed1-1e-12;cnt2+=c>=observed_omni-1e-12
  ret.update({f'{scope}_rotation_p_M1':(1+cnt1)/(args.rotations+1),
   f'{scope}_rotation_p_M2total':(1+cnt2)/(args.rotations+1),
   f'{scope}_rotation_null95_M1':float(np.quantile(vals1,.95)),
   f'{scope}_rotation_null95_M2total':float(np.quantile(vals2,.95))})
 return ret

def module_design(o,meta,arr,names,name,state_cols=None,common=None):
 j=names.index(name);sel=np.isfinite(arr[:,j])
 if common is not None:sel &= common
 if state_cols:
  for c in state_cols:sel &= np.isfinite(meta[c].to_numpy(float))
 # IMPORTANT: build_design includes every 'state_*' column; do not accidentally
 # control for marker proxies in baseline mode.
 base=KEY+(['n_eff'] if 'n_eff' in meta.columns else [])
 z=meta.loc[sel,base].copy()
 if state_cols:
  for i,c in enumerate(state_cols):z['state_'+str(i)]=meta.loc[sel,c].to_numpy(float)
 z['score']=arr[sel,j].astype(float)
 return o.build_design(z) if len(z)>=100 else None

def validation(args,root,out,o):
 design=jread(out/'00_DESIGN.json');nn=jread(out/'02_FIXED_NULL_GENE_SETS.json')['sets']
 meta=pd.read_csv(out/'01_ATLAS_UNIT_METADATA.csv.gz')
 pack=np.load(out/'01_ALL_PATHWAY_SCORES_WIDE.npz',allow_pickle=False)
 arr=pack['scores'];names=pack['pathway'].astype(str).tolist()
 assertok(len(meta)==len(arr),'Scored atlas row count mismatch')
 state_markers=[x for x in design['state_markers'] if x in names]
 for k in state_markers:meta['state_proxy_'+k]=arr[:,names.index(k)]
 meta,orth=external_state(meta,args.external_state_table)
 results=[];matches=[];rots=[];covariate_audit=[]
 # Each target has its own disjoint RNA panels. Baseline, adjusted and null
 # comparisons use a fixed COMMON row set for that particular target only.
 # Requiring complete data for all ~90 state panels would select a tiny,
 # potentially biased subset and is scientifically unnecessary.
 for target in TARGETS:
  short=f'{TARGETS.index(target):02d}'
  nullnames=sorted(x for x in nn if x.startswith('FIXNULL__'+short+'__'))
  assertok(len(nullnames)==args.null_sets,f'Wrong number of controls for {target}')
  required=design['target_state_mapping'][target]
  candidates=sorted('state_proxy_'+x for x in required if 'state_proxy_'+x in meta)
  assertok(len(candidates)>=2,
   f'{target}: fewer than two scored disjoint state panels; inspect coverage audit')
  # Choose exactly two nonoverlapping state panels based on COMPLETE-CASE
  # COVERAGE ONLY, never test outcomes/p-values or model effect sizes.
  # This avoids an artificial complete-case bottleneck caused by demanding
  # 3-5 optional markers simultaneously in all donor/tissue/cohort strata.
  from itertools import combinations
  options=[]
  basic=np.isfinite(arr[:,names.index(target)])
  for c in orth:basic &= np.isfinite(meta[c].to_numpy(float))
  for pair in combinations(candidates,2):
   take=basic.copy()
   for c in pair:take &= np.isfinite(meta[c].to_numpy(float))
   d=meta.loc[take]
   valid=(len(d)>=100 and d.donor.astype(str).nunique()>=10 and
     d.tissue.nunique()>=4 and d.lineage.nunique()==len(o.LINEAGES))
   if valid:options.append((int(take.sum()),pair,take))
  assertok(options,
   f'{target}: no two nonoverlapping RNA-state panels jointly cover 5 lineages, >=10 donors and >=100 units')
  options.sort(key=lambda x:(-x[0],x[1]))
  ncommon,selected,common_rows=options[0]
  proxy_cols=list(selected)
  controls=proxy_cols+orth
  covariate_audit.append(dict(pathway=target,n_common_rows=ncommon,
   n_rna_state_panels=len(proxy_cols),rna_state_panels=';'.join(proxy_cols),
   n_orthogonal_panels=len(orth),orthogonal_panels=';'.join(orth),
   n_scored_candidate_panels=len(candidates),n_viable_pairs=len(options),
   selection_rule='max_complete_case_rows_two_panels_no_outcome_information'))
  test_modes=[('baseline_common_rows',[]),('disjoint_RNA_proxies',proxy_cols)]
  if orth:
   test_modes.extend([('orthogonal_only',orth),('joint',controls)])
  for mode,fields in test_modes:
   local=meta.copy()
   for n,c in enumerate(fields):local['state_'+str(n)]=meta[c]
   # Fixed gene-set nulls must have same model rows; permit per-set missingness
   # only with paired target refit on its INTERSECTION, not different denominators.
   des=module_design(o,local,arr,names,target,fields,common_rows)
   assertok(des is not None,f'No viable target design: {target}/{mode}')
   fit=o.fit_models(des)
   assertok(fit is not None,f'No identifiable target fit: {target}/{mode}')
   obs1=fit['deltaR2_M1_vs_M0'];obs2=fit['deltaR2_M2_vs_M1']
   tab=dict(pathway=target,mode=mode,n_donors=fit['n_donors'],n_units=fit['n_units'],
    deltaR2_M1=obs1,deltaR2_M2=obs2,original_main_n_units=int(meta.shape[0]),
    state_fields=';'.join(fields),orthogonal_covariate_available=bool(orth))
   # Report effective covariates AFTER block-centering: a block-constant
   # clinical characteristic has no within-block identifying power.
   tab['estimated_n_nuisance_columns']=int(des['X0'].shape[1])
   wc=o.wild_boot(des,fit,args.wild_permutations,np.random.default_rng(args.seed+TARGETS.index(target)*59+len(fields)))
   tab.update(wc)
   vals1=[];vals2=[];nsame=0;err=0
   for idx,nm in enumerate(nullnames):
    j=names.index(nm)
    pair=np.isfinite(arr[:,j])&np.isfinite(arr[:,names.index(target)])&common_rows
    for f in fields:pair &= np.isfinite(local[f].to_numpy(float))
    if np.sum(pair)<80:err+=1;continue
    # Paired ORIGINAL and NULL on exactly the same observations.
    d0=module_design(o,local,arr,names,target,fields,pair)
    dn=module_design(o,local,arr,names,nm,fields,pair)
    if d0 is None or dn is None:err+=1;continue
    f0=o.fit_models(d0);fn=o.fit_models(dn)
    if f0 is None or fn is None:err+=1;continue
    assertok(len(d0['y'])==len(dn['y']) and d0['z'][KEY].reset_index(drop=True).equals(dn['z'][KEY].reset_index(drop=True)),
     'Paired target-null unit alignment failed')
    val1=fn['deltaR2_M1_vs_M0']-f0['deltaR2_M1_vs_M0']
    val2=fn['deltaR2_M2_vs_M1']-f0['deltaR2_M2_vs_M1']
    vals1.append(val1);vals2.append(val2);nsame+=1
    if mode=='baseline_common_rows':
     matches.append(dict(pathway=target,null=nm,n_units=int(len(d0['y'])),
      null_M1=fn['deltaR2_M1_vs_M0'],target_M1_paired=f0['deltaR2_M1_vs_M0'],
      null_M2=fn['deltaR2_M2_vs_M1'],target_M2_paired=f0['deltaR2_M2_vs_M1'],
      null_minus_target_M1=val1,null_minus_target_M2=val2))
   assertok(nsame>=max(30,args.null_sets//2),f'Only {nsame}/{args.null_sets} matched null models for {target}/{mode}')
   # Pairwise superiority of null over the same-row target. Exchangeability
   # is heuristic because matching ignores network correlations; exploratory.
   tab.update(n_valid_fixed_nulls=nsame,n_unusable_nulls=err,
     fixed_empirical_p_M1=(1+sum(v>=0 for v in vals1))/(nsame+1),
     fixed_empirical_p_M2=(1+sum(v>=0 for v in vals2))/(nsame+1),
     fixed_null_minus_target_median_M1=float(np.median(vals1)),
     fixed_null_minus_target_median_M2=float(np.median(vals2)))
   results.append(tab)
   log('VALIDATE',target,mode,'valid nulls',nsame,'M1',round(obs1,4),'M2',round(obs2,4))
   if mode=='baseline_common_rows' or (mode=='joint' and orth) or (mode=='disjoint_RNA_proxies' and not orth):
    rt=phase_rotations(args,o,des,args.seed+TARGETS.index(target)*137+len(fields))
    rots.append(dict(pathway=target,mode=mode,**rt))
  cwrite(out/'04_RESULTS_PARTIAL.csv',pd.DataFrame(results))
 cwrite(out/'04_TARGET_SPECIFIC_STATE_COVERAGE_AUDIT.csv',pd.DataFrame(covariate_audit))
 cwrite(out/'04_FIXED_MATCHED_NULL_PAIRED_DISTRIBUTION.csv.gz',pd.DataFrame(matches))
 cwrite(out/'04_FIXED_NULL_AND_STATE_RESULTS.csv',pd.DataFrame(results))
 cwrite(out/'04_PHASE_ROTATION_RESULTS.csv',pd.DataFrame(rots))
 a=pd.DataFrame(results);a['wild_BHq_M1_within_test_panel']=np.nan;a['wild_BHq_M2_within_test_panel']=np.nan
 for mode,g in a.groupby('mode'):
  a.loc[g.index,'wild_BHq_M1_within_test_panel']=o.bh(g.wild_p_M1)
  a.loc[g.index,'wild_BHq_M2_within_test_panel']=o.bh(g.wild_p_M2)
 cwrite(out/'04_FIXED_NULL_AND_STATE_RESULTS.csv',a)
 rr=pd.DataFrame(rots)
 for mode,g in rr.groupby('mode'):
  for p in [c for c in rr.columns if c.endswith('_p_M1') or c.endswith('_p_M2total')]:
   rr.loc[g.index,'BHq_'+p]=o.bh(g[p])
 cwrite(out/'04_PHASE_ROTATION_RESULTS.csv',rr)
 # Reuse already completed, independent held-tissue/donor-disjoint CV. No CV
 # retraining on the selected v2 targets; source is fully identified.
 oldcv=root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1'/'04_TISSUE_DONOR_DISJOINT_WITHIN_BLOCK_CV.csv'
 if oldcv.exists():
  cc=pd.read_csv(oldcv)
  cc=cc[cc.pathway.isin(TARGETS)]
  cvsum=cc.groupby('pathway',as_index=False).agg(n_held_tissues=('held_tissue','nunique'),
   prior_cv_M1=('delta_RMSE_M1','mean'),prior_cv_M2=('delta_RMSE_M2','mean'),
   prior_cv_fraction_M1_positive=('delta_RMSE_M1',lambda x:float(np.mean(x>0))),
   prior_cv_fraction_M2_positive=('delta_RMSE_M2',lambda x:float(np.mean(x>0))))
  cwrite(out/'04_CROSSCHECK_PRIOR_HELD_TISSUE_CV.csv',a.merge(cvsum,on='pathway',how='left'))
 report(out,design,a,rr,orth)
 if not args.no_figures:render_figures(out,a,rr)
 jwrite(out/'00_STATUS.json',dict(state='COMPLETE',version=VERSION,
  n_tested_pathways=len(TARGETS),n_fixed_nulls_per_target=args.null_sets,
  n_mode_results=len(a),n_rotations=args.rotations,n_informative_cohorts=design['n_informative_cohorts'],
  n_units=len(meta),orthogonal_covariates=orth,completed=datetime.now().isoformat()))
 log('COMPLETE',out)

def report(out,design,a,r,orth):
 lines=['# Figure 6 Strict Validation v2','',
  '**This is SAME-ATLAS sensitivity, not independent time or biological validation.**','',
  '- Frozen B0, original 2444 units, and original source pathway-score reproduction were checked.',
  '- Original GMT targets chosen from previous in-atlas discoveries; post-selection bias remains.',
  '- Fixed null genes matched ONCE globally, shared unchanged across cohorts. Null gene effects may still differ by gene-gene covariance.',
  '- Null p = within-pair empirical rank over the same score rows, not whole-reactome inference.',
  '- Donor-phase and block-phase rotations test absolute B0 phase alignment (M1, M2-vs-M0 omnibus), not conditional M2-vs-M1.',
  '- M2 conditional association uses M1-null donor wild sign flips; small donor number limits asymptotic precision.',
  '- State panels avoid literal tested-target/CLOCK17 genes, are defined separately for each target, and are NOT independent biological measurements.',
  '- Exactly 2 candidate RNA state panels are chosen per target using complete-case donor/lineage coverage only (not outcomes).',
  '- Orthogonal state measurements supplied: '+('; '.join(orth) if orth else 'NONE, so independent cell-state adjustment NOT established'),
  '- Phase dependence does not establish true circadian oscillation or causal CLOCK effects.','',
  '## Summary: empirical fixed-null and donor wild at nominal 0.05 (preselected tests only)','']
 for mode,g in a.groupby('mode'):
  lines+=['### '+mode,
   f'- M1: fixed-null p<=0.05 {sum(g.fixed_empirical_p_M1<=.05)}/{len(g)}, donor wild p<=0.05 {sum(g.wild_p_M1<=.05)}/{len(g)}',
   f'- M2: fixed-null p<=0.05 {sum(g.fixed_empirical_p_M2<=.05)}/{len(g)}, donor wild p<=0.05 {sum(g.wild_p_M2<=.05)}/{len(g)}','']
 lines+=['## Key files','',
  '04_FIXED_NULL_AND_STATE_RESULTS.csv (per-target sensitivity, BH within fixed selected panel)',
  '04_PHASE_ROTATION_RESULTS.csv (donor-coherent and block independent phase rotation)',
  '04_FIXED_MATCHED_NULL_PAIRED_DISTRIBUTION.csv.gz (same-row target-null paired diagnostics)',
  '04_CROSSCHECK_PRIOR_HELD_TISSUE_CV.csv (prior CV crosscheck; no refitting)',
  '03_FROZEN_TARGET_SCORE_REPRODUCTION.csv (must have unmatched=0, error<1e-4)',
  '02_FIXED_NULL_MATCH_AUDIT.csv.gz (fixed gene memberships and match distances)',
  '00_STATE_PROXY_GENE_DISJOINT_AUDIT.csv (target-wise overlap exclusions)',
  '04_TARGET_SPECIFIC_STATE_COVERAGE_AUDIT.csv (target-specific scored controls and complete-case rows)','',
  '## Interpretation checklist','',
  '1. Prefer effect size and held-tissue CV over in-sample significant fraction.',
  '2. Check both donor- and block-rotation; tests answer different null hypotheses.',
  '3. An M2 Wald/wild result can be amplitude-based, not only phase shift.',
  '4. Require actual independent time-stamped biological replicates for rhythm claims.',
  '5. Report number of unique donors (typically ~21), not 2444 independent units.',
  '6. Fixed-set nulls are matched controls, NOT a causal test or true held-out cohort.','']
 (out/'INTERPRETATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def render_figures(out,results,rotations):
 """Separate reviewer-ready exploratory validation panels, PNG/PDF/SVG."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 dest=out/'FIGURES';dest.mkdir(exist_ok=True)
 def save(fig,name):
  for ext in ('png','pdf','svg'):
   fig.savefig(dest/(name+'.'+ext),bbox_inches='tight',dpi=320 if ext=='png' else None)
  plt.close(fig)
 def label(x):return x.replace('HALLMARK_','').replace('REACTOME_','R:').replace('_',' ')[:55]
 a=results[results['mode']=='baseline_common_rows'].copy()
 a=a.sort_values('fixed_empirical_p_M1',ascending=False)
 yy=np.arange(len(a));fig,ax=plt.subplots(figsize=(9,7.6))
 ax.scatter(a['fixed_empirical_p_M1'],yy,s=34,label='M1 fixed-set empirical p',color='#4E79A7')
 ax.scatter(a['fixed_empirical_p_M2'],yy,s=34,marker='s',label='M2 fixed-set empirical p',color='#E15759')
 ax.axvline(.05,color='gray',ls='--',lw=1)
 ax.set_yticks(yy);ax.set_yticklabels([label(t) for t in a.pathway],fontsize=7.5)
 ax.set_xlabel('Paired target-versus-fixed matched-null empirical p (exploratory)')
 ax.set_xlim(0,1.03);ax.legend(fontsize=8,loc='lower right')
 ax.set_title('A | Cohort-invariant matched-gene negative controls')
 save(fig,'A_fixed_matched_null_empirical_p')
 for term in ('M1','M2'):
  cols=['baseline_common_rows','disjoint_RNA_proxies']
  if 'joint' in set(results['mode']):cols.append('joint')
  sub=results[results['mode'].isin(cols)].copy()
  pv=sub.pivot(index='pathway',columns='mode',values='deltaR2_'+term).reindex(TARGETS)
  if len(pv)==0:continue
  vals=pv.iloc[:,0]
  fig,ax=plt.subplots(figsize=(9,7.6));ys=np.arange(len(pv))
  colors={'baseline_common_rows':'#4E79A7','disjoint_RNA_proxies':'#E15759','joint':'#59A14F'}
  for c in cols:
   if c not in pv:continue
   ax.scatter(pv[c].to_numpy(float),ys,label=c.replace('_',' '),s=24,color=colors[c],alpha=.9)
  ax.set_yticks(ys);ax.set_yticklabels([label(t) for t in pv.index],fontsize=7.5)
  ax.set_xlabel('Within-block added explained fraction (ΔR²)')
  ax.set_title(f'B | {term} comparison before/after measured state controls')
  ax.legend(fontsize=8,loc='lower right');save(fig,'B_'+term+'_state_sensitivity')
 r=rotations[rotations['mode']=='baseline_common_rows'].copy().set_index('pathway').reindex(TARGETS)
 for term in ('M1','M2total'):
  pcols=['donor_rotation_p_'+term,'block_rotation_p_'+term]
  fig,ax=plt.subplots(figsize=(9,7.6));ys=np.arange(len(r))
  ax.scatter(r[pcols[0]],ys,label='Donor-coherent rotation',color='#4E79A7',s=27)
  ax.scatter(r[pcols[1]],ys,label='Independent block rotation',color='#E15759',s=27,marker='s')
  ax.axvline(.05,color='gray',ls='--',lw=1)
  ax.set_yticks(ys);ax.set_yticklabels([label(t) for t in r.index],fontsize=7.5)
  ax.set_xlim(0,1.03);ax.set_xlabel('Empirical phase-rotation p (exploratory)')
  ax.legend(fontsize=8);ax.set_title('C | '+term+' inferred-phase alignment nulls')
  save(fig,'C_'+term+'_donor_block_rotation')

def selftest():
 # Algebraic invariant: simultaneous shared donor phase shift by 0 leaves design
 # scores unchanged, as does all block shift by 0.
 class O:LINEAGES=['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
 n=60;codes=np.repeat(np.arange(10),6)
 z=pd.DataFrame({'phase_center_h':np.tile(np.arange(6)*4+2,10),
 'lineage':np.repeat(O.LINEAGES*2,6),'donor':np.repeat(['a','b']*5,6)})
 co=np.cos(2*np.pi*z.phase_center_h.to_numpy(float)/24);si=np.sin(2*np.pi*z.phase_center_h.to_numpy(float)/24)
 X0=np.zeros((n,0));X1=np.column_stack([X0,co,si]);X2=np.column_stack([X1]+[(z.lineage.eq(l).to_numpy(float)[:,None]*np.column_stack([co,si])) for l in O.LINEAGES[1:]])
 d=dict(z=z,block_codes=codes,n_donors=2,nblocks=10,X0=X0,X1=X1,X2=X2)
 a,b,c=phase_design_with_shift(O,d,np.zeros(2,int),'donor')
 assert np.allclose(b,X1) and np.allclose(c,X2)
 a,b,c=phase_design_with_shift(O,d,np.zeros(10,int),'block')
 assert np.allclose(b,X1) and np.allclose(c,X2)
 print('SELFTEST_PASS; phase-design zero-shift invariance')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--project-root',required=False,default='.')
 ap.add_argument('--hallmark-gmt',default=None);ap.add_argument('--reactome-gmt',default=None)
 ap.add_argument('--stage',choices=['all','design','profile','nulls','score','validate','selftest'],default='all')
 ap.add_argument('--resume',action='store_true')
 ap.add_argument('--null-sets',type=int,default=99)
 ap.add_argument('--rotations',type=int,default=499)
 ap.add_argument('--wild-permutations',type=int,default=499)
 ap.add_argument('--profile-cells',type=int,default=512)
 ap.add_argument('--max-pool',type=int,default=5500)
 ap.add_argument('--h5ad-block',type=int,default=192)
 ap.add_argument('--seed',type=int,default=20261012)
 ap.add_argument('--min_R',type=float,default=.3)
 ap.add_argument('--min-detect',type=int,default=2)
 ap.add_argument('--min-coverage',type=float,default=.45)
 ap.add_argument('--min-bin-weight',type=float,default=8)
 ap.add_argument('--min-bin-effective',type=float,default=8)
 ap.add_argument('--no-figures',action='store_true')
 ap.add_argument('--external-state-table',default=None,
  help='Optional INDEPENDENTLY MEASURED unit-level table with KEY columns + state_orthogonal_*; otherwise RNA proxies only')
 args=ap.parse_args()
 if args.stage=='selftest':selftest();return
 root=Path(args.project_root).expanduser().resolve()
 out=root/'TABULA_FIG6_STRICT_VALIDATION_V2_1';out.mkdir(parents=True,exist_ok=True)
 status=out/'00_STATUS.json'
 try:
  o,phase,orig,helper=load_modules(root)
  oldout=root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1'
  assertok(jread(oldout/'00_STATUS.json')['state']=='COMPLETE','Original M012 analysis not COMPLETE')
  assertok(jread(oldout/'01_HALLMARK_SOURCE_REPRODUCTION_AUDIT.json')['status']=='PASS',
   'Original hallmark replication FAIL')
  cohort,oldmeta,oldaudit=get_manifest(root,oldout,o)
  if args.stage in ('all','design'):
   if not (args.resume and (out/'00_DESIGN.json').exists()):
    jwrite(status,dict(state='DESIGN_RUNNING',version=VERSION))
    prepare(args,root,out,o,phase,orig,cohort,oldmeta)
   jwrite(status,dict(state='DESIGN_COMPLETE',version=VERSION))
  if args.stage in ('all','profile'):
   if not (args.resume and (out/'01_GLOBAL_GENE_MATCHING_FEATURES.csv.gz').exists()):
    jwrite(status,dict(state='PROFILE_RUNNING',version=VERSION))
    profile(args,root,out,o,phase,orig,helper,cohort)
    jwrite(status,dict(state='PROFILE_COMPLETE',version=VERSION))
  if args.stage in ('all','nulls'):
   if not (args.resume and (out/'02_FIXED_NULL_GENE_SETS.json').exists()):
    jwrite(status,dict(state='NULL_GENERATION_RUNNING',version=VERSION))
    make_nulls(args,out)
    jwrite(status,dict(state='NULL_GENERATION_COMPLETE',version=VERSION))
  if args.stage in ('all','score'):
   if not (args.resume and (out/'03_SCORE_SIGNATURE.json').exists()):
    jwrite(status,dict(state='SCORING_RUNNING',version=VERSION))
    scoring(args,root,out,o,phase,orig,helper,cohort,oldmeta)
    jwrite(status,dict(state='SCORING_COMPLETE',version=VERSION))
  if args.stage in ('all','validate'):
   assertok((out/'03_SCORE_SIGNATURE.json').exists(),'No successful null score extraction; run --stage score')
   jwrite(status,dict(state='VALIDATING',version=VERSION))
   validation(args,root,out,o)
 except Exception as e:
  jwrite(status,dict(state='FAILED',version=VERSION,error=str(e),traceback=traceback.format_exc()[-5000:]))
  raise

if __name__=='__main__':main()
