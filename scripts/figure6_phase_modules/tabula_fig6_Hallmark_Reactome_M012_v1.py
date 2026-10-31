#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabula Sapiens Figure 6: lineage identity versus inferred CLOCK-state association.

All modules in human Hallmark + human Reactome tested with common M0/M1/M2.
Tabula only, frozen cohort-specific B0 posterior. No true sampling time and no
claim of causal circadian regulation. No GSE292219 data.

M0: fixed donor×tissue×assay×lineage intercept, and optionally within-block
    effective-size/phase-bin confidence proxy; block intercepts are removed
    by within-block demeaning.
M1: M0 + common sin(theta), cos(theta).
M2: M1 + lineage-specific sin(theta), cos(theta) deviations (8 d.f.).

Model outcomes: cohort-standardized non-clock module mean-z scores aggregated
into donor×lineage×B0 posterior phase bins. Donor-balanced weighted least
squares; donor-cluster sandwich Wald tests.  Higher-tier donor-wild-bootstrap
and donor-disjoint held-tissue validation run for all Hallmark and top Reactome.

Matched gene-set nulls (optional, on by default): selected a priori Hallmark
and Reactome pathway names, per-cohort mean/SD/lineage-variability-matched
non-target genes, fixed within cohort but may vary across cohorts.  This is
not a genome-wide fixed-set negative control nor experimental validation.

REQUIREMENTS: place script alongside
  tabula_final_figure_clock_outputs_v1.py
  tabula_phase_resolved_six_arm_overnight.py
  tabula_core17_input_audit.py
and previous 08_PHASE_ATLAS/TABULA_B0_RUN_MANIFEST.csv.
Input GMTs must be HUMAN symbols: h.all.*.Hs.symbols.gmt and
c2.cp.reactome.*.Hs.symbols.gmt. NEVER substitute .Mm. GMT.

python tabula_fig6_Hallmark_Reactome_M012_v1.py --project-root "$PWD" \\
 --hallmark-gmt "$PWD/h.all.v2026.1.Hs.symbols.gmt" \\
 --reactome-gmt "$PWD/c2.cp.reactome.v2026.1.Hs.symbols.gmt" \\
 --stage all --resume

The source, old output, and original frozen B0 files are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import f as f_dist
from scipy.stats import norm as normal_dist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINEAGES=['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
L_COL={'B cell':'#4E79A7','T cell':'#E15759','Macrophage/Monocyte':'#59A14F','Fibroblast':'#F28E2B','Endothelial':'#9467BD'}
W=2*np.pi/24
BLOCK_KEYS=['tissue','assay','donor','lineage']
UNIT_KEYS=BLOCK_KEYS+['phase_bin','phase_center_h']
COV_TESTS=['HALLMARK_MYC_TARGETS_V1','HALLMARK_MTORC1_SIGNALING','HALLMARK_OXIDATIVE_PHOSPHORYLATION',
 'HALLMARK_UNFOLDED_PROTEIN_RESPONSE','HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY','HALLMARK_DNA_REPAIR',
 'HALLMARK_TNFA_SIGNALING_VIA_NFKB','HALLMARK_IL6_JAK_STAT3_SIGNALING',
 'HALLMARK_INTERFERON_GAMMA_RESPONSE','HALLMARK_INFLAMMATORY_RESPONSE']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':350})

def log(*x):print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def write_table(z,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 z.to_csv(p,index=False,compression='gzip' if str(p).endswith('.gz') else None)
def write_json(d,p):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def hash_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def bh(a):
 p=np.asarray(a,float);q=np.full(len(p),np.nan);ix=np.flatnonzero(np.isfinite(p))
 if len(ix):
  order=ix[np.argsort(p[ix])];v=p[order]*len(order)/np.arange(1,len(order)+1)
  q[order]=np.clip(np.minimum.accumulate(v[::-1])[::-1],0,1)
 return q

def mod_import(p,name):
 spec=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m

def resolve_gmt(root,explicit,kind):
 if explicit:
  p=Path(explicit).expanduser().resolve()
  if not p.is_file():raise FileNotFoundError(f'{kind} GMT not found: {p}')
 else:
  if kind=='Hallmark':names=['h.all.*.Hs.symbols.gmt']
  else:names=['c2.cp.reactome.*.Hs.symbols.gmt']
  options=sorted(set(x for pattern in names for x in root.glob(pattern)))
  if len(options)!=1:raise FileNotFoundError(
   f'Expected ONE human {kind} GMT in katv2, found {len(options)}. '
   f'Use --{kind.lower()}-gmt PATH. Download the HUMAN MSigDB collection; mouse .Mm is not suitable.')
  p=options[0]
 name=p.name.lower()
 if not re.search(r'(?:^|\.)hs(?:\.|$)',name):raise ValueError(f'Only HUMAN Hs GMT is accepted for Tabula: {p.name}')
 if kind=='Hallmark' and not name.startswith('h.all.'):raise ValueError('Hallmark GMT must be h.all.*.Hs.symbols.gmt')
 if kind=='Reactome' and not ('reactome' in name and 'c2.cp.' in name):raise ValueError('Reactome GMT must be c2.cp.reactome.*.Hs.symbols.gmt')
 return p

def gmt_read(path,orig,min_genes,max_genes):
 sets={};audit=[]
 with open(path,encoding='utf-8-sig') as f:
  for line in f:
   x=line.rstrip('\n\r').split('\t')
   if len(x)<3:continue
   name=x[0].strip()
   genes=list(dict.fromkeys(orig.canon(g) for g in x[2:] if str(g).strip()))
   genes=[g for g in genes if g not in orig.CLOCK17 and g not in orig.TF_MAIN]
   eligible=min_genes<=len(genes)<=max_genes
   audit.append({'collection':'Hallmark' if name.startswith('HALLMARK_') else 'Reactome',
                 'pathway':name,'n_after_exclusion':len(genes),'eligible':eligible})
   if eligible:sets[name]=genes
 return sets,pd.DataFrame(audit)

def select_priority(sets,max_reactome):
 # Explicit lexical rule decided before looking at new Reactome statistics.
 hits=[]
 patterns=[r'NFKB|NF_KB|TNFR|TNF_',r'INTERLEUKIN|IL_6|JAK_STAT',
           r'RESPIRATORY_ELECTRON_TRANSPORT|OXIDATIVE_PHOSPHORYLATION|TCA_CYCLE',
           r'UNFOLDED_PROTEIN_RESPONSE|ER_STRESS|CHAPERONE']
 for patt in patterns:
  group=sorted(x for x in sets if x.startswith('REACTOME_') and re.search(patt,x,re.I))
  hits.extend(group[:max_reactome])
 return [m for m in COV_TESTS if m in sets]+list(dict.fromkeys(hits))

def weighted_lstsq(X,y,w):
 XX=np.asarray(X,float)*np.sqrt(w)[:,None];yy=y*np.sqrt(w)
 coef=np.linalg.lstsq(XX,yy,rcond=None)[0]
 pred=X@coef;res=y-pred;sse=float(np.dot(w,res*res))
 return coef,pred,res,sse

def build_design(df,include_neff=True):
 z=df.copy().sort_values(BLOCK_KEYS+['phase_bin']).reset_index(drop=True)
 z['donor']=z['donor'].astype(str)
 z['block']=z[BLOCK_KEYS].astype(str).agg('|'.join,axis=1)
 # Only blocks with >=4 distinct posterior phase bins; avoid sparse-phase fits.
 cnt=z.groupby('block').phase_bin.transform('nunique')
 z=z[cnt>=4].copy().reset_index(drop=True)
 if z.empty:return None
 y=pd.to_numeric(z.score,errors='coerce').to_numpy(float)
 keep=np.isfinite(y)&np.isfinite(z.phase_center_h.to_numpy(float))
 z=z.loc[keep].reset_index(drop=True);y=y[keep]
 if z.empty:return None
 bc=np.cos(W*z.phase_center_h.to_numpy(float));bs=np.sin(W*z.phase_center_h.to_numpy(float))
 codes,keys=pd.factorize(z['block'],sort=True)
 donors, donor_keys=pd.factorize(z.donor,sort=True)
 size=np.bincount(codes);ndonor=np.bincount(donors)
 # donor-balanced: within each donor, each block equally and each phase bin equally weighted.
 nd_b=np.bincount(pd.factorize(z[['donor','block']].drop_duplicates().donor)[0]) if False else None
 block_donor=z.groupby('block',sort=True).donor.first();nblocks=block_donor.value_counts()
 w=np.array([1/(size[bb]*int(nblocks.loc[z.donor.iloc[i]])) for i,bb in enumerate(codes)],float)
 w*=len(donor_keys)/w.sum()
 def centre(v):
  v=np.asarray(v,float)
  mu=np.bincount(codes,weights=v,minlength=len(keys))/np.maximum(size,1)
  return v-mu[codes]
 yy=centre(y)
 x0=np.zeros((len(z),0))
 if include_neff and 'n_eff' in z:
  n=pd.to_numeric(z.n_eff,errors='coerce').to_numpy(float)
  if np.isfinite(n).all():
   xx=centre(np.log1p(np.maximum(n,0)))
   if np.std(xx)>1e-8:x0=xx[:,None]
 # Optional cell-state controls: these scores are fixed non-clock gene sets,
 # computed from the same cells. They are SENSITIVITY covariates, not independently
 # measured biological cell states; overlap with targets is screened upstream.
 state_vars=sorted(c for c in z.columns if c.startswith('state_'))
 for state in state_vars:
  v=pd.to_numeric(z[state],errors='coerce').to_numpy(float)
  if not np.isfinite(v).all():raise ValueError('Nonfinite cell-state sensitivity covariate '+state)
  x=centre(v)
  if np.std(x)>1e-8:x0=np.column_stack([x0,x])
 c=centre(bc);s=centre(bs)
 X1=np.column_stack([x0,c,s]);X0=x0
 ln=z.lineage.astype(str).to_numpy()
 X2=np.column_stack([X1]+[vv* np.column_stack((c,s)) for ll in LINEAGES[1:] for vv in [(ln==ll).astype(float)[:,None]]])
 # Correction: lineage interaction raw phi terms must be de-meaned within blocks;
 # centered common sin and cos are already block-centered (lineage constant in block).
 return dict(z=z,y=yy,X0=X0,X1=X1,X2=X2,w=w,donor_codes=donors,donors=donor_keys,
  nblocks=len(keys),n_tissues=z.tissue.nunique(),block_codes=codes,block_keys=keys,
  n_lineages=z.lineage.nunique(),n_donors=len(donor_keys))

def cluster_wald(X,y,w,donor_codes,from_idx):
 """CR1 donor-cluster Wald for added regressors. Descriptive with G~22."""
 n,k=X.shape;G=int(np.max(donor_codes))+1 if len(donor_codes) else 0
 if k==0 or G<=k-1:return np.nan
 sw=np.sqrt(w);A=X*sw[:,None];b=y*sw
 if np.linalg.matrix_rank(A)<k:return np.nan
 coef=np.linalg.lstsq(A,b,rcond=None)[0];r=b-A@coef
 bread=np.linalg.pinv(A.T@A)
 meat=np.zeros((k,k))
 for d in range(G):
  take=donor_codes==d
  u=A[take].T@r[take]
  meat+=np.outer(u,u)
 V=bread@meat@bread*(G/(G-1))*(max(n-1,1)/max(n-k,1))
 Z=coef[from_idx:];VZ=V[from_idx:,from_idx:]
 q=len(Z)
 if np.linalg.matrix_rank(VZ)<q:return np.nan
 stat=float(Z@np.linalg.pinv(VZ)@Z/q)
 return float(f_dist.sf(max(stat,0),q,G-1))

def fit_models(data):
 y=data['y'];w=data['w'];m0=data['X0'];m1=data['X1'];m2=data['X2']
 if data['n_donors']<8 or data['n_lineages']<5 or data['nblocks']<20:return None
 out=[]
 for X in (m0,m1,m2):
  beta,pred,res,sse=weighted_lstsq(X,y,w)
  out.append((beta,pred,res,sse))
 denom=float(np.dot(y*w,y))
 if denom<1e-9:return None
 s0,s1,s2=[t[3] for t in out]
 d1=(s0-s1)/denom;d2=(s1-s2)/denom
 p1=cluster_wald(m1,y,w,data['donor_codes'],m0.shape[1])
 p2=cluster_wald(m2,y,w,data['donor_codes'],m1.shape[1])
 return {'deltaR2_M1_vs_M0':float(d1),'deltaR2_M2_vs_M1':float(d2),
  'withinR2_M0':float(1-s0/denom),'withinR2_M1':float(1-s1/denom),'withinR2_M2':float(1-s2/denom),
  'p_donor_cluster_M1':p1,'p_donor_cluster_M2':p2,
  'n_units':len(y),'n_donors':data['n_donors'],'n_blocks':data['nblocks'],
  'n_tissues':data['n_tissues'],'n_lineages':data['n_lineages'],
  'beta_common_cos':float(out[1][0][-2]),'beta_common_sin':float(out[1][0][-1]),
  'phase_M1_h':float(np.angle(out[1][0][-2]+1j*out[1][0][-1])%(2*np.pi)/W),
  'amp_M1':float(np.hypot(*out[1][0][-2:])),
  'full_rank_M2':int(np.linalg.matrix_rank(m2))==m2.shape[1],
  '_fits':out}

def wild_boot(data,fit,rep,rng):
 """Donor wild sign flip: M0-null for M1, M1-null for M2. Exploratory p."""
 if rep<=0:return {'wild_p_M1':np.nan,'wild_p_M2':np.nan}
 y=data['y'];w=data['w'];X0=data['X0'];X1=data['X1'];X2=data['X2'];dc=data['donor_codes']
 fits=fit['_fits'];denom=float(np.dot(y*w,y));obs1=(fits[0][3]-fits[1][3])/denom;obs2=(fits[1][3]-fits[2][3])/denom
 greater1=0;greater2=0
 for j in range(rep):
  signs=rng.choice([-1.,1.],size=data['n_donors'])[dc]
  y0=fits[0][1]+fits[0][2]*signs
  _,_,_,a=weighted_lstsq(X0,y0,w);_,_,_,b=weighted_lstsq(X1,y0,w)
  t1=(a-b)/denom
  y1=fits[1][1]+fits[1][2]*signs
  _,_,_,c=weighted_lstsq(X1,y1,w);_,_,_,d=weighted_lstsq(X2,y1,w)
  t2=(c-d)/denom
  greater1+=t1>=obs1-1e-12;greater2+=t2>=obs2-1e-12
 return {'wild_p_M1':(1+greater1)/(rep+1),'wild_p_M2':(1+greater2)/(rep+1)}

def donor_boot(data,rep,rng):
 # Resample unique donors and refit using their original records. Duplicate draws
 # receive independent bootstrap donor labels to keep effective sampling honest.
 if rep<=0:return dict(ci_low_M1=np.nan,ci_high_M1=np.nan,ci_low_M2=np.nan,ci_high_M2=np.nan)
 z=data['z'];Ds=np.asarray(data['donors']);vals=[]
 groups={don:z[z.donor.astype(str).eq(str(don))] for don in Ds}
 for i in range(rep):
  draw=rng.choice(Ds,len(Ds),replace=True)
  chunks=[]
  for j,dd in enumerate(draw):
   q=groups[dd].copy();q['donor']='bootstrap_'+str(j);chunks.append(q)
  ds=build_design(pd.concat(chunks,ignore_index=True))
  if ds is None:continue
  f=fit_models(ds)
  if f is not None:vals.append((f['deltaR2_M1_vs_M0'],f['deltaR2_M2_vs_M1']))
 if len(vals)<max(30,int(rep*.5)):return dict(ci_low_M1=np.nan,ci_high_M1=np.nan,ci_low_M2=np.nan,ci_high_M2=np.nan)
 a=np.array(vals)
 return dict(ci_low_M1=float(np.quantile(a[:,0],.025)),ci_high_M1=float(np.quantile(a[:,0],.975)),
  ci_low_M2=float(np.quantile(a[:,1],.025)),ci_high_M2=float(np.quantile(a[:,1],.975)))

# ---- Raw Tabula extraction (frozen B0 alignment is checked, never refitted) ----

def match_nulls(module_names,modules,mu,sd,lin_spread,pool_indices,good,genes,n_ctrl,rng):
 """Match each target gene to a non-target gene using cohort-specific mean, SD,
 lineage dispersion of expression. Gene identity can vary across cohorts.
 Matching uses observed data ONLY for nuisance gene statistics, never outcomes
 from harmonic models. Results labeled as within-cohort matched controls.
 """
 if n_ctrl<=0:return {},[]
 pool=np.asarray([k for k in pool_indices if good[k]],int)
 if len(pool)<100:return {},[]
 Z=np.column_stack([np.log1p(np.maximum(mu,0)),np.log1p(np.maximum(sd,0)),np.log1p(np.maximum(np.nan_to_num(lin_spread,nan=0),0))])
 scales=np.maximum(np.nanstd(Z[pool],axis=0),1e-3)
 Z=Z/scales
 by_target={};mapping=[]
 for name in module_names:
  if name not in modules:continue
  tgt=np.asarray([k for k in modules[name] if good[k]],int)
  if len(tgt)<10:continue
  allowed=np.asarray([k for k in pool if k not in set(tgt)],int)
  if len(allowed)<len(tgt):continue
  # Per-target candidate neighbor cache, avoids redoing distances every control.
  neighbor=[]
  for g in tgt:
   dd=np.sum((Z[allowed]-Z[g])**2,axis=1)
   shortlist=allowed[np.argpartition(dd,min(len(dd)-1,40))[:min(len(dd),41)]]
   neighbor.append(shortlist)
  for i in range(n_ctrl):
   chosen=[];used=set()
   for cand in neighbor:
    aa=[int(v) for v in cand if int(v) not in used]
    if not aa:
     aa=[int(v) for v in allowed if int(v) not in used]
    if not aa:break
    g=int(rng.choice(aa[:min(30,len(aa))]));chosen.append(g);used.add(g)
   if len(chosen)!=len(tgt):continue
   key=f'NULL__{name}__{i:02d}'
   by_target[key]=chosen
   # Save overlap and descriptive statistic; detailed genes in audit only for
   # first control per pathway to avoid an enormous output.
   if i==0:
    mapping.append(dict(target=name,control=key,n_genes=len(tgt),
     median_matching_distance=float(np.median(np.sum((Z[tgt]-Z[chosen])**2,axis=1))),
     example_genes=';'.join(genes[g] for g in chosen[:10])))
 return by_target,mapping

def extract_cohort_fast(root,orig,prev,helper,row,sets,union,priority,args,signature,cache):
 import h5py
 tissue=str(row['tissue_general']);assay=str(row['assay']);tag=re.sub('[^a-z0-9]+','_',tissue.lower())+'__'+re.sub('[^a-z0-9]+','_',assay.lower())
 f_npz=cache/(tag+'.npz');f_units=cache/(tag+'_UNITS.csv.gz');f_audit=cache/(tag+'.json')
 if args.resume and f_npz.exists() and f_units.exists() and f_audit.exists():
  old=read_json(f_audit)
  if old.get('signature')==signature and old.get('status')=='SUCCESS':
   log('CACHE',tag);return old
 frozen=prev.path_reloc(row['output_dir'],root)
 phfile=frozen/'CELL_PHASE.csv.gz';post=frozen/'PHASE_POSTERIOR.npz';qcfile=frozen/'QC_SUMMARY.json'
 for p in (phfile,post,qcfile):
  if not p.exists():raise FileNotFoundError(str(p))
 pred=pd.read_csv(phfile,usecols=['cell_id','theta_B0_h','R_B0','total_raw_counts'])
 if len(pred)!=int(row['n_cells']):raise ValueError('Manifest cell count mismatch')
 _,P=prev.posterior(post,len(pred))
 R=np.abs(P@np.exp(2j*np.pi*np.arange(48)/48))
 if np.max(np.abs(R-pred.R_B0.to_numpy(float)))>1e-6:raise ValueError('Frozen B0 posterior R mismatch')
 pqc=read_json(qcfile);source=prev.path_reloc(pqc.get('source_h5ad',pqc.get('source_file','')),root)
 if not source.exists():raise FileNotFoundError(f'Missing input H5AD {source}')
 with h5py.File(source,'r') as h:
  joined,_,_,_=helper.exact_join(pred,h,assay)
  dc=next((x for x in prev.DONOR_FIELDS if prev.read_col(h['obs'],x) is not None),None)
  lc=next((x for x in prev.CELLTYPE_FIELDS if prev.read_col(h['obs'],x) is not None),None)
  if not dc or not lc:raise ValueError(f'No donor/lineage metadata {tag}')
  donor=np.asarray(prev.read_col(h['obs'],dc))[joined]
  lin=np.array([prev.standard_lineage(x) for x in prev.read_col(h['obs'],lc)[joined]],dtype=object)
  possible=(R>=args.min_R)&np.isin(lin,LINEAGES)&np.array([str(s).strip().lower() not in ('','nan','none','na','unknown') for s in donor])
  # Return sample-wide unique mappings using exactly the same 74G/Tabula helper.
  mapped,field=orig.make_target_indices(prev,h['var'],union)
  if len(mapped)<len(prev.CORE_DETECTION):raise ValueError('CORE17 mapping unexpectedly incomplete')
  if field=='var_names':syms=prev.read_index(h['var'])
  else:syms=prev.read_col(h['var'],field)
  allnames=list(dict.fromkeys(orig.canon(s) for s in syms))
  observed={g for g in allnames if g not in orig.CLOCK17 and g not in orig.TF_MAIN}
  extra=sorted(observed-set(union))
  rng=np.random.default_rng(args.seed + sum(tag.encode()))
  if args.extra_background_genes>0 and len(extra)>args.extra_background_genes:extra=rng.choice(extra,args.extra_background_genes,replace=False).tolist()
  if args.extra_background_genes==0:extra=[]
  mapping,field=orig.make_target_indices(prev,h['var'],sorted(set(union)|set(extra)))
  usable={name:[g for g in gs if g in mapping] for name,gs in sets.items()}
  usable={name:gs for name,gs in usable.items() if len(gs)>=max(5,int(np.ceil(args.min_coverage*len(sets[name]))))}
  selected=sorted(set().union(*map(set,usable.values()))) if usable else []
  background=sorted(g for g in extra if g in mapping)
  genes=sorted(set(selected)|set(background));n=len(genes)
  if not n or not usable:raise ValueError(f'No eligible genes/pathways {tag}')
  core_cols=[mapping[g] for g in prev.CORE_DETECTION]
  cols=[mapping[g] for g in genes]+core_cols
  ii=[];n_elig=0;mu=np.zeros(n);vv=np.zeros(n)
  bylin=np.zeros((len(LINEAGES),n));nlin=np.zeros(len(LINEAGES))
  expected=pred.total_raw_counts.to_numpy(float)
  log('PASS1',tag,'read genes',n,'pathways',len(usable))
  for st in range(0,len(pred),args.h5ad_block):
   take=np.arange(st,min(st+args.h5ad_block,len(pred)))
   take=take[possible[take]]
   if not len(take):continue
   raw,sums=prev.read_counts(h['X'],np.asarray(joined)[take],cols,args.h5ad_block)
   if not np.allclose(sums,expected[take],rtol=1e-5,atol=1e-3):raise ValueError(f'B0 raw count sums mismatch {tag}')
   keep=(raw[:,n:]>0).sum(axis=1)>=args.min_detect
   take=take[keep];raw=raw[keep];sums=sums[keep]
   if not len(take):continue
   x=np.log1p(1e4*raw[:,:n]/np.maximum(sums[:,None],1))
   mu+=x.sum(axis=0);vv+=np.sum(x*x,axis=0);n_elig+=len(take);ii.extend(take.tolist())
   for j,l in enumerate(LINEAGES):
    mask=lin[take]==l
    if mask.any():bylin[j]+=x[mask].sum(axis=0);nlin[j]+=int(mask.sum())
  if n_elig<args.min_cells:raise ValueError(f'Too few eligible cells: {tag}: {n_elig}')
  mu/=n_elig;sd=np.sqrt(np.maximum(vv/n_elig-mu*mu,0))
  lin_means=np.divide(bylin,nlin[:,None],out=np.full_like(bylin,np.nan),where=nlin[:,None]>0)
  lin_spread=np.nanstd(lin_means,axis=0)
  good=sd>1e-6;gene_idx={g:i for i,g in enumerate(genes)}
  mods={name:[gene_idx[g] for g in gs if good[gene_idx[g]]] for name,gs in usable.items()}
  mods={name:ix for name,ix in mods.items() if len(ix)>=max(5,int(np.ceil(args.min_coverage*len(sets[name]))))}
  if not mods:raise ValueError('No pathway passed SD/coverage')
  pool=[gene_idx[g] for g in background if good[gene_idx[g]]]
  matched,mapping_rows=match_nulls(priority,mods,mu,sd,lin_spread,pool,good,genes,args.matched_nulls,rng)
  mods.update(matched)
  names=sorted(mods);ridx=[];cidx=[];dat=[]
  for j,name in enumerate(names):
   ix=mods[name];ridx.extend(ix);cidx.extend([j]*len(ix));dat.extend([1/len(ix)]*len(ix))
  M=sp.csr_matrix((np.asarray(dat,np.float32),(ridx,cidx)),shape=(n,len(names)),dtype=np.float32)
  chosen=np.asarray(ii,int);bp=P.reshape(len(P),6,8).sum(axis=2)
  codes,uniq=pd.factorize(pd.MultiIndex.from_frame(pd.DataFrame({'donor':donor[chosen].astype(str),'lineage':lin[chosen].astype(str)})),sort=True)
  ng=len(uniq);bins=ng*6
  agg=np.zeros((bins,len(names)),dtype=np.float64)
  ws=np.zeros(bins);ws2=np.zeros(bins);counts=np.bincount(codes,minlength=ng)
  log('PASS2',tag,'eligible cells',n_elig,'sets incl nulls',len(names))
  for st in range(0,len(chosen),args.h5ad_block):
   take=chosen[st:st+args.h5ad_block];gcode=codes[st:st+args.h5ad_block]
   raw,sums=prev.read_counts(h['X'],np.asarray(joined)[take],cols,args.h5ad_block)
   z=(np.log1p(1e4*raw[:,:n]/np.maximum(sums[:,None],1))-mu)/np.maximum(sd,1e-12)
   scores=np.asarray(z.astype(np.float32)@M,dtype=np.float32)
   weights=bp[take].astype(np.float32)
   rr=np.repeat(gcode*6,6)+np.tile(np.arange(6),len(gcode));cc=np.repeat(np.arange(len(gcode)),6)
   ww=weights.ravel();ws+=np.bincount(rr,weights=ww,minlength=bins)
   ws2+=np.bincount(rr,weights=ww*ww,minlength=bins)
   GG=sp.csr_matrix((ww,(rr,cc)),shape=(bins,len(gcode)),dtype=np.float32)
   agg+=(GG@scores).astype(np.float64)
  neff=np.divide(ws*ws,ws2,out=np.zeros(bins),where=ws2>1e-12)
  inds=np.where((ws>=args.min_bin_weight)&(neff>=args.min_bin_effective))[0]
  arr=np.divide(agg[inds],ws[inds,None],out=np.full((len(inds),len(names)),np.nan),where=ws[inds,None]>1e-12)
  meta=[]
  for k in inds:
   bb,phase_bin=divmod(int(k),6);d,l=uniq[bb]
   meta.append((tissue,assay,str(d),str(l),phase_bin,float(orig.BINS[phase_bin]),float(neff[k]),float(ws[k]),int(counts[bb])))
  df=pd.DataFrame(meta,columns=UNIT_KEYS+['n_eff','weight_sum','n_source_cells'])
  write_table(df,f_units)
  np.savez_compressed(f_npz,scores=arr.astype(np.float32),pathway=np.array(names))
  outcome=dict(status='SUCCESS',signature=signature,tissue=tissue,assay=assay,tag=tag,n_units=len(df),
   n_cells_used=n_elig,n_genes=n,n_measured_genes=sum(good),n_pathways=len(usable),n_pathways_passed=len(mods)-len(matched),
   n_nulls=len(matched),n_null_targets=sum(name in mods for name in priority),n_source_cells=len(pred),
   input_h5ad=str(source),annotation_field=field,note='B0 FROZEN; raw H5AD counts re-normalized log1p(CP10k) like old Hallmark code',
   null_audit=mapping_rows)
  write_json(outcome,f_audit)
  return outcome

def extract_atlas(root,sets,priority,args,out,signature):
 cache=out/'CACHE_PER_COHORT';cache.mkdir(parents=True,exist_ok=True)
 old=mod_import(root/'tabula_final_figure_clock_outputs_v1.py','old_fig6_extractor')
 prev=mod_import(root/'tabula_phase_resolved_six_arm_overnight.py','old_clock_phase')
 helper,_=prev.loader_helper(root)
 manifest=root/'08_PHASE_ATLAS'/'TABULA_B0_RUN_MANIFEST.csv'
 if not manifest.exists():raise FileNotFoundError(str(manifest))
 d=pd.read_csv(manifest);d=d[d.status.eq('COMPLETE')].copy()
 if args.pilot:d=d.head(args.pilot)
 if d.empty:raise RuntimeError('No COMPLETE frozen B0 cohorts in manifest')
 union=sorted(set().union(*map(set,sets.values())))
 if len(union)>args.max_gene_union:raise ValueError(
  f'Matched collections require {len(union)} unique genes > --max-gene-union {args.max_gene_union}. '
  'Raise knowingly or use --reactome-regex pre-registered category restriction.')
 audits=[]
 for i,(_,r) in enumerate(d.iterrows(),1):
  log(f'[{i}/{len(d)}]',r['tissue_general'],r['assay'])
  try:q=extract_cohort_fast(root,old,prev,helper,r,sets,union,priority,args,signature,cache)
  except Exception as exc:
   log('FAILED cohort',str(exc));q={'status':'FAILED','tissue':str(r['tissue_general']),
    'assay':str(r['assay']),'error':str(exc),'traceback':traceback.format_exc()[-1200:]}
  audits.append({k:v for k,v in q.items() if k!='null_audit'})
  write_table(pd.DataFrame(audits),out/'00_PER_COHORT_EXTRACTION_AUDIT.csv')
 failures=[x for x in audits if x['status']!='SUCCESS']
 if failures and not args.allow_cohort_failures:
  raise RuntimeError(f'{len(failures)} cohort(s) failed. Inspect 00_PER_COHORT_EXTRACTION_AUDIT.csv; '
    'not proceeding on silent subset. To proceed explicitly use --allow-cohort-failures.')
 if len(audits)-len(failures)<max(3,len(d)//2):raise RuntimeError('Insufficient successful cohorts')
 return assemble_cached(cache,sets,priority,out,signature)

def assemble_cached(cache,sets,priority,out,signature):
 cachefiles=sorted(cache.glob('*.npz'))
 # A fixed global column list is reconstructed from actual cached cohorts only.
 allnames=set();arrs=[];metas=[]
 for file in cachefiles:
  au=file.with_suffix('.json');ufile=cache/(file.stem+'_UNITS.csv.gz')
  if not au.exists() or not ufile.exists():continue
  audit=read_json(au)
  if audit.get('status')!='SUCCESS' or audit.get('signature')!=signature:continue
  a=np.load(file,allow_pickle=False);ns=a['pathway'].astype(str).tolist();allnames.update(ns)
  mm=pd.read_csv(ufile);mat=a['scores'];
  if len(mm)!=len(mat):raise ValueError('Matrix/row mismatch '+str(file))
  metas.append(mm);arrs.append((ns,mat))
 if not metas:raise RuntimeError('No cached module data')
 names=sorted(allnames);idx={m:i for i,m in enumerate(names)}
 total=sum(len(x) for x in metas)
 mat=np.full((total,len(names)),np.nan,dtype=np.float32)
 k=0
 for meta,(cols,v) in zip(metas,arrs):
  mat[k:k+len(meta),[idx[m] for m in cols]]=v
  k+=len(meta)
 tab=pd.concat(metas,ignore_index=True)
 if tab.duplicated(UNIT_KEYS).any():raise ValueError('Duplicate donor+tissue+lineage+phase unit keys')
 write_table(tab,out/'01_ATLAS_UNIT_METADATA.csv.gz')
 np.savez_compressed(out/'01_ALL_PATHWAY_SCORES_WIDE.npz',scores=mat,pathway=np.array(names))
 write_table(pd.DataFrame({'pathway':names,'collection':['MatchedNull' if m.startswith('NULL__') else ('Hallmark' if m.startswith('HALLMARK_') else 'Reactome') for m in names],
  'n_aggregated_units':[int(np.isfinite(mat[:,i]).sum()) for i in range(len(names))]}),out/'01_MODULE_AVAILABILITY.csv')
 log('COMBINED',total,'unit rows',len(names),'modules incl matched nulls')
 return tab,mat,names

# ---- All module nested models, multiple tests and confirmation tiers ----

def hallmark_reproduction_audit(meta,mat,names,root,out):
 """Compare newly extracted Hallmark pathway scores with already frozen output
 at donor×tissue×assay×lineage×phase-bin resolution. Fail on divergence.
 """
 prior=root/'TABULA_FINAL_CLOCK_OUTPUT_FIGURE'/'03_TABULA_MODULE_UNIT_SCORES.csv.gz'
 if not prior.exists():
  log('WARNING: old Hallmark reference missing; audit skipped');return {'status':'NO_PRIOR_HALLMARK_FILE'}
 old=pd.read_csv(prior,usecols=UNIT_KEYS+['pathway','score'])
 old=old[old.pathway.astype(str).str.startswith('HALLMARK_')].copy()
 hh=[m for m in names if m.startswith('HALLMARK_')]
 if len(hh)<40:raise ValueError('Expected >=40 Hallmark modules for reproduction audit')
 parts=[]
 for m in hh:
  v=mat[:,names.index(m)]
  d=meta[UNIT_KEYS].copy();d['pathway']=m;d['new_score']=v
  parts.append(d[np.isfinite(v)])
 now=pd.concat(parts,ignore_index=True)
 for t in (old,now):
  t['donor']=t.donor.astype(str)
  t['phase_bin']=t.phase_bin.astype(int)
  t['phase_center_h']=t.phase_center_h.astype(float)
 keys=UNIT_KEYS+['pathway']
 joined=old.merge(now,on=keys,how='outer',validate='one_to_one',indicator=True)
 nmissing=int(joined._merge.ne('both').sum())
 v=joined[joined._merge.eq('both')]
 diff=np.abs(v.score.to_numpy(float)-v.new_score.to_numpy(float))
 result=dict(status='PASS' if nmissing==0 and len(v)>0 and np.max(diff)<=1e-4 else 'FAIL',
  n_old_rows=len(old),n_new_rows=len(now),n_matched=len(v),n_unmatched=nmissing,
  max_abs_difference=float(np.max(diff)) if len(diff) else None,
  tolerance=1e-4,
  note='74G fixed; full project original Hallmark extraction must reproduce before new Reactome comparison')
 write_json(result,out/'01_HALLMARK_SOURCE_REPRODUCTION_AUDIT.json')
 if result['status']=='FAIL':
  write_table(joined.loc[joined._merge.ne('both')].head(100),out/'01_HALLMARK_AUDIT_UNMATCHED_FIRST100.csv')
  raise RuntimeError('Hallmark reproduction failed: '+json.dumps(result))
 log('HALLMARK_REPRODUCTION',result)
 return result

def models_all(meta,matrix,names,args,out):
 res=[];fits_by_name={};start=time.time()
 for j,name in enumerate(names):
  vals=matrix[:,j];ix=np.isfinite(vals)
  if ix.sum()<args.min_units:continue
  d=meta.loc[ix].copy();d['score']=vals[ix].astype(float)
  design=build_design(d,include_neff=not args.skip_effective_size_covariate)
  if design is None:continue
  if design['n_donors']<args.min_donors or design['n_tissues']<args.min_tissues:continue
  f=fit_models(design)
  if f is None:continue
  record={k:v for k,v in f.items() if k!='_fits'}
  record.update(pathway=name,collection=('MatchedNull' if name.startswith('NULL__') else 'Hallmark' if name.startswith('HALLMARK_') else 'Reactome'))
  res.append(record)
  if j%150==0:log('M012 fit',j+1,'/',len(names),'elapsed',int(time.time()-start),'s')
 if not res:raise RuntimeError('M0/M1/M2 produced no viable model. Check min donors/tissues and data.')
 a=pd.DataFrame(res)
 for c in ('Hallmark','Reactome'):
  mask=a.collection.eq(c)
  for p,col in [('p_donor_cluster_M1','BHq_M1_within_collection'),('p_donor_cluster_M2','BHq_M2_within_collection')]:a.loc[mask,col]=bh(a.loc[mask,p])
 for p,q in [('p_donor_cluster_M1','BHq_M1_all_sets'),('p_donor_cluster_M2','BHq_M2_all_sets')]:
  istrue=a.collection.isin(['Hallmark','Reactome']);a.loc[istrue,q]=bh(a.loc[istrue,p])
 a['priority']=a.pathway.isin(COV_TESTS)
 a['M1_q_le_05']=a.BHq_M1_within_collection<=.05
 a['M2_q_le_05']=a.BHq_M2_within_collection<=.05
 a=a.sort_values(['collection','BHq_M2_within_collection','deltaR2_M2_vs_M1'],ascending=[True,True,False])
 write_table(a,out/'02_ALL_HALLMARK_REACTOME_M012.csv.gz')
 write_table(a[a.collection!='MatchedNull'],out/'02_M012_MAIN_RESULTS.csv')
 # Nulls have cohort-varying gene membership: only exploratory comparison.
 controls=[]
 for m in a.loc[a.collection.ne('MatchedNull'),'pathway']:
  cc=a[a.pathway.str.startswith('NULL__'+m+'__',na=False)]
  if cc.empty:continue
  hit=a[a.pathway.eq(m)].iloc[0]
  controls.append(dict(pathway=m,n_matched_control_sets=len(cc),
   empirical_p_M1=(1+int((cc.deltaR2_M1_vs_M0>=hit.deltaR2_M1_vs_M0).sum()))/(1+len(cc)),
   empirical_p_M2=(1+int((cc.deltaR2_M2_vs_M1>=hit.deltaR2_M2_vs_M1).sum()))/(1+len(cc)),
   observed_M1=hit.deltaR2_M1_vs_M0,observed_M2=hit.deltaR2_M2_vs_M1,
   null_median_M1=float(cc.deltaR2_M1_vs_M0.median()),null_median_M2=float(cc.deltaR2_M2_vs_M1.median()),
   caveat='NULL GENE MEMBERSHIP RESELECTED PER COHORT; EXPLORATORY ONLY; MAY UNDERESTIMATE NULL COHERENCE'))
 write_table(pd.DataFrame(controls),out/'02_MATCHED_GENE_SET_EXPLORATORY_NULL.csv')
 eligible=a[a.collection.ne('MatchedNull')].copy()
 selected=set(COV_TESTS)&set(eligible.pathway)
 for field in ['deltaR2_M1_vs_M0','deltaR2_M2_vs_M1']:
  for fam in ['Hallmark','Reactome']:
   z=eligible[eligible.collection.eq(fam)].sort_values(field,ascending=False).head(args.confirm_top)
   selected.update(z.pathway)
 for fam in ['Hallmark','Reactome']:
  for field in ['BHq_M1_within_collection','BHq_M2_within_collection']:
   selected.update(eligible[eligible.collection.eq(fam)].sort_values(field).head(args.confirm_top).pathway)
 # Also confirm targets of matched gene-set controls.
 selected.update([m for m in eligible.pathway if any(t in m for t in ['REACTOME_SIGNALING_BY_INTERLEUKINS'])])
 sel=eligible[eligible.pathway.isin(selected)].copy()
 log('Confirmation tier:',len(sel),'gene sets, wild bootstrap',args.wild_permutations,
  'donor bootstrap',args.bootstraps)
 outrecords=[]
 for k,ref in enumerate(sel.itertuples(index=False),1):
  j=names.index(ref.pathway);vals=matrix[:,j];ix=np.isfinite(vals)
  d=meta.loc[ix].copy();d['score']=vals[ix].astype(float)
  design=build_design(d,include_neff=not args.skip_effective_size_covariate)
  if design is None:continue
  fit=fit_models(design)
  if fit is None:continue
  rng=np.random.default_rng(args.seed + sum(ref.pathway.encode()))
  v=wild_boot(design,fit,args.wild_permutations,rng)
  b=donor_boot(design,args.bootstraps,rng)
  outrecords.append(dict(pathway=ref.pathway,collection=ref.collection,**v,**b))
  if k%10==0:log('CONFIRM',k,'/',len(sel))
 confirmed=pd.DataFrame(outrecords)
 if not confirmed.empty:
  confirmed=confirmed.merge(sel,on=['pathway','collection'],how='left',validate='one_to_one')
  for family in ['Hallmark','Reactome']:
   sub=confirmed.collection.eq(family)
   for p,q in [('wild_p_M1','BHq_confirmatory_wild_M1_WITHIN_SELECTED'),('wild_p_M2','BHq_confirmatory_wild_M2_WITHIN_SELECTED')]:
    confirmed.loc[sub,q]=bh(confirmed.loc[sub,p])
  # NOTE: selected by effects/significance. Wild-bootstrap q within selected is NOT
  # genome-wide, cannot be presented as independent post-selection confirmation.
 write_table(confirmed,out/'03_SELECTED_DONOR_BOOTSTRAP_WILD.csv')
 return a,confirmed

def cell_state_sensitivity(meta,matrix,names,sets,args,out):
 """Nested M0c/M1c/M2c, with available cell-cycle/stress pathway scores
 in M0c. Never use control score identical to outcome; skip highly overlapping
 gene sets. Controls are measured in the same Tabula atlas, observational.
 """
 qc=['HALLMARK_G2M_CHECKPOINT','HALLMARK_E2F_TARGETS','HALLMARK_UNFOLDED_PROTEIN_RESPONSE']
 found=[(m,names.index(m)) for m in qc if m in names]
 if len(found)<2:
  log('No adequate state-proxy sets; state sensitivity skipped')
  write_table(pd.DataFrame(),out/'03_CELL_STATE_SENSITIVITY.csv');return pd.DataFrame()
 results=[]
 for j,m in enumerate(names):
  if m.startswith('NULL__') or m not in sets:continue
  vv=matrix[:,j];take=np.isfinite(vv)
  available=[]
  for cn,cj in found:
   if cn==m:continue
   overlap=len(set(sets[m])&set(sets[cn]))
   if overlap/max(1,min(len(sets[m]),len(sets[cn])))>.20:continue
   take &= np.isfinite(matrix[:,cj]);available.append((cn,cj))
  if len(available)<1 or take.sum()<args.min_units:continue
  z=meta.loc[take].copy();z['score']=vv[take].astype(float)
  for i,(cn,cj) in enumerate(available):z['state_'+str(i)]=matrix[take,cj]
  design=build_design(z,include_neff=not args.skip_effective_size_covariate)
  if design is None or design['n_donors']<args.min_donors or design['n_tissues']<args.min_tissues:continue
  fit=fit_models(design)
  if fit is None:continue
  results.append(dict(pathway=m,collection='Hallmark' if m.startswith('HALLMARK_') else 'Reactome',
   state_proxy_sets=';'.join(c for c,_ in available),n_state_controls=len(available),
   state_adjusted_deltaR2_M1=fit['deltaR2_M1_vs_M0'],
   state_adjusted_deltaR2_M2=fit['deltaR2_M2_vs_M1'],
   state_adjusted_p_M1=fit['p_donor_cluster_M1'],state_adjusted_p_M2=fit['p_donor_cluster_M2'],
   n_donors=fit['n_donors'],n_tissues=fit['n_tissues'],n_units=fit['n_units'],
   warning='M0c cell-cycle/stress proxies from same dataset; not independent cell-state measurements'))
 if not results:return pd.DataFrame()
 d=pd.DataFrame(results)
 for family in ['Hallmark','Reactome']:
  k=d.collection.eq(family)
  d.loc[k,'state_adjusted_BHq_M1']=bh(d.loc[k,'state_adjusted_p_M1'])
  d.loc[k,'state_adjusted_BHq_M2']=bh(d.loc[k,'state_adjusted_p_M2'])
 write_table(d,out/'03_CELL_STATE_SENSITIVITY.csv')
 return d

def withheld_tissue_shape_cv(meta,matrix,names,selected,args,out):
 """Within-block centered module-shape prediction under held tissue/donor split.
 By construction test-module block mean is subtracted. It is *not* absolute
 prediction or valid performance assessment for raw pathway expression level.
 """
 rows=[]
 for i,name in enumerate(sorted(selected)):
  if name not in names:continue
  j=names.index(name);v=matrix[:,j];use=np.isfinite(v)
  z=meta.loc[use].copy();z['score']=v[use].astype(float)
  if z.tissue.nunique()<6:continue
  for held in sorted(z.tissue.unique()):
   held_donors=set(z.loc[z.tissue.eq(held),'donor'].astype(str))
   train=z[~z.tissue.eq(held) & ~z.donor.astype(str).isin(held_donors)].copy()
   test=z[z.tissue.eq(held)].copy()
   if len(train)<100 or len(test)<10:continue
   a=build_design(train,include_neff=not args.skip_effective_size_covariate)
   b=build_design(test,include_neff=not args.skip_effective_size_covariate)
   if a is None or b is None or a['n_donors']<8 or a['n_lineages']<4 or b['n_lineages']<2:continue
   errs=[]
   for mm in ['X0','X1','X2']:
    X=a[mm];Q=b[mm]
    if X.shape[1]!=Q.shape[1]:errs.append(np.nan);continue
    coef=np.linalg.lstsq(X*np.sqrt(a['w'])[:,None],a['y']*np.sqrt(a['w']),rcond=None)[0]
    pred=Q@coef
    errs.append(float(np.sqrt(np.average((b['y']-pred)**2,weights=b['w']))))
   if not np.isfinite(errs).all():continue
   rows.append(dict(pathway=name,collection='Hallmark' if name.startswith('HALLMARK_') else 'Reactome',
    held_tissue=held,n_train_donors=a['n_donors'],n_test_donors=b['n_donors'],n_test_units=len(b['y']),
    RMSE_M0=errs[0],RMSE_M1=errs[1],RMSE_M2=errs[2],
    delta_RMSE_M1=errs[0]-errs[1],delta_RMSE_M2=errs[1]-errs[2],
    validation='TISSUE_AND_DONOR_DISJOINT__TEST_BLOCK_MEAN_REMOVED'))
  if (i+1)%15==0:log('CV',i+1,'/',len(selected))
 cv=pd.DataFrame(rows);write_table(cv,out/'04_TISSUE_DONOR_DISJOINT_WITHIN_BLOCK_CV.csv')
 if cv.empty:return cv,pd.DataFrame()
 sm=cv.groupby(['pathway','collection'],as_index=False).agg(
  n_held_tissues=('held_tissue','nunique'),mean_deltaRMSE_M1=('delta_RMSE_M1','mean'),
  mean_deltaRMSE_M2=('delta_RMSE_M2','mean'),fraction_tissues_M1_better=('delta_RMSE_M1',lambda a:float(np.mean(a>0))),
  fraction_tissues_M2_better=('delta_RMSE_M2',lambda a:float(np.mean(a>0))))
 write_table(sm,out/'04_CV_SUMMARY.csv')
 return cv,sm

def lineage_harmonic_profiles(meta,matrix,names,confirmed,args,out):
 # Keep only candidate gene sets, report whether lineage slopes are actually
 # identifiable. Interpret phase only if robust nonzero module amplitude.
 if confirmed.empty:return pd.DataFrame()
 rows=[]
 for name in confirmed.pathway:
  v=matrix[:,names.index(name)];ok=np.isfinite(v)
  z=meta.loc[ok].copy();z['score']=v[ok].astype(float)
  for lineage in LINEAGES:
   d=z[z.lineage.eq(lineage)].copy()
   if d.donor.nunique()<args.min_donors//2 or len(d)<32:continue
   dat=build_design(d,include_neff=not args.skip_effective_size_covariate)
   if dat is None:continue
   # Only M1 makes sense for single lineage. Donor-balanced group-FE shape.
   X=dat['X1'];beta,_,_,sse=weighted_lstsq(X,dat['y'],dat['w'])
   bc,bs=beta[-2:];mag=float(np.hypot(bc,bs));ph=float((np.arctan2(bs,bc)%(2*np.pi))/W)
   rows.append(dict(pathway=name,lineage=lineage,n_donors=dat['n_donors'],n_tissues=dat['n_tissues'],
    amplitude=mag,descriptive_peak_phase_h=ph,
    donor_cluster_p=cluster_wald(X,dat['y'],dat['w'],dat['donor_codes'],X.shape[1]-2)))
 ans=pd.DataFrame(rows)
 if not ans.empty:ans['BHq_per_pathway_5_lineages']=ans.groupby('pathway').donor_cluster_p.transform(bh)
 write_table(ans,out/'05_LINEAGE_PHASE_AND_AMPLITUDE_DESCRIPTIVE.csv')
 return ans

# ---- Publication-oriented independent panels, not composite figures ----

def savefig(fig,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg']:
  fig.savefig(str(path)+'.'+ext,dpi=350 if ext=='png' else None,bbox_inches='tight',pad_inches=.18)
 plt.close(fig)

def pname(s):
 return s.replace('HALLMARK_','').replace('REACTOME_','').replace('_',' ').title()

def plot_figures(main,confirmed,cvs,profile,null,out,max_show):
 figdir=out/'FIGURES';figdir.mkdir(parents=True,exist_ok=True)
 true=main[main.collection.isin(['Hallmark','Reactome'])].copy()
 for fam in ['Hallmark','Reactome']:
  d=true[true.collection.eq(fam)]
  if d.empty:continue
  fig,ax=plt.subplots(figsize=(6.7,5.0))
  a=d.deltaR2_M1_vs_M0.to_numpy(float);b=d.deltaR2_M2_vs_M1.to_numpy(float)
  c=np.where(d.BHq_M2_within_collection<=.05,'#B54D59','#A4AAB0')
  ax.scatter(a,b,s=19,c=c,alpha=.73,edgecolors='none')
  ax.axhline(0,color='0.7',lw=.7);ax.axvline(0,color='0.7',lw=.7)
  ax.set_xlabel('M1 − M0: added within-block R² (common inferred phase)')
  ax.set_ylabel('M2 − M1: added within-block R² (lineage × phase)')
  ax.set_title(f'{fam} | Whole collection: model increments')
  savefig(fig,figdir/f'A_{fam}_M012_effect_plane')
  for metric,p in [('deltaR2_M1_vs_M0','B'),('deltaR2_M2_vs_M1','C')]:
   top=d.sort_values(metric,ascending=False).head(max_show).iloc[::-1]
   fig,ax=plt.subplots(figsize=(10,max(4,.34*len(top)+1)))
   cols=['#59A14F' if q<=.05 else '#9AA0A6' for q in top.BHq_M1_within_collection if metric=='deltaR2_M1_vs_M0'] if metric=='deltaR2_M1_vs_M0' else ['#E15759' if q<=.05 else '#9AA0A6' for q in top.BHq_M2_within_collection]
   ax.barh(np.arange(len(top)),top[metric],color=cols,height=.78)
   ax.set_yticks(np.arange(len(top)));ax.set_yticklabels([pname(s)[:64] for s in top.pathway],fontsize=7.8)
   ax.set_xlabel('Incremental within-donor-block explained fraction')
   ax.set_title(f'{fam} | '+('Common inferred-phase component' if metric=='deltaR2_M1_vs_M0' else 'Lineage-specific phase interaction'))
   savefig(fig,figdir/f'{p}_{fam}_top_{max_show}')
 if not confirmed.empty:
  z=confirmed.sort_values('deltaR2_M2_vs_M1',ascending=False).head(max_show).iloc[::-1]
  fig,ax=plt.subplots(figsize=(10,max(4,len(z)*.34+1)))
  xx=z.deltaR2_M2_vs_M1.to_numpy(float);low=z.ci_low_M2.to_numpy(float);hi=z.ci_high_M2.to_numpy(float)
  ax.errorbar(xx,np.arange(len(z)),xerr=[np.maximum(xx-low,0),np.maximum(hi-xx,0)],fmt='o',c='#4E79A7',ecolor='#B5C5D7',capsize=2,markersize=4)
  ax.axvline(0,color='.5',lw=.8)
  ax.set_yticks(np.arange(len(z)));ax.set_yticklabels([pname(p)[:67] for p in z.pathway],fontsize=7.5)
  ax.set_xlabel('M2−M1 within-block ΔR² (95% donor bootstrap, selected sets)')
  ax.set_title('D | Donor-level interaction uncertainty')
  savefig(fig,figdir/'D_selected_M2_donor_bootstrap')
 if not cvs.empty:
  d=cvs[cvs.n_held_tissues>=4].sort_values('mean_deltaRMSE_M2').tail(max_show)
  if len(d):
   fig,ax=plt.subplots(figsize=(9,max(4,len(d)*.34+1)))
   ax.barh(np.arange(len(d)),d.mean_deltaRMSE_M2,color=['#59A14F' if x>0 else '#E15759' for x in d.mean_deltaRMSE_M2])
   ax.axvline(0,color='.5',lw=.8)
   ax.set_yticks(np.arange(len(d)));ax.set_yticklabels([pname(p)[:55] for p in d.pathway],fontsize=7.5)
   ax.set_xlabel('Held-tissue, donor-disjoint ΔRMSE(M1 − M2), within-block centered output')
   ax.set_title('E | Cross-tissue transfer of interaction coefficients')
   savefig(fig,figdir/'E_tissue_donor_disjoint_CV')
 if not profile.empty:
  tops=confirmed[confirmed.collection.eq('Hallmark')].sort_values('deltaR2_M2_vs_M1',ascending=False).head(5).pathway
  for k,p in enumerate(tops):
   d=profile[profile.pathway.eq(p)]
   if d.empty:continue
   fig,ax=plt.subplots(figsize=(7.8,4.5))
   x=np.linspace(0,24,200)
   for row in d.itertuples(index=False):
    # This curve represents only phase-associated variation. Amplitude/phase
    # summary is descriptive and not an observed true-time trajectory.
    y=row.amplitude*np.cos(W*(x-row.descriptive_peak_phase_h))
    ax.plot(x,y,lw=2,label=row.lineage,c=L_COL.get(row.lineage,'gray'))
   ax.axhline(0,color='.7',lw=.7);ax.set_xlim(0,24);ax.set_xticks([0,4,8,12,16,20,24])
   ax.set_xlabel('B0 inferred phase (h; NOT sampled time)');ax.set_ylabel('Within-block harmonic fit')
   ax.set_title(f'F{k+1} | {pname(p)[:58]}');ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=8)
   savefig(fig,figdir/f'F_{k+1:02d}_{p[:65]}')
 if not null.empty:
  zz=null.sort_values('empirical_p_M2').head(max_show).iloc[::-1]
  fig,ax=plt.subplots(figsize=(9,max(4,.37*len(zz)+1)))
  ax.scatter(zz.observed_M2,np.arange(len(zz)),label='Observed pathway',c='#4E79A7',s=30)
  ax.scatter(zz.null_median_M2,np.arange(len(zz)),label='Matched-gene controls (cohort-varying)',c='#E15759',s=24,marker='x')
  ax.set_yticks(np.arange(len(zz)));ax.set_yticklabels([pname(p)[:55] for p in zz.pathway],fontsize=8)
  ax.set_xlabel('M2−M1 ΔR²');ax.set_title('G | Exploratory, cohort-specific matched gene-set controls')
  ax.legend(loc='lower right',fontsize=7);savefig(fig,figdir/'G_EXPLORATORY_matched_controls')

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 ap.add_argument('--project-root',required=True)
 ap.add_argument('--hallmark-gmt',default=None);ap.add_argument('--reactome-gmt',default=None)
 ap.add_argument('--reactome-regex',default='',help='Optional pre-specified Reactome restriction; empty runs all sets')
 ap.add_argument('--stage',choices=['all','extract','models'],default='all')
 ap.add_argument('--resume',action='store_true');ap.add_argument('--pilot',type=int,default=0,help='DEV only: first N cohorts; do not call this a full atlas result')
 ap.add_argument('--allow-cohort-failures',action='store_true')
 ap.add_argument('--min-pathway-genes',type=int,default=10);ap.add_argument('--max-pathway-genes',type=int,default=300)
 ap.add_argument('--min-coverage',type=float,default=.45);ap.add_argument('--max-gene-union',type=int,default=25000)
 ap.add_argument('--min_R',type=float,default=.3);ap.add_argument('--min_detect',type=int,default=2)
 ap.add_argument('--min_bin_weight',type=float,default=8);ap.add_argument('--min_bin_effective',type=float,default=8)
 ap.add_argument('--h5ad-block',type=int,default=192);ap.add_argument('--min-cells',type=int,default=50)
 ap.add_argument('--extra-background-genes',type=int,default=2500)
 ap.add_argument('--matched-nulls',type=int,default=15,help='Per selected target per cohort, explorative controls')
 ap.add_argument('--reactome-null-regex-max',type=int,default=2,help='Per prespecified Reactome keyword class')
 ap.add_argument('--min-units',type=int,default=80);ap.add_argument('--min-donors',type=int,default=10)
 ap.add_argument('--min-tissues',type=int,default=4);ap.add_argument('--skip-effective-size-covariate',action='store_true')
 ap.add_argument('--confirm-top',type=int,default=15);ap.add_argument('--wild-permutations',type=int,default=499)
 ap.add_argument('--bootstraps',type=int,default=250);ap.add_argument('--max-plot-pathways',type=int,default=22)
 ap.add_argument('--seed',type=int,default=20261012)
 ap.add_argument('--no-figures',action='store_true')
 ap.add_argument('--skip-cell-state-sensitivity',action='store_true')
 ap.add_argument('--skip-hallmark-audit',action='store_true',help='Disable strict old/new Hallmark reproduction check (not recommended)')
 args=ap.parse_args(argv)
 root=Path(args.project_root).expanduser().resolve()
 out=root/'TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1';out.mkdir(parents=True,exist_ok=True)
 status=dict(state='RUNNING',start=datetime.now().isoformat(),project='TABULA ONLY',
  phase='FROZEN COHORT-SPECIFIC B0 INFERRED; NOT TRUE TIME',pilot=args.pilot,
  confounding='Association not causal. Clock17-derived phase can capture cell identity/states.',
  tests='M0 donor×tissue×assay×lineage FE; M1 shared harmonic; M2 lineage×harmonic',
  secondary_null='Cohort-specific gene-matched sets (exploratory; gene identity differs by cohort)')
 write_json(status,out/'00_STATUS.json')
 h=resolve_gmt(root,args.hallmark_gmt,'Hallmark');r=resolve_gmt(root,args.reactome_gmt,'Reactome')
 source=mod_import(root/'tabula_final_figure_clock_outputs_v1.py','gmt_source')
 allH,qa=gmt_read(h,source,args.min_pathway_genes,args.max_pathway_genes)
 allR,qb=gmt_read(r,source,args.min_pathway_genes,args.max_pathway_genes)
 if args.reactome_regex:
  rexp=re.compile(args.reactome_regex,re.I);allR={k:v for k,v in allR.items() if rexp.search(k)}
 if len(allH)<40 or len(allR)<10:raise ValueError(f'Expected full pathway libraries, got Hallmark {len(allH)}, Reactome {len(allR)}')
 dup=set(allH)&set(allR)
 if dup:raise ValueError('Pathway names overlap: '+repr(list(dup)[:8]))
 sets={**allH,**allR};priority=select_priority(sets,args.reactome_null_regex_max)
 union=set().union(*map(set,sets.values()))
 write_table(pd.concat([qa,qb],ignore_index=True),out/'00_GMT_PATHWAY_LIBRARY_AUDIT.csv')
 signature=hashlib.sha256(json.dumps(dict(h=hash_file(h),r=hash_file(r),regex=args.reactome_regex,
  min_R=args.min_R,min_detect=args.min_detect,min_pathway_genes=args.min_pathway_genes,max_pathway_genes=args.max_pathway_genes,min_bin_weight=args.min_bin_weight,
  min_bin_effective=args.min_bin_effective,min_coverage=args.min_coverage,
  extra=args.extra_background_genes,nulls=args.matched_nulls,seed=args.seed,version='v1.2',priority=priority),sort_keys=True).encode()).hexdigest()
 status.update(hallmark_gmt=str(h),reactome_gmt=str(r),n_hallmark=len(allH),n_reactome=len(allR),
  n_union_genes=len(union),n_null_target_pathways=len(priority),signature=signature)
 write_json(status,out/'00_STATUS.json')
 if args.stage in ('all','extract'):
  if len(union)>args.max_gene_union:raise RuntimeError(f'Gene union {len(union)} exceeds max {args.max_gene_union}')
  meta,arr,names=extract_atlas(root,sets,priority,args,out,signature)
  status['extraction']='COMPLETE';status['n_aggregated_units']=len(meta)
  write_json(status,out/'00_STATUS.json')
  if not args.skip_hallmark_audit and not args.pilot:
   status['hallmark_reproduction']=hallmark_reproduction_audit(meta,arr,names,root,out)
   write_json(status,out/'00_STATUS.json')
  write_json({'signature':signature,'created':datetime.now().isoformat(),'cohorts':'PILOT' if args.pilot else 'FULL'},out/'01_EXTRACTION_SIGNATURE.json')
  if args.stage=='extract':status['state']='EXTRACTION_COMPLETE';write_json(status,out/'00_STATUS.json');return
 else:
  mfile=out/'01_ATLAS_UNIT_METADATA.csv.gz';afile=out/'01_ALL_PATHWAY_SCORES_WIDE.npz'
  if not mfile.exists() or not afile.exists():raise FileNotFoundError('Run --stage extract first')
  saved=out/'01_EXTRACTION_SIGNATURE.json'
  if not saved.exists() or read_json(saved).get('signature')!=signature:
   raise RuntimeError('Extraction signature does not match GMT/configuration; rerun --stage extract.')
  meta=pd.read_csv(mfile);d=np.load(afile,allow_pickle=False);arr=d['scores'];names=d['pathway'].astype(str).tolist()
  if len(meta)!=len(arr):raise ValueError('Saved module data index mismatch')
 if args.pilot:log('WARNING: PILOT analysis only; do not use pilot p-values for publication')
 mainres,confirmed=models_all(meta,arr,names,args,out)
 state_sens=cell_state_sensitivity(meta,arr,names,sets,args,out) if not args.skip_cell_state_sensitivity else pd.DataFrame()
 status['models']='COMPLETE';status['n_tested_models']=len(mainres)
 write_json(status,out/'00_STATUS.json')
 selected=confirmed.pathway.tolist() if not confirmed.empty else []
 # Validate all Hallmark plus selected Reactome, never only post-selected effect peaks.
 selected=set(selected)|set(x for x in names if x.startswith('HALLMARK_'))
 cv,cvsum=withheld_tissue_shape_cv(meta,arr,names,selected,args,out)
 profiles=lineage_harmonic_profiles(meta,arr,names,confirmed,args,out)
 null_path=out/'02_MATCHED_GENE_SET_EXPLORATORY_NULL.csv';null=pd.read_csv(null_path) if null_path.exists() and null_path.stat().st_size>10 else pd.DataFrame()
 if not args.no_figures:plot_figures(mainres,confirmed,cvsum,profiles,null,out,args.max_plot_pathways)
 hall=mainres[mainres.collection.eq('Hallmark')];react=mainres[mainres.collection.eq('Reactome')]
 lines=['# Tabula Figure 6 | lineage identity versus B0 clock-state association','',
  '**CAUTION: phases are cohort-specific inferred B0, NOT observed circadian time.**','',
  '## Same models for human Hallmark and Reactome','',
  'M0 = donor×tissue×assay×lineage fixed baseline, within-block effective-size proxy;',
  'M1 = M0 + shared cos/sin of inferred phase;',
  'M2 = M1 + lineage×cos/sin interactions (8 additional coefficients).','',
  'All responses are pathway means of cohort-standardized log1p(CP10k) non-CLOCK17 gene expression.',
  'Every subject is aggregated to donor×lineage×tissue×assay×posterior-phase-bin; donor is cluster for robust tests.',
  'ΔR² measures phase-related variation *within* donor×lineage×tissue×assay blocks, not between-lineage mean differences.','',
  '## Preliminary results (exploratory, whole-library BH on donor-cluster Wald)','',
  f'Hallmark tested: {len(hall)}; M1 BH≤.05: {int((hall.BHq_M1_within_collection<=.05).sum())}; M2 BH≤.05: {int((hall.BHq_M2_within_collection<=.05).sum())}',
  f'Reactome tested: {len(react)}; M1 BH≤.05: {int((react.BHq_M1_within_collection<=.05).sum())}; M2 BH≤.05: {int((react.BHq_M2_within_collection<=.05).sum())}',
  f'Selected confirmation (post-selection, NOT independent): {len(confirmed)} modules.',
  f'Held tissue and donor disjoint CV: {len(cv)} folds (within-block centered outcomes; NOT absolute scores).','',
  '## Limitations / interpretation safeguards','',
  '1. Frozen B0 was estimated from Clock17; this is neither a time label nor causal clock intervention.',
  '2. Donor-cluster Wald can be imperfect with few donors; donor wild and bootstrap on selected sets are sensitivity checks.',
  '3. Selected gene sets were chosen in the same data; post-selection p values require caution.',
  '4. Matched random gene identities change across cohorts: only exploratory; a *fixed* genome-wide null panel needs separate work.',
  '5. Covariates for cell cycle, stress, cytokine activity are NOT automatically adjusted unless independently measured;',
  '   do not claim adjustment for such confounding merely because M0 includes lineage.',
  '6. Reactome gene sets overlap extensively. All q values account for multiplicity, not overlapping-set confounding.',
  '7. CV test block means are removed: this measures transfer of within-block phase *shape*, not prediction of unseen absolute expression.',
  '8. Significant interaction terms can reflect amplitude AND phase differences; peak-shift significance needs separate amplitude-stable tests.',
  '9. Source extraction fails closed on original count-total vs frozen B0 checks and missing cohorts by default.','',
  '## Key outputs','',
  '02_M012_MAIN_RESULTS.csv : full Hallmark + Reactome M0,M1,M2 fits','03_SELECTED_DONOR_BOOTSTRAP_WILD.csv : donor resampling and wild bootstrap',
  '04_TISSUE_DONOR_DISJOINT_WITHIN_BLOCK_CV.csv : held-out predictions',
  '02_MATCHED_GENE_SET_EXPLORATORY_NULL.csv : matched nuisance controls (exploratory)',
  '03_CELL_STATE_SENSITIVITY.csv : auxiliary nested models with overlapping-gene-screened cell-cycle/stress score proxies',
  '05_LINEAGE_PHASE_AND_AMPLITUDE_DESCRIPTIVE.csv : descriptive inferred-phase peaks',
  'FIGURES/: individual publication-style PNG/PDF/SVG panels','']
 (out/'INTERPRETATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 status.update(state='COMPLETE',finish=datetime.now().isoformat(),
  n_state_sensitivity_sets=len(state_sens),n_matched_gene_controls=args.matched_nulls,n_selected_confirmed=len(confirmed),n_CV_folds=len(cv),
  n_panels=len(list((out/'FIGURES').glob('*.png'))) if not args.no_figures else 0)
 write_json(status,out/'00_STATUS.json')
 log('COMPLETE',out,'Hallmark',len(hall),'Reactome',len(react),'CV',len(cv))

if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();sys.exit(1)
