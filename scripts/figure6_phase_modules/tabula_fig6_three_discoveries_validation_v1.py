#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabula-only Figure 6: validate three candidate discoveries, with independent panels.

Requires OUTPUTS from completed tabula_final_figure_clock_outputs_v1.py.
Does NOT rerun B0, read GSE292219, or mutate previous files.
All phases are INFERRED B0 (not observed circadian times). All TF links observational.

Full dependencies: numpy, pandas, scipy, matplotlib, scikit-learn (optional for plots: no).
Usage:
 python -u tabula_fig6_three_discoveries_validation_v1.py --project-root "$PWD" --permutations 1999 --bootstraps 1000
No figures test: append --no-figures.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

LINEAGES=['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
ARMS=['EBOX_POS','EBOX_NEG','RORE_ACT','RORE_REP','DBOX_ACT','DBOX_REP']
PALETTE={'B cell':'#4E79A7','T cell':'#E15759','Macrophage/Monocyte':'#59A14F',
         'Fibroblast':'#F28E2B','Endothelial':'#9467BD'}
W=2*np.pi/24
FAMILY={
 'HALLMARK_MYC_TARGETS_V1':'Shared candidate','HALLMARK_MTORC1_SIGNALING':'Shared candidate',
 'HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY':'Shared candidate',
 'HALLMARK_UNFOLDED_PROTEIN_RESPONSE':'Shared candidate',
 'HALLMARK_OXIDATIVE_PHOSPHORYLATION':'Shared candidate',
 'HALLMARK_DNA_REPAIR':'Shared candidate',
 'HALLMARK_TNFA_SIGNALING_VIA_NFKB':'Divergent candidate',
 'HALLMARK_INTERFERON_GAMMA_RESPONSE':'Divergent candidate',
 'HALLMARK_INFLAMMATORY_RESPONSE':'Divergent candidate',
 'HALLMARK_IL6_JAK_STAT3_SIGNALING':'Divergent candidate'}
CANDIDATE_ASSOC=[
 ('Fibroblast','DBOX_REP','HALLMARK_TNFA_SIGNALING_VIA_NFKB'),
 ('Fibroblast','DBOX_REP','HALLMARK_IL6_JAK_STAT3_SIGNALING'),
 ('Fibroblast','DBOX_REP','HALLMARK_INTERFERON_GAMMA_RESPONSE'),
 ('Fibroblast','DBOX_REP','HALLMARK_HYPOXIA'),
 ('Macrophage/Monocyte','DBOX_REP','HALLMARK_TNFA_SIGNALING_VIA_NFKB'),
 ('B cell','RORE_ACT','HALLMARK_UNFOLDED_PROTEIN_RESPONSE'),
 ('B cell','RORE_ACT','HALLMARK_GLYCOLYSIS')]
plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'axes.spines.right':False,
 'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.dpi':140})

def label(p):return str(p).replace('HALLMARK_','').replace('_',' ').replace('SIGNALING VIA ','').title()
def outcsv(d,p):
 p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False)
def figsave(fig,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 for e in ['png','pdf','svg']:fig.savefig(str(path)+'.'+e,dpi=300 if e=='png' else None,bbox_inches='tight',pad_inches=.13)
 plt.close(fig)
def bh(p):
 p=np.asarray(p,float);ans=np.full(len(p),np.nan);idx=np.flatnonzero(np.isfinite(p))
 if len(idx):
  s=idx[np.argsort(p[idx])];z=p[s]*len(s)/np.arange(1,len(s)+1)
  ans[s]=np.minimum(1,np.minimum.accumulate(z[::-1])[::-1])
 return ans

def phase(z):return float((np.angle(z)%(2*np.pi))/W) if np.isfinite(z.real+z.imag) and abs(z)>1e-14 else np.nan
def circdiff(a,b):return ((a-b+12)%24)-12

def harmonic_fit(t,y):
 t=np.asarray(t,float);y=np.asarray(y,float)
 okay=np.isfinite(t)&np.isfinite(y)
 t=t[okay];y=y[okay]
 if len(y)<4:return (np.nan,np.nan,np.nan)
 X=np.column_stack([np.ones(len(t)),np.cos(W*t),np.sin(W*t)])
 if np.linalg.matrix_rank(X)<3:return (np.nan,np.nan,np.nan)
 b=np.linalg.lstsq(X,y,rcond=None)[0];return (float(b[1]),float(b[2]),float(b[0]))

def donor_complex(block_df):
 # One donor has equal weight irrespective of how many tissues or assays contribute.
 d=block_df.copy();d['z']=d['bc'].to_numpy(float)+1j*d['bs'].to_numpy(float)
 return d.groupby(['pathway','lineage','donor'],as_index=False).agg(
  bc=('bc','mean'),bs=('bs','mean'),n_tissues=('tissue','nunique'),n_blocks=('tissue','size'))

def block_curves(mod):
 rows=[]
 for (m,l,ti,a,d),s in mod.groupby(['pathway','lineage','tissue','assay','donor'],sort=False):
  s=s.sort_values('phase_center_h')
  if s.phase_bin.nunique()<4:continue
  bc,bs,base=harmonic_fit(s.phase_center_h,s.score)
  if not np.isfinite(bc+bs):continue
  rows.append(dict(pathway=m,lineage=l,tissue=ti,assay=a,donor=str(d),
   n_bins=int(s.phase_bin.nunique()),bc=bc,bs=bs,amp=float(np.hypot(bc,bs)),phase_h=phase(complex(bc,bs))))
 return pd.DataFrame(rows)

def phase_stats(blocks,boot,rng):
 donors=donor_complex(blocks)
 records=[];boots={}
 for mod,d in donors.groupby('pathway',sort=False):
  cohorts=d.pivot_table(index='donor',columns='lineage',values=['bc','bs'],aggfunc='first')
  n=len(cohorts);cc=np.full((n,len(LINEAGES)),np.nan+1j*np.nan)
  for j,l in enumerate(LINEAGES):
   if ('bc',l) in cohorts and ('bs',l) in cohorts:
    cc[:,j]=cohorts[('bc',l)].to_numpy(float)+1j*cohorts[('bs',l)].to_numpy(float)
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   avg=np.nanmean(cc,axis=0)
  ok=np.isfinite(avg.real)&(np.abs(avg)>1e-8)
  R=np.abs(np.nanmean(avg[ok]/np.abs(avg[ok]))) if ok.sum()>=3 else np.nan
  out={'pathway':mod,'family':FAMILY.get(mod,'Other Hallmark'),'n_lineages':int(ok.sum()),
   'n_unique_donors':int(n),'phase_concentration_R':float(R)}
  for j,l in enumerate(LINEAGES):
   out[f'phase_{l}']=phase(avg[j]) if ok[j] else np.nan
   out[f'amp_{l}']=float(abs(avg[j])) if ok[j] else np.nan
   out[f'n_donors_{l}']=int(np.isfinite(cc[:,j].real).sum())
  vals=[]
  for k in range(boot):
   ix=rng.integers(0,n,n)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning)
    v=np.nanmean(cc[ix],axis=0)
   good=np.isfinite(v.real)&(abs(v)>1e-8)
   vals.append(float(abs(np.nanmean(v[good]/abs(v[good])))) if good.sum()>=3 else np.nan)
  vals=np.array(vals,float);out['R_boot_ci_low']=float(np.nanquantile(vals,.025)) if np.isfinite(vals).sum()>25 else np.nan
  out['R_boot_ci_high']=float(np.nanquantile(vals,.975)) if np.isfinite(vals).sum()>25 else np.nan
  records.append(out);boots[mod]=vals
 return pd.DataFrame(records),donors,boots

def family_controls(R, rng):
 z=R.set_index('pathway')
 sh=[p for p,k in FAMILY.items() if k=='Shared candidate' and p in z.index]
 di=[p for p,k in FAMILY.items() if k=='Divergent candidate' and p in z.index]
 if len(sh)!=6 or len(di)!=4:return pd.DataFrame([dict(status='CANDIDATE_SET_INCOMPLETE')])
 ten=sh+di;values=z.loc[ten,'phase_concentration_R'].to_numpy(float)
 observed=float(np.nanmean(values[:6])-np.nanmean(values[6:]))
 exact=[]
 for co in itertools.combinations(range(10),6):
  a=np.array(list(co));b=np.array([i for i in range(10) if i not in co]);exact.append(float(np.nanmean(values[a])-np.nanmean(values[b])))
 null=np.array(exact)
 others=z.loc[[p for p in z.index if p not in ten],'phase_concentration_R'].dropna().to_numpy(float)
 # 5000 random subsets of all 50, module-level comparator (NOT gene-matched controls)
 null2=[]
 allv=z.phase_concentration_R.dropna().to_numpy(float)
 if len(allv)>=10:
  for k in range(5000):
   ix=rng.choice(len(allv),size=10,replace=False)
   null2.append(float(allv[ix[:6]].mean()-allv[ix[6:]].mean()))
 return pd.DataFrame([dict(status='EXPLORATORY_POST_HOC_GROUPS',n_shared=6,n_divergent=4,
  observed_shared_minus_divergent_R=observed,
  exact_210_label_permutation_p=float(np.mean(null>=observed-1e-12)),
  all_hallmark_resampling_p=float((1+sum(np.array(null2)>=observed))/(len(null2)+1)),
  other_hallmarks_median_R=float(np.median(others)) if len(others) else np.nan,
  note='Families were selected using previous observed results, so p-values are descriptive and selection-biased; no matched random GENE sets.')])

def tissue_sensitivity(blocks,focus):
 rows=[]
 for mod in focus:
  d=blocks[blocks.pathway.eq(mod)]
  if d.empty:continue
  for tissue in sorted(d.tissue.unique()):
   x=d[d.tissue.ne(tissue)]
   q=donor_complex(x)
   if q.empty:continue
   mean=q.groupby('lineage').agg(bc=('bc','mean'),bs=('bs','mean'),n_donors=('donor','nunique'))
   vv=np.array([complex(mean.loc[l,'bc'],mean.loc[l,'bs']) for l in LINEAGES if l in mean.index and mean.loc[l,'n_donors']>=4])
   if len(vv)<3:continue
   R=abs(np.mean(vv/abs(vv)))
   rows.append(dict(pathway=mod,held_tissue=tissue,R=float(R),n_lineages=len(vv),n_blocks=len(x)))
 return pd.DataFrame(rows)

def matched_lineage_contrasts(blocks,perms,boot,rng):
 # Pair lineage within same donor+tissue+assay, then aggregate multiple blocks to one donor.
 rows=[]
 for mod,d in blocks.groupby('pathway',sort=False):
  p=d.pivot_table(index=['donor','tissue','assay'],columns='lineage',values=['bc','bs'],aggfunc='first')
  for A,B in itertools.combinations(LINEAGES,2):
   if ('bc',A) not in p or ('bc',B) not in p or ('bs',A) not in p or ('bs',B) not in p:continue
   x=p[[('bc',A),('bs',A),('bc',B),('bs',B)]].dropna()
   if x.empty:continue
   delta=(x[('bc',A)].to_numpy()+1j*x[('bs',A)].to_numpy())-(x[('bc',B)].to_numpy()+1j*x[('bs',B)].to_numpy())
   dd=pd.DataFrame({'donor':x.index.get_level_values('donor'),'real':delta.real,'imag':delta.imag})
   q=dd.groupby('donor')[['real','imag']].mean()
   vals=q.real.to_numpy()+1j*q.imag.to_numpy();n=len(vals)
   if n<6:continue
   obs=abs(vals.mean()); signs=rng.choice([-1.,1.],size=(perms,n))
   null=np.abs((signs@vals)/n);pv=float((1+np.sum(null>=obs-1e-12))/(perms+1))
   ix=rng.integers(0,n,size=(boot,n));z=np.abs(vals[ix].mean(axis=1))
   # For interpretation, pair average phase separately, not phase of (A-B) interpreted as a phase shift.
   vA=p[[('bc',A),('bs',A)]].dropna();vB=p[[('bc',B),('bs',B)]].dropna()
   phA=phase(complex(vA[('bc',A)].mean(),vA[('bs',A)].mean()));phB=phase(complex(vB[('bc',B)].mean(),vB[('bs',B)].mean()))
   rows.append(dict(pathway=mod,lineage_A=A,lineage_B=B,n_matched_donors=n,
    n_matched_blocks=len(dd),interaction_vector_magnitude=obs,ci_low=float(np.quantile(z,.025)),
    ci_high=float(np.quantile(z,.975)),donor_signflip_p=pv,
    phase_A_h=phA,phase_B_h=phB,phase_A_minus_B_h=circdiff(phA,phB),
    note='Tests different phase-resolved harmonic vectors (amplitude and/or phase), NOT a pure phase-shift null.'))
 res=pd.DataFrame(rows)
 if not res.empty:res['BH_q_all_pair_pathway']=bh(res.donor_signflip_p)
 return res

def make_rel(wide):
 d=wide.copy();missing=[a for a in ARMS if a not in d]
 if missing:raise ValueError('Missing six arms in matched wide: '+repr(missing))
 d['clock_mean']=d[ARMS].mean(axis=1)
 for a in ARMS:d['relative_'+a]=d[a]-d['clock_mean']
 if 'block' not in d:d['block']=d.tissue.astype(str)+'|'+d.assay.astype(str)+'|'+d.donor.astype(str)
 d['donor']=d.donor.astype(str);return d

def residualize(d,features,mode):
 # Exact block fixed-intercept removal then nuisance phase regressions. All columns complete.
 z=d[features].astype(float)
 z=z-z.groupby(d['block']).transform('mean')
 theta=W*d.phase_center_h.to_numpy(float)
 C=np.cos(theta);S=np.sin(theta)
 reg=[C,S]
 if 'n_eff_median' in d:
  ne=np.log1p(np.maximum(pd.to_numeric(d.n_eff_median,errors='coerce').fillna(1).to_numpy(float),0))
  reg.append(ne)
 if mode=='tissue_phase':
  tissues=d.tissue.astype(str).to_numpy()
  unique,counts=np.unique(tissues,return_counts=True)
  for ti,n in zip(unique,counts):
   if n<14:continue
   mask=(tissues==ti).astype(float);reg += [C*mask,S*mask]
 if mode=='tissue_phase' and 'clock_mean' in z.columns:
  reg.append(z['clock_mean'].to_numpy(float))  # ACTUAL adjustment, not merely reporting it
 H=np.column_stack(reg)
 # Remove block means also from nuisance matrix (FWL fixed effects).
 hh=pd.DataFrame(H,index=d.index)
 H=(hh-hh.groupby(d['block']).transform('mean')).to_numpy(float)
 H=np.column_stack([H,np.ones(len(H))])
 valid=np.isfinite(H).all(axis=1)
 if not valid.all():raise ValueError('Nonfinite nuisance matrix')
 # A small ridge avoids ill-conditioning from tissue-phase interactions.
 M=H.T@H+1e-7*np.eye(H.shape[1]); coef=np.linalg.solve(M,H.T@z.to_numpy())
 out=z.to_numpy()-H@coef
 return pd.DataFrame(out,columns=features,index=d.index)

def stat_xy(d,x,y,perms,boot,rng):
 """One donor one statistical unit, pooling its available tissue blocks evenly."""
 xx=np.asarray(x,float);yy=np.asarray(y,float)
 rows=[];within=[]
 for block,s in d.groupby('block'):
  idx=d.index.get_indexer(s.index);a=xx[idx];b=yy[idx]
  if len(idx)<4 or np.std(a)<1e-9 or np.std(b)<1e-9:continue
  rows.append((str(s.donor.iloc[0]),float(np.mean(a*b)),float(np.mean(a*a)),float(np.mean(b*b)),str(s.tissue.iloc[0])))
  within.append(np.corrcoef(a,b)[0,1])
 if not rows:return dict(r=np.nan,p=np.nan,low=np.nan,high=np.nan,n_donors=0,n_blocks=0,block_sign_fraction=np.nan)
 q=pd.DataFrame(rows,columns=['donor','xy','xx','yy','tissue'])
 g=q.groupby('donor')[['xy','xx','yy']].mean().to_numpy(float)
 n=len(g)
 if n<6:return dict(r=np.nan,p=np.nan,low=np.nan,high=np.nan,n_donors=n,n_blocks=len(q),block_sign_fraction=np.nan)
 m=g.mean(axis=0);r=float(m[0]/np.sqrt(max(m[1]*m[2],1e-16)))
 signs=rng.choice([-1,1],size=(perms,n));null=(signs@g[:,0])/n
 p=float((1+np.sum(np.abs(null)>=abs(m[0])-1e-12))/(perms+1))
 ix=rng.integers(0,n,size=(boot,n));boots=g[ix].mean(axis=1)
 rc=boots[:,0]/np.sqrt(np.maximum(boots[:,1]*boots[:,2],1e-16))
 return dict(r=r,p=p,low=float(np.quantile(rc,.025)),high=float(np.quantile(rc,.975)),
  n_donors=n,n_blocks=len(q),n_tissues=q.tissue.nunique(),block_sign_fraction=float(np.mean(np.sign(within)==np.sign(r))))

def assoc_validate(wide,mods,perms,boot,rng,fast=False):
 res=[]; split=[];candidate=[];tfrows=[]
 chosen={(l,a,m) for l,a,m in CANDIDATE_ASSOC}
 for li in LINEAGES:
  part=wide[wide.lineage.eq(li)].copy()
  if part.empty:continue
  for j,mod in enumerate(mods):
   if mod not in part:continue
   for mode in ['global_phase','tissue_phase']:
    cols=['relative_'+a for a in ARMS]+[mod]
    extra=[]
    if mode=='tissue_phase':extra=['clock_mean']  # explicit global arm baseline nuisance control
    sub=part.replace([np.inf,-np.inf],np.nan).dropna(subset=cols+extra+['phase_center_h']).copy()
    if len(sub)<48 or sub.donor.nunique()<6:continue
    feats=cols+extra
    rr=residualize(sub,feats,mode)
    y=rr[mod].to_numpy(float)
    for arm in ARMS:
     x=rr['relative_'+arm].to_numpy(float)
     rrng=np.random.default_rng(int(rng.integers(2**31-1)))
     stat=stat_xy(sub,x,y,perms,boot,rrng)
     rec=dict(lineage=li,pathway=mod,arm=arm,adjustment=mode,**stat)
     res.append(rec)
     key=(li,arm,mod)
     if key in chosen:
      candidate.append(rec.copy())
     # tissue-split replication, especially for candidates, but save for all tests.
     if mode=='global_phase':
      tis=sorted(sub.tissue.unique());sides={t:int(hashlib.sha256(str(t).encode()).hexdigest(),16)%2 for t in tis}
      ss=[]
      for half in (0,1):
       take=sub.tissue.map(sides).to_numpy()==half
       if take.sum()<18: ss.append(np.nan);continue
       tmp=sub.loc[take];xa=x[take];ya=y[take]
       groups=tmp.groupby('donor').indices
       cs=[np.mean(xa[ix]*ya[ix]) for ix in groups.values()]
       xs=[np.mean(xa[ix]**2) for ix in groups.values()]
       ys=[np.mean(ya[ix]**2) for ix in groups.values()]
       ss.append(float(np.mean(cs)/np.sqrt(np.mean(xs)*np.mean(ys))) if len(cs)>=4 and np.mean(xs)*np.mean(ys)>1e-14 else np.nan)
      split.append(dict(lineage=li,pathway=mod,arm=arm,r_split0=ss[0],r_split1=ss[1],
       both_splits_same_sign=bool(np.isfinite(ss).all() and ss[0]*ss[1]>0)))
    # TF-output conditional correlations in candidate modules, with same per-donor test.
    if mode=='global_phase' and any(l==li and m==mod for l,_,m in chosen):
     tf_candidates=[t for t in ['RUNX1','CREM','MEF2A','TBX21','PPARG','TFAP2C','ERG','SPI1'] if t in part]
     if tf_candidates:
      cols2=tf_candidates+[mod]
      tt=part.replace([np.inf,-np.inf],np.nan).dropna(subset=cols2+['phase_center_h']).copy()
      if len(tt)>=40:
       ttr=residualize(tt,cols2,'global_phase')
       for tf in tf_candidates:
        st=stat_xy(tt,ttr[tf].to_numpy(),ttr[mod].to_numpy(),perms,boot,rng)
        tfrows.append(dict(lineage=li,pathway=mod,TF=tf,**st))
   if j%10==0:print(f'  association {li}: {j+1}/{len(mods)} pathways',flush=True)
 res=pd.DataFrame(res);split=pd.DataFrame(split);candidate=pd.DataFrame(candidate);tfrows=pd.DataFrame(tfrows)
 if not res.empty:
  for mode in res.adjustment.unique():
   idx=res.adjustment.eq(mode);res.loc[idx,'BH_q_1500_within_adjustment']=bh(res.loc[idx,'p'])
 if not tfrows.empty:tfrows['BH_q_all_candidate_TF_output_tests']=bh(tfrows.p)
 if not candidate.empty and not res.empty:
  candidate=candidate.merge(res[['lineage','arm','pathway','adjustment','BH_q_1500_within_adjustment']],
    on=['lineage','arm','pathway','adjustment'],how='left')
 return res,split,candidate,tfrows

def tissue_specific_candidate(wide,candidates,perms,boot,rng):
 rows=[]
 for li,arm,mod in CANDIDATE_ASSOC:
  part=wide[wide.lineage.eq(li)].copy()
  if mod not in part:continue
  for ti,z in part.groupby('tissue'):
   cols=['relative_'+arm,mod,'clock_mean']
   d=z.replace([np.inf,-np.inf],np.nan).dropna(subset=cols+['phase_center_h']).copy()
   if d.donor.nunique()<4 or d.block.nunique()<4 or len(d)<18:continue
   rr=residualize(d,cols,'global_phase')
   st=stat_xy(d,rr['relative_'+arm].to_numpy(),rr[mod].to_numpy(),min(perms,499),boot,rng)
   rows.append(dict(lineage=li,arm=arm,pathway=mod,tissue=ti,**st))
 return pd.DataFrame(rows)

def raw_vs_relative_candidates(wide,perms,boot,rng):
 """Distinguish NFIL3/arm absolute abundance, overall clock-arm level, and compositional relative architecture."""
 rows=[]
 for li,arm,mod in CANDIDATE_ASSOC:
  if mod not in wide:continue
  d=wide[wide.lineage.eq(li)].replace([np.inf,-np.inf],np.nan).dropna(
   subset=[arm,'relative_'+arm,'clock_mean',mod,'phase_center_h']).copy()
  if len(d)<40:continue
  for mode in ['global_phase','tissue_phase']:
   f=[arm,'relative_'+arm,'clock_mean',mod]
   r=residualize(d,f,mode)
   for feature in [arm,'relative_'+arm]:
    st=stat_xy(d,r[feature].to_numpy(),r[mod].to_numpy(),perms,boot,rng)
    rows.append(dict(lineage=li,pathway=mod,arm=arm,feature=feature,
       control=mode, **st))
 res=pd.DataFrame(rows)
 if not res.empty:res['BH_q_28_tests']=bh(res.p)
 return res

def candidate_circular_permutation(wide,perms,rng):
 # Secondary null: shift relative-arm residual curves independently within each donor+tissue+assay block.
 rows=[]
 for li,arm,mod in CANDIDATE_ASSOC:
  if mod not in wide:continue
  d=wide[wide.lineage.eq(li)].replace([np.inf,-np.inf],np.nan).dropna(subset=['relative_'+arm,mod,'phase_center_h']).copy()
  if len(d)<40:continue
  r=residualize(d,['relative_'+arm,mod],'global_phase')
  xx=r['relative_'+arm].to_numpy(float); yy=r[mod].to_numpy(float)
  blocks=[]
  for _,z in d.groupby('block'):
   idx=d.index.get_indexer(z.sort_values('phase_center_h').index)
   if len(idx)>=4:blocks.append(idx)
  if len(blocks)<6:continue
  cov=np.array([np.mean(xx[ix]*yy[ix]) for ix in blocks]);obs=float(cov.mean())
  null=[]
  for _ in range(perms):
   cc=[]
   for ix in blocks:
    sh=int(rng.integers(len(ix)))
    cc.append(float(np.mean(np.roll(xx[ix],sh)*yy[ix])))
   null.append(float(np.mean(cc)))
  pv=(1+np.sum(np.abs(null)>=abs(obs)-1e-12))/(perms+1)
  rows.append(dict(lineage=li,arm=arm,pathway=mod,n_blocks=len(blocks),observed_mean_covariance=obs,
   within_block_circular_shift_p=pv,note='Secondary dependence test; six inferred phase bins, no independent true-time labels.'))
 res=pd.DataFrame(rows)
 if not res.empty:res['BH_q_7_candidate_tests']=bh(res.within_block_circular_shift_p)
 return res

def CV_validate(cv,boot,perms,rng):
 rows=[]; tissue=[]
 for pathway,d in cv.groupby('pathway'):
  val=d.delta_RMSE.to_numpy(float);n=len(val)
  if not n:continue
  ix=rng.integers(n,size=(boot,n));means=val[ix].mean(axis=1)
  null=np.mean(val*rng.choice([-1,1],size=(perms,n)),axis=1)
  p=(1+np.sum(null>=val.mean()-1e-12))/(perms+1)
  rows.append(dict(pathway=pathway,n_heldout_tissues=n,mean_delta_RMSE=float(val.mean()),
   median_delta_RMSE=float(np.median(val)),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),
   fraction_tissues_positive=float(np.mean(val>0)),tissue_sign_test_p=float(binomtest(int((val>0).sum()),n,.5,alternative='greater').pvalue),
   exploratory_tissue_signflip_p=float(p),n_test_total=int(d.n_test.sum()),
   scheme=';'.join(sorted(d.validation_scheme.astype(str).unique()))))
  for z in d.itertuples(index=False):tissue.append(dict(pathway=pathway,tissue=z.held_out_group,delta_RMSE=z.delta_RMSE,
   rmse_baseline=z.rmse_baseline,rmse_plus_six_arm=z.rmse_plus_six_arm,
   n_test=z.n_test,n_test_donors=z.n_test_donors))
 res=pd.DataFrame(rows)
 if not res.empty:
  res['BH_q_tissue_sign']=bh(res.tissue_sign_test_p)
  res['BH_q_exploratory_signflip']=bh(res.exploratory_tissue_signflip_p)
 return res,pd.DataFrame(tissue)

def triads_join(assoc,tfout,triad):
 if assoc.empty or triad.empty:return pd.DataFrame()
 a=assoc[assoc.adjustment.eq('global_phase')][['lineage','arm','pathway','r','low','high','p','BH_q_1500_within_adjustment','n_donors']].rename(columns={'r':'arm_output_r','p':'arm_output_p','n_donors':'arm_output_n_donors'})
 t=triad.merge(a,on=['lineage','arm','pathway'],how='left')
 if not tfout.empty:
  tf=tfout[['lineage','TF','pathway','r','p','BH_q_all_candidate_TF_output_tests']].rename(columns={'r':'TF_output_recomputed_r','p':'TF_output_donor_p'})
  t=t.merge(tf,on=['lineage','TF','pathway'],how='left')
 key={(l,arm,mod) for l,arm,mod in CANDIDATE_ASSOC}
 t=t[t.apply(lambda x:(x.lineage,x.arm,x.pathway) in key,axis=1)].copy()
 if not t.empty:t=t.sort_values('triad_min_abs_r',ascending=False)
 return t

def visuals(out,ph,blocks,pares,assoc,candidate,rawcheck,cv,cvfold,triads,tissuecand,split,figdir):
 # A: candidates and all-pathway phase coherence; horizontal forest of R bootstrap.
 p=ph.sort_values('phase_concentration_R',ascending=True)
 fig,ax=plt.subplots(figsize=(9.1,13.4)); ys=np.arange(len(p))
 for i,(_,z) in enumerate(p.iterrows()):
  c={'Shared candidate':'#447C69','Divergent candidate':'#C76958'}.get(z.family,'#9DA3AC')
  if np.isfinite(z.R_boot_ci_low):ax.plot([z.R_boot_ci_low,z.R_boot_ci_high],[i,i],color=c,alpha=.53,lw=1.2)
  ax.scatter(z.phase_concentration_R,i,c=c,s=26 if z.family!='Other Hallmark' else 9,zorder=3)
 ax.set_yticks(ys);ax.set_yticklabels([label(z) for z in p.pathway],fontsize=7)
 ax.set_xlim(0,1.03);ax.set_xlabel('Cross-lineage circular phase concentration (R)')
 ax.set_title('A | Phase-coherence continuum across 50 Hallmarks (inferred B0)')
 ax.grid(axis='x',alpha=.14);figsave(fig,figdir/'A_all_Hallmarks_phase_coherence')
 # B: phase heatmap only for 10 preselected modules, no numbers inside cells.
 selected=[p for p in FAMILY if p in set(ph.pathway)]; q=ph.set_index('pathway').reindex(selected)
 dat=np.array([[q.loc[m,f'phase_{li}'] for li in LINEAGES] for m in selected],float)
 fig,ax=plt.subplots(figsize=(9,5.6)); cmap=plt.colormaps['twilight_shifted'].copy();cmap.set_bad('#ECEDEF')
 im=ax.imshow(np.ma.masked_invalid(dat),vmin=0,vmax=24,cmap=cmap,aspect='auto')
 ax.set_xticks(np.arange(5));ax.set_xticklabels(['B cell','T cell','Mac/Mono','Fibroblast','Endothelial'],rotation=25,ha='right')
 ax.set_yticks(np.arange(len(selected)));ax.set_yticklabels([label(m) for m in selected],fontsize=8)
 ax.set_title('B | Inferred-phase output peaks (no true collection time)')
 fig.colorbar(im,ax=ax,shrink=.8,label='Inferred peak (h; 0 h = 24 h)');figsave(fig,figdir/'B_inferred_phase_output_heatmap')
 # B2: pairwise phase-resolved vector contrasts for ten modules: fraction BH supported.
 f=pares[pares.pathway.isin(selected)] if not pares.empty else pd.DataFrame()
 if not f.empty:
  pi=f.pivot_table(index='pathway',columns=['lineage_A','lineage_B'],values='interaction_vector_magnitude')
  arr=pi.reindex(selected).to_numpy();fig,ax=plt.subplots(figsize=(11,5.5));im=ax.imshow(arr,aspect='auto',cmap='magma',vmin=0)
  ax.set_yticks(range(len(selected)));ax.set_yticklabels([label(m) for m in selected],fontsize=8)
  ax.set_xticks(range(len(pi.columns)));ax.set_xticklabels([f'{a.split()[0]}–{b.split()[0]}' for a,b in pi.columns],rotation=45,ha='right',fontsize=8)
  for j,m in enumerate(selected):
   for k,(a,b) in enumerate(pi.columns):
    ss=f[(f.pathway==m)&(f.lineage_A==a)&(f.lineage_B==b)]
    if len(ss) and float(ss.BH_q_all_pair_pathway.iloc[0])<=.05:ax.plot(k,j,'.',color='white',markersize=6)
  fig.colorbar(im,ax=ax,label='Harmonic-vector contrast magnitude')
  ax.set_title('B2 | Matched-donor lineage differences; dots: BH q≤0.05')
  figsave(fig,figdir/'B2_matched_donor_lineage_harmonic_contrasts')
 # C: association candidate display, adjusted comparison.
 if not candidate.empty:
  q=candidate.sort_values(['lineage','pathway','arm','adjustment'])
  keys=list(dict.fromkeys((l,a,m) for l,a,m in CANDIDATE_ASSOC if ((q.lineage==l)&(q.arm==a)&(q.pathway==m)).any()))
  fig,ax=plt.subplots(figsize=(10,max(4,len(keys)*.56+1.2))); yy=np.arange(len(keys));
  for k,(l,a,m) in enumerate(keys):
   zz=q[(q.lineage==l)&(q.arm==a)&(q.pathway==m)]
   for mode,off,c in [('global_phase',-.11,'#4E79A7'),('tissue_phase',+.11,'#F28E2B')]:
    h=zz[zz.adjustment==mode];
    if h.empty:continue
    z=h.iloc[0];ax.plot([z.low,z.high],[k+off,k+off],color=c,lw=2)
    ax.scatter(z.r,k+off,color=c,s=30,marker='o' if z.BH_q_1500_within_adjustment<=.05 else 'o',facecolors=c if z.BH_q_1500_within_adjustment<=.05 else 'white',zorder=3)
  ax.axvline(0,color='#555',lw=1);ax.set_yticks(yy)
  ax.set_yticklabels([f'{l} | {a} | {label(m)}' for l,a,m in keys],fontsize=7.9)
  ax.invert_yaxis();ax.set_xlabel('Donor-block weighted conditional correlation, 95% bootstrap CI')
  ax.set_title('C | Candidate clock–function coupling; filled = BH q≤0.05')
  ax.scatter([],[],color='#4E79A7',label='Global phase');ax.scatter([],[],color='#F28E2B',label='Tissue-specific phase + mean clock arm')
  ax.legend(frameon=False,loc='lower right',fontsize=8)
  figsave(fig,figdir/'C_candidate_coupling_donor_CI')
 # C2: six arms vs 10 selected functions, separated lineage, global phase residual.
 if not assoc.empty:
  qq=assoc[assoc.adjustment=='global_phase']
  for li in LINEAGES:
   sub=qq[qq.lineage.eq(li)];sel=[p for p in selected if p in set(sub.pathway)]
   if not sel:continue
   mat=np.array([[sub.loc[(sub.pathway==m)&(sub.arm==a),'r'].iloc[0] if ((sub.pathway==m)&(sub.arm==a)).any() else np.nan for a in ARMS] for m in sel])
   fig,ax=plt.subplots(figsize=(7.6,5.3)); mx=max(.15,float(np.nanmax(abs(mat))))
   im=ax.imshow(mat,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-mx,vcenter=0,vmax=mx),aspect='auto')
   ax.set_xticks(range(6));ax.set_xticklabels([x.replace('_',' ') for x in ARMS],rotation=40,ha='right',fontsize=8)
   ax.set_yticks(range(len(sel)));ax.set_yticklabels([label(m) for m in sel],fontsize=8)
   for j,m in enumerate(sel):
    for k,a in enumerate(ARMS):
     ss=sub[(sub.pathway==m)&(sub.arm==a)]
     if len(ss) and ss.BH_q_1500_within_adjustment.iloc[0]<=.05:ax.plot(k,j,'.',color='black',markersize=5)
   fig.colorbar(im,ax=ax,shrink=.8,label='Conditional donor-weighted r')
   ax.set_title(f'C2 | {li}: clock-arm/output association; BH dots')
   figsave(fig,figdir/('C2_'+li.lower().replace(' ','_').replace('/','_')+'_arm_output'))
 # C3: compositional control, raw vs relative arm and stringent tissue-clock-mean adjustment.
 if not rawcheck.empty:
  keys=list(dict.fromkeys((l,a,m) for l,a,m in CANDIDATE_ASSOC))
  fig,ax=plt.subplots(figsize=(9.2,6.4))
  for k,(li,arm,mod) in enumerate(keys):
   q=rawcheck[(rawcheck.lineage==li)&(rawcheck.arm==arm)&(rawcheck.pathway==mod)&
       (rawcheck.control=='tissue_phase')]
   for name,offset,col in [(arm,-.13,'#777777'),('relative_'+arm,+.13,'#4E79A7')]:
    t=q[q.feature==name]
    if t.empty:continue
    z=t.iloc[0];ax.plot([z.low,z.high],[k+offset,k+offset],color=col,alpha=.8)
    ax.scatter(z.r,k+offset,color=col,s=28)
  ax.axvline(0,lw=.9,color='black');ax.set_yticks(np.arange(len(keys)))
  ax.set_yticklabels([f'{l} / {a} / {label(m)}' for l,a,m in keys],fontsize=7.5)
  ax.invert_yaxis();ax.set_xlabel('Tissue-phase + overall arm-level adjusted donor r')
  ax.set_title('C3 | Raw arm vs six-arm-relative metric; distinct estimands')
  ax.scatter([],[],c='#777777',label='Raw arm');ax.scatter([],[],c='#4E79A7',label='Relative arm')
  ax.legend(frameon=False,loc='lower right')
  figsave(fig,figdir/'C3_raw_vs_relative_arm_sensitivity')
 # D: Tissue estimate stability, selected pairs; no one tissue claim if insufficient donors.
 if not tissuecand.empty:
  tt=tissuecand.copy();tt=tt[np.isfinite(tt.r)]
  chosen=[c for c in CANDIDATE_ASSOC if c[2] in tt.pathway.values];chosen=chosen[:5]
  for li,arm,mod in chosen:
   s=tt[(tt.lineage==li)&(tt.arm==arm)&(tt.pathway==mod)].sort_values('r')
   if len(s)<2:continue
   fig,ax=plt.subplots(figsize=(8,max(3.8,len(s)*.30+1)))
   for j,z in enumerate(s.itertuples()):
    ax.plot([z.low,z.high],[j,j],color='#77929E',lw=1.5)
    ax.scatter(z.r,j,color=PALETTE[li],s=18)
   ax.axvline(0,color='black',lw=.8);ax.set_yticks(np.arange(len(s)));ax.set_yticklabels(s.tissue,fontsize=7.4)
   ax.set_xlabel('Within-tissue donor-block r (exploratory; small n possible)');ax.set_title(f'D | {li} / {arm} / {label(mod)}')
   figsave(fig,figdir/('D_'+li.lower().split()[0]+'_'+arm+'_'+mod.replace('HALLMARK_','').lower()))
 # E: all 50 pathway predictive delta RMSE confidence; if 95% includes zero can't imply improvement.
 if not cv.empty:
  d=cv.sort_values('mean_delta_RMSE');fig,ax=plt.subplots(figsize=(9,13));ys=np.arange(len(d))
  for i,row in enumerate(d.itertuples()):
   c='#447C69' if row.ci_low>0 else ('#C76958' if row.ci_high<0 else '#9BA0A9')
   ax.plot([row.ci_low,row.ci_high],[i,i],color=c,lw=1.2)
   ax.scatter(row.mean_delta_RMSE,i,color=c,s=13)
  ax.axvline(0,color='black',lw=.9)
  ax.set_yticks(ys);ax.set_yticklabels([label(x) for x in d.pathway],fontsize=7)
  ax.set_xlabel('Held-tissue-out ΔRMSE = RMSE(base) − RMSE(+arms)')
  ax.set_title('E | Whole-atlas incremental prediction; 95% tissue bootstrap')
  figsave(fig,figdir/'E_all_Hallmarks_held_tissue_predictive_effect')
 # E2 by tissue, fraction better among 50 pathways.
 if not cvfold.empty:
  e=cvfold.groupby('tissue').agg(fraction_positive=('delta_RMSE',lambda x:np.mean(x>0)),mean_delta=('delta_RMSE','mean')).sort_values('fraction_positive')
  fig,ax=plt.subplots(figsize=(9,7.5));ax.barh(range(len(e)),e.fraction_positive,color='#6D8B99')
  ax.axvline(.5,color='#B45F5F',ls='--',lw=1);ax.set_yticks(range(len(e)));ax.set_yticklabels(e.index,fontsize=8)
  ax.set_xlim(0,1);ax.set_xlabel('Fraction of the 50 Hallmarks improved by six-arm addition')
  ax.set_title('E2 | Prediction gains are tissue dependent')
  figsave(fig,figdir/'E2_tissue_prediction_heterogeneity')
 # TF: only observable triads, not causal arrows.
 if not triads.empty:
  t=triads[(triads.lineage=='Fibroblast')&(triads.arm=='DBOX_REP')&(triads.pathway=='HALLMARK_TNFA_SIGNALING_VIA_NFKB')].copy()
  if len(t):
   t=t.sort_values('triad_min_abs_r',ascending=True)
   fig,ax=plt.subplots(figsize=(7.6,4.6));y=np.arange(len(t))
   ax.scatter(t.TF_clock_prior_r,y,color='#4E79A7',label='TF–arm prior')
   ax.scatter(t.arm_output_r,y+.13,color='#59A14F',label='Arm–output donor r')
   if 'TF_output_recomputed_r' in t.columns:
    ax.scatter(t.TF_output_recomputed_r,y-.13,color='#E15759',label='TF–output donor r')
   else:
    ax.scatter(t.TF_output_partial_r,y-.13,color='#E15759',marker='x',label='TF–output old exploratory r (no retest)')
   ax.axvline(0,color='#555',lw=.8);ax.set_yticks(y);ax.set_yticklabels(t.TF)
   ax.set_xlabel('Conditional expression association (not causality)')
   ax.set_title('F | Observational TF–arm–output candidate coherence')
   ax.legend(frameon=False,fontsize=8);figsave(fig,figdir/'F_TF_clock_output_observational_edges')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--project-root',type=Path,required=True)
 ap.add_argument('--source-dir',type=Path,default=None,help='Defaults to TABULA_FINAL_CLOCK_OUTPUT_FIGURE')
 ap.add_argument('--output-dir',type=Path,default=None)
 ap.add_argument('--permutations',type=int,default=1999)
 ap.add_argument('--bootstraps',type=int,default=1000)
 ap.add_argument('--seed',type=int,default=20261012)
 ap.add_argument('--no-figures',action='store_true')
 args=ap.parse_args();root=args.project_root.resolve()
 source=(args.source_dir or root/'TABULA_FINAL_CLOCK_OUTPUT_FIGURE').resolve()
 out=(args.output_dir or root/'TABULA_FIG6_THREE_DISCOVERIES_V1').resolve()
 if out==source:raise ValueError('Output must not overwrite source directory')
 out.mkdir(parents=True,exist_ok=True);figdir=out/'FIGURES'
 log=lambda *s:print(datetime.now().strftime('%F %T'),*s,flush=True)
 required={
 'module_units':source/'03_TABULA_MODULE_UNIT_SCORES.csv.gz',
 'matched_wide':source/'03_TABULA_CLOCK_OUTPUT_MATCHED_WIDE.csv.gz',
 'phase_old':source/'03_TABULA_DONOR_PHASE_OUTPUT_HARMONICS.csv',
 'cv_old':source/'04_HELD_TISSUE_OUT_PAIRED_SCORES.csv',
 'assoc_old':source/'05_CLOCK_ARM_OUTPUT_CONDITIONAL_ASSOCIATIONS.csv',
 'triads_old':source/'05_TF_CLOCK_OUTPUT_TRIADS_OBSERVATIONAL.csv'}
 missing=[str(v) for v in required.values() if not v.exists()]
 if missing:raise FileNotFoundError('Missing completed Tabula source files:\n'+'\n'.join(missing)+
  '\nRe-use the finished TABULA_FINAL_CLOCK_OUTPUT_FIGURE. Do not point at GSE output.')
 state={'state':'RUNNING','source':str(source),'created':datetime.now().isoformat(),
 'warning':'Inferred B0 phase only. Phase assignments were fitted from Clock17; associations observational.',
 'assay':'Tabula only, excludes GSE292219','input_paths':{k:str(v) for k,v in required.items()}}
 (out/'00_STATUS.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
 rng=np.random.default_rng(args.seed)
 log('Reading existing precomputed Tabula files; NO raw H5AD and NO B0 refit')
 mod=pd.read_csv(required['module_units']);wide=pd.read_csv(required['matched_wide']);cv=pd.read_csv(required['cv_old']);triad=pd.read_csv(required['triads_old'])
 need_mod={'tissue','assay','donor','lineage','pathway','score','phase_center_h','phase_bin'}
 need_wide={'tissue','assay','donor','lineage','phase_center_h'}|set(ARMS)
 if not need_mod.issubset(mod.columns):raise ValueError('MODULE schema missing '+str(need_mod-set(mod.columns)))
 if not need_wide.issubset(wide.columns):raise ValueError('WIDE schema missing '+str(need_wide-set(wide.columns)))
 if len(mod)<100 or len(wide)<100:raise ValueError('Unexpectedly small file; do not analyze pilot subset as full atlas')
 mod=mod[mod.lineage.isin(LINEAGES)].copy();mod.donor=mod.donor.astype(str)
 if mod.duplicated(['tissue','assay','donor','lineage','phase_bin','pathway']).any():raise ValueError('Duplicated module-unit keys')
 wide=make_rel(wide[wide.lineage.isin(LINEAGES)].copy())
 mods=sorted(set(mod.pathway.unique())&set(wide.columns))
 if len(mods)<40:raise ValueError(f'Only {len(mods)} pathways; expected full Hallmark 50. Check source.')
 state.update(n_module_rows=len(mod),n_wide_rows=len(wide),n_modules=len(mods),n_tissues=int(wide.tissue.nunique()),n_unique_donors=int(wide.donor.nunique()))
 log('1/3 donor-level phase summaries and shared vs divergent contrast')
 blocks=block_curves(mod);outcsv(blocks,out/'01_BLOCK_LEVEL_HARMONIC_COEFFICIENTS.csv')
 ph,donors,boots=phase_stats(blocks,args.bootstraps,rng)
 outcsv(donors,out/'01_DONOR_LEVEL_HARMONIC_COEFFICIENTS.csv');outcsv(ph,out/'01_ALL_50_LINEAGE_PHASE_CONCENTRATION.csv')
 outcsv(family_controls(ph,rng),out/'01_PRESPECIFIED_FAMILY_EXPLORATORY_COMPARISON.csv')
 outcsv(tissue_sensitivity(blocks,list(FAMILY)),out/'01_LEAVE_TISSUE_OUT_PHASE_COHERENCE.csv')
 pares=matched_lineage_contrasts(blocks,args.permutations,args.bootstraps,rng)
 outcsv(pares,out/'01_MATCHED_DONOR_LINEAGE_PHASE_VECTOR_INTERACTIONS.csv')
 log('2/3 donor-block conditions, tissue-phase controls, tissue split, circular-shift negative control')
 assoc,split,candidates,tfout=assoc_validate(wide,mods,args.permutations,args.bootstraps,rng)
 for x,f in [(assoc,'02_ALL_1500_DONOR_BLOCK_ASSOCIATIONS.csv'),(split,'02_TISSUE_SPLIT_REPLICATION.csv'),
   (candidates,'02_SEVEN_CANDIDATES_STRICT_COMPARISON.csv'),(tfout,'02_CANDIDATE_TF_OUTPUT_DONOR_TESTS.csv')]:outcsv(x,out/f)
 tissuecand=tissue_specific_candidate(wide,candidates,args.permutations,args.bootstraps,rng)
 outcsv(tissuecand,out/'02_CANDIDATE_WITHIN_TISSUE_EFFECTS.csv')
 circ=candidate_circular_permutation(wide,args.permutations,rng);outcsv(circ,out/'02_CANDIDATE_WITHIN_BLOCK_PHASE_ROTATION_NULL.csv')
 rawcheck=raw_vs_relative_candidates(wide,args.permutations,args.bootstraps,rng)
 outcsv(rawcheck,out/'02_CANDIDATE_RAW_VS_RELATIVE_ARM_SENSITIVITY.csv')
 tftriads=triads_join(assoc,tfout,triad);outcsv(tftriads,out/'02_TF_CLOCK_OUTPUT_TRIADS_RETESTED.csv')
 log('3/3 tissue-held-out predictor generality and tissue heterogeneity')
 cvsum,cvfold=CV_validate(cv,args.bootstraps,args.permutations,rng)
 outcsv(cvsum,out/'03_ALL_50_PREDICTIVE_GENERALIZATION.csv')
 outcsv(cvfold,out/'03_TISSUE_BY_PATHWAY_PREDICTION_DELTA.csv')
 report=[]
 report.append('# Tabula Figure 6 — three discoveries: independent validation')
 report+=['','IMPORTANT: all six-arm and functional phases are inferred from the SAME frozen cohort-specific B0 model, not observed time.',
 'Never infer biochemical loop activity or causal TF→clock→output relationships.',
 'Pathway sets overlap, and phase derives from CLOCK17. No claim of independent time validation.',
 '','## Discovery 1: phase-sharing versus lineage-divergence']
 report.append(f'- Number of donor-level harmonic summaries: {len(donors)}; 50-pathway concentration records: {len(ph)}.')
 if len(ph):
  for fam in ['Shared candidate','Divergent candidate']:
   q=ph[ph.family.eq(fam)].phase_concentration_R.dropna()
   report.append(f'- {fam}: n={len(q)}, median R={np.median(q):.3f}' if len(q) else f'- {fam}: no data')
  report.append('- Family candidates were chosen after looking at earlier Tabula findings; label permutations are DESCRIPTIVE/selection-biased.')
 report.append('- Matched-donor contrasts test differences in harmonic vectors, which can reflect amplitude as well as peak phase.')
 report+=['','## Discovery 2: selective clock-arm/output coupling']
 if not assoc.empty:
  for mode in assoc.adjustment.unique():
   d=assoc[assoc.adjustment.eq(mode)];k=(d.BH_q_1500_within_adjustment<=.05).sum()
   report.append(f'- {mode}: {k}/{len(d)} tests BH≤.05 (donor sign-flip of residual covariance; observational).')
 report.append('- Candidate edges were selected from the earlier data, so these analyses are in-dataset robustness checks, not independent replication.')
 report.append('- Tissue-phase residualization, phase-bin circular-shift negative controls and tissue-split checks are saved separately.')
 report.append('- Relative six-arm scores are compositional; DBOX_REP may represent NFIL3 alone.')
 report+=['','## Discovery 3: held-out tissue predictive generality']
 if not cvsum.empty:
  report.append(f'- {len(cvsum)} pathways, {len(cvfold)} held-tissue×pathway evaluations.')
  report.append(f'- Mean ΔRMSE={cvsum.mean_delta_RMSE.mean():.6f}; pathways with bootstrap CI >0: {sum(cvsum.ci_low>0)}.')
  report.append('- Per-tissue bootstrap and sign tests are sensitivity metrics: leave-tissue-out training folds overlap and tissues are not independent experiments.')
 report+=['','## What is NOT tested here',
 '- Genuine gene expression-matched random GENE SET nulls: cannot be generated from pathway-level pseudobulk CSV. Requires separate gene-level extraction.',
 '- Independent real-time rhythms: frozen B0 phase is derived from clock gene expression.',
 '- Causal TF regulation or independent experiments.',
 '- Unbiased post-selection statistical inference for pathways highlighted in prior figures.',
 '','## Panels',
 'A: 50 Hallmarks phase-coherence forest; B: five-lineage inferred phase heatmap; B2: paired-donor harmonic contrasts.',
 'C: candidate association bootstrap; C2: six-arm × pathway heatmaps; C3: raw vs relative arm comparison.',
 'D: tissue-wise effect stability; E/E2: held-tissue prediction forest and heterogeneity; F: observational TF triad.',
 'All panels separate PNG/PDF/SVG; heatmaps have no in-cell numbers.']
 (out/'INTERPRETATION_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 if not args.no_figures:
  log('Rendering separate figures; no composite main figure')
  visuals(out,ph,blocks,pares,assoc,candidates,rawcheck,cvsum,cvfold,tftriads,tissuecand,split,figdir)
 state.update(state='COMPLETE',finished=datetime.now().isoformat(),n_block_harmonics=len(blocks),n_phase_tests=len(pares),
  n_association_tests=len(assoc),n_tissue_cv_folds=len(cv),n_new_panels=len(list(figdir.glob('*.png'))) if not args.no_figures else 0,
  any_pathway_positive_heldout_CI=bool((cvsum.ci_low>0).any()),
  full_gene_matched_random_gene_set_controls='NOT_TESTED_REQUIRES_GENE_LEVEL_EXTRACTION')
 (out/'00_STATUS.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
 log('COMPLETE',out)

if __name__=='__main__': main()
