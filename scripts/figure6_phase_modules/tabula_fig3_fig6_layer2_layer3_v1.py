#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig3 frozen-ontology bridge: tissue-associated vs inferred-phase-associated pathway programs.

Inputs (READ ONLY):
 - corrected_full_analysis_v2/00_freeze/A4_8_V3_FINAL_ONTOLOGY_20260920/
     01_V3_PATHWAY_ONTOLOGY.csv (98 pathways/69 redundant families)
     04_FORMAL_HITS_WITH_V3_ONTOLOGY.csv (127 Fig3 formal unit-pathway hits)
 - katv2/TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1/
     01_ATLAS_UNIT_METADATA.csv.gz
     01_ALL_PATHWAY_SCORES_WIDE.npz
     02_M012_MAIN_RESULTS.csv (optional consistency check)
 - katv2/tabula_fig6_Hallmark_Reactome_M012_v1.py (original frozen model).

Layer 2: M0 donor×tissue×assay×lineage FE + within-block effective size,
         then add common sin/cos(B0 inferred phase) (M1).
Layer 3: M1 plus 4 non-reference lineage×sin/cos interaction pairs (M2).

PRIMARY FAMILY UNIT: one representative pathway fixed by Fig3, 69 families.
98 pathway full-panel results (legacy model and covariate sensitivity) as sensitivity;
this script does NOT pretend family representatives are new composite scores.

Evidence:
 - donor wild-sign-flip restricted-null incremental tests: M1 vs M0 / M2 vs M1.
 - held-tissue, all-tissue-donors-disjoint CV of within-block centered shapes.
 - descriptive fitted lineage amplitude and peak of M2 (B0 *not* real ZT).
 - Fig3 frozen tissue pair correspondence beta vs Layer2/3 within-context effects.

Does not claim causal direction, independent true-time replication, pathway specificity,
or independence of the Fig3 selected 98 pathways from Tabula itself.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANNER='FIG3_FIG6_LAYER2_LAYER3_BRIDGE_V1'
LN=['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
LC={'B cell':'#4E79A7','T cell':'#E15759','Macrophage/Monocyte':'#59A14F','Fibroblast':'#F28E2B','Endothelial':'#9467BD'}
W=2*np.pi/24

def log(*x):print(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def write(df,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);df.to_csv(p,index=False,compression='gzip' if str(p).endswith('.gz') else None)
def js(obj,p):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def imp(path):
 s=importlib.util.spec_from_file_location('fig6_original_m012_v1_bridge',str(path))
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def ontology_path(root,external,explicit):
 if explicit:
  p=Path(explicit).expanduser().resolve()
  if not p.exists():raise FileNotFoundError(p)
  return p
 candidates=[external/'00_freeze'/'A4_8_V3_FINAL_ONTOLOGY_20260920'/'01_V3_PATHWAY_ONTOLOGY.csv',
             external/'10_A4_8_V3_FINAL_ONTOLOGY'/'01_V3_PATHWAY_ONTOLOGY.csv',
             root/'corrected_full_analysis_v2'/'00_freeze'/'A4_8_V3_FINAL_ONTOLOGY_20260920'/'01_V3_PATHWAY_ONTOLOGY.csv']
 matches=[p for p in candidates if p.is_file()]
 if len(matches)==1:return matches[0]
 if len(matches)>1:
  hs={digest(p) for p in matches}
  if len(hs)!=1:raise RuntimeError('Conflicting frozen ontologies; provide --ontology')
  return matches[0]
 # An exact-name fallback is bounded to the previously frozen corrected analysis dir
 matches=list(external.rglob('01_V3_PATHWAY_ONTOLOGY.csv')) if external.exists() else []
 if len(matches)==1:return matches[0]
 raise FileNotFoundError('Need exact frozen 01_V3_PATHWAY_ONTOLOGY.csv. Pass --ontology PATH; candidates='+str(matches[:6]))

def load_freeze(path):
 q=pd.read_csv(path,low_memory=False)
 required=['collection','pathway','complete_linkage_cluster_id','representative_collection','representative_pathway','v3_primary_module_id','v3_primary_module_label']
 miss=set(required)-set(q.columns)
 if miss:raise ValueError('Frozen ontology missing '+str(miss))
 if q[['collection','pathway']].duplicated().any():raise ValueError('Duplicate frozen pathway name')
 if q[['collection','pathway']].drop_duplicates().shape[0]!=98 or q.complete_linkage_cluster_id.nunique()!=69:
  raise ValueError('Expected exactly Fig3 frozen 98 distinct pathways / 69 families; found %s/%s' %(len(q),q.complete_linkage_cluster_id.nunique()))
 q['pathway']=q.pathway.astype(str);q['collection']=q.collection.astype(str)
 fam=[]
 for fid,g in q.groupby('complete_linkage_cluster_id',sort=True):
  reps=g[['representative_collection','representative_pathway']].drop_duplicates()
  if len(reps)!=1:raise ValueError('Ambiguous representative '+str(fid))
  c,p=reps.iloc[0]
  exists=((g.collection==c)&(g.pathway==p)).any()
  if not exists:raise ValueError('Representative not part of frozen family '+str(fid)+': '+str(p))
  r=g[(g.collection==c)&(g.pathway==p)].iloc[0]
  fam.append({'family_id':str(fid),'pathway':str(p),'collection':str(c),
              'module_id':str(r.v3_primary_module_id),'module_label':str(r.v3_primary_module_label),
              'n_pathways_in_family':len(g),
              'n_modules_in_family':g.v3_primary_module_id.nunique()})
 fam=pd.DataFrame(fam).sort_values(['module_id','family_id']).reset_index(drop=True)
 return q,fam

def input_matrices(root,output_subdir):
 a=root/output_subdir
 files={'meta':a/'01_ATLAS_UNIT_METADATA.csv.gz','matrix':a/'01_ALL_PATHWAY_SCORES_WIDE.npz',
        'legacy':a/'02_M012_MAIN_RESULTS.csv'}
 for k in ('meta','matrix'):
  if not files[k].exists():raise FileNotFoundError('Missing previously computed Tabula matrix: '+str(files[k]))
 meta=pd.read_csv(files['meta'],low_memory=False)
 z=np.load(files['matrix'],allow_pickle=False)
 cols=z['pathway'].astype(str).tolist(); mat=z['scores']
 if mat.shape!=(len(meta),len(cols)):raise ValueError('Matrix shape and metadata mismatch')
 if len(cols)!=len(set(cols)):raise ValueError('Duplicated score matrix pathway columns')
 req={'donor','tissue','assay','lineage','phase_bin','phase_center_h','n_eff'}
 if req-set(meta.columns):raise ValueError('Missing metadata: '+str(req-set(meta.columns)))
 if meta[['tissue','assay','donor','lineage','phase_bin']].duplicated().any():raise ValueError('Duplicated analysis unit keys')
 if set(meta.lineage.dropna().astype(str))-set(LN):raise ValueError('Unexpected lineages in frozen matrix')
 return files,meta,mat,cols

def build_single(meta,vec,mod):
 mask=np.isfinite(vec)
 d=meta.loc[mask].copy();d['score']=vec[mask]
 D=mod.build_design(d,include_neff=True)
 if D is None:return None
 if D['n_donors']<8 or D['n_tissues']<5 or D['n_lineages']<5 or D['nblocks']<20:return None
 return D

def fit_family(mod,meta,vec,seed,wild):
 D=build_single(meta,vec,mod)
 if D is None:return dict(status='INSUFFICIENT_BLOCK_DONOR_LINEAGE_COVERAGE')
 fit=mod.fit_models(D)
 if fit is None:return dict(status='MODEL_UNIDENTIFIABLE')
 if not bool(fit['full_rank_M2']):return dict(status='M2_RANK_DEFICIENT')
 sig={k:v for k,v in fit.items() if k!='_fits'}
 sig['status']='FIT'
 sig.update(mod.wild_boot(D,fit,wild,np.random.default_rng(seed)))
 # M2 curve coefficients use reference lineage + interactions, NOT unadjusted single-lineage descriptions.
 common=fit['_fits'][2][0]
 nuisance_count=D['X0'].shape[1]
 bc,bs=float(common[nuisance_count]),float(common[nuisance_count+1])
 prof=[]
 for li,lin in enumerate(LN):
  cc,ss=bc,bs
  if li:
   cc+=float(common[nuisance_count+2+(li-1)*2]);ss+=float(common[nuisance_count+3+(li-1)*2])
  amp=float(np.hypot(cc,ss))
  peak=float(np.arctan2(ss,cc)%(2*np.pi)/W)
  prof.append(dict(lineage=lin,amplitude=amp,peak_B0_h=peak,beta_cos=cc,beta_sin=ss,
                   phase_interpretation='DESCRIPTIVE_B0_NOT_OBSERVED_ZT'))
 return sig,prof

def cv_family(mod,meta,vec,min_held=3):
 use=np.isfinite(vec);z=meta.loc[use].copy();z['score']=vec[use]
 out=[]
 for tissue in sorted(z.tissue.dropna().astype(str).unique()):
  held_donors=set(z.loc[z.tissue.astype(str).eq(tissue),'donor'].astype(str))
  tr=z[~z.tissue.astype(str).eq(tissue)&~z.donor.astype(str).isin(held_donors)].copy()
  te=z[z.tissue.astype(str).eq(tissue)].copy()
  if len(tr)<100 or len(te)<10:continue
  a=mod.build_design(tr,include_neff=True);b=mod.build_design(te,include_neff=True)
  if a is None or b is None or a['n_donors']<8 or a['n_lineages']<4 or b['n_lineages']<2:continue
  if set(a['donors']).intersection(set(b['donors'])):raise RuntimeError('CV donor leakage: '+tissue)
  vals=[]
  for k in ('X0','X1','X2'):
   X,Q=a[k],b[k]
   if X.shape[1]!=Q.shape[1] or (X.shape[1] and np.linalg.matrix_rank(X)<X.shape[1]):
    vals.append(np.nan);continue
   coeff=mod.weighted_lstsq(X,a['y'],a['w'])[0] if X.shape[1] else np.array([])
   yhat=Q@coeff
   vals.append(float(np.sqrt(np.average((b['y']-yhat)**2,weights=b['w']))))
  if not np.all(np.isfinite(vals)):continue
  out.append(dict(held_tissue=tissue,n_train_donors=a['n_donors'],n_test_donors=b['n_donors'],
                  n_test_units=len(b['y']),rmse_M0=vals[0],rmse_M1=vals[1],rmse_M2=vals[2],
                  delta_rmse_M1=vals[0]-vals[1],delta_rmse_M2=vals[1]-vals[2],
                  validation='TISSUE_AND_DONOR_DISJOINT_TEST_BLOCK_CENTERED'))
 return out

def family_x98(q,fam,all_main,mod):
 d=q.merge(all_main,on=['collection','pathway'],how='left',validate='one_to_one',indicator=True)
 d['family_id']=d.complete_linkage_cluster_id.astype(str)
 m={p:i for i,p in enumerate(mod)}
 d['family_representative']=d.apply(lambda r:r.pathway==r.representative_pathway and r.collection==r.representative_collection,axis=1)
 d['original_Fig6_mode']='M0_M1_M2_UNADJUSTED_STATE'
 return d

def join_fig3_formal(ontology, external_dir):
 p=ontology.parent/'04_FORMAL_HITS_WITH_V3_ONTOLOGY.csv'
 if not p.exists():return None,p
 t=pd.read_csv(p,low_memory=False)
 if 'beta_standardized_pairwise_correspondence' not in t:raise ValueError('Fig3 formal hits missing original frozen correspondence beta')
 if not {'collection','pathway','complete_linkage_cluster_id'}.issubset(t.columns):raise ValueError('Fig3 formal-hit file missing pathway/family keys')
 return t,p

def fig3_link(frozen,formal):
 valid=formal.merge(frozen[['collection','pathway','complete_linkage_cluster_id','v3_primary_module_id']],
                    on=['collection','pathway'],how='inner',suffixes=('','_freeze'))
 if len(valid)!=len(formal):raise ValueError('Some Fig3 formal hits do not map to frozen ontology')
 beta='beta_standardized_pairwise_correspondence'
 group=valid.groupby('complete_linkage_cluster_id').agg(
  fig3_formal_n_hits=(beta,'size'),
  fig3_formal_median_beta=(beta,'median'),
  fig3_formal_max_abs_beta=(beta,lambda x:float(np.nanmax(np.abs(x)))),
  fig3_formal_n_cellunits=('cell_unit_id','nunique') if 'cell_unit_id' in valid else (beta,'size'),
 ).reset_index().rename(columns={'complete_linkage_cluster_id':'family_id'})
 group['family_id']=group.family_id.astype(str)
 return group

def summarize_family(fit,cv,q,main_all):
 merged=fit.merge(cv,on='family_id',how='left',validate='one_to_one')
 met=['deltaR2_M1_vs_M0','deltaR2_M2_vs_M1']
 old=q[['pathway','collection','complete_linkage_cluster_id']].merge(main_all[['pathway','collection']+met],on=['pathway','collection'],how='left')
 old['family_id']=old.complete_linkage_cluster_id.astype(str)
 fm=old.groupby('family_id').agg(family_n_paths=('pathway','size'),
  family_n_scores_available=(met[0],lambda x:int(x.notna().sum())),
  family_med_M1_deltaR2=(met[0],'median'),family_med_M2_deltaR2=(met[1],'median'),
  family_m1_max=(met[0],'max'),family_m2_max=(met[1],'max')).reset_index()
 merged=merged.merge(fm,on='family_id',how='left',validate='one_to_one')
 return merged

def graph(df,profiles,out,with_text):
 a=df[df.status.eq('FIT')].copy()
 if a.empty:return
 # fig: Fig3 static correspondence vs phase and lineage component; gene/pathway overlap retained, no causality inference.
 for xx,stem,c in [('deltaR2_M1_vs_M0','layer2_shared','#4E79A7'),('deltaR2_M2_vs_M1','layer3_lineage','#E15759')]:
  fig,ax=plt.subplots(figsize=(5.5,4.5));dat=a[np.isfinite(a[xx])&np.isfinite(a.fig3_formal_median_beta)] if 'fig3_formal_median_beta' in a else a.iloc[:0]
  if len(dat):
   ax.scatter(dat.fig3_formal_median_beta,dat[xx],s=37,color=c,alpha=.77,edgecolor='white',lw=.5)
  else:
   ax.text(.5,.5,'Fig3 frozen beta file not available',transform=ax.transAxes,ha='center',va='center') if with_text else None
  ax.spines[['top','right']].set_visible(False)
  if with_text:
   ax.set_xlabel('Fig3 tissue-pair clock–pathway correspondence (frozen beta)')
   ax.set_ylabel('Additional within-context explained variance (ΔR²)')
   ax.set_title('Layer 2: shared inferred-phase associations' if xx.endswith('M0') else 'Layer 3: lineage-dependent associations',loc='left')
  else:ax.set_xticklabels([]);ax.set_yticklabels([])
  for fmt in ('png','pdf','svg'):fig.savefig(out/f'FIG_{stem}_{"text" if with_text else "notext"}.{fmt}',dpi=400,bbox_inches='tight')
  plt.close(fig)
 if profiles.empty:return
 # Five lineage amplitudes measured using conditional M2 coefficients, no fake confidence ribbon.
 top=a[a.status.eq('FIT')].sort_values('deltaR2_M2_vs_M1',ascending=False).head(9).family_id.tolist()
 sub=profiles[profiles.family_id.isin(top)].copy()
 if sub.empty:return
 plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axs=plt.subplots(len(top),1,figsize=(7.4,max(5.4,1.55*len(top))),sharex=True,squeeze=False)
 x=np.linspace(0,24,200)
 for i,fid in enumerate(top):
  ax=axs[i,0]
  rows=sub[sub.family_id.eq(fid)]
  for lin in LN:
   r=rows[rows.lineage.eq(lin)]
   if r.empty:continue
   r=r.iloc[0];curve=r.beta_cos*np.cos(W*x)+r.beta_sin*np.sin(W*x)
   ax.plot(x,curve,lw=1.55,color=LC[lin],label=lin)
  ax.axhline(0,color='#BBBBBB',lw=.6)
  ax.spines[['top','right']].set_visible(False)
  if with_text:ax.set_ylabel('Within-block score');ax.set_title(f'{fid} | {rows.pathway.iloc[0]}',fontsize=8,loc='left')
  else:ax.set_yticklabels([])
 axs[-1,0].set_xticks([0,6,12,18,24])
 if with_text:
  axs[-1,0].set_xlabel('Frozen B0 inferred phase (hours; not observed ZT)')
  h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.5,1.015))
 else:axs[-1,0].set_xticklabels([])
 fig.tight_layout()
 for fmt in ('png','pdf','svg'):fig.savefig(out/f'FIG_layer3_conditional_lineage_curves_{"text" if with_text else "notext"}.{fmt}',dpi=400,bbox_inches='tight')
 plt.close(fig)

def selftest():
 # Temporary local project simulates 5 lineage × 6 bins × 16 distinct donors, multiple tissues.
 import tempfile
 with tempfile.TemporaryDirectory() as q:
  root=Path(q); code=Path(__file__).resolve().parent/'reference'/'tabula_fig6_Hallmark_Reactome_M012_v1.py'
  if not code.exists():code=Path(__file__).resolve().parent/'tabula_fig6_Hallmark_Reactome_M012_v1.py'
  if not code.exists():print('SELFTEST schema only: original model script not in sandbox; full selftest at project root');return
  mod=imp(code);rr=np.random.default_rng(40);rows=[];vec=[]
  for donor in range(16):
   for lin in LN:
    for b in range(6):
     theta=4*b+2
     rows.append(dict(donor='d'+str(donor),tissue='t'+str(donor%5),assay='10x',lineage=lin,phase_bin=b,phase_center_h=theta,n_eff=100))
     vec.append(.4*np.cos(W*theta)+(.45 if lin=='Macrophage/Monocyte' else 0)*np.sin(W*theta)+rr.normal(scale=.25))
  m=pd.DataFrame(rows);dat=build_single(m,np.asarray(vec),mod);assert dat and dat['n_donors']==16
  fit=mod.fit_models(dat);assert fit and fit['deltaR2_M1_vs_M0']>.2 and fit['deltaR2_M2_vs_M1']>.01
  sb,pr=fit_family(mod,m,np.asarray(vec),4,39)
  assert sb['wild_p_M1']<=1 and len(pr)==5 and np.isfinite(pr[0]['peak_B0_h'])
  cv=cv_family(mod,m,np.asarray(vec));assert len(cv)>=2
  log('SELFTEST_PASS','units',len(m),'M1',round(fit['deltaR2_M1_vs_M0'],3),'M2',round(fit['deltaR2_M2_vs_M1'],3),'held_folds',len(cv))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--project-root',default='.')
 ap.add_argument('--external-analysis-root',default=None,help='The parent corrected_full_analysis_v2 folder (not katv2)')
 ap.add_argument('--ontology',default=None)
 ap.add_argument('--source-folder',default='TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1')
 ap.add_argument('--output-folder',default=BANNER)
 ap.add_argument('--wild',type=int,default=399,help='Donor-level restricted-null sign flips per frozen representative')
 ap.add_argument('--seed',type=int,default=260913)
 ap.add_argument('--resume',action='store_true')
 ap.add_argument('--stage',choices=['all','audit','fit','cv','report','selftest'],default='all')
 ap.add_argument('--min-cv-folds',type=int,default=4)
 args=ap.parse_args()
 if args.stage=='selftest':selftest();return
 root=Path(args.project_root).expanduser().resolve()
 ext=(Path(args.external_analysis_root).resolve() if args.external_analysis_root else root.parent/'corrected_full_analysis_v2')
 origp=root/'tabula_fig6_Hallmark_Reactome_M012_v1.py'
 if not origp.is_file():raise FileNotFoundError('Place original M012 model script in katv2: '+str(origp))
 orig=imp(origp)
 ontp=ontology_path(root,ext,args.ontology);frozen,families=load_freeze(ontp)
 files,meta,matrix,names=input_matrices(root,args.source_folder)
 nameidx={x:i for i,x in enumerate(names)}
 legacy=(pd.read_csv(files['legacy']) if files['legacy'].exists() else pd.DataFrame())
 if legacy.empty or not {'pathway','collection','deltaR2_M1_vs_M0','deltaR2_M2_vs_M1'}.issubset(legacy.columns):
  raise ValueError('Need full M012 results for 98-pathway sensitivity and score audit')
 out=root/args.output_folder;out.mkdir(parents=True,exist_ok=True)
 sig={'version':'v1.0','ontology_sha256':digest(ontp),'matrix_sha256':digest(files['matrix']),
      'meta_sha256':digest(files['meta']),'original_model_sha256':digest(origp),
      'legacy_sha256':digest(files['legacy']),'wild':args.wild,'seed':args.seed,'min_cv_folds':args.min_cv_folds}
 sigfile=out/'00_INPUT_SIGNATURE.json'
 if sigfile.exists() and json.loads(sigfile.read_text())!=sig:raise RuntimeError('Previous output generated with different inputs/config. Choose a NEW --output-folder.')
 js(sig,sigfile)
 write(frozen,out/'00_FROZEN_FIG3_98_PATHWAYS.csv')
 write(families,out/'00_FROZEN_FIG3_69_REPRESENTATIVES.csv')
 availability=[]
 for _,r in families.iterrows():
  path=r.pathway
  availability.append(dict(family_id=r.family_id,representative_pathway=path,
                           present_in_saved_scores=path in nameidx,
                           n_available_units=int(np.isfinite(matrix[:,nameidx[path]]).sum()) if path in nameidx else 0))
 ava=pd.DataFrame(availability);write(ava,out/'00_REPRESENTATIVE_AVAILABILITY.csv')
 if ava.present_in_saved_scores.mean()<.8:raise RuntimeError('Below 80% family representative availability. Investigate scoring coverage before proceeding.')
 old=family_x98(frozen,families,legacy,names);write(old,out/'01_FIG3_FROZEN_98_x_LEGACY_M012.csv')
 formal,fp=join_fig3_formal(ontp,ext)
 if formal is not None:
  cross=fig3_link(frozen,formal)
  write(cross,out/'01_FIG3_STATIC_CORRESPONDENCE_69.csv')
 else:
  log('WARNING','Fig3 frozen formal hit table absent; static-versus-phase correlation not estimable:',fp)
  cross=pd.DataFrame(columns=['family_id','fig3_formal_median_beta','fig3_formal_n_hits'])
 if args.stage=='audit':
  log('AUDIT COMPLETE',len(frozen),'pathways',len(families),'families',int(ava.present_in_saved_scores.sum()),'available');return
 partial=out/'02_LAYER23_PARTIAL.csv';profpart=out/'02_LAYER3_PROFILES_PARTIAL.csv'
 if partial.exists() and args.resume:fitrows=pd.read_csv(partial).to_dict('records')
 else:fitrows=[]
 if profpart.exists() and args.resume:profile_rows=pd.read_csv(profpart).to_dict('records')
 else:profile_rows=[]
 done=set(r['family_id'] for r in fitrows)
 if args.stage in ('all','fit'):
  for i,r in families.iterrows():
   fid=r.family_id
   if fid in done:continue
   path=r.pathway
   if path not in nameidx:
    f=dict(status='NOT_SCORED',family_id=fid,pathway=path)
    ps=[]
   else:
    try:
     result=fit_family(orig,meta,matrix[:,nameidx[path]],args.seed+i*1009,args.wild)
     if isinstance(result,tuple):f,ps=result
     else:f,ps=result,[]
     f.update(family_id=fid,pathway=path,collection=r.collection,module_id=r.module_id,module_label=r.module_label)
    except Exception as e:
     raise RuntimeError(f'Failed family {fid}/{path} (not silently skipped)') from e
   fitrows.append(f)
   for p in ps:profile_rows.append(dict(family_id=fid,pathway=path,module_id=r.module_id,module_label=r.module_label,**p))
   write(pd.DataFrame(fitrows),partial)
   if profile_rows:write(pd.DataFrame(profile_rows),profpart)
   log('FIT',i+1,'/',len(families),fid,path,f.get('status'),'M1',round(float(f.get('deltaR2_M1_vs_M0',np.nan)),4),'M2',round(float(f.get('deltaR2_M2_vs_M1',np.nan)),4))
  if not fitrows:raise RuntimeError('No family fits')
  fs=pd.DataFrame(fitrows)
  # Reproduce old nested-model effect sizes when the eligible-unit set is identical.
  # A mismatch means this is NOT the same model/data matrix: stop rather than relabel it.
  oldref=legacy[['pathway','n_units','deltaR2_M1_vs_M0','deltaR2_M2_vs_M1']].copy() if 'n_units' in legacy.columns else pd.DataFrame()
  if not oldref.empty:
   ra=fs.merge(oldref,on='pathway',how='left',suffixes=('_recomputed','_original'),validate='many_to_one')
   same=ra.status.eq('FIT')&ra.n_units_recomputed.eq(ra.n_units_original)
   audit=ra.loc[same,['pathway','n_units_recomputed','deltaR2_M1_vs_M0_recomputed','deltaR2_M1_vs_M0_original','deltaR2_M2_vs_M1_recomputed','deltaR2_M2_vs_M1_original']].copy()
   audit['abs_diff_M1']=(audit.deltaR2_M1_vs_M0_recomputed-audit.deltaR2_M1_vs_M0_original).abs()
   audit['abs_diff_M2']=(audit.deltaR2_M2_vs_M1_recomputed-audit.deltaR2_M2_vs_M1_original).abs()
   write(audit,out/'01_FROZEN_M012_REPRODUCTION_AUDIT.csv')
   if len(audit) and max(audit[['abs_diff_M1','abs_diff_M2']].max())>1e-5:
    raise RuntimeError('M1/M2 results do not reproduce saved original despite matching n_units; inspect 01_FROZEN_M012_REPRODUCTION_AUDIT.csv')
  for src,dst in [('wild_p_M1','wild_BHq_M1_69_families'),('wild_p_M2','wild_BHq_M2_69_families')]:
   fs[dst]=orig.bh(pd.to_numeric(fs.get(src,pd.Series(np.nan,index=fs.index)),errors='coerce').to_numpy(float))
  write(fs,out/'02_LAYER2_LAYER3_FAMILY_FITS_69.csv')
  if profile_rows:write(pd.DataFrame(profile_rows),out/'02_LAYER3_M2_LINEAGE_CONDITIONAL_PROFILES.csv')
  log('FIT COMPLETE',fs.status.value_counts().to_dict())
 if args.stage in ('all','cv'):
  fitsfile=out/'02_LAYER2_LAYER3_FAMILY_FITS_69.csv'
  if not fitsfile.exists():raise FileNotFoundError('Run --stage fit first')
  fits=pd.read_csv(fitsfile)
  foldfile=out/'03_CV_HELD_TISSUE_DONOR_DISJOINT_PARTIAL.csv.gz'
  folds=pd.read_csv(foldfile) if args.resume and foldfile.exists() else pd.DataFrame()
  completed=set(folds.family_id.astype(str).unique()) if not folds.empty else set()
  for i,r in families.iterrows():
   fid=r.family_id;path=r.pathway
   if fid in completed:continue
   status=fits.loc[fits.family_id.eq(fid),'status']
   if status.empty or status.iloc[0]!='FIT':continue
   if path not in nameidx:continue
   cvrows=cv_family(orig,meta,matrix[:,nameidx[path]])
   for c in cvrows:c.update(family_id=fid,pathway=path,module_id=r.module_id)
   if cvrows:folds=pd.concat([folds,pd.DataFrame(cvrows)],ignore_index=True)
   # Mark CV-attempted even if there are no valid folds
   completed.add(fid)
   if not folds.empty:write(folds,foldfile)
   log('CV',i+1,'/',len(families),fid,'folds',len(cvrows))
  if not folds.empty:
   folds=folds.drop_duplicates(['family_id','held_tissue'])
   write(folds,out/'03_CV_HELD_TISSUE_DONOR_DISJOINT.csv.gz')
   summary=folds.groupby('family_id').agg(n_held_tissues=('held_tissue','nunique'),
     mean_deltaRMSE_M1=('delta_rmse_M1','mean'),median_deltaRMSE_M1=('delta_rmse_M1','median'),
     frac_held_better_M1=('delta_rmse_M1',lambda x:float(np.mean(x>0))),
     mean_deltaRMSE_M2=('delta_rmse_M2','mean'),median_deltaRMSE_M2=('delta_rmse_M2','median'),
     frac_held_better_M2=('delta_rmse_M2',lambda x:float(np.mean(x>0)))).reset_index()
   summary['passes_min_folds']=summary.n_held_tissues>=args.min_cv_folds
   write(summary,out/'03_CV_FAMILY_SUMMARY.csv')
  else:log('WARNING','No CV folds available')
  log('CV COMPLETE',len(folds),'fold records')
 if args.stage in ('all','report'):
  fitsfile=out/'02_LAYER2_LAYER3_FAMILY_FITS_69.csv'
  if not fitsfile.exists():raise FileNotFoundError('Run --stage fit first')
  fs=pd.read_csv(fitsfile)
  cvf=out/'03_CV_FAMILY_SUMMARY.csv';cv=pd.read_csv(cvf) if cvf.exists() else pd.DataFrame(columns=['family_id'])
  df=summarize_family(fs,cv,frozen,legacy).merge(cross,on='family_id',how='left')
  df['has_phase_fit']=df.status.eq('FIT')
  df['m1_internal_wild_q05']=df.has_phase_fit & df.wild_BHq_M1_69_families.le(.05)
  df['m2_internal_wild_q05']=df.has_phase_fit & df.wild_BHq_M2_69_families.le(.05)
  df['m1_cv_positive']=df.mean_deltaRMSE_M1.gt(0)&df.n_held_tissues.ge(args.min_cv_folds) if 'mean_deltaRMSE_M1' in df else False
  df['m2_cv_positive']=df.mean_deltaRMSE_M2.gt(0)&df.n_held_tissues.ge(args.min_cv_folds) if 'mean_deltaRMSE_M2' in df else False
  df['m1_evidence']='NOT_ESTIMABLE'
  df.loc[df.has_phase_fit,'m1_evidence']='MODEL_FIT_ONLY'
  df.loc[df.m1_internal_wild_q05,'m1_evidence']='INTERNAL_CONDITIONAL_SUPPORT'
  df.loc[df.m1_internal_wild_q05&df.m1_cv_positive,'m1_evidence']='INTERNAL_SUPPORT_PLUS_EXPLORATORY_CV'
  df['m2_evidence']='NOT_ESTIMABLE'
  df.loc[df.has_phase_fit,'m2_evidence']='MODEL_FIT_ONLY'
  df.loc[df.m2_internal_wild_q05,'m2_evidence']='INTERNAL_CONDITIONAL_SUPPORT'
  df.loc[df.m2_internal_wild_q05&df.m2_cv_positive,'m2_evidence']='INTERNAL_SUPPORT_PLUS_EXPLORATORY_CV'
  write(df,out/'04_INTEGRATED_FIG3_LAYER2_LAYER3_EVIDENCE.csv')
  if len(df)>1 and df.fig3_formal_median_beta.notna().sum()>5:
   from scipy.stats import spearmanr
   corr=[]
   for col in ['deltaR2_M1_vs_M0','deltaR2_M2_vs_M1']:
    tmp=df[['fig3_formal_median_beta',col]].dropna()
    if len(tmp)>5:
     rho,p=spearmanr(tmp.iloc[:,0],tmp.iloc[:,1]);corr.append(dict(phase_dim=col,n_families=len(tmp),spearman_rho=rho,naive_p_NOT_CONFIRMATORY=p))
   write(pd.DataFrame(corr),out/'04_DESCRIPTIVE_FIG3_VS_LAYER23_ASSOCIATION.csv')
  counts=df.groupby(['module_id','module_label'],dropna=False).agg(n_families=('family_id','nunique'),
    n_layer2_internal=('m1_internal_wild_q05','sum'),n_layer3_internal=('m2_internal_wild_q05','sum'),
    n_layer2_cv_positive=('m1_cv_positive','sum'),n_layer3_cv_positive=('m2_cv_positive','sum')).reset_index()
  write(counts,out/'04_BIOLOGICAL_MODULE_SUMMARY.csv')
  pr=out/'02_LAYER3_M2_LINEAGE_CONDITIONAL_PROFILES.csv'
  prof=pd.read_csv(pr) if pr.exists() else pd.DataFrame()
  figs=out/'FIGURES';figs.mkdir(exist_ok=True)
  for label in (True,False):graph(df,prof,figs,label)
  report=['# Fig3 frozen biology × inferred clock-state layers 2 & 3','',
   '## Frozen inputs',f'- Fig3 pathways: {len(frozen)}; redundancy families: {len(families)}',
   f'- Fig3 formal-hit table available: {formal is not None}',
   f'- Tabula units: {len(meta)}; M012 pathways: {len(legacy)}',
   '','## Layer 2: shared inferred-phase term after donor × tissue × assay × lineage baselines',
   f'- Families fitting model: {int(df.has_phase_fit.sum())} / {len(df)}',
   f'- Family representatives donor-wild BH q<0.05: {int(df.m1_internal_wild_q05.sum())}',
   f'- Of those, positive descriptive held-tissue CV: {int((df.m1_internal_wild_q05&df.m1_cv_positive).sum())}',
   '','## Layer 3: lineage × inferred-phase interaction conditional on Layer 2',
   f'- Donor-wild BH q<0.05: {int(df.m2_internal_wild_q05.sum())}',
   f'- Of those, positive descriptive held-tissue CV: {int((df.m2_internal_wild_q05&df.m2_cv_positive).sum())}',
   '','## Scientific restrictions',
   '- Fig3 pathways are post-selected from the SAME Tabula dataset; phase tests are internal, not independent confirmation.',
   '- Fig3 ontology biological modules are annotations, not independent composite pathway scores.',
   '- 69 families: one frozen representative pathway each; 98-pathway analysis is sensitivity, not 98 independent confirmations.',
   '- Mean ΔRMSE improvement alone is not a formal CV significance test (training overlap across folds).',
   '- B0 phase was inferred cohort-wise from clock expression; not observed time. Cross-cohort orientation may differ.',
   '- M1/M2 associations do not establish a causal effect of six-arm architecture on any pathway.',
   '- No fixed covariance-matched null comparison is re-performed in this bridge. Do not label pathway specificity proven.',
   '- Fig3 beta vs layer23 correlation is descriptive and selected-sample; naive P is not confirmatory.',
   '- An M2 term can encode amplitude and/or phase differences; do not infer phase differences when amplitude ~0.',
   '','## Files','- 04_INTEGRATED_FIG3_LAYER2_LAYER3_EVIDENCE.csv',
   '- 04_BIOLOGICAL_MODULE_SUMMARY.csv','- 02_LAYER3_M2_LINEAGE_CONDITIONAL_PROFILES.csv',
   '- 03_CV_FAMILY_SUMMARY.csv','- 01_FIG3_FROZEN_98_x_LEGACY_M012.csv']
  (out/'INTERPRETATION_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
  js(dict(status='COMPLETE_INTERNAL_BRIDGE_NOT_EXTERNAL',n_fig3_paths=len(frozen),n_families=len(families),
          n_fit=int(df.has_phase_fit.sum()),n_layer2_wild_q05=int(df.m1_internal_wild_q05.sum()),
          n_layer3_wild_q05=int(df.m2_internal_wild_q05.sum()),n_phase_units=len(meta),
          n_fig3_hit_rows=len(formal) if formal is not None else None),out/'00_STATUS.json')
  log('REPORT COMPLETE',out)

if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();sys.exit(1)
