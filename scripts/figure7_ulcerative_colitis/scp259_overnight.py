#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SCP259 (Smillie Cell 2019) disease clock architecture - frozen five lineages.

Reads original authors' matrix-market gene x cell counts, metadata and fine-cluster
-> broad-lineage mapping. Streams matrices WITHOUT dense single-cell conversion.
Primary unit: donor x biopsy x lineage, analysis unit: donor x state x lineage.
Keeps Healthy, UC noninflamed and UC inflamed distinct. Frozen Stage-70 six arms.
All association, TF and pathway results are observational, not circadian rhythms.

Usage: python scp259_overnight.py --phase all --root OUT --project-root katv2
      python scp259_overnight.py --phase download --root OUT --scp-command-file bulk_curl.sh
      python scp259_overnight.py --phase analyze --root OUT --hallmark-gmt h.all...gmt
      python scp259_overnight.py --phase selftest --root OUT
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, logging, math, os, re, shutil, subprocess, sys, traceback
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINEAGES=['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
ARM={
 'EBOX_POS':['ARNTL','CLOCK','NPAS2'],
 'EBOX_NEG':['PER1','PER2','PER3','CRY1','CRY2'],
 'RORE_ACT':['RORA','RORB','RORC'],
 'RORE_REP':['NR1D1','NR1D2'],
 'DBOX_ACT':['DBP','TEF','HLF'],
 'DBOX_REP':['NFIL3']}
CLOCK17=[x for vv in ARM.values() for x in vv]
ARMS=list(ARM)
STATES=['Healthy','UC_noninflamed','UC_inflamed']
COMPARE=[('UC_noninflamed','Healthy'),('UC_inflamed','Healthy'),('UC_inflamed','UC_noninflamed')]
COL={'Healthy':'#587f94','UC_noninflamed':'#d5a854','UC_inflamed':'#c9666d'}
LINCOL={'B cell':'#5485C6','T cell':'#CE5E6B','Macrophage/Monocyte':'#E1A041','Fibroblast':'#65B19C','Endothelial':'#8A65BC'}
TF_CANDIDATES='BACH2 PAX5 EBF1 IRF4 IRF8 PRDM1 XBP1 TBX21 CREM JUN FOS STAT1 STAT3 STAT5A STAT5B NFKB1 NFKB2 REL RELA RELB SPI1 KLF2 KLF4 ETS1 TCF3 TCF7 RUNX1 RUNX3 MYC ATF3 BATF BCL6 FOXP1 NFE2L2 HIF1A GATA3 SMAD3 RORA NR4A1'.split()
ALIASES={'BMAL1':'ARNTL','E4BP4':'NFIL3'}
AUTHOR_CLUSTER_MAP={
 'WNT2B+ Fos-hi':'Fibroblast','WNT2B+ Fos-lo 1':'Fibroblast','WNT2B+ Fos-lo 2':'Fibroblast',
 'RSPO3+':'Fibroblast','WNT5B+ 1':'Fibroblast','WNT5B+ 2':'Fibroblast',
 'Inflammatory Fibroblasts':'Fibroblast','Myofibroblasts':'Fibroblast',
 'Endothelial':'Endothelial','Microvascular':'Endothelial','Post-capillary Venules':'Endothelial',
 'Macrophages':'Macrophage/Monocyte','Inflammatory Monocytes':'Macrophage/Monocyte',
 'Cycling Monocytes':'Macrophage/Monocyte',
 'CD4+ Activated Fos-hi':'T cell','CD4+ Activated Fos-lo':'T cell',
 'CD4+ Memory':'T cell','Tregs':'T cell','CD4+ PD1+':'T cell','CD8+ IELs':'T cell',
 'CD8+ LP':'T cell','CD8+ IL17+':'T cell','MT-hi':'T cell','Cycling T':'T cell',
 'Plasma':'B cell','Follicular':'B cell','GC':'B cell','Cycling B':'B cell'}
# The manuscript's broad T_cells group also has NKs/ILCs and broad Myeloid has
# DC/mast. We purposefully exclude them rather than silently mixing lineages.
EXCLUDED=['Pericytes','DC1','DC2','CD69+ Mast','CD69- Mast','NKs','ILCs']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

def log(s):print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def norm(x):return ALIASES.get(str(x).strip().upper(),str(x).strip().upper())
def canon_health(x):
 s=str(x).strip().lower().replace('_','-').replace(' ','-')
 if s in ('healthy','control','normal','healthy-control','healthy-individual'):return 'Healthy'
 if s in ('non-inflamed','noninflamed','uninflamed','non-inflamed-uc','uc-noninflamed','uc-non-inflamed'):return 'UC_noninflamed'
 if s in ('inflamed','uc-inflamed','inflammation','inflammatory'):return 'UC_inflamed'
 return None

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(obj,p):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def gopen(p,mode='rt'):
 p=Path(p);return gzip.open(p,mode,encoding='utf-8') if p.suffix=='.gz' and 't' in mode else (gzip.open(p,mode) if p.suffix=='.gz' else open(p,mode,encoding='utf-8') if 't' in mode else open(p,mode))
def ffind(root,name):
 p=root/name
 if p.is_file():return p
 p=root/(name+'.gz')
 if p.is_file():return p
 matches=list(root.rglob(name))+list(root.rglob(name+'.gz'))
 matches=[x for x in matches if x.is_file()]
 if len(matches)==1:return matches[0]
 if len(matches)>1:raise RuntimeError(f'Ambiguous multiple copies of {name}: {matches}')
 return None

def blocked_files(raw,root):
 required=['all.meta2.txt','Imm.genes.tsv','Imm.barcodes2.tsv','gene_sorted-Imm.matrix.mtx',
           'Fib.genes.tsv','Fib.barcodes2.tsv','gene_sorted-Fib.matrix.mtx']
 missing=[x for x in required if ffind(raw,x) is None]
 if missing:
  msg=('SCP259 official processed count files are not present.\n\n'
       'Public authors list (https://github.com/cssmillie/ulcerative_colitis):\n'
       '\n'.join(f'  - {x}' for x in required)+'\n\nMissing:\n'+ '\n'.join(f'  - {x}' for x in missing)+
       '\n\nOpen https://singlecell.broadinstitute.org/single_cell/study/SCP259\n'
       'Sign in > Download > Bulk download > Get download command (Immune + Stromal + metadata).\n'
       'Run that official command from the directory: '+str(raw)+'\n'
       'Or save the supplied shell commands into a file and pass --scp-command-file FILE.\n'
       'The SCP bulk URL expires after about 30 minutes. Do not fabricate permanent URLs.\n'
       'We do NOT need Epi files or author training Seurat objects.\n')
  (root/'BLOCKED_DOWNLOAD.txt').write_text(msg,encoding='utf-8')
  return missing
 (root/'BLOCKED_DOWNLOAD.txt').unlink(missing_ok=True)
 return []

def load_gmt(paths):
 sets={}
 for p in paths:
  if not p:continue
  p=Path(p)
  if not p.exists():raise FileNotFoundError(f'GMT missing: {p}')
  with p.open() as f:
   for line in f:
    parts=line.strip().split('\t')
    if len(parts)<7:continue
    gn=parts[0]
    if gn.startswith('HALLMARK_') or gn.startswith('REACTOME_'):
     sets[gn]=list(dict.fromkeys(norm(x) for x in parts[2:]))
 return sets

def load_meta(raw):
 mp=ffind(raw,'all.meta2.txt')
 m=pd.read_csv(mp,sep='\t',index_col=0,low_memory=False)
 m.index=m.index.astype(str).str.strip()
 if not m.index.is_unique:raise ValueError('all.meta2.txt cell IDs are not unique')
 req=['Subject','Sample','Health','Cluster']
 absent=[x for x in req if x not in m]
 if absent:raise ValueError(f'Author metadata missing {absent}, available: {list(m.columns)}')
 m=m[req].copy()
 for c in req:m[c]=m[c].astype(str).str.strip()
 m['condition']=m.Health.map(canon_health)
 if m.condition.isna().any():
  raise ValueError('Unknown author Health categories: '+repr(m.loc[m.condition.isna(),'Health'].value_counts().to_dict()))
 m['lineage']=m.Cluster.map(AUTHOR_CLUSTER_MAP)
 if (m['lineage'].notna()).sum()<100:raise ValueError('Author Cluster labels could not be resolved (check column labels vs author paper)')
 if set(m[m.lineage.notna()].lineage.unique())!=set(LINEAGES):
  raise ValueError(f'All five lineages not present in author labels: {m.lineage.value_counts().to_dict()}')
 # A donor can be healthy OR UC, not both; within UC it can have both biopsy states.
 check=m.groupby('Subject').condition.apply(lambda s:set(s.unique()))
 for d,cs in check.items():
  if 'Healthy' in cs and len(cs)>1:raise ValueError(f'Donor {d} mixes healthy and UC states')
 # Each sample corresponds to one donor and one biopsy condition.
 for key in ('Subject','condition'):
  if m.groupby('Sample')[key].nunique().max()>1:
   raise ValueError(f'Sample is not uniquely associated with {key}')
 return m

def load_tsv_lines(path):
 with gopen(path,'rt') as f:return [line.rstrip('\r\n') for line in f]

def source_unit_table(m,compartment):
 # Only use samples with one of the five mapped lineages and matching compartment.
 selected=m[m.lineage.notna()].copy()
 # Mapping table for compartment, never guess from cell barcode.
 if compartment=='Imm':selected=selected[selected.lineage.isin(LINEAGES[:3])]
 else:selected=selected[selected.lineage.isin(LINEAGES[3:])]
 unit_cols=['Subject','Sample','condition','lineage']
 unit=selected[unit_cols].drop_duplicates().sort_values(unit_cols).reset_index(drop=True)
 unit['unit_id']=np.arange(len(unit),dtype=np.int32)
 return unit

def stream_compartment(raw,m,comp,out,genes_interest,chunk=400000,hash_matrices=False):
 dest=out/'checkpoints'/f'{comp}_pseudobulk.npz';info=out/'checkpoints'/f'{comp}_units.csv'
 sigfile=out/'checkpoints'/f'{comp}_source_signature.json'
 src_paths=[ffind(raw,f'{comp}.genes.tsv'),ffind(raw,f'{comp}.barcodes2.tsv'),ffind(raw,f'gene_sorted-{comp}.matrix.mtx'),ffind(raw,'all.meta2.txt')]
 signature={str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in src_paths}
 if dest.exists() and info.exists():
  old=json.loads(sigfile.read_text(encoding='utf-8')) if sigfile.exists() else None
  if old!=signature:
   log(f'{comp} source signature missing/changed: invalidating old checkpoint')
   dest.unlink(missing_ok=True);info.unlink(missing_ok=True);sigfile.unlink(missing_ok=True)
 if dest.exists() and info.exists():
  log(f'Resuming existing {comp} pseudobulk checkpoint')
  with np.load(dest,allow_pickle=False) as z:
   return pd.read_csv(info,dtype=str),list(z['genes'].astype(str)),z['counts'].astype(np.float64),z['library_counts'].astype(np.float64),z['n_cells'].astype(int),dict(source='checkpoint',compartment=comp)
 gene_path=ffind(raw,f'{comp}.genes.tsv');bar_path=ffind(raw,f'{comp}.barcodes2.tsv');matrix_path=ffind(raw,f'gene_sorted-{comp}.matrix.mtx')
 genes_raw=load_tsv_lines(gene_path);barcodes=load_tsv_lines(bar_path)
 if not pd.Index(barcodes).is_unique:raise ValueError(f'Duplicate barcodes in {comp}')
 # Author readLines() implies one symbol per line. Preserve exact names; some exports
 # have two columns (ENSG, symbol), in which case use the second symbol column.
 gene_symbols=[]
 for x in genes_raw:
  p=x.split('\t');gene_symbols.append(norm(p[1] if len(p)>1 and p[0].upper().startswith('ENSG') else p[0]))
 target=sorted(set(gene_symbols)&set(genes_interest))
 if len(set(CLOCK17)&set(target))<12:raise ValueError(f'Only {len(set(CLOCK17)&set(target))}/17 clock genes in {comp}; incompatible features')
 unit=source_unit_table(m,comp)
 lookup={(r.Subject,r.Sample,r.condition,r.lineage):int(r.unit_id) for r in unit.itertuples(index=False)}
 ix=m.index.get_indexer(pd.Index(barcodes))
 match=ix>=0
 if match.mean()<.97:raise ValueError(f'{comp} exact barcode match only {match.mean():.3%}; cannot positional join')
 cell_unit=np.full(len(barcodes),-1,np.int32)
 for j,idx in enumerate(ix):
  if idx<0:continue
  r=m.iloc[idx]
  if pd.isna(r.lineage):continue
  if comp=='Imm' and r.lineage not in LINEAGES[:3]:continue
  if comp=='Fib' and r.lineage not in LINEAGES[3:]:continue
  cell_unit[j]=lookup[(r.Subject,r.Sample,r.condition,r.lineage)]
 nunit=len(unit);n_cells=np.bincount(cell_unit[cell_unit>=0],minlength=nunit)
 gene_id={g:i for i,g in enumerate(target)}
 target_ix=np.array([gene_id.get(g,-1) for g in gene_symbols],np.int32)
 counts=np.zeros((nunit,len(target)),dtype=np.float64)
 library=np.zeros(nunit,dtype=np.float64)
 with gopen(matrix_path,'rt') as fh:
  while True:
   line=fh.readline()
   if not line:raise ValueError(f'Invalid {comp} MatrixMarket header')
   if not line.startswith('%'):break
  dims=[int(x) for x in line.strip().split()]
  if len(dims)!=3 or dims[:2]!=[len(genes_raw),len(barcodes)]:
   raise ValueError(f'{comp} mtx dimensions {dims} disagree genes={len(genes_raw)} barcodes={len(barcodes)}')
  nnz=dims[2];seen=0
  it=pd.read_csv(fh,sep=r'\s+',header=None,names=['g','c','v'],chunksize=chunk,
                 dtype={'g':'int32','c':'int32','v':'float64'})
  for k,z in enumerate(it,1):
   g=z.g.to_numpy(dtype=np.int64)-1;c=z.c.to_numpy(dtype=np.int64)-1;v=z.v.to_numpy(dtype=np.float64)
   if len(g)==0:continue
   if g.min()<0 or g.max()>=len(genes_raw) or c.min()<0 or c.max()>=len(barcodes):
    raise ValueError(f'Out of range coordinate in {comp} matrix')
   uid=cell_unit[c];good=uid>=0
   if good.any():
    library+=np.bincount(uid[good],weights=v[good],minlength=nunit)
    ti=target_ix[g];sel=good&(ti>=0)
    if sel.any():np.add.at(counts,(uid[sel],ti[sel]),v[sel])
   seen+=len(z)
   if k%30==0:log(f'{comp} MTX {seen:,}/{nnz:,} entries ({100*seen/nnz:.1f}%)')
 if seen!=nnz:raise ValueError(f'{comp} MatrixMarket entry count mismatch {seen} != {nnz}')
 if np.any((n_cells>0)&(library<=0)):raise ValueError(f'{comp} empty libraries for sampled units')
 dest.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest,genes=np.array(target,dtype=str),counts=counts,library_counts=library,n_cells=n_cells)
 unit.to_csv(info,index=False)
 dump(signature,sigfile)
 aud={'source':'original_SCP259','compartment':comp,'n_barcodes':len(barcodes),'matched_fraction':float(match.mean()),
      'selected_cells':int((cell_unit>=0).sum()),'n_units':nunit,'n_genes_original':len(genes_raw),'n_genes_target':len(target),
      'mtx_nonzeros':seen,'mtx_path':str(matrix_path),'mtx_size_bytes':matrix_path.stat().st_size,
      'mtx_sha256':sha(matrix_path) if hash_matrices else 'NOT_COMPUTED_ADD_--hash-matrices',
      'gene_sha256':sha(gene_path),'barcode_sha256':sha(bar_path)}
 dump(aud,out/'audit'/f'{comp}_source_audit.json')
 return unit,target,counts,library,n_cells,aud

def combine_compartments(comp_results,meta):
 common_genes=set.intersection(*(set(res[1]) for res in comp_results))
 missing_clock=set(CLOCK17)-common_genes
 if len(missing_clock)>2:
  raise ValueError(f'More than 2 CLOCK17 genes missing in at least one compartment: {sorted(missing_clock)}; cannot substitute zero expression')
 allgenes=sorted(common_genes)
 full=[];units=[]
 for unit,gn,C,L,N,_ in comp_results:
  src=pd.Index(gn).get_indexer(allgenes)
  if (src<0).any():raise RuntimeError('Common-gene join failed')
  x=C[:,src]
  z=unit.copy();z['n_cells']=N;z['library_counts']=L
  full.append(x);units.append(z)
 d=pd.concat(units,ignore_index=True);raw_counts=np.vstack(full)
 # The 2 compartments never contain same biological broad lineage by design.
 if d.duplicated(['Subject','Sample','condition','lineage']).any():
  raise RuntimeError('Duplicate biopsy/lineage units across compartments')
 lib=d.library_counts.to_numpy(float)
 if (lib<=0).any():
  positive=lib>0
  d=d.loc[positive].reset_index(drop=True);raw_counts=raw_counts[positive];lib=lib[positive]
 logcpm=np.log1p(raw_counts/np.maximum(lib[:,None],1)*1e6)
 d['unit_key']=[f'{a}|{b}|{c}|{e}' for a,b,c,e in zip(d.Subject,d.Sample,d.condition,d.lineage)]
 return d,allgenes,logcpm

def donor_average(unit,genes,expr,min_cells=20):
 # Each biopsy is a real sample; donor is replication unit. Multiple biopsies
 # from one donor-condition-lineage are averaged with EQUAL biopsy weights.
 yes=unit.n_cells.astype(int).to_numpy()>=min_cells
 u=unit.loc[yes].reset_index(drop=True);X=expr[yes]
 keys=['Subject','condition','lineage']
 rows=[];arrays=[]
 for kk,sub in u.groupby(keys,sort=True):
  idx=sub.index.to_numpy();arrays.append(X[idx].mean(0))
  rows.append(dict(zip(keys,kk),n_biopsies=len(sub),n_cells=int(sub.n_cells.astype(int).sum()),
                   samples=';'.join(sorted(sub.Sample.astype(str)))))
 return pd.DataFrame(rows),np.vstack(arrays) if arrays else np.empty((0,len(genes)))

def build_features(d,genes,X,sets,min_match=5,sd_floor=.25):
 gm={g:i for i,g in enumerate(genes)}
 healthy=d.condition.eq('Healthy').to_numpy()
 n_ref=int(healthy.sum())
 if n_ref<10:raise ValueError(f'Only {n_ref} healthy donor-lineage observations; no stable reference')
 mu=X[healthy].mean(0);sd=X[healthy].std(0,ddof=1)
 # Single fixed reference across *all* healthy lineages, preserving normal
 # lineage-specific profiles (NOT within-lineage healthy zeroing).
 sigma=np.maximum(sd,sd_floor)
 Z=(X-mu)/sigma
 scores={};coverage=[]
 for a,alist in ARM.items():
  used=[g for g in alist if g in gm]
  if len(used)!=len(alist):log(f'WARNING {a} missing genes: {sorted(set(alist)-set(used))}')
  if not used:raise ValueError(f'Arm {a} has zero genes')
  scores[a]=Z[:,[gm[g] for g in used]].mean(1)
  coverage.append({'feature':a,'n_present':len(used),'n_expected':len(alist),'genes':';'.join(used),'type':'six_arm'})
 for g in CLOCK17:
  if g in gm:scores['GENE_'+g]=Z[:,gm[g]]
 for tf in TF_CANDIDATES:
  if tf in gm and tf not in CLOCK17:scores['TF_'+tf]=Z[:,gm[tf]]
 for p,ss in sets.items():
  good=[x for x in ss if x in gm and x not in CLOCK17]
  if len(good)>=min_match:
   scores['PATH_'+p]=Z[:,[gm[g] for g in good]].mean(1)
  coverage.append({'feature':p,'n_present':len(good),'n_expected':len(ss),'genes':';'.join(good),
                   'type':'pathway','usable':len(good)>=min_match})
 feat=pd.concat([d.reset_index(drop=True),pd.DataFrame(scores)],axis=1)
 feat['ARM_GLOBAL']=feat[ARMS].mean(axis=1)
 for a in ARMS:feat['SHAPE_'+a]=feat[a]-feat['ARM_GLOBAL']
 feat['SIX_ARM_DISTANCE']=np.nan
 for li in LINEAGES:
  c=feat.lineage.eq(li);ref=feat[c&feat.condition.eq('Healthy')]
  if len(ref)<2:continue
  center=ref[ARMS].mean().to_numpy(float)
  mat=feat.loc[c,ARMS].to_numpy(float)
  feat.loc[c,'SIX_ARM_DISTANCE']=np.sqrt(((mat-center)**2).mean(1))
  # Control LOO to avoid systematic control advantage (centroid includes self).
  for i in ref.index:
   rr=ref[ref.index!=i]
   cc=rr[ARMS].mean().to_numpy(float)
   x=feat.loc[i,ARMS].to_numpy(float)
   feat.loc[i,'SIX_ARM_DISTANCE']=np.sqrt(np.mean((x-cc)**2))
 return feat,pd.DataFrame(coverage),pd.DataFrame({'gene':genes,'healthy_mean_logCPM':mu,'healthy_sd_logCPM':sd,'sd_used':sigma}),gm

def bh(p):
 p=np.asarray(p,float);v=np.isfinite(p);out=np.full(len(p),np.nan)
 if v.sum()==0:return out
 order=np.argsort(p[v]);ps=p[v][order];n=len(ps)
 qs=np.minimum.accumulate((ps*n/np.arange(1,n+1))[::-1])[::-1]
 oo=np.empty(n);oo[order]=np.clip(qs,0,1);out[v]=oo
 return out

def test_one(x,y,paired=False,seed=0,n_boot=1500):
 x=np.asarray(x,float);y=np.asarray(y,float)
 if len(x)<3 or len(y)<3:return np.nan,np.nan,np.nan,np.nan,'INSUFFICIENT_DONORS'
 delta=float(x.mean()-y.mean())
 if paired:
  if len(x)!=len(y):raise ValueError('Paired requires same length')
  try:p=float(wilcoxon(x-y,alternative='two-sided',zero_method='wilcox').pvalue) if np.any(x!=y) else 1.
  except ValueError:p=np.nan
 else:p=float(mannwhitneyu(x,y,alternative='two-sided').pvalue)
 rng=np.random.default_rng(seed)
 if paired:
  dd=x-y;ix=rng.integers(0,len(dd),(n_boot,len(dd)));bs=dd[ix].mean(1)
 else:
  ix=rng.integers(0,len(x),(n_boot,len(x)));iy=rng.integers(0,len(y),(n_boot,len(y)))
  bs=x[ix].mean(1)-y[iy].mean(1)
 ci=np.percentile(bs,[2.5,97.5])
 return delta,p,float(ci[0]),float(ci[1]),'OK'

def comparisons(f,features,seed=119,n_boot=1500):
 results=[]
 for cmp_idx,(a,b) in enumerate(COMPARE):
  for lin in LINEAGES:
   A=f[f.lineage.eq(lin)&f.condition.eq(a)].set_index('Subject')
   B=f[f.lineage.eq(lin)&f.condition.eq(b)].set_index('Subject')
   for k,feature in enumerate(features):
    if feature not in f:continue
    if (a,b)==('UC_inflamed','UC_noninflamed'):
     matched=A.index.intersection(B.index);x=A.loc[matched,feature].to_numpy(float);y=B.loc[matched,feature].to_numpy(float);paired=True
    else:x=A[feature].to_numpy(float);y=B[feature].to_numpy(float);paired=False
    mask=np.isfinite(x)&np.isfinite(y) if paired else None
    if paired:x=x[mask];y=y[mask]
    else:x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    delta,p,lo,hi,st=test_one(x,y,paired,seed+cmp_idx*10000+k*17+LINEAGES.index(lin),n_boot)
    results.append({'contrast':f'{a}__vs__{b}','lineage':lin,'feature':feature,'delta':delta,
                    'p':p,'ci_low':lo,'ci_high':hi,'n_a':len(x),'n_b':len(y),
                    'paired':paired,'status':st,'a':a,'b':b})
 r=pd.DataFrame(results)
 # One prespecified family = 5 lineages x 6 arms for each comparison. Separate
 # other families; never present exploratory TF or 50 pathways as prespecified.
 def kind(q):
  if q in ARMS:return 'ARM'
  if q.startswith('SHAPE_'):return 'SHAPE'
  if q.startswith('GENE_'):return 'GENE17'
  if q.startswith('TF_'):return 'TF_EXP'
  if q.startswith('PATH_'):return 'PATHWAY'
  return 'OTHER'
 r['family']=r.feature.map(kind);r['q_bh']=np.nan
 for (contrast,fam),ix in r.groupby(['contrast','family']).groups.items():
  r.loc[ix,'q_bh']=bh(r.loc[ix,'p'])
 return r

def tf_arm_links(f):
 results=[]
 cols=[c for c in f if c.startswith('TF_')]
 # Within-group Spearman only, without pooling healthy and UC to fabricate
 # correlations driven by group mean shifts. No TF causality claim.
 for condition in STATES:
  for lin in LINEAGES:
   g=f[(f.condition==condition)&(f.lineage==lin)]
   if len(g)<5:continue
   for tf in cols:
    for arm in ARMS:
     xx=g[tf].to_numpy(float);yy=g[arm].to_numpy(float)
     if len(np.unique(xx))<3 or len(np.unique(yy))<3:continue
     rho,p=spearmanr(xx,yy)
     results.append({'condition':condition,'lineage':lin,'TF':tf[3:],'arm':arm,
                     'spearman_rho':rho,'p':p,'n_donors':len(g)})
 out=pd.DataFrame(results)
 if len(out):out['q_bh']=bh(out.p)
 return out

def figure_saver(fig,out,stem):
 for ext in ['png','pdf','svg']:
  fig.savefig(out/f'{stem}.{ext}',dpi=250 if ext=='png' else None,bbox_inches='tight',pad_inches=.12)
 plt.close(fig)

def plot_all(f,tests,out):
 out.mkdir(parents=True,exist_ok=True)
 # 3 condition x 5 lineage six arm heatmap, identical z color scale.
 vmin=-2.25;vmax=2.25
 fig,ax=plt.subplots(1,3,figsize=(15,4),sharex=True,sharey=True,layout='constrained')
 for i,condition in enumerate(STATES):
  z=f[f.condition.eq(condition)].groupby('lineage')[ARMS].mean().reindex(LINEAGES).to_numpy(float)
  im=ax[i].imshow(z,vmin=vmin,vmax=vmax,cmap='RdBu_r',aspect='auto')
  ax[i].set_title(condition.replace('_',' '));ax[i].set_xticks(range(6),ARMS,rotation=45,ha='right',fontsize=8)
  ax[i].set_yticks(range(5),LINEAGES)
 fig.colorbar(im,ax=ax.ravel().tolist(),shrink=.8,label='Healthy-reference gene z; arm mean')
 figure_saver(fig,out,'FIG01_FIVE_LINEAGE_SIX_ARM_3STATES')
 # 5 lineages x 2 contrasts forest arm effects (healthy reference)
 subset=tests[(tests.family=='ARM')&tests.contrast.isin([f'{s}__vs__Healthy' for s in STATES[1:]])]
 fig,axs=plt.subplots(1,5,figsize=(17,5),sharey=True,layout='constrained')
 for ax,li in zip(axs,LINEAGES):
  x=subset[subset.lineage.eq(li)]
  for j,s in enumerate(STATES[1:]):
   z=x[x.contrast.eq(f'{s}__vs__Healthy')].set_index('feature').reindex(ARMS)
   y=np.arange(6)+(-.13 if j==0 else .13)
   ds=z.delta.to_numpy(float);lo=z.ci_low.to_numpy(float);hi=z.ci_high.to_numpy(float)
   for yy,d,l,h in zip(y,ds,lo,hi):
    if np.isfinite(d) and np.isfinite(l) and np.isfinite(h):
     ax.plot([l,h],[yy,yy],color=COL[s],lw=2);ax.scatter([d],[yy],s=25,color=COL[s],zorder=3)
  ax.axvline(0,color='#758695',lw=.7);ax.set_title(li,color=LINCOL[li]);ax.set_yticks(np.arange(6),ARMS)
  ax.invert_yaxis();ax.set_xlabel('Disease − Healthy score')
 axs[-1].plot([],[],color=COL['UC_noninflamed'],label='UC noninflamed')
 axs[-1].plot([],[],color=COL['UC_inflamed'],label='UC inflamed')
 fig.legend(*axs[-1].get_legend_handles_labels(),loc='upper right',frameon=False)
 figure_saver(fig,out,'FIG02_FIVE_LINEAGE_DISEASE_ARM_EFFECTS')
 # B cell donor dots, no subtype explorations
 b=f[f.lineage.eq('B cell')]
 fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
 for aa,arm in zip(axs.flat,ARMS):
  for k,cond in enumerate(STATES):
   xx=b[b.condition.eq(cond)][arm].to_numpy(float)
   rng=np.random.default_rng(2026+k)
   aa.scatter(np.full(len(xx),k)+rng.uniform(-.12,.12,len(xx)),xx,color=COL[cond],s=24,alpha=.85)
   if len(xx):aa.plot([k-.2,k+.2],[xx.mean()]*2,color=COL[cond],lw=2)
  aa.set_title(arm);aa.set_xticks(range(3),['Healthy','UC noninfl.','UC inflam.'],rotation=24,ha='right');aa.grid(axis='y',alpha=.1)
 figure_saver(fig,out,'FIG03_B_CELL_DONOR_SIX_ARMS')
 # Six arm distance with leave-one-out healthy controls
 fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
 for i,lin in enumerate(LINEAGES):
  for j,st in enumerate(STATES):
   z=f[(f.lineage.eq(lin))&(f.condition.eq(st))]['SIX_ARM_DISTANCE'].dropna().to_numpy()
   rng=np.random.default_rng(i*10+j)
   xx=i+(j-1)*.23+rng.uniform(-.055,.055,len(z))
   ax.scatter(xx,z,color=COL[st],s=20,alpha=.8)
   if len(z):ax.plot([i+(j-1)*.23-.08,i+(j-1)*.23+.08],[z.mean()]*2,color=COL[st],lw=2)
 ax.set_xticks(range(5),LINEAGES,rotation=15,ha='right');ax.set_ylabel('Six-arm RMS distance from healthy centroid\n(control uses leave-one-out)')
 for st in STATES:ax.scatter([],[],color=COL[st],label=st)
 ax.legend(frameon=False,fontsize=8,ncol=3,loc='upper right')
 figure_saver(fig,out,'FIG04_SIX_ARM_REFERENCE_DISTANCE')
 # Static 17 clock genes, 3 states
 gene=[f'GENE_{g}' for g in CLOCK17 if f'GENE_{g}' in f]
 fig,axs=plt.subplots(1,3,figsize=(18,4.5),sharex=True,sharey=True,layout='constrained')
 for i,st in enumerate(STATES):
  xx=f[f.condition.eq(st)].groupby('lineage')[gene].mean().reindex(LINEAGES).to_numpy(float)
  im=axs[i].imshow(xx,cmap='RdBu_r',vmin=-2.25,vmax=2.25,aspect='auto');axs[i].set_title(st)
  axs[i].set_xticks(range(len(gene)),[g[5:] for g in gene],rotation=65,ha='right',fontsize=7)
  axs[i].set_yticks(range(5),LINEAGES)
 fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7,label='Healthy-reference logCPM z')
 figure_saver(fig,out,'FIG05_CLOCK17_STATIC_3STATES')

def reference_diagnostic(f,p,out):
 if not p:return 'NOT_RUN_NO_FROZEN_TABULA_REFERENCE'
 r=pd.read_csv(p)
 required={'lineage','arm','value'}
 if not required.issubset(r):return 'BLOCKED_REFERENCE_SCHEMA_REQUIRE_lineage_arm_value'
 if r.duplicated(['lineage','arm']).any():return 'BLOCKED_REFERENCE_DUPLICATES'
 healthy=f[f.condition.eq('Healthy')].groupby('lineage')[ARMS].mean().reindex(LINEAGES)
 h=healthy.sub(healthy.mean(axis=1),axis=0)
 rows=[]
 for li in LINEAGES:
  sub=r[r.lineage.eq(li)].set_index('arm').reindex(ARMS)
  if not sub.value.notna().all():continue
  x=h.loc[li].to_numpy(float);y=sub.value.astype(float).to_numpy()
  rho,pv=spearmanr(x,y)
  rows.append({'lineage':li,'healthy_fingerprint_vs_tabula_spearman':rho,'p_descriptive_only':pv,
               'metric_note':'RELATIVE_ARM_RANKS_ONLY; unknown cross-study normalization; not a formal replication'})
 pd.DataFrame(rows).to_csv(out/'06_OPTIONAL_TABULA_REFERENCE_RANK_CONCORDANCE.csv',index=False)
 return 'DESCRIPTIVE_RANKS_ONLY' if len(rows)==5 else 'PARTIAL_REFERENCE'

def find_gmts(a,project_root):
 files=[]
 for name in ('hallmark_gmt','reactome_gmt'):
  z=getattr(a,name)
  if z:files.append(Path(z).expanduser().resolve())
 if not files:
  for pattern in ('h.all.*.Hs.symbols.gmt','h.all*.gmt'):
   t=list(project_root.glob(pattern))
   if t:files.append(sorted(t)[-1]);break
 return files

def report(out,f,r,other,reference_status):
 arms=r[r.family.eq('ARM')].sort_values('p')
 pth=r[r.family.eq('PATHWAY')]
 status={
  'status':'ANALYSIS_COMPLETE','study':'SCP259 Smillie et al Cell 2019','donor_units':int(f.Subject.nunique()),
  'donor_condition_lineage_units':len(f),'counts_by_condition_lineage':f.groupby(['condition','lineage']).Subject.nunique().to_dict(),
  'arm_tests':int(arms.p.notna().sum()),'arm_significant_BH_q_lt_0p05':int((arms.q_bh<.05).sum()),
  'pathway_tests':int(pth.p.notna().sum()),'tabula_reference_status':reference_status,
  'note':'STATIC architecture only. Healthy is not inferred ZT. Biological donor is unit, nested biopsies averaged.'}
 dump({**{k:v for k,v in status.items() if k!='counts_by_condition_lineage'},
        'counts_by_condition_lineage':[{'condition':c,'lineage':l,'n_donors':int(n)} for (c,l),n in status['counts_by_condition_lineage'].items()]},out/'RUN_STATUS.json')
 lines=['# SCP259 disease-associated static clock architecture','','**ANALYSIS_COMPLETE** = completed computation, not validated disease reprogramming.','',
        '## Design','- Five predefined broad lineages, source Smillie author Cluster labels; no new reclustering or subtype study.',
        '- Independent units: donor; multiple biopsies within a donor×condition×lineage are averaged equally.',
        '- Three observed states: Healthy, UC noninflamed and UC inflamed. Within-UC inflamed/noninflamed uses matched donors only.',
        '- Frozen original Stage-70 17-gene six-arm map. Expression: pooled raw counts / all-gene library-size log1p(CPM), donor-averaged.',
        '- Healthy-reference gene standardization pooled across healthy *lineages* (preserves normal lineage architecture).',
        '- Six-arm tests: 5 × 6 BH for each contrast; effect with donor bootstrap 95% interval; no pseudo-replicated cell p-values.',
        '- TF expression and pathway analyses are exploratory; association is not TF activity or causal rewiring.','',
        f'## Coverage\n- Donors: {f.Subject.nunique()}; donor×condition×lineage units: {len(f)}',
        f'- 6-arm q<0.05: {(arms.q_bh<.05).sum()} / {arms.p.notna().sum()} evaluable',
        f'- Tabula frozen reference: {reference_status}', '',
        '## Leading nominal six-arm signals (NOT confirmed unless BH significant)']
 for rr in arms.head(12).itertuples(index=False):
  lines.append(f'- {rr.contrast}, {rr.lineage}, {rr.feature}: delta={rr.delta:.3f}, p={rr.p:.4g}, q={rr.q_bh:.4g}, donors={rr.n_a}/{rr.n_b}')
 lines+=['','## Interpretation bounds','- Cannot identify rhythmicity, phase, amplitude or biochemical feedback strength from this static dataset.',
         '- Do not interpret absence of FDR significance as evidence of unchanged clocks.',
         '- B-cell lineage intentionally pools author B categories; changes can reflect composition of states within this broad lineage.',
         '- Healthy-vs-UC comparison is unpaired. UC inflamed-vs-noninflamed uses donor matching only where actually available.',
         '- No TF→six-arm→pathway causal inference; downstream pathway scores are gene-set expression summaries.',
         '- If no explicit Tabula reference with defined comparable feature space: normal Tabula replication not established.',
         '- Check available clock-gene coverage in 02_FEATURE_COVERAGE.csv before any biological interpretation.',
         '', '## Files','- 01_BIOPSY_LINEAGE_PSEUDOBULK_METADATA.csv', '- 02_DONOR_CONDITION_LINEAGE_SCORES.csv',
         '- 03_SIX_ARM_AND_GENE_TESTS.csv', '- 04_ALL_TESTS_WITH_FDR.csv', '- 05_TF_ARM_WITHIN_CONDITION.csv',
         '- FIGURES/*.png/pdf/svg','']
 (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def analyze(a,root,raw):
 missing=blocked_files(raw,root)
 if missing:raise FileNotFoundError('SCP259 original files missing: '+', '.join(missing))
 out=root/'results';out.mkdir(parents=True,exist_ok=True)
 m=load_meta(raw)
 m.groupby(['condition','lineage'],dropna=False).size().rename('cells').reset_index().to_csv(out/'00_AUTHOR_CELL_LABEL_COVERAGE.csv',index=False)
 m.to_csv(root/'audit'/'AUTHOR_METADATA_USED.csv.gz',compression='gzip')
 gmts=find_gmts(a,Path(a.project_root).resolve())
 sets=load_gmt(gmts)
 dump({'source':'SCP259','gmt_paths':[str(x) for x in gmts],
       'n_gene_sets':len(sets),'TF_candidates':TF_CANDIDATES,'arms':ARM,
       'note':'GTEx/Tabula reference NOT bundled; missing reference not silently replaced with SCP healthy.'},root/'audit'/'FROZEN_ANALYSIS_DEFINITIONS.json')
 gene_interest=set(CLOCK17)|set(TF_CANDIDATES)|set().union(*[set(v) for v in sets.values()]) if sets else set(CLOCK17)|set(TF_CANDIDATES)
 cps=[]
 for comp in ['Imm','Fib']:
  log(f'Loading {comp} genes/barcodes then streaming full original MatrixMarket counts')
  cps.append(stream_compartment(raw,m,comp,root,gene_interest,a.chunk,a.hash_matrices))
 U,G,X=combine_compartments(cps,m)
 U.to_csv(out/'01_BIOPSY_LINEAGE_PSEUDOBULK_METADATA.csv',index=False)
 d,x=donor_average(U,G,X,min_cells=a.min_cells)
 if d.empty:raise RuntimeError('No donor-condition-lineage after minimum cell count')
 feat,coverage,ref,_=build_features(d,G,x,sets,min_match=a.min_pathway_genes)
 coverage.to_csv(out/'02_FEATURE_COVERAGE.csv',index=False)
 ref.to_csv(out/'02_HEALTHY_REFERENCE_GENE_PARAMETERS.csv',index=False)
 feat.to_csv(out/'02_DONOR_CONDITION_LINEAGE_SCORES.csv',index=False)
 features=ARMS+[f'SHAPE_{v}' for v in ARMS]+['ARM_GLOBAL','SIX_ARM_DISTANCE']
 features +=[c for c in feat if c.startswith(('GENE_','TF_','PATH_'))]
 tests=comparisons(feat,features,a.seed,a.bootstrap)
 tests.to_csv(out/'04_ALL_TESTS_WITH_FDR.csv',index=False)
 tests[tests.family.isin(['ARM','SHAPE','GENE17','OTHER'])].to_csv(out/'03_SIX_ARM_AND_GENE_TESTS.csv',index=False)
 tf_links=tf_arm_links(feat);tf_links.to_csv(out/'05_TF_ARM_WITHIN_CONDITION.csv',index=False)
 if not sets:(out/'PATHWAY_NOT_RUN.txt').write_text('Hallmark / Reactome GMT not supplied or found: no pathway claims.\n')
 tabstate=reference_diagnostic(feat,a.tabula_reference,out)
 plot_all(feat,tests,out/'FIGURES')
 report(out,feat,tests,{},tabstate)
 shutil.copy2(out/'RUN_STATUS.json',root/'RUN_STATUS.json')
 # Invariant audit: donor group sample count cannot be a single-cell pseudoreplicate.
 dump({'n_original_metadata_cells':len(m),'n_target_cells_metadata':int(m.lineage.notna().sum()),
       'n_target_donor_biopsies':len(U),'n_donor_condition_lineage':len(d),
       'n_donors':int(d.Subject.nunique()),'n_original_biopsies':int(m.Sample.nunique()),
       'n_data_gene_features':len(G),'n_pathway_sets':len(sets),'min_cell_count':a.min_cells,
       'normalization':'log1p(CPM of aggregated raw counts; all-gene library size); mean over biopsies to donor',
       'gene_z_reference':'pooled healthy donors+lineages fixed gene mean and sd; floor=0.25',
       'author_source':'https://github.com/cssmillie/ulcerative_colitis'},root/'audit'/'FINAL_DATA_AUDIT.json')
 log('SCP259 ANALYSIS COMPLETE: '+str(out/'REPORT.md'))

def selftest(a,root):
 # Minimal original-format fixture with deliberately designed healthy / inflamed
 # groups. Checks data loading, paired contrasts, original mapping, figures.
 raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(777)
 genes=CLOCK17+['CREM','STAT3','BACH2','IRF4','JUN','FOS','TNF','IL6','CXCL13','CD79A']
 clusters={'B cell':'Follicular','T cell':'CD4+ Memory','Macrophage/Monocyte':'Macrophages',
           'Fibroblast':'WNT5B+ 1','Endothelial':'Endothelial'}
 meta=[];comp_cols={'Imm':[],'Fib':[]};comp_vals={'Imm':[],'Fib':[]}
 for s in range(12):
  donor=f'D{s:02d}';cond=('Healthy' if s<6 else ('Inflamed' if s<9 else 'Non-inflamed'))
  for li in LINEAGES:
   comp='Imm' if li in LINEAGES[:3] else 'Fib'
   for j in range(25):
    bar=f'{donor}_{li.replace("/","_").replace(" ","")}_{j}'
    meta.append({'cell_id':bar,'Subject':donor,'Sample':f'{donor}_BIO1',
                 'Health':cond,'Cluster':clusters[li]})
    comp_cols[comp].append(bar)
    v=rng.poisson(2,size=len(genes)).astype(int)
    if cond=='Inflamed' and li=='B cell':v[0:3]+=2
    comp_vals[comp].append(v)
 pd.DataFrame(meta).set_index('cell_id').to_csv(raw/'all.meta2.txt',sep='\t')
 for comp in ['Imm','Fib']:
  (raw/f'{comp}.genes.tsv').write_text('\n'.join(genes)+'\n')
  (raw/f'{comp}.barcodes2.tsv').write_text('\n'.join(comp_cols[comp])+'\n')
  val=np.asarray(comp_vals[comp],dtype=int).T
  ii,jj=np.where(val>0)
  with open(raw/f'gene_sorted-{comp}.matrix.mtx','w') as fw:
   fw.write('%%MatrixMarket matrix coordinate integer general\n% synthetic test only\n')
   fw.write(f'{len(genes)} {val.shape[1]} {len(ii)}\n')
   for i,j in zip(ii,jj):fw.write(f'{i+1} {j+1} {val[i,j]}\n')
 log('Selftest mock created (not real UC data). Running analysis...')
 analyze(a,root,raw)
 (root/'SELFTEST_ONLY_NOT_BIOLOGICAL_DATA').write_text('Only code validation; no SCP259 real counts used.\n')

def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 ap.add_argument('--phase',choices=['all','download','analyze','selftest'],default='all')
 ap.add_argument('--root',default='TABULA_SCP259_DISEASE_OVERNIGHT')
 ap.add_argument('--project-root',default='.')
 ap.add_argument('--raw-dir',default=None,help='Directory containing SCP259 original all.meta2.txt and Imm/Fib 10x MTX files')
 ap.add_argument('--scp-command-file',help='Trusted shell script copied from portal official Bulk Download command; valid ~30 min')
 ap.add_argument('--hallmark-gmt',default=None);ap.add_argument('--reactome-gmt',default=None)
 ap.add_argument('--tabula-reference',default=None,help='Optional frozen lineage,arm,value CSV; relative-arm rank check only')
 ap.add_argument('--chunk',type=int,default=400000);ap.add_argument('--min-cells',type=int,default=20)
 ap.add_argument('--min-pathway-genes',type=int,default=5);ap.add_argument('--bootstrap',type=int,default=1500)
 ap.add_argument('--seed',type=int,default=119);ap.add_argument('--hash-matrices',action='store_true')
 args=ap.parse_args()
 root=Path(args.root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
 raw=Path(args.raw_dir).expanduser().resolve() if args.raw_dir else root/'raw'
 raw.mkdir(parents=True,exist_ok=True);(root/'audit').mkdir(exist_ok=True)
 try:
  if args.phase=='selftest':selftest(args,root);return 0
  if args.phase in ('all','download'):
   if args.scp_command_file:
    cp=Path(args.scp_command_file).expanduser().resolve()
    if not cp.is_file():raise FileNotFoundError(f'Official SCP curl shell command not found: {cp}')
    log(f'Running user-supplied official SCP bulk commands in {raw}')
    subprocess.run(['bash',str(cp)],cwd=raw,check=True)
   missing=blocked_files(raw,root)
   if missing:
    log(f'BLOCKED_DOWNLOAD: {len(missing)} required files missing. See {root/"BLOCKED_DOWNLOAD.txt"}')
    dump({'state':'BLOCKED_DOWNLOAD','missing':missing},root/'RUN_STATUS.json')
    return 2
   log('All SCP259 required source files are present')
   if args.phase=='download':return 0
  if args.phase in ('all','analyze'):analyze(args,root,raw)
  return 0
 except Exception as exc:
  log(f'BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}')
  dump({'state':'BLOCKED_OR_FAILED','error':repr(exc),'traceback':traceback.format_exc()},root/'RUN_STATUS.json')
  return 1
if __name__=='__main__':sys.exit(main())
