#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabula frozen-source biological figure expansion: seven themes, genuinely new analyses.
Inputs are frozen Stage62/64/78/79.2, Fig3/6, SCP259; no raw data refitting.
Must not infer observed circadian rhythms from inferred B0 phase. 2026-10-20.
"""
from __future__ import annotations
import argparse, json, os, sys, pathlib, hashlib, warnings
from collections import OrderedDict
import numpy as np, pandas as pd
import scipy.stats as st
import scipy.special as special
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mplcolors
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT=pathlib.Path(__file__).resolve().parent
LINS=['b_cell','t_cell','macrophage_monocyte','fibroblast','endothelial']
DISPLAY=['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
TO_DISPLAY=dict(zip(LINS,DISPLAY)); TO_SHORT=dict(zip(LINS,['B','T','Mac','Fib','Endo']))
TO_INTERNAL=dict(zip(DISPLAY,LINS))
COLORS=dict(zip(LINS,['#4F87B2','#C36C86','#E6AA59','#6EA994','#8B79B9']))
HOUR_CMAP='twilight_shifted'
C_BLUE='#4F87B2';C_RED='#CA6F79';C_GOLD='#DAAC62';C_GREEN='#75A994'
ARMS=['EBOX_POS','EBOX_NEG','RORE_ACT','RORE_REP','DBOX_ACT','DBOX_REP']
ARM_LABEL={'EBOX_POS':'E-box +','EBOX_NEG':'E-box −','RORE_ACT':'RORE +','RORE_REP':'RORE −','DBOX_ACT':'D-box +','DBOX_REP':'D-box −'}
GENES={'EBOX_POS':['ARNTL','CLOCK','NPAS2'],'EBOX_NEG':['PER1','PER2','PER3','CRY1','CRY2'],'RORE_ACT':['RORA','RORB','RORC'],'RORE_REP':['NR1D1','NR1D2'],'DBOX_ACT':['DBP','TEF','HLF'],'DBOX_REP':['NFIL3']}
CLOCK17=sum(GENES.values(),[])
CONDITIONS=['Healthy','UC_noninflamed','UC_inflamed']
CONDC={'Healthy':'#568AA9','UC_noninflamed':'#E4B563','UC_inflamed':'#CA6F79'}
# Curated before seeing results; fixed cell-function terms for UC, do not choose by p-value.
UC_FUNC=OrderedDict([
('b_cell',['HALLMARK_E2F_TARGETS','HALLMARK_G2M_CHECKPOINT','HALLMARK_MITOTIC_SPINDLE','HALLMARK_IL2_STAT5_SIGNALING','HALLMARK_TNFA_SIGNALING_VIA_NFKB','HALLMARK_INTERFERON_GAMMA_RESPONSE']),
('t_cell',['HALLMARK_IL2_STAT5_SIGNALING','HALLMARK_G2M_CHECKPOINT','HALLMARK_INTERFERON_GAMMA_RESPONSE','HALLMARK_TNFA_SIGNALING_VIA_NFKB','HALLMARK_APOPTOSIS','HALLMARK_HYPOXIA']),
('macrophage_monocyte',['HALLMARK_INFLAMMATORY_RESPONSE','HALLMARK_TNFA_SIGNALING_VIA_NFKB','HALLMARK_IL6_JAK_STAT3_SIGNALING','HALLMARK_COMPLEMENT','HALLMARK_HYPOXIA','HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY']),
('fibroblast',['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','HALLMARK_TGF_BETA_SIGNALING','HALLMARK_ANGIOGENESIS','HALLMARK_UNFOLDED_PROTEIN_RESPONSE','HALLMARK_MTORC1_SIGNALING','HALLMARK_HYPOXIA']),
('endothelial',['HALLMARK_ANGIOGENESIS','HALLMARK_COAGULATION','HALLMARK_HYPOXIA','HALLMARK_TNFA_SIGNALING_VIA_NFKB','HALLMARK_IL6_JAK_STAT3_SIGNALING','HALLMARK_COMPLEMENT']),
])
MODULE_FOCUS=['M01','M03','M04','M06','M07','M08','M09','M10','M11','M12']
TISSUES=['blood','bone marrow','colon','lung','adipose tissue','kidney','skin of abdomen','lymph node','liver','thymus']
SOURCES={
 'obs62':'corrected_full_analysis_v2/62_STATIC_3LOOP_ARCHITECTURE_20260930/source_data/STATIC_3LOOP_OBSERVATION_SCORES.csv',
 'gene62':'corrected_full_analysis_v2/62_STATIC_3LOOP_ARCHITECTURE_20260930/source_data/Fig_A_loop_ordered_clock17_heatmap_source.csv',
 'tissue62':'corrected_full_analysis_v2/62_STATIC_3LOOP_ARCHITECTURE_20260930/06_TISSUE_STABILITY/STATIC_3LOOP_LINEAGE_TISSUE_ARCHITECTURE.csv',
 'scores64':'corrected_full_analysis_v2/64_RESIDUALIZED_COUPLING_MATCHED_CONTRASTS_20260930/source_data/DONOR_TISSUE_LINEAGE_SCORES.csv',
 'ref78':'corrected_full_analysis_v2/78_LINEAGE_INTRINSIC_3LOOP_VALIDATION/01_TABLES/Tabula_TISSUE_RESIDUAL_LINEAGE_PROTOTYPE.csv',
 'net79':'corrected_full_analysis_v2/79_FIX2_STRICT_INTERACTION_AND_TF_20261004/00_INPUTS/79_FIX2_DOROTHEA_HS_ABC.csv',
 'meta79':'corrected_full_analysis_v2/79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_20261004/01_CACHE/79_2_aggregated_metadata.csv',
 'expr79':'corrected_full_analysis_v2/79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_20261004/01_CACHE/79_2_aggregated_expression.npz',
 'activity79':'corrected_full_analysis_v2/79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_20261004/01_CACHE/79_2_activity_clock17_excluded.npz',
 'arm79':'corrected_full_analysis_v2/79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_20261004/01_CACHE/79_2_aligned_sixarm.csv',
 'assoc79':'corrected_full_analysis_v2/79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_20261004/02_TABLES/79_2_ACTIVITY_CLOCK17_EXCLUDED_WITHIN_LINEAGE_ASSOCIATION.csv',
 'priority80':'corrected_full_analysis_v2/80_FIX3_LINEAGE_SPECIFIC_EVIDENCE_STRENGTH_20261005/01_TABLES/80_FIX3_TOP8_PER_LINEAGE.csv',
 'fig3set':'katv2/FIG3_FIG6_LAYER2_LAYER3_BRIDGE_W999/00_FROZEN_FIG3_98_PATHWAYS.csv',
 'fig3family':'katv2/FIG3_FIG6_LAYER2_LAYER3_BRIDGE_W999/00_FROZEN_FIG3_69_REPRESENTATIVES.csv',
 'fig3evid':'katv2/FIG3_FIG6_LAYER2_LAYER3_BRIDGE_W999/04_INTEGRATED_FIG3_LAYER2_LAYER3_EVIDENCE.csv',
 'fig6meta':'katv2/TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1/01_ATLAS_UNIT_METADATA.csv.gz',
 'fig6npz':'katv2/TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1/01_ALL_PATHWAY_SCORES_WIDE.npz',
 'fig6phase':'katv2/FIG3_FIG6_LAYER2_LAYER3_BRIDGE_W999/02_LAYER3_M2_LINEAGE_CONDITIONAL_PROFILES.csv',
 'fig6tissue':'katv2/FIG3_LAYER23_FAMILY_RECOVERY_BMAC_STABILITY_V2/03_WITHIN_TISSUE_ASSAY.csv',
 'uc':'katv2/TABULA_SCP259_DISEASE_OVERNIGHT/results/02_DONOR_CONDITION_LINEAGE_SCORES.csv',
 'uctests':'katv2/TABULA_SCP259_DISEASE_OVERNIGHT/results/04_ALL_TESTS_WITH_FDR.csv'
}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12,'axes.labelsize':10.5,'figure.titlesize':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':240,'axes.linewidth':.75,'xtick.major.size':3,'ytick.major.size':3})

class Pipeline:
 def __init__(self,args):
  self.a=args;self.root=pathlib.Path(args.project_root).resolve();self.out=pathlib.Path(args.outdir).resolve();
  self.pan=self.out/'PANELS';self.tables=self.out/'TABLES';self.logs=self.out/'LOGS';
  for p in (self.pan,self.tables,self.logs):p.mkdir(parents=True,exist_ok=True)
  self.logfile=self.logs/'PROGRESS.log';self.logfile.write_text('START\n',encoding='utf-8')
  self.files={k:self.root/path for k,path in SOURCES.items()};self.man=[]
  miss=[f'{k}: {v}' for k,v in self.files.items() if not v.is_file()]
  if miss:raise FileNotFoundError('Missing frozen source files:\n'+'\n'.join(miss)+'\nProvide --project-root as the parent directory containing katv2 and corrected_full_analysis_v2.')
  self.write_json('SOURCE_PATHS_AND_SIZES.json',{k:{'path':str(p),'bytes':p.stat().st_size} for k,p in self.files.items()})
 def log(self,*s):
  msg='[BIO-V3] '+' '.join(str(x) for x in s);print(msg,flush=True)
  with self.logfile.open('a',encoding='utf-8') as f:f.write(msg+'\n')
 def csv(self,df,name):
  df.to_csv(self.tables/name,index=False,encoding='utf-8-sig');self.log('TABLE',name,'rows=',len(df))
 def write_json(self,name,obj):
  (self.logs/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 def chart(self,stem,make):
  for text in (True,False):
   fig=make(text)
   if not text:
    # retain figure content and geometry, remove only text-bearing elements
    for t in list(fig.texts):t.set_visible(False)
    for ax in fig.axes:
     ax.title.set_visible(False);ax.xaxis.label.set_visible(False);ax.yaxis.label.set_visible(False)
     for t in list(ax.texts):t.set_visible(False)
     for tick in list(ax.get_xticklabels())+list(ax.get_yticklabels()):tick.set_visible(False)
     if ax.get_legend() is not None:ax.get_legend().set_visible(False)
     # colorbar standalone axis tick labels handled above
   suff='text' if text else 'notext'
   for ext in ('png','svg','pdf'):
    self.log('RENDER_START',stem,suff,ext)
    fig.savefig(self.pan/(stem+'_'+suff+'.'+ext),bbox_inches='tight',pad_inches=.13)
    self.log('RENDER_DONE',stem,suff,ext)
   plt.close(fig)
  self.man.append(stem);self.log('PANEL',stem,'text/notext PNG SVG PDF')
 def source(self,key):return pd.read_csv(self.files[key])

# plot helpers
def plot_matrix(ax,arr,xt,yt,lo=-1,hi=1,cmap='RdBu_r',annot=None,fmt=''):
 arr=np.asarray(arr,dtype=float)
 im=ax.imshow(arr,aspect='auto',cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
 ax.set_xticks(np.arange(arr.shape[1]),xt,rotation=55,ha='right')
 ax.set_yticks(np.arange(arr.shape[0]),yt)
 if annot is not None:
  for i in range(arr.shape[0]):
   for j in range(arr.shape[1]):
    v=annot[i,j]
    if v is not None and str(v).strip(): ax.text(j,i,str(v),ha='center',va='center',fontsize=8,color='#292D30')
 return im

def cbar(fig,im,ax,label):
 cb=fig.colorbar(im,ax=ax,shrink=.77,aspect=25,pad=.025);cb.ax.set_ylabel(label,rotation=90)
 return cb

def ylabel(path):
 x=path.replace('HALLMARK_','').replace('REACTOME_','').replace('_',' ').title()
 x=x.replace('Tnfa','TNFα').replace('Nfkb','NFκB').replace('Il6','IL6').replace('Jak','JAK').replace('Stat','STAT').replace('G2M','G2/M')
 return x

def bh(x):
 v=np.asarray(x,dtype=float);q=np.full(len(v),np.nan);m=np.isfinite(v)
 if m.any():q[m]=multipletests(v[m],method='fdr_bh')[1]
 return q

def paired_key(df,lin_col='lineage',don='donor_id'):
 return df.assign(_key=df[don].astype(str)+'||'+df['tissue'].astype(str)+'||'+df[lin_col].astype(str))

def safe_float(s):return pd.to_numeric(s,errors='coerce')

# 1. Tissue microenvironments + lineage architecture, *Stage64*.
def analyze_tissue(P):
 d=P.source('scores64');P.log('Stage64 rows',len(d))
 d['ARM_GLOBAL']=d[ARMS].mean(axis=1)
 for a in ARMS:d[f'SHAPE_{a}']=d[a]-d['ARM_GLOBAL']
 rows=[]
 for (t,l),z in d.groupby(['tissue','priority_lineage']):
  for a in ARMS:
   rows.append({'tissue':t,'lineage':l,'arm':a,'mean_raw':z[a].mean(),'mean_shape':z[f'SHAPE_{a}'].mean(),'n_donors':z.donor_id.nunique()})
 out=pd.DataFrame(rows);P.csv(out,'01_TISSUE_BY_LINEAGE_SIX_ARM_REAL.csv')
 # pick tissues on coverage first; not on effect sizes, avoid cherry-picking
 cover=d.groupby('tissue').agg(n_donors=('donor_id','nunique'),n_lineages=('priority_lineage','nunique'),n_units=('donor_id','size')).reset_index()
 cover['curated']=cover.tissue.isin(TISSUES);cover=cover.sort_values(['curated','n_lineages','n_donors'],ascending=False)
 select=cover[(cover.curated)&(cover.n_lineages>=4)&(cover.n_donors>=3)].tissue.tolist()[:7]
 if len(select)<4:
  select=cover[(cover.n_lineages>=4)&(cover.n_donors>=3)].tissue.tolist()[:7]
 P.csv(cover,'01_TISSUE_COVERAGE_AND_SELECTION.csv');P.log('selected tissues',select)
 def make(text):
  fig,axs=plt.subplots(len(select),1,figsize=(11,1.85*len(select)),sharex=True,layout='constrained')
  for i,(ax,t) in enumerate(zip(np.atleast_1d(axs),select)):
   mat=out[(out.tissue==t)].pivot(index='lineage',columns='arm',values='mean_shape').reindex(LINS)[ARMS].to_numpy()
   im=plot_matrix(ax,mat,ARMS,[TO_DISPLAY[l] for l in LINS],lo=-.65,hi=.65)
   ax.set_ylabel(t,fontsize=10,rotation=0,labelpad=65,ha='right',va='center')
   ax.set_xticklabels([] if i<len(select)-1 else [ARM_LABEL[x] for x in ARMS])
   for i2,l in enumerate(LINS):
    nd=d[(d.tissue==t)&(d.priority_lineage==l)].donor_id.nunique()
    if text and nd:ax.text(5.8,i2,f'n={nd}',va='center',ha='left',fontsize=8)
  if text:
   fig.suptitle('S2-new | Tissue-context-specific six-arm relative architecture (Tabula Stage64)')
   cbar(fig,im,axs,'Mean arm score minus six-arm mean')
  return fig
 P.chart('S2_TISSUE_MICROENVIRONMENT_SIX_ARM',make)
 def make2(text):
  z=out[out.tissue.isin(select)&(out.n_donors>=2)]
  fig,axes=plt.subplots(1,5,figsize=(17,4.6),layout='constrained',sharey=True)
  for ax,l in zip(axes,LINS):
   s=z[z.lineage==l].pivot(index='tissue',columns='arm',values='mean_shape').reindex(index=select,columns=ARMS)
   im=plot_matrix(ax,s, [ARM_LABEL[x] for x in ARMS],select,lo=-.65,hi=.65)
   ax.set_title(TO_DISPLAY[l],color=COLORS[l],fontweight='bold')
   if ax is not axes[0]:ax.set_yticklabels([])
  if text:fig.suptitle('S2-new | Same functional tissue context, distinct lineage-specific arm configurations');cbar(fig,im,axes,'Within-lineage relative arm score')
  return fig
 P.chart('S2_TISSUE_LINEAGE_RESPONSE_FACETS',make2)
 return d,out

# 2. CLOCK17 gene-level decomposition using frozen Stage62 source.
def analyze_clock17(P,arm64):
 obs=P.source('obs62');obs=obs[obs.priority_lineage.isin(LINS)].copy()
 # observed gene-score matrix: aggregate equally over cell_types within donor-tissue-lineage (avoid pseudoreplication)
 keys=['donor_id','tissue','priority_lineage']; zcols=['Z_'+g for g in CLOCK17]
 ag=obs.groupby(keys,as_index=False)[zcols].mean()
 joined=arm64.merge(ag,on=keys,validate='1:1',how='left',indicator=True)
 P.log('Stage62 gene join',joined._merge.value_counts().to_dict());P.csv(joined.drop(columns=['_merge']),'02_STAGE64_CLOCK17_DONOR_TISSUE_JOIN.csv')
 # directly check algebraic agreement between arm and mean component z after aggregation
 comparison=[]
 for a,gs in GENES.items():
  diff=joined[a]-joined[['Z_'+g for g in gs]].mean(axis=1)
  comparison.append({'arm':a,'mae':diff.abs().mean(),'pearson_r':joined[a].corr(joined[['Z_'+g for g in gs]].mean(axis=1))})
 P.csv(pd.DataFrame(comparison),'02_CLOCK17_ARM_DECOMPOSITION_CHECK.csv')
 # frozen original gene expression mapped to 5 lineages, center per lineage for contrasts
 prof=joined.groupby('priority_lineage')[zcols].mean().reindex(LINS)
 prof.columns=CLOCK17
 prof_long=prof.stack().reset_index(name='mean_gene_z').rename(columns={'priority_lineage':'lineage','level_1':'gene'})
 P.csv(prof_long,'02_TABULA_CLOCK17_5LINEAGE_GENE_PROFILE.csv')
 comp=[]
 for l in LINS:
  for a,gs in GENES.items():
   for g in gs:
    value=prof.loc[l,g];arm_val=prof.loc[l,gs].mean()
    comp.append({'lineage':l,'arm':a,'gene':g,'gene_z':value,'arm_gene_mean':arm_val,'gene_minus_arm_mean':value-arm_val,'n_genes_arm':len(gs)})
 P.csv(pd.DataFrame(comp),'02_ALL_17GENE_TO_6ARM_COMPONENTS.csv')
 def make(text):
  fig,ax=plt.subplots(figsize=(13.7,4.7),layout='constrained')
  im=plot_matrix(ax,prof[CLOCK17].to_numpy(),CLOCK17,[TO_DISPLAY[x] for x in LINS],lo=-1.35,hi=1.35)
  j=0
  for arm,gs in GENES.items():
   ax.add_patch(Rectangle((j-.5,-.5),len(gs),len(LINS),edgecolor='#5F6970',linewidth=1.35,fill=False))
   if text:ax.text(j+(len(gs)-1)/2,-1.05,ARM_LABEL[arm],ha='center',fontsize=10)
   j+=len(gs)
  if text:ax.set_title('S4-new | All 17 clock genes underlying the frozen Tabula six-arm architecture');cbar(fig,im,ax,'Frozen Stage62 CLOCK17 gene z-score')
  return fig
 P.chart('S4_CLOCK17_ALL_SIX_ARMS_GENE_LEVEL',make)
 def make2(text):
  # gene contribution relative to the arm its belongs to, makes opposing gene effects visible
  dt=pd.DataFrame(comp)
  fig,axs=plt.subplots(1,5,figsize=(18,5.5),sharey=True,layout='constrained')
  for ax,l in zip(axs,LINS):
   s=dt[dt.lineage==l].set_index('gene').reindex(CLOCK17)
   vals=s.gene_minus_arm_mean.to_numpy()
   ax.barh(np.arange(len(vals)),vals,color=[C_BLUE if v>=0 else C_RED for v in vals],alpha=.82)
   ax.axvline(0,color='#A0A0A0',lw=.8)
   ax.set_yticks(np.arange(len(vals)),CLOCK17);ax.invert_yaxis();ax.set_title(TO_DISPLAY[l],color=COLORS[l],fontweight='bold')
   if ax is not axs[0]:ax.set_yticklabels([])
  if text:fig.suptitle('S4-new | Which genes lead versus lag their own six-arm component?');axs[2].set_xlabel('Gene z-score minus its frozen arm mean')
  return fig
 P.chart('S4_GENE_WITHIN_ARM_COMPONENT_DEVIATION',make2)
 return joined

# 3. Exact matched pathway and six-arm data, cluster-robust donor tests.
def build_pathway_join(P,arm64):
 meta=P.source('fig6meta');z=np.load(P.files['fig6npz'],allow_pickle=False)
 scores=z['scores'];names=z['pathway'].astype(str)
 if scores.shape[0]!=len(meta):raise AssertionError(f'Fig6 matrix/meta mismatch: {scores.shape[0]} vs {len(meta)}')
 keep=np.array([not (name.startswith('NULL_') or name.startswith('RANDOM_')) and (name.startswith('HALLMARK_') or name.startswith('REACTOME_')) for name in names])
 scores=scores[:,keep];names=names[keep];P.log('Fig6 real pathways',len(names),'phase bins',len(meta))
 # phase weights from posterior weight_sum; reduction across phase bins is NOT a clock-timed contrast.
 meta=meta.copy();meta['lineage']=meta.lineage.map(TO_INTERNAL)
 if meta.lineage.isna().any():raise AssertionError('Unmapped Fig6 lineages')
 keys=['donor','tissue','lineage']
 # aggregate within assay (weighted by posterior bin mass), then average across assays to avoid dominance
 ameta=meta.groupby(keys+['assay'],sort=True).agg(posterior_mass=('weight_sum','sum'),phase_bin_count=('phase_bin','size'),n_source_cells=('n_source_cells','max')).reset_index()
 S=pd.DataFrame(scores,columns=names);S[keys+['assay']]=meta[keys+['assay']].values
 S['_w']=meta.weight_sum.to_numpy(dtype=float)
 group=S.groupby(keys+['assay'],sort=True)
 weighted=(group[names.tolist()].apply(lambda v: pd.Series(np.average(v.to_numpy(),weights=S.loc[v.index,'_w'].to_numpy(),axis=0),index=v.columns)) if False else None)
 # faster vectorized weighted sums
 S[names]=S[names].multiply(S['_w'],axis=0)
 sums=S.groupby(keys+['assay'],sort=True)[list(names)+['_w']].sum().reset_index()
 sums[names]=sums[names].div(sums['_w'],axis=0)
 a=sums.groupby(keys,as_index=False)[list(names)].mean()
 # means across *assays*, not raw cells; matching lineages now in one naming scheme
 arm=arm64.rename(columns={'donor_id':'donor','priority_lineage':'lineage'}).copy()
 m=arm.merge(a,on=keys,how='inner',validate='one_to_one')
 P.csv(ameta,'03_FIG6_PHASE_BIN_TO_ASSAY_PROVENANCE.csv')
 P.csv(a[keys],'03_FIG6_AGGREGATED_MATCHABLE_KEYS.csv')
 P.log('MATCHED Stage64×Fig6',len(m),'donors',m.donor.nunique(),'tissues',m.tissue.nunique(),'lineages',m.lineage.nunique())
 P.csv(m,'03_MATCHED_SIXARM_AND_FUNCTION_PER_DONOR_TISSUE.csv')
 P.write_json('03_JOIN_METADATA.json',{'stage64':len(arm),'fig6_bins':len(meta),'fig6_donor_tissue_lineage':len(a),'matched_rows':len(m),'matched_donors':m.donor.nunique(),'matched_tissues':m.tissue.nunique(),'matched_per_lineage':m.lineage.value_counts().to_dict(),'reduction':'B0 bin scores weighted within assay by posterior weight_sum; then equal-weight assay means. No phase claims.', 'score_scale':'Fig6 gene z from log1p(CP10k), standardized within tissue assay before pathway averaging.'})
 return m,list(names)

def analyze_arm_pathway(P,m,pathnames):
 curated=[]
 fig3=P.source('fig3set')
 selected=fig3[(fig3.collection.isin(['Hallmark','Reactome']))&(fig3.pathway.isin(pathnames))]
 # fixed biological modules; restrict to representative pathways to minimize redundancy
 families=P.source('fig3family')
 chosen=families[families.pathway.isin(pathnames)].copy()
 chosen=chosen[chosen.module_id.astype(str).str[:3].isin(MODULE_FOCUS)]
 # user goal representative immune/prolif/metabolism/ECM/ER etc not choose by significance
 if len(chosen)<12:chosen=families[families.pathway.isin(pathnames)].head(30)
 chosen=chosen.drop_duplicates('pathway').sort_values(['module_id','pathway']).head(36)
 P.csv(chosen,'03_PREDEFINED_BIOLOGY_REPRESENTATIVE_MODULES.csv')
 m=m.copy();m['ARM_GLOBAL']=m[ARMS].mean(axis=1)
 for ar in ARMS:m['SHAPE_'+ar]=m[ar]-m['ARM_GLOBAL']
 rows=[]
 # Frisch–Waugh residualization avoids running 216 separate formula models.
 # The residualized regression is algebraically equivalent to y~x+FE,
 # followed by donor-cluster CR1 sandwich SE with t(df=n_donor-1).
 from scipy.linalg import qr as qr_pivot
 for path in chosen.pathway:
  columns=['donor','tissue','lineage',path]+['SHAPE_'+ar for ar in ARMS]
  xx=m[columns].dropna().copy()
  if len(xx)<80:continue
  groups=xx.donor.astype(str).to_numpy()
  G=len(np.unique(groups));N=len(xx)
  cov=pd.get_dummies(xx[['lineage','tissue','donor']].astype(str),drop_first=True,dtype=float)
  C=np.column_stack([np.ones(N),cov.to_numpy(dtype=float)])
  Q,R,piv=qr_pivot(C,mode='economic',pivoting=True)
  tol=max(C.shape)*np.finfo(float).eps*abs(R[0,0]);rank=int(np.sum(np.abs(np.diag(R))>tol))
  Q=Q[:,:rank]
  ys=xx[path].to_numpy(dtype=float);ys=(ys-np.mean(ys))/max(np.std(ys,ddof=1),1e-8)
  Y=ys-Q@(Q.T@ys)
  X0=xx[['SHAPE_'+ar for ar in ARMS]].to_numpy(dtype=float)
  sd=np.std(X0,axis=0,ddof=1);X0=(X0-np.mean(X0,axis=0))/np.maximum(sd,1e-8)
  X=X0-Q@(Q.T@X0)
  for ia,arm in enumerate(ARMS):
   vx=X[:,ia];den=float(vx@vx)
   if den < 1e-10:continue
   beta=float((vx@Y)/den);err=Y-beta*vx
   clusters=np.array([np.sum(vx[groups==g]*err[groups==g]) for g in np.unique(groups)])
   # small-cluster CR1 (K includes FE+one arm), using t reference with G-1 df
   correction=(G/(G-1))*((N-1)/max(N-rank-1,1))
   se=np.sqrt(correction*np.sum(clusters**2)/den**2)
   t=beta/se if se>0 else np.nan
   pv=2*st.t.sf(abs(t),G-1) if np.isfinite(t) else np.nan
   rows.append({'pathway':path,'arm':arm,'beta_std':beta,'SE_cluster':se,'p_donor_df':pv,'n_rows':N,'n_donors':G,'n_tissues':xx.tissue.nunique(),'n_FE_columns_rank':rank+1,'model':'pathway~shape_arm+C(lineage)+C(tissue)+C(donor);CR1 donor-cluster SE'})
 test=pd.DataFrame(rows)
 if len(test)==0:raise RuntimeError('No matched-pathway regression succeeded')
 test['BH_q_all_arm_pathway']=bh(test.p_donor_df)
 P.csv(test,'03_ARM_TO_HALLMARK_REACTOME_FE_DONOR_CLUSTER.csv')
 # supplementary 5 lineage × arm × functional module observational profiles: not independently inferential due sparse donors.
 desc=[]
 for l in LINS:
  x=m[m.lineage==l]
  for path in chosen.pathway:
   for ar in ARMS:
    z=x[['SHAPE_'+ar,path]].dropna()
    rho,p=(st.spearmanr(z.iloc[:,0],z.iloc[:,1]) if len(z)>=8 else (np.nan,np.nan))
    desc.append({'lineage':l,'pathway':path,'arm':ar,'rho_unadjusted_descriptive':rho,'n_matched_donor_tissue':len(z),'warning':'descriptive only; donor/tissue adjusted inference is pooled table'})
 desc=pd.DataFrame(desc);P.csv(desc,'03_FIVE_LINEAGE_DESCRIPTIVE_ARM_FUNCTION_COUPLING.csv')
 # summary broad modules across curated 69 with top n modules, one represent pathway per module
 chosen_mod=chosen.groupby('module_id',as_index=False).first()
 chosenpaths=chosen_mod.pathway.tolist()[:12]
 def make(text):
  show=test[test.pathway.isin(chosenpaths)].copy();show['s']=show.pathway.map({x:i for i,x in enumerate(chosenpaths)})
  mat=show.pivot(index='pathway',columns='arm',values='beta_std').reindex(index=chosenpaths,columns=ARMS)
  qmat=show.pivot(index='pathway',columns='arm',values='BH_q_all_arm_pathway').reindex(index=chosenpaths,columns=ARMS)
  fig,ax=plt.subplots(figsize=(11.5,max(5.8,.58*len(mat))),layout='constrained')
  atext=np.full(mat.shape,'',dtype=object)
  for i in range(mat.shape[0]):
   for j in range(mat.shape[1]):
    q=qmat.iat[i,j];atext[i,j]='**' if np.isfinite(q) and q<.01 else '*' if np.isfinite(q) and q<.05 else ''
  im=plot_matrix(ax,mat, [ARM_LABEL[a] for a in ARMS],[ylabel(x) for x in chosenpaths],lo=-.6,hi=.6,annot=atext if text else None)
  if text:ax.set_title('S3-new | Six-arm shape versus pathway programs within matched Tabula samples\nDonor/tissue/lineage fixed effects; donor-cluster SE');cbar(fig,im,ax,'Standardized partial regression coefficient')
  return fig
 P.chart('S3_TABULA_ARM_FUNCTION_ADJUSTED_LINKS',make)
 # faceted by lineage - descriptives not inferential
 func=[p for p in chosenpaths][:7]
 def make2(text):
  fig,axs=plt.subplots(1,5,figsize=(20,7.1),layout='constrained')
  for ax,l in zip(axs,LINS):
   z=desc[(desc.lineage==l)&(desc.pathway.isin(func))]
   mat=z.pivot(index='pathway',columns='arm',values='rho_unadjusted_descriptive').reindex(index=func,columns=ARMS)
   im=plot_matrix(ax,mat,[ARM_LABEL[a] for a in ARMS],[ylabel(p) for p in func],lo=-.75,hi=.75)
   ax.set_title(f'{TO_DISPLAY[l]} (n={len(m[m.lineage==l])})',color=COLORS[l],fontweight='bold')
   if ax is not axs[0]:ax.set_yticklabels([])
  if text:fig.suptitle('S3-new | Five lineage views of clock-arm–functional associations (descriptive)');cbar(fig,im,axs,'Descriptive Spearman ρ (not FDR-tested)')
  return fig
 P.chart('S3_FIVE_LINEAGE_ARM_FUNCTION_MAP',make2)
 return m,test

# 4. TF regulon real-target pathways, using original signed DoRothEA and frozen Stage80.
def analyze_tf(P):
 net=P.source('net79');pri=P.source('priority80');geneset=P.source('fig3set')
 with np.load(P.files['expr79'],allow_pickle=True) as dat:gene_universe=set(map(str,dat['genes'].tolist()))
 # These 79.2 arrays were created locally by the frozen original script and use object metadata.
 P.log('Stage79.2 original gene matrix background',len(gene_universe))
 gene_universe={x.upper() for x in gene_universe}-set(CLOCK17)
 net=net.copy();net['source']=net.source.astype(str).str.upper();net['target']=net.target.astype(str).str.upper()
 net=net[net.target.isin(gene_universe)&(~net.target.isin(CLOCK17))]
 # use top4 per lineage from frozen Stage80, but require actual signed targets >=5
 nct=net.groupby('source').target.nunique()
 pri=pri[pri.lineage.isin(LINS)].sort_values(['lineage','rank_within_lineage']).copy()
 # Biological TF focus was pre-specified from the paper's Fig5/S5 storyline;
 # candidates must also be present in the ORIGINAL frozen Stage80 top8 for their lineage.
 TF_FOCUS=['PPARG','CREM','SPI1','RUNX1','ERG','TBX21','MEF2A','PRDM1','POU2F1','TWIST1','TFAP2C','FLI1','PAX5','MITF','NANOG']
 eligible=pri[pri.TF.isin(nct[nct>=5].index)].copy()
 eligible['_focus_rank']=eligible.TF.map({tf:i for i,tf in enumerate(TF_FOCUS)}).fillna(len(TF_FOCUS))
 pr=eligible.sort_values(['lineage','_focus_rank','rank_within_lineage']).groupby('lineage',sort=False).head(4).drop(columns=['_focus_rank'])
 P.csv(pr,'04_FROZEN_LINEAGE_TF_CANDIDATES.csv')
 universe=gene_universe
 sets=geneset.copy();sets['geneset']=sets.genes_scored.fillna('').apply(lambda x:set(str(x).upper().split(';')) & universe)
 best=[]
 for tf in pr.TF.unique():
  tg=set(net[net.source==tf].target)
  for _,pw in sets.iterrows():
   gs=pw.geneset
   if len(gs)<7:continue
   overlap=tg&gs
   # Crucial: include ZERO-overlap hypotheses in BH family. Filtering
   # overlaps before testing would improperly inflate enrichment significance.
   p=st.hypergeom.sf(len(overlap)-1,len(universe),len(gs),len(tg))
   best.append({'TF':tf,'pathway':pw.pathway,'module_id':pw.v3_primary_module_id,'module_label':pw.v3_primary_module_label,'n_targets':len(tg),'n_geneset':len(gs),'n_overlap':len(overlap),'overlap_genes':';'.join(sorted(overlap)),'p_hypergeom':p,'fold_enrichment':(len(overlap)/len(tg))/(len(gs)/len(universe))})
 enr=pd.DataFrame(best)
 if len(enr):
  enr['q_across_all_tested']=bh(enr.p_hypergeom)
 else: P.log('WARNING no valid TF×pathway enrichment')
 P.csv(enr,'04_TF_REGULON_TO_FUNCTIONAL_PATHWAYS_ENRICHMENT.csv')
 net_export=net[net.source.isin(pr.TF.unique())].copy();P.csv(net_export,'04_SELECTED_TF_SIGNED_TARGET_NETWORK_NO_CLOCK17.csv')
 def make(text):
  fig,axs=plt.subplots(1,5,figsize=(19.5,6.4),layout='constrained')
  for ax,l in zip(axs,LINS):
   s=pr[pr.lineage==l];tf=s.TF.tolist();piv=enr[enr.TF.isin(tf)]
   # reduce pathways to frozen named modules, min q across members - descriptive pathway-to-module max evidence
   val=piv.groupby(['TF','module_id'])['q_across_all_tested'].min().reset_index()
   mod=val.groupby('module_id')['q_across_all_tested'].min().sort_values().index.tolist()[:8]
   zz=val.pivot(index='TF',columns='module_id',values='q_across_all_tested').reindex(index=tf,columns=mod)
   dat=-np.log10(np.maximum(zz.to_numpy(float),1e-12));dat=np.minimum(dat,6)
   im=plot_matrix(ax,dat,[s[:3] for s in mod],tf,lo=0,hi=5,cmap='YlOrRd')
   ax.set_title(TO_DISPLAY[l],color=COLORS[l],fontweight='bold')
  if text:fig.suptitle('S5-new | Functional gene sets represented among CLOCK17-excluded TF regulon targets\nFrozen DoRothEA A–C; selected TF from Stage80, not TF-expression proxies');cbar(fig,im,axs,'−log10 minimum pathway FDR within module')
  return fig
 P.chart('S5_TF_REGULON_BIOLOGICAL_PATHWAY_ENRICHMENT',make)
 # frozen TF-arm evidence and target functions for focused candidates
 def make2(text):
  fig,axes=plt.subplots(1,5,figsize=(18,6),layout='constrained')
  for ax,l in zip(axes,LINS):
   s=pr[pr.lineage==l].copy();s=s.sort_values('rank_within_lineage')
   x=s.within_lineage_r.to_numpy();y=np.arange(len(s))
   ax.axvline(0,color='#ABB3BA',lw=.9)
   ax.scatter(x,y,s=115,color=COLORS[l],alpha=.9)
   for i,xv in enumerate(x):ax.plot([0,xv],[i,i],color=COLORS[l],lw=1.7)
   ax.set_yticks(y,s.TF);ax.invert_yaxis();ax.set_xlim(-.85,.85);ax.set_title(TO_DISPLAY[l],color=COLORS[l])
   for i,arm in enumerate(s.arm):
    if text:ax.text(.86,i,ARM_LABEL.get(arm,arm),va='center',ha='left',transform=ax.get_yaxis_transform(),fontsize=8)
  if text:fig.suptitle('S5-new | Frozen lineage-selective TF regulon associations and their preferred clock arms');axes[2].set_xlabel('Frozen within-lineage TF activity–arm correlation')
  return fig
 P.chart('S5_TF_REGULON_PREFERRED_ARM_BY_LINEAGE',make2)
 return enr

# 5. Fig6: actual frozen B0-inferred phase summaries across biology modules.
def analyze_phase(P):
 phase=P.source('fig6phase');frozen=P.source('fig3evid');tt=P.source('fig6tissue')
 # Restrict to frozen represented pathways, do not refit M2 and do not mix inferred with real clock sampling.
 phase=phase[phase.lineage.isin(DISPLAY)].copy();phase['module_prefix']=phase.module_id.str[:3]
 phase=phase[phase.module_prefix.isin(MODULE_FOCUS)]
 P.csv(phase,'05_INFERRED_B0_FUNCTION_LINEAGE_PHASE_AND_AMPLITUDE.csv')
 # choose one representative pathway per frozen functional module, based on original frozen family ID lexical priority
 representative=phase[['module_id','module_label','pathway']].drop_duplicates().sort_values(['module_id','pathway']).groupby('module_id').first().reset_index()
 paths=representative.pathway.tolist()[:12];p=phase[phase.pathway.isin(paths)].copy();
 # clipping amplitudes for sizes only after computing; transparent if smaller
 def make(text):
  fig,axs=plt.subplots(1,2,figsize=(16,6.5),layout='constrained')
  bins=p.pivot(index='pathway',columns='lineage',values='peak_B0_h').reindex(index=paths,columns=DISPLAY)
  amps=p.pivot(index='pathway',columns='lineage',values='amplitude').reindex(index=paths,columns=DISPLAY)
  im=plot_matrix(axs[0],bins.to_numpy(),DISPLAY,[ylabel(x) for x in paths],lo=0,hi=24,cmap=HOUR_CMAP)
  cbar(fig,im,axs[0],'Frozen inferred B0 phase (hours; not observed time)')
  im2=plot_matrix(axs[1],amps.to_numpy(),DISPLAY,[ylabel(x) for x in paths],lo=0,hi=np.nanquantile(amps.to_numpy(),.95),cmap='YlGnBu')
  cbar(fig,im2,axs[1],'Model harmonic amplitude (descriptive)')
  if text:axs[0].set_title('Predicted expression peak on inferred B0');axs[1].set_title('Corresponding predicted amplitude');fig.suptitle('S6-new | Functional programs across five lineages in the inferred-clock coordinate')
  return fig
 P.chart('S6_FUNCTIONAL_MODULE_PHASE_AND_AMPLITUDE',make)
 # Tissue-specific B–macro signed inferred phase difference on preselected three immune pathways; no physical-time interpretation
 focus=['HALLMARK_INFLAMMATORY_RESPONSE','HALLMARK_TNFA_SIGNALING_VIA_NFKB','HALLMARK_IL6_JAK_STAT3_SIGNALING']
 sub=tt[(tt.pathway.isin(focus))&(tt.status=='FIT')&(tt.n_independent_donors>=2)].copy()
 sub=sub.sort_values(['pathway','n_independent_donors'],ascending=[True,False])
 tissue_counts=sub.groupby('tissue').n_independent_donors.sum().sort_values(ascending=False)
 top=tissue_counts.index.tolist()[:12]
 P.csv(sub,'05_MATCHED_TISSUE_B_CELL_MAC_FUNCTION_PHASE_GAPS.csv')
 def make2(text):
  mat=sub[sub.tissue.isin(top)].pivot_table(index='tissue',columns='pathway',values='signed_M_minus_B_h',aggfunc='median').reindex(index=top,columns=focus)
  fig,ax=plt.subplots(figsize=(10.2,6),layout='constrained')
  im=plot_matrix(ax,mat, [ylabel(x) for x in focus],top,lo=-12,hi=12,cmap='RdBu_r')
  if text:ax.set_title('S6-new | Tissue-dependent inferred B cell–macrophage functional offsets\nSigned circular phase gap (B0 coordinates, not measured zeitgeber time)');cbar(fig,im,ax,'Macrophage minus B-cell modeled gap (h)')
  return fig
 P.chart('S6_TISSUE_BMAC_INFERRED_FUNCTION_PHASE_OFFSETS',make2)
 return phase

# 6. UC: complete CLOCK17 patient-level molecular contributions for supported arm-shape changes.
def analyze_uc(P):
 d=P.source('uc');d=d[d.lineage.isin(DISPLAY)].copy()
 clockcols=['GENE_'+x for x in CLOCK17]
 allcols=['PATH_'+x for x in set(sum(UC_FUNC.values(),[]))]
 for col in clockcols+ARMS+allcols:
  if col not in d.columns:raise AssertionError('Missing SCP259 original donor column '+col)
 d['ARM_GLOBAL_RECHECK']=d[ARMS].mean(axis=1)
 maximum=(d.ARM_GLOBAL-d.ARM_GLOBAL_RECHECK).abs().max();P.log('SCP259 ARM_GLOBAL consistency',maximum)
 # Difference between individual gene and six-arm global mean; member gene residual arithmetic produces arm shape exactly
 for gene in CLOCK17:d[f'RELGENE_{gene}']=d['GENE_'+gene]-d.ARM_GLOBAL_RECHECK
 chk=[]
 for ar,gs in GENES.items():
  residual=d[[f'RELGENE_{g}' for g in gs]].mean(axis=1)-d['SHAPE_'+ar]
  chk.append({'arm':ar,'mean_abs_error':residual.abs().mean(),'max_abs_error':residual.abs().max()})
 P.csv(pd.DataFrame(chk),'06_UC_CLOCK17_ARITHMETIC_CONSISTENCY.csv')
 rows=[]
 for l in DISPLAY:
  a=d[(d.lineage==l)&(d.condition=='UC_inflamed')];b=d[(d.lineage==l)&(d.condition=='Healthy')]
  for ar,gs in GENES.items():
   for g in gs:
    a1=a[f'RELGENE_{g}'].dropna();b1=b[f'RELGENE_{g}'].dropna();
    pv=st.mannwhitneyu(a1,b1,alternative='two-sided').pvalue if len(a1)>=4 and len(b1)>=4 else np.nan
    rows.append({'lineage':l,'arm':ar,'gene':g,'delta_rel_gene':a1.mean()-b1.mean(),'p':pv,'n_uc':len(a1),'n_healthy':len(b1)})
 tests=pd.DataFrame(rows);tests['q_all_85']=bh(tests.p)
 P.csv(tests,'06_UC_ALL_85_CLOCK17_RELATIVE_GENE_EFFECTS.csv')
 diff=tests.pivot(index='lineage',columns='gene',values='delta_rel_gene').reindex(index=DISPLAY,columns=CLOCK17)
 qmat=tests.pivot(index='lineage',columns='gene',values='q_all_85').reindex(index=DISPLAY,columns=CLOCK17)
 def make(text):
  fig,ax=plt.subplots(figsize=(14.3,5),layout='constrained')
  stars=np.where(qmat.to_numpy() <.05,'*','')
  im=plot_matrix(ax,diff,CLOCK17,DISPLAY,lo=-1.1,hi=1.1,annot=stars if text else None)
  idx=0
  for ar,gs in GENES.items():
   ax.add_patch(Rectangle((idx-.5,-.5),len(gs),5,fill=False,edgecolor='#636969',linewidth=1.15))
   if text:ax.text(idx+(len(gs)-1)/2,-1,ARM_LABEL[ar],ha='center',fontsize=10)
   idx+=len(gs)
  if text:ax.set_title('Fig7E-new | Gene-level drivers of altered relative six-arm architecture in UC\nHealthy vs inflamed UC, 30 independent patients (12 + 18)');cbar(fig,im,ax,'UC inflamed − Healthy (gene score − global arm score)')
  return fig
 P.chart('FIG7E_UC_RELATIVE_CLOCK17_GENE_DECOMPOSITION',make)
 # selectively show nonredundant supported arms with all their genes, never exclude nonsignificant genes within an arm
 focus=[('B cell','EBOX_POS'),('Fibroblast','EBOX_POS'),('Fibroblast','DBOX_ACT'),('Endothelial','EBOX_POS')]
 def make2(text):
  fig,axs=plt.subplots(1,4,figsize=(15.4,4.5),layout='constrained')
  for ax,(lin,ar) in zip(axs,focus):
   s=tests[(tests.lineage==lin)&(tests.arm==ar)].set_index('gene').reindex(GENES[ar]);y=np.arange(len(s));v=s.delta_rel_gene.to_numpy()
   ax.barh(y,v,color=[C_BLUE if x>=0 else C_RED for x in v],height=.55,alpha=.88)
   ax.set_yticks(y,s.index);ax.invert_yaxis();ax.axvline(0,color='#9AA2AA',lw=.85)
   if text:ax.set_title(lin+'\n'+ARM_LABEL[ar]);
  if text:fig.suptitle('S7-new | Complete individual CLOCK17 contributions for selected UC-altered arms');axs[1].set_xlabel('Relative gene difference, UC inflamed vs Healthy')
  return fig
 P.chart('S7_UC_SELECTED_ARMS_GENE_CONTRIBUTIONS',make2)
 # change definition of relative pathway: each patient's 50 hallmark common shift removed, rather than conflating inflammation with global upregulation
 pathcols=[c for c in d if c.startswith('PATH_HALLMARK_')]
 d['ALL_HALLMARK_MEAN']=d[pathcols].mean(axis=1)
 for pw in set(sum(UC_FUNC.values(),[])):
  col='PATH_'+pw;d['REL_'+col]=d[col]-d.ALL_HALLMARK_MEAN
 prow=[];corr=[]
 for l,ps in UC_FUNC.items():
  line=TO_DISPLAY[l]
  for pw in ps:
   col='REL_PATH_'+pw
   a=d[(d.lineage==line)&(d.condition=='UC_inflamed')][col]
   b=d[(d.lineage==line)&(d.condition=='Healthy')][col]
   pv=st.mannwhitneyu(a,b,alternative='two-sided').pvalue if len(a)>=4 and len(b)>=4 else np.nan
   prow.append({'lineage':line,'pathway':pw,'delta_relative_pathway':a.mean()-b.mean(),'p':pv,'n_healthy':len(b),'n_uc_inflamed':len(a)})
   for ar in ARMS:
    sub=d[(d.lineage==line)&(d.condition=='UC_inflamed')]
    x=sub[f'SHAPE_{ar}'];y=sub[col];z=pd.DataFrame({'x':x,'y':y,'glb':sub.ARM_GLOBAL_RECHECK,'cells':np.log1p(sub.n_cells)}).dropna()
    # use rank-based partial correlation controlling common clock shift and recovered cell counts
    if len(z)>7:
     R=z.rank();M=np.column_stack([np.ones(len(R)),R['glb'].to_numpy(),R['cells'].to_numpy()])
     rx=R.x.to_numpy()-M@np.linalg.lstsq(M,R.x.to_numpy(),rcond=None)[0]
     ry=R.y.to_numpy()-M@np.linalg.lstsq(M,R.y.to_numpy(),rcond=None)[0]
     rho=np.corrcoef(rx,ry)[0,1];dof=len(z)-4
     pp=2*st.t.sf(abs(rho)*np.sqrt(dof/(max(1e-12,1-rho*rho))),dof) if np.isfinite(rho) else np.nan
    else:rho=pp=np.nan
    corr.append({'lineage':line,'arm':ar,'pathway':pw,'rho_partial_rank':rho,'p_approx':pp,'n_uc_inflamed':len(z),'covariates':'ARM_GLOBAL_RECHECK;log1p(n_cells)'})
 path=pd.DataFrame(prow);path['q_30_predefined_tests']=bh(path.p)
 cor=pd.DataFrame(corr);cor['q_180_exploratory']=bh(cor.p_approx)
 P.csv(path,'06_UC_5LINEAGE_PREDEFINED_RELATIVE_FUNCTION_EFFECTS.csv')
 P.csv(cor,'06_UC_5LINEAGE_SHAPE_TO_FUNCTION_PATIENT_CORRELATIONS.csv')
 def make3(text):
  # A single matrix has consistent panel spacing and clearly distinguishes
  # pre-specified but untested lineage/pathway pairs (grey), not zero.
  all_paths=list(dict.fromkeys(sum(UC_FUNC.values(),[])))
  mat=path.pivot(index='pathway',columns='lineage',values='delta_relative_pathway').reindex(index=all_paths,columns=DISPLAY)
  qmat=path.pivot(index='pathway',columns='lineage',values='q_30_predefined_tests').reindex(index=all_paths,columns=DISPLAY)
  cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#E7E9EB')
  fig,ax=plt.subplots(figsize=(10.5,8.3),layout='constrained')
  vmax=max(.48,float(np.nanquantile(np.abs(mat.to_numpy()),.94)))
  im=ax.imshow(np.ma.masked_invalid(mat.to_numpy()),aspect='auto',vmin=-vmax,vmax=vmax,cmap=cmap)
  ax.set_xticks(range(5),DISPLAY,rotation=25,ha='right')
  ax.set_yticks(range(len(all_paths)),[ylabel(x) for x in all_paths])
  for i in range(len(all_paths)):
   for j in range(5):
    v=qmat.iat[i,j]
    if text and np.isfinite(v) and v < .05:
     ax.text(j,i,'**' if v<.01 else '*',ha='center',va='center',fontsize=12,color='#262B30')
  if text:
   ax.set_title('Fig7F-new | UC-associated functional remodeling across the five lineages\nCurated programs; 50-Hallmark common-expression shift removed; 30 tests')
   cbar(fig,im,ax,'UC inflamed − Healthy (relative pathway score)')
   fig.text(.02,.005,'Grey = pathway not preselected for this lineage, not zero effect; symbols = BH q<0.05 across all 30 comparisons.',fontsize=9)
  return fig
 P.chart('FIG7F_UC_LINEAGE_SPECIFIC_RELATIVE_FUNCTIONS',make3)
 def make4(text):
  fig,axs=plt.subplots(1,5,figsize=(20,7),layout='constrained')
  for ax,(l,ps) in zip(axs,UC_FUNC.items()):
   line=TO_DISPLAY[l];s=cor[cor.lineage==line];mat=s.pivot(index='pathway',columns='arm',values='rho_partial_rank').reindex(index=ps,columns=ARMS)
   q=s.pivot(index='pathway',columns='arm',values='q_180_exploratory').reindex(index=ps,columns=ARMS)
   sig=np.where(q.to_numpy()<.05,'*','')
   im=plot_matrix(ax,mat,[ARM_LABEL[a] for a in ARMS],[ylabel(x) for x in ps],lo=-.85,hi=.85,annot=sig if text else None)
   ax.set_title(line,color=COLORS[l])
  if text:fig.suptitle('S7-new | Patient-level clock shape–functional associations in inflamed UC\nExploratory partial-rank correlations; BH over all 180 tests');cbar(fig,im,axs,'Partial Spearman rho (exploratory)')
  return fig
 P.chart('S7_UC_FIVE_LINEAGE_CLOCK_FUNCTION_COUPLING',make4)
 return tests,path,cor

def write_report(P,source_stats):
 d=[]
 d.append('# Full-project Tabula biological extension: source-grounded run report')
 d.append('')
 d.append('These are **additional biological analyses**, not redraws of existing Fig1–Fig7. Data: original frozen Tabula Stage62/64/78/79.2, Fig3/6, and SCP259 donor tables.')
 d.append('')
 d.append('## Scientific constraints')
 d.append('- Stage64 six arms are **static transcriptional abundance summaries**, not oscillatory activities.')
 d.append('- Fig6 phase bins were inferred by B0 and must **never** be equated with real sampling time or endogenous oscillation.')
 d.append('- Fig6 per-bin pathway scores are re-aggregated across phase bins by posterior mass within assay; equal-weight assays, then joined to Stage64 on donor+tissue+lineage. The scores are gene-z (within tissue/assay in frozen extraction), not GSVA/Reactome activation.')
 d.append('- Fig3/4 partial regressions control lineage, tissue, donor fixed effects; SE clustered by donor. FDR over every program×six-arm hypothesis actually tested. They are exploratory associations, not causal pathways.')
 d.append('- TF activity uses frozen Stage79.2 clock-excluded regulons; enrichment uses actual DoRothEA A–C target edges (excluding all CLOCK17), intersected with measured Stage79.2 gene universe and frozen Fig3 curated gene sets. TF target enrichment does not prove activation or binding.')
 d.append('- SCP259 uses 12 Healthy vs 18 UC donors for unpaired contrasts, 18 UC donors for correlations. All 85 relative CLOCK17 gene contrasts FDR-controlled together; 30 curated relative Hallmark contrasts corrected as a family; 180 exploratory associations corrected together.')
 d.append('- When no association is significant, plots still show actual results; no labels imply positive biology without evidence.')
 d.append('')
 d.append('## Primary new panels')
 for panel in P.man:d.append('- `'+panel+'`')
 d.append('')
 d.append('## Matched-data provenance')
 d.append('```json\n'+json.dumps(source_stats,ensure_ascii=False,indent=2)+'\n```')
 (P.out/'RESULTS_AND_INTERPRETATION.md').write_text('\n'.join(d)+'\n',encoding='utf-8')

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--project-root',required=True,help='Parent of katv2 and corrected_full_analysis_v2')
 ap.add_argument('--outdir',required=True,help='Writable full-result output directory')
 a=ap.parse_args();p=Pipeline(a)
 p.log('Starting full frozen-source biological analysis with 7 thematic steps')
 tissue,prof=analyze_tissue(p)
 gene=analyze_clock17(p,tissue)
 matched,names=build_pathway_join(p,tissue)
 matched,coup=analyze_arm_pathway(p,matched,names)
 enr=analyze_tf(p)
 phase=analyze_phase(p)
 ucgene,ucpath,uccor=analyze_uc(p)
 summary={'n_new_panel_themes':len(p.man),'new_panel_files':len(list(p.pan.glob('*.png'))),'tabula_stage64_rows':len(tissue),'tabula_gene_matched_rows':len(gene),'pathway_matched_rows':len(matched),'matched_donors':matched.donor.nunique(),'matched_tissues':matched.tissue.nunique(),'tf_target_enrichment_tests':len(enr),'uc_gene_tests':len(ucgene),'uc_function_tests':len(ucpath),'uc_function_pairwise_associations':len(uccor),'arm_pathway_cluster_q05':int((coup.BH_q_all_arm_pathway<.05).sum()),'uc_gene_q05':int((ucgene.q_all_85<.05).sum()),'uc_function_q05':int((ucpath.q_30_predefined_tests<.05).sum()),'uc_function_assoc_q05':int((uccor.q_180_exploratory<.05).sum()),'panels':p.man}
 p.write_json('RUN_SUMMARY.json',summary);write_report(p,summary);p.log('COMPLETE',json.dumps(summary,ensure_ascii=False));print('COMPLETE')

if __name__=='__main__':main()
