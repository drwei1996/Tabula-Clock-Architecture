#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SCP259 five-lineage static CLOCK17/six-arm figure extension (no raw reprocessing).

Uses ONLY already-generated SCP259 donor×condition×lineage scores.
- Predefined 5 broad lineages, 6 original arms, 3 comparisons.
- Wilcoxon paired UC inflamed/noninflamed on exact same donors; unrelated Healthy
  comparisons use donor-level Mann–Whitney U.
- Original arm, mean-subtracted arm shape, healthy-centroid distances (raw and
  shape only) with patient-level bootstrap CI and honest FDR families.
- TF expression and Hallmark/Reactome expression score contrasts.
- Relative pathway scores = each pathway - mean of *available* pathway scores
  from the same source feature panel and same donor (not enrichment NES).
- Within-state TF-shape and pathway-shape relationships, exploratory only.
- Reports which proposed TF -> clock -> pathway chains remain unsupported.
No B0, true ZT, rhythms, regulon activity, intervention, or causality claims.

Usage:
 python scp259_figure_extension_v1.py --scp-root \
   "$PWD/TABULA_SCP259_DISEASE_OVERNIGHT" \
   --outdir "$PWD/TABULA_SCP259_FIGURE_EXTENSION_V1"

Options: --bootstrap 2000 --seed 42 --min-donors 5 --no-pdf
Python: numpy pandas scipy matplotlib (no seaborn / statsmodels required)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, rankdata, t as tdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

LINEAGES=['B cell','T cell','Macrophage/Monocyte','Fibroblast','Endothelial']
ARMS=['EBOX_POS','EBOX_NEG','RORE_ACT','RORE_REP','DBOX_ACT','DBOX_REP']
STATES=['Healthy','UC_noninflamed','UC_inflamed']
COMP=[('UC_noninflamed','Healthy'),('UC_inflamed','Healthy'),('UC_inflamed','UC_noninflamed')]
COL={'Healthy':'#5C879E','UC_noninflamed':'#DBAC51','UC_inflamed':'#CB6770'}
LC={'B cell':'#5485C6','T cell':'#CE5E6B','Macrophage/Monocyte':'#E1A041','Fibroblast':'#65B19C','Endothelial':'#8A65BC'}
ARM_SHORT={'EBOX_POS':'E-box +','EBOX_NEG':'E-box -','RORE_ACT':'RORE +','RORE_REP':'RORE -','DBOX_ACT':'D-box +','DBOX_REP':'D-box -'}
# Fixed clinically interpretable Hallmark panel for *display only*. All 50 sets
# remain tested and BH adjusted in complete CSV. Selection does not change q.
CLINICAL_PATHWAYS=[
 'HALLMARK_G2M_CHECKPOINT','HALLMARK_E2F_TARGETS','HALLMARK_MITOTIC_SPINDLE',
 'HALLMARK_IL6_JAK_STAT3_SIGNALING','HALLMARK_TNFA_SIGNALING_VIA_NFKB',
 'HALLMARK_INTERFERON_ALPHA_RESPONSE','HALLMARK_INTERFERON_GAMMA_RESPONSE',
 'HALLMARK_IL2_STAT5_SIGNALING','HALLMARK_COMPLEMENT',
 'HALLMARK_INFLAMMATORY_RESPONSE','HALLMARK_PI3K_AKT_MTOR_SIGNALING',
 'HALLMARK_MTORC1_SIGNALING','HALLMARK_ALLOGRAFT_REJECTION',
 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']
CREDIT='Exploratory donor-level, static transcriptional signatures; no causal/regulon/oscillation inference'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

def hsh(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def bh(p):
    p=np.asarray(p,float); ok=np.isfinite(p); ans=np.full(len(p),np.nan)
    if not ok.any():return ans
    tmp=p[ok]; idx=np.argsort(tmp); s=tmp[idx]
    q=np.minimum.accumulate((s*len(s)/np.arange(1,len(s)+1))[::-1])[::-1]
    inv=np.empty_like(q);inv[idx]=np.clip(q,0,1);ans[ok]=inv
    return ans

def read_sources(args):
    src=Path(args.scp_root).expanduser().resolve()
    path=src/'results'/'02_DONOR_CONDITION_LINEAGE_SCORES.csv'
    if not path.exists():
        raise FileNotFoundError(f'Required frozen donor-level input missing: {path}')
    x=pd.read_csv(path)
    need={'Subject','condition','lineage','n_cells',*ARMS}
    missing=need-set(x.columns)
    if missing:raise ValueError('Donor CSV missing: '+', '.join(sorted(missing)))
    if x[list(need&set(x.columns))].isna().any().any():
        raise ValueError('Missing values in grouping/6-arm source columns; stop rather than silently impute')
    states=set(x.condition.astype(str)); lins=set(x.lineage.astype(str))
    if states!=set(STATES): raise ValueError(f'Unexpected states {states}')
    if lins!=set(LINEAGES): raise ValueError(f'Unexpected lineages {lins}')
    dup=x.duplicated(['Subject','condition','lineage'])
    if dup.any():raise ValueError('Multiple rows for same donor-condition-lineage; donor independence violated')
    for a in ARMS:x[a]=pd.to_numeric(x[a],errors='raise')
    if (x.n_cells.astype(float)<=0).any():raise ValueError('Invalid n_cells')
    x['ARM_GLOBAL_RECALC']=x[ARMS].mean(axis=1)
    for a in ARMS:x['SHAPE_'+a]=x[a]-x['ARM_GLOBAL_RECALC']
    # Preserve pre-existing values for numerical agreement audit
    if 'ARM_GLOBAL' in x:
        v=np.max(np.abs(x['ARM_GLOBAL'].to_numpy(float)-x.ARM_GLOBAL_RECALC.to_numpy(float)))
        if v>1e-7:raise ValueError(f'Existing ARM_GLOBAL differs from recalculated value: {v}')
    # Use original raw-scored per-donor pathway features, do not invent pathway activity.
    paths=sorted(c for c in x if c.startswith('PATH_') and x[c].notna().all())
    tfs=sorted(c for c in x if c.startswith('TF_') and x[c].notna().all())
    if not tfs:raise ValueError('No TF_* columns; cannot perform requested TF analysis')
    if not paths: warnings.warn('No PATH_* columns: missing Hallmark GMT at original run; will produce TF and six-arm only')
    if paths:
        x['PATH_PANEL_GLOBAL']=x[paths].mean(axis=1)
        for p in paths:x['REL_'+p]=x[p]-x['PATH_PANEL_GLOBAL']
    # healthy centroid: one per lineage; healthy LOO distances to avoid self-inclusion bias
    x['SIX_ARM_DISTANCE_RECALC']=np.nan
    x['SHAPE_DISTANCE_LOO']=np.nan
    for lin in LINEAGES:
        sub=x[x.lineage==lin]
        health=sub[sub.condition=='Healthy']; n=len(health)
        if n<3:raise ValueError(f'Healthy {lin} n={n}<3; no stable reference')
        for typ,cols,target in [('raw',ARMS,'SIX_ARM_DISTANCE_RECALC'),
                                ('shape',['SHAPE_'+a for a in ARMS],'SHAPE_DISTANCE_LOO')]:
            baseline=health[cols].to_numpy(float); center=baseline.mean(axis=0)
            block=sub[cols].to_numpy(float)
            dist=np.sqrt(np.mean((block-center)**2,axis=1))
            x.loc[sub.index,target]=dist
            for i,row in enumerate(health.index):
                loo_center=(baseline.sum(axis=0)-baseline[i])/(n-1)
                x.loc[row,target]=np.sqrt(np.mean((baseline[i]-loo_center)**2))
    if 'SIX_ARM_DISTANCE' in x:
        err=np.nanmax(np.abs(x.SIX_ARM_DISTANCE_RECALC-x.SIX_ARM_DISTANCE))
        if err>1e-6:raise ValueError(f'Recomputed source SIX_ARM_DISTANCE inconsistent: {err:.4g}')
    return x,path,paths,tfs

def contrast_vector(g,a,b,var,minimum=3):
    A=g[g.condition==a].set_index('Subject')
    B=g[g.condition==b].set_index('Subject')
    paired=(a=='UC_inflamed' and b=='UC_noninflamed')
    if paired:
        ids=A.index.intersection(B.index).sort_values()
        yy=A.loc[ids,var].to_numpy(float);zz=B.loc[ids,var].to_numpy(float)
        both=np.isfinite(yy)&np.isfinite(zz);y=yy[both];z=zz[both]
        donors=';'.join(ids[both].astype(str))
    else:
        y=A[var].to_numpy(float);z=B[var].to_numpy(float)
        y=y[np.isfinite(y)];z=z[np.isfinite(z)];donors='UNPAIRED'
    if len(y)<minimum or len(z)<minimum:
        return y,z,paired,'INSUFFICIENT_DONORS',donors
    return y,z,paired,'OK',donors

def test_vals(a,b,paired,nboot,seed):
    d=float(np.mean(a)-np.mean(b))
    if paired:
        if np.allclose(a,b,atol=0,rtol=0):p=1.
        else:
            try:p=float(wilcoxon(a-b,alternative='two-sided',zero_method='wilcox',method='auto').pvalue)
            except ValueError:p=np.nan
    else:
        p=float(mannwhitneyu(a,b,alternative='two-sided',method='auto').pvalue)
    rng=np.random.default_rng(seed)
    if paired:
        diff=a-b;sim=diff[rng.integers(len(diff),size=(nboot,len(diff)))].mean(axis=1)
    else:
        sim=(a[rng.integers(len(a),size=(nboot,len(a)))].mean(axis=1)
             -b[rng.integers(len(b),size=(nboot,len(b)))].mean(axis=1))
    lo,hi=np.percentile(sim,[2.5,97.5])
    return d,p,float(lo),float(hi)

def kind(col):
    if col in ARMS:return 'ARM'
    if col.startswith('SHAPE_') and col[6:] in ARMS:return 'SHAPE'
    if col in ('SIX_ARM_DISTANCE_RECALC','SHAPE_DISTANCE_LOO'):return 'DISTANCE'
    if col=='ARM_GLOBAL_RECALC':return 'ARM_GLOBAL'
    if col.startswith('TF_'):return 'TF_EXPRESSION'
    if col.startswith('REL_PATH_'):return 'PATHWAY_RELATIVE'
    if col=='PATH_PANEL_GLOBAL':return 'PATHWAY_PANEL_GLOBAL'
    if col.startswith('PATH_'):return 'PATHWAY_RAW'
    raise ValueError(col)

def tests_all(x, paths,tfs, nboot,seed):
    core=ARMS+['SHAPE_'+a for a in ARMS]+['ARM_GLOBAL_RECALC',
                                        'SIX_ARM_DISTANCE_RECALC','SHAPE_DISTANCE_LOO']
    cols=core+tfs+paths+['REL_'+p for p in paths]+(['PATH_PANEL_GLOBAL'] if paths else [])
    rows=[]
    for ci,(a,b) in enumerate(COMP):
        for li,lin in enumerate(LINEAGES):
            g=x[x.lineage==lin]
            for vi,var in enumerate(cols):
                y,z,pair,status,ids=contrast_vector(g,a,b,var)
                if status=='OK':
                    delta,p,lo,hi=test_vals(y,z,pair,nboot,seed+ci*100000+li*10000+vi)
                else:delta=p=lo=hi=np.nan
                rows.append(dict(contrast=f'{a}__vs__{b}',lineage=lin,feature=var,
                    family=kind(var),delta=delta,p=p,ci_low=lo,ci_high=hi,
                    n_a=len(y),n_b=len(z),paired=pair,matched_donors=ids,status=status))
    t=pd.DataFrame(rows);t['q_bh']=np.nan
    # Predefine entire family per comparison: ARM & SHAPE 5*6, DISTANCE 5*2,
    # TF 5*38, pathway original 5*50, relative 5*50. Pair comparisons are
    # distinct questions; also export globally BH across three comparisons.
    for (c,k),ind in t.groupby(['contrast','family']).groups.items():
        t.loc[ind,'q_bh']=bh(t.loc[ind,'p'])
    t['q_bh_all_three_contrasts']=np.nan
    for k,ind in t.groupby('family').groups.items():
        t.loc[ind,'q_bh_all_three_contrasts']=bh(t.loc[ind,'p'])
    return t

def matched_counts(x):
    rows=[]
    for lin in LINEAGES:
        g=x[x.lineage==lin]
        hc=set(g.loc[g.condition=='Healthy','Subject']); non=set(g.loc[g.condition=='UC_noninflamed','Subject']); infl=set(g.loc[g.condition=='UC_inflamed','Subject'])
        rows.append(dict(lineage=lin,n_healthy=len(hc),n_UC_noninflamed=len(non),n_UC_inflamed=len(infl),
                         n_UC_paired=len(non&infl),n_UC_only_non=len(non-infl),n_UC_only_infl=len(infl-non)))
    return pd.DataFrame(rows)

def residual_rank_corr(d,a,b,covariates=('ARM_GLOBAL_RECALC','n_cells')):
    # Rank partial Pearson = a partial Spearman *descriptor*. Approximate
    # parametric p-values (exploratory; not randomization-causal).
    y=d[a].to_numpy(float);z=d[b].to_numpy(float)
    cv=[d[c].to_numpy(float) for c in covariates]
    good=np.isfinite(y)&np.isfinite(z)
    for c in cv:good&=np.isfinite(c)
    y=y[good];z=z[good];cv=[c[good] for c in cv]
    n=len(y);k=len(cv)
    if n<max(9,k+6) or np.unique(y).size<4 or np.unique(z).size<4:return np.nan,np.nan,n
    # Avoid log zero n_cells and explicitly condition on donor sampling depth.
    cv=[np.log1p(v) if c=='n_cells' else v for c,v in zip(covariates,cv)]
    mat=np.column_stack([np.ones(n)]+[rankdata(c) for c in cv])
    if np.linalg.matrix_rank(mat)<mat.shape[1]:return np.nan,np.nan,n
    ry=rankdata(y);rz=rankdata(z)
    ey=ry-mat@np.linalg.lstsq(mat,ry,rcond=None)[0]
    ez=rz-mat@np.linalg.lstsq(mat,rz,rcond=None)[0]
    sy=np.linalg.norm(ey);sz=np.linalg.norm(ez)
    if min(sy,sz)<1e-10:return np.nan,np.nan,n
    rho=float(np.clip(np.dot(ey,ez)/(sy*sz),-1,1))
    df=n-k-2
    if df<=0:return rho,np.nan,n
    tval=abs(rho)*np.sqrt(df/max(1e-12,1-rho*rho))
    p=float(2*tdist.sf(tval,df))
    return rho,p,n

def partial_corr_matrix(group, left_cols, right_cols):
    """Fast rank-partial correlation matrix, keeping covariate adjustment identical.

    For nonfinite data return NaN pairwise, never coerce missing data to zero.
    """
    allcols=list(dict.fromkeys(left_cols+right_cols+['ARM_GLOBAL_RECALC','n_cells']))
    df=group[allcols].copy()
    n=len(df);k=2
    rho=np.full((len(left_cols),len(right_cols)),np.nan)
    pv=rho.copy()
    if n<9:return rho,pv
    if not np.isfinite(df.astype(float).to_numpy()).all():
        # Missingness not expected in upstream exported scores; perform safe
        # slower pairwise association only where available.
        for i,l in enumerate(left_cols):
            for j,r in enumerate(right_cols):
                rho[i,j],pv[i,j],_=residual_rank_corr(group,l,r)
        return rho,pv
    cov=np.column_stack([np.ones(n),rankdata(df.ARM_GLOBAL_RECALC.to_numpy(float)),
                               rankdata(np.log1p(df.n_cells.to_numpy(float)))])
    if np.linalg.matrix_rank(cov)<3:return rho,pv
    q,_=np.linalg.qr(cov,mode='reduced')
    A=np.column_stack([rankdata(df[c].to_numpy(float)) for c in left_cols])
    B=np.column_stack([rankdata(df[c].to_numpy(float)) for c in right_cols])
    A=A-q@(q.T@A);B=B-q@(q.T@B)
    na=np.linalg.norm(A,axis=0);nb=np.linalg.norm(B,axis=0)
    valid=(na[:,None]>1e-9)&(nb[None,:]>1e-9)
    denom=np.outer(na,nb)
    rho[valid]=np.clip((A.T@B)[valid]/denom[valid],-1,1)
    dfree=n-k-2
    tv=np.abs(rho)*np.sqrt(dfree/np.maximum(1e-14,1-rho*rho))
    pv[valid]=2*tdist.sf(tv[valid],dfree)
    return rho,pv


def association_scan(x,tfs,paths,min_donors):
    rows=[]
    for state in STATES:
        for lin in LINEAGES:
            d=x[(x.condition==state)&(x.lineage==lin)]
            if len(d)<min_donors:continue
            for kind_,features in [('TF_SHAPE',tfs),('PATH_SHAPE_RELATIVE',['REL_'+p for p in paths])]:
                if not features:continue
                shape_cols=['SHAPE_'+a for a in ARMS]
                rho,p=partial_corr_matrix(d,shape_cols,features)
                for i,arm in enumerate(ARMS):
                    for j,ft in enumerate(features):
                        rows.append(dict(condition=state,lineage=lin,arm=arm,
                         partner=ft,link_type=kind_,rho_partial_rank=rho[i,j],p_approx=p[i,j],
                         n_donors=len(d),controlling='ARM_GLOBAL_RECALC;log1p(n_cells)'))
    a=pd.DataFrame(rows)
    if a.empty:return a
    a['q_bh']=np.nan
    for _,ix in a.groupby(['condition','link_type']).groups.items():
        a.loc[ix,'q_bh']=bh(a.loc[ix,'p_approx'])
    return a


def tf_path_bcell_links(x,tfs,paths):
    out=[]
    if not paths:return pd.DataFrame(columns=['condition','TF','pathway','rho_partial_rank','p_approx','n_donors','q_bh'])
    for state in STATES:
        g=x[(x.lineage=='B cell')&(x.condition==state)]
        rhs=['REL_'+p for p in paths]
        rho,p=partial_corr_matrix(g,tfs,rhs)
        for i,tf in enumerate(tfs):
            for j,path in enumerate(paths):
                out.append(dict(condition=state,TF=tf,pathway=path,
                                rho_partial_rank=rho[i,j],p_approx=p[i,j],n_donors=len(g)))
    out=pd.DataFrame(out);out['q_bh']=np.nan
    for state,idx in out.groupby('condition').groups.items():out.loc[idx,'q_bh']=bh(out.loc[idx,'p_approx'])
    return out

def ordered_summary(t,assoc,tfpath):
    a=t[t.contrast=='UC_inflamed__vs__Healthy'].copy()
    # descriptive candidate ranking; DOES NOT aggregate link p-values into a
    # nonexistent causal statistical test. Show every chain's individual evidence.
    out=[]
    bshape=a[(a.lineage=='B cell')&(a.feature.str.startswith('SHAPE_'))&(a.q_bh<.05)]
    if bshape.empty:bshape=a[(a.lineage=='B cell')&(a.feature.str.startswith('SHAPE_'))].sort_values('p').head(1)
    for _,sh in bshape.iterrows():
        arm=sh.feature.removeprefix('SHAPE_')
        tlink=assoc[(assoc.condition=='UC_inflamed')&(assoc.lineage=='B cell')&(assoc.arm==arm)&(assoc.link_type=='TF_SHAPE')]
        plink=assoc[(assoc.condition=='UC_inflamed')&(assoc.lineage=='B cell')&(assoc.arm==arm)&(assoc.link_type=='PATH_SHAPE_RELATIVE')]
        tlink=tlink.dropna(subset=['p_approx']).sort_values('p_approx').head(12)
        plink=plink.dropna(subset=['p_approx']).sort_values('p_approx').head(12)
        for _,tf in tlink.iterrows():
            q=a[(a.lineage=='B cell')&(a.feature==tf.partner)]
            if q.empty:continue
            q=q.iloc[0]
            for _,pa in plink.iterrows():
                pr=a[(a.lineage=='B cell')&(a.feature==pa.partner)]
                if pr.empty:continue
                pr=pr.iloc[0]
                z=tfpath[(tfpath.condition=='UC_inflamed')&(tfpath.TF==tf.partner)&(tfpath.pathway==pa.partner.removeprefix('REL_'))]
                z=z.iloc[0] if not z.empty else None
                out.append(dict(lineage='B cell',arm=arm,TF=tf.partner.removeprefix('TF_'),pathway=pa.partner.removeprefix('REL_PATH_'),
                    shape_delta=sh.delta,shape_q=sh.q_bh,
                    tf_expression_delta=q.delta,tf_expression_q=q.q_bh,
                    relative_pathway_delta=pr.delta,relative_pathway_q=pr.q_bh,
                    TF_to_SHAPE_partial_rho=tf.rho_partial_rank,TF_to_SHAPE_q=tf.q_bh,
                    SHAPE_to_PATH_partial_rho=pa.rho_partial_rank,SHAPE_to_PATH_q=pa.q_bh,
                    TF_to_PATH_partial_rho=z.rho_partial_rank if z is not None else np.nan,
                    TF_to_PATH_q=z.q_bh if z is not None else np.nan,
                    interpretation='Exploratory non-causal coassociation; no TF perturbation or time series'))
    res=pd.DataFrame(out)
    if not res.empty:
        res['clinically_interpretable_pathway']=res['pathway'].isin(CLINICAL_PATHWAYS)
        res['n_nominal_links']=((res['TF_to_SHAPE_q']<.10).astype(int)+(res['SHAPE_to_PATH_q']<.10).astype(int)+(res['TF_to_PATH_q']<.10).astype(int))
        res['candidate_rank_metric']=res[['shape_q','tf_expression_q','relative_pathway_q','TF_to_SHAPE_q','SHAPE_to_PATH_q']].fillna(1).clip(lower=1e-6).apply(lambda z:float(np.mean(-np.log10(z))),axis=1)
        res=res.sort_values(['clinically_interpretable_pathway','n_nominal_links','candidate_rank_metric'],ascending=False)
    return res

def figsave(fig,out,name,no_pdf):
    exts=['png','svg']+([] if no_pdf else ['pdf'])
    for ext in exts:fig.savefig(out/f'{name}.{ext}',dpi=270 if ext=='png' else None,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)

def forest(t,out,no_pdf):
    contrast='UC_inflamed__vs__Healthy'
    # shape-specific forest: avoid incorrectly treating global shift as architecture
    sub=t[(t.contrast==contrast)&(t.family=='SHAPE')]
    fig,axes=plt.subplots(1,5,figsize=(16,4.6),sharey=True)
    for k,(lin,ax) in enumerate(zip(LINEAGES,axes)):
        z=sub[sub.lineage==lin].set_index('feature').reindex(['SHAPE_'+a for a in ARMS])
        y=np.arange(6);eff=z.delta.to_numpy(float);lo=z.ci_low.to_numpy(float);hi=z.ci_high.to_numpy(float)
        ax.axvline(0,color='#B0BBC5',lw=.8)
        for j in range(6):
            ok=z.q_bh.iloc[j]<.05 if np.isfinite(z.q_bh.iloc[j]) else False
            ax.errorbar(eff[j],j,xerr=[[max(0,eff[j]-lo[j])],[max(0,hi[j]-eff[j])]],fmt='D' if ok else 'o',
                        color=LC[lin],alpha=1. if ok else .48,markersize=6 if ok else 4,capthick=.8,capsize=2,lw=1.7)
            if ok:ax.text(hi[j]+.025,j,'*',fontsize=10,color=LC[lin],va='center')
        if k==0:ax.set_yticks(y,[ARM_SHORT[a] for a in ARMS])
        ax.invert_yaxis();ax.set_title(lin,color=LC[lin],fontsize=10)
        ax.grid(axis='x',alpha=.13)
    fig.suptitle('A | UC inflamed vs Healthy: relative six-arm shape contrasts',y=1.04,fontsize=13)
    fig.supxlabel('Patient-level mean difference (six-arm mean removed); 95% bootstrap CI | * BH q < 0.05')
    fig.tight_layout()
    figsave(fig,out,'FIG_A_SHAPE_EFFECT_FOREST',no_pdf)

def bcell_scores(x,t,out,no_pdf):
    d=x[x.lineage=='B cell']
    f=t[(t.lineage=='B cell')&(t.family=='SHAPE')]
    fig,axes=plt.subplots(2,3,figsize=(13,7.1),sharex=True)
    rng=np.random.default_rng(51)
    for ax,arm in zip(axes.flat,ARMS):
        feat='SHAPE_'+arm
        for k,state in enumerate(STATES):
            a=d[d.condition==state][feat].to_numpy(float)
            jit=rng.uniform(-.15,.15,len(a));ax.scatter(k+jit,a,s=18,color=COL[state],alpha=.75,zorder=3)
            ax.plot([k-.25,k+.25],[np.mean(a)]*2,color=COL[state],lw=2.4)
        sig=f[(f.feature==feat)&(f.contrast=='UC_inflamed__vs__Healthy')]
        q=float(sig.q_bh.iloc[0]) if not sig.empty else np.nan
        ax.set_title(f'{ARM_SHORT[arm]} | UC infl. vs Healthy q={q:.3g}',fontsize=10)
        ax.set_xticks(range(3),['Healthy','UC noninfl.','UC infl.'],rotation=22)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('B | B cell relative six-arm scores: one point = one donor',fontsize=13)
    fig.tight_layout();figsave(fig,out,'FIG_B_BCELL_DONOR_SHAPES',no_pdf)

def distance_figure(x,t,out,no_pdf):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    rng=np.random.default_rng(19)
    for ax,(feature,title) in zip(axes,[('SIX_ARM_DISTANCE_RECALC','Raw six-arm distance'),('SHAPE_DISTANCE_LOO','Relative-shape distance')]):
        for j,lin in enumerate(LINEAGES):
            d=x[x.lineage==lin]
            for k,state in enumerate(STATES):
                v=d.loc[d.condition==state,feature].to_numpy(float)
                center=j+(k-1)*.23
                ax.scatter(center+rng.uniform(-.052,.052,len(v)),v,color=COL[state],s=13,alpha=.63)
                ax.plot([center-.08,center+.08],[v.mean()]*2,lw=2,color=COL[state])
        ax.set_xticks(range(5),['B','T','Macro','Fibro','Endo']);ax.set_title(title)
        ax.set_ylabel('RMS from lineage-specific Healthy centroid (Healthy LOO)')
        ax.grid(axis='y',alpha=.15)
    handles=[plt.Line2D([0],[0],marker='o',color=c,lw=0,label=s) for s,c in COL.items()]
    axes[1].legend(handles=handles,frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('C | Healthy-centroid departures: descriptive donor-level distances',fontsize=12.5)
    fig.tight_layout();figsave(fig,out,'FIG_C_RAW_AND_SHAPE_DISTANCES',no_pdf)
    # Complete statistics, all 3 contrasts
    t[(t.family=='DISTANCE')].to_csv(out/'SOURCE_DISTANCES_ALL_CONTRASTS.csv',index=False)

def tf_fig(t,assoc,out,no_pdf):
    lin='B cell';cmp='UC_inflamed__vs__Healthy'
    d=t[(t.contrast==cmp)&(t.lineage==lin)&(t.family=='TF_EXPRESSION')].copy()
    d=d.sort_values('q_bh').head(15).sort_values('delta')
    links=assoc[(assoc.condition=='UC_inflamed')&(assoc.lineage=='B cell')&(assoc.arm=='EBOX_POS')&(assoc.link_type=='TF_SHAPE')]
    fig,axes=plt.subplots(1,2,figsize=(12.3,5.3))
    ax=axes[0];y=np.arange(len(d)); ax.axvline(0,color='#AFBAC4',lw=1)
    for j,(_,r) in enumerate(d.iterrows()):
        col=LC['B cell'] if r.q_bh<.05 else '#AAB8C8';
        ax.errorbar(r.delta,j,xerr=[[max(0,r.delta-r.ci_low)],[max(0,r.ci_high-r.delta)]],fmt='o',color=col,capsize=2)
    ax.set_yticks(y,d.feature.str.removeprefix('TF_'))
    ax.set_title('TF expression difference\nUC inflamed - Healthy')
    ax.set_xlabel('Healthy-referenced gene z | CI; blue q < 0.05')
    ax=axes[1];z=links.dropna(subset=['p_approx']).sort_values('p_approx').head(15).sort_values('rho_partial_rank')
    yy=np.arange(len(z));ax.barh(yy,z.rho_partial_rank,color=[LC['B cell'] if v<.05 else '#9CB3C5' for v in z.q_bh])
    ax.set_yticks(yy,z.partner.str.removeprefix('TF_'))
    ax.axvline(0,lw=.8,color='#AAB4BF');ax.set_xlim(-1,1)
    ax.set_title('Within UC inflamed: TF vs E-box+ SHAPE\npartial rank correlations')
    ax.set_xlabel('Condition on arm global + cell-count proxy; blue q < 0.05')
    fig.suptitle('D | B-cell TF candidates: expression vs architecture association',fontsize=13)
    fig.tight_layout();figsave(fig,out,'FIG_D_BCELL_TF_CANDIDATES',no_pdf)

def pathway_fig(t,assoc,paths,out,no_pdf):
    if not paths:return
    cmp='UC_inflamed__vs__Healthy'
    sub=t[(t.contrast==cmp)&(t.lineage=='B cell')&(t.family=='PATHWAY_RELATIVE')].copy()
    if sub.empty:return
    sub['path']=sub.feature.str.removeprefix('REL_PATH_HALLMARK_').str.removeprefix('REL_PATH_')
    # Figure displays fixed clinically relevant pathways; this is not
    # outcome-driven selection. ALL 50 sets / q-values are preserved in CSV.
    sub=sub[sub.feature.str.removeprefix('REL_PATH_').isin(CLINICAL_PATHWAYS)].copy()
    draw=sub.set_index('feature').reindex(['REL_PATH_'+v for v in CLINICAL_PATHWAYS]).dropna(subset=['delta']).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(14,6.4))
    y=np.arange(len(draw))
    ax=axes[0];ax.axvline(0,color='#9DAFBF',lw=.8)
    for i,(_,r) in enumerate(draw.iterrows()):
        col='#C96570' if r.q_bh<.05 else '#92AABD'
        ax.errorbar(r.delta,i,xerr=[[max(0,r.delta-r.ci_low)],[max(0,r.ci_high-r.delta)]],fmt='o',color=col,capsize=2,lw=1.4)
    ax.set_yticks(y,draw.path.str.replace('_',' ',regex=False),fontsize=7.5)
    ax.invert_yaxis();ax.set_title('B cell: curated immune/proliferation programs\nUC inflamed - Healthy')
    ax.set_xlabel('Relative pathway score: red = BH q < 0.05')
    ax=axes[1]
    assocB=assoc[(assoc.condition=='UC_inflamed')&(assoc.lineage=='B cell')&(assoc.arm=='EBOX_POS')&(assoc.link_type=='PATH_SHAPE_RELATIVE')].dropna(subset=['p_approx']).copy()
    assocB=assocB[assocB.partner.str.removeprefix('REL_PATH_').isin(CLINICAL_PATHWAYS)].copy()
    assocB=assocB.set_index('partner').reindex(['REL_PATH_'+v for v in CLINICAL_PATHWAYS]).dropna(subset=['rho_partial_rank']).reset_index()
    labs=assocB.partner.str.removeprefix('REL_PATH_HALLMARK_').str.removeprefix('REL_PATH_').str.replace('_',' ',regex=False)
    ax.barh(np.arange(len(assocB)),assocB.rho_partial_rank,color=[LC['B cell'] if q<.05 else '#92AABD' for q in assocB.q_bh])
    ax.set_yticks(np.arange(len(assocB)),labs,fontsize=7.5);ax.axvline(0,color='#9DAFBF',lw=.8);ax.set_xlim(-1,1)
    ax.invert_yaxis();ax.set_xlabel('Partial rank rho (E-box+ SHAPE vs relative pathway)')
    ax.set_title('Same 14 programs within UC inflamed\ncontrol global arm + sampling depth')
    fig.suptitle('E | B-cell expression-program disturbances (NOT pathway activation)',fontsize=13)
    fig.tight_layout();figsave(fig,out,'FIG_E_BCELL_RELATIVE_PATHWAYS',no_pdf)

def paired_fig(t,out,no_pdf):
    sub=t[(t.contrast=='UC_inflamed__vs__UC_noninflamed')&(t.family=='SHAPE')]
    fig,axes=plt.subplots(1,5,figsize=(15.6,4.6),sharey=True)
    for k,(lin,ax) in enumerate(zip(LINEAGES,axes)):
        s=sub[sub.lineage==lin].set_index('feature').reindex(['SHAPE_'+a for a in ARMS])
        ax.axvline(0,color='#9FAEBC',lw=.9)
        for j,r in enumerate(s.itertuples()):
            eff=r.delta
            if not np.isfinite(eff):continue
            ax.errorbar(eff,j,xerr=[[max(0,eff-r.ci_low)],[max(0,r.ci_high-eff)]],
                        fmt='D' if r.q_bh<.05 else 'o',color=LC[lin],alpha=1 if r.q_bh<.05 else .65,capsize=2,lw=1.5)
        if k==0:ax.set_yticks(range(6),[ARM_SHORT[a] for a in ARMS])
        ax.invert_yaxis();ax.set_title(lin,color=LC[lin]);ax.grid(axis='x',alpha=.15)
    fig.suptitle('F | Matched UC donors: inflamed vs noninflamed relative six-arm changes',fontsize=13)
    fig.supxlabel('Within-donor mean difference with 95% paired bootstrap CI (5 x 6 BH per contrast)')
    fig.tight_layout();figsave(fig,out,'FIG_F_PAIRED_UC_SHAPE_COMPARISON',no_pdf)

def render_report(x,t,associations,triplets,paths,tfs,out,src):
    q=lambda name: t[(t.family==name)&(t.q_bh<.05)].shape[0]
    cm=lambda comp,fam: t[(t.contrast==comp)&(t.family==fam)&(t.q_bh<.05)].shape[0]
    ss=t[(t.lineage=='B cell')&(t.feature=='SHAPE_EBOX_POS')]
    v=ss[ss.contrast=='UC_inflamed__vs__Healthy'];v=v.iloc[0] if len(v) else None
    br=t[(t.family=='PATHWAY_RELATIVE')&(t.lineage=='B cell')&(t.contrast=='UC_inflamed__vs__Healthy')]
    br=br.sort_values('q_bh').head(12)
    tf=t[(t.family=='TF_EXPRESSION')&(t.lineage=='B cell')&(t.contrast=='UC_inflamed__vs__Healthy')].sort_values('q_bh').head(12)
    lines=['# SCP259 | Figure-extension / patient-level static clock analysis','',
      '**These are observational transcriptional data, not a TF perturbation study or measured rhythmicity.**','',
      '## Inputs and sample unit',f'- Frozen donor×state×lineage input: `{src}`',
      f'- {x.Subject.nunique()} unique donors; {len(x)} donor×state×lineage rows.',
      '- Five broad author-defined lineages, original six arms; no B-cell subtype analysis.',
      '- Healthy vs UC uses independent donors. Inflamed vs noninflamed uses **matched UC donors**.',
      '- Per-comparison Mann–Whitney U / paired Wilcoxon and 95% donor bootstrap CI.',
      '- BH FDR within each pre-defined contrast×feature family (both 3-contrast and per-contrast q provided).','',
      '## Six-arm and healthy-distance evidence',
      f'- Shape family significant (all comparisons): **{q("SHAPE")}/{t[(t.family=="SHAPE")].p.notna().sum()}**.',
      f'- Matched UC inflamed vs noninflamed: **{cm("UC_inflamed__vs__UC_noninflamed","ARM")} ARM**, **{cm("UC_inflamed__vs__UC_noninflamed","SHAPE")} SHAPE** q<0.05.',
      f'- Distance significance across three contrasts: **{q("DISTANCE")}** of {t[(t.family=="DISTANCE")].p.notna().sum()} tests.',
      '- Raw six-arm distance mixes global clock-expression displacement and arm configuration changes; SHAPE_DISTANCE_LOO is more specific to relative 6-arm configuration.',
      '- Healthy donor distances use leave-one-out reference to avoid including that donor in its own centroid.','']
    if v is not None:lines += [f'- B-cell inflamed vs Healthy EBOX_POS relative SHAPE: delta={v.delta:.4f}, p={v.p:.4g}, q={v.q_bh:.4g}, n={v.n_a}/{v.n_b}.','']
    lines += ['## TF and Hallmark/Reactome expression-program exploration',
       f'- {len(tfs)} TF expression features; {len(paths)} available gene-set scores.',
       '- `PATH_` values = average Healthy-reference gene z-score excluding CLOCK17. **Not NES, TF activity, flux, phosphoprotein signaling or measured pathway activation.**',
       '- `REL_PATH_` = pathway score minus each donor’s mean across the same available pathway panel. This is a sensitivity diagnostic for broad global upshifts, **not** a canonical GSVA/ssGSEA score.',
       '- `TF_SHAPE` and `PATH_SHAPE_RELATIVE` are **within-condition partial rank correlations** adjusted for overall six-arm score and log1p recovered cell count.',
       '- Approximate partial-correlation P-values and BH q are exploratory and cannot establish causal direction.',
       f'- UC inflamed vs Healthy: B-cell relative pathways q<0.05: {(br.q_bh<.05).sum()} among top-12; all evaluated B-cell sets={(t[(t.family=="PATHWAY_RELATIVE")&(t.lineage=="B cell")&(t.contrast=="UC_inflamed__vs__Healthy")&(t.q_bh<.05)]).shape[0]}.','',
       '### B-cell candidate TF differential expression (ranked exploratory)',
       '|TF|Delta inflamed - Healthy|BH q|','|---|---:|---:|']
    for r in tf.itertuples():lines.append(f'|{r.feature.removeprefix("TF_")}|{r.delta:.3f}|{r.q_bh:.4g}|')
    lines += ['','### B-cell relative pathway-score contrasts (top 12 by q; direction retained)',
             '|Pathway|Relative score delta|BH q|','|---|---:|---:|']
    for r in br.itertuples():lines.append(f'|{r.feature.removeprefix("REL_PATH_")}|{r.delta:.3f}|{r.q_bh:.4g}|')
    lines += ['','## Critical inference limits',
      '1. TF and pathway scores come from same pseudobulk expression data; coassociation does not prove transcriptional regulation, signal activation or mediation.',
      '2. A significant TF expression difference + arm difference + pathway difference are three parallel associations, not a causal TF→clock→pathway chain.',
      '3. Broad changes in nearly all Hallmark sets indicate global score shift; interpret relative-pathway results with their reference panel and gene overlap in mind.',
      '4. Disease/healthy comparisons are observational and not randomized, and donor-level cell recovery may reflect sampling composition.',
      '5. Healthy centroid is estimated from 12 controls and includes uncertainty, not a clinical predictor. No independent validation.',
      '', '## Deliverables',
      '- `01_ALL_DONOR_LEVEL_FEATURES_WITH_DISTANCES.csv`: donor scores with extra descriptors.',
      '- `02_ALL_CONTRASTS_EFFECT_CI_FDR.csv`: formal 3-group comparisons for arms, shapes, distances, TF, and raw/relative pathways.',
      '- `03_PAIRED_UC_DONOR_COVERAGE.csv`: matched-sample coverage.',
      '- `04_TF_AND_PATHWAY_ARM_SHAPE_ASSOCIATIONS.csv`: conditional partial correlations, multiple-testing adjusted.',
      '- `05_BCELL_TF_PATHWAY_ASSOCIATIONS.csv`: exploratory within-B-cell correlations.',
      '- `06_BCELL_HYPOTHESIS_CANDIDATES.csv`: all possible shortlisted descriptive triplets, NOT causal chains.',
      '- `06B_BCELL_CLINICALLY_INTERPRETABLE_CANDIDATES.csv`: predefined immune/proliferation subset; same whole-panel FDR.',
      '- `FIGURES`: individually exported panel A–F in PNG, SVG and PDF.',
      '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

def main():
    pa=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument('--scp-root',required=True)
    pa.add_argument('--outdir',required=True)
    pa.add_argument('--bootstrap',type=int,default=1500)
    pa.add_argument('--seed',type=int,default=20261017)
    pa.add_argument('--min-donors',type=int,default=8)
    pa.add_argument('--no-pdf',action='store_true')
    args=pa.parse_args()
    if args.bootstrap<100:pa.error('--bootstrap must be >=100')
    out=Path(args.outdir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    fig=out/'FIGURES';fig.mkdir(exist_ok=True)
    x,src,paths,tfs=read_sources(args)
    print(f'[SOURCE] {src} | {len(x)} donor×condition×lineage rows, {len(paths)} pathway scores, {len(tfs)} TF scores',flush=True)
    counts=matched_counts(x);counts.to_csv(out/'03_PAIRED_UC_DONOR_COVERAGE.csv',index=False)
    x.to_csv(out/'01_ALL_DONOR_LEVEL_FEATURES_WITH_DISTANCES.csv',index=False)
    t=tests_all(x,paths,tfs,args.bootstrap,args.seed)
    t.to_csv(out/'02_ALL_CONTRASTS_EFFECT_CI_FDR.csv',index=False)
    print(f'[CONTRASTS] {len(t)} contrasts, FDR computed',flush=True)
    ass=association_scan(x,tfs,paths,args.min_donors)
    ass.to_csv(out/'04_TF_AND_PATHWAY_ARM_SHAPE_ASSOCIATIONS.csv',index=False)
    tfpath=tf_path_bcell_links(x,tfs,paths) if paths else pd.DataFrame()
    tfpath.to_csv(out/'05_BCELL_TF_PATHWAY_ASSOCIATIONS.csv',index=False)
    cand=ordered_summary(t,ass,tfpath)
    cand.to_csv(out/'06_BCELL_HYPOTHESIS_CANDIDATES.csv',index=False)
    if not cand.empty:
        cand[cand.clinically_interpretable_pathway].to_csv(out/'06B_BCELL_CLINICALLY_INTERPRETABLE_CANDIDATES.csv',index=False)
    forest(t,fig,args.no_pdf)
    bcell_scores(x,t,fig,args.no_pdf)
    distance_figure(x,t,fig,args.no_pdf)
    tf_fig(t,ass,fig,args.no_pdf)
    pathway_fig(t,ass,paths,fig,args.no_pdf)
    paired_fig(t,fig,args.no_pdf)
    render_report(x,t,ass,cand,paths,tfs,out,src)
    info={'input':str(src),'sha256':hsh(src),'rows':len(x),'n_donors':x.Subject.nunique(),
          'TF_features':len(tfs),'pathways':len(paths),'bootstrap':args.bootstrap,
          'cross_condition_matching':'Healthy unpaired; UC inflamed/noninflamed paired donor ID',
          'feature_definitions':'Original six arms and study-specific Healthy-reference z, no raw data reload',
          'BH':'within each contrast/family; also BH across three contrasts/family',
          'gene_set_score':'mean healthy-referenced gene z, CLOCK17 excluded, NOT gene set activity',
          'relative_path_score':'per donor gene-set score minus mean of entire available gene-set panel',
          'shape':'arm - six-arm mean, no oscillation measured',
          'healthy_distance':'control LOO for Healthy, full Healthy centroid for UC cases',
          'status':'COMPLETE_COMPUTATIONAL_EXPLORATION_NOT_VALIDATED_CAUSAL',
          'scientific_limit':'No causal TF->six-arm->pathway or circadian phase inference'}
    (out/'RUN_SUMMARY.json').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[DONE]',out,flush=True)
    print('[B CELL SHAPE EBOX_POS]',t[(t.lineage=='B cell')&(t.feature=='SHAPE_EBOX_POS')][['contrast','delta','q_bh']].to_string(index=False),flush=True)

if __name__=='__main__':main()
