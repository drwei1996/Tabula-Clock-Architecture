#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
49_run_celltype_adjusted_tissue_DEG_enrichment.py

Primary question:
Within each current major lineage, which genes remain tissue-associated after
controlling original fine cell identity?

Primary gene model:
  reduced: expression ~ donor + original_cell_type + log1p(n_cells)
  full:    expression ~ donor + original_cell_type + tissue + log1p(n_cells)

Primary gene:
  BH-FDR < 0.05 AND incremental adjusted R2 from tissue >= 0.02

Support:
  original cell type >=2 tissues, >=3 donors, >=6 rows
  retained tissue >=3 donors
  gene prevalence >=20%

Pathway ORA:
  one-sided Fisher exact test on primary genes
  BH within lineage x collection
  primary enrichment = FDR<0.05 AND OR>=1.5 AND overlap>=5 genes

This is static-expression analysis only.
"""

import os
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
from scipy.stats import f as fdist, fisher_exact
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
FREEZE = ROOT / "00_freeze"
DEP = PROJECT / "results" / "phaseA_clock_architecture" / "dependencies"

META = FREEZE / "observation_metadata_corrected_v3_FROZEN.tsv"
NPZ = DEP / "broad_normalized_mean.npz"
HALLMARK = DEP / "h.all.v2026.1.Hs.symbols.gmt"
REACTOME = DEP / "c2.cp.reactome.v2026.1.Hs.symbols.gmt"

OUT = ROOT / "49_CELLTYPE_ADJUSTED_TISSUE_DEG_ENRICHMENT_20260922"
FIG = OUT / "figures"
TAB = OUT / "tables"
GENE = TAB / "gene_level"
PATH = TAB / "pathway_level"
SUP = TAB / "support"
AUD = OUT / "audit"
for d in [OUT, FIG, TAB, GENE, PATH, SUP, AUD]:
    d.mkdir(parents=True, exist_ok=True)

LINEAGES = ["b_cell","t_cell","macrophage_monocyte","fibroblast","endothelial"]
LABEL = {
    "b_cell":"B cell","t_cell":"T cell",
    "macrophage_monocyte":"Macrophage/Monocyte",
    "fibroblast":"Fibroblast","endothelial":"Endothelial",
}
COL = {
    "b_cell":"#4E79A7","t_cell":"#E15759",
    "macrophage_monocyte":"#59A14F",
    "fibroblast":"#F28E2B","endothelial":"#9467BD",
}
EXPECTED = {
    "b_cell":153,"t_cell":437,"macrophage_monocyte":280,
    "fibroblast":148,"endothelial":206,
}
SENS = [0.00,0.01,0.02,0.05,0.10]
EPS = 1e-12

mpl.rcParams.update({
    "font.family":"Arial","font.size":8,"axes.titlesize":11,
    "axes.labelsize":9,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":7,"axes.linewidth":0.8,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
    "savefig.dpi":600,"figure.facecolor":"white","axes.facecolor":"white",
})

def bh(p):
    p=np.asarray(p,float); q=np.ones_like(p)
    ok=np.isfinite(p); pv=np.clip(p[ok],0,1)
    if len(pv)==0: return q
    o=np.argsort(pv); r=pv[o]; n=len(r)
    a=r*n/np.arange(1,n+1); a=np.minimum.accumulate(a[::-1])[::-1]; a=np.clip(a,0,1)
    tmp=np.empty_like(a); tmp[o]=a; q[ok]=tmp
    return q

def basis(X):
    u,s,_=np.linalg.svd(np.asarray(X,float),full_matrices=False)
    if len(s)==0 or s[0]<=0: return np.empty((len(X),0)),0
    tol=s[0]*max(X.shape)*np.finfo(float).eps
    r=int((s>tol).sum())
    return u[:,:r],r

def sse(Y,Q):
    y2=np.sum(Y.astype(float)**2,axis=0)
    if Q.shape[1]==0: return y2
    z=Q.T@Y
    return np.maximum(y2-np.sum(z.astype(float)**2,axis=0),0)

def gmt(path, collection):
    out=[]
    with path.open("r",encoding="utf-8",errors="replace") as f:
        for line in f:
            z=line.rstrip("\n").split("\t")
            if len(z)>=3:
                out.append({
                    "collection":collection,"pathway":z[0],"description":z[1],
                    "genes":tuple(dict.fromkeys(g for g in z[2:] if g))
                })
    return out

def savefig(fig, name):
    for ext in ["png","pdf","svg"]:
        kw={"bbox_inches":"tight","pad_inches":0.12,"facecolor":"white"}
        if ext=="png": kw["dpi"]=600
        fig.savefig(FIG/f"{name}.{ext}",**kw)
    plt.close(fig)

def clean(ax):
    ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def load():
    for p in [META,NPZ,HALLMARK,REACTOME]:
        if not p.exists(): raise FileNotFoundError(p)
    m=pd.read_csv(META,sep="\t",low_memory=False)
    req={"group_id","donor_id","tissue","cell_type","priority_lineage_corrected","n_cells"}
    if req-set(m.columns): raise RuntimeError(f"Missing metadata columns: {req-set(m.columns)}")
    m["group_id"]=m["group_id"].astype(str)
    for c in ["donor_id","tissue","cell_type"]: m[c]=m[c].astype(str)
    m["n_cells"]=pd.to_numeric(m["n_cells"],errors="coerce")
    if len(m)!=2030: raise RuntimeError(f"Expected 2030 rows, found {len(m)}")
    obs=m.priority_lineage_corrected.dropna().value_counts().to_dict()
    if obs!=EXPECTED: raise RuntimeError(f"Frozen lineage counts changed: {obs}")
    d=np.load(NPZ,allow_pickle=True)
    ids=d["group_id"].astype(str); genes=d["genes"].astype(str); X=d["expression"]
    if X.shape!=(2030,59410): raise RuntimeError(f"Unexpected matrix shape {X.shape}")
    m=m.set_index("group_id")
    order=pd.Index(ids).get_indexer(m.index.astype(str))
    if (order<0).any(): raise RuntimeError("group_id alignment failed")
    X=X[order]
    if not np.isfinite(X).all(): raise RuntimeError("non-finite expression")
    return m,X,genes

def support_filter(m, lin, a):
    p=m[m.priority_lineage_corrected.eq(lin)].copy()
    s=(p.groupby("cell_type").agg(
        n_rows=("cell_type","size"),n_donors=("donor_id","nunique"),n_tissues=("tissue","nunique")
    ).reset_index())
    s["keep"]=(s.n_rows.ge(a.min_celltype_rows)&s.n_donors.ge(a.min_celltype_donors)&s.n_tissues.ge(a.min_celltype_tissues))
    p=p[p.cell_type.isin(set(s.loc[s.keep,"cell_type"]))].copy()
    ts=(p.groupby("tissue").donor_id.nunique().rename("n_donors").reset_index())
    ts["keep"]=ts.n_donors.ge(a.min_tissue_donors)
    p=p[p.tissue.isin(set(ts.loc[ts.keep,"tissue"]))].copy()
    # one re-check after tissue restriction
    s2=(p.groupby("cell_type").agg(
        n_rows=("cell_type","size"),n_donors=("donor_id","nunique"),n_tissues=("tissue","nunique")
    ).reset_index())
    s2["keep"]=(s2.n_rows.ge(a.min_celltype_rows)&s2.n_donors.ge(a.min_celltype_donors)&s2.n_tissues.ge(a.min_celltype_tissues))
    p=p[p.cell_type.isin(set(s2.loc[s2.keep,"cell_type"]))].copy()
    ts2=(p.groupby("tissue").donor_id.nunique().rename("n_donors").reset_index())
    p=p[p.tissue.isin(set(ts2.loc[ts2.n_donors.ge(a.min_tissue_donors),"tissue"]))].copy()
    s.to_csv(SUP/f"{lin}_celltype_support_initial.csv",index=False)
    s2.to_csv(SUP/f"{lin}_celltype_support_final.csv",index=False)
    ts2.to_csv(SUP/f"{lin}_tissue_support_final.csv",index=False)
    if p.empty or p.tissue.nunique()<2: raise RuntimeError(f"{lin}: insufficient supported rows")
    return p

def design(p, with_celltype):
    n=len(p); cols=[np.ones((n,1))]
    cols.append(pd.get_dummies(p.donor_id,drop_first=True,dtype=float).to_numpy())
    if with_celltype:
        cols.append(pd.get_dummies(p.cell_type,drop_first=True,dtype=float).to_numpy())
    z=np.log1p(p.n_cells.to_numpy(float)); z=(z-z.mean())[:,None]; cols.append(z)
    red=np.column_stack(cols)
    tis=pd.get_dummies(p.tissue,drop_first=True,dtype=float).to_numpy()
    full=np.column_stack(cols+[tis])
    return red,full

def gene_test(Y, genes, p, with_celltype, min_prev, min_delta):
    Xr,Xf=design(p,with_celltype)
    Qr,rr=basis(Xr); Qf,rf=basis(Xf)
    n=len(p); df1=rf-rr; df2=n-rf
    if df1<=0 or df2<=2: raise RuntimeError(f"Bad df: tissue={df1}, residual={df2}")
    sr=sse(Y,Qr); sf=sse(Y,Qf)
    mu=Y.astype(float).mean(0); st=np.sum((Y.astype(float)-mu[None,:])**2,axis=0)
    prev=(Y>0).mean(0); sd=Y.std(0,ddof=1)
    test=np.isfinite(sd)&(sd>EPS)&(prev>=min_prev)&np.isfinite(st)&(st>EPS)&(sr>EPS)
    F=np.full(len(genes),np.nan); P=np.ones(len(genes))
    mse=sf/df2; ok=test&np.isfinite(mse)&(mse>EPS)
    F[ok]=((np.maximum(sr[ok]-sf[ok],0)/df1)/mse[ok]); P[ok]=fdist.sf(F[ok],df1,df2)
    Q=bh(P)
    ar=np.full(len(genes),np.nan); af=np.full(len(genes),np.nan)
    ar[test]=1-(sr[test]/(n-rr))/(st[test]/(n-1))
    af[test]=1-(sf[test]/(n-rf))/(st[test]/(n-1))
    delta=af-ar
    resp=test&(Q<0.05)&np.isfinite(delta)&(delta>=min_delta)
    out=pd.DataFrame({
        "gene":genes,"testable":test,"prevalence":prev,"F_tissue":F,
        "df_tissue":df1,"df_residual":df2,"P_tissue":P,"FDR_tissue":Q,
        "adjR2_reduced":ar,"adjR2_full":af,"delta_adjR2_tissue":delta,
        "tissue_responsive_primary":resp,
    })
    info={"n_rows":n,"n_donors":p.donor_id.nunique(),"n_tissues":p.tissue.nunique(),
          "n_cell_types":p.cell_type.nunique(),"with_celltype":with_celltype,
          "rank_reduced":rr,"rank_full":rf,"df_tissue":df1,"df_residual":df2,
          "n_testable":int(test.sum()),"n_primary":int(resp.sum())}
    return out,info

def ora(gs, sets, collection, a):
    test=gs.testable.to_numpy(bool); resp=gs.tissue_responsive_primary.to_numpy(bool)
    names=gs.gene.astype(str).to_numpy(); U=set(names[test]); R=set(names[resp]); N=len(U); K=len(R)
    rows=[]
    for rec in sets:
        Mset=set(rec["genes"])&U; M=len(Mset)
        if M<a.min_pathway_genes:
            rows.append({"collection":collection,"pathway":rec["pathway"],"description":rec["description"],
                         "testable_pathway":False,"n_universe":N,"n_responsive":K,"n_pathway_testable":M,
                         "n_overlap_responsive":0,"odds_ratio_corrected":np.nan,"log2_odds_ratio_corrected":np.nan,
                         "P_enrichment":1.0,"overlap_genes":""})
            continue
        ov=sorted(Mset&R); aa=len(ov); b=M-aa; c=K-aa; d=N-aa-b-c
        try: _,pv=fisher_exact([[aa,b],[c,d]],alternative="greater")
        except Exception: pv=1.0
        OR=((aa+0.5)*(d+0.5))/((b+0.5)*(c+0.5))
        rows.append({"collection":collection,"pathway":rec["pathway"],"description":rec["description"],
                     "testable_pathway":True,"n_universe":N,"n_responsive":K,"n_pathway_testable":M,
                     "n_overlap_responsive":aa,"odds_ratio_corrected":OR,
                     "log2_odds_ratio_corrected":math.log2(OR) if OR>0 else np.nan,
                     "P_enrichment":pv,"overlap_genes":";".join(ov)})
    o=pd.DataFrame(rows); o["FDR_enrichment"]=bh(o.P_enrichment.to_numpy(float))
    o["FDR_supported"]=o.testable_pathway&o.FDR_enrichment.lt(0.05)
    o["primary_supported"]=o.FDR_supported&o.odds_ratio_corrected.ge(a.min_pathway_or)&o.n_overlap_responsive.ge(a.min_responsive_overlap)
    return o

def select_paths(df, collection, maxn, lookup=None, jac=0.5):
    z=df[df.collection.eq(collection)]
    s=(z.groupby("pathway").agg(
        n_primary=("primary_supported","sum"),n_fdr=("FDR_supported","sum"),
        max_log2OR=("log2_odds_ratio_corrected","max"),min_FDR=("FDR_enrichment","min")
    ).reset_index())
    c=s[s.n_primary.gt(0)].copy()
    if c.empty: c=s[s.n_fdr.gt(0)].copy()
    if c.empty: c=s.copy()
    c=c.sort_values(["n_primary","n_fdr","max_log2OR","min_FDR","pathway"],ascending=[False,False,False,True,True])
    if collection=="Hallmark": return c.head(maxn).pathway.tolist()
    out=[]
    for r in c.itertuples():
        g=lookup.get(r.pathway,())
        if all(len(set(g)&set(lookup.get(p,())))/max(1,len(set(g)|set(lookup.get(p,()))))<=jac for p in out):
            out.append(r.pathway)
        if len(out)>=maxn: break
    return out

def heatmap(df, selected, collection, name, title):
    if not selected: return
    z=df[df.collection.eq(collection)&df.pathway.isin(selected)]
    eff=z.pivot_table(index="pathway",columns="lineage",values="log2_odds_ratio_corrected",aggfunc="first")
    pri=z.pivot_table(index="pathway",columns="lineage",values="primary_supported",aggfunc="first")
    fdr=z.pivot_table(index="pathway",columns="lineage",values="FDR_supported",aggfunc="first")
    for lin in LINEAGES:
        if lin not in eff: eff[lin]=np.nan
        if lin not in pri: pri[lin]=False
        if lin not in fdr: fdr[lin]=False
    eff=eff[LINEAGES].reindex(selected); pri=pri[LINEAGES].reindex(selected).fillna(False); fdr=fdr[LINEAGES].reindex(selected).fillna(False)
    M=eff.to_numpy(float); finite=M[np.isfinite(M)]
    vmax=max(0.5,float(np.nanpercentile(np.abs(finite),98))); cmap=mpl.colormaps["RdBu_r"].copy(); cmap.set_bad("#F2F2F2")
    fig,ax=plt.subplots(figsize=(6.5,max(3.6,0.31*len(selected)+1.35)))
    im=ax.imshow(np.ma.masked_invalid(M),aspect="auto",interpolation="nearest",cmap=cmap,vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(5)); ax.set_xticklabels([LABEL[x] for x in LINEAGES],rotation=28,ha="right")
    labs=[p.replace("HALLMARK_","").replace("REACTOME_","").replace("_"," ") for p in selected]
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs,fontsize=6 if len(labs)<=20 else 5.5)
    P=pri.to_numpy(bool); Q=fdr.to_numpy(bool)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i,j]: ax.scatter(j,i,s=18,c="black",linewidths=0)
            elif Q[i,j]: ax.scatter(j,i,s=13,facecolors="none",edgecolors="black",linewidths=0.7)
    ax.set_title(title,loc="left",pad=8); ax.grid(False); ax.tick_params(axis="both",length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.025); cb.set_label("log2 enrichment odds ratio",fontsize=8)
    fig.text(0.995,0.01,"Filled: FDR<0.05 + OR≥1.5 + overlap≥5; open: FDR<0.05 only",ha="right",fontsize=6.2)
    savefig(fig,name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--min-celltype-tissues",type=int,default=2)
    ap.add_argument("--min-celltype-donors",type=int,default=3)
    ap.add_argument("--min-celltype-rows",type=int,default=6)
    ap.add_argument("--min-tissue-donors",type=int,default=3)
    ap.add_argument("--min-prevalence",type=float,default=0.20)
    ap.add_argument("--min-delta-adjr2",type=float,default=0.02)
    ap.add_argument("--min-pathway-genes",type=int,default=10)
    ap.add_argument("--min-pathway-or",type=float,default=1.50)
    ap.add_argument("--min-responsive-overlap",type=int,default=5)
    ap.add_argument("--max-hallmark-display",type=int,default=20)
    ap.add_argument("--max-reactome-display",type=int,default=20)
    ap.add_argument("--reactome-display-jaccard",type=float,default=0.50)
    a=ap.parse_args()

    print("="*110)
    print("CELL-TYPE-ADJUSTED TISSUE-RESPONSIVE GENE + PATHWAY ANALYSIS")
    print("="*110)
    print(f"Primary gene: FDR<0.05 + delta adjusted R2 >= {a.min_delta_adjr2}")
    print(f"Primary pathway: FDR<0.05 + OR >= {a.min_pathway_or} + overlap >= {a.min_responsive_overlap}")

    m,X,genes=load(); pos=pd.Series(np.arange(len(m)),index=m.index)
    H=gmt(HALLMARK,"Hallmark"); R=gmt(REACTOME,"Reactome")
    if len(H)!=50 or len(R)!=1839: raise RuntimeError(f"Unexpected GMT sizes: {len(H)}, {len(R)}")
    Rlookup={r["pathway"]:r["genes"] for r in R}

    burden=[]; compare=[]; sens=[]; designs=[]; allp=[]

    for lin in LINEAGES:
        print("\n"+"-"*110+f"\n[{lin}] filtering fine-identity support...\n"+"-"*110)
        p=support_filter(m,lin,a); idx=pos.loc[p.index].to_numpy(int); Y=X[idx].astype(np.float32,copy=False)
        print(f"[{lin}] rows={len(p)}, donors={p.donor_id.nunique()}, tissues={p.tissue.nunique()}, cell types={p.cell_type.nunique()}")

        unadj,du=gene_test(Y,genes,p,False,a.min_prevalence,a.min_delta_adjr2)
        adj,da=gene_test(Y,genes,p,True,a.min_prevalence,a.min_delta_adjr2)
        adj.insert(0,"lineage",lin)
        adj.to_csv(GENE/f"{lin}_celltype_adjusted_all_gene_tissue_effects.csv.gz",index=False,compression="gzip")
        adj[adj.tissue_responsive_primary].sort_values(["delta_adjR2_tissue","FDR_tissue"],ascending=[False,True]).to_csv(
            GENE/f"{lin}_PRIMARY_celltype_adjusted_tissue_responsive_genes.csv",index=False)

        nt=int(adj.testable.sum()); na=int(adj.tissue_responsive_primary.sum()); nu=int(unadj.tissue_responsive_primary.sum())
        burden.append({"lineage":lin,"lineage_label":LABEL[lin],"n_rows":len(p),"n_donors":p.donor_id.nunique(),
                       "n_tissues":p.tissue.nunique(),"n_original_cell_types":p.cell_type.nunique(),
                       "n_testable_genes":nt,"n_primary_adjusted_genes":na,
                       "adjusted_fraction_testable":na/nt if nt else np.nan,
                       "median_delta_adjR2_testable":adj.loc[adj.testable,"delta_adjR2_tissue"].median()})
        compare.append({"lineage":lin,"lineage_label":LABEL[lin],"n_testable_genes":nt,
                        "n_unadjusted_primary":nu,"n_celltype_adjusted_primary":na,
                        "unadjusted_fraction":nu/nt if nt else np.nan,"adjusted_fraction":na/nt if nt else np.nan})
        du.update({"lineage":lin,"model":"same_rows_without_celltype"}); da.update({"lineage":lin,"model":"PRIMARY_with_celltype"})
        designs += [du,da]
        for t in SENS:
            n=int((adj.testable&adj.FDR_tissue.lt(0.05)&adj.delta_adjR2_tissue.ge(t)).sum())
            sens.append({"lineage":lin,"lineage_label":LABEL[lin],"delta_adjR2_threshold":t,
                         "n_genes":n,"n_testable":nt,"fraction_testable":n/nt if nt else np.nan})
        print(f"[{lin}] testable={nt:,}; same-row unadjusted={nu:,}; adjusted={na:,}; tissue df={da['df_tissue']}; residual df={da['df_residual']}")

        h=ora(adj,H,"Hallmark",a); h.insert(0,"lineage",lin)
        r=ora(adj,R,"Reactome",a); r.insert(0,"lineage",lin)
        h.to_csv(PATH/f"{lin}_Hallmark_celltype_adjusted_ORA.csv",index=False)
        r.to_csv(PATH/f"{lin}_Reactome_celltype_adjusted_ORA.csv",index=False)
        allp += [h,r]
        print(f"[{lin}] Hallmark FDR-only={int(h.FDR_supported.sum())}, primary={int(h.primary_supported.sum())}")
        print(f"[{lin}] Reactome FDR-only={int(r.FDR_supported.sum())}, primary={int(r.primary_supported.sum())}")

    B=pd.DataFrame(burden); C=pd.DataFrame(compare); S=pd.DataFrame(sens); D=pd.DataFrame(designs); P=pd.concat(allp,ignore_index=True)
    B.to_csv(TAB/"01_PRIMARY_celltype_adjusted_gene_burden.csv",index=False)
    C.to_csv(TAB/"02_same_rows_before_after_celltype_adjustment.csv",index=False)
    S.to_csv(TAB/"03_delta_adjR2_threshold_sensitivity.csv",index=False)
    D.to_csv(TAB/"04_design_rank_and_df_audit.csv",index=False)
    P.to_csv(PATH/"ALL_LINEAGES_ALL_PATHWAYS_celltype_adjusted_ORA.csv.gz",index=False,compression="gzip")

    # Fig3E
    e=B.set_index("lineage").loc[LINEAGES].reset_index(); y=np.arange(5); pct=100*e.adjusted_fraction_testable.to_numpy()
    fig,ax=plt.subplots(figsize=(5.8,3.0))
    ax.barh(y,pct,height=0.68,color=[COL[x] for x in e.lineage],edgecolor="none")
    for i,r in enumerate(e.itertuples()):
        ax.text(100*r.adjusted_fraction_testable+0.25,i,f"{r.n_primary_adjusted_genes:,}/{r.n_testable_genes:,}",va="center",fontsize=7.2)
    ax.set_yticks(y); ax.set_yticklabels(e.lineage_label); ax.invert_yaxis()
    ax.set_xlabel("Cell-identity-adjusted tissue-responsive genes (% of testable genes)")
    ax.set_title("Tissue-associated expression differences after controlling original cell identity",loc="left",pad=7)
    ax.set_xlim(0,min(100,max(pct)*1.25+2)); clean(ax); savefig(fig,"Fig3E_celltype_adjusted_tissue_responsive_gene_burden")

    hs=select_paths(P,"Hallmark",a.max_hallmark_display)
    pd.DataFrame({"display_order":range(1,len(hs)+1),"pathway":hs}).to_csv(TAB/"Fig3F_selected_Hallmark.csv",index=False)
    heatmap(P,hs,"Hallmark","Fig3F_Hallmark_celltype_adjusted_tissue_enrichment",
            "Hallmark enrichment after controlling original cell identity")

    rs=select_paths(P,"Reactome",a.max_reactome_display,Rlookup,a.reactome_display_jaccard)
    pd.DataFrame({"display_order":range(1,len(rs)+1),"pathway":rs}).to_csv(TAB/"Fig3G_selected_Reactome.csv",index=False)
    heatmap(P,rs,"Reactome","Fig3G_Reactome_celltype_adjusted_tissue_enrichment",
            "Representative Reactome enrichment after controlling original cell identity")

    # Supp before/after
    c=C.set_index("lineage").loc[LINEAGES].reset_index(); xloc=np.arange(5); w=0.35
    fig,ax=plt.subplots(figsize=(6.1,3.1))
    ax.bar(xloc-w/2,100*c.unadjusted_fraction,width=w,color="#C7CDD4",edgecolor="none",label="Without cell-type adjustment")
    ax.bar(xloc+w/2,100*c.adjusted_fraction,width=w,color=[COL[x] for x in c.lineage],edgecolor="none",label="With cell-type adjustment")
    ax.set_xticks(xloc); ax.set_xticklabels(c.lineage_label,rotation=28,ha="right")
    ax.set_ylabel("Primary tissue-responsive genes (% of testable genes)")
    ax.set_title("Effect of controlling original cell identity",loc="left",pad=7)
    ax.legend(frameon=False); clean(ax); savefig(fig,"Supp_same_rows_before_after_celltype_adjustment")

    # Supp effect sensitivity
    fig,ax=plt.subplots(figsize=(5.8,3.2))
    for lin in LINEAGES:
        z=S[S.lineage.eq(lin)].sort_values("delta_adjR2_threshold")
        ax.plot(z.delta_adjR2_threshold,100*z.fraction_testable,marker="o",linewidth=1.4,markersize=4,color=COL[lin],label=LABEL[lin])
    ax.axvline(a.min_delta_adjr2,ls="--",lw=0.9,color="#666666")
    ax.set_xlabel("Minimum incremental adjusted R² from tissue")
    ax.set_ylabel("FDR-supported genes (% of testable genes)")
    ax.set_title("Sensitivity to tissue-effect size threshold",loc="left",pad=7)
    ax.legend(frameon=False); clean(ax); savefig(fig,"Supp_delta_adjR2_threshold_sensitivity")

    summary=(P.groupby(["lineage","collection"]).agg(
        n_tested=("testable_pathway","sum"),n_FDR_only=("FDR_supported","sum"),
        n_primary=("primary_supported","sum"),median_log2OR=("log2_odds_ratio_corrected","median")
    ).reset_index())
    summary.to_csv(TAB/"05_pathway_summary.csv",index=False)

    manifest={
        "primary_model":{"reduced":"expression ~ donor + original_cell_type + log1p(n_cells)",
                         "full":"expression ~ donor + original_cell_type + tissue + log1p(n_cells)",
                         "gene_rule":f"FDR<0.05 and delta adjusted R2 >= {a.min_delta_adjr2}"},
        "support":{"min_celltype_tissues":a.min_celltype_tissues,"min_celltype_donors":a.min_celltype_donors,
                   "min_celltype_rows":a.min_celltype_rows,"min_tissue_donors":a.min_tissue_donors,
                   "min_prevalence":a.min_prevalence},
        "pathway_rule":{"FDR":0.05,"min_OR":a.min_pathway_or,"min_overlap":a.min_responsive_overlap},
        "static_expression_only":True,"clock_distance_used":False,"strict_A4_7_modified":False,
    }
    (AUD/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

    print("\n"+"="*110+"\nANALYSIS COMPLETE\n"+"="*110)
    print("\nPRIMARY GENE BURDEN")
    print(B[["lineage_label","n_rows","n_donors","n_tissues","n_original_cell_types","n_testable_genes",
             "n_primary_adjusted_genes","adjusted_fraction_testable","median_delta_adjR2_testable"]].to_string(
        index=False,formatters={"adjusted_fraction_testable":lambda x:f"{100*x:.1f}%",
                                "median_delta_adjR2_testable":lambda x:f"{x:.4f}"}))
    print("\nBEFORE/AFTER CELL-TYPE ADJUSTMENT")
    print(C.to_string(index=False,formatters={"unadjusted_fraction":lambda x:f"{100*x:.1f}%",
                                             "adjusted_fraction":lambda x:f"{100*x:.1f}%"}))
    print("\nPATHWAY SUMMARY")
    print(summary.to_string(index=False))
    print("\nMAIN FIGURES")
    print(FIG/"Fig3E_celltype_adjusted_tissue_responsive_gene_burden.png")
    print(FIG/"Fig3F_Hallmark_celltype_adjusted_tissue_enrichment.png")
    print(FIG/"Fig3G_Reactome_celltype_adjusted_tissue_enrichment.png")
    print("="*110)

if __name__=="__main__":
    main()
