#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
50_run_matched_competitive_tissue_pathway_enrichment.py

Primary analysis:
Use the continuous original-cell-type-adjusted tissue effect from script 49
(delta_adjR2_tissue), rather than DEG/non-DEG thresholding.

For each lineage:
- rank all testable non-clock genes by delta_adjR2_tissue
- convert ranks to inverse-normal scores
- test each pathway against expression-matched background genes
- matching variables: atlas-wide mean, SD, prevalence (4 quantile bins each)
- pathway null preserves its matching-bin composition

Primary pathway support:
    matched-competitive FDR < 0.05 and z > 0

Expanded17 clock genes are excluded from both pathway scoring and background.
Static-expression analysis only; no rhythm/phase/causal claim.
"""

import os
from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
PREV = ROOT / "49_CELLTYPE_ADJUSTED_TISSUE_DEG_ENRICHMENT_20260922"
GENE_DIR = PREV / "tables" / "gene_level"
DEP = PROJECT / "results" / "phaseA_clock_architecture" / "dependencies"

NPZ = DEP / "broad_normalized_mean.npz"
HALLMARK = DEP / "h.all.v2026.1.Hs.symbols.gmt"
REACTOME = DEP / "c2.cp.reactome.v2026.1.Hs.symbols.gmt"

OUT = ROOT / "50_MATCHED_COMPETITIVE_TISSUE_PATHWAYS_20260922"
FIG = OUT / "figures"
TAB = OUT / "tables"
AUD = OUT / "audit"
for d in [OUT, FIG, TAB, AUD]:
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
CLOCK17 = {
    "ARNTL","CLOCK","NPAS2","PER1","PER2","PER3","CRY1","CRY2",
    "NR1D1","NR1D2","RORA","RORB","RORC","DBP","TEF","HLF","NFIL3"
}
EPS = 1e-12

mpl.rcParams.update({
    "font.family":"Arial","font.size":8,"axes.titlesize":11,
    "axes.labelsize":9,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":7,"axes.linewidth":0.8,"pdf.fonttype":42,
    "ps.fonttype":42,"svg.fonttype":"none","savefig.dpi":600,
    "figure.facecolor":"white","axes.facecolor":"white",
})

def bh(p):
    p=np.asarray(p,float); q=np.ones_like(p); ok=np.isfinite(p); pv=np.clip(p[ok],0,1)
    if len(pv)==0: return q
    o=np.argsort(pv); r=pv[o]; n=len(r)
    a=r*n/np.arange(1,n+1); a=np.minimum.accumulate(a[::-1])[::-1]; a=np.clip(a,0,1)
    t=np.empty_like(a); t[o]=a; q[ok]=t
    return q

def qbins(v,n):
    r=pd.Series(np.asarray(v,float)).rank(method="average",pct=True).to_numpy()
    return np.clip(np.floor(r*n).astype(np.int16),0,n-1)

def gmt(path,collection):
    out=[]
    with path.open("r",encoding="utf-8",errors="replace") as f:
        for line in f:
            z=line.rstrip("\n").split("\t")
            if len(z)>=3:
                out.append({"collection":collection,"pathway":z[0],"description":z[1],
                            "genes":tuple(dict.fromkeys(x for x in z[2:] if x))})
    return out

def save(fig,name):
    for ext in ["png","pdf","svg"]:
        kw={"bbox_inches":"tight","pad_inches":0.12,"facecolor":"white"}
        if ext=="png": kw["dpi"]=600
        fig.savefig(FIG/f"{name}.{ext}",**kw)
    plt.close(fig)

def clean(ax):
    ax.grid(False); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def short(x):
    return str(x).replace("HALLMARK_","").replace("REACTOME_","").replace("_"," ")

def jaccard(a,b):
    a=set(a); b=set(b)
    return len(a&b)/len(a|b) if (a or b) else 0.0

def global_chars(nbins):
    d=np.load(NPZ,allow_pickle=True)
    genes=d["genes"].astype(str); X=d["expression"]
    if X.shape[1]!=59410: raise RuntimeError(f"Unexpected expression shape {X.shape}")
    mean=X.mean(0).astype(float); sd=X.std(0,ddof=1).astype(float); prev=(X>0).mean(0).astype(float)
    variable=np.isfinite(mean)&np.isfinite(sd)&(sd>EPS)&np.isfinite(prev)
    b1=qbins(mean,nbins); b2=qbins(sd,nbins); b3=qbins(prev,nbins)
    code=((b1.astype(int)*nbins+b2.astype(int))*nbins+b3.astype(int))
    return pd.DataFrame({
        "gene":genes,"global_mean":mean,"global_sd":sd,"global_prevalence":prev,
        "match_code":code,"global_variable":variable,
        "is_clock":[g in CLOCK17 for g in genes]
    })

def lineage_table(lin,G):
    p=GENE_DIR/f"{lin}_celltype_adjusted_all_gene_tissue_effects.csv.gz"
    if not p.exists(): raise FileNotFoundError(f"Run script 49 first; missing {p}")
    x=pd.read_csv(p,low_memory=False)
    x["gene"]=x["gene"].astype(str)
    x["testable"]=x["testable"].astype(str).str.lower().isin(["true","1","t","yes"])
    x["delta_adjR2_tissue"]=pd.to_numeric(x["delta_adjR2_tissue"],errors="coerce")
    x=x.merge(G,on="gene",how="left",validate="one_to_one")
    x["universe"]=x.testable & x.global_variable.fillna(False) & ~x.is_clock.fillna(False) & np.isfinite(x.delta_adjR2_tissue)
    idx=x.universe.to_numpy(bool)
    vals=x.loc[idx,"delta_adjR2_tissue"].to_numpy(float)
    ranks=pd.Series(vals).rank(method="average").to_numpy(float)
    u=(ranks-0.5)/len(ranks)
    x["rank_pct"]=np.nan; x["score"]=np.nan
    x.loc[idx,"rank_pct"]=u
    x.loc[idx,"score"]=norm.ppf(np.clip(u,1e-6,1-1e-6))
    return x

def code_coord(code,n):
    return np.array([code//(n*n),(code%(n*n))//n,code%n],int)

def get_pool(code,target_mask,pools,coords,needed,min_extra):
    exact=pools.get(int(code),np.array([],int))
    exact=exact[~target_mask[exact]]
    if len(exact)>=needed+min_extra: return exact,0
    tc=coords[int(code)]
    dist=np.abs(coords-tc[None,:]).sum(1)
    gather=[]; maxd=0
    for c in np.argsort(dist,kind="stable"):
        p=pools.get(int(c))
        if p is None: continue
        p=p[~target_mask[p]]
        if len(p):
            gather.append(p); maxd=int(dist[c])
        if sum(len(a) for a in gather)>=needed+min_extra: break
    return (np.unique(np.concatenate(gather)),maxd) if gather else (np.array([],int),np.nan)

def competitive(tab,sets,collection,nbins,min_genes,min_extra):
    U=tab[tab.universe]
    gene2idx={g:int(i) for i,g in zip(U.index,U.gene)}
    pools={int(c):p.index.to_numpy(int) for c,p in U.groupby("match_code")}
    coords=np.vstack([code_coord(i,nbins) for i in range(nbins**3)])
    scores=tab.score.to_numpy(float); codes=tab.match_code.to_numpy()
    rows=[]
    for rec in sets:
        tg=[g for g in rec["genes"] if g not in CLOCK17 and g in gene2idx]
        tidx=np.array([gene2idx[g] for g in tg],int)
        if len(tidx)<min_genes:
            rows.append({"collection":collection,"pathway":rec["pathway"],"description":rec["description"],
                         "testable_pathway":False,"n_genes":len(tidx),"observed_mean_score":np.nan,
                         "observed_mean_rank_pct":np.nan,"null_mean":np.nan,"null_sd":np.nan,
                         "matched_z":np.nan,"P_matched":1.0,"max_match_bin_distance":np.nan,
                         "genes_scored":";".join(tg)})
            continue
        mask=np.zeros(len(tab),bool); mask[tidx]=True
        uc,cnt=np.unique(codes[tidx],return_counts=True)
        obs=float(np.mean(scores[tidx])); obs_pct=float(np.mean(tab.loc[tidx,"rank_pct"].to_numpy(float)))
        sm=0.0; sv=0.0; maxdist=0; valid=True
        for code,n_take in zip(uc,cnt):
            pool,dist=get_pool(int(code),mask,pools,coords,int(n_take),min_extra)
            vals=scores[pool]; vals=vals[np.isfinite(vals)]
            N=len(vals); n=int(n_take)
            if N<n or N==0:
                valid=False; break
            mu=float(vals.mean()); var=float(vals.var(ddof=0))
            sm += n*mu
            if N>1:
                sv += n*var*(N-n)/(N-1)
            if np.isfinite(dist): maxdist=max(maxdist,int(dist))
        m=len(tidx)
        if valid:
            null_mean=sm/m; null_sd=math.sqrt(max(sv/(m*m),0))
            z=(obs-null_mean)/null_sd if null_sd>EPS else np.nan
            p=float(norm.sf(z)) if np.isfinite(z) else 1.0
        else:
            null_mean=null_sd=z=np.nan; p=1.0
        rows.append({"collection":collection,"pathway":rec["pathway"],"description":rec["description"],
                     "testable_pathway":valid,"n_genes":m,"observed_mean_score":obs,
                     "observed_mean_rank_pct":obs_pct,"null_mean":null_mean,"null_sd":null_sd,
                     "matched_z":z,"P_matched":p,"max_match_bin_distance":maxdist if valid else np.nan,
                     "genes_scored":";".join(tg)})
    out=pd.DataFrame(rows)
    out["FDR_matched"]=bh(out.P_matched.to_numpy(float))
    out["supported"]=out.testable_pathway & out.FDR_matched.lt(0.05) & out.matched_z.gt(0)
    return out

def select_hall(R,maxn):
    x=R[R.collection.eq("Hallmark")]
    s=x.groupby("pathway").agg(n_supported=("supported","sum"),max_z=("matched_z","max"),min_fdr=("FDR_matched","min")).reset_index()
    h=s[s.n_supported.gt(0)].copy()
    if h.empty: h=s.copy()
    return h.sort_values(["n_supported","max_z","min_fdr","pathway"],ascending=[False,False,True,True]).head(maxn).pathway.tolist()

def select_react(R,lookup,maxn,jmax):
    x=R[R.collection.eq("Reactome")]
    s=x.groupby("pathway").agg(n_supported=("supported","sum"),max_z=("matched_z","max"),min_fdr=("FDR_matched","min")).reset_index()
    c=s[s.n_supported.gt(0)].copy()
    if c.empty: c=s.copy()
    c=c.sort_values(["n_supported","max_z","min_fdr","pathway"],ascending=[False,False,True,True])
    out=[]
    for r in c.itertuples():
        if all(jaccard(lookup.get(r.pathway,()),lookup.get(p,()))<=jmax for p in out):
            out.append(r.pathway)
        if len(out)>=maxn: break
    return out

def heatmap(R,selected,collection,name,title):
    z=R[R.collection.eq(collection)&R.pathway.isin(selected)]
    M=z.pivot_table(index="pathway",columns="lineage",values="matched_z",aggfunc="first")
    Q=z.pivot_table(index="pathway",columns="lineage",values="FDR_matched",aggfunc="first")
    for lin in LINEAGES:
        if lin not in M: M[lin]=np.nan
        if lin not in Q: Q[lin]=np.nan
    M=M[LINEAGES].reindex(selected); Q=Q[LINEAGES].reindex(selected)
    A=M.to_numpy(float); finite=A[np.isfinite(A)]
    if len(finite)==0: return
    vmax=max(2.0,float(np.nanpercentile(np.abs(finite),98)))
    cmap=mpl.colormaps["RdBu_r"].copy(); cmap.set_bad("#F2F2F2")
    fig,ax=plt.subplots(figsize=(6.5,max(3.8,0.31*len(selected)+1.35)))
    im=ax.imshow(np.ma.masked_invalid(A),aspect="auto",interpolation="nearest",cmap=cmap,vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(5)); ax.set_xticklabels([LABEL[x] for x in LINEAGES],rotation=28,ha="right")
    ax.set_yticks(range(len(selected))); ax.set_yticklabels([short(p) for p in selected],fontsize=6 if len(selected)<=20 else 5.5)
    q=Q.to_numpy(float)
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if np.isfinite(q[i,j]) and q[i,j]<0.05 and np.isfinite(A[i,j]) and A[i,j]>0:
                ax.scatter(j,i,s=15,c="black",linewidths=0)
    ax.set_title(title,loc="left",pad=8); ax.grid(False); ax.tick_params(axis="both",length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    cb=fig.colorbar(im,ax=ax,fraction=0.035,pad=0.025); cb.set_label("Matched competitive enrichment z",fontsize=8)
    fig.text(0.995,0.01,"Black dot: matched-competitive FDR < 0.05",ha="right",fontsize=6.4)
    save(fig,name)

def plot_E(tabs):
    rows=[]
    for lin,t in tabs.items():
        x=t.loc[t.universe,"delta_adjR2_tissue"].to_numpy(float)
        rows.append({"lineage":lin,"lineage_label":LABEL[lin],"median":np.nanmedian(x),
                     "q25":np.nanquantile(x,.25),"q75":np.nanquantile(x,.75),
                     "q90":np.nanquantile(x,.90),"n_genes":len(x)})
    s=pd.DataFrame(rows).set_index("lineage").loc[LINEAGES].reset_index()
    s.to_csv(TAB/"Fig3E_continuous_gene_tissue_effect_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.9,3.1)); y=np.arange(5)
    for i,r in enumerate(s.itertuples()):
        ax.plot([r.q25,r.q75],[i,i],lw=5,solid_capstyle="butt",color=COL[r.lineage])
        ax.scatter(r.median,i,s=34,c="white",edgecolor=COL[r.lineage],lw=1.2,zorder=3)
        ax.scatter(r.q90,i,s=28,c=COL[r.lineage],edgecolor="white",lw=.5,zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(s.lineage_label); ax.invert_yaxis()
    ax.set_xlabel("Incremental adjusted R² attributable to tissue")
    ax.set_title("Continuous tissue-effect magnitude after controlling original cell identity",loc="left",pad=7)
    clean(ax)
    ax.legend(handles=[
        Line2D([0],[0],marker="o",color="none",markerfacecolor="white",markeredgecolor="#555",markersize=5,label="Median"),
        Line2D([0],[0],lw=5,color="#999",label="IQR"),
        Line2D([0],[0],marker="o",color="none",markerfacecolor="#555",markeredgecolor="white",markersize=5,label="90th percentile")
    ],frameon=False,loc="best")
    save(fig,"Fig3E_continuous_celltype_adjusted_tissue_effect")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--match-bins",type=int,default=4)
    ap.add_argument("--min-pathway-genes",type=int,default=10)
    ap.add_argument("--min-extra-match-genes",type=int,default=20)
    ap.add_argument("--max-hallmark-display",type=int,default=20)
    ap.add_argument("--max-reactome-display",type=int,default=20)
    ap.add_argument("--reactome-display-jaccard",type=float,default=.50)
    a=ap.parse_args()

    print("="*112)
    print("MATCHED-COMPETITIVE CONTINUOUS TISSUE-EFFECT PATHWAY ANALYSIS")
    print("="*112)
    print("Primary support: matched-competitive FDR<0.05 and z>0")
    print(f"Matching: mean + SD + prevalence, {a.match_bins} quantile bins/dimension")
    print("Expanded17 clock genes excluded from pathways and background.")

    G=global_chars(a.match_bins)
    H=gmt(HALLMARK,"Hallmark"); RR=gmt(REACTOME,"Reactome")
    if len(H)!=50 or len(RR)!=1839: raise RuntimeError(f"Unexpected GMT sizes {len(H)}, {len(RR)}")
    lookup={r["pathway"]:tuple(g for g in r["genes"] if g not in CLOCK17) for r in RR}

    tabs={}; allres=[]
    for lin in LINEAGES:
        print("\n"+"-"*112+f"\n[{lin}] loading script49 gene effects...\n"+"-"*112)
        t=lineage_table(lin,G); tabs[lin]=t
        print(f"[{lin}] testable non-clock universe={int(t.universe.sum()):,}")
        h=competitive(t,H,"Hallmark",a.match_bins,a.min_pathway_genes,a.min_extra_match_genes); h.insert(0,"lineage",lin)
        r=competitive(t,RR,"Reactome",a.match_bins,a.min_pathway_genes,a.min_extra_match_genes); r.insert(0,"lineage",lin)
        h.to_csv(TAB/f"{lin}_Hallmark_matched_competitive.csv",index=False)
        r.to_csv(TAB/f"{lin}_Reactome_matched_competitive.csv",index=False)
        allres += [h,r]
        print(f"[{lin}] Hallmark supported={int(h.supported.sum())}/{int(h.testable_pathway.sum())}")
        print(f"[{lin}] Reactome supported={int(r.supported.sum())}/{int(r.testable_pathway.sum())}")

    R=pd.concat(allres,ignore_index=True)
    R.to_csv(TAB/"ALL_LINEAGES_matched_competitive_pathways.csv.gz",index=False,compression="gzip")

    plot_E(tabs)
    hs=select_hall(R,a.max_hallmark_display)
    pd.DataFrame({"display_order":range(1,len(hs)+1),"pathway":hs}).to_csv(TAB/"Fig3F_selected_Hallmark.csv",index=False)
    heatmap(R,hs,"Hallmark","Fig3F_Hallmark_matched_competitive_tissue_enrichment",
            "Hallmark programs with tissue effects beyond matched gene background")

    rs=select_react(R,lookup,a.max_reactome_display,a.reactome_display_jaccard)
    pd.DataFrame({"display_order":range(1,len(rs)+1),"pathway":rs}).to_csv(TAB/"Fig3G_selected_Reactome.csv",index=False)
    heatmap(R,rs,"Reactome","Fig3G_Reactome_matched_competitive_tissue_enrichment",
            "Representative Reactome programs with tissue effects beyond matched background")

    S=R.groupby(["lineage","collection"]).agg(
        n_testable=("testable_pathway","sum"),n_supported=("supported","sum"),
        median_z=("matched_z","median"),median_match_distance=("max_match_bin_distance","median")
    ).reset_index()
    S["supported_fraction"]=S.n_supported/S.n_testable
    S.to_csv(TAB/"01_matched_competitive_hit_summary.csv",index=False)

    A=R[R.testable_pathway].groupby(["lineage","collection"]).agg(
        median_match_distance=("max_match_bin_distance","median"),
        max_match_distance=("max_match_bin_distance","max"),
        fraction_exact_match=("max_match_bin_distance",lambda x:float(np.mean(np.asarray(x,float)==0))),
        median_pathway_genes=("n_genes","median")
    ).reset_index()
    A.to_csv(AUD/"match_quality_summary.csv",index=False)

    manifest={
        "analysis":"matched-competitive enrichment of continuous cell-type-adjusted tissue effect",
        "gene_score":"delta_adjR2_tissue from script49 -> within-lineage inverse-normal rank",
        "matching":["atlas-wide mean expression","atlas-wide expression SD","atlas-wide prevalence"],
        "clock_genes_excluded":sorted(CLOCK17),
        "primary_rule":"FDR_matched < 0.05 and matched_z > 0",
        "null":"stratified finite-population competitive null preserving pathway matching-bin composition",
        "reactome_redundancy":"display-only Jaccard filtering; statistics unchanged",
        "nonclaims":["not DEG ORA","not pathway-clock correspondence","not clock-specificity","not rhythmicity","not causal"]
    }
    (AUD/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

    print("\n"+"="*112+"\nANALYSIS COMPLETE\n"+"="*112)
    print("\nHIT SUMMARY")
    print(S.to_string(index=False,formatters={
        "supported_fraction":lambda x:f"{100*x:.1f}%",
        "median_z":lambda x:f"{x:.3f}",
        "median_match_distance":lambda x:f"{x:.1f}"
    }))
    print("\nMATCH QUALITY")
    print(A.to_string(index=False,formatters={"fraction_exact_match":lambda x:f"{100*x:.1f}%"}))
    print("\nDIAGNOSTIC")
    for r in S.itertuples():
        state="SATURATED" if r.supported_fraction>=.80 else ("VERY SPARSE" if r.supported_fraction<=.01 else "INTERMEDIATE")
        print(f"{r.lineage:24s} {r.collection:9s}: {int(r.n_supported):4d}/{int(r.n_testable):4d} ({100*r.supported_fraction:5.1f}%) {state}")
    print("\nMAIN FIGURES")
    print(FIG/"Fig3E_continuous_celltype_adjusted_tissue_effect.png")
    print(FIG/"Fig3F_Hallmark_matched_competitive_tissue_enrichment.png")
    print(FIG/"Fig3G_Reactome_matched_competitive_tissue_enrichment.png")
    print("="*112)

if __name__=="__main__":
    main()
