
# -*- coding: utf-8 -*-
"""
79_run_78_1_correction_and_TF_discovery.py

Integrated follow-up:
A) replace saturated interaction R2 with cross-validated interaction evidence
B) screen TF expression / optional DoRothEA activity against six-arm architecture

No old outputs are modified.
"""

from __future__ import annotations
import os
import itertools, json, math, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
OUT = ROOT / "79_78_1_CORRECTION_AND_TF_DISCOVERY_20261004"
AUDIT = OUT / "00_AUDIT"
CORR = OUT / "01_78_1_CORRECTION"
TFDIR = OUT / "02_TF_DISCOVERY"
FIG = OUT / "03_FIGURES"
REPORT = OUT / "04_REPORT"
INPUTS = OUT / "00_INPUTS"
for p in [OUT,AUDIT,CORR,TFDIR,FIG,REPORT,INPUTS]: p.mkdir(parents=True, exist_ok=True)

META = ROOT / "00_freeze" / "observation_metadata_corrected_v3_FROZEN.tsv"
NPZ = PROJECT / "results" / "phaseA_clock_architecture" / "dependencies" / "broad_normalized_mean.npz"
RUN78 = ROOT / "78_LINEAGE_INTRINSIC_3LOOP_VALIDATION"
TABULA6 = RUN78 / "01_TABLES" / "Tabula_DONOR_TISSUE_LINEAGE_SIXARM.csv"
HCL6 = RUN78 / "01_TABLES" / "HCL_DONOR_TISSUE_LINEAGE_SIXARM.csv"
AHCA6 = RUN78 / "01_TABLES" / "AHCA_DONOR_TISSUE_LINEAGE_SIXARM.csv"
GSE292 = ROOT / "76_GSE292219_CLOCK_TIME_VS_LINEAGE_STORY" / "01_SOURCE_DATA" / "76_LOOP_ARM_AND_BALANCE_SCORES.csv"
DOROTHEA = INPUTS / "79_DOROTHEA_HS_ABC.csv"

LINEAGES = ["b_cell","t_cell","macrophage_monocyte","fibroblast","endothelial"]
ARMS = ["EBOX_POS","EBOX_NEG","RORE_ACT","RORE_REP","DBOX_ACT","DBOX_REP"]
ARM_GENES = {
    "EBOX_POS":["ARNTL","CLOCK","NPAS2"],
    "EBOX_NEG":["PER1","PER2","PER3","CRY1","CRY2"],
    "RORE_ACT":["RORA","RORB","RORC"],
    "RORE_REP":["NR1D1","NR1D2"],
    "DBOX_ACT":["DBP","TEF","HLF"],
    "DBOX_REP":["NFIL3"],
}
CLOCK17 = sorted({g for v in ARM_GENES.values() for g in v})
N_PERM = 500
SEED = 20261004
EPS = 1e-12

FALLBACK_TFS = sorted(set("""
SPI1 CEBPA CEBPB IRF1 IRF4 IRF7 IRF8 STAT1 STAT3 STAT5A STAT5B
RELA RELB NFKB1 NFKB2 JUN JUNB JUND FOS FOSB FOSL1 FOSL2 ATF3 ATF4
CREB1 ETS1 ETS2 ERG FLI1 GATA2 GATA3 RUNX1 RUNX3 TBX21 EOMES BCL6
PRDM1 PAX5 IKZF1 IKZF3 TCF3 LEF1 FOXO1 FOXO3 FOXA1 FOXA2 FOXC1
FOXC2 KLF2 KLF4 KLF5 KLF6 KLF9 TEAD1 TEAD2 TEAD3 TEAD4 YAP1 WWTR1
SMAD1 SMAD2 SMAD3 SMAD4 SMAD5 HIF1A EPAS1 ARNT MYC MAX SREBF1
SREBF2 PPARG PPARD PPARA RXRA ESR1 AR NR3C1 SOX4 SOX9 SOX17 SOX18
TWIST1 SNAI1 SNAI2 ZEB1 ZEB2 MEF2A MEF2C SRF XBP1 HSF1 NFE2L2
""".split()))

def log(x):
    s=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {x}"
    print(s,flush=True)
    with (OUT/"79_console.log").open("a",encoding="utf-8") as f:f.write(s+"\n")

def bh(p):
    p=np.asarray(p,float); q=np.full_like(p,np.nan); ok=np.isfinite(p)
    x=p[ok]
    if not len(x): return q
    o=np.argsort(x); z=x[o]*len(x)/np.arange(1,len(x)+1)
    z=np.minimum.accumulate(z[::-1])[::-1]; z=np.clip(z,0,1)
    back=np.empty(len(x)); back[o]=z; q[ok]=back; return q

def norm_lineage(x):
    s=str(x).strip()
    if s in LINEAGES:return s
    z=s.lower().replace("/","_").replace(" ","_")
    if z in LINEAGES:return z
    if "macrophage" in z or "monocyte" in z:return "macrophage_monocyte"
    if "fibro" in z:return "fibroblast"
    if "endothel" in z:return "endothelial"
    if z in {"b_cell","b_cells","bcell"}:return "b_cell"
    if z in {"t_cell","t_cells","tcell"}:return "t_cell"
    return None

def savefig(fig,name):
    for ext in ["png","pdf"]:
        fig.savefig(FIG/f"{name}.{ext}",dpi=600,bbox_inches="tight",facecolor="white")
    plt.close(fig)

# ------------------------------------------------------------------
# A1. static donor-held-out interaction CV
# ------------------------------------------------------------------
def cv_static(name,path):
    d=pd.read_csv(path)
    d=d.dropna(subset=ARMS).copy()
    donors=sorted(d.donor_id.astype(str).unique())
    if len(donors)<2:
        return pd.DataFrame([{"dataset":name,"status":"NA_SINGLE_DONOR","n_donors":len(donors)}]),pd.DataFrame()

    rows=[]
    for donor in donors:
        tr=d[d.donor_id.astype(str)!=donor].copy()
        te=d[d.donor_id.astype(str)==donor].copy()
        for x in [tr,te]:
            x["LT"]=x.lineage.astype(str)+"||"+x.tissue.astype(str)

        Ytr=tr[ARMS].to_numpy(float); Yte=te[ARMS].to_numpy(float)
        mu=Ytr.mean(0); sd=Ytr.std(0); sd[sd<EPS]=1
        Ytr=(Ytr-mu)/sd; Yte=(Yte-mu)/sd

        vals={}
        for tag,cols in [("add",["lineage","tissue"]),("int",["lineage","tissue","LT"])]:
            enc=OneHotEncoder(handle_unknown="ignore",drop="first",sparse_output=False)
            Xtr=enc.fit_transform(tr[cols]); Xte=enc.transform(te[cols])
            m=LinearRegression().fit(Xtr,Ytr); pred=m.predict(Xte)
            vals[tag]=float(np.mean((Yte-pred)**2))
        rows.append({"dataset":name,"heldout_donor":donor,"mse_additive":vals["add"],
                     "mse_interaction":vals["int"],"gain":vals["add"]-vals["int"]})

    f=pd.DataFrame(rows); delta=f.gain.to_numpy(float)
    try:p=float(stats.wilcoxon(delta,alternative="greater").pvalue)
    except:p=np.nan
    s=pd.DataFrame([{"dataset":name,"status":"OK","n_donors":len(donors),
                     "mean_additive_MSE":f.mse_additive.mean(),
                     "mean_interaction_MSE":f.mse_interaction.mean(),
                     "mean_gain":delta.mean(),
                     "fraction_donors_better":np.mean(delta>0),
                     "paired_one_sided_P":p}])
    return s,f

# ------------------------------------------------------------------
# A2. GSE292 harmonic interaction leave-one-ZT-out
# ------------------------------------------------------------------
def hdesign(d,interaction):
    L=pd.get_dummies(pd.Categorical(d.lineage,categories=LINEAGES),drop_first=True,dtype=float).to_numpy()
    r=2*np.pi*d.ZT.to_numpy(float)/24
    s=np.sin(r)[:,None]; c=np.cos(r)[:,None]
    parts=[np.ones((len(d),1)),L,s,c]
    if interaction:parts += [L*s,L*c]
    return np.column_stack(parts)

def cv_gse292():
    d=pd.read_csv(GSE292)[["lineage","ZT",*ARMS]].copy()
    d["lineage"]=d.lineage.map(norm_lineage)
    d=d[d.lineage.isin(LINEAGES)].dropna()
    rows=[]
    for zt in sorted(d.ZT.unique()):
        tr=d[d.ZT!=zt]; te=d[d.ZT==zt]
        Ytr=tr[ARMS].to_numpy(float); Yte=te[ARMS].to_numpy(float)
        mu=Ytr.mean(0);sd=Ytr.std(0);sd[sd<EPS]=1
        Ytr=(Ytr-mu)/sd;Yte=(Yte-mu)/sd
        Xa,Xi=hdesign(tr,False),hdesign(tr,True)
        Xta,Xti=hdesign(te,False),hdesign(te,True)
        ba=np.linalg.lstsq(Xa,Ytr,rcond=None)[0];bi=np.linalg.lstsq(Xi,Ytr,rcond=None)[0]
        ma=float(np.mean((Yte-Xta@ba)**2));mi=float(np.mean((Yte-Xti@bi)**2))
        rows.append({"heldout_ZT":zt,"mse_additive":ma,"mse_harmonic_interaction":mi,"gain":ma-mi})
    f=pd.DataFrame(rows)
    # exact sign-flip (6 folds)
    obs=f.gain.mean(); null=[]
    for sg in itertools.product([-1,1],repeat=len(f)):
        null.append(np.mean(f.gain.to_numpy()*np.asarray(sg)))
    p=float(np.mean(np.asarray(null)>=obs-1e-12))
    s=pd.DataFrame([{"dataset":"GSE292219","n_ZT_folds":len(f),
                     "mean_additive_MSE":f.mse_additive.mean(),
                     "mean_interaction_MSE":f.mse_harmonic_interaction.mean(),
                     "mean_gain":obs,"fraction_ZT_better":np.mean(f.gain>0),
                     "exact_signflip_P":p}])
    return s,f

# ------------------------------------------------------------------
# B. network + frozen expression
# ------------------------------------------------------------------
def load_net():
    if DOROTHEA.exists():
        n=pd.read_csv(DOROTHEA)
        n.columns=[c.lower() for c in n.columns]
        src="source" if "source" in n.columns else "tf"
        out=pd.DataFrame({"source":n[src].astype(str).str.upper(),
                          "target":n.target.astype(str).str.upper(),
                          "mor":pd.to_numeric(n.mor,errors="coerce"),
                          "confidence":n.confidence.astype(str) if "confidence" in n else "NA"})
        out=out.dropna().drop_duplicates(["source","target"])
        return out,"DOROTHEA_ABC"
    return pd.DataFrame(),"FALLBACK_EXPRESSION_ONLY"

def load_tabula_agg(net):
    meta=pd.read_csv(META,sep="\t")
    src="priority_lineage_corrected" if "priority_lineage_corrected" in meta else "priority_lineage"
    meta["lineage"]=meta[src].map(norm_lineage)
    z=np.load(NPZ,allow_pickle=True,mmap_mode="r")
    ids=z["group_id"].astype(str);genes=z["genes"].astype(str);X=z["expression"]
    meta["group_id"]=meta.group_id.astype(str)
    order=pd.Index(ids).get_indexer(meta.group_id)
    if (order<0).any():raise RuntimeError("metadata/expression alignment failed")
    X=X[order]
    keep=meta.lineage.isin(LINEAGES).to_numpy()
    m=meta.loc[keep,["donor_id","tissue","lineage"]].reset_index(drop=True)
    pos=np.where(keep)[0]

    gm={str(g).upper():i for i,g in enumerate(genes)}
    if len(net):
        tfu=sorted(set(net.source)-set(CLOCK17))
        needed=set(tfu)|set(net.target)
    else:
        tfu=sorted(set(FALLBACK_TFS)-set(CLOCK17));needed=set(tfu)
    selected=sorted(g for g in needed if g in gm)
    idx=np.array([gm[g] for g in selected],int)

    vals=[];keys=[]
    tmp=m.copy();tmp["_i"]=np.arange(len(tmp))
    for key,g in tmp.groupby(["donor_id","tissue","lineage"],sort=False):
        oi=pos[g._i.to_numpy(int)]
        vals.append(np.asarray(X[np.ix_(oi,idx)],float).mean(0));keys.append(key)
    M=np.vstack(vals); am=pd.DataFrame(keys,columns=["donor_id","tissue","lineage"])
    am["key"]=am.donor_id.astype(str)+"||"+am.tissue.astype(str)+"||"+am.lineage.astype(str)

    six=pd.read_csv(TABULA6);six["lineage"]=six.lineage.map(norm_lineage)
    six["key"]=six.donor_id.astype(str)+"||"+six.tissue.astype(str)+"||"+six.lineage.astype(str)
    if set(am.key)!=set(six.key):raise RuntimeError("aggregate keys differ from analysis 78")
    reidx=pd.Index(am.key).get_indexer(six.key)
    return am.iloc[reidx].reset_index(drop=True),M[reidx],selected,six.reset_index(drop=True),tfu

def tf_expression(M,genes,tfu):
    gm={g:i for i,g in enumerate(genes)}
    names=[t for t in tfu if t in gm]
    return (M[:,[gm[t] for t in names]] if names else np.empty((len(M),0))),names

def ulm_activity(M,genes,net):
    if net.empty:return np.empty((len(M),0)),[]
    gm={g:i for i,g in enumerate(genes)}
    Z=(M-M.mean(0))/np.where(M.std(0)>EPS,M.std(0),1)
    acts=[];names=[]
    for tf,d in net.groupby("source"):
        if tf in CLOCK17:continue
        d=d[d.target.isin(gm)]
        if len(d)<10:continue
        ii=np.array([gm[t] for t in d.target]);w=d.mor.to_numpy(float);wc=w-w.mean()
        den=np.dot(wc,wc)
        if den<EPS:continue
        Y=Z[:,ii];yc=Y-Y.mean(1,keepdims=True);b=(yc@wc)/den
        res=yc-b[:,None]*wc;df=max(len(w)-2,1)
        se=np.sqrt(np.maximum(np.sum(res**2,1)/df/den,EPS))
        acts.append(b/se);names.append(tf)
    return (np.column_stack(acts) if acts else np.empty((len(M),0))),names

def dummies(meta,cols):
    parts=[np.ones((len(meta),1))]
    for c in cols:
        x=pd.get_dummies(meta[c].astype(str),drop_first=True,dtype=float)
        if x.shape[1]:parts.append(x.to_numpy())
    return np.column_stack(parts)

def sse(y,X):
    b=np.linalg.lstsq(X,y,rcond=None)[0];r=y-X@b
    return float(np.dot(r,r)),np.linalg.matrix_rank(X)

def lineage_stats(meta,P,names,ptype):
    xr=dummies(meta,["donor_id","tissue"]);xf=dummies(meta,["donor_id","tissue","lineage"])
    rows=[]
    for j,t in enumerate(names):
        sr,rr=sse(P[:,j],xr);sf,rf=sse(P[:,j],xf)
        dfn=rf-rr;dfd=len(meta)-rf
        pr2=max(0,(sr-sf)/sr) if sr>EPS else np.nan
        F=((sr-sf)/dfn)/(sf/dfd) if dfn>0 and dfd>0 and sf>EPS else np.nan
        p=float(stats.f.sf(F,dfn,dfd)) if np.isfinite(F) else np.nan
        rows.append({"predictor_type":ptype,"TF":t,"lineage_partial_R2":pr2,"P":p})
    o=pd.DataFrame(rows);o["FDR"]=bh(o.P);return o

def blocks_from_meta(meta):
    key=meta.donor_id.astype(str)+"||"+meta.tissue.astype(str)
    return [np.array(v,int) for v in key.groupby(key,sort=False).groups.values() if len(v)>=2]

def center_blocks(A,blocks):
    out=np.asarray(A,float).copy()
    for i in blocks:out[i]-=out[i].mean(0,keepdims=True)
    return out

def assoc(meta,six,P,names,ptype):
    blocks=blocks_from_meta(meta)
    Pc=center_blocks(P,blocks);Yc=center_blocks(six[ARMS].to_numpy(float),blocks)
    Pz=(Pc-Pc.mean(0))/np.where(Pc.std(0)>EPS,Pc.std(0),1)
    Yz=(Yc-Yc.mean(0))/np.where(Yc.std(0)>EPS,Yc.std(0),1)
    R=(Pz.T@Yz)/(len(meta)-1)
    rows=[]
    for j,t in enumerate(names):
        for a,arm in enumerate(ARMS):
            rows.append({"predictor_type":ptype,"TF":t,"arm":arm,"matched_r":R[j,a],"matched_R2":R[j,a]**2})
    out=pd.DataFrame(rows)
    if not len(names):
        out["perm_P"]=[];out["maxT_FWER"]=[];return out

    rng=np.random.default_rng(SEED+(0 if ptype=="expression" else 1000))
    exc=np.zeros_like(R,int);mxexc=np.zeros_like(R,int);obs=np.abs(R)
    for k in range(N_PERM):
        Q=Pz.copy()
        for b in blocks:Q[b]=Q[rng.permutation(b)]
        Rp=np.abs((Q.T@Yz)/(len(meta)-1))
        exc+=(Rp>=obs-1e-12);mx=Rp.max();mxexc+=(mx>=obs-1e-12)
        if (k+1)%50==0:log(f"{ptype} permutation {k+1}/{N_PERM}")
    emp=(exc+1)/(N_PERM+1);mxf=(mxexc+1)/(N_PERM+1)
    out["perm_P"]=[emp[names.index(r.TF),ARMS.index(r.arm)] for r in out.itertuples()]
    out["maxT_FWER"]=[mxf[names.index(r.TF),ARMS.index(r.arm)] for r in out.itertuples()]
    out["perm_FDR"]=bh(out.perm_P)
    return out

def lineage_means(meta,P,names,ptype):
    Pc=center_blocks(P,blocks_from_meta(meta))
    rows=[]
    for lin in LINEAGES:
        mask=meta.lineage.eq(lin).to_numpy()
        for j,t in enumerate(names):
            rows.append({"predictor_type":ptype,"TF":t,"lineage":lin,"mean_centered":Pc[mask,j].mean()})
    return pd.DataFrame(rows)

def direct_targets(net):
    if net.empty:return pd.DataFrame(columns=["TF","n_clock_targets","clock_targets"])
    rows=[]
    for tf,d in net.groupby("source"):
        tar=sorted(set(d.target)&set(CLOCK17))
        rows.append({"TF":tf,"n_clock_targets":len(tar),"clock_targets":",".join(tar)})
    return pd.DataFrame(rows)

def priority(expr_lin,act_lin,expr_a,act_a,direct):
    def best(a,prefix):
        if a.empty:return pd.DataFrame(columns=["TF"])
        x=a.sort_values(["maxT_FWER","perm_P","matched_R2"],ascending=[True,True,False]).groupby("TF",as_index=False).first()
        return x[["TF","arm","matched_r","matched_R2","perm_P","maxT_FWER"]].rename(columns={c:f"{prefix}_{c}" for c in ["arm","matched_r","matched_R2","perm_P","maxT_FWER"]})
    def lin(x,prefix):
        if x.empty:return pd.DataFrame(columns=["TF"])
        return x[["TF","lineage_partial_R2","FDR"]].rename(columns={"lineage_partial_R2":f"{prefix}_lineage_R2","FDR":f"{prefix}_lineage_FDR"})
    fs=[best(expr_a,"expr"),best(act_a,"act"),lin(expr_lin,"expr"),lin(act_lin,"act"),direct]
    alltf=sorted(set().union(*[set(x.TF) for x in fs if len(x)]))
    o=pd.DataFrame({"TF":alltf})
    for x in fs:
        if len(x):o=o.merge(x,on="TF",how="left")
    score=np.zeros(len(o))
    score+=2*(o.get("act_maxT_FWER",pd.Series(1,index=o.index)).fillna(1)<.05)
    score+=1*(o.get("expr_maxT_FWER",pd.Series(1,index=o.index)).fillna(1)<.05)
    score+=1*(o.get("act_lineage_FDR",pd.Series(1,index=o.index)).fillna(1)<.05)
    score+=1*(o.get("expr_lineage_FDR",pd.Series(1,index=o.index)).fillna(1)<.05)
    score+=1*(o.get("n_clock_targets",pd.Series(0,index=o.index)).fillna(0)>0)
    if "expr_arm" in o and "act_arm" in o:
        agree=o.expr_arm.notna()&o.act_arm.notna()&(o.expr_arm==o.act_arm)
        score+=agree.astype(int);o["expr_activity_arm_agree"]=agree
    o["evidence_score"]=score
    o["priority"]=np.where(score>=5,"A",np.where(score>=3,"B",np.where(score>=1,"C","LOW")))
    return o.sort_values(["evidence_score","TF"],ascending=[False,True])

def main():
    log("79 START")
    for p in [TABULA6,HCL6,AHCA6,GSE292,META,NPZ]:
        if not p.exists():raise FileNotFoundError(p)

    ss=[];ff=[]
    for name,path in [("Tabula",TABULA6),("HCL",HCL6),("AHCA",AHCA6)]:
        s,f=cv_static(name,path);ss.append(s)
        if len(f):ff.append(f)
    ss=pd.concat(ss,ignore_index=True);ff=pd.concat(ff,ignore_index=True) if ff else pd.DataFrame()
    ss.to_csv(CORR/"79_static_interaction_CV_summary.csv",index=False)
    ff.to_csv(CORR/"79_static_interaction_CV_folds.csv",index=False)

    gs,gf=cv_gse292()
    gs.to_csv(CORR/"79_GSE292_harmonic_interaction_CV_summary.csv",index=False)
    gf.to_csv(CORR/"79_GSE292_harmonic_interaction_CV_folds.csv",index=False)

    net,mode=load_net();log(f"network mode={mode}, edges={len(net)}")
    meta,M,genes,six,tfu=load_tabula_agg(net)
    Pexpr,en=tf_expression(M,genes,tfu);Pact,an=ulm_activity(M,genes,net)
    log(f"TF expression={len(en)}; activity={len(an)}")

    expr_lin=lineage_stats(meta,Pexpr,en,"expression")
    act_lin=lineage_stats(meta,Pact,an,"activity") if len(an) else pd.DataFrame()
    expr_a=assoc(meta,six,Pexpr,en,"expression")
    act_a=assoc(meta,six,Pact,an,"activity") if len(an) else pd.DataFrame()

    expr_lin.to_csv(TFDIR/"79_TF_expression_lineage_effect.csv",index=False)
    act_lin.to_csv(TFDIR/"79_TF_activity_lineage_effect.csv",index=False)
    expr_a.to_csv(TFDIR/"79_TF_expression_arm_association.csv",index=False)
    act_a.to_csv(TFDIR/"79_TF_activity_arm_association.csv",index=False)

    lm=pd.concat([lineage_means(meta,Pexpr,en,"expression"),
                  lineage_means(meta,Pact,an,"activity") if len(an) else pd.DataFrame()],ignore_index=True)
    lm.to_csv(TFDIR/"79_TF_lineage_centered_means.csv",index=False)

    direct=direct_targets(net);direct.to_csv(TFDIR/"79_DOROTHEA_direct_clock_targets.csv",index=False)
    pr=priority(expr_lin,act_lin,expr_a,act_a,direct)
    pr.to_csv(TFDIR/"79_TF_CANDIDATE_PRIORITY.csv",index=False)

    # figures
    d=ss[ss.status=="OK"]
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(d.dataset,d.mean_gain);ax.axhline(0,color="black",lw=.8)
    ax.set_ylabel("Donor-held-out MSE gain\n(additive − interaction)")
    ax.set_title("78.1: Lineage×Tissue generalization")
    savefig(fig,"79_F1_static_interaction_CV")

    fig,ax=plt.subplots(figsize=(4.5,4))
    ax.bar(["GSE292219"],[gs.mean_gain.iloc[0]]);ax.axhline(0,color="black",lw=.8)
    ax.set_ylabel("LO-ZT MSE gain\n(additive − harmonic interaction)")
    ax.set_title("78.1: Lineage×Time generalization")
    savefig(fig,"79_F2_GSE292_harmonic_interaction")

    top=pr.head(20)
    fig,ax=plt.subplots(figsize=(6,max(4,0.28*len(top)+1)))
    y=np.arange(len(top));ax.barh(y,top.evidence_score);ax.set_yticks(y);ax.set_yticklabels(top.TF);ax.invert_yaxis()
    ax.set_xlabel("Integrated evidence score");ax.set_title("TF candidates")
    savefig(fig,"79_F3_TF_candidates")

    # report
    report = f"""# 79 — 78.1 correction + TF discovery

## Network mode
**{mode}**

## Static interaction correction
{ss.to_string(index=False)}

## GSE292219 harmonic interaction correction
{gs.to_string(index=False)}

## TF analysis
TF-expression predictors: {len(en)}
TF-activity predictors: {len(an)}

Top candidates:
{pr.head(25).to_string(index=False)}

## Claim boundary
The corrected interaction tests are cross-validated and replace the saturated in-sample interaction R2.
TF results are regulatory candidate associations, not causal evidence. Perturbation and/or chromatin occupancy
validation is required before claiming a TF rewires the clock.
"""
    (REPORT/"79_FINAL_REPORT.md").write_text(report,encoding="utf-8")
    (OUT/"79_COMPLETE.json").write_text(json.dumps({
        "status":"COMPLETE","network_mode":mode,"n_tf_expression":len(en),"n_tf_activity":len(an),
        "n_priority_A":int((pr.priority=="A").sum()),"n_priority_B":int((pr.priority=="B").sum())
    },indent=2),encoding="utf-8")
    log("79 COMPLETE")

if __name__=="__main__":
    try:main()
    except Exception:
        (OUT/"79_FAILED.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print(traceback.format_exc(),flush=True)
        raise
