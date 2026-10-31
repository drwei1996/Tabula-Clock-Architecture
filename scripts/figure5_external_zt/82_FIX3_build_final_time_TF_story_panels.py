#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re, shutil, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ============================================================
# 0. PATHS
# ============================================================
PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
OUT = ROOT / "82_FIX3_FINAL_TIME_VALIDATION_TF_REWIRING_FIGURE_20261005"
MAIN = OUT / "MAIN"
SUPP = OUT / "SUPPLEMENT"
SRC = OUT / "SOURCE_DATA"
AUDIT = OUT / "AUDIT"
for d in [OUT, MAIN, SUPP, SRC, AUDIT]: d.mkdir(parents=True, exist_ok=True)

LINEAGES = ["B cell","T cell","Macrophage/Monocyte","Fibroblast","Endothelial"]
LINEAGE_COLORS = {
    "B cell":"#4E79A7","T cell":"#E15759","Macrophage/Monocyte":"#59A14F",
    "Fibroblast":"#F28E2B","Endothelial":"#9467BD",
}
MAIN_GENES = ["Arntl","Per2","Cry1","Nr1d1","Dbp","Nfil3"]
MAIN_GENE_ALIASES = {
    "Arntl": ["Arntl", "Bmal1"],
    "Per2": ["Per2"],
    "Cry1": ["Cry1"],
    "Nr1d1": ["Nr1d1", "Rev-erba", "ReverbA", "RevErbA"],
    "Dbp": ["Dbp"],
    "Nfil3": ["Nfil3", "E4bp4"],
}
LINEAGE_MARKERS = {
    "B cell": "o",
    "T cell": "s",
    "Macrophage/Monocyte": "^",
    "Fibroblast": "D",
    "Endothelial": "v",
}
MAIN_TFS = ["CREM","MEF2A","RUNX1","TBX21","PPARG","TFAP2C","ERG","SPI1"]
CLASS_COLORS = {
    "SIGN_SWITCHING":"#B2182B","LINEAGE_SELECTIVE":"#EF8A62",
    "SHARED_SIGN_HETEROGENEOUS":"#67A9CF","PAN_LINEAGE_POSITIVE":"#2166AC",
    "PAN_LINEAGE_NEGATIVE":"#4393C3","WEAK_OR_UNSTABLE":"#BDBDBD",
}

from matplotlib import font_manager
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((x for x in ["Arial","Liberation Sans","DejaVu Sans"] if x in avail), "sans-serif")
mpl.rcParams.update({
    "font.family":FONT,"font.size":8.5,"axes.titlesize":10,"axes.labelsize":9,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,
    "axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
    "savefig.dpi":600,"figure.facecolor":"white","savefig.facecolor":"white",
})

manifest=[]; warnings=[]; errors=[]

def norm(x): return re.sub(r"[^a-z0-9]+","",str(x).lower())
def clean(ax): ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(False)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def add_manifest(panel,action,src,out,note=""):
    srcp=Path(src) if src else None
    manifest.append({"panel":panel,"action":action,"source":"" if srcp is None else str(srcp),
                     "source_sha256":"" if srcp is None or not srcp.exists() else sha(srcp),
                     "output":str(out),"note":note})
def save(fig,stem,folder=MAIN):
    for ext in ['png','pdf','svg']:
        p=folder/f"{stem}.{ext}"
        fig.savefig(p,dpi=600 if ext=='png' else None,bbox_inches='tight',pad_inches=0.08)
    plt.close(fig)
def write_source(df,stem):
    p=SRC/f"{stem}.csv"; df.to_csv(p,index=False); return p

def stage_dirs(prefix):
    # Prefer direct analysis folders, but also allow one nested level because
    # some later stages were archived under a wrapper directory.
    hits=[]
    for p in ROOT.iterdir():
        if p.is_dir() and p.name.lower().startswith(prefix.lower()):
            hits.append(p)
    if not hits:
        for p in ROOT.glob('*/*'):
            if p.is_dir() and p.name.lower().startswith(prefix.lower()):
                hits.append(p)
    return sorted(set(hits))
def files_for(prefixes,patterns):
    out=[]; seen=set()
    for pre in prefixes:
        for r in stage_dirs(pre):
            for pat in patterns:
                for p in r.rglob(pat):
                    if p not in seen: seen.add(p); out.append(p)
    return out

def safe_header(p):
    try:
        sep='\t' if p.suffix.lower()=='.tsv' else ','
        return list(pd.read_csv(p,sep=sep,nrows=2).columns)
    except Exception:
        try: return list(pd.read_csv(p,sep=None,engine='python',nrows=2).columns)
        except Exception: return []
def readtab(p):
    sep='\t' if p.suffix.lower()=='.tsv' else ','
    try: return pd.read_csv(p,sep=sep)
    except Exception: return pd.read_csv(p,sep=None,engine='python')
def pick(cols, exact=(), contains=()):
    m={norm(c):c for c in cols}
    for e in exact:
        if norm(e) in m: return m[norm(e)]
    for c in cols:
        nc=norm(c)
        if all(norm(x) in nc for x in contains): return c
    return None

def linstd(x):
    n=norm(x)
    return {"bcell":"B cell","tcell":"T cell","macrophagemonocyte":"Macrophage/Monocyte",
            "macrophage":"Macrophage/Monocyte","monocyte":"Macrophage/Monocyte",
            "fibroblast":"Fibroblast","endothelial":"Endothelial","endothelialcell":"Endothelial"}.get(n,str(x))
def boolish(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(bool,np.bool_)): return bool(x)
    if isinstance(x,(int,float,np.number)): return float(x)>0
    s=str(x).strip().lower()
    if s in {'true','yes','1','supported','ok','evaluable'}: return True
    if s in {'false','no','0','unsupported','not_evaluable','na','none'}: return False
    return np.nan

def copy_first(tokens,dest,folder=MAIN,required=True):
    hits=[]
    for p in ROOT.rglob('*.png'):
        if OUT in p.parents: continue
        n=p.name.lower()
        if all(t.lower() in n for t in tokens): hits.append(p)
    if not hits:
        if required: raise FileNotFoundError(f"No PNG matches {tokens}")
        return None
    src=sorted(hits,key=lambda p:(p.stat().st_size,p.stat().st_mtime),reverse=True)[0]
    dst=folder/dest; shutil.copy2(src,dst)
    add_manifest(dest,'COPY_FINALIZED',src,dst,'Copied unchanged from finalized upstream output.')
    return src

# ============================================================
# A. SHARED TEMPORAL PROGRAM — REDRAW
# ============================================================
def _read_dynamic_candidate(p):
    if p.name.lower().endswith('.pkl'):
        return pd.read_pickle(p)
    return readtab(p)

def _gene_col_from_aliases(cols, gene):
    aliases = MAIN_GENE_ALIASES.get(gene, [gene])
    amap = {norm(c): c for c in cols}
    for a in aliases:
        if norm(a) in amap:
            return amap[norm(a)]
    return None

def _dynamic_candidate_to_long(df, p):
    """
    Convert a candidate lineage x ZT expression table to:
        lineage, ZT, gene, value

    Crucially, candidates are evaluated by ACTUAL five-lineage coverage rather
    than by filename. This prevents the older 3-lineage pseudobulk table from
    silently feeding Panel A.
    """
    cols = list(df.columns)
    lin = pick(cols, exact=('lineage','major_lineage','priority_lineage'))
    zt  = pick(cols, exact=('ZT','zt','time','time_h','time_hours'))
    if lin is None or zt is None:
        return None, {"file": str(p), "usable": False, "reason": "no_lineage_or_ZT"}

    z = df.copy()
    z[lin] = z[lin].map(linstd)
    z[zt] = pd.to_numeric(z[zt], errors='coerce')
    z = z[z[lin].isin(LINEAGES)].copy()
    if z.empty:
        return None, {"file": str(p), "usable": False, "reason": "no_target_lineages"}

    gc = pick(cols, exact=('gene','gene_symbol','symbol'))
    vc = pick(cols, exact=('value','expression','expr','mean_expression',
                          'mean_expr','z','centered','centered_z'))

    rows = []

    # Long format.
    if gc is not None and vc is not None:
        sg = z.loc[:, gc]
        sv = z.loc[:, vc]
        if isinstance(sg, pd.DataFrame): sg = sg.iloc[:,0]
        if isinstance(sv, pd.DataFrame): sv = sv.iloc[:,0]

        alias_to_gene = {}
        for g, aliases in MAIN_GENE_ALIASES.items():
            for a in aliases:
                alias_to_gene[norm(a)] = g

        tmp = pd.DataFrame({
            "lineage": z[lin].to_numpy(),
            "ZT": z[zt].to_numpy(),
            "__gene": sg.astype(str).map(norm).to_numpy(),
            "value": pd.to_numeric(sv, errors='coerce').to_numpy(),
        })
        tmp["gene"] = tmp["__gene"].map(alias_to_gene)
        tmp = tmp.dropna(subset=["gene","value","ZT","lineage"])
        rows.append(tmp[["lineage","ZT","gene","value"]])

    # Wide format.
    else:
        for g in MAIN_GENES:
            c = _gene_col_from_aliases(cols, g)
            if c is None:
                continue
            sv = z.loc[:, c]
            if isinstance(sv, pd.DataFrame):
                sv = sv.iloc[:,0]
            rows.append(pd.DataFrame({
                "lineage": z[lin].to_numpy(),
                "ZT": z[zt].to_numpy(),
                "gene": g,
                "value": pd.to_numeric(sv, errors='coerce').to_numpy(),
            }))

    if not rows:
        return None, {"file": str(p), "usable": False, "reason": "no_selected_genes"}

    d = pd.concat(rows, ignore_index=True)
    d = (
        d.dropna(subset=["lineage","ZT","gene","value"])
         .groupby(["lineage","ZT","gene"], as_index=False)["value"].mean()
    )

    if d.empty:
        return None, {"file": str(p), "usable": False, "reason": "empty_after_cleaning"}

    nlin = int(d["lineage"].nunique())
    ngen = int(d["gene"].nunique())
    zt_per_lineage = d.groupby("lineage")["ZT"].nunique().to_dict()
    min_zt = int(min(zt_per_lineage.values())) if zt_per_lineage else 0

    gene_lineage = (
        d.groupby("gene")["lineage"].nunique()
        .reindex(MAIN_GENES, fill_value=0)
        .to_dict()
    )
    genes_all5 = sum(int(v == 5) for v in gene_lineage.values())

    return d, {
        "file": str(p),
        "usable": True,
        "n_lineages": nlin,
        "n_genes": ngen,
        "min_ZT_per_lineage": min_zt,
        "genes_with_all5_lineages": genes_all5,
        "lineages": ";".join(sorted(d["lineage"].unique())),
        "genes": ";".join(sorted(d["gene"].unique())),
    }

def load_dynamic():
    """
    FIX3:
    Search the later frozen dynamic-validation outputs first and REQUIRE:
      - all five lineages
      - all six requested clock genes
      - >=4 ZT per lineage (normally six)

    If no candidate satisfies this, STOP rather than draw a misleading
    three-lineage panel.
    """
    roots = []
    for pre in ['76','75','74G','74','73','71','70']:
        roots.extend(stage_dirs(pre))

    # Search table-like sources only. Candidate tables here are small lineage x
    # time summaries, not raw cell matrices.
    cand = []
    seen = set()
    for r in roots:
        for pat in ['*.pkl','*.csv','*.csv.gz','*.tsv']:
            for p in r.rglob(pat):
                if p in seen:
                    continue
                seen.add(p)
                lname = (p.name + " " + str(p.parent)).lower()
                # Strongly favor expression / trajectory / pseudobulk tables;
                # deprioritize pure metric / audit tables.
                if not any(k in lname for k in [
                    'lineage','trajectory','pseudobulk','expression','expr','clock','dynamic','zt'
                ]):
                    continue
                try:
                    if p.stat().st_size > 500 * 1024 * 1024:
                        continue
                except Exception:
                    continue
                cand.append(p)

    # Explicitly include the old canonical pickle, but it must pass the same
    # five-lineage coverage gate.
    for p in ROOT.rglob('GSE292219_lineage_ZT_mean_norm10k.pkl'):
        if p not in seen:
            cand.append(p); seen.add(p)

    audit = []
    good = []
    for p in cand:
        try:
            df = _read_dynamic_candidate(p)
            d, meta = _dynamic_candidate_to_long(df, p)
        except Exception as e:
            audit.append({
                "file": str(p), "usable": False,
                "reason": f"{type(e).__name__}: {e}"
            })
            continue

        audit.append(meta)
        if d is None:
            continue

        # Hard scientific gate for the main figure.
        full_gene_cov = (
            d.groupby("gene")["lineage"].nunique()
             .reindex(MAIN_GENES, fill_value=0)
        )
        if (
            d["lineage"].nunique() == 5
            and d["gene"].nunique() == 6
            and (full_gene_cov == 5).all()
            and d.groupby("lineage")["ZT"].nunique().min() >= 4
        ):
            # Later stages preferred; then more complete ZT coverage.
            pathlow = str(p).lower()
            stage_score = (
                70 if '76_' in pathlow or '\\76' in pathlow else
                60 if '75_' in pathlow or '\\75' in pathlow else
                50 if '74g' in pathlow else
                40 if '74_' in pathlow or '\\74' in pathlow else
                30 if '73_' in pathlow or '\\73' in pathlow else
                20 if '71_' in pathlow or '\\71' in pathlow else
                10
            )
            zt_score = int(d.groupby("lineage")["ZT"].nunique().min())
            good.append((stage_score, zt_score, p, d))

    pd.DataFrame(audit).to_csv(AUDIT / "A_SOURCE_AUDIT.csv", index=False)

    if not good:
        raise RuntimeError(
            "Panel A: no source table contains all 5 lineages x all 6 requested "
            "clock genes with >=4 ZT/lineage. See AUDIT/A_SOURCE_AUDIT.csv."
        )

    good.sort(key=lambda x: (x[0], x[1], x[2].stat().st_mtime), reverse=True)
    _, _, p, d = good[0]

    # Lineage-center and scale separately for each gene x lineage trajectory.
    def zfun(x):
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return x - x.mean()
        return (x - x.mean()) / sd

    d = d.copy()
    d["centered_z"] = (
        d.groupby(["gene","lineage"])["value"]
         .transform(zfun)
    )

    # Final audit gate: every gene must still contain every lineage.
    final_cov = (
        d.groupby("gene")["lineage"].nunique()
         .reindex(MAIN_GENES, fill_value=0)
    )
    if not (final_cov == 5).all():
        raise RuntimeError(
            "Panel A final five-lineage gate failed after transformation."
        )

    pd.DataFrame([{
        "selected_source": str(p),
        "n_lineages": d["lineage"].nunique(),
        "n_genes": d["gene"].nunique(),
        "min_ZT_per_lineage": d.groupby("lineage")["ZT"].nunique().min(),
    }]).to_csv(AUDIT / "A_SELECTED_SOURCE.csv", index=False)

    return d, p

def harmonic(t,y,grid):
    t=np.asarray(t,float); y=np.asarray(y,float); ok=np.isfinite(t)&np.isfinite(y); t=t[ok]; y=y[ok]
    if len(t)<4: return np.full_like(grid,np.nan),np.nan
    w=2*np.pi/24.; X=np.c_[np.ones(len(t)),np.cos(w*t),np.sin(w*t)]; b=np.linalg.lstsq(X,y,rcond=None)[0]; Xg=np.c_[np.ones(len(grid)),np.cos(w*grid),np.sin(w*grid)]
    return Xg@b,(np.arctan2(b[2],b[1])%(2*np.pi))*24/(2*np.pi)

def panel_A():
    d,src=load_dynamic()
    genes=MAIN_GENES[:]  # hard-gated in load_dynamic()
    grid=np.linspace(0,24,200)
    phases=[]

    fig=plt.figure(figsize=(9.2,5.5))
    gs=fig.add_gridspec(
        2,4,
        width_ratios=[1,1,1,0.88],
        wspace=.38,hspace=.46
    )

    for i,g in enumerate(genes):
        ax=fig.add_subplot(gs[i//3,i%3])
        qg=d[d.gene.eq(g)].copy()

        for zorder,lin in enumerate(LINEAGES, start=2):
            q=qg[qg.lineage.eq(lin)].sort_values('ZT')
            if q.empty:
                raise RuntimeError(f"Panel A lost lineage {lin} for gene {g}")

            pred,ph=harmonic(q.ZT,q.centered_z,grid)
            phases.append({'gene':g,'lineage':lin,'phase_h':ph})

            ax.plot(
                grid,pred,lw=1.35,
                color=LINEAGE_COLORS[lin],
                alpha=.82,zorder=zorder
            )
            ax.scatter(
                q.ZT,q.centered_z,
                s=20,marker=LINEAGE_MARKERS[lin],
                color=LINEAGE_COLORS[lin],
                edgecolor='white',linewidth=.45,
                alpha=.95,zorder=10+zorder
            )

        ax.set_title(g,fontstyle='italic',pad=3)
        ax.set_xlim(0,24)
        ax.set_xticks([0,6,12,18,24])
        ax.axhline(0,color='#C7C7C7',lw=.65,zorder=0)
        if i//3==1:
            ax.set_xlabel('Zeitgeber time (h)')
        if i%3==0:
            ax.set_ylabel('Lineage-centered\nexpression (z)')
        clean(ax)

    # Phase dispersion inset.
    ax=fig.add_subplot(gs[:,3])
    ph=pd.DataFrame(phases)
    rr=[]
    for g in genes:
        v=ph.loc[ph.gene.eq(g),'phase_h'].dropna().to_numpy(float)
        if len(v) != 5:
            raise RuntimeError(f"Panel A phase inset: {g} has {len(v)} lineages, expected 5")
        R=np.clip(abs(np.mean(np.exp(1j*2*np.pi*v/24))),1e-9,1)
        sd=np.sqrt(-2*np.log(R))*24/(2*np.pi)
        rr.append((g,sd))
    rr=pd.DataFrame(rr,columns=['gene','circular_phase_SD_h'])
    y=np.arange(len(rr))
    ax.hlines(y,0,rr.circular_phase_SD_h,color='#C7C7C7',lw=1.1)
    ax.scatter(rr.circular_phase_SD_h,y,s=38,color='#555555',zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(rr.gene,fontstyle='italic')
    ax.invert_yaxis()
    ax.set_xlabel('Cross-lineage\nphase dispersion (h)')
    ax.set_title('Phase coherence',pad=4)
    clean(ax)

    handles=[
        Line2D(
            [0],[0],
            color=LINEAGE_COLORS[l],lw=1.8,
            marker=LINEAGE_MARKERS[l],markersize=4.8,
            markerfacecolor=LINEAGE_COLORS[l],
            markeredgecolor='white',
            label=l
        )
        for l in LINEAGES
    ]
    fig.legend(
        handles=handles,loc='upper center',
        bbox_to_anchor=(.43,1.02),
        ncol=5,frameon=False,columnspacing=1.3
    )
    fig.suptitle(
        'A  Time-stamped liver reveals a broadly shared temporal program across lineages',
        x=.01,y=1.06,ha='left',fontsize=11,fontweight='bold'
    )

    stem='MAIN_A_shared_temporal_program'
    save(fig,stem)
    write_source(d,stem+'_trajectories_source')
    write_source(rr,stem+'_phase_dispersion_source')
    add_manifest(
        'A','REDRAW',src,MAIN/f'{stem}.png',
        'All five lineages are required for all six genes; marker shapes distinguish overlapping trajectories.'
    )

# ============================================================
# B. TIME REMOVAL — REDRAW STRICT RESULT
# ============================================================
def _num(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _candidate_text(df):
    # compact row-wise lower-case text used only for locating candidate result rows
    return df.astype(str).apply(lambda r: ' | '.join(r.values).lower(), axis=1)


def _extract_B_from_dataframe(d, p):
    cols=list(d.columns)
    nmap={norm(c):c for c in cols}

    def col_by_terms(required=(), anyof=(), forbidden=()):
        for c in cols:
            nc=norm(c)
            if all(norm(x) in nc for x in required) and (not anyof or any(norm(x) in nc for x in anyof)) and not any(norm(x) in nc for x in forbidden):
                return c
        return None

    # ---------- Case 1: permutation/null draws ----------
    ba_cols=[c for c in cols if norm(c) in {'ba','balancedaccuracy','balancedacc','score'} or 'balancedaccuracy' in norm(c)]
    type_cols=[c for c in cols if norm(c) in {'type','kind','label','group','status','iterationtype','drawtype'}]
    for bac in ba_cols:
        vals=pd.to_numeric(d[bac],errors='coerce')
        for tc in type_cols:
            lab=d[tc].astype(str).str.lower()
            om=lab.str.contains(r'obs|observed',regex=True,na=False)
            nm=lab.str.contains(r'null|perm',regex=True,na=False)
            if om.any() and nm.sum()>=20 and vals[om].notna().any():
                obs=float(vals[om].dropna().iloc[0]); null=vals[nm].dropna().to_numpy(float)
                return dict(obs=obs,null=null,mean=float(np.mean(null)),q95=float(np.quantile(null,.95)),
                            p=float((1+(null>=obs).sum())/(1+len(null))),src=p)

    # ---------- Case 2: conventional wide one-row summary ----------
    obsc=next((c for c in cols if ('observed' in norm(c) or norm(c).startswith('obs')) and any(k in norm(c) for k in ['balanced','accuracy','ba','score'])),None)
    if obsc is None:
        obsc=next((c for c in cols if norm(c) in {'observed','obs','observedba','observedscore'}),None)
    meanc=next((c for c in cols if any(k in norm(c) for k in ['null','perm']) and 'mean' in norm(c)),None)
    q95c=next((c for c in cols if any(k in norm(c) for k in ['null','perm']) and any(k in norm(c) for k in ['95','q95','quantile95','upper95'])),None)
    pc=next((c for c in cols if norm(c) in {'p','pvalue','permutationp','empiricalp','permp','permutationpvalue','empiricalpvalue'} or ('perm' in norm(c) and 'p' in norm(c))),None)

    if obsc and (meanc or q95c):
        # Prefer row whose labels explicitly mention lineage + time/ZT/residualization.
        text=_candidate_text(d)
        hit=text.str.contains('lineage',na=False) & text.str.contains(r'zt|time|resid|leave.one|held.out',regex=True,na=False)
        rows=d[hit] if hit.any() else d
        for _,r in rows.iterrows():
            obs=_num(r[obsc]); mean=_num(r[meanc]) if meanc else np.nan; q95=_num(r[q95c]) if q95c else np.nan; pv=_num(r[pc]) if pc else np.nan
            if np.isfinite(obs) and (np.isfinite(mean) or np.isfinite(q95)):
                return dict(obs=obs,null=None,mean=mean,q95=q95,p=pv,src=p)

    # ---------- Case 3: long metric/value result table ----------
    labelcols=[c for c in cols if any(k in norm(c) for k in ['metric','measure','statistic','endpoint','analysis','test','name','key','quantity'])]
    valcols=[c for c in cols if norm(c) in {'value','estimate','score','stat','result'}]
    if labelcols and valcols:
        txt=_candidate_text(d)
        # keep rows from the lineage-after-time-removal analysis when such labels exist
        context=txt.str.contains('lineage',na=False) & txt.str.contains(r'zt|time|resid|leave.one|held.out',regex=True,na=False)
        q=d[context].copy() if context.any() else d.copy()
        qt=_candidate_text(q)
        vc=valcols[0]
        def findval(regex):
            m=qt.str.contains(regex,regex=True,na=False)
            if not m.any(): return np.nan
            vv=pd.to_numeric(q.loc[m,vc],errors='coerce').dropna()
            return float(vv.iloc[0]) if len(vv) else np.nan
        obs=findval(r'observed.*(balanced|accuracy|\bba\b)|(balanced|accuracy|\bba\b).*observed')
        mean=findval(r'(null|perm).*(mean|average)|(mean|average).*(null|perm)')
        q95=findval(r'(null|perm).*(95|q95|quantile)|(95|q95|quantile).*(null|perm)')
        pv=findval(r'(empirical|permutation|perm).*\bp\b|\bp\b.*(empirical|permutation|perm)')
        if np.isfinite(obs) and (np.isfinite(mean) or np.isfinite(q95)):
            return dict(obs=obs,null=None,mean=mean,q95=q95,p=pv,src=p)

    # ---------- Case 4: held-out true/predicted labels ----------
    truec=next((c for c in cols if norm(c) in {'ytrue','truelineage','trueclass','lineagetrue'}),None)
    predc=next((c for c in cols if norm(c) in {'ypred','predlineage','predictedlineage','predictedclass','lineagepred'}),None)
    if truec and predc and any(k in (p.name+' '+str(p.parent)).lower() for k in ['leave','held','resid','time']):
        try:
            from sklearn.metrics import balanced_accuracy_score
            q=d[[truec,predc]].dropna()
            if len(q)>=10:
                obs=float(balanced_accuracy_score(q[truec].astype(str),q[predc].astype(str)))
                # Exact null cannot be reconstructed without the frozen permutation draws;
                # do not invent it. Return None only if the same table carries null fields.
        except Exception:
            pass

    return None


def _extract_B_from_text(p):
    try:
        txt=p.read_text(encoding='utf-8',errors='ignore')
    except Exception:
        return None
    low=txt.lower()
    if 'gse292219' not in low and 'zt' not in low:
        return None
    if 'lineage' not in low:
        return None

    # Search paragraph/line windows that mention lineage classification and time removal.
    lines=txt.splitlines()
    windows=[]
    for i,line in enumerate(lines):
        ll=line.lower()
        if 'lineage' in ll and any(k in ll for k in ['balanced','accuracy','classification','predict']):
            windows.append('\n'.join(lines[max(0,i-6):min(len(lines),i+10)]))
    if not windows:
        windows=[txt]

    def grab(patterns, block):
        for pat in patterns:
            m=re.search(pat,block,re.I|re.S)
            if m:
                try: return float(m.group(1))
                except Exception: pass
        return np.nan

    for w in windows:
        obs=grab([
            r'observed[^\n]{0,80}?(?:balanced\s*accuracy|\bBA\b)[^0-9]{0,20}([0-9]*\.[0-9]+)',
            r'(?:balanced\s*accuracy|\bBA\b)[^\n]{0,50}?observed[^0-9]{0,20}([0-9]*\.[0-9]+)',
            r'(?:observed|real)[^0-9]{0,20}([0-9]*\.[0-9]+)',
        ],w)
        mean=grab([
            r'null[^\n]{0,50}?(?:mean|average)[^0-9]{0,20}([0-9]*\.[0-9]+)',
            r'(?:mean|average)[^\n]{0,30}?null[^0-9]{0,20}([0-9]*\.[0-9]+)',
        ],w)
        q95=grab([
            r'(?:null|perm)[^\n]{0,60}?(?:95|q95|quantile)[^0-9]{0,20}([0-9]*\.[0-9]+)',
            r'(?:95|q95|quantile)[^\n]{0,30}?(?:null|perm)[^0-9]{0,20}([0-9]*\.[0-9]+)',
        ],w)
        pv=grab([
            r'(?:permutation|empirical)[^\n]{0,50}?\bP\b\s*[=:]?\s*([0-9]*\.[0-9]+)',
            r'\bP\b\s*[=:]\s*([0-9]*\.[0-9]+)',
        ],w)
        if np.isfinite(obs) and (np.isfinite(mean) or np.isfinite(q95)):
            return dict(obs=obs,null=None,mean=mean,q95=q95,p=pv,src=p)
    return None


def find_B():
    """
    FIX2: resolve Panel B from the exact frozen 78 outputs first.

    Canonical files:
      78_E_GSE292219_ZT_RESIDUAL_PORTABILITY.csv
      GSE292219_ZT_LOTO_NULL.csv.gz
      GSE292219_ZT_LOTO_PREDICTIONS.csv

    No approximate values are hard-coded.
    """
    table_root = (
        ROOT /
        "78_LINEAGE_INTRINSIC_3LOOP_VALIDATION" /
        "01_TABLES"
    )
    summary_path = table_root / "78_E_GSE292219_ZT_RESIDUAL_PORTABILITY.csv"
    null_path = table_root / "GSE292219_ZT_LOTO_NULL.csv.gz"
    pred_path = table_root / "GSE292219_ZT_LOTO_PREDICTIONS.csv"

    if summary_path.exists():
        s = readtab(summary_path)

        required = [
            "zt_residual_loto_balanced_accuracy",
            "null_accuracy_mean",
            "null_accuracy_q95",
            "perm_p_accuracy_upper",
        ]
        missing = [c for c in required if c not in s.columns]
        if missing:
            raise RuntimeError(
                "Canonical Panel B summary exists but is missing columns: "
                + ", ".join(missing)
            )

        if "dataset" in s.columns:
            hit = s[
                s["dataset"].astype(str)
                .str.contains("GSE292219", case=False, na=False)
            ]
            if len(hit) == 1:
                row = hit.iloc[0]
            elif len(s) == 1:
                row = s.iloc[0]
            else:
                raise RuntimeError(
                    "Canonical Panel B summary has no unique GSE292219 row."
                )
        elif len(s) == 1:
            row = s.iloc[0]
        else:
            raise RuntimeError(
                "Canonical Panel B summary lacks dataset column and has >1 row."
            )

        obs = _num(row["zt_residual_loto_balanced_accuracy"])
        mean = _num(row["null_accuracy_mean"])
        q95 = _num(row["null_accuracy_q95"])
        pv = _num(row["perm_p_accuracy_upper"])

        if not all(np.isfinite(v) for v in [obs, mean, q95, pv]):
            raise RuntimeError(
                "Canonical Panel B summary contains non-finite frozen metrics."
            )

        audit = [{
            "check": "summary_file",
            "status": "PASS",
            "source": str(summary_path),
            "value": f"obs={obs}; null_mean={mean}; null_q95={q95}; P={pv}",
        }]

        null = None
        if null_path.exists():
            nd = readtab(null_path)
            if "balanced_accuracy" not in nd.columns:
                raise RuntimeError(
                    "Canonical null file exists but lacks balanced_accuracy."
                )
            null = pd.to_numeric(
                nd["balanced_accuracy"], errors="coerce"
            ).dropna().to_numpy(float)

            if len(null) < 20:
                raise RuntimeError(
                    f"Canonical null file has too few valid draws: {len(null)}"
                )

            calc_mean = float(np.mean(null))
            calc_q95 = float(np.quantile(null, 0.95))

            mean_ok = np.isclose(calc_mean, mean, rtol=0, atol=5e-4)
            q95_ok = np.isclose(calc_q95, q95, rtol=0, atol=5e-4)

            audit += [
                {
                    "check": "null_mean_crosscheck",
                    "status": "PASS" if mean_ok else "FAIL",
                    "source": str(null_path),
                    "value": f"summary={mean}; recomputed={calc_mean}",
                },
                {
                    "check": "null_q95_crosscheck",
                    "status": "PASS" if q95_ok else "FAIL",
                    "source": str(null_path),
                    "value": f"summary={q95}; recomputed={calc_q95}",
                },
            ]
            if not (mean_ok and q95_ok):
                pd.DataFrame(audit).to_csv(
                    AUDIT / "B_CANONICAL_CROSSCHECK.csv", index=False
                )
                raise RuntimeError(
                    "Panel B frozen null distribution does not reproduce "
                    "summary mean/q95 within tolerance."
                )

        if pred_path.exists():
            pr = readtab(pred_path)
            if not {"true", "pred"}.issubset(pr.columns):
                raise RuntimeError(
                    "Canonical prediction file exists but lacks true/pred columns."
                )
            from sklearn.metrics import balanced_accuracy_score
            qq = pr[["true", "pred"]].dropna()
            pred_obs = float(
                balanced_accuracy_score(
                    qq["true"].astype(str),
                    qq["pred"].astype(str),
                )
            )
            pred_ok = np.isclose(pred_obs, obs, rtol=0, atol=1e-10)
            audit.append({
                "check": "observed_BA_crosscheck",
                "status": "PASS" if pred_ok else "FAIL",
                "source": str(pred_path),
                "value": f"summary={obs}; recomputed={pred_obs}",
            })
            if not pred_ok:
                pd.DataFrame(audit).to_csv(
                    AUDIT / "B_CANONICAL_CROSSCHECK.csv", index=False
                )
                raise RuntimeError(
                    "Panel B prediction table does not reproduce frozen "
                    "observed balanced accuracy."
                )

        pd.DataFrame(audit).to_csv(
            AUDIT / "B_CANONICAL_CROSSCHECK.csv", index=False
        )

        return dict(
            obs=obs,
            null=null,
            mean=mean,
            q95=q95,
            p=pv,
            src=summary_path,
            null_src=null_path if null_path.exists() else None,
            pred_src=pred_path if pred_path.exists() else None,
        )

    # Fallback search for renamed installations.
    candidates=[]
    roots=[]
    roots += stage_dirs('78') + stage_dirs('79')
    script_dir=ROOT/'scripts'
    if script_dir.exists(): roots.append(script_dir)

    seen=set()
    for r in roots:
        for pat in ['*.csv','*.csv.gz','*.tsv','*.txt','*.md']:
            for p in r.rglob(pat):
                if p in seen: continue
                seen.add(p)
                name=(p.name+' '+str(p.parent)).lower()
                score=sum(w for k,w in [
                    ('gse292219',12),('lineage',8),('balanced',8),('accuracy',6),('classification',6),
                    ('zt',6),('time',4),('resid',8),('leave',6),('held',5),('perm',6),('null',5),('78_f4',10)
                ] if k in name)
                candidates.append((score,p))

    # Also include any 78/79 text-like file under ROOT if stage folder naming changed.
    for p in ROOT.rglob('*'):
        if not p.is_file() or p in seen: continue
        if p.suffix.lower() not in {'.csv','.tsv','.txt','.md'} and not p.name.lower().endswith('.csv.gz'): continue
        low=(p.name+' '+str(p.parent)).lower()
        if re.search(r'(^|[\\/_ -])(78|79)([\\/_ -]|$)',low):
            score=sum(w for k,w in [('gse292219',12),('lineage',8),('balanced',8),('accuracy',6),('zt',6),('resid',8),('perm',6),('null',5)] if k in low)
            candidates.append((score,p)); seen.add(p)

    candidates=sorted(candidates,key=lambda x:(x[0],x[1].stat().st_mtime if x[1].exists() else 0),reverse=True)

    audit_rows=[]
    for score,p in candidates:
        if p.suffix.lower() in {'.txt','.md'}:
            r=_extract_B_from_text(p)
            audit_rows.append({'score':score,'file':str(p),'kind':'text','resolved':r is not None,'columns':''})
        else:
            try:
                d=readtab(p)
                r=_extract_B_from_dataframe(d,p)
                audit_rows.append({'score':score,'file':str(p),'kind':'table','resolved':r is not None,'columns':' | '.join(map(str,d.columns[:40]))})
            except Exception as e:
                r=None
                audit_rows.append({'score':score,'file':str(p),'kind':'table_read_error','resolved':False,'columns':str(e)})
        if r is not None:
            pd.DataFrame(audit_rows).to_csv(AUDIT/'B_SOURCE_SCAN.csv',index=False)
            return r

    pd.DataFrame(audit_rows).to_csv(AUDIT/'B_SOURCE_SCAN.csv',index=False)
    raise FileNotFoundError(
        'Exact time-residualized lineage-prediction source not discovered. '
        'A diagnostic file was written to AUDIT/B_SOURCE_SCAN.csv; no approximate values were substituted.'
    )

def panel_B():
    r=find_B()
    fig,ax=plt.subplots(figsize=(5.0,3.8))

    if r['null'] is None:
        raise RuntimeError(
            "Panel B FIX3 requires the frozen GSE292219_ZT_LOTO_NULL.csv.gz "
            "distribution, not only summary values."
        )

    null=np.asarray(r['null'],float)
    bins=min(35,max(16,int(np.sqrt(len(null)))))
    counts,edges,_=ax.hist(
        null,bins=bins,density=True,
        color='#D0D0D0',edgecolor='white',linewidth=.55
    )

    ymax=max(counts) if len(counts) else 1.0
    ax.set_ylim(0,ymax*1.24)

    ax.axvline(
        r['q95'],color='#7A7A7A',
        ls='--',lw=1.4
    )
    ax.axvline(
        r['obs'],color='#2F4B7C',
        lw=2.4
    )

    # Direct labels replace a crowded legend.
    ax.text(
        r['q95'],ymax*1.10,'Null 95th',
        rotation=90,ha='right',va='top',
        fontsize=7.5,color='#666666'
    )
    ax.text(
        r['obs'],ymax*1.10,'Observed',
        rotation=90,ha='left',va='top',
        fontsize=7.5,color='#2F4B7C'
    )

    stat_text = (
        f"Observed BA = {r['obs']:.3f}\n"
        f"Null mean = {r['mean']:.3f}\n"
        f"Null q95 = {r['q95']:.3f}\n"
        f"Permutation P = {r['p']:.3g}"
    )
    ax.text(
        .03,.96,stat_text,
        transform=ax.transAxes,
        ha='left',va='top',
        fontsize=7.8,
        bbox=dict(
            boxstyle='round,pad=0.30',
            facecolor='white',edgecolor='#D9D9D9',
            linewidth=.7,alpha=.96
        )
    )

    ax.set_xlabel('Leave-one-ZT-out balanced accuracy')
    ax.set_ylabel('Permutation density')
    ax.set_title(
        'B  Lineage-associated clock structure persists after explicit time removal',
        loc='left',fontweight='bold',pad=10
    )
    clean(ax)

    stem='MAIN_B_time_residualized_lineage_prediction'
    save(fig,stem)

    source_row=pd.DataFrame([{
        'observed_balanced_accuracy':r['obs'],
        'null_mean':r['mean'],
        'null_q95':r['q95'],
        'empirical_P':r['p'],
        'summary_source_file':str(r['src']),
        'null_draw_source_file':str(r.get('null_src') or ''),
        'prediction_source_file':str(r.get('pred_src') or ''),
        'n_null_draws':len(null),
    }])
    write_source(source_row,stem+'_source')

    add_manifest(
        'B','REDRAW',r['src'],MAIN/f'{stem}.png',
        'Frozen 78 time-residualized leave-one-ZT-out lineage portability; direct line labels avoid overlap.'
    )
    if r.get('null_src'):
        add_manifest(
            'B_null','READ_ONLY',r['null_src'],MAIN/f'{stem}.png',
            'Frozen permutation draws.'
        )
    if r.get('pred_src'):
        add_manifest(
            'B_predictions','READ_ONLY',r['pred_src'],MAIN/f'{stem}.png',
            'Frozen held-out predictions.'
        )

# ============================================================
# C/D. COPY FINALIZED 78 PANELS
# ============================================================
def panels_CD():
    """
    FIX3: redraw C/D from the exact frozen 78 source tables so the final main
    figure uses A–F labels and one visual language rather than copying legacy
    F2/F3 panels.
    """
    troot = ROOT / '78_LINEAGE_INTRINSIC_3LOOP_VALIDATION' / '01_TABLES'

    # -------------------------
    # C. Tissue-held-out transfer
    # -------------------------
    pc = troot / '78_B_TISSUE_HELD_OUT_LINEAGE_PORTABILITY.csv'
    if not pc.exists():
        raise FileNotFoundError(pc)
    c = readtab(pc)

    need = {
        'dataset','loto_balanced_accuracy',
        'null_balanced_accuracy_mean',
        'null_balanced_accuracy_q95',
        'perm_p_accuracy_upper'
    }
    if not need.issubset(c.columns):
        raise RuntimeError(
            'Panel C source missing columns: ' +
            ', '.join(sorted(need-set(c.columns)))
        )

    order=['Tabula','HCL','AHCA']
    c=c[c.dataset.astype(str).isin(order)].copy()
    c['dataset']=pd.Categorical(c['dataset'],categories=order,ordered=True)
    c=c.sort_values('dataset').reset_index(drop=True)

    fig,ax=plt.subplots(figsize=(5.2,3.8))
    x=np.arange(len(c))
    obs=pd.to_numeric(c['loto_balanced_accuracy'],errors='coerce').to_numpy(float)
    nullm=pd.to_numeric(c['null_balanced_accuracy_mean'],errors='coerce').to_numpy(float)
    q95=pd.to_numeric(c['null_balanced_accuracy_q95'],errors='coerce').to_numpy(float)
    pv=pd.to_numeric(c['perm_p_accuracy_upper'],errors='coerce').to_numpy(float)

    ax.bar(x,obs,width=.62,color='#4E79A7',alpha=.95,label='Observed')
    ax.scatter(x,nullm,s=42,color='#F28E2B',zorder=4,label='Permutation mean')
    for xi,yy in zip(x,q95):
        ax.hlines(yy,xi-.08,xi+.08,color='#59A14F',lw=2.0,zorder=5)
    for xi,yy,pp in zip(x,obs,pv):
        ax.text(
            xi,yy+.035,
            f'P = {pp:.4g}',
            ha='center',va='bottom',fontsize=7.5
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0,max(1.02,float(np.nanmax(obs)+.12)))
    ax.set_ylabel('Tissue-held-out balanced accuracy')
    ax.set_title(
        'C  Clock architecture predicts lineage in unseen tissues and independent atlases',
        loc='left',fontweight='bold',pad=10
    )
    clean(ax)

    handles=[
        Rectangle((0,0),1,1,facecolor='#4E79A7',edgecolor='none',label='Observed'),
        Line2D([0],[0],marker='o',linestyle='None',color='#F28E2B',label='Permutation mean'),
        Line2D([0],[0],color='#59A14F',lw=2,label='Permutation 95th percentile'),
    ]
    ax.legend(handles=handles,frameon=False,loc='upper left',fontsize=7.1)

    stem='MAIN_C_tissue_held_out_portability'
    save(fig,stem)
    write_source(c,stem+'_source')
    add_manifest(
        'C','REDRAW',pc,MAIN/f'{stem}.png',
        'Exact frozen 78 tissue-held-out portability.'
    )

    # -------------------------
    # D. Cross-atlas six-arm prototypes
    # -------------------------
    proto_files = {
        'Tabula': troot / 'Tabula_TISSUE_RESIDUAL_LINEAGE_PROTOTYPE.csv',
        'HCL': troot / 'HCL_TISSUE_RESIDUAL_LINEAGE_PROTOTYPE.csv',
        'AHCA': troot / 'AHCA_TISSUE_RESIDUAL_LINEAGE_PROTOTYPE.csv',
    }
    pdata={}
    for ds,pp in proto_files.items():
        if not pp.exists():
            raise FileNotFoundError(pp)
        z=readtab(pp)
        need={'lineage','EBOX_POS','EBOX_NEG','RORE_ACT','RORE_REP','DBOX_ACT','DBOX_REP'}
        if not need.issubset(z.columns):
            raise RuntimeError(
                f'Panel D {ds} source missing columns: ' +
                ', '.join(sorted(need-set(z.columns)))
            )
        z=z.copy()
        z['lineage']=z['lineage'].map(linstd)
        z=z[z.lineage.isin(LINEAGES)].copy()
        z=z.set_index('lineage').reindex(LINEAGES)
        pdata[ds]=z

    arms=['EBOX_POS','EBOX_NEG','RORE_ACT','RORE_REP','DBOX_ACT','DBOX_REP']
    allv=np.concatenate([
        pdata[ds][arms].to_numpy(float).ravel()
        for ds in order
    ])
    vmax=max(1.0,float(np.nanmax(np.abs(allv))))

    fig,axs=plt.subplots(
        1,3,figsize=(8.8,3.8),
        gridspec_kw={'wspace':.38}
    )
    im=None
    for ax,ds in zip(axs,order):
        z=pdata[ds]
        im=ax.imshow(
            z[arms].to_numpy(float),
            cmap='RdBu_r',
            norm=TwoSlopeNorm(vcenter=0,vmin=-vmax,vmax=vmax),
            aspect='auto'
        )
        ax.set_title(ds,pad=5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(
            ['E-box +','E-box −','RORE +','RORE −','D-box +','D-box −'],
            rotation=55,ha='right'
        )
        ax.set_yticks(range(5))
        ax.set_yticklabels(LINEAGES)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(.65)

    cb=fig.colorbar(
        im,ax=axs.ravel().tolist(),
        fraction=.025,pad=.025
    )
    cb.set_label('Tissue-residual lineage prototype (z)')

    fig.suptitle(
        'D  Six-arm clock architectures show partial cross-atlas lineage conservation',
        x=.01,y=1.03,ha='left',
        fontsize=11,fontweight='bold'
    )

    stem='MAIN_D_cross_atlas_six_arm_architecture'
    save(fig,stem)

    source_rows=[]
    for ds in order:
        zz=pdata[ds].reset_index().copy()
        zz.insert(0,'dataset',ds)
        source_rows.append(zz)
    write_source(pd.concat(source_rows,ignore_index=True),stem+'_source')

    for ds,pp in proto_files.items():
        add_manifest(
            f'D_{ds}','REDRAW',pp,MAIN/f'{stem}.png',
            'Exact frozen 78 tissue-residual lineage prototype.'
        )

# ============================================================
# E. SAME-TF / SAME-ARM REWIRING — REDRAW
# ============================================================
def find_E_tables():
    coup=[]; cls=[]
    for p in files_for(['81'],['*.csv','*.csv.gz','*.tsv']):
        cols=safe_header(p); tf=pick(cols,exact=('TF','tf','transcription_factor')); lin=pick(cols,exact=('lineage','major_lineage')); arm=pick(cols,exact=('arm','frozen_arm','target_arm','act_arm'))
        rc=next((c for c in cols if norm(c) in {'r','correlation','residualr','samearmr','matchedr','lineager'}),None)
        cc=next((c for c in cols if 'class' in norm(c) or 'rewiring' in norm(c)),None); fdr=next((c for c in cols if 'fdr' in norm(c) and 'heter' in norm(c)),None)
        if tf and lin and rc:
            s=10+(10 if 'specific' in p.name.lower() else 0)+(8 if 'rewir' in p.name.lower() else 0)+(4 if arm else 0); coup.append((s,p,tf,lin,rc,arm))
        if tf and cc:
            s=10+(10 if 'classification' in p.name.lower() else 0)+(8 if 'tf_level' in p.name.lower() else 0); cls.append((s,p,tf,cc,arm,fdr))
    if not coup: raise FileNotFoundError('No 81 same-TF/same-arm coupling table found')
    return sorted(coup,reverse=True)[0], (sorted(cls,reverse=True)[0] if cls else None)

def panel_E():
    C,K=find_E_tables()
    _,p,tf,lin,rc,arm=C
    d=readtab(p)
    d['TF']=d[tf].astype(str).str.upper()
    d['lineage']=d[lin].map(linstd)
    d['r']=pd.to_numeric(d[rc],errors='coerce')
    d['arm']='' if arm is None else d[arm].astype(str)

    info={}
    csrc=None
    if K:
        _,cp,ctf,cc,carm,cfdr=K
        c=readtab(cp); csrc=cp
        for _,row in c.iterrows():
            info[str(row[ctf]).upper()]={
                'class':str(row[cc]),
                'arm':'' if carm is None else str(row[carm]),
                'heterogeneity_FDR':np.nan if cfdr is None else pd.to_numeric(
                    pd.Series([row[cfdr]]),errors='coerce'
                ).iloc[0]
            }

    # FIX3: require the eight preselected interpretable TFs.
    missing=[t for t in MAIN_TFS if t not in set(d.TF)]
    if missing:
        raise RuntimeError(
            "Panel E missing preselected TFs: " + ", ".join(missing)
        )
    chosen=MAIN_TFS[:]

    dd=d[d.TF.isin(chosen)&d.lineage.isin(LINEAGES)].copy()
    mat=(
        dd.pivot_table(
            index='TF',columns='lineage',values='r',aggfunc='mean'
        )
        .reindex(index=chosen,columns=LINEAGES)
    )

    if mat.isna().any().any():
        bad=np.argwhere(mat.isna().to_numpy())
        raise RuntimeError(
            f"Panel E has missing TF-lineage cells: {len(bad)}"
        )

    arms={}
    for t in chosen:
        a=info.get(t,{}).get('arm','')
        if not a or a in {'nan','None'}:
            q=dd.loc[dd.TF.eq(t),'arm']
            q=q[~q.isin(['','nan','None'])]
            a=q.mode().iloc[0] if len(q) else ''
        arms[t]=a
    labels=[f"{t}  [{arms[t]}]" if arms[t] else t for t in chosen]

    fig,(ax,axg)=plt.subplots(
        1,2,figsize=(8.2,4.8),
        gridspec_kw={'width_ratios':[5.6,1.25],'wspace':.10}
    )

    vmax=max(.4,min(1.,np.nanmax(np.abs(mat.values))))
    im=ax.imshow(
        mat.values,cmap='RdBu_r',
        norm=TwoSlopeNorm(vcenter=0,vmin=-vmax,vmax=vmax),
        aspect='auto'
    )
    ax.set_xticks(range(5))
    ax.set_xticklabels(['B','T','Mac/Mono','Fib','Endo'])
    ax.set_yticks(range(len(chosen)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Lineage')
    ax.set_title(
        'Same TF → same frozen clock arm',
        pad=6
    )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v=float(mat.iloc[i,j])
            ax.text(
                j,i,f'{v:.2f}',
                ha='center',va='center',fontsize=6.8,
                color='white' if abs(v)>.55*vmax else 'black'
            )

    # Visual split between shared-sign examples and rewiring examples.
    ax.axhline(3.5,color='white',lw=3.0)
    ax.axhline(3.5,color='#666666',lw=.75)

    # Horizontal colorbar prevents collision with the interpretation column.
    cb=fig.colorbar(
        im,ax=ax,orientation='horizontal',
        fraction=.065,pad=.13,aspect=30
    )
    cb.set_label('Residual TF–clock-arm coupling (r)')

    # Right-side interpretation blocks, not a per-row class list.
    axg.set_xlim(0,1)
    axg.set_ylim(-.5,7.5)
    axg.invert_yaxis()
    axg.axis('off')

    axg.add_patch(Rectangle(
        (.05,-.45),.90,3.90,
        facecolor='#EEF3F8',edgecolor='#AAB7C4',lw=.8
    ))
    axg.text(
        .50,1.50,
        'Shared-sign\nbackbone',
        ha='center',va='center',
        fontsize=8.2,fontweight='bold'
    )
    axg.text(
        .50,2.45,
        'same direction\nacross lineages',
        ha='center',va='center',
        fontsize=6.8,color='#666666'
    )

    axg.add_patch(Rectangle(
        (.05,3.55),.90,3.90,
        facecolor='#FCEDE5',edgecolor='#D9A68E',lw=.8
    ))
    axg.text(
        .50,5.45,
        'Lineage-selective\nrewiring',
        ha='center',va='center',
        fontsize=8.2,fontweight='bold'
    )
    axg.text(
        .50,6.45,
        'direction and/or\nmagnitude changes',
        ha='center',va='center',
        fontsize=6.8,color='#666666'
    )

    fig.suptitle(
        'E  Shared-sign TF–clock coupling coexists with lineage-selective rewiring',
        x=.01,y=1.02,ha='left',
        fontsize=11,fontweight='bold'
    )

    stem='MAIN_E_sameTF_sameArm_rewiring'
    save(fig,stem)
    write_source(dd[['TF','lineage','r','arm']],stem+'_source')
    add_manifest(
        'E','REDRAW',p,MAIN/f'{stem}.png',
        'Main panel groups four shared-sign examples and four rewiring examples; detailed 81 classes remain source/supplement information.'
    )
    if csrc:
        add_manifest(
            'E_class_source','READ_ONLY',csrc,MAIN/f'{stem}.png',
            'Detailed 81 TF-level classes retained for provenance but not printed in the main panel.'
        )
    return chosen,info

# ============================================================
# F. ORTHOGONAL EVIDENCE MATRIX — REDRAW
# ============================================================
def collect_assoc_strict(tfs):
    """
    Exact 79.2 source:
      79_2_Q2_LINEAGE_ADJUSTMENT_SUMMARY.csv

    Use adj_adjusted_r rather than the old heuristic "largest absolute r from
    any 79 table". This is the intended Clock17-excluded + lineage-adjusted
    safeguard result.
    """
    hits=list(ROOT.rglob('79_2_Q2_LINEAGE_ADJUSTMENT_SUMMARY.csv'))
    if not hits:
        raise FileNotFoundError(
            '79_2_Q2_LINEAGE_ADJUSTMENT_SUMMARY.csv not found'
        )
    p=sorted(hits,key=lambda x:x.stat().st_mtime,reverse=True)[0]
    d=readtab(p)

    need={'TF','adj_adjusted_r'}
    if not need.issubset(d.columns):
        raise RuntimeError(
            f'79.2 Q2 table missing columns: {sorted(need-set(d.columns))}'
        )

    d=d.copy()
    d['TF']=d['TF'].astype(str).str.upper()
    d=d[d.TF.isin(tfs)].copy()

    cols=['TF','adj_adjusted_r']
    for c in [
        'adj_arm','adj_perm_P','adj_perm_FDR','adj_maxT_FWER',
        'persists_after_lineage_adjustment','mechanism_tier'
    ]:
        if c in d.columns:
            cols.append(c)

    out=d[cols].copy()
    out=out.drop_duplicates('TF',keep='first')
    out['adj_adjusted_r']=pd.to_numeric(
        out['adj_adjusted_r'],errors='coerce'
    )
    return out,p

def _exact_col(cols, names):
    m={norm(c):c for c in cols}
    for x in names:
        if norm(x) in m:
            return m[norm(x)]
    return None

def collect_ext_strict(tfs):
    """
    Conservative external-evidence parser.

    Unlike FIX2, numeric values are NEVER interpreted as profile counts merely
    because a column name contains "profile". Counts are used only from exact
    whitelisted count columns, or are computed from an explicit profile-level
    KnockTF table.

    The function also writes F_EXTERNAL_SOURCE_AUDIT.csv.
    """
    rec={
        t:{
            'binding_evaluable':np.nan,
            'binding_support':np.nan,
            'perturb_evaluable':np.nan,
            'perturb_support':np.nan,
            'perturb_n_profiles':np.nan,
            'perturb_n_concordant':np.nan,
            'perturb_concordance':np.nan,
            'binding_source':'',
            'perturb_source':'',
        }
        for t in tfs
    }
    audit=[]

    files=files_for(['80'],['*.csv','*.csv.gz','*.tsv'])
    for p in files:
        cols=safe_header(p)
        tfc=_exact_col(cols,['TF','transcription_factor'])
        if tfc is None:
            continue

        try:
            d=readtab(p)
        except Exception as e:
            audit.append({
                'file':str(p),'used':False,'reason':f'read_error:{e}'
            })
            continue

        d=d.copy()
        d['__TF']=d[tfc].astype(str).str.upper()
        q=d[d.__TF.isin(tfs)].copy()
        if q.empty:
            continue

        cmap={norm(c):c for c in cols}
        used=False
        reasons=[]

        # -----------------------------------------------------
        # A. Generic domain/evaluable/support summary
        # -----------------------------------------------------
        domain_c=_exact_col(cols,['domain','evidence_domain','source_domain'])
        eval_c=_exact_col(cols,['evaluable','is_evaluable'])
        support_c=_exact_col(cols,['support','supported','is_supported'])

        if domain_c and eval_c and support_c:
            for _,row in q.iterrows():
                T=row['__TF']
                domain=str(row[domain_c]).lower()
                ev=boolish(row[eval_c])
                sp=boolish(row[support_c])
                if 'bind' in domain or 'chip' in domain:
                    if not pd.isna(ev): rec[T]['binding_evaluable']=bool(ev)
                    if not pd.isna(sp): rec[T]['binding_support']=bool(sp)
                    rec[T]['binding_source']=str(p)
                    used=True
                if 'perturb' in domain or 'knocktf' in domain:
                    if not pd.isna(ev): rec[T]['perturb_evaluable']=bool(ev)
                    if not pd.isna(sp): rec[T]['perturb_support']=bool(sp)
                    rec[T]['perturb_source']=str(p)
                    used=True
            if used:
                reasons.append('domain_summary')

        # -----------------------------------------------------
        # B. Explicit wide summary columns only
        # -----------------------------------------------------
        be=_exact_col(cols,[
            'binding_evaluable','binding_eval',
            'chip_evaluable','chip_eval'
        ])
        bs=_exact_col(cols,[
            'binding_support','binding_supported',
            'chip_support','chip_supported'
        ])
        pe=_exact_col(cols,[
            'perturb_evaluable','perturbation_evaluable',
            'knocktf_evaluable','perturb_eval'
        ])
        ps=_exact_col(cols,[
            'perturb_support','perturbation_support',
            'knocktf_support','perturb_supported'
        ])
        npf=_exact_col(cols,[
            'n_profiles','n_profiles_used','n_total_profiles',
            'n_perturbation_profiles','n_knocktf_profiles'
        ])
        nco=_exact_col(cols,[
            'n_concordant','n_concordant_profiles',
            'n_direction_consistent','n_consistent_profiles'
        ])
        frc=_exact_col(cols,[
            'concordant_fraction','fraction_concordant',
            'direction_consistency','direction_concordance',
            'same_sign_fraction'
        ])

        if any(x is not None for x in [be,bs,pe,ps,npf,nco,frc]):
            for T,sub in q.groupby('__TF'):
                row=sub.iloc[0]
                if be:
                    v=boolish(row[be])
                    if not pd.isna(v): rec[T]['binding_evaluable']=bool(v)
                    rec[T]['binding_source']=str(p); used=True
                if bs:
                    v=boolish(row[bs])
                    if not pd.isna(v): rec[T]['binding_support']=bool(v)
                    rec[T]['binding_source']=str(p); used=True
                if pe:
                    v=boolish(row[pe])
                    if not pd.isna(v): rec[T]['perturb_evaluable']=bool(v)
                    rec[T]['perturb_source']=str(p); used=True
                if ps:
                    v=boolish(row[ps])
                    if not pd.isna(v): rec[T]['perturb_support']=bool(v)
                    rec[T]['perturb_source']=str(p); used=True
                if npf:
                    v=pd.to_numeric(pd.Series([row[npf]]),errors='coerce').iloc[0]
                    if pd.notna(v): rec[T]['perturb_n_profiles']=float(v)
                    rec[T]['perturb_source']=str(p); used=True
                if nco:
                    v=pd.to_numeric(pd.Series([row[nco]]),errors='coerce').iloc[0]
                    if pd.notna(v): rec[T]['perturb_n_concordant']=float(v)
                    rec[T]['perturb_source']=str(p); used=True
                if frc:
                    v=pd.to_numeric(pd.Series([row[frc]]),errors='coerce').iloc[0]
                    if pd.notna(v):
                        v=float(v)
                        if v>1 and v<=100: v/=100.
                        rec[T]['perturb_concordance']=v
                    rec[T]['perturb_source']=str(p); used=True
            if used:
                reasons.append('explicit_summary_columns')

        # -----------------------------------------------------
        # C. Explicit KnockTF profile-level table
        # -----------------------------------------------------
        lname=p.name.lower()
        if 'knocktf' in lname:
            prof=_exact_col(cols,[
                'profile_id','profile','experiment_id','experiment','dataset_id'
            ])
            cons=_exact_col(cols,[
                'direction_consistent','direction_concordant',
                'concordant','consistent'
            ])
            if prof and cons:
                for T,sub in q.groupby('__TF'):
                    ss=sub[[prof,cons]].dropna(subset=[prof]).copy()
                    if len(ss):
                        ss=ss.drop_duplicates(prof)
                        vals=ss[cons].map(boolish)
                        valid=vals.notna()
                        if valid.any():
                            n=int(valid.sum())
                            k=int(vals[valid].astype(bool).sum())
                            rec[T]['perturb_evaluable']=True
                            rec[T]['perturb_n_profiles']=n
                            rec[T]['perturb_n_concordant']=k
                            rec[T]['perturb_concordance']=k/n
                            rec[T]['perturb_support']=bool(k>0)
                            rec[T]['perturb_source']=str(p)
                            used=True
                if used:
                    reasons.append('knocktf_profile_level')

        # -----------------------------------------------------
        # D. Explicit binding-detail table: a returned TF/clock-target row is
        # support for that TF. This does not infer "no support" from absence.
        # -----------------------------------------------------
        if any(k in lname for k in ['chip','binding']):
            target=_exact_col(cols,[
                'clock_target','target_gene','target','gene'
            ])
            if target:
                for T,sub in q.groupby('__TF'):
                    vals=sub[target].dropna().astype(str)
                    vals=vals[vals.str.len()>0]
                    if len(vals):
                        rec[T]['binding_evaluable']=True
                        rec[T]['binding_support']=True
                        rec[T]['binding_source']=str(p)
                        used=True
                if used:
                    reasons.append('binding_detail_rows')

        audit.append({
            'file':str(p),
            'used':bool(used),
            'reason':';'.join(sorted(set(reasons))) if reasons else 'no_strict_schema_match',
            'columns':' | '.join(map(str,cols[:50]))
        })

    pd.DataFrame(audit).to_csv(
        AUDIT/'F_EXTERNAL_SOURCE_AUDIT.csv',index=False
    )

    out=[]
    for T in tfs:
        z=rec[T].copy()
        z['TF']=T
        out.append(z)
    return pd.DataFrame(out)

def _support_text(evaluable,support):
    if pd.isna(evaluable) or not bool(evaluable):
        return 'Not evaluable'
    if pd.isna(support):
        return 'Evaluable'
    return 'Supported' if bool(support) else 'No support'

def panel_F(tfs,info):
    a,asrc=collect_assoc_strict(tfs)
    e=collect_ext_strict(tfs)
    d=(
        pd.DataFrame({'TF':tfs})
        .merge(a,on='TF',how='left')
        .merge(e,on='TF',how='left')
    )

    fig,ax=plt.subplots(figsize=(8.6,4.9))
    ax.set_xlim(-.5,2.5)
    ax.set_ylim(-.5,len(d)-.5)
    ax.invert_yaxis()
    ax.set_xticks([0,1,2])
    ax.set_xticklabels([
        'Clock17-excluded +\nlineage-adjusted association',
        'Public TF binding\nto clock target(s)',
        'KnockTF perturbation'
    ])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.TF)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Background table cells.
    for i in range(len(d)):
        for j in range(3):
            ax.add_patch(Rectangle(
                (j-.47,i-.43),.94,.86,
                facecolor='white',
                edgecolor='#E1E1E1',lw=.8
            ))

    max_abs=np.nanmax(np.abs(pd.to_numeric(d['adj_adjusted_r'],errors='coerce')))
    if not np.isfinite(max_abs) or max_abs==0:
        max_abs=1.0

    for i,row in d.iterrows():
        # Column 1: exact safeguarded residual association.
        r=pd.to_numeric(pd.Series([row.get('adj_adjusted_r',np.nan)]),errors='coerce').iloc[0]
        if pd.notna(r):
            frac=min(1.0,abs(float(r))/max_abs)
            cmap=plt.get_cmap('RdBu_r')
            color=cmap(.5 + .45*np.sign(r)*frac)
            ax.add_patch(Rectangle(
                (-.43,i-.34),.86,.68,
                facecolor=color,edgecolor='none',alpha=.92
            ))
            ax.text(
                0,i,f"r = {float(r):+.2f}",
                ha='center',va='center',
                fontsize=7.5,
                color='white' if abs(float(r))>.45*max_abs else 'black',
                fontweight='bold'
            )
        else:
            ax.text(0,i,'Not resolved',ha='center',va='center',fontsize=7,color='#777')

        # Column 2: public binding.
        bt=_support_text(
            row.get('binding_evaluable',np.nan),
            row.get('binding_support',np.nan)
        )
        bcolor=(
            '#E8F1E8' if bt=='Supported' else
            '#F4F4F4' if bt=='Not evaluable' else
            '#F7ECEC'
        )
        ax.add_patch(Rectangle(
            (1-.43,i-.34),.86,.68,
            facecolor=bcolor,edgecolor='none'
        ))
        ax.text(
            1,i,bt,
            ha='center',va='center',
            fontsize=7.1,
            color='#2E5E2E' if bt=='Supported' else '#666666'
        )

        # Column 3: perturbation. Show raw profile-level information only when
        # it was resolved from an exact profile-count schema.
        pe=row.get('perturb_evaluable',np.nan)
        ps=row.get('perturb_support',np.nan)
        n=row.get('perturb_n_profiles',np.nan)
        k=row.get('perturb_n_concordant',np.nan)
        frac=row.get('perturb_concordance',np.nan)

        if pd.isna(pe) or not bool(pe):
            ptxt='Not evaluable'
            pcolor='#F4F4F4'
        elif pd.notna(n) and pd.notna(k):
            ptxt=f"{int(round(k))}/{int(round(n))} concordant"
            pcolor='#EEE7F5' if int(round(k))>0 else '#F7ECEC'
        elif pd.notna(n) and pd.notna(frac):
            kk=int(round(float(frac)*float(n)))
            ptxt=f"{kk}/{int(round(n))} concordant"
            pcolor='#EEE7F5' if kk>0 else '#F7ECEC'
        elif pd.notna(ps):
            ptxt='Supported' if bool(ps) else 'Evaluated, mixed/none'
            pcolor='#EEE7F5' if bool(ps) else '#F7ECEC'
        else:
            ptxt='Evaluable'
            pcolor='#F4F4F4'

        ax.add_patch(Rectangle(
            (2-.43,i-.34),.86,.68,
            facecolor=pcolor,edgecolor='none'
        ))
        ax.text(
            2,i,ptxt,
            ha='center',va='center',
            fontsize=7.0,
            color='#5E3C76' if 'concordant' in ptxt or ptxt=='Supported' else '#666666'
        )

    # Separator mirrors Panel E's conceptual split.
    ax.axhline(3.5,color='#777777',lw=.8)
    ax.text(
        -0.04,1.5,'shared-sign examples',
        transform=ax.get_yaxis_transform(),
        ha='right',va='center',
        fontsize=6.8,color='#5B6F82'
    )
    ax.text(
        -0.04,5.5,'rewiring examples',
        transform=ax.get_yaxis_transform(),
        ha='right',va='center',
        fontsize=6.8,color='#A46449'
    )

    ax.set_title(
        'F  Orthogonal evidence supports candidate TF–clock links',
        loc='left',fontweight='bold',pad=12
    )

    fig.text(
        .125,.015,
        'Association: TF activity–clock-arm residual correlation after Clock17 exclusion and lineage adjustment.  '
        'Binding: public TF→clock-target evidence.  Perturbation: KnockTF profile-level direction when evaluable.',
        ha='left',va='bottom',fontsize=6.7,color='#555555'
    )
    fig.text(
        .125,-.012,
        '“Not evaluable” means the frozen external resource did not provide an interpretable test for that TF; no composite evidence score is used.',
        ha='left',va='bottom',fontsize=6.7,color='#777777'
    )

    stem='MAIN_F_orthogonal_regulatory_evidence'
    save(fig,stem)
    write_source(d,stem+'_source')

    add_manifest(
        'F','REDRAW',asrc,MAIN/f'{stem}.png',
        'FIX3 uses the exact 79.2 lineage-adjusted table and conservative external-evidence parsing; no heuristic profile counts.'
    )

# ============================================================
# SUPPLEMENT
# ============================================================
def supplement_copy():
    jobs=[
        (['73a','highconf','cell','counts'],'S2_73_highconf_cell_counts.png'),
        (['73c','clock17','phase','heatmap'],'S3A_73_clock17_phase_heatmap.png'),
        (['73c','clock17','r2','heatmap'],'S3B_73_clock17_R2_heatmap.png'),
        (['73e','six','arm','phase','heatmap'],'S3C_73_six_arm_phase_heatmap.png'),
        (['74g','fig2a','pairwise','lineage','distance'],'S4A_74G_pairwise_lineage_distance.png'),
        (['74g','fig3b','component','variability','heatmap'],'S4B_74G_component_variability.png'),
        (['76_core_raw','umap','lineage'],'S5A_76_core_raw_lineage_UMAP.png'),
        (['76_core_time_resid','umap','lineage'],'S5B_76_core_time_resid_lineage_UMAP.png'),
        (['76_expanded_raw','umap','lineage'],'S5C_76_expanded_raw_lineage_UMAP.png'),
        (['76_expanded_time_resid','umap','lineage'],'S5D_76_expanded_time_resid_lineage_UMAP.png'),
        (['79','clock17','excluded'],'S7_79_2_Clock17_excluded_sensitivity.png'),
        (['79','residual'],'S8_79_2_residualized_TF_arm_coupling.png'),
        (['80','fix2b','external','evidence'],'S9A_80_external_evaluability_support.png'),
        (['80','fix2c','knocktf'],'S9B_80_KnockTF_profiles.png'),
        (['77_f5','phase','jitter'],'S10_OPTIONAL_77_phase_jitter.png'),
    ]
    for tok,dest in jobs:
        try: copy_first(tok,dest,SUPP,required=False)
        except Exception as e: warnings.append(f'{dest}: {e}')

def supp_S6():
    vals=pd.DataFrame([
        {'dataset':'Tabula','mean_gain':0.043021,'P':0.141865},
        {'dataset':'HCL','mean_gain':-0.058167,'P':0.719299},
        {'dataset':'GSE292219','mean_gain':-0.265563,'P':1.0},
    ])
    reps=[p for p in files_for(['79'],['*.md','*.txt']) if p.name.lower()=='79_fix2_report.md']; src=reps[0] if reps else None
    fig,ax=plt.subplots(figsize=(4.6,3.4)); x=np.arange(3); ax.bar(x,vals.mean_gain,color=['#4E79A7','#59A14F','#9467BD'],width=.62); ax.axhline(0,color='#555',lw=.8); ax.set_xticks(x); ax.set_xticklabels(vals.dataset); ax.set_ylabel('Held-out interaction gain\n(additive MSE − interaction MSE)')
    for i,r in vals.iterrows():
        y=r.mean_gain; ax.text(i,y+(.012 if y>=0 else -.018),('P = 1' if r.P==1 else f'P = {r.P:.3f}'),ha='center',va='bottom' if y>=0 else 'top',fontsize=7)
    ax.set_title('S6  Strict held-out tests do not support generalizable L×context interactions',loc='left',fontweight='bold',pad=7); clean(ax); stem='S6_79_strict_interaction_audit'; save(fig,stem,SUPP); write_source(vals,stem+'_source'); add_manifest('S6','REDRAW',src,SUPP/f'{stem}.png','Exact strict 79 FIX2 report values; AHCA is non-estimable (single donor).')

# ============================================================
# OUTPUT DOCS
# ============================================================
def docs():
    readme=f'''# 82 final time-validation → TF-rewiring figure package\n\nMain figure:\nA five-lineage shared time program → B lineage structure persists after time removal → C tissue-held-out portability → D partial cross-atlas six-arm conservation → E shared-sign vs rewired TF–clock coupling → F explicit orthogonal evidence table.\n\nMain claim: **Shared circadian timing coexists with stable lineage-associated clock architecture and lineage-selective TF–clock coupling.**\n\nClaim boundary: TF–clock-arm coupling is observational. Binding/perturbation data support candidate regulatory direction but do not establish universal causality.\n\nFrozen palette:\n- B cell #4E79A7\n- T cell #E15759\n- Macrophage/Monocyte #59A14F\n- Fibroblast #F28E2B\n- Endothelial #9467BD\n\nNew panels export PNG 600 dpi + PDF + SVG. Existing finalized 78 panels are copied unchanged.\n'''
    (OUT/'README.md').write_text(readme,encoding='utf-8'); pd.DataFrame(manifest).to_csv(OUT/'PANEL_MANIFEST.csv',index=False); (AUDIT/'WARNINGS.txt').write_text('\n'.join(warnings) if warnings else 'NONE\n',encoding='utf-8'); (AUDIT/'ERRORS.txt').write_text('\n'.join(errors) if errors else 'NONE\n',encoding='utf-8')

# ============================================================
# MAIN
# ============================================================
def run(label,fn):
    print(f'[{label}]')
    try: return fn()
    except Exception as e: errors.append(f'{label}: {type(e).__name__}: {e}'); print('  ERROR:',e); return None

def main():
    if not ROOT.exists(): raise FileNotFoundError(ROOT)
    print('='*88); print('82 FIX3 FINAL TIME-VALIDATION / TF-REWIRING FIGURE PACKAGE'); print('='*88); print('OUT:',OUT)
    run('A redraw',panel_A); run('B redraw',panel_B); run('C/D copy finalized 78',panels_CD)
    e=run('E redraw',panel_E)
    if e: run('F redraw',lambda:panel_F(e[0],e[1]))
    else: errors.append('F: skipped because E could not be resolved')
    run('Supplement copy',supplement_copy); run('S6 redraw',supp_S6); docs()
    print('\nMAIN:',MAIN); print('SUPPLEMENT:',SUPP); print('MANIFEST:',OUT/'PANEL_MANIFEST.csv')
    if errors:
        print('\nCOMPLETED WITH UNRESOLVED PANELS — no data were fabricated.'); print('\n'.join(errors)); raise RuntimeError('See AUDIT/ERRORS.txt')
    print('\n82_FIX3_FINAL_FIGURE_PACKAGE_COMPLETE = TRUE')

if __name__=='__main__': main()
