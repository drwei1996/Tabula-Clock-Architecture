#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure S6G v3 | FIVE lineage clocks showing a SINGLE predominant B0 time per tissue.

Each circle represents exactly ONE lineage (B, T, Macrophage/Monocyte,
Fibroblast, Endothelial). Each circle contains ALL tissues as angular sectors,
with 12 concentric radial phase tracks (outer: B0 0-2 h; inner: 22-24 h).
Within each tissue×lineage the tile intensity encodes the donor-equal mean
frozen 48-bin posterior aggregated to 12×2-h bins.

DATA: Reuses the VERIFIED V4 output CSV if present:
  FIGS6_G_ALL_TISSUES_CIRCULAR_V4/FIGS6_G_all_tissues_phase_source_data.csv
If absent, reads the ACTUAL frozen manifest schema from:
  08_PHASE_ATLAS/TABULA_B0_RUN_MANIFEST.csv
  columns: status,tissue_general,assay,n_cells,...,output_dir
Then reads CELL_PHASE.csv.gz, PHASE_POSTERIOR.npz and H5AD metadata via
strict full unique cell-ID joins (no guessed lineage from CELL_PHASE).

Frozen B0 was independently fitted per tissue×assay cohort: the same bin
position across tissue sectors does not imply an aligned real-world time.
This is a descriptive atlas, NOT observed circadian timing validation.

Outputs text and notext PNG/PDF/SVG, plus source-data, coverage and audit.

Usage:
python plot_FigS6G_five_lineage_all_tissues_v2.py \
  --data-root "$PWD" \
  --outdir FIGS6_G_FIVE_LINEAGE_CIRCOS_V2
"""
from __future__ import annotations

import argparse
import json
import re
from matplotlib.colors import to_rgb
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LINEAGES = ['B cell', 'T cell', 'Macrophage/Monocyte', 'Fibroblast', 'Endothelial']
SHORT = ['B cell', 'T cell', 'Macro', 'Fibro', 'Endo']
COLORS = ['#5485C6', '#CE5E6B', '#E1A041', '#65B19C', '#8A65BC']
PHASE_CENTERS = np.arange(1, 24, 2, dtype=float)
ID_FIELDS = ['cell_id', 'census_cell_id', 'soma_joinid', 'barcode', 'cell_name']
DONOR_FIELDS = ['donor_id','donor','donor_identifier','donor_uuid','individual_id',
                'individual','donor_label','donor_name']
CELLTYPE_FIELDS = ['cell_type','cell_type_label','cell_type_name',
                   'cell_ontology_class','cell_type_ontology_term']

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
    'pdf.fonttype':42,'ps.fonttype':42,
})


def st(x):
    return x.decode('utf-8','replace') if isinstance(x,bytes) else str(x)


def h5_index(g):
    name=st(g.attrs.get('_index','_index'))
    if name not in g: name='_index'
    if name not in g: raise ValueError(f'No index dataset in {g.name}')
    return np.asarray([st(x) for x in g[name][...]],dtype=object)


def h5_col(g,name):
    if name not in g: return None
    x=g[name]
    if isinstance(x,h5py.Dataset) and x.ndim==1:
        return np.asarray([st(y) for y in x[...]],dtype=object)
    if isinstance(x,h5py.Group) and {'codes','categories'}.issubset(x.keys()):
        c=np.asarray(x['codes'][...],dtype=int)
        cats=np.asarray([st(y) for y in x['categories'][...]],dtype=object)
        result=np.empty(len(c),dtype=object)
        for i,j in enumerate(c):result[i]=cats[j] if 0<=j<len(cats) else ''
        return result
    return None


def relocate_path(p,root):
    p=Path(str(p))
    if p.exists():return p
    t=str(p).replace('\\','/')
    for marker in ['/08_PHASE_ATLAS/','/03_TABULA_INPUT/']:
        if marker in t:
            return root/marker.strip('/')/t.split(marker,1)[1]
    return root/p if not p.is_absolute() else p


def manifest(root):
    for p in [root/'08_PHASE_ATLAS'/'TABULA_B0_RUN_MANIFEST.csv',
              root/'TABULA_B0_RUN_MANIFEST.csv']:
        if p.is_file():
            d=pd.read_csv(p)
            req={'tissue_general','assay','output_dir','status'}
            if not req.issubset(d.columns):raise ValueError(f'Manifest missing {req-set(d.columns)}')
            return d[d.status.eq('COMPLETE')].copy(),p
    raise FileNotFoundError('Cannot locate frozen B0 manifest under 08_PHASE_ATLAS')


def lineage_of(celltype):
    t=str(celltype).lower()
    hits=[]
    if re.search(r'\bb[\s-]*cells?\b|plasmablast|plasma cell',t):hits.append('B cell')
    if re.search(r'\bt[\s-]*cells?\b|cd[48]-positive.*lymphocyte|alpha-beta t',t):hits.append('T cell')
    if re.search(r'macrophage|monocyte|kupffer',t):hits.append('Macrophage/Monocyte')
    if 'fibroblast' in t:hits.append('Fibroblast')
    if 'endothelial' in t:hits.append('Endothelial')
    return hits[0] if len(set(hits))==1 else ''


def aligned_obs(pred,h5,assay):
    ids=pd.Index(pred['cell_id'].astype(str))
    if not ids.is_unique:raise ValueError('Duplicate frozen cell_id')
    obs=h5['obs']
    possible={'obs_names':h5_index(obs)}
    for key in ID_FIELDS:
        v=h5_col(obs,key)
        if v is not None:possible[key]=v
    choices=[]
    for key,v in possible.items():
        ix=pd.Index(v.astype(str))
        if not ix.is_unique:continue
        rows=ix.get_indexer(ids)
        if np.all(rows>=0):choices.append((key,rows))
    if not choices:
        raise ValueError('No FULL unique cell-ID match to source H5AD. Refusing positional alignment.')
    name,rows=choices[0]
    assaycol=h5_col(obs,'assay')
    if assaycol is not None:
        if not np.all(assaycol[rows]==assay):
            raise ValueError('Frozen cell-ID mapping disagrees with source H5AD assay')
        if np.count_nonzero(assaycol==assay)!=len(rows):
            raise ValueError('Source H5AD assay cell count does not match frozen cohort')
    dfield=next((k for k in DONOR_FIELDS if h5_col(obs,k) is not None),None)
    cfield=next((k for k in CELLTYPE_FIELDS if h5_col(obs,k) is not None),None)
    if dfield is None or cfield is None:
        raise ValueError(f'Missing donor/cell-type annotation in {h5.filename}')
    donors=h5_col(obs,dfield)[rows]
    types=h5_col(obs,cfield)[rows]
    lineage=np.asarray([lineage_of(x) for x in types],dtype=object)
    return donors,lineage,dict(cell_ID_match_field=name,donor_field=dfield,cell_type_field=cfield)


def posterior_matrix_12(path,n,R_ref):
    with np.load(path,allow_pickle=False) as obj:
        names=[k for k in obj.files if obj[k].ndim==2 and obj[k].shape==(n,48)]
        if not names:raise ValueError(f'Expected an N×48 posterior matrix in {path}')
        names.sort(key=lambda x:x not in ('posterior','q','q_theta','phase_posterior'))
        key=names[0]
        a=np.asarray(obj[key],dtype=np.float64)
    if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Invalid posterior values')
    if np.max(np.abs(a.sum(axis=1)-1.0))>1e-4:
        raise ValueError('Posterior rows are not normalized')
    circ=np.exp(2j*np.pi*np.arange(48)/48)
    calc_R=np.abs(a@circ)
    diff=float(np.max(np.abs(calc_R-R_ref)))
    if diff>2e-5:raise ValueError(f'Frozen posterior vs R_B0 mismatch: {diff:.5g}')
    return a.reshape(n,12,4).sum(axis=2),key


def cohort_profile(root,row,min_r=0.0,min_cells=60,min_donors=1):
    tissue=str(row['tissue_general']);assay=str(row['assay'])
    run=relocate_path(row['output_dir'],root)
    cp=run/'CELL_PHASE.csv.gz';pp=run/'PHASE_POSTERIOR.npz';qc=run/'QC_SUMMARY.json'
    for path in [cp,pp,qc]:
        if not path.is_file():raise FileNotFoundError(f'Missing frozen B0 file: {path}')
    pred=pd.read_csv(cp,usecols=['cell_id','R_B0'])
    if 'n_cells' in row and pd.notna(row['n_cells']) and len(pred)!=int(row['n_cells']):
        raise ValueError('Manifest and CELL_PHASE cell counts disagree')
    q=json.loads(qc.read_text(encoding='utf8'))
    source=relocate_path(q.get('source_h5ad') or q.get('source_file') or '',root)
    if not source.is_file():raise FileNotFoundError(f'Missing source H5AD: {source}')
    with h5py.File(source,'r') as h5:
        donors,lineage,audit=aligned_obs(pred,h5,assay)
    probs,posterior_key=posterior_matrix_12(pp,len(pred),pred.R_B0.to_numpy(float))
    valid=(pred.R_B0.to_numpy(float)>=min_r) & np.isin(lineage,LINEAGES)
    valid &= np.asarray([str(x).strip().lower() not in ('','nan','none','unknown','na') for x in donors])
    data=[]
    meta=[]
    for lin in LINEAGES:
        sub=np.flatnonzero(valid & (lineage==lin))
        if len(sub)<min_cells:
            meta.append(dict(tissue=tissue,assay=assay,lineage=lin,
                             n_cells=len(sub),n_donors=0,displayed=False,reason='LOW_CELL_COUNT'))
            continue
        donor_avgs=[];individual_counts=[]
        for name in np.unique(donors[sub]):
            ix=sub[donors[sub]==name]
            if not len(ix):continue
            pp_d=probs[ix].mean(axis=0)
            donor_avgs.append(pp_d)
            individual_counts.append(len(ix))
        if len(donor_avgs)<min_donors:
            meta.append(dict(tissue=tissue,assay=assay,lineage=lin,
                             n_cells=len(sub),n_donors=len(donor_avgs),displayed=False,reason='LOW_DONOR_COUNT'))
            continue
        prob=np.mean(np.vstack(donor_avgs),axis=0)
        if not np.isclose(prob.sum(),1,atol=1e-6):raise AssertionError('Bad donor-equal distribution normalization')
        meta.append(dict(tissue=tissue,assay=assay,lineage=lin,
                         n_cells=len(sub),n_donors=len(donor_avgs),displayed=True,reason='OK'))
        for j,v in enumerate(prob):
            data.append(dict(tissue=tissue,assay=assay,lineage=lin,
                             phase_bin=j,phase_center_h=float(PHASE_CENTERS[j]),
                             donor_equal_posterior_fraction=float(v),
                             n_eligible_cells=int(len(sub)),n_contributing_donors=int(len(donor_avgs))))
    audit.update(dict(tissue=tissue,assay=assay,source_h5ad=str(source),
                      frozen_run_dir=str(run),n_frozen_cells=int(len(pred)),
                      posterior_key=posterior_key,bin_width_h=2.0,
                      min_r=min_r,min_cells=min_cells,min_donors=min_donors))
    return pd.DataFrame(data),pd.DataFrame(meta),audit





# ------------------------------ FIG S6G V3 ----------------------------------
# 5 independent clocks, one per lineage.
# Tissue = angular sector. Time = radial location of a single marker AND
# cyclic time color on the outer tissue band. This IS NOT a posterior heatmap.
# Input remains the same 12-bin donor-equal frozen B0 posterior summaries.
# Circular mean of posterior (default) or modal bin (option) is descriptive.
# ---------------------------------------------------------------------------
from matplotlib.colors import to_rgba, to_hex, Normalize
from matplotlib import cm
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D

NAVY='#223750'
MUTED='#6C7D91'
GREY='#DDE2E9'
BACKGROUND='#FFFFFF'
TIME_CMAP=plt.get_cmap('twilight_shifted')
TIME_CMAP_NAME='twilight_shifted'


def load_all_tissues(root, selected, args):
    samples=[]; metadata=[]; audits=[]; failed=[]
    cache_csv=Path(args.outdir)/'FIGS6_G_all_tissues_phase_source_data.csv'
    cache_meta=Path(args.outdir)/'FIGS6_G_lineage_coverage.csv'
    if args.cache_only:
        if not cache_csv.exists() or not cache_meta.exists():
            raise FileNotFoundError('Cache data missing. Run without --cache-only first.')
        return pd.read_csv(cache_csv),pd.read_csv(cache_meta),[],[]
    n_total=len(selected)
    for i,(_,row) in enumerate(selected.iterrows(),start=1):
        tissue=str(row['tissue_general'])
        assay=str(row['assay'])
        print(f'[{i}/{n_total}] {tissue} | {assay} ...',flush=True)
        try:
            d, m, audit=cohort_profile(root,row,args.min_r,args.min_cells,args.min_donors)
            if not d.empty:
                samples.append(d)
            metadata.append(m)
            audits.append(audit)
            supported=[r for r in m.itertuples(index=False) if r.displayed]
            print(f'  OK: {len(supported)}/5 lineages with data; missing lineages appear grey',flush=True)
        except Exception as e:
            failed.append(dict(tissue=tissue,assay=assay,error_type=type(e).__name__,reason=str(e)))
            print(f'  ERROR: {type(e).__name__}: {e}',flush=True)
            if not args.skip_missing:
                raise RuntimeError(
                    f'Failed to read {tissue} | {assay}. Run --skip-missing to continue and log unavailable cohorts.'
                ) from e
    if not samples:
        raise RuntimeError('No lineage posterior profiles passed minimum coverage; lower --min-cells and inspect cohort annotations.')
    data=pd.concat(samples,ignore_index=True)
    meta=pd.concat(metadata,ignore_index=True)
    data.to_csv(cache_csv,index=False)
    meta.to_csv(cache_meta,index=False)
    (Path(args.outdir)/'FIGS6_G_run_audit.json').write_text(
        json.dumps({'assay':args.assay,'selected_cohorts':n_total,'successful_cohorts':len(audits),
                    'posterior_bins':12,'bin_width_h':2.0,
                    'quantification':'donor-equal mean of per-cell frozen posterior probabilities',
                    'interpretation':'cohort-specific inferred B0 only; NOT observed time',
                    'cohorts':audits},indent=2,ensure_ascii=False),encoding='utf-8'
    )
    pd.DataFrame(failed,columns=['tissue','assay','error_type','reason']).to_csv(
        Path(args.outdir)/'FIGS6_G_skipped_cohorts.csv',index=False)
    return data,meta,audits,failed


def prepare_arrays(data,meta,order):
    lookup={}
    for (tissue,lineage),sub in data.groupby(['tissue','lineage'],sort=False):
        a=np.full(12,np.nan)
        for r in sub.itertuples(index=False):
            a[int(r.phase_bin)]=float(r.donor_equal_posterior_fraction)
        if np.isfinite(a).all():
            lookup[(str(tissue),str(lineage))]=a
    meta_lookup={(str(r.tissue),str(r.lineage)):r for r in meta.itertuples(index=False)}
    return lookup,meta_lookup


def text_orientation(theta):
    # 0 at top, clockwise; tangent orientation with left-half flip for legibility.
    angle_deg=(90-np.degrees(theta))
    if 90 < (np.degrees(theta)%360) < 270:
        angle_deg+=180
        ha='right'
    else:
        ha='left'
    return angle_deg,ha


def data_from_v4_cache(root: Path, args):
    """Reuse actual V4 saved values. NEVER load demo/mock data automatically."""
    candidates = []
    if args.source_csv:
        candidates.append(Path(args.source_csv).expanduser())
    else:
        candidates += [
            root / 'FIGS6_G_ALL_TISSUES_CIRCULAR_V4' / 'FIGS6_G_all_tissues_phase_source_data.csv',
            root / 'FIGS6_G_CIRCULAR_V4' / 'FIGS6_G_all_tissues_phase_source_data.csv',
            root / 'FIGS6_G_ALL_TISSUES_CIRCULAR_V4_1' / 'FIGS6_G_all_tissues_phase_source_data.csv',
            root / 'FIGS6_G_FIVE_LINEAGE_CIRCOS_V2' / 'FIGS6_G_five_lineage_phase_source_data.csv',
        ]
    for csv_path in candidates:
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        if not csv_path.is_file():
            continue
        d = pd.read_csv(csv_path)
        needed = {'tissue', 'lineage', 'phase_bin', 'donor_equal_posterior_fraction'}
        if not needed.issubset(d):
            raise ValueError(f'Invalid V4 source CSV {csv_path}, missing: {needed-set(d)}')
        d = d.copy()
        if 'assay' in d:
            d = d[d['assay'].astype(str).eq(args.assay)].copy()
        d = d[d['lineage'].isin(LINEAGES)].copy()
        if d.empty:
            raise ValueError(f'V4 source CSV contains no rows for selected assay {args.assay!r}: {csv_path}')
        d['phase_bin'] = pd.to_numeric(d['phase_bin'],errors='raise').astype(int)
        if not d.phase_bin.between(0,11).all():
            raise ValueError('V4 cached phase_bin is not 0..11')
        d['donor_equal_posterior_fraction'] = pd.to_numeric(
            d['donor_equal_posterior_fraction'],errors='raise')
        if d[['tissue','lineage','phase_bin']].duplicated().any():
            raise ValueError('Duplicate tissue×lineage×phase_bin in V4 cache: would pool cohorts ambiguously')
        for (t,l), g in d.groupby(['tissue','lineage']):
            if len(g)!=12 or not np.isclose(g.donor_equal_posterior_fraction.sum(),1.,atol=2e-3):
                raise ValueError(f'Incomplete/non-normalized posterior fraction {t} | {l}')
        meta_file = csv_path.parent/'FIGS6_G_lineage_coverage.csv'
        if meta_file.is_file():
            meta = pd.read_csv(meta_file)
        else:
            meta = (d[['tissue','assay','lineage','n_eligible_cells','n_contributing_donors']]
                    .drop_duplicates().rename(columns={'n_eligible_cells':'n_cells',
                                                       'n_contributing_donors':'n_donors'}))
            meta['displayed']=True
            meta['reason']='CACHED_OK'
        print(f'[CACHE] Using frozen posterior summaries: {csv_path}', flush=True)
        return d, meta, csv_path
    if args.source_csv:
        raise FileNotFoundError('Explicit --source-csv does not exist. Check path, e.g. '
                                'FIGS6_G_ALL_TISSUES_CIRCULAR_V4/FIGS6_G_all_tissues_phase_source_data.csv')
    return None


def tissue_order(d, args):
    """All tissues, even those without all five lineages."""
    observed = sorted(set(d['tissue'].astype(str)))
    manually = [x.strip() for x in args.tissue_order.split(',') if x.strip()]
    if not manually:
        return observed
    return [x for x in manually if x in observed] + [x for x in observed if x not in manually]


def paper_tissue_label(s):
    # Shorten DISPLAY labels only; source CSV keeps the original tissue identifiers.
    short={
        'adipose tissue':'Adipose',
        'bladder organ':'Bladder',
        'bone marrow':'Bone marrow',
        'endocrine gland':'Endocrine',
        'exocrine gland':'Exocrine',
        'large intestine':'Large intestine',
        'lymph node':'Lymph node',
        'musculature':'Muscle',
        'peripheral nervous system':'PNS',
        'prostate gland':'Prostate',
        'respiratory system':'Respiratory',
        'salivary gland':'Salivary',
        'sensory system':'Sensory',
        'skin of body':'Skin',
        'small intestine':'Small intestine',
    }
    return short.get(str(s).lower(),str(s).capitalize())




def summarize_inferred_times(data, meta, tissues, args):
    """One descriptive representative B0 hour for each tissue×lineage.

    circular-mean: atan2(sum p sinθ, sum p cosθ), with resultant R.
    modal-bin: time center of largest 2-h posterior bin; R still reported.

    Both are derived from actual B0 inferred posterior, NOT sampling ZT.
    """
    prob,_=prepare_arrays(data,meta,tissues)
    rows=[]
    angle=2*np.pi*PHASE_CENTERS/24.
    exp=np.exp(1j*angle)
    metaidx={(str(z.tissue),str(z.lineage)):z for z in meta.itertuples(index=False)}
    for t in tissues:
        for lin in LINEAGES:
            a=prob.get((t,lin))
            if a is None or not np.isfinite(a).all():
                rows.append(dict(tissue=t,lineage=lin,assay=args.assay,
                                 summary_method=args.time_summary,
                                 inferred_B0_hour=np.nan,representative_B0_time='',
                                 circular_R=np.nan,modal_bin_h=np.nan,
                                 diffuse=True,n_eligible_cells=0,n_contributing_donors=0,
                                 status='NO_DATA'))
                continue
            a=np.array(a,float); a=np.maximum(0,a)
            if a.sum()<=0:raise ValueError(f'Zero posterior weight for {t} x {lin}')
            a/=a.sum()
            z=np.sum(a*exp)
            circ_hour=(np.angle(z)%(2*np.pi))*24./(2*np.pi)
            resultant=float(np.abs(z))
            modal_bin=float(PHASE_CENTERS[int(np.argmax(a))])
            hour=float(circ_hour if args.time_summary=='circular-mean' else modal_bin)
            midx=metaidx.get((t,lin))
            nc=int(getattr(midx,'n_cells',getattr(midx,'n_eligible_cells',0))) if midx is not None else 0
            nd=int(getattr(midx,'n_donors',getattr(midx,'n_contributing_donors',0))) if midx is not None else 0
            hr=int(hour)%24; minute=int(round((hour%1)*60))
            if minute==60:hr=(hr+1)%24;minute=0
            rows.append(dict(tissue=t,lineage=lin,assay=args.assay,
                             summary_method=args.time_summary,
                             inferred_B0_hour=hour,
                             representative_B0_time=f'{hr:02d}:{minute:02d}',
                             circular_R=resultant,modal_bin_h=modal_bin,
                             diffuse=bool(resultant<args.diffuse_threshold),
                             n_eligible_cells=nc,n_contributing_donors=nd,
                             status='OK'))
    df=pd.DataFrame(rows)
    if int((df.status=='OK').sum())==0:
        raise RuntimeError('No supported tissue-lineage inference to show.')
    return df


def draw_clock(ax, lineage, lin_color, tissues, lookup, args, with_text):
    """Ring outside = clock-hue for inferred time.
    Marker RADIAL position = inferred hour; radial guide lines encode 0,6,12,18,24h.
    Marker is open if the cohort-level inferred distribution has low resultant R.
    """
    n=len(tissues)
    gap=np.deg2rad(max(.35,min(.95,15/n)))
    width=(2*np.pi-n*gap)/n
    # Radius position as hours increases: inner 00, outer 24.
    r_inner,r_outer=.50,1.06
    band_bottom=1.11
    band_height=.14
    label_radius=1.37 if with_text else 1.27
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0,1.70 if with_text else 1.30)
    ax.set_xticks([]);ax.set_yticks([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('white')
    theta=np.linspace(0,2*np.pi,700)
    ax.fill(theta,np.full_like(theta,r_inner-.025),color='#FAFBFC',zorder=1)
    ax.plot(theta,np.full_like(theta,r_inner-.025),lw=.8,color='#CFD8E3',zorder=2)
    ax.plot(theta,np.full_like(theta,band_bottom+band_height+.018),lw=.6,color='#D5DCE6',zorder=2)
    for hh in [0,6,12,18,24]:
        rr=r_inner+(r_outer-r_inner)*hh/24
        ax.plot(theta,np.full_like(theta,rr),lw=.48,
                color='#D7DEE8' if hh not in (0,24) else '#B9C6D5',zorder=1,alpha=.9)
    n_valid=0;n_diffuse=0
    for i,tissue in enumerate(tissues):
        start=i*(width+gap)
        mid=start+width/2
        cell=lookup.get((tissue,lineage))
        has=cell is not None and cell.status=='OK'
        hour=float(cell.inferred_B0_hour) if has else np.nan
        diffuse=bool(cell.diffuse) if has else False
        # Inner backbone sector: lineage tint, without probability heatmap.
        ax.bar(mid,r_outer-r_inner,width=width*.96,bottom=r_inner,
               align='center',color=to_rgba(lin_color,.075) if has else '#EEF1F5',
               edgecolor='none',zorder=2)
        if has:
            n_valid+=1
            if diffuse:n_diffuse+=1
            # Outer annulus encodes inferred local B0 time (cyclic hue), NOT probability.
            phase_color=TIME_CMAP(hour/24)
            ax.bar(mid,band_height,width=width*.96,bottom=band_bottom,
                   align='center',color=phase_color,edgecolor='none',zorder=4)
            # Thin radial spine and salient clock-position marker.
            rr=r_inner+(r_outer-r_inner)*(hour/24.)
            ax.plot([mid,mid],[r_inner+.013,rr],color=lin_color,lw=2.2,
                    alpha=.70,solid_capstyle='round',zorder=5)
            ax.scatter([mid],[rr],s=57,
                       facecolor='white' if diffuse else lin_color,
                       edgecolor=lin_color if diffuse else 'white',
                       linewidth=1.5 if diffuse else .9,zorder=10)
        else:
            ax.bar(mid,band_height,width=width*.96,bottom=band_bottom,
                   align='center',color='#E6EAF0',edgecolor='none',zorder=4)
        # Small tissue edge marker so sectors stay legible.
        ax.plot([start,start],[r_inner-.006,band_bottom+band_height+.008],
                color='white',lw=1.35,zorder=6)
        if with_text and args.labels=='all':
            deg=np.rad2deg(mid)%360
            rot=90-deg
            if 90<deg<270:
                rot+=180; ha='right'
            else:
                ha='left'
            ax.text(mid,label_radius,paper_tissue_label(tissue),color=NAVY,
                    fontsize=5.7 if n>=28 else 7.,rotation=rot,rotation_mode='anchor',
                    ha=ha,va='center',zorder=15)
    if with_text:
        ax.text(.5,.55,{'Macrophage/Monocyte':'Macrophage /\nMonocyte'}.get(lineage,lineage),
                color=lin_color,fontsize=11,fontweight='bold',ha='center',va='center',
                transform=ax.transAxes,zorder=50)
        ax.text(.5,.47,f'{n_valid}/{n} tissues',fontsize=7.7,color=MUTED,
                transform=ax.transAxes,ha='center',va='center',zorder=50)
        # Display radial scale on just the first circle, to avoid crowding 5.
    return n_valid,n_diffuse


def create_figure(phases, order, outdir, args, with_text):
    args_global_dpi[0]=args.dpi
    rows={(str(z.tissue),str(z.lineage)):z for z in phases.itertuples(index=False)}
    if args.layout=='row':
        fig=plt.figure(figsize=(28,7.5),facecolor='white')
        positions=[(.013+.196*k,.12,.187,.76) for k in range(5)]
    else:
        fig=plt.figure(figsize=(18.6,14.7),facecolor='white')
        positions=[(.023+.324*k,.52,.310,.40) for k in range(3)]
        positions +=[(.19+.325*k,.06,.310,.40) for k in range(2)]
    for k,(lin,color,pos) in enumerate(zip(LINEAGES,COLORS,positions)):
        ax=fig.add_axes(pos,projection='polar')
        draw_clock(ax,lin,color,order,rows,args,with_text)
    if with_text:
        fig.text(.033,.979,'S6G',fontsize=17,color=NAVY,fontweight='bold',va='top')
        fig.text(.106,.979,'Representative inferred B0 clock time by tissue and lineage',
                 fontsize=14,color=NAVY,va='top')
        fig.text(.107,.947,f'{args.assay} · {len(order)} tissue sectors · {args.time_summary} · frozen Tabula B0',
                 fontsize=9,color=MUTED,va='top')
        fig.text(.047,.455 if args.layout=='grid' else .10,
                 'Each sector = one tissue. Outer sector hue + marker location = representative inferred B0 time.\n'
                 'Solid dot: concentrated estimate. Open dot: diffuse distribution; time is less interpretable.',
                 fontsize=9.1,color=NAVY,va='top')
        fig.text(.047,.024 if args.layout=='grid' else .052,
                 'Hours are COHORT-SPECIFIC inferred B0, not observed ZT. Independently fitted tissue phases are not globally aligned.',
                 fontsize=8.5,color=MUTED,va='bottom')
        # Genuine 24-h phase color gradient (wraps at 24h): no posterior probability legend.
        gradient_ax=fig.add_axes([.70,.472,.205,.014] if args.layout=='grid' else [.76,.88,.175,.02])
        gradient=np.linspace(0,1,512)[None,:]
        gradient_ax.imshow(gradient,aspect='auto',cmap=TIME_CMAP_NAME,origin='lower',extent=(0,24,0,1))
        gradient_ax.set_xlim(0,24)
        gradient_ax.set_yticks([])
        gradient_ax.set_xticks([0,6,12,18,24]);gradient_ax.tick_params(labelsize=8,length=2)
        for sp in gradient_ax.spines.values():sp.set_visible(False)
        gy=.490 if args.layout=='grid' else .925
        fig.text(.70 if args.layout=='grid' else .76,gy,'Inferred B0 time (h)',
                 fontsize=9.1,color=NAVY,fontweight='bold')
        # One compact radial time guide, demonstrating the dot r position.
        xx,yy=(.80,.33) if args.layout=='grid' else (.78,.20)
        guide=fig.add_axes([xx,yy,.085,.065])
        guide.set_xlim(0,1);guide.set_ylim(0,1);guide.axis('off')
        guide.plot([.5,.5],[.13,.90],color='#A6B7C9',lw=2)
        for j,hh in enumerate([0,6,12,18,24]):
            ry=.13+.77*j/4
            guide.plot([.45,.55],[ry,ry],color='#8296AB',lw=1.1)
            guide.text(.65,ry,f'{hh:02d} h',va='center',ha='left',fontsize=7.3,color=MUTED)
        fig.text(xx,yy+.068,'Dot radius',fontsize=8,color=NAVY)
    tag='text' if with_text else 'notext'
    stem=Path(outdir)/f'FIGS6_G_five_lineage_predicted_B0_times_{tag}'
    save_fig(fig,stem)


args_global_dpi=[280]

def save_fig(fig,stem):
    for ext in ['png','svg','pdf']:
        opts={'bbox_inches':'tight','pad_inches':.14,'facecolor':'white'}
        if ext=='png':opts['dpi']=args_global_dpi[0]
        fig.savefig(str(stem)+'.'+ext,**opts)
    plt.close(fig)


def main():
    pa=argparse.ArgumentParser(description='Figure S6G: 5 lineage clocks, all tissues, ONE inferred B0 time per supported tissue.')
    pa.add_argument('--data-root',default='.',help='katv2 root directory')
    pa.add_argument('--outdir',default='FIGS6_G_PREDICTED_B0_TIMES_V3')
    pa.add_argument('--assay',default="10x 3' v3")
    pa.add_argument('--source-csv',default='',help='V4 posterior CSV, optional. Prefer cached local output.')
    pa.add_argument('--force-reload',action='store_true',help='Read frozen manifest and original phase/posterior/H5AD instead of V4 cache')
    pa.add_argument('--min-r',type=float,default=0.)
    pa.add_argument('--min-cells',type=int,default=20)
    pa.add_argument('--min-donors',type=int,default=1)
    pa.add_argument('--skip-missing',action='store_true')
    pa.add_argument('--cache-only',action='store_true',help=argparse.SUPPRESS)
    pa.add_argument('--max-tissues',type=int,default=0)
    pa.add_argument('--tissue-order',default='')
    pa.add_argument('--labels',choices=['all','none'],default='all')
    pa.add_argument('--layout',choices=['grid','row'],default='grid')
    pa.add_argument('--time-summary',choices=['circular-mean','modal-bin'],default='circular-mean',
                    help='Representative inferred time: circular mean from pooled posterior or most populated 2h bin.')
    pa.add_argument('--diffuse-threshold',type=float,default=.15,
                    help='Mark R<threshold as open circles; no unique strong phase mode. Default 0.15.')
    pa.add_argument('--dpi',type=int,default=250)
    args=pa.parse_args()
    if args.diffuse_threshold<0 or args.diffuse_threshold>1:raise ValueError('--diffuse-threshold must be in [0,1]')
    root=Path(args.data_root).expanduser().resolve()
    out=Path(args.outdir).expanduser().resolve();out.mkdir(exist_ok=True,parents=True)
    cache=None if args.force_reload else data_from_v4_cache(root,args)
    if cache is not None:
        data,meta,origin=cache
        selected=sorted(data.tissue.astype(str).unique())
        origin=str(origin)
    else:
        mf,manifest_path=manifest(root)
        cohort=mf[mf.assay.astype(str).eq(args.assay)].sort_values('tissue_general')
        if cohort.empty:raise RuntimeError(f'No COMPLETE B0 cohorts for {args.assay}')
        if cohort.tissue_general.duplicated().any():raise ValueError('Duplicate tissue runs; no automatic pooling of independent B0 fits')
        if args.max_tissues>0:cohort=cohort.head(args.max_tissues)
        data,meta,_,_=load_all_tissues(root,cohort,args)
        selected=sorted(cohort.tissue_general.astype(str).unique())
        origin=str(manifest_path)
    if args.max_tissues>0:
        selected=selected[:args.max_tissues]
        data=data[data.tissue.isin(selected)]
        meta=meta[meta.tissue.isin(selected)]
    requested=[s.strip() for s in args.tissue_order.split(',') if s.strip()]
    order=[x for x in requested if x in selected]+[x for x in selected if x not in requested]
    df=summarize_inferred_times(data,meta,order,args)
    df.to_csv(out/'FIGS6_G_tissue_lineage_inferred_B0_time_estimates.csv',index=False)
    data.to_csv(out/'FIGS6_G_source_posterior_fraction_12bins.csv',index=False)
    meta.to_csv(out/'FIGS6_G_lineage_support.csv',index=False)
    pd.DataFrame({'tissue':order,'angular_sector_order':range(len(order))}).to_csv(out/'FIGS6_G_sector_order.csv',index=False)
    create_figure(df,order,out,args,True)
    create_figure(df,order,out,args,False)
    count=(df.status=='OK').sum();dif=((df.status=='OK') & df.diffuse).sum()
    (out/'README_FIGS6G_v3.txt').write_text(
        f'Source: {origin}\nAssay: {args.assay}\nTissue sectors: {len(order)}\n'
        f'Method for representative B0 phase: {args.time_summary}\n'
        f'Valid tissue-lineage combos: {count}; diffuse by R<{args.diffuse_threshold}: {dif}.\n'
        'The location of each marker and the hue of the outer sector encode the REPRESENTATIVE inferred B0 hour, NOT a posterior probability.\n'
        'Circular mean of a multimodal/diffuse distribution may be misleading. Open markers indicate low resultant R.\n'
        'B0 separately fit per tissue x assay; cross-tissue absolute hours are not automatically aligned and are not observed ZT.\n',encoding='utf-8')
    print(f'DONE: {len(order)} tissues, {count} supported lineage*tissue points; {dif} diffuse.')
    print('Output:',out)

if __name__=='__main__':
    main()
