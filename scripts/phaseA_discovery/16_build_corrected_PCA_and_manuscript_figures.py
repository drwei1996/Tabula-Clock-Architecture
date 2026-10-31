#!/usr/bin/env python
from __future__ import annotations
import os

"""
Build the manuscript-ready Tabula figure set from the completed publication suite.

Goals
-----
1) Add a NEW corrected PCA from the frozen standardized static clock-architecture input.
2) Reorganize all available outputs into:
      Figure 1
      Figure 2
      Supplementary
3) Create composite manuscript preview figures while preserving panel-level PDF/SVG/PNG.
4) Do NOT modify formal statistics, ontology, redundancy families, or frozen results.

Static-expression language only. No phase/rhythm/amplitude/oscillation claims.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"
SUITE = ROOT / "12_COMPLETE_PUBLICATION_FIGURE_SUITE_20260920"

CURRENT_MAIN = SUITE / "01_current_corrected_main"
CURRENT_SUPP = SUITE / "02_current_corrected_supplementary"
LEGACY = SUITE / "03_legacy_revised_architecture"
HIST_PHASEA = SUITE / "04_historical_phaseA_old_mapping"
HIST_A5 = SUITE / "05_historical_A5_old_mapping"

SOURCE = SUITE / "source_data"
AUDIT = SUITE / "audit"

OUT = ROOT / "13_MANUSCRIPT_FIGURES_20260920"
FIG1 = OUT / "Figure_1"
FIG2 = OUT / "Figure_2"
SUPP = OUT / "Supplementary"
PANEL = OUT / "panel_files"
SRCOUT = OUT / "source_data"
AUDOUT = OUT / "audit"

for d in [FIG1, FIG2, SUPP, PANEL, SRCOUT, AUDOUT]:
    d.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

COL = {
    "blue": "#4E79A7",
    "red": "#E15759",
    "green": "#59A14F",
    "orange": "#F28E2B",
    "purple": "#9467BD",
    "gray": "#9EA4AA",
    "lightgray": "#E8EAED",
    "dark": "#30343B",
}

LINEAGE_ORDER = ["b_cell","t_cell","macrophage_monocyte","fibroblast","endothelial"]
LINEAGE_DISPLAY = {
    "b_cell":"B cell",
    "t_cell":"T cell",
    "macrophage_monocyte":"Macrophage/monocyte",
    "fibroblast":"Fibroblast",
    "endothelial":"Endothelial",
    "other":"Other",
}
LINEAGE_COLORS = {
    "b_cell":"#4E79A7",
    "t_cell":"#E15759",
    "macrophage_monocyte":"#59A14F",
    "fibroblast":"#F28E2B",
    "endothelial":"#9467BD",
    "other":"#B9B9B9",
}


# =============================================================================
# HELPERS
# =============================================================================

def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save(fig, pathstem: Path):
    fig.savefig(pathstem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(pathstem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(pathstem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def read_csv_auto(path: Path):
    if not path.exists():
        return None
    sep = "\t" if path.suffix.lower() in [".tsv",".txt"] else ","
    return pd.read_csv(path, sep=sep, low_memory=False)

def first_hit(patterns, roots):
    hits=[]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            hits.extend([p for p in root.rglob(pattern) if SUITE not in p.parents and OUT not in p.parents])
    if not hits:
        return None
    # prefer shorter deterministic paths
    hits = sorted(set(hits), key=lambda p:(len(p.parts), str(p).lower()))
    return hits[0]

def copy_panel(stem, source_dir, dest_dir, new_name=None):
    new_name = new_name or stem
    copied=[]
    for ext in [".pdf",".svg",".png"]:
        src=source_dir/f"{stem}{ext}"
        if src.exists():
            dst=dest_dir/f"{new_name}{ext}"
            shutil.copy2(src,dst)
            copied.append(str(dst))
    # source CSV if present
    src_csv=SOURCE/f"{stem}.csv"
    if src_csv.exists():
        shutil.copy2(src_csv,SRCOUT/f"{new_name}.csv")
    return copied

def add_panel_label(ax, label):
    ax.text(-0.06,1.04,label,transform=ax.transAxes,
            ha="right",va="bottom",fontsize=11,weight="bold")


# =============================================================================
# 1. NEW CORRECTED PCA
# =============================================================================

std_path = first_hit(
    ["clock_architecture_UMAP_clock_architecture_standardized_input.tsv"],
    [PROJECT, ROOT]
)
meta_path = ROOT / "00_freeze" / "observation_metadata_corrected_v3_FROZEN.tsv"

if std_path is None:
    raise FileNotFoundError("Frozen standardized clock-architecture input was not found.")
if not meta_path.exists():
    raise FileNotFoundError(meta_path)

std = pd.read_csv(std_path, sep="\t")
meta = pd.read_csv(meta_path, sep="\t")

if "group_id" not in std.columns or "group_id" not in meta.columns:
    raise ValueError("group_id is required in both standardized input and corrected metadata.")

feature_cols=[c for c in std.columns if c!="group_id"]
X=std[feature_cols].to_numpy(dtype=float)
if not np.isfinite(X).all():
    raise ValueError("Standardized PCA matrix contains non-finite values.")

pca=PCA(n_components=2)
coords=pca.fit_transform(X)

pc = pd.DataFrame({
    "group_id": std["group_id"].astype(str),
    "PC1": coords[:,0],
    "PC2": coords[:,1],
})
pc = pc.merge(
    meta[["group_id","priority_lineage_corrected","tissue","donor_id","cell_type"]],
    on="group_id",
    how="left",
    validate="one_to_one"
)
pc["priority_lineage_corrected"]=pc["priority_lineage_corrected"].fillna("other").astype(str)

pc.to_csv(SRCOUT/"C08_PCA_frozen_clock_architecture_corrected_source.csv",index=False)

pca_info = {
    "input": str(std_path),
    "n_observations": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    "note": "PCA newly computed from the frozen standardized static clock-architecture input; corrected lineage used only as metadata coloring."
}
(AUDOUT/"C08_PCA_metadata.json").write_text(json.dumps(pca_info,indent=2),encoding="utf-8")

fig=plt.figure(figsize=(5.1,4.1))
ax=fig.add_axes([.13,.15,.64,.73])
lax=fig.add_axes([.80,.12,.18,.78]);lax.axis("off")
for lin in ["other"]+LINEAGE_ORDER:
    d=pc[pc["priority_lineage_corrected"].eq(lin)]
    if d.empty:
        continue
    ax.scatter(d.PC1,d.PC2,s=10,
               color=LINEAGE_COLORS.get(lin,COL["gray"]),
               alpha=.30 if lin=="other" else .78,
               edgecolors="none",rasterized=True)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Static clock-architecture PCA")
clean_axis(ax)
handles=[
    Line2D([],[],marker="o",ls="",markersize=4.5,
           color=LINEAGE_COLORS[x],label=LINEAGE_DISPLAY[x])
    for x in ["other"]+LINEAGE_ORDER if (pc["priority_lineage_corrected"]==x).any()
]
lax.legend(handles=handles,title="Corrected major lineage",
           frameon=False,loc="center left",fontsize=6.5,title_fontsize=7)
fig.text(.5,.035,
         "Frozen standardized clock-architecture features; corrected lineage used for display only.",
         ha="center",fontsize=6.5,color="#555555")
save(fig, CURRENT_MAIN/"C08_PCA_frozen_clock_architecture_corrected")


# =============================================================================
# 2. MANUSCRIPT PANEL ASSIGNMENT
# =============================================================================

# Figure 1 — static architecture + remodeling
FIG1_PANELS = [
    ("A","C08_PCA_frozen_clock_architecture_corrected",CURRENT_MAIN,
     "Corrected PCA of frozen static clock-architecture features"),
    ("B","C01_A1_corrected_lineage_information",CURRENT_MAIN,
     "Clock-associated lineage information relative to matched non-clock genes"),
    ("C","C02_A2_corrected_tissue_remodeling",CURRENT_MAIN,
     "Within-lineage tissue remodeling versus between-lineage separation"),
]

# Figure 2 — pathway correspondence synthesis
FIG2_PANELS = [
    ("A","C04_module_cellunit_landscape",CURRENT_MAIN,
     "Biological-module landscape across formal cell units"),
    ("B","C05_Bcell_module_landscape",CURRENT_MAIN,
     "B-cell-focused module landscape"),
    ("C","C06_Hallmark_Reactome_convergence",CURRENT_MAIN,
     "Hallmark–Reactome descriptive convergence"),
    ("D","C07_module_composition",CURRENT_MAIN,
     "Final coherent redundancy-family composition"),
]

# Current corrected supplementary priorities
SUPP_CURRENT = [
    ("S1","C00_lineage_mapping_correction",CURRENT_SUPP,
     "Lineage-mapping correction audit"),
    ("S2","C03_A47_formal_collection_summary",CURRENT_MAIN,
     "Formal Hallmark/Reactome hit-count overview"),
    ("S3","S_C01_full_69_family_heatmap",CURRENT_SUPP,
     "All 69 coherent redundancy families"),
    ("S4","S_C02_beta_vs_specificity",CURRENT_SUPP,
     "Formal-hit effect size versus specificity"),
    ("S5","S_C03_Hallmark_Reactome_overlap",CURRENT_SUPP,
     "Cross-database gene-set overlap"),
    ("S6","S_C04_v1_vs_v2_cluster_size",CURRENT_SUPP,
     "Connected-component versus complete-linkage redundancy structure"),
    ("S7","S_C05_remaining_other_unclassified",CURRENT_SUPP,
     "Residual Other/Unclassified pathways"),
]

# Revised coordinate displays. These are valid display layers because coordinates are frozen,
# while corrected lineage recoloring does not change the underlying embedding.
REVISED_DISPLAY = []
for p in sorted(LEGACY.glob("L10_UMAP_*_by_corrected_lineage.png")):
    REVISED_DISPLAY.append((f"S{8+len(REVISED_DISPLAY)}",p.stem,LEGACY,
                            "Frozen UMAP coordinates recolored with corrected major-lineage metadata"))

# Other revised embedding views — tissue/donor/original cell-type — supplement only.
for pattern,desc in [
    ("L11_UMAP_*_by_tissue.png","Frozen UMAP coordinates colored by tissue context"),
    ("L11_UMAP_*_by_donor.png","Frozen UMAP coordinates colored by donor"),
    ("L12_UMAP_*_by_original_cell_type.png","Frozen UMAP coordinates colored by original cell-type annotation"),
]:
    for p in sorted(LEGACY.glob(pattern)):
        REVISED_DISPLAY.append((f"S{8+len(REVISED_DISPLAY)}",p.stem,LEGACY,desc))

# Historical figures are retained as audit/sensitivity supplements, not current corrected inference.
HISTORICAL = []
for source_dir, prefix, desc in [
    (HIST_PHASEA,"HPA","Historical Phase-A figure; old lineage mapping; audit/sensitivity only"),
    (HIST_A5,"HA5","Historical A5 variance-architecture figure; old lineage mapping; audit/sensitivity only"),
]:
    for p in sorted(source_dir.glob("*.png")):
        HISTORICAL.append((prefix,p.stem,source_dir,desc))


# =============================================================================
# 3. COPY PANEL FILES INTO ORGANIZED DIRECTORIES
# =============================================================================

assignment_rows=[]

for label,stem,srcdir,desc in FIG1_PANELS:
    new=f"Figure1{label}_{stem}"
    copied=copy_panel(stem,srcdir,FIG1,new)
    assignment_rows.append({
        "destination":"Figure 1","panel":label,"source_stem":stem,
        "new_stem":new,"status":"current corrected","description":desc,
        "files_copied":len(copied)
    })

for label,stem,srcdir,desc in FIG2_PANELS:
    new=f"Figure2{label}_{stem}"
    copied=copy_panel(stem,srcdir,FIG2,new)
    assignment_rows.append({
        "destination":"Figure 2","panel":label,"source_stem":stem,
        "new_stem":new,"status":"current corrected","description":desc,
        "files_copied":len(copied)
    })

for label,stem,srcdir,desc in SUPP_CURRENT + REVISED_DISPLAY:
    new=f"{label}_{stem}"
    copied=copy_panel(stem,srcdir,SUPP,new)
    assignment_rows.append({
        "destination":"Supplementary","panel":label,"source_stem":stem,
        "new_stem":new,
        "status":"current corrected" if not stem.startswith("L") else "frozen coordinates / corrected display",
        "description":desc,"files_copied":len(copied)
    })

hist_counter=1
for prefix,stem,srcdir,desc in HISTORICAL:
    label=f"S_HIST_{hist_counter:02d}"
    new=f"{label}_{stem}"
    copied=copy_panel(stem,srcdir,SUPP,new)
    assignment_rows.append({
        "destination":"Supplementary historical audit","panel":label,"source_stem":stem,
        "new_stem":new,"status":"historical old mapping",
        "description":desc,"files_copied":len(copied)
    })
    hist_counter+=1

assign=pd.DataFrame(assignment_rows)
assign.to_csv(OUT/"MANUSCRIPT_FIGURE_ASSIGNMENT.csv",index=False)


# =============================================================================
# 4. COMPOSITE PREVIEW FIGURES
# =============================================================================

def load_png(stem, srcdir):
    p=srcdir/f"{stem}.png"
    if not p.exists():
        raise FileNotFoundError(p)
    return plt.imread(p)

# Figure 1 composite: A full width, B/C below
fig=plt.figure(figsize=(11.0,8.0))
gs=fig.add_gridspec(2,2,height_ratios=[1.05,1.0],hspace=.16,wspace=.10,
                    left=.035,right=.985,bottom=.045,top=.96)
specs=[
    ("A",FIG1_PANELS[0],gs[0,:]),
    ("B",FIG1_PANELS[1],gs[1,0]),
    ("C",FIG1_PANELS[2],gs[1,1]),
]
for letter,(_,stem,srcdir,_),slot in specs:
    ax=fig.add_subplot(slot)
    ax.imshow(load_png(stem,srcdir))
    ax.axis("off")
    ax.text(-.01,1.01,letter,transform=ax.transAxes,fontsize=14,weight="bold",
            ha="right",va="bottom")
fig.suptitle("Figure 1 | Static clock-associated architecture and tissue remodeling",
             fontsize=13,y=.995)
save(fig,FIG1/"Figure1_COMPOSITE_PREVIEW")

# Figure 2 composite: 2 x 2
fig=plt.figure(figsize=(12.0,9.3))
gs=fig.add_gridspec(2,2,hspace=.10,wspace=.08,left=.025,right=.985,bottom=.035,top=.965)
for (letter,stem,srcdir,_),slot in zip(FIG2_PANELS,[gs[0,0],gs[0,1],gs[1,0],gs[1,1]]):
    ax=fig.add_subplot(slot)
    ax.imshow(load_png(stem,srcdir))
    ax.axis("off")
    ax.text(-.01,1.01,letter,transform=ax.transAxes,fontsize=14,weight="bold",
            ha="right",va="bottom")
fig.suptitle("Figure 2 | Pathway correspondence and biological-module synthesis",
             fontsize=13,y=.995)
save(fig,FIG2/"Figure2_COMPOSITE_PREVIEW")


# =============================================================================
# 5. SUPPLEMENTARY INDEX + LEGENDS
# =============================================================================

def md_table(df):
    cols=df.columns.tolist()
    lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    for row in df.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(str(x).replace("|","/") for x in row)+" |")
    return "\n".join(lines)

current_supp=assign[assign["destination"].eq("Supplementary")][
    ["panel","new_stem","status","description"]
].copy()
historical_supp=assign[assign["destination"].eq("Supplementary historical audit")][
    ["panel","new_stem","status","description"]
].copy()

readme=f"""# Manuscript figure organization

## Figure 1
**Static clock-associated architecture and tissue remodeling**

- A. Corrected PCA from the frozen standardized static clock-architecture feature matrix.
- B. Corrected A1 matched-null lineage-information comparison.
- C. Corrected A2 tissue-associated remodeling / between-lineage separation.

## Figure 2
**Pathway correspondence and biological-module synthesis**

- A. Module × formal cell-unit landscape.
- B. B-cell-focused module landscape.
- C. Hallmark–Reactome descriptive convergence.
- D. Final coherent redundancy-family composition.

## Supplementary
Current corrected / frozen-coordinate display figures:

{md_table(current_supp)}

## Historical audit/sensitivity
These figures are retained because they are useful for audit, robustness, or methodological context,
but they predate the corrected priority-lineage mapping. They are **not** promoted to current corrected inference.

{md_table(historical_supp)}

## Interpretation guardrails
- Tabula Sapiens is static expression; do not describe phase, rhythm, amplitude, oscillation, or phase shift.
- Broad/fine levels are hierarchical resolutions of the same atlas, not independent replication.
- Hallmark–Reactome overlap is descriptive convergence, not independent experimental replication.
- M99 Other/Unclassified has convergence = NA.
"""
(OUT/"README_MANUSCRIPT_FIGURES.md").write_text(readme,encoding="utf-8")

print("="*120)
print("MANUSCRIPT FIGURE REORGANIZATION COMPLETE")
print("="*120)
print(f"OUTPUT ROOT:       {OUT}")
print(f"NEW CORRECTED PCA: {CURRENT_MAIN/'C08_PCA_frozen_clock_architecture_corrected.pdf'}")
print(f"FIGURE 1:          {FIG1/'Figure1_COMPOSITE_PREVIEW.pdf'}")
print(f"FIGURE 2:          {FIG2/'Figure2_COMPOSITE_PREVIEW.pdf'}")
print(f"SUPPLEMENTARY DIR: {SUPP}")
print(f"ASSIGNMENT TABLE:  {OUT/'MANUSCRIPT_FIGURE_ASSIGNMENT.csv'}")
print("="*120)
