#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import os

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

RAW_A47 = (
    ROOT
    / "06_A4_7_within_donor_identity_pathways"
    / "01_broad_lineage"
    / "all_broad_pairwise_pathway_screens.csv"
)

V3_FREEZE = (
    ROOT
    / "00_freeze"
    / "A4_8_V3_FINAL_ONTOLOGY_20260920"
)

STRICT_HITS = V3_FREEZE / "04_FORMAL_HITS_WITH_V3_ONTOLOGY.csv"
STRICT_ONTOLOGY = V3_FREEZE / "01_V3_PATHWAY_ONTOLOGY.csv"

OUT = ROOT / "47_FIG3_EFG_TOP_PATHWAYS_20260921"
FIG = OUT / "figures"
SRC = OUT / "source_data"
AUD = OUT / "audit"

TOP_N_PER_LINEAGE = 10
TOP_N_SUPP = 8

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

LINEAGES = [
    "b_cell",
    "t_cell",
    "macrophage_monocyte",
    "fibroblast",
    "endothelial",
]

LINEAGE_LABEL = {
    "b_cell": "B cell",
    "t_cell": "T cell",
    "macrophage_monocyte": "Macrophage/Monocyte",
    "fibroblast": "Fibroblast",
    "endothelial": "Endothelial",
}

LINEAGE_COLORS = {
    "b_cell": "#4E79A7",
    "t_cell": "#E15759",
    "macrophage_monocyte": "#59A14F",
    "fibroblast": "#F28E2B",
    "endothelial": "#9467BD",
}

MODULE_LABELS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE":
        "Cytokine/TNF/NF-κB/innate immune",
    "M02_MAPK_RTK_GROWTH_FACTOR":
        "MAPK/RTK/growth factor",
    "M03_TGFB_SMAD":
        "TGF-β/SMAD",
    "M04_CELL_DEATH_APOPTOSIS":
        "Cell death/apoptosis",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING":
        "Rho/cytoskeleton/adhesion/remodeling",
    "M06_TRAFFICKING_ER_PROTEOSTASIS":
        "Trafficking/ER/proteostasis",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR":
        "Antigen/interferon/immune effector",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION":
        "Transcription/chromatin/differentiation",
    "M09_METABOLIC_OXIDATIVE_STRESS":
        "Metabolic/oxidative stress",
    "M10_CELL_CYCLE_GROWTH":
        "Cell cycle/growth",
    "M11_OTHER_SIGNALING_SPECIALIZED":
        "Other specialized signaling",
    "M99_OTHER_UNCLASSIFIED":
        "Other/unclassified",
}

MODULE_PRIORITY = list(MODULE_LABELS.keys())

DISPLAY_MODULE_ORDER = [
    "M02_MAPK_RTK_GROWTH_FACTOR",
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING",
    "M09_METABOLIC_OXIDATIVE_STRESS",
    "M11_OTHER_SIGNALING_SPECIALIZED",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION",
    "M10_CELL_CYCLE_GROWTH",
    "M04_CELL_DEATH_APOPTOSIS",
    "M03_TGFB_SMAD",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR",
    "M99_OTHER_UNCLASSIFIED",
]

MODULE_COLORS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE": "#59A89D",
    "M02_MAPK_RTK_GROWTH_FACTOR": "#4E6A9E",
    "M03_TGFB_SMAD": "#B07AA1",
    "M04_CELL_DEATH_APOPTOSIS": "#D97562",
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING": "#76B7B2",
    "M06_TRAFFICKING_ER_PROTEOSTASIS": "#E9968A",
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR": "#91C47C",
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION": "#8A89A6",
    "M09_METABOLIC_OXIDATIVE_STRESS": "#E0BE63",
    "M10_CELL_CYCLE_GROWTH": "#EDC948",
    "M11_OTHER_SIGNALING_SPECIALIZED": "#86BCB6",
    "M99_OTHER_UNCLASSIFIED": "#9EA4AA",
}

KEYWORDS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE": [
        "TNF", "TNFR", "NFKB", "NF_KB", "NF_KAPPA_B",
        "TOLL", "TLR", "MYD88", "IRAK", "RIPK",
        "INFLAMMATORY", "CYTOKINE", "INTERLEUKIN",
        "IL1", "IL_1", "IL17", "IL_17",
        "IL2_STAT5", "IL_2_STAT5",
        "IL6_JAK_STAT3", "IL_6_JAK_STAT3",
    ],
    "M02_MAPK_RTK_GROWTH_FACTOR": [
        "MAPK", "MAPK1", "MAPK3", "RAF", "RAS",
        "RECEPTOR_TYROSINE_KINASE", "RTK",
        "NTRK", "EGFR", "ERBB", "FGFR", "MET",
        "SPRY", "GROWTH_FACTOR", "ERYTHROPOIETIN",
        "PHOSPHOLIPASE_C_GAMMA",
    ],
    "M03_TGFB_SMAD": [
        "TGFB", "TGF_BETA", "TGF_BETA_RECEPTOR", "SMAD",
    ],
    "M04_CELL_DEATH_APOPTOSIS": [
        "APOPTOSIS", "APOPTOTIC", "DEATH_RECEPTOR",
        "BH3", "PUMA", "CASPASE", "NECROSIS",
        "NECROPTOSIS", "PROAPOPTOTIC",
    ],
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING": [
        "RHO", "RAC", "ROCK", "KTN1", "CYTOSKELET",
        "INTEGRIN", "EPHRIN", "SEMAPHORIN",
        "EPITHELIAL_MESENCHYMAL_TRANSITION",
        "ADHESION",
    ],
    "M06_TRAFFICKING_ER_PROTEOSTASIS": [
        "MEMBRANE_TRAFFICKING", "VESICLE", "GOLGI",
        "ER_RETROGRADE", "RETROGRADE_TRANSPORT",
        "UBIQUITIN", "PROTEASOME", "AUTOPHAGY",
        "IRE1", "CHAPERONE", "POST_TRANSLATIONAL",
        "CLATHRIN", "UNFOLDED_PROTEIN_RESPONSE",
        "RAB_GERANYLGERANYLATION",
    ],
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR": [
        "ANTIGEN", "MHC", "INTERFERON", "DDX58",
        "IFIH1", "IMMUNE_EFFECTOR",
    ],
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION": [
        "ESTROGEN", "ANDROGEN", "HDAC", "HISTONE",
        "CHROMATIN", "BETA_CATENIN", "TCF",
        "RUNX", "TRANSCRIPTIONAL_REGULATION",
        "GENE_EXPRESSION", "DIFFERENTIATION",
    ],
    "M09_METABOLIC_OXIDATIVE_STRESS": [
        "HEME", "HMOX1", "NFE2L2",
        "OXIDATIVE_PHOSPHORYLATION",
        "CHOLESTEROL", "FATTY_ACID", "AMINO_ACID",
        "CYP", "METABOLISM", "HYPOXIA",
    ],
    "M10_CELL_CYCLE_GROWTH": [
        "MITOTIC", "CELL_CYCLE", "MTOR", "PI3K", "P53",
    ],
    "M11_OTHER_SIGNALING_SPECIALIZED": [
        "NOTCH", "HEDGEHOG", "ACTIVATION_OF_SMO",
        "G_ALPHA", "SECOND_MESSENGER", "CHANNEL",
        "NEURONAL_SYSTEM", "PLATELET_ACTIVATION",
    ],
}

MANUAL_PRIMARY_OVERRIDES = {
    "REACTOME_NUCLEAR_EVENTS_KINASE_AND_TRANSCRIPTION_FACTOR_ACTIVATION":
        "M02_MAPK_RTK_GROWTH_FACTOR",
    "REACTOME_MEMBRANE_TRAFFICKING":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_COPI_DEPENDENT_GOLGI_TO_ER_RETROGRADE_TRAFFIC":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_RUNX1_REGULATES_GENES_INVOLVED_IN_MEGAKARYOCYTE_DIFFERENTIATION_AND_PLATELET_FUNCTION":
        "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION",
    "HALLMARK_IL2_STAT5_SIGNALING":
        "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING":
        "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION":
        "M05_RHO_CYTOSKELETON_ADHESION_REMODELING",
    "REACTOME_REGULATION_OF_HMOX1_EXPRESSION_AND_ACTIVITY":
        "M09_METABOLIC_OXIDATIVE_STRESS",
    "REACTOME_REGULATION_OF_NFE2L2_GENE_EXPRESSION":
        "M09_METABOLIC_OXIDATIVE_STRESS",
    "REACTOME_ACTIVATION_OF_SMO":
        "M11_OTHER_SIGNALING_SPECIALIZED",
    "REACTOME_ERYTHROPOIETIN_ACTIVATES_PHOSPHOLIPASE_C_GAMMA_PLCG":
        "M02_MAPK_RTK_GROWTH_FACTOR",
    "REACTOME_RAB_GERANYLGERANYLATION":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_NEURONAL_SYSTEM":
        "M11_OTHER_SIGNALING_SPECIALIZED",
    "REACTOME_PLATELET_ACTIVATION_SIGNALING_AND_AGGREGATION":
        "M11_OTHER_SIGNALING_SPECIALIZED",
}

INTENTIONAL_OTHER = {
    "HALLMARK_UV_RESPONSE_DN",
    "HALLMARK_UV_RESPONSE_UP",
    "REACTOME_DENGUE_VIRUS_GENOME_TRANSLATION_AND_REPLICATION",
    "REACTOME_PROCESSING_OF_CAPPED_INTRON_CONTAINING_PRE_MRNA",
}

def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )

def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_figure(fig, stem: str):
    for ext in ["png", "pdf", "svg"]:
        kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.12,
            "facecolor": "white",
        }
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kwargs)
    plt.close(fig)

def short_pathway(x: str) -> str:
    return (
        str(x)
        .replace("HALLMARK_", "")
        .replace("REACTOME_", "")
        .replace("_", " ")
    )

def assign_module(pathway: str, frozen_lookup: dict[str, str]):
    pw = str(pathway)
    if pw in frozen_lookup:
        return frozen_lookup[pw], "frozen_v3"

    up = pw.upper()

    if up in MANUAL_PRIMARY_OVERRIDES:
        return MANUAL_PRIMARY_OVERRIDES[up], "v3_manual_rule_extended"

    if up in INTENTIONAL_OTHER:
        return "M99_OTHER_UNCLASSIFIED", "v3_intentional_other"

    matches = []
    for module in MODULE_PRIORITY:
        if module == "M99_OTHER_UNCLASSIFIED":
            continue
        if any(k in up for k in KEYWORDS.get(module, [])):
            matches.append(module)

    if matches:
        return matches[0], "v3_keyword_rule_extended"

    return "M99_OTHER_UNCLASSIFIED", "no_rule_match"

for p in [RAW_A47, STRICT_HITS, STRICT_ONTOLOGY]:
    if not p.exists():
        raise FileNotFoundError(p)

for d in [OUT, FIG, SRC, AUD]:
    d.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(RAW_A47, low_memory=False)

required = {
    "level",
    "lineage",
    "collection",
    "pathway",
    "pathway_testable",
    "beta_standardized_pairwise_correspondence",
    "pairwise_correspondence_FDR_collection",
    "supported_positive_pairwise_correspondence",
}

missing = required - set(raw.columns)
if missing:
    raise RuntimeError(
        "A4.7 full broad table missing required columns: "
        + ", ".join(sorted(missing))
    )

raw = raw[
    raw["level"].astype(str).str.lower().eq("broad")
    & raw["lineage"].isin(LINEAGES)
].copy()

raw["pathway_testable"] = as_bool(raw["pathway_testable"])
raw["supported_positive_pairwise_correspondence"] = as_bool(
    raw["supported_positive_pairwise_correspondence"]
)

raw["beta_standardized_pairwise_correspondence"] = pd.to_numeric(
    raw["beta_standardized_pairwise_correspondence"],
    errors="coerce",
)

raw["pairwise_correspondence_FDR_collection"] = pd.to_numeric(
    raw["pairwise_correspondence_FDR_collection"],
    errors="coerce",
)

expected_relaxed = (
    raw["pathway_testable"]
    & raw["beta_standardized_pairwise_correspondence"].gt(0)
    & raw["pairwise_correspondence_FDR_collection"].lt(0.05)
)

flag_mismatch = (
    expected_relaxed
    != raw["supported_positive_pairwise_correspondence"]
)

if flag_mismatch.any():
    raise RuntimeError(
        f"Relaxed A4.7 flag mismatch in {int(flag_mismatch.sum())} rows."
    )

relaxed = raw[
    raw["supported_positive_pairwise_correspondence"]
].copy()

if relaxed.duplicated(
    ["lineage", "collection", "pathway"]
).any():
    raise RuntimeError(
        "Duplicate lineage x collection x pathway rows detected."
    )

strict = pd.read_csv(STRICT_HITS, low_memory=False)
strict_ontology = pd.read_csv(STRICT_ONTOLOGY, low_memory=False)

if "lineage" not in strict.columns:
    if "cell_unit_id" not in strict.columns:
        raise RuntimeError("Strict table lacks lineage/cell_unit_id.")
    strict["lineage"] = (
        strict["cell_unit_id"]
        .astype(str)
        .str.replace("broad::", "", regex=False)
    )

if "level" in strict.columns:
    strict = strict[
        strict["level"].astype(str).str.lower().eq("broad")
    ].copy()
elif "cell_unit_id" in strict.columns:
    strict = strict[
        strict["cell_unit_id"].astype(str).str.startswith("broad::")
    ].copy()

strict = strict[strict["lineage"].isin(LINEAGES)].copy()

if "v3_primary_module_id" not in strict_ontology.columns:
    raise RuntimeError(
        "Frozen ontology lacks v3_primary_module_id."
    )

frozen_lookup = {
    str(r.pathway): str(r.v3_primary_module_id)
    for r in strict_ontology.itertuples()
}

summary_rows = []

for lin in LINEAGES:
    z_all = raw[
        (raw["lineage"] == lin)
        & raw["pathway_testable"]
    ]
    z_rel = relaxed[relaxed["lineage"] == lin]
    z_str = strict[strict["lineage"] == lin]

    n_testable = z_all[["collection", "pathway"]].drop_duplicates().shape[0]
    n_relaxed = z_rel[["collection", "pathway"]].drop_duplicates().shape[0]
    n_strict = z_str[["collection", "pathway"]].drop_duplicates().shape[0]

    summary_rows.append({
        "lineage": lin,
        "lineage_label": LINEAGE_LABEL[lin],
        "n_testable": n_testable,
        "n_relaxed": n_relaxed,
        "relaxed_fraction": n_relaxed / n_testable if n_testable else np.nan,
        "n_strict": n_strict,
        "specificity_retention": n_strict / n_relaxed if n_relaxed else np.nan,
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SRC / "Fig3E_and_method_lineage_summary.csv",
    index=False,
)

top_parts = []

for lin in LINEAGES:
    z = relaxed[relaxed["lineage"] == lin].copy()

    z = z.sort_values(
        [
            "beta_standardized_pairwise_correspondence",
            "pairwise_correspondence_FDR_collection",
            "collection",
            "pathway",
        ],
        ascending=[False, True, True, True],
    )

    z = z.head(TOP_N_PER_LINEAGE).copy()
    z["rank_within_lineage"] = np.arange(1, len(z) + 1)
    top_parts.append(z)

top_long = pd.concat(top_parts, ignore_index=True)

top_long.to_csv(
    SRC / "Fig3F_topN_per_lineage_long.csv",
    index=False,
)

top_union = (
    top_long[["collection", "pathway"]]
    .drop_duplicates()
    .copy()
)

nomination_counts = (
    top_long.groupby(["collection", "pathway"])["lineage"]
    .nunique()
    .rename("n_lineages_nominating_topN")
    .reset_index()
)

top_union = top_union.merge(
    nomination_counts,
    on=["collection", "pathway"],
    how="left",
    validate="one_to_one",
)

print("=" * 108)
print("FIG3 E/F/G — TOP PATHWAY VERSION")
print("=" * 108)
print(summary.to_string(index=False))
print()
print(f"Top N per lineage:                 {TOP_N_PER_LINEAGE}")
print(f"Union of top pathways:             {len(top_union)}")

selected_relaxed = relaxed.merge(
    top_union[["collection", "pathway"]],
    on=["collection", "pathway"],
    how="inner",
    validate="many_to_one",
)

heat = selected_relaxed.pivot_table(
    index=["collection", "pathway"],
    columns="lineage",
    values="beta_standardized_pairwise_correspondence",
    aggfunc="first",
)

for lin in LINEAGES:
    if lin not in heat.columns:
        heat[lin] = np.nan

heat = heat[LINEAGES]

row_meta = (
    heat.max(axis=1, skipna=True)
    .rename("max_beta")
    .reset_index()
)

dominant = []
for idx, row in heat.iterrows():
    vals = row.to_numpy(float)
    if np.all(~np.isfinite(vals)):
        dominant.append(None)
    else:
        dominant.append(LINEAGES[int(np.nanargmax(vals))])

row_meta["dominant_lineage"] = dominant
row_meta = row_meta.merge(
    top_union,
    on=["collection", "pathway"],
    how="left",
    validate="one_to_one",
)

row_meta["dominant_lineage_order"] = row_meta["dominant_lineage"].map(
    {x: i for i, x in enumerate(LINEAGES)}
)

row_meta = row_meta.sort_values(
    [
        "dominant_lineage_order",
        "max_beta",
        "n_lineages_nominating_topN",
        "pathway",
    ],
    ascending=[True, False, False, True],
)

ordered_index = pd.MultiIndex.from_frame(
    row_meta[["collection", "pathway"]]
)

heat = heat.reindex(ordered_index)

heat_export = heat.reset_index()
heat_export = heat_export.merge(
    row_meta[
        [
            "collection",
            "pathway",
            "dominant_lineage",
            "n_lineages_nominating_topN",
            "max_beta",
        ]
    ],
    on=["collection", "pathway"],
    how="left",
)

heat_export.to_csv(
    SRC / "Fig3F_top_pathway_heatmap_source.csv",
    index=False,
)

module_rows = []

for r in top_union.itertuples(index=False):
    module_id, source = assign_module(
        r.pathway,
        frozen_lookup,
    )

    module_rows.append({
        "collection": r.collection,
        "pathway": r.pathway,
        "module_id": module_id,
        "module_label": MODULE_LABELS[module_id],
        "assignment_source": source,
        "n_lineages_nominating_topN":
            int(r.n_lineages_nominating_topN),
    })

top_modules = pd.DataFrame(module_rows)

top_modules.to_csv(
    SRC / "Fig3G_top_union_module_mapping.csv",
    index=False,
)

unclassified = top_modules[
    top_modules["module_id"].eq("M99_OTHER_UNCLASSIFIED")
].copy()

unclassified.to_csv(
    AUD / "top_union_unclassified_for_manual_review.csv",
    index=False,
)

unclassified_fraction = len(unclassified) / len(top_modules) if len(top_modules) else np.nan

# Fig3E
e = summary.copy()
e["_order"] = e["lineage"].map(
    {x: i for i, x in enumerate(LINEAGES)}
)
e = e.sort_values("_order")

fig, ax = plt.subplots(figsize=(5.7, 3.0))

y = np.arange(len(e))
pct = 100 * e["relaxed_fraction"].to_numpy(float)

ax.barh(
    y,
    pct,
    height=0.68,
    color=[LINEAGE_COLORS[x] for x in e["lineage"]],
    edgecolor="none",
)

for yi, r in enumerate(e.itertuples()):
    ax.text(
        float(r.relaxed_fraction) * 100 + 0.8,
        yi,
        f"{int(r.n_relaxed)}/{int(r.n_testable)}",
        va="center",
        ha="left",
        fontsize=7.5,
    )

ax.set_yticks(y)
ax.set_yticklabels(e["lineage_label"])
ax.invert_yaxis()
ax.set_xlim(0, 105)
ax.set_xlabel(
    "FDR-supported positive pathways (% of testable pathways)"
)
ax.set_title(
    "Broad pathway correspondence across major lineages\n"
    "(without specificity filtering)",
    loc="left",
    pad=7,
)

clean_axis(ax)
save_figure(fig, "Fig3E_relaxed_correspondence_breadth")

# Fig3F
mat = heat.to_numpy(float)
finite = mat[np.isfinite(mat)]

if len(finite) == 0:
    raise RuntimeError("No finite beta values for Fig3F.")

vmin = float(np.nanpercentile(finite, 2))
vmax = float(np.nanpercentile(finite, 98))

if vmin == vmax:
    vmax = vmin + 1e-6

cmap = mpl.colormaps["viridis"].copy()
cmap.set_bad("white")

fig_h = max(5.2, 0.22 * len(heat) + 1.5)

fig, ax = plt.subplots(
    figsize=(6.4, fig_h)
)

im = ax.imshow(
    np.ma.masked_invalid(mat),
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
)

ax.set_xticks(np.arange(len(LINEAGES)))
ax.set_xticklabels(
    [LINEAGE_LABEL[x] for x in LINEAGES],
    rotation=28,
    ha="right",
)

row_labels = [
    short_pathway(pw)
    for coll, pw in heat.index
]

ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=5.7)

ax.set_title(
    f"Top pathway correspondences across major lineages\n"
    f"(union of top {TOP_N_PER_LINEAGE} per lineage; no specificity filter)",
    loc="left",
    pad=8,
)

ax.grid(False)
ax.tick_params(axis="both", length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.025,
)

cbar.set_label(
    "Standardized correspondence β",
    fontsize=8,
)

fig.text(
    0.995,
    0.01,
    "Blank = not FDR-supported positive in that lineage",
    ha="right",
    va="bottom",
    fontsize=6.5,
)

save_figure(
    fig,
    "Fig3F_top_pathway_correspondence_heatmap",
)

# Fig3G
g = (
    top_modules.groupby(
        ["module_id", "module_label"],
        observed=True,
    )
    .size()
    .reset_index(name="n_unique_pathways")
)

# -------------------------------------------------------------
# Do not display Other/unclassified in Fig3G
# Important: pathways are excluded from visualization only,
# not reassigned or deleted from the audit.
# -------------------------------------------------------------
g = g[
    g["module_id"] != "M99_OTHER_UNCLASSIFIED"
].copy()

order_lookup = {
    m: i
    for i, m in enumerate(DISPLAY_MODULE_ORDER)
}

g["_order"] = g["module_id"].map(order_lookup).fillna(999)

g = g.sort_values(
    ["n_unique_pathways", "_order"],
    ascending=[False, True],
).reset_index(drop=True)

g.to_csv(
    SRC / "Fig3G_top_pathway_module_composition_source.csv",
    index=False,
)

fig, ax = plt.subplots(
    figsize=(5.8, max(3.5, 0.34 * len(g) + 0.9))
)

y = np.arange(len(g))

ax.barh(
    y,
    g["n_unique_pathways"],
    height=0.70,
    color=[MODULE_COLORS[m] for m in g["module_id"]],
    edgecolor="none",
)

maxv = max(1, int(g["n_unique_pathways"].max()))

for yi, v in zip(y, g["n_unique_pathways"]):
    ax.text(
        v + 0.10,
        yi,
        str(int(v)),
        va="center",
        ha="left",
        fontsize=7,
    )

ax.set_yticks(y)
ax.set_yticklabels(g["module_label"])
ax.invert_yaxis()

ax.set_xlabel("Classified top-ranked pathways")

ax.set_title(
    "Biological composition of classified top-ranked pathway correspondences",
    loc="left",
    pad=7,
)

ax.set_xlim(0, maxv * 1.18 + 0.5)

clean_axis(ax)

save_figure(
    fig,
    "Fig3G_top_pathway_module_composition",
)

# Supplement B-like
supp_parts = []

fig, axes = plt.subplots(
    nrows=len(LINEAGES),
    ncols=1,
    figsize=(8.6, 8.8),
)

for ax, lin in zip(axes, LINEAGES):
    z = relaxed[
        relaxed["lineage"] == lin
    ].copy()

    z = z.sort_values(
        [
            "beta_standardized_pairwise_correspondence",
            "pairwise_correspondence_FDR_collection",
            "collection",
            "pathway",
        ],
        ascending=[False, True, True, True],
    ).head(TOP_N_SUPP)

    z = z.sort_values(
        "beta_standardized_pairwise_correspondence",
        ascending=True,
    )

    z["display_pathway"] = z["pathway"].map(short_pathway)
    z["lineage_label"] = LINEAGE_LABEL[lin]
    supp_parts.append(z)

    y = np.arange(len(z))

    ax.barh(
        y,
        z["beta_standardized_pairwise_correspondence"],
        height=0.68,
        color=LINEAGE_COLORS[lin],
        edgecolor="none",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        z["display_pathway"],
        fontsize=6.3,
    )

    ax.set_xlabel(
        "Standardized correspondence β"
    )

    ax.set_title(
        LINEAGE_LABEL[lin],
        loc="left",
        fontsize=9,
        pad=3,
    )

    clean_axis(ax)

fig.suptitle(
    "Top pathway correspondences without specificity filtering",
    x=0.01,
    y=0.995,
    ha="left",
    fontsize=11,
    fontweight="bold",
)

fig.subplots_adjust(
    left=0.48,
    right=0.98,
    top=0.94,
    bottom=0.07,
    hspace=0.90,
)

supp_b = pd.concat(
    supp_parts,
    ignore_index=True,
)

supp_b.to_csv(
    SRC / "Supp_B_like_top_pathways_no_specificity.csv",
    index=False,
)

save_figure(
    fig,
    "Supp_B_like_top_pathways_no_specificity",
)

# Supplement method 1
fig, ax = plt.subplots(
    figsize=(6.2, 3.1)
)

x = np.arange(len(summary))
w = 0.36

ax.bar(
    x - w / 2,
    summary["n_relaxed"],
    width=w,
    color="#B8C1CC",
    edgecolor="none",
    label="No specificity filter",
)

ax.bar(
    x + w / 2,
    summary["n_strict"],
    width=w,
    color="#3D485A",
    edgecolor="none",
    label="Specificity-supported",
)

max_relaxed = max(summary["n_relaxed"])

for xi, a, b in zip(
    x,
    summary["n_relaxed"],
    summary["n_strict"],
):
    ax.text(
        xi - w / 2,
        a + max_relaxed * 0.012,
        str(int(a)),
        ha="center",
        va="bottom",
        fontsize=7,
    )
    ax.text(
        xi + w / 2,
        b + max_relaxed * 0.012,
        str(int(b)),
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax.set_xticks(x)

ax.set_xticklabels(
    summary["lineage_label"],
    rotation=28,
    ha="right",
)

ax.set_ylabel("Unique pathways")

ax.set_title(
    "Pathway correspondence before and after specificity filtering",
    loc="left",
    pad=7,
)

ax.legend(
    frameon=False,
    loc="upper right",
)

clean_axis(ax)
save_figure(
    fig,
    "Supp_Method1_relaxed_vs_strict_pathway_counts",
)

# Supplement method 2
fig, ax = plt.subplots(
    figsize=(5.8, 2.9)
)

for i, r in summary.iterrows():
    if not np.isfinite(
        r["specificity_retention"]
    ):
        continue

    ax.scatter(
        i,
        r["specificity_retention"],
        s=85,
        color=LINEAGE_COLORS[r["lineage"]],
        edgecolor="white",
        linewidth=0.65,
        zorder=3,
    )

    ax.text(
        i,
        r["specificity_retention"] + 0.00035,
        f'{r["specificity_retention"]:.3%}',
        ha="center",
        va="bottom",
        fontsize=7,
    )

ax.set_xticks(np.arange(len(summary)))

ax.set_xticklabels(
    summary["lineage_label"],
    rotation=28,
    ha="right",
)

max_ret = float(
    np.nanmax(summary["specificity_retention"])
)

ax.set_ylim(
    0,
    max(0.015, max_ret * 1.28),
)

ax.set_ylabel(
    "Specificity-supported / relaxed"
)

ax.set_title(
    "Specificity retention across major lineages",
    loc="left",
    pad=7,
)

clean_axis(ax)
save_figure(
    fig,
    "Supp_Method2_specificity_retention",
)

assignment_summary = (
    top_modules["assignment_source"]
    .value_counts(dropna=False)
    .rename_axis("assignment_source")
    .reset_index(name="n_top_union_pathways")
)

assignment_summary.to_csv(
    AUD / "top_union_module_assignment_coverage.csv",
    index=False,
)

audit = {
    "status": "COMPLETED",
    "raw_A4_7_source": str(RAW_A47),
    "strict_hits_source": str(STRICT_HITS),
    "strict_ontology_source": str(STRICT_ONTOLOGY),
    "relaxed_definition": (
        "pathway_testable AND beta_standardized_pairwise_correspondence > 0 "
        "AND pairwise_correspondence_FDR_collection < 0.05"
    ),
    "specificity_filter_used_for_Fig3E": False,
    "specificity_filter_used_for_Fig3F": False,
    "specificity_filter_used_for_Fig3G": False,
    "top_n_per_lineage_for_Fig3F": TOP_N_PER_LINEAGE,
    "top_union_unique_pathways": int(len(top_union)),
    "reclustering_performed": False,
    "pathway_pair_overlap_performed": False,
    "strict_results_modified": False,
    "Fig3F_blank_cell_meaning": (
        "Pathway was not FDR-supported positive in that lineage."
    ),
    "Fig3G_scope": (
        "Descriptive biological-module composition of the small set of "
        "unique pathways shown in Fig3F."
    ),
    "module_mapping": (
        "Frozen V3 assignment preserved for strict pathways; "
        "relaxed-only top pathways mapped using the same V3 manual/keyword rules."
    ),
    "top_union_unclassified_fraction": (
        float(unclassified_fraction)
        if np.isfinite(unclassified_fraction)
        else None
    ),
    "warning": (
        "Module labels for relaxed-only top pathways are descriptive post-hoc "
        "extensions and should be manually reviewed before manuscript freeze."
    ),
}

(AUD / "audit.json").write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("TOP-UNION MODULE ASSIGNMENT COVERAGE")
print(
    assignment_summary.to_string(
        index=False
    )
)

print(
    f"\nTop-union Other/unclassified: "
    f"{len(unclassified)}/{len(top_modules)} "
    f"({unclassified_fraction:.1%})"
)

if unclassified_fraction > 0.30:
    print(
        "WARNING: >30% of Fig3F union is Other/unclassified. "
        "Inspect audit/top_union_unclassified_for_manual_review.csv "
        "before using Fig3G in the main figure."
    )
else:
    print(
        "Module coverage is acceptable for descriptive Fig3G, "
        "but manual review is still recommended."
    )

print()
print("MAIN FIGURES")
print(FIG / "Fig3E_relaxed_correspondence_breadth.png")
print(FIG / "Fig3F_top_pathway_correspondence_heatmap.png")
print(FIG / "Fig3G_top_pathway_module_composition.png")

print()
print("SUPPLEMENTARY")
print(FIG / "Supp_B_like_top_pathways_no_specificity.png")
print(FIG / "Supp_Method1_relaxed_vs_strict_pathway_counts.png")
print(FIG / "Supp_Method2_specificity_retention.png")

print()
print("No pathway-pair overlap calculation; no complete-linkage clustering.")
print("=" * 108)
