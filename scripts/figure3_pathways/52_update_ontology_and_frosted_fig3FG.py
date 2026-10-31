#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
52_update_ontology_and_frosted_fig3FG.py

目的
----
按最新审美要求，重画 3 类图：

A. 原 ontology landscape（对应你给的图1）
   - 把 "Collection representation" 和 "Formal cell-unit coverage"
     两个图例都放到右下角外侧
   - 删除 "Both collections" 图例与菱形表示
   - 如果某个 family 同时包含 Hallmark 和 Reactome，则直接按
     representative_pathway 所属 collection 决定点形状
     （HALLMARK_=圆形；REACTOME_=方形）

B. matched-competitive representative pathways（对应你给的图2）
   - y 轴标签不再写 "Hallmark | ..." / "Reactome | ..."
   - 改成你要的 CL 编号形式，例如：
       CL001 | KRAS SIGNALING UP
       CL002 | TNFA SIGNALING VIA NFKB
   - collection 仍然只通过图例里的圆形/方形体现
   - supporting lineages 仍通过点大小体现
   - 输出一份 CL 编号映射表，便于正文和图注引用

C. Fig3F / Fig3G heatmap
   - 不再用显著性黑点
   - 对 FDR >= 0.05 的 cell 做更强的 frosted-glass 处理：
       1) 更高 alpha 的冷白色遮罩
       2) 轻度蓝灰哑光内层
       3) 斜向磨砂纹理
       4) 半透明边框
   - 目标：一眼能看出“这个格子不显著”，但仍保留底层颜色趋势

本脚本只重画，不重跑统计分析。
"""

from __future__ import annotations
import os

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


# =============================================================================
# 基础路径
# =============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
ROOT = PROJECT / "corrected_full_analysis_v2"

# 34：原 ontology landscape
SRC34 = ROOT / "34_TABULA_STYLE_CANDIDATE_ADDITIONS_20260920" / "05_PATHWAY_ONTOLOGY_LANDSCAPE"

# 50：matched competitive 主结果
SRC50 = ROOT / "50_MATCHED_COMPETITIVE_TISSUE_PATHWAYS_20260922"
SRC50_TAB = SRC50 / "tables"

# frozen V3 ontology（用于模块赋值）
V3_FREEZE = ROOT / "00_freeze" / "A4_8_V3_FINAL_ONTOLOGY_20260920"
STRICT_ONTOLOGY = V3_FREEZE / "01_V3_PATHWAY_ONTOLOGY.csv"

OUT = ROOT / "52_RESTYLE_ONTOLOGY_AND_FIG3FG_20260922"
FIG = OUT / "figures"
TAB = OUT / "source_data"
AUD = OUT / "audit"

for d in [OUT, FIG, TAB, AUD]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 常量
# =============================================================================

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

MODULE_DISPLAY_ORDER = [
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
    "M99_OTHER_UNCLASSIFIED": "#A0A4AA",
}

MODULE_PRIORITY = list(MODULE_LABELS.keys())

KEYWORDS = {
    "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE": [
        "TNF", "TNFR", "NFKB", "NF_KB", "NF_KAPPA_B",
        "TOLL", "TLR", "MYD88", "IRAK", "RIPK",
        "INFLAMMATORY", "CYTOKINE", "INTERLEUKIN",
        "IL1", "IL_1", "IL17", "IL_17",
        "IL2_STAT5", "IL_2_STAT5",
        "IL6_JAK_STAT3", "IL_6_JAK_STAT3",
        "INNATE_IMMUNE",
    ],
    "M02_MAPK_RTK_GROWTH_FACTOR": [
        "MAPK", "RAF", "RAS", "RECEPTOR_TYROSINE_KINASE", "RTK",
        "NTRK", "EGFR", "ERBB", "FGFR", "MET",
        "SPRY", "GROWTH_FACTOR", "ERYTHROPOIETIN",
        "PHOSPHOLIPASE_C_GAMMA",
    ],
    "M03_TGFB_SMAD": [
        "TGFB", "TGF_BETA", "TGF_BETA_RECEPTOR", "SMAD",
    ],
    "M04_CELL_DEATH_APOPTOSIS": [
        "APOPTOSIS", "APOPTOTIC", "DEATH_RECEPTOR",
        "BH3", "PUMA", "CASPASE", "NECROSIS", "NECROPTOSIS",
    ],
    "M05_RHO_CYTOSKELETON_ADHESION_REMODELING": [
        "RHO", "RAC", "ROCK", "CYTOSKELET", "INTEGRIN",
        "EPHRIN", "SEMAPHORIN", "ADHESION",
        "EPITHELIAL_MESENCHYMAL_TRANSITION",
    ],
    "M06_TRAFFICKING_ER_PROTEOSTASIS": [
        "MEMBRANE_TRAFFICKING", "VESICLE", "GOLGI", "ER_RETROGRADE",
        "RETROGRADE_TRANSPORT", "UBIQUITIN", "PROTEASOME", "AUTOPHAGY",
        "IRE1", "CHAPERONE", "POST_TRANSLATIONAL", "CLATHRIN",
        "UNFOLDED_PROTEIN_RESPONSE", "HEAT_STRESS",
    ],
    "M07_ANTIGEN_INTERFERON_IMMUNE_EFFECTOR": [
        "ANTIGEN", "MHC", "INTERFERON", "DDX58", "IFIH1",
        "IMMUNE_EFFECTOR", "ADAPTIVE_IMMUNE", "T_CELL_ACTIVATION",
        "NEUTROPHIL", "DEGRANULATION", "PD_L1", "CD274",
    ],
    "M08_TRANSCRIPTION_CHROMATIN_DIFFERENTIATION": [
        "ESTROGEN", "ANDROGEN", "HDAC", "HISTONE", "CHROMATIN",
        "BETA_CATENIN", "TCF", "RUNX", "TRANSCRIPTIONAL_REGULATION",
        "GENE_EXPRESSION", "DIFFERENTIATION",
    ],
    "M09_METABOLIC_OXIDATIVE_STRESS": [
        "HEME", "HMOX1", "NFE2L2", "KEAP1", "OXIDATIVE_PHOSPHORYLATION",
        "CHOLESTEROL", "FATTY_ACID", "AMINO_ACID", "CYP",
        "METABOLISM", "HYPOXIA", "RESPIRATORY_ELECTRON_TRANSPORT",
    ],
    "M10_CELL_CYCLE_GROWTH": [
        "MITOTIC", "CELL_CYCLE", "MTOR", "PI3K", "P53",
    ],
    "M11_OTHER_SIGNALING_SPECIALIZED": [
        "NOTCH", "HEDGEHOG", "ACTIVATION_OF_SMO", "G_ALPHA",
        "SECOND_MESSENGER", "CHANNEL", "NEURONAL_SYSTEM",
        "PLATELET_ACTIVATION", "GPCR", "HEMOSTASIS", "ATTENUATION_PHASE",
        "CILIUM",
    ],
}

MANUAL_OVERRIDES = {
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION":
        "M05_RHO_CYTOSKELETON_ADHESION_REMODELING",
    "HALLMARK_IL2_STAT5_SIGNALING":
        "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING":
        "M01_CYTOKINE_TNF_NFKB_INNATE_IMMUNE",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_MEMBRANE_TRAFFICKING":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_COPI_DEPENDENT_GOLGI_TO_ER_RETROGRADE_TRAFFIC":
        "M06_TRAFFICKING_ER_PROTEOSTASIS",
    "REACTOME_REGULATION_OF_HMOX1_EXPRESSION_AND_ACTIVITY":
        "M09_METABOLIC_OXIDATIVE_STRESS",
    "REACTOME_REGULATION_OF_NFE2L2_GENE_EXPRESSION":
        "M09_METABOLIC_OXIDATIVE_STRESS",
    "REACTOME_ACTIVATION_OF_SMO":
        "M11_OTHER_SIGNALING_SPECIALIZED",
    "REACTOME_NEURONAL_SYSTEM":
        "M11_OTHER_SIGNALING_SPECIALIZED",
}


# =============================================================================
# style
# =============================================================================

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# helpers
# =============================================================================

def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.12, "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FIG / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def wrap_label(x, width=44):
    return "\n".join(textwrap.wrap(str(x), width=width, break_long_words=False, break_on_hyphens=False))


def pretty_pathway(x):
    return str(x).replace("HALLMARK_", "").replace("REACTOME_", "").replace("_", " ")


def as_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def marker_from_pathway(pathway):
    up = str(pathway).upper()
    if up.startswith("HALLMARK_"):
        return "o"
    if up.startswith("REACTOME_"):
        return "s"
    return "o"


def load_frozen_module_lookup():
    if not STRICT_ONTOLOGY.exists():
        return {}
    x = pd.read_csv(STRICT_ONTOLOGY, low_memory=False)
    if "pathway" not in x.columns or "v3_primary_module_id" not in x.columns:
        return {}
    return {
        str(r.pathway): str(r.v3_primary_module_id)
        for r in x.itertuples(index=False)
        if pd.notna(r.v3_primary_module_id)
    }


def assign_module(pathway, frozen_lookup):
    p = str(pathway)
    if p in frozen_lookup:
        return frozen_lookup[p], "frozen_v3"

    up = p.upper()
    if up in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[up], "manual_override"

    for module in MODULE_PRIORITY:
        if module == "M99_OTHER_UNCLASSIFIED":
            continue
        if any(key in up for key in KEYWORDS.get(module, [])):
            return module, "keyword"

    return "M99_OTHER_UNCLASSIFIED", "no_rule_match"


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate files exist:\n" + "\n".join(str(x) for x in paths))


def add_outside_legends(fig, ax, include_collection=True, include_size=True, size_values=(1, 3, 6), size_prefix="cell unit"):
    # 右下角外侧，上下堆叠
    collection_handles = []
    if include_collection:
        collection_handles = [
            Line2D([], [], marker="o", ls="", color="#555555", label="Hallmark"),
            Line2D([], [], marker="s", ls="", color="#555555", label="Reactome"),
        ]
        leg1 = ax.legend(
            handles=collection_handles,
            frameon=False,
            bbox_to_anchor=(1.02, 0.08),
            loc="lower left",
            title="Collection",
            title_fontsize=7,
            fontsize=7,
            borderaxespad=0.0,
            handletextpad=0.6,
            labelspacing=0.35,
        )
        ax.add_artist(leg1)

    if include_size:
        size_handles = [
            ax.scatter([], [], s=38 + 34 * n, color="#A7A7A7", edgecolor="white", linewidth=0.7,
                       label=f"{n} {size_prefix}" if n == 1 else f"{n} {size_prefix}s")
            for n in size_values
        ]
        ax.legend(
            handles=size_handles,
            frameon=False,
            bbox_to_anchor=(1.02, 0.20),
            loc="lower left",
            title="Supporting lineages" if size_prefix == "lineage" else "Formal cell-unit coverage",
            title_fontsize=7,
            fontsize=7,
            borderaxespad=0.0,
            scatterpoints=1,
            labelspacing=0.35,
        )


# =============================================================================
# A. redraw original ontology landscape (图1)
# =============================================================================

def load_ontology_family_tables():
    selected_candidates = [
        SRC34 / "C05_selected_family_ontology.csv",
        SRC34 / "tables" / "C05_selected_family_ontology.csv",
    ]
    all_candidates = [
        SRC34 / "C05_all_family_ontology.csv",
        SRC34 / "tables" / "C05_all_family_ontology.csv",
    ]

    selected = pd.read_csv(find_first_existing(selected_candidates), low_memory=False)

    all_df = None
    for p in all_candidates:
        if p.exists():
            all_df = pd.read_csv(p, low_memory=False)
            break

    return selected, all_df


def family_label(row):
    cid = str(row["complete_linkage_cluster_id"])
    rep = pretty_pathway(row["representative_pathway"])
    return f"{cid} | {rep}"


def plot_ontology_bubble(df, stem, title, label_fs=6.2):
    if df is None or df.empty:
        return

    required = {
        "complete_linkage_cluster_id",
        "representative_pathway",
        "display_module_id",
        "median_beta",
        "n_formal_cell_units",
    }
    miss = required - set(df.columns)
    if miss:
        raise RuntimeError(f"{stem} missing columns: {sorted(miss)}")

    mod_rank = {m: i for i, m in enumerate(MODULE_DISPLAY_ORDER)}

    dd = df.copy()
    dd["_module_rank"] = dd["display_module_id"].map(mod_rank).fillna(999)

    # 如果存在原脚本排序字段，保留；否则退化为模块内 coverage + beta 排序
    sort_cols = ["_module_rank"]
    ascending = [True]

    if "n_formal_cell_units" in dd.columns:
        sort_cols.append("n_formal_cell_units")
        ascending.append(False)
    if "median_beta" in dd.columns:
        sort_cols.append("median_beta")
        ascending.append(False)
    if "complete_linkage_cluster_id" in dd.columns:
        sort_cols.append("complete_linkage_cluster_id")
        ascending.append(True)

    dd = dd.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)

    fig_h = max(6.5, min(22.0, 2.0 + 0.32 * len(dd)))
    fig, ax = plt.subplots(figsize=(9.1, fig_h))
    y = np.arange(len(dd))

    # 背景 module 带
    for mod, grp in dd.groupby("display_module_id", sort=False):
        idx = grp.index.to_numpy()
        lo = idx.min() - 0.5
        hi = idx.max() + 0.5
        col = MODULE_COLORS.get(str(mod), "#BBBBBB")
        ax.axhspan(lo, hi, color=col, alpha=0.09, zorder=0)

        ax.text(
            1.01,
            1 - ((lo + hi) / 2 + 0.5) / len(dd),
            MODULE_LABELS.get(str(mod), str(mod)),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.2,
            color=col,
            clip_on=False,
        )

    for i, r in dd.iterrows():
        shape = marker_from_pathway(r["representative_pathway"])
        ax.scatter(
            float(r["median_beta"]),
            i,
            s=34 + 34 * float(r["n_formal_cell_units"]),
            marker=shape,
            color=MODULE_COLORS.get(str(r["display_module_id"]), "#999999"),
            edgecolor="white",
            linewidth=0.65,
            alpha=0.92,
            zorder=3,
        )

    labels = [family_label(r) for _, r in dd.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, 52) for x in labels], fontsize=label_fs)
    ax.set_ylim(len(dd) - 0.4, -0.6)
    ax.set_xlabel("Median standardized pairwise correspondence β")
    ax.set_title(title, pad=10)
    clean_axis(ax)

    fig.subplots_adjust(left=0.45 if len(dd) <= 30 else 0.48, right=0.80, bottom=0.09, top=0.95)
    add_outside_legends(fig, ax, include_collection=True, include_size=True, size_values=(1, 3, 6), size_prefix="cell unit")

    dd.to_csv(TAB / f"{stem}_source.csv", index=False)
    save_figure(fig, stem)


# =============================================================================
# B. matched competitive representative pathways (图2)
# =============================================================================

def load_matched_inputs():
    results = pd.read_csv(SRC50_TAB / "ALL_LINEAGES_matched_competitive_pathways.csv.gz", low_memory=False)
    results["supported"] = as_bool(results["supported"])
    results["matched_z"] = pd.to_numeric(results["matched_z"], errors="coerce")
    results["FDR_matched"] = pd.to_numeric(results["FDR_matched"], errors="coerce")

    fsel = pd.read_csv(SRC50_TAB / "Fig3F_selected_Hallmark.csv", low_memory=False)
    gsel = pd.read_csv(SRC50_TAB / "Fig3G_selected_Reactome.csv", low_memory=False)

    return results, fsel, gsel


def build_matched_representative_table(results, fsel, gsel, per_module=2, max_total=20):
    frozen_lookup = load_frozen_module_lookup()
    main_selected = set(fsel["pathway"].astype(str)) | set(gsel["pathway"].astype(str))

    supp = results[results["supported"] & results["matched_z"].gt(0)].copy()

    summary = (
        supp.groupby(["collection", "pathway"], observed=True)
        .agg(
            n_supporting_lineages=("lineage", "nunique"),
            median_supported_z=("matched_z", "median"),
            max_supported_z=("matched_z", "max"),
            supporting_lineages=("lineage", lambda x: ";".join(sorted(set(str(v) for v in x)))),
        )
        .reset_index()
    )

    mids = []
    srcs = []
    for p in summary["pathway"].astype(str):
        mid, src = assign_module(p, frozen_lookup)
        mids.append(mid)
        srcs.append(src)

    summary["module_id"] = mids
    summary["module_label"] = summary["module_id"].map(MODULE_LABELS)
    summary["assignment_source"] = srcs
    summary["selected_in_main_heatmaps"] = summary["pathway"].astype(str).isin(main_selected)

    summary.to_csv(AUD / "matched_pathway_module_assignment_all.csv", index=False)

    candidate = summary[~summary["module_id"].eq("M99_OTHER_UNCLASSIFIED")].copy()
    mod_order = {m: i for i, m in enumerate(MODULE_DISPLAY_ORDER)}
    candidate["_module_order"] = candidate["module_id"].map(mod_order).fillna(999)

    candidate = candidate.sort_values(
        ["_module_order", "selected_in_main_heatmaps", "n_supporting_lineages", "median_supported_z", "max_supported_z", "pathway"],
        ascending=[True, False, False, False, False, True],
    )

    selected_parts = []
    for m in MODULE_DISPLAY_ORDER:
        if m == "M99_OTHER_UNCLASSIFIED":
            continue
        z = candidate[candidate["module_id"].eq(m)].head(per_module)
        if len(z):
            selected_parts.append(z)

    if not selected_parts:
        raise RuntimeError("No classified matched pathways available.")

    selected = pd.concat(selected_parts, ignore_index=True)

    if len(selected) > max_total:
        first = (
            selected.sort_values(
                ["_module_order", "selected_in_main_heatmaps", "n_supporting_lineages", "median_supported_z"],
                ascending=[True, False, False, False],
            )
            .groupby("module_id", observed=True, sort=False)
            .head(1)
        )
        used = set(zip(first["collection"], first["pathway"]))
        remaining = selected[
            ~selected.apply(lambda r: (r["collection"], r["pathway"]) in used, axis=1)
        ].sort_values(
            ["selected_in_main_heatmaps", "n_supporting_lineages", "median_supported_z", "max_supported_z"],
            ascending=[False, False, False, False],
        )
        need = max(0, max_total - len(first))
        selected = pd.concat([first, remaining.head(need)], ignore_index=True)

    selected["_module_order"] = selected["module_id"].map(mod_order).fillna(999)
    selected = selected.sort_values(
        ["_module_order", "median_supported_z", "n_supporting_lineages", "pathway"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    # 这里给你新的 CL 编号
    selected["plot_cl_id"] = [f"CL{i:03d}" for i in range(1, len(selected) + 1)]
    selected["plot_label"] = selected.apply(
        lambda r: f"{r['plot_cl_id']} | {pretty_pathway(r['pathway'])}",
        axis=1,
    )

    selected.to_csv(TAB / "Supp_matched_representative_pathways_with_CL_source.csv", index=False)

    mapping = selected[
        [
            "plot_cl_id",
            "collection",
            "pathway",
            "module_id",
            "module_label",
            "n_supporting_lineages",
            "supporting_lineages",
            "median_supported_z",
        ]
    ].copy()
    mapping.to_csv(TAB / "Supp_matched_CL_id_mapping.csv", index=False)

    return selected


def plot_matched_representative(selected, stem="Supp_representative_matched_pathways_by_module_CL"):
    modules_present = [
        m for m in MODULE_DISPLAY_ORDER
        if m in set(selected["module_id"]) and m != "M99_OTHER_UNCLASSIFIED"
    ]

    ordered_rows = []
    module_ranges = []
    cursor = 0

    for m in modules_present:
        z = selected[selected["module_id"].eq(m)].copy()
        z = z.sort_values(
            ["median_supported_z", "n_supporting_lineages", "pathway"],
            ascending=[False, False, True],
        )
        start = cursor
        for _, r in z.iterrows():
            ordered_rows.append(r)
            cursor += 1
        end = cursor - 1
        module_ranges.append((m, start, end))

    p = pd.DataFrame(ordered_rows).reset_index(drop=True)

    fig_h = max(6.2, 0.43 * len(p) + 1.8)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    plt.subplots_adjust(left=0.43, right=0.79, bottom=0.15, top=0.94)

    n = len(p)
    y = np.arange(n)

    xmin = float(np.nanmin(p["median_supported_z"]))
    xmax = float(np.nanmax(p["median_supported_z"]))
    xr = max(1.0, xmax - xmin)

    x_left = max(0.0, xmin - 0.16 * xr)
    x_right = xmax + 0.18 * xr

    for mod, start, end in module_ranges:
        c = MODULE_COLORS[mod]
        ax.axhspan(start - 0.48, end + 0.48, color=c, alpha=0.09, zorder=0)
        mid = (start + end) / 2
        ax.text(
            1.015,
            1 - ((mid + 0.5) / n),
            MODULE_LABELS[mod],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.4,
            color=c,
            clip_on=False,
        )
        if end < n - 1:
            ax.axhline(end + 0.5, color=c, alpha=0.22, linewidth=0.7, zorder=0.5)

    size_map = {1: 44, 2: 68, 3: 92, 4: 122, 5: 152}

    for yi, r in enumerate(p.itertuples(index=False)):
        marker = "o" if str(r.collection) == "Hallmark" else "s"
        ax.scatter(
            r.median_supported_z,
            yi,
            s=size_map.get(int(r.n_supporting_lineages), 44),
            marker=marker,
            color=MODULE_COLORS[r.module_id],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.96,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, 48) for x in p["plot_label"]], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("Median matched competitive enrichment z\n(among supporting lineages)")
    ax.set_title("Representative matched-competitive pathways organized by biological module", loc="left", pad=8)
    clean_axis(ax)

    # 右下角图例：先 collection，再 lineage size
    collection_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="white", markersize=7, label="Hallmark"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#555555", markeredgecolor="white", markersize=7, label="Reactome"),
    ]
    leg1 = ax.legend(
        handles=collection_handles,
        title="Collection",
        frameon=False,
        bbox_to_anchor=(1.02, 0.06),
        loc="lower left",
        borderaxespad=0,
    )
    ax.add_artist(leg1)

    size_handles = [
        ax.scatter([], [], s=size_map[k], color="#A7A7A7", edgecolor="white", linewidth=0.7,
                   label=f"{k} lineage" if k == 1 else f"{k} lineages")
        for k in [1, 3, 5]
    ]
    ax.legend(
        handles=size_handles,
        title="Supporting lineages",
        frameon=False,
        bbox_to_anchor=(1.26, 0.06),
        loc="lower left",
        borderaxespad=0,
    )

    p.to_csv(TAB / f"{stem}_ordered_source.csv", index=False)
    save_figure(fig, stem)


# =============================================================================
# C. stronger frosted heatmap (Fig3F/G)
# =============================================================================

def draw_frosted_cell(ax, x0, y0, alpha=0.84):
    # 外层高 alpha 冷白 veil
    ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            1.0, 1.0,
            facecolor=(0.97, 0.98, 0.99, alpha),
            edgecolor=(1.0, 1.0, 1.0, 0.92),
            linewidth=0.9,
            zorder=2.0,
        )
    )

    # 内层轻度蓝灰哑光层
    ax.add_patch(
        Rectangle(
            (x0 - 0.44, y0 - 0.44),
            0.88, 0.88,
            facecolor=(0.88, 0.92, 0.96, 0.18),
            edgecolor="none",
            zorder=2.05,
        )
    )

    # 两条磨砂斜纹，让人更容易一眼识别
    line_specs = [
        ((x0 - 0.42, y0 + 0.16), (x0 - 0.10, y0 + 0.42)),
        ((x0 - 0.12, y0 - 0.34), (x0 + 0.28, y0 - 0.02)),
        ((x0 + 0.03, y0 + 0.10), (x0 + 0.34, y0 + 0.38)),
    ]
    for (x1, y1), (x2, y2) in line_specs:
        ax.plot(
            [x1, x2], [y1, y2],
            color=(1, 1, 1, 0.45),
            linewidth=1.1,
            solid_capstyle="round",
            zorder=2.1,
        )

    # 轻微边框
    ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            1.0, 1.0,
            facecolor="none",
            edgecolor=(0.83, 0.87, 0.91, 0.65),
            linewidth=0.7,
            zorder=2.2,
        )
    )


def plot_frosted_heatmap(results, selected, collection, stem, title, z_limit=6.0, frost_alpha=0.84):
    order = selected.sort_values("display_order")["pathway"].astype(str).tolist()

    z = results[
        results["collection"].eq(collection) &
        results["pathway"].isin(order)
    ].copy()

    effect = z.pivot_table(index="pathway", columns="lineage", values="matched_z", aggfunc="first")
    qmat = z.pivot_table(index="pathway", columns="lineage", values="FDR_matched", aggfunc="first")

    for lin in LINEAGES:
        if lin not in effect.columns:
            effect[lin] = np.nan
        if lin not in qmat.columns:
            qmat[lin] = np.nan

    effect = effect[LINEAGES].reindex(order)
    qmat = qmat[LINEAGES].reindex(order)

    mat = effect.to_numpy(float)
    q = qmat.to_numpy(float)

    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#F2F2F2")

    fig_h = max(4.1, 0.31 * len(order) + 1.35)
    fig, ax = plt.subplots(figsize=(6.6, fig_h))

    im = ax.imshow(
        np.ma.masked_invalid(mat),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-float(z_limit),
        vmax=float(z_limit),
        zorder=1,
    )

    nonsig = (~np.isfinite(q)) | (q >= 0.05)
    for i in range(nonsig.shape[0]):
        for j in range(nonsig.shape[1]):
            if nonsig[i, j]:
                draw_frosted_cell(ax, j, i, alpha=frost_alpha)

    ax.set_xticks(np.arange(len(LINEAGES)))
    ax.set_xticklabels([LINEAGE_LABEL[x] for x in LINEAGES], rotation=28, ha="right")

    if collection == "Reactome":
        yt = [wrap_label(pretty_pathway(p), 50) for p in order]
        yfs = 5.7
    else:
        yt = [wrap_label(pretty_pathway(p), 42) for p in order]
        yfs = 6.2

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(yt, fontsize=yfs)
    ax.set_title(title, loc="left", pad=8)

    ax.tick_params(axis="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Matched competitive enrichment z", fontsize=8)

    fig.text(
        0.995, 0.01,
        "Clear cell: FDR < 0.05   |   Frosted cell: FDR ≥ 0.05",
        ha="right", va="bottom", fontsize=6.2
    )

    pd.concat(
        [
            effect.add_suffix("__matched_z"),
            qmat.add_suffix("__FDR"),
        ],
        axis=1
    ).reset_index().to_csv(TAB / f"{stem}_source.csv", index=False)

    save_figure(fig, stem)


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frost-alpha", type=float, default=0.84)
    ap.add_argument("--z-limit", type=float, default=6.0)
    ap.add_argument("--matched-per-module", type=int, default=2)
    ap.add_argument("--matched-max-total", type=int, default=20)
    args = ap.parse_args()

    print("=" * 110)
    print("RESTYLE ONTOLOGY + MATCHED REPRESENTATIVE + STRONGER FROSTED FIG3F/G")
    print("=" * 110)
    print("No statistics are recomputed; only figure styles / labels are updated.")
    print()

    # A. ontology (图1)
    selected_family, all_family = load_ontology_family_tables()
    plot_ontology_bubble(
        selected_family,
        stem="Supp_C05A_selected_pathway_family_ontology_legend_outside",
        title="Representative coherent pathway families organized by final biological module",
        label_fs=6.4,
    )
    if all_family is not None and len(all_family):
        plot_ontology_bubble(
            all_family,
            stem="Supp_C05B_all_pathway_family_ontology_legend_outside",
            title="All formal coherent pathway families organized by final biological module",
            label_fs=5.2,
        )

    # B + C
    results, fsel, gsel = load_matched_inputs()

    matched_selected = build_matched_representative_table(
        results, fsel, gsel,
        per_module=args.matched_per_module,
        max_total=args.matched_max_total,
    )
    plot_matched_representative(matched_selected)

    plot_frosted_heatmap(
        results=results,
        selected=fsel,
        collection="Hallmark",
        stem="Fig3F_Hallmark_frosted_stronger",
        title="Hallmark programs with tissue effects beyond matched gene background",
        z_limit=args.z_limit,
        frost_alpha=args.frost_alpha,
    )
    plot_frosted_heatmap(
        results=results,
        selected=gsel,
        collection="Reactome",
        stem="Fig3G_Reactome_frosted_stronger",
        title="Representative Reactome programs with tissue effects beyond matched background",
        z_limit=args.z_limit,
        frost_alpha=args.frost_alpha,
    )

    manifest = {
        "recomputed_statistics": False,
        "changes": {
            "ontology_plot": [
                "Legends moved outside to lower-right",
                "Removed 'Both collections' legend entry",
                "Marker shape determined by representative_pathway collection only",
            ],
            "matched_plot": [
                "Assigned new CL-style IDs",
                "Y-axis labels no longer show collection prefix",
                "Collection still encoded by shape legend",
            ],
            "Fig3F_G": [
                "Removed significance dots",
                "Strengthened frosted overlay for FDR>=0.05 cells",
            ],
        },
    }
    (AUD / "restyle_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("OUTPUT FIGURES")
    print(FIG / "Supp_C05A_selected_pathway_family_ontology_legend_outside.png")
    if (FIG / "Supp_C05B_all_pathway_family_ontology_legend_outside.png").exists():
        print(FIG / "Supp_C05B_all_pathway_family_ontology_legend_outside.png")
    print(FIG / "Supp_representative_matched_pathways_by_module_CL.png")
    print(FIG / "Fig3F_Hallmark_frosted_stronger.png")
    print(FIG / "Fig3G_Reactome_frosted_stronger.png")
    print()
    print("SOURCE TABLES")
    print(TAB / "Supp_matched_CL_id_mapping.csv")
    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
