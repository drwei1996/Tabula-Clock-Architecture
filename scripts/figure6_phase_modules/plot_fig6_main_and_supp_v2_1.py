# -*- coding: utf-8 -*-
"""
Figure 6 main biology panels + supplementary robustness panels
Author: ChatGPT
Version: v2

目的：
1) 主图只画生物学发现（B/C/D/E）
2) robustness 放附图
3) 每个 panel 输出：
   - 带字版（_text）
   - 不带字版（_notext）
4) 尽量保持统一、简洁的论文图风格

运行：
python plot_fig6_main_and_supp_v2.py --project-root "<TABULA_PROJECT_ROOT>/katv2" --outdir FIG6_FINAL_PANELS_V2

依赖：
pandas, numpy, matplotlib
"""

import os
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# =========================
# 全局风格（你可以按你旧fig再微调）
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

BASE_FONTSIZE = 9
TITLE_FONTSIZE = 11
PANEL_FONTSIZE = 16
TICK_FONTSIZE = 8
LEGEND_FONTSIZE = 8

DARK = "#20334D"
GREY = "#7A8797"
LIGHT_GREY = "#D9E1EA"
GRID = "#E8EDF3"

# 你可以替换成你之前 fig 的固定配色
LINEAGE_COLORS = {
    "B cell": "#3B82C4",
    "T cell": "#E25555",
    "Monocyte/Macrophage": "#E79A2D",
    "Fibroblast": "#42B7A5",
    "Endothelial": "#9B69C1",
}

# 主图通路颜色
METABOLIC_COLORS = {
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "#1E97AE",
    "HALLMARK_MTORC1_SIGNALING": "#6C8EBF",
    "HALLMARK_GLYCOLYSIS": "#6AAE61",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": "#9A58C5",
}

INFLAMMATORY_COLORS = {
    "HALLMARK_INFLAMMATORY_RESPONSE": "#D34D4D",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": "#E07B24",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "#B04E8D",
}

# 主图通路选择
MAIN_METABOLIC = [
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
]

MAIN_INFLAMMATORY = [
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
]

# E panel 用的炎症 pathway（主图）
HEATMAP_INFLAMMATORY = [
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
]

LINEAGE_ORDER = ["B cell", "T cell", "Monocyte/Macrophage", "Fibroblast", "Endothelial"]


# =========================
# 工具函数
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def first_existing(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing columns. Candidates tried: {candidates}")
    return None


def nice_name(pathway):
    x = pathway.replace("HALLMARK_", "").replace("REACTOME_", "")
    x = x.replace("_", " ").title()
    x = x.replace("Tnfa", "TNFα").replace("Nfkb", "NFκB").replace("Il6", "IL6")
    x = x.replace("Jak", "JAK").replace("Stat3", "STAT3").replace("Mtorc1", "mTORC1")
    x = x.replace("Oxidative Phosphorylation", "Oxidative phosphorylation")
    x = x.replace("Unfolded Protein Response", "Unfolded protein response")
    x = x.replace("Inflammatory Response", "Inflammatory response")
    return x


def save_figure(fig, outbase):
    fig.savefig(outbase + ".png", dpi=450, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    fig.savefig(outbase + ".svg", bbox_inches="tight")
    plt.close(fig)


def strip_all_text(fig):
    # 去掉所有文本，生成“不带字版”
    if hasattr(fig, "_suptitle") and fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for txt in list(ax.texts):
            txt.set_visible(False)


def base_axis_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#98A6B5")
    ax.spines["bottom"].set_color("#98A6B5")
    ax.tick_params(labelsize=TICK_FONTSIZE, colors=DARK)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_panel(ax, letter, title=None, with_text=True):
    if with_text:
        ax.text(
            -0.18, 1.12, letter,
            transform=ax.transAxes,
            fontsize=PANEL_FONTSIZE,
            fontweight="bold",
            color=DARK,
            va="top",
            ha="left",
        )
        if title:
            ax.text(
                0.0, 1.10, title,
                transform=ax.transAxes,
                fontsize=TITLE_FONTSIZE,
                fontweight="bold",
                color=DARK,
                va="top",
                ha="left",
            )


def load_required_tables(project_root):
    paths = {
        "main_results": os.path.join(project_root, "TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1", "02_M012_MAIN_RESULTS.csv"),
        "fixed_state": os.path.join(project_root, "TABULA_FIG6_STRICT_VALIDATION_V2_1", "04_FIXED_NULL_AND_STATE_RESULTS.csv"),
        "phase_rotation": os.path.join(project_root, "TABULA_FIG6_STRICT_VALIDATION_V2_1", "04_PHASE_ROTATION_RESULTS.csv"),
        "descriptive": os.path.join(project_root, "TABULA_FIG6_IDENTITY_VS_CLOCK_HALLMARK_REACTOME_V1", "05_LINEAGE_PHASE_AND_AMPLITUDE_DESCRIPTIVE.csv"),
        "m2_conditional": os.path.join(project_root, "TABULA_FIG6_M2_CONDITIONAL_NULL_CV_V3", "02_CONDITIONAL_M2_RESULTS.csv"),
        "m2_cv": os.path.join(project_root, "TABULA_FIG6_M2_CONDITIONAL_NULL_CV_V3", "03_M2_CV_SUMMARY.csv"),
        "m2_common_support": os.path.join(project_root, "TABULA_FIG6_M2_COMMON_SUPPORT_V4", "03_COMMON_SUPPORT_M2_RESULTS.csv"),
        "gse_true_zt": os.path.join(project_root, "FIG6_GSE292219_TRUE_ZT_TARGETED_EXTERNAL_V1", "02_ALL_18_GSE_M012_REAL_ZT.csv"),
        "gse_true_zt_summary": os.path.join(project_root, "FIG6_GSE292219_TRUE_ZT_TARGETED_EXTERNAL_V1", "04_PRIMARY_BACKGROUND_ADJUSTED_CROSSMETHOD_SUMMARY.csv"),
    }

    out = {}
    for k, p in paths.items():
        if os.path.exists(p):
            out[k] = pd.read_csv(p)
        else:
            out[k] = None
            print(f"[WARN] file not found: {p}")
    return out


# =========================
# Panel B 主图
# 生物学发现为主，不强调robustness
# =========================
def plot_panel_B_main(main_results, outdir, with_text=True):
    df = main_results.copy()

    pathway_col = first_existing(df, ["pathway", "Pathway"])
    xcol = first_existing(df, ["deltaR2_M1_vs_M0", "deltaR2_M1", "M1_minus_M0_deltaR2"])
    ycol = first_existing(df, ["deltaR2_M2_vs_M1", "deltaR2_M2", "M2_minus_M1_deltaR2"])
    collection_col = first_existing(df, ["collection", "Collection"], required=False)

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    base_axis_style(ax)

    # 背景点
    ax.scatter(
        df[xcol], df[ycol],
        s=18, c="#D8E1EB", alpha=0.8,
        edgecolors="none", zorder=1
    )

    # 主图只高亮生物学重点
    highlight = MAIN_METABOLIC + MAIN_INFLAMMATORY
    hdf = df[df[pathway_col].isin(highlight)].copy()

    for _, r in hdf.iterrows():
        pw = r[pathway_col]
        if pw in METABOLIC_COLORS:
            c = METABOLIC_COLORS[pw]
        else:
            c = INFLAMMATORY_COLORS[pw]
        ax.scatter(r[xcol], r[ycol], s=72, color=c, edgecolor="white", linewidth=0.9, zorder=3)
        if with_text:
            dx = 0.01
            dy = 0.005
            ax.text(r[xcol] + dx, r[ycol] + dy, nice_name(pw), fontsize=7.5, color=DARK)

    ax.axvline(0, color=LIGHT_GREY, linewidth=1)
    ax.axhline(0, color=LIGHT_GREY, linewidth=1)

    xmax = max(0.40, float(df[xcol].max()) * 1.08)
    ymax = max(0.32, float(df[ycol].max()) * 1.10)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)

    if with_text:
        ax.set_xlabel("Shared association score (M1−M0 ΔR²)", fontsize=9, color=DARK)
        ax.set_ylabel("Lineage-modulated association score (M2−M1 ΔR²)", fontsize=9, color=DARK)

    label_panel(
        ax, "B",
        title="Clock-phase-associated functional programs" if with_text else None,
        with_text=with_text
    )

    if with_text:
        # 左上角加 very small biological annotation
        ax.text(
            0.02, 0.98,
            "Metabolic pathways cluster toward\nstrong shared associations;\n"
            "inflammatory pathways show stronger\nlineage-modulated associations.",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, color=GREY
        )

    outbase = os.path.join(outdir, f"FIG6_B_main_biology_{'text' if with_text else 'notext'}")
    save_figure(fig, outbase)


# =========================
# Panel C 主图
# shared metabolic programs
# =========================
def plot_panel_C_main(fixed_state, outdir, with_text=True):
    df = fixed_state.copy()

    pathway_col = first_existing(df, ["pathway", "Pathway"])
    mode_col = first_existing(df, ["mode", "Mode"])
    m1_col = first_existing(df, ["deltaR2_M1", "observed_M1_deltaR2", "M1_deltaR2"])

    # mode
    baseline_mode = "baseline_common_rows"
    adjusted_mode = "disjoint_RNA_proxies"

    sub = df[df[pathway_col].isin(MAIN_METABOLIC)].copy()

    pivot = sub.pivot_table(index=pathway_col, columns=mode_col, values=m1_col, aggfunc="first")
    for p in MAIN_METABOLIC:
        if p not in pivot.index:
            pivot.loc[p, :] = np.nan
    pivot = pivot.loc[MAIN_METABOLIC]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    base_axis_style(ax)

    y = np.arange(len(MAIN_METABOLIC))[::-1]
    labels = [nice_name(x) for x in MAIN_METABOLIC]

    for i, pw in enumerate(MAIN_METABOLIC):
        c = METABOLIC_COLORS[pw]
        yv = y[i]
        x0 = pivot.loc[pw, baseline_mode] if baseline_mode in pivot.columns else np.nan
        x1 = pivot.loc[pw, adjusted_mode] if adjusted_mode in pivot.columns else np.nan

        # 连线
        if np.isfinite(x0) and np.isfinite(x1):
            ax.plot([x0, x1], [yv, yv], color="#A9B8C8", linewidth=2.0, zorder=1)

        # 原始（空心）
        if np.isfinite(x0):
            ax.scatter(x0, yv, s=88, facecolors="white", edgecolors=c, linewidth=1.6, zorder=3)
        # 调整后（实心）
        if np.isfinite(x1):
            ax.scatter(x1, yv, s=88, facecolors=c, edgecolors="white", linewidth=0.8, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels if with_text else [])
    ax.set_xlim(0, max(0.40, np.nanmax(pivot.values) * 1.15))
    ax.set_ylim(-0.7, len(MAIN_METABOLIC) - 0.3)

    if with_text:
        ax.set_xlabel("Shared association score (M1 ΔR²)", fontsize=9, color=DARK)
        ax.set_ylabel("")
    else:
        ax.set_ylabel("")

    label_panel(
        ax, "C",
        title="Shared metabolic programs across lineages" if with_text else None,
        with_text=with_text
    )

    if with_text:
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
                       markeredgecolor=DARK, markersize=7, label="Original"),
            plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=DARK,
                       markeredgecolor="white", markersize=7, label="Adjusted"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")

    outbase = os.path.join(outdir, f"FIG6_C_main_metabolic_shared_{'text' if with_text else 'notext'}")
    save_figure(fig, outbase)


# =========================
# D 主图
# 直接画生物学上的lineage curves
# 用 descriptive amplitude + peak phase 重建
# =========================
def reconstruct_curve(amplitude, peak_h, x):
    # 简单余弦重建：y = A * cos(2π(x - peak)/24)
    return amplitude * np.cos(2 * np.pi * (x - peak_h) / 24.0)


def plot_panel_D_main(descriptive, outdir, with_text=True):
    if descriptive is None:
        print("[WARN] descriptive file missing, skip panel D.")
        return

    df = descriptive.copy()
    pathway_col = first_existing(df, ["pathway", "Pathway"])
    lineage_col = first_existing(df, ["lineage", "Lineage"])
    amp_col = first_existing(df, ["amplitude", "Amplitude"])
    peak_col = first_existing(df, ["descriptive_peak_phase_h", "peak_phase_h", "peak_phase"])

    fig = plt.figure(figsize=(6.2, 5.5))
    gs = GridSpec(3, 1, figure=fig, hspace=0.45)

    x = np.linspace(0, 24, 241)

    for i, pw in enumerate(MAIN_INFLAMMATORY):
        ax = fig.add_subplot(gs[i, 0])
        base_axis_style(ax)

        sub = df[df[pathway_col] == pw].copy()

        for lin in LINEAGE_ORDER:
            ss = sub[sub[lineage_col] == lin]
            if ss.shape[0] == 0:
                continue
            amp = float(ss.iloc[0][amp_col])
            peak = float(ss.iloc[0][peak_col])
            y = reconstruct_curve(amp, peak, x)
            c = LINEAGE_COLORS.get(lin, "#999999")
            ax.plot(x, y, color=c, linewidth=2.0, label=lin)
            # 淡淡阴影，只做视觉统一
            ax.fill_between(x, y - 0.10 * max(abs(amp), 1e-3), y + 0.10 * max(abs(amp), 1e-3),
                            color=c, alpha=0.10, linewidth=0)

        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.axhline(0, color=LIGHT_GREY, linewidth=1.0)

        if with_text:
            ax.set_ylabel("Relative score", fontsize=8, color=DARK)
            ax.set_title(nice_name(pw), fontsize=10, color=DARK, pad=4, fontweight="bold")
            if i == 2:
                ax.set_xlabel("Inferred clock phase (hours)", fontsize=9, color=DARK)
            else:
                ax.set_xlabel("")
        else:
            ax.set_ylabel("")
            ax.set_xlabel("")

        if i == 0:
            label_panel(
                ax, "D",
                title="Lineage-modulated inflammatory programs" if with_text else None,
                with_text=with_text
            )

        if with_text and i == 2:
            leg = ax.legend(
                frameon=False, ncol=3, fontsize=8, loc="upper center",
                bbox_to_anchor=(0.5, -0.30), columnspacing=1.0, handlelength=1.5
            )

    outbase = os.path.join(outdir, f"FIG6_D_main_inflammatory_lineage_curves_{'text' if with_text else 'notext'}")
    save_figure(fig, outbase)


# =========================
# E 主图
# 生物学总结：amplitude / peak-phase heatmap
# =========================
def normalize_rows(mat):
    out = mat.copy().astype(float)
    for i in range(out.shape[0]):
        vals = out[i, :]
        if np.all(np.isnan(vals)):
            continue
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if vmax > vmin:
            out[i, :] = (vals - vmin) / (vmax - vmin)
        else:
            out[i, :] = 0.5
    return out


def plot_panel_E_main(descriptive, outdir, with_text=True):
    if descriptive is None:
        print("[WARN] descriptive file missing, skip panel E.")
        return

    df = descriptive.copy()
    pathway_col = first_existing(df, ["pathway", "Pathway"])
    lineage_col = first_existing(df, ["lineage", "Lineage"])
    amp_col = first_existing(df, ["amplitude", "Amplitude"])
    peak_col = first_existing(df, ["descriptive_peak_phase_h", "peak_phase_h", "peak_phase"])

    pathways = HEATMAP_INFLAMMATORY
    amps = np.full((len(pathways), len(LINEAGE_ORDER)), np.nan)
    peaks = np.full((len(pathways), len(LINEAGE_ORDER)), np.nan)

    for i, pw in enumerate(pathways):
        for j, lin in enumerate(LINEAGE_ORDER):
            ss = df[(df[pathway_col] == pw) & (df[lineage_col] == lin)]
            if ss.shape[0] == 0:
                continue
            amps[i, j] = float(ss.iloc[0][amp_col])
            peaks[i, j] = float(ss.iloc[0][peak_col])

    amp_show = normalize_rows(np.abs(amps))
    peak_show = peaks.copy()

    fig = plt.figure(figsize=(6.4, 3.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.35)

    # amplitude heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(amp_show, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(LINEAGE_ORDER)))
    ax1.set_yticks(np.arange(len(pathways)))
    ax1.set_xticklabels(LINEAGE_ORDER if with_text else [], rotation=35, ha="right", fontsize=8)
    ax1.set_yticklabels([nice_name(x) for x in pathways] if with_text else [], fontsize=8)
    ax1.tick_params(length=0)
    for s in ax1.spines.values():
        s.set_color("#98A6B5")
    if with_text:
        ax1.set_title("Relative amplitude across lineages", fontsize=10, color=DARK, pad=6)
    label_panel(ax1, "E", title="Biological summary of inflammatory heterogeneity" if with_text else None, with_text=with_text)

    # peak heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(peak_show, aspect="auto", cmap="viridis", vmin=0, vmax=24)
    ax2.set_xticks(np.arange(len(LINEAGE_ORDER)))
    ax2.set_yticks(np.arange(len(pathways)))
    ax2.set_xticklabels(LINEAGE_ORDER if with_text else [], rotation=35, ha="right", fontsize=8)
    ax2.set_yticklabels([] if with_text else [])
    ax2.tick_params(length=0)
    for s in ax2.spines.values():
        s.set_color("#98A6B5")
    if with_text:
        ax2.set_title("Descriptive peak phase (hours)", fontsize=10, color=DARK, pad=6)

    # colorbars
    if with_text:
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=7)
        cbar1.set_label("Scaled amplitude", fontsize=8, color=DARK)
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=7)
        cbar2.set_label("Phase (h)", fontsize=8, color=DARK)

    outbase = os.path.join(outdir, f"FIG6_E_main_inflammatory_heatmap_{'text' if with_text else 'notext'}")
    save_figure(fig, outbase)


# =========================
# Supplementary robustness
# =========================
def plot_supp_fixed_state_and_rotation(fixed_state, phase_rotation, outdir, with_text=True):
    if fixed_state is None or phase_rotation is None:
        print("[WARN] missing files for supplementary robustness panel S1.")
        return

    fs = fixed_state.copy()
    pr = phase_rotation.copy()

    pcol_fs = first_existing(fs, ["pathway", "Pathway"])
    mode_col = first_existing(fs, ["mode", "Mode"])
    m1_col = first_existing(fs, ["deltaR2_M1", "observed_M1_deltaR2", "M1_deltaR2"], required=False)
    m2_col = first_existing(fs, ["deltaR2_M2", "observed_M2_deltaR2", "M2_deltaR2", "observed_M2_deltaR2"], required=False)

    pcol_pr = first_existing(pr, ["pathway", "Pathway"])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax1, ax2 = axes

    # S1A: metabolic M1 before/after adjustment
    base_axis_style(ax1)
    sub = fs[fs[pcol_fs].isin(MAIN_METABOLIC)].copy()
    pivot = sub.pivot_table(index=pcol_fs, columns=mode_col, values=m1_col, aggfunc="first")
    y = np.arange(len(MAIN_METABOLIC))[::-1]
    for i, pw in enumerate(MAIN_METABOLIC):
        c = METABOLIC_COLORS[pw]
        x0 = pivot.loc[pw, "baseline_common_rows"] if ("baseline_common_rows" in pivot.columns and pw in pivot.index) else np.nan
        x1 = pivot.loc[pw, "disjoint_RNA_proxies"] if ("disjoint_RNA_proxies" in pivot.columns and pw in pivot.index) else np.nan
        yy = y[i]
        if np.isfinite(x0) and np.isfinite(x1):
            ax1.plot([x0, x1], [yy, yy], color="#B7C4D1", linewidth=2)
        if np.isfinite(x0):
            ax1.scatter(x0, yy, s=70, facecolors="white", edgecolors=c, linewidth=1.5)
        if np.isfinite(x1):
            ax1.scatter(x1, yy, s=70, facecolors=c, edgecolors="white", linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([nice_name(x) for x in MAIN_METABOLIC] if with_text else [])
    if with_text:
        ax1.set_xlabel("M1 ΔR²", fontsize=9)
        ax1.set_title("S1A  Metabolic M1: original vs adjusted", fontsize=10, color=DARK)

    # S1B: inflammatory M2 before/after adjustment
    base_axis_style(ax2)
    if m2_col is not None:
        sub2 = fs[fs[pcol_fs].isin(MAIN_INFLAMMATORY)].copy()
        pivot2 = sub2.pivot_table(index=pcol_fs, columns=mode_col, values=m2_col, aggfunc="first")
        y2 = np.arange(len(MAIN_INFLAMMATORY))[::-1]
        for i, pw in enumerate(MAIN_INFLAMMATORY):
            c = INFLAMMATORY_COLORS[pw]
            x0 = pivot2.loc[pw, "baseline_common_rows"] if ("baseline_common_rows" in pivot2.columns and pw in pivot2.index) else np.nan
            x1 = pivot2.loc[pw, "disjoint_RNA_proxies"] if ("disjoint_RNA_proxies" in pivot2.columns and pw in pivot2.index) else np.nan
            yy = y2[i]
            if np.isfinite(x0) and np.isfinite(x1):
                ax2.plot([x0, x1], [yy, yy], color="#B7C4D1", linewidth=2)
            if np.isfinite(x0):
                ax2.scatter(x0, yy, s=70, facecolors="white", edgecolors=c, linewidth=1.5)
            if np.isfinite(x1):
                ax2.scatter(x1, yy, s=70, facecolors=c, edgecolors="white", linewidth=0.8)
        ax2.set_yticks(y2)
        ax2.set_yticklabels([nice_name(x) for x in MAIN_INFLAMMATORY] if with_text else [])
        if with_text:
            ax2.set_xlabel("M2-related score", fontsize=9)
            ax2.set_title("S1B  Inflammatory programs: original vs adjusted", fontsize=10, color=DARK)

    outbase = os.path.join(outdir, f"FIG6_SUPP_S1_fixed_state_rotation_{'text' if with_text else 'notext'}")
    if not with_text:
        strip_all_text(fig)
    save_figure(fig, outbase)


def plot_supp_m2_conditional_and_support(m2_conditional, m2_common_support, outdir, with_text=True):
    """V2.1: always use adjusted mode, distinguish no support from negative results."""
    if m2_conditional is None or m2_common_support is None:
        print("[WARN] missing files for supplementary robustness panel S2.")
        return

    def adjusted_three(df, name):
        required = {'pathway', 'mode'}
        if not required.issubset(df.columns):
            raise ValueError(f"{name}: missing columns: {sorted(required-set(df.columns))}")
        # Both V3 and V4 have exactly two rows per pathway: baseline, adjusted.
        # Do NOT pivot or reindex until model mode has been fixed.
        mode = 'adjusted' if 'adjusted' in set(df['mode'].astype(str)) else 'disjoint_RNA_proxies'
        t = df.loc[df['mode'].astype(str).eq(mode) & df['pathway'].isin(MAIN_INFLAMMATORY)].copy()
        if t['pathway'].duplicated().any():
            dups = t.loc[t['pathway'].duplicated(False), 'pathway'].tolist()
            raise ValueError(f"{name}: duplicate pathway after selecting mode={mode}: {dups}")
        missing = sorted(set(MAIN_INFLAMMATORY)-set(t['pathway']))
        if missing:
            raise ValueError(f"{name}: missing adjusted pathways: {missing}")
        return t.set_index('pathway').loc[MAIN_INFLAMMATORY].reset_index()

    cdf = adjusted_three(m2_conditional, 'V3 conditional M2')
    sdf = adjusted_three(m2_common_support, 'V4 common support')
    if 'observed_M2_deltaR2' not in cdf.columns:
        raise KeyError('V3: missing observed_M2_deltaR2')
    if not {'status', 'empirical_rank_p', 'n_joint_supported'}.issubset(sdf.columns):
        raise KeyError('V4: missing status/empirical_rank_p/n_joint_supported')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={'width_ratios': [1, 1.30]})
    for ax in (ax1, ax2):
        base_axis_style(ax)
    yy = np.arange(len(MAIN_INFLAMMATORY))[::-1]

    for i, pw in enumerate(MAIN_INFLAMMATORY):
        row = cdf.loc[cdf['pathway'].eq(pw)].iloc[0]
        val = float(row['observed_M2_deltaR2'])
        ax1.barh(yy[i], val, color=INFLAMMATORY_COLORS.get(pw, '#888888'),
                 height=0.53, alpha=0.83, zorder=3)
    ax1.set_yticks(yy)
    short_labels = {'HALLMARK_INFLAMMATORY_RESPONSE': 'Inflammatory response',
                    'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNFα / NFκB',
                    'HALLMARK_IL6_JAK_STAT3_SIGNALING': 'IL6–JAK–STAT3'}
    ax1.set_yticklabels([short_labels.get(p, nice_name(p)) for p in MAIN_INFLAMMATORY] if with_text else [])
    ax1.set_ylim(-0.7, len(yy)-0.3)
    ax1.set_xlim(0, max(0.30, float(np.nanmax(cdf['observed_M2_deltaR2'])) * 1.18))
    if with_text:
        ax1.set_title('S2A  Adjusted M2−M1 association', fontsize=10, color=DARK)
        ax1.set_xlabel('Incremental within-block ΔR²', fontsize=9)

    # Only show empirical p-values for SUPPORTED_EXPLORATORY rows.
    # NO_COMMON_SUPPORT means untestable, NOT non-significant and NOT p=1.
    ax2.set_xlim(0, 1.04)
    ax2.set_ylim(-0.7, len(yy)-0.3)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([])
    ax2.axvline(0.05, linewidth=1.2, linestyle='--', color='#B5B5B5', zorder=1)
    n_supported = 0
    for i, pw in enumerate(MAIN_INFLAMMATORY):
        r = sdf.loc[sdf['pathway'].eq(pw)].iloc[0]
        status = str(r['status'])
        color = INFLAMMATORY_COLORS.get(pw, '#888888')
        if status == 'SUPPORTED_EXPLORATORY':
            p = pd.to_numeric(pd.Series([r['empirical_rank_p']]), errors='coerce').iloc[0]
            if not np.isfinite(p):
                raise ValueError(f'{pw}: SUPPORTED but empirical_rank_p is missing')
            ax2.scatter([float(p)], [yy[i]], color=color, s=68,
                        edgecolor='white', linewidth=0.8, zorder=4)
            n_supported += 1
            if with_text:
                ax2.text(min(float(p)+0.04, 0.96), yy[i], f'P = {float(p):.3f}',
                         color=DARK, fontsize=8, va='center')
        elif status == 'NO_COMMON_SUPPORT':
            ax2.hlines(yy[i], 0.20, 0.78, color='#D2D8DD', lw=1.5, linestyles='dotted', zorder=2)
            if with_text:
                count = pd.to_numeric(pd.Series([r['n_joint_supported']]), errors='coerce').iloc[0]
                count = int(count) if np.isfinite(count) else 0
                ax2.text(0.48, yy[i], f'NOT CALIBRATED  (joint matched: {count})',
                         fontsize=7.8, va='center', ha='center', color=GREY,
                         bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
        else:
            raise ValueError(f'{pw}: unexpected V4 status {status!r}')
    if with_text:
        ax2.set_title('S2B  V4 matched-null common support', fontsize=10, color=DARK)
        ax2.set_xlabel('Exploratory empirical P (supported pathways only)', fontsize=9)
    print(f'[S2] mode=adjusted, pathways={len(MAIN_INFLAMMATORY)}, '
          f'common-support={n_supported}, no-support={len(MAIN_INFLAMMATORY)-n_supported}')
    fig.subplots_adjust(left=0.20, right=0.97, wspace=0.18, bottom=0.20, top=0.82)
    if not with_text:
        strip_all_text(fig)
    outbase = os.path.join(outdir, f"FIG6_SUPP_S2_conditional_support_{'text' if with_text else 'notext'}")
    save_figure(fig, outbase)



def plot_supp_external_validation(gse_true_zt_summary, outdir, with_text=True):
    """V2.1: actual GSE292219 summary columns, matched source vs strict-10k.
    Descriptive sensitivity ONLY: one biological sample per ZT, no replication.
    """
    if gse_true_zt_summary is None:
        print('[WARN] missing GSE true-ZT summary, skip supplementary external panel.')
        return
    df = gse_true_zt_summary.copy()
    required = {'pathway', 'deltaR2_M1_M0', 'deltaR2_M2_M1',
                'strict10k_M1_deltaR2', 'strict10k_M2_deltaR2'}
    absent = sorted(required-set(df.columns))
    if absent:
        print(f'[WARN] GSE summary missing columns, skip S3: {absent}')
        return
    if 'adjustment' in df.columns:
        df = df.loc[df['adjustment'].eq('global_background_subtracted')].copy()
    if 'method' in df.columns:
        df = df.loc[df['method'].eq('source')].copy()
    if df['pathway'].duplicated().any():
        raise ValueError('GSE summary contains duplicate pathway after filtering method/adjustment')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.7, 3.9))
    for ax in (ax1, ax2):
        base_axis_style(ax)
        ax.set_xticks([0, 1])
        ax.set_xlim(-0.17, 1.17)
        ax.set_xticklabels(['Source', 'Strict-10k'] if with_text else [])
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=0.75)
    for ax, pathways, col0, col1, cmap, label, header in (
        (ax1, MAIN_METABOLIC, 'deltaR2_M1_M0', 'strict10k_M1_deltaR2',
         METABOLIC_COLORS, 'Shared association ΔR²', 'S3A  Metabolic programs'),
        (ax2, MAIN_INFLAMMATORY, 'deltaR2_M2_M1', 'strict10k_M2_deltaR2',
         INFLAMMATORY_COLORS, 'Lineage interaction ΔR²', 'S3B  Inflammatory programs'),
    ):
        n = 0
        for pw in pathways:
            sel = df.loc[df['pathway'].eq(pw)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            ys = [pd.to_numeric(pd.Series([r[col0]]), errors='coerce').iloc[0],
                  pd.to_numeric(pd.Series([r[col1]]), errors='coerce').iloc[0]]
            if not np.all(np.isfinite(ys)):
                continue
            c = cmap[pw]
            ax.plot([0, 1], ys, color=c, linewidth=1.5, alpha=.85,
                    marker='o', ms=5.3, label=nice_name(pw))
            n += 1
        if with_text:
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(header, fontsize=10, color=DARK)
            if n:
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -0.20),
                          ncol=2, frameon=False, fontsize=7)
    if with_text:
        fig.suptitle('GSE292219 true-ZT descriptive sensitivity (no time-point replicates)',
                     fontsize=10, color=DARK)
    else:
        strip_all_text(fig)
    outbase = os.path.join(outdir, f"FIG6_SUPP_S3_external_trueZT_{'text' if with_text else 'notext'}")
    save_figure(fig, outbase)



# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, help="katv2 project root")
    parser.add_argument("--outdir", default="FIG6_FINAL_PANELS_V2", help="output directory")
    args = parser.parse_args()

    outdir = os.path.join(args.project_root, args.outdir)
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "supplementary"))

    tabs = load_required_tables(args.project_root)

    # ---------------------
    # Main figure
    # ---------------------
    if tabs["main_results"] is not None:
        plot_panel_B_main(tabs["main_results"], outdir, with_text=True)
        plot_panel_B_main(tabs["main_results"], outdir, with_text=False)

    if tabs["fixed_state"] is not None:
        plot_panel_C_main(tabs["fixed_state"], outdir, with_text=True)
        plot_panel_C_main(tabs["fixed_state"], outdir, with_text=False)

    if tabs["descriptive"] is not None:
        plot_panel_D_main(tabs["descriptive"], outdir, with_text=True)
        plot_panel_D_main(tabs["descriptive"], outdir, with_text=False)
        plot_panel_E_main(tabs["descriptive"], outdir, with_text=True)
        plot_panel_E_main(tabs["descriptive"], outdir, with_text=False)

    # ---------------------
    # Supplementary robustness
    # ---------------------
    supp_dir = os.path.join(outdir, "supplementary")

    plot_supp_fixed_state_and_rotation(
        tabs["fixed_state"], tabs["phase_rotation"], supp_dir, with_text=True
    )
    plot_supp_fixed_state_and_rotation(
        tabs["fixed_state"], tabs["phase_rotation"], supp_dir, with_text=False
    )

    plot_supp_m2_conditional_and_support(
        tabs["m2_conditional"], tabs["m2_common_support"], supp_dir, with_text=True
    )
    plot_supp_m2_conditional_and_support(
        tabs["m2_conditional"], tabs["m2_common_support"], supp_dir, with_text=False
    )

    plot_supp_external_validation(
        tabs["gse_true_zt_summary"], supp_dir, with_text=True
    )
    plot_supp_external_validation(
        tabs["gse_true_zt_summary"], supp_dir, with_text=False
    )

    print("[DONE] Figure 6 main biology panels and supplementary robustness panels generated.")
    print(f"[OUTDIR] {outdir}")


if __name__ == "__main__":
    main()