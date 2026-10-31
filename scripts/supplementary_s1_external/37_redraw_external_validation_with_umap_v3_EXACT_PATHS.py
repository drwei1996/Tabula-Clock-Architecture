# -*- coding: utf-8 -*-
"""
37_redraw_external_validation_with_umap_v3_EXACT_PATHS.py

Uses ONLY exact files verified by the local structure check.
No recursive guessing. No H5AD required.

Inputs (verified):
- FINAL_SUMMARY.csv
- 01_HCL/source_data/pseudobulk_standardized.csv
- 02_AHCA_15ORG/source_data/pseudobulk_standardized.csv
- 03_CROSS_DATASET/lineage_remodeling_cross_dataset.csv

Main panels:
A HCL pseudobulk UMAP
B AHCA pseudobulk UMAP
C Major-lineage backbone (within vs between)
D Lineage-remodeling rank trajectory across Tabula/HCL/AHCA
E Tabula vs HCL rank concordance
F Tabula vs AHCA rank concordance

Important:
- Output should be a SHORT Windows path to avoid the ~260-char path problem.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    from sklearn.decomposition import PCA


LINEAGE_ORDER = [
    "b_cell",
    "t_cell",
    "macrophage_monocyte",
    "fibroblast",
    "endothelial",
]

LINEAGE_DISPLAY = {
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

DATASET_ORDER = ["Tabula", "HCL", "AHCA"]


def choose_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for x in ["Arial", "Liberation Sans", "DejaVu Sans"]:
        if x in names:
            return x
    return "sans-serif"


FONT = choose_font()
mpl.rcParams.update({
    "font.family": FONT,
    "font.size": 8,
    "axes.titlesize": 10.5,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,
    "legend.fontsize": 6.7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def clean_axis(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in [".png", ".pdf", ".svg"]:
        kwargs = dict(bbox_inches="tight", pad_inches=0.06, facecolor="white")
        if ext == ".png":
            kwargs["dpi"] = 600
        fig.savefig(outdir / f"{stem}{ext}", **kwargs)
    plt.close(fig)


def resolve_root(root: Path) -> Path:
    root = root.resolve()
    if (root / "clock_external_validation").is_dir() and not (root / "01_HCL").is_dir():
        root = root / "clock_external_validation"

    required = [
        root / "FINAL_SUMMARY.csv",
        root / "01_HCL" / "source_data" / "pseudobulk_standardized.csv",
        root / "02_AHCA_15ORG" / "source_data" / "pseudobulk_standardized.csv",
        root / "03_CROSS_DATASET" / "lineage_remodeling_cross_dataset.csv",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(
            "Verified input file(s) missing:\n" + "\n".join(missing)
        )
    return root


def load_inputs(root: Path):
    summary_path = root / "FINAL_SUMMARY.csv"
    hcl_path = root / "01_HCL" / "source_data" / "pseudobulk_standardized.csv"
    ahca_path = root / "02_AHCA_15ORG" / "source_data" / "pseudobulk_standardized.csv"
    cross_path = root / "03_CROSS_DATASET" / "lineage_remodeling_cross_dataset.csv"

    summary = pd.read_csv(summary_path)
    summary = summary[summary["dataset"].isin(["HCL", "AHCA"])].copy()
    summary["dataset"] = pd.Categorical(
        summary["dataset"], categories=["HCL", "AHCA"], ordered=True
    )
    summary = summary.sort_values("dataset").reset_index(drop=True)

    hcl = pd.read_csv(hcl_path, low_memory=False)
    ahca = pd.read_csv(ahca_path, low_memory=False)

    for name, df, expected_n in [("HCL", hcl, 237), ("AHCA", ahca, 141)]:
        feats = [c for c in df.columns if c.startswith("expr_")]
        if len(df) != expected_n:
            raise RuntimeError(f"{name}: n={len(df)}, expected {expected_n}")
        if len(feats) != 17:
            raise RuntimeError(f"{name}: found {len(feats)} expr_ features, expected 17")
        if "major_lineage" not in df.columns:
            raise RuntimeError(f"{name}: missing major_lineage")
        unexpected = sorted(set(df["major_lineage"].dropna()) - set(LINEAGE_ORDER))
        if unexpected:
            raise RuntimeError(f"{name}: unexpected major_lineage values: {unexpected}")

    # This is the verified wide table:
    # major_lineage, display, Tabula, HCL, AHCA
    cross = pd.read_csv(cross_path)
    expected_cols = {"major_lineage", "Tabula", "HCL", "AHCA"}
    if not expected_cols.issubset(cross.columns):
        raise RuntimeError(
            f"Unexpected cross-dataset table columns: {list(cross.columns)}"
        )
    cross = cross[cross["major_lineage"].isin(LINEAGE_ORDER)].copy()

    rows = []
    for _, r in cross.iterrows():
        for ds in DATASET_ORDER:
            rows.append({
                "dataset": ds,
                "major_lineage": r["major_lineage"],
                "distance": float(r[ds]),
            })
    dist = pd.DataFrame(rows)

    return summary, hcl, ahca, dist, {
        "summary": summary_path,
        "HCL": hcl_path,
        "AHCA": ahca_path,
        "lineage_remodeling": cross_path,
    }


def compute_umap(df: pd.DataFrame):
    feats = [c for c in df.columns if c.startswith("expr_")]
    X = df[feats].to_numpy(float)

    if HAS_UMAP:
        model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.35,
            metric="euclidean",
            random_state=42,
        )
        xy = model.fit_transform(X)
        method = "UMAP"
    else:
        model = PCA(n_components=2, random_state=42)
        xy = model.fit_transform(X)
        method = "PCA"

    out = df.copy()
    out["dim1"] = xy[:, 0]
    out["dim2"] = xy[:, 1]
    return out, method


def add_ranks(dist: pd.DataFrame):
    d = dist.copy()
    d["rank"] = np.nan
    for ds, idx in d.groupby("dataset", sort=False).groups.items():
        d.loc[idx, "rank"] = (
            d.loc[idx, "distance"]
            .rank(method="average", ascending=True)
            .to_numpy()
        )
    return d


def exact_perm_p(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rho = float(spearmanr(x, y).statistic)
    vals = []
    for yp in itertools.permutations(y.tolist()):
        vals.append(float(spearmanr(x, np.asarray(yp)).statistic))
    vals = np.asarray(vals)
    p = float(np.mean(np.abs(vals) >= abs(rho) - 1e-15))
    return rho, p


def add_letter(ax, letter):
    ax.text(
        -0.14, 1.05, letter,
        transform=ax.transAxes,
        fontsize=12.5, fontweight="bold",
        va="top", ha="left",
    )


def plot_umap(ax, d, dataset, method, letter):
    add_letter(ax, letter)
    for lin in LINEAGE_ORDER:
        p = d[d["major_lineage"].eq(lin)]
        ax.scatter(
            p["dim1"], p["dim2"],
            s=19, alpha=0.82,
            color=LINEAGE_COLORS[lin],
            edgecolor="none",
            label=LINEAGE_DISPLAY[lin],
        )
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"{dataset} pseudobulk")
    clean_axis(ax)


def plot_dumbbell(ax, summary, letter):
    add_letter(ax, letter)
    x = np.arange(len(summary))
    for i, r in summary.iterrows():
        wi = float(r["within_lineage_median"])
        be = float(r["between_lineage_median"])
        ax.plot([i, i], [wi, be], color="#B8B8B8", lw=1.6, zorder=1)
        ax.scatter(
            i, wi, s=58,
            facecolor="white", edgecolor="#555555",
            linewidth=1.2, zorder=3,
        )
        ax.scatter(
            i, be, s=60,
            facecolor="#555555", edgecolor="#555555",
            linewidth=0.8, zorder=3,
        )
        ax.text(
            i + 0.07, (wi + be) / 2,
            rf"$\Delta$ = {be-wi:.2f}",
            va="center", fontsize=7.1, color="#444444",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"].astype(str))
    ax.set_ylabel("Median Euclidean distance")
    ax.set_title("Major-lineage backbone")
    clean_axis(ax)


def plot_rank_trajectory(ax, ranked, letter):
    add_letter(ax, letter)
    xmap = {"Tabula": 0, "HCL": 1, "AHCA": 2}
    for lin in LINEAGE_ORDER:
        p = ranked[ranked["major_lineage"].eq(lin)].copy()
        p["x"] = p["dataset"].map(xmap)
        p = p.sort_values("x")
        ax.plot(
            p["x"], p["rank"],
            color=LINEAGE_COLORS[lin],
            lw=1.35, alpha=0.9,
        )
        ax.scatter(
            p["x"], p["rank"],
            s=34, color=LINEAGE_COLORS[lin],
            edgecolor="white", linewidth=0.5,
            zorder=3,
        )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.6, 5.4)
    ax.set_ylabel("Tissue-remodeling rank\n(1 = lowest, 5 = highest)")
    ax.set_title("Lineage remodeling hierarchy")
    clean_axis(ax)


def plot_rank_scatter(ax, ranked, external, letter):
    add_letter(ax, letter)
    tab = ranked[ranked["dataset"].eq("Tabula")].set_index("major_lineage")
    ext = ranked[ranked["dataset"].eq(external)].set_index("major_lineage")

    x, y = [], []
    for lin in LINEAGE_ORDER:
        xv = float(tab.loc[lin, "rank"])
        yv = float(ext.loc[lin, "rank"])
        x.append(xv)
        y.append(yv)
        ax.scatter(
            xv, yv, s=48,
            color=LINEAGE_COLORS[lin],
            edgecolor="white", linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            LINEAGE_DISPLAY[lin], (xv, yv),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6.5,
        )

    rho, p = exact_perm_p(x, y)
    ax.plot([0.7, 5.3], [0.7, 5.3], ls="--", lw=0.8, color="#B5B5B5")
    ax.set_xlim(0.65, 5.35)
    ax.set_ylim(0.65, 5.35)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Tabula remodeling rank")
    ax.set_ylabel(f"{external} remodeling rank")
    ax.set_title(f"Tabula vs {external}")
    ax.text(
        0.04, 0.96,
        f"Spearman ρ = {rho:.2f}\nExact P = {p:.4f}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7.0,
    )
    ax.set_aspect("equal", adjustable="box")
    clean_axis(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--outdir",
        required=True,
        help="Use a SHORT path on Windows, e.g. .\\external_validation_figures_v3"
    )
    args = ap.parse_args()

    root = resolve_root(Path(args.root))
    out = Path(args.outdir).resolve()
    figs = out / "figures"
    src = out / "source_data"
    figs.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("EXTERNAL VALIDATION REDRAW V3 — EXACT PATHS")
    print("=" * 100)
    print("ROOT:", root)
    print("OUT :", out)
    print("FONT:", FONT)
    print("UMAP:", HAS_UMAP)

    summary, hcl, ahca, dist, sources = load_inputs(root)

    print("[SOURCE] FINAL_SUMMARY:", sources["summary"])
    print("[SOURCE] HCL PB      :", sources["HCL"], "| n=237 | p=17")
    print("[SOURCE] AHCA PB     :", sources["AHCA"], "| n=141 | p=17")
    print("[SOURCE] remodeling  :", sources["lineage_remodeling"])

    ranked = add_ranks(dist)
    hcl_emb, hcl_method = compute_umap(hcl)
    ahca_emb, ahca_method = compute_umap(ahca)

    # exact plot source data
    summary.to_csv(src / "panel_C_dataset_summary.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(src / "panels_D_E_F_lineage_distances.csv", index=False, encoding="utf-8-sig")
    ranked.to_csv(src / "panels_D_E_F_lineage_ranks.csv", index=False, encoding="utf-8-sig")

    for name, emb in [("HCL", hcl_emb), ("AHCA", ahca_emb)]:
        keep = [
            c for c in ["donor_id", "tissue", "cell_type", "major_lineage", "n_cells"]
            if c in emb.columns
        ]
        emb[keep + ["dim1", "dim2"]].to_csv(
            src / f"panel_{name}_embedding.csv",
            index=False, encoding="utf-8-sig",
        )

    # Composite
    fig = plt.figure(figsize=(8.8, 10.7))
    gs = fig.add_gridspec(3, 2, wspace=0.38, hspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]

    plot_umap(axes[0], hcl_emb, "HCL", hcl_method, "A")
    plot_umap(axes[1], ahca_emb, "AHCA", ahca_method, "B")
    plot_dumbbell(axes[2], summary, "C")
    plot_rank_trajectory(axes[3], ranked, "D")
    plot_rank_scatter(axes[4], ranked, "HCL", "E")
    plot_rank_scatter(axes[5], ranked, "AHCA", "F")

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=LINEAGE_COLORS[l],
            markeredgecolor="none",
            markersize=5.5,
            label=LINEAGE_DISPLAY[l],
        )
        for l in LINEAGE_ORDER
    ]
    fig.legend(
        handles=handles,
        ncol=5, frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.08, right=0.92)
    save_fig(fig, figs, "EXT_VALIDATION_MAIN_WITH_UMAP_2x3_V3")

    # Separate panels
    panels = [
        ("Panel_A_HCL_UMAP", lambda ax: plot_umap(ax, hcl_emb, "HCL", hcl_method, "A")),
        ("Panel_B_AHCA_UMAP", lambda ax: plot_umap(ax, ahca_emb, "AHCA", ahca_method, "B")),
        ("Panel_C_major_lineage_backbone", lambda ax: plot_dumbbell(ax, summary, "C")),
        ("Panel_D_rank_trajectory", lambda ax: plot_rank_trajectory(ax, ranked, "D")),
        ("Panel_E_Tabula_vs_HCL_rank", lambda ax: plot_rank_scatter(ax, ranked, "HCL", "E")),
        ("Panel_F_Tabula_vs_AHCA_rank", lambda ax: plot_rank_scatter(ax, ranked, "AHCA", "F")),
    ]

    for stem, fn in panels:
        fig, ax = plt.subplots(figsize=(4.5, 3.9))
        fn(ax)
        fig.tight_layout()
        save_fig(fig, figs, stem)

    (out / "RUN_RECORD.json").write_text(
        json.dumps({
            "root": str(root),
            "outdir": str(out),
            "sources": {k: str(v) for k, v in sources.items()},
            "HCL_n": 237,
            "AHCA_n": 141,
            "features": 17,
            "HCL_embedding_method": hcl_method,
            "AHCA_embedding_method": ahca_method,
            "font": FONT,
            "note": "No H5AD used. Exact verified source paths only.",
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DONE")
    print("OPEN FIRST:", figs / "EXT_VALIDATION_MAIN_WITH_UMAP_2x3_V3.png")
    print("=" * 100)


if __name__ == "__main__":
    main()
