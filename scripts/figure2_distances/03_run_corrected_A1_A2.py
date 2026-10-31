from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import ast
import hashlib
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import t as student_t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from threadpoolctl import threadpool_limits


# ============================================================================
# Paths
# ============================================================================

PROJECT = Path(os.environ.get("TABULA_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
CORR = PROJECT / "corrected_full_analysis_v2"
FREEZE = CORR / "00_freeze"
OUT_A1 = CORR / "01_A1"
OUT_A2 = CORR / "02_A2"
COMPARE = CORR / "comparison_legacy_vs_corrected"
REPORT = CORR / "reports"

LEGACY = PROJECT / "results" / "phaseA_clock_architecture"
LEGACY_DEP = LEGACY / "dependencies"

V1 = PROJECT.parent / "TabulaCircadian_deliverables"
V2 = PROJECT.parent / "TabulaCircadian_deliverables_v2"

FROZEN_META = FREEZE / "observation_metadata_corrected_v3_FROZEN.tsv"
NPZ_PATH = LEGACY_DEP / "broad_normalized_mean.npz"
FEATURE_PATH = V1 / "source_data" / "pseudobulk__clock_architecture_features.tsv"
RUNNER = V2 / "scripts" / "revision_v2_runner.py"
STANDARDIZED = (
    PROJECT
    / "figures_revised"
    / "clock_architecture_umap"
    / "source_data"
    / "clock_architecture_UMAP_clock_architecture_standardized_input.tsv"
)

OLD_A1 = LEGACY / "01_matched_random_controls"
OLD_A1_UNATTENDED = OLD_A1 / "unattended"
OLD_A2 = PROJECT / "figures_revised" / "clock_centroid_distances"

SEED = 20260913
BOOTSTRAP_SEED = 20260913
METRICS = ["balanced_accuracy", "macro_f1", "silhouette", "distance_ratio"]

WITHIN = "Within lineage / between tissues"
BETWEEN = "Between lineages / within tissue"
DELTA_LABEL = "Delta: between - within"


for _p in [OUT_A1, OUT_A2, COMPARE, REPORT]:
    _p.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Utilities
# ============================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.empty(len(p), dtype=float)
    q[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    return np.minimum(q, 1.0)


def historical_spec():
    """Extract only frozen constants + get_feature_matrix from historical V2 runner."""
    # Must include the upstream constants used to construct the derived
    # feature-column constants. This mirrors the historical
    # clock_architecture_umap.original_spec() dependency set.
    constants = {
        "SEED",
        "POSITIVE_LIMB",
        "NEGATIVE_LIMB",
        "AUXILIARY_CLOCK",
        "CORE_CLOCK",
        "CLOCK_GENES",
        "CORE_FEATURE_COLS",
        "EXPANDED_FEATURE_COLS",
        "MODULE_COLS",
        "CLOCK_FEATURE_COLS",
        "MIN_CELLS_PER_GROUP",
        "LINEAGE_ORDER",
    }
    funcs = {"get_feature_matrix"}

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    nodes = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in funcs:
            nodes.append(node)

    env = {
        "np": np,
        "pd": pd,
        "StandardScaler": StandardScaler,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), env)

    missing = (constants | funcs) - set(env)
    if missing:
        raise RuntimeError(f"Could not extract historical specification: {sorted(missing)}")
    return env


def load_inputs():
    if not FROZEN_META.exists():
        raise FileNotFoundError(FROZEN_META)
    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)

    meta = pd.read_csv(FROZEN_META, sep="\t")

    if "priority_lineage_corrected" not in meta:
        raise RuntimeError("Frozen metadata lacks priority_lineage_corrected.")

    meta = meta.rename(
        columns={"priority_lineage_corrected": "priority_lineage"}
    ).set_index("group_id")

    data = np.load(NPZ_PATH, allow_pickle=True)
    genes = data["genes"].astype(str)
    ids = data["group_id"].astype(str)
    x = data["expression"]

    if x.shape != (2030, 59410):
        raise RuntimeError(f"Unexpected broad matrix shape: {x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError("Broad expression matrix contains non-finite values.")

    if set(ids) != set(meta.index.astype(str)):
        raise RuntimeError("Frozen metadata and broad matrix group_id sets differ.")

    order = pd.Index(ids).get_indexer(meta.index.astype(str))
    if (order < 0).any():
        raise RuntimeError("Could not align frozen metadata to broad matrix.")

    x = x[order]

    if len(meta) != 2030:
        raise RuntimeError(f"Expected 2030 metadata rows, observed {len(meta)}.")

    corrected_n = int(meta.priority_lineage.notna().sum())
    if corrected_n != 1224:
        raise RuntimeError(
            f"Frozen corrected priority membership changed: expected 1224, observed {corrected_n}."
        )

    expected_counts = {
        "b_cell": 153,
        "endothelial": 206,
        "fibroblast": 148,
        "macrophage_monocyte": 280,
        "t_cell": 437,
    }
    observed_counts = meta.priority_lineage.dropna().value_counts().to_dict()
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Frozen lineage counts changed.\nExpected={expected_counts}\nObserved={observed_counts}"
        )

    return meta, x, genes


# ============================================================================
# A1: same historical primary matched panels, corrected lineage membership
# ============================================================================

def build_corrected_folds(meta: pd.DataFrame):
    """
    Preserve historical donor->fold assignment and only remove the 78 bad rows.
    No donor is reassigned to a different test fold.
    """
    cm = meta[meta.priority_lineage.notna()].copy()

    manifest_path = OLD_A1 / "classification_split_manifest.csv"
    manifest = pd.read_csv(manifest_path).set_index("group_id")

    missing = cm.index.difference(manifest.index)
    if len(missing):
        raise RuntimeError(f"Corrected priority IDs missing from historical split manifest: {len(missing)}")

    split = manifest.loc[cm.index].copy()

    if not np.array_equal(
        split["lineage"].astype(str).to_numpy(),
        cm.priority_lineage.astype(str).to_numpy(),
    ):
        bad = split.index[
            split["lineage"].astype(str).to_numpy()
            != cm.priority_lineage.astype(str).to_numpy()
        ]
        raise RuntimeError(f"Retained rows changed lineage assignment: {len(bad)}")

    folds = []
    for f in sorted(split.test_fold.unique()):
        te = np.flatnonzero(split.test_fold.to_numpy() == f)
        tr = np.flatnonzero(split.test_fold.to_numpy() != f)
        if set(cm.iloc[tr].donor_id) & set(cm.iloc[te].donor_id):
            raise RuntimeError(f"Donor leakage in historical fold {f}")
        folds.append((int(f), tr, te))

    corrected_manifest = split.reset_index()[
        ["group_id", "donor_id", "test_fold", "lineage"]
    ]
    corrected_manifest.to_csv(
        OUT_A1 / "classification_split_manifest_corrected.tsv",
        sep="\t",
        index=False,
    )

    return cm, folds


def a1_support(meta: pd.DataFrame):
    cm = meta[meta.priority_lineage.notna()].copy()

    groups = []
    keys = []

    for key, part in cm.groupby(["priority_lineage", "tissue"], sort=True):
        if part.donor_id.nunique() >= 3:
            keys.append(key)
            groups.append(meta.index.get_indexer(part.index))

    within = []
    between = []

    for a, b in itertools.combinations(range(len(keys)), 2):
        if keys[a][0] == keys[b][0]:
            within.append((a, b))
        elif keys[a][1] == keys[b][1]:
            between.append((a, b))

    within = np.asarray(within, dtype=int)
    between = np.asarray(between, dtype=int)

    if len(within) == 0 or len(between) == 0:
        raise RuntimeError("Corrected support produced no within/between comparisons.")

    support = pd.DataFrame(
        [
            {
                "priority_lineage": key[0],
                "tissue": key[1],
                "n_observations": len(cm[(cm.priority_lineage == key[0]) & (cm.tissue == key[1])]),
                "n_donors": cm[
                    (cm.priority_lineage == key[0]) & (cm.tissue == key[1])
                ].donor_id.nunique(),
            }
            for key in keys
        ]
    )
    support.to_csv(OUT_A1 / "corrected_lineage_tissue_support.tsv", sep="\t", index=False)

    return groups, keys, within, between


def build_matching(meta, x, genes, env):
    means = x.mean(0)
    variances = x.var(0)

    valid = (means > 0) & (variances > 1e-12)

    expanded = [v[5:] for v in env["EXPANDED_FEATURE_COLS"]]
    universe = np.flatnonzero(valid & ~np.isin(genes, expanded))

    stats = np.column_stack(
        [
            np.log10(np.maximum(means, 1e-300)),
            np.log10(np.maximum(variances, 1e-300)),
        ]
    )
    stats = (stats - stats[universe].mean(0)) / stats[universe].std(0)

    gene_index = {g: i for i, g in enumerate(genes)}

    pools = {}
    for g in expanded:
        if g not in gene_index:
            raise RuntimeError(f"Clock target missing from broad matrix: {g}")
        gi = gene_index[g]
        order = np.argsort(
            ((stats[universe] - stats[gi]) ** 2).sum(1),
            kind="stable",
        )
        pools[g] = universe[order[:100]]

    pd.DataFrame(
        {
            "gene": genes,
            "mean": means,
            "variance": variances,
            "eligible_nonclock": np.isin(np.arange(len(genes)), universe),
        }
    ).to_csv(
        OUT_A1 / "gene_universe_and_matching_statistics.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "clock_gene": g,
                "candidate": genes[j],
                "rank": rank + 1,
                "distance": float(np.linalg.norm(stats[j] - stats[gene_index[g]])),
            }
            for g, ids in pools.items()
            for rank, j in enumerate(ids)
        ]
    ).to_csv(
        OUT_A1 / "matching_candidate_pools.tsv",
        sep="\t",
        index=False,
    )

    return stats, pools, gene_index


def validate_and_load_historical_panels(env, pools, genes):
    """
    Load the actual frozen PRIMARY panel assignments from the completed
    unattended A1 run, then independently regenerate them from the historical
    RNG procedure and require exact equality.

    Historical files:
      01_matched_random_controls/unattended/primary_core_genes.csv
      01_matched_random_controls/unattended/primary_expanded_genes.csv
    """
    source_files = [
        OLD_A1_UNATTENDED / "primary_core_genes.csv",
        OLD_A1_UNATTENDED / "primary_expanded_genes.csv",
    ]

    missing = [p for p in source_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Historical unattended PRIMARY matched-panel files are missing:\n"
            + "\n".join(str(p) for p in missing)
        )

    saved = pd.concat([pd.read_csv(p) for p in source_files], ignore_index=True)

    required = {"panel", "variant", "iteration", "clock_gene", "matched_gene"}
    if not required.issubset(saved.columns):
        raise RuntimeError(
            "Unexpected historical primary panel schema. "
            f"Required={sorted(required)}; observed={saved.columns.tolist()}"
        )

    saved = saved.loc[
        saved["variant"].astype(str).eq("primary"),
        ["panel", "variant", "iteration", "clock_gene", "matched_gene"],
    ].reset_index(drop=True)

    # Exact primary procedure from phaseA_unattended.py::matched_controls():
    # one RNG seeded 20260913, Core first then Expanded, no duplicate genes
    # within a panel.
    rng = np.random.default_rng(SEED)
    regenerated = []

    for panel, key in [
        ("Core clock", "CORE_FEATURE_COLS"),
        ("Expanded clock", "EXPANDED_FEATURE_COLS"),
    ]:
        targets = [v[5:] for v in env[key]]

        for iteration in range(1, 1001):
            choice = []

            for g in targets:
                pool = pools[g]
                available = pool[~np.isin(pool, choice)]
                j = int(rng.choice(available))
                choice.append(j)

                regenerated.append(
                    {
                        "panel": panel,
                        "variant": "primary",
                        "iteration": iteration,
                        "clock_gene": g,
                        "matched_gene": genes[j],
                    }
                )

    regenerated = pd.DataFrame(regenerated)

    pd.testing.assert_frame_equal(
        saved,
        regenerated,
        check_dtype=False,
        check_like=False,
    )

    regenerated.to_csv(
        OUT_A1 / "matched_panel_genes_primary_FROZEN.tsv",
        sep="\t",
        index=False,
    )

    write_json(
        OUT_A1 / "matched_panel_gene_provenance.json",
        {
            "source_files": [str(p) for p in source_files],
            "source_sha256": {p.name: sha256(p) for p in source_files},
            "variant": "primary",
            "seed": SEED,
            "core_panels": 1000,
            "expanded_panels": 1000,
            "exact_regeneration_check": "PASS",
        },
    )

    return regenerated


def run_a1(meta, x, genes, env):
    print("\n[A1] Preparing corrected membership and frozen folds...", flush=True)

    cm, folds = build_corrected_folds(meta)
    ii = meta.index.get_indexer(cm.index)
    labels = cm.priority_lineage.astype(str).to_numpy()

    groups, keys, within, between = a1_support(meta)

    _, pools, gene_index = build_matching(meta, x, genes, env)
    panel_assignments = validate_and_load_historical_panels(env, pools, genes)

    support_manifest = {
        "corrected_priority_observations": len(cm),
        "supported_lineage_tissue_groups": len(keys),
        "within_lineage_between_tissue_pairs": len(within),
        "between_lineage_within_tissue_pairs": len(between),
        "n_donors_priority": int(cm.donor_id.nunique()),
        "n_tissues_priority": int(cm.tissue.nunique()),
        "folds_preserved_from_legacy": True,
        "matched_panel_gene_assignments_exactly_preserved": True,
    }
    write_json(OUT_A1 / "A1_corrected_support_manifest.json", support_manifest)

    def evaluate(ids):
        values = x[:, ids]
        z = StandardScaler().fit_transform(values)

        xx = values[ii]
        pred = np.empty(len(ii), dtype=object)

        for _, tr, te in folds:
            scale = StandardScaler()
            train = scale.fit_transform(xx[tr])
            test = scale.transform(xx[te])

            model = RidgeClassifier(alpha=1, class_weight="balanced")
            model.fit(train, labels[tr])
            pred[te] = model.predict(test)

        cent = np.array([z[g].mean(0) for g in groups])

        d_between = np.linalg.norm(
            cent[between[:, 0]] - cent[between[:, 1]],
            axis=1,
        )
        d_within = np.linalg.norm(
            cent[within[:, 0]] - cent[within[:, 1]],
            axis=1,
        )

        return {
            "balanced_accuracy": balanced_accuracy_score(labels, pred),
            "macro_f1": f1_score(labels, pred, average="macro"),
            "silhouette": silhouette_score(z[ii], labels),
            "distance_ratio": float(np.median(d_between) / np.median(d_within)),
        }

    observed_rows = []
    null_rows_all = []

    for panel, key in [
        ("Core clock", "CORE_FEATURE_COLS"),
        ("Expanded clock", "EXPANDED_FEATURE_COLS"),
    ]:
        targets = [v[5:] for v in env[key]]
        ids = [gene_index[g] for g in targets]

        print(f"[A1] Observed metrics: {panel}", flush=True)
        obs = evaluate(ids)

        observed_rows.append(
            {
                "panel": panel,
                "n_observations": len(ii),
                "n_genes": len(ids),
                **obs,
                "n_null_panels": 1000,
                "null_status": "COMPLETE_CORRECTED_MEMBERSHIP",
            }
        )

        checkpoint = OUT_A1 / (
            "null_core_checkpoint.tsv"
            if panel == "Core clock"
            else "null_expanded_checkpoint.tsv"
        )

        if checkpoint.exists():
            existing = pd.read_csv(checkpoint, sep="\t")
            existing = existing[existing["panel"] == panel].copy()
            done = set(existing.iteration.astype(int))
            rows = existing.to_dict("records")
            print(
                f"[A1] Resuming {panel}: {len(done)}/1000 iterations already present.",
                flush=True,
            )
        else:
            done = set()
            rows = []

        psel = panel_assignments[panel_assignments.panel == panel]

        for iteration in range(1, 1001):
            if iteration in done:
                continue

            sub = psel[psel.iteration == iteration].set_index("clock_gene")
            if set(sub.index) != set(targets):
                raise RuntimeError(
                    f"{panel} iteration {iteration}: historical target set mismatch."
                )

            selected_genes = sub.loc[targets, "matched_gene"].tolist()
            selected_ids = [gene_index[g] for g in selected_genes]

            result = {
                "panel": panel,
                "iteration": iteration,
                **evaluate(selected_ids),
            }
            rows.append(result)

            if iteration % 50 == 0:
                pd.DataFrame(rows).sort_values("iteration").to_csv(
                    checkpoint,
                    sep="\t",
                    index=False,
                )
                print(f"[A1] {panel}: {iteration}/1000", flush=True)

        panel_null = pd.DataFrame(rows).sort_values("iteration")
        if len(panel_null) != 1000 or panel_null.iteration.nunique() != 1000:
            raise RuntimeError(f"{panel}: incomplete null panel results.")

        panel_null.to_csv(checkpoint, sep="\t", index=False)
        null_rows_all.append(panel_null)

    observed = pd.DataFrame(observed_rows)
    null = pd.concat(null_rows_all, ignore_index=True)

    summary_rows = []
    for r in observed.itertuples(index=False):
        for metric in METRICS:
            vals = null.loc[null.panel == r.panel, metric].to_numpy(float)
            value = float(getattr(r, metric))

            summary_rows.append(
                {
                    "panel": r.panel,
                    "metric": metric,
                    "observed": value,
                    "null_mean": float(vals.mean()),
                    "null_median": float(np.median(vals)),
                    "null_sd": float(vals.std(ddof=1)),
                    "null_q025": float(np.quantile(vals, 0.025)),
                    "null_q975": float(np.quantile(vals, 0.975)),
                    "empirical_percentile": float(
                        100 * (np.sum(vals < value) + 0.5 * np.sum(vals == value)) / len(vals)
                    ),
                    "empirical_P": float(
                        (1 + np.sum(vals >= value)) / (len(vals) + 1)
                    ),
                    "Z": float((value - vals.mean()) / vals.std(ddof=1)),
                    "n_null_panels": len(vals),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["FDR_8_metrics"] = bh(summary.empirical_P.to_numpy())

    observed.to_csv(OUT_A1 / "observed_metrics_corrected.tsv", sep="\t", index=False)
    null.to_csv(OUT_A1 / "empirical_null_distributions_corrected.tsv", sep="\t", index=False)
    summary.to_csv(OUT_A1 / "empirical_null_summary_corrected.tsv", sep="\t", index=False)

    # Prefer the completed unattended historical A1 summary.
    old_summary_candidates = [
        OLD_A1_UNATTENDED / "empirical_null_summary.csv",
        OLD_A1 / "empirical_null_summary.csv",
    ]
    old_summary_path = next((p for p in old_summary_candidates if p.exists()), None)

    if old_summary_path is not None:
        old = pd.read_csv(old_summary_path)

        if "variant" in old.columns:
            old = old[old["variant"].astype(str).eq("primary")].copy()

        rename = {}
        if "P" in old.columns and "empirical_P" not in old.columns:
            rename["P"] = "empirical_P"
        if "percentile" in old.columns and "empirical_percentile" not in old.columns:
            rename["percentile"] = "empirical_percentile"
        if "BH_8_metrics" in old.columns and "FDR_8_metrics" not in old.columns:
            rename["BH_8_metrics"] = "FDR_8_metrics"
        old = old.rename(columns=rename)

        merged = old.merge(
            summary,
            on=["panel", "metric"],
            suffixes=("_legacy", "_corrected"),
            validate="one_to_one",
        )

        for col in [
            "observed",
            "null_mean",
            "null_median",
            "empirical_percentile",
            "empirical_P",
            "Z",
        ]:
            if f"{col}_legacy" in merged and f"{col}_corrected" in merged:
                merged[f"delta_{col}"] = (
                    merged[f"{col}_corrected"] - merged[f"{col}_legacy"]
                )

        merged["legacy_source_file"] = str(old_summary_path)

        merged.to_csv(
            COMPARE / "A1_legacy_vs_corrected.tsv",
            sep="\t",
            index=False,
        )

    print("[A1] COMPLETE", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary, support_manifest


# ============================================================================
# A2: corrected membership, same frozen 24-feature standardized expression
# ============================================================================

def corrected_centroid_tables(meta, features, threshold, lineage_order):
    primary = meta[meta.priority_lineage.notna()].copy()
    combined = primary.join(features)
    cols = features.columns.tolist()

    records = []
    vectors = []

    for (lineage, tissue), part in combined.groupby(
        ["priority_lineage", "tissue"],
        sort=True,
    ):
        centroid = part[cols].mean().to_numpy(float)
        donor_count = part.donor_id.nunique()

        records.append(
            {
                "lineage": lineage,
                "tissue": tissue,
                "n_donors": int(donor_count),
                "n_observations": len(part),
                "n_cell_types": int(part.cell_type.nunique()),
                "adequate_support": bool(donor_count >= threshold),
                "support": (
                    "adequate" if donor_count >= threshold
                    else "moderate" if donor_count == 2
                    else "low"
                ),
                "donor_ids": ";".join(sorted(part.donor_id.astype(str).unique())),
            }
        )
        vectors.append(centroid)

    table = pd.DataFrame(records)
    matrix = np.vstack(vectors)

    inclusion_rows = []
    for lin in lineage_order:
        allrows = combined[combined.priority_lineage == lin]
        tab = table[table.lineage == lin]
        eligible = tab[tab.adequate_support]
        included = allrows[allrows.tissue.isin(eligible.tissue)]

        inclusion_rows.append(
            {
                "lineage": lin,
                "all_tissues": len(tab),
                "all_donors": int(allrows.donor_id.nunique()),
                "all_observations": len(allrows),
                "supported_tissues": len(eligible),
                "supported_donors": int(included.donor_id.nunique()),
                "supported_observations": len(included),
                "excluded_low_support_tissues": len(tab) - len(eligible),
            }
        )

    return table, matrix, pd.DataFrame(inclusion_rows)


def corrected_pair_tables(centroids, matrix, lineage_order):
    supported = centroids[centroids.adequate_support]
    records = []

    for lin in lineage_order:
        indices = supported.index[supported.lineage == lin].tolist()
        for a, b in itertools.combinations(indices, 2):
            left = centroids.loc[a]
            right = centroids.loc[b]

            records.append(
                {
                    "comparison_type": WITHIN,
                    "lineage": lin,
                    "tissue_A": left.tissue,
                    "tissue_B": right.tissue,
                    "tissue": "",
                    "lineage_A": lin,
                    "lineage_B": lin,
                    "distance": float(np.linalg.norm(matrix[a] - matrix[b])),
                    "n_donors_A": int(left.n_donors),
                    "n_donors_B": int(right.n_donors),
                    "n_shared_donors": len(
                        set(left.donor_ids.split(";")) & set(right.donor_ids.split(";"))
                    ),
                    "centroid_index_A": a,
                    "centroid_index_B": b,
                }
            )

    for tissue, part in supported.groupby("tissue", sort=True):
        for a, b in itertools.combinations(part.index, 2):
            left = centroids.loc[a]
            right = centroids.loc[b]

            records.append(
                {
                    "comparison_type": BETWEEN,
                    "lineage": "",
                    "tissue_A": tissue,
                    "tissue_B": tissue,
                    "tissue": tissue,
                    "lineage_A": left.lineage,
                    "lineage_B": right.lineage,
                    "distance": float(np.linalg.norm(matrix[a] - matrix[b])),
                    "n_donors_A": int(left.n_donors),
                    "n_donors_B": int(right.n_donors),
                    "n_shared_donors": len(
                        set(left.donor_ids.split(";")) & set(right.donor_ids.split(";"))
                    ),
                    "centroid_index_A": a,
                    "centroid_index_B": b,
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("No corrected A2 distance pairs were generated.")
    return out


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    halfway = cumulative[-1] / 2
    i = np.searchsorted(cumulative, halfway, side="left")

    if i + 1 < len(values) and np.isclose(cumulative[i], halfway, rtol=1e-13, atol=0):
        return float((values[i] + values[i + 1]) / 2)

    return float(values[i])


def bootstrap_distances(meta, features, centroids, pairs, lineage_order, iterations=5000, seed=BOOTSTRAP_SEED):
    if iterations < 5000:
        raise ValueError("At least 5000 bootstrap iterations required.")

    donors = sorted(
        meta.loc[meta.priority_lineage.notna(), "donor_id"].astype(str).unique()
    )
    donor_index = {d: i for i, d in enumerate(donors)}

    supported = centroids[centroids.adequate_support]
    tissues = sorted(supported.tissue.unique())
    tissue_index = {t: i for i, t in enumerate(tissues)}

    group_index = {(r.lineage, r.tissue): i for i, r in centroids.iterrows()}

    counts = np.zeros((len(centroids), len(donors)))
    sums = np.zeros((len(centroids), len(donors), features.shape[1]))

    for group_id, row in meta[meta.priority_lineage.notna()].iterrows():
        g = group_index[(str(row.priority_lineage), row.tissue)]
        d = donor_index[str(row.donor_id)]
        counts[g, d] += 1
        sums[g, d] += features.loc[group_id].to_numpy(float)

    point = sums.sum(axis=1) / counts.sum(axis=1)[:, None]

    a = pairs.centroid_index_A.to_numpy(int)
    b = pairs.centroid_index_B.to_numpy(int)

    np.testing.assert_allclose(
        np.linalg.norm(point[a] - point[b], axis=1),
        pairs.distance.to_numpy(float),
        rtol=1e-13,
        atol=1e-13,
    )

    ta = np.array([tissue_index[t] for t in pairs.tissue_A])
    tb = np.array([tissue_index[t] for t in pairs.tissue_B])

    within = pairs.comparison_type.eq(WITHIN).to_numpy()
    between = ~within

    lineage_masks = {
        lin: within & pairs.lineage.eq(lin).to_numpy()
        for lin in lineage_order
    }

    rng = np.random.default_rng(seed)
    records = []

    for iteration in range(iterations):
        donor_weight = rng.exponential(1, size=len(donors))
        tissue_weight = rng.exponential(1, size=len(tissues))

        denominator = counts @ donor_weight
        group_means = (
            np.einsum("gdf,d->gf", sums, donor_weight, optimize=False)
            / denominator[:, None]
        )

        distances = np.linalg.norm(group_means[a] - group_means[b], axis=1)

        weights = tissue_weight[ta].copy()
        weights[within] *= tissue_weight[tb[within]]

        mw = weighted_median(distances[within], weights[within])
        mb = weighted_median(distances[between], weights[between])

        rec = {
            "iteration": iteration + 1,
            "median_within": mw,
            "median_between": mb,
            "delta": mb - mw,
            "donor_only_delta": float(
                np.median(distances[between]) - np.median(distances[within])
            ),
        }

        for lin, mask in lineage_masks.items():
            rec[f"within_{lin}"] = weighted_median(distances[mask], weights[mask])

        records.append(rec)

        if (iteration + 1) % 1000 == 0:
            print(f"[A2] Bootstrap {iteration+1}/{iterations}", flush=True)

    result = pd.DataFrame(records)
    if not np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("Non-finite bootstrap result.")

    return result


def summarize_distances(pairs, boot, lineage_order):
    rows = []

    for typ, col in [(WITHIN, "median_within"), (BETWEEN, "median_between")]:
        values = pairs.loc[pairs.comparison_type == typ, "distance"]

        rows.append(
            {
                "summary_type": "overall",
                "comparison_type": typ,
                "lineage": "",
                "n_comparisons": len(values),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "bootstrap_95_ci_low": float(boot[col].quantile(0.025)),
                "bootstrap_95_ci_high": float(boot[col].quantile(0.975)),
            }
        )

    delta = rows[1]["median"] - rows[0]["median"]

    rows.append(
        {
            "summary_type": "effect",
            "comparison_type": DELTA_LABEL,
            "lineage": "",
            "n_comparisons": len(pairs),
            "median": delta,
            "q25": np.nan,
            "q75": np.nan,
            "bootstrap_95_ci_low": float(boot.delta.quantile(0.025)),
            "bootstrap_95_ci_high": float(boot.delta.quantile(0.975)),
        }
    )

    for lin in lineage_order:
        values = pairs.loc[
            (pairs.comparison_type == WITHIN) & (pairs.lineage == lin),
            "distance",
        ]

        rows.append(
            {
                "summary_type": "lineage_within",
                "comparison_type": WITHIN,
                "lineage": lin,
                "n_comparisons": len(values),
                "median": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "bootstrap_95_ci_low": float(boot[f"within_{lin}"].quantile(0.025)),
                "bootstrap_95_ci_high": float(boot[f"within_{lin}"].quantile(0.975)),
            }
        )

    return pd.DataFrame(rows)


def adjusted_r2(meta, y, terms):
    pieces = [np.ones((len(meta), 1))]

    for term in terms:
        value = (
            meta.cell_type.astype(str) + "||" + meta.tissue.astype(str)
            if term == "celltype_tissue"
            else meta[term].astype(str)
        )
        pieces.append(pd.get_dummies(value, dtype=float).to_numpy())

    design = np.column_stack(pieces)

    beta, _, rank, _ = lstsq(
        design,
        y,
        cond=max(design.shape) * np.finfo(float).eps,
        lapack_driver="gelsd",
    )

    sse = ((y - design @ beta) ** 2).sum(axis=0)
    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)

    return 1 - sse / sst * (len(meta) - 1) / (len(meta) - rank), int(rank)


def effects(meta, y):
    main, rank = adjusted_r2(meta, y, ["cell_type", "tissue", "donor_id"])

    reduced = [
        adjusted_r2(meta, y, terms)[0]
        for terms in [
            ["tissue", "donor_id"],
            ["cell_type", "donor_id"],
            ["cell_type", "tissue"],
        ]
    ]

    full, fullrank = adjusted_r2(meta, y, ["celltype_tissue", "donor_id"])

    return np.column_stack([main - r for r in reduced] + [full - main]), (rank, fullrank)


def stability(meta, y, lineage_order):
    sse = np.zeros(y.shape[1])
    baseline = sse.copy()

    labels = meta.priority_lineage.to_numpy()

    for tissue in sorted(meta.tissue.unique()):
        test = meta.tissue.eq(tissue).to_numpy()
        train = ~test
        grand = y[train].mean(axis=0)

        for lin in lineage_order:
            target = test & (labels == lin)
            reference = train & (labels == lin)

            if target.any():
                if not reference.any():
                    raise ValueError(
                        f"No cross-tissue lineage reference for {lin}, tissue={tissue}"
                    )

                sse += ((y[target] - y[reference].mean(axis=0)) ** 2).sum(axis=0)
                baseline += ((y[target] - grand) ** 2).sum(axis=0)

    return 1 - sse / baseline


def remodeling(meta, y, lineage_order):
    result = []

    for lin in lineage_order:
        mask = meta.priority_lineage.eq(lin).to_numpy()
        part = meta.loc[mask]

        full, _ = adjusted_r2(part, y[mask], ["tissue", "donor_id"])
        reduced, _ = adjusted_r2(part, y[mask], ["donor_id"])
        result.append(full - reduced)

    return np.mean(result, axis=0), np.asarray(result)


def jackknife_interval(point, draws):
    draws = np.asarray(draws)
    n = len(draws)

    se = np.sqrt(
        (n - 1) / n * ((draws - draws.mean(axis=0)) ** 2).sum(axis=0)
    )

    q = student_t.ppf(0.975, n - 1)
    return point - q * se, point + q * se, se


def run_feature_scores(meta, env, supported_keys):
    raw = pd.read_csv(FEATURE_PATH, sep="\t")

    if raw.group_id.duplicated().any():
        raise RuntimeError("Duplicate group_id in historical feature table.")

    valid = (
        raw[raw.n_cells >= env["MIN_CELLS_PER_GROUP"]]
        .set_index("group_id")
        .loc[meta.index]
    )

    feature_cols = list(
        dict.fromkeys(env["CLOCK_FEATURE_COLS"] + env["MODULE_COLS"])
    )

    prepared, retained = env["get_feature_matrix"](valid.copy(), feature_cols)

    if retained != feature_cols:
        raise RuntimeError("Historical feature preparation dropped/reordered features.")

    prepared.index = meta.index
    y = prepared.to_numpy(float)

    point, ranks = effects(meta, y)

    legacy_feature = pd.read_csv(
        LEGACY / "statistics" / "feature_effects_and_scores.csv"
    ).set_index("feature")

    effect_names = [
        "cell_identity_unique_adj_r2",
        "tissue_context_unique_adj_r2",
        "donor_unique_adj_r2",
        "cell_identity_tissue_context_interaction_adj_r2",
    ]

    np.testing.assert_allclose(
        point,
        legacy_feature.loc[feature_cols, effect_names].to_numpy(),
        atol=2e-10,
        rtol=2e-10,
    )

    keep = np.array(
        [
            (row.priority_lineage, row.tissue) in supported_keys
            for row in meta.itertuples()
        ]
    )

    sm = meta.loc[keep].copy()
    sy = y[keep]

    lineage_order = list(env["LINEAGE_ORDER"])

    stability_point = stability(sm, sy, lineage_order)
    remodeling_point, bylineage = remodeling(sm, sy, lineage_order)

    score_donors = sorted(sm.donor_id.unique())
    score_draws = []

    for i, donor in enumerate(score_donors, 1):
        ds = sm.donor_id.ne(donor).to_numpy()

        score_draws.append(
            np.column_stack(
                [
                    stability(sm.loc[ds], sy[ds], lineage_order),
                    remodeling(sm.loc[ds], sy[ds], lineage_order)[0],
                ]
            )
        )

        print(
            f"[A2] Feature-score donor deletion {i}/{len(score_donors)}",
            flush=True,
        )

    score_draws = np.asarray(score_draws)

    feature_table = legacy_feature.loc[feature_cols].reset_index().copy()

    feature_table["score_n_observations"] = len(sm)
    feature_table["score_n_donors"] = sm.donor_id.nunique()
    feature_table["score_n_tissues"] = sm.tissue.nunique()

    for j, (name, value) in enumerate(
        [
            ("lineage_stability", stability_point),
            ("tissue_remodeling", remodeling_point),
        ]
    ):
        low, high, _ = jackknife_interval(value, score_draws[:, :, j])

        feature_table[name] = value
        feature_table[f"{name}_ci_low"] = low
        feature_table[f"{name}_ci_high"] = high
        feature_table[f"{name}_deletion_min"] = score_draws[:, :, j].min(axis=0)
        feature_table[f"{name}_deletion_max"] = score_draws[:, :, j].max(axis=0)

    bylineage_table = pd.DataFrame(
        [
            {
                "lineage": lin,
                "feature": f,
                "tissue_unique_adj_r2": bylineage[i, j],
            }
            for i, lin in enumerate(lineage_order)
            for j, f in enumerate(feature_cols)
        ]
    )

    return feature_table, bylineage_table, {
        "n_supported_observations": len(sm),
        "n_supported_donors": int(sm.donor_id.nunique()),
        "n_supported_tissues": int(sm.tissue.nunique()),
        "main_rank": ranks[0],
        "interaction_rank": ranks[1],
        "all_2030_effects_reproduced": True,
    }


def distance_delta_fixed_support(meta, features, centroids, pairs):
    joined = meta.join(features)

    means = (
        joined[joined.priority_lineage.notna()]
        .groupby(["priority_lineage", "tissue"])[list(features.columns)]
        .mean()
    )

    supported = centroids[centroids.adequate_support]
    vectors = np.array(
        [
            means.loc[(r.lineage, r.tissue)].to_numpy(float)
            for r in supported.itertuples()
        ]
    )

    mapping = {
        old_index: new_index
        for new_index, old_index in enumerate(supported.index)
    }

    a = pairs.centroid_index_A.map(mapping).to_numpy(int)
    b = pairs.centroid_index_B.map(mapping).to_numpy(int)

    d = np.linalg.norm(vectors[a] - vectors[b], axis=1)
    within = pairs.comparison_type.eq(WITHIN).to_numpy()

    return float(np.median(d[~within]) - np.median(d[within]))


def run_a2(meta, env, bootstrap_iterations=5000):
    print("\n[A2] Loading frozen 24-feature standardized matrix...", flush=True)

    saved = pd.read_csv(
        STANDARDIZED,
        sep="\t",
        float_precision="round_trip",
    ).set_index("group_id")

    missing = meta.index.difference(saved.index)
    if len(missing):
        raise RuntimeError(f"Standardized matrix missing {len(missing)} frozen observations.")

    features = saved.loc[meta.index].copy()

    if len(features) != 2030:
        raise RuntimeError("Expected 2030 rows in frozen standardized feature matrix.")

    threshold = 3
    lineage_order = list(env["LINEAGE_ORDER"])

    centroids, matrix, inclusion = corrected_centroid_tables(
        meta, features, threshold, lineage_order
    )

    pairs = corrected_pair_tables(centroids, matrix, lineage_order)

    supported = centroids[centroids.adequate_support]
    supported_keys = set(zip(supported.lineage, supported.tissue))

    print(
        f"[A2] Corrected support: {len(supported)} lineage×tissue groups; "
        f"{(pairs.comparison_type == WITHIN).sum()} within pairs; "
        f"{(pairs.comparison_type == BETWEEN).sum()} between pairs.",
        flush=True,
    )

    boot = bootstrap_distances(
        meta,
        features,
        centroids,
        pairs,
        lineage_order,
        iterations=bootstrap_iterations,
        seed=BOOTSTRAP_SEED,
    )

    summary = summarize_distances(pairs, boot, lineage_order)

    centroids.to_csv(OUT_A2 / "corrected_centroid_support.tsv", sep="\t", index=False)
    inclusion.to_csv(OUT_A2 / "corrected_lineage_inclusion.tsv", sep="\t", index=False)
    pairs.to_csv(OUT_A2 / "corrected_between_within_distances.tsv", sep="\t", index=False)
    boot.to_csv(OUT_A2 / "corrected_distance_bootstrap_5000.tsv", sep="\t", index=False)
    summary.to_csv(OUT_A2 / "corrected_distance_summary.tsv", sep="\t", index=False)

    plasticity = []

    for lin in lineage_order:
        idx = supported.index[supported.lineage == lin]
        vec = matrix[idx]

        center = vec.mean(axis=0)
        displacement = np.linalg.norm(vec - center, axis=1)

        sr = summary[
            (summary.summary_type == "lineage_within")
            & (summary.lineage == lin)
        ].iloc[0]

        inc = inclusion[inclusion.lineage == lin].iloc[0]

        plasticity.append(
            {
                "lineage": lin,
                "supported_tissues": len(idx),
                "donors": int(inc.supported_donors),
                "observations": int(inc.supported_observations),
                "median_cross_tissue_distance": float(sr["median"]),
                "ci_low": float(sr.bootstrap_95_ci_low),
                "ci_high": float(sr.bootstrap_95_ci_high),
                "median_centroid_displacement": float(np.median(displacement)),
                "rms_tissue_dispersion": float(
                    np.sqrt(np.mean(displacement * displacement))
                ),
            }
        )

    plasticity = pd.DataFrame(plasticity)
    plasticity.to_csv(
        OUT_A2 / "corrected_lineage_plasticity.tsv",
        sep="\t",
        index=False,
    )

    sensitivity = []

    for donor in sorted(meta.donor_id.unique()):
        mask = meta.donor_id.ne(donor)
        delta = distance_delta_fixed_support(
            meta.loc[mask],
            features.loc[mask],
            centroids,
            pairs,
        )

        sensitivity.append(
            {
                "deletion": "donor",
                "identifier": donor,
                "delta": delta,
                "n_pairs": len(pairs),
            }
        )

    for tissue in sorted(supported.tissue.unique()):
        pp = pairs[
            pairs.tissue_A.ne(tissue)
            & pairs.tissue_B.ne(tissue)
        ]

        delta = (
            pp.loc[pp.comparison_type == BETWEEN, "distance"].median()
            - pp.loc[pp.comparison_type == WITHIN, "distance"].median()
        )

        sensitivity.append(
            {
                "deletion": "tissue",
                "identifier": tissue,
                "delta": float(delta),
                "n_pairs": len(pp),
            }
        )

    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(
        OUT_A2 / "corrected_distance_driver_sensitivity.tsv",
        sep="\t",
        index=False,
    )

    print("[A2] Recomputing membership-dependent feature scores...", flush=True)

    feature_table, bylineage_table, score_manifest = run_feature_scores(
        meta, env, supported_keys
    )

    feature_table.to_csv(
        OUT_A2 / "corrected_feature_effects_and_scores.tsv",
        sep="\t",
        index=False,
    )
    bylineage_table.to_csv(
        OUT_A2 / "corrected_lineage_feature_tissue_effects.tsv",
        sep="\t",
        index=False,
    )

    manifest = {
        "corrected_priority_observations": int(meta.priority_lineage.notna().sum()),
        "supported_lineage_tissue_groups": len(supported),
        "within_lineage_between_tissue_pairs": int(
            (pairs.comparison_type == WITHIN).sum()
        ),
        "between_lineage_within_tissue_pairs": int(
            (pairs.comparison_type == BETWEEN).sum()
        ),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "standardized_24_feature_matrix_reused_exactly": True,
        "feature_scores": score_manifest,
    }
    write_json(OUT_A2 / "A2_corrected_manifest.json", manifest)

    legacy_summary_path = LEGACY / "statistics" / "distance_summary.csv"
    if legacy_summary_path.exists():
        old = pd.read_csv(legacy_summary_path)

        keys = ["summary_type", "comparison_type", "lineage"]
        old2 = old.copy()
        new2 = summary.copy()

        for c in keys:
            old2[c] = old2[c].fillna("").astype(str)
            new2[c] = new2[c].fillna("").astype(str)

        merged = old2.merge(
            new2,
            on=keys,
            suffixes=("_legacy", "_corrected"),
            validate="one_to_one",
        )

        for col in [
            "n_comparisons",
            "median",
            "bootstrap_95_ci_low",
            "bootstrap_95_ci_high",
        ]:
            a = f"{col}_legacy"
            b = f"{col}_corrected"
            if a in merged and b in merged:
                merged[f"delta_{col}"] = merged[b] - merged[a]

        merged.to_csv(
            COMPARE / "A2_distance_legacy_vs_corrected.tsv",
            sep="\t",
            index=False,
        )

    legacy_feature_path = LEGACY / "statistics" / "feature_effects_and_scores.csv"
    if legacy_feature_path.exists():
        oldf = pd.read_csv(legacy_feature_path)[
            ["feature", "lineage_stability", "tissue_remodeling"]
        ]

        newf = feature_table[
            ["feature", "lineage_stability", "tissue_remodeling"]
        ]

        mf = oldf.merge(
            newf,
            on="feature",
            suffixes=("_legacy", "_corrected"),
            validate="one_to_one",
        )

        mf["delta_lineage_stability"] = (
            mf.lineage_stability_corrected - mf.lineage_stability_legacy
        )
        mf["delta_tissue_remodeling"] = (
            mf.tissue_remodeling_corrected - mf.tissue_remodeling_legacy
        )

        mf.to_csv(
            COMPARE / "A2_feature_scores_legacy_vs_corrected.tsv",
            sep="\t",
            index=False,
        )

    print("[A2] COMPLETE", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n[A2] Corrected lineage plasticity:")
    print(plasticity.to_string(index=False), flush=True)

    return summary, manifest


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Corrected Tabula Phase A A1/A2 numerical rerun. "
            "Historical methods are preserved; only frozen priority-lineage membership changes."
        )
    )
    parser.add_argument(
        "--skip-a1",
        action="store_true",
        help="Skip corrected A1 matched-null rerun.",
    )
    parser.add_argument(
        "--skip-a2",
        action="store_true",
        help="Skip corrected A2 distance/feature-score rerun.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
        help="A2 crossed donor/tissue multiplier bootstrap iterations; minimum 5000.",
    )

    args = parser.parse_args()

    env = historical_spec()
    meta, x, genes = load_inputs()

    lineage_order = list(env["LINEAGE_ORDER"])

    expected_lineages = {
        "b_cell",
        "t_cell",
        "macrophage_monocyte",
        "fibroblast",
        "endothelial",
    }
    if set(lineage_order) != expected_lineages:
        raise RuntimeError(f"Unexpected historical LINEAGE_ORDER: {lineage_order}")

    run_manifest = {
        "status": "STARTED",
        "mapping_status": "FROZEN",
        "corrected_priority_n": int(meta.priority_lineage.notna().sum()),
        "lineage_order": lineage_order,
        "immutable_inputs": {
            "frozen_corrected_metadata": {
                "path": str(FROZEN_META),
                "sha256": sha256(FROZEN_META),
            },
            "broad_normalized_mean": {
                "path": str(NPZ_PATH),
                "sha256": sha256(NPZ_PATH),
            },
            "historical_revision_v2_runner": {
                "path": str(RUNNER),
                "sha256": sha256(RUNNER),
            },
            "standardized_24_feature_matrix": {
                "path": str(STANDARDIZED),
                "sha256": sha256(STANDARDIZED),
            },
        },
        "principle": (
            "Historical analysis semantics, feature definitions, donor folds, "
            "matched-panel gene assignments, standardization and support threshold "
            "are preserved. Only the frozen corrected priority-lineage membership "
            "is substituted; lineage-dependent support and statistics are recomputed."
        ),
    }
    write_json(REPORT / "A1_A2_corrected_rerun_manifest.json", run_manifest)

    results = {}

    with threadpool_limits(limits=1):
        if not args.skip_a1:
            _, a1_manifest = run_a1(meta, x, genes, env)
            results["A1"] = a1_manifest

        if not args.skip_a2:
            _, a2_manifest = run_a2(
                meta,
                env,
                bootstrap_iterations=args.bootstrap_iterations,
            )
            results["A2"] = a2_manifest

    run_manifest["status"] = "COMPLETE"
    run_manifest["results"] = results
    write_json(REPORT / "A1_A2_corrected_rerun_manifest.json", run_manifest)

    print("\n" + "=" * 78)
    print("CORRECTED A1/A2 RERUN COMPLETE")
    print("=" * 78)
    print(f"Corrected priority observations: {meta.priority_lineage.notna().sum()}")
    print(f"A1 output: {OUT_A1}")
    print(f"A2 output: {OUT_A2}")
    print(f"Comparison output: {COMPARE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
