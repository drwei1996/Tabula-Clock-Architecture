"""Offline smoke test for the Tabula-Clock-Architecture release.

Verifies, without any downloads:
1. the three gene sets load and differ EXACTLY as the paper requires
   (anchor17 contains BHLHE41/FBXO21 and lacks RORA/RORB; static17 the reverse;
   core8 is a subset of static17);
2. the six-arm mapping covers the static 17 disjointly (6 arms, 17 slots);
3. frozen configs parse and record the prespecified seed/feature sets;
4. every shipped source-data file is non-empty and has a header;
5. key analysis scripts referenced by docs/figure_code_mapping.csv exist.

Exit code 0 = PASS.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_rows(p):
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    errs = []

    # ---- 1. gene sets -----------------------------------------------------
    core8 = read_rows(REPO / "gene_sets" / "clock_core8.csv")
    static17 = read_rows(REPO / "gene_sets" / "clock_static17.csv")
    anchor17 = read_rows(REPO / "gene_sets" / "anchorphase_core17_reference.csv")
    c8 = {r["gene"] for r in core8}
    s17 = {r["gene"] for r in static17}
    a17 = {r["anchor_gene_symbol_as_used"] for r in anchor17}
    assert len(c8) == 8, len(c8)
    assert len(s17) == 17, len(s17)
    assert len(a17) == 17, len(a17)
    assert c8 <= s17, "core8 must be a subset of static17"
    a17u = {g.upper() for g in a17}
    assert a17u - s17 == {"BHLHE41", "FBXO21"}, a17u - s17
    assert s17 - a17u == {"RORA", "RORB"}, s17 - a17u
    phis = [float(r["prior_acrophase_h"]) for r in anchor17]
    assert all(0 <= p < 24 for p in phis)

    # ---- 2. six-arm mapping -----------------------------------------------
    arms = read_rows(REPO / "gene_sets" / "six_arm_mapping.csv")
    assert len(arms) == 6, arms
    members = []
    for r in arms:
        members += r["genes"].split(";")
    assert len(members) == 17 and set(members) == s17, "six arms must partition the static 17"

    # ---- 3. configs ---------------------------------------------------------
    proto = json.loads((REPO / "configs" / "prespecified_protocol.json").read_text())
    assert proto["seed"] == 20260913
    fs = proto["feature_sets"]
    assert len(fs["CORE_FEATURE_COLS"]) == 8 and len(fs["EXPANDED_FEATURE_COLS"]) == 17
    for f in ["input_manifest.json", "software_versions.json"]:
        json.loads((REPO / "configs" / f).read_text())

    # ---- 4. source data readable -------------------------------------------
    sd_root = REPO / "source_data"
    n_tables = 0
    for p in sorted(sd_root.rglob("*")):
        if p.suffix in {".csv", ".tsv", ".gz", ".json"}:
            if p.stat().st_size == 0:
                errs.append(f"empty source data: {p}")
                continue
            n_tables += 1
            if p.suffix == ".json":
                json.loads(p.read_text())
            elif p.suffix == ".gz":
                with gzip.open(p, "rt") as fh:
                    head = fh.readline()
                assert head.strip(), p
            else:
                with open(p, encoding="utf-8-sig") as fh:
                    head = fh.readline()
                assert "," in head or "\t" in head, p
    assert n_tables >= 40, n_tables

    # ---- 5. key scripts exist ----------------------------------------------
    must = [
        "scripts/phaseA_discovery/phaseA_clock_architecture.py",
        "scripts/phaseA_discovery/clock_architecture_umap.py",
        "scripts/figure2_distances/03_run_corrected_A1_A2.py",
        "scripts/figure2_distances/42_run_corrected_24feature_variance_partition.py",
        "scripts/figure3_pathways/52_update_ontology_and_frosted_fig3FG.py",
        "scripts/figure4_six_arm/62_run_static_3loop_architecture.py",
        "scripts/figure4_six_arm/64_run_residualized_coupling_and_matched_contrasts.py",
        "scripts/figure5_external_zt/78_LINEAGE_INTRINSIC_3LOOP_VALIDATION.py",
        "scripts/figure5_external_zt/82_FIX3_build_final_time_TF_story_panels.py",
        "scripts/figure6_phase_modules/tabula_fig6_strict_validation_v2_1.py",
        "scripts/figure6_phase_modules/tabula_fig6_M2_conditional_null_cv_v3.py",
        "scripts/figure6_phase_modules/plot_fig6_panels_ABCDE_v5_2.py",
        "scripts/figure7_ulcerative_colitis/scp259_overnight.py",
        "scripts/supplementary_s1_external/58_run_HCL_AHCA_Fig1_Fig2_single_panels.py",
        "scripts/supplementary_s3_tf_coupling/79_2_CLOCK_EXCLUDED_LINEAGE_ADJUSTED_TF_OVERNIGHT.py",
        "scripts/supplementary_s4_phase_families/tabula_predicted_phase_full_story_v2.py",
        "metadata/dataset_manifest.csv",
        "metadata/lineage_mapping.csv",
    ]
    for rel in must:
        if not (REPO / rel).exists():
            errs.append(f"missing: {rel}")

    # ---- 6. Fig 6e per-fold table present (the BLOCKED-render panel) --------
    per_fold = REPO / "source_data" / "Figure6" / "03_HELD_TISSUE_DONOR_DISJOINT_M2_CV_per_fold.csv"
    rows = read_rows(per_fold)
    folds = {r["held_tissue"] for r in rows if "held_tissue" in r}
    assert len(folds) >= 26, f"expected >=26 held tissues, got {len(folds)}"

    if errs:
        for e in errs:
            print("ERROR:", e)
        return 1
    print(f"SMOKE TEST PASS — gene sets OK, {n_tables} source tables readable, "
          f"{len(must)} key paths present, Fig6e per-fold table has {len(folds)} folds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
