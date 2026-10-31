# Tabula-Clock-Architecture

**Lineage-associated organization of clock gene expression across human tissues**

Code and data release for the Tabula Sapiens circadian-architecture paper (Nature Communications submission). The paper maps how clock-gene expression is organized across ~2,030 donor–tissue–lineage units of the **Tabula Sapiens 2.0** atlas, validates the architecture in three external human resources (HCL, AHCA, and the true-ZT cohort GSE292219), resolves a six-arm E-box/RORE/D-box regulatory decomposition, reconstructs phase-resolved functional modules with the AnchorPhase model, and characterizes their disruption in ulcerative colitis (SCP259).

> **Companion paper.** The phase-inference method used for Figure 6 (AnchorPhase / KATv2-B0) is published separately as a methods paper with its own repository. This repository consumes it as a pinned external dependency; see §AnchorPhase dependency and `docs/anchorphase_dependency.md`.

---

## 1. Study and figures

| Figure | Content |
|---|---|
| Fig 1 | core-8 / expanded-17 clock UMAPs, 17-gene lineage-ordered heatmap, 5-lineage heatmaps (a–d) |
| Fig 2 | between- vs within-lineage expression distance 1.595 [0.926–1.902], cross-tissue architecture, 24-feature variance partition (a–e) |
| Fig 3 | identity-vs-tissue pathway coupling (Hallmark/Reactome matched-competitive nulls), module ontology 127/98/69/11 (a–f) |
| Fig 4 | three-loop and six-arm scores, activator/repressor balance, matched lineage contrasts, residualized coupling, incremental R² (a–f) |
| Fig 5 | GSE292219 true-ZT trajectories, leave-one-ZT-out portability, three-atlas LOTO 0.77/0.53/0.85, cross-atlas prototypes r=.848/.790, TF coupling (a–e) |
| Fig 6 | AnchorPhase-phase-resolved functional modules: counts (a), shared programs (b), inflammatory composite + B-vs-Mac amplitude (c), tissue-specific curves (d), M0/M1/M2 26-fold out-of-sample gain (e). **Five panels a–e; there is no panel 6f.** |
| Fig 7 | SCP259 ulcerative colitis: 3-state scores, arm contrasts, six-arm distance to healthy reference, B-cell TF differences, curated functional contrasts (a–f) |
| Suppl. Figs 1–5 | AHCA/HCL external validation; leave-one-gene-out robustness; B0 phase landscape + TF coupling; phase families & module estimates; UC B-cell subtypes |

Final frozen TIFFs: `Figure1.tif` … `Figure7.tif`, `Figure S1.tif` … `Figure S5.tif` (in the journal submission package; panel-level source data are here under `source_data/`).

## 2. Repository layout

```
gene_sets/            the three DISTINCT gene sets (see §Gene sets — do not conflate)
metadata/             lineage mapping, dataset manifest, panel gene definitions, B0 run manifest
scripts/
  preprocessing/            Tabula 2.0 download helper
  phaseA_discovery/         prespecified Phase-A pipeline (UMAP, nulls, pathways, variance architecture)
  figure2_distances/        A1/A2 bootstrap distances + variance partition
  figure3_pathways/         module landscape + matched-competitive enrichment chains
  figure4_six_arm/          static three-loop / six-arm chains 62–68
  figure5_external_zt/      73/74/76/78/81/82_FIX3 chains (GSE292219 + external atlases + TF)
  figure6_phase_modules/    AnchorPhase-dependent module analyses + M0/M1/M2 validation + Fig6 plots
  figure7_ulcerative_colitis/  SCP259 disease pipeline
  supplementary_s1_external/   HCL/AHCA replication chains
  supplementary_s3_tf_coupling/  DoRothEA clock-excluded TF coupling
  supplementary_s4_phase_families/  B0 phase-atlas derived families
source_data/          Figure1..Figure7 + SupplementaryFigure1..5 (per-panel tables)
configs/              prespecified protocol (seeds, feature sets), input/software manifests
docs/                 figure→code map, dataset accessions, workflow, AnchorPhase dependency
tests/                gene-set integrity + config load smoke tests
```

## 3. Gene sets — three distinct lists (never conflate)

| Set | File | n | Definition |
|---|---|---|---|
| Core clock 8 | `gene_sets/clock_core8.csv` | 8 | ARNTL, CLOCK, NPAS2, PER1–3, CRY1–2 (static analyses backbone) |
| Static expanded 17 | `gene_sets/clock_static17.csv` | 17 | core 8 + NR1D1/2, RORA/B/C, DBP, TEF, HLF, NFIL3 — human symbols; powers the 24-feature space and the six-arm grouping |
| AnchorPhase anchor 17 | `gene_sets/anchorphase_core17_reference.csv` | 17 | the model's fixed-anchor list **in mouse-case symbols as actually used by the B0 fits on Tabula** (Arntl … Tef). Contains **BHLHE41 and FBXO21; does NOT contain RORA or RORB** |

The two 17-gene lists share 15 genes and differ by exactly {BHLHE41, FBXO21} vs {RORA, RORB}. `gene_sets/six_arm_mapping.csv` defines the E-box/RORE/D-box activator–repressor arms over the static 17.

## 4. Installation

```bash
# core analysis environment (frozen pins are in docs/)
pip install numpy pandas scipy scikit-learn matplotlib statsmodels h5py anndata umap-learn numba
# AnchorPhase-dependent analyses (Figure 6, S4) additionally need the AnchorPhase repo:
pip install -e /path/to/AnchorPhase
```

Exact frozen versions: `configs/software_versions.json` (discovery stage), `docs/requirements-*-20260916.txt` (plotting/UMAP overlays), and the AnchorPhase `environment.yml` for the phase model.

## 5. Minimal running example

```bash
python tests/test_smoke_release.py
```

Verifies offline: the three gene sets load and differ exactly as specified; configs parse; every shipped source-data file is readable with a header; key scripts exist. No downloads required.

## 6. Full workflow (top-level order)

1. **Preprocess** — `scripts/preprocessing/download_tabula_by_tissue.py` (CELLxGENE, per-tissue h5ad; pseudobulk to donor×tissue×assay×lineage units; ≥20 cells/unit). Frozen unit metadata: `configs/mapping_freeze_manifest_v3.json` + `metadata/lineage_mapping.csv`.
2. **Phase A discovery** (Figs 1–3 backbone, S2) — `scripts/phaseA_discovery/phaseA_clock_architecture.py` then `phaseA_unattended.py` (matched-random controls, backbone remodeling, unbiased pathway screens, A5 variance architecture). Prespecified protocol & seed 20260913: `configs/prespecified_protocol.json`.
3. **Figure 2** — `scripts/figure2_distances/03_run_corrected_A1_A2.py` (crossed donor/tissue bootstrap, 5,000 iters, seed 20260913), `41_redraw_fig2_four_candidates_clean.py`, `42_run_corrected_24feature_variance_partition.py` (finite-design hierarchical Shapley SS).
4. **Figure 3** — chains 43→52 in `scripts/figure3_pathways/` (tissue-responsive / celltype-adjusted / matched-competitive enrichment; ontology freeze v3: module counts 127/98/69/11).
5. **Figure 4** — chains 62→68 in `scripts/figure4_six_arm/` (static three-loop + six-arm scores from the expanded-17 grouping; residualized coupling; matched same-donor/same-tissue contrasts).
6. **Figure 5** — `scripts/figure5_external_zt/`: 73/74 (GSE292219 mirror), 76 (ZT story), 78 (lineage-intrinsic validation incl. leave-one-ZT-out with within-ZT label permutation ×5,000 seed 20261004, three-atlas LOTO, prototypes exact 120-perm), 81 (TF coupling), 82_FIX3 (final Figure-5 panel assembly).
7. **Figure 6** — `scripts/figure6_phase_modules/`: `tabula_fig3_fig6_layer2_layer3_v1.py` (layer bridge), `tabula_fig6_strict_validation_v2_1*.py` (M0/M1/M2 with fixed matched gene-set nulls), `tabula_fig6_M2_conditional_null_cv_v3.py` (**26-fold held-tissue donor-disjoint CV — per-fold rows shipped in `source_data/Figure6/03_HELD_TISSUE_DONOR_DISJOINT_M2_CV_per_fold.csv`**), `tabula_fig6_M2_common_support_v4.py`, `plot_fig6_panels_ABCDE_v5_2.py` (panels a–d render), `plot_fig6_main_and_supp_v2_1.py`.
8. **Figure 7 + S5** — `scripts/figure7_ulcerative_colitis/`: `scp259_overnight.py` (six-arm + TF tests on donor×condition×lineage), `run_tabula_scp259_biology_extension.py`, `scp259_figure_extension_v1.py` (distance/forest/TF/pathway panels).
9. **S1** — `scripts/supplementary_s1_external/` (HCL/AHCA Fig1/Fig2-like replication, AHCA Fig3, formal external validation figures).

## 7. Source Data

`source_data/Figure{1..7}/` and `source_data/SupplementaryFigure{1..5}/` hold the frozen per-panel tables backing every quantitative panel (25 core tables carry SHA-256 in the audit `SOURCE_DATA_INDEX.csv`; panel-level mapping in `docs/figure_code_mapping.csv`). Large single-cell matrices are not redistributed — see `docs/dataset_accessions.md`.

## 8. Frozen versions

- Atlas: Tabula Sapiens 2.0 (Cell 2026; CELLxGENE collection `e5f58829-1a66-40b5-a624-9046778e74f5`).
- Pathways: MSigDB v2026.1 (Hallmark + Reactome c2.cp).
- TF: DoRothEA A/B/C regulons (Garcia-Alonso et al. 2019), clock-17 genes excluded.
- Phase model: AnchorPhase/KATv2-B0 corrected-offset, frozen 2026-10-07; 60-cohort phase atlas over Tabula (`metadata/anchorphase_phase_atlas_run_manifest.csv`).

## 9. Seeds & statistics

| Analysis | Seed | Resampling |
|---|---|---|
| A2 distances / UMAP | 20260913 | 5,000 bootstrap |
| LOTO / ZT permutations | 20261004 (+4242 for ZT draws) | 5,000 perms; exact 120-perm cross-atlas; 720 external true-ZT |
| phase-matched six-arm | 20261010 | 4,999 perms |
| phase-model engine | 20261007 | deterministic EM |
Eligibility: posterior R ≥ 0.3 and ≥2 anchored genes; ≥20 cells per pseudobulk. BH FDR throughout; per-family nulls as stated per panel.

## 10. Compute

Static analyses (Figs 1–5, S1–S3): laptop-scale (≤64 GB RAM; the 24-feature partition and pathway screens are minutes–hours CPU). Figure 6 phase-model stages: CPU (torch, 1 thread/fit, ~1 core-day for the full atlas). SCP259 overnight chain: ~2–4 h CPU.

## 11. Reproduction limits (documented, not hidden)

- **Figure 6e (M0/M1/M2 per-fold panel):** the per-fold 26-fold CV **data** are shipped (`source_data/Figure6/03_HELD_TISSUE_DONOR_DISJOINT_M2_CV_per_fold.csv`, produced by `tabula_fig6_M2_conditional_null_cv_v3.py`), and the validation framework is complete — but the **specific panel-render script that drew the final 6e graphic from those per-fold rows was not recovered** from the analysis tree (searched project-wide; recorded as the single BLOCKED provenance item in the V4 verification register). Re-rendering 6e pixel-exactly therefore requires re-plotting from the shipped table; all numbers on the panel are verifiable from it.
- **Figure 6c (right scatter):** panel content verified; the amplitude CSVs are shipped, the exact render PNG for the right half was not pinned (left-half curves fully traced).
- Raw matrices (Tabula 2.0, HCL, AHCA, GSE292219, SCP259) and MSigDB GMTs must be downloaded (accessions in `docs/dataset_accessions.md`).
- Full-resolution frozen TIFFs live in the journal submission package, not in this repository.

## 12. Citation

```bibtex
@software{tabulaclock2026,
  title  = {Lineage-associated organization of clock gene expression across
            human tissues — analysis code and source data},
  author = {Tabula Clock Architecture authors},
  year   = {2026},
  version = {1.0.0},
  doi    = {ZENODO-DOI-PLACEHOLDER},
  url    = {https://github.com/PLACEHOLDER-ORG/Tabula-Clock-Architecture}
}
```

## 13. License

Code: MIT. Derived tables: CC BY 4.0. Third-party data under original terms (not redistributed). See `LICENSE`.

## 14. Contact / contributions

Analysis questions: open a GitHub issue in this repository. Model questions (AnchorPhase): the companion repository.

## 15. Relationship to the AnchorPhase methods paper

AnchorPhase = the phase-inference method (methods paper, separate repository: model, benchmarks, external validation, colorectal-cancer application). This repository = the biological application to human atlases. Overlap is deliberate and one-directional: Figure 6 / S4 here consume AnchorPhase outputs through the pinned release; no model-development code is duplicated. Cite both papers independently.
