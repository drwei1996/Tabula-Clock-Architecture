# Analysis workflow (stage order and entry points)

Numbers in brackets are the frozen chain ids used in the analysis tree (kept here so results can be matched to the audit registers).

```
[Tabula 2.0 h5ad per tissue]
   └─ download_tabula_by_tissue.py ─→ pseudobulk units (≥20 cells; donor×tissue×assay×lineage)
        ├─ phaseA_clock_architecture.py + phaseA_unattended.py     [Phase A; seed 20260913]
        │    ├─ core8/static17 standardization (prespecified protocol)
        │    ├─ matched-random gene-panel controls (empirical nulls)
        │    ├─ backbone remodeling + centroid distances
        │    ├─ unbiased Hallmark/Reactome screens
        │    └─ A5 variance architecture
        ├─ Figure 1  ← clock_architecture_umap.py + 16/21/34/35 chains (UMAPs, heatmaps)
        ├─ Figure 2  ← 03_run_corrected_A1_A2.py (bootstrap 5000, seed 20260913)
        │              41_redraw_fig2_four_candidates_clean.py
        │              42_run_corrected_24feature_variance_partition.py (Shapley SS)
        ├─ Figure 3  ← 43 (module landscape) → 47/48/49/50 (enrichment chains)
        │              → 51 (redesign) → 52 (ontology freeze v3; 127/98/69/11)
        ├─ Figure 4  ← 62 (static 3-loop + six-arm) → 63 (deepdive/jackknife)
        │              → 64 (residualized coupling + matched contrasts) → 65–68 (render)
        ├─ Figure 5  ← 73/74 (GSE292219 mirror) → 76 (ZT story) → 77 FIX1
        │              → 78 (lineage-intrinsic validation; LOTO/perms seed 20261004)
        │              → 79/80 (DoRothEA clock-excluded TF) → 81 (TF coupling)
        │              → 82_FIX3 (final Figure-5 assembly)
        ├─ Suppl S1  ← 37/53–61 (HCL & AHCA replication + formal external validation)
        ├─ Suppl S2  ← Phase A robustness (H_A11 leave-one-gene-out; A5 robustness)
        ├─ Suppl S3  ← 79_2/80/81 tables + FULL_STORY Fig5U phase landscape
        │
        ├─ [AnchorPhase KATv2-B0 on 60 cohorts]  (external dependency)
        │    └─ 08_PHASE_ATLAS per-cohort CELL_PHASE.csv.gz
        │         ├─ Figure 6 ← tabula_fig3_fig6_layer2_layer3_v1.py (bridge W999)
        │         │            tabula_fig6_Hallmark_Reactome_M012_v1.py (identity-vs-clock)
        │         │            tabula_fig6_strict_validation_v2_1*.py (fixed matched nulls)
        │         │            tabula_fig6_M2_conditional_null_cv_v3.py (26-fold CV → 6e data)
        │         │            tabula_fig6_M2_common_support_v4.py
        │         │            plot_fig6_panels_ABCDE_v5_2.py (renders a–d)
        │         │            plot_fig6_main_and_supp_v2_1.py
        │         └─ Suppl S4 ← tabula_predicted_phase_full_story_v2.py (phase families)
        │                      tabula_full_biology_v3.py (module heatmaps)
        └─ Figure 7 / Suppl S5 ← scp259_overnight.py → run_tabula_scp259_biology_extension.py
                                  → scp259_figure_extension_v1.py (SCP259 UC)
```

## Stage-to-figure claims (key numbers → producing script)

| Claim (as printed on panel) | Producing script | Frozen table (shipped) |
|---|---|---|
| Fig2a 1.595 [0.926–1.902] | `03_run_corrected_A1_A2.py` | `source_data/Figure2/Fig2a_b_bootstrap5000.tsv` |
| Fig3e modules 127/98/69/11 | `52_update_ontology_and_frosted_fig3FG.py` | ontology freeze v3 (see audit CSV) |
| Fig5b LOO-ZT 0.433, P=0.018 | `78_LINEAGE_INTRINSIC_3LOOP_VALIDATION.py` §7 | `Fig5b_ZT_LOTO.csv` + `Fig5b_ZT_null_draws.gz` |
| Fig5c 0.77/0.53/0.85 (P=0.0002) | `78_...py` §4 | `Fig5c_atlas_LOTO.csv` |
| Fig5d r=.848/.790, P=.0083 | `78_...py` §6 + `82_FIX3` §D | `Fig5d_concordance.csv`, `82FIX3_MAIN_D...` |
| Fig6a 1,468 modules | `tabula_fig6_strict_validation_v2_1*.py` | `Fig6_provenance.json` |
| Fig6e M0/M1/M2 26-fold gain | `tabula_fig6_M2_conditional_null_cv_v3.py` | `03_HELD_TISSUE_DONOR_DISJOINT_M2_CV_per_fold.csv` + `03_M2_CV_SUMMARY.csv` |
| Fig7b 26/90 arms q<0.05; fib 1.798, B 0.628 (q=.0061) | `scp259_figure_extension_v1.py` | `Fig7_contrasts_all.csv` |
| S1e rho=1.00 (P=.0167); HCL 0.90 (P=.0833) | `55_draw_HCL_AHCA_formal_external_validation_v3.py` | `S1e_rank_concordance.csv` |
