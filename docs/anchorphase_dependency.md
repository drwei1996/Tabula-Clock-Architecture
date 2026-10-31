# AnchorPhase dependency (Figure 6, Supplementary Figure 4)

## What is used

AnchorPhase (KATv2-B0, corrected-offset variant) is an **external, separately published dependency** of this repository. It is used for exactly two things:

1. per-cell circadian phase posteriors over the 60 Tabula Sapiens 2.0 assay×tissue cohorts (`katv2/08_PHASE_ATLAS` in the analysis tree; run manifest shipped at `metadata/anchorphase_phase_atlas_run_manifest.csv`);
2. those phases as the time axis for the functional-module analyses of Figure 6 (module harmonic fits, M0/M1/M2 comparison) and the phase families of Supplementary Figure 4.

It is **not** used for Figures 1–5 static analyses or Figure 7 six-arm scores.

## Pinning

| Field | Value |
|---|---|
| Software | AnchorPhase, planned release `v1.0.0` |
| Frozen model | KATv2-B0 corrected offset (task 10A), frozen 2026-10-07 |
| Anchor priors | 17 genes, fixed phi_g; SHA-256 of prior source recorded in the AnchorPhase release (`configs/ACROPHASE_PRIOR_FREEZE.json`) |
| Engine seed | 20261007 (deterministic EM/L-BFGS) |
| Repository | https://github.com/drwei1996/AnchorPhase |
| Git pin | `6c1ec50498d05e6d9321d9770813dc7804024f81` |
| Archive DOI | Pending Zenodo archival of the GitHub v1.0.0 release |
  
## Input/output contract

- Input: raw single-cell UMI counts for the 17 anchor genes, together with total UMI counts across all genes for each cell. The sequencing-depth offset is calculated using the total UMI count.
- Output consumed here: per-cell `phase_mean_h` (0–24 h) and `posterior_R`. Downstream analyses retain cells with posterior R ≥ 0.3 and at least two detected anchor genes. Eligible cells are then aggregated when required for downstream analyses.
- Example rebuild call (after installing AnchorPhase):

```bash
python scripts/figure6_phase_modules/tabula_fig6_strict_validation_v2_1.py   # consumes the frozen atlas scores
python scripts/figure6_phase_modules/tabula_fig6_M2_conditional_null_cv_v3.py # 26-fold held-tissue CV (6e data)
```

## What is deliberately NOT here

No model-development code (training benchmarks, comparators, falsification pipeline) and no GSE166555 application code — those belong exclusively to the AnchorPhase methods repository. The two papers cite each other by versioned reference; nothing is copied between repositories except this declaration.
