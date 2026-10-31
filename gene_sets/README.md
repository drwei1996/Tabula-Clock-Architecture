# The three gene sets — read before using any of them

This paper uses **three distinct clock-gene lists**. They are NOT interchangeable, and two of them happen to both contain 17 genes while differing in membership. All three definitions below are programmatic (generated from the frozen protocol JSON, the frozen six-arm audit CSV, and the AnchorPhase prior freeze; see `tests/test_smoke_release.py` which asserts the differences).

## 1. `clock_core8.csv` — core clock, 8 genes (human symbols)

ARNTL, CLOCK, NPAS2, PER1, PER2, PER3, CRY1, CRY2.
Used for: core-8 UMAPs (Fig 1a), distance backbones, classification baselines.

## 2. `clock_static17.csv` — expanded static-expression set, 17 genes (human symbols)

core 8 **plus** NR1D1, NR1D2, RORA, RORB, RORC, DBP, TEF, HLF, NFIL3.
Used for: the 24-feature standardized expression space (Figs 1b–d, 2, S2), the six-arm grouping (`six_arm_mapping.csv` — this grouping is defined OVER the static 17), three-loop scores (Fig 4), external atlas replication (S1), and arm scores in Fig 5/7.
**This is a static-expression set. It is not the AnchorPhase anchor set.**

## 3. `anchorphase_core17_reference.csv` — AnchorPhase model anchors, 17 genes (mouse-case symbols as used by the frozen B0 fits)

Arntl, Bhlhe41, Clock, Cry1, Cry2, Dbp, Fbxo21, Hlf, Nfil3, Npas2, Nr1d1, Nr1d2, Per1, Per2, Per3, Rorc, Tef — with frozen external acrophase priors (phi_g, hours) from the AnchorPhase release.

Membership difference vs set 2 (program-verified):
- anchor set contains **BHLHE41, FBXO21** — not in the static 17;
- static 17 contains **RORA, RORB** — not anchor genes.

Used ONLY for: AnchorPhase/KATv2-B0 phase inference feeding Figure 6 modules, S4 phase families, and the 60-cohort phase atlas.

## 4. `six_arm_mapping.csv`

E-box / RORE / D-box activator–repressor arms over the static 17 (EBOX_POS: ARNTL;CLOCK;NPAS2 … DBOX_REP: NFIL3). Frozen from `62_STATIC_3LOOP_SCORE_DEFINITIONS.csv`.
