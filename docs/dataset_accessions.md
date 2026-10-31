# Dataset accessions and download notes

| Dataset | Stable ID | Access | What to download |
|---|---|---|---|
| **Tabula Sapiens 2.0** (discovery atlas) | CELLxGENE collection `e5f58829-1a66-40b5-a624-9046778e74f5`; "All Cells" dataset `53d208b0-2cfd-4366-9866-c3c6114081bc`; figshare DOI `10.6084/m9.figshare.27921984.v1`; Cell 2026 `10.1016/j.cell.2026.08.010` (PMID 42685689) | https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5 | per-tissue h5ad via `scripts/preprocessing/download_tabula_by_tissue.py`; build donor×tissue×assay×lineage pseudobulks (≥20 cells/unit; 2,030 analyzed units) |
| HCL | Han et al., Nature 2020, `10.1038/s41586-020-2157-4` | original publication data links | tissue-level matrices -> pseudobulk lineage units |
| AHCA | **GSE159929** (He et al., Genome Biol 21:294, 2020) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159929 | count matrices + metadata; 59 units / 15 tissues / single donor structure |
| GSE292219 | **GSE292219**, samples GSM8853297–GSM8853302 (Veltri AJ et al., PMID 42397025) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE292219 | lineage×ZT mean normalized-10k matrices (6 ZT × 5 lineages); frozen as `75_74G_lineage_ZT_mean_norm10k.pkl` in the analysis tree |
| SCP259 | **SCP259** | https://singlecell.broadinstitute.org/single_cell/study/SCP259 | raw matrices + author cell labels (healthy / UC-uninflamed / UC-inflamed; 235 donor×condition×lineage units) |
| MSigDB Hallmark | `h.all.v2026.1.Hs.symbols.gmt` | https://www.gsea-msigdb.org (free registration; **not redistributable**) | the exact v2026.1 GMT |
| MSigDB Reactome | `c2.cp.reactome.v2026.1.Hs.symbols.gmt` | same | the exact v2026.1 GMT |
| DoRothEA | A/B/C human regulons (Garcia-Alonso et al., Genome Res 29:1363–1375, 2019) | https://github.com/saezlab/dorothea | regulon table; clock-17 genes excluded before scoring |

## Version pinning

- The **analysed atlas is Tabula Sapiens 2.0 (2026 Cell paper)**. The 2022 Science Tabula Sapiens (GSE201333, `10.1126/science.abl4896`) is the precursor resource and is cited as such; it is NOT the analysed version.
- MSigDB v2026.1 throughout; other versions change the pathway universe and the 127/98/69/11 module counts.
- Frozen unit metadata: `configs/mapping_freeze_manifest_v3.json`, `metadata/lineage_mapping.csv` (official cell-type → broad class crosswalk).

## Pseudobulk construction (frozen)

Per prespecified protocol (`configs/prespecified_protocol.json`): donor × tissue × assay × lineage units, minimum 20 cells, lineage set = B cell, T cell, macrophage/monocyte, fibroblast, endothelial (priority five passing thresholds), seed 20260913.
