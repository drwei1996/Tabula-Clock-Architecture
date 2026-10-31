"""Independent tests/audit for completed local A1/A3; does not mutate frozen outputs."""
import unittest,json
from phaseA_unattended import *
from test_phaseA_resume import run as resume_regression

class CompletedPhaseATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=pd.read_csv(A1/'empirical_null_summary.csv');cls.null=pd.read_csv(A1/'empirical_null_distributions.csv')
        cls.results=pd.read_csv(A3/'all_pathway_associations.csv');cls.definitions=pd.read_csv(A3/'pathway_coverage.csv')

    def test_existing_resume_numerics(self):resume_regression()

    def test_legacy_numerics_and_frozen_figure_scope(self):
        # Historical suite assumed exactly5 total figures. New figures are authorized;
        # retain every numeric test and scope its figure-count assertion to its original files.
        from test_phaseA_clock_architecture import PhaseATests
        for name in ['test_adjusted_r2_against_direct_svd','test_stability_perfect_and_constant_lineage','test_remodeling_detects_tissue_signal','test_jackknife_formula']:
            case=PhaseATests(name);getattr(case,name)()
        original=['FigA2_clock_lineage_classifier_confusion','FigA3_clock_architecture_between_vs_within_distance','FigA4_lineage_specific_tissue_plasticity','FigA5_conserved_backbone_vs_context_remodeling','FigA8_integrated_clock_architecture_model']
        baseline=json.loads((AUD/'initial_frozen_hashes.json').read_text())
        self.assertEqual(len(original),5)
        for stem in original:
            path=FIG/(stem+'.png');self.assertTrue(path.exists());self.assertEqual(sha(path),baseline[str(path)])

    def test_integrity_and_dimensions(self):
        audit=json.loads((AUD/'dependency_integrity.json').read_text())
        self.assertEqual(audit['shape'],[2030,59410]);self.assertEqual(audit['original_gene_overlap'],68)
        self.assertLess(audit['max_discrepancy'],1e-14)
        self.assertEqual(frozen(),audit['A2_frozen'])
        for name,digest in audit['dependency_sha256'].items():self.assertEqual(sha(DEP/name),digest)

    def test_donor_splits_and_original_metrics(self):
        manifest=pd.read_csv(OUT/'01_matched_random_controls/classification_split_manifest.csv')
        self.assertEqual(len(manifest),1302);self.assertTrue(manifest.groupby('donor_id').test_fold.nunique().eq(1).all())
        observed=pd.read_csv(OUT/'01_matched_random_controls/observed_metrics.csv').set_index('panel')
        meta,x,genes=load_data();ev=Evaluator(meta,x);idx={g:i for i,g in enumerate(genes)};pr=protocol()
        self.assertEqual(len(ev.groups),97);self.assertEqual(len(ev.w),928);self.assertEqual(len(ev.b),139)
        for panel,key in SETS.items():
            metric=ev.evaluate([idx[c[5:]] for c in pr['feature_sets'][key]])
            for name in METRICS:self.assertAlmostEqual(metric[name],observed.loc[panel,name],places=10)

    def test_panel_counts_clock_exclusion_and_matching(self):
        clock={s[5:] for s in protocol()['feature_sets']['EXPANDED_FEATURE_COLS']}
        pools=pd.read_csv(A1/'candidate_pools.csv');stats=pd.read_csv(A1/'gene_matching_statistics.csv').set_index('gene')
        for variant,seed,k,dim in VARIANTS:
            for panel,slug,size in [('Core clock','core',8),('Expanded clock','expanded',17)]:
                p=pd.read_csv(A1/f'{variant}_{slug}_genes.csv');self.assertEqual(len(p),1000*size)
                self.assertTrue(p.groupby('iteration').matched_gene.nunique().eq(size).all());self.assertFalse(p.matched_gene.isin(clock).any())
                self.assertTrue(stats.loc[p.matched_gene,'eligible_nonclock'].all())
                allowed=set(zip(pools[pools.variant.eq(variant)].clock_gene,pools[pools.variant.eq(variant)].matched_gene))
                self.assertTrue(all(pair in allowed for pair in zip(p.clock_gene,p.matched_gene)))
                a=stats.loc[p.matched_gene,['standardized_log_mean','standardized_log_variance']].to_numpy();b=stats.loc[p.clock_gene,['standardized_log_mean','standardized_log_variance']].to_numpy()
                q=pd.read_csv(A1/'matching_quality.csv');q=q[q.variant.eq(variant)&q.panel.eq(panel)]
                expected=np.abs(a-b).reshape(1000,size,2).mean(1)
                np.testing.assert_allclose(expected,q[['mean_abs_logmean_z_difference','mean_abs_logvariance_z_difference']],rtol=1e-12,atol=1e-12)

    def test_empirical_p_and_bh_independently(self):
        self.assertEqual(len(self.null),8000)
        for r in self.summary.itertuples():
            values=self.null[self.null.panel.eq(r.panel)&self.null.variant.eq(r.variant)][r.metric].to_numpy()
            self.assertEqual(len(values),1000);self.assertAlmostEqual(r.P,(1+np.count_nonzero(values>=r.observed))/1001)
            self.assertAlmostEqual(r.null_sd,np.std(values,ddof=1))
        for variant,p in self.summary.groupby('variant'):
            np.testing.assert_allclose(p.BH_8_metrics,false_discovery_control(p.P.to_numpy()),rtol=1e-12)
        for collection,p in self.results.groupby('collection'):
            np.testing.assert_allclose(p.FDR,false_discovery_control(p.P.to_numpy()),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(p.tissue_adjusted_FDR,false_discovery_control(p.tissue_adjusted_P.to_numpy()),rtol=1e-12,atol=1e-12)

    def test_complete_collections_coverage_and_scoring(self):
        clock={s[5:] for s in protocol()['feature_sets']['EXPANDED_FEATURE_COLS']}
        self.assertEqual(self.definitions.groupby('collection').size().to_dict(),{'Hallmark':50,'Reactome':1839})
        self.assertEqual(self.results.groupby('collection').size().to_dict(),{'Hallmark':250,'Reactome':9195})
        for r in self.definitions.itertuples():
            members=set(str(r.genes_scored).split(';')) if pd.notna(r.genes_scored) else set()
            self.assertFalse(clock&members);self.assertEqual(len(members),r.n_scored);self.assertAlmostEqual(r.coverage,len(members)/r.n_original)
        # Independently verify one aggregate pathway score directly from the original matrix.
        meta,x,genes=load_data();phenotype=pd.read_csv(OUT/'03_unbiased_pathways/clock_remodeling_phenotype_donor_tissue_lineage.csv');p=phenotype.iloc[0]
        r=self.definitions[self.definitions.collection.eq('Hallmark')].iloc[0];idx={g:i for i,g in enumerate(genes)};ids=[idx[g] for g in r.genes_scored.split(';')]
        rows=meta.priority_lineage.eq(p.lineage)&meta.donor_id.eq(p.donor_id)&meta.tissue.eq(p.tissue)
        xx=x[:,ids];expected=((xx-xx.mean(0))/xx.std(0))[rows].mean()
        score=np.load(A3/'hallmark_scores.npz');j=int(np.flatnonzero(score['pathways']==r.pathway)[0]);self.assertAlmostEqual(expected,score['scores'][0,j],places=12)

    def test_sensitivity_sizes_and_pathway_deletion_summaries(self):
        genes=pd.read_csv(A1/'leave_one_gene_out.csv');self.assertEqual(len(genes),25)
        deletions=pd.read_csv(A1/'donor_tissue_deletions.csv');self.assertEqual(len(deletions),106)
        self.assertEqual(deletions.groupby('deletion').identifier.nunique().to_dict(),{'donor':24,'tissue':29})
        sizes=pd.read_csv(A1/'size_control_genes.csv');self.assertTrue(sizes.groupby(['kind','iteration']).gene.nunique().eq(8).all())
        for collection in ['Hallmark','Reactome']:
            for lin in LINEAGE_ORDER:
                data=np.load(A3/f'{collection.lower()}_{lin}_deletion_effects.npz');rows=self.results[self.results.collection.eq(collection)&self.results.lineage.eq(lin)].set_index('pathway').loc[data['pathways']]
                for kind in ['donor','tissue']:
                    arr=data[kind+'_beta'];np.testing.assert_allclose(rows[kind+'_sign_fraction'],np.mean(np.sign(arr)==np.sign(rows.beta.to_numpy()),axis=0))
                    np.testing.assert_allclose(rows[kind+'_beta_min'],np.min(arr,axis=0),equal_nan=True)

    def test_final_figures_and_frozen_files(self):
        result=final_integrity();self.assertGreaterEqual(len(result['figures_with_sources_and_all_formats']),16)
        self.assertEqual(result['unexpected_changed_files'],[])

if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main(verbosity=2)
