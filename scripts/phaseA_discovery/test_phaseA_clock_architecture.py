"""Independent synthetic/sanity checks for new Phase A core functions."""
import unittest
from phaseA_clock_architecture import *

class PhaseATests(unittest.TestCase):
    def test_adjusted_r2_against_direct_svd(self):
        rng=np.random.default_rng(3)
        m=pd.DataFrame({'donor_id':np.repeat(['a','b','c'],10),'tissue':np.tile(['x','y'],15)})
        y=rng.normal(size=(30,2))
        d=np.column_stack([np.ones(30),pd.get_dummies(m.donor_id,dtype=float),pd.get_dummies(m.tissue,dtype=float)])
        coef=np.linalg.pinv(d)@y; rank=np.linalg.matrix_rank(d)
        expected=1-((y-d@coef)**2).sum(0)/((y-y.mean(0))**2).sum(0)*29/(30-rank)
        actual,r=adjusted_r2(m,y,['donor_id','tissue'])
        np.testing.assert_allclose(actual,expected,atol=1e-12);self.assertEqual(r,rank)

    def test_stability_perfect_and_constant_lineage(self):
        m=pd.DataFrame([{'priority_lineage':l,'tissue':t,'donor_id':d} for l in LINEAGE_ORDER for t in ['x','y','z'] for d in ['a','b','c']])
        y=np.array([[LINEAGE_ORDER.index(l)] for l in m.priority_lineage],float)
        np.testing.assert_allclose(stability(m,y),1)
        y=np.array([[['x','y','z'].index(t)] for t in m.tissue],float)
        np.testing.assert_allclose(stability(m,y),0)

    def test_remodeling_detects_tissue_signal(self):
        rng=np.random.default_rng(12)
        m=pd.DataFrame([{'priority_lineage':l,'tissue':t,'donor_id':d} for l in LINEAGE_ORDER for t in ['x','y','z'] for d in ['a','b','c','d']])
        y=np.array([[['x','y','z'].index(t)] for t in m.tissue],float)+rng.normal(0,.01,(len(m),1))
        mean,each=remodeling(m,y);self.assertGreater(mean[0],.95);self.assertEqual(each.shape,(5,1))

    def test_jackknife_formula(self):
        values=np.array([1.,2.,3.,4.,5.]);draws=np.array([np.delete(values,i).mean() for i in range(5)])
        low,high,se=jackknife_interval(values.mean(),draws)
        self.assertAlmostEqual(se,values.std(ddof=1)/np.sqrt(5));self.assertLess(low,3);self.assertGreater(high,3)

    def test_outputs_and_donor_separation(self):
        manifest=pd.read_csv(OUT/'01_matched_random_controls/classification_split_manifest.csv')
        self.assertTrue(manifest.groupby('donor_id').test_fold.nunique().eq(1).all())
        self.assertEqual(len(manifest),1302);self.assertFalse(manifest.group_id.duplicated().any())
        metrics=pd.read_csv(OUT/'01_matched_random_controls/observed_metrics.csv')
        self.assertTrue(metrics.empirical_P.isna().all());self.assertTrue(metrics.n_null_panels.eq(0).all())
        for panel in metrics.panel:
            p=pd.read_csv(OUT/'source_data/FigA2_predictions.csv');p=p[p.panel.eq(panel)]
            self.assertAlmostEqual(balanced_accuracy_score(p.observed,p.predicted),metrics.set_index('panel').loc[panel,'balanced_accuracy'])
        scores=pd.read_csv(OUT/'statistics/feature_effects_and_scores.csv');self.assertEqual(len(scores),25)
        self.assertTrue(np.isfinite(scores.select_dtypes('number')).all().all())
        s=pd.read_csv(OUT/'statistics/distance_driver_sensitivity.csv');self.assertEqual(len(s),53)
        self.assertTrue(s.delta.gt(0).all())
        phenotype=pd.read_csv(OUT/'03_unbiased_pathways/clock_remodeling_phenotype_original_observations.csv')
        self.assertEqual(len(phenotype),936);self.assertTrue(phenotype.distance.ge(0).all())
        for i in [1,6,7]:self.assertEqual(len(list((OUT/'figures').glob(f'FigA{i}_*'))),0)
        self.assertEqual(len(list((OUT/'figures').glob('*.png'))),5)

if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main(verbosity=2)
