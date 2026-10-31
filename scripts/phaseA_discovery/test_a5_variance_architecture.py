import unittest
from a5_variance_architecture import *

class TestPartition(unittest.TestCase):
    def setUp(self):
        rows=list(itertools.product(['L1','L2','L3'],['T1','T2','T3','T4'],['D1','D2','D3'],range(2)))
        self.meta=pd.DataFrame(rows,columns=['priority_lineage','tissue','donor_id','replicate']);self.model=Partition(self.meta)
    def test_pure_factors_and_interaction(self):
        l=self.meta.priority_lineage.map({'L1':-1,'L2':0,'L3':1}).to_numpy();t=self.meta.tissue.map({'T1':-1,'T2':-1,'T3':1,'T4':1}).to_numpy();d=self.meta.donor_id.map({'D1':-1,'D2':0,'D3':1}).to_numpy()
        y=np.column_stack([l,t,d,l*t,(-1.)**self.meta.replicate])
        components,f,var=self.model.fit(y)
        np.testing.assert_allclose(f,np.eye(5),atol=1e-12);np.testing.assert_allclose(components.sum(1),var,atol=1e-12)
    def test_unbalanced_direct_projection_and_invariance(self):
        meta=self.meta.iloc[np.arange(len(self.meta))%5!=0].reset_index(drop=True);model=Partition(meta);y=np.random.default_rng(42).normal(size=(len(meta),4));c,f,v=model.fit(y)
        _,f2,_=model.fit(3*y+100);np.testing.assert_allclose(f,f2,atol=1e-12)
        design=np.column_stack([np.ones(len(meta)),pd.get_dummies(meta.priority_lineage+'|'+meta.tissue,dtype=float),pd.get_dummies(meta.donor_id,dtype=float)])
        residual=y-design@np.linalg.lstsq(design,y,rcond=None)[0];np.testing.assert_allclose(c[:,4],np.mean(residual**2,axis=0),atol=1e-12)
        np.testing.assert_allclose(c.sum(1),v,atol=1e-12)
        perm=np.random.default_rng(1).permutation(len(meta));_,fp,_=Partition(meta.iloc[perm]).fit(y[perm]);np.testing.assert_allclose(f,fp,atol=1e-12)
    def test_zero_and_empirical_ties(self):
        _,f,_=self.model.fit(np.ones((len(self.meta),1)));self.assertTrue(np.isnan(f).all())
        self.assertEqual(empirical([1,1,1],1,True),1);self.assertEqual(empirical([1,2,3],4,False),.25)
    def test_missing_combinations(self):
        m=self.meta[~((self.meta.priority_lineage=='L1')&(self.meta.tissue=='T1'))];model=Partition(m)
        self.assertEqual(model.diagnostics['observed_pairs'],11);self.assertGreater(model.diagnostics['residual_df'],0)

if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
