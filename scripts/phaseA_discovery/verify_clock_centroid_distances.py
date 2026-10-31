"""Independent validation of saved centroid/distance outputs and visible points."""
import json
from itertools import combinations

import clock_centroid_distances as work
from clock_centroid_distances import np,pd,plt,OUTPUT,WITHIN,BETWEEN,LINEAGE_ORDER
from scipy.spatial.distance import euclidean
from PIL import Image

def main():
    meta,x,_,threshold=work.load_frozen_inputs()
    table=pd.read_csv(OUTPUT/'clock_centroid_tissue_displacement.csv')
    pairs=pd.read_csv(OUTPUT/'clock_between_within_distances.csv').fillna('')
    summary=pd.read_csv(OUTPUT/'clock_distance_summary.csv')
    boot=pd.read_csv(OUTPUT/'clock_distance_bootstrap_draws.csv')
    expected_centroids=[]
    for _,row in table.iterrows():
        ids=meta.index[meta.priority_lineage.eq(row.lineage)&meta.tissue.eq(row.tissue)]
        center=x.loc[ids].to_numpy().mean(axis=0)
        expected_centroids.append(center)
        lineage_ids=meta.index[meta.priority_lineage.eq(row.lineage)]
        overall=x.loc[lineage_ids].to_numpy().mean(axis=0)
        np.testing.assert_allclose(row.distance_to_lineage_centroid,euclidean(center,overall),rtol=1e-12)
        np.testing.assert_allclose([row.centroid_PC1,row.centroid_PC2],meta.loc[ids,['PC1','PC2']].mean(),rtol=1e-12,atol=1e-14)
        assert len(ids)==row.n_observations and meta.loc[ids].donor_id.nunique()==row.n_donors
    expected_centroids=np.array(expected_centroids)
    support=table[table.n_donors>=threshold]
    expected_within=sum(len(list(combinations(g.index,2))) for _,g in support.groupby('lineage'))
    expected_between=sum(len(list(combinations(g.index,2))) for _,g in support.groupby('tissue'))
    assert pairs.comparison_type.eq(WITHIN).sum()==expected_within
    assert pairs.comparison_type.eq(BETWEEN).sum()==expected_between
    assert not pairs.duplicated(['comparison_type','centroid_index_A','centroid_index_B']).any()
    for _,row in pairs.iterrows():
        a=int(row.centroid_index_A); b=int(row.centroid_index_B)
        assert table.loc[a].n_donors>=threshold and table.loc[b].n_donors>=threshold
        if row.comparison_type==WITHIN:
            assert table.loc[a].lineage==table.loc[b].lineage and table.loc[a].tissue!=table.loc[b].tissue
        else:
            assert table.loc[a].tissue==table.loc[b].tissue and table.loc[a].lineage!=table.loc[b].lineage
        np.testing.assert_allclose(row.distance,euclidean(expected_centroids[a],expected_centroids[b]),rtol=1e-12)
    # Equal block weights recover the standard sample median; ties and even n included.
    for values in [np.array([1.,3.,8.]),np.array([1.,3.,8.,10.]),np.array([1.,1.,1.,3.])]:
        assert work.weighted_median(values,np.ones(len(values)))==np.median(values)
    assert len(boot)>=5000 and np.isfinite(boot.to_numpy()).all()
    np.testing.assert_allclose(boot.delta,boot.median_between-boot.median_within,rtol=1e-12,atol=1e-14)
    effect=summary[summary.summary_type.eq('effect')].iloc[0]
    np.testing.assert_allclose([effect.bootstrap_95_ci_low,effect.bootstrap_95_ci_high],boot.delta.quantile([.025,.975]),rtol=1e-12)
    work.apply_publication_style()
    fig=work.figure_B(pairs,summary)
    n_B=0
    for collection in fig.axes[0].collections:
        if collection.get_gid() in (WITHIN,BETWEEN):
            expected=pairs[pairs.comparison_type.eq(collection.get_gid())].distance
            np.testing.assert_array_equal(collection.get_offsets()[:,1],expected)
            n_B+=len(expected)
    assert n_B==len(pairs); plt.close(fig)
    fig=work.figure_supp(pairs); n_supp=0
    for collection in fig.axes[0].collections:
        if collection.get_gid() in LINEAGE_ORDER:
            expected=pairs[pairs.comparison_type.eq(WITHIN)&pairs.lineage.eq(collection.get_gid())].distance
            np.testing.assert_array_equal(collection.get_offsets()[:,0],expected)
            n_supp+=len(expected)
    assert n_supp==expected_within; plt.close(fig)
    images={}
    for name in ['fig_clock_centroid_tissue_displacement','fig_between_vs_within_clock_distance','fig_within_tissue_distance_by_lineage_supp']:
        with Image.open(OUTPUT/f'{name}.png') as im:
            dpi=im.info['dpi']; assert min(dpi)>299.9
            images[name]={'dimensions':im.size,'dpi':dpi}
        assert (OUTPUT/f'{name}.pdf').read_bytes().startswith(b'%PDF-')
    result={'centroids_independently_verified':len(table),'all_qualified_pairs_present':len(pairs),
            'within_pairs':expected_within,'between_pairs':expected_between,
            'all_pairs_visible_in_figure_B':n_B,'all_within_pairs_visible_in_supplement':n_supp,
            'bootstrap_iterations':len(boot),'bootstrap_CI_matches_draws':True,'images':images}
    (OUTPUT/'independent_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
