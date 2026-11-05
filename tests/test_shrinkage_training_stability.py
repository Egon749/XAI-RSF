import numpy as np
import pytest
from dataclasses import replace
from crforest import CompetingRiskForest,CIFExpansion
from crforest.core import cluster_bootstrap,target_support_allows
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_shrinkage_stability_v1.common import resample,draw_indices,pair_stats,configuration
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage

def test_outer_draw_keeps_real_company_identity():
    d={'X':np.arange(120).reshape(60,2),'groups':np.arange(60),'time':np.arange(60)+1.,'event':np.arange(60)%3}
    a,ix=resample(d,2023,1);b,jx=resample(d,2023,1)
    np.testing.assert_array_equal(ix,jx)
    np.testing.assert_array_equal(a['groups'],d['groups'][ix])
    assert len(ix)==60 and len(np.unique(ix))<60
    assert not np.array_equal(ix,draw_indices(60,2023,2))
    with pytest.raises(ValueError):draw_indices(60,2023,0)
    inner=cluster_bootstrap(a['groups'],17,0)
    assert set(a['groups'][inner])<=set(a['groups'])
    # Two copies of ONE event company do not satisfy minimum2.
    assert not target_support_allows(np.array([True,True,False]),np.array([1,1,2]),np.array([7,7,8]),1,2)

def test_paired_resample_shrinkage_and_fixed_reference(tmp_path):
    rng=np.random.default_rng(991)
    n=80
    d={'X':rng.normal(size=(n,74)),'groups':np.arange(n),'time':rng.integers(1,700,n).astype(float),'event':np.arange(n)%3}
    d['X'][::7,2]=np.nan
    tr,ix=resample(d,2024,1)
    f=CompetingRiskForest(replace(configuration(17),n_trees=3)).fit(tr['X'],tr['time'],tr['event'],tr['groups'])
    for s in (0,10,18):
        q=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],s)
        assert check_shrinkage(f,q,tr,s)['independent_shrinkage_error']<1e-12
        ex=CIFExpansion.from_forest(q,d['X'])
        t=np.unique(np.r_[0,q.grid_,365,730]);p=q.predict(d['X'],t)
        total=np.broadcast_to(ex._coeff(t)[1],p.shape).copy()
        for key in [(j,) for j in ex.mains]+ex.pairs:total+=ex.component(d['X'],key,t)
        np.testing.assert_allclose(total,p,atol=1e-10,rtol=0)
        assert p.min()>=-1e-12 and p.sum(-1).max()<=1+1e-12
        assert np.diff(p,axis=1).min()>=-1e-12
    assert f.trees_==shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],0).trees_

def test_absolute_sd_alone_can_be_misleading():
    p=np.array([[.1,.2,.4],[.12,.19,.45],[.09,.23,.38]])
    a=pair_stats(p);b=pair_stats(p*.1)
    assert b['mean_company_sd']==pytest.approx(.1*a['mean_company_sd'])
    assert b['median_pair_rank_correlation']==pytest.approx(a['median_pair_rank_correlation'])
    assert b['between_company_sd']==pytest.approx(.1*a['between_company_sd'])
def test_constant_rank_not_claimed_perfect():
    z=pair_stats(np.full((4,5),.1))
    assert np.isnan(z['median_pair_rank_correlation'])
    assert z['undefined_rank_pairs']==6
