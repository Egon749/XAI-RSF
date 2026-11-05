import copy
import numpy as np
import pytest
from crforest import CompetingRiskForest, ForestConfig, CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_shrinkage_strength_v1.common import STRENGTHS, select_strength
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage

def test_complete_strength_grid_preserves_structure_and_exactness():
    rng=np.random.default_rng(614)
    X=rng.normal(size=(60,4));X[::7,0]=np.nan
    t=rng.integers(1,35,60).astype(float);e=rng.integers(0,3,60);g=np.arange(60)
    tr=dict(X=X,time=t,event=e,groups=g)
    raw=CompetingRiskForest(ForestConfig(n_trees=2,max_depth=4,path_budget=2,min_leaf=4,
        min_groups=3,min_events=1,mtry=4,n_thresholds=3,seed=17,cut_strategy='shared_quantile')).fit(X,t,e,g)
    before=copy.deepcopy(raw.trees_);last_weight=2.
    for strength in STRENGTHS:
        f=shrink_leaf_cifs(raw,X,t,e,g,strength)
        ck=check_shrinkage(raw,f,tr,strength);assert ck['mean_leaf_weight']<=last_weight
        last_weight=ck['mean_leaf_weight'];assert raw.trees_==before
        ex=CIFExpansion.from_forest(f,X);times=np.r_[0.,f.grid_,100.];pred=f.predict(X,times)
        assert pred.min()>=0 and pred.sum(-1).max()<=1+1e-12 and np.diff(pred,axis=1).min()>=-1e-12
        np.testing.assert_allclose(ex.predict(X,times),pred,atol=1e-10,rtol=0)
    with pytest.raises(AssertionError):check_shrinkage(raw,f,tr,10)
    with pytest.raises(ValueError):shrink_leaf_cifs(f,X,t,e,g,10)

def test_forward_selection_ties_and_no_event_fallback():
    rows=[dict(strength=s,within_year_auc=.7,ap=.2,ipcw_brier=.1) for s in STRENGTHS]
    assert select_strength(rows)[0]==10
    rows[0]['within_year_auc']=.8;assert select_strength(rows)[0]==0
    for r in rows:r['within_year_auc']=float('nan')
    assert select_strength(rows)[0]==10
