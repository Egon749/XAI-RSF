import numpy as np
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_shrinkage_fine_v1.common import STRENGTHS,REUSED,NEW,select_strength
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage

def test_exact_integer_grid_and_reuse_partition():
    assert STRENGTHS==tuple(range(21))
    assert len(REUSED)==6 and len(NEW)==15 and not set(REUSED)&set(NEW)
    assert set(REUSED)|set(NEW)==set(STRENGTHS)
    rows=[dict(strength=s,within_year_auc=.8,ap=.3,ipcw_brier=.1) for s in STRENGTHS]
    assert select_strength(rows)[0]==10

def test_each_integer_preserves_raw_structure_probabilities_and_full_functions():
    rng=np.random.default_rng(1105);X=rng.normal(size=(50,3));X[::8,1]=np.nan
    t=rng.integers(1,30,50).astype(float);e=rng.integers(0,3,50);g=np.arange(50)
    raw=CompetingRiskForest(ForestConfig(n_trees=2,max_depth=3,path_budget=2,min_leaf=4,min_events=1,
        min_groups=3,mtry=3,n_thresholds=3,seed=17,cut_strategy='shared_quantile')).fit(X,t,e,g)
    tr=dict(X=X,time=t,event=e,groups=g);times=np.r_[0.,raw.grid_,100.];last=2.
    for strength in STRENGTHS:
        sh=shrink_leaf_cifs(raw,X,t,e,g,strength);c=check_shrinkage(raw,sh,tr,strength)
        assert c['mean_leaf_weight']<=last;last=c['mean_leaf_weight']
        p=sh.predict(X,times);ex=CIFExpansion.from_forest(sh,X)
        np.testing.assert_allclose(ex.predict(X,times),p,atol=1e-10,rtol=0)
        assert p.min()>=0 and p.sum(-1).max()<=1+1e-12 and np.diff(p,axis=1).min()>=-1e-12
