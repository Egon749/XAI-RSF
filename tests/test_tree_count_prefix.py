import json
import numpy as np
import pytest
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_tree_count_v1.nested import grow_range,assemble,prefix

def test_nested_prefix_and_disjoint_chunks_equal_fresh_fit_with_shrinkage(tmp_path):
    rng=np.random.default_rng(1003);x=rng.normal(size=(90,5));x[::9,1]=np.nan
    tr=dict(X=x,time=rng.integers(1,400,90).astype(float),event=np.tile([0,1,2],30),groups=np.arange(90))
    cfg=ForestConfig(n_trees=8,path_budget=2,max_depth=3,min_leaf=5,min_groups=3,min_events=1,
        mtry=5,tree_feature_count=4,n_thresholds=3,cut_strategy='shared_quantile',seed=42,split_min_target_groups=2)
    from dataclasses import replace
    old=CompetingRiskForest(replace(cfg,n_trees=3)).fit(x,tr['time'],tr['event'],tr['groups'])
    baseline=grow_range(cfg,tr,0,3)
    old_counts=[v['candidate_evaluations']for v in baseline]
    path=tmp_path/'old.json';old.save(path);old=CompetingRiskForest.load(path)
    chunks=grow_range(cfg,tr,5,8)+grow_range(cfg,tr,3,5)
    extended,counts=assemble(old,chunks,8,old_counts)
    fresh=CompetingRiskForest(cfg).fit(x,tr['time'],tr['event'],tr['groups'])
    assert json.dumps(extended.trees_)==json.dumps(fresh.trees_)
    assert extended.candidate_evaluations_==fresh.candidate_evaluations_
    a=shrink_leaf_cifs(extended,x,tr['time'],tr['event'],tr['groups'],10)
    b=shrink_leaf_cifs(fresh,x,tr['time'],tr['event'],tr['groups'],10)
    assert json.dumps(a.trees_)==json.dumps(b.trees_)
    for count in [3,4,8]:
        f=prefix(extended,count,counts);s=prefix(a,count,counts)
        direct=CompetingRiskForest(replace(cfg,n_trees=count)).fit(x,tr['time'],tr['event'],tr['groups'])
        assert json.dumps(f.trees_)==json.dumps(direct.trees_)
        shrink=shrink_leaf_cifs(direct,x,tr['time'],tr['event'],tr['groups'],10)
        assert json.dumps(s.trees_)==json.dumps(shrink.trees_)
        ex=CIFExpansion.from_forest(s,x);p=s.predict(x)
        np.testing.assert_allclose(ex.predict(x),p,atol=1e-10,rtol=0)
        assert p.min()>=0 and p.sum(-1).max()<=1+1e-12 and np.diff(p,axis=1).min()>=-1e-12
    with pytest.raises(ValueError,match='Missing or duplicated'):assemble(old,chunks+chunks[:1],8,old_counts)
    with pytest.raises(ValueError,match='Missing or duplicated'):assemble(old,chunks[:-1],8,old_counts)
    with pytest.raises(ValueError,match='raw AJ'):assemble(prefix(a,3,counts),chunks,8,old_counts)
