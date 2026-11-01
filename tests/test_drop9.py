from dataclasses import replace
import numpy as np
import pytest
from crforest import CompetingRiskForest,ForestConfig
from crforest.partition import shared_quantile_cuts
from experiments.post_st_drop9_v1.common import DROP_NAMES,kept_columns,reduced


def test_drop_exact_fields_preserve_every_other_input_and_cut():
    names=[f'field{i}' for i in range(74)]+DROP_NAMES
    rng=np.random.default_rng(4209);X=rng.normal(size=(40,83));X[::4,3]=np.nan
    data=dict(X=X,event=np.arange(40)%3,groups=np.arange(40),time=np.arange(40)+1)
    keep=kept_columns(names);d=reduced(data,keep)
    assert len(keep)==74 and d['X'].shape==(40,74)
    for k in data:
        np.testing.assert_array_equal(d[k],data[k][:,keep] if k=='X' else data[k])
    before=shared_quantile_cuts(X,5);after=shared_quantile_cuts(d['X'],5)
    for i,j in enumerate(keep):np.testing.assert_array_equal(before[j],after[i])
    with pytest.raises(ValueError):kept_columns(names[:-1])


def test_mtry_upper_bound_above_reduced_dimension_changes_no_search():
    rng=np.random.default_rng(7409);X=rng.normal(size=(80,74));X[::7,2]=np.nan
    t=rng.integers(1,100,80);e=rng.integers(0,3,80);groups=np.repeat(np.arange(40),2)
    cfg=ForestConfig(n_trees=3,max_depth=4,path_budget=2,min_leaf=5,min_groups=2,min_events=1,
        mtry=75,n_thresholds=5,tree_feature_count=25,cut_strategy='shared_quantile',seed=42)
    a=CompetingRiskForest(cfg).fit(X,t,e,groups)
    b=CompetingRiskForest(replace(cfg,mtry=74)).fit(X,t,e,groups)
    assert a.trees_==b.trees_ and a.candidate_evaluations_==b.candidate_evaluations_
    np.testing.assert_array_equal(a.predict(X),b.predict(X))
    for x,y in zip(a.inbag_,b.inbag_):np.testing.assert_array_equal(x,y)
    for bad in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError):CompetingRiskForest(replace(cfg,mtry=bad)).fit(X,t,e,groups)
