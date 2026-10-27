from dataclasses import replace
import numpy as np
import pytest
from crforest import ForestConfig, CompetingRiskForest, CIFExpansion
from crforest.partition import (shared_quantile_cuts, shared_refined_cuts,
    node_shared_candidates, tree_feature_subset)
from experiments.shared_partition_v1.engine import fit_shard, assemble


def test_balanced_subspaces_cover_all_inputs_without_bootstrap_dependence():
    blocks=[tree_feature_subset(75,25,42,i) for i in range(3)]
    np.testing.assert_array_equal(np.sort(np.concatenate(blocks)),np.arange(75))
    for i in range(32):
        v=tree_feature_subset(75,25,42,i)
        assert len(v)==len(set(v))==25
        np.testing.assert_array_equal(v,tree_feature_subset(75,25,42,i))
    counts=np.bincount(np.concatenate([tree_feature_subset(75,25,42,i) for i in range(32)]),minlength=75)
    assert counts.min()==10 and counts.max()==11
    for i in range(5):assert len(set(tree_feature_subset(7,3,42,i)))==3


def test_refinement_keeps_anchors_and_reaches_within_tail_nodes():
    X=np.arange(10000.)[:,None]
    old=shared_quantile_cuts(X,5)[0];new=shared_refined_cuts(X,5)[0]
    assert set(old)<=set(new) and set(new)<=set(X[:,0])
    assert len(new)==27
    node=X[9500:,0]
    selected=node_shared_candidates(new,node,5)
    assert 0<len(selected)<=5 and set(selected)<=set(new)
    assert not len(old[(old>=node.min())&(old<node.max())])
    assert len(node_shared_candidates(new,np.array([np.nan]),5))==0
    # Multiplicity-sensitive nearest quantile, tie goes to lower cut.
    np.testing.assert_array_equal(node_shared_candidates(np.array([0.,2.,4.,6.]),np.array([0.,1.,1.,1.,7.]),1),[0.])


@pytest.mark.parametrize('strategy',['shared_quantile','shared_refined'])
@pytest.mark.parametrize('subset',[None,3])
def test_new_options_preserve_probability_exact_compilation_and_sharding(strategy,subset,tmp_path):
    rng=np.random.default_rng(914);X=rng.normal(size=(250,6));X[::7,0]=np.nan
    t=rng.integers(1,25,len(X)).astype(float);e=rng.integers(0,3,len(X));g=np.arange(len(X))//2
    cfg=ForestConfig(n_trees=6,max_depth=5,path_budget=2,min_leaf=8,min_groups=3,min_events=1,
        mtry=6,n_thresholds=5,cut_strategy=strategy,tree_feature_count=subset)
    f=CompetingRiskForest(cfg).fit(X,t,e,g)
    for i,tree in enumerate(f.trees_):
        allowed=set(tree_feature_subset(6,subset,cfg.seed,i))
        assert all(n['feature'] in allowed for n in tree if 'feature' in n)
    ex=CIFExpansion.from_forest(f,X);p=f.predict(X)
    np.testing.assert_allclose(ex.predict(X),p,atol=1e-13,rtol=0)
    assert p.min()>=0 and p.sum(-1).max()<=1+1e-13 and np.diff(p,axis=1).min()>=-1e-13
    d=dict(X=X,time=t,event=e,groups=g)
    joined=assemble([(i,fit_shard(cfg,d,i,2)) for i in [4,0,2]])
    assert f.trees_==joined.trees_ and f.audit_==joined.audit_
    f.save(tmp_path/'f.json');joined.save(tmp_path/'j.json')
    assert (tmp_path/'f.json').read_bytes()==(tmp_path/'j.json').read_bytes()
    np.testing.assert_array_equal(CompetingRiskForest.load(tmp_path/'f.json').predict(X),p)
    assert f.audit_['path_violations']==0


def test_invalid_subspace_rejected():
    X=np.zeros((10,2));t=np.arange(1,11);e=np.ones(10)
    for count in [0,-1,3,1.5,True]:
        with pytest.raises(ValueError,match='tree feature count'):
            CompetingRiskForest(ForestConfig(tree_feature_count=count)).fit(X,t,e)
