from dataclasses import replace
import json
import numpy as np
import pytest
from crforest import ForestConfig, CompetingRiskForest, CIFExpansion, HigherOrderCIFExpansion
from crforest.partition import tree_feature_subset
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.common import configuration, pairing, NEW

def test_iid_is_uniform_without_replacement_on_separate_reproducible_stream():
    for seed in [42,101,202]:
        for tree in range(32):
            expected=np.sort(np.random.default_rng(np.random.SeedSequence([seed,tree,1731])).choice(83,25,replace=False))
            actual=tree_feature_subset(83,25,seed,tree,'iid')
            np.testing.assert_array_equal(actual,expected)
            assert len(set(actual))==25
    with pytest.raises(ValueError): ForestConfig(tree_feature_strategy='unknown')

def test_exact_parameter_differences_and_bootstrap_pairing():
    from dataclasses import asdict
    base=configuration(42,'shallow_parent'); groups=np.arange(100)//2
    expected=dict(q_unrestricted={'path_budget'},local_cuts={'cut_strategy'},iid_features={'tree_feature_strategy'})
    for arm in NEW:
        cfg=configuration(42,arm);a,b=asdict(cfg),asdict(base)
        assert {k for k in a if a[k]!=b[k]}==expected[arm]
        for x,y in zip(pairing(groups,cfg),pairing(groups,base)):
            assert x['bootstrap_sha256']==y['bootstrap_sha256']
            assert x['multiplicity_sha256']==y['multiplicity_sha256']
            if arm!='iid_features':assert x['features_sha256']==y['features_sha256']

def test_old_missing_config_field_compatibility_and_iid_full_reconstruction(tmp_path):
    rng=np.random.default_rng(992);X=rng.normal(size=(100,6));X[::7,0]=np.nan
    t=rng.integers(1,30,100).astype(float);e=rng.integers(0,3,100);g=np.arange(100)//2
    base=ForestConfig(n_trees=3,max_depth=4,path_budget=2,min_leaf=5,min_groups=2,min_events=1,mtry=6,
        tree_feature_count=4,n_thresholds=3,cut_strategy='shared_quantile')
    f=CompetingRiskForest(base).fit(X,t,e,g);f.save(tmp_path/'current.json')
    legacy=json.loads((tmp_path/'current.json').read_text());legacy['config'].pop('tree_feature_strategy')
    (tmp_path/'legacy.json').write_text(json.dumps(legacy))
    old=CompetingRiskForest.load(tmp_path/'legacy.json')
    assert old.config.tree_feature_strategy=='balanced'
    np.testing.assert_array_equal(old.predict(X),f.predict(X))
    for strategy,q in [('iid',2),('balanced',None)]:
        cfg=replace(base,tree_feature_strategy=strategy,path_budget=q)
        raw=CompetingRiskForest(cfg).fit(X,t,e,g)
        for i,tree in enumerate(raw.trees_):
            allowed=set(tree_feature_subset(6,4,base.seed,i,strategy))
            assert all(n['feature'] in allowed for n in tree if 'feature' in n)
            np.testing.assert_array_equal(raw.inbag_[i],f.inbag_[i])
        shrunk=shrink_leaf_cifs(raw,X,t,e,g,10)
        ex=(CIFExpansion if q==2 else HigherOrderCIFExpansion).from_forest(shrunk,X)
        pred=shrunk.predict(X)
        np.testing.assert_allclose(ex.predict(X),pred,atol=1e-10,rtol=0)
        assert pred.min()>=0 and pred.sum(-1).max()<=1+1e-12 and np.diff(pred,axis=1).min()>=-1e-12
        with pytest.raises(ValueError):shrink_leaf_cifs(shrunk,X,t,e,g,10)
        from experiments.post_st_ablation_v1.verification import replay
        rows,_,_,_,audit=replay(shrunk,dict(X=X,time=t,event=e,groups=g))
        assert sum(r['routed_attempts'] for r in rows)==raw.candidate_evaluations_
        assert audit['max_leaf_estimator_error']<1e-12
