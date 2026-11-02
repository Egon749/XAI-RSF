from dataclasses import replace
import json
import numpy as np
import pytest
from crforest import CompetingRiskForest, ForestConfig, CIFExpansion
from crforest.core import _split_parent, logrank_cr_score, target_support_allows
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components


@pytest.mark.parametrize('weights', [(1,1),(1,0),(0,1),(2,1)])
def test_quadratic_score_matches_independent_event_time_calculation(weights):
    t=np.array([1,1,2,2,3,4,5,6.]);e=np.array([1,0,2,1,2,0,1,2])
    left=np.array([1,0,1,1,0,1,0,0],bool)
    u,v=independent_components(t,e,left);w=np.asarray(weights,float)
    expected=np.sqrt(np.sum(w*w*np.divide(u*u,v,out=np.zeros_like(v),where=v>1e-18))/np.sum(w*w))
    for scale in [1,3]:
        got=logrank_cr_score(left,e,_split_parent(t,e,2),w*scale,'causewise_quadratic')
        assert got==pytest.approx(expected,abs=1e-14)


def test_opposed_cause_signals_cancel_only_in_legacy_score():
    t=np.ones(80);e=np.r_[np.ones(40,int),np.full(40,2)];left=np.arange(80)<40
    p=_split_parent(t,e,2)
    assert logrank_cr_score(left,e,p)==0
    assert logrank_cr_score(left,e,p,split_score='causewise_quadratic')>8
    assert logrank_cr_score(left,np.zeros(80,int),_split_parent(t,np.zeros(80,int),2),
                           split_score='causewise_quadratic')==0


def test_support_is_unique_companies_and_allows_zero_target_side():
    e=np.array([1,1,1,2,2,0]);g=np.array(['a','a','a','b','c','d']);left=np.array([1,1,1,0,0,0],bool)
    assert target_support_allows(left,e,g,minimum=0)
    assert not target_support_allows(left,e,g,minimum=2)
    g[1]='z'
    assert target_support_allows(left,e,g,minimum=2)
    e[:3]=2
    assert target_support_allows(left,e,g,minimum=2)
    # Two target companies split 1:1 must be rejected, not counted together.
    e=np.array([1,0,1,0]);g=np.arange(4);left=np.array([1,1,0,0],bool)
    assert not target_support_allows(left,e,g,minimum=2)


@pytest.mark.parametrize('settings',[{'split_score':'bad'},{'split_min_target_groups':-1},
    {'split_min_target_groups':True},{'split_min_target_groups':1.5},
    {'split_target_cause':0},{'split_target_cause':3},{'split_target_cause':True}])
def test_invalid_score_and_support_configuration(settings):
    with pytest.raises(ValueError):ForestConfig(**settings)


def test_all_four_modes_keep_pairing_aj_and_complete_q2_functions(tmp_path):
    rng=np.random.default_rng(1202);x=rng.normal(size=(100,4));x[::7,2]=np.nan
    t=rng.integers(1,25,len(x)).astype(float);e=rng.choice([0,1,2],len(x),p=[.3,.2,.5]);g=np.arange(len(x))
    cfg=ForestConfig(n_trees=3,max_depth=4,path_budget=2,min_leaf=4,min_groups=3,min_events=1,
                     mtry=4,n_thresholds=4,cut_strategy='shared_quantile',tree_feature_count=3)
    reference=None
    for i,(score,guard) in enumerate([('signed_sum',0),('causewise_quadratic',0),('signed_sum',2),('causewise_quadratic',2)]):
        f=CompetingRiskForest(replace(cfg,split_score=score,split_min_target_groups=guard)).fit(x,t,e,g)
        if reference is None:reference=f
        for a,b in zip(reference.inbag_,f.inbag_):np.testing.assert_array_equal(a,b)
        if guard:
            for tree in f.trees_:
                for node in tree:
                    if 'cif' not in node:
                        for side in ['left','right']:
                            n=tree[node[side]]['cause_company_support'][0];assert n==0 or n>=2
        sh=shrink_leaf_cifs(f,x,t,e,g,10)
        from experiments.post_st_cause_support_v1.verification import replay
        nodes,features,pairs,leaves,check=replay(sh,dict(X=x,time=t,event=e,groups=g))
        assert check['max_score_error']<1e-12
        for kind,model in [('raw',f),('shrunk',sh)]:
            grid=np.unique(np.r_[0,model.grid_,30]);p=model.predict(x,grid)
            ex=CIFExpansion.from_forest(model,x)
            total=np.broadcast_to(ex._coeff(grid)[1],p.shape).copy()
            for key in [(j,) for j in ex.mains]+ex.pairs:total+=ex.component(x,key,grid)
            np.testing.assert_allclose(total,p,atol=1e-12,rtol=0)
            assert model.audit()['max_path_variables']<=2 and p.min()>=0
            assert p.sum(-1).max()<=1+1e-12 and np.diff(p,axis=1).min()>=-1e-12
            path=tmp_path/f'{i}_{kind}.json';model.save(path)
            np.testing.assert_array_equal(CompetingRiskForest.load(path).predict(x,grid),p)
    path=tmp_path/'legacy.json';reference.save(path);payload=json.loads(path.read_text())
    for name in ['split_score','split_min_target_groups','split_target_cause']:payload['config'].pop(name)
    path.write_text(json.dumps(payload));old=CompetingRiskForest.load(path)
    assert old.config.split_score=='signed_sum' and old.config.split_min_target_groups==0
    np.testing.assert_array_equal(old.predict(x),reference.predict(x))

