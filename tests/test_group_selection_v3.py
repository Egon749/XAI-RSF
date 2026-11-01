import json
import numpy as np
import pytest
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_group_selection_v3.coverage import anchored_sets,original_tree
from experiments.post_st_group_selection_v3.selection import group_guard,protected_family,PeriodEvaluator


def sample():
    rng=np.random.default_rng(22);X=rng.normal(size=(90,83));X[:8,5]=np.nan
    return X,rng.uniform(1,100,90),rng.integers(0,3,90),np.arange(90)


def config():return ForestConfig(n_trees=6,max_depth=4,path_budget=2,min_leaf=5,min_groups=3,min_events=1,mtry=75,n_thresholds=5,tree_feature_count=25,cut_strategy='shared_quantile',seed=42)


def test_default_compatibility_pairing_and_exact_unaffected_trees(tmp_path):
    X,t,e,g=sample();cfg=config();full=CompetingRiskForest(cfg).fit(X,t,e,g)
    sets,_=anchored_sets(range(83),42,6)
    explicit=CompetingRiskForest(cfg).fit(X,t,e,g,tree_feature_sets=sets)
    assert explicit.trees_==full.trees_
    keep=tuple(j for j in range(83) if j!=71);sets,rows=anchored_sets(keep,42,6,groups=[{'members':[68,71]}])
    reduced=CompetingRiskForest(cfg).fit(X[:,keep],t,e,g,tree_feature_sets=sets)
    for a,b,row,nodes,base in zip(full.inbag_,reduced.inbag_,rows,reduced.trees_,full.trees_):
        np.testing.assert_array_equal(a,b)
        assert set(row['anchor'])&set(keep)<=set(row['actual']) and len(row['actual'])==25
        if row['unchanged']:assert original_tree(nodes,keep)==base
    shrunk=shrink_leaf_cifs(reduced,X[:,keep],t,e,g,10)
    path=tmp_path/'forest.json';shrunk.save(path);loaded=CompetingRiskForest.load(path)
    assert loaded.tree_feature_sets_==sets
    np.testing.assert_array_equal(loaded.predict(X[:,keep]),shrunk.predict(X[:,keep]))
    # Existing exporter tests cover every native time and explicit component sums.
    assert shrunk.audit()['max_path_variables']<=2
    pred=shrunk.predict(X[:,keep]);assert pred.min()>=0 and pred.sum(2).max()<=1+1e-12 and np.diff(pred,axis=1).min()>=-1e-12
    ordinary=tmp_path/'ordinary.json';full.save(ordinary);assert 'tree_feature_sets' not in json.loads(ordinary.read_text())
    assert CompetingRiskForest.load(ordinary).tree_feature_sets_ is None

@pytest.mark.parametrize('bad',[[[0]*25]*6,[[0,1]]*6,[[83]+list(range(24))]*6,[[float(i) for i in range(25)]]*6,[[True]*25]*6])
def test_schedule_validation(bad):
    with pytest.raises(ValueError):CompetingRiskForest(config()).fit(*sample(),tree_feature_sets=bad)


def test_group_protection_distinguishes_stable_and_tiny_or_sparse():
    a=np.linspace(-.01,.01,50);ss=[a*.95,a,a*1.05]
    assert group_guard(a,a,ss,.2,10)['passed']
    assert not group_guard(a,np.zeros(50),ss,.2,10)['passed']
    assert not group_guard(a,a,ss,.2,2)['qualified']
    assert not group_guard(a*.01,np.zeros(50),np.array(ss)*.01,.2,10)['qualified']
    assert not group_guard(a,a,[a,np.zeros(50),np.zeros(50)],.2,10)['qualified']
    # Constant baseline shifts do not change centered group information.
    assert group_guard(a,a+.04,ss,.2,10)['passed']


def test_proxy_family_stays_within_predefined_family():
    x=np.arange(50,dtype=float);X=np.stack([x,-x,x,x*x],1)
    assert protected_family(X,[0],['market','market','financial','market'])==[0,1,3]


def test_period_guard_does_not_average_away_recovery_damage():
    ev=object.__new__(PeriodEvaluator)
    rows=[dict(cohort=y,horizon=24,cause=2,ranking_guard=True,target_events=10,auc=.8,ap=.8,brier=.2,captured=3) for y in [2020,2021]]
    new=[dict(rows[0],brier=.206),dict(rows[1],brier=.199)]
    ev.summary=lambda keep:dict(periods=rows if len(keep)==83 else new,cells=100 if len(keep)==83 else 90)
    result=ev.compare(tuple(range(82)))
    assert result['performance_failures']==['2020_H24_C2_brier']
