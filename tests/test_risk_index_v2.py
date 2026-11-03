import inspect
import numpy as np
from crforest import CompetingRiskForest,ForestConfig,cluster_bootstrap
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_risk_index_v2.oob import multiplicities,predict_tree,oob_predictions,threshold_table,verify_membership


def fixture():
    r=np.random.default_rng(801);X=r.normal(size=(72,4));time=r.integers(1,35,len(X)).astype(float)
    event=r.choice([0,1,2],len(X));groups=np.repeat(np.arange(36),2)
    cfg=ForestConfig(n_trees=24,max_depth=3,path_budget=2,min_leaf=4,min_groups=2,min_events=1,mtry=4,n_thresholds=3,seed=42,cut_strategy='shared_quantile')
    return X,time,event,groups,cfg


def test_multiplicity_oob_and_group_repetition():
    groups=np.repeat(np.arange(30),2)
    repeated=False
    for ti in range(20):
        counts=multiplicities(groups,42,ti);rows=cluster_bootstrap(groups,42,ti)
        np.testing.assert_array_equal(counts==0,~np.isin(np.arange(len(groups)),rows))
        assert (counts[0::2]==counts[1::2]).all()
        repeated|=bool((counts>1).any());assert not np.any((counts>1)&(counts==0))
    assert repeated


def test_refit_changed_company_outcomes_do_not_change_its_oob_trees():
    X,t,e,g,cfg=fixture();f=CompetingRiskForest(cfg).fit(X,t,e,g)
    original=shrink_leaf_cifs(f,X,t,e,g,10)
    altered=e.copy();altered[g==0]=(altered[g==0]+1)%3
    f2=CompetingRiskForest(cfg).fit(X,t,altered,g);changed=shrink_leaf_cifs(f2,X,t,altered,g,10)
    times=np.arange(0,40,dtype=float);checked=0
    for ti in range(cfg.n_trees):
        counts=multiplicities(g,cfg.seed,ti)
        if counts[g==0].sum()==0:
            np.testing.assert_array_equal(predict_tree(original,ti,X[g==0],times),predict_tree(changed,ti,X[g==0],times));checked+=1
    assert checked>0
    z,counts=oob_predictions([original],X,g,times)
    explicit=[]
    for ti in range(cfg.n_trees):
        if multiplicities(g,cfg.seed,ti)[0]==0:explicit.append(predict_tree(original,ti,X[:1],times)[0])
    np.testing.assert_allclose(z[0],np.mean(explicit,axis=0),rtol=0,atol=1e-15)
    assert counts[0]==len(explicit)
    info,_=verify_membership(original,X,t,e,g);assert info['max_leaf_parent_error']<=1e-10


def test_thresholds_no_outcome_or_eval_input_and_sparse_support():
    r=np.random.default_rng(22);pred=r.uniform(0,.3,(40,2,2));counts=np.full(40,25);mu=np.full((2,2),.1)
    context=dict(time=r.integers(1,20,40),event=r.integers(0,3,40))
    before=threshold_table(pred,counts,mu)[0]
    context['event']=r.integers(0,3,40);context['time']=r.integers(100,200,40)
    after=threshold_table(pred,counts,mu)[0];np.testing.assert_array_equal(before,after)
    assert list(inspect.signature(threshold_table).parameters)==['pred','count','mu']
    counts[:11]=9;thresholds,valid,_=threshold_table(pred,counts,mu)
    assert valid.sum()==29 and np.isnan(thresholds).all()
    counts[10]=10;thresholds,valid,_=threshold_table(pred,counts,mu)
    assert valid.sum()==30 and np.isfinite(thresholds).all()
