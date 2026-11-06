import numpy as np
import pandas as pd
from experiments.pre_st_warning_v1.clock import stop_at,endpoint,available_values
from experiments.pre_st_warning_v1.raw_scan import normalize,MARKET
from experiments.pre_st_warning_v1.metrics import binary_scores,independent_weights,metrics
from experiments.post_st_prognosis_v1.common import endpoint_weights
from sklearn.metrics import roc_auc_score,average_precision_score

def test_future_training_events_are_censored_not_normal_labels():
    t,e,end=stop_at('2022-01-01','2026-08-31','2023-02-01',1,'2023-01-01')
    assert t==364 and e==0 and end==pd.Timestamp('2022-12-31')
    known,y=endpoint([t],[e],[365],1)
    assert not known[0] and not y[0]
    assert stop_at('2022-01-01','2026-08-31','2024-02-01',1,'2023-01-01')==(t,e,end)

def test_calendar_horizon_and_competing_event():
    assert stop_at('2024-01-01','2026-08-31',pd.NaT,0)[0]==731
    assert stop_at('2023-01-01','2026-08-31',pd.NaT,0)[0]==731
    known,y=endpoint([12,364,365],[2,0,0],[365,365,365],1)
    np.testing.assert_array_equal(known,[True,False,True]);assert not y.any()

def test_derived_feature_requires_both_disclosure_dates():
    row=pd.Series(dict(available=pd.Timestamp('2022-10-20'),prev_available=pd.Timestamp('2023-02-01'),roa=.1,revenue_growth=.4))
    v,d=available_values(row,['roa','revenue_growth'],pd.Timestamp('2023-01-01'))
    assert v[0]==.1 and np.isnan(v[1]) and np.isnat(d[1])

def test_raw_header_rows_and_non_a_shares_are_removed():
    rows=[['证券代码','交易日期']+[None]*10,['1','2022-12-30']+[1]*8+[1,1],['2','2022-12-30']+[1]*8+[2,1]]
    d=normalize(pd.DataFrame(rows,columns=MARKET),'market')
    assert d.Stkcd.tolist()==['000001']

def test_warning_metrics_ties_censoring_competing_events():
    t=np.array([2,3,4,12,12,24,6.]);e=np.array([1,0,2,0,1,0,2]);p=np.array([.4,.2,.2,.1,.4,.1,.3])
    w,k,_=endpoint_weights(t,e,12)
    np.testing.assert_allclose(w,independent_weights(t,e,12),rtol=0,atol=1e-15)
    y=(e==1)&(t<=12);a,ap=binary_scores(y[k],p[k])
    assert abs(a-roc_auc_score(y[k],p[k]))<1e-15
    assert abs(ap-average_precision_score(y[k],p[k]))<1e-15
    z=metrics(t,e,p,np.ones(7),np.arange(7),np.arange(7),12,1)
    assert z['unknown']==1 and z['top20_checked']==2 and z['top20_captured']==2
    assert z['within_year_auc']==z['auc']

def test_pooled_auc_is_not_average_of_year_auc():
    z=metrics(np.ones(4)*12,np.array([1,0,1,0]),np.array([.9,.8,.2,.1]),np.array([2023,2023,2024,2024]),np.arange(4),np.arange(4),12,1)
    assert z['within_year_auc']==1 and z['auc']==.75

def test_coefficient_cache_preserves_ordinary_functions(tmp_path):
    from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
    from experiments.pre_st_warning_v1.verification import CachedExpansion
    rng=np.random.default_rng(5);X=rng.normal(size=(60,3));X[::5,0]=np.nan
    f=CompetingRiskForest(ForestConfig(n_trees=3,max_depth=3,path_budget=2,min_leaf=5,min_groups=3,min_events=1)).fit(X,rng.integers(1,100,60),rng.integers(0,3,60))
    a=CIFExpansion.from_forest(f,X);b=CachedExpansion.from_forest(f,X)
    for times in [None,[0,1,25,200],[0,1,25,200]]:
        np.testing.assert_array_equal(a.predict(X,times),b.predict(X,times))
        for key in [(j,) for j in a.mains]+a.pairs:np.testing.assert_array_equal(a.component(X,key,times),b.component(X,key,times))
    b.save(tmp_path/'f.npz');c=CachedExpansion.load(tmp_path/'f.npz')
    np.testing.assert_array_equal(a.predict(X),c.predict(X))
