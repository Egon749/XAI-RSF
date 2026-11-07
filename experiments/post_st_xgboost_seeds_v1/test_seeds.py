import numpy as np
import pandas as pd
from .common import *

def test_exact_selected_parameter_inheritance_and_distinct_seeds():
    p=read(HERE/'FROZEN_PROTOCOL.json');assert p['new_seeds']==list(SEEDS)
    assert not set(SEEDS)&set(OLD_SEEDS)
    old=pd.read_csv(ROOT/'experiments/post_st_c061_xgboost_v1/COMPUTE_AND_PARAMETERS.csv')
    for y in YEARS:
        for h in (12,24):
            for row in old[(old.year==y)&(old.months==h)].to_dict('records'):
                assert all(row[k]==v for k,v in config(y,h).items())

def test_fixed_seed_xgboost_reproducible_with_native_missing():
    import pytest
    pytest.importorskip('xgboost')
    from experiments.post_st_c061_xgboost_v1.pipeline import fit,predict
    rng=np.random.default_rng(123);n=50;x=rng.normal(size=(n,4));x[::5,0]=np.nan
    d=dict(X=x,analysis_time=np.full(n,12.),event=np.where(np.arange(n)%7==0,1,0),groups=np.arange(n),label_eligible=np.ones(n,bool))
    c=dict(max_depth=2,num_boost_round=4,**{'lambda':1.})
    a,_=fit(d,12,c,42);b,_=fit(d,12,c,42)
    np.testing.assert_array_equal(predict(a,x),predict(b,x))
