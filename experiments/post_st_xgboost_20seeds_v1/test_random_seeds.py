import numpy as np
import pandas as pd
from .common import *

def test_draw_partition_and_parameter_inheritance():
    assert list(SEEDS)==draw_seeds()
    assert len(SEEDS)==len(set(SEEDS))==20
    assert not set(SEEDS)&set(OLD_SEEDS+PRECEDING_SEEDS)
    assert tuple(s for batch in seed_batches() for s in batch)==SEEDS
    jobs=[(y,h,s) for y in YEARS for h in (12,24) for batch in seed_batches() for s in batch]
    assert len(jobs)==len(set(jobs))==120
    old=pd.read_csv(ROOT/'experiments/post_st_c061_xgboost_v1/COMPUTE_AND_PARAMETERS.csv')
    for y in YEARS:
        for h in (12,24):
            for row in old[(old.year==y)&(old.months==h)].to_dict('records'):
                assert all(row[k]==v for k,v in config(y,h).items())

def test_seed_summary_keeps_undefined_and_mean_is_not_ensemble_metric():
    assert summarize([np.nan]*20)['undefined']==20
    assert np.isnan(summarize([np.nan]*20)['median'])
    s=summarize([1.,2.,3.,np.nan])
    assert s['median']==2 and s['defined']==3 and s['undefined']==1
    from sklearn.metrics import roc_auc_score
    y=np.array([0,1,0,1]);a=np.array([.1,.2,.3,.4]);b=np.array([.2,.9,.1,.0])
    assert np.mean([roc_auc_score(y,a),roc_auc_score(y,b)])!=roc_auc_score(y,(a+b)/2)
