import numpy as np
import pandas as pd
import pytest
from .common import recensor
from experiments.post_st_eswa_baselines_v1.common import Horizon

def small():
    t=np.array([3.,8.,12.,24.,4.,15.]);e=np.array([1,2,0,0,0,1])
    return dict(X=np.arange(18,dtype=float).reshape(6,3),analysis_time=t,event=e,label_eligible=np.ones(6,bool),groups=np.arange(6))

def test_unknowns_not_negatives_and_competing_events_known():
    d=small();hd=Horizon(d,12,'IPCW')
    np.testing.assert_array_equal(hd.y,[1,0,0,0,-1,0])
    fit=np.arange(5);ix,y,w=hd.weighted(fit,np.arange(6))
    assert 4 not in ix and w[0]==1 and (w>0).all()
    d2={k:v.copy() for k,v in d.items()};d2['analysis_time'][5]=1
    _,_,w2=Horizon(d2,12,'IPCW').weighted(fit,np.arange(5))
    np.testing.assert_array_equal(w[:4],w2) # held-out followup never changes fitting G

def test_inner_cutoff_recensors_events():
    m=pd.DataFrame(dict(origin=pd.to_datetime(['2019-01-01','2019-06-01','2020-02-01']),stop=pd.to_datetime(['2020-03-01','2019-09-01','2021-01-01'])))
    d=dict(X=np.ones((3,2)),time=np.ones(3),event=np.array([1,2,1]),groups=np.arange(3),analysis_time=np.ones(3))
    a=recensor(d,m,2020);assert a['event'].tolist()==[0,2] and len(a['X'])==2
    m2=m.copy();m2.loc[0,'stop']=pd.Timestamp('2021-04-01');d['event'][0]=2
    b=recensor(d,m2,2020)
    for k in a:np.testing.assert_array_equal(a[k],b[k])

def test_real_xgboost_missing_reload_and_finite_selection(tmp_path):
    xgb=pytest.importorskip('xgboost')
    from .pipeline import fit,predict,choose,candidates
    rng=np.random.default_rng(74);d=small()
    d={k:np.tile(v,(20,1)) if k=='X' else np.tile(v,20) for k,v in d.items()}
    d['groups']=np.arange(len(d['X']));d['X']+=rng.normal(size=d['X'].shape);d['X'][::3,0]=np.nan
    c=dict(max_depth=2,num_boost_round=4,**{'lambda':1.})
    model,_=fit(d,12,c,17);p=predict(model,d['X']);model.save_model(tmp_path/'model.ubj');copy=xgb.Booster();copy.load_model(tmp_path/'model.ubj')
    np.testing.assert_array_equal(p,predict(copy,d['X']));assert len(candidates())==18
    rows=[dict(max_depth=2,num_boost_round=96,**{'lambda':1.},mean_ap=.3,mean_brier=.1),dict(max_depth=1,num_boost_round=32,**{'lambda':10.},mean_ap=.3,mean_brier=.1)]
    best,_=choose(rows);assert best['num_boost_round']==32
    assert choose([])[0]==dict(max_depth=2,num_boost_round=96,**{'lambda':10.})
