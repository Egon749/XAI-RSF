import numpy as np
import pytest
from .common import Horizon,transform,transform_stats
from experiments.post_st_prognosis_v1.common import endpoint_weights
def sample():
    t=np.array([2,3,4,5,6,7,12,16.]);e=np.array([2,0,1,0,2,1,0,1])
    return dict(analysis_time=t,event=e,groups=np.arange(len(t)),X=np.ones((len(t),2)),label_eligible=np.ones(len(t),bool))
def test_competing_and_positive_training_information():
    d=sample();hd=Horizon(d,12,'IPCW');ix=np.arange(8);k,y,w=hd.weighted(ix,ix);ref,known,_=endpoint_weights(d['analysis_time'],d['event'],12)
    np.testing.assert_array_equal(k,np.flatnonzero(known));np.testing.assert_allclose(w,ref[k]);assert hd.y.tolist()==[0,-1,1,-1,0,1,0,0]
    a=np.arange(6);b=np.array([6,7]);before=hd.weighted(a,b)[2];hd.e[b]=0;np.testing.assert_array_equal(before,hd.weighted(a,b)[2])
def test_schedule_derivatives():
    pytest.importorskip('xgboost');from .proboost import objective,beta_at
    z=np.array([-1.,-.1,.7,1.5]);y=np.array([0,1,0,1]);w=np.array([.9,1.2,.7,2.]);eps=1e-4
    for beta in [5.,2.,.2]:
        a,g,h=objective(z,y,w,beta,.75);b=objective(z+eps,y,w,beta,.75)[0];c=objective(z-eps,y,w,beta,.75)[0]
        np.testing.assert_allclose((b-c)/(2*eps),g,atol=1e-7);np.testing.assert_allclose((b-2*a+c)/eps**2,h,atol=5e-7);assert (h>0).all()
    assert np.isclose(beta_at(47,48,5,.2),.2);assert beta_at(0,48,5,.2)==5
def test_chen_mechanisms_and_smote_provenance():
    pytest.importorskip('hyperopt');from .stacking import inertia,feasible,fweights,smote
    v=inertia(np.ones(3),np.array([.1,.5,.9]));assert v[0]>1 and v[-1]<1
    assert feasible(np.array([1,0,1,0,1])) and not feasible([1,0,0,0,1]);np.testing.assert_allclose(fweights(np.array([.1,.2,.7]),np.ones(3,bool)),[.1,.2,.7])
    rng=np.random.default_rng(1);x=rng.random((24,4));x[:,2:]=0;y=np.r_[np.ones(8),np.zeros(16)];w=np.linspace(.8,2.,24)
    xx,yy,ww,n=smote(x,y,w,np.array([False,True]),5);assert n==8;np.testing.assert_array_equal(xx[-8:,1],x[:8,1]);np.testing.assert_array_equal(ww[-8:],w[:8]);assert np.all(yy[-8:]==1)
    assert smote(x,y*(np.arange(24)<5),w,np.array([False,True]),5)[3]==0
