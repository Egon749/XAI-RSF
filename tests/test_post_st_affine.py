import numpy as np
import pytest
from experiments.post_st_affine_calibration_v1.calibration import fit_affine,transform,shape_at,A
from experiments.post_st_prognosis_v1.common import endpoint_weights,classify_first


def test_exact_function_sum_and_order_after_affine():
    rng=np.random.default_rng(51);x=rng.uniform(-1,1,(100,3));t=np.linspace(0,1,19)
    mu=np.array([.15,.25]);main=np.c_[.03*x[:,0]+.02*x[:,1],-.02*x[:,1]]
    pair=np.c_[.01*x[:,0]*x[:,1],.02*x[:,1]*x[:,2]]
    f=(mu+main+pair)[:,None,:]*t[None,:,None];g=np.broadcast_to(t[:,None],(len(t),2))
    cal={'c':[.05,.04],'b':[.8,.7]};z=transform(f,g,cal)
    direct=transform(mu*t[:,None],g,cal)+(main+pair)[:,None,:]*t[None,:,None]*cal['b']
    np.testing.assert_allclose(z,direct,rtol=0,atol=1e-15)
    for k in range(2):assert np.array_equal(np.argsort(z[:,-1,k]),np.argsort(f[:,-1,k]))
    assert (z[:,0]==0).all() and (np.diff(z,axis=1)>=0).all() and (z.sum(2)<=1).all()


def test_simplex_vertices_certify_every_convex_combination():
    vertices=np.array([[0.,0.],[1.,0.],[0.,1.]])
    cal={'c':[.1,.1],'b':[.8,.4]};z=transform(vertices,np.ones(2),cal)
    assert (z>=0).all() and (z.sum(1)<=1).all()
    assert (transform(vertices,np.ones(2),{'c':[.1,0.],'b':[1.,1.]}).sum(1)>1).any()


def test_unknown_outcomes_have_zero_calibration_weight():
    time=np.array([5.,8.,12.,12.]);event=np.array([0,1,0,2]);w,known,_=endpoint_weights(time,event,12.)
    assert w[0]==0 and not known[0]
    p=np.array([[.2,.3],[.4,.2],[.1,.2],[.1,.5]]);y=np.c_[event==1,event==2].astype(float)
    a=fit_affine(p,y,w);y[0]=[1.,0.];b=fit_affine(p,y,w)
    np.testing.assert_allclose(a['c']+a['b'],b['c']+b['b'],rtol=0,atol=1e-12)
    assert a['gap']<=1e-8 and a['objective_after']<=a['objective_before']+1e-10


def test_fitted_affine_joint_probabilities_and_gap():
    rng=np.random.default_rng(20);p=rng.dirichlet([2,4,10],200)[:,:2]
    event=rng.choice([0,1,2],200,p=[.85,.10,.05]);y=np.c_[event==1,event==2]
    cal=fit_affine(p,y,np.ones(200));z=np.r_[cal['c'],cal['b']]
    assert (A@z<=1+1e-10).all() and min(cal['b'])>0 and cal['gap']<1e-8
    pred=transform(p,np.ones(2),cal);assert pred.min()>=0 and pred.sum(1).max()<=1+1e-10


def test_shape_origin_monotonicity_and_calendar_horizon_equality():
    g=shape_at([0.,90.,365.,366.,730.],[20.,90.,360.,366.],np.array([[.01,.02],[.02,.04],[.03,.1],[.05,.12]]))
    assert (g[0]==0).all() and (np.diff(g,axis=0)>=0).all()
    np.testing.assert_array_equal(g[2:],np.ones((3,2)))


def test_calibration_cutoff_excludes_later_event():
    a=classify_first('2022-05-01',[('2023-02-01',1)],'2022-12-31',months=12)
    assert a['event']==0 and a['reason']=='administrative_censoring'


def test_reject_incompatible_probabilities():
    with pytest.raises(ValueError):fit_affine([[.8,.8]],[[1,0]],[1])


def test_constant_predictions_and_no_calibration_events_remain_defined():
    p=np.broadcast_to([.1,.2],(20,2));cal=fit_affine(p,np.zeros_like(p),np.ones(20))
    assert cal['gap']<1e-8 and cal['slope_floor_active']
    assert np.max(transform(p,np.ones(2),cal))<1e-6
