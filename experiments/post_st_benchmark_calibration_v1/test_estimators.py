import numpy as np
from .estimators import transform,estimate,regression_predict,censor_left

def test_finegray_aj_limit_and_competing_risk_sets():
    x=np.zeros((5,1));t=np.array([1,2,3,4,5]);e=np.array([2,0,1,1,0])
    model=estimate(x,t,e,1,.1,True)
    # At time 3: three event-free firms and one retained competitor weighted 3/4.
    np.testing.assert_allclose(model['jump'],[1/3.75,1/2.75])
    ordinary=estimate(x,t,e,1,.1,False);np.testing.assert_allclose(ordinary['jump'],[1/3,1/2])
    np.testing.assert_allclose(censor_left(t,e)[1],[1,1,.75,.75,.75])
def test_joint_hazards_and_scaling():
    m=dict(beta=np.array([0.]),time=np.array([1]),jump=np.array([1.]))
    p=regression_predict([m,m],np.ones((1,1)),np.array([0,1,2]),False)
    np.testing.assert_allclose(p[0,1],[(1-np.exp(-2))/2]*2)
    assert p.min()==0 and p.sum(-1).max()<1
    a,b,f=transform(np.array([[np.nan,1],[2,1],[4,1]]),np.array([[100,77]]))
    assert a.shape==(3,2) and f['median'][0]==3 and b[0,0]>10
