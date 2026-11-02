import numpy as np
import pytest
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.affine import fit_affine,forest_envelope,transform,transformed_expansion,hull
from experiments.post_st_temporal_affine_v1.calibration import fit
from experiments.post_st_temporal_affine_v1.common import cohorts,periods

def example():
    rng=np.random.default_rng(221102);p=rng.dirichlet([2,4,8],100)[:,:2]
    y=np.eye(3)[rng.choice(3,100,p=[.25,.45,.30])][:,:2];v=hull(np.vstack([p,[[.5,.4],[.1,.8]]]))
    return p,y,v

def test_no_temporal_prior_matches_previous_frozen_calibration():
    p,y,v=example();a=fit_affine(p,y,v,y.sum(0));b=fit(p,y,v,y.sum(0))
    np.testing.assert_allclose(np.r_[a['c'],a['b']],np.r_[b['c'],b['b']],atol=1e-10,rtol=0)
    np.testing.assert_allclose(a['objective_after'],b['objective_after'],atol=1e-12,rtol=0)

def test_temporal_penalty_reduces_scaled_distance_and_has_certified_optimum():
    p,y,v=example();prior={'c':[.03,.01],'b':[1.2,.8]}
    a=fit(p,y,v,y.sum(0),prior,temporal_companies=0);b=fit(p,y,v,y.sum(0),prior,temporal_companies=200)
    target=np.r_[prior['c'],prior['b']];scale=np.array(b['parameter_scales'])
    dist=lambda z:np.sum((scale*(np.r_[z['c'],z['b']]-target))**2)
    assert dist(b)<=dist(a)+1e-12 and b['gap']<=1e-8 and b['primal_violation']<=1e-10
    assert b['objective_after']==pytest.approx(b['data_loss']+b['identity_penalty']+b['temporal_penalty'],abs=1e-12)

def test_duplicate_companies_cannot_unlock_sparse_slope_and_future_prior_not_needed():
    p,y,v=example();prior={'c':[.2,.0],'b':[3.,.6]}
    a=fit(p,y,v,[3,9],prior);assert a['b']==[1.,1.]
    assert a['frozen_slope_for_sparse_cause']==[True,True]
    assert cohorts(2023,12)==[2020,2021] and cohorts(2025,24)==[2021,2022]
    assert cohorts(2022,24)==[2019] and periods(12)==[2021,2022,2023,2024,2025]

def test_affine_full_functions_probability_and_reload(tmp_path):
    rng=np.random.default_rng(302);x=rng.normal(size=(90,4));x[::9,0]=np.nan
    t=rng.integers(1,25,90);e=rng.choice([0,1,2],90);g=np.arange(90)
    f=CompetingRiskForest(ForestConfig(n_trees=3,max_depth=3,path_budget=2,min_leaf=4,min_groups=3,min_events=1)).fit(x,t,e,g)
    f=shrink_leaf_cifs(f,x,t,e,g,10);v=forest_envelope(f);p=f.predict(x,[12])[:,0,:];y=np.column_stack([(e==k)&(t<=12) for k in [1,2]])
    cal=fit(p,y,v,y.sum(0),{'c':[.01,.02],'b':[1.,1.]});cal.update(shape_grid=[0,6,12,24],shape_values=[0,.4,1,1])
    ex=transformed_expansion(CIFExpansion.from_forest(f,x),cal);grid=np.unique(np.r_[0,f.grid_,cal['shape_grid'],30])
    expected=transform(f.predict(x,grid),grid,cal);total=np.broadcast_to(ex._coeff(grid)[1],expected.shape).copy()
    for k in [(j,) for j in ex.mains]+ex.pairs:total+=ex.component(x,k,grid)
    np.testing.assert_allclose(total,expected,atol=1e-12,rtol=0)
    assert expected.min()>=-1e-12 and expected.sum(-1).max()<=1+1e-12 and np.diff(expected,axis=1).min()>=-1e-12
    ex.save(tmp_path/'functions.npz');np.testing.assert_allclose(CIFExpansion.load(tmp_path/'functions.npz').predict(x,grid),expected,atol=1e-12,rtol=0)

@pytest.mark.parametrize('bad',[[-1,2],[101,2],[1.5,2]])
def test_invalid_distinct_support_rejected(bad):
    p,y,v=example()
    with pytest.raises(ValueError):fit(p,y,v,bad)


def test_quadratic_polish_does_not_force_nearly_active_facet():
    from experiments.post_st_temporal_affine_v1.calibration import _polish
    H=np.eye(4);linear=-np.array([.2,.2,1.,1.])
    G=np.array([[1.,0,0,0],[1.,1e-7,0,0]]);h=np.array([0.,3e-8])
    fun=lambda z:.5*z@H@z+linear@z
    z=_polish(np.array([-1e-9,.20000002,1.,1.]),H,linear,G,h,fun)
    np.testing.assert_allclose(z,[0,.2,1,1],atol=1e-12,rtol=0)
    interior=_polish(np.array([.1999999,.20000002,1.,1.]),H,linear,np.eye(4),np.full(4,2.),fun)
    np.testing.assert_allclose(interior,-linear,atol=1e-12,rtol=0)
