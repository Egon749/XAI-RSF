from dataclasses import asdict
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from experiments.post_st_level_coverage_v1.common import configuration,calibration_years
from experiments.post_st_level_coverage_v1.affine import hull,forest_envelope,fit_affine,transform,transformed_expansion

def test_only_feature_eligibility_changes():
    a,b=asdict(configuration(42,'reference')),asdict(configuration(42))
    assert [k for k in a if a[k]!=b[k]]==['tree_feature_count']
    assert a['path_budget']==b['path_budget']==2 and b['mtry']==75 and b['tree_feature_count']==83

@pytest.mark.parametrize('year,horizon,expected',[(2023,24,[2019,2020]),(2024,24,[2020,2021]),(2023,12,[2020,2021]),(2024,12,[2021,2022])])
def test_maturity_selection_uses_calendar_only(year,horizon,expected):
    assert calibration_years(year,horizon)==expected

def test_minkowski_certificate_encloses_all_combinations():
    leaves=[np.array([[.1,.8],[.5,.2],[0.,0.]]),np.array([[.4,.4],[.2,.7]])]
    f=SimpleNamespace(trees_=[[dict(cif=[p]) for p in points] for points in leaves])
    vertices=forest_envelope(f)
    for b in [np.array([.5,2.]),np.array([3.,.2]),np.ones(2)]:
        actual=max(np.mean(pair,axis=0)@b for pair in product(*leaves))
        assert abs(actual-max(vertices@b))<1e-12
    assert hull([[0,0],[.1,.1],[.2,.2]]).shape==(2,2)

def test_sparse_slopes_and_probability_certificate():
    p=np.tile([.08,.4],(30,1));y=np.zeros_like(p);y[:3,0]=1;y[3:20,1]=1
    vertices=np.array([[0.,0.],[.3,.6],[.1,.8]])
    cal=fit_affine(p,y,vertices,[3,17])
    assert cal['b'][0]==1 and cal['gap']<=1e-8 and cal['primal_violation']<=1e-10
    for v in vertices:
        z=np.array(cal['c'])+np.array(cal['b'])*v
        assert z.min()>=0 and z.sum()<=1+1e-10
    assert cal['objective_after']<=cal['objective_before']+1e-12

def test_affine_complete_functions_and_reload(tmp_path):
    rng=np.random.default_rng(44);X=rng.normal(size=(100,3));t=rng.integers(1,40,100);e=rng.integers(0,3,100)
    f=CompetingRiskForest(ForestConfig(n_trees=3,max_depth=3,path_budget=2,min_leaf=5,min_groups=3,min_events=1,mtry=3,n_thresholds=3)).fit(X,t,e)
    ex=CIFExpansion.from_forest(f,X)
    cal=dict(c=[.1,.05],b=[.6,.6],shape_grid=[0.,10.,20.,50.],shape_values=[0.,.2,.5,1.])
    new=transformed_expansion(ex,cal);grid=np.unique(np.r_[0.,f.grid_,cal['shape_grid'],100.])
    expected=transform(f.predict(X,grid),grid,cal)
    total=np.broadcast_to(new._coeff(grid)[1],expected.shape).copy()
    for k in [(j,) for j in new.mains]+new.pairs:total+=new.component(X,k,grid)
    np.testing.assert_allclose(total,expected,atol=1e-12,rtol=0)
    np.testing.assert_allclose(new.predict(X,grid),expected,atol=1e-12,rtol=0)
    new.save(tmp_path/'functions.npz')
    np.testing.assert_array_equal(CIFExpansion.load(tmp_path/'functions.npz').predict(X,grid),new.predict(X,grid))
    assert expected.min()>=0 and expected.sum(-1).max()<=1 and np.diff(expected,axis=1).min()>=0

def test_affine_rejects_invalid_support():
    with pytest.raises(ValueError,match='support'):
        fit_affine(np.array([[.1,.2]]),np.array([[1.,0.]]),np.array([[.1,.2]]),[0,0])

def test_certificate_rejects_invalid_native_leaf():
    f=SimpleNamespace(trees_=[[dict(cif=[[.2,.3],[.1,.4]])]])
    with pytest.raises(ValueError,match='leaf CIF'):forest_envelope(f)
