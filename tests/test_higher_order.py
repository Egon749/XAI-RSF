from itertools import product
from dataclasses import replace
import numpy as np
import pytest
from crforest import ForestConfig,CompetingRiskForest,CIFExpansion,HigherOrderCIFExpansion
from experiments.shared_partition_v1.engine import fit_shard,assemble


def planted(q):
    f=CompetingRiskForest(ForestConfig(n_trees=1,path_budget=q,max_depth=q))
    f.grid_=np.array([1.,2.]);f.n_features_=q;f.candidate_evaluations_=0;nodes=[]
    def grow(depth,all_right):
        i=len(nodes);n=dict(ancestors=list(range(depth)),depth=depth);nodes.append(n)
        if depth==q:
            n['cif']=[[.01+.1*all_right,.01],[.02+.2*all_right,.03]]
        else:
            n.update(feature=depth,threshold=0.,missing_left=depth%2==0)
            n['left']=grow(depth+1,False);n['right']=grow(depth+1,all_right)
        return i
    grow(0,True);f.trees_=[nodes];f.audit_=f.audit();return f


@pytest.mark.parametrize('q',[3,4])
def test_genuine_high_order_exact_all_cells_centering_and_reload(q,tmp_path):
    f=planted(q);ref=np.array(list(product([-1.,0.,1.,np.nan],repeat=q)))
    ex=HigherOrderCIFExpansion.from_forest(f,ref);times=[0.,1.,1.5,2.,3.]
    expected=f.predict(ref,times);total=np.broadcast_to(ex._coeff(times)[1],expected.shape).copy()
    assert len(ex.keys)==2**q-1
    for key in ex.keys:
        c=ex.component(ref,key,times);total+=c
        # Integrate one active coordinate while fixing every other coordinate.
        cube=c.reshape(*([4]*q),len(times),2)
        for axis in key:np.testing.assert_allclose(cube.mean(axis=axis),0,atol=1e-14)
    np.testing.assert_allclose(total,expected,atol=1e-14,rtol=0)
    np.testing.assert_allclose(ex.predict(ref,times),expected,atol=1e-14,rtol=0)
    orders=ex.predict_by_order(ref,times)
    np.testing.assert_allclose(orders.sum(1),expected,atol=1e-14,rtol=0)
    assert abs(orders[:,q]).max()>.001  # Cannot pass by silently exporting only q2.
    assert abs(orders[:,:3].sum(1)-expected).max()>.001
    # Independently known highest-order formula for this planted product model.
    indicator=np.column_stack([(ref[:,j]>0)|(np.isnan(ref[:,j])&(j%2==1)) for j in range(q)])
    formula=.2*np.prod(indicator-indicator.mean(0),axis=1)
    np.testing.assert_allclose(ex.component(ref,tuple(range(q)),[2])[:,0,0],formula,atol=1e-14)
    ex.save(tmp_path/'functions.npz');loaded=HigherOrderCIFExpansion.load(tmp_path/'functions.npz')
    np.testing.assert_array_equal(loaded.predict(ref,times),ex.predict(ref,times))
    cc=ex.complexity(ref);assert cc[f'order{q}_functions']==1
    assert cc['spatial_table_cells']==sum(3**len(k) for k in ex.keys)
    for key in ex.keys:
        for j in key:np.testing.assert_array_equal(ex.cuts(key,j),[0.])


@pytest.mark.parametrize('q',[2,3,4])
def test_fitted_forest_missing_shared_cuts_and_shard_parity(q,tmp_path):
    rng=np.random.default_rng(941);X=rng.normal(size=(250,5));X[::11,1]=np.nan
    t=rng.integers(1,15,len(X)).astype(float);event=rng.integers(0,3,len(X));groups=np.arange(len(X))//2
    cfg=ForestConfig(n_trees=4,max_depth=6,path_budget=q,min_leaf=8,min_events=1,min_groups=3,mtry=5,
                     n_thresholds=3,cut_strategy='shared_quantile',tree_feature_count=5)
    f=CompetingRiskForest(cfg).fit(X,t,event,groups);assert f.audit()['max_path_variables']<=q
    ex=HigherOrderCIFExpansion.from_forest(f,X);times=[0,2,8,15]
    np.testing.assert_allclose(ex.predict(X,times),f.predict(X,times),atol=1e-13,rtol=0)
    total=np.broadcast_to(ex._coeff(times)[1],f.predict(X,times).shape).copy()
    for key in ex.keys:total+=ex.component(X,key,times)
    np.testing.assert_allclose(total,f.predict(X,times),atol=1e-13,rtol=0)
    data=dict(X=X,time=t,event=event,groups=groups)
    joined=assemble([(2,fit_shard(cfg,data,2,2)),(0,fit_shard(cfg,data,0,2))])
    f.save(tmp_path/'serial.json');joined.save(tmp_path/'shards.json')
    assert (tmp_path/'serial.json').read_bytes()==(tmp_path/'shards.json').read_bytes()
    if q==2:
        old=CIFExpansion.from_forest(f,X)
        for key in ex.keys:np.testing.assert_allclose(old.component(X,key,times),ex.component(X,key,times),atol=1e-14,rtol=0)
        a,b=old.complexity(X),ex.complexity(X)
        for k in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction']:assert a[k]==b[k]


def test_high_order_export_validation_and_stump():
    f=planted(4);ex=HigherOrderCIFExpansion.from_forest(f,np.zeros((4,4)))
    for key in [(0,0),(2,1),(-1,),(4,),(),(0,1,2,3,4)]:
        with pytest.raises(ValueError):ex.component(np.zeros((2,4)),key)
    with pytest.raises(ValueError):ex.predict(np.full((1,4),np.inf))
    X=np.zeros((20,4));t=np.arange(1,21);e=np.ones(20,int)
    f=CompetingRiskForest(ForestConfig(n_trees=1,path_budget=4,max_depth=0)).fit(X,t,e)
    ex=HigherOrderCIFExpansion.from_forest(f,X)
    assert not ex.keys
    np.testing.assert_allclose(ex.predict(X),f.predict(X),atol=1e-14)
    assert ex.complexity(X)['spatial_table_cells']==0
