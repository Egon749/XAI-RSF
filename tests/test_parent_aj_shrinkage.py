import copy
import numpy as np
import pytest
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,cluster_bootstrap,aalen_johansen
from crforest.shrinkage import shrink_leaf_cifs


def hand_tree(seed=8):
    X=np.array([[-1.],[np.nan],[1.],[2.]])
    time=np.array([1.,2.,1.,2.]);event=np.array([1,0,2,0]);groups=np.arange(4)
    f=CompetingRiskForest(ForestConfig(n_trees=1,max_depth=1,path_budget=2,min_leaf=1,min_events=0,min_groups=1,mtry=1,seed=seed))
    f.n_features_=1;f.grid_=np.array([1.]);f.candidate_evaluations_=2
    rows=cluster_bootstrap(groups,seed,0)
    def node(ids,depth):
        return dict(n=len(ids),groups=len(np.unique(groups[ids])),depth=depth,ancestors=[0] if depth else [],
            events=int((event[ids]>0).sum()),cause_company_support=[len(np.unique(groups[ids[event[ids]==k]])) for k in [1,2]])
    l=rows[(X[rows,0]<=0)|np.isnan(X[rows,0])];r=rows[X[rows,0]>0]
    root=node(rows,0);root.update(feature=0,threshold=0.,missing_left=True,score=1.,left=1,right=2)
    left=node(l,1);right=node(r,1)
    left['cif']=aalen_johansen(time[l],event[l],f.grid_).tolist()
    right['cif']=aalen_johansen(time[r],event[r],f.grid_).tolist()
    f.trees_=[[root,left,right]];f.audit_=f.audit()
    return f,X,time,event,groups


def test_hand_calculated_two_cause_mixture_with_missing_route():
    f,X,t,e,g=hand_tree();before=copy.deepcopy(f.trees_)
    z=shrink_leaf_cifs(f,X,t,e,g,strength=2)
    # Four distinct bootstrap companies. Parent=(1/4,1/4), leaves=(1/2,0),(0,1/2).
    np.testing.assert_array_equal(z.predict(X,[1.])[:,0],[[.375,.125],[.375,.125],[.125,.375],[.125,.375]])
    assert f.trees_==before and z.audit_==f.audit_
    assert all(n['leaf_shrinkage']['weight']==.5 for n in z.trees_[0][1:])


def test_bootstrap_duplicates_do_not_inflate_shrinkage_support():
    f,X,t,e,g=hand_tree(seed=0)
    np.testing.assert_array_equal(cluster_bootstrap(g,0,0),[1,2,2,3])
    z=shrink_leaf_cifs(f,X,t,e,g,strength=2)
    assert z.trees_[0][1]['leaf_shrinkage']['weight']==1/3
    assert z.trees_[0][2]['leaf_shrinkage']['weight']==.5
    np.testing.assert_allclose(z.predict(X,[1.])[[0,2],0],[[0,1/3],[0,7/12]],atol=1e-15,rtol=0)


def test_zero_strength_is_byte_identical_noop(tmp_path):
    f,X,t,e,g=hand_tree();z=shrink_leaf_cifs(f,X,t,e,g,strength=0)
    f.save(tmp_path/'a.json');z.save(tmp_path/'b.json')
    assert (tmp_path/'a.json').read_bytes()==(tmp_path/'b.json').read_bytes()
    assert z is not f


@pytest.mark.parametrize('strength',[-1,np.nan,np.inf,True,[1,2]])
def test_invalid_strength_rejected(strength):
    f,X,t,e,g=hand_tree()
    with pytest.raises(ValueError):shrink_leaf_cifs(f,X,t,e,g,strength)


def test_wrong_training_data_and_double_shrink_rejected():
    f,X,t,e,g=hand_tree()
    changed=e.copy();changed[0]=2
    with pytest.raises(ValueError):shrink_leaf_cifs(f,X,t,changed,g,2)
    changed_X=X.copy();changed_X[0]=1
    with pytest.raises(ValueError):shrink_leaf_cifs(f,changed_X,t,e,g,2)
    z=shrink_leaf_cifs(f,X,t,e,g,2)
    with pytest.raises(ValueError):shrink_leaf_cifs(z,X,t,e,g,2)


def test_root_leaf_no_shrink_and_no_event_grid():
    X=np.arange(20.)[:,None];t=np.arange(1.,21.);e=np.zeros(20,int);g=np.arange(20)
    f=CompetingRiskForest(ForestConfig(n_trees=2,max_depth=0,path_budget=2,mtry=1)).fit(X,t,e,g)
    z=shrink_leaf_cifs(f,X,t,e,g,10)
    np.testing.assert_array_equal(z.predict(X,[0,10,100]),f.predict(X,[0,10,100]))
    assert all(tree[0]['leaf_shrinkage']['weight']==1 for tree in z.trees_)


def test_q2_full_export_reload_and_curves_remain_valid(tmp_path):
    rng=np.random.default_rng(925);X=rng.normal(size=(90,4));X[::9,1]=np.nan
    t=rng.integers(1,50,90).astype(float);e=rng.integers(0,3,90);g=np.arange(90)
    cfg=ForestConfig(n_trees=5,max_depth=4,path_budget=2,min_leaf=5,min_events=1,min_groups=3,mtry=4,n_thresholds=3,seed=42,cut_strategy='shared_quantile')
    f=CompetingRiskForest(cfg).fit(X,t,e,g);z=shrink_leaf_cifs(f,X,t,e,g,10)
    times=np.unique(np.r_[0,z.grid_,100]);pred=z.predict(X,times)
    assert pred.min()>=0 and pred.sum(-1).max()<=1+1e-12 and np.diff(pred,axis=1).min()>=-1e-12
    assert z.audit_==f.audit_ and z.audit_['max_path_variables']<=2
    ex=CIFExpansion.from_forest(z,X)
    np.testing.assert_allclose(ex.predict(X,times),pred,atol=1e-12,rtol=0)
    terms=sum((ex.component(X,k,times) for k in [(j,) for j in ex.mains]+ex.pairs),np.zeros_like(pred))
    np.testing.assert_allclose(ex._coeff(times)[1]+terms,pred,atol=1e-12,rtol=0)
    z.save(tmp_path/'model.json');ex.save(tmp_path/'functions.npz')
    again=CompetingRiskForest.load(tmp_path/'model.json')
    assert again.trees_==z.trees_
    np.testing.assert_array_equal(again.predict(X,times),pred)
    np.testing.assert_allclose(CIFExpansion.load(tmp_path/'functions.npz').predict(X,times),pred,atol=1e-12,rtol=0)
