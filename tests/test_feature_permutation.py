from types import SimpleNamespace
import itertools
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CIFExpansion
from experiments.post_st_feature_permutation_v1.common import (
    score_batch,bootstrap_effect,replacement_lookup,logical_closure,evaluation_context,prohibit_fit)


def test_metric_ties_censor_weights_and_no_known_outcome():
    p=np.array([[.1,.3,.3,.7,.4],[.8,.2,.6,.6,.2]])
    y=np.array([0,1,0,1,1],bool);w=np.array([1.,2.,0.,1.5,1.])
    got=score_batch(p,y,w);use=w>0
    for i in range(len(p)):
        assert abs(got['auc'][i]-roc_auc_score(y[use],p[i,use],sample_weight=w[use]))<1e-14
        assert abs(got['ap'][i]-average_precision_score(y[use],p[i,use],sample_weight=w[use]))<1e-14
        assert abs(got['brier'][i]-np.sum(w*(p[i]-y)**2)/len(y))<1e-14
    assert np.isnan(score_batch(p,y,np.zeros(5))['brier']).all()


def test_averaging_permutation_metrics_is_not_metric_of_average_prediction():
    y=np.array([True,False]);p=np.array([[.8,.9],[.2,.1]]);original=np.array([.8,.1]);w=np.ones(2)
    # Mean AUC is .5, while AUC of mean predictions is .5 here (tie). Use AP
    # to distinguish the nonlinear aggregation explicitly: .75 versus .5.
    assert score_batch(p,y,w)['ap'].mean()==.75
    assert score_batch(p.mean(0),y,w)['ap'][0]==.5
    boot=bootstrap_effect(original,p,y,w,np.array([[1,1],[2,0],[0,2]]))
    assert boot[0,0]==.5 and np.isnan(boot[1:,0]).all()


def test_bootstrap_equals_explicit_company_resampling():
    y=np.array([0,1,0,1],bool);w=np.ones(4);original=np.array([.1,.7,.4,.6])
    p=np.array([[.4,.7,.1,.6],[.7,.1,.4,.6]])
    draws=np.array([[2,1,0,1],[0,1,1,2]])
    got=bootstrap_effect(original,p,y,w,draws)
    for i,counts in enumerate(draws):
        ix=np.repeat(np.arange(4),counts);b=score_batch(original[ix],y[ix],w[ix]);z=score_batch(p[:,ix],y[ix],w[ix])
        assert abs(got[i,0]-(b['auc'][0]-z['auc'].mean()))<1e-14
        assert abs(got[i,1]-(z['brier'].mean()-b['brier'][0]))<1e-14


def test_exact_lookup_all_donors_missing_and_two_variable_replacement():
    tree=[dict(feature=0,threshold=0.,missing_left=True,left=1,right=4),
          dict(feature=1,threshold=1.,missing_left=False,left=2,right=3),
          dict(cif=[[.1,.1],[.2,.2]]),dict(cif=[[.2,.1],[.3,.2]]),dict(cif=[[.3,.1],[.4,.2]])]
    f=SimpleNamespace(n_features_=3,grid_=np.array([1.,2.]),trees_=[tree],audit=lambda:dict(max_path_variables=2))
    X=np.array([[-1.,0.,1.],[-1.,2.,2.],[1.,2.,0.],[np.nan,np.nan,1.]])
    ex=CIFExpansion.from_forest(f,X);base=ex.predict(X)
    for group in [[0],[1],[2],[0,1],[0,2]]:
        lookup=replacement_lookup(ex,X,group,times=np.array([1.,2.]))
        for i,j in itertools.product(range(len(X)),repeat=2):
            xx=X[[i]].copy();xx[:,group]=X[j,group]
            np.testing.assert_allclose(base[i]+lookup[i,j],ex.predict(xx)[0],atol=1e-14,rtol=0)


def test_horizon_unsupported_even_if_some_early_events_known():
    labels=dict(analysis_time=np.array([3.,7.,8.]),event=np.array([1,0,0]),label_eligible=np.ones(3,bool))
    y,w,known,supported,diag=evaluation_context(labels,24,1)
    assert known.tolist()==[True,False,False] and not supported and w[1:].sum()==0
    labels['event']=np.array([1,2,2])
    assert evaluation_context(labels,24,1)[3]


def test_audit_linked_fields_and_fit_prohibited():
    assert logical_closure([78])==list(range(76,83))
    assert logical_closure([0,78])==[0]+list(range(76,83))
    with pytest.raises(RuntimeError,match='prohibits'):prohibit_fit()
