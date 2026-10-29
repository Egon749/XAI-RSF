"""Hand-constructed trees only: no new model fit is needed for this audit."""
import copy
from itertools import product
import numpy as np
import pytest
pytest.importorskip('shap',reason='install .[benchmarks] for the complete explanation audit')
from experiments.shap_alignment_v1.methods import (
    model_dict,product_adapter,predict,route,coalition_interactions,
    product_components,default_tree_shap,shapley_weights,rsf_adapter,load_rsf_export)


def correlated_pair_fixture():
    ref=np.repeat(np.array(list(product([-1.,.5,2.],[-1.,1.],[0.]))),[8,2,1,3,3,7],axis=0)
    t=dict(features=np.array([0,1,-1,-1,0,-1,-1]),
           children_left=np.array([1,2,-1,-1,5,-1,-1]),
           children_right=np.array([4,3,-1,-1,6,-1,-1]),
           children_default=np.array([1,2,-1,-1,5,-1,-1]),
           thresholds=np.array([0.,0.,0.,0.,1.,0.,0.]),
           values=np.array([0.,0.,.1,.4,0.,.7,.9])[:,None],
           node_sample_weight=np.array([24.,10.,8.,2.,14.,4.,10.]))
    return model_dict([t]),ref


def brute_full_coalitions(model,X,bg):
    p=X.shape[1];w,ww,pairs=shapley_weights(p)
    v=np.zeros((len(X),2**p))
    for r,x in enumerate(X):
        for mask in range(2**p):
            z=bg.copy();cols=[j for j in range(p) if mask>>j&1];z[:,cols]=x[cols]
            v[r,mask]=predict(model,z).mean()
    phi=v@w;Phi=np.zeros((len(X),p,p));inter=v@ww
    for a,(j,l) in enumerate(pairs):Phi[:,j,l]=Phi[:,l,j]=inter[:,a]
    ii=np.arange(p);Phi[:,ii,ii]=phi-Phi.sum(2)
    return v[:,0],phi,Phi


def test_product_tree_shap_matches_cartesian_coalitions_and_q2_functions():
    m,ref=correlated_pair_fixture();x=ref[[0,8,10,14]]
    bg=np.array([[a,b,0.] for a,b in product(ref[:,0],ref[:,1])])
    q=product_adapter(m,ref);z=default_tree_shap(q,x)
    base,phi,Phi=brute_full_coalitions(m,x,bg)
    bb,pp,PP=coalition_interactions(q,x)
    np.testing.assert_allclose(z['phi'],phi,rtol=0,atol=1e-14)
    np.testing.assert_allclose(z['Phi'],Phi,rtol=0,atol=1e-14)
    np.testing.assert_allclose(pp,phi,rtol=0,atol=1e-14)
    np.testing.assert_allclose(PP,Phi,rtol=0,atol=1e-14)
    np.testing.assert_allclose(bb,base,rtol=0,atol=1e-14)
    c=product_components(m,x,ref)
    for key,v in c['terms'].items():
        got=Phi[:,key[0],key[0]] if len(key)==1 else 2*Phi[:,key[0],key[1]]
        np.testing.assert_allclose(got,v,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(predict(q,x),predict(m,x))
    old=default_tree_shap(m,x)
    assert np.max(abs(old['Phi']-Phi))>.001
    assert np.max(abs(old['base']+old['phi'].sum(1)-predict(m,x)))<1e-14


def test_default_tree_shap_matches_independent_path_game_including_repeated_split():
    m,ref=correlated_pair_fixture();x=ref[:12]
    z=default_tree_shap(m,x);base,phi,Phi=coalition_interactions(m,x)
    np.testing.assert_allclose(phi,z['phi'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(Phi,z['Phi'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(base,z['base'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(z['adapter_prediction'],predict(m,x),atol=1e-14,rtol=0)


def test_missing_only_split_and_q2_native_prediction_are_unchanged():
    m,_=correlated_pair_fixture();t=copy.deepcopy(m['trees'][0])
    # A missing-only stump is an exact supported region, not median imputation.
    t={k:v[:3].copy() for k,v in t.items()};t['features'][:]=[0,-1,-1]
    t['children_left'][:]=[1,-1,-1];t['children_right'][:]=[2,-1,-1]
    t['children_default'][:]=[1,-1,-1];t['thresholds'][:]=[-np.inf,0,0]
    t['values'][:,0]=[0,.7,.2];t['node_sample_weight'][:]=[4,1,3]
    m=model_dict([t]);ref=np.array([[np.nan],[0.],[1.],[2.]])
    q=product_adapter(m,ref);z=default_tree_shap(q,ref)
    np.testing.assert_allclose(predict(m,ref),[.7,.2,.2,.2],atol=0,rtol=0)
    np.testing.assert_allclose(z['adapter_prediction'],predict(m,ref),atol=1e-14,rtol=0)
    np.testing.assert_allclose(z['base']+z['phi'].sum(1),predict(m,ref),atol=1e-14,rtol=0)


def test_three_way_signal_is_allocated_to_features_and_shap_interaction_matrix():
    ref=np.array(list(product([-1.,1.],repeat=3)));nodes=[]
    def grow(d,signs):
        i=len(nodes);nodes.append(None)
        if d==3:
            nodes[i]=(-1,-1,-1,0.,.2+.1*np.prod(signs),1.)
        else:
            left=grow(d+1,signs+[-1]);right=grow(d+1,signs+[1])
            nodes[i]=(d,left,right,0.,0.,2**(3-d))
        return i
    grow(0,[]);a=np.array(nodes)
    t=dict(features=a[:,0].astype(int),children_left=a[:,1].astype(int),children_right=a[:,2].astype(int),
           children_default=a[:,1].astype(int),thresholds=a[:,3],values=a[:,4,None],node_sample_weight=a[:,5])
    m=model_dict([t]);z=default_tree_shap(m,ref);signal=.1*ref.prod(1)
    np.testing.assert_allclose(z['phi'],np.repeat(signal[:,None]/3,3,axis=1),atol=1e-14,rtol=0)
    for j in range(3):
        np.testing.assert_allclose(z['Phi'][:,j,j],-signal/6,atol=1e-14,rtol=0)
        for l in range(j+1,3):np.testing.assert_allclose(z['Phi'][:,j,l],signal/4,atol=1e-14,rtol=0)
    cc=product_components(m,ref,ref)
    np.testing.assert_allclose(cc['order'][:,1:3],0,atol=1e-14,rtol=0)
    np.testing.assert_allclose(cc['order'][:,3],signal,atol=1e-14,rtol=0)


def test_rsf_terminal_na_threshold_is_a_nonpredictive_placeholder():
    raw=dict(ntree=1,trees=[dict(feature=[0,-1,-1],threshold=[.5,'NA','NA'],
        values=[0,.1,.8],cover=[5,2,3],leaf_id=[1,1,2])])
    m=rsf_adapter(raw)
    np.testing.assert_array_equal(predict(m,np.array([[.5],[.6]])),[.1,.8])


def test_binary_rsf_payload_preserves_cut_boundary_lost_by_text(tmp_path):
    import json
    cut=np.nextafter(.5,np.inf)
    raw=dict(ntree=1,trees=[dict(feature=[0,-1,-1],threshold=[.5,'NA','NA'],
        values=[0,.1,.8],cover=[5,2,3],leaf_id=[1,1,2])])
    (tmp_path/'forest_export_PRIVATE.json').write_text(json.dumps(raw))
    np.array([cut,np.nan,np.nan],dtype='<f8').tofile(tmp_path/'tree_001_cuts_PRIVATE.bin')
    np.array([0.,.1,.8],dtype='<f8').tofile(tmp_path/'tree_001_values_PRIVATE.bin')
    m=rsf_adapter(load_rsf_export(tmp_path))
    np.testing.assert_array_equal(predict(m,np.array([[cut],[np.nextafter(cut,np.inf)]])),[.1,.8])
