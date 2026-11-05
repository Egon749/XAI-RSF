"""Independent direct-parent estimator replay for arbitrary frozen strength."""
import numpy as np
from crforest.core import cluster_bootstrap
from experiments.post_st_leaf_shallow_v1.audit import direct_aj

def check_shrinkage(raw, derived, tr, strength):
    x,t,e,g=[tr[k] for k in ['X','time','event','groups']]
    assert raw.config==derived.config and raw.n_features_==derived.n_features_
    np.testing.assert_array_equal(raw.grid_,derived.grid_)
    errors=[];leaves=[]
    for ti,(a,b) in enumerate(zip(raw.trees_,derived.trees_)):
        assert len(a)==len(b)
        for na,nb in zip(a,b):
            assert {k:v for k,v in na.items() if k!='cif'}=={k:v for k,v in nb.items() if k not in ['cif','leaf_shrinkage']}
        def visit(index,ids,parent=None,parent_index=None):
            na=a[index];nb=b[index];m=len(np.unique(g[ids]))
            assert m==na['groups'] and len(ids)==na['n']
            support=[len(np.unique(g[ids[e[ids]==k]])) for k in [1,2]]
            assert support==na['cause_company_support']
            curve=direct_aj(t[ids],e[ids],raw.grid_)
            if 'cif' in na:
                w=1. if parent is None or strength==0 else m/(m+strength)
                prior=curve if parent is None else parent
                target=w*curve+(1-w)*prior
                errors.append(float(abs(target-np.asarray(nb['cif'])).max()))
                np.testing.assert_allclose(na['cif'],curve,rtol=0,atol=1e-13)
                if strength:
                    meta=nb['leaf_shrinkage'];assert meta['strength']==strength and meta['weight']==w and meta['parent_node']==parent_index
                    assert meta['unique_inbag_companies']==m
                    np.testing.assert_allclose(meta['raw_parent_cif'],prior,rtol=0,atol=1e-13)
                else:assert na==nb
                leaves.append(dict(unique_companies=m,weight=w,termination_companies=support[0],
                    borrowed_target=bool(support[0]==0 and target[:,0].max()>0)))
                return
            z=x[ids,na['feature']];left=(z<=na['threshold'])|(np.isnan(z)&na['missing_left'])
            visit(na['left'],ids[left],curve,index);visit(na['right'],ids[~left],curve,index)
        visit(0,cluster_bootstrap(g,raw.config.seed,ti))
    err=max(errors);assert err<=1e-10
    return dict(independent_shrinkage_error=err,structure_unchanged=True,leaves=len(leaves),
        mean_leaf_weight=float(np.mean([r['weight'] for r in leaves])),
        minimum_leaf_weight=float(min(r['weight'] for r in leaves)),
        zero_target_leaves=sum(r['termination_companies']==0 for r in leaves),
        borrowed_target_leaves=sum(r['borrowed_target'] for r in leaves))
