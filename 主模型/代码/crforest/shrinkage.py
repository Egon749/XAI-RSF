"""Opt-in one-level parent-AJ mixing for saved forests, with unchanged partitions.

This changes the leaf estimator, not the split search. It is not a fitted
calibration layer or a reproduction of the recursive hierarchical-shrinkage
algorithm. Use the same original training data/order and bootstrap seed.
"""
from copy import deepcopy
import numpy as np
from .core import aalen_johansen, cluster_bootstrap


def shrink_leaf_cifs(forest, X, time, event, groups, strength=10.0):
    """Return a copy with w*leaf_AJ+(1-w)*direct_parent_AJ at each leaf.

    w = distinct in-bag companies / (distinct companies + strength).
    The weight is constant over time and shared by all causes, preserving CIF
    validity. Parent AJ uses the SAME bootstrap rows, before any shrinkage.
    Roots that are already leaves stay unchanged. Zero strength is an exact
    no-op copy. Existing forest/defaults are never modified.
    """
    if isinstance(strength, (bool, np.bool_)) or not np.isscalar(strength):
        raise ValueError('strength must be a finite nonnegative scalar')
    strength=float(strength)
    if not np.isfinite(strength) or strength<0:
        raise ValueError('strength must be a finite nonnegative scalar')
    if any('leaf_shrinkage' in n for tree in forest.trees_ for n in tree):
        raise ValueError('already regularized; use the original unshrunk forest')
    X=np.asarray(X,float);time=np.asarray(time,float);event=np.asarray(event);groups=np.asarray(groups)
    n=len(X)
    if (X.ndim!=2 or X.shape[1]!=forest.n_features_ or np.isinf(X).any() or n==0 or
        time.shape!=(n,) or event.shape!=(n,) or groups.shape!=(n,) or
        not np.isfinite(time).all() or (time<=0).any() or not np.isfinite(event).all() or
        (event!=np.floor(event)).any() or (event<0).any() or (event>forest.config.n_causes).any()):
        raise ValueError('original aligned finite-or-missing training data required')
    event=event.astype(int)
    expected_grid=np.unique(time[event>0])
    if not len(expected_grid):expected_grid=np.array([0.])
    if not np.array_equal(forest.grid_,expected_grid):
        raise ValueError('training event grid differs from the saved forest')
    result=deepcopy(forest)
    for tree_id,tree in enumerate(result.trees_):
        sampled=cluster_bootstrap(groups,forest.config.seed,tree_id)
        def visit(index,ids,parent_index=None,parent_curve=None):
            node=tree[index];unique=int(len(np.unique(groups[ids])))
            causes=[int(len(np.unique(groups[ids[event[ids]==k]]))) for k in range(1,forest.config.n_causes+1)]
            if node['n']!=len(ids) or node['groups']!=unique or node['cause_company_support']!=causes:
                raise ValueError('training/bootstrap node membership differs from saved forest')
            raw=aalen_johansen(time[ids],event[ids],forest.grid_,forest.config.n_causes)
            if 'cif' in node:
                if not np.allclose(raw,np.asarray(node['cif']),atol=1e-14,rtol=0):
                    raise ValueError('training outcomes do not reproduce the saved leaf AJ')
                if strength==0:return
                weight=1. if parent_curve is None else unique/(unique+strength)
                prior=raw if parent_curve is None else parent_curve
                mixed=weight*raw+(1-weight)*prior
                if (mixed.min() < -1e-12 or mixed.sum(1).max()>1+1e-12 or
                    (len(mixed)>1 and np.diff(mixed,axis=0).min() < -1e-12)):
                    raise AssertionError('invalid mixed CIF; no clipping is permitted')
                node['cif']=mixed.tolist()
                node['leaf_shrinkage']=dict(method='direct_parent_AJ_convex_mix_v1',strength=strength,
                    weight=float(weight),unique_inbag_companies=unique,parent_node=parent_index,
                    raw_leaf_cif=raw.tolist(),raw_parent_cif=prior.tolist())
                return
            z=X[ids,node['feature']]
            left=(z<=node['threshold'])|(np.isnan(z)&node['missing_left'])
            visit(node['left'],ids[left],index,raw)
            visit(node['right'],ids[~left],index,raw)
        visit(0,sampled)
    result.audit_=result.audit()
    return result
