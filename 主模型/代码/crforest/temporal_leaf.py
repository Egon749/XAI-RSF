"""Opt-in case-weighted AJ leaf replacement; split partitions remain fixed."""
from copy import deepcopy
import numpy as np
from .core import aalen_johansen,cluster_bootstrap


def weighted_aalen_johansen(time,event,grid,weights,n_causes=2):
    time=np.asarray(time,float);event=np.asarray(event);grid=np.asarray(grid,float)
    weights=np.asarray(weights,float)
    if weights.shape!=time.shape or not np.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('one strictly positive finite case weight per observation required')
    reference=aalen_johansen(time,event,grid,n_causes)
    if np.all(weights==weights[0]):return reference
    tt,inv=np.unique(time,return_inverse=True)
    total=np.bincount(inv,weights=weights,minlength=len(tt))
    risk=np.cumsum(total[::-1])[::-1]
    deaths=np.column_stack([np.bincount(inv,weights=weights*(event==k),minlength=len(tt)) for k in range(1,n_causes+1)])
    hazard=deaths/risk[:,None]
    survival_before=np.r_[1.,np.cumprod(1-hazard.sum(1))[:-1]]
    values=np.cumsum(survival_before[:,None]*hazard,axis=0)
    index=np.searchsorted(tt,grid,side='right')-1
    ans=np.zeros_like(reference);valid=index>=0;ans[valid]=values[index[valid]]
    return ans


def temporal_leaf_cifs(forest,X,time,event,groups,weights,strength=10.):
    """Copy an unshrunk forest; replace raw leaf/parent AJ by weighted AJ.

    Original bootstrap multiplicities remain. The SAME positive company weight
    is used in every risk set and both causes; no event duplication or target
    reweighting. Direct-parent mixing retains n_unique/(n_unique+strength), so
    only calendar case weighting changes relative to the matched baseline.
    """
    if isinstance(strength,(bool,np.bool_)) or not np.isscalar(strength):
        raise ValueError('finite nonnegative strength required')
    strength=float(strength)
    if not np.isfinite(strength) or strength<0:raise ValueError('finite nonnegative strength required')
    if any('leaf_shrinkage' in node for tree in forest.trees_ for node in tree):
        raise ValueError('use original unshrunk forest, not an already updated forest')
    X=np.asarray(X,float);time=np.asarray(time,float);event=np.asarray(event);groups=np.asarray(groups);weights=np.asarray(weights,float)
    n=len(X)
    if X.ndim!=2 or n==0 or X.shape[1]!=forest.n_features_ or np.isinf(X).any() or groups.shape!=(n,) or time.shape!=(n,):
        raise ValueError('aligned original finite-or-missing training data required')
    weighted_aalen_johansen(time,event,forest.grid_,weights,forest.config.n_causes)
    expected=np.unique(time[event>0]);expected=expected if len(expected) else np.array([0.])
    if not np.array_equal(expected,forest.grid_):raise ValueError('event grid differs from saved forest')
    # Cluster weights must not differ across records of the same company.
    for g in np.unique(groups):
        if np.ptp(weights[groups==g])>0:raise ValueError('case weights must be constant within company')
    result=deepcopy(forest)
    for tree_id,tree in enumerate(result.trees_):
        sampled=cluster_bootstrap(groups,forest.config.seed,tree_id)
        def visit(index,ids,parent_index=None,parent_curve=None):
            node=tree[index];unique,inverse=np.unique(groups[ids],return_inverse=True)
            causes=[int(len(np.unique(groups[ids[event[ids]==k]]))) for k in range(1,forest.config.n_causes+1)]
            if node['n']!=len(ids) or node['groups']!=len(unique) or node['cause_company_support']!=causes:
                raise ValueError('bootstrap node membership differs from saved forest')
            curve=weighted_aalen_johansen(time[ids],event[ids],forest.grid_,weights[ids],forest.config.n_causes)
            if 'cif' in node:
                raw=aalen_johansen(time[ids],event[ids],forest.grid_,forest.config.n_causes)
                if not np.allclose(raw,node['cif'],atol=1e-14,rtol=0):raise ValueError('original outcomes do not reproduce saved leaf')
                alpha=1. if parent_curve is None else len(unique)/(len(unique)+strength)
                prior=curve if parent_curve is None else parent_curve
                mixed=alpha*curve+(1-alpha)*prior
                if mixed.min()<-1e-12 or mixed.sum(1).max()>1+1e-12 or (len(mixed)>1 and np.diff(mixed,axis=0).min()<-1e-12):
                    raise AssertionError('invalid weighted CIF; clipping is prohibited')
                wg=np.bincount(inverse,weights=weights[ids]);effective=float(wg.sum()**2/(wg@wg))
                node['cif']=mixed.tolist()
                node['leaf_shrinkage']=dict(method='time_weighted_direct_parent_AJ_v1',strength=strength,weight=float(alpha),
                    unique_inbag_companies=len(unique),effective_weighted_companies=effective,parent_node=parent_index,
                    case_weight_min=float(weights[ids].min()),case_weight_max=float(weights[ids].max()),
                    raw_leaf_cif=curve.tolist(),raw_parent_cif=prior.tolist())
                return
            z=X[ids,node['feature']];left=(z<=node['threshold'])|(np.isnan(z)&node['missing_left'])
            visit(node['left'],ids[left],index,curve);visit(node['right'],ids[~left],index,curve)
        visit(0,sampled)
    result.audit_=result.audit()
    return result
