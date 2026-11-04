"""Exact grouped explanation for arbitrary firm rows and native/prediction times."""
import numpy as np
from .common import PAIRS,allocate

def grouped_components(expansion,X,times,feature_groups):
    """Return the unchanged product-reference base, A (8), I (28) and C (8).
    Group axis is last. For large cohorts, call in row batches. No term or
    missing state is dropped; group codes are 0..7 in model input order.
    """
    groups=np.asarray(feature_groups)
    if groups.shape!=(expansion.n_features,) or not np.issubdtype(groups.dtype,np.integer) or set(groups)!=set(range(8)):
        raise ValueError('one zero-based group per input; all eight groups required')
    _,mu=expansion._coeff(times)
    blocks=np.zeros((len(X),*mu.shape,36),dtype=float)
    for j in expansion.mains:
        blocks[...,groups[j]]+=expansion.component(X,(j,),times)
    for j,k in expansion.pairs:
        a,b=sorted((groups[j],groups[k]))
        slot=a if a==b else 8+PAIRS.index((a,b))
        blocks[...,slot]+=expansion.component(X,(j,k),times)
    return mu,blocks[...,:8],blocks[...,8:],allocate(blocks)
