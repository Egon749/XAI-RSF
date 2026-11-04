import numpy as np
from .bootstrap import assessment
from experiments.post_st_requested_seeds_v1.analyze import fast_metrics
def test_replicated_company_counts_match_legacy_tied_rankings():
    y=np.array([1,0,1,0,0,1,0]);p=np.array([.2,.2,.3,.4,.5,.5,.5]);yr=np.array([1,1,1,1,2,2,2]);ids=np.array(list('abcdefg'))
    ix=np.array([0,0,2,3,4,5,5]);weights=np.bincount(ix,minlength=len(y))[None,:]
    val=assessment(weights,y,p,yr,ids)[0];orig=fast_metrics(y[ix].astype(bool),p[ix],yr[ix],ids[ix])
    np.testing.assert_allclose(val[[0,1,2,3,8,9]],orig[[0,8,1,2,6,7]],atol=1e-14,rtol=0)
