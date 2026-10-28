import numpy as np
import pytest
from experiments.bootstrap_stability_50_v1.intervals import paired_u_interval,anchored_interval


def test_jackknife_matches_explicit_deletions():
    rng=np.random.default_rng(992);x=rng.normal(size=(11,7));h=np.sqrt(((x[:,None]-x[None,:])**2).mean(2))
    result=paired_u_interval(h)
    leave=[]
    for i in range(len(h)):
        sub=np.delete(np.delete(h,i,0),i,1);leave.append(sub[np.triu_indices(len(sub),1)].mean())
    expected=np.sqrt((len(h)-1)/len(h)*np.sum((leave-np.mean(leave))**2))
    assert result['jackknife_se']==pytest.approx(expected)
    assert result['estimate']==pytest.approx(h[np.triu_indices(len(h),1)].mean())


def test_mean_kernel_has_mean_standard_error_and_pair_invariance():
    x=np.array([-3.,1.,4.,7.,8.,12.]);h=(x[:,None]+x[None,:])/2;np.fill_diagonal(h,0)
    result=paired_u_interval(h)
    assert result['estimate']==pytest.approx(x.mean())
    assert result['jackknife_se']==pytest.approx(x.std(ddof=1)/np.sqrt(len(x)))
    ix=[3,5,2,4,0,1];other=paired_u_interval(h[np.ix_(ix,ix)])
    for k in ['estimate','jackknife_se','ci_low','ci_high']:assert result[k]==pytest.approx(other[k])
    reverse=paired_u_interval(-h);assert reverse['ci_low']==pytest.approx(-result['ci_high'])


def test_intervals_reject_invalid_pairs_and_distinguish_distribution():
    with pytest.raises(ValueError):paired_u_interval(np.ones((5,5)))
    with pytest.raises(ValueError):paired_u_interval(np.zeros((3,3)))
    h=np.zeros((5,5));h[0,1]=1
    with pytest.raises(ValueError):paired_u_interval(h)
    d=np.linspace(-1,1,50);r=anchored_interval(d)
    assert r['distribution_low']==pytest.approx(-.95)
    assert r['ci_low']>r['distribution_low']
    assert r['n_replicates']==50
