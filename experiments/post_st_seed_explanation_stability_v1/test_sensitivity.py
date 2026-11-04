"""Small independent mathematical checks for this finite sensitivity audit."""
import numpy as np
from .common import TRIPLES
from .similarity import SHARED,jaccards,correlations,midpercent
from .geometry import stats

def test_exhaustive_overlap_by_counting():
    assert len(set(TRIPLES))==120
    assert [int((SHARED==i).sum()) for i in (0,1,2)]==[2100,3780,1260]
    assert [sum(i in s for s in TRIPLES) for i in range(10)]==[36]*10

def test_empty_union_ties_and_anchor_midrank():
    assert np.isnan(jaccards(np.zeros((2,3)))[0,1])
    assert jaccards([[1,0,1],[1,1,0]])[0,1]==1/3
    assert midpercent([1,2,2,3],2)==50
    assert np.isnan(correlations([[1,1],[0,1]])[0,1])

def test_supported_shape_and_off_support_reversal():
    x=np.tile([-.01,.02,2.,-2.],(120,1));x[::2,2:]*=-1
    n=np.array([3,1,0,0]);p=np.ones(120,bool)
    a,b=stats(x,n,p,'observed')
    np.testing.assert_allclose([a['shape_correlation'],b['sign_agreement'],b['rms_difference']],[1,1,0],atol=1e-10)
    assert b['common_pairs']==2100
    assert a['mean_abs_median']==.0125
    assert stats(x,n,p,'unsupported')[0]['sign_agreement']<=a['sign_agreement']
