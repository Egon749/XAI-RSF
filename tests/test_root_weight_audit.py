import numpy as np
from crforest.core import _split_parent, logrank_cr_score
from experiments.post_st_rule_features_v1.audit_root_weights import components, independent_components


def test_opposite_outcomes_cancel_signed_composite_without_erasing_target_score():
    t=np.ones(20);e=np.r_[np.ones(10,dtype=int),np.full(10,2)]
    left=np.arange(20)<10;parent=_split_parent(t,e,2)
    u,v=components(left,e,parent)
    np.testing.assert_allclose(u,[5.,-5.])
    np.testing.assert_allclose(v,[25/19,25/19])
    assert logrank_cr_score(left,e,parent,[1,1])==0
    assert logrank_cr_score(left,e,parent,[1,0])>4
    assert logrank_cr_score(left,e,parent,[0,1])>4


def test_components_match_independent_risk_sets_with_censoring_and_ties():
    t=np.array([1,1,2,2,3,4,5,6.]);e=np.array([1,0,2,1,2,0,1,2])
    left=np.array([1,0,1,1,0,1,0,0],bool)
    u,v=components(left,e,_split_parent(t,e,2));ui,vi=independent_components(t,e,left)
    np.testing.assert_allclose(u,ui,rtol=0,atol=1e-14)
    np.testing.assert_allclose(v,vi,rtol=0,atol=1e-14)
    for w in [[1,1],[1,0],[0,1]]:
        w=np.array(w);expected=abs(w@u)/np.sqrt((w*w)@v)
        np.testing.assert_allclose(logrank_cr_score(left,e,_split_parent(t,e,2),w),expected,rtol=0,atol=1e-14)
