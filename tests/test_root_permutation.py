import numpy as np
from crforest.core import _split_parent,logrank_cr_score
from experiments.post_st_permutation_v1.permutation import ScoreCache,permutation_indices


def test_pair_permutation_respects_year_blocks_and_is_reproducible():
    year=np.repeat([2019,2020,2021],[4,3,5])
    p=permutation_indices(year,'within_entry_year_pair',42)
    np.testing.assert_array_equal(year[p],year)
    np.testing.assert_array_equal(np.sort(p),np.arange(len(year)))
    np.testing.assert_array_equal(p,permutation_indices(year,'within_entry_year_pair',42))
    assert not np.array_equal(p,permutation_indices(year,'within_entry_year_pair',43))


def test_batch_scores_preserve_original_statistics_after_joint_permutation():
    t=np.array([1,1,2,2,3,4,5,6.]);e=np.array([1,0,2,1,2,0,1,2])
    masks=np.array([[1,0,1,1,0,1,0,0],[0,1,1,1,0,0,0,1]],bool)
    cache=ScoreCache(masks,[0,1],t,e,2,min_events=1)
    for p in [np.arange(8),np.array([4,2,0,7,1,6,5,3])]:
        scores,valid=cache.scores(p);parent=_split_parent(t[p],e[p],2)
        for i in np.flatnonzero(valid):
            for k,w in enumerate([[1,1],[1,0],[0,1]]):
                np.testing.assert_allclose(scores[i,k],logrank_cr_score(masks[i],e[p],parent,w),rtol=0,atol=1e-12)


def test_event_support_is_rechecked_not_frozen_from_observed_labels():
    t=np.arange(1.,9.);e=np.array([1,2,0,0,1,2,0,0]);mask=np.array([[1,1,1,1,0,0,0,0]],bool)
    cache=ScoreCache(mask,[0],t,e,1,min_events=2)
    _,v0=cache.scores(np.arange(8));s1,v1=cache.scores(np.array([0,1,4,5,2,3,6,7]))
    assert v0[0] and not v1[0]
    np.testing.assert_array_equal(s1,0.)
