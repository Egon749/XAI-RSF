import numpy as np
import pytest
pytest.importorskip('hyperopt')
from experiments.post_st_eswa_baselines_v1.common import Horizon
from experiments.post_st_eswa_baselines_v1.stacking import Search
from .resampling import BootstrapHorizon,BootstrapSearch,draw_rows

def sample():
    e=np.r_[np.ones(8),np.full(20,2),np.zeros(8)].astype(int)
    rng=np.random.default_rng(1)
    return dict(groups=np.array([str(x) for x in rng.permutation(len(e))]),X=rng.normal(size=(len(e),4)),analysis_time=np.minimum(np.arange(len(e))+1,12),event=e,label_eligible=np.ones(len(e),bool))
def test_identity_preserves_original_partition_and_augmentation():
    d=sample();ix=np.arange(len(d['groups']));original=Horizon(d,12,'IPCW');boot=BootstrapHorizon(d,12,ix)
    for expected,observed in zip(original.split(ix,3,741),boot.split(ix,3,741),strict=True):
        for a,b in zip(expected,observed):np.testing.assert_array_equal(a,b)
    s,t=Search(original).design(ix,np.arange(4)),BootstrapSearch(boot).design(ix,np.arange(4))
    for a,b in zip(s[1:],t[1:]):np.testing.assert_array_equal(a,b)
def test_repeats_never_cross_folds_or_inflate_support():
    d=sample();rows=np.r_[np.repeat([0,1,2],6),np.arange(8,36)];h=BootstrapHorizon(d,12,rows);ix=np.arange(len(rows))
    assert h.positive_firms(ix)==3
    for train,val in h.split(ix,3,741):assert not set(rows[train])&set(rows[val])
    s=BootstrapSearch(h);assert s.design(ix,np.arange(4))[-1]==0
    assert len(s.select(ix))==4 or np.array_equal(s.select(ix),np.arange(74))
def test_reproducible_draws():
    assert len({draw_rows(2023,b,309).tobytes() for b in range(1,51)})==50
    np.testing.assert_array_equal(draw_rows(2025,1,390),draw_rows(2025,1,390))
