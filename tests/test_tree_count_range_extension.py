from dataclasses import asdict
from experiments.post_st_tree_count_70_120_v1.common import COUNTS,NEW_COUNTS,configuration

def test_70_120_grid_adds_only_three_missing_equal_seed_prefixes():
    assert COUNTS==list(range(24,41))
    assert [3*n for n in COUNTS]==list(range(72,121,3))
    assert sorted(set(COUNTS)-set(range(27,41)))==NEW_COUNTS==[24,25,26]
    ref=asdict(configuration(42,32))
    for n in COUNTS:
        cfg=asdict(configuration(42,n))
        assert all(v==ref[k]for k,v in cfg.items()if k!='n_trees')
