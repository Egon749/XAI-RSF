import pandas as pd
import pytest
from experiments.post_st_tree_count_local_v1.common import COUNTS,configuration
from experiments.post_st_tree_count_local_v1.selection import rank_candidates
from experiments.post_st_tree_count_local_v1.worker import forbidden_fit
from dataclasses import asdict

def test_local_grid_preserves_equal_seed_sizes_and_all_other_parameters():
    assert COUNTS==list(range(27,41))
    assert [3*n for n in COUNTS]==list(range(81,121,3))
    ref=asdict(configuration(42,32))
    for n in COUNTS:
        cfg=asdict(configuration(42,n))
        assert {k:v for k,v in cfg.items()if k!='n_trees'}=={k:v for k,v in ref.items()if k!='n_trees'}

def test_descriptive_ranking_filters_endpoint_and_uses_frozen_tiebreaks():
    base=dict(scope='pooled',seed='ensemble',version='shrunk',months=12,cause=1,supported=True,within_year_auc=.8,captured=13,ap=.3,brier=.2)
    rows=[dict(base,arm='t32'),dict(base,arm='t33'),dict(base,arm='t31',captured=12),
        dict(base,arm='t30',within_year_auc=.81),dict(base,arm='t29',months=24,within_year_auc=.99),
        dict(base,arm='t28',cause=2,within_year_auc=.99),dict(base,arm='t27',scope='2024',within_year_auc=.99)]
    z=rank_candidates(pd.DataFrame(rows));assert z.arm.tolist()==['t30','t32','t33','t31']
    assert z.status.eq('EXPOSED_TEST_POINT_ORDER_NOT_ADOPTED').all()
    rows[0]['within_year_auc']=float('nan')
    with pytest.raises(ValueError,match='Undefined'):rank_candidates(pd.DataFrame(rows))

def test_local_worker_blocks_new_fitting():
    with pytest.raises(RuntimeError,match='forbids'):forbidden_fit()

