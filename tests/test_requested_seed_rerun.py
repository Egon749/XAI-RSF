from dataclasses import asdict
from experiments.post_st_requested_seeds_v1.common import *

def test_requested_seed_grid_always_refits_and_preserves_budget():
    assert set(SEEDS)=={1009,503,17,202,101,419,300,42} and len(SEEDS)==8
    assert REUSED_SEEDS==[] and len(HISTORICAL_SEEDS)==7 and TREES==32
    rows=task_rows('pilot')+task_rows('formal')
    assert len(rows)==len({(z['year'],z['seed'])for z in rows})==24
    assert task_rows('pilot')==[dict(year=2024,seed=300)]
    ref=asdict(configuration(42,32))
    for s in SEEDS:assert all(v==ref[k]for k,v in asdict(configuration(s,32)).items()if k!='seed')
