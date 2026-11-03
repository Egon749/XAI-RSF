from dataclasses import asdict
from experiments.post_st_seed_stability20_v1.common import SEEDS,REUSED_SEEDS,TREES,PILOT,configuration,task_rows

def test_finite_seed_matrix_and_reuse():
    assert len(SEEDS)==len(set(SEEDS))==20 and set(REUSED_SEEDS)<set(SEEDS) and TREES==32
    rows=task_rows('pilot')+task_rows('formal')
    assert len(rows)==60 and len({(r['year'],r['seed'])for r in rows})==60
    assert len([r for r in rows if r['seed']not in REUSED_SEEDS])==51
    assert task_rows('pilot')==[dict(year=PILOT[0],seed=PILOT[1])]

def test_seed_and_budget_only_changes():
    ref=asdict(configuration(42,32))
    for seed in SEEDS:
        cfg=asdict(configuration(seed,TREES))
        assert all(v==ref[k]for k,v in cfg.items()if k!='seed')
