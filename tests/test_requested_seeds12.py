from dataclasses import asdict
from experiments.post_st_requested_seeds12_v1.common import *

def test_requested_twelve_grid_and_reuse_accounting():
    assert SEEDS==[17,1009,503,42,101,99,1000,889,24,555,499,1010]
    assert len(set(SEEDS))==12 and set(NEW_SEEDS).isdisjoint(REUSED_SEEDS)
    rows=task_rows('pilot')+task_rows('formal')
    assert len(rows)==21 and len({(r['year'],r['seed'])for r in rows})==21
    assert {(r['year'],r['seed'])for r in rows}=={(y,s)for y in YEARS for s in NEW_SEEDS}
    assert len(task_rows('ensemble'))==3 and 12*TREES==384

def test_requested_twelve_only_changes_seed():
    base=asdict(configuration(42,32));base.pop('seed')
    for seed in SEEDS:
        cfg=asdict(configuration(seed,32));assert cfg.pop('seed')==seed;assert cfg==base
