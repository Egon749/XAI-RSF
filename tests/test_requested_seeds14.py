from dataclasses import asdict
from experiments.post_st_requested_seeds14_v1.common import *

def test_fourteen_seed_grid_and_independent_version_exports():
    assert SEEDS==[98,17,503,5001,2570,22,33,15,99,95,500,2580,97,94]
    assert len(REQUESTED_SEEDS)==15 and REQUESTED_SEEDS.count(95)==2 and SEEDS.count(95)==1
    assert len(set(SEEDS))==14 and set(NEW_SEEDS).isdisjoint(REUSED_SEEDS)
    rows=task_rows('pilot')+task_rows('formal')
    assert len(rows)==18 and {(r['year'],r['seed'])for r in rows}=={(y,s)for y in YEARS for s in NEW_SEEDS}
    assert len(REUSED_SEEDS)*len(YEARS)==24 and len(SEEDS)*TREES==448
    exports=task_rows('ensemble')
    assert len(exports)==6 and {(r['year'],r['version'])for r in exports}=={(y,v)for y in YEARS for v in ['raw','shrunk']}

def test_fourteen_seed_configuration_only_changes_seed():
    baseline=asdict(configuration(42,32));baseline.pop('seed')
    for seed in SEEDS:
        cfg=asdict(configuration(seed,32));assert cfg.pop('seed')==seed;assert cfg==baseline
