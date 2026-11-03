from dataclasses import asdict
from experiments.post_st_requested_seeds11_v1.common import *

def test_eleven_seed_grid_and_independent_version_exports():
    assert SEEDS==[17,1009,503,101,3003,3509,2903,2560,3721,4590,5001]
    assert len(set(SEEDS))==11 and set(NEW_SEEDS).isdisjoint(REUSED_SEEDS)
    rows=task_rows('pilot')+task_rows('formal')
    assert len(rows)==21 and {(r['year'],r['seed'])for r in rows}=={(y,s)for y in YEARS for s in NEW_SEEDS}
    assert len(REUSED_SEEDS)*len(YEARS)==12 and len(SEEDS)*TREES==352
    exports=task_rows('ensemble')
    assert len(exports)==6 and {(r['year'],r['version'])for r in exports}=={(y,v)for y in YEARS for v in ['raw','shrunk']}

def test_eleven_seed_configuration_only_changes_seed():
    baseline=asdict(configuration(42,32));baseline.pop('seed')
    for seed in SEEDS:
        cfg=asdict(configuration(seed,32));assert cfg.pop('seed')==seed;assert cfg==baseline
