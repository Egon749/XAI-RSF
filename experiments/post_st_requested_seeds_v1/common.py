from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_tree_count_v1.common import configuration,pairing,array_hash,arrays,sha,SOURCE,TIMES,ANALYSIS_SEED
NAME='post_st_requested_seeds_v1'
REPO=Path(__file__).resolve().parents[2]
YEARS=[2023,2024,2025]
SEEDS=[1009,503,17,202,101,419,300,42]
HISTORICAL_SEEDS=[s for s in SEEDS if s!=300]
REFERENCE_SEEDS=[42,101,202]
REUSED_SEEDS=[]  # User explicitly requested rerunning, including existing seeds.
TREES=32
PILOT=(2024,300)
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False,default=str);f.write('\n')
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def previous_path():return Path(load(REPO/'experiments/post_st_seed_stability20_v1/run_location.json')['private_run'])
def task_rows(stage):
    if stage=='pilot':return [dict(year=PILOT[0],seed=PILOT[1])]
    if stage!='formal':raise ValueError(stage)
    return [dict(year=y,seed=s)for y in YEARS for s in SEEDS if(y,s)!=PILOT]
