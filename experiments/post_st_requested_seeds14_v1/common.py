from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays,sha,write,load,configuration,pairing,array_hash,SOURCE,TIMES,
    ANALYSIS_SEED,
)
from experiments.post_st_requested_seeds16_v2.common import run_path as previous_path
from experiments.post_st_requested_seeds12_v1.common import run_path as extra_previous_path
NAME='post_st_requested_seeds14_v1'
REPO=Path(__file__).resolve().parents[2]
SOURCE_COMMIT='7c7c424cda7b39291f6317e64468416f02899f0c'
YEARS=[2023,2024,2025]
REQUESTED_SEEDS=[98,17,503,5001,2570,22,33,15,99,95,500,2580,97,95,94]
SEEDS=list(dict.fromkeys(REQUESTED_SEEDS))
REUSED_SEEDS=[98,17,503,5001,2570,22,15,99]
NEW_SEEDS=[s for s in SEEDS if s not in REUSED_SEEDS]
REFERENCE_SEEDS=[42,101,202]
TREES=32
PILOT=(2024,33)
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def task_rows(stage):
    if stage=='pilot':return [dict(year=PILOT[0],seed=PILOT[1])]
    if stage=='formal':return [dict(year=y,seed=s)for y in YEARS for s in NEW_SEEDS if(y,s)!=PILOT]
    if stage=='ensemble':return [dict(year=y,version=v)for y in YEARS for v in ['raw','shrunk']]
    raise ValueError(stage)
