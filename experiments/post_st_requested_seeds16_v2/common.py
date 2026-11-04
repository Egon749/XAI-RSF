from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays,sha,write,load,configuration,pairing,array_hash,SOURCE,TIMES,
    ANALYSIS_SEED,
)
from experiments.post_st_requested_seeds16_v1.common import run_path as previous_path
NAME='post_st_requested_seeds16_v2'
REPO=Path(__file__).resolve().parents[2]
SOURCE_COMMIT='22ee65b2358281a589b27c27b7ae48cb159ffbb6'
YEARS=[2023,2024,2025]
SEEDS=[505,98,17,503,5001,2570,15,19,20,5399,22,5000,4999,4998,2500,100]
REUSED_SEEDS=[505,98,17,503,5001,2570,15,5399]
NEW_SEEDS=[s for s in SEEDS if s not in REUSED_SEEDS]
REFERENCE_SEEDS=[42,101,202]
TREES=32
PILOT=(2024,19)
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def task_rows(stage):
    if stage=='pilot':return [dict(year=PILOT[0],seed=PILOT[1])]
    if stage=='formal':return [dict(year=y,seed=s)for y in YEARS for s in NEW_SEEDS if(y,s)!=PILOT]
    if stage=='ensemble':return [dict(year=y,version=v)for y in YEARS for v in ['raw','shrunk']]
    raise ValueError(stage)
