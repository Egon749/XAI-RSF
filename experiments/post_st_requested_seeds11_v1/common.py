from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays,sha,write,load,configuration,pairing,array_hash,SOURCE,TIMES,
    ANALYSIS_SEED,run_path as previous_path,
)
NAME='post_st_requested_seeds11_v1'
REPO=Path(__file__).resolve().parents[2]
SOURCE_COMMIT='ad61c801c232d54a20ad119ba02db07ecf3df197'
YEARS=[2023,2024,2025]
SEEDS=[17,1009,503,101,3003,3509,2903,2560,3721,4590,5001]
REUSED_SEEDS=[17,1009,503,101]
NEW_SEEDS=[s for s in SEEDS if s not in REUSED_SEEDS]
REFERENCE_SEEDS=[42,101,202]
TREES=32
PILOT=(2024,3003)
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def task_rows(stage):
    if stage=='pilot':return [dict(year=PILOT[0],seed=PILOT[1])]
    if stage=='formal':return [dict(year=y,seed=s)for y in YEARS for s in NEW_SEEDS if(y,s)!=PILOT]
    if stage=='ensemble':return [dict(year=y,version=v)for y in YEARS for v in ['raw','shrunk']]
    raise ValueError(stage)
