from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays,sha,write,load,configuration,pairing,array_hash,SOURCE,TIMES,
    ANALYSIS_SEED,
)
from experiments.post_st_requested_seeds11_v1.common import run_path as previous_path
NAME='post_st_requested_seeds16_v1'
REPO=Path(__file__).resolve().parents[2]
SOURCE_COMMIT='7cac54cb618a776bfc839c427a96754a365df96c'
YEARS=[2023,2024,2025]
SEEDS=[5001,503,17,2570,5399,39,486,15,8,5100,6270,116,111,98,505,507]
REUSED_SEEDS=[5001,503,17]
NEW_SEEDS=[s for s in SEEDS if s not in REUSED_SEEDS]
REFERENCE_SEEDS=[42,101,202]
TREES=32
PILOT=(2024,2570)
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def task_rows(stage):
    if stage=='pilot':return [dict(year=PILOT[0],seed=PILOT[1])]
    if stage=='formal':return [dict(year=y,seed=s)for y in YEARS for s in NEW_SEEDS if(y,s)!=PILOT]
    if stage=='ensemble':return [dict(year=y,version=v)for y in YEARS for v in ['raw','shrunk']]
    raise ValueError(stage)
