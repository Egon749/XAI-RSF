from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays,sha,write,load,configuration,pairing,array_hash,SOURCE,TIMES,
    ANALYSIS_SEED,run_path as previous_path,
)
NAME='post_st_requested_seeds12_v1'
REPO=Path(__file__).resolve().parents[2]
SOURCE_COMMIT='7b17941b15d233e44f8aa6276a5bc0d99c1a6eba'
YEARS=[2023,2024,2025]
SEEDS=[17,1009,503,42,101,99,1000,889,24,555,499,1010]
REUSED_SEEDS=[17,1009,503,42,101]
NEW_SEEDS=[s for s in SEEDS if s not in REUSED_SEEDS]
REFERENCE_SEEDS=[42,101,202]
TREES=32
PILOT=(2024,99)
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def task_rows(stage):
    if stage=='pilot':return [dict(year=PILOT[0],seed=PILOT[1])]
    if stage=='formal':return [dict(year=y,seed=s)for y in YEARS for s in NEW_SEEDS if(y,s)!=PILOT]
    if stage=='ensemble':return [dict(year=y)for y in YEARS]
    raise ValueError(stage)
