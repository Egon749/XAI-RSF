"""Fixed equal-prediction combinations; no fitting or selection."""
from pathlib import Path
import json
import numpy as np
from experiments.post_st_requested_seeds_v1.common import (
    arrays, sha, write, load, YEARS, TIMES, ANALYSIS_SEED,
    run_path as source_path,
)
REPO=Path(__file__).resolve().parents[2]
NAME='post_st_seed_ensemble_v1'
SOURCE_COMMIT='341b3a33854dc9d52b8d28b75bb7dc81c1f213d6'
ARMS={'reference96':[42,101,202],
      'requested7_224':[1009,503,17,202,101,419,300],
      'requested8_256':[1009,503,17,202,101,419,300,42]}
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def average_predictions(records):
    if not records:raise ValueError('No prediction members')
    first=records[0]
    for z in records:
        np.testing.assert_array_equal(z['sample_ids'],first['sample_ids'])
        np.testing.assert_array_equal(z['times'],first['times'])
        if z['cif'].shape!=first['cif'].shape or not np.isfinite(z['cif']).all():
            raise ValueError('Invalid prediction shape or values')
    return np.mean([z['cif']for z in records],axis=0)
def tasks():return [dict(year=y,arm=a)for y in YEARS for a in ARMS if a!='reference96']
