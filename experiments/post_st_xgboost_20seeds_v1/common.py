from pathlib import Path
import random
import numpy as np
from experiments.post_st_c061_xgboost_v1.common import read,write,sha,arrays,YEARS,source
from experiments.post_st_c061_xgboost_v1.common import run as previous_run
from experiments.post_st_xgboost_seeds_v1.common import run as preceding_run
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=tuple(read(HERE/'FROZEN_PROTOCOL.json')['new_seeds'])
OLD_SEEDS=(17,155,8679)
PRECEDING_SEEDS=(42,101,202)
def run():return Path(read(HERE/'run_location.json')['private_run'])
def config(y,h):return read(HERE/'FROZEN_PROTOCOL.json')['configurations'][f'{y}/{h}']
def draw_seeds():return random.Random(2026110701).sample(range(1,2147483647),20)
def seed_batches():return [SEEDS[i:i+5] for i in range(0,len(SEEDS),5)]
def summarize(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    out=dict(defined=len(a),undefined=len(values)-len(a))
    keys=['min','p05','median','p95','max']
    out.update(dict(zip(keys,np.quantile(a,[0,.05,.5,.95,1]) if len(a) else [np.nan]*5)))
    out.update(mean=np.mean(a) if len(a) else np.nan,sd=np.std(a,ddof=1) if len(a)>1 else np.nan)
    return out
