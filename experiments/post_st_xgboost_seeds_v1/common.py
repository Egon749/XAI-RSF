from pathlib import Path
from experiments.post_st_c061_xgboost_v1.common import read,write,sha,arrays,YEARS,source
from experiments.post_st_c061_xgboost_v1.common import run as previous_run
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=(42,101,202)
OLD_SEEDS=(17,155,8679)
def run():return Path(read(HERE/'run_location.json')['private_run'])
def config(y,h):return read(HERE/'FROZEN_PROTOCOL.json')['configurations'][f'{y}/{h}']
