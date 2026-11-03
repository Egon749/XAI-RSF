from pathlib import Path
import json
import numpy as np
from experiments.post_st_drop9_support_v1.common import configuration as old_configuration
from experiments.post_st_ablation_v1.common import pairing,array_hash
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions

NAME='post_st_followup_cap_v1'
REPO=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/drop9_support_20261102_p4gfo7x4')
METADATA_SOURCE=SOURCE.parent/'cause_support_20261102_run1'
YEARS=[2023,2024,2025]
SEEDS=[42,101,202]
ARM_CAPS={'cap12':12,'cap24':24,'cap36':36,'uncapped':None}
ARMS=list(ARM_CAPS)
NEW_ARMS=['cap12','cap36','uncapped']
TIMES=np.array([365.,366.,730.,731.])
ANALYSIS_SEED=2026110201

def load(path):return json.loads(Path(path).read_text())
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def configuration(seed,arm,trees=32):
    if arm not in ARMS:raise ValueError(arm)
    return old_configuration(seed,trees=trees)
