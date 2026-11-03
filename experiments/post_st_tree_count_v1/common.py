from pathlib import Path
from dataclasses import asdict,replace
import json
import numpy as np
from experiments.post_st_drop9_support_v1.common import configuration as original_configuration
from experiments.post_st_ablation_v1.common import pairing,array_hash
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions

NAME='post_st_tree_count_v1'
REPO=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/drop9_support_20261102_p4gfo7x4')
YEARS=[2023,2024,2025]
SEEDS=[42,101,202]
COUNTS=[8,16,32,64,128,256]
TIMES=np.array([365.,366.,730.,731.])
ANALYSIS_SEED=2026110201
def load(p):return json.loads(Path(p).read_text())
def run_path():return Path(load(REPO/'experiments'/NAME/'run_location.json')['private_run'])
def configuration(seed,trees):return original_configuration(seed,trees=trees)
