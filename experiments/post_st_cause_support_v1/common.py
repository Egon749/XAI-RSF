from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from experiments.post_st_ablation_v1.common import configuration as baseline_configuration, pairing, array_hash
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizon_predictions

NAME='post_st_cause_support_v1'
REPO=Path(__file__).resolve().parents[2]
RUN=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/cause_support_20261102_run1')
PARENT=Path('/home/oliver/MainDiskData/xai_rsf_dropcolumn_selection_20261101_EzUsFtiO')
META=Path('/home/oliver/MainDiskData/xai_rsf_group_selection_v3_20261101_DfdNL6cc/input')
YEARS=[2023,2024,2025]
SEEDS=[42,101,202]
ARMS=['baseline','quadratic','support','quadratic_support']
TIMES=np.array([365.,366.,730.,731.])
ANALYSIS_SEED=2026110201

def load(p):return json.loads(Path(p).read_text())

def configuration(seed,arm,trees=32):
    if arm not in ARMS:raise ValueError(arm)
    return replace(baseline_configuration(seed,'shallow_parent'),n_trees=trees,
        split_score='causewise_quadratic' if arm in ['quadratic','quadratic_support'] else 'signed_sum',
        split_min_target_groups=2 if arm in ['support','quadratic_support'] else 0,split_target_cause=1)

