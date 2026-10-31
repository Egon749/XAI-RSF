"""Two separate, finite changes; the enterprise reference remains immutable."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from experiments.post_st_ablation_v1.common import configuration as base_configuration, pairing, array_hash
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizon_predictions

NAME='post_st_level_coverage_v1'
YEARS=[2023,2024]; SEEDS=[42,101,202]; TIMES=np.array([365.,366.,730.,731.])
HISTORY=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj')
ROLLING=Path('/home/oliver/MainDiskData/xai_rsf_temporal_leaf_20261028__kvhg19v')
SIMULATION=Path('/home/oliver/MainDiskData/xai_rsf_structure_recovery_20261030_eqjtVhJ4')
REFERENCE_SHA='aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65'

def configuration(seed,arm='wide83'):
    cfg=base_configuration(seed,'shallow_parent')
    if arm=='wide83': return replace(cfg,tree_feature_count=83)
    if arm=='reference': return cfg
    raise ValueError(arm)

def calibration_years(year,months):
    """Last TWO fully mature calendar-entry cohorts, selected without outcomes."""
    return [year-months//12-2,year-months//12-1]

def outputs(run,stage):
    gate=json.loads((run/f'audit/{stage}_gate.json').read_text()); assert gate['status']=='PASS'
    return [run/'results'/stage/str(gate['job'])/f'task-{i}/output' for i in range(gate['tasks'])]
