"""Frozen candidate-pool grid; reuse the authoritative forest/evaluator."""
from pathlib import Path
from dataclasses import replace,asdict
import json
import numpy as np
import pandas as pd
from experiments.post_st_training_start_v1.common import endpoint,supported
from experiments.post_st_drop9_support_v1.common import configuration as reference_configuration
from experiments.post_st_prognosis_v1.common import arrays,sha,write
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
COUNTS=tuple(range(10,36))
REFERENCE=25
YEARS=(2023,2024,2025)
HISTORICAL=(2020,2021,2022)
ALL_YEARS=HISTORICAL+YEARS
SEEDS=(17,503,22)
TIMES=np.array([365.,366.,730.,731.])
def load(p):return json.loads(Path(p).read_text())
def run():return Path(load(D/'run_location.json')['private_run'])
def source():return Path(load(D.parent/'post_st_seed_explanation_stability_v1/run_location.json')['private_run'])
def configuration(seed,pool_count):
    if pool_count not in COUNTS:raise ValueError('outside frozen count grid')
    return replace(reference_configuration(seed),tree_feature_count=pool_count)
def select_count(rows):
    candidates=[r for r in rows if np.isfinite(r['within_year_auc'])]
    if not candidates:return REFERENCE,'NO_ESTIMABLE_HISTORICAL_AUC_RETURN_REFERENCE'
    def key(r):return (-r['within_year_auc'],-r['ap'] if np.isfinite(r['ap']) else float('inf'),r['ipcw_brier'],abs(r['pool_count']-REFERENCE),r['pool_count'])
    return int(min(candidates,key=key)['pool_count']),'EXPLORATORY_FORWARD_SELECTION'
