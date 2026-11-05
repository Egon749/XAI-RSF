"""Frozen finite direct-parent shrinkage experiment; no split changes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiments.post_st_training_start_v1.common import endpoint, supported, filter_history
from experiments.post_st_drop9_support_v1.common import configuration
from experiments.post_st_prognosis_v1.common import arrays, sha, write, calendar_elapsed

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
STRENGTHS = tuple(range(21))
REUSED = (0, 1, 2, 5, 10, 20)
NEW = tuple(s for s in STRENGTHS if s not in REUSED)
YEARS = (2023, 2024, 2025)
HISTORICAL = (2020, 2021, 2022)
ALL_YEARS = HISTORICAL + YEARS
SEEDS = (17, 503, 22)
TIMES = np.array([365., 366., 730., 731.])

def load(path):
    return json.loads(Path(path).read_text())

def run():
    return Path(load(D / 'run_location.json')['private_run'])

def source():
    return Path(load(D.parent / 'post_st_seed_explanation_stability_v1/run_location.json')['private_run'])

def select_strength(rows):
    """No outer outcomes. Max 12m within-year AUC; deterministic tie breaks."""
    candidates = [r for r in rows if np.isfinite(r['within_year_auc'])]
    if not candidates:
        return 10, 'NO_ESTIMABLE_HISTORICAL_AUC_RETURN_REFERENCE'
    def key(r):
        return (-r['within_year_auc'], -r['ap'] if np.isfinite(r['ap']) else float('inf'),
                r['ipcw_brier'], abs(r['strength']-10), r['strength'])
    return int(min(candidates, key=key)['strength']), 'EXPLORATORY_FORWARD_SELECTION'
