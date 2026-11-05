"""Finite entry-window ablation; same frozen company records and clocks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from crforest import ForestConfig
from experiments.post_st_prognosis_v1.common import arrays, sha, write
from experiments.post_st_drop9_support_v1.common import configuration

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
STARTS = tuple(range(2012, 2021))
YEARS = (2023, 2024, 2025)
SEEDS = (17, 503, 22)
TIMES = np.array([365., 366., 730., 731.])

def load(p):
    return json.loads(Path(p).read_text())

def run():
    return Path(load(D / 'run_location.json')['private_run'])

def source():
    return Path(load(D.parent / 'post_st_seed_explanation_stability_v1/run_location.json')['private_run'])

def filter_history(data, meta, start, cutoff):
    """Slice rows only: never mature, reconstruct or extend their outcomes."""
    if not meta.sample_id.is_unique or len(set(data['groups'])) != len(meta):
        raise ValueError('Expected one row per company')
    np.testing.assert_array_equal(meta.sample_id, data['sample_ids'])
    origin = pd.to_datetime(meta.origin)
    stop = pd.to_datetime(meta.stop)
    if not ((origin < pd.Timestamp(cutoff, 1, 1)).all() and
            (stop < pd.Timestamp(cutoff, 1, 1)).all()):
        raise ValueError('Own historical snapshot exceeds cutoff')
    np.testing.assert_array_equal(meta.entry_year, origin.dt.year)
    if not ((stop >= origin).all() and
            (stop <= origin.map(lambda t: t + pd.DateOffset(months=24))).all()):
        raise ValueError('Invalid original followup cap')
    mask = (origin.dt.year >= start).to_numpy()
    # Assert the snapshot schema instead of accidentally slicing column metadata.
    required = {'X', 'time', 'event', 'groups', 'sample_ids'}
    allowed = required | {'year', 'h12', 'h24', 'label_eligible', 'chain_gap_present', 'analysis_time'}
    if not required <= set(data) <= allowed or any(v.shape[0] != len(meta) for v in data.values()):
        raise ValueError(f'Unexpected training keys {set(data)}')
    return {k: v[mask].copy() for k, v in data.items()}, mask

def endpoint(cif, days):
    ix = np.searchsorted(TIMES, days)
    np.testing.assert_array_equal(TIMES[ix], days)
    return cif[np.arange(len(days)), ix]

def supported(year, horizon):
    return not (year == 2025 and horizon == 24)
