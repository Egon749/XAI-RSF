"""Definitions for the finite, read-only triple audit."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from experiments.post_st_requested_seeds10_v1.common import SEEDS, YEARS, TIMES, arrays, sha, run_path as source_path

HERE = Path(__file__).resolve().parent
NAME = HERE.name
ROOT = HERE.parents[1]
BASE = ROOT / 'experiments/post_st_requested_seeds10_v1'
TRIPLES = list(combinations(range(len(SEEDS)), 3))
NAMES = [f'C{i+1:03d}' for i in range(len(TRIPLES))]
VERSIONS = ('raw', 'shrunk')
MATERIAL = .001
def read(path):
    return json.loads(Path(path).read_text())
def dump(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
def run_path():
    return Path(read(HERE/'run_location.json')['private_run'])
def distributions(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict.fromkeys(('min','p05','median','p95','max','mean','sd'), np.nan) | {'valid':0}
    return dict(zip(('min','p05','median','p95','max'),np.quantile(x,[0,.05,.5,.95,1]))) | dict(mean=x.mean(),sd=x.std(ddof=1) if len(x)>1 else np.nan,valid=len(x))
