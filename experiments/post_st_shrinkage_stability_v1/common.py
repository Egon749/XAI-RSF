"""Fixed paired company perturbations; original forest engine is unchanged."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays, sha, write
from experiments.post_st_drop9_support_v1.common import configuration
from experiments.post_st_training_start_v1.common import endpoint, supported, filter_history
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
YEARS=(2023,2024,2025)
SEEDS=(17,503,22)
STRENGTHS=(0,10,18)
REPEATS=50
TIMES=np.array([365.,366.,730.,731.])
DRAW_SEED=20261105
def load(p):return json.loads(Path(p).read_text())
def run():return Path(load(D/'run_location.json')['private_run'])
def draw_indices(n,year,repeat):
    if not 1<=repeat<=REPEATS:raise ValueError('repeat outside frozen range')
    return np.sort(np.random.default_rng(np.random.SeedSequence([DRAW_SEED,year,repeat,811])).integers(n,size=n))
def resample(data,year,repeat):
    if len(np.unique(data['groups']))!=len(data['groups']):raise ValueError('one original row per company required')
    ix=draw_indices(len(data['groups']),year,repeat)
    # Preserve identity and all multiplicities. Never relabel copies as independent firms.
    return {k:v[ix].copy() for k,v in data.items()},ix
def top_indices(p,ids):
    return np.lexsort((ids,-np.asarray(p)))[:int(np.ceil(.2*len(p)))]
def pair_stats(values):
    """Across-replicate correlation, Jaccard and score spread on fixed companies."""
    from scipy.stats import rankdata
    p=np.asarray(values,float)
    ranks=np.array([rankdata(v,method='average') for v in p])
    good=np.std(ranks,axis=1)>1e-14
    corr=np.corrcoef(ranks[good]) if good.sum()>1 else np.empty((0,0))
    vals=corr[np.triu_indices(len(corr),1)] if len(corr) else np.array([])
    return dict(mean_company_sd=float(p.std(0,ddof=1).mean()),
      median_company_sd=float(np.median(p.std(0,ddof=1))),
      p95_company_sd=float(np.quantile(p.std(0,ddof=1),.95)),
      between_company_sd=float(p.mean(0).std(ddof=1)),
      mean_rank_sd=float(ranks.std(0,ddof=1).mean()),
      median_pair_rank_correlation=float(np.median(vals)) if len(vals) else np.nan,
      undefined_rank_pairs=int(len(p)*(len(p)-1)//2-len(vals)))
