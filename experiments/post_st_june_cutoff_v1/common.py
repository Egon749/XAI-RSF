from pathlib import Path
import numpy as np
import pandas as pd
from experiments.post_st_drop9_v1.common import (
    SEEDS,TIMES,DROP_NAMES,REFERENCE_SHA,kept_columns,reduced,
    configuration,pairing,array_hash,arrays,sha,write,horizon_predictions)
from experiments.post_st_next_open_info_v1.prepare import label_record,packed

NAME='post_st_june_cutoff_v1'
YEARS=[2023,2024,2025]
ARMS=['reference83','drop9']
BASE=Path('/home/oliver/MainDiskData/xai_rsf_2025_shallow_parent_20261029_pSED0uC5')
OBSERVED_END=pd.Timestamp('2026-06-30')
ENTRY_COVERAGE_END=pd.Timestamp('2025-12-31')


def masks(origins,cutoff):
    cutoff=pd.Timestamp(cutoff)
    return origins.le(cutoff), origins.gt(cutoff)&origins.le(cutoff+pd.DateOffset(months=12))


def relabel(meta,events,reviews,cap,endpoint_months=None):
    rows=[]
    for r in meta.to_dict('records'):
        limit=pd.Timestamp(cap)
        if endpoint_months is not None:limit=min(limit,pd.Timestamp(r['origin'])+pd.DateOffset(months=endpoint_months))
        rows.append({**r,**label_record(r,events[r['group']],reviews,limit)})
    return pd.DataFrame(rows,index=meta.index)


def evaluation_status(n,cohort_complete):
    if not cohort_complete:return 'BLOCKED_INCOMPLETE_ENTRY_COHORT'
    if n<2:return 'INSUFFICIENT_EVALUATION_SAMPLE'
    return 'SUPPORTED'
