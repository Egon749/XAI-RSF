"""One prespecified deletion set, one outer year; no configuration search."""
from pathlib import Path
from experiments.post_st_ablation_v1.common import configuration, pairing, array_hash
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizon_predictions
import numpy as np

NAME='post_st_drop9_v1'
SEEDS=[42,101,202]
TIMES=np.array([365.,366.,730.,731.])
HISTORY=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj')
REFERENCE_SHA='aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65'
DROP_NAMES=['net_margin','operating_profit_to_assets','net_profit_to_assets','cfo_to_current_liabilities',
    'working_capital_to_assets','profit_total_growth','total_mktcap_change_3m','downside_vol_3m','total_mktcap_change_12m']


def kept_columns(names):
    if len(names)!=83 or len(set(names))!=83 or not set(DROP_NAMES)<=set(names):
        raise ValueError('Expected frozen83-column schema and the fixed nine fields')
    return [j for j,n in enumerate(names) if n not in DROP_NAMES]


def reduced(data,keep):
    result={k:v.copy() for k,v in data.items()}
    result['X']=data['X'][:,keep].copy()
    return result
