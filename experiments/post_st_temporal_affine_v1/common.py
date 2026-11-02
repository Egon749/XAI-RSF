from pathlib import Path
import json
import numpy as np
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions,calendar_elapsed
from experiments.post_st_ablation_v1.common import configuration
from experiments.post_st_paper_features_v1.common import references,history

NAME='post_st_temporal_affine_v1'
REPO=Path(__file__).resolve().parents[2]
RUN=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/temporal_affine_20261102_run1')
PARENT=Path('/home/oliver/MainDiskData/xai_rsf_dropcolumn_selection_20261101_EzUsFtiO')
CAUSE=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/cause_support_20261102_run1')
OLD=Path('/home/oliver/MainDiskData/xai_rsf_level_coverage_20261031_ckCB3u1l')
YEARS=[2023,2024,2025];SEEDS=[42,101,202];ARMS=['baseline','independent','temporal']
TIMES=np.array([365.,366.,730.,731.]);ANALYSIS_SEED=2026110202;RESAMPLE_SEED=2026110212
REFERENCE_SHA='aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65'

def load(p):return json.loads(Path(p).read_text())
def periods(h):return list(range(2020+h//12,2026))
def cohorts(year,h):return [v for v in [year-h//12-2,year-h//12-1] if v>=2019]

def block_forest_fits():
    from crforest import CompetingRiskForest
    def forbidden(*args,**kwargs):raise RuntimeError('Forest training is forbidden in this calibration-only experiment')
    CompetingRiskForest.fit=forbidden
