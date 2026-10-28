import os,sys,json
from pathlib import Path
import numpy as np,pandas as pd,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import apply_patch
from support import arrays,sha,PATCH
R=Path(os.environ['REVISION_RUN']);I=R/'input'
def test_only_authorized_cause_rows():
 a=arrays(I/'L0_ORIGINAL/final_train_PRIVATE.npz');b=arrays(I/'L1_ADJUDICATED/final_train_PRIVATE.npz');ix=a['event']!=b['event'];assert ix.sum()==12;assert (a['event'][ix]==1).all() and (b['event'][ix]==2).all()
 for k in a:
  if k!='event':np.testing.assert_array_equal(a[k],b[k])
def test_eval_byte_identical():
 for name in ['final_eval_PRIVATE.npz','final_eval_metadata_PRIVATE.parquet']:assert sha(I/'L0_ORIGINAL'/name)==sha(I/'L1_ADJUDICATED'/name)
def test_patch_idempotence_guard():
 d=pd.read_parquet(I/'L1_ADJUDICATED/final_train_metadata_PRIVATE.parquet');z=arrays(I/'L1_ADJUDICATED/final_train_PRIVATE.npz');p=pd.read_parquet(PATCH/'private/label_patch_PRIVATE.parquet')
 with pytest.raises(ValueError,match='already applied'):apply_patch(d,z,p,'same','same')
def test_hash_guard():
 d=pd.read_parquet(I/'L0_ORIGINAL/final_train_metadata_PRIVATE.parquet');z=arrays(I/'L0_ORIGINAL/final_train_PRIVATE.npz');p=pd.read_parquet(PATCH/'private/label_patch_PRIVATE.parquet')
 with pytest.raises(ValueError,match='hash'):apply_patch(d,z,p,'expected','different')
def test_effective_endpoint_not_legacy_binary_view():
 a=pd.read_parquet(I/'L0_ORIGINAL/final_train_metadata_PRIVATE.parquet');b=pd.read_parquet(I/'L1_ADJUDICATED/final_train_metadata_PRIVATE.parquet');ix=a.event!=b.event
 np.testing.assert_array_equal(a.label_eligible|(a.event==2),b.label_eligible|(b.event==2));assert b.loc[ix,'label_observed_first'].isna().all();assert not b.loc[ix,'label_eligible'].any()
def test_source_uncertainty_not_fixed():
 a=pd.read_parquet(I/'L0_ORIGINAL/final_train_metadata_PRIVATE.parquet');b=pd.read_parquet(I/'L1_ADJUDICATED/final_train_metadata_PRIVATE.parquet')
 for col in ['first_history_complete','origin_normal_observed','stop_date','event_date','outcome_available','duration','event_bin','complete_months']:pd.testing.assert_series_equal(a[col],b[col])
def test_monthly_mask_change_only_cause():
 a=arrays(I/'L0_ORIGINAL/cause_masks_PRIVATE.npz');b=arrays(I/'L1_ADJUDICATED/cause_masks_PRIVATE.npz');np.testing.assert_array_equal(a['any_event'],b['any_event']);np.testing.assert_array_equal(a['endpoint_known'],b['endpoint_known']);assert np.sum(a['monthly_labels']!=b['monthly_labels'])==12
