"""Apply only the certified patch to a new copy; leave every source read-only."""
from pathlib import Path
import sys,json,shutil,subprocess
import numpy as np,pandas as pd
from support import *

def apply_patch(d,z,p,expected_hash,actual_hash):
 if actual_hash!=expected_hash:raise ValueError('Source hash mismatch')
 if p.sample_id.duplicated().any():raise ValueError('Duplicate patch row')
 ix=d.index[d.sample_id.isin(p.sample_id)].to_numpy()
 if len(ix)!=12 or p.event_id.nunique()!=3 or p.group.nunique()!=3:raise ValueError('Unexpected patch extent')
 if not d.iloc[ix].event.eq(1).all() or not (z['event'][ix]==1).all():raise ValueError('Patch already applied or old labels differ')
 if set(d[d.event_id.isin(p.event_id)].sample_id)!=set(p.sample_id):raise ValueError('Incomplete event windows')
 changed=d.copy(deep=True);out={k:v.copy() for k,v in z.items()};out['event'][ix]=2
 changed.loc[ix,'event']=2;changed.loc[ix,'status']='non_target_competing_event';changed.loc[ix,'previous_label']='non_target_first_event'
 changed.loc[ix,'label_observed_first']=pd.NA;changed.loc[ix,'label_eligible']=False;changed.loc[ix,'three_view_eligible']=False;changed.loc[ix,'followup_reason']='observed_non_target_first_ST'
 return changed,out,ix

def counts(d):return dict(records=len(d),groups=d.group.nunique(),target_events=d.loc[d.event.eq(1),'event_id'].nunique(),competing_events=d.loc[d.event.eq(2),'event_id'].nunique(),target_rows=int(d.event.eq(1).sum()),competing_rows=int(d.event.eq(2).sum()))
def main(run):
 run=Path(run);inp=run/'input';assert not (inp/'data_version_manifest.json').exists(),'Never reapply to a completed version'
 refs=[PATCH/'private/label_patch_PRIVATE.parquet',PATCH/'private/label_patch_manifest.json',PATCH/'private/label_adjudication_PRIVATE.parquet',PATCH/'audit/label_patch_validation.json',ORIGINAL/'reports/execution_receipt.json',ORIGINAL/'input/plan.json',ORIGINAL/'audit/immutable_input_hashes.json',BASE/'input/plan.json']
 refs+=list((BASE/'input/legacy').glob('*.py'))
 source_definition=Path('/home/oliver/LLMworkspace/early-warning-12m-server-isolated/experiments/early_warning_12m/source_reference/prepare_cohort.py')
 refs += [source_definition,Path('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1/input/audit_first_followup_v3.py'),Path('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1/sample_2012plus_v1/summary.json')]
 for role in ['train','eval']:
  for suffix in ['PRIVATE.npz','metadata_PRIVATE.parquet']:refs += [BASE/f'input/final_{role}_{suffix}',ORIGINAL/f'input/final_{role}_{suffix}']
 refs += [BASE/'input/M2_PRIVATE.npz',BASE/'input/code/experiments/shared_partition_v1/input_fields.json']
 refs += [p for p in (ORIGINAL/'input/code').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
 models=[]
 for p in sorted((ORIGINAL/'results/assembled').glob('*/task-*/output/summary.json')):
  s=json.loads(p.read_text());key='q2' if s['variant']=='coverage' else s['variant'];folder=p.parent
  refs+=list(folder.glob('*'));f=json.loads((folder/'forest_PRIVATE.json').read_text())
  models.append(dict(model=key,variant=s['variant'],seed=s['seed'],source_folder=str(folder),model_sha256=sha(folder/'forest_PRIVATE.json'),prediction_sha256=sha(folder/'predictions_PRIVATE.npz'),training_code_sha=TRAIN_SHA))
 register(run,[p for p in refs if p.is_file()]);assert len(models)==6
 receipt=json.loads((ORIGINAL/'reports/execution_receipt.json').read_text());assert receipt['training_code']==TRAIN_SHA
 frozen=json.loads((ORIGINAL/'input/plan.json').read_text());assert sha(ORIGINAL/'input/plan.json')==receipt['plan_sha256']
 for name,hh in frozen['source_hashes'].items():assert sha(ORIGINAL/'input'/name)==hh,name
 for m in models:
  key=f"{m['variant']}/{m['seed']}";assert m['model_sha256']==receipt['model_hashes'][key]
  actual=json.loads((Path(m['source_folder'])/'forest_PRIVATE.json').read_text());assert actual['config']==frozen['configs'][f"final/{m['seed']}/{m['variant']}"]
 shutil.copytree(ORIGINAL/'input/code',inp/'frozen_code',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
 for rel in ['crforest/core.py','crforest/functions.py','crforest/partition.py','experiments/shared_partition_v1/engine.py']:
  gitbytes=subprocess.check_output(['git','show',TRAIN_SHA+':'+rel]);assert gitbytes==(inp/'frozen_code'/rel).read_bytes()
 shutil.copytree(BASE/'input/legacy',inp/'legacy',ignore=shutil.ignore_patterns('__pycache__'))
 # Additional legacy rank helper for the already-tested conditional bootstrap.
 rank=Path('/home/oliver/LLMworkspace/early-warning-12m-server-isolated/experiments/early_warning_12m/bootstrap.py');register(run,[rank]);shutil.copy2(rank,inp/'legacy/bootstrap.py')
 for label in ['L0_ORIGINAL','L1_ADJUDICATED']:(inp/label).mkdir()
 for role in ['train','eval']:
  for suffix in ['PRIVATE.npz','metadata_PRIVATE.parquet']:
   name=f'final_{role}_{suffix}';assert sha(BASE/'input'/name)==sha(ORIGINAL/'input'/name)
   shutil.copy2(BASE/'input'/name,inp/'L0_ORIGINAL'/name)
   if role=='eval':shutil.copy2(BASE/'input'/name,inp/'L1_ADJUDICATED'/name)
 pm=json.loads((PATCH/'private/label_patch_manifest.json').read_text());p=pd.read_parquet(PATCH/'private/label_patch_PRIVATE.parquet');assert sha(PATCH/'private/label_patch_PRIVATE.parquet')==pm['patch_sha256']
 d=pd.read_parquet(inp/'L0_ORIGINAL/final_train_metadata_PRIVATE.parquet');z=arrays(inp/'L0_ORIGINAL/final_train_PRIVATE.npz');assert sha(inp/'L0_ORIGINAL/final_train_PRIVATE.npz')==pm['array_sha256'];np.testing.assert_array_equal(z['sample_ids'],d.sample_id)
 # Use the published validator as an additional independent gate, without its write entrypoint.
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'label_audit_clock_v1'));from validate_label_patch import validate
 validate(p,d,sha(inp/'L0_ORIGINAL/final_train_metadata_PRIVATE.parquet'),pm['metadata_sha256'])
 l1,z1,ix=apply_patch(d,z,p,pm['metadata_sha256'],sha(inp/'L0_ORIGINAL/final_train_metadata_PRIVATE.parquet'))
 allowed=['event','status','previous_label','label_observed_first','label_eligible','three_view_eligible','followup_reason'];changes=[]
 for col in d:
  eq=d[col].eq(l1[col]).fillna(False)|(d[col].isna()&l1[col].isna());changed=np.flatnonzero(~eq.to_numpy(bool))
  if col not in allowed:assert len(changed)==0,col
  assert set(changed)<=set(ix)
  changes.append(dict(field=col,changed_rows=len(changed),action='CAUSE_DERIVED_ONLY' if col in allowed else 'PRESERVED',semantics='legacy target/survivor view; competing known through event==2' if col in ['label_eligible','three_view_eligible','label_observed_first'] else 'source-defined status/reason or immutable original field'))
 for k in z:
  if k!='event':np.testing.assert_array_equal(z[k],z1[k])
 np.testing.assert_array_equal(z['event']>0,z1['event']>0)
 sys.path.insert(0,str(inp/'legacy'));from common import endpoint_weights,monthly_labels
 from endpoint_guard import source_consistent_endpoint_weights
 w0,a0=source_consistent_endpoint_weights(d,endpoint_weights);w1,a1=source_consistent_endpoint_weights(l1,endpoint_weights);np.testing.assert_array_equal(w0,w1)
 np.testing.assert_array_equal(d.label_eligible|(d.event==2),l1.label_eligible|(l1.event==2))
 y0=monthly_labels(d);y1=monthly_labels(l1);assert np.sum(y0!=y1)==12 and (y0[y0!=y1]==1).all() and (y1[y0!=y1]==2).all()
 np.savez_compressed(inp/'L1_ADJUDICATED/final_train_PRIVATE.npz',**z1);l1.to_parquet(inp/'L1_ADJUDICATED/final_train_metadata_PRIVATE.parquet',index=False)
 for label,dd,zz,yy in [('L0_ORIGINAL',d,z,y0),('L1_ADJUDICATED',l1,z1,y1)]:
  np.savez_compressed(inp/label/'cause_masks_PRIVATE.npz',target=zz['event']==1,competing=zz['event']==2,any_event=zz['event']>0,endpoint_known=dd.label_eligible.to_numpy(bool)|(zz['event']==2),monthly_labels=yy)
 pd.DataFrame([dict(version=k,**counts(dd)) for k,dd in [('L0_ORIGINAL',d),('L1_ADJUDICATED',l1)]]).to_csv(run/'reports/label_counts_before_after.csv',index=False)
 assert list(counts(l1).values())==[56458,2577,174,34,642,129]
 write(inp/'label_dependency_map.json',{'fields':changes,'cause_masks':'derived afresh from event; no stale cache reused','legacy_views':'label_observed_first becomes NA, label_eligible/three_view_eligible false for competing, following existing schema; NOT normal. effective endpoint-known union and IPCW weights unchanged.','previous_label':'current upstream cause classification rebuilt; original values preserved in private field_change ledger and L0.','first_history_and_origin_uncertainty':'unchanged; no risk-set reentry','source_definition_sha256':sha(source_definition)})
 change=d.iloc[ix][['sample_id','event_id']].copy()
 for c in allowed:change['old_'+c]=d.iloc[ix][c].to_numpy();change['new_'+c]=l1.iloc[ix][c].to_numpy()
 change.to_parquet(run/'private/field_changes_PRIVATE.parquet',index=False)
 files={str(f.relative_to(inp)):{'sha256':sha(f)} for label in ['L0_ORIGINAL','L1_ADJUDICATED'] for f in (inp/label).iterdir()}
 sem={label:{k:semantic(v) for k,v in zz.items()} for label,zz in [('L0_ORIGINAL',z),('L1_ADJUDICATED',z1)]}
 write(inp/'patch_application_receipt.json',dict(applied_to_new_copy=True,original_modified=False,events=3,groups=3,records=12,changed_fields=changes,patch_sha256=pm['patch_sha256'],any_event_state_unchanged=True,ipcw_weights_unchanged=True,effective_endpoint_eligibility_unchanged=True,monthly_label_changed_cells=12,repeat_apply='ERROR',unknown_or_early_censored_qualification_unchanged=True))
 shutil.copy2(BASE/'input/M2_PRIVATE.npz',inp/'M2_PRIVATE.npz');shutil.copy2(BASE/'input/code/experiments/shared_partition_v1/input_fields.json',inp/'input_fields.json')
 write(inp/'model_provenance_PRIVATE.json',models)
 write(inp/'plan.json',dict(configs=frozen['configs'],seeds=[42,101,202],source_counts=frozen['source_counts'],training_code=TRAIN_SHA,formal_L1_fits=6,control_policy='reuse only after reload match',tree_shard_size=4,bootstrap_replicates=1000,bootstrap_seed=20261025,primary='three-seed probability ensemble',reference='all original training X with equal row weights, same for L0/L1'))
 write(inp/'data_version_manifest.json',dict(data_status=['PROVISIONAL','POST_EXPOSURE_EXPLORATORY'],versions=['L0_ORIGINAL','L1_ADJUDICATED'],source_input=str(BASE/'input'),patch_source=str(PATCH/'private/label_patch_PRIVATE.parquet'),patch_sha256=pm['patch_sha256'],files=files,array_semantic_hashes=sem,feature_order_sha256=sha(inp/'input_fields.json'),X_and_missing_positions_identical=True,all_eval_files_byte_identical=True,no_new_features=True,unresolved_events_unchanged=47,default_future_working_version='L1_ADJUDICATED',selection_basis='adjudicated evidence, not model performance'))
 print('L1 APPLIED',counts(l1),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
