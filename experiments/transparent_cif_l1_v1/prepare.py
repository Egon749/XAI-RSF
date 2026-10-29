"""Read-only L1 contract, explicit reference mapping and frozen inference protocol."""
from pathlib import Path
import sys,json,shutil,subprocess
import numpy as np,pandas as pd
from crforest.transparent import file_sha256 as sha,write_json,group_roles
from experiments.transparent_cif_v2.run import training_contract,feature_contract
SOURCE=Path('/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1')
PARENT=SOURCE.parent

def main(run):
 inp=run/'input';ledger={}
 def record(p):ledger[str(p)]=sha(p)
 hm=json.loads((SOURCE/'handoff_manifest.json').read_text())
 for name,h in hm['files'].items():assert sha(SOURCE/name)==h;record(SOURCE/name)
 for p in SOURCE.rglob('*'):
  if p.is_file() and '__pycache__' not in str(p):record(p)
 record(PARENT/'input/data_version_manifest.json');record(PARENT/'reports/execution_receipt.json')
 a,d,h,eligible=training_contract(SOURCE,'2022-01-01');fields=feature_contract(SOURCE/'features.json',75)
 assert (len(d),d.group.nunique(),a['X'].shape[1],d.loc[d.event==1,'event_id'].nunique(),d.loc[d.event==2,'event_id'].nunique(),int((d.event==1).sum()),int((d.event==2).sum()))==(56458,2577,75,174,34,642,129)
 # Evaluation is verified by frozen container hashes here; outcomes opened only after all fits freeze.
 for name in ['final_eval_PRIVATE.npz','final_eval_metadata_PRIVATE.parquet']:
  assert sha(SOURCE/name)==sha(PARENT/'input/L0_ORIGINAL'/name)
 original=np.load(PARENT/'input/L0_ORIGINAL/final_train_PRIVATE.npz');record(PARENT/'input/L0_ORIGINAL/final_train_PRIVATE.npz')
 for name in ['X','time','groups','sample_ids','horizon']:np.testing.assert_array_equal(a[name],original[name])
 assert [f['name'] for f in fields]==json.loads((PARENT/'input/input_fields.json').read_text())['ordered_names']
 st,wt=group_roles(a['groups'],.2,20261028);assert not set(a['groups'][st])&set(a['groups'][wt]);roles=[]
 for name,ix in [('all',np.arange(len(d))),('structure',st),('weight_holdout',wt)]:
  dd=d.iloc[ix];roles.append(dict(role=name,records=len(dd),groups=dd.group.nunique(),target_rows=int((dd.event==1).sum()),competing_rows=int((dd.event==2).sum()),target_events=dd.loc[dd.event==1,'event_id'].nunique(),competing_events=dd.loc[dd.event==2,'event_id'].nunique(),endpoint_eligible=int(eligible[ix].sum())))
 pd.DataFrame(roles).to_csv(run/'reports/role_support.csv',index=False)
 index=json.loads((PARENT/'audit/models_index.json').read_text());refs=[]
 for model in ['q2','unrestricted']:
  for seed in [42,101,202]:
   path=SOURCE/f'reference_L1_q2_{seed}_PRIVATE.npz' if model=='q2' else Path(index[f'L1/{model}/{seed}'])/'predictions_PRIVATE.npz'
   record(path);refs.append(dict(model=f'L1_{model}_fulltrain_reference',seed=seed,path=str(path),sha256=sha(path),role='full-training practical reference, not role-matched mechanism control'))
 write_json(inp/'reference_manifest.json',refs)
 config=json.loads(Path('experiments/transparent_cif_v2/config.json').read_text());assert config['pair']==dict(max_pairs=64,max_trees=16,max_cuts=3,min_score=0.,seed=20261028)
 shutil.copy2('experiments/transparent_cif_v2/config.json',inp/'frozen_config.json')
 comparisons={'base_weight_effect':['base_weighted','base_equal'],'ordinary_weight_effect':['ordinary_extra_weighted','ordinary_extra_equal'],'pair_weight_effect':['pair_weighted','pair_equal'],'pair_equal_effect':['pair_equal','ordinary_extra_equal'],'PRIMARY_pair_weighted_effect':['pair_weighted','ordinary_extra_weighted'],'total_effect':['pair_weighted','base_equal'],'practical_vs_fulltrain_q2':['pair_weighted','L1_q2_fulltrain_reference']}
 write_json(inp/'inference_protocol.json',dict(contrasts=comparisons,primary_mechanism='PRIMARY_pair_weighted_effect',bootstrap_replicates=1000,seed=20261025,lists='fixed',nuisance='fixed',primary_aggregation='three-seed probability average',primary_weighting='record_equal',multiplicity='All percentile 95% intervals unadjusted exploratory; no selective significance claims; all prespecified contrasts/years/endpoints retained',primary_endpoints=['AP','IPCW_Brier','Top20_company_hit_fraction','Top20_event_coverage','Top20_lead3m','Top20_lead6m'],fit_count=3,weight_solves=9,ebm_candidates=18,no_pilots=True))
 write_json(run/'audit/source_hashes_before.json',ledger);write_json(run/'audit/data_contract.json',dict(source=str(SOURCE),status='PROVISIONAL_POST_EXPOSURE_EXPLORATORY',features=75,group_overlap=0,eval_outcomes_opened=False,eval_hashes_verified=True,training_counts_verified=True,patch_reapplied=False,cutoff='2022-01-01',horizon_days=np.unique(h).tolist(),unknown_preserved=True))
 print(pd.DataFrame(roles).to_string(index=False))
if __name__=='__main__':main(Path(sys.argv[1]))
