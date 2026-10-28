"""No fits: pair-stream, environment and next-stage input contract verification."""
import os,sys,json,shutil,subprocess,platform,importlib.metadata
from pathlib import Path
import numpy as np
from support import arrays,write,sha,register,BASE

def main(run):
 inp=run/'input';sys.path.insert(0,str(inp/'frozen_code'))
 from crforest import CompetingRiskForest
 from crforest.core import cluster_bootstrap
 from crforest.partition import tree_feature_subset
 from dataclasses import asdict
 a=arrays(inp/'L0_ORIGINAL/final_train_PRIVATE.npz');b=arrays(inp/'L1_ADJUDICATED/final_train_PRIVATE.npz');index=json.loads((run/'audit/models_index.json').read_text());pairs=[]
 for model in ['q2','unrestricted']:
  for seed in [42,101,202]:
   f0=CompetingRiskForest.load(Path(index[f'L0/{model}/{seed}'])/'forest_PRIVATE.json');f1=CompetingRiskForest.load(Path(index[f'L1/{model}/{seed}'])/'forest_PRIVATE.json')
   assert asdict(f0.config)==asdict(f1.config);np.testing.assert_array_equal(f0.grid_,f1.grid_)
   if f0.shared_cuts_ is not None:
    for c,d in zip(f0.shared_cuts_,f1.shared_cuts_):np.testing.assert_array_equal(c,d)
   for tree in range(32):
    np.testing.assert_array_equal(cluster_bootstrap(a['groups'],seed,tree),cluster_bootstrap(b['groups'],seed,tree))
    np.testing.assert_array_equal(tree_feature_subset(a['X'].shape[1],f0.config.tree_feature_count,seed,tree),tree_feature_subset(b['X'].shape[1],f1.config.tree_feature_count,seed,tree))
   pairs.append(dict(model=model,seed=seed,trees=32,config_identical=True,company_draws_identical=True,eligible_features_identical=True,sharedcuts_identical=True,time_grid_identical=True,native_times=len(f0.grid_)))
 write(run/'audit/paired_randomness.json',pairs)
 # Create a separately versioned adapter, never mutate the frozen fit input plan.
 handoff=run/'handoff_L1';handoff.mkdir(exist_ok=False)
 for role in ['train','eval']:
  for suffix in ['PRIVATE.npz','metadata_PRIVATE.parquet']:
   name=f'final_{role}_{suffix}';shutil.copy2(inp/'L1_ADJUDICATED'/name,handoff/name)
 original_plan=BASE/'input/plan.json';original=json.loads(original_plan.read_text())
 plan=dict(status='PROVISIONAL_POST_EXPOSURE_EXPLORATORY',label_revision='L1_ADJUDICATED',label_revision_manifest=str(inp/'data_version_manifest.json'),seeds=[42,101,202],forest_configs=original['forest_configs'],original_plan_sha256=sha(original_plan),source_counts={'final_train':{'records':56458,'groups':2577,'target_events':174,'competing_events':34,'target_rows':642,'competing_rows':129},'final_eval':{'records':23799,'target_events':79}},source_hashes={p.name:sha(p) for p in handoff.iterdir() if p.is_file()},scope='Input adapter only; no new experiment authorization; original baseline protocol not automatically inherited')
 write(handoff/'plan.json',plan)
 feature=Path('/home/oliver/MainDiskData/xai_rsf_transparent_cif_v2_20261028_p3k845e7/input/features.json');register(run,[feature]);shutil.copy2(feature,handoff/'features.json')
 # Eval/reference data are unchanged historical evidence, not matched revised controls.
 shutil.copy2(inp/'M2_PRIVATE.npz',handoff/'M2_PRIVATE.npz');shutil.copytree(inp/'legacy',handoff/'legacy')
 for version in ['L0','L1']:
  for seed in [42,101,202]:shutil.copy2(Path(index[f'{version}/q2/{seed}'])/'predictions_PRIVATE.npz',handoff/f'reference_{version}_q2_{seed}_PRIVATE.npz')
 write(handoff/'reference_mapping.json',{'primary_revision_reference':'reference_L1_q2_{seed}_PRIVATE.npz','historical_q2':'reference_L0_q2_{seed}_PRIVATE.npz','M2':'M2_PRIVATE.npz; frozen full-sector historical reference only','reference_warning':'full-training q2 models, not role-matched transparent_cif_v2 controls','labels':'L1 is selected by adjudication evidence, never by performance','features_unchanged':True})
 # Run next-stage schema contracts in another interpreter with current code, no train entrypoint.
 repo=Path(__file__).resolve().parents[2]
 script="""import json,sys,numpy as np
from pathlib import Path
from experiments.transparent_cif_v2.run import training_contract,feature_contract
p=Path(sys.argv[1]);a,d,h,e=training_contract(p,'2022-01-01');f=feature_contract(p/'features.json',75)
assert [x['name'] for x in f]==json.loads(Path(sys.argv[2]).read_text())['ordered_names']
print(json.dumps(dict(training_contract='PASS',feature_contract='PASS',records=len(d),features=len(f),fit_invoked=False,eligible_endpoints=int(e.sum()),horizon_days=np.unique(h).tolist())))
"""
 env=dict(os.environ,PYTHONPATH=str(repo));result=subprocess.check_output([sys.executable,'-c',script,str(handoff),str(inp/'input_fields.json')],cwd=repo,env=env,text=True)
 contract=json.loads(result);write(run/'audit/handoff_contract.json',contract)
 manifest=dict(label_version='L1_ADJUDICATED',source_input=str(handoff),data_manifest=str(inp/'data_version_manifest.json'),files={p.name:sha(p) for p in handoff.iterdir() if p.is_file()},source_features_sha256=sha(feature),training_contract=contract,transparent_training_executed=False)
 write(handoff/'handoff_manifest.json',manifest)
 env=dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','pyarrow','pytest']});write(run/'environment/oliver.json',env)
 print(json.dumps(contract))
if __name__=='__main__':main(Path(sys.argv[1]))
