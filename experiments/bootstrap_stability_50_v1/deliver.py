"""Publish only a fixed allowlist of aggregate artifacts into the research branch."""
from pathlib import Path
import argparse,json,hashlib,shutil,re
import pandas as pd
from PIL import Image


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(run):
    target=Path(__file__).parent;out=run/'reports';r=json.loads((out/'execution_receipt.json').read_text());assert r['status']=='PASS' and r['new_tasks']==84 and r['paired_replicates']==50
    for name,h in r['results_hashes'].items():assert sha(out/name)==h
    # Analysis scripts may be delivered after the frozen training commit, but
    # the exact scripts used are identified separately by their content hashes.
    for name,h in r['analysis_source_hashes'].items():assert sha(target/name)==h
    names=['README.md','STABILITY_RESULTS_ZH.md','stability_intervals.png','stability_intervals.pdf','formal_gate.json','execution_receipt.json','paired_stability.csv','original_anchor_stability.csv','background_stability.csv','stability_intervals.csv','original_anchor_intervals.csv','background_intervals.csv','native_q2_stability.csv','task_diagnostics.csv','model_hashes.csv']
    for name in names:shutil.copy2(out/name,target/name)
    a=(run/'audit/requirements_oliver.lock').read_text();b=(run/'audit/requirements_lemon.lock').read_text();assert a==b
    (target/'requirements.lock').write_text(a)
    env=json.loads((run/'audit/environment_oliver.json').read_text());env.update(lemon_same_locked_packages=True,oliver_python='/home/oliver/.venvs/xai-baselines-20261027/bin/python',lemon_python='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python',thread_policy='one physical worker per fit; Slurm oliver 8P+16E, lemon 8 physical; BLAS/OMP/MKL=1; NFS; 6GiB virtual-address-space limit')
    (target/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    plan=json.loads((run/'input/plan.json').read_text());freeze=json.loads((run/'audit/freeze.json').read_text())
    frozen=dict(training_code_commit=freeze['code_commit'],historical_parent_code_commit=plan['code_commit'],historical_field_notice='input/plan.json code_commit is inherited parent provenance; freeze.json is execution authority',forest_configs=plan['forest_configs'],bootstrap_replicates=50,bootstrap_models=plan['bootstrap_models'],new_fits=84,reused_fits=16,model_seed=42,company_sampling_seeds=list(range(8000,8050)),fixed_anchors=32,fixed_background=16,horizon_days=365,primary_metric='normalized_rmse',contrast='q2 minus unrestricted',interval='conditional MC delete-one-paired-replicate jackknife t',prefixes=[8,30,50],source_counts=plan['source_counts'],training_input_sha256=freeze['input_hashes']['final_train_PRIVATE.npz'],evaluation_input_sha256=freeze['input_hashes']['final_eval_PRIVATE.npz'],references_sha256=freeze['input_hashes']['references_PRIVATE.npz'],protocol_sha256=sha(run/'input/code/experiments/bootstrap_stability_50_v1/PROTOCOL_ZH.md'),status='EXPLORATORY_PROVISIONAL',stopping='exactly 50 paired fits; no outcome-selected stopping')
    (target/'frozen_protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    shutil.copy2(run/'audit/reuse_compatibility.json',target/'reuse_compatibility.json')
    pairs=pd.read_csv(target/'paired_stability.csv');d=pd.read_csv(target/'task_diagnostics.csv');s=pd.read_csv(target/'stability_intervals.csv')
    assert len(pairs)==1225 and len(d)==100 and len(s)==18 and len(d.query('reused==False'))==84
    assert set(zip(d.model,d.replicate))=={(m,i) for m in ['q2','unrestricted'] for i in range(50)}
    assert (d.affinity_count==1).all() and (d.reload_error<1e-12).all() and (d.shap_additivity_error<1e-10).all()
    assert (s.ci_low<=s.estimate).all() and (s.ci_high>=s.estimate).all()
    im=Image.open(target/'stability_intervals.png');im.verify()
    testlog=(target/'tests_final.txt').read_text();match=re.search(r'(\d+) passed',testlog)
    assert match and int(match.group(1))>=58 and not re.search(r'\d+ (failed|error|skipped)',testlog)
    validation=dict(status='PASS',new_fits=84,reused_fits=16,paired_replicates=50,dependent_comparison_pairs=1225,all_models_identified=True,all_replicates_included=True,same_packages_on_both_hosts=True,all_worker_affinity_one=True,unpublished_private_arrays=True,full_tests=int(match.group(1)),full_test_log_sha256=sha(target/'tests_final.txt'),independent_R_checks=2,code_inputs_and_outputs_hashed=True,overall_research_decision='INCONCLUSIVE',source_results_sha256={name:sha(out/name) for name in names},postprocessing_sha256={p.name:sha(p) for p in target.glob('*.py')})
    (target/'delivery_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if not isinstance(v,dict)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
