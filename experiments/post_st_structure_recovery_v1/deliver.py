"""Publish synthetic aggregate evidence; never copy fitted parameters or records."""
from pathlib import Path
import argparse,json,shutil
import pandas as pd
from .common import *

def deliver(run,repo):
    dest=repo/'experiments/post_st_structure_recovery_v1';results=dest/'results'
    validation=json.loads((run/'reports/independent_validation.json').read_text());assert validation['status']=='PASS'
    assert not results.exists();results.mkdir()
    for p in (run/'reports').iterdir():
        if p.is_file():shutil.copy2(p,results/p.name)
    for name in ['pre_fit_freeze.json','pilot_gate.json','formal_gate.json','runtime_hosts.json','analysis_environment.json','analysis_addendum.json','analysis_code_before_effect_inspection.json','first_effect_inspection.json','prefit_tests.log','final_tests.log']:
        shutil.copy2(run/'audit'/name,results/name)
    m=pd.read_csv(results/'metric_summary.csv',dtype={'seed':str});m=m[(m.seed=='ensemble')&(m.variant=='shrunk')&(m.day==730)&(m.cause==1)]
    data=pd.read_csv(results/'data_identity.csv');rows=[]
    fields=['true_pair_amplitude_ratio','true_pair_function_l2_error','true_pair_detected','true_pair_positive_detected','pair_precision','pair_recall','all_fp','main_fp','pair_fp','higher_fp','joint_risk_mse','exact_product_risk_mse','expected_brier','mean_prediction','true_mean','spatial_cells_including_missing']
    for scenario in SCENARIOS:
        d=data[data.scenario==scenario]
        for arm in ARMS:
            g=m[(m.scenario==scenario)&(m.arm==arm)].set_index('metric')
            row=dict(scenario=scenario,arm=arm,training_n=SCENARIOS[scenario]['n'],repetitions=len(d),mean_target_events=d.target_events.mean(),min_target_events=d.target_events.min(),max_target_events=d.target_events.max())
            for metric in fields:
                r=g.loc[metric];row[metric]=r['mean'];row[metric+'_low95']=r.low95;row[metric+'_high95']=r.high95
                if metric in ['true_pair_detected','true_pair_positive_detected','pair_recall']:
                    row[metric+'_wilson_low95']=r.rate_wilson_low95;row[metric+'_wilson_high95']=r.rate_wilson_high95
            rows.append(row)
    pd.DataFrame(rows).to_csv(results/'main_comparison.csv',index=False)
    tasks=pd.read_csv(results/'task_manifest.csv');manifest=[]
    for row in tasks.itertuples():
        folder=Path(row.output);hashes=json.loads((folder/'hashes.json').read_text())
        for name,digest in hashes.items():
            if name.endswith(('forest.json.gz','population_functions.npz')):
                child=Path(name).parts[0];arm,seed,variant=child.rsplit('_',2)
                manifest.append(dict(scenario=row.scenario,rep=row.rep,arm=arm,seed=seed,variant=variant,artifact=Path(name).name,
                    sha256=digest,bytes=(folder/name).stat().st_size,private_relative_path=f'task-{row.task}/output/{name}'))
    artifacts=pd.DataFrame(manifest);assert (artifacts.artifact=='forest.json.gz').sum()==5400
    assert (artifacts.artifact=='population_functions.npz').sum()==7200
    artifacts.to_csv(results/'model_artifact_hashes.csv.gz',index=False,compression='gzip')
    receipt=json.loads((run/'reports/execution_summary.json').read_text())
    receipt.update(validation=validation,protocol_sha256=sha(dest/'frozen_protocol.json'),
        numerical_tests_before_fits='168 passed; independent R checks enabled; one existing EBM warning',
        unit_development_corrections=['Signal-row expectation corrected12->8 before freeze; no tolerance changed','Wilson interval endpoint made analytically exact at0/1; descriptive analysis only'],
        public_data='Synthetic aggregate results and artifact hashes only; no enterprise inputs, outcomes, parameters or predictions',
        analysis_files_sha256={name:sha(dest/name) for name in ['analyze.py','plots.py','validate.py','attenuation_audit.py','report.py','deliver.py']},
        reporting_addendum='Uniform Wilson intervals supplement unchanged frozen t intervals; recorded before formal effect estimates were read',
        reference_model_sha256=sha(repo/'CURRENT_POST_ST_MODEL.json'))
    write(dest/'execution_receipt.json',receipt)
    write(results/'public_artifact_hashes.json',{p.name:sha(p) for p in results.iterdir() if p.is_file()})
    print(json.dumps(dict(public_files=len(list(results.iterdir())),public_bytes=sum(p.stat().st_size for p in results.iterdir()),native_model_hashes=5400,function_export_hashes=7200)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();deliver(a.run,a.repo)
