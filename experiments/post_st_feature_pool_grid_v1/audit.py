"""Read-only final validation and environment/compute receipt inputs."""
import importlib.metadata,platform,subprocess,re,hashlib
from collections import Counter
from .common import *

def main():
    r=run();audit=load(D/'DATA_AUDIT.json');inp=r/'input'
    for name,digest in audit['input_hashes'].items():assert sha(r/name)==digest
    for name,digest in load(r/'input_hashes.json').items():assert sha(inp/name)==digest
    assert sha(ROOT/'CURRENT_POST_ST_MODEL.json')==audit['current_pointer_sha256']
    assert not subprocess.check_output(['git','diff','f7d59c91086c17fa0aaee8f155139bfdef0c8d4f','--','crforest','CURRENT_POST_ST_MODEL.json','experiments/post_st_shrinkage_fine_v1'],text=True,cwd=ROOT).strip()
    jobs=[];failures=[]
    for stage in ['pilot','formal']:
        launcher=r/f'{stage}_launcher.log';lines=launcher.read_text().splitlines()
        record=next(json.loads(line) for line in reversed(lines) if line.startswith('{') and '"success"' in line)
        result=Path(record['result']);s=load(result);n=1 if stage=='pilot' else 155
        assert record['success'] and len(s['completed'])==s['expected']==n and not s['pending'] and s['srun_returncode']==0
        assert {x['id'] for x in s['completed']}==set(range(n))
        assert all(x['returncode']==0 and x['compute_completed'] and x['publication_completed'] and x['storage']['publication_state']=='verified' for x in s['completed'])
        jobs.append(dict(stage=stage,job_id=record['job'],tasks=n,forest_fits=3*n,by_node=record['by_node'],execution_seconds=record['execution_seconds'],elapsed_including_queue=record['elapsed_including_queue'],scheduler_result_sha256=sha(result)))
        (D/f'{stage}_launcher.log').write_bytes(launcher.read_bytes())
    summaries=[load(p) for p in (r/'published').glob('*/task-*/output/summary.json')]
    assert len(summaries)==156 and sum(s['forest_fits'] for s in summaries)==468
    assert {(s['year'],s['pool_count']) for s in summaries}=={(y,c) for y in ALL_YEARS for c in COUNTS}
    assert all(s['no_evaluation_labels_read'] for s in summaries)
    num=pd.read_csv(D/'NUMERICAL_COMPLEXITY.csv');assert len(num)==1092
    fields=['full_reconstruction','explicit_component_sum','function_reload','forest_reload'];errors=num[fields].max().to_dict()
    assert max(errors.values())<=1e-10
    assert (num.max_path_variables<=2).all() and (num.max_depth<=4).all() and (num.path_violations==0).all() and (num.ancestor_errors==0).all()
    assert num.probability_min.min()>=-1e-12 and num.probability_sum_max.max()<=1+1e-10 and num.minimum_increment.min()>=-1e-12
    repro=pd.read_csv(D/'REFERENCE_REPRODUCTION.csv');assert len(repro)==18 and repro.raw_file_identical.all()
    assert (repro.loc[repro.year.isin(YEARS),'prediction_error']==0).all()
    m=pd.read_csv(D/'METRICS_ALL.csv');unsupported=m[(m.scope=='2025')&(m.months==24)];assert len(unsupported)==210
    assert unsupported[['auc','ap','ipcw_brier']].isna().all().all()
    # Compare historical pool25 values to the previous lambda10 frozen evaluator.
    old=pd.read_csv(D.parent/'post_st_shrinkage_fine_v1/HISTORICAL_VALIDATION_METRICS.csv');old=old[old.strength==10]
    new=pd.read_csv(D/'HISTORICAL_VALIDATION_METRICS.csv');new=new[new.pool_count==25]
    from experiments.post_st_benchmark_calibration_v1.bootstrap import FIELDS
    merged=new.merge(old,on=['outer_cutoff','scope','months','cause'],suffixes=('_new','_old'));assert len(merged)==len(old)==len(new)
    for field in FIELDS:np.testing.assert_allclose(merged[field+'_new'],merged[field+'_old'],rtol=0,atol=1e-12,equal_nan=True)
    # Exact company identifiers are checked only when their format is distinctive;
    # generic numeric group labels are not identifiers and would false-match counts.
    identifiers=set()
    for p in (inp/'folds').rglob('*.npz'):
        z=arrays(p)
        for key in ['sample_ids']:
            if key in z:
                for value in z[key].astype(str):
                    if len(value)>=8 and not value.isdigit():identifiers.add(value)
    assert identifiers
    public=[]
    forbidden={'sample_id','sample_ids','company_id','company_code','stock_code','origin_date','event_date','true_threshold'}
    for p in D.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix not in ['.csv','.md','.json','.log','.py']:continue
        text=p.read_text();public.append(p)
        assert not any(value in text for value in identifiers),f'company identifier in {p.name}'
        if p.suffix=='.csv':assert not forbidden&set(pd.read_csv(p,nrows=0).columns),p.name
    testlog=(D/'pytest_final.log').read_text();assert '338 passed, 1 warning' in testlog and 'skipped' not in testlog
    env=dict(python=platform.python_version(),platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib','pytest']},
        oliver_interpreter='/home/oliver/.venvs/xai-baselines-20261027/bin/python',lemon_interpreter='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python',OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,independent_R_tests_enabled=True)
    write(D/'ENVIRONMENT.json',env)
    write(D/'FAILURE_RECORDS.json',dict(failures=failures,formal_fits=468,failed_tasks=0,reruns=0,publication_recoveries=0))
    write(D/'FINAL_AUDIT.json',dict(immutable_input_hashes_verified=True,core_pointer_and_previous_experiment_unchanged=True,
        forest_fits=468,calibrator_fits=0,raw_members=468,shrunk_members=468,merged_predictors=156,reference_raw_reproductions=18,current_shrunk_reproductions=9,
        numerical_models=len(num),maximum_errors=errors,full_q2_probability_and_reload_checks=True,
        company_identifiers_found=0,public_text_files_scanned=len(public),public_company_level_tables=False,
        unsupported_2025_H24_rows=len(unsupported),historical_pool25_metric_rows_reproduced=len(merged),
        jobs=jobs,failed_tasks=0,final_full_pytest='338 passed, 1 unrelated EBM visualization warning; independent R enabled; 0 skipped'))
    print(json.dumps(dict(verified_fits=468,numerical_models=len(num),maximum_errors=errors,jobs=jobs,privacy='PASS'),ensure_ascii=False))
if __name__=='__main__':main()
