"""Final read-only checks of the completed finite run and aggregate release."""
import csv,subprocess
from .common import *

def main():
    r=run();files=[r/'pilot/summary.json']+list((r/'published').glob('*/task-*/output/summary.json'))
    assert len(files)==27
    summaries=[load(p) for p in files];assert len({(s['year'],s['earliest_entry_year']) for s in summaries})==27
    assert sum(s['forest_fits'] for s in summaries)==81
    for s in summaries:
        tr=r/f"input/folds/{s['year']}/{s['earliest_entry_year']}/train_PRIVATE.npz"
        assert sha(tr)==s['source_train_sha256']
        for member in s['members']:
            assert member['forest_fits']==1
            for ck in member['checks'].values():
                assert ck['max_path_variables']<=2 and ck['max_depth']<=4
                assert max(ck[k] for k in ['full_reconstruction','explicit_component_sum','forest_reload','function_reload'])<=1e-10
                assert ck['probability_min']>=-1e-12 and ck['probability_sum_max']<=1+1e-10 and ck['minimum_increment']>=-1e-12
    sched=load(r/'slurm_results_PRIVATE.json')
    assert sched['pending']==[] and len(sched['completed'])==sched['expected']==26 and sched['srun_returncode']==0
    assert {x['id'] for x in sched['completed']}==set(range(26))
    assert all(x['returncode']==0 and x['compute_completed'] and x['publication_completed'] for x in sched['completed'])
    audit=load(D/'DATA_AUDIT.json');private_identifiers=set()
    for y in YEARS:
        trpath=source()/f'input/folds/{y}/train_PRIVATE.npz';evpath=source()/f'input/folds/{y}/eval_PRIVATE.npz'
        assert sha(trpath)==audit['input_hashes'][f'{y}/source_train'] and sha(evpath)==audit['input_hashes'][f'{y}/eval_X']
        tr=arrays(trpath);labels=arrays(source()/f'input/evaluation_input/{y}_test_PRIVATE.npz')
        private_identifiers.update(map(str,tr['sample_ids']));private_identifiers.update(map(str,tr['groups']))
        private_identifiers.update(map(str,labels['sample_ids']));private_identifiers.update(map(str,labels['groups']))
        assert not set(tr['groups'])&set(labels['groups'])
    for p in D.glob('*.csv'):
        with p.open() as f:
            reader=csv.DictReader(f)
            assert not set(reader.fieldnames)&{'sample_id','company_id','company_code','code','group','origin','stop','threshold'}
            for row in reader:
                assert not set(row.values())&private_identifiers, f'Private identifier in {p.name}'
    assert not list(D.rglob('*_PRIVATE.npz')) and not list(D.rglob('*_PRIVATE.parquet')) and not list(D.rglob('forest_PRIVATE.json'))
    assert sha(ROOT/'CURRENT_POST_ST_MODEL.json')==audit['current_pointer_sha256']
    diff=subprocess.check_output(['git','diff','3e58f6dfc3407692da00ca7775a4cc9f5e4de647','--','crforest','CURRENT_POST_ST_MODEL.json'],cwd=ROOT)
    assert not diff
    checks=pd.read_csv(D/'NUMERICAL_COMPLEXITY.csv');assert len(checks)==189
    met=pd.read_csv(D/'METRICS_ENSEMBLE.csv');assert len(met)==9*4*2*2
    na=met[(met.scope=='2025')&(met.months==24)]
    assert na[['auc','ap','ipcw_brier','observed','top20_captured']].isna().all().all()
    re=pd.read_csv(D/'HISTORICAL_REPRODUCTION.csv');assert len(re)==9 and re.trees_identical.all()
    assert (re.all_row_time_error==0).all() and (re.saved_prediction_error==0).all()
    result=dict(status='PASS',actual_forests=81,completed_window_years=27,ensemble_models=27,exact_model_exports=189,
        original_C039_reproductions=9,failed_fits=0,final_log_failure_recovered_without_refitting=True,core_and_pointer_unchanged=True,
        public_company_identifier_cells=0,company_disjoint_each_cutoff=True,source_inputs_unchanged=True,
        max_full_reconstruction=float(checks.full_reconstruction.max()),max_component_sum=float(checks.explicit_component_sum.max()),
        max_reload=float(checks.forest_reload.max()),two_nodes=list(sorted({x['host'] for x in sched['completed']})),
        publication_verified_tasks=26,unsupported_2025_h24_performance='NA')
    write(D/'FINAL_VALIDATION.json',result);print(json.dumps(result))

if __name__=='__main__':main()
