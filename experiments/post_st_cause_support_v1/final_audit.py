"""Verify completed outputs and protected historical inputs, without new fits."""
from itertools import product
import pandas as pd
from .common import *

def main():
    for dirname,manifest in [('input','immutable_input_hashes.json'),('analysis_input','immutable_analysis_hashes.json')]:
        for n,h in load(RUN/'audit'/manifest).items():assert sha(RUN/dirname/n)==h
    for n,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(n))==h
    pointer=load(RUN/'audit/current_pointer.json');assert sha(Path(pointer['path']))==pointer['sha256']
    counts={};hosts={}
    for stage,expected in [('pilot',2),('formal',36),('ensemble',24)]:
        rows=load(RUN/f'audit/registry_{stage}.json');gate=load(RUN/f'audit/gate_{stage}.json');assert len(rows)==expected and gate['status']=='PASS'
        assert sha(Path(gate['report']))==gate['report_sha256']
        rpt=load(Path(gate['report']));assert len(rpt['completed'])==expected and not rpt['pending']
        for task in rpt['completed']:assert task['compute_completed'] and task['publication_completed'] and task['returncode']==0
        for r in rows:
            p=Path(r['path']);s=load(p/'summary.json');assert s['status']=='COMPLETE'
            for name,h in load(p/'hashes.json').items():assert sha(p/name)==h
        counts[stage]=len(rows);hosts[stage]=sorted({r['host'] for r in rows})
    assert hosts['formal']==['lemon','oliverdesktop']
    rows=load(RUN/'audit/registry_formal.json')
    assert {(r['year'],r['arm'],r['seed']) for r in rows}==set(product(YEARS,ARMS,SEEDS))
    public=REPO/'experiments'/NAME;m=pd.read_csv(public/'METRICS_ALL.csv');assert len(m)==512
    unsupported=m[(m.scope=='2025')&(m.months==24)]
    assert not unsupported.supported.any() and unsupported[['auc','ap','brier','captured','coverage']].isna().all().all()
    none=m[(m.scope=='2023')&(m.months==12)&(m.cause==2)]
    assert none[['auc','ap']].isna().all().all() and none.brier.notna().all()
    fv=load(public/'function_validation.json');assert fv['max_explicit_sum_error']<=1e-10 and fv['max_path_variables']==2
    assert fv['original_reference_exact_matches']==9 and fv['max_forest_reload_error']==0
    costs=pd.read_csv(public/'COMPUTE_COST.csv');assert len(costs)==36 and costs.reference_trees_identical.eq(True).sum()==9
    guard=pd.read_csv(public/'LEAF_SUPPORT.csv');assert guard.loc[guard.arm.str.contains('support'),'one_target_leaves'].eq(0).all()
    pairs=pd.read_csv(public/'PAIRING_CHECKS.csv');assert len(pairs)==36 and pairs.bootstrap_multiplicities_and_feature_sets_identical.all()
    ci=pd.read_csv(public/'PAIRED_BOOTSTRAP_95.csv');assert ci.repeats.eq(1000).all() and (ci.valid_repeats+ci.undefined_repeats).eq(1000).all()
    for file in public.rglob('*'):
        assert not (file.is_file() and ('_PRIVATE' in file.name or file.suffix in ['.npz','.parquet']))
    for file in public.glob('*.csv'):
        headers=pd.read_csv(file,nrows=0).columns
        assert not {'sample_id','company_id','code','threshold','cif','coefficients'}&set(headers)
    receipt=load(public/'execution_receipt.json')
    for n,h in receipt['public_results_sha256'].items():assert sha(public/n)==h
    out=dict(status='PASS',verified_expected_tasks=counts,actual_hosts=hosts,all_input_and_output_hashes_valid=True,
        old_models_and_current_pointer_unchanged=True,complete_arms_and_seeds=True,unsupported_metrics_preserved=True,
        no_private_records_parameters_or_thresholds_in_public_directory=True,forest_fits=0)
    write(public/'FINAL_AUDIT.json',out);write(RUN/'audit/final_audit.json',out);print(json.dumps(out))

if __name__=='__main__':main()
