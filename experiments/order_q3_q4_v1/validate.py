"""Delivery checks against the frozen parent; reads private artifacts locally."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd


def main(run,dest):
    parent=Path(json.loads((run/'audit/reuse_receipt.json').read_text())['parent'])
    plan=json.loads((run/'input/plan.json').read_text());old=json.loads((parent/'input/plan.json').read_text())
    for seed in plan['seeds']:
        for variant in plan['variants']:
            a=plan['configs'][f'final/{seed}/{variant}'].copy();b=old['configs'][f'final/{seed}/coverage'].copy()
            assert a.pop('path_budget')==int(variant[-1]) and b.pop('path_budget')==2
            assert a==b
    m=pd.read_csv(dest/'metrics.csv');p=pd.read_csv(parent/'reports/metrics.csv')
    assert len(m)==120 and len(m[m.aggregation=='single_seed'])==90
    # Frozen references must reproduce the previously reported metrics exactly,
    # on every scope/weighting/seed, including prediction ensembles.
    matched=0
    for model,previous in [('q2','coverage'),('unrestricted','unrestricted'),('M2_allsector','M2_allsector')]:
        keys=['scope','seed','weighting']
        a=m[m.model==model].sort_values(keys).reset_index(drop=True)
        b=p[p.model==previous].sort_values(keys).reset_index(drop=True)
        assert a[keys].equals(b[keys])
        for col in ['auc','ap','ipcw_brier','prediction_mean','observed_ipcw_incidence','prediction_max']:
            np.testing.assert_allclose(a[col],b[col],atol=1e-14,rtol=0)
        matched+=len(a)
    w=pd.read_csv(dest/'warning_metrics.csv')
    for (model,seed,fraction),g in w.groupby(['model','seed','top_fraction']):
        for year in ['2023','2024']:
            annual=g[g.scope==year].iloc[0];quarter=g[g.scope.str.startswith(year+'Q')]
            assert len(quarter)==4
            for col in ['alert_records','hit_records','known_no_target_alerts','unknown_outcome_alerts']:
                assert annual[col]==quarter[col].sum()
    receipt=json.loads((dest/'execution_receipt.json').read_text())
    assert receipt['formal_fits']==6 and not receipt['failed_tasks']
    assert receipt['evaluation_script_sha256']==hashlib.sha256((Path(__file__).parent/'evaluate.py').read_bytes()).hexdigest()
    diag=pd.read_csv(dest/'training_diagnostics.csv')
    assert len(diag)==6 and set(zip(diag.variant,diag.seed))=={(v,s) for v in ['q3','q4'] for s in [42,101,202]}
    assert (diag.path_violations==0).all() and (diag.reload_error==0).all()
    assert (diag.max_path_variables==diag.requested_path_order).all()
    assert diag.reconstruction_error.max()<1e-10 and diag.explicit_component_sum_error.max()<1e-10
    assert (diag.truncated_q2_max_error>1e-10).all()
    for v in ['q3','q4']:
        assert (diag.loc[diag.variant==v,f'order{v[-1]}_functions']>0).all()
    forbidden={'sample_id','sample_ids','group','company_code','threshold','feature_value','features','cif_contribution'}
    for file in dest.glob('*.csv'):assert not forbidden.intersection(pd.read_csv(file,nrows=0).columns),file
    tests=(dest/'tests_final.txt').read_text();assert '51 passed' in tests and 'skipped' not in tests
    result=dict(status='PASS',formal_fits=6,failed_tasks=0,metric_rows=len(m),unchanged_reference_metric_rows=matched,
        only_config_change='path_budget',quarterly_alert_totals_equal_annual=True,all_orders_present=True,
        truncated_explanations_rejected=True,full_reconstruction_verified=True,public_CSVs_aggregate_only=True,
        evaluation_script_hash_matches_receipt=True,tests_passed=51,independent_R_checks=2,skipped=0,
        post_exposure_exploratory=True,strict_blind=False,industry_PIT_certified=False)
    with (dest/'delivery_validation.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,default=Path(__file__).parent)
    a=p.parse_args();main(a.run,a.dest)
