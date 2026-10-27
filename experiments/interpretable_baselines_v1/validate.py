from pathlib import Path
import argparse,json,hashlib
import pandas as pd
import numpy as np


def main(run):
    out=run/'reports';inp=run/'input';plan=json.loads((inp/'plan.json').read_text());receipt=json.loads((out/'execution_receipt.json').read_text())
    m=pd.read_csv(out/'metrics.csv');w=pd.read_csv(out/'warning_metrics.csv');d=pd.read_csv(out/'task_diagnostics.csv')
    assert len(m)==192 and len(d)==261 and not receipt['failed_tasks']
    parent=pd.read_csv(Path(plan['order_run'])/'reports/metrics.csv');matched=0
    for model in ['q2','q3','q4','unrestricted','M2_allsector']:
        keys=['scope','seed','weighting'];a=m[m.model==model].sort_values(keys).reset_index(drop=True);b=parent[parent.model==model].sort_values(keys).reset_index(drop=True)
        assert a[keys].equals(b[keys])
        for col in ['auc','ap','ipcw_brier','prediction_mean','prediction_max']:np.testing.assert_allclose(a[col],b[col],rtol=0,atol=1e-14)
        matched+=len(a)
    assert np.array_equal(w.alert_records,w.hit_records+w.known_no_target_alerts+w.unknown_outcome_alerts)
    assert (w.detected_events<=w.independent_events).all()
    for _,g in w.groupby(['model','seed','top_fraction']):
        for year in ['2023','2024']:
            annual=g[g.scope==year].iloc[0];quarter=g[g.scope.str.startswith(year+'Q')];assert len(quarter)==4
            for col in ['alert_records','hit_records','known_no_target_alerts','unknown_outcome_alerts']:assert annual[col]==quarter[col].sum()
    bs=d[d.kind=='bootstrap'];assert len(bs)==24
    for _,g in bs.groupby('replicate'):assert g.resample_sha256.nunique()==1 and len(g)==3
    assert len(d[d.kind=='simulation'])==210 and len(d[d.kind=='shap'])==18 and len(d[d.kind=='baseline'])==9
    assert d.reconstruction_error.dropna().max()<1e-10 and d.additivity_max_error.dropna().max()<1e-10
    assert d.reload_error.dropna().max()<1e-12
    c=pd.read_csv(out/'native_complexity.csv');assert len(c)==18 and set(c.native_scale)=={'CIF','log_odds'}
    forbidden={'sample_id','sample_ids','company_code','group','threshold','features','cif_contribution'}
    for p in out.glob('*.csv'):assert not forbidden.intersection(pd.read_csv(p,nrows=0).columns),p
    assert receipt['evaluation_script_sha256']==hashlib.sha256((Path(__file__).parent/'evaluate.py').read_bytes()).hexdigest()
    result=dict(status='PASS',formal_tasks=261,formal_fits=243,failed_tasks=0,unchanged_reference_metric_rows=matched,common_cluster_draws_verified=8,
        quarterly_totals_verified=True,unknowns_not_treated_as_normal=True,probability_vs_logit_scales_separate=True,
        full_reconstruction_passed=True,public_csvs_aggregate_only=True,evaluation_script_hash_matches=True)
    with (out/'delivery_validation.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
