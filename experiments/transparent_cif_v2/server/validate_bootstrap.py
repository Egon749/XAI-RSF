"""Independent point/repeated-company validation against sklearn and published metrics."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .bootstrap import setup

def main(a):
    source=a.run/'bootstrap_input';groups,calc=setup(source);point=calc(np.ones(len(groups)));metrics=pd.read_csv(a.run/'evaluation/metrics.csv');warnings=pd.read_csv(a.run/'evaluation/warning_company_quarter.csv');errors=[]
    for row in metrics.itertuples():
        for m in ['auc','ap','ipcw_brier']:
            key=f'{row.scope}|{row.model}|{row.seed}|{row.weighting}|{m}';errors.append(abs(point[key]-getattr(row,m)))
    for row in warnings[warnings.scope.astype(str).isin(['2023','2024','pooled'])].itertuples():
        for m,field in [('company_hit_fraction','company_hit_fraction'),('event_coverage','event_coverage'),('alert_companies','alert_company_groups'),('alert_records','alert_records'),('unknown_alert_records','unknown_outcome_alerts'),('lead3m_coverage','announcement_coverage_3m'),('lead6m_coverage','announcement_coverage_6m')]:
            key=f'{row.scope}|{row.model}|{row.seed}|Top{int(row.top_fraction*100)}|{m}';errors.append(abs(point[key]-getattr(row,field)))
    assert max(errors)<1e-10
    # A nontrivial common company draw; compare weighted ranking to sklearn.
    from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    d=pd.read_parquet(source/'metadata.parquet');pred=dict(np.load(source/'predictions.npz'));rng=np.random.default_rng(13827);counts=np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups));got=calc(counts);lookup=dict(zip(groups,counts));p=pred['base_equal|42'];y=(d.event.to_numpy()==1).astype(int);ipcw,_=source_consistent_endpoint_weights(d,endpoint_weights);rw=d.group.astype(str).map(lookup).to_numpy();w=rw*ipcw;keep=w>0
    for metric,fn in [('auc',roc_auc_score),('ap',average_precision_score)]:assert abs(got[f'pooled|base_equal|42|record_equal|{metric}']-fn(y[keep],p[keep],sample_weight=w[keep]))<1e-10
    with (a.run/'audit/bootstrap_validation.json').open('x') as f:json.dump(dict(status='PASS',point_comparisons=len(errors),max_point_error=max(errors),nontrivial_draw_sklearn_check=True),f,indent=2)
    print('Bootstrap point and repeated-company checks passed',len(errors),max(errors))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args())
