"""Read-only hash, aggregation and public-artifact checks; never refits."""
from pathlib import Path
import argparse,datetime,json,re
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,write


def verify(root,public):
    receipt=json.loads((public/'execution_receipt.json').read_text())
    assert receipt['tasks_succeeded']==20 and receipt['new_forest_fits']==34
    for row in pd.read_csv(public/'results/source_hashes.csv').itertuples():assert sha(Path(row.path))==row.sha256,row.path
    for row in pd.read_csv(public/'artifact_manifest.csv').itertuples():assert sha(public/row.file)==row.sha256,row.file
    for name,h in receipt['preserved_public_sha256'].items():assert sha(Path(name))==h
    assert receipt['calibration_fits']==0 and not receipt['new_2025_evaluation']
    selected=pd.read_csv(public/'results/metrics_with_selected_policy.csv');original=pd.read_csv(public/'results/metrics.csv')
    selected['scope']=selected.scope.astype(str);original['scope']=original.scope.astype(str)
    for year,arm in [('2023','baseline'),('2024','time_w4')]:
        a=selected[(selected.scope==year)&(selected.arm=='selected_policy')].sort_values(['seed','horizon_months','cause'])
        b=original[(original.scope==year)&(original.arm==arm)].sort_values(['seed','horizon_months','cause'])
        np.testing.assert_allclose(a[['auc','ap','brier','risk_mean']],b[['auc','ap','brier','risk_mean']],atol=1e-12,rtol=0,equal_nan=True)
    keys=['arm','seed','horizon_months','cause']
    for _,d in selected.groupby(keys):
        p=d[d.scope=='pooled'].iloc[0];years=d[d.scope!='pooled']
        assert p.companies==years.companies.sum()
        assert p.known_target_events==years.known_target_events.sum()
        for key in ['brier','risk_mean','ipcw_observed_incidence']:
            assert abs(p[key]-np.average(years[key],weights=years.companies))<1e-12
    nodes=pd.read_csv(public/'results/node_audit.csv')
    for year in [2023,2024]:
        base=nodes[(nodes.year==year)&(nodes.arm=='baseline')].sort_values('seed')
        for arm in ['time_w2','time_w4']:
            d=nodes[(nodes.year==year)&(nodes.arm==arm)].sort_values('seed')
            np.testing.assert_array_equal(base[['leaves','zero_target_leaves']],d[['leaves','zero_target_leaves']])
    assert len(original)==192 and len(selected)==240
    # Explicit public outputs are aggregates, source hashes, logs, figures,
    # generic source code, protocol and receipts; never serialized models.
    forbidden_columns={'sample_id','sample_ids','company_id','code','origin','stop','event_date','threshold','X'}
    for path in public.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:continue
        assert '_PRIVATE' not in path.name and path.suffix not in {'.npz','.parquet','.pkl','.joblib'},path
        if path.suffix=='.csv':assert not forbidden_columns.intersection(pd.read_csv(path,nrows=0).columns),path
        if path.suffix=='.md':
            text=path.read_text();assert not any(ord(c)<32 and c not in '\n\r\t' for c in text),path
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if not link.startswith(('https:','http:','#','/')):assert (path.parent/link.split('#')[0]).exists(),(path,link)
    assert '139 passed' in (public/'results/full_tests_delivery_final.log').read_text()
    audit=dict(status='PASS',source_hashes_verified=len(pd.read_csv(public/'results/source_hashes.csv')),
        public_artifacts_verified=len(pd.read_csv(public/'artifact_manifest.csv')),original_results_preserved=True,
        selected_policy_matches_frozen_yearly_selection=True,pooled_additive_metrics_reconciled=True,
        public_company_detail_files_absent=True,all_local_report_links_exist=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    write(root/'audit'/f'delivery_verification_{stamp}.json',audit);print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();verify(a.run,a.public)
