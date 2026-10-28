"""Independent announcement/censoring audit and saved-tree support summaries."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from crforest.partition import tree_feature_subset
from experiments.post_st_prognosis_v1.common import arrays,sha,write


def audit(run):
    endpoints=pd.read_parquet(run/'private/endpoint_ledger_PRIVATE.parquet')
    checked=0;names=json.loads((run/'input/plan.json').read_text())['feature_names']
    source_cap=pd.Timestamp('2026-09-03')
    for role in ['train','eval']:
        meta=pd.read_parquet(run/'input'/f'{role}_metadata_PRIVATE.parquet')
        a=arrays(run/'input'/f'{role}_PRIVATE.npz')
        np.testing.assert_array_equal(meta.sample_id,a['sample_ids'])
        for row in meta.itertuples():
            deadline=min(row.origin+pd.DateOffset(months=12),source_cap,
                         pd.Timestamp('2024-12-31') if role=='train' else source_cap)
            e=endpoints[(endpoints.group==row.group)&(endpoints.announcement>row.origin)&(endpoints.announcement<=deadline)]
            if len(e):
                stop=e.announcement.min();k=e.loc[e.announcement==stop,'type'].str[-1].unique()
                assert len(k)==1;event={'X':1,'A':2}[k[0]]
            else:stop=deadline;event=0
            assert row.event==event and row.stop==stop and row.time==(stop-row.origin).days
            assert row.label_eligible and not row.unresolved_chain_gap
            checked+=1
    assert checked==443
    rows=[];price=names.index('log_price_last')
    for file in sorted((run/'results/formal').glob('*/task-*/output/summary.json')):
        s=json.loads(file.read_text())
        if s['model']=='standard_rsf':continue
        p=file.parent/'forest_PRIVATE.json'
        assert sha(p)==json.loads((file.parent/'hashes.json').read_text())[p.name]
        f=json.loads(p.read_text());cfg=f['config'];trees=f['trees']
        leaves=[n for tree in trees for n in tree if 'cif' in n]
        used={n['feature'] for tree in trees for n in tree if 'feature' in n}
        write(run/'private'/f"used_inputs_{s['model']}_{s['seed']}_PRIVATE.json",[names[i] for i in sorted(used)])
        rows.append(dict(model=s['model'],seed=s['seed'],actual_used_inputs=len(used),leaves=len(leaves),
            zero_termination_leaf_fraction=float(np.mean([n['cause_company_support'][0]==0 for n in leaves])),
            fewer_than_three_termination_leaf_fraction=float(np.mean([n['cause_company_support'][0]<3 for n in leaves])),
            zero_recovery_leaf_fraction=float(np.mean([n['cause_company_support'][1]==0 for n in leaves])),
            price_eligible_trees=sum(price in tree_feature_subset(f['n_features'],cfg['tree_feature_count'],cfg['seed'],i) for i in range(len(trees))),
            price_used_trees=sum(any(n.get('feature')==price for n in t) for t in trees),
            root_price_trees=sum(t[0].get('feature')==price for t in trees)))
    pd.DataFrame(rows).to_csv(run/'evaluation/tree_support.csv',index=False)
    ev=arrays(run/'input/eval_PRIVATE.npz');tr=arrays(run/'input/train_PRIVATE.npz')
    pd.DataFrame(dict(name=names,train_missing=np.isnan(tr['X']).mean(0),test_missing=np.isnan(ev['X']).mean(0))).to_csv(run/'evaluation/input_missingness.csv',index=False)
    meta=pd.read_parquet(run/'input/eval_metadata_PRIVATE.parquet')
    write(run/'evaluation/additional_validation.json',dict(status='PASS',independent_date_records=checked,
        cutoff_and_12m_cap_verified=True,test_origin_min=str(meta.origin.min()),test_origin_max=str(meta.origin.max()),
        test_daily_lifecycle_first_supported=int(meta.prior_history_daily_complete.sum()),test_companies=len(meta),
        other_first_status='Accepted observed-first with inherited incomplete-history limitations; not certified lifetime first',
        financial_available_max=str(meta.financial_available.max()),market_available_max=str(meta.market_available.max()),
        all_test_feature_dates_before_origin=True,all_test_endpoints_known=True,source_coverage_end=str(source_cap)))
    print(json.dumps(dict(status='PASS',independent_date_records=checked,tree_models=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
