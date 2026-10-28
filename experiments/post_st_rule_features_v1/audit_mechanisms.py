"""Inspect immutable trees and endpoint dates; no fitting, tuning or replacement.

Price-using versus other-tree averages are post-exposure diagnostic partitions,
not proposed/selectable models. Both partitions and every seed are reported.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from crforest.partition import tree_feature_subset
from experiments.post_st_prognosis_v1.common import arrays, sha, write, calendar_elapsed, endpoint_weights


def tree_predictions(nodes, X, grid, horizons):
    """Traverse saved leaves at each company's calendar horizon, without fitting."""
    ix=np.searchsorted(grid,horizons,side='right')-1
    ans=np.zeros((len(X),2))
    for row in range(len(X)):
        at=0
        while 'cif' not in nodes[at]:
            n=nodes[at];v=X[row,n['feature']]
            left=v<=n['threshold'] or (np.isnan(v) and n['missing_left'])
            at=n['left' if left else 'right']
        if ix[row]>=0:ans[row]=nodes[at]['cif'][ix[row]]
    return ans


def run(base,raw_audit,out):
    out.mkdir(exist_ok=True);private=out/'private';private.mkdir(exist_ok=False)
    reports=out/'reports';reports.mkdir(exist_ok=False)
    tr=arrays(base/'input/train_PRIVATE.npz');ev=arrays(base/'input/eval_PRIVATE.npz')
    assert ((ev['event']>0)|(ev['time']>=ev['h24'])).all()
    names=pd.read_csv(base/'reports/feature_manifest.csv').name.tolist()
    price=names.index('log_price_last')
    original=json.loads(Path('experiments/post_st_prognosis_v1/execution_receipt.json').read_text())
    model_hash={(x['model'],x['seed']):x for x in original['model_hashes']}
    source_hashes={str(base/'input'/f):sha(base/'input'/f) for f in original['model_input_hashes']}
    for f,digest in original['model_input_hashes'].items():assert source_hashes[str(base/'input'/f)]==digest
    summaries=[];metrics=[];errors=[]
    for directory in sorted((base/'results/formal/2534').glob('task-*/output')):
        summary=json.loads((directory/'summary.json').read_text())
        if summary['model']=='standard_rsf':continue
        path=directory/'forest_PRIVATE.json';f=json.loads(path.read_text());cfg=f['config']
        identity=(summary['model'],cfg['seed']);assert sha(path)==model_hash[identity]['model_sha256']
        saved=arrays(directory/'predictions_PRIVATE.npz')
        assert sha(directory/'predictions_PRIVATE.npz')==model_hash[identity]['prediction_sha256']
        source_hashes[str(path)]=sha(path)
        source_hashes[str(directory/'predictions_PRIVATE.npz')]=sha(directory/'predictions_PRIVATE.npz')
        details=[];all_predictions=[]
        for tid,nodes in enumerate(f['trees']):
            eligible=price in tree_feature_subset(f['n_features'],cfg['tree_feature_count'],cfg['seed'],tid)
            splits=[n for n in nodes if 'feature' in n];leaves=[n for n in nodes if 'cif' in n]
            used=any(n['feature']==price for n in splits);assert not used or eligible
            details.append(dict(tree=tid,price_eligible=eligible,price_used=used,
                root_price=nodes[0].get('feature')==price,leaves=len(leaves),
                zero_termination_leaves=sum(n['cause_company_support'][0]==0 for n in leaves),
                fewer_than_three_termination_companies=sum(n['cause_company_support'][0]<3 for n in leaves),
                leaves_fewer_than_20_bootstrap_rows=sum(n['n']<20 for n in leaves),
                leaves_below_total_event_stop=sum(n['events']<2*cfg['min_events'] for n in leaves),
                leaves_at_depth_cap=sum(n['depth']==cfg['max_depth'] for n in leaves),
                root_independent_termination_companies=nodes[0]['cause_company_support'][0],
                min_leaf_bootstrap_rows=min(n['n'] for n in leaves),
                min_leaf_groups=min(n['groups'] for n in leaves)))
            all_predictions.append(tree_predictions(nodes,ev['X'],f['grid'],ev['h24']))
        a=pd.DataFrame(details);pred=np.array(all_predictions)
        ix=np.searchsorted(saved['times'],ev['h24']);assert np.array_equal(saved['times'][ix],ev['h24'])
        reference=saved['cif'][np.arange(len(ev['X'])),ix]
        error=float(np.max(abs(pred.mean(0)-reference)));assert error<1e-12;errors.append(error)
        prefix=f'{identity[0]}_{identity[1]}'
        a.to_csv(private/f'{prefix}_tree_audit_PRIVATE.csv',index=False)
        np.savez_compressed(private/f'{prefix}_tree_predictions_PRIVATE.npz',cif=pred,sample_ids=ev['sample_ids'])
        summaries.append(dict(model=identity[0],seed=identity[1],trees=len(a),
            price_eligible_trees=int(a.price_eligible.sum()),price_used_trees=int(a.price_used.sum()),
            price_root_trees=int(a.root_price.sum()),leaves=int(a.leaves.sum()),
            average_leaves_per_tree=float(a.leaves.mean()),
            zero_termination_leaf_fraction=float(a.zero_termination_leaves.sum()/a.leaves.sum()),
            fewer_than_three_termination_companies_leaf_fraction=float(a.fewer_than_three_termination_companies.sum()/a.leaves.sum()),
            rows_below_20_leaf_fraction=float(a.leaves_fewer_than_20_bootstrap_rows.sum()/a.leaves.sum()),
            below_total_event_stop_leaf_fraction=float(a.leaves_below_total_event_stop.sum()/a.leaves.sum()),
            depth_cap_leaf_fraction=float(a.leaves_at_depth_cap.sum()/a.leaves.sum()),
            root_termination_company_min=int(a.root_independent_termination_companies.min()),
            root_termination_company_max=int(a.root_independent_termination_companies.max()),
            observed_min_leaf_bootstrap_rows=int(a.min_leaf_bootstrap_rows.min()),
            observed_min_leaf_groups=int(a.min_leaf_groups.min()),
            configured_min_leaf=cfg['min_leaf'],configured_min_groups=cfg['min_groups'],configured_min_total_events=cfg['min_events']))
        for group,trees in [('all',np.ones(len(a),bool)),('uses_price',a.price_used.to_numpy()),('does_not_use_price',~a.price_used.to_numpy())]:
            risk=pred[trees].mean(0)[:,0] if trees.any() else None
            for scope,mask in [('2023',ev['year']==2023),('2024',ev['year']==2024),('pooled',np.ones(len(ev['X']),bool))]:
                y=ev['event'][mask]==1;p=risk[mask] if risk is not None else None
                metrics.append(dict(model=identity[0],seed=identity[1],scope=scope,tree_group=group,tree_count=int(trees.sum()),n=int(mask.sum()),events=int(y.sum()),auc=roc_auc_score(y,p) if p is not None else None,ap=average_precision_score(y,p) if p is not None else None,mean_risk=float(p.mean()) if p is not None else None,diagnostic_only=True,unavailable_reason='no_saved_trees_in_group' if p is None else ''))
    pd.DataFrame(summaries).to_csv(reports/'tree_support_and_price_coverage.csv',index=False)
    pd.DataFrame(metrics).to_csv(reports/'fixed_tree_group_ranking.csv',index=False)
    rows=pd.concat([pd.read_parquet(base/'input'/f'{role}_metadata_PRIVATE.parquet') for role in ['train','eval']])
    train_meta=rows[rows.split.eq('train')]
    assert np.array_equal(train_meta.sample_id.to_numpy(str),tr['sample_ids'])
    elapsed=np.array([calendar_elapsed(o,s) for o,s in zip(train_meta.origin,train_meta.stop)])
    weights,known,_=endpoint_weights(elapsed,tr['event'],24,tr['label_eligible'])
    low_price=-tr['X'][:,price];assert np.isfinite(low_price).all()
    target=(tr['event']==1)&(elapsed<=24)
    pd.DataFrame([dict(scope='training',score='lower_price_ranking',known=int(known.sum()),
        unknown=int((~known).sum()),termination_events=int(target.sum()),
        auc=roc_auc_score(target[known],low_price[known],sample_weight=weights[known]),
        ap=average_precision_score(target[known],low_price[known],sample_weight=weights[known]),
        weighting='same calendar-month IPCW as original endpoint evaluation',
        interpretation='descriptive training-data signal; not independent validation')]).to_csv(reports/'training_price_ranking.csv',index=False)
    grounds_path=raw_audit/'direct_raw_analysis/private/termination_grounds_PRIVATE.parquet'
    source_hashes[str(grounds_path)]=sha(grounds_path)
    ground=pd.read_parquet(grounds_path).merge(rows[['sample_id','stop']],on='sample_id',validate='one_to_one')
    ground['termination_year']=ground.stop.dt.year
    ground.groupby(['scope','ground','termination_year']).size().rename('observed_events').reset_index().to_csv(reports/'entry_cohort_vs_termination_year.csv',index=False)
    code_hashes={str(p):sha(p) for p in [Path(__file__),Path('crforest/core.py'),Path('crforest/partition.py')]}
    for filename in ['core.py','partition.py']:
        assert sha(Path('crforest')/filename)==sha(base/'input/code/crforest'/filename)
    receipt=dict(status='COMPLETE_DIAGNOSTIC_ONLY',new_research_fits=0,new_training_inputs=False,old_predictions_replaced=False,
        cohorts_changed=False,trees_inspected=sum(s['trees'] for s in summaries),python_models_inspected=len(summaries),
        source_sha256=source_hashes,code_sha256=code_hashes,max_saved_prediction_reconstruction_error=max(errors),
        split_event_rule='event > 0, pooled recovery and termination; bootstrap multiplicity counts',
        subgroup_rule='presence of an actual price split in a fixed existing tree; both subgroups and every seed retained',
        interpretation='post-exposure diagnosis; no subtree selection, no promotion to a new predictive model',
        artifact_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    write(reports/'receipt.json',receipt)
    print(pd.DataFrame(summaries).to_json(orient='records'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--raw-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.base,a.raw_audit,a.out)
