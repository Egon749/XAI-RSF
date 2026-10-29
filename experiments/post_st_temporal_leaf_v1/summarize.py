"""Summarize fixed forecasts, including the pre-test selected yearly policy.

This module never fits, selects parameters or reads any 2025 company records.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,horizon_predictions,sha,write
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .stages import artifacts


def summarize(root):
    selection=json.loads((root/'audit/selection_freeze.json').read_text())
    protocol=json.loads((root/'input/protocol.json').read_text())
    assert protocol==json.loads(Path(__file__).with_name('frozen_protocol.json').read_text())
    assert json.loads((root/'evaluation/independent_audit.json').read_text())['status']=='PASS'
    dest=root/'summary';dest.mkdir()
    old=Path(protocol['reference_run']);data={};forecasts={}
    for year in protocol['outer_years']:
        data[str(year)]=arrays(old/'evaluation_input'/f'{year}_test_PRIVATE.npz')
    for year,_,seed,folder in artifacts(root,'formal'):
        arm=selection['chosen'][str(year)]['arm']
        forecasts[(str(year),seed)]=arrays(folder/arm/'predictions_PRIVATE.npz')
    risks={}
    for year in map(str,protocol['outer_years']):
        members=[forecasts[(year,str(s))] for s in protocol['seeds']]
        forecasts[(year,'ensemble')]=dict(sample_ids=members[0]['sample_ids'],times=members[0]['times'],cif=np.mean([p['cif'] for p in members],axis=0))
        for seed in [*map(str,protocol['seeds']),'ensemble']:
            p=forecasts[(year,seed)];ev=data[year]
            np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids'])
            for h in [12,24]:risks[(year,seed,h)]=horizon_predictions(p['cif'],p['times'],ev[f'h{h}'])
    data['pooled']={k:np.concatenate([data[str(y)][k] for y in protocol['outer_years']]) for k in data['2023']}
    for seed in [*map(str,protocol['seeds']),'ensemble']:
        for h in [12,24]:risks[('pooled',seed,h)]=np.concatenate([risks[(str(y),seed,h)] for y in protocol['outer_years']])
    rows=[];warnings=[];bins=[]
    for (year,seed,h),risk in risks.items():
        ev=data[year]
        for k,cause in [(1,'termination'),(2,'recovery')]:
            common=dict(scope=year,arm='selected_policy',seed=seed,horizon_months=h,cause=cause,weighting='one_company_one_record')
            m,w,target,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),h),risk[:,k-1],k,ev['label_eligible'])
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            np.testing.assert_allclose([m[z] for z in ['auc','ap','brier']],direct_metrics(target,risk[:,k-1]),atol=1e-12,rtol=0,equal_nan=True)
            rows.append(dict(**common,**m,observed_minus_predicted_pp=100*(m['ipcw_observed_incidence']-m['risk_mean'])))
            if seed=='ensemble':
                for b,ix in enumerate(np.array_split(np.lexsort((ev['sample_ids'],risk[:,k-1])),4)):
                    bins.append(dict(**common,bin=b+1,n=len(ix),predicted=float(risk[ix,k-1].mean()),observed=float(target[ix].mean()),target_events=int(target[ix].sum())))
            if k!=1:continue
            for fraction in [.2,.25]:
                chosen=np.zeros(len(risk),bool)
                for y in np.unique(ev['year']):
                    ix=np.flatnonzero(ev['year']==y);order=np.lexsort((ev['sample_ids'][ix],-risk[ix,0]))
                    chosen[ix[order[:int(np.ceil(len(ix)*fraction))]]]=True
                warnings.append(dict(**common,top_fraction=fraction,checked=int(chosen.sum()),captured=int((chosen&target).sum()),target_events=int(target.sum()),
                    coverage=float((chosen&target).sum()/target.sum()),known_non_target=int((chosen&~target).sum()),unknown=0,
                    lead3=int((chosen&target&(ev['analysis_time']>=3)).sum()),lead6=int((chosen&target&(ev['analysis_time']>=6)).sum()),repeat_alerts=0))
    metrics=pd.concat([pd.read_csv(root/'evaluation/metrics.csv'),pd.DataFrame(rows)],ignore_index=True)
    warnings=pd.concat([pd.read_csv(root/'evaluation/warning_metrics.csv'),pd.DataFrame(warnings)],ignore_index=True)
    bins=pd.concat([pd.read_csv(root/'evaluation/calibration_bins.csv'),pd.DataFrame(bins)],ignore_index=True)
    for name,frame in [('metrics_with_selected_policy',metrics),('warnings_with_selected_policy',warnings),('bins_with_selected_policy',bins)]:
        frame.to_csv(dest/f'{name}.csv',index=False)
    numeric=['auc','ap','brier','risk_mean','observed_minus_predicted_pp']
    ranges=metrics[metrics.seed.astype(str).isin(list(map(str,protocol['seeds'])))].groupby(['scope','arm','horizon_months','cause'])[numeric].agg(['mean','min','max','std'])
    ranges.columns=['_'.join(x) for x in ranges.columns];ranges.reset_index().to_csv(dest/'seed_metric_ranges.csv',index=False)
    ensemble=metrics[(metrics.seed=='ensemble')&(metrics.cause=='termination')].copy()
    for fraction,label in [(.2,'top20'),(.25,'top25')]:
        ww=warnings[(warnings.seed=='ensemble')&(warnings.top_fraction==fraction)][['scope','arm','horizon_months','checked','captured','coverage','lead3','lead6']]
        ww=ww.rename(columns={k:f'{label}_{k}' for k in ['checked','captured','coverage','lead3','lead6']})
        ensemble=ensemble.merge(ww,on=['scope','arm','horizon_months'],validate='one_to_one')
    ensemble.to_csv(dest/'termination_ensemble_summary.csv',index=False)
    leaves=pd.read_csv(root/'evaluation/node_audit.csv');summary=[]
    for (year,arm),d in leaves.groupby(['year','arm']):
        summary.append(dict(year=year,arm=arm,leaves=int(d.leaves.sum()),zero_target_leaves=int(d.zero_target_leaves.sum()),
            zero_target_fraction=float(d.zero_target_leaves.sum()/d.leaves.sum()),at_most_one_target_fraction=float(d.at_most_one_target_leaves.sum()/d.leaves.sum()),
            min_unique_company_support=int(d.unique_company_support_min.min()),mean_seed_median_unique_support=float(d.unique_company_support_median.mean()),
            mean_seed_median_effective_support=float(d.effective_company_support_median.mean())))
    pd.DataFrame(summary).to_csv(dest/'leaf_support_summary.csv',index=False)
    val=pd.read_csv(root/'reports/rolling_validation_metrics.csv')
    val=val[(val.seed=='ensemble')&(val.cause=='termination')&val.horizon_supported]
    base=val[val.arm=='baseline'][['outer_year','validation_entry_year','horizon_months','brier','risk_mean']].rename(columns={'brier':'baseline_brier','risk_mean':'baseline_risk_mean'})
    drivers=val.merge(base,on=['outer_year','validation_entry_year','horizon_months'],validate='many_to_one')
    drivers['brier_difference']=drivers.brier-drivers.baseline_brier
    drivers['risk_mean_difference']=drivers.risk_mean-drivers.baseline_risk_mean
    drivers.to_csv(dest/'validation_selection_drivers.csv',index=False)
    write(dest/'summary_receipt.json',dict(status='COMPLETE',no_new_fit=True,selection_sha256=sha(root/'audit/selection_freeze.json'),
        selected_yearly_arms={y:v['arm'] for y,v in selection['chosen'].items()},pooled_scores_recomputed_on_concatenated_predictions=True,
        seed_ranges_are_not_confidence_intervals=True,rank_policy='retrospective within-entry-year; not an online annual quota policy',
        unique_training_companies_union=len(set(arrays(root/'input/formal/2023/train_PRIVATE.npz')['groups'])|set(arrays(root/'input/formal/2024/train_PRIVATE.npz')['groups'])),
        input_protocol_canonical_equality=True))
    print(ensemble[['scope','arm','horizon_months','auc','ap','brier','risk_mean','top20_captured','known_target_events']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);summarize(p.parse_args().run)
