"""Fixed-prediction, fixed-slope evaluation after protocol freeze and O/E gate."""
from pathlib import Path
import json, os, traceback
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from experiments.post_st_reporting_drift_v1.analyze import Reader, write_json
from experiments.post_st_reporting_drift_v1.metrics import stratified_metrics, paired_auc, interval, alert_metrics
from experiments.post_st_ablation_v1.uncertainty import fast_metrics
from experiments.post_st_feature_permutation_v1.common import evaluation_context

PUBLIC=Path(__file__).resolve().parent
YEARS=[2023,2024,2025]
ARMS=['reference','recovery_only','both_2y','both_1y','uncertified_2y']


def metrics(y,p,years,ids):
    fast=fast_metrics(y,p,years,ids)
    alert=alert_metrics(y,p,years,ids)
    oe=alert['observed_expected']
    return dict(**stratified_metrics(y,p,years,ids),ap=fast[1],brier=fast[2],
                **alert,abs_log_oe=abs(np.log(oe)) if oe>0 else np.inf,
                mean_minus_observed=alert['mean_predicted']-alert['observed'])


def main():
    loc=json.loads((PUBLIC/'run_location.json').read_text());run=Path(os.environ.get('CALENDAR_LEVEL_RUN',loc['private_run']));source=Path(loc['source_run'])
    out=run/'evaluation';out.mkdir();reader=Reader()
    gates=reader.json(run/'prepared/REFERENCE_GATE.json');assert len(gates)==5 and all(g['passed'] for g in gates)
    specs=reader.json(run/'prepared/specifications.json')
    refs=reader.arrays(run/'prepared/reference_risks_PRIVATE.npz')
    draws=reader.arrays(run/'prepared/bootstrap_indices_PRIVATE.npz')
    labs={y:reader.arrays(source/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    risks={(y,h,a):refs[f'{y}_{h}']*np.array(specs[str(y)][a]['b']) for y in YEARS for h in [12,24] for a in specs[str(y)]}
    np.savez_compressed(out/'mapped_horizon_risks_PRIVATE.npz',**{f'{y}_{h}_{a}':p for (y,h,a),p in risks.items()})
    rows=[];intervals=[];rank_checks=[];scenario_rows=[]
    measures=['observed_expected','abs_log_oe','brier','mean_minus_observed','within_year_auc','pooled_auc','ap','precision','coverage','captured']
    for h in [12,24]:
        for scope in [2023,2024,2025,'pooled']:
            ys=([2023,2024,2025] if h==12 else [2023,2024]) if scope=='pooled' else [scope]
            head=dict(scope=str(scope),months=h,years_included=','.join(map(str,ys)))
            if scope==2025 and h==24:
                for a in ARMS:
                    for k in [1,2,0]: rows.append(dict(**head,arm=a,cause=k,supported=False,status='NA_NO_24_MONTH_FOLLOWUP',**{v:np.nan for v in measures+['mean_predicted','observed','joint_brier']}))
                scenario_rows.append(dict(year=2025,months=24,supported=False,observed=np.nan,low_mean=risks[2025,24,'recovery_only'][:,0].mean(),middle_mean=risks[2025,24,'both_2y'][:,0].mean(),high_mean=risks[2025,24,'scenario_high'][:,0].mean(),observed_inside=np.nan))
                continue
            ids=np.concatenate([labs[y]['sample_ids'] for y in ys]); years=np.concatenate([np.full(len(labs[y]['X']),y) for y in ys])
            contexts={k:[evaluation_context(labs[y],h,k) for y in ys] for k in [1,2]}
            assert all(s and known.all() and (w==1).all() for cs in contexts.values() for _,w,known,s,_ in cs)
            outcomes=np.column_stack([np.concatenate([c[0] for c in contexts[k]]) for k in [1,2]])
            predictions={a:np.concatenate([risks[y,h,a] for y in ys]) for a in ARMS}
            offsets=np.r_[0,np.cumsum([len(labs[y]['X']) for y in ys])]
            ix_draws=np.concatenate([draws[str(y)]+offsets[i] for i,y in enumerate(ys)],axis=1)
            for k in [1,2]:
                y=outcomes[:,k-1];actual={}
                boot={a:{m:[] for m in measures} for a in ARMS}
                for a,pair in predictions.items():
                    p=pair[:,k-1];actual[a]=metrics(y,p,years,ids)
                    rows.append(dict(**head,arm=a,cause=k,supported=True,status='DIAGNOSTIC_UNCERTIFIED_SQUARED_ERROR_ONLY' if a=='uncertified_2y' else 'CERTIFIED',companies=len(y),events=int(y.sum()),**actual[a]))
                    for yr in ys:
                        sel=years==yr;reference=predictions['reference'][sel,k-1]
                        # Exact rank and tie-group equality; independent AUC check.
                        np.testing.assert_array_equal(np.argsort(p[sel],kind='stable'),np.argsort(reference,kind='stable'))
                        np.testing.assert_array_equal(np.diff(np.sort(p[sel]))==0,np.diff(np.sort(reference))==0)
                        m=metrics(y[sel],p[sel],years[sel],ids[sel]);r=metrics(y[sel],reference,years[sel],ids[sel])
                        for name in ['pooled_auc','ap','captured','precision','coverage']:
                            np.testing.assert_allclose(m[name],r[name],rtol=0,atol=0,equal_nan=True)
                        independent=roc_auc_score(y[sel],p[sel]) if len(np.unique(y[sel]))==2 else np.nan
                        counted,_,_=paired_auc(y[sel],p[sel]);np.testing.assert_allclose([m['pooled_auc'],counted],[independent,independent],atol=1e-14,rtol=0,equal_nan=True)
                        rank_checks.append(dict(scope=str(scope),year=yr,months=h,cause=k,arm=a,exact_ranking_match=True,sklearn_pair_agreement=True))
                    for ix in ix_draws:
                        bm=metrics(y[ix],p[ix],years[ix],ids[ix])
                        for name in measures:boot[a][name].append(bm[name])
                for a in ARMS:
                    np.testing.assert_allclose(actual[a]['within_year_auc'],actual['reference']['within_year_auc'],atol=0,rtol=0,equal_nan=True)
                    for name in measures:
                        b=np.array(boot[a][name]);ref=np.array(boot['reference'][name])
                        with np.errstate(invalid='ignore'):dif=b-ref;estimate=actual[a][name]-actual['reference'][name]
                        intervals.append(dict(**head,arm=a,cause=k,metric=name,comparison='arm_minus_reference',estimate=estimate,**interval(dif)))
                if scope!='pooled' and k==1:
                    low=float(risks[scope,h,'recovery_only'][:,0].mean());mid=float(risks[scope,h,'both_2y'][:,0].mean());high=float(risks[scope,h,'scenario_high'][:,0].mean());obs=float(y.mean())
                    scenario_rows.append(dict(year=scope,months=h,supported=True,observed=obs,low_mean=low,middle_mean=mid,high_mean=high,observed_inside=bool(low<=obs<=high),ordered=bool(low<=mid<=high),requested_g=specs[str(scope)]['uncertified_2y']['b'][0],certified_g=specs[str(scope)]['both_2y']['b'][0],required_oe=actual['reference']['observed_expected'],required_minus_requested=actual['reference']['observed_expected']-specs[str(scope)]['uncertified_2y']['b'][0]))
            # Joint two-cause squared-error sum. No event-free third term.
            loss={a:np.sum((p-outcomes)**2,axis=1) for a,p in predictions.items()}
            for a in ARMS:
                rows.append(dict(**head,arm=a,cause=0,supported=True,status='DIAGNOSTIC_UNCERTIFIED_SQUARED_ERROR_ONLY' if a=='uncertified_2y' else 'CERTIFIED',joint_brier=float(loss[a].mean()),companies=len(ids)))
                diff=loss[a]-loss['reference'];values=diff[ix_draws].mean(1)
                intervals.append(dict(**head,arm=a,cause=0,metric='joint_brier_sum_two_causes',comparison='arm_minus_reference',estimate=float(diff.mean()),**interval(values)))
    pd.DataFrame(rows).to_csv(out/'ARM_METRICS_PRIVATE.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'PAIRED_DIFFERENCES_95.csv',index=False)
    pd.DataFrame(scenario_rows).to_csv(out/'LEVEL_SCENARIOS.csv',index=False)
    pd.DataFrame(rank_checks).to_csv(out/'RANKING_CHECKS.csv',index=False)
    reader.finish();write_json(out/'input_hashes_PRIVATE.json',reader.hashes)
    print(pd.DataFrame(rows).query("cause==1 and scope!='pooled' and arm in ['reference','both_2y']")[["scope","months","arm","mean_predicted","observed_expected","brier"]].to_string(index=False))


if __name__=='__main__':
    try:main()
    except Exception:
        loc=json.loads((PUBLIC/'run_location.json').read_text());(Path(os.environ.get('CALENDAR_LEVEL_RUN',loc['private_run']))/'EVALUATION_FAILURE.txt').write_text(traceback.format_exc());raise
