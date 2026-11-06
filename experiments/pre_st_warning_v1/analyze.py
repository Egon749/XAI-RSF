"""Evaluate frozen predictions; never fit a model or use results for selection."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import aalen_johansen,CIFExpansion
from experiments.post_st_prognosis_v1.common import arrays,calendar_elapsed,endpoint_weights,horizon_predictions
from .metrics import metrics,independent_weights
from .raw_scan import sha
from .stage import successful

TIMES=np.array([365.,366.,730.,731.])
SCOPES=['2023','2024','2025','pooled']

def censor_meta(run,year):
    d=pd.read_parquet(run/f'evaluation_input/{year}_eval_PRIVATE.parquet')
    d['analysis_time']=[calendar_elapsed(o,s) for o,s in zip(d.origin,d.stop)]
    return d

def evaluate(run):
    started=time.perf_counter();formal=successful(run,'formal',9);ensembles=successful(run,'ensemble',3)
    meta={y:censor_meta(run,y) for y in [2023,2024,2025]};forecast={};validation=[];cost=[]
    for root,s in formal+ensembles:
        y=s['year'];seed=str(s['seed'])
        for version in ['raw','shrunk']:
            p=root/version;z=arrays(p/'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'],meta[y].sample_id.to_numpy(str))
            assert np.array_equal(z['times'],TIMES)
            forecast[(y,seed,version)]=z['cif']
            check=json.loads((p/'verification.json').read_text())
            validation.append(dict(year=y,seed=seed,version=version,**check,model_sha256=sha(p/'forest_PRIVATE.json'),functions_sha256=sha(p/'functions_PRIVATE.npz')))
        cost.append({k:v for k,v in s.items() if k not in ['checks','raw_checks','independent_checks','shrinkage_check','source_members','environment']})
    rows=[];calibration=[];null=[];unavailable=[]
    for seed in ['17','503','22','ensemble']:
        for version in ['raw','shrunk']:
            for h in [12,24]:
                for scope in SCOPES:
                    years=([2023,2024,2025] if h==12 else [2023,2024]) if scope=='pooled' else [int(scope)]
                    d=pd.concat([meta[y] for y in years],ignore_index=True)
                    p=np.concatenate([horizon_predictions(forecast[(y,seed,version)],TIMES,meta[y][f'h{h}'].to_numpy()) for y in years])
                    if scope=='2025' and h==24:
                        for k in [1,2]:
                            row=dict(scope=scope,seed=seed,version=version,horizon_months=h,cause=k,supported=False,records=len(d),companies=d.group.nunique(),
                                risk_mean=float(p[:,k-1].mean()),risk_max=float(p[:,k-1].max()),note='2025 24-month horizon not mature; performance NA')
                            rows.append(row)
                        continue
                    for k in [1,2]:
                        z=metrics(d.analysis_time,d.event,p[:,k-1],d.year,d.sample_id,d.group,h,k)
                        z.update(scope=scope,seed=seed,version=version,horizon_months=h,cause=k,supported=True,note='EXPLORATORY_PROVISIONAL; known-endpoint AUC/AP; IPCW Brier; AJ incidence')
                        rows.append(z)
                        if seed=='ensemble' and version=='shrunk':
                            w,known,_=endpoint_weights(d.analysis_time,d.event,h);y=((d.event==k)&(d.analysis_time<=h)).to_numpy()
                            if y[known].any() and (~y[known]).any():
                                assert abs(z['auc']-roc_auc_score(y[known],p[known,k-1]))<1e-14
                                assert abs(z['ap']-average_precision_score(y[known],p[known,k-1]))<1e-14
                            np.testing.assert_allclose(w,independent_weights(d.analysis_time,d.event,h),rtol=0,atol=1e-13)
                            order=np.argsort(p[:,k-1],kind='stable')
                            for b,ix in enumerate(np.array_split(order,5),1):
                                tt=d.analysis_time.to_numpy()[ix];ee=d.event.to_numpy()[ix]
                                observed=float(aalen_johansen(tt,ee,[h])[0,k-1]);known_bin=((ee>0)&(tt<=h))|(tt>=h)
                                calibration.append(dict(scope=scope,horizon_months=h,cause=k,quintile=b,n=len(ix),
                                    mean_prediction=float(p[ix,k-1].mean()),observed_aj=observed,events=int(((ee==k)&(tt<=h)).sum()),unknown=int((~known_bin).sum())))
                    if seed=='ensemble' and version=='shrunk':
                        for k in [1,2]:
                            const=[]
                            for yr in years:
                                tr=pd.read_parquet(run/f'evaluation_input/{yr}_train_PRIVATE.parquet')
                                tt=np.array([calendar_elapsed(o,s) for o,s in zip(tr.origin,tr.stop)])
                                v=float(aalen_johansen(tt,tr.event,[h])[0,k-1]);const.extend([v]*len(meta[yr]))
                            z=metrics(d.analysis_time,d.event,np.array(const),d.year,d.sample_id,d.group,h,k)
                            null.append(dict(scope=scope,horizon_months=h,cause=k,**z))
    pd.DataFrame(rows).to_csv(run/'reports/METRICS_ALL.csv',index=False)
    pd.DataFrame(calibration).to_csv(run/'reports/CALIBRATION.csv',index=False)
    pd.DataFrame(null).to_csv(run/'reports/NULL_AJ_BASELINE.csv',index=False)
    pd.DataFrame(validation).to_csv(run/'reports/FUNCTION_VALIDATION.csv',index=False)
    pd.DataFrame(cost).to_csv(run/'reports/COMPUTE.csv',index=False)
    complexity_cols=['year','seed','version','trees','max_path_variables','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','function_bytes','zero_support_cell_fraction','low_support_cell_fraction']
    df=pd.DataFrame(validation);df[[c for c in complexity_cols if c in df]].to_csv(run/'reports/EXPLANATION_COMPLEXITY.csv',index=False)
    importance=[];schema=json.loads(Path(__file__).with_name('feature_schema.json').read_text())
    for root,s in ensembles:
        y=s['year'];ex=CIFExpansion.load(root/'shrunk/functions_PRIVATE.npz');X=arrays(run/f'prepared/folds/{y}/train_PRIVATE.npz')['X']
        for key in [(j,) for j in ex.mains]+ex.pairs:
            terms=ex.component(X,key,[365,730]);a=np.abs(terms).mean(axis=0)
            for h,ih in [(12,0),(24,1)]:
                for k in [1,2]:importance.append(dict(year=y,horizon_months=h,cause=k,order=len(key),features=' × '.join(schema['names'][j] for j in key),mean_absolute_cif_contribution=float(a[ih,k-1])))
    pd.DataFrame(importance).to_csv(run/'reports/FUNCTION_IMPORTANCE.csv',index=False)
    # Save only alignment keys and predictions privately; never export this file.
    np.savez_compressed(run/'private/evaluation_predictions_PRIVATE.npz',**{f'{y}_{s}_{v}':p for (y,s,v),p in forecast.items()})
    (run/'audit/evaluation_checks.json').write_text(json.dumps(dict(sklearn_auc_ap=True,independent_reverse_km=True,
        metadata_alignment=True,forecast_count=len(forecast),seconds=time.perf_counter()-started,forest_fits=0),indent=2))
    return meta,forecast

def bootstrap(run,meta,forecast,only_scope=None,only_horizon=None,out=None):
    out=run/'reports' if out is None else out;out.mkdir(parents=True,exist_ok=True)
    rows=[];arrays_out={};draw_hashes={}
    fields=['auc','ap','ipcw_brier','risk_mean','observed_aj','observed_expected','calibration_gap','top20_coverage','top25_coverage','within_year_auc']
    for h in [12,24]:
        for scope in SCOPES:
            if scope=='2025' and h==24:continue
            if only_scope is not None and scope!=only_scope:continue
            if only_horizon is not None and h!=only_horizon:continue
            rng=np.random.default_rng(np.random.SeedSequence([2026110501,h,0 if scope=='pooled' else int(scope)]))
            years=([2023,2024,2025] if h==12 else [2023,2024]) if scope=='pooled' else [int(scope)]
            d=pd.concat([meta[y] for y in years],ignore_index=True)
            p=np.concatenate([horizon_predictions(forecast[(y,'ensemble','shrunk')],TIMES,meta[y][f'h{h}'].to_numpy()) for y in years])
            groups,inverse=np.unique(d.group,return_inverse=True);blocks=[np.flatnonzero(inverse==i) for i in range(len(groups))]
            draws=rng.integers(len(groups),size=(1000,len(groups)),dtype=np.int32)
            file=out/f'bootstrap_{scope}_{h}_PRIVATE.npz';np.savez_compressed(file,groups=groups,draws=draws);draw_hashes[file.name]=sha(file)
            t,e,year,ids,g=[d[c].to_numpy() for c in ['analysis_time','event','year','sample_id','group']]
            actual=[metrics(t,e,p[:,k-1],year,ids,g,h,k) for k in [1,2]]
            values=np.empty((1000,2,len(fields)))
            for b in range(1000):
                ix=np.concatenate([blocks[i] for i in draws[b]])
                for ki,k in enumerate([1,2]):
                    z=metrics(t[ix],e[ix],p[ix,k-1],year[ix],ids[ix],g[ix],h,k)
                    values[b,ki]=[z[f] for f in fields]
            for ki,k in enumerate([1,2]):
                for j,f in enumerate(fields):
                    a=values[:,ki,j];valid=np.isfinite(a);lo,hi=np.quantile(a[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
                    rows.append(dict(scope=scope,horizon_months=h,cause=k,metric=f,estimate=actual[ki][f],lower95=lo,upper95=hi,
                        replicates=1000,undefined_replicates=int((~valid).sum()),conditional_on_fixed_predictions=True))
            arrays_out[f'{scope}_{h}']=values
            print('bootstrap_complete',scope,h,flush=True)
    pd.DataFrame(rows).to_csv(out/'BOOTSTRAP_95.csv',index=False)
    np.savez_compressed(out/'bootstrap_metrics_PRIVATE.npz',**arrays_out)
    (out/'bootstrap.json').write_text(json.dumps(dict(seed=2026110501,replicates=1000,unit='company',
        pooled_cross_year_company_dependence_preserved=True,fixed_models=True,draw_hashes=draw_hashes),indent=2))

def main(run):
    meta,forecast=evaluate(run);bootstrap(run,meta,forecast)
    d=pd.read_csv(run/'reports/METRICS_ALL.csv');print(d[(d.seed=='ensemble')&(d.version=='shrunk')&(d.cause==1)][['scope','horizon_months','auc','ap','ipcw_brier','top20_captured','target_records','risk_mean','observed_aj']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['all','evaluate','bootstrap'],default='all')
    p.add_argument('--scope',choices=SCOPES);p.add_argument('--horizon',type=int,choices=[12,24]);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.stage=='evaluate':evaluate(a.run)
    elif a.stage=='bootstrap':
        start=time.perf_counter()
        meta={y:censor_meta(a.run,y) for y in [2023,2024,2025]};saved=arrays(a.run/'private/evaluation_predictions_PRIVATE.npz')
        forecast={(int(k.split('_')[0]),k.split('_')[1],k.split('_')[2]):v for k,v in saved.items()}
        bootstrap(a.run,meta,forecast,a.scope,a.horizon,a.out)
        if a.out:
            (a.out/'summary.json').write_text(json.dumps(dict(status='COMPLETE',scope=a.scope,horizon=a.horizon,replicates=1000,forest_fits=0,seconds=time.perf_counter()-start),indent=2))
    else:main(a.run)
