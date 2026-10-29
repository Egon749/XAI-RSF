"""Frozen-prediction, year-stratified paired company bootstrap; no model fit."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .common import *
from .analyze import contrasts,selected

METRICS=['auc','ap','brier','risk_mean','calibration_gap','absolute_calibration_gap','top20_captured','top20_coverage']

def fast_metrics(y,p,year,ids):
    """Exact tied-score binary metrics, for this verified complete-endpoint sample."""
    y=np.asarray(y,bool);p=np.asarray(p,float);pos=int(y.sum());neg=len(y)-pos
    order=np.argsort(p,kind='stable');scores=p[order];yy=y[order]
    starts=np.r_[0,np.flatnonzero(scores[1:]!=scores[:-1])+1]
    counts=np.diff(np.r_[starts,len(y)]);tp=np.add.reduceat(yy.astype(float),starts);fp=counts-tp
    auc=float(np.sum(tp*(np.cumsum(fp)-fp+fp/2))/(pos*neg)) if pos and neg else np.nan
    cp=np.cumsum(tp[::-1]);cn=np.cumsum(counts[::-1])
    ap=float(np.sum(tp[::-1]/pos*cp/cn)) if pos else np.nan
    chosen=selected(p,year,ids);captured=int(y[chosen].sum());gap=float(p.mean()-y.mean())
    return np.array([auc,ap,np.mean((p-y)**2),p.mean(),gap,abs(gap),captured,captured/pos if pos else np.nan])

def bootstrap(run):
    started=time.perf_counter();out=run/'evaluation';protocol=json.loads((run/'audit/protocol.json').read_text())['bootstrap']
    nrep=protocol['replicates'];rng=np.random.default_rng(protocol['analysis_seed'])
    d={str(y):arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz') for y in YEARS}
    assert all(v['label_eligible'].all() and np.all((v['event']>0)|(v['analysis_time']>=24)) for v in d.values())
    idx={str(y):rng.integers(len(d[str(y)]['X']),size=(nrep,len(d[str(y)]['X']))) for y in YEARS}
    index_path=run/'private/bootstrap_indices_PRIVATE.npz'
    if index_path.exists():
        saved=arrays(index_path)
        for k,v in idx.items():np.testing.assert_array_equal(saved[k],v)
    else:np.savez_compressed(index_path,**idx)
    forecasts=arrays(run/'private/evaluation_forecasts_PRIVATE.npz');weights=contrasts();arms=ARMS
    matrix=np.array([[v.get(a,0) for a in arms] for v in weights.values()],float)
    rows=[];rep_rows=[]
    for scope in ['2023','2024','pooled']:
        years=list(d) if scope=='pooled' else [scope]
        y=np.concatenate([(d[s]['event']==1)&(d[s]['analysis_time']<=24) for s in years])
        yy=np.concatenate([d[s]['year'] for s in years]);ids=np.concatenate([d[s]['sample_ids'] for s in years])
        p=np.array([forecasts[f'{scope}|{a}|ensemble|24'][:,0] for a in arms])
        # Check the optimized evaluator against the existing/sklearn definitions.
        estimate=np.array([fast_metrics(y,z,yy,ids) for z in p])
        for ai in range(len(arms)):
            np.testing.assert_allclose(estimate[ai,:2],[roc_auc_score(y,p[ai]),average_precision_score(y,p[ai])],atol=1e-14,rtol=0)
        actual=matrix@estimate;values=np.empty((nrep,len(weights),len(METRICS)))
        for b in range(nrep):
            ix=np.concatenate([idx[s][b]+sum(len(d[z]['X']) for z in years[:i]) for i,s in enumerate(years)])
            sample=np.array([fast_metrics(y[ix],z[ix],yy[ix],ids[ix]) for z in p])
            # Avoid zero*NaN spreading undefined metrics from unrelated arms.
            delta=np.array([sum(w*sample[arms.index(a)] for a,w in v.items()) for v in weights.values()])
            values[b]=delta
        for j,label in enumerate(weights):
            for k,name in enumerate(METRICS):
                v=values[:,j,k];ok=np.isfinite(v);bounds=np.quantile(v[ok],[.025,.975]) if ok.any() else [np.nan,np.nan]
                rows.append(dict(scope=scope,contrast=label,metric=name,estimate=float(actual[j,k]),lower95=float(bounds[0]),upper95=float(bounds[1]),replicates=nrep,valid_replicates=int(ok.sum()),undefined_replicates=int((~ok).sum()),analysis_seed=protocol['analysis_seed']))
        np.savez_compressed(run/'private'/f'bootstrap_differences_{scope}_PRIVATE.npz',values=values,contrasts=np.array(list(weights)),metrics=np.array(METRICS))
    df=pd.DataFrame(rows);df.to_csv(out/'paired_bootstrap_intervals.csv',index=False)
    write(out/'bootstrap_validation.json',dict(status='PASS',replicates=nrep,analysis_seed=protocol['analysis_seed'],indices_sha256=sha(run/'private/bootstrap_indices_PRIVATE.npz'),all_arms_common_indices=True,year_stratified=True,company_unit=True,models_fixed=True,forest_fits=0,seconds=time.perf_counter()-started,interval_rows=len(rows),max_undefined_replicates=int(df.undefined_replicates.max()),uncertainty='conditional test-sample variability; excludes training/model-selection/adaptive-development uncertainty'))
    print(df[(df.scope=='pooled')&df.contrast.isin([a+'_minus_reference' for a in NEW])&df.metric.isin(['auc','ap','brier','top20_coverage'])].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);bootstrap(p.parse_args().run)
