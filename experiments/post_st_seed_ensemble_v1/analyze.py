"""Recompute performance from fixed equal-prediction means, without fitting."""
from itertools import product
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_requested_seeds_v1.analyze import fast_metrics,METRICS
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_reporting_drift_v1.metrics import paired_auc
from experiments.post_st_prognosis_v1.common import horizon_predictions
from .common import *
def main():
    r=run_path();root=r/'input';rows=[];validation=[];cost=[];ensemble={};risks={}
    for n,h in load(r/'audit/input_hashes.json').items():assert sha(root/n)==h
    for n,h in load(r/'audit/source_hashes.json').items():assert sha(Path(n))==h
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==load(r/'audit/current_pointer.json')['sha256']
    records=load(r/'audit/registry.json');assert len(records)==6
    for rec in records:
        p=Path(rec['path']);s=load(p/'summary.json')
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        ensemble[rec['year'],rec['arm']]=arrays(p/'representation/predictions_PRIVATE.npz')
        validation.append(dict(year=rec['year'],arm=rec['arm'],mean_error=s['full_mean_error'],**s['checks']))
        cost.append(dict(year=rec['year'],arm=rec['arm'],job=rec['job'],host=s['host'],seconds=s['seconds'],peak_rss_MiB=s['peak_rss_MiB'],forest_fits=0))
    labels={y:arrays(r/f'evaluation_input/{y}_test_PRIVATE.npz')for y in YEARS}
    for y in YEARS:ensemble[y,'reference96']=arrays(root/f'reference/{y}/predictions_PRIVATE.npz')
    for (y,a),z in ensemble.items():
        pred=average_predictions([arrays(root/f'members/{y}/{s}/predictions_PRIVATE.npz')for s in ARMS[a]])
        np.testing.assert_allclose(pred,z['cif'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(z['sample_ids'],labels[y]['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
        # Evaluate the arithmetic mean itself, not metric means; merged-tree
        # roundoff is independently checked and cannot change the saved inputs.
        for h in [12,24]:risks[y,a,h]=horizon_predictions(pred,TIMES,labels[y][f'h{h}'])
    for y,a,h,c in product(YEARS,ARMS,[12,24],[1,2]):
        p=risks[y,a,h][:,c-1];z=metrics(labels[y],p,h,c);yy,w,known,supported,_=evaluation_context(labels[y],h,c)
        if supported:
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            expected=[roc_auc_score(yy,p)if yy.any()and(~yy).any()else np.nan,average_precision_score(yy,p)if yy.any()else np.nan,np.mean((p-yy)**2)]
            np.testing.assert_allclose([z[k]for k in ['auc','ap','brier']],expected,atol=1e-13,rtol=0,equal_nan=True)
            np.testing.assert_allclose(z['auc'],paired_auc(yy,p)[0],atol=1e-13,rtol=0,equal_nan=True)
        rows.append(dict(**z,scope=str(y),arm=a,trees=32*len(ARMS[a]),months=h,cause=c,within_year_auc=z['auc'],equal_year_auc=z['auc'],
            observed_expected=z['observed']/z['risk_mean']if z['risk_mean']>0 else np.nan,calibration_gap=z['risk_mean']-z['observed']))
    for a,h,c in product(ARMS,[12,24],[1,2]):
        years=YEARS if h==12 else[2023,2024];p=np.concatenate([risks[y,a,h][:,c-1]for y in years])
        yy=np.concatenate([evaluation_context(labels[y],h,c)[0]for y in years]);yrs=np.concatenate([np.full(len(labels[y]['X']),y)for y in years]);ids=np.concatenate([labels[y]['sample_ids']for y in years])
        z=fast_metrics(yy,p,yrs,ids)
        np.testing.assert_allclose(z[:3],[roc_auc_score(yy,p),average_precision_score(yy,p),np.mean((p-yy)**2)],atol=1e-13,rtol=0)
        rows.append(dict(scope='pooled',arm=a,trees=32*len(ARMS[a]),months=h,cause=c,years_included=','.join(map(str,years)),companies=len(p),target_events=int(yy.sum()),unknown=0,supported=True,reason='',
            observed=float(yy.mean()),risk_mean=z[3],auc=z[0],ap=z[1],brier=z[2],within_year_auc=z[8],equal_year_auc=z[9],captured=int(z[6]),coverage=z[7],checked=sum(int(np.ceil(.2*len(labels[y]['X'])))for y in years),calibration_gap=z[4],observed_expected=float(yy.mean()/p.mean())))
    df=pd.DataFrame(rows);old=pd.read_csv(REPO/'experiments/post_st_requested_seeds_v1/METRICS_ALL.csv')
    old=old[(old.seed.astype(str)=='reference96')&(old.version=='shrunk')]
    for x in old.to_dict('records'):
        rr=df[(df.scope==str(x['scope']))&(df.arm=='reference96')&(df.months==x['months'])&(df.cause==x['cause'])];assert len(rr)==1
        for k in ['auc','ap','brier','risk_mean','captured','coverage']:np.testing.assert_allclose(rr.iloc[0][k],x[k],atol=1e-13,rtol=0,equal_nan=True)
    df.to_csv(r/'reports/ENSEMBLE_METRICS.csv',index=False)
    pd.DataFrame(validation).to_csv(r/'reports/FUNCTION_VALIDATION.csv',index=False);pd.DataFrame(cost).to_csv(r/'reports/COMPUTE_COST.csv',index=False)
    np.savez_compressed(r/'private/ensemble_risks_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risks.items()})
    draws=arrays(r/'private/bootstrap_indices_PRIVATE.npz');rng=np.random.default_rng(ANALYSIS_SEED)
    for y in YEARS:np.testing.assert_array_equal(draws[str(y)],rng.integers(len(labels[y]['X']),size=(1000,len(labels[y]['X']))))
    intervals=[];arms=list(ARMS)
    for h,c,scope in product([12,24],[1,2],[*YEARS,'pooled']):
        if h==24 and scope==2025:continue
        years=(YEARS if h==12 else[2023,2024])if scope=='pooled'else[scope]
        yy=np.concatenate([evaluation_context(labels[y],h,c)[0]for y in years]);yrs=np.concatenate([np.full(len(labels[y]['X']),y)for y in years]);ids=np.concatenate([labels[y]['sample_ids']for y in years])
        pp=np.array([np.concatenate([risks[y,a,h][:,c-1]for y in years])for a in arms]);actual=np.array([fast_metrics(yy,p,yrs,ids)for p in pp]);values=np.empty((1000,len(arms),len(METRICS)))
        for b in range(1000):
            ix=np.concatenate([draws[str(y)][b]+sum(len(labels[z]['X'])for z in years[:i])for i,y in enumerate(years)])
            values[b]=[fast_metrics(yy[ix],p[ix],yrs[ix],ids[ix])for p in pp]
        for i,a in enumerate(arms[1:],1):
            diff=values[:,i]-values[:,0];point=actual[i]-actual[0]
            for j,m in enumerate(METRICS):
                z=diff[:,j];ok=np.isfinite(z);ci=np.quantile(z[ok],[.025,.975])if ok.any()else[np.nan,np.nan]
                intervals.append(dict(scope=scope,arm=a,reference='reference96',months=h,cause=c,metric=m,estimate=point[j],lower95=ci[0],upper95=ci[1],repeats=1000,valid_repeats=int(ok.sum()),undefined_repeats=int((~ok).sum()),analysis_seed=ANALYSIS_SEED))
    pd.DataFrame(intervals).to_csv(r/'reports/PAIRED_DIFFERENCES_95.csv',index=False)
    write(r/'reports/validation.json',dict(status='PASS',forest_fits=0,calibration_fits=0,derived_ensembles=6,reference96_reproduced=True,
        source_hashes_unchanged=True,all_rows_all_native_times_both_causes=True,unsupported2025H24_not_scored=True,
        max_reconstruction=max(z['full_reconstruction']for z in validation),max_explicit_sum=max(z['explicit_component_sum']for z in validation),
        max_mean_error=max(z['mean_error']for z in validation),max_probability_sum=max(z['probability_sum_max']for z in validation)))
    print(df[(df.cause==1)&(df.scope!='pooled')][['scope','months','arm','auc','ap','brier','captured','target_events']].to_string(index=False))
if __name__=='__main__':main()
