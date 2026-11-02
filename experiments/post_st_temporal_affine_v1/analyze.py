from itertools import product
import argparse,time
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_ablation_v1.uncertainty import fast_metrics,METRICS
from experiments.post_st_level_coverage_v1.affine import transform
from .common import *

CONTRASTS={'independent_minus_baseline':{'independent':1,'baseline':-1},
           'temporal_minus_baseline':{'temporal':1,'baseline':-1},'temporal_minus_independent':{'temporal':1,'independent':-1}}
def save(name,rows):pd.DataFrame(rows).to_csv(RUN/f'reports/{name}.csv',index=False)

def point_outputs():
    paths={};cal={};checks=[]
    for r in load(RUN/'audit/registry_point.json'):
        p=Path(r['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        for c in load(p/'calibrators_PRIVATE.json'):cal[(c['year'],c['months'],c['arm'])]=c
        checks.extend(load(p/'numerical_checks.json'))
        for y,a in product(YEARS,ARMS):paths[(y,r['months'],a)]=p/f'{y}_{a}'
    return paths,cal,checks

def evaluate():
    block_forest_fits();start=time.perf_counter();paths,cal,checks=point_outputs()
    labels={y:arrays(RUN/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS};risk={};rows=[];bins=[];seed_rows=[];same=[]
    for (y,h,a),p in paths.items():
        z=arrays(p/'eval_predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],labels[y]['sample_ids'])
        risk[(y,h,a)]=horizon_predictions(z['cif'],TIMES,labels[y][f'h{h}'])
        for c in [1,2]:
            pp=risk[(y,h,a)][:,c-1];m=metrics(labels[y],pp,h,c);yy,w,known,supported,_=evaluation_context(labels[y],h,c)
            if supported:
                assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
                expected=[roc_auc_score(yy,pp) if yy.any() and (~yy).any() else np.nan,average_precision_score(yy,pp) if yy.any() else np.nan,np.mean((pp-yy)**2)]
                np.testing.assert_allclose([m[k] for k in ['auc','ap','brier']],expected,atol=1e-13,rtol=0,equal_nan=True)
                for b,ix in enumerate(np.array_split(np.lexsort((labels[y]['sample_ids'],pp)),4)):
                    bins.append(dict(year=y,months=h,arm=a,cause=c,bin=b+1,companies=len(ix),risk_mean=float(pp[ix].mean()),observed=float(yy[ix].mean()),events=int(yy[ix].sum())))
            m.update(scope=str(y),months=h,arm=a,cause=c,years_included=str(y),calibration_gap=m['risk_mean']-m['observed'],absolute_calibration_gap=abs(m['risk_mean']-m['observed']));rows.append(m)
    for h,a,c in product([12,24],ARMS,[1,2]):
        ys=YEARS if h==12 else [2023,2024];pp=np.concatenate([risk[(y,h,a)][:,c-1] for y in ys]);yy=np.concatenate([evaluation_context(labels[y],h,c)[0] for y in ys])
        year=np.concatenate([np.full(len(labels[y]['X']),y) for y in ys]);ids=np.concatenate([labels[y]['sample_ids'] for y in ys]);v=fast_metrics(yy,pp,year,ids)
        np.testing.assert_allclose(v[:3],[roc_auc_score(yy,pp),average_precision_score(yy,pp),np.mean((pp-yy)**2)],atol=1e-13,rtol=0)
        rows.append(dict(scope='pooled',months=h,arm=a,cause=c,years_included=','.join(map(str,ys)),companies=len(pp),target_events=int(yy.sum()),
            unknown=0,supported=True,risk_mean=v[3],observed=float(yy.mean()),auc=v[0],ap=v[1],brier=v[2],calibration_gap=v[4],absolute_calibration_gap=v[5],captured=int(v[6]),coverage=v[7],checked=sum(int(np.ceil(.2*len(labels[y]['X']))) for y in ys),reason=''))
    df=pd.DataFrame(rows)
    for y,h,c in product(YEARS,[12,24],[1,2]):
        d=df[(df.scope==str(y))&(df.months==h)&(df.cause==c)].set_index('arm')
        for a in ['independent','temporal']:
            fields=['auc','ap','captured'];np.testing.assert_allclose(d.loc[a,fields].to_numpy(float),d.loc['baseline',fields].to_numpy(float),atol=1e-13,rtol=0,equal_nan=True)
            pp=risk[(y,h,a)][:,c-1];base=risk[(y,h,'baseline')][:,c-1];ids=labels[y]['sample_ids']
            assert np.array_equal(np.lexsort((ids,-pp)),np.lexsort((ids,-base)))
            same.append(dict(year=y,months=h,cause=c,arm=a,entire_ranking_and_top20_identical=True))
    for y,s,h,c in product(YEARS,SEEDS,[12,24],[1,2]):
        z=arrays(RUN/f'input/reference/{y}/{s}/predictions_PRIVATE.npz');p=horizon_predictions(z['cif'],TIMES,labels[y][f'h{h}'])[:,c-1]
        seed_rows.append(dict(year=y,seed=s,months=h,cause=c,**metrics(labels[y],p,h,c)))
    save('METRICS',rows);save('BASELINE_SEED_METRICS',seed_rows);save('RANK_PRESERVATION',same);save('CALIBRATION_QUARTILES',bins)
    differences=[];fields=['auc','ap','brier','risk_mean','captured','coverage','calibration_gap','absolute_calibration_gap']
    for (scope,h,c),g in df.groupby(['scope','months','cause']):
        g=g.set_index('arm')
        for name,w in CONTRASTS.items():differences.append(dict(scope=scope,months=h,cause=c,contrast=name,**{k:sum(v*g.loc[a,k] for a,v in w.items()) for k in fields}))
    save('PAIRED_DIFFERENCES',differences)
    # Private numerical predictions, no model changes.
    np.savez_compressed(RUN/'private/endpoint_risks_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risk.items()})
    rng=np.random.default_rng(ANALYSIS_SEED);draws={y:rng.integers(len(labels[y]['X']),size=(1000,len(labels[y]['X']))) for y in YEARS}
    np.savez_compressed(RUN/'private/test_bootstrap_indices_PRIVATE.npz',**{str(y):v for y,v in draws.items()});intervals=[]
    for scope,h,c in product([*YEARS,'pooled'],[12,24],[1,2]):
        if scope==2025 and h==24:continue
        ys=([*YEARS] if h==12 else [2023,2024]) if scope=='pooled' else [scope]
        yy=np.concatenate([evaluation_context(labels[y],h,c)[0] for y in ys]);year=np.concatenate([np.full(len(labels[y]['X']),y) for y in ys]);ids=np.concatenate([labels[y]['sample_ids'] for y in ys])
        preds={a:np.concatenate([risk[(y,h,a)][:,c-1] for y in ys]) for a in ARMS};actual={a:fast_metrics(yy,p,year,ids) for a,p in preds.items()};values=np.empty((1000,3,len(METRICS)))
        for rep in range(1000):
            ix=np.concatenate([draws[y][rep]+sum(len(labels[v]['X']) for v in ys[:i]) for i,y in enumerate(ys)])
            z={a:fast_metrics(yy[ix],p[ix],year[ix],ids[ix]) for a,p in preds.items()}
            values[rep]=[sum(w*z[a] for a,w in weights.items()) for weights in CONTRASTS.values()]
        for i,(name,weights) in enumerate(CONTRASTS.items()):
            estimate=sum(w*actual[a] for a,w in weights.items())
            for j,m in enumerate(METRICS):
                v=values[:,i,j];valid=np.isfinite(v);bounds=np.quantile(v[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
                intervals.append(dict(scope=scope,months=h,cause=c,contrast=name,metric=m,estimate=estimate[j],lower95=bounds[0],upper95=bounds[1],repeats=1000,valid_repeats=int(valid.sum()),undefined_repeats=int((~valid).sum())))
        np.savez_compressed(RUN/f'private/test_bootstrap_{scope}_{h}_{c}_PRIVATE.npz',values=values)
    save('PAIRED_BOOTSTRAP_95',intervals)
    # Independent historical control must reproduce the previous experiment.
    legacy=[]
    for folder in (OLD/'results/formal').glob('*/task-*/output'):
        s=load(folder/'summary.json')
        if s.get('mode')!='calibrate':continue
        y,h=s['year'],s['horizon'];z=arrays(folder/'calibrated/predictions_PRIVATE.npz');current=arrays(paths[(y,h,'independent')]/'eval_predictions_PRIVATE.npz')
        err=float(abs(z['cif']-current['cif']).max());assert err<=1e-8
        legacy.append(dict(year=y,months=h,source=str(folder),max_prediction_error=err))
    assert len(legacy)==4;save('LEGACY_CALIBRATION_REPRODUCTION',legacy)
    # These are distinct horizon-targeted maps; audit the pair of reported endpoints.
    horizon_checks=[]
    for y,a in product(YEARS,ARMS):
        for role in ['train','eval']:
            d=arrays(RUN/f'input/folds/{y}/{role}_PRIVATE.npz')
            if role=='eval':horizons=labels[y]
            else:horizons=d
            r={h:horizon_predictions(arrays(paths[(y,h,a)]/f'{role}_predictions_PRIVATE.npz')['cif'],TIMES,horizons[f'h{h}']) for h in [12,24]}
            for c in [0,1]:
                delta=r[12][:,c]-r[24][:,c]
                horizon_checks.append(dict(year=y,arm=a,role=role,cause=c+1,companies=len(delta),violations=int((delta>1e-10).sum()),max_H12_above_H24=float(max(0,delta.max()))))
    save('CROSS_HORIZON_CHECKS',horizon_checks);save('NUMERICAL_COMPLEXITY',checks)
    write(RUN/'reports/metric_validation.json',dict(status='PASS',rows=len(df),unit_IPCW_on_supported_endpoints=True,pooled_recomputed=True,rank_checks=len(same),legacy_controls=len(legacy),
        legacy_max_error=max(x['max_prediction_error'] for x in legacy),test_bootstrap_repeats=1000,analysis_seed=ANALYSIS_SEED,
        test_bootstrap_indices_sha256=sha(RUN/'private/test_bootstrap_indices_PRIVATE.npz'),cross_horizon_violations=sum(x['violations'] for x in horizon_checks),seconds=time.perf_counter()-start,new_forests=0))
    print(df[df.cause==1][['scope','months','arm','auc','ap','brier','risk_mean','observed','captured']].round(6).to_string(index=False),flush=True)

def stability():
    block_forest_fits();paths,point,checks=point_outputs();tasks=[];records={};diagnostics=[];hashes=[]
    for stage in ['point','bootstrap']:
        for r in load(RUN/f'audit/registry_{stage}.json'):
            p=Path(r['path']);s=load(p/'summary.json');assert s['status']=='COMPLETE' and s['new_forests']==0;tasks.append(s)
            for name,h in load(p/'hashes.json').items():assert sha(p/name)==h
            for cal in load(p/'calibrators_PRIVATE.json'):
                k=(cal['year'],cal['months'],cal['arm'],cal['repeat']);records[k]=cal
                diagnostics.append({k:cal[k] for k in ['year','months','arm','repeat','companies','unique_companies','gap','primal_violation','iterations','optimizer_success','objective_before','objective_after','data_loss','identity_penalty','temporal_penalty','fit_seconds','certificate_vertices','maximum_terminal_sum_bound']}|
                    dict(unique_target_events=cal['unique_event_counts'][0],unique_recovery_events=cal['unique_event_counts'][1],target_slope_fixed=cal['frozen_slope_for_sparse_cause'][0],recovery_slope_fixed=cal['frozen_slope_for_sparse_cause'][1]))
                file=p/f'{cal["year"]}_{cal["arm"]}_calibrator_PRIVATE.json';hashes.append(dict(year=cal['year'],months=cal['months'],arm=cal['arm'],repeat=cal['repeat'],calibrator_sha256=sha(file)))
    assert len(records)==918 and sum(s['new_calibrators'] for s in tasks)==918
    spreads=[];metrics_rows=[];step=[];samples={}
    for y,h,a in product(YEARS,[12,24],['independent','temporal']):
        cc=[records[(y,h,a,r)] for r in range(50)];theta=np.array([np.r_[c['c'],c['b']] for c in cc])
        lab=arrays(RUN/f'evaluation_input/{y}_test_PRIVATE.npz');base=arrays(paths[(y,h,'baseline')]/'eval_predictions_PRIVATE.npz')['cif'];rs=[]
        for r,cal in enumerate(cc):
            curve=transform(base,TIMES,cal);pp=horizon_predictions(curve,TIMES,lab[f'h{h}']);rs.append(pp)
            for cause in [1,2]:metrics_rows.append(dict(year=y,months=h,arm=a,repeat=r,cause=cause,**metrics(lab,pp[:,cause-1],h,cause)))
        rs=np.array(rs);samples[(y,h,a)]=rs
        for c in [0,1]:
            spreads.append(dict(year=y,months=h,arm=a,cause=c+1,replicates=50,offset_sd=float(theta[:,c].std(ddof=1)),slope_sd=float(theta[:,c+2].std(ddof=1)),
                slope_fixed_replicates=sum(z['frozen_slope_for_sparse_cause'][c] for z in cc),prediction_sd_mean=float(rs[:,:,c].std(0,ddof=1).mean()),
                risk_mean_sd=float(rs[:,:,c].mean(1).std(ddof=1)),max_prediction_sd=float(rs[:,:,c].std(0,ddof=1).max())))
    # Adjacent parameter changes are diagnostics, not calibration error or independent periods.
    for h,a,rep in product([12,24],['independent','temporal'],[-1,*range(50)]):
        for y in periods(h)[1:]:
            c=records[(y,h,a,rep)];p=records[(y-1,h,a,rep)];dc=np.asarray(c['c'])-p['c'];db=np.asarray(c['b'])-p['b']
            step.append(dict(year=y,months=h,arm=a,repeat=rep,offset_step_l2=float(np.linalg.norm(dc)),slope_step_l2=float(np.linalg.norm(db))))
    sd=pd.DataFrame(spreads);paired=[]
    for y,h,c in product(YEARS,[12,24],[1,2]):
        ix=sd[(sd.year==y)&(sd.months==h)&(sd.cause==c)].set_index('arm')
        paired.append(dict(year=y,months=h,cause=c,**{k:float(ix.loc['temporal',k]-ix.loc['independent',k]) for k in ['offset_sd','slope_sd','prediction_sd_mean','risk_mean_sd']}))
    for name,rows in [('CALIBRATOR_DIAGNOSTICS',diagnostics),('CALIBRATOR_HASHES',hashes),('COMPUTE_COST',tasks),('CALIBRATION_ESTIMATE_STABILITY',spreads),('STABILITY_DIFFERENCES',paired),('CALIBRATION_RESAMPLE_METRICS',metrics_rows),('PARAMETER_STEP_CHANGES',step)]:save(name,rows)
    np.savez_compressed(RUN/'private/calibration_bootstrap_predictions_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in samples.items()})
    function_hashes=[]
    for (y,h,a),p in paths.items():function_hashes.append(dict(year=y,months=h,arm=a,function_sha256=sha(p/'functions_PRIVATE.npz'),fixed_ensemble_forest_sha256=sha(p/'fixed_forest_PRIVATE.json'),prediction_sha256=sha(p/'eval_predictions_PRIVATE.npz')))
    save('FUNCTION_HASHES',function_hashes)
    d=pd.DataFrame(diagnostics);n=pd.DataFrame(checks)
    write(RUN/'reports/numerical_validation.json',dict(status='PASS',new_forests=0,point_calibrators=18,resampled_calibrators=900,complete_chain_repetitions_per_horizon=50,
        max_gap=float(abs(d.gap).max()),max_primal_violation=float(d.primal_violation.max()),optimizer_unsuccessful_but_certified=int((~d.optimizer_success).sum()),
        complete_predictors_checked=len(n),max_full_reconstruction=float(n.reconstruction.max()),max_component_sum=float(n.explicit_component_sum.max()),max_function_reload=float(n.function_reload.max()),max_forest_reload=float(n.forest_reload.max()),
        max_path_variables=int(n.max_path_variables.max()),max_probability_sum=float(n.sum_max.max()),min_probability=float(n.probability_min.min()),minimum_time_increment=float(n.minimum_increment.min()),
        all_train_eval_rows_and_native_shape_times=True,all_input_probability_certificates=True,clipping=False,seed_specific_calibrators=False))
    print(sd.round(6).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metrics','stability']);a=p.parse_args();(evaluate if a.stage=='metrics' else stability)()
