"""Fixed saved-prediction evaluation; never select or ensemble the twenty seeds."""
from itertools import product,combinations
import argparse,time
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_ablation_v1.uncertainty import fast_metrics as original_metrics,METRICS as ORIGINAL_METRICS
from experiments.post_st_reporting_drift_v1.metrics import stratified_metrics,paired_auc
from experiments.post_st_prognosis_v1.common import horizon_predictions
from .common import *

ARMS=[str(s)for s in SEEDS]+['reference96']
METRICS=[*ORIGINAL_METRICS,'within_year_auc','equal_year_auc']
def fast_metrics(y,p,year,ids):
    z=original_metrics(y,p,year,ids);s=stratified_metrics(y,p,year,ids)
    return np.r_[z,s['within_year_auc'],s['equal_year_auc']]
def save(name,rows):pd.DataFrame(rows).to_csv(run_path()/'reports'/f'{name}.csv',index=False)

def read():
    r=run_path();records=load(r/'audit/registry_pilot.json')+load(r/'audit/registry_formal.json');paths={};pred={}
    assert len(records)==60 and len({(x['year'],x['seed'])for x in records})==60
    for rec in records:
        p=Path(rec['path']);s=load(p/'summary.json');assert s['status']=='COMPLETE'and s['trees']==32
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        for v in ['raw','shrunk']:paths[rec['year'],str(rec['seed']),v]=p/v
    for rec in load(r/'input/old_ensemble_registry.json'):
        p=Path(rec['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        paths[rec['year'],'reference96',rec['version']]=p/'representation'
    assert len(paths)==126
    for key,p in paths.items():
        z=arrays(p/'predictions_PRIVATE.npz');y,a,v=key;ev=arrays(r/f'input/folds/{y}/eval_PRIVATE.npz')
        np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
        pred[key]=z['cif']
    for y,v in product(YEARS,['raw','shrunk']):
        np.testing.assert_allclose(pred[y,'reference96',v],np.mean([pred[y,str(s),v]for s in REUSED_SEEDS],axis=0),atol=1e-12,rtol=0)
    return records,paths,pred

def evaluate():
    start=time.perf_counter();r=run_path();records,paths,pred=read()
    labels={y:arrays(r/f'evaluation_input/{y}_test_PRIVATE.npz')for y in YEARS};risks={};rows=[];checks=[]
    for y,a,v in pred:
        for h in [12,24]:risks[y,a,v,h]=horizon_predictions(pred[y,a,v],TIMES,labels[y][f'h{h}'])
    for y,a,v,h,c in product(YEARS,ARMS,['raw','shrunk'],[12,24],[1,2]):
        p=risks[y,a,v,h][:,c-1];z=metrics(labels[y],p,h,c);yy,w,known,supported,_=evaluation_context(labels[y],h,c)
        if supported:
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            expected=[roc_auc_score(yy,p)if yy.any()and(~yy).any()else np.nan,average_precision_score(yy,p)if yy.any()else np.nan,np.mean((p-yy)**2)]
            np.testing.assert_allclose([z[k]for k in ['auc','ap','brier']],expected,rtol=0,atol=1e-13,equal_nan=True)
            np.testing.assert_allclose(z['auc'],paired_auc(yy,p)[0],rtol=0,atol=1e-13,equal_nan=True)
            checks.append(dict(year=y,seed=a,version=v,months=h,cause=c,unit_IPCW=True,independent_metrics_match=True))
        rows.append(dict(**z,scope=str(y),seed=a,trees=96 if a=='reference96'else 32,version=v,months=h,cause=c,
            within_year_auc=z['auc'],equal_year_auc=z['auc'],observed_expected=z['observed']/z['risk_mean']if z['risk_mean']>0 else np.nan,
            calibration_gap=z['risk_mean']-z['observed'],absolute_calibration_gap=abs(z['risk_mean']-z['observed'])))
    for a,v,h,c in product(ARMS,['raw','shrunk'],[12,24],[1,2]):
        years=YEARS if h==12 else [2023,2024]
        p=np.concatenate([risks[y,a,v,h][:,c-1]for y in years]);yy=np.concatenate([evaluation_context(labels[y],h,c)[0]for y in years])
        yrs=np.concatenate([np.full(len(labels[y]['X']),y)for y in years]);ids=np.concatenate([labels[y]['sample_ids']for y in years]);z=fast_metrics(yy,p,yrs,ids)
        np.testing.assert_allclose(z[:3],[roc_auc_score(yy,p),average_precision_score(yy,p),np.mean((yy-p)**2)],atol=1e-13,rtol=0)
        rows.append(dict(scope='pooled',seed=a,trees=96 if a=='reference96'else 32,version=v,months=h,cause=c,years_included=','.join(map(str,years)),
            companies=len(p),target_events=int(yy.sum()),unknown=0,supported=True,reason='',observed=float(yy.mean()),risk_mean=z[3],auc=z[0],within_year_auc=z[8],equal_year_auc=z[9],ap=z[1],brier=z[2],
            captured=int(z[6]),coverage=z[7],checked=sum(int(np.ceil(.2*len(labels[y]['X'])))for y in years),calibration_gap=z[4],absolute_calibration_gap=z[5],observed_expected=float(yy.mean()/p.mean())if p.mean()>0 else np.nan))
    df=pd.DataFrame(rows)
    old=pd.read_csv(REPO/'experiments/post_st_drop9_support_v1/METRICS_ALL.csv');old=old[old.arm.eq('drop9_support')]
    for x in old.to_dict('records'):
        a='reference96'if x['seed']=='ensemble'else str(x['seed']);rr=df[df.scope.eq(str(x['scope']))&df.seed.eq(a)&df.version.eq(x['version'])&df.months.eq(x['months'])&df.cause.eq(x['cause'])];assert len(rr)==1
        for k in ['auc','ap','brier','risk_mean','captured','coverage']:np.testing.assert_allclose(rr.iloc[0][k],x[k],atol=1e-13,rtol=0,equal_nan=True)
    save('METRICS_ALL',df);save('METRIC_INDEPENDENT_CHECKS',checks)
    for y in YEARS:save(f'YEAR_{y}_RESULTS',df[df.scope.eq(str(y))&df.version.eq('shrunk')])
    save('POOLED_RESULTS',df[df.scope.eq('pooled')&df.version.eq('shrunk')])
    stats=[]
    for (scope,v,h,c),g in df[~df.seed.eq('reference96')].groupby(['scope','version','months','cause'],sort=False):
        for m in ['auc','within_year_auc','ap','brier','captured','coverage','risk_mean','observed_expected']:
            z=g[m].to_numpy(float);ok=np.isfinite(z);q=np.quantile(z[ok],[.025,.25,.5,.75,.975])if ok.any()else[np.nan]*5
            stats.append(dict(scope=scope,version=v,months=h,cause=c,metric=m,n_valid=int(ok.sum()),n_undefined=int((~ok).sum()),
                mean=float(z[ok].mean())if ok.any()else np.nan,sd=float(z[ok].std(ddof=1))if ok.sum()>1 else np.nan,
                minimum=float(z[ok].min())if ok.any()else np.nan,maximum=float(z[ok].max())if ok.any()else np.nan,
                q025=q[0],q25=q[1],median=q[2],q75=q[3],q975=q[4],meaning='algorithmic_seed_distribution_not_confidence_interval'))
    save('SEED_METRIC_SUMMARY',stats)
    variation=[]
    for y,v,h,c in product(YEARS,['raw','shrunk'],[12,24],[1,2]):
        pp=np.array([risks[y,str(s),v,h][:,c-1]for s in SEEDS]);std=pp.std(0,ddof=1)
        variation.append(dict(year=y,version=v,months=h,cause=c,seeds=20,mean_company_sd=float(std.mean()),max_company_sd=float(std.max()),no_outcomes_used=True))
    save('PREDICTION_VARIABILITY',variation)
    np.savez_compressed(r/'private/risks_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risks.items()})
    draws=arrays(r/'private/bootstrap_indices_PRIVATE.npz');rng=np.random.default_rng(ANALYSIS_SEED)
    for y in YEARS:np.testing.assert_array_equal(draws[str(y)],rng.integers(len(labels[y]['X']),size=(1000,len(labels[y]['X']))))
    intervals=[]
    for h,c,scope in product([12,24],[1,2],[*YEARS,'pooled']):
        if h==24 and scope==2025:continue
        years=(YEARS if h==12 else [2023,2024])if scope=='pooled'else[scope]
        yy=np.concatenate([evaluation_context(labels[y],h,c)[0]for y in years]);yrs=np.concatenate([np.full(len(labels[y]['X']),y)for y in years]);ids=np.concatenate([labels[y]['sample_ids']for y in years])
        pp=np.array([np.concatenate([risks[y,a,'shrunk',h][:,c-1]for y in years])for a in ARMS]);actual=np.array([fast_metrics(yy,p,yrs,ids)for p in pp]);values=np.empty((1000,len(ARMS),len(METRICS)))
        for b in range(1000):
            ix=np.concatenate([draws[str(y)][b]+sum(len(labels[z]['X'])for z in years[:i])for i,y in enumerate(years)])
            values[b]=[fast_metrics(yy[ix],p[ix],yrs[ix],ids[ix])for p in pp]
        for i,a in enumerate(ARMS[:-1]):
            diff=values[:,i]-values[:,-1];point=actual[i]-actual[-1]
            for j,m in enumerate(METRICS):
                z=diff[:,j];ok=np.isfinite(z);ci=np.quantile(z[ok],[.025,.975])if ok.any()else[np.nan,np.nan]
                intervals.append(dict(scope=scope,seed=a,reference='reference96',months=h,cause=c,metric=m,estimate=point[j],lower95=ci[0],upper95=ci[1],repeats=1000,valid_repeats=int(ok.sum()),undefined_repeats=int((~ok).sum()),analysis_seed=ANALYSIS_SEED))
    save('PAIRED_DIFFERENCES_95',intervals)
    write(r/'reports/metric_validation.json',dict(status='PASS',reference96_and_original3seeds_reproduced=True,independent_checks=len(checks),pooled_recomputed=True,all_supported_IPCW_one=True,unsupported2025H24_not_scored=True,twenty_seed_ensemble_created=False,seconds=time.perf_counter()-start))

def functions():
    r=run_path();records,paths,pred=read();checks=[];hashes=[];cost=[];stable=[];support=[];coverage=[]
    for (y,a,v),p in paths.items():
        z=load(p/'verification.json');assert max(z[k]for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])<=1e-10
        assert z['max_path_variables']<=2 and z['probability_min']>=-1e-12 and z['probability_sum_max']<=1+1e-10 and z['minimum_increment']>=-1e-12
        checks.append(dict(year=y,seed=a,version=v,**z));hashes.append(dict(year=y,seed=a,version=v,forest_sha256=sha(p/'forest_PRIVATE.json'),function_sha256=sha(p/'functions_PRIVATE.npz'),prediction_sha256=sha(p/'predictions_PRIVATE.npz')))
    for rec in records:
        p=Path(rec['path']);z=load(p/'summary.json')
        cost.append(dict(**{k:v for k,v in rec.items()if k!='path'},**{k:z[k]for k in ['forest_fits','new_trees','reused','fit_seconds','shrink_seconds','replay_seconds','seconds','memory_slot_wait_seconds','peak_rss_MiB','actual_scored_candidates','guard_rejected_candidates']}))
        for label in ['feature_coverage','leaf_support']:
            dd=pd.read_csv(p/f'{label}_PRIVATE.csv');dd['year']=rec['year'];dd['seed']=rec['seed']
            if label=='feature_coverage':coverage.extend(dd.to_dict('records'))
            else:support.append(dict(year=rec['year'],seed=rec['seed'],leaves=len(dd),median_unique_companies=float(dd.unique_companies.median()),zero_target_leaves=int(dd.target_companies.eq(0).sum()),one_target_leaves=int(dd.target_companies.eq(1).sum()),borrowed_positive_target_leaves=int(dd.borrowed_positive_target.sum())))
    assert sum(z['forest_fits']for z in cost)==51 and sum(z['new_trees']for z in cost)==1632
    for y,v in product(YEARS,['raw','shrunk']):
        info={s:(load(paths[y,str(s),v]/'complexity_PRIVATE.json'),arrays(paths[y,str(s),v]/'components_PRIVATE.npz'))for s in SEEDS}
        for s1,s2 in combinations(SEEDS,2):
            c1,t1=info[s1];c2,t2=info[s2];i1={k:i for i,k in enumerate(t1['keys'])};i2={k:i for i,k in enumerate(t2['keys'])}
            for order in [1,2]:
                sets=[{tuple(z['features'])for z in c['functions']if z['order']==order and z['nonzero']}for c in [c1,c2]];jacc=len(sets[0]&sets[1])/len(sets[0]|sets[1])if sets[0]|sets[1]else np.nan
                keys=[k for k in sorted(i1.keys()&i2.keys())if len(k.split(','))==order]
                for hi,day in [(0,365),(2,730)]:
                    for cause in [1,2]:
                        corr=[];rms=[]
                        for k in keys:
                            l=t1['terms'][:,i1[k],hi,cause-1];rr=t2['terms'][:,i2[k],hi,cause-1];rms.append(float(np.sqrt(np.mean((l-rr)**2))))
                            if l.std()>1e-12 and rr.std()>1e-12:corr.append(float(np.corrcoef(l,rr)[0,1]))
                        stable.append(dict(year=y,version=v,seed_a=s1,seed_b=s2,day=day,cause=cause,order=order,jaccard=jacc,common_functions=len(keys),median_correlation=np.median(corr)if corr else np.nan,mean_rms_difference=np.mean(rms)if rms else np.nan))
    for name,z in [('NUMERICAL_COMPLEXITY',checks),('MODEL_HASHES',hashes),('COMPUTE_COST',cost),('FUNCTION_STABILITY',stable),('LEAF_SUPPORT',support),('FEATURE_COVERAGE',coverage)]:save(name,z)
    z=pd.DataFrame(checks);write(r/'reports/function_validation.json',dict(status='PASS',representations=len(z),forest_fits=51,new_trees=1632,max_full_reconstruction=float(z.full_reconstruction.max()),max_explicit_sum=float(z.explicit_component_sum.max()),max_path_variables=int(z.max_path_variables.max()),max_probability_sum=float(z.probability_sum_max.max()),all_rows_all_native_times_both_causes=True))

def audit():
    r=run_path();rows=[]
    for n,h in load(r/'audit/input_hashes.json').items():assert sha(r/'input'/n)==h
    for n,h in load(r/'audit/source_hashes.json').items():assert sha(Path(n))==h
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==load(r/'audit/current_pointer.json')['sha256']
    old=load(r/'input/old_registry.json')
    for y in YEARS:
        tr=arrays(r/f'input/folds/{y}/train_PRIVATE.npz');ev=arrays(r/f'input/folds/{y}/eval_PRIVATE.npz');labels=arrays(r/f'evaluation_input/{y}_test_PRIVATE.npz')
        np.testing.assert_array_equal(ev['sample_ids'],labels['sample_ids']);np.testing.assert_array_equal(ev['X'],labels['X'][:,load(SOURCE/'audit/keep_PRIVATE.json')]);assert not set(tr['groups'])&set(labels['groups'])
        for s in REUSED_SEEDS:
            original=Path(old[f'{y}|{s}'])
            for v in ['raw','shrunk']:assert sha(r/f'input/reused/{y}/{s}/{v}_PRIVATE.json')==sha(original/f'{v}/forest_PRIVATE.json')
            np.testing.assert_equal(pairing(tr['groups'],configuration(s,32),74),load(original/'pairing.json'))
        rows.append(dict(year=y,train_companies=len(tr['groups']),train_target_events=int((tr['event']==1).sum()),train_recovery_events=int((tr['event']==2).sum()),eval_companies=len(ev['X']),training_sha256=sha(r/f'input/folds/{y}/train_PRIVATE.npz'),disjoint_train_test=True,original3seeds_exact=True))
    save('DATA_AUDIT',rows);write(r/'reports/data_validation.json',dict(status='PASS',immutable_inputs_and_sources=True,original3seed_models_byte_identical=True,disjoint_train_test_groups=True,evaluation74_columns_match=True,current_pointer_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metrics','functions','audit']);a=p.parse_args();{'metrics':evaluate,'functions':functions,'audit':audit}[a.stage]()
