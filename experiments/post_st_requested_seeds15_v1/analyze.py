"""Evaluate every fixed seed and equal-prediction ensemble; never select winners."""
from itertools import product,combinations
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_requested_seeds_v1.analyze import fast_metrics,METRICS
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_reporting_drift_v1.metrics import paired_auc
from experiments.post_st_prognosis_v1.common import horizon_predictions
from .common import *
ARMS=[*map(str,SEEDS),ENSEMBLE,'reference96']
def save(name,rows):pd.DataFrame(rows).to_csv(run_path()/'reports'/f'{name}.csv',index=False)
def read():
    r=run_path();reused=load(r/'input/reused_registry.json');fresh=load(r/'audit/registry_pilot.json')+load(r/'audit/registry_formal.json')
    records=[dict(**z,reused=True)for z in reused]+[dict(**z,reused=False)for z in fresh]
    assert len(records)==N_FITS+N_REUSED and {(z['year'],z['seed'])for z in records}==set(product(YEARS,SEEDS));paths={};pred={}
    for rec in records:
        p=Path(rec['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        for v in ['raw','shrunk']:paths[rec['year'],str(rec['seed']),v]=p/v
    for rec in load(r/'audit/registry_ensemble.json'):
        p=Path(rec['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        v=rec['version'];paths[rec['year'],ENSEMBLE,v]=p/v
    for rec in load(r/'input/old_ensemble_registry.json'):
        p=Path(rec['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        paths[rec['year'],'reference96',rec['version']]=p/'representation'
    assert len(paths)==len(YEARS)*len(ARMS)*2
    for key,p in paths.items():
        y,a,v=key;z=arrays(p/'predictions_PRIVATE.npz');ev=arrays(r/f'input/folds/{y}/eval_PRIVATE.npz')
        np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES);pred[key]=z['cif']
    for y,v in product(YEARS,['raw','shrunk']):
        mean=np.mean([pred[y,str(s),v]for s in SEEDS],axis=0)
        np.testing.assert_allclose(mean,pred[y,ENSEMBLE,v],atol=1e-12,rtol=0);pred[y,ENSEMBLE,v]=mean
    return records,paths,pred
def evaluate(pred):
    r=run_path();labels={y:arrays(r/f'evaluation_input/{y}_test_PRIVATE.npz')for y in YEARS};risks={};rows=[];checks=[]
    for (y,a,v),p in pred.items():
        for h in [12,24]:risks[y,a,v,h]=horizon_predictions(p,TIMES,labels[y][f'h{h}'])
    for y,a,v,h,c in product(YEARS,ARMS,['raw','shrunk'],[12,24],[1,2]):
        p=risks[y,a,v,h][:,c-1];z=metrics(labels[y],p,h,c);yy,w,known,supported,_=evaluation_context(labels[y],h,c)
        if supported:
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            expected=[roc_auc_score(yy,p)if yy.any()and(~yy).any()else np.nan,average_precision_score(yy,p)if yy.any()else np.nan,np.mean((p-yy)**2)]
            np.testing.assert_allclose([z[k]for k in ['auc','ap','brier']],expected,atol=1e-13,rtol=0,equal_nan=True)
            np.testing.assert_allclose(z['auc'],paired_auc(yy,p)[0],atol=1e-13,rtol=0,equal_nan=True)
            checks.append(dict(year=y,seed=a,version=v,months=h,cause=c,unit_IPCW=True,independent_match=True))
        rows.append(dict(**z,scope=str(y),seed=a,trees=ENSEMBLE_TREES if a==ENSEMBLE else 96 if a=='reference96'else 32,version=v,months=h,cause=c,
            within_year_auc=z['auc'],equal_year_auc=z['auc'],observed_expected=z['observed']/z['risk_mean']if z['risk_mean']>0 else np.nan,calibration_gap=z['risk_mean']-z['observed']))
    for a,v,h,c in product(ARMS,['raw','shrunk'],[12,24],[1,2]):
        years=YEARS if h==12 else[2023,2024];p=np.concatenate([risks[y,a,v,h][:,c-1]for y in years]);yy=np.concatenate([evaluation_context(labels[y],h,c)[0]for y in years])
        yrs=np.concatenate([np.full(len(labels[y]['X']),y)for y in years]);ids=np.concatenate([labels[y]['sample_ids']for y in years]);z=fast_metrics(yy,p,yrs,ids)
        np.testing.assert_allclose(z[:3],[roc_auc_score(yy,p),average_precision_score(yy,p),np.mean((yy-p)**2)],atol=1e-13,rtol=0)
        rows.append(dict(scope='pooled',seed=a,trees=ENSEMBLE_TREES if a==ENSEMBLE else 96 if a=='reference96'else 32,version=v,months=h,cause=c,years_included=','.join(map(str,years)),
            companies=len(p),target_events=int(yy.sum()),unknown=0,supported=True,reason='',observed=float(yy.mean()),risk_mean=z[3],auc=z[0],ap=z[1],brier=z[2],within_year_auc=z[8],equal_year_auc=z[9],
            captured=int(z[6]),coverage=z[7],checked=sum(int(np.ceil(.2*len(labels[y]['X'])))for y in years),calibration_gap=z[4],observed_expected=float(yy.mean()/p.mean())))
    df=pd.DataFrame(rows);old=pd.read_csv(REPO/'experiments/post_st_requested_seeds14_v1/METRICS_ALL.csv');old=old[old.seed.astype(str).isin([*map(str,REUSED_SEEDS),'reference96'])]
    for x in old.to_dict('records'):
        rr=df[(df.scope==str(x['scope']))&(df.seed==str(x['seed']))&(df.version==x['version'])&(df.months==x['months'])&(df.cause==x['cause'])];assert len(rr)==1
        for k in ['auc','ap','brier','risk_mean','captured','coverage']:np.testing.assert_allclose(rr.iloc[0][k],x[k],atol=1e-13,rtol=0,equal_nan=True)
    save('METRICS_ALL',df);save('METRIC_INDEPENDENT_CHECKS',checks)
    for y in YEARS:save(f'YEAR_{y}_RESULTS',df[(df.scope==str(y))&(df.version=='shrunk')])
    save('POOLED_RESULTS',df[(df.scope=='pooled')&(df.version=='shrunk')])
    np.savez_compressed(r/'private/risks_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risks.items()})
    draws=arrays(r/'private/bootstrap_indices_PRIVATE.npz');rng=np.random.default_rng(ANALYSIS_SEED)
    for y in YEARS:np.testing.assert_array_equal(draws[str(y)],rng.integers(len(labels[y]['X']),size=(1000,len(labels[y]['X']))))
    intervals=[]
    for h,c,scope in product([12,24],[1,2],[*YEARS,'pooled']):
        if h==24 and scope==2025:continue
        years=(YEARS if h==12 else[2023,2024])if scope=='pooled'else[scope]
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
    variability=[]
    for y,v,h,c in product(YEARS,['raw','shrunk'],[12,24],[1,2]):
        sd=np.array([risks[y,str(s),v,h][:,c-1]for s in SEEDS]).std(0,ddof=1)
        variability.append(dict(year=y,version=v,months=h,cause=c,seeds=len(SEEDS),mean_company_sd=float(sd.mean()),max_company_sd=float(sd.max()),interpretation='selected_seed_variability_not_unbiased_distribution'))
    save('PREDICTION_VARIABILITY',variability)
    write(r/'reports/metric_validation.json',dict(status='PASS',reference96_and_six_reused_seeds_reproduced=True,independent_checks=len(checks),pooled_recomputed=True,all_supported_IPCW_one=True,unsupported2025H24_not_scored=True,ensemble_size=ENSEMBLE_TREES))
def functions(records,paths):
    r=run_path();checks=[];hashes=[];cost=[];stable=[];support=[];coverage=[]
    for (y,a,v),p in paths.items():
        z=load(p/'verification.json');assert max(z[k]for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])<=1e-10
        assert z['max_path_variables']<=2 and z['probability_min']>=-1e-12 and z['probability_sum_max']<=1+1e-10 and z['minimum_increment']>=-1e-12
        checks.append(dict(year=y,seed=a,version=v,**z));hashes.append(dict(year=y,seed=a,version=v,forest_sha256=sha(p/'forest_PRIVATE.json'),function_sha256=sha(p/'functions_PRIVATE.npz'),prediction_sha256=sha(p/'predictions_PRIVATE.npz')))
    for rec in records:
        p=Path(rec['path']);z=load(p/'summary.json');reused=rec['reused']
        cost.append(dict(year=rec['year'],seed=rec['seed'],reused=reused,new_forest_fits=int(not reused),new_trees=0 if reused else 32,job=rec['job'],host=rec['host'],
            fit_seconds=np.nan if reused else z['fit_seconds'],shrink_seconds=np.nan if reused else z['shrink_seconds'],peak_rss_MiB=np.nan if reused else z['peak_rss_MiB'],seconds=np.nan if reused else z['seconds']))
        for label in ['feature_coverage','leaf_support']:
            dd=pd.read_csv(p/f'{label}_PRIVATE.csv');dd['year']=rec['year'];dd['seed']=rec['seed']
            if label=='feature_coverage':coverage.extend(dd.to_dict('records'))
            else:support.append(dict(year=rec['year'],seed=rec['seed'],leaves=len(dd),median_unique_companies=float(dd.unique_companies.median()),zero_target_leaves=int(dd.target_companies.eq(0).sum()),one_target_leaves=int(dd.target_companies.eq(1).sum()),borrowed_positive_target_leaves=int(dd.borrowed_positive_target.sum())))
    assert sum(z['new_forest_fits']for z in cost)==N_FITS and sum(z['new_trees']for z in cost)==N_FITS*TREES
    fresh=[load(Path(rec['path'])/'summary.json')for rec in records if not rec['reused']]
    write(r/'reports/NEW_FIT_DIAGNOSTICS.json',dict(
        forest_fits=sum(s['forest_fits']for s in fresh),
        max_score_error=max(s['independent']['max_score_error']for s in fresh),
        max_leaf_estimator_error=max(s['independent']['max_leaf_estimator_error']for s in fresh),
        max_raw_parent_error=max(s['independent']['max_raw_parent_error']for s in fresh),
        total_scored_candidates=sum(s['actual_scored_candidates']for s in fresh),
        total_guard_rejected_candidates=sum(s['guard_rejected_candidates']for s in fresh),
        max_peak_rss_MiB=max(s['peak_rss_MiB']for s in fresh)))
    for rec in load(r/'audit/registry_ensemble.json'):
        z=load(Path(rec['path'])/'summary.json');cost.append(dict(year=rec['year'],seed=ENSEMBLE,version=rec['version'],reused=False,new_forest_fits=0,new_trees=0,job=rec['job'],host=rec['host'],fit_seconds=0,shrink_seconds=0,seconds=z['seconds'],peak_rss_MiB=z['peak_rss_MiB']))
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
    z=pd.DataFrame(checks);write(r/'reports/function_validation.json',dict(status='PASS',representations=len(z),forest_fits=N_FITS,reused_forests=N_REUSED,new_trees=N_FITS*TREES,ensemble_exports=6,max_full_reconstruction=float(z.full_reconstruction.max()),max_explicit_sum=float(z.explicit_component_sum.max()),max_path_variables=int(z.max_path_variables.max()),max_probability_sum=float(z.probability_sum_max.max()),all_rows_all_native_times_both_causes=True))
def audit():
    r=run_path();rows=[]
    for root,name in [('input','input_hashes'),('ensemble_input','ensemble_input_hashes')]:
        for n,h in load(r/f'audit/{name}.json').items():assert sha(r/root/n)==h
    for n,h in load(r/'audit/source_hashes.json').items():assert sha(Path(n))==h
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==load(r/'audit/current_pointer.json')['sha256']
    for y in YEARS:
        tr=arrays(r/f'input/folds/{y}/train_PRIVATE.npz');ev=arrays(r/f'input/folds/{y}/eval_PRIVATE.npz');labels=arrays(r/f'evaluation_input/{y}_test_PRIVATE.npz')
        np.testing.assert_array_equal(ev['sample_ids'],labels['sample_ids']);np.testing.assert_array_equal(ev['X'],labels['X'][:,load(SOURCE/'audit/keep_PRIVATE.json')]);assert not set(tr['groups'])&set(labels['groups'])
        rows.append(dict(year=y,train_companies=len(tr['groups']),train_target_events=int((tr['event']==1).sum()),train_recovery_events=int((tr['event']==2).sum()),eval_companies=len(ev['X']),training_sha256=sha(r/f'input/folds/{y}/train_PRIVATE.npz'),disjoint_train_test=True))
    save('DATA_AUDIT',rows);write(r/'reports/data_validation.json',dict(status='PASS',immutable_inputs_and_sources=True,disjoint_train_test_groups=True,evaluation74_columns_match=True,current_pointer_unchanged=True))
def main():
    audit();records,paths,pred=read();functions(records,paths);evaluate(pred)
    print('PASS: 27 fits, 18 reused forests, 6 ensemble exports, 102 representations verified')
if __name__=='__main__':main()
