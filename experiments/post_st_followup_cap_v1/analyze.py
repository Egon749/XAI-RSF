"""Unified fixed-prediction evaluation; never fit, calibrate or select a model."""
from itertools import product,combinations
import argparse,time
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_ablation_v1.uncertainty import fast_metrics as original_metrics,METRICS as ORIGINAL_METRICS
from experiments.post_st_reporting_drift_v1.metrics import stratified_metrics,paired_auc
from .common import *
RUN=run_path()
METRICS=[*ORIGINAL_METRICS,'within_year_auc','equal_year_auc']

def fast_metrics(y,p,year,ids):
    z=original_metrics(y,p,year,ids);s=stratified_metrics(y,p,year,ids)
    return np.r_[z,s['within_year_auc'],s['equal_year_auc']]

def save(name,rows):pd.DataFrame(rows).to_csv(RUN/'reports'/f'{name}.csv',index=False)

def read():
    records=load(RUN/'audit/registry_formal.json');assert len(records)==36
    paths={};pred={}
    for r in records:
        p=Path(r['path']);assert load(p/'summary.json')['status']=='COMPLETE'
        for name,h in load(p/'hashes.json').items():assert sha(p/name)==h
        for v in ['raw','shrunk']:
            key=(r['year'],r['arm'],r['seed'],v);paths[key]=p/v
            z=arrays(p/v/'predictions_PRIVATE.npz');ev=arrays(RUN/f'input/folds/{r["year"]}/{r["arm"]}/eval_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
            pred[key]=z['cif']
    for y,a,v in product(YEARS,ARMS,['raw','shrunk']):
        pred[(y,a,'ensemble',v)]=np.mean([pred[(y,a,s,v)] for s in SEEDS],axis=0)
    return records,paths,pred

def evaluate():
    started=time.perf_counter()
    for name,h in load(RUN/'audit/immutable_input_hashes.json').items():assert sha(RUN/'input'/name)==h
    for name,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(name))==h
    records,paths,pred=read()
    labels={y:arrays(RUN/f'evaluation_input/{y}_test_PRIVATE.npz')for y in YEARS}
    risks={};rows=[];checks=[];stability=[];differences=[]
    for y,a,s,v in pred:
        for h in [12,24]:risks[(y,a,s,v,h)]=horizon_predictions(pred[(y,a,s,v)],TIMES,labels[y][f'h{h}'])
    for y,a,s,v,h,c in product(YEARS,ARMS,[*SEEDS,'ensemble'],['raw','shrunk'],[12,24],[1,2]):
        p=risks[(y,a,s,v,h)][:,c-1]
        if a=='cap12' and h==24:
            r=dict(companies=len(p),supported=False,reason='OUTSIDE_TRAINING_FOLLOWUP_WINDOW',
                **{k:np.nan for k in ['auc','ap','brier','risk_mean','observed','target_events','unknown','captured','coverage','checked']})
        else:
            r=metrics(labels[y],p,h,c)
            yy,w,known,supported,_=evaluation_context(labels[y],h,c)
            if supported:
                assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
                expected=[roc_auc_score(yy,p) if yy.any() and (~yy).any() else np.nan,
                    average_precision_score(yy,p) if yy.any() else np.nan,np.mean((p-yy)**2)]
                np.testing.assert_allclose([r[k]for k in ['auc','ap','brier']],expected,rtol=0,atol=1e-13,equal_nan=True)
                np.testing.assert_allclose(r['auc'],paired_auc(yy,p)[0],rtol=0,atol=1e-13,equal_nan=True)
                checks.append(dict(year=y,arm=a,seed=s,version=v,months=h,cause=c,unit_IPCW=True,metrics_match=True))
        rows.append(dict(**r,scope=str(y),arm=a,seed=s,version=v,months=h,cause=c,
            within_year_auc=r['auc'],equal_year_auc=r['auc'],
            observed_expected=r['observed']/r['risk_mean'] if r['risk_mean']>0 else np.nan,
            calibration_gap=r['risk_mean']-r['observed'],absolute_calibration_gap=abs(r['risk_mean']-r['observed'])))
    for a,s,v,h,c in product(ARMS,[*SEEDS,'ensemble'],['raw','shrunk'],[12,24],[1,2]):
        if a=='cap12' and h==24:continue
        years=YEARS if h==12 else [2023,2024]
        p=np.concatenate([risks[(y,a,s,v,h)][:,c-1]for y in years])
        yy=np.concatenate([evaluation_context(labels[y],h,c)[0]for y in years])
        yrs=np.concatenate([np.full(len(labels[y]['X']),y)for y in years]);ids=np.concatenate([labels[y]['sample_ids']for y in years])
        z=fast_metrics(yy,p,yrs,ids)
        np.testing.assert_allclose(z[:3],[roc_auc_score(yy,p),average_precision_score(yy,p),np.mean((yy-p)**2)],atol=1e-13,rtol=0)
        rows.append(dict(scope='pooled',arm=a,seed=s,version=v,months=h,cause=c,years_included=','.join(map(str,years)),
            companies=len(p),target_events=int(yy.sum()),unknown=0,supported=True,reason='',observed=float(yy.mean()),
            risk_mean=z[3],auc=z[0],within_year_auc=z[8],equal_year_auc=z[9],ap=z[1],brier=z[2],
            captured=int(z[6]),coverage=z[7],checked=sum(int(np.ceil(.2*len(labels[y]['X'])))for y in years),
            calibration_gap=z[4],absolute_calibration_gap=z[5],observed_expected=float(yy.mean()/p.mean())if p.mean()>0 else np.nan))
    df=pd.DataFrame(rows);save('METRICS_ALL',df);save('METRICS_ENSEMBLE',df[df.seed.eq('ensemble')]);save('METRIC_INDEPENDENT_CHECKS',checks)
    # Exact old24 reproduction is mandatory before interpreting any new arm.
    old=pd.read_csv(REPO/'experiments/post_st_drop9_support_v1/METRICS_ALL.csv')
    old=old[old.arm.eq('drop9_support')]
    for r in old.to_dict('records'):
        rr=df[(df.scope==str(r['scope']))&(df.arm=='cap24')&(df.seed.astype(str)==str(r['seed']))&
            (df.version==r['version'])&(df.months==r['months'])&(df.cause==r['cause'])]
        assert len(rr)==1
        for key in ['auc','ap','brier','risk_mean','captured','coverage']:
            np.testing.assert_allclose(rr.iloc[0][key],r[key],atol=1e-13,rtol=0,equal_nan=True)
    for (scope,seed,v,h,c),g in df.groupby(['scope','seed','version','months','cause'],sort=False):
        g=g.set_index('arm')
        for a in NEW_ARMS:
            if a not in g.index or not bool(g.loc[a,'supported']):continue
            differences.append(dict(scope=scope,seed=seed,version=v,months=h,cause=c,contrast=a+'_minus_cap24',
                **{k:g.loc[a,k]-g.loc['cap24',k]for k in ['auc','within_year_auc','ap','brier','risk_mean','captured','coverage','calibration_gap']}))
    save('PAIRED_DIFFERENCES',differences)
    for y,a,v,h,c in product(YEARS,ARMS,['raw','shrunk'],[12,24],[1,2]):
        if a=='cap12' and h==24:continue
        z=np.array([risks[(y,a,s,v,h)][:,c-1]for s in SEEDS]).std(0,ddof=1)
        stability.append(dict(year=y,arm=a,version=v,months=h,cause=c,mean_prediction_sd=z.mean(),max_prediction_sd=z.max()))
    save('PREDICTION_STABILITY',stability)
    np.savez_compressed(RUN/'private/risks_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risks.items()})
    draws=arrays(RUN/'private/bootstrap_indices_PRIVATE.npz');rng=np.random.default_rng(ANALYSIS_SEED)
    for y in YEARS:np.testing.assert_array_equal(draws[str(y)],rng.integers(len(labels[y]['X']),size=(1000,len(labels[y]['X']))))
    intervals=[]
    for h,c,scope in product([12,24],[1,2],[*YEARS,'pooled']):
        if h==24 and scope==2025:continue
        years=(YEARS if h==12 else [2023,2024])if scope=='pooled' else [scope]
        arms=ARMS if h==12 else ['cap24','cap36','uncapped']
        yy=np.concatenate([evaluation_context(labels[y],h,c)[0]for y in years]);yrs=np.concatenate([np.full(len(labels[y]['X']),y)for y in years]);ids=np.concatenate([labels[y]['sample_ids']for y in years])
        pp=np.array([np.concatenate([risks[(y,a,'ensemble','shrunk',h)][:,c-1]for y in years])for a in arms])
        actual=np.array([fast_metrics(yy,p,yrs,ids)for p in pp]); values=np.empty((1000,len(arms),len(METRICS)))
        for b in range(1000):
            ix=np.concatenate([draws[str(y)][b]+sum(len(labels[v]['X'])for v in years[:i])for i,y in enumerate(years)])
            values[b]=[fast_metrics(yy[ix],p[ix],yrs[ix],ids[ix])for p in pp]
        for i,a in enumerate(arms):
            if a=='cap24':continue
            diff=values[:,i]-values[:,arms.index('cap24')];point=actual[i]-actual[arms.index('cap24')]
            for j,m in enumerate(METRICS):
                v=diff[:,j];ok=np.isfinite(v);ci=np.quantile(v[ok],[.025,.975])if ok.any()else[np.nan,np.nan]
                intervals.append(dict(scope=scope,months=h,cause=c,contrast=a+'_minus_cap24',metric=m,
                    estimate=point[j],lower95=ci[0],upper95=ci[1],repeats=1000,valid_repeats=int(ok.sum()),undefined_repeats=int((~ok).sum()),analysis_seed=ANALYSIS_SEED))
    save('PAIRED_BOOTSTRAP_95',intervals)
    write(RUN/'reports/metric_validation.json',dict(status='PASS',forest_fits=0,cap24_reproduced=True,
        all_supported_IPCW_one=True,independent_metric_checks=len(checks),pooled_recomputed=True,
        cap12_beyond_training_support_not_scored=True,unsupported2025H24_not_scored=True,
        bootstrap_index_sha256=sha(RUN/'private/bootstrap_indices_PRIVATE.npz'),seconds=time.perf_counter()-started))
    print(df[(df.seed=='ensemble')&(df.version=='shrunk')&(df.cause==1)&(df.months==12)][['scope','arm','auc','ap','brier','captured','target_events','risk_mean']].to_string(index=False),flush=True)

def functions():
    records,paths,_=read();checks=[];cost=[];support=[];hashes=[];info={};stability=[]
    for r in load(RUN/'audit/registry_ensemble.json'):
        p=Path(r['path'])
        for name,h in load(p/'hashes.json').items():assert sha(p/name)==h
        paths[(r['year'],r['arm'],'ensemble',r['version'])]=p/'representation'
    for (y,a,s,v),p in paths.items():
        z=load(p/'verification.json')
        assert max(z[k]for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])<=1e-10
        assert z['max_path_variables']<=2 and z['probability_sum_max']<=1+1e-10 and z['probability_min']>=-1e-12 and z['minimum_increment']>=-1e-12
        checks.append(dict(year=y,arm=a,seed=s,version=v,reused=a=='cap24',**z))
        info[(y,a,s,v)]=(load(p/'complexity_PRIVATE.json'),arrays(p/'components_PRIVATE.npz'))
        hashes.append(dict(year=y,arm=a,seed=s,version=v,forest_sha256=sha(p/'forest_PRIVATE.json'),function_sha256=sha(p/'functions_PRIVATE.npz'),prediction_sha256=sha(p/'predictions_PRIVATE.npz')))
    for r in records:
        p=Path(r['path']);z=load(p/'summary.json');base={k:r[k]for k in ['year','arm','seed']}
        cost.append(dict(**base,reused=r['arm']=='cap24',**{k:z[k]for k in ['host','slurm_job','fit_seconds','shrink_seconds','seconds','peak_rss_MiB','actual_scored_candidates']},**z['independent']))
        leaves=pd.read_csv(p/'leaf_support_PRIVATE.csv')
        support.append(dict(**base,leaves=len(leaves),median_unique_companies=leaves.unique_companies.median(),
            zero_target_leaves=int(leaves.target_companies.eq(0).sum()),one_target_leaves=int(leaves.target_companies.eq(1).sum()),
            borrowed_positive_target_leaves=int(leaves.borrowed_positive_target.sum())))
        tr=arrays(RUN/f'input/folds/{r["year"]}/{r["arm"]}/train_PRIVATE.npz')
        assert load(p/'pairing.json')==pairing(tr['groups'],configuration(r['seed'],r['arm']),74)
    for y,a,v,s1,s2 in [(y,a,v,*pair)for y,a,v in product(YEARS,ARMS,['raw','shrunk'])for pair in combinations(SEEDS,2)]:
        c1,t1=info[(y,a,s1,v)];c2,t2=info[(y,a,s2,v)]
        i1={k:i for i,k in enumerate(t1['keys'])};i2={k:i for i,k in enumerate(t2['keys'])}
        for order in [1,2]:
            sets=[{tuple(r['features'])for r in comp['functions']if r['order']==order and r['nonzero']}for comp in [c1,c2]]
            jacc=len(sets[0]&sets[1])/len(sets[0]|sets[1])if sets[0]|sets[1]else np.nan
            for hi,day in [(0,365),(2,730)]:
                if a=='cap12' and day==730:continue
                for cause in [1,2]:
                    keys=[k for k in sorted(i1.keys()&i2.keys())if len(k.split(','))==order];corr=[];rms=[]
                    for k in keys:
                        l=t1['terms'][:,i1[k],hi,cause-1];rr=t2['terms'][:,i2[k],hi,cause-1]
                        rms.append(float(np.sqrt(np.mean((l-rr)**2))))
                        if l.std()>1e-12 and rr.std()>1e-12:corr.append(float(np.corrcoef(l,rr)[0,1]))
                    stability.append(dict(year=y,arm=a,version=v,seed_a=s1,seed_b=s2,day=day,cause=cause,order=order,jaccard=jacc,
                        common_functions=len(keys),median_correlation=np.median(corr)if corr else np.nan,mean_rms_difference=np.mean(rms)if rms else np.nan))
    for name,data in [('NUMERICAL_COMPLEXITY',checks),('COMPUTE_COST',cost),('LEAF_SUPPORT',support),('MODEL_HASHES',hashes),('FUNCTION_STABILITY',stability)]:save(name,data)
    df=pd.DataFrame(checks)
    write(RUN/'reports/function_validation.json',dict(status='PASS',verified_representations=len(df),
        max_reconstruction_error=float(df.full_reconstruction.max()),max_explicit_sum_error=float(df.explicit_component_sum.max()),
        max_path_variables=int(df.max_path_variables.max()),all_rows_all_native_times_both_causes=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metrics','functions']);a=p.parse_args()
    {'metrics':evaluate,'functions':functions}[a.stage]()
