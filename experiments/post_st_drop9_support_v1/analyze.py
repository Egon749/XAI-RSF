"""Unified fixed-model evaluation and function diagnostics. No fitting API used."""
from pathlib import Path
from itertools import product,combinations
import argparse,time,shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CompetingRiskForest
from crforest.core import cluster_bootstrap
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_ablation_v1.uncertainty import fast_metrics as original_metrics,METRICS as ORIGINAL_METRICS
from experiments.post_st_reporting_drift_v1.metrics import stratified_metrics
from .common import *

ARMS=['reference83','support83','drop9','drop9_support']
METRICS=[*ORIGINAL_METRICS,'within_year_auc','equal_year_auc']
CONTRASTS={
 '74support_minus_74':{'drop9_support':1,'drop9':-1},
 '74support_minus_83support':{'drop9_support':1,'support83':-1},
 '74support_minus_83':{'drop9_support':1,'reference83':-1},
 'deletion_support_interaction':{'drop9_support':1,'drop9':-1,'support83':-1,'reference83':1}}

def fast_metrics(y,p,year,ids):
    z=original_metrics(y,p,year,ids);s=stratified_metrics(y,p,year,ids)
    return np.r_[z,s['within_year_auc'],s['equal_year_auc']]

def save(name,rows):
    pd.DataFrame(rows).to_csv(RUN/'reports'/f'{name}.csv',index=False)

def read():
    rows=load(RUN/'audit/registry_formal.json');assert len(rows)==9
    paths={};pred={}
    reused=load(RUN/'audit/reused_artifacts.json')
    for r in reused:paths[(r['year'],r['arm'],r['seed'],r['version'])]=Path(r['path'])
    for r in rows:
        p=Path(r['path']);assert load(p/'summary.json')['status']=='COMPLETE'
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        for v in ['raw','shrunk']:paths[(r['year'],r['arm'],r['seed'],v)]=p/v
    assert len(paths)==72
    for key,p in paths.items():
        z=arrays(p/'predictions_PRIVATE.npz');lab=arrays(RUN/f'evaluation_input/{key[0]}_test_PRIVATE.npz')
        np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
        pred[key]=z['cif']
    for y,a,v in product(YEARS,ARMS,['raw','shrunk']):pred[(y,a,'ensemble',v)]=np.mean([pred[(y,a,s,v)] for s in SEEDS],axis=0)
    return rows,paths,pred

def evaluate():
    started=time.perf_counter();(RUN/'private').mkdir(exist_ok=True)
    for n,h in load(RUN/'audit/immutable_input_hashes.json').items():assert sha(RUN/'input'/n)==h
    for n,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(n))==h
    records,paths,pred=read();labels={y:arrays(RUN/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    risks={};result=[];bins=[];seed_sd=[];validations=[]
    for y,a,s,v in pred:
        for h in [12,24]:risks[(y,a,s,v,h)]=horizon_predictions(pred[(y,a,s,v)],TIMES,labels[y][f'h{h}'])
    for y,a,s,v,h,c in product(YEARS,ARMS,[*SEEDS,'ensemble'],['raw','shrunk'],[12,24],[1,2]):
        p=risks[(y,a,s,v,h)][:,c-1];r=metrics(labels[y],p,h,c)
        yy,w,known,supported,_=evaluation_context(labels[y],h,c)
        if supported:
            # All currently supported endpoints are complete; IPCW is exactly1.
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            expected=[roc_auc_score(yy,p) if yy.any() and (~yy).any() else np.nan,
                average_precision_score(yy,p) if yy.any() else np.nan,np.mean((p-yy)**2)]
            np.testing.assert_allclose([r[k] for k in ['auc','ap','brier']],expected,rtol=0,atol=1e-13,equal_nan=True)
            validations.append(dict(year=y,arm=a,seed=s,version=v,months=h,cause=c,unit_IPCW=True,independent_metrics_match=True))
        r.update(scope=str(y),arm=a,seed=s,version=v,months=h,cause=c,years_included=str(y),
                 within_year_auc=r['auc'],equal_year_auc=r['auc'],calibration_gap=r['risk_mean']-r['observed'],absolute_calibration_gap=abs(r['risk_mean']-r['observed']))
        result.append(r)
        if s=='ensemble' and supported:
            for i,ix in enumerate(np.array_split(np.lexsort((labels[y]['sample_ids'],p)),4)):
                bins.append(dict(scope=y,arm=a,version=v,months=h,cause=c,bin=i+1,companies=len(ix),
                    risk_mean=float(p[ix].mean()),observed=float(yy[ix].mean()),events=int(yy[ix].sum())))
    for a,s,v,h,c in product(ARMS,[*SEEDS,'ensemble'],['raw','shrunk'],[12,24],[1,2]):
        years=YEARS if h==12 else [2023,2024]
        p=np.concatenate([risks[(y,a,s,v,h)][:,c-1] for y in years]);yy=np.concatenate([evaluation_context(labels[y],h,c)[0] for y in years])
        year=np.concatenate([np.full(len(labels[y]['X']),y) for y in years]);ids=np.concatenate([labels[y]['sample_ids'] for y in years])
        z=fast_metrics(yy,p,year,ids);np.testing.assert_allclose(z[:3],[roc_auc_score(yy,p),average_precision_score(yy,p),np.mean((yy-p)**2)],atol=1e-13,rtol=0)
        result.append(dict(scope='pooled',arm=a,seed=s,version=v,months=h,cause=c,years_included=','.join(map(str,years)),
            companies=len(p),target_events=int(yy.sum()),unknown=0,supported=True,observed=float(yy.mean()),risk_mean=float(z[3]),
            auc=z[0],within_year_auc=z[8],equal_year_auc=z[9],ap=z[1],brier=z[2],captured=int(z[6]),coverage=z[7],checked=sum(int(np.ceil(.2*len(labels[y]['X']))) for y in years),
            calibration_gap=z[4],absolute_calibration_gap=z[5],reason=''))
    df=pd.DataFrame(result);save('METRICS_ALL',result);save('METRICS_ENSEMBLE',df[df.seed.eq('ensemble')]);save('CALIBRATION_QUARTILES',bins)
    save('METRIC_INDEPENDENT_CHECKS',validations)
    delta=[]
    fields=['within_year_auc','equal_year_auc','auc','ap','brier','risk_mean','captured','coverage','calibration_gap','absolute_calibration_gap']
    for (scope,s,v,h,c),gg in df.groupby(['scope','seed','version','months','cause'],sort=False):
        g=gg.set_index('arm')
        for name,weights in CONTRASTS.items():
            delta.append(dict(scope=scope,seed=s,version=v,months=h,cause=c,contrast=name,
                **{k:sum(w*g.loc[a,k] for a,w in weights.items()) for k in fields}))
    save('PAIRED_DIFFERENCES',delta)
    for y,a,v,h,c in product(YEARS,ARMS,['raw','shrunk'],[12,24],[1,2]):
        pp=np.array([risks[(y,a,s,v,h)][:,c-1] for s in SEEDS]);sd=pp.std(0,ddof=1)
        seed_sd.append(dict(year=y,arm=a,version=v,months=h,cause=c,mean_prediction_sd=float(sd.mean()),max_prediction_sd=float(sd.max()),median_prediction_sd=float(np.median(sd))))
    save('PREDICTION_STABILITY',seed_sd)
    np.savez_compressed(RUN/'private/risks_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risks.items()})
    # All comparisons share exactly the same fixed company draws, stratified by year.
    rng=np.random.default_rng(ANALYSIS_SEED);draws={y:rng.integers(len(labels[y]['X']),size=(1000,len(labels[y]['X']))) for y in YEARS}
    file=RUN/'private/bootstrap_indices_PRIVATE.npz'
    if file.exists():
        for y in YEARS:np.testing.assert_array_equal(arrays(file)[str(y)],draws[y])
    else:np.savez_compressed(file,**{str(k):v for k,v in draws.items()})
    intervals=[]
    for h,c,scope in product([12,24],[1,2],[*YEARS,'pooled']):
        if h==24 and scope==2025:continue
        years=([*YEARS] if h==12 else [2023,2024]) if scope=='pooled' else [scope]
        yy=np.concatenate([evaluation_context(labels[y],h,c)[0] for y in years]);year=np.concatenate([np.full(len(labels[y]['X']),y) for y in years])
        ids=np.concatenate([labels[y]['sample_ids'] for y in years]);p=np.array([np.concatenate([risks[(y,a,'ensemble','shrunk',h)][:,c-1] for y in years]) for a in ARMS])
        actual=np.array([fast_metrics(yy,pp,year,ids) for pp in p]);all_values=np.empty((1000,len(CONTRASTS),len(METRICS)))
        for b in range(1000):
            ix=np.concatenate([draws[y][b]+sum(len(labels[v]['X']) for v in years[:i]) for i,y in enumerate(years)])
            z=np.array([fast_metrics(yy[ix],pp[ix],year[ix],ids[ix]) for pp in p])
            all_values[b]=np.array([sum(w*z[ARMS.index(a)] for a,w in weights.items()) for weights in CONTRASTS.values()])
        for i,(name,weights) in enumerate(CONTRASTS.items()):
            estimate=sum(w*actual[ARMS.index(a)] for a,w in weights.items())
            for j,metric in enumerate(METRICS):
                values=all_values[:,i,j];valid=np.isfinite(values);bounds=np.quantile(values[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
                intervals.append(dict(scope=scope,months=h,cause=c,contrast=name,metric=metric,estimate=estimate[j],
                    lower95=bounds[0],upper95=bounds[1],repeats=1000,valid_repeats=int(valid.sum()),undefined_repeats=int((~valid).sum()),analysis_seed=ANALYSIS_SEED))
        np.savez_compressed(RUN/f'private/bootstrap_{scope}_{h}_{c}_PRIVATE.npz',values=all_values)
    save('PAIRED_BOOTSTRAP_95',intervals)
    write(RUN/'reports/metric_validation.json',dict(status='PASS',forest_fits=0,metric_rows=len(df),all_supported_IPCW_weights_one=True,
        independent_sklearn_and_Brier_checks=len(validations),pooled_recomputed=True,top20_annual_quotas=[8,9,11],
        common_bootstrap_indices_sha256=sha(file),repeats=1000,analysis_seed=ANALYSIS_SEED,seconds=time.perf_counter()-started))
    print(df[(df.seed=='ensemble')&(df.version=='shrunk')&(df.cause==1)][['scope','months','arm','auc','ap','brier','captured','target_events','risk_mean']].to_string(index=False),flush=True)


def functions():
    records,paths,_=read();checks=[];cost=[];support=[];search=[];hashes=[];term_info={}
    for r in load(SOURCE/'audit/registry_ensemble.json'):
        if r['arm'] not in ['baseline','support']:continue
        paths[(r['year'],'reference83' if r['arm']=='baseline' else 'support83','ensemble',r['version'])]=Path(r['path'])/'representation'
    for r in load(RUN/'audit/registry_ensemble.json'):
        p=Path(r['path'])
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        paths[(r['year'],'drop9_support','ensemble',r['version'])]=p/'representation'
    for y in YEARS:
        name='post_st_drop9_v1' if y==2024 else 'post_st_drop9_transport_v1'
        oldrun=Path(load(REPO/'experiments'/name/'execution_receipt.json')['private_run'])
        paths[(y,'drop9','ensemble','shrunk')]=oldrun/'private'/('verify_drop9_ensemble' if y==2024 else f'verify_{y}_drop9_ensemble')
    for (y,a,s,v),p in paths.items():
        z=load(p/'verification.json');assert max(z[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload'])<=1e-10
        assert z['max_path_variables']<=2 and z['probability_sum_max']<=1+1e-10 and z['probability_min']>=-1e-12 and z['minimum_increment']>=-1e-12
        checks.append(dict(year=y,arm=a,seed=s,version=v,verification_source='new' if a=='drop9_support' else 'reused',**z))
        term_info[(y,a,s,v)]=(load(p/'complexity_PRIVATE.json'),arrays(p/'components_PRIVATE.npz'))
        hashes.append(dict(year=y,arm=a,seed=s,version=v,forest_sha256=sha(p/'forest_PRIVATE.json'),function_sha256=sha(p/'functions_PRIVATE.npz'),prediction_sha256=sha(p/'predictions_PRIVATE.npz')))
    for r in records:
        p=Path(r['path']);z=load(p/'summary.json');meta={k:r[k] for k in ['year','arm','seed']}
        cost.append(dict(**meta,**{k:z[k] for k in ['host','slurm_job','fit_seconds','shrink_seconds','replay_seconds','seconds','peak_rss_MiB','actual_scored_candidates']},**z['independent']))
        leaves=pd.read_csv(p/'leaf_support_PRIVATE.csv');nodes=pd.read_csv(p/'node_search_PRIVATE.csv')
        support.append(dict(**meta,leaves=len(leaves),unique_companies_min=int(leaves.unique_companies.min()),unique_companies_median=float(leaves.unique_companies.median()),target_median=float(leaves.target_companies.median()),recovery_median=float(leaves.recovery_companies.median()),zero_target_leaves=int(leaves.target_companies.eq(0).sum()),one_target_leaves=int(leaves.target_companies.eq(1).sum()),borrowed_positive_target_leaves=int(leaves.borrowed_positive_target.sum())))
        search.append(dict(**meta,search_nodes=len(nodes),**{k:int(nodes[k].sum()) for k in ['finite_cuts','routed_attempts','scored_candidates','guard_rejected','guard_changed_winner','guard_stopped_split']}))
        # Verify stored worker pairing against the historical74 configuration.
        tr=arrays(RUN/f'input/folds/{r["year"]}/train_PRIVATE.npz')
        old=CompetingRiskForest.load(paths[(r['year'],'drop9',r['seed'],'shrunk')]/'forest_PRIVATE.json')
        assert load(p/'pairing.json')==pairing(tr['groups'],old.config,74)
    stability=[]
    for y,a,v in product(YEARS,ARMS,['raw','shrunk']):
        for s1,s2 in combinations(SEEDS,2):
            c1,t1=term_info[(y,a,s1,v)];c2,t2=term_info[(y,a,s2,v)]
            sets=lambda cc,o:{tuple(r['features']) for r in cc['functions'] if r['order']==o and r['nonzero']}
            i1={k:i for i,k in enumerate(t1['keys'])};i2={k:i for i,k in enumerate(t2['keys'])}
            for o in [1,2]:
                l,r=sets(c1,o),sets(c2,o);jacc=len(l&r)/len(l|r) if l|r else np.nan
                for hi,day in [(0,365),(2,730)]:
                    for c in [0,1]:
                        common=[k for k in sorted(i1.keys()&i2.keys()) if len(k.split(','))==o];corr=[];rms=[]
                        for k in common:
                            x=t1['terms'][:,i1[k],hi,c];z=t2['terms'][:,i2[k],hi,c];rms.append(float(np.sqrt(np.mean((x-z)**2))))
                            if x.std()>1e-12 and z.std()>1e-12:corr.append(float(np.corrcoef(x,z)[0,1]))
                        stability.append(dict(year=y,arm=a,version=v,seed_a=s1,seed_b=s2,day=day,cause=c+1,order=o,jaccard=jacc,common_functions=len(common),nonconstant_functions=len(corr),median_correlation=float(np.median(corr)) if corr else np.nan,mean_rms_difference=float(np.mean(rms)) if rms else np.nan))
    for name,rows in [('NUMERICAL_COMPLEXITY',checks),('COMPUTE_COST',cost),('LEAF_SUPPORT',support),('SEARCH_DIAGNOSTICS',search),('MODEL_HASHES',hashes),('FUNCTION_STABILITY',stability)]:save(name,rows)
    df=pd.DataFrame(checks);fresh=df[df.arm.eq('drop9_support')]
    write(RUN/'reports/function_validation.json',dict(status='PASS',new_single_seed_representations=18,new_ensemble_representations=6,reused_verified_representations=len(df)-len(fresh),max_reconstruction_error=float(fresh.full_reconstruction.max()),max_explicit_sum_error=float(fresh.explicit_component_sum.max()),max_function_reload_error=float(fresh.function_reload.max()),max_forest_reload_error=float(fresh.forest_reload.max()),max_path_variables=int(fresh.max_path_variables.max()),all_native_times_and_all_train_eval_rows=True,no_clipping=True,forest_fits_in_analysis=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metrics','functions']);a=p.parse_args();{'metrics':evaluate,'functions':functions}[a.stage]()
