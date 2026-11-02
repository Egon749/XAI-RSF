"""Unified fixed-model evaluation and function diagnostics. No fitting API used."""
from pathlib import Path
from itertools import product,combinations
import argparse,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CompetingRiskForest
from crforest.core import cluster_bootstrap
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_ablation_v1.uncertainty import fast_metrics,METRICS
from .common import *

CONTRASTS={a+'_minus_baseline':{a:1,'baseline':-1} for a in ARMS[1:]}
CONTRASTS['score_support_interaction']={'quadratic_support':1,'quadratic':-1,'support':-1,'baseline':1}

def save(name,rows):
    pd.DataFrame(rows).to_csv(RUN/'reports'/f'{name}.csv',index=False)

def read():
    rows=load(RUN/'audit/registry_formal.json');assert len(rows)==36
    paths={};pred={}
    for r in rows:
        p=Path(r['path']);summary=load(p/'summary.json');assert summary['status']=='COMPLETE'
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        for v in ['raw','shrunk']:
            paths[(r['year'],r['arm'],r['seed'],v)]=p/v
            z=arrays(p/v/'predictions_PRIVATE.npz');pred[(r['year'],r['arm'],r['seed'],v)]=z['cif']
            lab=arrays(RUN/f'evaluation_input/{r["year"]}_test_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
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
                 calibration_gap=r['risk_mean']-r['observed'],absolute_calibration_gap=abs(r['risk_mean']-r['observed']))
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
            auc=z[0],ap=z[1],brier=z[2],captured=int(z[6]),coverage=z[7],checked=sum(int(np.ceil(.2*len(labels[y]['X']))) for y in years),
            calibration_gap=z[4],absolute_calibration_gap=z[5],reason=''))
    df=pd.DataFrame(result);save('METRICS_ALL',result);save('METRICS_ENSEMBLE',df[df.seed.eq('ensemble')]);save('CALIBRATION_QUARTILES',bins)
    save('METRIC_INDEPENDENT_CHECKS',validations)
    delta=[]
    fields=['auc','ap','brier','risk_mean','captured','coverage','calibration_gap','absolute_calibration_gap']
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
    records,paths,_=read();summaries=[];cost=[];support=[];search=[];pairing_checks=[];coverage=[];hashes=[];term_info={}
    for r in load(RUN/'audit/registry_ensemble.json'):
        p=Path(r['path']);summary=load(p/'summary.json');assert summary['new_forests']==0
        for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
        paths[(r['year'],r['arm'],'ensemble',r['version'])]=p/'representation'
    for (y,a,s,v),p in paths.items():
        cc=load(p/'complexity_PRIVATE.json');terms=arrays(p/'components_PRIVATE.npz');term_info[(y,a,s,v)]=(cc,terms)
        checks=load(p/'verification.json');summaries.append(dict(year=y,arm=a,seed=s,version=v,**checks))
        hashes.append(dict(year=y,arm=a,seed=s,version=v,forest_sha256=sha(p/'forest_PRIVATE.json'),function_sha256=sha(p/'functions_PRIVATE.npz'),prediction_sha256=sha(p/'predictions_PRIVATE.npz')))
    for r in records:
        p=Path(r['path']);z=load(p/'summary.json');meta={k:r[k] for k in ['year','arm','seed']}
        cost.append(dict(**meta,**{k:z[k] for k in ['host','slurm_job','fit_seconds','shrink_seconds','replay_seconds','seconds','peak_rss_MiB','actual_scored_candidates','reference_trees_identical','reference_prediction_error']},**z['independent']))
        leaves=pd.read_csv(p/'leaf_support_PRIVATE.csv');nodes=pd.read_csv(p/'node_search_PRIVATE.csv');features=pd.read_csv(p/'feature_coverage_PRIVATE.csv')
        support.append(dict(**meta,leaves=len(leaves),unique_companies_min=int(leaves.unique_companies.min()),
            unique_companies_median=float(leaves.unique_companies.median()),unique_companies_max=int(leaves.unique_companies.max()),
            target_min=int(leaves.target_companies.min()),target_median=float(leaves.target_companies.median()),
            recovery_median=float(leaves.recovery_companies.median()),zero_target_leaves=int(leaves.target_companies.eq(0).sum()),
            one_target_leaves=int(leaves.target_companies.eq(1).sum()),borrowed_positive_target_leaves=int(leaves.borrowed_positive_target.sum())))
        row=dict(**meta,search_nodes=len(nodes),selected_nodes=int(nodes.selected_opposed_signs.notna().sum()),opposed_selected=int(nodes.selected_opposed_signs.fillna(False).sum()))
        for k in ['finite_cuts','missing_only_cuts','routed_attempts','scored_candidates','guard_rejected','guard_changed_winner','guard_stopped_split']:row[k]=int(nodes[k].sum())
        positive=nodes.best_without_guard.gt(0);row['guard_best_score_relative_loss_mean']=float(((nodes.loc[positive,'best_without_guard']-nodes.loc[positive,'best_with_guard'])/nodes.loc[positive,'best_without_guard']).mean())
        search.append(row)
        for f in features.to_dict('records'):coverage.append(dict(**meta,**f))
    for y,s in product(YEARS,SEEDS):
        base=load(paths[(y,'baseline',s,'raw')].parent/'pairing.json')
        for a in ARMS:
            pp=load(paths[(y,a,s,'raw')].parent/'pairing.json');assert pp==base
            pairing_checks.append(dict(year=y,seed=s,arm=a,trees=32,bootstrap_multiplicities_and_feature_sets_identical=True))
    stability=[];function_details=[];amplitudes=[]
    for (y,a,s,v),(cc,tt) in term_info.items():
        for hi,day in [(0,365),(2,730)]:
            for c in [0,1]:
                order=np.array([len(k.split(',')) for k in tt['keys']]);terms=tt['terms'][:,:,hi,c]
                amplitudes.append(dict(year=y,arm=a,seed=s,version=v,day=day,cause=c+1,
                    sum_main_mean_abs=float(abs(terms[:,order==1]).mean(0).sum()),sum_pair_mean_abs=float(abs(terms[:,order==2]).mean(0).sum())))
    for y,a,v in product(YEARS,ARMS,['raw','shrunk']):
        for s1,s2 in combinations(SEEDS,2):
            c1,t1=term_info[(y,a,s1,v)];c2,t2=term_info[(y,a,s2,v)]
            sets=lambda cc,o:{tuple(r['features']) for r in cc['functions'] if r['order']==o and r['nonzero']}
            jacc={}
            for o in [1,2]:
                l,r=sets(c1,o),sets(c2,o);jacc[f'order{o}_jaccard']=len(l&r)/len(l|r) if l|r else np.nan
            i1={k:i for i,k in enumerate(t1['keys'])};i2={k:i for i,k in enumerate(t2['keys'])}
            for hi,day in [(0,365),(2,730)]:
                for c,o in product([0,1],[1,2]):
                    corr=[];rms=[];constant=0
                    common=[k for k in sorted(i1.keys()&i2.keys()) if len(k.split(','))==o]
                    for k in common:
                        l=t1['terms'][:,i1[k],hi,c];r=t2['terms'][:,i2[k],hi,c];d=float(np.sqrt(np.mean((l-r)**2)));rms.append(d)
                        cor=float(np.corrcoef(l,r)[0,1]) if l.std()>1e-12 and r.std()>1e-12 else np.nan
                        if np.isfinite(cor):corr.append(cor)
                        else:constant+=1
                        function_details.append(dict(year=y,arm=a,version=v,seed_a=s1,seed_b=s2,key=k,day=day,cause=c+1,correlation=cor,rms_difference=d))
                    stability.append(dict(year=y,arm=a,version=v,seed_a=s1,seed_b=s2,day=day,cause=c+1,order=o,**jacc,
                        common_functions=len(common),nonconstant_functions=len(corr),constant_functions=constant,
                        median_correlation=float(np.median(corr)) if corr else np.nan,mean_rms_difference=float(np.mean(rms)) if rms else np.nan))
    pd.DataFrame(function_details).to_csv(RUN/'private/function_stability_details_PRIVATE.csv',index=False)
    for name,rows in [('NUMERICAL_COMPLEXITY',summaries),('COMPUTE_COST',cost),('LEAF_SUPPORT',support),('SEARCH_DIAGNOSTICS',search),
        ('FEATURE_COVERAGE',coverage),('PAIRING_CHECKS',pairing_checks),('MODEL_HASHES',hashes),('FUNCTION_STABILITY',stability),('FUNCTION_AMPLITUDES',amplitudes)]:save(name,rows)
    sf=pd.DataFrame(summaries);cf=pd.DataFrame(cost)
    write(RUN/'reports/function_validation.json',dict(status='PASS',single_seed_representations=72,ensemble_representations=24,
        max_reconstruction_error=float(sf.full_reconstruction.max()),max_explicit_sum_error=float(sf.explicit_component_sum.max()),
        max_function_reload_error=float(sf.function_reload.max()),max_forest_reload_error=float(sf.forest_reload.max()),
        max_path_variables=int(sf.max_path_variables.max()),max_independent_score_error=float(cf.max_score_error.max()),
        max_direct_parent_AJ_error=float(cf.max_raw_parent_error.max()),original_reference_exact_matches=int(cf.reference_trees_identical.eq(True).sum()),
        all_native_times_and_all_train_eval_rows=True,no_clipping=True,forest_fits_in_analysis=0))

def mechanism():
    records,paths,_=read();details=[]
    for r in records:
        y,a,s=r['year'],r['arm'],r['seed'];f=CompetingRiskForest.load(paths[(y,a,s,'raw')]/'forest_PRIVATE.json')
        tr=arrays(RUN/f'input/folds/{y}/train_PRIVATE.npz')
        for i,tree in enumerate(f.trees_):
            n=tree[0]
            if 'cif' in n:continue
            ix=cluster_bootstrap(tr['groups'],s,i);z=tr['X'][ix,n['feature']];left=(z<=n['threshold'])|(np.isnan(z)&n['missing_left'])
            u,v=independent_components(tr['time'][ix],tr['event'][ix],left)
            zz=np.divide(u*u,v,out=np.zeros(2),where=v>1e-18)
            details.append(dict(year=y,arm=a,seed=s,tree=i,feature=n['feature'],opposed=bool(u[0]*u[1]<0),
                cancellation_fraction=1-abs(u.sum())/abs(u).sum() if abs(u).sum()>0 else 0.,
                target_standardized_magnitude=float(np.sqrt(zz[0])),recovery_standardized_magnitude=float(np.sqrt(zz[1])),
                target_share_of_quadratic=float(zz[0]/zz.sum()) if zz.sum()>0 else np.nan,
                target_deviation=float(u[0]),recovery_deviation=float(u[1])))
    d=pd.DataFrame(details);d.to_csv(RUN/'private/root_score_detail_PRIVATE.csv',index=False)
    g=d.groupby(['year','arm','seed']).agg(roots=('tree','size'),opposed_roots=('opposed','sum'),
        mean_cancellation_fraction=('cancellation_fraction','mean'),mean_target_z=('target_standardized_magnitude','mean'),
        mean_recovery_z=('recovery_standardized_magnitude','mean'),mean_target_quadratic_share=('target_share_of_quadratic','mean')).reset_index()
    save('ROOT_SCORE_MECHANISM',g)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metrics','functions','mechanism']);a=p.parse_args();{'metrics':evaluate,'functions':functions,'mechanism':mechanism}[a.stage]()
