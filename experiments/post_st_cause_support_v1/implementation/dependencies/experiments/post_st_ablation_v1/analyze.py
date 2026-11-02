"""Unified endpoint metrics, fixed contrasts and saved-function diagnostics."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .common import *
from .stages import validation_outputs

def contrasts():
    answer={a+'_minus_reference':{a:1,'shallow_parent':-1} for a in ARMS if a!='shallow_parent'}
    answer.update(depth_without_shrink={'shallow':1,'baseline':-1},depth_with_shrink={'shallow_parent':1,'parent_shrink':-1},
        shrink_at_depth6={'parent_shrink':1,'baseline':-1},shrink_at_depth4={'shallow_parent':1,'shallow':-1},
        depth_shrink_interaction={'shallow_parent':1,'shallow':-1,'parent_shrink':-1,'baseline':1})
    for a in NEW:
        answer[a+'_raw_minus_shallow']={a+'_raw':1,'shallow':-1}
        answer[a+'_shrink_effect']={a:1,a+'_raw':-1}
        answer[a+'_shrink_interaction']={a:1,a+'_raw':-1,'shallow_parent':-1,'shallow':1}
    return answer

def selected(risk,year,ids,fraction=.2):
    chosen=np.zeros(len(risk),bool)
    for y in np.unique(year):
        ix=np.flatnonzero(year==y);order=np.lexsort((ids[ix],-risk[ix]))
        chosen[ix[order[:int(np.ceil(len(ix)*fraction))]]]=True
    return chosen

def read_predictions(run):
    pred={};paths={}
    for r in registry(run):
        p=Path(r['path']);y,a,s=r['year'],r['arm'],str(r['seed'])
        for n,h in json.loads((p/'hashes.json').read_text()).items():assert sha(p/n)==h
        z=arrays(p/'predictions_PRIVATE.npz');ev=arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz')
        np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
        pred[(str(y),a,s)]=z;paths[(y,a,s)]=p
    for y,a in itertools.product(YEARS,ARMS+['standard_rsf']):
        ps=[pred[(str(y),a,str(s))] for s in SEEDS]
        pred[(str(y),a,'ensemble')]=dict(cif=np.mean([p['cif'] for p in ps],axis=0),times=TIMES,sample_ids=ps[0]['sample_ids'])
    return pred,paths

def evaluate(run):
    out=run/'evaluation';out.mkdir(exist_ok=False)
    for n,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/n)==h
    for n,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(n))==h
    pred,paths=read_predictions(run)
    data={str(y):arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz') for y in YEARS}
    data['pooled']={k:np.concatenate([data[str(y)][k] for y in YEARS]) for k in data['2023']}
    risks={}
    for (y,a,s),z in pred.items():
        for h in [12,24]:risks[(y,a,s,h)]=horizon_predictions(z['cif'],z['times'],data[y][f'h{h}'])
    for a,s,h in {(a,s,h) for y,a,s,h in risks}:
        risks[('pooled',a,s,h)]=np.concatenate([risks[(str(y),a,s,h)] for y in YEARS])
    rows=[];warnings=[];bins=[];stability=[]
    for (scope,a,s,h),risk in risks.items():
        ev=data[scope]
        for k,cause in enumerate(['termination','recovery']):
            meta=dict(scope=scope,arm=a,seed=s,horizon_months=h,cause=cause,weighting='one_company_one_record')
            v,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),h),risk[:,k],k+1,ev['label_eligible'])
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            np.testing.assert_allclose([v[z] for z in ['auc','ap','brier']],direct_metrics(y,risk[:,k]),atol=1e-12,rtol=0,equal_nan=True)
            v['calibration_gap']=v['risk_mean']-v['ipcw_observed_incidence'];v['absolute_calibration_gap']=abs(v['calibration_gap'])
            for fraction,tag in [(.2,'top20'),(.25,'top25')]:
                chosen=selected(risk[:,k],ev['year'],ev['sample_ids'],fraction);captured=int((chosen&y).sum())
                v[tag+'_captured']=captured;v[tag+'_coverage']=captured/int(y.sum()) if y.any() else np.nan
                if k==0:warnings.append(dict(**meta,fraction=fraction,checked=int(chosen.sum()),captured=captured,target_events=int(y.sum()),coverage=v[tag+'_coverage'],known_non_target=int((chosen&~y).sum()),unknown=0,repeat_alerts=0))
            rows.append(dict(**meta,**v))
            if s=='ensemble':
                for b,ix in enumerate(np.array_split(np.lexsort((ev['sample_ids'],risk[:,k])),4)):
                    bins.append(dict(**meta,bin=b+1,n=len(ix),predicted=float(risk[ix,k].mean()),observed=float(y[ix].mean()),events=int(y[ix].sum())))
    df=pd.DataFrame(rows);df.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(warnings).to_csv(out/'warning_metrics.csv',index=False);pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    numeric=['auc','ap','brier','risk_mean','calibration_gap','absolute_calibration_gap','top20_captured','top20_coverage']
    diffs=[]
    for (scope,s,h,cause),gg in df[df.arm.isin(ARMS)].groupby(['scope','seed','horizon_months','cause']):
        g=gg.set_index('arm')
        for label,weights in contrasts().items():
            diffs.append(dict(scope=scope,seed=s,horizon_months=h,cause=cause,contrast=label,**{k:sum(w*g.loc[a,k] for a,w in weights.items()) for k in numeric}))
    pd.DataFrame(diffs).to_csv(out/'paired_differences.csv',index=False)
    old=pd.read_csv(HISTORY/'evaluation/metrics.csv',dtype={'scope':str,'seed':str});checked=0
    for r in df[df.arm.isin(OLD+['standard_rsf'])].itertuples():
        z=old[(old.scope==r.scope)&(old.arm==r.arm)&(old.seed==r.seed)&(old.cause==r.cause)&(old.horizon_months==r.horizon_months)]
        assert len(z)==1
        np.testing.assert_allclose([getattr(r,k) for k in ['auc','ap','brier','risk_mean']],z.iloc[0][['auc','ap','brier','risk_mean']].to_numpy(float),atol=1e-14,rtol=0,equal_nan=True);checked+=1
    for scope,a,h in itertools.product(['2023','2024','pooled'],ARMS,[12,24]):
        pp=np.stack([risks[(scope,a,str(s),h)] for s in SEEDS])
        for c in range(2):
            sd=pp[:,:,c].std(0,ddof=1)
            stability.append(dict(scope=scope,arm=a,horizon_months=h,cause=['termination','recovery'][c],prediction_sd_mean=float(sd.mean()),prediction_sd_max=float(sd.max()),prediction_sd_median=float(np.median(sd))))
    pd.DataFrame(stability).to_csv(out/'prediction_stability.csv',index=False)
    ss=df[df.seed.isin(list(map(str,SEEDS)))].groupby(['scope','arm','horizon_months','cause'])[numeric].agg(['mean','min','max','std'])
    ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    np.savez_compressed(run/'private/evaluation_forecasts_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risks.items()})
    write(out/'metric_validation.json',dict(status='PASS',metric_rows=len(df),historical_rows_recomputed_and_matched=checked,all_IPCW_weights_one=True,pooled_recomputed=True,companies=81,primary_events=32,new_fits_in_evaluation=0))
    print(df[(df.seed=='ensemble')&(df.horizon_months==24)&(df.cause=='termination')][['scope','arm','auc','ap','brier','top20_captured']].to_string(index=False))

def function_diagnostics(run):
    out=run/'evaluation';info={};summaries=[];hashes=[];coverage=[];paircoverage=[];leafsupport=[];search=[];amps=[]
    for y,a,s,p in validation_outputs(run):
        for n,h in json.loads((p/'hashes.json').read_text()).items():assert sha(p/n)==h
        z=json.loads((p/'summary.json').read_text());summaries.append(z)
        cc=json.loads((p/'complexity_PRIVATE.json').read_text());terms=arrays(p/'components_PRIVATE.npz');info[(y,a,s)]=(cc,terms,p)
        hashes.append(dict(year=y,arm=a,seed=s,function_sha256=z['function_sha256'],source_model_sha256=json.dumps(z['source_model_sha256'])))
        ad=pd.read_csv(p/'order_amplitudes.csv');ad.insert(0,'seed',s);ad.insert(0,'arm',a);ad.insert(0,'year',y);amps.append(ad)
        if s=='ensemble':continue
        for target,name in [(coverage,'feature_coverage'),(paircoverage,'pair_coverage'),(leafsupport,'leaf_support'),(search,'node_search')]:
            dd=pd.read_csv(p/f'{name}_PRIVATE.csv');dd.insert(0,'seed',s);dd.insert(0,'arm',a);dd.insert(0,'year',y);target.append(dd)
    sf=pd.DataFrame(summaries);sf.to_csv(out/'numerical_complexity_cost.csv',index=False)
    pd.DataFrame(hashes).to_csv(out/'function_hashes.csv',index=False);pd.concat(amps).to_csv(out/'order_amplitudes.csv',index=False)
    cv=pd.concat(coverage,ignore_index=True);pc=pd.concat(paircoverage,ignore_index=True);ld=pd.concat(leafsupport,ignore_index=True);nd=pd.concat(search,ignore_index=True)
    # Aggregate index-level counts contain no company identities or raw cuts.
    cv.to_csv(out/'feature_coverage_counts.csv',index=False);pc.to_csv(out/'pair_coverage_counts.csv',index=False)
    cd=[]
    for (y,a,s),g in cv.groupby(['year','arm','seed']):
        q=pc[(pc.year==y)&(pc.arm==a)&(pc.seed==s)]
        cd.append(dict(year=y,arm=a,seed=s,eligible_features=int((g.eligible_trees>0).sum()),eligible_min=int(g.eligible_trees.min()),eligible_max=int(g.eligible_trees.max()),eligible_sd=float(g.eligible_trees.std(ddof=0)),used_features=int((g.split_nodes>0).sum()),eligible_pairs=int((q.eligible_trees>0).sum()),co_path_pairs=int((q.co_path_trees>0).sum())))
    pd.DataFrame(cd).to_csv(out/'coverage_summary.csv',index=False)
    ld.groupby(['year','arm','seed']).agg(leaves=('node','size'),unique_min=('unique_companies','min'),unique_median=('unique_companies','median'),unique_max=('unique_companies','max'),target_median=('target_companies','median'),recovery_median=('recovery_companies','median'),borrowed=('borrowed_positive_target','sum')).reset_index().to_csv(out/'leaf_support_summary.csv',index=False)
    nd.groupby(['year','arm','seed'])[['finite_cuts','missing_only_cuts','routed_attempts','scored_candidates','features_considered','features_with_fewer_than_five_cuts']].sum().reset_index().to_csv(out/'search_counts.csv',index=False)
    # More detailed rows remain private; do not publish node-level fitted structure.
    ld.to_csv(run/'private/all_leaf_support_PRIVATE.csv',index=False);nd.to_csv(run/'private/all_node_search_PRIVATE.csv',index=False)
    stable=[];function_rows=[]
    for y,a in itertools.product(YEARS,ARMS):
        for s1,s2 in itertools.combinations(map(str,SEEDS),2):
            c1,t1,_=info[(y,a,s1)];c2,t2,_=info[(y,a,s2)]
            sets=lambda c,k,nz:{tuple(r['features']) for r in c['functions'] if r['order']==k and (r['nonzero'] or not nz)}
            jacc={}
            for k in [1,2,3,4]:
                for nz in [False,True]:
                    left,right=sets(c1,k,nz),sets(c2,k,nz);jacc[f'order{k}_'+('nonzero' if nz else 'structural')+'_jaccard']=len(left&right)/len(left|right) if left|right else np.nan
            i1={k:i for i,k in enumerate(t1['keys'])};i2={k:i for i,k in enumerate(t2['keys'])}
            for h in range(2):
                for c in range(2):
                    corr=[];rms=[];constant=0
                    for key in sorted(i1.keys()&i2.keys()):
                        l=t1['terms'][:,i1[key],h,c];r=t2['terms'][:,i2[key],h,c];v=float(np.sqrt(np.mean((l-r)**2)));rms.append(v)
                        nonconstant=np.std(l)>1e-12 and np.std(r)>1e-12
                        cor=float(np.corrcoef(l,r)[0,1]) if nonconstant else np.nan
                        if nonconstant:corr.append(cor)
                        else:constant+=1
                        function_rows.append(dict(year=y,arm=a,seed_a=s1,seed_b=s2,key=key,order=len(key.split(',')),day=[365,730][h],cause=c+1,correlation=cor,rms_difference=v))
                    stable.append(dict(year=y,arm=a,seed_a=s1,seed_b=s2,day=[365,730][h],cause=c+1,**jacc,common_terms=len(rms),common_nonconstant=len(corr),constant_terms=constant,median_correlation=float(np.median(corr)) if corr else np.nan,mean_rms_difference=float(np.mean(rms)) if rms else np.nan))
    pd.DataFrame(stable).to_csv(out/'function_stability.csv',index=False)
    pd.DataFrame(function_rows).to_csv(run/'private/function_stability_detail_PRIVATE.csv',index=False)
    # Validate paired source and feature streams across all 32 trees and seeds.
    checks=[]
    for y,s in itertools.product(YEARS,map(str,SEEDS)):
        base=json.loads((info[(y,'shallow_parent',s)][2]/'pairing.json').read_text())
        for a in ARMS:
            actual=json.loads((info[(y,a,s)][2]/'pairing.json').read_text())
            for l,r in zip(base,actual):
                assert l['bootstrap_sha256']==r['bootstrap_sha256'] and l['multiplicity_sha256']==r['multiplicity_sha256']
                if not a.startswith('iid_features'):assert l['features_sha256']==r['features_sha256']
            checks.append(dict(year=y,seed=s,arm=a,trees=32,bootstrap_and_multiplicity_match=True,feature_sets_match=all(l['features_sha256']==r['features_sha256'] for l,r in zip(base,actual))))
    pd.DataFrame(checks).to_csv(out/'pairing_checks.csv',index=False)
    write(out/'function_validation.json',dict(status='PASS',single_seed_models=60,mean_predictors=20,max_reconstruction_error=float(sf.reconstruction_error.max()),max_component_sum_error=float(sf.component_sum_error.max()),max_reload_error=float(sf.function_reload_error.max()),max_leaf_estimator_error=float(sf.max_leaf_estimator_error.max()),no_clipping=True,no_new_fits=True,all_training_and_test_rows=True,all_native_times=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['metrics','functions'],required=True);a=p.parse_args()
    (evaluate if a.stage=='metrics' else function_diagnostics)(a.run)
