"""All five cumulative stages: unchanged evaluator, full audit, paired intervals."""
from pathlib import Path
from dataclasses import asdict
import argparse,itertools,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from experiments.post_st_ablation_v1.analyze import selected
from experiments.post_st_ablation_v1.uncertainty import fast_metrics,METRICS
from .common import *

def analyze(run):
    out=run/'evaluation';out.mkdir(exist_ok=False)
    for n,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(n))==h
    for n,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/n)==h
    data={str(y):arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz') for y in YEARS}
    data['pooled']={k:np.concatenate([data[str(y)][k] for y in YEARS]) for k in data['2023']}
    predictions={};models={};model_hashes=[];costs=[]
    for r in registry(run):
        folder=Path(r['path']);y,a,s=r['year'],r['arm'],str(r['seed'])
        for name,h in json.loads((folder/'hashes.json').read_text()).items():assert sha(folder/name)==h
        p=arrays(folder/'predictions_PRIVATE.npz');np.testing.assert_array_equal(p['sample_ids'],data[str(y)]['sample_ids'])
        predictions[(str(y),a,s)]=p;f=CompetingRiskForest.load(folder/'forest_PRIVATE.json')
        assert asdict(f.config)==asdict(configuration(int(s),a));models[(y,a,s)]=f
        model_hashes.append(dict(year=y,arm=a,seed=s,reused=r['reuse'],model_sha256=sha(folder/'forest_PRIVATE.json'),prediction_sha256=sha(folder/'predictions_PRIVATE.npz')))
        meta=json.loads((folder/'summary.json').read_text());costs.append(dict(year=y,arm=a,seed=s,reused=r['reuse'],host=meta.get('host'),forest_fit_seconds=meta.get('forest_fit_seconds',meta.get('base_fit_seconds')),shrinkage_seconds=meta.get('shrinkage_seconds',meta.get('leaf_update_seconds')),prediction_seconds=meta.get('prediction_seconds'),source_model_export_seconds=meta.get('model_export_seconds')))
    assert len(models)==30
    for y,a in itertools.product(YEARS,STAGES):
        members=[predictions[(str(y),a,str(s))] for s in SEEDS]
        assert all(np.array_equal(p['times'],members[0]['times']) for p in members)
        predictions[(str(y),a,'ensemble')]=dict(times=members[0]['times'],cif=np.mean([p['cif'] for p in members],axis=0))
    risks={}
    for (y,a,s),p in predictions.items():
        for h in [12,24]:risks[(y,a,s,h)]=horizon_predictions(p['cif'],p['times'],data[y][f'h{h}'])
    for a,s,h in {(a,s,h) for y,a,s,h in risks}:
        risks[('pooled',a,s,h)]=np.concatenate([risks[(str(y),a,s,h)] for y in YEARS])
    rows=[];warnings=[];bins=[]
    for (scope,a,s,h),risk in sorted(risks.items()):
        ev=data[scope]
        for k,cause in enumerate(['termination','recovery']):
            meta=dict(scope=scope,arm=a,seed=s,horizon_months=h,cause=cause)
            result,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),h),risk[:,k],k+1,ev['label_eligible'])
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            np.testing.assert_allclose([result[z] for z in ['auc','ap','brier']],direct_metrics(y,risk[:,k]),atol=1e-12,rtol=0,equal_nan=True)
            result['calibration_gap']=result['risk_mean']-result['ipcw_observed_incidence'];result['absolute_calibration_gap']=abs(result['calibration_gap'])
            for fraction,tag in [(.2,'top20'),(.25,'top25')]:
                chosen=selected(risk[:,k],ev['year'],ev['sample_ids'],fraction);hit=int((chosen&y).sum())
                result[tag+'_captured']=hit;result[tag+'_coverage']=hit/int(y.sum()) if y.any() else np.nan
                if k==0:warnings.append(dict(**meta,fraction=fraction,checked=int(chosen.sum()),captured=hit,total_events=int(y.sum()),coverage=result[tag+'_coverage'],known_non_target=int((chosen&~y).sum()),unknown=0))
            rows.append(dict(**meta,**result))
            if s=='ensemble':
                for b,ix in enumerate(np.array_split(np.lexsort((ev['sample_ids'],risk[:,k])),4)):
                    bins.append(dict(**meta,bin=b+1,n=len(ix),risk_mean=float(risk[ix,k].mean()),observed=float(y[ix].mean()),events=int(y[ix].sum())))
    df=pd.DataFrame(rows);assert len(df)==240
    df.to_csv(out/'metrics.csv',index=False);pd.DataFrame(warnings).to_csv(out/'warning_metrics.csv',index=False);pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    old=pd.read_csv(PREVIOUS/'evaluation/metrics.csv',dtype={'scope':str,'seed':str});matched=0
    for row in df[df.arm.isin(['s0','s1'])].itertuples():
        name={'s0':'shallow_parent','s1':'q_unrestricted'}[row.arm]
        z=old[(old.scope==row.scope)&(old.arm==name)&(old.seed==row.seed)&(old.horizon_months==row.horizon_months)&(old.cause==row.cause)]
        assert len(z)==1
        np.testing.assert_allclose([getattr(row,k) for k in METRICS],z.iloc[0][METRICS].to_numpy(float),atol=1e-14,rtol=0,equal_nan=True);matched+=1
    differences=[]
    for (scope,s,h,cause),g in df.groupby(['scope','seed','horizon_months','cause']):
        g=g.set_index('arm')
        for name,weights in contrasts().items():differences.append(dict(scope=scope,seed=s,horizon_months=h,cause=cause,contrast=name,**{k:sum(w*g.loc[a,k] for a,w in weights.items()) for k in METRICS}))
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    predsd=[]
    for scope,a,h in itertools.product(['2023','2024','pooled'],STAGES,[12,24]):
        pp=np.stack([risks[(scope,a,str(s),h)] for s in SEEDS]);sd=pp.std(0,ddof=1)
        for k in range(2):predsd.append(dict(scope=scope,arm=a,horizon_months=h,cause=k+1,seed_sd_mean=float(sd[:,k].mean()),seed_sd_max=float(sd[:,k].max())))
    pd.DataFrame(predsd).to_csv(out/'prediction_stability.csv',index=False)
    ss=df[df.seed!='ensemble'].groupby(['scope','arm','horizon_months','cause'])[METRICS].agg(['min','mean','max','std']);ss.columns=['_'.join(c) for c in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    np.savez_compressed(run/'private/forecasts_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in risks.items()})
    pd.DataFrame(model_hashes).to_csv(out/'model_hashes.csv',index=False);pd.DataFrame(costs).to_csv(out/'model_costs.csv',index=False)
    # Validate pairing and verify that the final step changes only the leaf CIFs.
    paired=[];geometry=[]
    strip=lambda f:[[{k:v for k,v in n.items() if k not in ['cif','leaf_shrinkage']} for n in t] for t in f.trees_]
    for y,s in itertools.product(YEARS,map(str,SEEDS)):
        tr=arrays(run/'input/folds'/str(y)/'train_PRIVATE.npz');base=pairing(tr['groups'],models[(y,'s0',s)].config)
        for a in STAGES:
            p=pairing(tr['groups'],models[(y,a,s)].config)
            assert all(l['bootstrap_sha256']==r['bootstrap_sha256'] and l['multiplicity_sha256']==r['multiplicity_sha256'] for l,r in zip(base,p))
            if a in ['s0','s1','s2']:assert p==base
            paired.append(dict(year=y,arm=a,seed=s,trees=32,bootstrap_equal=True,features_equal_to_s0=all(l['features_sha256']==r['features_sha256'] for l,r in zip(base,p))))
        assert strip(models[(y,'s3',s)])==strip(models[(y,'s4',s)])
        assert all('leaf_shrinkage' not in n for t in models[(y,'s4',s)].trees_ for n in t)
        geometry.append(dict(year=y,seed=s,s3_s4_geometry_identical=True,s4_unshrunk=True))
    pd.DataFrame(paired).to_csv(out/'pairing_checks.csv',index=False);pd.DataFrame(geometry).to_csv(out/'final_step_geometry.csv',index=False)
    diagnostics(run)
    bootstrap(run,risks,data)
    write(out/'validation.json',dict(status='PASS',new_forest_fits=12,new_shrunk_models=12,reused_chain_models=12,chain_single_models=30,chain_ensembles=10,matched_historical_metric_rows=matched,metric_rows=len(df),all_test_endpoints_known=True,IPCW_weights_one=True,independent_metrics_verified=True,reference_unchanged=True))
    print(df[(df.seed=='ensemble')&(df.horizon_months==24)&(df.cause=='termination')][['scope','arm','auc','ap','brier','top20_captured']].to_string(index=False))

def diagnostics(run):
    out=run/'evaluation';summaries=[];artifacts={};coverage=[];amps=[]
    for y,a,s,p in verification_outputs(run):
        for n,h in json.loads((p/'hashes.json').read_text()).items():assert sha(p/n)==h
        row=json.loads((p/'summary.json').read_text());row['complete_cells_with_baseline']=row['spatial_table_cells']+1
        row['complete_scalars_with_baseline']=row['scalar_table_cells']+row['native_time_count']*2;summaries.append(row)
        artifacts[(y,a,s)]=(json.loads((p/'complexity_PRIVATE.json').read_text()),arrays(p/'components_PRIVATE.npz'))
        aa=pd.read_csv(p/'order_amplitudes.csv');aa.insert(0,'seed',s);aa.insert(0,'arm',a);aa.insert(0,'year',y);amps.append(aa)
        if s!='ensemble':
            cv=pd.read_csv(p/'feature_coverage_PRIVATE.csv');coverage.append(dict(year=y,arm=a,seed=s,eligible_min=int(cv.eligible_trees.min()),eligible_max=int(cv.eligible_trees.max()),eligible_sd=float(cv.eligible_trees.std(ddof=0)),actual_split_features=int((cv.split_nodes>0).sum())))
    cc=pd.DataFrame(summaries);assert len(cc)==40
    cc.to_csv(out/'numerical_complexity_cost.csv',index=False);cc[cc.seed=='ensemble'].to_csv(out/'ensemble_complexity.csv',index=False);cc[cc.seed!='ensemble'].to_csv(out/'single_seed_complexity.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage_summary.csv',index=False);pd.concat(amps).to_csv(out/'order_amplitudes.csv',index=False)
    stability=[]
    for y,a in itertools.product(YEARS,STAGES):
        for sa,sb in itertools.combinations(map(str,SEEDS),2):
            ca,ta=artifacts[(y,a,sa)];cb,tb=artifacts[(y,a,sb)];ia={k:i for i,k in enumerate(ta['keys'])};ib={k:i for i,k in enumerate(tb['keys'])}
            for order in range(1,5):
                left={tuple(r['features']) for r in ca['functions'] if r['order']==order and r['nonzero']};right={tuple(r['features']) for r in cb['functions'] if r['order']==order and r['nonzero']}
                keys=[k for k in ia.keys()&ib.keys() if len(k.split(','))==order]
                for hi in range(2):
                    for k in range(2):
                        cor=[];rms=[]
                        for key in keys:
                            l=ta['terms'][:,ia[key],hi,k];r=tb['terms'][:,ib[key],hi,k];rms.append(float(np.sqrt(np.mean((l-r)**2))))
                            if np.std(l)>1e-12 and np.std(r)>1e-12:cor.append(float(np.corrcoef(l,r)[0,1]))
                        stability.append(dict(year=y,arm=a,seed_a=sa,seed_b=sb,order=order,day=[365,730][hi],cause=k+1,nonzero_jaccard=len(left&right)/len(left|right) if left|right else np.nan,common_terms=len(keys),nonconstant_terms=len(cor),median_correlation=float(np.median(cor)) if cor else np.nan,mean_rms_difference=float(np.mean(rms)) if rms else np.nan))
    pd.DataFrame(stability).to_csv(out/'function_stability.csv',index=False)
    executions=[]
    for stage in ['formal','verify']:
        gate=json.loads((run/f'audit/{stage}_gate.json').read_text());job=gate['job']
        report=json.loads((run/f'{stage}.jsonl.results-{job}.json').read_text())
        for item in report['completed']:
            folder=run/'results'/stage/str(job)/f'task-{item["id"]}/output';z=json.loads((folder/'summary.json').read_text())
            executions.append(dict(stage=stage,job=job,task_id=item['id'],year=z['year'],arm=z.get('arm',z.get('design_arm')),seed=z['seed'],host=z['host'],seconds=z['seconds'],peak_rss_MiB=z['peak_rss_MiB'],new_fits=z['new_forest_fits'],returncode=item['returncode'],publication_completed=item['publication_completed']))
    ed=pd.DataFrame(executions);assert ed.new_fits.sum()==12 and len(ed)==52 and ed.publication_completed.all() and ed.returncode.max()==0
    ed.to_csv(out/'execution_tasks.csv',index=False)
    write(out/'function_validation.json',dict(status='PASS',models=30,ensembles=10,all_training_test_rows_and_native_times=True,both_causes=True,max_component_sum_error=float(cc.component_sum_error.max()),max_reconstruction_error=float(cc.reconstruction_error.max()),max_reload_error=float(cc.function_reload_error.max()),max_leaf_estimator_error=float(cc.max_leaf_estimator_error.max()),no_clipping=True))

def bootstrap(run,risks,data):
    idx=arrays(run/'private/bootstrap_indices_PRIVATE.npz');expected=arrays(PREVIOUS/'private/bootstrap_indices_PRIVATE.npz')
    for k in idx:np.testing.assert_array_equal(idx[k],expected[k])
    weights=contrasts();rows=[];previousdiff=pd.read_csv(run/'evaluation/paired_differences.csv',dtype={'scope':str,'seed':str})
    for scope in ['2023','2024','pooled']:
        years=list(map(str,YEARS)) if scope=='pooled' else [scope];ev=data[scope]
        y=(ev['event']==1)&(ev['analysis_time']<=24);yy=ev['year'];ids=ev['sample_ids']
        pp=np.array([risks[(scope,a,'ensemble',24)][:,0] for a in STAGES])
        points=np.array([fast_metrics(y,p,yy,ids) for p in pp]);values=np.empty((1000,len(weights),len(METRICS)))
        for b in range(1000):
            ix=np.concatenate([idx[s][b]+sum(len(data[z]['X']) for z in years[:i]) for i,s in enumerate(years)])
            result=np.array([fast_metrics(y[ix],p[ix],yy[ix],ids[ix]) for p in pp])
            values[b]=np.array([sum(w*result[STAGES.index(a)] for a,w in ww.items()) for ww in weights.values()])
        for j,(name,ww) in enumerate(weights.items()):
            point=sum(w*points[STAGES.index(a)] for a,w in ww.items())
            z=previousdiff[(previousdiff.scope==scope)&(previousdiff.seed=='ensemble')&(previousdiff.horizon_months==24)&(previousdiff.cause=='termination')&(previousdiff.contrast==name)]
            np.testing.assert_allclose(point,z.iloc[0][METRICS].to_numpy(float),atol=1e-13,rtol=0)
            for k,m in enumerate(METRICS):
                v=values[:,j,k];ok=np.isfinite(v);lo,hi=np.quantile(v[ok],[.025,.975]) if ok.any() else [np.nan,np.nan]
                rows.append(dict(scope=scope,contrast=name,metric=m,estimate=float(point[k]),lower95=float(lo),upper95=float(hi),valid_replicates=int(ok.sum()),undefined_replicates=int((~ok).sum())))
        np.savez_compressed(run/'private'/f'bootstrap_differences_{scope}_PRIVATE.npz',values=values,contrasts=np.array(list(weights)),metrics=np.array(METRICS))
    pd.DataFrame(rows).to_csv(run/'evaluation/paired_bootstrap_intervals.csv',index=False)
    write(run/'evaluation/bootstrap_validation.json',dict(status='PASS',replicates=1000,analysis_seed=2026102901,previous_indices_reused_exactly=True,indices_sha256=sha(run/'private/bootstrap_indices_PRIVATE.npz'),company_unit=True,year_stratified=True,all_stages_common_indices=True,forest_fits=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);analyze(p.parse_args().run)
