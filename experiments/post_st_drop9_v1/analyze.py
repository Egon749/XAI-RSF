"""One locked2024 comparison; preserve all seeds and unfavorable metrics."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .common import *


def warning(risk,event,time,ids,h=24):
    order=np.lexsort((ids,-risk));chosen=order[:int(np.ceil(.2*len(risk)))];target=(event==1)&(time<=h)
    known=(event>0)|(time>=h);captured=int(target[chosen].sum())
    return dict(checked=len(chosen),captured=captured,target_events=int(target.sum()),coverage=captured/target.sum() if target.any() else np.nan,
        known_other_checked=int((known[chosen]&~target[chosen]).sum()),unknown_checked=int((~known[chosen]).sum()))


def analyze(run,repo):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS' and gate['tasks']==3
    out=run/'reports';evals=arrays(run/'evaluation_input/test_PRIVATE.npz');evx=arrays(run/'input/eval_PRIVATE.npz')
    np.testing.assert_array_equal(evals['sample_ids'],evx['sample_ids'])
    original=arrays(run/'input/reference_train_PRIVATE.npz');reducedtr=arrays(run/'input/train_PRIVATE.npz')
    assert not set(evals['groups'])&set(original['groups']) and len(evals['X'])==45
    schema=json.loads((run/'input/feature_map.json').read_text());keep=[r['original_index'] for r in schema if not r['removed']]
    np.testing.assert_array_equal(evals['X'][:,keep],evx['X'])
    models={};pred={};validation=[];fits=[];features=[];source_models=[]
    for seed in SEEDS:
        p=run/'input/reference'/str(seed);models[('reference83',str(seed))]=CompetingRiskForest.load(p/'forest_PRIVATE.json')
        saved=arrays(p/'predictions_PRIVATE.npz');np.testing.assert_array_equal(saved['times'],TIMES)
        np.testing.assert_array_equal(saved['sample_ids'],evals['sample_ids']);pred[('reference83',str(seed))]=saved['cif']
        np.testing.assert_array_equal(models[('reference83',str(seed))].predict(evals['X'],TIMES),saved['cif'])
        source_models.append(dict(arm='reference83',seed=seed,forest_sha256=sha(p/'forest_PRIVATE.json'),predictions_sha256=sha(p/'predictions_PRIVATE.npz')))
    for i,seed in enumerate(SEEDS):
        p=run/'results/formal'/str(gate['job'])/f'task-{i}/output';s=json.loads((p/'summary.json').read_text());assert s['seed']==seed
        fits.append({k:v for k,v in s.items() if k not in ['checks','replay_checks']})
        for kind in ['raw','shrunk']:
            validation.append(dict(arm='drop9_'+kind,seed=str(seed),**s['checks'][kind]))
        models[('drop9',str(seed))]=CompetingRiskForest.load(p/'shrunk/forest_PRIVATE.json')
        z=arrays(p/'shrunk/predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],evals['sample_ids']);pred[('drop9',str(seed))]=z['cif']
        source_models.append(dict(arm='drop9',seed=seed,forest_sha256=sha(p/'shrunk/forest_PRIVATE.json'),predictions_sha256=sha(p/'shrunk/predictions_PRIVATE.npz'),raw_forest_sha256=sha(p/'raw/forest_PRIVATE.json')))
    for arm in ['reference83','drop9']:
        tr=original if arm=='reference83' else reducedtr
        ev=dict(X=evals['X'] if arm=='reference83' else evx['X'],sample_ids=evals['sample_ids'])
        for seed in SEEDS:
            f=models[(arm,str(seed))]
            for j in range(f.n_features_):
                original_j=j if arm=='reference83' else keep[j]
                features.append(dict(arm=arm,seed=seed,original_index=original_j,name=schema[original_j]['name'],label=schema[original_j]['label'],
                    split_trees=sum(any(n.get('feature')==j for n in tree) for tree in f.trees_),split_nodes=sum(n.get('feature')==j for tree in f.trees_ for n in tree)))
            if arm=='reference83':
                z=verify_export(f,tr,ev,run/'private'/f'verify_reference_{seed}');validation.append(dict(arm=arm,seed=str(seed),**z))
        members=[models[(arm,str(s))] for s in SEEDS]
        pred[(arm,'ensemble')]=np.mean([pred[(arm,str(s))] for s in SEEDS],axis=0)
        joined=merge(members);z=verify_export(joined,tr,ev,run/'private'/f'verify_{arm}_ensemble')
        np.testing.assert_allclose(joined.predict(ev['X'],TIMES),pred[(arm,'ensemble')],atol=1e-13,rtol=0)
        validation.append(dict(arm=arm,seed='ensemble',**z))
    rows=[];warnings=[];bins=[];risks={}
    for (arm,seed),p in pred.items():
        for h in [12,24]:
            risk=horizon_predictions(p,TIMES,evals[f'h{h}']);risks[(arm,seed,h)]=risk
            for c in [1,2]:
                tags=dict(year=2024,arm=arm,seed=seed,horizon_months=h,cause=c)
                result,w,y,known=metric(evals['analysis_time'],evals['event'],np.full(len(risk),h),risk[:,c-1],c,evals['label_eligible'])
                assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
                np.testing.assert_allclose([result[k] for k in ['auc','ap','brier']],direct_metrics(y,risk[:,c-1]),atol=1e-12,rtol=0,equal_nan=True)
                rows.append(dict(**tags,**result,calibration_gap=result['risk_mean']-result['ipcw_observed_incidence']))
                if c==1:warnings.append(dict(**tags,**warning(risk[:,0],evals['event'],evals['analysis_time'],evals['sample_ids'],h)))
                if seed=='ensemble':
                    for i,ix in enumerate(np.array_split(np.argsort(risk[:,c-1],kind='stable'),4)):
                        bins.append(dict(**tags,bin=i+1,n=len(ix),predicted=float(risk[ix,c-1].mean()),observed=float(y[ix].mean()),events=int(y[ix].sum())))
    df=pd.DataFrame(rows);wf=pd.DataFrame(warnings)
    prior=pd.read_csv(repo/'experiments/post_st_leaf_shallow_v1/results/metrics.csv',dtype={'scope':str,'seed':str})
    for r in df[df.arm=='reference83'].itertuples():
        old=prior[(prior.scope=='2024')&(prior.arm=='shallow_parent')&(prior.seed==r.seed)&(prior.horizon_months==r.horizon_months)&(prior.cause==('termination' if r.cause==1 else 'recovery'))]
        assert len(old)==1
        np.testing.assert_allclose([getattr(r,k) for k in ['auc','ap','brier','risk_mean']],old.iloc[0][['auc','ap','brier','risk_mean']].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
    for name,table in [('metrics',df),('warnings',wf),('calibration_bins',pd.DataFrame(bins)),('function_validation',pd.DataFrame(validation)),
            ('execution_tasks',pd.DataFrame(fits)),('actual_feature_usage',pd.DataFrame(features)),('model_hashes',pd.DataFrame(source_models))]:table.to_csv(out/f'{name}.csv',index=False)
    protocol=json.loads((run/'input/protocol.json').read_text());rng=np.random.default_rng(protocol['uncertainty']['seed'])
    draws=rng.integers(45,size=(protocol['uncertainty']['repeats'],45));np.savez_compressed(run/'private/bootstrap_indices_PRIVATE.npz',indices=draws)
    boot=[];old=risks[('reference83','ensemble',24)][:,0];new=risks[('drop9','ensemble',24)][:,0]
    for i,ix in enumerate(draws):
        values=[]
        for p in [old,new]:
            result,_,_,_=metric(evals['analysis_time'][ix],evals['event'][ix],np.full(len(ix),24),p[ix],1,evals['label_eligible'][ix])
            result.update(warning(p[ix],evals['event'][ix],evals['analysis_time'][ix],evals['sample_ids'][ix]))
            values.append(result)
        boot.append(dict(replicate=i,**{k:values[1][k]-values[0][k] for k in ['auc','ap','brier','risk_mean','captured','coverage']}))
    bd=pd.DataFrame(boot);bd.to_csv(out/'bootstrap_differences.csv',index=False)
    contrast=[];main=df[(df.seed=='ensemble')&(df.horizon_months==24)&(df.cause==1)].set_index('arm')
    mainw=wf[(wf.seed=='ensemble')&(wf.horizon_months==24)].set_index('arm')
    for k in ['auc','ap','brier','risk_mean','captured','coverage']:
        vals=mainw[k] if k in ['captured','coverage'] else main[k];valid=bd[k].dropna()
        lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
        contrast.append(dict(metric=k,reference=vals['reference83'],drop9=vals['drop9'],difference=vals['drop9']-vals['reference83'],lower95=lo,upper95=hi,undefined_bootstraps=int(bd[k].isna().sum()),
            direction='negative improves' if k=='brier' else 'not a performance direction by itself' if k=='risk_mean' else 'positive improves'))
    pd.DataFrame(contrast).to_csv(out/'paired_differences.csv',index=False)
    # Save company forecasts only in private storage.
    np.savez_compressed(run/'private/risks_PRIVATE.npz',**{'_'.join(map(str,k)):v for k,v in risks.items()})
    for p,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(p))==h
    for p,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/p)==h
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    maxerror=max(max(r[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload']) for r in validation)
    receipt=dict(status='PASS',new_forests=3,derived_models=3,original_forests_reused=3,formal_tasks=3,
        source_hashes_unchanged=True,current_reference_unchanged=True,bootstrap_indices_paired=True,bootstrap_repeats=1000,reference_metric_rows_reproduced=16,
        evaluation_companies=45,target_events24=int(((evals['event']==1)&(evals['analysis_time']<=24)).sum()),unknown24=0,
        training_companies=345,training_target_events=26,maximum_explanation_error=maxerror,original_code_pit_status='PROVISIONAL')
    write(run/'audit/validation.json',receipt)
    print(main[['auc','ap','brier','risk_mean']].to_string());print(mainw[['checked','captured','coverage']].to_string());print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run,Path(__file__).resolve().parents[2])
