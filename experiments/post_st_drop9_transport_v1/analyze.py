"""Evaluate saved annual predictions without any further fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge,replay
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from experiments.post_st_drop9_v1.analyze import warning
from .common import *


def analyze(run,repo):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS' and gate['tasks']==6
    protocol=json.loads((run/'input/protocol.json').read_text())
    rows=[];warnings=[];bins=[];checks=[];tasks=[];hashes=[];support=[];leaf_support=[];usage=[];spread=[];contrasts=[];boot=[]
    for year in YEARS:
        root=run/f'input/folds/{year}';ev=arrays(run/f'evaluation_input/{year}_test_PRIVATE.npz')
        meta=pd.read_parquet(run/f'evaluation_input/{year}_test_metadata_PRIVATE.parquet')
        old=arrays(root/'reference_train_PRIVATE.npz');tr=arrays(root/'train_PRIVATE.npz');xx=arrays(root/'eval_PRIVATE.npz')
        np.testing.assert_array_equal(ev['sample_ids'],xx['sample_ids']);np.testing.assert_array_equal(ev['sample_ids'],meta.sample_id)
        assert meta.group.is_unique and not set(ev['groups'])&set(old['groups'])
        assert (meta.origin.dt.year==year).all()
        if year==2025:assert (meta.stop<=pd.Timestamp('2026-06-30')).all()
        fmap=json.loads((root/'feature_map.json').read_text());keep=[v['original_index'] for v in fmap if not v['removed']]
        np.testing.assert_array_equal(ev['X'][:,keep],xx['X'])
        models={};pred={};risks={}
        for i,seed in enumerate(SEEDS):
            reference=root/f'reference/{seed}'
            new=run/'results/formal'/str(gate['job'])/f'task-{YEARS.index(year)*3+i}/output'
            s=json.loads((new/'summary.json').read_text());assert s['year']==year and s['seed']==seed
            tasks.append({k:v for k,v in s.items() if k not in ['checks','replay_checks']})
            for kind in ['raw','shrunk']:checks.append(dict(year=year,arm='drop9_'+kind,seed=str(seed),**s['checks'][kind]))
            for arm,folder in [('reference83',reference),('drop9',new/'shrunk')]:
                f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');models[(arm,str(seed))]=f
                z=arrays(folder/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['times'],TIMES)
                np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);pred[(arm,str(seed))]=z['cif']
                np.testing.assert_array_equal(f.predict(ev['X'] if arm=='reference83' else xx['X'],TIMES),z['cif'])
                hashes.append(dict(year=year,arm=arm,seed=seed,forest_sha256=sha(folder/'forest_PRIVATE.json'),
                    prediction_sha256=sha(folder/'predictions_PRIVATE.npz')))
                if arm=='reference83':
                    result=verify_export(f,old,dict(X=ev['X'],sample_ids=ev['sample_ids']),run/'private'/f'verify_{year}_reference_{seed}')
                    checks.append(dict(year=year,arm=arm,seed=str(seed),**result))
                    _,_,_,ld,rc=replay(f,old);assert max(rc.values())<1e-10;ld=pd.DataFrame(ld)
                else:ld=pd.read_csv(new/'leaf_support_PRIVATE.csv')
                leaf_support.append(dict(year=year,arm=arm,seed=seed,leaves=len(ld),
                    min_unique_companies=int(ld.unique_companies.min()),median_unique_companies=float(ld.unique_companies.median()),
                    zero_target_leaves=int((ld.target_companies==0).sum()),borrowed_positive_target_leaves=int(ld.borrowed_positive_target.sum())))
                for j in range(f.n_features_):
                    jj=j if arm=='reference83' else keep[j]
                    usage.append(dict(year=year,arm=arm,seed=seed,name=fmap[jj]['name'],label=fmap[jj]['label'],original_index=jj,
                        split_trees=sum(any(n.get('feature')==j for n in tree) for tree in f.trees_),
                        split_nodes=sum(n.get('feature')==j for tree in f.trees_ for n in tree)))
        for arm in ['reference83','drop9']:
            members=[models[(arm,str(s))] for s in SEEDS];f=merge(members)
            pred[(arm,'ensemble')]=np.mean([pred[(arm,str(s))] for s in SEEDS],axis=0)
            data=old if arm=='reference83' else tr;test=dict(X=ev['X'] if arm=='reference83' else xx['X'],sample_ids=ev['sample_ids'])
            z=verify_export(f,data,test,run/'private'/f'verify_{year}_{arm}_ensemble')
            checks.append(dict(year=year,arm=arm,seed='ensemble',**z))
            np.testing.assert_allclose(f.predict(test['X'],TIMES),pred[(arm,'ensemble')],atol=1e-13,rtol=0)
        for h in [12,24]:
            hs=horizon_support(ev,h);support.append(dict(year=year,**hs))
            for (arm,seed),p in pred.items():
                risk=horizon_predictions(p,TIMES,ev[f'h{h}']);risks[(arm,seed,h)]=risk
                for cause in [1,2]:
                    tags=dict(year=year,arm=arm,seed=seed,horizon_months=h,cause=cause)
                    if not hs['supported']:
                        rows.append(dict(**tags,status='UNSUPPORTED_HORIZON',companies=len(risk),auc=np.nan,ap=np.nan,brier=np.nan,
                            risk_mean=float(risk[:,cause-1].mean()),unknown_outcomes=hs['unknown'],calibration_gap=np.nan))
                        continue
                    m,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),h),risk[:,cause-1],cause,ev['label_eligible'])
                    assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
                    np.testing.assert_allclose([m[k] for k in ['auc','ap','brier']],direct_metrics(y,risk[:,cause-1]),atol=1e-12,rtol=0,equal_nan=True)
                    rows.append(dict(**tags,**m,status='SUPPORTED',calibration_gap=m['risk_mean']-m['ipcw_observed_incidence']))
                    if cause==1:warnings.append(dict(**tags,**warning(risk[:,0],ev['event'],ev['analysis_time'],ev['sample_ids'],h)))
                    if seed=='ensemble':
                        for q,ix in enumerate(np.array_split(np.argsort(risk[:,cause-1],kind='stable'),4)):
                            bins.append(dict(**tags,quartile=q+1,n=len(ix),predicted=float(risk[ix,cause-1].mean()),observed=float(y[ix].mean()),events=int(y[ix].sum())))
            for arm in ['reference83','drop9']:
                sd=np.stack([risks[(arm,str(s),h)][:,0] for s in SEEDS]).std(0,ddof=1)
                spread.append(dict(year=year,arm=arm,horizon_months=h,company_seed_prediction_sd_mean=float(sd.mean()),company_seed_prediction_sd_max=float(sd.max())))
        # One common company-resampling matrix per year, also reused across supported horizons.
        rng=np.random.default_rng(np.random.SeedSequence([protocol['uncertainty']['seed'],year]));n=len(ev['X'])
        draws=rng.integers(n,size=(protocol['uncertainty']['repeats'],n))
        np.savez_compressed(run/f'private/bootstrap_{year}_PRIVATE.npz',indices=draws)
        np.savez_compressed(run/f'private/risks_{year}_PRIVATE.npz',**{'_'.join(map(str,k)):v for k,v in risks.items()})
        for h in [12,24]:
            if not horizon_support(ev,h)['supported']:continue
            local=[]
            for i,ix in enumerate(draws):
                values=[]
                for arm in ['reference83','drop9']:
                    p=risks[(arm,'ensemble',h)][:,0]
                    m,_,_,_=metric(ev['analysis_time'][ix],ev['event'][ix],np.full(len(ix),h),p[ix],1,ev['label_eligible'][ix])
                    m.update(warning(p[ix],ev['event'][ix],ev['analysis_time'][ix],ev['sample_ids'][ix],h));values.append(m)
                local.append(dict(year=year,horizon_months=h,replicate=i,**{k:values[1][k]-values[0][k] for k in ['auc','ap','brier','risk_mean','captured','coverage']}))
            boot.extend(local);bd=pd.DataFrame(local)
            for k in ['auc','ap','brier','risk_mean','captured','coverage']:
                means=[]
                for arm in ['reference83','drop9']:
                    candidates=warnings if k in ['captured','coverage'] else rows
                    r=next(r for r in candidates if r['year']==year and r['horizon_months']==h and r['cause']==1 and r['seed']=='ensemble' and r['arm']==arm)
                    means.append(r[k])
                valid=bd[k].dropna();lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
                contrasts.append(dict(year=year,horizon_months=h,metric=k,reference=means[0],drop9=means[1],difference=means[1]-means[0],lower95=lo,upper95=hi,
                    undefined_bootstraps=int(bd[k].isna().sum()),direction='negative improves' if k=='brier' else 'level only' if k=='risk_mean' else 'positive improves'))
    df=pd.DataFrame(rows)
    for year in YEARS:
        prior=pd.read_csv(repo/('experiments/post_st_leaf_shallow_v1/results/metrics.csv' if year==2023 else 'experiments/post_st_2025_shallow_parent_v1/results/metrics.csv'),dtype={'scope':str,'seed':str})
        for r in df[(df.year==year)&(df.arm=='reference83')].itertuples():
            mask=(prior.seed==r.seed)&(prior.horizon_months==r.horizon_months)&(prior.cause==('termination' if r.cause==1 else 'recovery'))
            mask&=((prior.scope=='2023')&(prior.arm=='shallow_parent')) if year==2023 else (prior.model=='xai_q2')
            assert mask.sum()==1
            np.testing.assert_allclose([getattr(r,k) for k in ['auc','ap','brier','risk_mean']],prior.loc[mask,['auc','ap','brier','risk_mean']].iloc[0].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True)
    for name,table in [('metrics',df),('warnings',pd.DataFrame(warnings)),('horizon_support',pd.DataFrame(support)),('calibration_bins',pd.DataFrame(bins)),
        ('function_validation',pd.DataFrame(checks)),('execution_tasks',pd.DataFrame(tasks)),('model_hashes',pd.DataFrame(hashes)),
        ('leaf_support_summary',pd.DataFrame(leaf_support)),('actual_feature_usage',pd.DataFrame(usage)),('prediction_seed_variation',pd.DataFrame(spread)),
        ('paired_differences',pd.DataFrame(contrasts)),('bootstrap_differences',pd.DataFrame(boot))]:table.to_csv(run/f'reports/{name}.csv',index=False)
    for p,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(p))==h
    for p,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/p)==h
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    error=max(max(r[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload']) for r in checks)
    assert df[(df.year==2025)&(df.horizon_months==24)][['auc','ap','brier']].isna().all().all()
    write(run/'audit/validation.json',dict(status='PASS',new_forests=6,derived_models=6,reused_forests=6,formal_tasks=6,reference_metric_rows_reproduced=32,
        source_hashes_unchanged=True,current_reference_unchanged=True,bootstrap_trees_paired=192,bootstrap_repeats_per_year=1000,
        max_explanation_error=error,unsupported_2025_24m_suppressed=True,no_new_calibration_or_selection=True))
    print(df[(df.seed=='ensemble')&(df.cause==1)][['year','arm','horizon_months','auc','ap','brier','risk_mean','status']].to_string(index=False))
    print(pd.DataFrame(warnings).query("seed=='ensemble'")[['year','arm','horizon_months','captured','target_events','coverage']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    analyze(a.run,Path(__file__).resolve().parents[2])
