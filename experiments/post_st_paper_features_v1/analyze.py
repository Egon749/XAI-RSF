"""Fixed-prediction annual12-month comparisons, no fitting in analysis."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from experiments.post_st_drop9_v1.analyze import warning
from .common import *


def analyze(run,repo):
    gate=json_load(run/'audit/formal_gate.json');assert gate['status']=='PASS' and gate['tasks']==9
    protocol=json_load(run/'input/protocol.json');arms=['reference83','drop9','paper20']
    rows=[];alerts=[];bins=[];checks=[];fits=[];hashes=[];complexities=[];leaves=[];spreads=[];boots=[];diffs=[]
    for year in YEARS:
        root=run/f'input/folds/{year}';tr=arrays(root/'train_PRIVATE.npz');evx=arrays(root/'eval_PRIVATE.npz')
        old=arrays(root/'reference_train_PRIVATE.npz');ev=arrays(run/f'evaluation_input/{year}_test_PRIVATE.npz')
        meta=pd.read_parquet(run/f'evaluation_input/{year}_test_metadata_PRIVATE.parquet');plan=json_load(root/'plan.json')
        np.testing.assert_array_equal(meta.sample_id,ev['sample_ids']);np.testing.assert_array_equal(evx['sample_ids'],ev['sample_ids'])
        assert meta.group.is_unique and not set(ev['groups'])&set(tr['groups']) and (meta.origin.dt.year==year).all()
        np.testing.assert_array_equal(transform(ev,plan['original_names'])['X'],evx['X'])
        keep=kept_columns(plan['original_names']);pred={};models={};risks={}
        for i,seed in enumerate(SEEDS):
            folder=run/'results/formal'/str(gate['job'])/f'task-{YEARS.index(year)*3+i}/output'
            s=json_load(folder/'summary.json');assert (s['year'],s['seed'])==(year,seed) and s['no_test_labels_read']
            assert max(s['replay_checks'].values())<1e-10
            fits.append({k:v for k,v in s.items() if k not in ['checks','replay_checks']})
            assert json_load(folder/'pairing.json')==pairing(tr['groups'],configuration(seed),20)
            for version in ['raw','shrunk']:checks.append(dict(year=year,seed=str(seed),version=version,**s['checks'][version]))
            ld=pd.read_csv(folder/'leaf_support_PRIVATE.csv')
            leaves.append(dict(year=year,seed=seed,leaves=len(ld),min_unique_companies=int(ld.unique_companies.min()),
                median_unique_companies=float(ld.unique_companies.median()),zero_target_leaves=int(ld.target_companies.eq(0).sum()),borrowed_positive_target_leaves=int(ld.borrowed_positive_target.sum())))
            for arm in arms:
                p=folder/'shrunk' if arm=='paper20' else root/'reference'/arm/str(seed)
                f=CompetingRiskForest.load(p/'forest_PRIVATE.json');z=arrays(p/'predictions_PRIVATE.npz')
                np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
                X=evx['X'] if arm=='paper20' else ev['X'] if arm=='reference83' else ev['X'][:,keep]
                np.testing.assert_array_equal(f.predict(X,TIMES),z['cif'])
                models[(arm,str(seed))]=f;pred[(arm,str(seed))]=z['cif']
                hashes.append(dict(year=year,arm=arm,seed=seed,forest_sha256=sha(p/'forest_PRIVATE.json'),predictions_sha256=sha(p/'predictions_PRIVATE.npz')))
        for arm in arms:
            f=merge([models[(arm,str(s))] for s in SEEDS]);pred[(arm,'ensemble')]=np.mean([pred[(arm,str(s))] for s in SEEDS],axis=0)
            data=tr if arm=='paper20' else old if arm=='reference83' else reduced(old,keep)
            X=evx['X'] if arm=='paper20' else ev['X'] if arm=='reference83' else ev['X'][:,keep]
            np.testing.assert_allclose(f.predict(X,TIMES),pred[(arm,'ensemble')],atol=1e-13,rtol=0)
            if arm=='paper20':
                check=verify_export(f,tr,evx,run/f'private/verify_{year}_ensemble');checks.append(dict(year=year,seed='ensemble',version='shrunk',**check))
                c={k:check[k] for k in ['main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','distinct_thresholds']}
            else:
                ex=CIFExpansion.from_forest(f,data['X']);c={k:v for k,v in ex.complexity(data['X'],support_min=10).items() if k!='functions'}
            complexities.append(dict(year=year,arm=arm,seed='ensemble',trees=96,**c))
        for (arm,seed),p in pred.items():
            risk=horizon_predictions(p,TIMES,ev['h12']);risks[(arm,seed)]=risk
            for cause in [1,2]:
                tags=dict(year=year,arm=arm,seed=seed,horizon_months=12,cause=cause)
                m,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),12),risk[:,cause-1],cause,ev['label_eligible'])
                assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
                np.testing.assert_allclose([m[k] for k in ['auc','ap','brier']],direct_metrics(y,risk[:,cause-1]),atol=1e-12,rtol=0,equal_nan=True)
                rows.append(dict(**tags,**m,calibration_gap=m['risk_mean']-m['ipcw_observed_incidence']))
                if cause==1:alerts.append(dict(**tags,**warning(risk[:,0],ev['event'],ev['analysis_time'],ev['sample_ids'],12)))
                if seed=='ensemble':
                    for q,ii in enumerate(np.array_split(np.argsort(risk[:,cause-1],kind='stable'),4)):
                        bins.append(dict(**tags,quartile=q+1,n=len(ii),predicted=float(risk[ii,cause-1].mean()),observed=float(y[ii].mean()),events=int(y[ii].sum())))
        np.savez_compressed(run/f'private/risks_{year}_PRIVATE.npz',**{'_'.join(k):v for k,v in risks.items()})
        for arm in arms:
            sd=np.stack([risks[(arm,str(s))][:,0] for s in SEEDS]).std(0,ddof=1)
            spreads.append(dict(year=year,arm=arm,company_seed_prediction_sd_mean=float(sd.mean()),company_seed_prediction_sd_max=float(sd.max())))
        rng=np.random.default_rng(np.random.SeedSequence([protocol['uncertainty']['seed'],year]));n=len(ev['X'])
        draws=rng.integers(n,size=(1000,n));np.savez_compressed(run/f'private/bootstrap_{year}_PRIVATE.npz',indices=draws)
        local=[]
        for i,ii in enumerate(draws):
            values={}
            for arm in arms:
                p=risks[(arm,'ensemble')][:,0]
                m,_,_,_=metric(ev['analysis_time'][ii],ev['event'][ii],np.full(n,12),p[ii],1,ev['label_eligible'][ii])
                m.update(warning(p[ii],ev['event'][ii],ev['analysis_time'][ii],ev['sample_ids'][ii],12));values[arm]=m
            for ref in arms[:2]:local.append(dict(year=year,reference_arm=ref,replicate=i,**{k:values['paper20'][k]-values[ref][k] for k in ['auc','ap','brier','risk_mean','captured','coverage']}))
        boots.extend(local);bd=pd.DataFrame(local)
        for ref in arms[:2]:
            for k in ['auc','ap','brier','risk_mean','captured','coverage']:
                candidate=alerts if k in ['captured','coverage'] else rows;vv=[]
                for arm in [ref,'paper20']:vv.append(next(r[k] for r in candidate if r['year']==year and r['arm']==arm and r['seed']=='ensemble' and r['cause']==1))
                valid=bd[bd.reference_arm==ref][k].dropna();lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
                diffs.append(dict(year=year,reference_arm=ref,metric=k,reference=vv[0],paper20=vv[1],difference=vv[1]-vv[0],lower95=lo,upper95=hi,undefined_bootstraps=1000-len(valid)))
    df=pd.DataFrame(rows);wa=pd.DataFrame(alerts)
    # Recompute every reused12-month prediction against its published metrics.
    prior=pd.concat([pd.read_csv(repo/f'experiments/{name}/results/metrics.csv',dtype={'seed':str}) for name in ['post_st_drop9_v1','post_st_drop9_transport_v1']],ignore_index=True)
    verified=0
    for r in df[df.arm!='paper20'].itertuples():
        z=prior[(prior.year==r.year)&(prior.arm==r.arm)&(prior.seed==r.seed)&(prior.horizon_months==12)&(prior.cause==r.cause)]
        assert len(z)==1
        np.testing.assert_allclose([getattr(r,k) for k in ['auc','ap','brier','risk_mean']],z[['auc','ap','brier','risk_mean']].iloc[0].to_numpy(float),atol=1e-12,rtol=0,equal_nan=True);verified+=1
    ensemble=df[(df.seed=='ensemble')&(df.cause==1)].merge(wa[['year','arm','seed','checked','captured','target_events','coverage']],on=['year','arm','seed'])
    for name,d in [('metrics',df),('warnings',wa),('ensemble_comparison',ensemble),('calibration_bins',pd.DataFrame(bins)),('function_validation',pd.DataFrame(checks)),
        ('function_complexity',pd.DataFrame(complexities)),('execution_tasks',pd.DataFrame(fits)),('model_hashes',pd.DataFrame(hashes)),('leaf_support_summary',pd.DataFrame(leaves)),
        ('prediction_seed_variation',pd.DataFrame(spreads)),('paired_differences',pd.DataFrame(diffs)),('bootstrap_differences',pd.DataFrame(boots))]:d.to_csv(run/f'reports/{name}.csv',index=False)
    for p,h in json_load(run/'audit/source_hashes.json').items():assert sha(Path(p))==h
    for p,h in json_load(run/'audit/immutable_input_hashes.json').items():assert sha(run/'input'/p)==h
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    error=max(max(r[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload']) for r in checks)
    write(run/'audit/validation.json',dict(status='PASS_PROVISIONAL_ADAPTED_SUBSET',new_forests=9,derived_models=9,reused_forests=18,reference_metric_rows_reproduced=verified,
        source_hashes_unchanged=True,current_reference_unchanged=True,bootstrap_tree_pairs=288,max_explanation_error=error,all_test_endpoints_known=True,
        unsupported_full54_claim=False,calibration_fits=0,search_trials=0))
    print(ensemble[['year','arm','auc','ap','brier','risk_mean','captured','target_events']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run,Path(__file__).resolve().parents[2])
