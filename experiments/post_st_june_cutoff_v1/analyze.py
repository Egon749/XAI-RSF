"""Evaluate only after all frozen fits publish; no model fitting or selection."""
from pathlib import Path
import argparse,json,itertools
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from experiments.post_st_drop9_v1.analyze import warning
from .common import *


def analyze(run,repo):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS' and gate['tasks']==18
    protocol=json.loads((run/'input/protocol.json').read_text())
    rows=[];warnings=[];bins=[];checks=[];tasks=[];hashes=[];support=[];leaves=[];usage=[];spread=[];contrasts=[];boot=[];stability=[]
    for year in YEARS:
        root=run/f'input/folds/{year}';ev=arrays(run/f'evaluation_input/{year}_test_PRIVATE.npz')
        meta=pd.read_parquet(run/f'evaluation_input/{year}_test_metadata_PRIVATE.parquet')
        np.testing.assert_array_equal(ev['sample_ids'],meta.sample_id)
        cutoff=pd.Timestamp(f'{year}-06-30')
        assert meta.group.is_unique and (meta.origin>cutoff).all() and (meta.origin<=cutoff+pd.DateOffset(months=12)).all()
        assert (meta.stop<=OBSERVED_END).all()
        status=json.loads((root/'reference83/plan.json').read_text())['evaluation_status']
        models={};pred={};risks={};parts={}
        for arm in ARMS:
            tr=arrays(root/arm/'train_PRIVATE.npz');xx=arrays(root/arm/'eval_PRIVATE.npz')
            plan=json.loads((root/arm/'plan.json').read_text());keep=plan['original_indices']
            np.testing.assert_array_equal(ev['sample_ids'],xx['sample_ids'])
            np.testing.assert_array_equal(ev['X'][:,keep],xx['X'])
            assert not set(ev['groups'])&set(tr['groups'])
            for i,seed in enumerate(SEEDS):
                task=YEARS.index(year)*6+ARMS.index(arm)*3+i
                folder=run/'results/formal'/str(gate['job'])/f'task-{task}/output'
                s=json.loads((folder/'summary.json').read_text());assert (s['year'],s['arm'],s['seed'])==(year,arm,seed)
                assert s['no_test_labels_read'] and max(s['replay_checks'].values())<1e-10
                tasks.append({k:v for k,v in s.items() if k not in ['checks','replay_checks']})
                expected=pairing(tr['groups'],configuration(seed,'shallow_parent'),tr['X'].shape[1])
                assert json.loads((folder/'pairing.json').read_text())==expected
                for kind in ['raw','shrunk']:checks.append(dict(year=year,arm=arm,version=kind,seed=str(seed),**s['checks'][kind]))
                modelpath=folder/'shrunk';f=CompetingRiskForest.load(modelpath/'forest_PRIVATE.json')
                models[(arm,str(seed))]=f
                z=arrays(modelpath/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['times'],TIMES)
                np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);pred[(arm,str(seed))]=z['cif']
                np.testing.assert_array_equal(f.predict(xx['X'],TIMES),z['cif'])
                parts[(arm,str(seed))]=(json.loads((modelpath/'complexity_PRIVATE.json').read_text()),arrays(modelpath/'components_PRIVATE.npz'))
                hashes.append(dict(year=year,arm=arm,seed=seed,forest_sha256=sha(modelpath/'forest_PRIVATE.json'),
                    prediction_sha256=sha(modelpath/'predictions_PRIVATE.npz'),raw_forest_sha256=sha(folder/'raw/forest_PRIVATE.json'),
                    functions_sha256=sha(modelpath/'functions_PRIVATE.npz')))
                ld=pd.read_csv(folder/'leaf_support_PRIVATE.csv')
                leaves.append(dict(year=year,arm=arm,seed=seed,leaves=len(ld),min_unique_companies=int(ld.unique_companies.min()),
                    median_unique_companies=float(ld.unique_companies.median()),zero_target_leaves=int((ld.target_companies==0).sum()),
                    borrowed_positive_target_leaves=int(ld.borrowed_positive_target.sum())))
                for j,name in enumerate(plan['feature_names']):
                    usage.append(dict(year=year,arm=arm,seed=seed,name=name,original_index=keep[j],
                        split_trees=sum(any(n.get('feature')==j for n in tree) for tree in f.trees_),
                        split_nodes=sum(n.get('feature')==j for tree in f.trees_ for n in tree)))
            f=merge([models[(arm,str(seed))] for seed in SEEDS])
            pred[(arm,'ensemble')]=np.mean([pred[(arm,str(seed))] for seed in SEEDS],axis=0)
            z=verify_export(f,tr,xx,run/'private'/f'verify_{year}_{arm}_ensemble')
            checks.append(dict(year=year,arm=arm,version='shrunk',seed='ensemble',**z))
            np.testing.assert_allclose(f.predict(xx['X'],TIMES),pred[(arm,'ensemble')],atol=1e-13,rtol=0)
            # Function comparisons use the same training empirical-product reference.
            for sa,sb in itertools.combinations(map(str,SEEDS),2):
                ca,xa=parts[(arm,sa)];cb,xb=parts[(arm,sb)]
                sets=lambda c,o:{tuple(v['features']) for v in c['functions'] if v['nonzero'] and v['order']==o}
                ma,mb,pa,pb=sets(ca,1),sets(cb,1),sets(ca,2),sets(cb,2)
                ia={k:i for i,k in enumerate(xa['keys'])};ib={k:i for i,k in enumerate(xb['keys'])};corr=[]
                for key in ia.keys()&ib.keys():
                    va=xa['terms'][:,ia[key],0,0];vb=xb['terms'][:,ib[key],0,0]
                    if np.std(va)>1e-12 and np.std(vb)>1e-12:corr.append(float(np.corrcoef(va,vb)[0,1]))
                stability.append(dict(year=year,arm=arm,seed_a=sa,seed_b=sb,
                    main_jaccard=len(ma&mb)/max(len(ma|mb),1),pair_jaccard=len(pa&pb)/max(len(pa|pb),1),
                    common_nonconstant_functions=len(corr),median_function_correlation=float(np.median(corr)) if corr else np.nan,
                    comparison='training reference, cause1 at day365; descriptive algorithm randomness'))
        # The public blocker is applied before any performance, event-count or
        # risk summaries, to avoid exposing the one available test company.
        for (arm,seed),p in pred.items():
            risk=horizon_predictions(p,TIMES,ev['h12']);risks[(arm,seed)]=risk
            for cause in [1,2]:
                tags=dict(year=year,training_cutoff=str(cutoff.date()),arm=arm,seed=seed,horizon_months=12,cause=cause)
                if status!='SUPPORTED':
                    rows.append(dict(**tags,status=status,companies=len(risk),auc=np.nan,ap=np.nan,brier=np.nan,
                        risk_mean=np.nan,ipcw_observed_incidence=np.nan,calibration_gap=np.nan));continue
                m,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),12),risk[:,cause-1],cause,ev['label_eligible'])
                assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
                np.testing.assert_allclose([m[k] for k in ['auc','ap','brier']],direct_metrics(y,risk[:,cause-1]),atol=1e-12,rtol=0,equal_nan=True)
                rows.append(dict(**tags,**m,status=status,calibration_gap=m['risk_mean']-m['ipcw_observed_incidence']))
                if cause==1:warnings.append(dict(**tags,**warning(risk[:,0],ev['event'],ev['analysis_time'],ev['sample_ids'],12)))
                if seed=='ensemble':
                    for q,ix in enumerate(np.array_split(np.argsort(risk[:,cause-1],kind='stable'),4)):
                        bins.append(dict(**tags,quartile=q+1,n=len(ix),predicted=float(risk[ix,cause-1].mean()),
                            observed=float(y[ix].mean()),events=int(y[ix].sum())))
        np.savez_compressed(run/f'private/risks_{year}_PRIVATE.npz',**{'_'.join(k):v for k,v in risks.items()})
        support.append(dict(year=year,status=status,companies_available=len(meta),cohort_complete=status=='SUPPORTED',
            complete_endpoint_companies=len(meta) if status=='SUPPORTED' else np.nan))
        if status!='SUPPORTED':continue
        for arm in ARMS:
            sd=np.stack([risks[(arm,str(s))][:,0] for s in SEEDS]).std(0,ddof=1)
            spread.append(dict(year=year,arm=arm,company_seed_prediction_sd_mean=float(sd.mean()),company_seed_prediction_sd_max=float(sd.max())))
        rng=np.random.default_rng(np.random.SeedSequence([protocol['uncertainty']['seed'],year]));n=len(meta)
        draws=rng.integers(n,size=(protocol['uncertainty']['repeats'],n))
        np.savez_compressed(run/f'private/bootstrap_{year}_PRIVATE.npz',indices=draws)
        local=[]
        for i,ix in enumerate(draws):
            values=[]
            for arm in ARMS:
                p=risks[(arm,'ensemble')][:,0]
                m,_,_,_=metric(ev['analysis_time'][ix],ev['event'][ix],np.full(n,12),p[ix],1,ev['label_eligible'][ix])
                m.update(warning(p[ix],ev['event'][ix],ev['analysis_time'][ix],ev['sample_ids'][ix],12));values.append(m)
            local.append(dict(year=year,replicate=i,**{k:values[1][k]-values[0][k] for k in ['auc','ap','brier','risk_mean','captured','coverage']}))
        boot.extend(local);bd=pd.DataFrame(local)
        for k in ['auc','ap','brier','risk_mean','captured','coverage']:
            values=[]
            for arm in ARMS:
                candidates=warnings if k in ['captured','coverage'] else rows
                r=next(r for r in candidates if r['year']==year and r['cause']==1 and r['seed']=='ensemble' and r['arm']==arm)
                values.append(r[k])
            valid=bd[k].dropna();lo,hi=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
            contrasts.append(dict(year=year,metric=k,reference=values[0],drop9=values[1],difference=values[1]-values[0],
                lower95=lo,upper95=hi,undefined_bootstraps=int(bd[k].isna().sum()),
                direction='negative improves' if k=='brier' else 'level only' if k=='risk_mean' else 'positive improves'))
    df=pd.DataFrame(rows)
    assert df[df.year==2025][['auc','ap','brier','risk_mean','ipcw_observed_incidence','known_target_events','known_other_outcomes','unknown_outcomes']].isna().all().all()
    for name,table in [('metrics',df),('warnings',pd.DataFrame(warnings)),('horizon_support',pd.DataFrame(support)),('calibration_bins',pd.DataFrame(bins)),
        ('function_validation',pd.DataFrame(checks)),('execution_tasks',pd.DataFrame(tasks)),('model_hashes',pd.DataFrame(hashes)),
        ('leaf_support_summary',pd.DataFrame(leaves)),('actual_feature_usage',pd.DataFrame(usage)),('prediction_seed_variation',pd.DataFrame(spread)),
        ('function_seed_stability',pd.DataFrame(stability)),('paired_differences',pd.DataFrame(contrasts)),('bootstrap_differences',pd.DataFrame(boot))]:
        table.to_csv(run/f'reports/{name}.csv',index=False)
    for p,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(p))==h
    for p,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/p)==h
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    error=max(max(r[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','forest_reload']) for r in checks)
    write(run/'audit/validation.json',dict(status='PASS_WITH_2025_EVALUATION_BLOCKER',new_forests=18,derived_models=18,reused_forests=0,formal_tasks=18,
        source_hashes_unchanged=True,current_reference_unchanged=True,bootstrap_tree_pairs=288,bootstrap_repeats_per_supported_window=1000,
        max_explanation_error=error,all_2025_test_metrics_suppressed=True,no_new_calibration_or_selection=True,supported_windows=2,blocked_windows=1))
    print(df[(df.seed=='ensemble')&(df.cause==1)][['year','arm','auc','ap','brier','risk_mean','status']].to_string(index=False))
    print(pd.DataFrame(warnings).query("seed=='ensemble'")[['year','arm','captured','target_events','coverage']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    analyze(a.run,Path(__file__).resolve().parents[2])
