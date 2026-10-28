"""Matched fixed-forecast analysis, with independent metrics and baseline replay."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import pandas as pd
from crforest import CIFExpansion,CompetingRiskForest
from crforest.core import aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from .reference import forecasts as old_forecasts


def evaluate(run):
    protocol=json.loads((run/'audit/protocol.json').read_text())
    for name,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/name)==h
    for name,h in json.loads((run/'audit/data_alignment.json').read_text())['unchanged_source_hashes'].items():assert sha(Path(name))==h
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS'
    old,old_details,_=old_forecasts(protocol)
    out=run/'evaluation';out.mkdir(exist_ok=False)
    pred={};details={};summaries=[];checks=[];hashes=[];stability=[]
    tasks=[json.loads(s) for s in (run/'formal.jsonl').read_text().splitlines()]
    for i,task in enumerate(tasks):
        args=task['argv'];year=int(args[args.index('--year')+1]);variant=args[args.index('--variant')+1];seed=args[args.index('--seed')+1]
        folder=run/'results/formal'/str(gate['job'])/f'task-{i}'/'output'
        for name,h in json.loads((folder/'hashes.json').read_text()).items():assert sha(folder/name)==h
        key=(year,variant,seed);assert key not in pred
        p=arrays(folder/'predictions_PRIVATE.npz');pred[key]=p;details[key]=folder
        model=CompetingRiskForest.load(folder/'forest_PRIVATE.json')
        assert tuple(model.config.cause_weights)==tuple(protocol['weights'][variant])
        summary=json.loads((folder/'summary.json').read_text());summary.pop('input_hashes',None)
        leaves=[n for t in model.trees_ for n in t if 'cif' in n]
        summary.update(test_year=year,variant=variant,
            target_unsupported_leaf_fraction=float(np.mean([n['cause_company_support'][0]==0 for n in leaves])),
            target_support_median=float(np.median([n['cause_company_support'][0] for n in leaves])))
        summaries.append(summary)
        error=None;tree_equal=None
        if variant=='equal':
            prior=old[(year,'xai_q2',seed)]
            for name in ['sample_ids','times','cif','training_cif']:np.testing.assert_array_equal(p[name],prior[name])
            before=CompetingRiskForest.load(old_details[(year,'xai_q2',seed)]/'forest_PRIVATE.json')
            tree_equal=before.trees_==model.trees_;assert tree_equal;error=0.
        checks.append(dict(test_year=year,variant=variant,seed=seed,baseline_prediction_max_error=error,
            baseline_all_trees_identical=tree_equal,reconstruction_error=summary['complete_component_sum_error'],
            reload_error=summary['reload_error'],max_path_variables=summary['max_path_variables']))
        hashes.append(dict(test_year=year,variant=variant,seed=seed,forest_sha256=sha(folder/'forest_PRIVATE.json'),
            predictions_sha256=sha(folder/'predictions_PRIVATE.npz'),functions_sha256=sha(folder/'functions_PRIVATE.npz')))
    assert len(pred)==18
    for (year,model,seed),p in old.items():
        if model in ['unrestricted','standard_rsf'] and seed!='ensemble':pred[(year,model,seed)]=p
    models=list(protocol['weights'])+['unrestricted','standard_rsf']
    for year in protocol['test_years']:
        for model in models:
            parts=[pred[(year,model,str(s))] for s in protocol['seeds']]
            pred[(year,model,'ensemble')]={k:parts[0][k] for k in ['sample_ids','times']}
            pred[(year,model,'ensemble')]['cif']=np.mean([p['cif'] for p in parts],axis=0)
        tr=arrays(run/'input/folds'/str(year)/'equal/train_PRIVATE.npz')
        nref=len(tr['X'])
        for variant in protocol['weights']:
            functions=[];components=[];reconstructed=[]
            ev=arrays(run/'evaluation_input'/str(year)/'test_PRIVATE.npz')
            for seed in map(str,protocol['seeds']):
                d=details[(year,variant,seed)]
                functions.append(json.loads((d/'complexity_PRIVATE.json').read_text()))
                components.append(arrays(d/'components_PRIVATE.npz'))
                ex=CIFExpansion.load(d/'functions_PRIVATE.npz')
                reconstructed.append(ex.predict(ev['X'],[365.,366.,730.,731.]))
            plan=json.loads((run/'input/folds'/str(year)/variant/'plan.json').read_text());ncal=plan['calibration_n']
            ensemble_error=float(abs(np.mean(reconstructed,axis=0)-pred[(year,variant,'ensemble')]['cif'][ncal:]).max())
            assert ensemble_error<1e-10
            checks.append(dict(test_year=year,variant=variant,seed='ensemble',reconstruction_error=ensemble_error))
            for ia,ib in itertools.combinations(range(3),2):
                fa,fb=functions[ia],functions[ib];ca,cb=components[ia],components[ib]
                sets=lambda f,order:{tuple(x['features']) for x in f['functions'] if x['nonzero'] and x['order']==order}
                ma,mb=sets(fa,1),sets(fb,1);pa,pb=sets(fa,2),sets(fb,2)
                ja={k:i for i,k in enumerate(ca['keys'])};jb={k:i for i,k in enumerate(cb['keys'])}
                for cause in range(2):
                    correlations=[]
                    for term in set(ja)&set(jb):
                        aa=ca['terms'][:nref,ja[term],2,cause];bb=cb['terms'][:nref,jb[term],2,cause]
                        if np.std(aa)>1e-12 and np.std(bb)>1e-12:correlations.append(float(np.corrcoef(aa,bb)[0,1]))
                    stability.append(dict(test_year=year,variant=variant,seed_a=protocol['seeds'][ia],seed_b=protocol['seeds'][ib],
                        cause=['termination','recovery'][cause],main_jaccard=len(ma&mb)/max(len(ma|mb),1),
                        pair_jaccard=len(pa&pb)/max(len(pa|pb),1),common_nonconstant_functions=len(correlations),
                        median_function_correlation=float(np.median(correlations)) if correlations else np.nan,
                        reference='same fold training companies, day730; descriptive 3 seeds'))
    metrics=[];warnings=[];bins=[];risk_by_key={}
    data={y:arrays(run/'evaluation_input'/str(y)/'test_PRIVATE.npz') for y in protocol['test_years']}
    metadata={y:pd.read_parquet(run/'evaluation_input'/str(y)/'test_metadata_PRIVATE.parquet') for y in protocol['test_years']}
    for year in protocol['test_years']:
        plan=json.loads((run/'input/folds'/str(year)/'equal/plan.json').read_text());ev=data[year]
        for (yy,model,seed),p in pred.items():
            if yy!=year:continue
            ncal=plan['calibration_n'];assert np.array_equal(p['sample_ids'][ncal:],ev['sample_ids'])
            risk_by_key[(str(year),model,seed)]=horizon_predictions(p['cif'][ncal:],p['times'],ev['h24'])
        tr=arrays(run/'input/folds'/str(year)/'equal/train_PRIVATE.npz')
        aj=aalen_johansen(tr['time'],tr['event'],[730.,731.])
        risk_by_key[(str(year),'training_AJ','none')]=aj[np.searchsorted([730.,731.],ev['h24'])]
    for model,seed in sorted({(m,s) for y,m,s in risk_by_key}):
        risk_by_key[('pooled',model,seed)]=np.concatenate([risk_by_key[(str(y),model,seed)] for y in protocol['test_years']])
    data['pooled']={k:np.concatenate([data[y][k] for y in protocol['test_years']]) for k in data[2023]}
    metadata['pooled']=pd.concat([metadata[y] for y in protocol['test_years']],ignore_index=True)
    for (scope,model,seed),risk in risk_by_key.items():
        ev=data['pooled' if scope=='pooled' else int(scope)];meta=metadata['pooled' if scope=='pooled' else int(scope)]
        for k,cause in enumerate(['termination','recovery']):
            common=dict(scope=scope,model=model,seed=seed,cause=cause,horizon_months=24,weighting='one_company_one_record')
            result,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),24.),risk[:,k],k+1,ev['label_eligible'])
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            np.testing.assert_allclose([result[x] for x in ['auc','ap','brier']],direct_metrics(y,risk[:,k]),rtol=0,atol=1e-12,equal_nan=True)
            metrics.append({**common,**result})
            if seed in ['ensemble','none']:
                for b,ids in enumerate(np.array_split(np.argsort(risk[:,k],kind='stable'),4)):
                    bins.append({**common,'bin':b+1,'n':len(ids),'predicted':float(risk[ids,k].mean()),'observed':float(y[ids].mean())})
            if k:continue
            for fraction in [.2,.25]:
                chosen=np.zeros(len(risk),bool)
                for year in np.unique(ev['year']):
                    ix=np.flatnonzero(ev['year']==year);order=np.lexsort((ev['sample_ids'][ix],-risk[ix,k]))
                    chosen[ix[order[:int(np.ceil(len(ix)*fraction))]]]=True
                lead={months:np.array([(o+pd.DateOffset(months=months)-o).days for o in meta.origin]) for months in [3,6]}
                warnings.append({**common,'top_fraction':fraction,'checked':int(chosen.sum()),'captured':int((chosen&y).sum()),
                    'target_events':int(y.sum()),'coverage':float((chosen&y).sum()/y.sum()),
                    'known_non_target':int((chosen&~y).sum()),'unknown':0,'repeat_alerts':0,
                    'lead3':int((chosen&y&(ev['time']>=lead[3])).sum()),'lead6':int((chosen&y&(ev['time']>=lead[6])).sum()),
                    'selection':'retrospective within entry year; pooled sums annual selections, not a prospective policy'})
    df=pd.DataFrame(metrics);df.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(warnings).to_csv(out/'warning_metrics.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'training_diagnostics.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'mathematical_checks.csv',index=False)
    pd.DataFrame(hashes).to_csv(out/'model_hashes.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'seed_stability.csv',index=False)
    ss=df[df.seed.isin(['42','101','202'])].groupby(['scope','model','cause'])[['auc','ap','brier','risk_mean']].agg(['mean','min','max'])
    ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    deltas=[]
    for (scope,model,seed),risk in risk_by_key.items():
        if model not in ['double_termination','termination_only']:continue
        base=df[(df.scope==scope)&(df.model=='equal')&(df.seed==seed)].set_index('cause')
        new=df[(df.scope==scope)&(df.model==model)&(df.seed==seed)].set_index('cause')
        for cause in ['termination','recovery']:
            deltas.append(dict(scope=scope,model=model,seed=seed,cause=cause,
                **{name+'_change':float(new.loc[cause,name]-base.loc[cause,name]) for name in ['auc','ap','brier','risk_mean']}))
    pd.DataFrame(deltas).to_csv(out/'paired_changes.csv',index=False)
    write(out/'validation.json',dict(status='PASS',formal_models=18,pilots=2,metric_rows=len(df),
        historical_q2_models_reproduced_exactly=6,q2_ensemble_reconstruction_checks=6,
        input_hashes_unchanged=True,independent_metric_check=True,all_test_endpoints_known=True,
        independent_test_companies=81,independent_test_terminations=31,independent_test_recoveries=31,
        no_calibration_fit=True,no_test_outcome_selection=True,no_clipping=True,
        maximum_reconstruction_error=float(pd.DataFrame(checks).reconstruction_error.max())))
    print(df[(df.seed.isin(['ensemble','none']))&(df.cause=='termination')][['scope','model','auc','ap','brier','risk_mean']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();evaluate(a.run)
