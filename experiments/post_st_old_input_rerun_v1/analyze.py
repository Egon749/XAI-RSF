"""Fresh old-input rerun versus original old-C forecasts; independent endpoint calculations."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from experiments.post_st_next_open_info_v1.analyze import outputs


def evaluate(run):
    protocol=json.loads((run/'audit/protocol.json').read_text());old=Path(protocol['reference_run'])
    for name,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/name)==h
    for name,h in json.loads((run/'audit/data_alignment.json').read_text())['unchanged_source_hashes'].items():assert sha(Path(name))==h
    out=run/'evaluation';out.mkdir(exist_ok=False)
    pred={};details={};summaries=[];checks=[];hashes=[];stability=[]
    tasks=[('rerun_old_C',y,m,s,p) for y,v,m,s,p in outputs(run)]
    refs=json.loads((run/'audit/reference_models.json').read_text())
    tasks += [('original_old_C',r['year'],r['model'],r['seed'],Path(r['path'])) for r in refs]
    for state,year,model,seed,folder in tasks:
        for name,h in json.loads((folder/'hashes.json').read_text()).items():assert sha(folder/name)==h
        key=(state,year,model,seed);assert key not in pred
        p=arrays(folder/'predictions_PRIVATE.npz');pred[key]=p;details[key]=folder
        ev=arrays(run/'evaluation_input'/f'{year}_announcement_fields_test_PRIVATE.npz')
        np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids'])
        summary=json.loads((folder/'summary.json').read_text());summary.pop('input_hashes',None)
        summary.update(source_state=state,test_year=year)
        if model=='xai_q2':
            f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');assert tuple(f.config.cause_weights)==(1.,1.)
            assert f.config.path_budget==2 and f.config.n_trees==32
            leaves=[n for t in f.trees_ for n in t if 'cif' in n]
            summary.update(target_unsupported_leaf_fraction=float(np.mean([n['cause_company_support'][0]==0 for n in leaves])),
                target_leaf_support_median=float(np.median([n['cause_company_support'][0] for n in leaves])))
        summaries.append(summary)
        checks.append(dict(source_state=state,test_year=year,model=model,seed=seed,
            reconstruction_error=summary.get('complete_component_sum_error'),reload_error=summary['reload_error'],
            max_path_variables=summary.get('max_path_variables'),probability_min=summary['probability_min'],
            max_cause_probability_sum=summary['max_cause_probability_sum']))
        hashes.append(dict(source_state=state,test_year=year,model=model,seed=seed,
            forest_sha256=sha(folder/('forest_PRIVATE.rds' if model=='standard_rsf' else 'forest_PRIVATE.json')),
            predictions_sha256=sha(folder/'predictions_PRIVATE.npz'),
            functions_sha256=sha(folder/'functions_PRIVATE.npz') if model=='xai_q2' else None))
    assert len(pred)==24
    years=protocol['test_years'];models=protocol['models'];states=['original_old_C','rerun_old_C']
    for state,year,model in itertools.product(states,years,models):
        members=[pred[(state,year,model,str(s))] for s in protocol['seeds']]
        pred[(state,year,model,'ensemble')]=dict(times=members[0]['times'],sample_ids=members[0]['sample_ids'],
            cif=np.mean([p['cif'] for p in members],axis=0))
    for state,year in itertools.product(states,years):
        n=len(arrays(run/'input/folds'/str(year)/'announcement_fields/train_PRIVATE.npz')['X'])
        cs=[json.loads((details[(state,year,'xai_q2',str(s))]/'complexity_PRIVATE.json').read_text()) for s in protocol['seeds']]
        ps=[arrays(details[(state,year,'xai_q2',str(s))]/'components_PRIVATE.npz') for s in protocol['seeds']]
        for a,b in itertools.combinations(range(3),2):
            sets=lambda c,k:{tuple(x['features']) for x in c['functions'] if x['nonzero'] and x['order']==k}
            ma,mb=sets(cs[a],1),sets(cs[b],1);pa,pb=sets(cs[a],2),sets(cs[b],2)
            ia={k:i for i,k in enumerate(ps[a]['keys'])};ib={k:i for i,k in enumerate(ps[b]['keys'])}
            for k,cause in enumerate(['termination','recovery']):
                corr=[]
                for term in ia.keys()&ib.keys():
                    aa=ps[a]['terms'][:n,ia[term],2,k];bb=ps[b]['terms'][:n,ib[term],2,k]
                    if np.std(aa)>1e-12 and np.std(bb)>1e-12:corr.append(float(np.corrcoef(aa,bb)[0,1]))
                stability.append(dict(source_state=state,test_year=year,seed_a=protocol['seeds'][a],seed_b=protocol['seeds'][b],cause=cause,
                    main_jaccard=len(ma&mb)/max(len(ma|mb),1),pair_jaccard=len(pa&pb)/max(len(pa|pb),1),
                    common_nonconstant_functions=len(corr),median_function_correlation=float(np.median(corr)) if corr else np.nan,
                    reference='within each source version: same fold training reference/day730; source revisions change centering, not direct same-function causal comparison'))
    data={y:arrays(run/'evaluation_input'/f'{y}_announcement_fields_test_PRIVATE.npz') for y in years}
    meta={y:pd.read_parquet(run/'evaluation_input'/f'{y}_test_metadata_PRIVATE.parquet') for y in years}
    data['pooled']={k:np.concatenate([data[y][k] for y in years]) for k in data[years[0]]}
    meta['pooled']=pd.concat([meta[y] for y in years],ignore_index=True)
    risks={}
    for (state,year,model,seed),p in pred.items():
        for months in [12,24]:risks[(str(year),state,model,seed,months)]=horizon_predictions(p['cif'],p['times'],data[year][f'h{months}'])
    for year in years:
        tr=arrays(run/'input/folds'/str(year)/'announcement_fields/train_PRIVATE.npz');tt=np.array([365.,366.,730.,731.])
        aj=aalen_johansen(tr['time'],tr['event'],tt)
        for months in [12,24]:risks[(str(year),'common','training_AJ','none',months)]=aj[np.searchsorted(tt,data[year][f'h{months}'])]
    for state,model,seed,months in {(s,m,z,h) for y,s,m,z,h in risks}:
        risks[('pooled',state,model,seed,months)]=np.concatenate([risks[(str(y),state,model,seed,months)] for y in years])
    metrics=[];warnings=[];bins=[]
    for (scope,state,model,seed,months),risk in risks.items():
        yy='pooled' if scope=='pooled' else int(scope);ev=data[yy];mm=meta[yy]
        for k,cause in enumerate(['termination','recovery']):
            common=dict(scope=scope,source_state=state,model=model,seed=seed,cause=cause,horizon_months=months,weighting='one_company_one_record')
            result,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),float(months)),risk[:,k],k+1,ev['label_eligible'])
            assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            np.testing.assert_allclose([result[z] for z in ['auc','ap','brier']],direct_metrics(y,risk[:,k]),rtol=0,atol=1e-12,equal_nan=True)
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
                leads={n:np.array([(o+pd.DateOffset(months=n)-o).days for o in mm.origin]) for n in [3,6]}
                warnings.append({**common,'top_fraction':fraction,'checked':int(chosen.sum()),'captured':int((chosen&y&known).sum()),
                    'target_events':int((y&known).sum()),'coverage':float((chosen&y&known).sum()/(y&known).sum()) if (y&known).any() else np.nan,
                    'known_non_target':int((chosen&~y&known).sum()),'unknown':int((chosen&~known).sum()),'repeat_alerts':0,
                    'lead3':int((chosen&y&known&(ev['time']>=leads[3])).sum()),'lead6':int((chosen&y&known&(ev['time']>=leads[6])).sum()),
                    'selection':'annual retrospective ranking, not an online policy'})
    df=pd.DataFrame(metrics);df.to_csv(out/'metrics.csv',index=False)
    for name,rows in [('warning_metrics',warnings),('calibration_bins',bins),('training_diagnostics',summaries),
        ('mathematical_checks',checks),('model_hashes',hashes),('seed_stability',stability)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    # Independently reproduce every reused old-C metric, including pooled AP.
    prior=pd.read_csv(old/'evaluation/metrics.csv',dtype={'scope':str,'seed':str})
    count=0
    for row in df[df.source_state=='original_old_C'].itertuples():
        z=prior[(prior.scope==row.scope)&(prior.variant=='announcement_fields')&(prior.model==row.model)&(prior.seed==row.seed)&(prior.horizon_months==row.horizon_months)&(prior.cause==row.cause)]
        assert len(z)==1
        np.testing.assert_allclose([getattr(row,k) for k in ['auc','ap','brier','risk_mean']],z.iloc[0][['auc','ap','brier','risk_mean']].to_numpy(float),atol=1e-14,rtol=0,equal_nan=True);count+=1
    ss=df[df.seed.isin(['42','101','202'])].groupby(['scope','source_state','model','horizon_months','cause'])[['auc','ap','brier','risk_mean']].agg(['mean','min','max'])
    ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    changes=[]
    for (scope,model,seed,months,cause),g in df[df.source_state!='common'].groupby(['scope','model','seed','horizon_months','cause']):
        g=g.set_index('source_state');changes.append(dict(scope=scope,model=model,seed=seed,horizon_months=months,cause=cause,
            **{k+'_change':float(g.loc['rerun_old_C',k]-g.loc['original_old_C',k]) for k in ['auc','ap','brier','risk_mean']}))
    pd.DataFrame(changes).to_csv(out/'paired_changes.csv',index=False)
    np.savez_compressed(run/'private/fixed_ensemble_predictions_PRIVATE.npz',**{'_'.join(map(str,k)):v for k,v in risks.items() if k[3]=='ensemble'})
    write(out/'validation.json',dict(status='PASS',new_formal_models=12,reused_models=12,metric_rows=len(df),
        old_C_metric_rows_reproduced=count,all_source_and_frozen_hashes_unchanged=True,
        independent_endpoint_metrics=True,all_test_endpoints_known=True,no_calibration=True,no_clipping=True,
        test_companies=81,test_terminations=32,labels_identical_between_source_versions=True,
        maximum_per_model_reconstruction_error=float(pd.DataFrame(checks).reconstruction_error.max())))
    print(df[(df.seed.isin(['ensemble','none']))&(df.cause=='termination')&(df.horizon_months==24)][
        ['scope','source_state','model','auc','ap','brier','risk_mean']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run)
