"""Frozen-forecast evaluation of all arms and seeds; no tuning or refitting."""
from pathlib import Path
import argparse, itertools, json
import numpy as np
import pandas as pd
from crforest import CIFExpansion, CompetingRiskForest
from crforest.core import aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics


def outputs(run):
    gate = json.loads((run/'audit/formal_gate.json').read_text()); assert gate['status']=='PASS'
    for i,line in enumerate((run/'formal.jsonl').read_text().splitlines()):
        a=json.loads(line)['argv']; get=lambda k:a[a.index('--'+k)+1]
        yield int(get('year')),get('variant'),get('model'),get('seed'),run/'results/formal'/str(gate['job'])/f'task-{i}'/'output'


def evaluate(run):
    protocol=json.loads((run/'audit/protocol.json').read_text())
    for name,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items(): assert sha(run/'input'/name)==h
    for name,h in json.loads((run/'audit/data_alignment.json').read_text())['unchanged_source_hashes'].items(): assert sha(Path(name))==h
    out=run/'evaluation';out.mkdir(exist_ok=False)
    pred={};details={};summaries=[];checks=[];hashes=[];stability=[]
    for year,variant,model,seed,folder in outputs(run):
        for name,h in json.loads((folder/'hashes.json').read_text()).items(): assert sha(folder/name)==h
        key=(year,variant,model,seed);assert key not in pred
        p=arrays(folder/'predictions_PRIVATE.npz');pred[key]=p;details[key]=folder
        ev=arrays(run/'evaluation_input'/f'{year}_{variant}_test_PRIVATE.npz')
        np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids'])
        s=json.loads((folder/'summary.json').read_text());s.pop('input_hashes',None)
        s.update(test_year=year,variant=variant)
        if model=='xai_q2':
            f=CompetingRiskForest.load(folder/'forest_PRIVATE.json')
            assert tuple(f.config.cause_weights)==(1.,1.) and f.config.path_budget==2
            leaves=[n for t in f.trees_ for n in t if 'cif' in n]
            s.update(target_unsupported_leaf_fraction=float(np.mean([n['cause_company_support'][0]==0 for n in leaves])),
                     target_leaf_support_median=float(np.median([n['cause_company_support'][0] for n in leaves])))
        summaries.append(s)
        checks.append(dict(test_year=year,variant=variant,model=model,seed=seed,
            reconstruction_error=s.get('complete_component_sum_error'),reload_error=s['reload_error'],
            max_path_variables=s.get('max_path_variables'),probability_min=s['probability_min'],
            max_cause_probability_sum=s['max_cause_probability_sum']))
        hashes.append(dict(test_year=year,variant=variant,model=model,seed=seed,
            forest_sha256=sha(folder/('forest_PRIVATE.rds' if model=='standard_rsf' else 'forest_PRIVATE.json')),
            predictions_sha256=sha(folder/'predictions_PRIVATE.npz'),
            functions_sha256=sha(folder/'functions_PRIVATE.npz') if model=='xai_q2' else None))
    assert len(pred)==protocol['formal_fits']
    years=protocol['test_years'];variants=protocol['variants'];models=protocol['models']
    for year,variant,model in itertools.product(years,variants,models):
        parts=[pred[(year,variant,model,str(s))] for s in protocol['seeds']]
        pred[(year,variant,model,'ensemble')]=dict(sample_ids=parts[0]['sample_ids'],times=parts[0]['times'],
            cif=np.mean([p['cif'] for p in parts],axis=0))
    for year,variant in itertools.product(years,variants):
        tr=arrays(run/'input/folds'/str(year)/variant/'train_PRIVATE.npz');n=len(tr['X'])
        cc=[json.loads((details[(year,variant,'xai_q2',str(s))]/'complexity_PRIVATE.json').read_text()) for s in protocol['seeds']]
        parts=[arrays(details[(year,variant,'xai_q2',str(s))]/'components_PRIVATE.npz') for s in protocol['seeds']]
        for a,b in itertools.combinations(range(3),2):
            sets=lambda c,k:{tuple(x['features']) for x in c['functions'] if x['nonzero'] and x['order']==k}
            ma,mb=sets(cc[a],1),sets(cc[b],1);pa,pb=sets(cc[a],2),sets(cc[b],2)
            ia={k:i for i,k in enumerate(parts[a]['keys'])};ib={k:i for i,k in enumerate(parts[b]['keys'])}
            for cause in range(2):
                cor=[]
                for k in ia.keys()&ib.keys():
                    va=parts[a]['terms'][:n,ia[k],2,cause];vb=parts[b]['terms'][:n,ib[k],2,cause]
                    if np.std(va)>1e-12 and np.std(vb)>1e-12:cor.append(float(np.corrcoef(va,vb)[0,1]))
                stability.append(dict(test_year=year,variant=variant,seed_a=protocol['seeds'][a],seed_b=protocol['seeds'][b],
                    cause=['termination','recovery'][cause],main_jaccard=len(ma&mb)/max(len(ma|mb),1),
                    pair_jaccard=len(pa&pb)/max(len(pa|pb),1),common_nonconstant_functions=len(cor),
                    median_function_correlation=float(np.median(cor)) if cor else np.nan,
                    reference='same fold and information-set training reference, day730; three seeds, descriptive'))
    data={y:arrays(run/'evaluation_input'/f'{y}_market_clock_test_PRIVATE.npz') for y in years}
    meta={y:pd.read_parquet(run/'evaluation_input'/f'{y}_test_metadata_PRIVATE.parquet') for y in years}
    data['pooled']={k:np.concatenate([data[y][k] for y in years]) for k in data[years[0]]}
    meta['pooled']=pd.concat([meta[y] for y in years],ignore_index=True)
    risks={}
    for (year,variant,model,seed),p in pred.items():
        for months in [12,24]:risks[(str(year),variant,model,seed,months)]=horizon_predictions(p['cif'],p['times'],data[year][f'h{months}'])
    for year in years:
        tr=arrays(run/'input/folds'/str(year)/'market_clock/train_PRIVATE.npz');tt=np.array([365.,366.,730.,731.])
        baseline=aalen_johansen(tr['time'],tr['event'],tt)
        for months in [12,24]:risks[(str(year),'common','training_AJ','none',months)]=baseline[np.searchsorted(tt,data[year][f'h{months}'])]
    keys={(v,m,s,h) for y,v,m,s,h in risks}
    for v,m,s,h in keys:risks[('pooled',v,m,s,h)]=np.concatenate([risks[(str(y),v,m,s,h)] for y in years])
    metrics=[];warnings=[];bins=[]
    for (scope,variant,model,seed,months),risk in risks.items():
        yy='pooled' if scope=='pooled' else int(scope);ev=data[yy];mm=meta[yy]
        for k,cause in enumerate(['termination','recovery']):
            common=dict(scope=scope,variant=variant,model=model,seed=seed,cause=cause,
                        horizon_months=months,weighting='one_company_one_record')
            result,w,y,known=metric(ev['analysis_time'],ev['event'],np.full(len(risk),float(months)),risk[:,k],k+1,ev['label_eligible'])
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
                leads={n:np.array([(o+pd.DateOffset(months=n)-o).days for o in mm.origin]) for n in [3,6]}
                warnings.append({**common,'top_fraction':fraction,'checked':int(chosen.sum()),'captured':int((chosen&y&known).sum()),
                    'target_events':int((y&known).sum()),'coverage':float((chosen&y&known).sum()/(y&known).sum()) if (y&known).any() else np.nan,
                    'known_non_target':int((chosen&~y&known).sum()),'unknown':int((chosen&~known).sum()),'repeat_alerts':0,
                    'lead3':int((chosen&y&known&(ev['time']>=leads[3])).sum()),'lead6':int((chosen&y&known&(ev['time']>=leads[6])).sum()),
                    'selection':'retrospective annual ranking; not an online alert policy'})
    df=pd.DataFrame(metrics);df.to_csv(out/'metrics.csv',index=False)
    for name,records in [('warning_metrics',warnings),('calibration_bins',bins),('training_diagnostics',summaries),
                         ('mathematical_checks',checks),('model_hashes',hashes),('seed_stability',stability)]:
        pd.DataFrame(records).to_csv(out/f'{name}.csv',index=False)
    ss=df[df.seed.isin(['42','101','202'])].groupby(['scope','variant','model','horizon_months','cause'])[['auc','ap','brier','risk_mean']].agg(['mean','min','max'])
    ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    deltas=[]
    for v,base in [('financial_refresh','market_clock'),('announcement_fields','financial_refresh'),('announcement_fields','market_clock')]:
        for (scope,model,seed,months,cause),g in df[df.variant.isin([v,base])].groupby(['scope','model','seed','horizon_months','cause']):
            g=g.set_index('variant');deltas.append(dict(scope=scope,model=model,seed=seed,horizon_months=months,cause=cause,variant=v,baseline=base,
                **{n+'_change':float(g.loc[v,n]-g.loc[base,n]) for n in ['auc','ap','brier','risk_mean']}))
    pd.DataFrame(deltas).to_csv(out/'paired_changes.csv',index=False)
    np.savez_compressed(run/'private/fixed_ensemble_predictions_PRIVATE.npz',
        **{'_'.join(map(str,k)):v for k,v in risks.items() if k[3]=='ensemble'})
    write(out/'validation.json',dict(status='PASS',formal_models=len(summaries),metric_rows=len(df),
        source_and_frozen_inputs_unchanged=True,independent_metric_checks=True,all_test_endpoints_known=True,
        per_arm_labels_and_company_order_identical=True,independent_test_companies=len(data['pooled']['groups']),
        independent_test_terminations=int((data['pooled']['event']==1).sum()),
        no_calibration=True,no_clipping=True,no_test_selection=True,
        maximum_reconstruction_error=float(pd.DataFrame(checks).reconstruction_error.max())))
    print(df[(df.seed.isin(['ensemble','none']))&(df.cause=='termination')&(df.horizon_months==24)][
        ['scope','variant','model','auc','ap','brier','risk_mean']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run)
