"""Evaluate every fixed arm/seed on identical endpoints, without fitting."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import pandas as pd
from crforest import aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics


def artifacts(run):
    refs=json.loads((run/'audit/reference_models.json').read_text())
    for r in refs:
        yield r['year'],'baseline' if r['model']=='xai_q2' else 'standard_rsf',str(r['seed']),Path(r['path'])
    job=json.loads((run/'audit/formal_gate.json').read_text())['job']
    for i,line in enumerate((run/'formal.jsonl').read_text().splitlines()):
        argv=json.loads(line)['argv'];get=lambda k:argv[argv.index(k)+1]
        y=int(get('--year'));s=get('--seed');d=int(get('--depth'))
        folder=run/'results/formal'/str(job)/f'task-{i}'/'output'
        if d==4:yield y,'shallow',s,folder/'raw'
        yield y,'parent_shrink' if d==6 else 'shallow_parent',s,folder/'shrunk'


def evaluate(run):
    protocol=json.loads((run/'audit/protocol.json').read_text());years=protocol['test_years']
    for name,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/name)==h
    for name,h in json.loads((run/'audit/data_alignment.json').read_text())['unchanged_source_hashes'].items():assert sha(Path(name))==h
    out=run/'evaluation';out.mkdir(exist_ok=False);pred={};folders={};summaries=[];hashes=[];stability=[]
    arms=list(protocol['arms'])+['standard_rsf'];seeds=[str(s) for s in protocol['seeds']]
    for year,arm,seed,folder in artifacts(run):
        for name,h in json.loads((folder/'hashes.json').read_text()).items():assert sha(folder/name)==h
        key=(year,arm,seed);assert key not in pred
        p=arrays(folder/'predictions_PRIVATE.npz');pred[key]=p;folders[key]=folder
        ev=arrays(run/'evaluation_input'/f'{year}_test_PRIVATE.npz')
        np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids'])
        summary=json.loads((folder/'summary.json').read_text());summary.pop('input_hashes',None)
        summary.update(arm=arm,test_year=year,seed=seed);summaries.append(summary)
        hashes.append(dict(test_year=year,arm=arm,seed=seed,
            forest_sha256=sha(folder/('forest_PRIVATE.rds' if arm=='standard_rsf' else 'forest_PRIVATE.json')),
            predictions_sha256=sha(folder/'predictions_PRIVATE.npz'),
            functions_sha256=sha(folder/'functions_PRIVATE.npz') if arm!='standard_rsf' else None))
    assert len(pred)==30
    for year,arm in itertools.product(years,arms):
        members=[pred[(year,arm,s)] for s in seeds]
        assert all(np.array_equal(p['times'],members[0]['times']) for p in members)
        pred[(year,arm,'ensemble')]=dict(times=members[0]['times'],sample_ids=members[0]['sample_ids'],cif=np.mean([p['cif'] for p in members],axis=0))
    for year,arm in itertools.product(years,protocol['arms']):
        n=len(arrays(run/'input/folds'/str(year)/'train_PRIVATE.npz')['X'])
        cs=[json.loads((folders[(year,arm,s)]/'complexity_PRIVATE.json').read_text()) for s in seeds]
        ps=[arrays(folders[(year,arm,s)]/'components_PRIVATE.npz') for s in seeds]
        for a,b in itertools.combinations(range(3),2):
            sets=lambda c,k:{tuple(x['features']) for x in c['functions'] if x['nonzero'] and x['order']==k}
            ma,mb=sets(cs[a],1),sets(cs[b],1);pa,pb=sets(cs[a],2),sets(cs[b],2)
            ia={k:i for i,k in enumerate(ps[a]['keys'])};ib={k:i for i,k in enumerate(ps[b]['keys'])}
            for k,cause in enumerate(['termination','recovery']):
                correlations=[]
                for term in ia.keys()&ib.keys():
                    aa=ps[a]['terms'][:n,ia[term],2,k];bb=ps[b]['terms'][:n,ib[term],2,k]
                    if np.std(aa)>1e-12 and np.std(bb)>1e-12:correlations.append(float(np.corrcoef(aa,bb)[0,1]))
                stability.append(dict(test_year=year,arm=arm,seed_a=seeds[a],seed_b=seeds[b],cause=cause,
                    main_jaccard=len(ma&mb)/max(len(ma|mb),1),pair_jaccard=len(pa&pb)/max(len(pa|pb),1),
                    common_nonconstant_functions=len(correlations),median_function_correlation=float(np.median(correlations)) if correlations else np.nan,
                    reference='same old-C training reference/day730; common nonconstant terms only'))
    data={y:arrays(run/'evaluation_input'/f'{y}_test_PRIVATE.npz') for y in years}
    meta={y:pd.read_parquet(run/'evaluation_input'/f'{y}_test_metadata_PRIVATE.parquet') for y in years}
    data['pooled']={k:np.concatenate([data[y][k] for y in years]) for k in data[years[0]]}
    meta['pooled']=pd.concat([meta[y] for y in years],ignore_index=True);risks={}
    for (year,arm,seed),p in pred.items():
        for months in [12,24]:risks[(str(year),arm,seed,months)]=horizon_predictions(p['cif'],p['times'],data[year][f'h{months}'])
    for year in years:
        tr=arrays(run/'input/folds'/str(year)/'train_PRIVATE.npz');tt=np.array([365.,366.,730.,731.]);aj=aalen_johansen(tr['time'],tr['event'],tt)
        for months in [12,24]:risks[(str(year),'training_AJ','none',months)]=aj[np.searchsorted(tt,data[year][f'h{months}'])]
    for arm,seed,months in {(a,s,h) for y,a,s,h in risks}:
        risks[('pooled',arm,seed,months)]=np.concatenate([risks[(str(y),arm,seed,months)] for y in years])
    metrics=[];warnings=[];bins=[]
    for (scope,arm,seed,months),risk in risks.items():
        yy='pooled' if scope=='pooled' else int(scope);ev=data[yy];mm=meta[yy]
        for k,cause in enumerate(['termination','recovery']):
            common=dict(scope=scope,arm=arm,seed=seed,cause=cause,horizon_months=months,weighting='one_company_one_record')
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
    df=pd.DataFrame(metrics);wf=pd.DataFrame(warnings);df.to_csv(out/'metrics.csv',index=False)
    for name,rows in [('warning_metrics',warnings),('calibration_bins',bins),('training_diagnostics',summaries),('model_hashes',hashes),('seed_stability',stability)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    old=Path(protocol['reference_run']);prior=pd.read_csv(old/'evaluation/metrics.csv',dtype={'scope':str,'seed':str});count=0
    for row in df[df.arm.isin(['baseline','standard_rsf'])].itertuples():
        m='xai_q2' if row.arm=='baseline' else 'standard_rsf'
        z=prior[(prior.scope==row.scope)&(prior.source_state=='rerun_old_C')&(prior.model==m)&(prior.seed==row.seed)&(prior.horizon_months==row.horizon_months)&(prior.cause==row.cause)]
        assert len(z)==1
        np.testing.assert_allclose([getattr(row,k) for k in ['auc','ap','brier','risk_mean']],z.iloc[0][['auc','ap','brier','risk_mean']].to_numpy(float),atol=1e-14,rtol=0,equal_nan=True);count+=1
    ss=df[df.seed.isin(seeds)].groupby(['scope','arm','horizon_months','cause'])[['auc','ap','brier','risk_mean']].agg(['mean','min','max'])
    ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    changes=[];gates=[]
    for (scope,seed,months,cause),g in df[df.arm.isin(protocol['arms'])].groupby(['scope','seed','horizon_months','cause']):
        g=g.set_index('arm')
        for arm in list(protocol['arms'])[1:]:
            delta={k+'_change':float(g.loc[arm,k]-g.loc['baseline',k]) for k in ['auc','ap','brier','risk_mean']}
            changes.append(dict(scope=scope,arm=arm,seed=seed,horizon_months=months,cause=cause,**delta))
            if scope!='pooled' and seed=='ensemble' and months==24 and cause=='termination':
                ww=wf[(wf.scope==scope)&(wf.seed==seed)&(wf.horizon_months==months)&(wf.top_fraction==.2)].set_index('arm')
                capture=int(ww.loc[arm,'captured']-ww.loc['baseline','captured'])
                gates.append(dict(scope=scope,arm=arm,**delta,top20_capture_change=capture,
                    practical_metric_gate=bool(delta['auc_change']>=-.02 and delta['ap_change']>=-.02 and delta['brier_change']<=1e-12 and capture>=0)))
    pd.DataFrame(changes).to_csv(out/'paired_changes.csv',index=False);pd.DataFrame(gates).to_csv(out/'retention_gate.csv',index=False)
    np.savez_compressed(run/'private/fixed_ensemble_predictions_PRIVATE.npz',**{'_'.join(map(str,k)):v for k,v in risks.items() if k[2]=='ensemble'})
    write(out/'validation.json',dict(status='PASS',q2_models=24,standard_references=6,new_formal_forest_fits=6,new_leaf_updates=12,
        metric_rows=len(df),reference_metric_rows_reproduced=count,independent_endpoint_metrics=True,all_test_endpoints_known=True,
        all_source_and_frozen_hashes_unchanged=True,no_calibration=True,no_clipping=True,test_companies=81,test_terminations=32,
        maximum_per_model_reconstruction_error=float(pd.DataFrame(summaries).complete_component_sum_error.max())))
    print(df[(df.seed.isin(['ensemble','none']))&(df.cause=='termination')&(df.horizon_months==24)][['scope','arm','auc','ap','brier','risk_mean']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run)
