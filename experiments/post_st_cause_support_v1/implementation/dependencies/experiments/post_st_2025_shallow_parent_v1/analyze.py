"""Evaluate supported endpoints only, keeping fixed predictions and every seed."""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from experiments.post_st_2025_shallow_parent_v1.prepare import horizon_summary


def outputs(run):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS'
    for i,line in enumerate((run/'formal.jsonl').read_text().splitlines()):
        argv=json.loads(line)['argv'];model=argv[argv.index('--model')+1];seed=argv[argv.index('--seed')+1]
        p=run/'results/formal'/str(gate['job'])/f'task-{i}'/'output'
        yield model,seed,p/'shrunk' if model=='xai_q2' else p


def evaluate(run):
    for n,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/n)==h
    for n,h in json.loads((run/'audit/data_alignment.json').read_text())['unchanged_source_hashes'].items():assert sha(n)==h
    out=run/'evaluation';out.mkdir(exist_ok=False)
    ev=arrays(run/'evaluation_input/2025_test_PRIVATE.npz');tr=arrays(run/'input/folds/2025/train_PRIVATE.npz')
    meta=pd.read_parquet(run/'evaluation_input/2025_test_metadata_PRIVATE.parquet')
    assert np.array_equal(ev['sample_ids'],meta.sample_id) and meta.group.is_unique
    pred={};diag=[];hashes=[];terms={};complexity={};folders={}
    for model,seed,p in outputs(run):
        for n,h in json.loads((p/'hashes.json').read_text()).items():assert sha(p/n)==h
        z=arrays(p/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids'])
        pred[(model,seed)]=z;folders[(model,seed)]=p
        s=json.loads((p/'summary.json').read_text());s.pop('input_hashes',None);s.update(model=model,seed=seed);diag.append(s)
        hashes.append(dict(model=model,seed=seed,forest_sha256=sha(p/('forest_PRIVATE.json' if model=='xai_q2' else 'forest_PRIVATE.rds')),
            prediction_sha256=sha(p/'predictions_PRIVATE.npz'),functions_sha256=sha(p/'functions_PRIVATE.npz') if model=='xai_q2' else None))
        if model=='xai_q2':
            terms[seed]=arrays(p/'components_PRIVATE.npz');complexity[seed]=json.loads((p/'complexity_PRIVATE.json').read_text())
    assert set(pred)=={(m,str(s)) for m in ['xai_q2','standard_rsf'] for s in [42,101,202]}
    for model in ['xai_q2','standard_rsf']:
        members=[pred[(model,str(s))] for s in [42,101,202]]
        assert all(np.array_equal(z['times'],members[0]['times']) for z in members)
        pred[(model,'ensemble')]=dict(times=members[0]['times'],cif=np.mean([z['cif'] for z in members],axis=0),
            training_cif=np.mean([z['training_cif'] for z in members],axis=0),sample_ids=members[0]['sample_ids'])
    # Aggregate the COMPLETE native term sets, including terms present in one seed only.
    union=sorted(set().union(*(set(x['keys'].tolist()) for x in terms.values())))
    template=next(iter(terms.values()));combined=np.zeros((len(tr['X'])+len(ev['X']),len(union),len(template['times']),2))
    baseline=np.mean([x['baseline'] for x in terms.values()],axis=0);index={k:i for i,k in enumerate(union)}
    for x in terms.values():
        for j,key in enumerate(x['keys']):combined[:,index[key]]+=x['terms'][:,j]/3
    expected=np.vstack([pred[('xai_q2','ensemble')]['training_cif'],pred[('xai_q2','ensemble')]['cif']])
    error=float(abs(baseline+combined.sum(1)-expected).max());assert error<1e-10
    np.savez_compressed(run/'private/ensemble_functions_PRIVATE.npz',keys=np.array(union),terms=combined,baseline=baseline,times=template['times'])
    metrics=[];warnings=[];bins=[];ranked=[];support=[]
    for months in [12,24]:
        ss=horizon_summary(meta,months);support.append(ss)
        for (model,seed),p in pred.items():
            risk=horizon_predictions(p['cif'],p['times'],ev[f'h{months}'])
            for cause,name in [(1,'termination'),(2,'recovery')]:
                common=dict(year=2025,model=model,seed=seed,horizon_months=months,cause=name,
                    observation_cutoff='2026-06-30',weighting='one_company_one_record')
                if not ss['endpoint_evaluation_supported']:
                    metrics.append({**common,'companies':len(meta),'auc':np.nan,'ap':np.nan,'brier':np.nan,
                        'risk_mean':float(risk[:,cause-1].mean()),'status':'UNSUPPORTED_HORIZON',
                        'unknown_outcomes':ss['unknown'],'note':'No company reaches24months; observed first events do not identify the remaining outcomes.'})
                    continue
                result,w,y,known=metric(ev['analysis_time'],ev['event'],float(months),risk[:,cause-1],cause,ev['label_eligible'])
                if known.all():
                    np.testing.assert_allclose(w,np.ones(len(w)),rtol=0,atol=1e-12)
                    np.testing.assert_allclose([result[k] for k in ['auc','ap','brier']],direct_metrics(y,risk[:,cause-1]),rtol=0,atol=5e-14)
                metrics.append({**common,**result,'mean_risk_minus_observed':result['risk_mean']-result['ipcw_observed_incidence'],'status':'SUPPORTED'})
                if seed=='ensemble':
                    for b,ids in enumerate(np.array_split(np.argsort(risk[:,cause-1],kind='stable'),4)):
                        bins.append({**common,'risk_quartile_low_to_high':b+1,'n':len(ids),'predicted_mean':float(risk[ids,cause-1].mean()),
                            'ipcw_observed':float(np.sum(w[ids]*y[ids])/len(ids)),'events':int((known[ids]&y[ids]).sum()),'unknown':int((~known[ids]).sum())})
                if cause!=1:continue
                order=np.lexsort((ev['sample_ids'],-risk[:,0]))
                if seed=='ensemble':
                    for rank,i in enumerate(order,1):ranked.append(dict(model=model,rank=rank,sample_id=ev['sample_ids'][i],
                        risk=float(risk[i,0]),event=int(ev['event'][i]),event_within_horizon=bool((ev['event'][i]>0)&(ev['analysis_time'][i]<=months)),
                        target=bool(known[i]&y[i]),known=bool(known[i]),horizon_months=months))
                for fraction in [.2,.25]:
                    choose=order[:int(np.ceil(len(meta)*fraction))];chosen=np.zeros(len(meta),bool);chosen[choose]=True
                    lead={m:np.array([(o+pd.DateOffset(months=m)-o).days for o in meta.origin]) for m in [3,6]}
                    target=known&y
                    warnings.append({**common,'top_fraction':fraction,'checked':len(choose),'captured':int((chosen&target).sum()),
                        'all_target_events':int(target.sum()),'coverage':float((chosen&target).sum()/target.sum()) if target.any() else np.nan,
                        'known_non_target':int((chosen&known&~y).sum()),'unknown':int((chosen&~known).sum()),
                        'lead3':int((chosen&target&(ev['time']>=lead[3])).sum()),'lead6':int((chosen&target&(ev['time']>=lead[6])).sum()),
                        'ranking':'retrospective within-entry-year; not an online alert policy'})
    stability=[]
    for a,b in itertools.combinations(['42','101','202'],2):
        select=lambda s,n:{tuple(z['features']) for z in complexity[s]['functions'] if z['nonzero'] and z['order']==n}
        for n in [1,2]:
            sa,sb=select(a,n),select(b,n)
            stability.append(dict(seed_a=a,seed_b=b,order=n,selected_a=len(sa),selected_b=len(sb),common=len(sa&sb),
                jaccard=len(sa&sb)/max(len(sa|sb),1),interpretation='selection overlap only, not proof of stable function shape'))
    for name,rows in [('metrics',metrics),('warning_metrics',warnings),('calibration_bins',bins),('training_diagnostics',diag),
        ('model_hashes',hashes),('horizon_support',support),('seed_stability',stability)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    pd.DataFrame(ranked).to_csv(run/'private/ranked_predictions_PRIVATE.csv',index=False)
    summary=pd.DataFrame(metrics);summary=summary[(summary.seed!='ensemble')&(summary.status=='SUPPORTED')]
    ss=summary.groupby(['model','horizon_months','cause'])[['auc','ap','brier','risk_mean']].agg(['mean','min','max','std'])
    ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(out/'seed_summary.csv',index=False)
    validation=dict(status='PASS',formal_models=6,seed_metric_rows=24,ensemble_metric_rows=8,
        supported_horizon_months=[x['horizon_months'] for x in support if x['endpoint_evaluation_supported']],
        unsupported_horizon_months=[x['horizon_months'] for x in support if not x['endpoint_evaluation_supported']],
        ensemble_full_function_reconstruction_error=error,ensemble_main_keys=sum(',' not in k for k in union),
        ensemble_pair_keys=sum(',' in k for k in union),all_frozen_sources_and_outputs_verified=True,
        unsupported_horizon_metrics_suppressed=True,unknown_outcomes_never_negative=True,new_calibration_fits=0,
        current_reference_unchanged=True)
    write(out/'validation.json',validation)
    print(pd.DataFrame(metrics).query("seed=='ensemble' and cause=='termination'")[['model','horizon_months','auc','ap','brier','risk_mean','status']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);evaluate(p.parse_args().run)
