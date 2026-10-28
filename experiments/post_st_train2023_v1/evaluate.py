"""Fixed-prediction evaluation; never chooses a model/configuration or refits."""
from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest.core import aalen_johansen
from experiments.post_st_prognosis_v1.common import arrays,sha,write,endpoint_weights,horizon_predictions,calendar_elapsed


def metric(time,event,h,p,cause,eligible):
    w,known,diag=endpoint_weights(time,event,h,eligible)
    y=(event==cause)&(time<=h)
    active=w>0; positives=int((known&y).sum()); negatives=int((known&~y).sum())
    auc=float(roc_auc_score(y[active],p[active],sample_weight=w[active])) if positives and negatives else np.nan
    ap=float(average_precision_score(y[active],p[active],sample_weight=w[active])) if positives else np.nan
    return dict(companies=len(p),known_target_events=positives,known_other_outcomes=negatives,
        unknown_outcomes=int((~known).sum()),auc=auc,ap=ap,
        brier=float(np.sum(w*(p-y)**2)/len(p)),risk_mean=float(p.mean()),risk_min=float(p.min()),risk_max=float(p.max()),
        ipcw_observed_incidence=float(np.sum(w*y)/len(p)),ipcw_min_G=diag['min_G'],
        metric_note='no observed positive events: AUC/AP undefined' if not positives else 'AUC undefined: no controls' if not negatives else ''),w,y,known


def evaluate(run,output):
    output.mkdir(exist_ok=False,parents=True)
    plan=json.loads((run/'input/plan.json').read_text())
    assert json.loads((run/'audit/formal_gate.json').read_text())['status']=='PASS'
    train=arrays(run/'input/train_PRIVATE.npz');ev=arrays(run/'input/eval_PRIVATE.npz')
    meta=pd.read_parquet(run/'input/eval_metadata_PRIVATE.parquet')
    trainmeta=pd.read_parquet(run/'input/train_metadata_PRIVATE.parquet')
    for data,dd in [(train,trainmeta),(ev,meta)]:
        data['analysis_time']=np.array([calendar_elapsed(o,s) for o,s in zip(dd.origin,dd.stop)])
        assert np.array_equal(data['time']>=data['h24'],data['analysis_time']>=24)
        assert np.array_equal(data['time']>=data['h12'],data['analysis_time']>=12)
    metrics=[];warnings=[];calibration=[];diagnostics=[];forecasts={};details={}
    files=sorted((run/'results/formal').glob('*/task-*/output/summary.json'));assert len(files)==9
    for path in files:
        s=json.loads(path.read_text());out=path.parent
        for name,digest in json.loads((out/'hashes.json').read_text()).items():assert sha(out/name)==digest
        p=arrays(out/'predictions_PRIVATE.npz');assert np.array_equal(p['sample_ids'],ev['sample_ids'])
        key=(s['model'],str(s['seed']));assert key not in forecasts
        forecasts[key]=p;details[key]=out
        diagnostics.append({('randomForestSRC_version' if k=='r_version' else k):v
                            for k,v in s.items() if k not in ['input_hashes']})
    for model in plan['variants']:
        members=[forecasts[(model,str(s))] for s in plan['seeds']]
        forecasts[(model,'ensemble')]=dict(times=members[0]['times'],cif=np.mean([p['cif'] for p in members],axis=0),
                                          training_cif=np.mean([p['training_cif'] for p in members],axis=0))
    times=np.array([365.,366.,730.,731.])
    baseline=aalen_johansen(train['time'],train['event'],times,n_causes=2)
    forecasts[('training_AJ','none')]=dict(times=times,cif=np.broadcast_to(baseline,(len(ev['X']),4,2)),
                                        training_cif=np.broadcast_to(baseline,(len(train['X']),4,2)))
    for (model,seed),p in forecasts.items():
        for scope in ['2024','train_apparent']:
            data=train if scope=='train_apparent' else ev
            mask=np.ones(len(data['X']),bool) if scope in ['pooled','train_apparent'] else data['year']==int(scope)
            pp=p['training_cif'] if scope=='train_apparent' else p['cif']
            for months in [12,24]:
                h=data[f'h{months}'];risks=horizon_predictions(pp,p['times'],h)
                for cause,name in [(1,'termination'),(2,'recovery')]:
                    common=dict(model=model,seed=seed,scope=scope,horizon_months=months,cause=name,weighting='one_company_one_record')
                    t=data['time'][mask];e=data['event'][mask];hh=h[mask];risk=risks[mask,cause-1];eligible=data['label_eligible'][mask]
                    result,w,y,known=metric(data['analysis_time'][mask],e,np.full(len(t),float(months)),risk,cause,eligible)
                    # For complete test endpoints all weights must be one. This
                    # catches spurious leap-day administrative censoring effects.
                    if known.all():np.testing.assert_allclose(w,np.ones(len(w)),atol=1e-12,rtol=0)
                    metrics.append({**common,**result})
                    if scope!='train_apparent' and seed in ['ensemble','none']:
                        ordering=np.argsort(risk,kind='stable')
                        for b,ids in enumerate(np.array_split(ordering,5)):
                            calibration.append({**common,'bin':b+1,'n':len(ids),'predicted':float(risk[ids].mean()),
                                'ipcw_observed':float(np.sum(w[ids]*y[ids])/len(ids)),
                                'events':int((known[ids]&y[ids]).sum()),'unknown':int((~known[ids]).sum())})
                    if scope=='train_apparent' or cause!=1:continue
                    yy=data['year'][mask];ids=data['sample_ids'][mask];mm=meta.loc[mask].reset_index(drop=True)
                    for fraction in [.20,.25]:
                        chosen=np.zeros(len(risk),bool)
                        for year in np.unique(yy):
                            index=np.flatnonzero(yy==year);order=np.lexsort((ids[index],-risk[index]));ncheck=int(np.ceil(len(index)*fraction))
                            chosen[index[order[:ncheck]]]=True
                        target=known&y
                        lead3=np.array([(o+pd.DateOffset(months=3)-o).days for o in mm.origin])
                        lead6=np.array([(o+pd.DateOffset(months=6)-o).days for o in mm.origin])
                        warnings.append({**common,'top_fraction':fraction,'checked_companies':int(chosen.sum()),
                            'captured_events':int((chosen&target).sum()),'all_target_events':int(target.sum()),
                            'event_coverage':float((chosen&target).sum()/target.sum()) if target.any() else np.nan,
                            'known_non_target_checked':int((chosen&known&~y).sum()),'unknown_checked':int((chosen&~known).sum()),
                            'identified_at_least_3_calendar_months_before_termination':int((chosen&target&(t>=lead3)).sum()),
                            'identified_at_least_6_calendar_months_before_termination':int((chosen&target&(t>=lead6)).sum()),
                            'repeated_alerts':0,'rank_policy':'retrospective within-entry-year, not an online policy'})
    df=pd.DataFrame(metrics);df.to_csv(output/'metrics.csv',index=False)
    numeric=['auc','ap','brier','risk_mean','ipcw_observed_incidence']
    summary=df[df.seed.isin(list(map(str,plan['seeds'])))].groupby(['model','scope','horizon_months','cause'])[numeric].agg(['mean','min','max','std'])
    summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(output/'metrics_summary.csv',index=False)
    pd.DataFrame(warnings).to_csv(output/'warning_metrics.csv',index=False)
    pd.DataFrame(calibration).to_csv(output/'calibration.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'training_diagnostics.csv',index=False)
    qrows=[];stability=[];parts={}
    for seed in map(str,plan['seeds']):
        out=details[('xai_q2',seed)];s=json.loads((out/'summary.json').read_text());cc=json.loads((out/'complexity_PRIVATE.json').read_text())
        components=arrays(out/'components_PRIVATE.npz');parts[seed]=(cc,components)
        for months in [12,24]:
            h=train[f'h{months}'];ix=np.searchsorted(components['times'],h);k=len(components['keys'])
            terms=components['terms'][:len(h)][np.arange(len(h))[:,None],np.arange(k)[None,:],ix[:,None],:]
            for cause in range(2):
                meanabs=np.mean(abs(terms[:,:,cause]),axis=0);order=np.array([len(x.split(',')) for x in components['keys']])
                qrows.append(dict(seed=seed,horizon_months=months,cause=['termination','recovery'][cause],
                    native_all_time_main_functions=s['main_functions'],native_all_time_pair_functions=s['interaction_functions'],
                    spatial_table_cells=s['spatial_table_cells'],scalar_table_cells=s['scalar_table_cells'],
                    reference_active_mains=int(((meanabs>1e-14)&(order==1)).sum()),reference_active_pairs=int(((meanabs>1e-14)&(order==2)).sum()),
                    reference_pair_absolute_share=float(meanabs[order==2].sum()/meanabs.sum()) if meanabs.sum()>0 else 0.,
                    reconstruction_error=s['complete_component_sum_error'],low_support_cell_fraction=s['low_support_cell_fraction'],
                    zero_support_cell_fraction=s['zero_support_cell_fraction']))
    for a,b in itertools.combinations(map(str,plan['seeds']),2):
        ca,xa=parts[a];cb,xb=parts[b]
        maina={tuple(x['features']) for x in ca['functions'] if x['nonzero'] and x['order']==1}
        mainb={tuple(x['features']) for x in cb['functions'] if x['nonzero'] and x['order']==1}
        paira={tuple(x['features']) for x in ca['functions'] if x['nonzero'] and x['order']==2}
        pairb={tuple(x['features']) for x in cb['functions'] if x['nonzero'] and x['order']==2}
        common=set(xa['keys'])&set(xb['keys']);ia={k:i for i,k in enumerate(xa['keys'])};ib={k:i for i,k in enumerate(xb['keys'])}
        for cause in range(2):
            corr=[]
            for k in common:
                va=xa['terms'][:len(train['X']),ia[k],2,cause];vb=xb['terms'][:len(train['X']),ib[k],2,cause]
                if np.std(va)>1e-12 and np.std(vb)>1e-12:corr.append(float(np.corrcoef(va,vb)[0,1]))
            stability.append(dict(seed_a=a,seed_b=b,cause=['termination','recovery'][cause],
                main_jaccard=len(maina&mainb)/max(len(maina|mainb),1),pair_jaccard=len(paira&pairb)/max(len(paira|pairb),1),
                common_nonconstant_functions=len(corr),median_function_correlation=float(np.median(corr)) if corr else np.nan,
                comparison='same fixed training reference, fixed day 730; 3 seeds, no confidence interval'))
    pd.DataFrame(qrows).to_csv(output/'interpretability.csv',index=False)
    pd.DataFrame(stability).to_csv(output/'seed_stability.csv',index=False)
    # Preserve predictions and all component identities exclusively in private storage.
    write(output/'evaluation_receipt.json',dict(status='COMPLETE',metrics_rows=len(df),formal_models=len(files),
        company_and_record_weights_identical=True,test_companies=len(ev['X']),training_companies=len(train['X']),
        known_test_h24=int(((ev['time']>=ev['h24'])|((ev['event']>0)&(ev['time']<=ev['h24']))).sum()),
        test_outcome_tuning=False,new_fits_in_evaluation=0,input_hashes=plan['source_hashes'],
        censoring_clock='exact calendar months',scope='2024 only; original calendar-month evaluation semantics preserved'))
    print(df[(df.seed.isin(['ensemble','none']))&(df.scope!='train_apparent')&(df.cause=='termination')][
        ['model','scope','horizon_months','companies','known_target_events','auc','ap','brier','risk_mean','ipcw_observed_incidence']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();evaluate(args.run,args.output)
