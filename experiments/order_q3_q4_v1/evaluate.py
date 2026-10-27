"""Manufacturing-only refits and frozen all-sector references on identical rows."""
from pathlib import Path
import argparse,hashlib,json,sys,platform,importlib.metadata
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)


def main(run):
    inp=run/'input';out=run/'reports';plan=json.loads((inp/'plan.json').read_text())
    assert not (out/'execution_receipt.json').exists(), 'Preserve completed evaluation; use a distinct output for any revision.'
    for name,digest in plan['source_hashes'].items():assert sha(inp/name)==digest
    immutable=json.loads((run/'audit/immutable_input_hashes.json').read_text())
    for name,digest in immutable.items():assert sha(inp/name)==digest
    m2_provenance=json.loads((run/'audit/M2_provenance.json').read_text())
    for item in m2_provenance:
        assert sha(item['path'])==item['sha256']
        assert item['train_records']==json.loads((run/'audit/data_alignment.json').read_text())['allsector_model_training_records']
    for stage in ['pilot','training','assembly']:
        assert json.loads((run/f'audit/{stage}_gate.json').read_text())['status']=='PASS'
    sys.path.insert(0,str(inp/'legacy'))
    from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    from evaluate import top_flags,warning_metrics,event_leads,lead_summary
    data=pd.read_parquet(inp/'final_eval_metadata_PRIVATE.parquet').reset_index(drop=True)
    assert set(data.year)=={2023,2024}
    assert data.sample_id.is_unique
    # Legacy label_eligible covers target/survivor endpoints; observed competing
    # events are known non-target results through their event code, independently.
    assert data.loc[data.event==1,'label_eligible'].all()
    preds={};diagnostics=[];model_hashes={};prediction_hashes={};paths={}
    for p in (run/'results/assembled').glob('*/task-*/output/summary.json'):
        s=json.loads(p.read_text());assert not s['pilot'] and s['status']=='COMPLETED_PROVISIONAL'
        key=(s['variant'],s['seed']);assert key not in preds
        paths[key]=p.parent
        for name,digest in json.loads((p.parent/'output_hashes.json').read_text()).items():assert sha(p.parent/name)==digest
        with np.load(p.parent/'predictions_PRIVATE.npz',allow_pickle=False) as z:
            assert np.array_equal(z['sample_ids'],data.sample_id.to_numpy())
            assert np.array_equal(z['groups'],data.group.to_numpy(dtype=str))
            assert np.array_equal(z['horizon_days'],(data.horizon-data.origin).dt.days.to_numpy())
            preds[key]=z['cif12'][:,0]
        diagnostics.append({k:v for k,v in s.items() if k not in ['training','validation']})
        model_hashes[f'{key[0]}/{key[1]}']=sha(p.parent/'forest_PRIVATE.json')
        prediction_hashes[f'{key[0]}/{key[1]}']=sha(p.parent/'predictions_PRIVATE.npz')
    assert set(preds)=={(v,s) for v in plan['variants'] for s in plan['seeds']}
    for v in ['q2','unrestricted']:
        for seed in plan['seeds']:
            with np.load(inp/f'reference_{v}_{seed}_PRIVATE.npz',allow_pickle=False) as z:
                assert np.array_equal(z['sample_ids'],data.sample_id)
                assert np.array_equal(z['horizon_days'],(data.horizon-data.origin).dt.days.to_numpy())
                preds[v,seed]=z['cif12'][:,0]
    for v in [*plan['variants'],'q2','unrestricted']:
        preds[v,-1]=np.mean([preds[v,s] for s in plan['seeds']],axis=0)
    with np.load(inp/'M2_PRIVATE.npz',allow_pickle=False) as m:
        assert np.array_equal(m['sample_ids'],data.sample_id)
        assert np.array_equal(m['seeds'],plan['seeds'])
        for j,s in enumerate(plan['seeds']):preds['M2_allsector',s]=m['probabilities'][j]
        preds['M2_allsector',-1]=m['ensemble']
        assert np.array_equal(np.mean(m['probabilities'],axis=0),m['ensemble'])
    metrics=[];warnings=[];calibration=[]
    scopes=[('2023',np.flatnonzero(data.year==2023)),('2024',np.flatnonzero(data.year==2024)),('pooled',np.arange(len(data)))]
    for scope,indices in scopes:
        d=data.iloc[indices].reset_index(drop=True)
        w,wa=source_consistent_endpoint_weights(d,endpoint_weights);y=(d.event.to_numpy()==1).astype(int)
        for (model,seed),prediction in sorted(preds.items()):
            p=prediction[indices];assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
            aggregation='seed_ensemble' if seed==-1 else 'single_seed'
            for weighting in ['record_equal','company_equal']:
                base=np.ones(len(d)) if weighting=='record_equal' else 1/d.groupby('group').group.transform('size').to_numpy()
                ww=w*base;keep=ww>0
                assert set(y[keep])=={0,1}
                metrics.append(dict(scope=scope,model=model,seed=seed,aggregation=aggregation,weighting=weighting,records=len(d),
                    auc=float(roc_auc_score(y[keep],p[keep],sample_weight=ww[keep])),
                    ap=float(average_precision_score(y[keep],p[keep],sample_weight=ww[keep])),
                    ipcw_brier=float(np.sum(ww*(y-p)**2)/base.sum()),
                    prediction_mean=float(np.average(p,weights=base)),observed_ipcw_incidence=float(np.sum(ww*y)/base.sum()),
                    prediction_min=float(p.min()),prediction_max=float(p.max()),prediction_p95=float(np.quantile(p,.95)),prediction_p99=float(np.quantile(p,.99)),
                    risk_quantile_weighting='record_equal, even on company-weighted metric rows',**wa))
                for b,ii in enumerate(np.array_split(np.lexsort((d.sample_id.to_numpy(),p)),10)):
                    calibration.append(dict(scope=scope,model=model,seed=seed,aggregation=aggregation,weighting=weighting,bin=b+1,n=len(ii),
                        prediction=float(np.average(p[ii],weights=base[ii])),observed=float(np.sum(ww[ii]*y[ii])/base[ii].sum()),known=int(keep[ii].sum())))
            for fraction in [.2,.25]:
                flags=top_flags(d,p,fraction)
                slices=[(scope,np.arange(len(d)))]
                if scope!='pooled':slices += [(str(q),np.asarray(ii)) for q,ii in d.groupby('quarter').indices.items()]
                for label,ii in slices:
                    dd=d.iloc[ii].reset_index(drop=True);ff=flags[ii]
                    warnings.append(dict(scope=label,model=model,seed=seed,aggregation=aggregation,top_fraction=fraction,
                        **warning_metrics(dd,ff),**lead_summary(event_leads(dd,ff))))
    m=pd.DataFrame(metrics);w=pd.DataFrame(warnings);diag=pd.DataFrame(diagnostics)
    assert np.array_equal(w.alert_records, w.hit_records+w.known_no_target_alerts+w.unknown_outcome_alerts)
    assert (w.detected_events<=w.independent_events).all()
    assert (w.announcement_detected_6m<=w.announcement_detected_3m).all()
    assert (w.announcement_detected_3m<=w.detected_events).all()
    m.to_csv(out/'metrics.csv',index=False);w.to_csv(out/'warning_metrics.csv',index=False)
    pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False);diag.to_csv(out/'training_diagnostics.csv',index=False)
    means=m[m.aggregation=='single_seed'].groupby(['scope','model','weighting'],as_index=False).agg(
        auc_mean=('auc','mean'),auc_min=('auc','min'),auc_max=('auc','max'),ap_mean=('ap','mean'),ap_min=('ap','min'),ap_max=('ap','max'),
        brier_mean=('ipcw_brier','mean'),prediction_mean=('prediction_mean','mean'),prediction_max_min=('prediction_max','min'),prediction_max_max=('prediction_max','max'))
    means.to_csv(out/'metrics_seed_summary.csv',index=False)
    m[m.aggregation=='seed_ensemble'].to_csv(out/'metrics_ensemble.csv',index=False)
    ws=w[(w.aggregation=='single_seed')&w.scope.isin(['2023','2024','pooled'])].groupby(['scope','model','top_fraction'],as_index=False).agg(
        events_mean=('detected_events','mean'),events_min=('detected_events','min'),events_max=('detected_events','max'),independent_events=('independent_events','first'),
        lead3_mean=('announcement_detected_3m','mean'),lead6_mean=('announcement_detected_6m','mean'),unknown_mean=('unknown_outcome_alerts','mean'),
        known_no_target_mean=('known_no_target_alerts','mean'),alert_records=('alert_records','first'),companies_mean=('alert_company_groups','mean'),repeats_mean=('repeated_alert_records','mean'))
    ws.to_csv(out/'warning_seed_summary.csv',index=False)
    w[(w.aggregation=='seed_ensemble')&w.scope.isin(['2023','2024','pooled'])].to_csv(out/'warning_ensemble.csv',index=False)
    contrasts=[]
    for scope,_ in scopes:
        for weighting in ['record_equal','company_equal']:
            for seed in plan['seeds']:
                a=m.query('scope==@scope and weighting==@weighting and seed==@seed').set_index('model')
                for lhs,rhs in [('q3','q2'),('q4','q2'),('q4','q3'),('q3','unrestricted'),('q4','unrestricted'),('q3','M2_allsector'),('q4','M2_allsector')]:
                    contrasts.append(dict(scope=scope,weighting=weighting,seed=seed,comparison=lhs+' minus '+rhs,**{k:float(a.loc[lhs,k]-a.loc[rhs,k]) for k in ['auc','ap','ipcw_brier']}))
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    # Same fixed training reference for order-specific conditional seed stability.
    from crforest import CIFExpansion,HigherOrderCIFExpansion
    with np.load(inp/'final_train_PRIVATE.npz',allow_pickle=False) as z:
        xx=z['X'][np.linspace(0,len(z['X'])-1,512,dtype=int)]
    stability=[]
    for model in ['q2',*plan['variants']]:
        expansions={s:(CIFExpansion.load(inp/f'reference_q2_{s}_functions_PRIVATE.npz') if model=='q2'
                       else HigherOrderCIFExpansion.load(paths[model,s]/'functions_PRIVATE.npz')) for s in plan['seeds']}
        for sa,sb in combinations(plan['seeds'],2):
            a,b=expansions[sa],expansions[sb]
            ak=set(a.keys if model!='q2' else [(j,) for j in a.mains]+a.pairs)
            bk=set(b.keys if model!='q2' else [(j,) for j in b.mains]+b.pairs)
            for order in range(1,int(model[-1])+1):
                aa={k for k in ak if len(k)==order};bb={k for k in bk if len(k)==order};corr=[];sign=[]
                for key in sorted(aa&bb):
                    u=a.component(xx,key,[365])[:,0,0];v=b.component(xx,key,[365])[:,0,0]
                    if u.std()>1e-12 and v.std()>1e-12:corr.append(float(np.corrcoef(u,v)[0,1]))
                    mask=(abs(u)>1e-12)|(abs(v)>1e-12)
                    if mask.any():sign.append(float(np.mean(np.sign(u[mask])==np.sign(v[mask]))))
                stability.append(dict(model=model,seed_a=sa,seed_b=sb,order=order,reference_n=512,horizon_days=365,
                    functions_a=len(aa),functions_b=len(bb),common=len(aa&bb),jaccard=len(aa&bb)/len(aa|bb) if aa|bb else None,
                    nonconstant_common=len(corr),median_correlation=float(np.median(corr)) if corr else None,
                    median_sign_agreement=float(np.median(sign)) if sign else None))
    pd.DataFrame(stability).to_csv(out/'seed_stability.csv',index=False)
    # All expected identities, complete horizons and no silent dropped samples.
    assert len(m)==3*20*2 and len(diag)==6
    for scope,ix in scopes:
        assert set(m.loc[m.scope==scope,'records'])=={len(ix)}
    audit=json.loads((run/'audit/data_alignment.json').read_text())
    publication=[json.loads((run/f'audit/{stage}_gate.json').read_text()) for stage in ['pilot','training','assembly']]
    write(out/'execution_receipt.json',dict(status='COMPLETED_PROVISIONAL_POST_EXPOSURE_Q3_Q4',run=str(run),formal_fits=6,pilot_fits=2,training_tasks=48,assembly_tasks=6,
        separate_evaluation_years=[2023,2024],training_code=plan['code_commit'],plan_sha256=sha(inp/'plan.json'),frozen_protocol_sha256=sha(run/'audit/frozen_protocol.json'),
        fit_cutoff='2022-01-01',training_years=[2012,2020],source_counts=audit['source_counts'],model_hashes=model_hashes,prediction_hashes=prediction_hashes,
        publication=publication,failed_tasks=[],M2_refit=False,allsector_references_refit=False,industry_clock_certified=False,strict_blind=False,requested_path_order_configurations=2,post_exposure_algorithm_comparison=True,identifiability_penalty='NONE',
        immutable_input_files_verified=len(immutable),M2_provenance_sha256=sha(run/'audit/M2_provenance.json'),
        maximum_reconstruction_error=float(diag.reconstruction_error.max()),maximum_explicit_component_sum_error=float(diag.explicit_component_sum_error.max()),fit_cpu_seconds=float(diag.fit_cpu_seconds.sum()),
        assembly_validation_peak_RSS_MiB=float(diag.peak_rss_MiB.max()),
        training_shard_peak_RSS_MiB=float(diag.training_shard_peak_rss_MiB.max()),
        evaluation_script_sha256=sha(Path(__file__))))
    write(out/'environment.json',dict(evaluation_python=platform.python_version(),
        evaluation_packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow','pytest']},
        training_hosts=sorted(publication[1]['by_host']),assembly_hosts=sorted(set(diag.host)),assembly_python=sorted(set(diag.python)),assembly_numpy=sorted(set(diag.numpy)),
        thread_policy='one CPU affinity per item; BLAS=1; physical Slurm pool, no GPU'))
    print(means.query("weighting=='record_equal'").to_string(index=False),flush=True)
    print(m.query("weighting=='record_equal' and aggregation=='seed_ensemble'")[['scope','model','auc','ap','ipcw_brier','prediction_max']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
