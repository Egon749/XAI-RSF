"""Common fixed-prediction evaluation and paired finite-design contrasts."""
from pathlib import Path
from itertools import combinations
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CompetingRiskForest,CIFExpansion


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def main(run):
    run=Path(run);inp=run/'input';out=run/'reports';plan=json.loads((inp/'plan.json').read_text())
    for name,digest in plan['source_hashes'].items():assert sha(inp/name)==digest
    recovery_path=run/'audit/recovery_plan.json'
    recovery_plan=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    stages=[('pilot',5),('training',156)]
    if recovery_plan:stages.append(('recovery',len(recovery_plan['tasks'])))
    stages.append(('assembly',18));publications=[]
    for stage,count in stages:
        reports=list(run.glob(f'{stage}.jsonl.results-*.json'));assert len(reports)==1
        pool=json.loads(reports[0].read_text());done=pool['completed']
        assert len(done)==count and {x['id'] for x in done}==set(range(count))
        failed=[x for x in done if x['returncode']!=0]
        if failed:
            assert stage=='training' and recovery_plan
            assert {x['id'] for x in failed}=={x['original_id'] for x in recovery_plan['tasks']}
            assert all('Out of range float values are not JSON compliant: -inf' in x['tail'] for x in failed)
        success=[x for x in done if x['returncode']==0]
        assert all(x['compute_completed'] and x['publication_completed'] for x in success)
        publications.append(dict(stage=stage,job=int(reports[0].stem.rsplit('-',1)[1]),tasks=count,
            elapsed_pool_seconds=pool['seconds'],mode=pool['mode'],
            completed_by_host={h:sum(x['host']==h for x in success) for h in sorted({x['host'] for x in done})},
            failed_original_ids=[x['id'] for x in failed],successful_tasks=len(success),
            allocated_physical_workers={h['host']:len(h['cpus']) for h in pool['hosts']},
            report_sha256=sha(reports[0]),compute_and_publication_verified=not failed))
    write(out/'publication_verification.json',publications)
    sys.path.insert(0,str(inp/'legacy'))
    from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    from evaluate import top_flags,warning_metrics,event_leads,lead_summary
    frames={f:pd.read_parquet(inp/f'{f}_eval/metadata_PRIVATE.parquet').reset_index(drop=True) for f in plan['folds']}
    paths={};summaries={};preds={};diag=[];parity=[];budget=[]
    for stage in ['training','recovery','assembled']:
        for p in (run/'results'/stage).glob('*/task-*/output/summary.json'):
            s=json.loads(p.read_text());assert not s['pilot'] and s['status']=='COMPLETED_PROVISIONAL'
            key=(s['fold'],s['seed'],s['variant']);assert key not in paths
            paths[key]=p.parent;summaries[key]=s
            for fn,h in json.loads((p.parent/'output_hashes.json').read_text()).items():assert sha(p.parent/fn)==h
            with np.load(p.parent/'predictions_PRIVATE.npz') as a:
                assert np.array_equal(a['sample_ids'],frames[key[0]].sample_id);preds[key]=a['cif12'][:,0]
            diag.append({k:v for k,v in s.items() if k not in ['training','validation']})
            log=json.loads((p.parent/'budget_log_PRIVATE.json').read_text())
            if log:
                rejected=[r for r in log if r['best_rejected'] is not None]
                changing=[r for r in rejected if r['best_rejected']['score']>(r['selected_score'] or 0)+1e-14]
                budget.append(dict(fold=key[0],seed=key[1],model=key[2],examined_nodes=len(log),rejected_candidates=sum(r['rejected_candidates'] for r in log),nodes_with_rejections=len(rejected),nodes_where_rejected_score_exceeds_selected=len(changing),first_rejection_tree=min((r['tree'] for r in rejected),default=None),max_best_rejected_score=max((r['best_rejected']['score'] for r in rejected),default=None),max_accepted_cells=max(r.get('cells_after',r['cells_before']) for r in log)))
    assert set(paths)=={(f,s,v) for f in plan['folds'] for s in plan['seeds'] for v in plan['variants']}
    mechanisms=[]
    for fold in plan['folds']:
        for seed in plan['seeds']:
            for a,b in [('local_budget','local_q2'),('shared_budget','shared_q2')]:
                aa=CompetingRiskForest.load(paths[fold,seed,a]/'forest_PRIVATE.json')
                bb=CompetingRiskForest.load(paths[fold,seed,b]/'forest_PRIVATE.json')
                mechanisms.append(dict(fold=fold,seed=seed,budget_model=a,reference=b,
                    identical_trees=aa.trees_==bb.trees_,
                    max_target_prediction_difference=float(abs(preds[fold,seed,a]-preds[fold,seed,b]).max()),
                    rejected_candidates=summaries[fold,seed,a]['budget_rejected_candidates']))
    pd.DataFrame(mechanisms).to_csv(out/'mechanism_audit.csv',index=False)
    # The old-seed controls must be identical despite the new dispatcher and opt-in code.
    for f in plan['folds']:
        for v,old_v in [('unrestricted','unrestricted'),('local_q2','q2')]:
            old=[]
            for p in (Path(plan['original_run'])/'results/formal').glob('*/task-*/output/summary.json'):
                s=json.loads(p.read_text())
                if s['fold']==f and s['variant']==old_v:old.append(p.parent)
            assert len(old)==1
            a=CompetingRiskForest.load(old[0]/'forest_PRIVATE.json');b=CompetingRiskForest.load(paths[f,42,v]/'forest_PRIVATE.json')
            assert a.trees_==b.trees_ and a.audit_==b.audit_
            with np.load(old[0]/'predictions_PRIVATE.npz') as z:np.testing.assert_array_equal(z['cif12'][:,0],preds[f,42,v])
            parity.append(dict(fold=f,variant=v,status='BITWISE_TREE_AND_PREDICTION_EQUAL'))
        with np.load(inp/f'{f}_M2_PRIVATE.npz') as a:
            assert np.array_equal(a['sample_ids'],frames[f].sample_id);preds[f,-1,'M2']=a['probabilities'][:,-1,1]
    frames['pooled']=pd.concat([frames[f] for f in plan['folds']],ignore_index=True)
    identities=[(s,v) for s in plan['seeds'] for v in plan['variants']]+[(-1,'M2')]
    for seed,v in identities:preds['pooled',seed,v]=np.concatenate([preds[f,seed,v] for f in plan['folds']])
    metrics=[];warnings=[];cal=[]
    for scope,d in frames.items():
        w,wa=source_consistent_endpoint_weights(d,endpoint_weights);y=(d.event.to_numpy()==1).astype(int)
        for weighting in ['record_equal','company_equal']:
            base=np.ones(len(d)) if weighting=='record_equal' else 1/d.groupby('group').group.transform('size').to_numpy();ww=w*base;keep=ww>0
            for seed,v in identities:
                p=preds[scope,seed,v]
                metrics.append(dict(scope=scope,seed=seed,model=v,weighting=weighting,records=len(d),auc=float(roc_auc_score(y[keep],p[keep],sample_weight=ww[keep])),ap=float(average_precision_score(y[keep],p[keep],sample_weight=ww[keep])),ipcw_brier=float(np.sum(ww*(y-p)**2)/base.sum()),prediction_mean=float(np.average(p,weights=base)),observed_ipcw_incidence=float(np.sum(ww*y)/base.sum()),**wa))
                for b,ii in enumerate(np.array_split(np.lexsort((d.sample_id.to_numpy(),p)),10)):
                    cal.append(dict(scope=scope,seed=seed,model=v,weighting=weighting,bin=b+1,n=len(ii),prediction=float(np.average(p[ii],weights=base[ii])),observed=float(np.sum(ww[ii]*y[ii])/base[ii].sum()),known=int(keep[ii].sum())))
        for seed,v in identities:
            for frac in [.2,.25]:
                flags=top_flags(d,preds[scope,seed,v],frac)
                scopes=[(scope,np.arange(len(d)))]
                if scope!='pooled':scopes += [(str(q),np.asarray(ii)) for q,ii in d.groupby('quarter').indices.items()]
                for sub,ii in scopes:
                    dd=d.iloc[ii].reset_index(drop=True);ff=flags[ii]
                    warnings.append(dict(scope=sub,seed=seed,model=v,top_fraction=frac,**warning_metrics(dd,ff),**lead_summary(event_leads(dd,ff))))
    m=pd.DataFrame(metrics);w=pd.DataFrame(warnings);d=pd.DataFrame(diag)
    prior=pd.read_csv(Path(plan['original_run'])/'reports/metrics.csv')
    metric_parity=[]
    for model,old,seed in [('M2','M2',-1),('unrestricted','unrestricted',42),('local_q2','q2',42)]:
        for scope in frames:
            for weighting in ['record_equal','company_equal']:
                a=m.query('model==@model and scope==@scope and weighting==@weighting and seed==@seed').iloc[0]
                b=prior.query('model==@old and scope==@scope and weighting==@weighting').iloc[0]
                fields=['auc','ap','ipcw_brier','prediction_mean','observed_ipcw_incidence']
                delta=max(abs(float(a[k])-float(b[k])) for k in fields);assert delta<1e-14
                metric_parity.append(dict(model=model,scope=scope,weighting=weighting,max_difference=delta))
    write(out/'legacy_metric_parity.json',metric_parity)
    m.to_csv(out/'metrics.csv',index=False);w.to_csv(out/'warning_metrics.csv',index=False);d.to_csv(out/'training_diagnostics.csv',index=False);pd.DataFrame(cal).to_csv(out/'calibration.csv',index=False);pd.DataFrame(budget).to_csv(out/'budget_diagnostics.csv',index=False)
    write(out/'legacy_control_parity.json',parity)
    summary=m.groupby(['scope','model','weighting'],as_index=False).agg(ap_mean=('ap','mean'),ap_min=('ap','min'),ap_max=('ap','max'),auc_mean=('auc','mean'),brier_mean=('ipcw_brier','mean'),prediction_mean=('prediction_mean','mean'))
    summary.to_csv(out/'metrics_summary.csv',index=False)
    warn=w[w.scope.isin(['V19','V20','pooled'])].groupby(['scope','model','top_fraction'],as_index=False).agg(events_mean=('detected_events','mean'),events_min=('detected_events','min'),events_max=('detected_events','max'),independent_events=('independent_events','first'),lead3_mean=('announcement_detected_3m','mean'),lead6_mean=('announcement_detected_6m','mean'))
    warn.to_csv(out/'warning_summary.csv',index=False)
    contrasts=[]
    for label,a,b in plan['contrasts']:
        for weighting in ['record_equal','company_equal']:
            for scope in ['V19','V20','pooled']:
                aa=m.query('model==@a and scope==@scope and weighting==@weighting').set_index('seed');bb=m.query('model==@b and scope==@scope and weighting==@weighting').set_index('seed')
                for seed in plan['seeds']:
                    contrasts.append(dict(contrast=label,scope=scope,weighting=weighting,seed=seed,ap_difference=aa.loc[seed,'ap']-bb.loc[seed,'ap'],brier_difference=aa.loc[seed,'ipcw_brier']-bb.loc[seed,'ipcw_brier'],auc_difference=aa.loc[seed,'auc']-bb.loc[seed,'auc']))
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    gates=[]
    for v in plan['variants'][1:]:
        dd=d[d.variant==v];maxcells=int(dd.spatial_table_cells.max())
        for scope in ['V19','V20','pooled']:
            a=summary.query("model==@v and scope==@scope and weighting=='record_equal'").iloc[0];b=summary.query("model=='unrestricted' and scope==@scope and weighting=='record_equal'").iloc[0]
            aa=warn.query('model==@v and scope==@scope and top_fraction==.2').iloc[0];bb=warn.query("model=='unrestricted' and scope==@scope and top_fraction==.2").iloc[0]
            ar=a.ap_mean/b.ap_mean;br=a.brier_mean/b.brier_mean;lost=bb.events_mean-aa.events_mean
            gates.append(dict(model=v,scope=scope,ap_retention=ar,brier_ratio=br,mean_events_lost=lost,max_spatial_cells=maxcells,prediction_gate=bool(ar>=.95 and br<=1.05 and lost<=3),complexity_gate=bool(maxcells<=1000)))
    pd.DataFrame(gates).to_csv(out/'acceptance_gates.csv',index=False)
    # Conditional algorithmic stability across seeds, using fixed TRAINING references.
    stability=[]
    for fold in plan['folds']:
        with np.load(inp/f'{fold}_train_PRIVATE.npz') as z:X=z['X']
        xx=X[np.linspace(0,len(X)-1,512,dtype=int)]
        for v in plan['variants'][1:]:
            ex={s:CIFExpansion.load(paths[fold,s,v]/'functions_PRIVATE.npz') for s in plan['seeds']}
            for sa,sb in combinations(plan['seeds'],2):
                a,b=ex[sa],ex[sb];am,bm=set(a.mains),set(b.mains);ap,bp=set(a.pairs),set(b.pairs)
                row=dict(fold=fold,model=v,seed_a=sa,seed_b=sb,main_jaccard=len(am&bm)/len(am|bm) if am|bm else None,pair_jaccard=len(ap&bp)/len(ap|bp) if ap|bp else None)
                for order,keys in [(1,[(j,) for j in sorted(am&bm)]),(2,sorted(ap&bp))]:
                    corr=[];sign=[]
                    for key in keys:
                        u=a.component(xx,key,[365])[:,0,0];z=b.component(xx,key,[365])[:,0,0]
                        if u.std()>1e-12 and z.std()>1e-12:corr.append(float(np.corrcoef(u,z)[0,1]))
                        mask=(abs(u)>1e-12)|(abs(z)>1e-12)
                        if mask.any():sign.append(float(np.mean(np.sign(u[mask])==np.sign(z[mask]))))
                    row[f'order{order}_median_correlation']=float(np.median(corr)) if corr else None
                    row[f'order{order}_median_sign_agreement']=float(np.median(sign)) if sign else None
                stability.append(row)
    pd.DataFrame(stability).to_csv(out/'seed_stability.csv',index=False)
    failures=[dict(stage='initial input preparation',reason='generated bytecode cache not in frozen manifest; canonical data untouched',fits_started=0)]
    if recovery_plan:failures.append(dict(stage='post-fit diagnostic serialization',reason=recovery_plan['reason'],original_attempts=len(recovery_plan['tasks']),recovered_same_configs=True,tasks=recovery_plan['tasks']))
    write(out/'execution_receipt.json',dict(status='COMPLETED_PROVISIONAL',formal_fits=len(paths),pilot_fits=5,training_tasks=156,recovery_tasks=len(recovery_plan['tasks']) if recovery_plan else 0,assembly_tasks=18,training_code=plan['code_commit'],recovery_plan=recovery_plan,plan_sha256=sha(inp/'plan.json'),source_hashes_verified=len(plan['source_hashes']),seeds=plan['seeds'],variants=plan['variants'],input_columns=75,training_records_fold_sum=112513,unique_training_records=62049,independent_training_events=182,validation_records=25689,independent_validation_events=135,model_hashes={f'{f}/{s}/{v}':sha(p/'forest_PRIVATE.json') for (f,s,v),p in paths.items()},publication=publications,test_outcomes_used=False,identifiability_penalty='NOT_TESTED',failures=failures,run=str(run)))
    print(summary.query("weighting=='record_equal'")[['scope','model','ap_mean','auc_mean','brier_mean']].to_string(index=False))
    print(pd.DataFrame(gates).to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(a.run)
