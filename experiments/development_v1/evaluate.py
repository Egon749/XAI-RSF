"""Fixed prediction evaluation, common endpoint guard and complete output receipt."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from crforest import CompetingRiskForest,CIFExpansion


def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(run):
    run=Path(run);out=run/'reports';inp=run/'input'
    sys.path.insert(0,str(inp/'legacy'))
    from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    from evaluate import top_flags,warning_metrics,event_leads,lead_summary
    models=['M2','unrestricted','depth2','q2','q1'];frames={};preds={};paths={};diag=[]
    for f in ['V19','V20']:
        frames[f]=pd.read_parquet(inp/f'{f}_eval/metadata_PRIVATE.parquet').reset_index(drop=True)
        with np.load(inp/f'{f}_M2_PRIVATE.npz') as a:
            assert np.array_equal(a['sample_ids'],frames[f].sample_id);preds[f,'M2']=a['probabilities'][:,-1,1]
        for name in models[1:]:
            found=[]
            for p in (run/'results/formal').glob('*/task-*/output/summary.json'):
                s=json.loads(p.read_text())
                if s['fold']==f and s['variant']==name:found.append((p,s))
            assert len(found)==1,(f,name,len(found));p,s=found[0];assert not s['pilot'] and s['status']=='COMPLETED_PROVISIONAL'
            paths[f,name]=p.parent;diag.append({k:v for k,v in s.items() if k not in ['training','validation']})
            with np.load(p.parent/'predictions_PRIVATE.npz') as a:
                assert np.array_equal(a['sample_ids'],frames[f].sample_id);preds[f,name]=a['cif12'][:,0]
            # Every fitted artifact is hashed independently after pool publication.
            for filename,digest in json.loads((p.parent/'output_hashes.json').read_text()).items():assert sha(p.parent/filename)==digest
    frames['pooled']=pd.concat([frames['V19'],frames['V20']],ignore_index=True)
    for name in models:preds['pooled',name]=np.r_[preds['V19',name],preds['V20',name]]
    metrics=[];cal=[];warnings=[]
    for scope,d in frames.items():
        w,wa=source_consistent_endpoint_weights(d,endpoint_weights);y=(d.event.to_numpy()==1).astype(int)
        for weighting in ['record_equal','company_equal']:
            base=np.ones(len(d)) if weighting=='record_equal' else 1/d.groupby('group').group.transform('size').to_numpy()
            ww=w*base;keep=ww>0
            for name in models:
                p=preds[scope,name]
                metrics.append(dict(scope=scope,model=name,weighting=weighting,records=len(d),groups=int(d.group.nunique()),target_records=int(y.sum()),target_events=int(d.loc[y==1,'event_id'].nunique()),auc=float(roc_auc_score(y[keep],p[keep],sample_weight=ww[keep])),ap=float(average_precision_score(y[keep],p[keep],sample_weight=ww[keep])),ipcw_brier=float(np.sum(ww*(y-p)**2)/base.sum()),prediction_mean=float(np.average(p,weights=base)),observed_ipcw_incidence=float(np.sum(ww*y)/base.sum()),**wa))
                order=np.lexsort((d.sample_id.to_numpy(),p))
                for b,ii in enumerate(np.array_split(order,10)):
                    cal.append(dict(scope=scope,model=name,weighting=weighting,bin=b+1,records=len(ii),predicted=float(np.average(p[ii],weights=base[ii])),observed_ipcw=float(np.sum(ww[ii]*y[ii])/base[ii].sum()),known=int(keep[ii].sum())))
        for name in models:
            for fraction in [.2,.25]:
                flags=top_flags(d,preds[scope,name],fraction)
                scopes=[(scope,np.arange(len(d)))]
                if scope!='pooled':scopes += [(str(q),np.asarray(ii)) for q,ii in d.groupby('quarter').indices.items()]
                for sub,ii in scopes:
                    dd=d.iloc[ii].reset_index(drop=True);ff=flags[ii]
                    warnings.append(dict(scope=sub,model=name,top_fraction=fraction,**warning_metrics(dd,ff),**lead_summary(event_leads(dd,ff))))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(cal).to_csv(out/'calibration.csv',index=False);pd.DataFrame(warnings).to_csv(out/'warning_metrics.csv',index=False)
    pd.DataFrame(diag).to_csv(out/'training_diagnostics.csv',index=False)
    # Common pre-V19 training reference for descriptive cross-fold explanations.
    # Evaluation rows are never used to fit the reference or the forests.
    with np.load(inp/'V19_train_PRIVATE.npz') as a:reference=a['X']
    selected=np.linspace(0,len(reference)-1,512,dtype=int);xx=reference[selected]
    stability=[];directions=[]
    for name in models[2:]:
        ex=[]
        for f in ['V19','V20']:ex.append(CIFExpansion.from_forest(CompetingRiskForest.load(paths[f,name]/'forest_PRIVATE.json'),reference))
        ma,mb=map(lambda z:set(z.mains),ex);pa,pb=map(lambda z:set(z.pairs),ex)
        stability.append(dict(model=name,main_jaccard=len(ma&mb)/len(ma|mb) if ma|mb else None,pair_jaccard=len(pa&pb)/len(pa|pb) if pa|pb else None,shared_mains=len(ma&mb),shared_pairs=len(pa&pb),note='descriptive across different calendar folds, not independent training uncertainty'))
        for key in [(j,) for j in sorted(ma&mb)]+sorted(pa&pb):
            aa=ex[0].component(xx,key,[365])[:,0,0];bb=ex[1].component(xx,key,[365])[:,0,0]
            active=(abs(aa)>1e-12)|(abs(bb)>1e-12)
            # Feature names/IDs and full functions remain private.
            directions.append(dict(model=name,order=len(key),common_reference_component_correlation=float(np.corrcoef(aa,bb)[0,1]) if aa.std()>1e-12 and bb.std()>1e-12 else None,nonzero_sign_agreement=float(np.mean(np.sign(aa[active])==np.sign(bb[active]))) if active.any() else None))
    pd.DataFrame(stability).to_csv(out/'function_stability.csv',index=False)
    pd.DataFrame(directions).to_csv(out/'component_agreement.csv',index=False)
    # Published simulation tasks and their fixed metrics.
    sm=[];ss=[]
    for p in sorted((run/'results/simulation').glob('*/task-*/output/summary.json')):
        s=json.loads(p.read_text());assert s['status']=='COMPLETED';ss.append(s);sm.append(pd.read_csv(p.parent/'metrics.csv'))
    assert len(ss)==18 and len({(s['scenario'],s['seed']) for s in ss})==18
    pd.concat(sm,ignore_index=True).to_csv(out/'simulation_metrics.csv',index=False)
    sd=[r for s in ss for r in s['models']]
    pd.DataFrame([{k:v for k,v in d.items() if k!='probability_audit'} for d in sd]).to_csv(out/'simulation_diagnostics.csv',index=False)
    train=pd.concat([pd.read_parquet(inp/f'{f}_train/metadata_PRIVATE.parquet') for f in ['V19','V20']],ignore_index=True)
    write(out/'execution_receipt.json',dict(status='COMPLETED_PROVISIONAL',run=str(run),fit_code_commit=json.loads((inp/'real_config.json').read_text())['code_commit'],formal_real_fits=8,simulation_fits=90,pilot_fits=2,training_records_fold_sum=len(train),unique_training_records=int(train.sample_id.nunique()),independent_training_target_events=int(train.loc[train.event==1,'event_id'].nunique()),validation_records=len(frames['pooled']),independent_validation_target_events=int(frames['pooled'].loc[frames['pooled'].event==1,'event_id'].nunique()),penalty_evaluation='NOT_TESTED',formal_tasks_verified=8,simulation_tasks_verified=18,real_prediction_scale='target CIF at each calendar-year horizon',source_config_sha256=sha(inp/'real_config.json'),model_hashes={f'{f}/{n}':sha(p/'forest_PRIVATE.json') for (f,n),p in paths.items()},data_status='PROVISIONAL',test_2022_2024_used=False,
        failures=[dict(stage='preparation',type='overbroad clock check',resolution='retain already-masked missing source provenance; no observed future content'),dict(stage='formal_attempt_0',job=2508,started_fits=8,completed_fits=0,type='Slurm node-state failure and automatic requeue',resolution='same immutable code/config/seeds; original interruption logs preserved')]))
    print(pd.DataFrame(metrics).query("weighting=='record_equal'")[['scope','model','auc','ap','ipcw_brier']].to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(a.run)
