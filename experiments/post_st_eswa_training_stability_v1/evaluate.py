import pandas as pd
import numpy as np
from experiments.post_st_eswa_baselines_v1.common import Horizon,load,sha,arrays,dump
from experiments.post_st_benchmark_calibration_v1.common import old,endpoints
from experiments.post_st_benchmark_calibration_v1.models import stored
from .run import D,ROOT,run
from .metrics import metrics,record,PERF,SCORES,position
from .resampling import draw_rows
YEARS=[2023,2024,2025];ALGOS=['PROBoost','Stacking']
def collect():
    r=run();entries={};audit=[];oldexp=ROOT/'experiments/post_st_eswa_baselines_v1'
    published=sorted(r.glob('published/*/task-*/output/done.json'))
    for f in published:
        p=f.parent;j=load(f);k=(j['year'],j['months'],j['rep'],j['model']);assert k not in entries;entries[k]=(p,j)
        assert (p.parent/'complete.json').exists()
        for row in load(p.parent/'complete.json')['manifest']:
            if len(row)==3:assert (p/row[0]).stat().st_size==row[1] and sha(p/row[0])==row[2]
        audit.append(j|dict(receipt_sha256=sha(p.parent/'complete.json')))
    expected={(y,h,b,m) for y in YEARS for h in (12,24) for b in range(1,51) for m in ALGOS}
    assert set(entries)==expected,'incomplete publication; do not compute partial final distributions'
    rows=[];obs={};test={};oob={};skipped={};paired={}
    for y in YEARS:
        tr=arrays(r/f'input/{y}/train.npz');ev=arrays(r/f'input/{y}/eval.npz')
        lab=arrays(old('post_st_seed_explanation_stability_v1')/f'input/evaluation_input/{y}_test_PRIVATE.npz')
        assert len(set(tr['groups'])&set(lab['groups']))==0
        np.testing.assert_array_equal(lab['sample_ids'],ev['sample_ids'])
        for h in (12,24):
            hd=Horizon(tr,h,'IPCW');ext=Horizon(lab,h,'complete');w=(ext.y>=0).astype(float);ok=(y,h)!=(2025,24)
            if ok:assert w.min()==1
            obs[y,h]=(ext.y,w,ev['sample_ids'])
            for b in range(1,51):
                ix=draw_rows(y,b,len(hd.x));om=np.setdiff1d(np.arange(len(ix)),ix);kk,yy,ww=hd.weighted(ix,om);iw=np.zeros(len(om));iw[np.searchsorted(om,kk)]=ww
                for model in ALGOS:
                    key=(y,h,b,model);p,j=entries[key];draw=arrays(p/'draw.npz')
                    np.testing.assert_array_equal(draw['rows'],ix);np.testing.assert_array_equal(draw['oob'],om)
                    paired.setdefault((y,b),set()).add(j['draw_sha256']);base=dict(scope=str(y),months=h,rep=b,model=model,status=j['status'],reason=j.get('reason',''),train_distinct_positive=j['train_distinct_positive'],selected_cv_mean_ap=j.get('selected_cv_mean_ap'),deployment_GA_F1=j.get('deployment_GA_F1'),avg_deployment_base_F1=j.get('avg_deployment_base_F1'))
                    if j['status']=='ok':
                        z=arrays(p/'scores.npz');np.testing.assert_array_equal(z['oob_ids'],tr['sample_ids'][om]);np.testing.assert_array_equal(z['test_ids'],ev['sample_ids']);assert j['reload_error']==0.
                        test[key]=record(y,ev['sample_ids'],ext.y,w,z['test'])
                        oob[key]=record(y,tr['sample_ids'][om],hd.y[om],iw,z['oob'])
                    else:skipped[key]=j['status']
                    rows.append(summarize_one(base,[key],test,oob,ok))
    assert all(len(v)==1 for v in paired.values()) and len(paired)==150
    for h in (12,24):
        years=YEARS if h==12 else YEARS[:2]
        for b in range(1,51):
            for m in ALGOS:
                keys=[(y,h,b,m) for y in years];bad=[str(k[0])+':'+skipped[k] for k in keys if k in skipped]
                base=dict(scope='pooled',months=h,rep=b,model=m,status='incomplete' if bad else 'ok',reason='|'.join(bad))
                rows.append(summarize_one(base,keys,test,oob,True))
    df=pd.DataFrame(rows);df.to_csv(D/'ALL_REPLICATES.csv',index=False);pd.DataFrame(audit).to_csv(D/'TRAINING_STATUS.csv',index=False)
    original=pd.concat([pd.read_csv(oldexp/f,dtype={'scope':str}) for f in ['METRICS_BY_YEAR.csv','POOLED_METRICS.csv']])
    dist=pd.read_csv(oldexp/'SCORE_DISTRIBUTIONS.csv',dtype={'scope':str})
    oscores=original.merge(dist[['model','months','scope','unique_scores','max']],on=['model','months','scope'],validate='one_to_one')
    # Verify exact original score minima, maxima and spans against matched private deployments.
    from pathlib import Path
    import os
    parent=Path(os.environ['ESWA_RUN']);orig_parts={};orig_cv={}
    for f in sorted(parent.glob('*published/*/task-*/output/result.json')):
        j=load(f);y,h,m=j['year'],j['months'],j['arm'];k=(y,h,m)
        if j['mode']!='IPCW':continue
        if k in orig_parts:assert f.parents[3].name=='repair_published'
        logs=pd.read_csv(f.parent/'training.csv')
        if m=='PROBoost':
            pick=logs.loc[logs.stage=='PROBoost_final','candidate'].iloc[0];orig_cv[str(y),h,m,'selected_cv_mean_ap']=logs[(logs.stage=='PROBoost_tune')&(logs.candidate==pick)].score.mean()
        else:
            import pickle
            with (f.parent/'stack.pkl').open('rb') as stream:bundle,_=pickle.load(stream)
            ga=logs[logs.stage.isin(['GA_search','GA_support_fallback'])].iloc[-1]
            orig_cv[str(y),h,m,'deployment_GA_F1']=ga.score if ga.stage=='GA_search' else np.nan
            orig_cv[str(y),h,m,'avg_deployment_base_F1']=np.mean(bundle['fold_F1']) if bundle['fold_F1'] else np.nan
        z=arrays(f.parent/'predictions.npz');truth,w,ids=obs[y,h];np.testing.assert_array_equal(z['ids'],ids);orig_parts[k]=record(y,ids,truth,w,z['score'])
    orig={}
    for h in (12,24):
        for sc in [*map(str,YEARS),'pooled']:
            ys=(YEARS if h==12 else YEARS[:2]) if sc=='pooled' else [int(sc)]
            for m in ALGOS:
                vals=metrics([orig_parts[y,h,m] for y in ys],(sc,h)!=('2025',24));orig[sc,h,m]=vals
                ref=oscores[(oscores.scope==sc)&(oscores.months==h)&(oscores.model==m+'/IPCW')].iloc[0]
                for field in PERF+['unique_scores']:
                    np.testing.assert_allclose(vals[field],ref[field],rtol=0,atol=1e-12,equal_nan=True)
    stats=[]
    targets=['test_'+x for x in PERF+SCORES]+['oob_'+x for x in PERF+SCORES]+['gap_'+x for x in PERF if x!='top20_captured']+['selected_cv_mean_ap','deployment_GA_F1','avg_deployment_base_F1']
    for (sc,h,m),g in df.groupby(['scope','months','model']):
        assert len(g)==50
        for v in targets:
            x=g[v].dropna().to_numpy(float);oldval=orig[sc,h,m].get(v[5:],np.nan) if v.startswith('test_') else orig_cv.get((sc,h,m,v),np.nan)
            summary=np.quantile(x,[0,.05,.5,.95,1.]) if len(x) else [np.nan]*5
            stats.append(dict(scope=sc,months=h,model=m,metric=v,scheduled=50,finite=len(x),undefined=50-len(x),original=oldval,original_midpercentile=position(x,oldval),**dict(zip(['minimum','p05','median','p95','maximum'],summary))))
    summary=pd.DataFrame(stats);summary.to_csv(D/'DISTRIBUTIONS.csv',index=False)
    compare_xai(obs,summary);dump(D/'ALIGNMENT_AUDIT.json',dict(full_job_count=len(entries),unique_year_draws=len(paired),all_model_horizon_draws_matched=True,original_metrics_reproduced=True,heldout_score_alignment=True,model_pointer_sha256=sha(ROOT/'CURRENT_POST_ST_MODEL.json')))
def summarize_one(base,keys,test,oob,ok):
    fields=PERF+SCORES
    available=all(k in test for k in keys)
    if available:
        ext=metrics([test[k] for k in keys],ok);internal=metrics([oob[k] for k in keys])
    else:ext=dict.fromkeys(fields,np.nan);internal=ext
    row=base|{'test_'+f:v for f,v in ext.items()}|{'oob_'+f:v for f,v in internal.items()}
    for f in PERF:
        if f!='top20_captured':row['gap_'+f]=internal[f]-ext[f]
    return row
def compare_xai(obs,summ):
    pred={};rows=[];records=[];archived=ROOT/'experiments/post_st_benchmark_calibration_v1'
    saved=pd.read_csv(archived/'METRICS_ALL.csv',dtype={'scope':str}).set_index(['scope','months','cause','model'])
    for y in YEARS:
        book,_=stored(y)
        for h in (12,24):
            lab=arrays(old('post_st_seed_explanation_stability_v1')/f'input/evaluation_input/{y}_test_PRIVATE.npz');truth,w,ids=obs[y,h]
            for name,cif in book.items():
                if name.startswith('triple/') or name in ('XAI/C039','XAI/anchor'):
                    pred[y,h,name]=record(y,ids,truth,w,endpoints(cif,lab[f'h{h}'])[:,0])
    names=sorted({n for y,h,n in pred})
    for h in (12,24):
        for sc in [*map(str,YEARS),'pooled']:
            ys=(YEARS if h==12 else YEARS[:2]) if sc=='pooled' else [int(sc)]
            for name in names:
                val=metrics([pred[y,h,name] for y in ys],(sc,h)!=('2025',24));records.append(dict(scope=sc,months=h,member=name,**val))
                row=saved.loc[(sc,h,1,name)]
                np.testing.assert_allclose([row['auc'],row['ap'],row['top20_captured']],[val[f] for f in ['auc','ap','top20_captured']],atol=1e-12,rtol=0,equal_nan=True)
    xai=pd.DataFrame(records);xai.to_csv(D/'XAI_FIXED_COMBINATIONS.csv',index=False)
    for (sc,h),g in xai.groupby(['scope','months']):
        v=g.set_index('member');triples=g[g.member.str.startswith('triple/')];assert len(triples)==120
        for f in PERF+SCORES:
            x=triples[f].dropna().to_numpy();q=np.quantile(x,[0,.05,.5,.95,1]) if len(x) else [np.nan]*5;origin=v.at['XAI/C039',f]
            rows.append(dict(scope=sc,months=h,model='XAI-RSF/120',metric='test_'+f,scheduled=120,finite=len(x),undefined=120-len(x),original=origin,original_midpercentile=position(x,origin),fixed_anchor=v.at['XAI/anchor',f],perturbation='overlapping seed triples; same training companies',**dict(zip(['minimum','p05','median','p95','maximum'],q))))
    c=summ[summ.metric.str.startswith('test_')].copy();c['perturbation']='training-company bootstrap; full search/refit';pd.concat([c,pd.DataFrame(rows)],ignore_index=True).to_csv(D/'SIDE_BY_SIDE.csv',index=False)
if __name__=='__main__':collect()
