import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_benchmark_calibration_v1.bootstrap import FIELDS,assessment
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_prognosis_v1.common import horizon_predictions
from .common import *

def jobs(root):
    result={};receipts=[]
    for f in sorted(root.glob('published/*/task-*/output/result.json')):
        p=f.parent;r=read(f);key=r['year'],r['months'];assert key not in result
        for row in read(p.parent/'complete.json')['manifest']:
            if len(row)==2:assert (p/row[0]).is_dir();continue
            fn,size,hh=row;assert (p/fn).stat().st_size==size and sha(p/fn)==hh
        result[key]=(p,r)
        receipts.append(dict(year=key[0],months=key[1],prediction_hash=sha(p/'predictions_PRIVATE.npz'),manifest_hash=sha(p.parent/'complete.json'),job_id=r['job_id'],hostname=r['hostname']))
    assert set(result)=={(y,h) for y in YEARS for h in (12,24)}
    return result,receipts

def main():
    new,receipts=jobs(run());old,old_receipts=jobs(previous_run());base=source()/'input'
    pd.DataFrame(receipts).to_csv(HERE/'PUBLICATION_CHECKS.csv',index=False)
    pd.DataFrame(old_receipts).to_csv(HERE/'PRIOR_OUTPUT_HASHES.csv',index=False)
    expected=read(ROOT/'experiments/post_st_c061_xgboost_v1/SOURCE_HASHES.json')
    draw_path=Path(read(ROOT/'experiments/post_st_cause_support_v1/run_location.json')['private_run'])/'private/bootstrap_indices_PRIVATE.npz'
    draws=arrays(draw_path);rng=np.random.default_rng(2026110201);books={};labels={};counts={};hashes={};spreads=[];coherence=[];cost=[]
    for y in YEARS:
        labels[y]=arrays(base/f'evaluation_input/{y}_test_PRIVATE.npz');lab=labels[y];ids=lab['sample_ids'];n=len(ids)
        tr=arrays(run()/f'input/{y}/train.npz');ev=arrays(run()/f'input/{y}/eval.npz');assert set(ev)=={'X','sample_ids'}
        np.testing.assert_array_equal(ids,ev['sample_ids']);assert not set(tr['groups'])&set(lab['groups'])
        for role in ('train','eval'):assert sha(run()/f'input/{y}/{role}.npz')==expected[f'{y}/{role}']
        hashes[f'{y}/labels']=sha(base/f'evaluation_input/{y}_test_PRIVATE.npz')
        np.testing.assert_array_equal(draws[str(y)],rng.integers(n,size=(1000,n)))
        counts[y]=np.vstack([np.ones(n,int),*[np.bincount(a,minlength=n) for a in draws[str(y)]]])
        members=[]
        for seed in OLD_SEEDS:
            p=base/f'{y}/{seed}/shrunk'
            for fn in ('forest_PRIVATE.json','predictions_PRIVATE.npz','functions_PRIVATE.npz'):assert sha(p/fn)==expected[f'{y}/{seed}/{fn}']
            z=arrays(p/'predictions_PRIVATE.npz');np.testing.assert_array_equal(ids,z['sample_ids']);members.append(z['cif'])
        cif=np.mean(members,axis=0)
        for h in (12,24):
            pp,rr=new[y,h];aa=arrays(pp/'predictions_PRIVATE.npz');bb=arrays(old[y,h][0]/'predictions_PRIVATE.npz')
            for d in (aa,bb):np.testing.assert_array_equal(d['sample_ids'],ids)
            np.testing.assert_array_equal(aa['ensemble'],np.mean([aa[str(s)] for s in SEEDS],axis=0))
            np.testing.assert_array_equal(bb['ensemble'],np.mean([bb[str(s)] for s in OLD_SEEDS],axis=0))
            books[y,h]={'C061':horizon_predictions(cif,z['times'],lab[f'h{h}'])[:,0],
                'XGBoost_old3':bb['ensemble'],'XGBoost_new3':aa['ensemble'],
                **{f'XGBoost/{s}':aa[str(s)] for s in SEEDS},**{f'XGBoost/{s}':bb[str(s)] for s in OLD_SEEDS}}
            assert rr['tuning_fits']==0 and rr['final_fits']==3
            for row in rr['models']:
                assert all(row[k]==v for k,v in config(y,h).items())
                cost.append(dict(year=y,months=h,**row,task_seconds=rr['seconds'],task_peak_RSS_MiB=rr['peak_RSS_MiB'],node=rr['hostname']))
            for name,seeds,data in [('old3',OLD_SEEDS,bb),('new3',SEEDS,aa)]:
                sd=np.std([data[str(s)] for s in seeds],axis=0,ddof=1)
                spreads.append(dict(year=y,months=h,group=name,mean_company_score_sd=sd.mean(),max_company_score_sd=sd.max()))
        for name in books[y,12]:
            diff=books[y,12][name]-books[y,24][name]
            coherence.append(dict(year=y,model=name,h12_gt_h24=int(sum(diff>1e-10)),max_h12_minus_h24=float(max(diff))))
    old_metrics=pd.read_csv(ROOT/'experiments/post_st_c061_xgboost_v1/METRICS_ALL.csv');rows=[];deltas=[];checks=[]
    for h in (12,24):
        for scope in [*YEARS,'pooled']:
            years=([2023,2024] if h==24 else list(YEARS)) if scope=='pooled' else [scope]
            supported=(scope,h)!=(2025,24);n=sum(len(labels[y]['X']) for y in years)
            yy=np.concatenate([(labels[y]['event']==1)&(labels[y]['analysis_time']<=h) for y in years]).astype(int)
            ids=np.concatenate([labels[y]['sample_ids'] for y in years]);yr=np.concatenate([np.full(len(labels[y]['X']),y) for y in years])
            count=np.column_stack([counts[y] for y in years]);quota=int(sum(np.ceil(.2*len(labels[y]['X'])) for y in years));values={}
            if supported:
                for y in years:
                    _,w,known,ok,_=evaluation_context(labels[y],h,1);assert ok and known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            for name in books[years[0],h]:
                pred=np.concatenate([books[y,h][name] for y in years]);metrics=dict.fromkeys(FIELDS,np.nan)
                if supported:
                    vv=assessment(count,yy,pred,yr,ids);values[name]=vv;metrics=dict(zip(FIELDS,vv[0]))
                    auc=roc_auc_score(yy,pred);ap=average_precision_score(yy,pred)
                    np.testing.assert_allclose([metrics['auc'],metrics['ap']],[auc,ap],rtol=0,atol=1e-12)
                    top=np.concatenate([np.flatnonzero(yr==y)[np.lexsort((ids[yr==y],-pred[yr==y]))[:int(np.ceil(.2*sum(yr==y)))]] for y in years])
                    assert yy[top].sum()==metrics['top20_captured']
                    historical_name='XGBoost' if name=='XGBoost_old3' else name
                    oldm=old_metrics[(old_metrics.scope.astype(str)==str(scope))&(old_metrics.months==h)&(old_metrics.model==historical_name)]
                    replay=not oldm.empty
                    if replay:np.testing.assert_allclose([metrics[f] for f in FIELDS],oldm.iloc[0][FIELDS].astype(float).to_numpy(),atol=1e-12,rtol=0)
                    checks.append(dict(scope=scope,months=h,model=name,auc_error=abs(metrics['auc']-auc),ap_error=abs(metrics['ap']-ap),top20_verified=True,prior_metrics_replayed=replay))
                rows.append(dict(scope=scope,months=h,model=name,companies=n,supported=supported,target_events=int(sum(yy)) if supported else np.nan,checked=quota if supported else np.nan,**metrics))
            if supported:
                for baseline in ('XGBoost_old3','C061'):
                    diff=values['XGBoost_new3']-values[baseline]
                    for j,metric in enumerate(FIELDS):
                        vals=diff[1:,j];ok=np.isfinite(vals);q=np.quantile(vals[ok],[.025,.975]) if ok.any() else [np.nan,np.nan]
                        deltas.append(dict(scope=scope,months=h,contrast='XGBoost_new3 minus '+baseline,metric=metric,difference=diff[0,j],lower95=q[0],upper95=q[1],undefined=int(sum(~ok))))
    df=pd.DataFrame(rows);df.to_csv(HERE/'METRICS_ALL.csv',index=False)
    df[df.model.isin([f'XGBoost/{s}' for s in SEEDS])].to_csv(HERE/'NEW_SEED_RESULTS.csv',index=False)
    df[df.model.isin(['C061','XGBoost_old3','XGBoost_new3'])].to_csv(HERE/'ENSEMBLE_COMPARISON.csv',index=False)
    pd.DataFrame(deltas).to_csv(HERE/'PAIRED_DIFFERENCES.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'METRIC_VALIDATION.csv',index=False)
    pd.DataFrame(cost).to_csv(HERE/'TRAINING_LOG.csv',index=False)
    pd.DataFrame(coherence).to_csv(HERE/'HORIZON_COHERENCE.csv',index=False)
    pd.DataFrame(spreads).to_csv(HERE/'PREDICTION_SPREAD.csv',index=False)
    write(HERE/'EVALUATION_HASHES.json',dict(labels=hashes,bootstrap=sha(draw_path),source_models_unchanged=True,prior_forecasts_replayed=True))

if __name__=='__main__':main()
