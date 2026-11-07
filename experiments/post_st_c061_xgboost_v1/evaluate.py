"""Frozen comparisons; called only after all historical choices and fits finish."""
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from experiments.post_st_benchmark_calibration_v1.bootstrap import FIELDS, assessment, ratio
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from .common import *

def collect():
    jobs={};pub=[];cost=[];tuning=[];logs=[]
    for f in sorted(run().glob('published/*/task-*/output/result.json')):
        p=f.parent;r=read(f);key=r['year'],r['months'];assert key not in jobs
        manifest=read(p.parent/'complete.json')['manifest']
        for row in manifest:
            if len(row)==2:assert (p/row[0]).is_dir();continue
            fn,size,digest=row;assert (p/fn).stat().st_size==size and sha(p/fn)==digest
        for name,digest in r['input_hashes'].items():assert sha(run()/f'input/{key[0]}'/name)==digest
        jobs[key]=(p,r)
        pub.append(dict(year=key[0],months=key[1],manifest_hash=sha(p.parent/'complete.json'),prediction_hash=sha(p/'predictions_PRIVATE.npz'),job_id=r['job_id'],node=r['hostname'],publication_verified=True))
        for record in r['final']:
            cost.append(dict(year=key[0],months=key[1],**record,**r['chosen'],selection_rule=r['selection_rule'],task_seconds=r['seconds'],task_peak_RSS_MiB=r['peak_RSS_MiB'],task_tuning_fits=r['tuning_fits'],node=r['hostname']))
        for filename,target in [('candidates.csv',tuning),('training_log.csv',logs)]:
            df=pd.read_csv(p/filename);df.insert(0,'year',key[0]);df.insert(1,'months',key[1]);target.append(df)
    assert set(jobs)=={(y,h) for y in YEARS for h in (12,24)}
    pd.DataFrame(pub).to_csv(HERE/'PUBLICATION_CHECKS.csv',index=False)
    pd.DataFrame(cost).to_csv(HERE/'COMPUTE_AND_PARAMETERS.csv',index=False)
    pd.concat(tuning).to_csv(HERE/'TUNING_ALL.csv',index=False);pd.concat(logs).to_csv(HERE/'TRAINING_LOG.csv',index=False)
    return jobs

def main():
    jobs=collect();base=source()/'input'
    draw_path=Path(read(ROOT/'experiments/post_st_cause_support_v1/run_location.json')['private_run'])/'private/bootstrap_indices_PRIVATE.npz'
    draws=arrays(draw_path);rng=np.random.default_rng(2026110201)
    labels={};counts={};books={};hashes={};monotonic=[];replay=[]
    expected=read(HERE/'SOURCE_HASHES.json')
    reference=pd.read_csv(ROOT/'experiments/post_st_seed_explanation_stability_v1/ALL_METRICS.csv')
    mapping=pd.read_csv(ROOT/'experiments/post_st_grouped_explanation_v1/FEATURE_GROUP_MAPPING.csv').sort_values('input_position').original_feature_id.to_numpy()
    for y in YEARS:
        lab=arrays(base/f'evaluation_input/{y}_test_PRIVATE.npz');labels[y]=lab
        tr=arrays(run()/f'input/{y}/train.npz');ev=arrays(run()/f'input/{y}/eval.npz')
        np.testing.assert_array_equal(lab['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(lab['X'][:,mapping],ev['X'])
        assert not set(tr['groups'])&set(lab['groups']) and len(set(lab['groups']))==len(lab['X'])
        n=len(lab['X']);np.testing.assert_array_equal(draws[str(y)],rng.integers(n,size=(1000,n)))
        counts[y]=np.vstack([np.ones(n,int),*[np.bincount(ix,minlength=n) for ix in draws[str(y)]]])
        hashes[f'{y}/labels']=sha(base/f'evaluation_input/{y}_test_PRIVATE.npz')
        cif=[]
        for s in SEEDS:
            p=base/f'{y}/{s}/shrunk'
            for fn in ('forest_PRIVATE.json','predictions_PRIVATE.npz','functions_PRIVATE.npz'):
                assert sha(p/fn)==expected[f'{y}/{s}/{fn}']
            z=arrays(p/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);cif.append(z['cif'])
        avg=np.mean(cif,axis=0)
        for h in (12,24):
            p,r=jobs[y,h];new=arrays(p/'predictions_PRIVATE.npz');np.testing.assert_array_equal(new.pop('sample_ids'),ev['sample_ids'])
            np.testing.assert_array_equal(new['ensemble'],np.mean([new[str(s)] for s in SEEDS],axis=0))
            book={'C061':horizon_predictions(avg,z['times'],lab[f'h{h}'])[:,0],
                  'XGBoost':new['ensemble'],**{f'XGBoost/{s}':new[str(s)] for s in SEEDS}}
            books[y,h]=book
        for model in books[y,12]:
            delta=books[y,12][model]-books[y,24][model]
            monotonic.append(dict(year=y,model=model,companies=n,h12_gt_h24=int(sum(delta>1e-10)),max_h12_minus_h24=float(max(delta)),scope='prediction legality diagnostic, including unsupported2025H24'))
    rows=[];paired=[];checks=[]
    for h in (12,24):
        for scope in [*YEARS,'pooled']:
            years=([2023,2024] if h==24 else list(YEARS)) if scope=='pooled' else [scope]
            supported=(scope,h)!=(2025,24)
            lab=[labels[y] for y in years]
            ids=np.concatenate([z['sample_ids'] for z in lab]);yr=np.concatenate([np.full(len(z['X']),y) for y,z in zip(years,lab)])
            yy=np.concatenate([(z['event']==1)&(z['analysis_time']<=h) for z in lab]).astype(int)
            count=np.column_stack([counts[y] for y in years]);quota=int(sum(np.ceil(.2*len(z['X'])) for z in lab));vectors={}
            if supported:
                for z in lab:
                    _,w,known,ok,_=evaluation_context(z,h,1);assert ok and known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
            for name in books[years[0],h]:
                p=np.concatenate([books[y,h][name] for y in years]);point=dict.fromkeys(FIELDS,np.nan)
                if supported:
                    values=assessment(count,yy,p,yr,ids);vectors[name]=values;point=dict(zip(FIELDS,values[0]))
                    np.testing.assert_allclose([point['auc'],point['ap']],[roc_auc_score(yy,p),average_precision_score(yy,p)],rtol=0,atol=1e-12)
                    top=np.concatenate([np.flatnonzero(yr==y)[np.lexsort((ids[yr==y],-p[yr==y]))[:int(np.ceil(.2*sum(yr==y)))]] for y in years])
                    assert yy[top].sum()==point['top20_captured']
                    checks.append(dict(scope=scope,months=h,model=name,sklearn_auc_error=abs(point['auc']-roc_auc_score(yy,p)),sklearn_ap_error=abs(point['ap']-average_precision_score(yy,p)),top20_independent_pass=True))
                    if name=='C061':
                        old=reference[(reference.scope.astype(str)==str(scope))&(reference.months==h)&(reference.model=='C061')&(reference.version=='shrunk')&(reference.cause==1)].iloc[0]
                        for oldf,newf in [('auc','auc'),('ap','ap'),('brier','ipcw_brier'),('captured','top20_captured'),('risk_mean','mean_predicted')]:
                            error=abs(old[oldf]-point[newf]);assert error<=1e-12
                            replay.append(dict(scope=scope,months=h,metric=newf,old=old[oldf],recomputed=point[newf],abs_error=error))
                rows.append(dict(scope=scope,months=h,model=name,companies=len(p),supported=supported,reason='' if supported else '2025H24 followup unsupported',target_events=int(sum(yy)) if supported else np.nan,checked=quota if supported else np.nan,**point))
            if supported:
                diff=vectors['XGBoost']-vectors['C061']
                for j,metric in enumerate(FIELDS):
                    v=diff[1:,j];good=np.isfinite(v)
                    q=np.quantile(v[good],[.025,.975]) if good.any() else [np.nan,np.nan]
                    paired.append(dict(scope=scope,months=h,metric=metric,contrast='XGBoost minus C061',difference=diff[0,j],lower95=q[0],upper95=q[1],defined=int(sum(good)),undefined=int(sum(~good)),conditional_fixed_predictions=True))
    all_metrics=pd.DataFrame(rows);all_metrics.to_csv(HERE/'METRICS_ALL.csv',index=False)
    all_metrics[all_metrics.model.isin(['C061','XGBoost'])].to_csv(HERE/'MODEL_COMPARISON.csv',index=False)
    pd.DataFrame(paired).to_csv(HERE/'PAIRED_DIFFERENCES.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'METRIC_VALIDATION.csv',index=False)
    pd.DataFrame(replay).to_csv(HERE/'C061_REPLAY.csv',index=False)
    pd.DataFrame(monotonic).to_csv(HERE/'HORIZON_COHERENCE.csv',index=False)
    write(HERE/'EVALUATION_HASHES.json',dict(labels=hashes,bootstrap_hash=sha(draw_path),matched_original_1000_draws=True,source_model_hashes_unchanged=True,training_eval_groups_disjoint=True,supported_outer_weights_all_one=True))
    write(run()/'aligned_predictions_PRIVATE.json',{'note':'Per-company arrays remain in original and new private output files; no export to Git.'})

if __name__=='__main__':main()
