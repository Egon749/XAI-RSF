import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_benchmark_calibration_v1.bootstrap import FIELDS,assessment,ratio
from experiments.post_st_benchmark_calibration_v1.models import stored
from experiments.post_st_benchmark_calibration_v1.common import endpoints
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from .common import *
MORE=['precision20','lift20']
def tasks(r):
    jobs={};files=[]
    for j in sorted(r.glob('*published/*/task-*/output/result.json')):
        v=load(j);p=j.parent;key=(v['year'],v['months'],v['arm'],v['mode'])
        repair=j.parents[3].name=='repair_published';affected=(key[0:3]==(2023,12,'Stacking'))
        if repair:assert key in jobs and affected
        else:assert key not in jobs
        manifest=load(p.parent/'complete.json')['manifest']
        for row in manifest:
            if len(row)==2:assert (p/row[0]).is_dir();continue
            f,size,digest=row;assert (p/f).stat().st_size==size and sha(p/f)==digest
        files.append(dict(current=repair or not affected,year=key[0],months=key[1],model=key[2],mode=key[3],predictions_hash=sha(p/'predictions.npz'),artifact_manifest_hash=sha(p.parent/'complete.json'),host=v['host'],slurm_job=v['job']))
        jobs[key]=(p,v)
    assert len(jobs)==24 and len(files)==26 and sum(j['current'] for j in files)==24
    pd.DataFrame(files).to_csv(D/'PUBLICATION_CHECKS.csv',index=False);return jobs
def main():
    r=run();jobs=tasks(r);base=old('post_st_seed_explanation_stability_v1')/'input';draws=arrays(old('post_st_cause_support_v1')/'private/bootstrap_indices_PRIVATE.npz');rng=np.random.default_rng(2026110201);books={};lab={};cnt={};mx=0;miss={};validation=[]
    for y in YEARS:
        lab[y]=arrays(base/f'evaluation_input/{y}_test_PRIVATE.npz');ids=lab[y]['sample_ids'];nn=len(ids);np.testing.assert_array_equal(rng.integers(nn,size=(1000,nn)),draws[str(y)]);cnt[y]=np.vstack([np.ones(nn,int),*[np.bincount(a,minlength=nn) for a in draws[str(y)]]]);refs,_=stored(y)
        tr=arrays(r/f'input/{y}/train.npz');ev=arrays(r/f'input/{y}/eval.npz');miss[y]=np.isnan(ev['X']).mean(1)>=np.median(np.isnan(tr['X']).mean(1))
        for h in (12,24):
            book={}
            for model in ('XAI/C039','XAI/anchor'):book[model]=endpoints(refs[model],lab[y][f'h{h}'])[:,0]
            for a in ('PROBoost','Stacking'):
                for mode in ('IPCW','complete'):
                    p,_=jobs[y,h,a,mode];z=arrays(p/'predictions.npz');np.testing.assert_array_equal(z['ids'],ids);book[a+'/'+mode]=z['score']
            books[y,h]=book
    records=[];paired=[];sub=[];dist=[]
    for h in (12,24):
        for scope in [*YEARS,'pooled']:
            years=[scope] if scope!='pooled' else list(YEARS[:2] if h==24 else YEARS);ok=(scope,h)!=(2025,24);ids=np.concatenate([lab[y]['sample_ids'] for y in years]);yr=np.concatenate([np.full(len(lab[y]['X']),y) for y in years]);yy=np.concatenate([evaluation_context(lab[y],h,1)[0] for y in years]).astype(int);count=np.column_stack([cnt[y] for y in years]);quota=int(sum(np.ceil(.2*len(lab[y]['X'])) for y in years));base_record=dict(scope=str(scope),months=h,n=len(yy),status='supported' if ok else 'NO_2025_H24_SUPPORT');vectors={}
            for name in books[years[0],h]:
                p=np.concatenate([books[y,h][name] for y in years]);point=dict.fromkeys(FIELDS+MORE,np.nan)
                if ok:
                    for y in years:
                        _,w,known,support,_=evaluation_context(lab[y],h,1);assert support and known.all() and (w==1).all()
                    vals=assessment(count,yy,p,yr,ids);pr=vals[:,8]/quota;vv=np.column_stack([vals,pr,ratio(pr,vals[:,5])]);vectors[name]=vv;point=dict(zip(FIELDS+MORE,vv[0]))
                    np.testing.assert_allclose([point['auc'],point['ap']],[roc_auc_score(yy,p),average_precision_score(yy,p)],rtol=0,atol=1e-12)
                    chosen=np.concatenate([np.flatnonzero(yr==y)[np.lexsort((ids[yr==y],-p[yr==y]))[:int(np.ceil(.2*sum(yr==y)))]] for y in years]);assert yy[chosen].sum()==point['top20_captured']
                    if scope!='pooled':
                        oldm=metrics(lab[scope],p,h,1);np.testing.assert_allclose([point['auc'],point['ap'],point['top20_captured']],[oldm['auc'],oldm['ap'],oldm['captured']],atol=1e-12,rtol=0)
                    m=np.concatenate([miss[y] for y in years])
                    for cat,mask in [('below_training_median',~m),('at_or_above_training_median',m)]:
                        n=int(sum(mask));pos=int(yy[mask].sum());allow=n>=10 and min(pos,n-pos)>=5
                        sub.append(dict(**base_record,model=name,missingness=cat,companies=n,observed_events=pos if allow else None,auc=roc_auc_score(yy[mask],p[mask]) if allow else None,ap=average_precision_score(yy[mask],p[mask]) if allow else None,estimable=allow))
                summary=dict(**base_record,model=name,model_selection='POST_SELECTED_EXPLORATORY' if name=='XAI/C039' else 'fixed reference' if name=='XAI/anchor' else 'historical-only classifier search',target_events=int(yy.sum()) if ok else np.nan,checked=quota if ok else np.nan,**{f:point[f] for f in ('auc','within_year_auc','ap','top20_captured','top20_coverage','precision20','lift20')})
                records.append(summary);dist.append(dict(**base_record,model=name,mean=p.mean(),q25=np.quantile(p,.25),median=np.median(p),q75=np.quantile(p,.75),max=p.max(),unique_scores=len(np.unique(p)),output_scale='CIF' if name.startswith('XAI') else 'fixed-horizon classifier score'))
            for left,right in [(x,y) for x in ('XAI/C039','XAI/anchor') for y in ['PROBoost/IPCW','Stacking/IPCW','PROBoost/complete','Stacking/complete']]+[('PROBoost/IPCW','Stacking/IPCW')]:
                for f in ['auc','within_year_auc','ap','top20_captured','top20_coverage','precision20','lift20']:
                    d=dict(**base_record,left=left,right=right,metric=f,reps=1000 if ok else 0)
                    if ok:
                        i=(FIELDS+MORE).index(f);z=vectors[left][:,i]-vectors[right][:,i];valid=z[1:][np.isfinite(z[1:])];bounds=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan];d.update(estimate=z[0],lower95=bounds[0],upper95=bounds[1],undefined=1000-len(valid))
                    paired.append(d)
    df=pd.DataFrame(records);df[df.scope!='pooled'].to_csv(D/'METRICS_BY_YEAR.csv',index=False);df[df.scope=='pooled'].to_csv(D/'POOLED_METRICS.csv',index=False);df[['scope','months','model','n','target_events','checked','top20_captured','top20_coverage','precision20','lift20','status']].to_csv(D/'TOP20_CAPTURE.csv',index=False);pd.DataFrame(paired).to_csv(D/'PAIRED_COMPARISONS.csv',index=False);pd.DataFrame(sub).to_csv(D/'MISSINGNESS_SENSITIVITY.csv',index=False);pd.DataFrame(dist).to_csv(D/'SCORE_DISTRIBUTIONS.csv',index=False)
if __name__=='__main__':main()
