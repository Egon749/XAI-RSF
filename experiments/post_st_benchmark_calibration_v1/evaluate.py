import warnings
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .models import *
from .bootstrap import FIELDS,assessment
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_reporting_drift_v1.metrics import paired_auc
from experiments.post_st_requested_seeds_v1.analyze import fast_metrics
def quant(x,qs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanquantile(x,qs,axis=0)
def main():
    work=jobs();r=run();prev=old('post_st_seed_explanation_stability_v1')/'input'
    labs={y:arrays(prev/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    books={};hashes={}
    for y in YEARS:
        books[y],hashes[str(y)]=collect(y,work)
        lab=labs[y];ev=arrays(r/f'input/{y}/eval.npz');np.testing.assert_array_equal(ev['sample_ids'],lab['sample_ids'])
        assert len(set(lab['groups']))==len(ev['X'])
    drawfile=old('post_st_cause_support_v1')/'private/bootstrap_indices_PRIVATE.npz';draws=arrays(drawfile);rng=np.random.default_rng(2026110201);rows=[];diffs=[];spread=[];cal=[];aud=[]
    weights={}
    for y in YEARS:
        n=len(labs[y]['X']);np.testing.assert_array_equal(draws[str(y)],rng.integers(n,size=(1000,n)))
        weights[y]=np.vstack([np.ones(n,int),*[np.bincount(q,minlength=n) for q in draws[str(y)]]])
    for h in (12,24):
        for k in (1,2):
            for scope in [*YEARS,'pooled']:
                ys=([scope] if scope!='pooled' else list(YEARS[:2] if h==24 else YEARS))
                ok=not (scope==2025 and h==24); labels=[];yr=[];ids=[];cnt=[]
                for y in ys:
                    yy,w,known,supported,_=evaluation_context(labs[y],h,k);assert supported==ok
                    if ok:assert known.all() and (w==1).all()
                    labels.extend(yy);ids.extend(labs[y]['sample_ids']);yr.extend([y]*len(yy));cnt.append(weights[y])
                yy,ids,yr=np.array(labels,int),np.array(ids),np.array(yr);assert len(set(ids))==len(yy)
                count=np.column_stack(cnt);all_values={};base=dict(scope=str(scope),months=h,cause=k,companies=len(yy),supported=ok)
                for name in books[ys[0]]:
                    p=np.concatenate([endpoints(books[y][name],labs[y][f'h{h}'])[:,k-1] for y in ys])
                    if not ok:
                        rows.append(dict(**base,model=name,**dict.fromkeys(FIELDS,np.nan),mean_prediction_only=float(p.mean()),reason='incomplete 2025 H24'))
                        if name=='XAI/C039':cal.append(dict(**base,mean_prediction_only=float(p.mean()),reason='incomplete 2025 H24'))
                        continue
                    vals=assessment(count,yy,p,yr,ids);all_values[name]=vals
                    point=dict(zip(FIELDS,vals[0]));full=dict(**base,model=name,**point,events=int(yy.sum()),checked=int(sum(np.ceil(.2*len(labs[y]['X'])) for y in ys)))
                    # Keep every frozen triple individually inspectable in addition to summaries.
                    rows.append(full)
                    if scope!='pooled':
                        ref=metrics(labs[scope],p,h,k);np.testing.assert_allclose([point[f] for f in ['auc','ap','ipcw_brier','mean_predicted','observed','top20_captured']],[ref[f] for f in ['auc','ap','brier','risk_mean','observed','captured']],atol=1e-12,rtol=0,equal_nan=True)
                    if name.startswith('triple/'):continue
                    basefast=fast_metrics(yy.astype(bool),p,yr,ids);np.testing.assert_allclose(vals[0,[0,1,2,3,8]],basefast[[0,8,1,2,6]],rtol=0,atol=1e-12,equal_nan=True)
                    if name=='XAI/C039':
                        capture=sum(yy[sorted(np.where(yr==y)[0],key=lambda i:(-p[i],ids[i]))[:int(np.ceil(.2*sum(yr==y)))]].sum() for y in ys)
                        assert capture==point['top20_captured']
                        num=sum(paired_auc(yy[yr==y],p[yr==y])[0]*sum(yy[yr==y])*sum(1-yy[yr==y]) for y in ys if len(set(yy[yr==y]))==2)
                        den=sum(sum(yy[yr==y])*sum(1-yy[yr==y]) for y in ys)
                        if den:assert abs(num/den-point['within_year_auc'])<1e-12
                        if yy.min()!=yy.max():np.testing.assert_allclose([point['auc'],point['ap']],[roc_auc_score(yy,p),average_precision_score(yy,p)])
                        aux={}
                        for f in ['observed_expected','calibration_gap','mean_predicted','observed','ipcw_brier']:
                            i=FIELDS.index(f);q=quant(vals[1:,i],[.025,.975]);aux[f+'_lower95'],aux[f+'_upper95']=q;aux[f+'_undefined']=sum(~np.isfinite(vals[1:,i]))
                        cal.append(full|aux);aud.append(dict(**base,auc=float(point['within_year_auc']) if den else None,top20=int(capture),independent=True))
                if not ok:
                    rows.append(dict(**base,model='XAI/median120',reason='incomplete 2025 H24'))
                    for f in FIELDS:
                        spread.append(dict(**base,metric=f,valid_triples=0))
                    for bench in ['AJ','FG','CSC','KNN_AJ','unrestricted/selected_seeds','unrestricted/anchor_seeds','RF/selected_seeds','RF/anchor_seeds','XAI/anchor']:
                        for ref in ['C039','median120']:
                            for f in FIELDS:diffs.append(dict(**base,reference=ref,comparator=bench,metric=f,repeats=0,reason='incomplete 2025 H24'))
                    continue
                cube=np.stack([all_values['triple/'+n] for n in NAMES]);median=quant(cube,[.5])[0]
                q=quant(cube[:,0,:],[.05,.5,.95])
                for i,f in enumerate(FIELDS):spread.append(dict(**base,metric=f,p05=q[0,i],median=q[1,i],p95=q[2,i],valid_triples=int(np.isfinite(cube[:,0,i]).sum())))
                rows.append(dict(**base,model='XAI/median120',**dict(zip(FIELDS,median[0])),events=yy.sum()))
                for name in ['AJ','FG','CSC','KNN_AJ','unrestricted/selected_seeds','unrestricted/anchor_seeds','RF/selected_seeds','RF/anchor_seeds','XAI/anchor']:
                    for ref,val in [('C039',all_values['XAI/C039']),('median120',median)]:
                        delta=val-all_values[name];bounds=quant(delta[1:],[.025,.975])
                        for i,f in enumerate(FIELDS):
                            diffs.append(dict(**base,reference=ref,comparator=name,metric=f,difference=delta[0,i],lower95=bounds[0,i],upper95=bounds[1,i],undefined=int((~np.isfinite(delta[1:,i])).sum()),repeats=1000))
    pd.DataFrame(rows).to_csv(D/'METRICS_ALL.csv',index=False);pd.DataFrame(diffs).to_csv(D/'PAIRED_DIFFERENCES.csv',index=False);pd.DataFrame(spread).to_csv(D/'SEED_DISTRIBUTION.csv',index=False);pd.DataFrame(cal).to_csv(D/'CALIBRATION.csv',index=False)
    write(D/'METRIC_VALIDATION.json',dict(draw_file_sha256=sha(drawfile),checks=aud,bootstrap='common original 1000 resamples; selection and training fixed',main_source_hashes={y:{a:z for a,z in v.items() if a in map(str,PICK)} for y,v in hashes.items()}))
if __name__=='__main__':main()
