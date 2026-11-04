"""Additional score audit transporting the training censoring distribution to complete evaluation cohorts."""
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_feature_permutation_v1.common import score_batch
from experiments.post_st_benchmark_calibration_v1.models import stored
from experiments.post_st_benchmark_calibration_v1.common import endpoints
from .evaluate import tasks
from .common import *
def main():
    r=run();jobs=tasks(r);data=[];base=old('post_st_seed_explanation_stability_v1')/'input'
    for y in YEARS:
        tr=arrays(r/f'input/{y}/train.npz');lab=arrays(base/f'evaluation_input/{y}_test_PRIVATE.npz');ref,_=stored(y);ev=arrays(r/f'input/{y}/eval.npz');np.testing.assert_array_equal(ev['sample_ids'],lab['sample_ids']);lab['X']=ev['X']
        for h in (12,24):
            if (y,h)==(2025,24):continue
            # First block alone is passed to the censoring estimator.
            fields=['X','groups','label_eligible','event','analysis_time'];combined={k:np.concatenate([tr[k],lab[k]]) for k in fields};start=len(tr['X'])
            hd=Horizon(combined,h,'IPCW');ix,yy,w=hd.weighted(np.arange(start),np.arange(start,len(hd.x)));assert len(yy)==len(lab['X'])
            for model in ['PROBoost/IPCW','Stacking/IPCW','XAI/C039','XAI/anchor']:
                if model.startswith('XAI'):p=endpoints(ref[model],lab[f'h{h}'])[:,0]
                else:p=arrays(jobs[y,h,model.split('/')[0],'IPCW'][0]/'predictions.npz')['score']
                data.append(dict(year=y,h=h,model=model,y=yy,w=w,p=p))
    rows=[]
    for h in [12,24]:
        for model in sorted({d['model'] for d in data}):
            cells=[d for d in data if d['model']==model and d['h']==h]
            for scope,group in [(str(d['year']),[d]) for d in cells]+[('pooled',cells)]:
                y,w,p=(np.concatenate([a[k] for a in group]) for k in ['y','w','p']);res=score_batch(p,y,w);weights=[sum(d['w'][d['y']==1])*sum(d['w'][d['y']==0]) for d in group]
                wy=np.average([roc_auc_score(d['y'],d['p'],sample_weight=d['w']) for d in group],weights=weights)
                np.testing.assert_allclose([res['auc'][0],res['ap'][0]],[roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w)],atol=1e-12,rtol=0)
                rows.append(dict(months=h,scope=scope,model=model,n=len(y),transported_train_G_auc=res['auc'][0],within_year_auc=wy,ap=res['ap'][0],weight_min=w.min(),weight_max=w.max()))
    pd.DataFrame(rows).to_csv(D/'TRAIN_G_RANKING_AUDIT.csv',index=False)
if __name__=='__main__':main()
