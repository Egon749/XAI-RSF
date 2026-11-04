"""Apply v2 disclosure gates independently to three-tier and binary summaries."""
import pandas as pd
from .common import *
from .models import stored
from .bootstrap import assessment
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_risk_index_v1.index import tiers
def main():
    r=run();inp=old('post_st_seed_explanation_stability_v1')/'input';dd=arrays(old('post_st_cause_support_v1')/'private/bootstrap_indices_PRIVATE.npz')
    high=[];diff=[];replay=[]
    for h in [12,24]:
        for index_cause in [1,2]:
            store={}
            for y in YEARS:
                pp=stored(y)[0]['XAI/C039'];lab=arrays(inp/f'evaluation_input/{y}_test_PRIVATE.npz')
                val=arrays(r/f'risk_index/{y}/scores_{h}.npz')['score'][:,index_cause-1];threshold=arrays(r/f'risk_index/{y}/oob.npz')['threshold'];tier=tiers(val,threshold[h//12-1,index_cause-1]);group=lab['groups'];ids=lab['sample_ids']
                yy,w,known,ok,_=evaluation_context(lab,h,index_cause)
                if ok:
                    a=assessment(np.ones((1,len(ids)),int),yy.astype(int),val,np.full(len(ids),y),ids);b=assessment(np.ones((1,len(ids)),int),yy.astype(int),endpoints(pp,lab[f'h{h}'])[:,index_cause-1],np.full(len(ids),y),ids)
                    np.testing.assert_array_equal(a[0,[0,1,2,8,9]],b[0,[0,1,2,8,9]])
                    replay.append(dict(year=y,months=h,cause=index_cause,auc_ap_top20_same=True))
                    store[y]=(tier,np.column_stack([evaluation_context(lab,h,k)[0] for k in (1,2)]))
            for scope in [*store,'pooled']:
                years=list(store) if scope=='pooled' else [scope]
                cat=np.concatenate([store[y][0] for y in years]);ev=np.vstack([store[y][1] for y in years])
                m=np.column_stack([np.vstack([np.ones(len(store[y][0]),int),*[np.bincount(ix,minlength=len(store[y][0])) for ix in dd[str(y)]]]) for y in years])
                for k in (1,2):
                    event=ev[:,k-1];base=dict(scope=scope,months=h,index_cause=index_cause,outcome_cause=k)
                    for label,groups in [('high_not_high',[cat=='high',cat!='high']),('high_low',[cat=='high',cat=='low',cat=='medium'])]:
                        ns=[sum(g) for g in groups];nn=[sum(event[g]) for g in groups];allow=all(n==0 or n>=5 for n in [*ns,*nn])
                        if label=='high_not_high':
                            for i,name in enumerate(['high','not_high']):
                                high.append(dict(**base,tier=name,companies=ns[i] if all(n==0 or n>=5 for n in ns) else None,events=nn[i] if allow else None,rate=nn[i]/ns[i] if allow and ns[i] else None,status='released' if allow else 'small cell/complement'))
                        if not allow or not min(ns[:2]):
                            diff.append(dict(**base,contrast=label,status='insufficient or small cell/complement'));continue
                        rates=[]
                        for g in groups[:2]:
                            den=m@g;rates.append(np.divide(m@(event&g),den,out=np.full(len(m),np.nan),where=den>0))
                        d=rates[0]-rates[1];v=d[1:][np.isfinite(d[1:])];q=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
                        diff.append(dict(**base,contrast=label,status='released',observed_difference=d[0],lower95=q[0],upper95=q[1],undefined=1000-len(v)))
    pd.DataFrame(diff).to_csv(D/'TIER_DIFFERENCES.csv',index=False);pd.DataFrame(high).to_csv(D/'HIGH_VS_REMAINDER.csv',index=False);pd.DataFrame(replay).to_csv(D/'INDEX_METRIC_REPLAY.csv',index=False)
if __name__=='__main__':main()
