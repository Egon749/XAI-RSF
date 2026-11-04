"""Company-level rankings on complete external endpoints or censor-weighted OOB outcomes."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
SCORES=['unique_scores','score_min','score_max','score_span']
PERF=['auc','within_year_auc','ap','top20_captured','top20_coverage','precision20','lift20']
def divide(a,b):return a/b if b>0 else np.nan
def metrics(data,performance=True):
    year,ids,y,w,p=(np.concatenate([d[k] for d in data]) for k in ['year','ids','y','w','p'])
    assert len(p) and np.isfinite(p).all()
    out=dict(n=len(p),unique_scores=len(np.unique(p)),score_min=p.min(),score_max=p.max(),score_span=np.ptp(p))
    if not performance:return out|dict.fromkeys(PERF,np.nan)
    known=(y>=0);assert np.array_equal(w>0,known)
    active=y[known];ww=w[known];pos=ww[active==1].sum();neg=ww[active==0].sum()
    auc=roc_auc_score(active,p[known],sample_weight=ww) if min(pos,neg)>0 else np.nan
    ap=average_precision_score(active,p[known],sample_weight=ww) if pos>0 else np.nan
    selected=np.zeros(len(p),bool);total=0.;pairs=0.
    for a in np.unique(year):
        ix=np.flatnonzero(year==a);order=ix[np.lexsort((ids[ix],-p[ix]))];selected[order[:int(np.ceil(.2*len(ix)))]]=True
        i=ix[known[ix]];weight=w[i][y[i]==1].sum()*w[i][y[i]==0].sum()
        if weight>0:
            val=roc_auc_score(y[i],p[i],sample_weight=w[i]);total+=val*weight;pairs+=weight
    caught=w[selected&(y==1)].sum();precision=divide(caught,w[selected].sum());coverage=divide(caught,pos)
    return out|dict(auc=auc,within_year_auc=divide(total,pairs),ap=ap,top20_captured=int(sum(selected&(y==1))),
        top20_coverage=coverage,precision20=precision,lift20=divide(precision,divide(pos,pos+neg)),known=int(known.sum()),
        events=int(sum(y==1)),unknown=int(sum(~known)),checked=int(sum(selected)),unknown_checked=int(sum(selected&~known)))
def record(y,ids,truth,weight,score):return dict(year=np.full(len(ids),y),ids=ids,y=truth,w=weight,p=score)
def position(x,orig):
    if not len(x) or not np.isfinite(orig):return np.nan
    t=np.isclose(x,orig,rtol=0,atol=1e-12)
    return 100*(np.count_nonzero((x<orig)&~t)+.5*t.sum())/len(x)
