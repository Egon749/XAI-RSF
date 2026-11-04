"""Full global grid with empirical joint support separated from structural mass."""
from itertools import product
import numpy as np
from .common import *
from .similarity import *

def cell_geometry(ex,key,X):
    cuts=[np.unique(np.concatenate([a.cuts(key,j) for a in ex if (j in a.mains if len(key)==1 else key in a.pairs)])) for j in key]
    reps=[ex[0].representatives(c) for c in cuts]; coords=np.array(list(product(*reps))); grid=np.zeros((len(coords),X.shape[1]))
    grid[:,list(key)]=coords
    bins=[]
    for j,cut in zip(key,cuts):
        b=np.searchsorted(cut,X[:,j],side='left');b[np.isnan(X[:,j])]=len(cut)+1;bins.append(b)
    shape=tuple(len(r) for r in reps)
    flat=np.ravel_multi_index(tuple(bins),shape)
    n=np.bincount(flat,minlength=len(grid))
    return grid,flat,n,shape

def med(a):
    a=np.asarray(a);ok=np.isfinite(a)
    return float(np.median(a[ok])) if ok.any() else np.nan

def stats(a,n,present,region):
    select=n>0 if region=='observed' else n==0
    if not select.any():return []
    x=a[:,select];w=n[select].astype(float) if region=='observed' else np.ones(x.shape[1]);w/=w.sum()
    mu=x@w;z=(x-mu[:,None])*np.sqrt(w);norm=np.linalg.norm(z,axis=1)
    corr=divide(z@z.T,norm[:,None]*norm)
    corr[(norm<1e-12) | (norm[:,None]<1e-12)]=np.nan
    # Symmetric identity for exact weighted RMS; zero tiny roundoff in squared distances only.
    weighted=x*np.sqrt(w); gram=weighted@weighted.T
    rms=np.sqrt(np.maximum(0,gram.diagonal()[:,None]+gram.diagonal()-2*gram))
    pos=(x>=MATERIAL).astype(float);neg=(x<=-MATERIAL).astype(float)
    common=((pos+neg)*w)@(pos+neg).T
    sign=divide((pos*w)@pos.T+(neg*w)@neg.T,common)
    high=(x>0)&(x>=np.quantile(x,.8,axis=1)[:,None]); high_overlap=jaccards(high)
    exists=present[LEFT]&present[RIGHT]
    amp=abs(x)@w
    rows=[]
    for scope in ('all','disjoint'):
        use=exists if scope=='all' else exists&(SHARED==0)
        def typical(mat):return med(mat[LEFT[use],RIGHT[use]])
        rows.append(dict(region=region,comparison=scope,common_pairs=int(use.sum()),valid_shape_pairs=int(np.isfinite(corr[LEFT[use],RIGHT[use]]).sum()),
                         shape_correlation=typical(corr),rms_difference=typical(rms),sign_agreement=typical(sign),
                         both_material_mass=typical(common),high_region_jaccard=typical(high_overlap),
                         mean_abs_p05=np.quantile(amp,.05),mean_abs_median=med(amp),mean_abs_p95=np.quantile(amp,.95),
                         nearzero_fraction=float((amp<MATERIAL).mean())))
    return rows

def geometry(year,version,members,train,names,masks,out):
    rows=[];supports=[];private=[];curve=[]
    keys=sorted({(j,) for a in members for j in a.mains}|{k for a in members for k in a.pairs})
    assert len(np.unique(train['groups']))==len(train['X'])
    for key in keys:
        grid,binids,n,shape=cell_geometry(members,key,train['X'])
        token=','.join(map(str,key)); label=' × '.join(names[j] for j in key)
        present=np.array([token in masks[k] for k in NAMES])
        mats=np.stack([a.component(grid,key,[365,730]) for a in members])
        values=mats[np.array(TRIPLES)].mean(1)
        for t,day in enumerate((365,730)):
            for c in (1,2):
                support=np.bincount(binids[(train['event']==c)&(train['time']<=day)],minlength=len(n))
                a=values[:,:,t,c-1];positive=(a>=MATERIAL).mean(0)>=.9
                base=dict(year=year,version=version,day=day,cause=c,order=len(key),term=label)
                important_support=bool(n[positive].sum()>=10 and support[positive].sum()>=3)
                supports.append(dict(**base,cells=len(n),occupied_fraction=np.mean(n>0),zero_fraction=np.mean(n==0),low_fraction=np.mean(n<10),
                     median_occupied_companies=med(n[n>0]),cells_with_ge3_cause_companies=int((support>=3).sum()),
                     positive_consensus_sample_gate=important_support,positive_consensus_cell_fraction=positive.mean(),
                     cells_with_both_ge10_companies_ge3_events=int(((n>=10)&(support>=3)).sum())))
                for i in range(len(n)):
                    private.append(dict(**base,cell=i,companies=int(n[i]),events=int(support[i]),positive_consensus=bool(positive[i])))
                for region in ('observed','unsupported'):
                    for summary in stats(a,n,present,region):
                        screen=(summary['region']=='observed' and summary['comparison']=='disjoint' and present.mean()>=.9 and
                                summary['mean_abs_median']>=MATERIAL and summary['shape_correlation']>=.8 and summary['sign_agreement']>=.9 and important_support)
                        rows.append(dict(**base,presence=present.mean(),**summary,screen_pass=screen))
                if len(key)==1:
                    for i in np.flatnonzero(n>=5):
                        curve.append(dict(**base,bin=int(i),missing=bool(i==len(n)-1),percentile_midpoint=(n[:i].sum()+.5*n[i])/max(1,n[:-1].sum()) if i<len(n)-1 else np.nan,
                                          p05=np.quantile(a[:,i],.05),median=med(a[:,i]),p95=np.quantile(a[:,i],.95)))
    import pandas as pd
    for filename,data in [('terms',rows),('support',supports),('cells_PRIVATE',private),('curve',curve)]:pd.DataFrame(data).to_csv(out/(filename+'.csv'),index=False)
    return keys
