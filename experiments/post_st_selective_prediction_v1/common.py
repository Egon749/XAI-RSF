from pathlib import Path
import hashlib,json
import numpy as np
from experiments.post_st_temporal_affine_v1.common import RUN as SOURCE,PARENT,REPO,TIMES,YEARS,SEEDS,REFERENCE_SHA,load,arrays,sha,write,horizon_predictions,block_forest_fits,cohorts

NAME='post_st_selective_prediction_v1'
RUN=Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/selective_prediction_20261102_run1')
ARMS=['combined','support','seed_sd','low_risk_control'];COVERAGES=[.9,.8,.6]
RAW=['seed_sd','function_sparse_share','leaf_company_penalty','leaf_cause_penalty','leaf_followup_penalty']
TIE_SEED=2026110221;RANDOM_SEED=2026110222;BOOT_SEED=2026110223;HISTORY_SEED=2026110224

def tie_values(ids):
    return np.array([int.from_bytes(hashlib.sha256(f'{TIE_SEED}|{v}'.encode()).digest()[:8],'big') for v in ids],dtype=np.uint64)

def ecdf(reference,values):
    r=np.sort(np.asarray(reference,float));v=np.asarray(values,float)
    if not len(r) or not np.isfinite(r).all() or not np.isfinite(v).all():raise ValueError('finite reference and values required')
    return (np.searchsorted(r,v,'left')+np.searchsorted(r,v,'right'))/(2*len(r))

def scores(reference,values,arm):
    if arm=='low_risk_control':return values[:,5].copy()
    if arm=='seed_sd':return values[:,0].copy()
    ix=range(5) if arm=='combined' else range(1,5)
    return np.mean([ecdf(reference[:,j],values[:,j]) for j in ix],axis=0)

def threshold(score,ids,coverage):
    if not 0<coverage<=1 or not len(score):raise ValueError('nonempty historical data and legal coverage required')
    tie=tie_values(ids);k=int(np.ceil(coverage*len(score)));order=np.lexsort((tie,score));i=order[k-1]
    return dict(value=float(score[i]),tie=int(tie[i]),historical_n=len(score),target=float(coverage))

def accept(score,ids,cut):
    s=np.asarray(score);return (s<cut['value'])|((s==cut['value'])&(tie_values(ids)<=cut['tie']))

def matched(score,ids,n):
    mask=np.zeros(len(score),bool);mask[np.lexsort((tie_values(ids),score))[:n]]=True;return mask

def matrix(d,h,c):return np.column_stack([d[f'{h}_{c}_{k}'] for k in RAW]+[d[f'{h}_{c}_risk']])

def cell_support(ex,key,reference,X,groups):
    """Empirical JOINT cell support, counting a company once, missing separate."""
    shape=[];rb=[];xb=[]
    for j in key:
        cuts=ex.cuts(key,j);shape.append(len(cuts)+2)
        for data,dest in [(reference,rb),(X,xb)]:
            b=np.searchsorted(cuts,data[:,j],side='left');b[np.isnan(data[:,j])]=len(cuts)+1;dest.append(b)
    train=np.ravel_multi_index(tuple(rb),shape);query=np.ravel_multi_index(tuple(xb),shape)
    pairs=np.unique(np.column_stack([train,np.unique(groups,return_inverse=True)[1]]),axis=0)
    count=np.bincount(pairs[:,0],minlength=int(np.prod(shape)))
    return count[query]
