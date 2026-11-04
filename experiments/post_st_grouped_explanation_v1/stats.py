"""Same estimator for the economic map and every negative control."""
import ctypes
from scipy.stats import rankdata
from .common import *
METS=['rank_rho','top_jaccard','l1','relative_l1','sign_agreement','joint_material_fraction','any_joint_material']
OVER=np.zeros((120,120),np.int32);OVER[LEFT,RIGHT]=SHARED
def meanfinite(x,axis=None):
    x=np.asarray(x);n=np.isfinite(x).sum(axis=axis);s=np.nansum(x,axis=axis)
    return np.divide(s,n,out=np.full(np.shape(s),np.nan),where=n>0)
def similarity(vals,k,individual=False):
    """vals shape is [firm, 120 triples, terms], zero ties receive average rank."""
    vals=np.ascontiguousarray(vals)
    rank=rankdata(abs(vals),axis=-1);rank-=rank.mean(-1,keepdims=True)
    norm=np.linalg.norm(rank,axis=-1,keepdims=True)
    rank=np.ascontiguousarray(np.divide(rank,norm,out=np.full_like(rank,np.nan),where=norm>0))
    best=np.argsort(-abs(vals),axis=-1,kind='stable')[...,:k]
    best=np.ascontiguousarray(np.where(np.take_along_axis(abs(vals),best,-1)>=.001,best,-1),dtype=np.int32)
    out=np.zeros((len(vals),3,len(METS),2));ax=np.zeros((len(vals),3,vals.shape[-1],2)) if individual else None
    fun=ctypes.CDLL(str(D/'kernel.so')).pair_scores;fun.restype=None
    ptr=lambda a:ctypes.c_void_p(a.ctypes.data) if a is not None else None
    fun(len(vals),vals.shape[-1],k,ptr(vals),ptr(rank),ptr(best),ptr(OVER),ptr(out),ptr(ax))
    return out,ax
def quant(x,q):
    x=np.asarray(x);z=x[np.isfinite(x)];return np.quantile(z,q) if len(z) else np.nan
def summary(values,k,level,economic,context):
    scored,axis=similarity(values,k,individual=economic and level in ('A8','C8','I28','C4'))
    rows=[]
    for s in (0,1,2):
        for i,key in enumerate(METS):
            v=scored[:,s,i,0];valid=scored[:,s,i,1]
            rows.append(dict(**context,level=level,top_k=k,shared=s,metric=key,mean=meanfinite(v),firms_defined=int(np.isfinite(v).sum()),valid_comparisons=int(valid.sum()),
                firm_p05=quant(v,.05) if economic else np.nan,firm_p95=quant(v,.95) if economic else np.nan))
    return rows,axis,scored
