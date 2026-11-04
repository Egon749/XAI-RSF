"""Fixed partial likelihoods and prediction maps. Times in actual days; Breslow for tied failures."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from crforest.core import aalen_johansen

def transform(train,test):
    if np.isinf(train).any() or np.isinf(test).any():raise ValueError('unexpected infinite input')
    med=np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0. for x in train.T])
    flags=np.isnan(train).any(0)&np.isfinite(train).any(0)
    def fill(x):return np.column_stack([np.where(np.isnan(x),med,x),np.isnan(x[:,flags])])
    a,b=fill(train),fill(test);sd=a.std(0);keep=sd>0;avg=a.mean(0)
    return (a[:,keep]-avg[keep])/sd[keep],(b[:,keep]-avg[keep])/sd[keep],dict(median=med,flags=flags,keep=keep,mean=avg,sd=sd)

def censor_left(time,event):
    times,inv,count=np.unique(time,return_inverse=True,return_counts=True)
    atrisk=np.cumsum(count[::-1])[::-1]; cens=np.bincount(inv,weights=event==0,minlength=len(times))
    return times,np.r_[1.,np.cumprod(1-cens/atrisk)[:-1]]

def estimate(x,t,e,k,penalty,fg):
    take=e==k;u,d=np.unique(t[take],return_counts=True)
    if not len(u):raise ValueError('no cause events in fit: partial likelihood undefined')
    risk=(t[None,:]>=u[:,None]).astype(float)
    if fg:
        ts,g=censor_left(t,e);prior=((e>0)&~take)[None,:]&(t[None,:]<u[:,None]);lo=g[np.searchsorted(ts,t)]
        assert np.all(lo>0)
        risk+=prior*(g[np.searchsorted(ts,u),None]/lo[None,:])
    logw=np.full_like(risk,-np.inf);np.log(risk,out=logw,where=risk>0)
    sufficient=x[take].mean(0);norm=d/d.sum()
    def loss(b):
        eta=x@b;den=logsumexp(logw+eta,axis=1)
        score=(norm@np.exp(logw+eta-den[:,None]))@x-sufficient+penalty*b
        return float(norm@den-sufficient@b+.5*penalty*(b@b)),score
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options=dict(maxiter=600,ftol=1e-13,gtol=1e-8))
    if not fit.success or np.max(abs(fit.jac))>1e-5:raise RuntimeError(f'partial likelihood failed: {fit.message}, gradient {np.max(abs(fit.jac))}')
    increments=d*np.exp(-logsumexp(logw+x@fit.x,axis=1))
    return dict(beta=fit.x,time=u,jump=increments,iterations=int(fit.nit),gradient=float(abs(fit.jac).max()))

def regression_predict(models,x,days,fg):
    ts=np.unique(np.concatenate([m['time'] for m in models]));jump=np.zeros((len(x),len(ts),2))
    for k,m in enumerate(models):jump[:,np.searchsorted(ts,m['time']),k]=np.exp(x@m['beta'])[:,None]*m['jump']
    if fg:full=-np.expm1(-jump.cumsum(axis=1))
    else:
        total=jump.sum(2);prev=np.column_stack([np.zeros(len(x)),np.cumsum(total,axis=1)[:,:-1]])
        scale=np.divide(-np.expm1(-total),total,out=np.ones_like(total),where=total>0)
        full=np.cumsum((np.exp(-prev)*scale)[:,:,None]*jump,axis=1)
    index=np.searchsorted(ts,days,side='right')-1
    ans=full[:,np.maximum(index,0)].copy();ans[:,index<0]=0
    return ans

def knn(x,z,t,e,k,days):
    order=np.argsort(cdist(z,x),kind='stable',axis=1)[:,:k]
    return np.stack([aalen_johansen(t[ix],e[ix],days,2) for ix in order])
