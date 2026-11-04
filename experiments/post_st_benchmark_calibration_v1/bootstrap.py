"""Exact fixed-prediction bootstrap using resampled row multiplicities, including ties."""
import numpy as np
FIELDS=['auc','within_year_auc','ap','ipcw_brier','mean_predicted','observed','calibration_gap','observed_expected','top20_captured','top20_coverage']
def ratio(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)!=0)
def ranks(count,y,p):
    order=np.argsort(p,kind='stable');s=p[order];starts=np.r_[0,1+np.flatnonzero(np.diff(s)!=0)]
    pop=np.add.reduceat(count[:,order],starts,axis=1);tp=np.add.reduceat(count[:,order]*y[order],starts,axis=1);fp=pop-tp
    numer=(tp*(fp.cumsum(1)-fp/2)).sum(1);den=tp.sum(1)*fp.sum(1)
    pos=tp[:,::-1];cum=pop[:,::-1].cumsum(1)
    precision=np.divide(pos.cumsum(1),cum,out=np.zeros(cum.shape),where=cum>0)
    return numer,den,ratio((pos*precision).sum(1),pos.sum(1))
def assessment(count,y,p,year,ids):
    n=count.sum(1);pos=count@y;obs=pos/n;pred=count@p/n;num,den,ap=ranks(count,y,p);sn=np.zeros(len(n));sd=sn.copy();caught=sn.copy()
    for a in np.unique(year):
        ix=np.flatnonzero(year==a);nn,dd,_=ranks(count[:,ix],y[ix],p[ix]);sn+=nn;sd+=dd
        order=ix[np.lexsort((ids[ix],-p[ix]))];m=count[:,order];quota=np.ceil(m.sum(1)*.2).astype(int)
        in_top=np.minimum(m,np.maximum(0,quota[:,None]-m.cumsum(1)+m));caught+=in_top@y[order]
    return np.column_stack([ratio(num,den),ratio(sn,sd),ap,count@((p-y)**2)/n,pred,obs,pred-obs,ratio(obs,pred),caught,ratio(caught,pos)])
