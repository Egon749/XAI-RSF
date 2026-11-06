"""Known-endpoint discrimination, competing-risk IPCW, and annual screening."""
import numpy as np
from crforest import aalen_johansen
from experiments.post_st_prognosis_v1.common import endpoint_weights

def binary_scores(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,float)
    if not len(y):return np.nan,np.nan
    pos=int(y.sum());neg=len(y)-pos;order=np.argsort(p,kind='stable');s=p[order];z=y[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];counts=np.diff(np.r_[starts,len(z)])
    tp=np.add.reduceat(z.astype(float),starts);fp=counts-tp
    auc=float(np.sum(tp*(np.cumsum(fp)-fp+fp/2))/(pos*neg)) if pos and neg else np.nan
    ap=float(np.sum(tp[::-1]/pos*np.cumsum(tp[::-1])/np.cumsum(counts[::-1]))) if pos else np.nan
    return auc,ap

def independent_weights(t,e,h):
    """Literal reverse-KM loop for independent evaluation verification."""
    t,e=np.asarray(t),np.asarray(e);dates=np.unique(t);before={};survival=1.
    for d in dates:
        before[d]=survival
        survival*=1-float(np.sum((t==d)&(e==0)))/np.sum(t>=d)
    def Gleft(d):
        value=1.
        for u in dates[dates<d]:value*=1-np.sum((t==u)&(e==0))/np.sum(t>=u)
        return value
    result=np.zeros(len(t));gh=Gleft(h)
    for i,(u,k) in enumerate(zip(t,e)):
        if k>0 and u<=h:result[i]=1/before[u]
        elif u>=h:result[i]=1/gh
    return result

def metrics(t,e,p,year,ids,groups,h,cause):
    t,e,p,year,ids,groups=map(np.asarray,(t,e,p,year,ids,groups))
    w,known,diag=endpoint_weights(t,e,h);y=(e==cause)&(t<=h)
    auc,ap=binary_scores(y[known],p[known]);aucw=[];pairs=[]
    for yr in np.unique(year):
        k=known&(year==yr);a,_=binary_scores(y[k],p[k]);n=int(y[k].sum())*int((~y[k]).sum())
        if n:aucw.append(a);pairs.append(n)
    aj=float(aalen_johansen(t,e,[h],n_causes=2)[0,cause-1])
    result=dict(records=len(t),companies=len(np.unique(groups)),known=int(known.sum()),unknown=int((~known).sum()),
        target_records=int(y.sum()),target_companies=len(np.unique(groups[y])),auc=auc,ap=ap,
        within_year_auc=float(np.average(aucw,weights=pairs)) if pairs else np.nan,
        equal_year_auc=float(np.mean(aucw)) if pairs else np.nan,informative_years=len(pairs),
        ipcw_brier=float(np.sum(w*(y-p)**2)/len(p)),risk_mean=float(p.mean()),risk_max=float(p.max()),
        observed_aj=aj,observed_known_rate=float(y[known].mean()) if known.any() else np.nan,
        observed_expected=aj/p.mean() if p.mean()>0 else np.nan,calibration_gap=float(p.mean()-aj),ipcw_min_G=diag['min_G'])
    for fraction in [.2,.25]:
        chosen=[]
        for yr in np.unique(year):
            ix=np.flatnonzero(year==yr);chosen.extend(ix[np.lexsort((ids[ix],-p[ix]))[:int(np.ceil(fraction*len(ix)))]])
        ix=np.array(chosen,int);prefix=f'top{int(fraction*100)}_'
        result.update({prefix+'checked':len(ix),prefix+'captured':int(y[ix].sum()),prefix+'unique_events':len(np.unique(groups[ix][y[ix]])),
            prefix+'coverage':float(y[ix].sum()/y.sum()) if y.sum() else np.nan,
            prefix+'known_non_target':int((known[ix]&~y[ix]).sum()),prefix+'unknown':int((~known[ix]).sum())})
    return result
