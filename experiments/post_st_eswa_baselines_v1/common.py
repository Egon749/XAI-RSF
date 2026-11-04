import hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score,f1_score
from experiments.post_st_benchmark_calibration_v1.common import old,arrays,sha,DAYS,YEARS
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
def run():return Path(os.environ['ESWA_RUN'])
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')
def load(p):return json.loads(Path(p).read_text())
def seed(ix,salt):
    return int.from_bytes(hashlib.sha256(np.array(ix,dtype='<i8').tobytes()+str(salt).encode()).digest()[:4],'little')%2147483646+1
class Horizon:
    def __init__(self,d,h,mode):
        self.d=d;self.x=d['X'];self.t=d['analysis_time'];self.e=d['event'];self.h=h;self.mode=mode
        self.y=np.where((self.e>0)&(self.t<=h),self.e==1,np.where(self.t>=h,0,-1)).astype(int)
        assert len(np.unique(d['groups']))==len(self.x) and d['label_eligible'].all()
    def weighted(self,fit,ix):
        keep=ix[self.y[ix]>=0];w=np.ones(len(keep))
        if self.mode=='IPCW':
            ts,inv,n=np.unique(self.t[fit],return_inverse=True,return_counts=True);risk=np.cumsum(n[::-1])[::-1];dc=np.bincount(inv,weights=self.e[fit]==0,minlength=len(n));g=np.cumprod(1-dc/risk)
            u=np.minimum(self.t[keep],self.h);j=np.searchsorted(ts,u,side='left')-1;den=np.where(j>=0,g[np.maximum(j,0)],1.)
            if np.any(den<=0):raise ValueError('IPCW positivity violation')
            w=1/den
        return keep,self.y[keep],w
    def split(self,ix,k,salt):
        k=min(k,int(sum(self.y[ix]==1)),int(sum(self.y[ix]!=1)))
        if k<2:return []
        cv=StratifiedKFold(n_splits=k,shuffle=True,random_state=seed(ix,salt))
        return [(ix[a],ix[b]) for a,b in cv.split(ix,self.y[ix]==1)]
def transform_stats(x):
    med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in x.T]);a=np.where(np.isnan(x),med,x)
    sd=a.std(0);sd[sd==0]=1.
    discrete=np.array([np.unique(v[np.isfinite(v)]).size<=2 for v in x.T])
    # All audit status, warning and missingness flags are inherited together during SMOTE.
    return med,a.mean(0),sd,discrete
def transform(x,st,cols):
    med,mean,sd,_=st;a=x[:,cols];miss=np.isnan(a)
    return np.column_stack([(np.where(miss,med[cols],a)-mean[cols])/sd[cols],miss.astype(float)])
def ap(y,p,w):return float(average_precision_score(y,p,sample_weight=w))
def f1(y,p,w):return float(f1_score(y,np.asarray(p)>=.5,sample_weight=w,zero_division=0))
