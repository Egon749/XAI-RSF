"""Preserve bootstrap multiplicities in the model and enforce company isolation and real-event support."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiments.post_st_eswa_baselines_v1.common import Horizon,seed,transform,transform_stats
from experiments.post_st_eswa_baselines_v1.stacking import Search,smote
def draw_rows(year,rep,n):
    rng=np.random.default_rng(np.random.SeedSequence([20261104,year,rep]))
    return np.sort(rng.integers(n,size=n))
def representatives(groups):
    return np.sort(np.unique(groups,return_index=True)[1])
class BootstrapHorizon(Horizon):
    def __init__(self,original,h,rows):
        super().__init__(original,h,'IPCW')
        self.d={k:v[rows] for k,v in original.items()};self.x=self.d['X'];self.t=self.t[rows];self.e=self.e[rows];self.y=self.y[rows]
    def positive_firms(self,ix):
        return len(np.unique(self.d['groups'][ix[self.y[ix]==1]]))
    def split(self,ix,k,salt):
        groups=self.d['groups'][ix];reps=representatives(groups);labels=self.y[ix[reps]]==1
        k=min(k,int(labels.sum()),int((~labels).sum()))
        if k<2:return []
        cv=StratifiedKFold(n_splits=k,shuffle=True,random_state=seed(ix,salt));out=[]
        for a,b in cv.split(reps,labels):
            inside=np.isin(groups,groups[reps[a]]);fit,val=ix[inside],ix[~inside]
            assert not np.intersect1d(self.d['groups'][fit],self.d['groups'][val]).size
            out.append((fit,val))
        return out
class BootstrapSearch(Search):
    def design(self,ix,cols):
        hd=self.h;kk,y,w=hd.weighted(ix,ix);st=transform_stats(hd.x[ix]);x=transform(hd.x[kk],st,cols)
        rep=representatives(hd.d['groups'][kk]);expanded,labels,weights,n=smote(x[rep],y[rep],w[rep],st[3][cols],seed(ix,912))
        if n:
            x=np.vstack([x,expanded[-n:]]);y=np.r_[y,labels[-n:]];w=np.r_[w,weights[-n:]]
        assert n<=hd.positive_firms(ix)
        return st,x,y,w,n
    def log(self,stage,ix,syn=0,**extra):
        super().log(stage,ix,syn,distinct_positive_firms=self.h.positive_firms(ix),**extra)
