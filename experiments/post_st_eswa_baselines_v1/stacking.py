"""Finite, fold-local reimplementation of the Chen IG–IPSO / improved stacking design."""
import time,pickle,warnings
from functools import partial
import numpy as np
from scipy.special import expit,xlogy
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from hyperopt import fmin,hp,tpe,Trials,STATUS_OK
from .common import seed,transform_stats,transform,ap,f1
LEARNERS=('RF','GBDT','LightGBM','XGBoost','AdaBoost')
PARAMS=([2,4,None],[1,2,3],[3,7,15],[1,2,3],[.25,.6,1.])
def learner(i,j,r):
    v=PARAMS[i][j]
    if i==0:return RandomForestClassifier(n_estimators=24,max_depth=v,random_state=r,n_jobs=1)
    if i==1:return GradientBoostingClassifier(n_estimators=24,max_depth=v,learning_rate=.08,random_state=r)
    if i==2:return LGBMClassifier(n_estimators=24,num_leaves=v,learning_rate=.08,verbosity=-1,n_jobs=1,random_state=r)
    if i==3:return XGBClassifier(n_estimators=24,max_depth=v,learning_rate=.08,tree_method='hist',n_jobs=1,random_state=r)
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=24,learning_rate=v,random_state=r)
def predict(m,x):
    if isinstance(m,LGBMClassifier):return m.booster_.predict(x,num_threads=1)
    return m.predict_proba(x)[:,1]
def entropy(pos,total):
    prob=np.divide(pos,total,out=np.zeros_like(np.asarray(pos,dtype=float)),where=total>0)
    return -(xlogy(prob,prob)+xlogy(1-prob,1-prob))/np.log(2)
def ig(hd,ix):
    keep,y,w=hd.weighted(ix,ix);base=entropy(np.sum(w*y),w.sum());gain=[]
    for col in hd.x[ix].T:
        obs=col[np.isfinite(col)];cut=np.unique(np.quantile(obs,[.25,.5,.75])) if len(obs) else []
        vals=col[hd.y[ix]>=0];b=np.searchsorted(cut,vals);b[np.isnan(vals)]=len(cut)+1
        pop=np.bincount(b,weights=w);pos=np.bincount(b,weights=w*y,minlength=len(pop))
        gain.append(base-(pop@entropy(pos,pop))/w.sum())
    return np.argsort(-np.array(gain),kind='stable')[:24]
def smote(x,y,w,discrete,r):
    pos=np.flatnonzero(y==1);n=len(pos)
    if n<6 or sum(y==0)<=n:return x,y,w,0
    rng=np.random.default_rng(r);d=x.shape[1]//2;dist=cdist(x[pos],x[pos]);np.fill_diagonal(dist,np.inf)
    order=np.argsort(dist,axis=1,kind='stable')[:,:min(n-1,5)];nn=min(n,int(sum(y==0))-n);a=np.arange(nn);b=order[a,rng.integers(order.shape[1],size=nn)];xa=x[pos[a]];xb=x[pos[b]]
    frac=rng.random((nn,1));fake=xa.copy();modifiable=(~discrete)[None,:]&(xa[:,d:]==0)&(xb[:,d:]==0)
    fake[:,:d]=np.where(modifiable,xa[:,:d]+frac*(xb[:,:d]-xa[:,:d]),xa[:,:d])
    return np.vstack([x,fake]),np.r_[y,np.ones(nn,int)],np.r_[w,w[pos[a]]],nn
def inertia(w,f):
    mean=f.mean();ratio=f/mean if mean>0 else np.ones_like(f)
    return w*(np.where(f<mean,2.,1.)-expit(ratio))
def mask_of(x):
    m=x>=.5
    if sum(m)<4:m[np.argsort(-x,kind='stable')[:4]]=True
    return m
def feasible(mask):return 3<=sum(mask)<=5
def fweights(f,mask):
    v=f*mask
    return v/v.sum() if v.sum()>0 else mask.astype(float)/sum(mask)
def lr(x,y,w):
    model=LogisticRegression(C=1.,solver='lbfgs',max_iter=500,tol=1e-8);model.fit(x,y,sample_weight=w)
    if max(model.n_iter_)>=500:raise RuntimeError('meta LR failed to converge')
    return model
class Search:
    def __init__(self,h):
        self.h=h;self.logs=[];self.select_cache={};self.base_cache={};self.audit=[]
    def log(self,stage,ix,syn=0,**extra):
        self.logs.append(dict(stage=stage,fit_rows=int(sum(self.h.y[ix]>=0)),fit_pos=int(sum(self.h.y[ix]==1)),synthetic=int(syn),**extra))
    def split(self,ix,k,salt):
        folds=self.h.split(ix,k,salt)
        for a,b in folds:
            assert not np.intersect1d(self.h.d['groups'][a],self.h.d['groups'][b]).size
            self.audit.append((salt,tuple(a),tuple(b)))
        return folds
    def design(self,ix,cols):
        hd=self.h;kk,y,w=hd.weighted(ix,ix);st=transform_stats(hd.x[ix]);a=transform(hd.x[kk],st,cols)
        x,y,w,n=smote(a,y,w,st[3][cols],seed(ix,912));return st,x,y,w,n
    def select(self,ix):
        key=tuple(ix)
        if key in self.select_cache:return self.select_cache[key]
        if sum(self.h.y[ix]==1)<4:
            cols=np.arange(74);self.log('IPSO_support_fallback',ix)
        else:
            splits=self.split(ix,2,512);data=[]
            for a,b in splits:
                ranks=ig(self.h,a);st,x,y,w,ns=self.design(a,ranks);v,yy,ww=self.h.weighted(a,b)
                data.append((a,x,y,w,ns,transform(self.h.x[v],st,ranks),yy,ww))
            rng=np.random.default_rng(seed(ix,513));cache={};best=None;value=-np.inf;stale=0;resets=0
            def init():
                x=rng.uniform(0,1,(4,24));return x,rng.uniform(-1,1,x.shape),np.full(4,.8),x.copy(),np.full(4,-np.inf)
            x,vel,iw,local,score=init()
            for step in range(4):
                old=value;f=[]
                for i,vec in enumerate(x):
                    mask=mask_of(vec);k=tuple(mask)
                    if k not in cache:
                        ids=np.r_[np.flatnonzero(mask),24+np.flatnonzero(mask)];vv=[]
                        for a,xx,yy,ww,ns,v,yv,wv in data:
                            model=RandomForestClassifier(n_estimators=12,max_depth=3,n_jobs=1,random_state=seed(a,321));model.fit(xx[:,ids],yy,sample_weight=ww)
                            vv.append(ap(yv,predict(model,v[:,ids]),wv));self.log('IPSO_proxy_fit',a,ns,validation_rows=len(yv),validation_pos=int(yv.sum()),trees=12,score=vv[-1])
                        cache[k]=float(np.mean(vv))
                    fit=cache[k];f.append(fit)
                    if fit>score[i]:local[i]=vec;score[i]=fit
                    if fit>value or (fit==value and sum(mask)<sum(mask_of(best))):best=vec.copy();value=fit
                stale=stale+1 if value==old else 0
                if step<3 and stale>=1 and not resets:
                    resets+=1;stale=0;x,vel,iw,local,score=init();continue
                iw=inertia(iw,np.array(f));vel=np.clip(iw[:,None]*vel+1.5*rng.random(x.shape)*(local-x)+1.5*rng.random(x.shape)*(best-x),-4,4);x=np.clip(x+vel,0,1)
            cols=ig(self.h,ix)[mask_of(best)];self.log('IPSO_search',ix,particles=4,generations=4,unique_masks=len(cache),resets=resets,score=value,chosen=len(cols))
        self.select_cache[key]=cols;return cols
    def tuned_base(self,ix):
        key=tuple(ix)
        if key in self.base_cache:return self.base_cache[key]
        hd=self.h;folds=self.split(ix,2,602) if sum(hd.y[ix]==1)>=4 else [];data=[]
        for a,b in folds:
            cols=self.select(a);st,x,y,w,ns=self.design(a,cols);v,yy,ww=hd.weighted(a,b);data.append((a,x,y,w,ns,transform(hd.x[v],st,cols),yy,ww))
        par=[]
        for i in range(5):
            if not folds:par.append(1);continue
            def loss(j):
                j=int(j);scores=[]
                for a,xx,yy,ww,ns,v,yv,wv in data:
                    model=learner(i,j,seed(a,i));model.fit(xx,yy,sample_weight=ww);scores.append(f1(yv,predict(model,v),wv));self.log('Hyperopt_fit',a,ns,learner=LEARNERS[i],setting=j,validation_rows=len(yv),validation_pos=int(yv.sum()),score=scores[-1],trees=24)
                return {'loss':-float(np.mean(scores)),'status':STATUS_OK}
            trial=Trials();fmin(loss,hp.choice('j',[0,1,2]),algo=partial(tpe.suggest,n_startup_jobs=1),max_evals=3,trials=trial,rstate=np.random.default_rng(seed(ix,700+i)),show_progressbar=False,verbose=False)
            j=int(trial.best_trial['misc']['vals']['j'][0]);par.append(j)
        if not folds:self.log('Hyperopt_support_fallback',ix)
        cols=self.select(ix);st,x,y,w,ns=self.design(ix,cols);models=[]
        for i,j in enumerate(par):
            est=learner(i,j,seed(ix,i));est.fit(x,y,sample_weight=w);models.append(est);self.log('base_refit',ix,ns,learner=LEARNERS[i],setting=j,trees=24)
        model=dict(cols=cols,st=st,models=models,settings=par);self.base_cache[key]=model
        return model
    def ga(self,ix,raw,folds):
        hd=self.h;cache={};rng=np.random.default_rng(seed(ix,754));data=[]
        for a,b in folds:
            inner=self.split(a,2,753)
            if not inner:
                self.log('GA_support_fallback',ix);return np.ones(5,bool)
            local=np.full((len(a),5),np.nan);rates=[]
            for aa,bb in inner:
                base=self.tuned_base(aa);local[np.searchsorted(a,bb)]=predict_base(base,hd.x[bb])
                valid,yv,wv=hd.weighted(aa,bb)
                rates.append([f1(yv,local[np.searchsorted(a,valid),i],wv) for i in range(5)])
            assert np.isfinite(local).all()
            train,y,w=hd.weighted(a,a);valid,yv,wv=hd.weighted(a,b)
            data.append((a,local[np.searchsorted(a,train)],y,w,raw[np.searchsorted(ix,valid)],yv,wv,np.mean(rates,axis=0)))
        def fitness(mask):
            kk=tuple(mask)
            if kk not in cache:
                scores=[]
                for a,x,y,w,v,yv,wv,f in data:
                    W=fweights(f,mask);est=lr(x*W,y,w);scores.append(f1(yv,predict(est,v*W),wv));self.log('GA_meta_fit',a,score=scores[-1],validation_rows=len(yv),validation_pos=int(yv.sum()))
                cache[kk]=float(np.mean(scores))
            return cache[kk]
        population=[np.ones(5,bool)]
        while len(population)<6:
            m=rng.random(5)<.7
            if feasible(m):population.append(m)
        for gen in range(3):
            ranked=sorted(population,key=lambda m:(-fitness(m),sum(m),tuple(m)));elite=ranked[:3]
            if gen==2:break
            population=elite.copy()
            while len(population)<6:
                a,b=rng.choice(3,2,replace=False);x=rng.integers(1,5);m=np.r_[elite[a][:x],elite[b][x:]].copy();m[rng.integers(5)]^=True
                if feasible(m):population.append(m)
        mask=np.array(min(cache,key=lambda k:(-cache[k],sum(k),k)));self.log('GA_search',ix,chromosomes=len(cache),selected=int(sum(mask)),score=cache[tuple(mask)])
        return mask
    def plan(self,ix):
        hd=self.h;folds=self.split(ix,2,681);raw=np.full((len(ix),5),np.nan);fold_ids=np.full(len(ix),-1);fs=[]
        for j,(a,b) in enumerate(folds):
            model=self.tuned_base(a);idx=np.searchsorted(ix,b);raw[idx]=predict_base(model,hd.x[b]);fold_ids[idx]=j
            v,y,w=hd.weighted(a,b);fs.append([f1(y,raw[np.searchsorted(ix,v),i],w) for i in range(5)])
        if folds:
            assert np.isfinite(raw).all() and (fold_ids>=0).all()
            f=np.mean(fs,axis=0);mask=self.ga(ix,raw,folds);weight=fweights(f,mask)
        else:
            f=np.zeros(5);mask=np.ones(5,bool);weight=np.full(5,.2);self.log('GA_support_fallback',ix)
        full=self.tuned_base(ix);full.update(mask=mask,weight=weight,fold_F1=fs);self.log('plan',ix,included=int(mask.sum()),uniform_fallback=int(sum(f*mask)==0));return full
def predict_base(bundle,x):
    xx=transform(x,bundle['st'],bundle['cols']);return np.column_stack([predict(m,xx) for m in bundle['models']])
def execute(hd,test,out):
    ix=np.arange(len(hd.x));s=Search(hd);oof=np.full((len(ix),5),np.nan);t=time.perf_counter();planlogs=[]
    for j,(a,b) in enumerate(s.split(ix,3,741)):
        plan=s.plan(a);oof[b]=predict_base(plan,hd.x[b])*plan['weight'];planlogs.append(dict(role='meta_oof',fold=j,n=len(a),cols=plan['cols'].tolist(),learners=plan['mask'].tolist(),weights=plan['weight'].tolist(),parameters=plan['settings']))
    assert np.isfinite(oof).all()
    known,y,w=hd.weighted(ix,ix);meta=lr(oof[known],y,w);s.log('final_meta_fit',ix);full=s.plan(ix)
    planlogs.append(dict(role='deployment',fold=-1,n=len(ix),cols=full['cols'].tolist(),learners=full['mask'].tolist(),weights=full['weight'].tolist(),parameters=full['settings']))
    ft=time.perf_counter()-t;file=out/'stack.pkl'
    with file.open('wb') as f:pickle.dump((full,meta),f)
    start=time.perf_counter();p=predict(meta,predict_base(full,test)*full['weight']);pt=time.perf_counter()-start
    with file.open('rb') as f:re,rm=pickle.load(f)
    np.testing.assert_array_equal(p,predict(rm,predict_base(re,test)*re['weight']))
    with (out/'fold_indices.pkl').open('wb') as f:pickle.dump(s.audit,f)
    return p,s.logs,dict(fit_seconds=ft,predict_seconds=pt,model_bytes=file.stat().st_size,selected=full['cols'].tolist(),plans=planlogs,reload_error=0.,disjoint_splits=len(s.audit))
