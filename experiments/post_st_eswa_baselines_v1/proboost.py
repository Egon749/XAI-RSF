import itertools,time
from scipy.special import expit
import xgboost as xgb
from .common import *
GRID=list(itertools.product([(2.,1.),(5.,.2)],[.25,.75],[2,3]))
T=48
def beta_at(t,n,start,end):
    if not (n>=2 and 0<=t<n and start>=end>0):raise ValueError('invalid schedule')
    return start-(start-end)*t/(n-1)
def objective(z,y,w,beta,c):
    a=beta*(z-c);p=expit(a)
    return w*(np.logaddexp(0.,a)-y*a),w*beta*(p-y),w*beta**2*p*expit(-a)
def fit(hd,ix,cfg):
    (start,end),c,depth=cfg;st=transform_stats(hd.x[ix]);cols=np.arange(74);known,y,w=hd.weighted(ix,ix)
    assert len(np.unique(y))==2
    dm=xgb.DMatrix(transform(hd.x[known],st,cols),label=y,nthread=1)
    bst=xgb.Booster(dict(booster='gbtree',tree_method='hist',max_depth=depth,learning_rate=.075,reg_lambda=1.,gamma=0.,min_child_weight=1.,subsample=1.,colsample_bytree=1.,base_score=0.,objective='reg:squarederror',nthread=1,seed=811),[dm])
    for it in range(T):
        z=bst.predict(dm,output_margin=True).astype(float);_,g,hs=objective(z,y,w,beta_at(it,T,start,end),c)
        bst.boost(dm,it,grad=g,hess=hs)
    return st,bst
def score(st,bst,x,cfg):
    margin=bst.predict(xgb.DMatrix(transform(x,st,np.arange(74)),nthread=1),output_margin=True).astype(float)
    return expit(cfg[0][1]*(margin-cfg[1]))
def execute(hd,test,out):
    ix=np.arange(len(hd.x));folds=hd.split(ix,3,741);logs=[];means=[]
    for n,cfg in enumerate(GRID):
        vals=[]
        for j,(a,b) in enumerate(folds):
            start=time.perf_counter();st,bst=fit(hd,a,cfg);v,y,w=hd.weighted(a,b);p=score(st,bst,hd.x[v],cfg);vals.append(ap(y,p,w));logs.append(dict(stage='PROBoost_tune',candidate=n,fold=j,fit_rows=sum(hd.y[a]>=0),fit_pos=sum(hd.y[a]==1),val_rows=len(v),val_pos=sum(y),score=vals[-1],trees=T,seconds=time.perf_counter()-start))
        means.append(np.mean(vals))
    ni=int(np.argmax(means));cfg=GRID[ni];t=time.perf_counter();st,bst=fit(hd,ix,cfg);ft=time.perf_counter()-t;bst.save_model(out/'proboost.ubj');np.savez(out/'preprocessing.npz',*st);tt=time.perf_counter();p=score(st,bst,test,cfg);pt=time.perf_counter()-tt
    other=xgb.Booster();other.load_model(out/'proboost.ubj');other.set_param({'nthread':1});np.testing.assert_array_equal(score(st,other,test,cfg),p)
    logs.append(dict(stage='PROBoost_final',candidate=ni,fold=-1,fit_rows=sum(hd.y>=0),fit_pos=sum(hd.y==1),trees=T,seconds=ft))
    return p,logs,dict(cfg=cfg,trees=T,fit_seconds=ft,predict_seconds=pt,model_bytes=(out/'proboost.ubj').stat().st_size+(out/'preprocessing.npz').stat().st_size,selected=list(range(74)),reload_error=0.)
