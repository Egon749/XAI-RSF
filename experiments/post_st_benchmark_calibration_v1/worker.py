"""Independent benchmark jobs; outer evaluation labels are not in the job input."""
import sys,time,socket,pickle,resource
import pandas as pd
from .common import *
from .estimators import transform,estimate,regression_predict,knn
from crforest import ForestConfig,CompetingRiskForest,HigherOrderCIFExpansion
from crforest.core import aalen_johansen
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_prognosis_v1.worker import reference_fit
GRIDS={'FG':[.01,.1,1.],'CSC':[.01,.1,1.],'KNN_AJ':[20,50,100],'RF':[5,10,20]}
def probability(p,independent=False):
    assert np.isfinite(p).all() and p.min()>=-1e-12 and p.max()<=1+1e-10 and np.diff(p,axis=1).min()>=-1e-12
    high=p.sum(-1)
    if not independent:assert high.max()<=1+1e-10
    return dict(max_sum=float(high.max()),rows_over_one=int((high>1+1e-10).any(1).sum()))
def conventional(arm,tr,z,param,seed,out):
    out.mkdir(parents=True);start=time.perf_counter();t,e=tr['time'],tr['event'];meta={}
    if arm=='AJ':
        values=aalen_johansen(t,e,DAYS,2);p=np.broadcast_to(values,(len(z),4,2)).copy()
        np.savez_compressed(out/'aj.npz',times=np.unique(t),curve=aalen_johansen(t,e,np.unique(t),2));meta['baseline_curves']=2
    else:
        x,test,prep=transform(tr['X'],z)
        np.savez_compressed(out/'preprocessing.npz',**prep);meta['effective_columns']=x.shape[1]
        if arm=='RF':
            cfg=dict(nodesize=int(param),nodedepth=None,nsplit=10,mtry=max(1,int(np.sqrt(x.shape[1]))))
            v,grid,info=reference_fit(dict(tr,X=x),test,{'standard_rsf':cfg},seed,32,out)
            ix=np.searchsorted(grid,DAYS,side='right')-1;p=v[:,np.maximum(ix,0)].copy();p[:,ix<0]=0;meta.update(leaves=info['leaves'],reload_error=info['reload_error'],mtry=cfg['mtry'])
        elif arm=='KNN_AJ':
            p=knn(x,test,t,e,int(param),DAYS);np.savez_compressed(out/'neighbors.npz',X=x,time=t,event=e,k=param);meta['reference_rows']=len(x)
        else:
            fg=arm=='FG';models=[estimate(x,t,e,k,param,fg) for k in (1,2)]
            with (out/'model.pkl').open('wb') as f:pickle.dump(models,f)
            with (out/'model.pkl').open('rb') as f:p=regression_predict(pickle.load(f),test,DAYS,fg)
            meta.update(link_terms=int(sum(np.count_nonzero(m['beta']) for m in models)),iterations=[m['iterations'] for m in models],max_gradient=max(m['gradient'] for m in models))
    return p,dict(**meta,**probability(p,arm=='FG'),seconds=time.perf_counter()-start)
def run_task(arm,y,seed,root,out):
    folder=root/str(y);tr=arrays(folder/'train.npz');ev=arrays(folder/'eval.npz');assert set(ev)=={'X','sample_ids'}
    start=time.perf_counter();log=[];models=[];risks={};param=None
    if arm=='unrestricted':
        cfg=read(folder/'config.json');cfg.update(seed=seed,path_budget=None);f=CompetingRiskForest(ForestConfig(**cfg)).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1)
        fitsec=time.perf_counter()-start;f.save(out/'raw_forest.json');f=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);f.save(out/'forest.json');f=CompetingRiskForest.load(out/'forest.json')
        xx=np.vstack([tr['X'],ev['X']]);ex=HigherOrderCIFExpansion.from_forest(f,tr['X']);ex.save(out/'functions.npz')
        tt=np.unique(np.r_[f.grid_,DAYS]);err=float(np.max(abs(ex.predict(xx,tt)-f.predict(xx,tt))));assert err<=1e-10
        cx=ex.complexity(tr['X']);del cx['functions'];p=f.predict(ev['X'],DAYS);risks[str(seed)]=p;models.append(dict(seed=seed,seconds=time.perf_counter()-start,forest_fit_seconds=fitsec,exact_error=err,**cx,**probability(p)))
    else:
        if arm in GRIDS:
            scores={}
            for param0 in GRIDS[arm]:
                losses=[]
                for v in [y-3,y-2]:
                    train=arrays(folder/f'{v}_fit.npz');val=arrays(folder/f'{v}_val.npz')
                    if min(sum(train['event']==k) for k in (1,2))==0:continue
                    p,info=conventional(arm,train,val['X'],param0,7301,out/f'tune_{v}_{param0}')
                    pred=endpoints(p,val['h12']);ll=[]
                    for k in (1,2):
                        yy,w,_,ok,_=evaluation_context(val,12,k)
                        if ok:ll.append(float(np.mean(w*(pred[:,k-1]-yy)**2)))
                    log.append(dict(validation_year=v,parameter=param0,brier=ll,**info))
                    if len(ll)==2:losses.append(np.mean(ll))
                if losses:scores[param0]=float(np.mean(losses))
            param=max((v for v,z in scores.items() if z<=min(scores.values())+1e-12),default=GRIDS[arm][1])
        for ss in ((*PICK,*ANCHOR) if arm=='RF' else [0]):
            p,info=conventional(arm,tr,ev['X'],param,ss,out/f'outer_{ss}');risks[str(ss)]=p;models.append(dict(seed=ss,**info))
    np.savez_compressed(out/'predictions.npz',sample_ids=ev['sample_ids'],**risks)
    return dict(arm=arm,year=y,seed=seed,parameter=param,tuning=log,models=models,seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,host=socket.gethostname(),job=os.environ.get('SLURM_JOB_ID'))
if __name__=='__main__':
    arm,y,ss=sys.argv[1:];out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True);write(out/'done.json',run_task(arm,int(y),int(ss),Path(os.environ['DATA_DIR']),out))
