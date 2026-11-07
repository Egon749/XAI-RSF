"""Finite historical-only selection; receives no outer labels."""
import itertools, time
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score
from experiments.post_st_eswa_baselines_v1.common import Horizon
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from .common import HERE, SEEDS, read

def candidates():
    g=read(HERE/'FROZEN_PROTOCOL.json')['grid']
    return [dict(max_depth=d, num_boost_round=r, **{'lambda':l})
            for d,r,l in itertools.product(g['max_depth'],g['num_boost_round'],g['lambda'])]

def fit(train, horizon, candidate, seed):
    hd=Horizon(train,horizon,'IPCW');ix=np.arange(len(hd.x));known,y,w=hd.weighted(ix,ix)
    if len(np.unique(y)) != 2: raise ValueError('Fitting labels must have both classes')
    if np.isinf(hd.x).any(): raise ValueError('Infinite input')
    params=read(HERE/'FROZEN_PROTOCOL.json')['fixed_parameters'].copy()
    params.update({k:v for k,v in candidate.items() if k!='num_boost_round'},seed=seed)
    start=time.perf_counter()
    model=xgb.train(params,xgb.DMatrix(hd.x[known],label=y,weight=w,missing=np.nan,nthread=1),num_boost_round=candidate['num_boost_round'])
    return model,dict(known=len(known),positive=int(sum(y)),unknown=len(ix)-len(known),weight_max=float(w.max()),fit_seconds=time.perf_counter()-start)

def predict(model,x):
    if np.isinf(x).any(): raise ValueError('Infinite input')
    p=model.predict(xgb.DMatrix(x,missing=np.nan,nthread=1)).astype(float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p

def metrics(validation, horizon, p):
    y,w,_,supported,reason=evaluation_context(validation,horizon,1)
    keep=w>0
    ap=float(average_precision_score(y[keep],p[keep],sample_weight=w[keep])) if len(np.unique(y[keep]))==2 else None
    brier=float(np.mean(w*(p-y)**2)) if supported else None
    return dict(ap=ap,brier=brier,known=int(sum(keep)),positive=int(sum(y[keep])),supported=bool(supported),reason=reason)

def choose(rows):
    usable=[r for r in rows if r['mean_ap'] is not None]
    objective='AP'
    if usable:
        best=max(r['mean_ap'] for r in usable)
        usable=[r for r in usable if r['mean_ap']>=best-1e-12]
    else:
        usable=[r for r in rows if r['mean_brier'] is not None];objective='Brier fallback'
    if not usable:return dict(max_depth=2,num_boost_round=96,**{'lambda':10.}), 'No supported fold: fixed fallback'
    bb=min(r['mean_brier'] for r in usable)
    usable=[r for r in usable if r['mean_brier']<=bb+1e-12]
    best=min(usable,key=lambda r:(r['num_boost_round'],r['max_depth'],-r['lambda']))
    return {k:best[k] for k in ('max_depth','num_boost_round','lambda')},objective

def select(folds,horizon,out):
    rows=[];log=[];count=0
    for ci,c in enumerate(candidates()):
        aps=[];briers=[]
        for vy,train,val in folds:
            hd=Horizon(train,horizon,'IPCW')
            if len(np.unique(hd.y[hd.y>=0]))<2:
                log.append(dict(candidate=ci,validation_year=vy,**c,skipped='single fitting class'));continue
            ps=[]
            for seed in SEEDS:
                model,info=fit(train,horizon,c,seed);count+=1
                model.save_model(out/f'tune_{ci}_{vy}_{seed}.ubj')
                ps.append(predict(model,val['X']))
                log.append(dict(candidate=ci,validation_year=vy,seed=seed,**c,**info,skipped=''))
            m=metrics(val,horizon,np.mean(ps,axis=0))
            log.append(dict(candidate=ci,validation_year=vy,seed='ensemble',**c,**m,skipped=''))
            if m['supported'] and m['ap'] is not None: aps.append(m['ap'])
            if m['brier'] is not None:briers.append(m['brier'])
        rows.append(dict(candidate=ci,**c,mean_ap=float(np.mean(aps)) if aps else None,
            mean_brier=float(np.mean(briers)) if briers else None,ap_folds=len(aps),brier_folds=len(briers)))
    chosen,rule=choose(rows)
    return chosen,rule,rows,log,count
