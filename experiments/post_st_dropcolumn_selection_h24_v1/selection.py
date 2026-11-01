"""Frozen H24 objective and guards, independent of outer outcome files."""
from pathlib import Path
import itertools,json
import numpy as np
from experiments.post_st_dropcolumn_selection_v1.common import (
    TIMES,SEEDS,arrays,load,metrics,evaluation_context,horizon_predictions,subset_hash)

SELECTION_SEED=2026110124

def gate24(rows):
    informative=[r for r in rows if r['horizon']==24 and r['mature'] and r['supported']
                 and r['train_termination']>=1 and r['termination']>=3 and r['controls']>=5]
    count=sum(r['termination'] for r in informative)
    return dict(passed=len(informative)>=2 and count>=10,cohorts=[r['cohort'] for r in informative],
                informative_cohorts=len(informative),target_events=count)

def eligible(rows,h):
    return [r['cohort'] for r in rows if r['horizon']==h and r['mature'] and r['supported']
            and r['train_termination']>=1 and r['termination']>=3 and r['controls']>=5]

def constraints(d):
    limits=dict(delta_ap24=(.005,None),delta_capture24=(0,None),ap24_lower90=(-.01,None),
        delta_auc12=(-.02,None),delta_ap12=(-.02,None),delta_capture12=(-1,None),
        delta_brier12=(None,.002),delta_brier24=(None,.002),
        recovery_delta_brier12=(None,.005),recovery_delta_brier24=(None,.005),
        seed_gain_fraction=(2/3,None),pair_jaccard_change=(-.1,None),spatial_cell_change=(None,0))
    failed=[]
    for k,(lo,hi) in limits.items():
        v=d[k]
        if not np.isfinite(v) or (lo is not None and v<lo-1e-12) or (hi is not None and v>hi+1e-12):failed.append(k)
    return failed

def initial_units(parent,outer):
    sets=load(parent/'input/subsets.json');arms=load(parent/'input/arms.json');units=set()
    for a in arms:
        if a['kind']=='single' or (a['kind']=='group' and a['outer']==outer):
            s=sets[a['subset_id']]
            if s['mode']!='count':continue
            removed=tuple(sorted(set(range(83))-set(s['keep'])))
            if removed:units.add(removed)
    return sorted(units)

def bootstrap_ap_difference(lab,a,b,counts,outer,year,repeat):
    """Common 500 company draws, retaining any outer-resample multiplicity."""
    y,w,_,supported,_=evaluation_context(lab,24,1)
    if not supported:return np.full(500,np.nan)
    active=np.flatnonzero(counts>0);rng=np.random.default_rng(np.random.SeedSequence([SELECTION_SEED,outer,year,repeat+1]))
    draws=rng.integers(len(active),size=(500,len(active)))
    resampled=np.stack([np.bincount(ix,minlength=len(active)) for ix in draws])
    ww=resampled*(w*counts)[active];yy=y[active]
    result=[]
    for risk in [a[active],b[active]]:
        order=np.argsort(-risk,kind='stable');p=risk[order];weight=ww[:,order];hits=weight*yy[order]
        tp=np.cumsum(hits,1);total=np.cumsum(weight,1);end=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1]
        precision=np.divide(tp[:,end],total[:,end],out=np.zeros_like(tp[:,end]),where=total[:,end]>0)
        pos=hits.sum(1)
        result.append(np.divide((np.diff(np.c_[np.zeros(500),tp[:,end]],axis=1)*precision).sum(1),pos,out=np.full(500,np.nan),where=pos>0))
    return result[1]-result[0]

class Evaluator:
    """store.get(keep,year) supplies seed/ensemble predictions and complexity."""
    def __init__(self,parent,outer,store,support,counts=None,repeat=-1):
        self.parent,self.outer,self.store,self.support,self.repeat=parent,outer,store,support,repeat
        self.cohorts={h:eligible(support,h) for h in [12,24]};self.labels={}
        for year in set(self.cohorts[12]+self.cohorts[24]):self.labels[year]=arrays(parent/f'input/folds/{year}/labels_{outer}_PRIVATE.npz')
        self.counts={v:np.ones(len(l['event']),int) if counts is None else counts[v] for v,l in self.labels.items()}
        self.cache={}

    def summary(self,keep):
        keep=tuple(keep)
        if keep in self.cache:return self.cache[keep]
        output={};values={};sizes=[];jaccards=[]
        for h in [12,24]:
            metrics_by_cause={c:[] for c in [1,2]};seed_ap=[[] for _ in SEEDS]
            for v in self.cohorts[h]:
                s=self.store.get(keep,v);lab=self.labels[v];ix=np.repeat(np.arange(len(lab['event'])),self.counts[v]);ll={k:a[ix] for k,a in lab.items()}
                risk=horizon_predictions(s['ensemble'],TIMES,lab[f'h{h}']);values[(v,h)]=risk[:,0]
                for c in [1,2]:metrics_by_cause[c].append(metrics(ll,risk[ix,c-1],h,c))
                for i,p in enumerate(s['seeds']):seed_ap[i].append(metrics(ll,horizon_predictions(p,TIMES,lab[f'h{h}'])[ix,0],h,1)['ap'])
                if h==24:sizes.append(s['cells']);jaccards.append(s['pair_jaccard'])
            for c,rr in metrics_by_cause.items():
                for metric in ['auc','ap','brier','captured']:
                    vv=[r[metric] for r in rr]
                    output[(h,c,metric)]=(float(np.sum(vv)) if metric=='captured' else float(np.mean(vv))) if vv else np.nan
            output[(h,'seed_ap')]=np.array([np.mean(x) if x else np.nan for x in seed_ap])
        result=dict(metrics=output,risks=values,cells=float(np.mean(sizes)) if sizes else np.nan,pair_jaccard=float(np.mean(jaccards)) if jaccards else np.nan)
        self.cache[keep]=result;return result

    def prefetch(self,keeps):
        self.store.ensure(keeps,sorted(set(self.cohorts[12]+self.cohorts[24])))

    def contrast(self,current,keep):
        a,b=self.summary(current),self.summary(keep);d=dict(keep=list(keep),subset_id=subset_hash(keep),features=len(keep))
        for h in [12,24]:
            for metric in ['auc','ap','brier','captured']:
                name='capture' if metric=='captured' else metric
                d[f'delta_{name}{h}']=b['metrics'][(h,1,metric)]-a['metrics'][(h,1,metric)]
            d[f'recovery_delta_brier{h}']=b['metrics'][(h,2,'brier')]-a['metrics'][(h,2,'brier')]
        samples=[bootstrap_ap_difference(self.labels[v],a['risks'][(v,24)],b['risks'][(v,24)],self.counts[v],self.outer,v,self.repeat) for v in self.cohorts[24]]
        boot=np.mean(samples,axis=0);finite=boot[np.isfinite(boot)]
        d['ap24_lower90']=float(np.quantile(finite,.05)) if len(finite) else np.nan;d['ap24_upper90']=float(np.quantile(finite,.95)) if len(finite) else np.nan
        d['undefined_bootstraps']=500-len(finite)
        gains=b['metrics'][(24,'seed_ap')]-a['metrics'][(24,'seed_ap')]
        d['seed_gain_fraction']=float(np.mean(gains>1e-12)) if np.isfinite(gains).all() else np.nan
        d['spatial_cell_change']=b['cells']-a['cells'];d['pair_jaccard_change']=b['pair_jaccard']-a['pair_jaccard']
        d['failed_guards']=constraints(d);d['approved']=not d['failed_guards'];return d

def select(evaluator,units,gate,groups=()):
    current=tuple(range(83));path=[];scans=[]
    if not gate['passed']:return current,[dict(round=0,action='STOP',reason='INSUFFICIENT_H24_SUPPORT',features=83)],scans
    if not evaluator.cohorts[12]:return current,[dict(round=0,action='STOP',reason='REQUIRED_H12_GUARDS_UNDEFINED',features=83)],scans
    evaluator.prefetch([current]+[tuple(j for j in current if j not in u) for u in units])
    initial=[evaluator.contrast(current,tuple(j for j in current if j not in u)) for u in units]
    for r in initial:r.update(round=0,phase='initial')
    scans.extend(initial)
    allowed=[r for r in initial if not(set(range(76,83))-set(r['keep'])) or not(set(range(76,83))&set(r['keep']))]
    ranked=sorted(allowed,key=lambda r:(-r['delta_ap24'] if np.isfinite(r['delta_ap24']) else np.inf,r['subset_id']))[:12]
    pool=[tuple(sorted(set(current)-set(r['keep']))) for r in ranked]
    for step in range(12):
        proposals=sorted({tuple(j for j in current if j not in u) for u in pool if set(u)<=set(current)}-{current,()})
        if not proposals:path.append(dict(round=step,action='STOP',reason='EMPTY_POOL',features=len(current)));break
        evaluator.prefetch([current]+proposals)
        rows=[r for r in initial if tuple(r['keep']) in proposals] if step==0 else [evaluator.contrast(current,p) for p in proposals]
        if step:
            for r in rows:r.update(round=step,phase='recursive')
            scans.extend(rows)
        valid=[r for r in rows if r['approved']]
        if not valid:path.append(dict(round=step,action='STOP',reason='NO_CANDIDATE_PASSES_GUARDS',features=len(current)));break
        winner=max(valid,key=lambda r:(r['delta_ap24'],r['delta_capture24'],-r['spatial_cell_change'],r['subset_id']))
        removed=sorted(set(current)-set(winner['keep']));current=tuple(winner['keep'])
        path.append(dict(round=step,action='DELETE',removed=removed,features=len(current),evidence=winner))
    if len(current)<83:
        removed=set(range(83))-set(current);initial_pool=set().union(*map(set,pool));repairs=set()
        for j in sorted(removed&initial_pool):
            restore=set(range(76,83)) if j>=76 else {j}
            repairs.add(tuple(sorted(set(current)|restore)))
            for group in groups:
                if j in group:
                    for old in sorted(set(group)&set(current)):
                        if j<76 and old<76:repairs.add(tuple(sorted((set(current)-{old})|{j})))
        proposals=sorted(repairs,key=lambda p:(tuple(sorted(set(p)-set(current))),p))[:12]
        evaluator.prefetch([current]+proposals)
        rows=[evaluator.contrast(current,p) for p in proposals]
        for r in rows:r.update(round=len(path),phase='repair')
        scans.extend(rows);valid=[r for r in rows if r['approved']]
        if valid:
            winner=max(valid,key=lambda r:(r['delta_ap24'],r['delta_capture24'],-r['spatial_cell_change'],r['subset_id']))
            current=tuple(winner['keep']);path.append(dict(round=len(path),action='REPAIR',features=len(current),evidence=winner))
        else:path.append(dict(round=len(path),action='STOP_REPAIR',reason='NO_REPAIR_PASSES_GUARDS',features=len(current)))
    return current,path,scans
