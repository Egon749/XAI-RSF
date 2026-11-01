"""Pre-cutoff, per-period prediction safeguards and stable group protection."""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from crforest import CIFExpansion
from experiments.post_st_group_selection_v2.selection import FULL,grouped_candidates,json_safe,InteractionCheck
from experiments.post_st_dropcolumn_selection_v1.common import arrays,load,TIMES,SEEDS,metrics,horizon_predictions,evaluation_context


def group_guard(base, candidate, seeds, share, event_support):
    a=np.asarray(base)-np.mean(base);b=np.asarray(candidate)-np.mean(candidate)
    ss=np.asarray(seeds);ss=ss-ss.mean(axis=1,keepdims=True)
    rms=float(np.sqrt(np.mean(a*a)));new_rms=float(np.sqrt(np.mean(b*b)))
    correlations=[float(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 and y.std()>1e-12 else 0. for x,y in itertools.combinations(ss,2)]
    stability=float(np.median(correlations));dispersion=float(np.sqrt(np.mean(np.var(ss,axis=0))))
    qualified=bool(event_support>=5 and rms>=.001 and share>=.01 and stability>=.7)
    distance=float(np.sqrt(np.mean((b-a)**2)));tolerance=max(.5*rms,dispersion,.001)
    return dict(qualified=qualified,passed=not qualified or distance<=tolerance+1e-12,
        baseline_rms=rms,candidate_rms=new_rms,absolute_contribution_share=share,training_event_companies=event_support,
        median_seed_correlation=stability,seed_dispersion=dispersion,distance=distance,tolerance=tolerance,
        reason='STABLE_IMPORTANT_GROUP' if qualified else 'NO_QUALIFIED_GROUP_ANCHOR')


def protected_family(X, group, feature_groups):
    """One-hop proxies, same economic family; no label or future-window use."""
    result=set(group)
    for j in range(X.shape[1]):
        for k in group:
            if feature_groups[j]!=feature_groups[k]:continue
            valid=np.isfinite(X[:,j])&np.isfinite(X[:,k])
            if valid.sum()<30:continue
            rho=pd.Series(X[valid,j]).corr(pd.Series(X[valid,k]),method='spearman')
            if np.isfinite(rho) and abs(rho)>=.8:result.add(j)
    return sorted(result)


def contributions(ex,X,keep,group):
    total=np.zeros((len(X),len(TIMES),2));main=total.copy();pair=total.copy();denom=total.copy()
    for key in [(j,) for j in ex.mains]+ex.pairs:
        value=ex.component(X[:,keep],key,TIMES);denom+=abs(value)
        if set(keep[j] for j in key)&set(group):
            total+=value
            if len(key)==1:main+=value
            else:pair+=value
    return total,main,pair,denom


class GroupCheck:
    def __init__(self,parent,store,cohorts,groups,draw=None):
        self.parent,self.store,self.cohorts,self.groups,self.draw=parent,store,cohorts,groups,draw
        self.feature_groups=[r['feature_group'] for r in load(parent/'input/features.json')];self.cache={}
    def __call__(self,keep):
        removed=set(FULL)-set(keep);group=next(g['members'] for g in self.groups if removed<=set(g['members']))
        rows=[]
        for cohort in self.cohorts:
            key=(cohort,tuple(group))
            data=arrays(self.parent/f'input/folds/{cohort}/train_PRIVATE.npz');X=data['X']
            active=np.ones(len(X),bool) if self.draw is None else np.isin(data['groups'],np.unique(self.draw))
            if key not in self.cache:
                H=protected_family(X[active],group,self.feature_groups)
                base_path=self.store.path(FULL,cohort)
                ex=CIFExpansion.load(base_path/'ensemble/functions_PRIVATE.npz');base=contributions(ex,X,FULL,H)
                seeds=[contributions(CIFExpansion.load(base_path/f'{s}/shrunk/functions_PRIVATE.npz'),X,FULL,H)[0] for s in SEEDS]
                self.cache[key]=H,base,seeds
            H,base,seeds=self.cache[key];new=contributions(CIFExpansion.load(self.store.path(keep,cohort)/'ensemble/functions_PRIVATE.npz'),X,keep,H)
            for h in [12,24]:
                ix=np.searchsorted(TIMES,data[f'h{h}']);rr=np.arange(len(X))
                for c in [1,2]:
                    a=base[0][rr,ix,c-1];b=new[0][rr,ix,c-1]
                    denominator=float(base[3][rr,ix,c-1].mean());share=float(abs(a).mean()/denominator) if denominator>0 else 0.
                    support=len(np.unique(data['groups'][active&(data['event']==c)]))
                    rows.append(dict(cohort=cohort,group=group,protected_family=H,horizon=h,cause=c,
                        baseline_main_rms=float(np.sqrt(np.mean(base[1][rr,ix,c-1]**2))),candidate_main_rms=float(np.sqrt(np.mean(new[1][rr,ix,c-1]**2))),
                        baseline_pair_rms=float(np.sqrt(np.mean(base[2][rr,ix,c-1]**2))),candidate_pair_rms=float(np.sqrt(np.mean(new[2][rr,ix,c-1]**2))),
                        **group_guard(a,b,[z[rr,ix,c-1] for z in seeds],share,support)))
        return rows


class PeriodEvaluator:
    def __init__(self,parent,outer,store,support,counts):
        self.parent,self.outer,self.store,self.counts=parent,outer,store,counts
        self.cohorts={h:sorted(r['cohort'] for r in support if r['horizon']==h and r['mature'] and r['supported'] and r['train_termination']>=1) for h in [12,24]}
        self.cache={}
    def summary(self,keep):
        keep=tuple(keep)
        if keep in self.cache:return self.cache[keep]
        rows=[];sizes=[]
        for h in [12,24]:
            for y in self.cohorts[h]:
                lab=arrays(self.parent/f'input/folds/{y}/labels_{self.outer}_PRIVATE.npz');ix=np.repeat(np.arange(len(lab['event'])),self.counts[y])
                if not len(ix):continue
                ll={k:v[ix] for k,v in lab.items()};bundle=self.store.get(keep,y)
                risk=horizon_predictions(bundle['ensemble'],TIMES,lab[f'h{h}'])
                active=self.counts[y]>0;unique={k:v[active] for k,v in lab.items()}
                for c in [1,2]:
                    yy,_,known,ok,_=evaluation_context(unique,h,c)
                    rows.append(dict(cohort=y,horizon=h,cause=c,ranking_guard=bool(ok and (yy&known).sum()>=3 and (~yy&known).sum()>=5),**metrics(ll,risk[ix,c-1],h,c)))
                if h==24:sizes.append(bundle['cells'])
        result=dict(periods=rows,cells=float(np.mean(sizes)) if sizes else np.nan);self.cache[keep]=result;return result
    def compare(self,keep):
        a=self.summary(FULL);b=self.summary(keep);rows=[];failed=[]
        for x,y in zip(a['periods'],b['periods']):
            assert (x['cohort'],x['horizon'],x['cause'])==(y['cohort'],y['horizon'],y['cause'])
            row=dict(cohort=x['cohort'],horizon=x['horizon'],cause=x['cause'],ranking_guard=x['ranking_guard'],events=x['target_events'])
            for metric in ['auc','ap','brier','captured']:
                row['baseline_'+metric]=x[metric];row['candidate_'+metric]=y[metric];row['delta_'+metric]=y[metric]-x[metric]
            reasons=[]
            if x['ranking_guard']:
                for metric,limit in [('auc',-.02),('ap',-.02),('captured',-1)]:
                    if not np.isfinite(row['delta_'+metric]) or row['delta_'+metric]<limit-1e-12:reasons.append(metric)
            if not np.isfinite(row['delta_brier']) or row['delta_brier']>(.002 if x['cause']==1 else .005)+1e-12:reasons.append('brier')
            row['failed_guards']=reasons;rows.append(row)
            failed.extend(f"{x['cohort']}_H{x['horizon']}_C{x['cause']}_{v}" for v in reasons)
        if not rows:failed.append('NO_PERIOD_EVIDENCE')
        if not np.isfinite(b['cells']) or b['cells']>=a['cells']:failed.append('NO_COMPLEXITY_REDUCTION')
        return dict(keep=list(keep),cells=b['cells'],spatial_cell_change=b['cells']-a['cells'],performance_failures=failed,periods=rows)


def choose(scans,guard):
    valid=[r for r in scans if not r['performance_failures'] and r[guard]]
    if not valid:return FULL,'NO_CANDIDATE_PASSES_FROZEN_GUARDS'
    row=min(valid,key=lambda r:(r['cells'],len(r['keep']),tuple(r['keep'])))
    return tuple(row['keep']),'EXPLORATORY_SELECTED'
