"""Chronologically honest selection gates and bounded, interaction-aware candidates."""
import numpy as np
import pandas as pd
from experiments.post_st_feature_permutation_v1.common import groups_from_training
FULL=tuple(range(83))
AUDIT=set(range(76,83))

def support_at(data, meta, cutoff, groups=None):
    """Only outcomes recorded before cutoff may count; duplicates never inflate events."""
    cap=pd.Timestamp(f'{cutoff}-01-01')
    active=np.ones(len(meta),bool) if groups is None else np.isin(data['groups'],np.unique(groups))
    rows=[]
    for cohort in range(2019,cutoff):
        val=active & meta.origin.dt.year.eq(cohort).to_numpy()
        if not val.any(): continue
        train=active & meta.origin.lt(pd.Timestamp(f'{cohort}-01-01')).to_numpy()
        train_event=(data['event']==1) & meta.stop.lt(pd.Timestamp(f'{cohort}-01-01')).to_numpy()
        for h in [12,24]:
            endpoint=meta.origin+pd.DateOffset(months=h)
            observed=meta.stop.lt(cap).to_numpy()
            target=(data['event']==1)&observed&meta.stop.le(endpoint).to_numpy()
            known=((data['event']>0)&observed&meta.stop.le(endpoint).to_numpy()) | endpoint.lt(cap).to_numpy()
            mature=bool(endpoint[val].lt(cap).all())
            rows.append(dict(cutoff=cutoff,cohort=cohort,horizon=h,companies=int(val.sum()),
                training_targets=int((train&train_event).sum()),targets=int((val&target).sum()),
                controls=int((val&known&~target).sum()),unknown=int((val&~known).sum()),mature=mature))
    return rows

def informative(rows,h=24):
    return [r for r in rows if r['horizon']==h and r['mature'] and r['unknown']==0
            and r['training_targets']>=1 and r['targets']>=3 and r['controls']>=5]

def discovery_gate(rows):
    rr=informative(rows)
    return dict(passed=len(rr)>=2 and sum(r['targets'] for r in rr)>=10,
                cohorts=[r['cohort'] for r in rr],targets=sum(r['targets'] for r in rr))

def time_gate(data,meta,cutoff,groups=None):
    history=support_at(data,meta,cutoff,groups)
    discovery=discovery_gate(history)
    confirmations=[]
    # A confirmation forecast must have selected features BEFORE its cohort starts.
    # Its selector may not use labels that only mature by today's outer cutoff.
    for row in informative(history):
        inner=discovery_gate(support_at(data,meta,row['cohort'],groups))
        confirmations.append(dict(cohort=row['cohort'],targets=row['targets'],
                                  selector_cutoff=row['cohort'],inner=inner,eligible=inner['passed']))
    return dict(discovery=discovery,confirmations=confirmations,
                passed=discovery['passed'] and any(c['eligible'] for c in confirmations),
                reason='PASS' if discovery['passed'] and any(c['eligible'] for c in confirmations)
                else 'INSUFFICIENT_DISCOVERY_SUPPORT' if not discovery['passed'] else 'NO_HONEST_FORWARD_CONFIRMATION')

def grouped_candidates(data,meta,cutoff,groups=None):
    """Recompute stable correlations per resample; no labels choose groups."""
    mult=np.ones(len(meta),int) if groups is None else np.array([np.sum(groups==g) for g in data['groups']])
    windows=list(range(2019,cutoff))[-3:]; seen={}
    for year in windows:
        ix=np.flatnonzero((mult>0)&meta.origin.lt(pd.Timestamp(f'{year}-01-01')).to_numpy())
        # Require independent-company overlap, even in resampled correlations.
        x=data['X'][np.repeat(ix,mult[ix])]
        _,edges,_=groups_from_training(x)
        for j,k,rho,_ in edges:
            n=int((np.isfinite(data['X'][ix,j])&np.isfinite(data['X'][ix,k])).sum())
            if j not in AUDIT and k not in AUDIT and n>=30:seen.setdefault((j,k),[]).append(year)
    edges=[pair for pair,ys in seen.items() if len(ys)==3]
    parent=list(range(83))
    def find(j):
        while parent[j]!=j: j=parent[j]
        return j
    for j,k in edges:parent[find(k)]=find(j)
    components={}
    for j in range(76):components.setdefault(find(j),[]).append(j)
    groups_out=[];candidates={FULL}
    ix=np.repeat(np.arange(len(meta)),mult)
    X=data['X'][ix]
    for members in components.values():
        if len(members)<2:continue
        # A connected component is not automatically a redundant clique.
        if any(tuple(sorted((j,k))) not in edges for j in members for k in members if j<k):continue
        first=min(members,key=lambda j:(-float(np.isfinite(X[:,j]).mean()),j))
        corr=pd.DataFrame(X[:,members]).corr(method='spearman',min_periods=30).to_numpy()
        second=min([j for j in members if j!=first],key=lambda j:(abs(corr[members.index(first),members.index(j)]),-float(np.isfinite(X[:,j]).mean()),j))
        groups_out.append(dict(members=members,one=[first],two=sorted([first,second])))
        for retain in [[first],sorted([first,second])]:
            candidates.add(tuple(j for j in FULL if j not in members or j in retain))
    return groups_out,sorted(candidates)

def performance_guards(d):
    limits={'delta_auc24':(-.02,None),'delta_ap24':(-.02,None),'delta_capture24':(-1,None),
        'delta_auc12':(-.02,None),'delta_ap12':(-.02,None),'delta_capture12':(-1,None),
        'delta_brier12':(None,.002),'delta_brier24':(None,.002),
        'recovery_delta_brier12':(None,.005),'recovery_delta_brier24':(None,.005)}
    failed=[k for k,(lo,hi) in limits.items() if not np.isfinite(d[k]) or (lo is not None and d[k]<lo) or (hi is not None and d[k]>hi)]
    if d['spatial_cell_change']>=0:failed.append('NO_COMPLEXITY_REDUCTION')
    return failed

def interaction_guard(base,new):
    """Same reference, group-aggregated pair contributions; a shape check, not causal truth."""
    a=np.asarray(base).ravel();b=np.asarray(new).ravel()
    amp=np.sqrt(np.mean(a*a));newamp=np.sqrt(np.mean(b*b))
    if amp<1e-8:return dict(passed=True,reason='NEGLIGIBLE_BASE_COMPONENT',ratio=None,correlation=None,relative_l2=None)
    ratio=newamp/amp
    corr=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else float('nan')
    distance=float(np.sqrt(np.mean((a-b)**2))/amp)
    return dict(passed=bool(.5<=ratio<=2 and np.isfinite(corr) and corr>=.7 and distance<=.5),
                reason='FUNCTION_SHAPE_AND_AMPLITUDE',ratio=float(ratio),correlation=corr,relative_l2=distance)

def choose(gate,candidates):
    if not gate['passed']:return FULL,gate['reason']
    valid=[c for c in candidates if c['discovery_pass'] and c['confirmation_pass'] and c['interaction_pass'] and c['stability_pass']]
    if not valid:return FULL,'NO_CONFIRMED_SAFE_REDUCTION'
    best=min(valid,key=lambda c:(c['cells'],len(c['keep']),tuple(c['keep'])))
    return tuple(best['keep']),'SELECTED'
