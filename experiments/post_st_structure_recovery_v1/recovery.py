"""Exact population-product L2 recovery, with independent known truth.

No Monte Carlo masker and no selection from test outcomes. The prediction is
unchanged by recentering; this only aligns learned and known functions.
"""
from collections import defaultdict
import numpy as np
from crforest import HigherOrderCIFExpansion
from .common import *

def population_expansion(forest,training,kind):
    ex=HigherOrderCIFExpansion.from_forest(forest,training)
    coefficients,_=ex._coeff(TIMES)
    ex.grid=TIMES.copy();ex.coefficients=coefficients
    ex.mass=interval_mass(ex.low,ex.high,kind)
    ex.mu=np.einsum('i,itk->tk',ex.mass.prod(1),ex.coefficients)
    return ex

def weights(ex,key,kind):
    ids,axes=ex._terms[key]
    selected=np.zeros((len(ids),ex.max_order),bool);selected[np.arange(len(ids))[:,None],axes]=True
    outside=np.where(selected,1.,ex.mass[ids]).prod(1)
    coeff=ex.coefficients[ids]*outside[:,None,None]
    moment=np.ones(len(ids));gram=np.ones((len(ids),len(ids)))
    for a in range(len(key)):
        lo=ex.low[ids,axes[:,a]];hi=ex.high[ids,axes[:,a]];m=ex.mass[ids,axes[:,a]]
        moment*=interval_first_moment(lo,hi,kind)
        gram*=interval_mass(np.maximum(lo[:,None],lo),np.minimum(hi[:,None],hi),kind)-m[:,None]*m
    norm=np.einsum('ihc,ij,jhc->hc',coeff,gram,coeff,optimize=True)
    assert norm.min()>=-1e-12
    # Only squared norms can acquire negative roundoff; predictions are untouched.
    norm=np.maximum(norm,0.)
    inner=np.einsum('i,ihc->hc',moment,coeff)
    return ids,coeff,moment,norm,inner

def assess(ex,scenario,rep):
    kind=population_kind(scenario);variance=1/3 if kind=='uniform' else 1.
    truth_by_cause={c:truth_terms(scenario,rep,c) for c in [1,2]};scale=time_scale(TIMES)
    keys=set(ex.keys)|set().union(*(set(t) for t in truth_by_cause.values()));keys.discard(())
    records=[];signal=[];totals=np.zeros((2,2));order_energy=np.zeros((5,2,2))
    for key in sorted(keys,key=lambda x:(len(x),x)):
        if key in ex._terms:_,_,_,norm,inner=weights(ex,key,kind)
        else:norm=np.zeros((2,2));inner=np.zeros((2,2))
        order=len(key);factor=variance**order;order_energy[order]+=norm
        for ci,c in enumerate([1,2]):
            theta=truth_by_cause[c].get(key,0.)*scale
            mse=norm[:,ci]+theta**2*factor-2*theta*inner[:,ci]
            assert mse.min()>=-1e-12
            totals[:,ci]+=np.maximum(mse,0.)
            for h,day in enumerate(TIMES):
                row=dict(day=int(day),cause=c,order=order,features=','.join(map(str,key)),true=key in truth_by_cause[c],
                    estimated_rms=float(np.sqrt(norm[h,ci])),true_rms=float(abs(theta[h])*np.sqrt(factor)),
                    function_l2_error=float(np.sqrt(max(mse[h],0))),coefficient=float(inner[h,ci]/factor),true_coefficient=float(theta[h]))
                records.append(row)
                if row['true']:
                    row=dict(row,amplitude_ratio=row['coefficient']/row['true_coefficient'],
                        sign_correct=row['coefficient']*row['true_coefficient']>0,
                        shape_cosine=float(inner[h,ci]*theta[h]/np.sqrt(norm[h,ci]*theta[h]**2*factor)) if norm[h,ci]>1e-24 else np.nan)
                    signal.append(row)
    baseline=np.column_stack([truth_by_cause[c][()]*scale for c in [1,2]])
    totals+=(ex.mu-baseline)**2
    summaries=[]
    for h,day in enumerate(TIMES):
        for ci,c in enumerate([1,2]):
            rr=[r for r in records if r['day']==day and r['cause']==c]
            for threshold in [.0025,.005,.01]:
                for focus in ['all','main','pair','higher']:
                    rows=[r for r in rr if focus=='all' or (focus=='main' and r['order']==1) or (focus=='pair' and r['order']==2) or (focus=='higher' and r['order']>=3)]
                    tp=sum(r['true'] and r['estimated_rms']>=threshold for r in rows)
                    fp=sum(not r['true'] and r['estimated_rms']>=threshold for r in rows)
                    fn=sum(r['true'] and r['estimated_rms']<threshold for r in rows)
                    summaries.append(dict(day=int(day),cause=c,threshold=threshold,focus=focus,tp=tp,fp=fp,fn=fn,
                        precision=tp/(tp+fp) if tp+fp else np.nan,recall=tp/(tp+fn) if tp+fn else np.nan,
                        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                        exact_product_risk_mse=float(totals[h,ci]),baseline=float(ex.mu[h,ci]),true_baseline=float(baseline[h,ci]),
                        higher_order_energy=float(order_energy[3:,h,ci].sum())))
    return records,signal,summaries

def tree_signal_audit(ex,forest,scenario,rep):
    """Identity: forest coefficient = sum of exact individual tree coefficients/T."""
    leaf_tree=np.concatenate([np.full(sum('cif' in n for n in t),i) for i,t in enumerate(forest.trees_)])
    kind=population_kind(scenario);var=1/3 if kind=='uniform' else 1.;rows=[]
    from crforest.partition import tree_feature_subset
    for key,theta in truth_terms(scenario,rep).items():
        if len(key)<2:continue
        values=np.zeros((len(forest.trees_),2));eligible=[]
        for i in range(len(forest.trees_)):
            # Called only on a single-seed forest. Seed streams remain original.
            fs=tree_feature_subset(P,forest.config.tree_feature_count,forest.config.seed,i,forest.config.tree_feature_strategy);eligible.append(set(key).issubset(fs))
        if key in ex._terms:
            ids,coeff,moment,_,inner=weights(ex,key,kind)
            for h in range(2):np.add.at(values[:,h],leaf_tree[ids],coeff[:,h,0]*moment*len(forest.trees_)/(var**len(key)))
            np.testing.assert_allclose(values.mean(0),inner[:,0]/var**len(key),atol=1e-13,rtol=0)
        for h,day in enumerate(TIMES):
            nz=abs(values[:,h])>1e-12
            for i in range(len(values)):rows.append(dict(day=int(day),order=len(key),features=','.join(map(str,key)),tree=i,eligible=eligible[i],coefficient=float(values[i,h]),true_coefficient=float(theta*time_scale([day])[0]),nonzero=bool(nz[i])))
    return rows
