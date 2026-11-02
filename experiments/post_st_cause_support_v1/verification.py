"""Saved-node audit and complete function verification, without fitting trees."""
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion,HigherOrderCIFExpansion
from crforest.core import _split_parent,logrank_cr_score
from crforest.partition import training_cut_dictionary,tree_feature_subset
from crforest.core import cluster_bootstrap
from experiments.post_st_leaf_shallow_v1.audit import direct_aj
from .common import *
from crforest.core import target_support_allows
from experiments.post_st_rule_features_v1.audit_root_weights import independent_components

def replay(f,tr):
    """Replay RNG/candidates at observed nodes. Never grow a replacement forest."""
    x,t,e,g=[tr[k] for k in ['X','time','event','groups']];cfg=f.config
    cuts=training_cut_dictionary(x,cfg);node_rows=[];feature_rows=[];pair_rows=[];leaf_rows=[]
    selected=np.zeros(x.shape[1],int);split_nodes=np.zeros_like(selected);split_trees=np.zeros_like(selected)
    pair_eligible=np.zeros((x.shape[1],x.shape[1]),int);pair_paths=np.zeros_like(pair_eligible)
    total_attempts=0;max_score_error=0.;max_leaf_error=0.;max_parent_error=0.
    for ti,tree in enumerate(f.trees_):
        eligible=tree_feature_subset(x.shape[1],cfg.tree_feature_count,cfg.seed,ti,cfg.tree_feature_strategy)
        selected[eligible]+=1;used_in_tree=set();paths_in_tree=set()
        for j,k in combinations(eligible,2):pair_eligible[j,k]+=1
        def visit(index,rows,used,depth,node_key,parent_curve=None):
            nonlocal total_attempts,max_score_error,max_leaf_error,max_parent_error
            n=tree[index];unique=len(np.unique(g[rows]));support=[int(len(np.unique(g[rows[e[rows]==k]]))) for k in [1,2]]
            assert n['groups']==unique and n['n']==len(rows) and n['cause_company_support']==support
            raw=direct_aj(t[rows],e[rows],f.grid_)
            if 'cif' in n:
                if 'leaf_shrinkage' in n:
                    sh=n['leaf_shrinkage'];alpha=1. if parent_curve is None else unique/(unique+10.)
                    prior=raw if parent_curve is None else parent_curve
                    assert sh['strength']==10 and sh['unique_inbag_companies']==unique and abs(sh['weight']-alpha)<1e-15
                    max_parent_error=max(max_parent_error,float(abs(np.asarray(sh['raw_parent_cif'])-prior).max()))
                    np.testing.assert_allclose(sh['raw_leaf_cif'],raw,atol=1e-13,rtol=0)
                    estimate=alpha*raw+(1-alpha)*prior
                else:alpha=1.;estimate=raw
                max_leaf_error=max(max_leaf_error,float(abs(np.asarray(n['cif'])-estimate).max()))
                leaf_rows.append(dict(tree=ti,node=index,unique_companies=unique,bootstrap_rows=len(rows),target_companies=support[0],recovery_companies=support[1],weight=alpha,borrowed_positive_target=bool(support[0]==0 and np.asarray(n['cif'])[:,0].max()>0)))
                paths_in_tree.update(combinations(sorted(used),2))
            stopping=(depth>=cfg.max_depth or len(rows)<2*cfg.min_leaf or n['events']<2*cfg.min_events or unique<2*cfg.min_groups)
            if not stopping:
                active=eligible if cfg.path_budget is None or len(used)<cfg.path_budget else np.array(sorted(used))
                rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,ti,node_key,1]))
                features=rng.choice(active,min(cfg.mtry or int(np.ceil(np.sqrt(x.shape[1]))),len(active)),replace=False)
                parent=_split_parent(t[rows],e[rows],cfg.n_causes);best=0.;winner=None;unconstrained_best=0.;unconstrained_winner=None
                nr=dict(tree=ti,node=index,depth=depth,features_considered=len(features),finite_cuts=0,missing_only_cuts=0,routed_attempts=0,scored_candidates=0,features_with_fewer_than_five_cuts=0,guard_rejected=0,target_company_support=support[0],recovery_company_support=support[1])
                for j in features:
                    z=x[rows,j];missing=np.isnan(z);vals=np.unique(z[~missing])
                    if cuts is None:
                        cc=vals[:-1]
                        if len(cc)>cfg.n_thresholds:cc=rng.choice(cc,cfg.n_thresholds,replace=False)
                    else:cc=cuts[j][(cuts[j]>=vals[0])&(cuts[j]<vals[-1])] if len(vals) else np.array([])
                    nr['finite_cuts']+=len(cc);nr['features_with_fewer_than_five_cuts']+=len(cc)<cfg.n_thresholds
                    candidates=[(float(c),route) for c in sorted(cc) for route in ([False,True] if missing.any() else [False])]
                    if missing.any() and len(vals):candidates.append((-np.inf,True));nr['missing_only_cuts']+=1
                    for c,ml in candidates:
                        nr['routed_attempts']+=1;left=(z<=c)|(missing&ml)
                        if min(left.sum(),(~left).sum())<cfg.min_leaf:continue
                        if any((e[rows[side]]>0).sum()<cfg.min_events for side in (left,~left)):continue
                        if any(len(np.unique(g[rows[side]]))<cfg.min_groups for side in (left,~left)):continue
                        nr['scored_candidates']+=1;score=logrank_cr_score(left,e[rows],parent,cfg.cause_weights,cfg.split_score)
                        if score>unconstrained_best+1e-14:unconstrained_best=score;unconstrained_winner=(int(j),c,bool(ml))
                        # Independently count target company sets rather than call production guard.
                        counts=[len(set(g[rows[side & (e[rows]==cfg.split_target_cause)]])) for side in [left,~left]]
                        allowed=all(count==0 or count>=cfg.split_min_target_groups for count in counts)
                        assert allowed==target_support_allows(left,e[rows],g[rows],cfg.split_target_cause,cfg.split_min_target_groups)
                        if not allowed:nr['guard_rejected']+=1;continue
                        if score>best+1e-14:best=score;winner=(int(j),c,bool(ml))
                if 'cif' in n:assert winner is None
                else:
                    assert winner==(n['feature'],n['threshold'],n['missing_left'])
                    max_score_error=max(max_score_error,abs(best-n['score']))
                    zz=x[rows,n['feature']];mask=(zz<=n['threshold'])|(np.isnan(zz)&n['missing_left'])
                    uu,vv=independent_components(t[rows],e[rows],mask);ww=np.ones(cfg.n_causes) if cfg.cause_weights is None else np.asarray(cfg.cause_weights)
                    if cfg.split_score=='causewise_quadratic':
                        independent=float(np.sqrt(np.sum(ww**2*np.divide(uu**2,vv,out=np.zeros_like(vv),where=vv>1e-18))/np.sum(ww**2)))
                    else:
                        den=float(np.sum(ww**2*vv));independent=float(abs(ww@uu)/np.sqrt(den)) if den>1e-18 else 0.
                    max_score_error=max(max_score_error,abs(independent-n['score']))
                    nr['selected_opposed_signs']=bool(uu[0]*uu[1]<0)
                nr.update(guard_changed_winner=winner!=unconstrained_winner,guard_stopped_split=winner is None and unconstrained_winner is not None,
                          best_without_guard=unconstrained_best,best_with_guard=best)
                if 'split_diagnostic' in n:
                    diag=n['split_diagnostic'];assert diag['scored_candidates']==nr['scored_candidates'] and diag['rejected_target_support']==nr['guard_rejected']
                total_attempts+=nr['routed_attempts'];node_rows.append(nr)
            if 'cif' not in n:
                j=n['feature'];split_nodes[j]+=1;used_in_tree.add(j)
                z=x[rows,j];left=(z<=n['threshold'])|(np.isnan(z)&n['missing_left'])
                visit(n['left'],rows[left],used|{j},depth+1,node_key*2,raw)
                visit(n['right'],rows[~left],used|{j},depth+1,node_key*2+1,raw)
        visit(0,cluster_bootstrap(g,cfg.seed,ti),set(),0,1)
        for j in used_in_tree:split_trees[j]+=1
        for j,k in paths_in_tree:pair_paths[j,k]+=1
    assert total_attempts==f.candidate_evaluations_,(total_attempts,f.candidate_evaluations_)
    assert max(max_leaf_error,max_parent_error)<1e-10 and max_score_error<1e-12
    for j in range(x.shape[1]):feature_rows.append(dict(feature=j,eligible_trees=int(selected[j]),split_trees=int(split_trees[j]),split_nodes=int(split_nodes[j])))
    for j,k in combinations(range(x.shape[1]),2):pair_rows.append(dict(feature_j=j,feature_k=k,eligible_trees=int(pair_eligible[j,k]),co_path_trees=int(pair_paths[j,k])))
    return node_rows,feature_rows,pair_rows,leaf_rows,dict(max_score_error=max_score_error,max_leaf_estimator_error=max_leaf_error,max_raw_parent_error=max_parent_error)
