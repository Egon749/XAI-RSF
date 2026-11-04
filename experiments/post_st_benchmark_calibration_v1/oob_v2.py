"""OOB reconstruction from saved trees. Threshold API has no outcome arguments."""
from dataclasses import replace
from copy import copy
import numpy as np
from crforest import cluster_bootstrap
from crforest.core import aalen_johansen
from experiments.post_st_risk_index_v1.index import risk_index, historical_thresholds


def multiplicities(groups, seed, tree_id):
    return np.bincount(cluster_bootstrap(groups, seed, tree_id), minlength=len(groups))


def predict_tree(forest,tree_id,X,times):
    single=copy(forest);single.trees_=[forest.trees_[tree_id]]
    single.config=replace(forest.config,n_trees=1)
    return single.predict(X,times)


def oob_predictions(members,X,groups,times=(365.,730.)):
    """Use original seed/tree IDs, average each company's own OOB trees equally."""
    X=np.asarray(X);groups=np.asarray(groups)
    total=np.zeros((len(X),len(times),members[0].config.n_causes));count=np.zeros(len(X),int)
    for f in members:
        for ti in range(len(f.trees_)):
            mask=multiplicities(groups,f.config.seed,ti)==0
            if mask.any():total[mask]+=predict_tree(f,ti,X[mask],times)
            count+=mask
    pred=np.full_like(total,np.nan);ok=count>0
    pred[ok]=total[ok]/count[ok,None,None]
    return pred,count


def threshold_table(pred,count,mu):
    """No outcome/evaluation inputs; minimum10 trees and30 training companies."""
    valid=np.asarray(count)>=10
    scores=risk_index(np.asarray(pred)[valid],mu)
    thresholds=np.full((2,2,2),np.nan)
    if valid.sum()>=30:
        for hi in range(2):
            for k in range(2):thresholds[hi,k]=historical_thresholds(scores[:,hi,k])
    return thresholds,valid,scores


def verify_membership(f,X,time,event,groups):
    """Independent replay of shrink_leaf_cifs membership/raw-direct-parent math.

    Outcomes are used only for verification of immutable saved estimator bytes;
    not accepted by oob_predictions or threshold_table.
    """
    X,time,event,groups=map(np.asarray,(X,time,event,groups))
    max_error=0.;checked=0;hashable_counts=[]
    for ti,tree in enumerate(f.trees_):
        sampled=cluster_bootstrap(groups,f.config.seed,ti)
        counts=np.bincount(sampled,minlength=len(X));hashable_counts.append(counts)
        def visit(node_id,rows,parent_raw=None):
            nonlocal checked,max_error
            node=tree[node_id];checked+=1;unique=len(np.unique(groups[rows]))
            assert node['n']==len(rows) and node['groups']==unique,'STOP: bootstrap/node size mismatch'
            assert node['events']==int((event[rows]>0).sum())
            assert node['cause_company_support']==[len(np.unique(groups[rows[event[rows]==k]])) for k in [1,2]]
            assert node['cause_event_rows']==[int((event[rows]==k).sum()) for k in [1,2]]
            raw=aalen_johansen(time[rows],event[rows],f.grid_,2)
            if 'cif' in node:
                sh=node['leaf_shrinkage'];assert sh['strength']==10 and sh['unique_inbag_companies']==unique
                weight=1. if parent_raw is None else unique/(unique+10.)
                assert sh['weight']==weight
                prior=raw if parent_raw is None else parent_raw
                error=max(float(abs(raw-sh['raw_leaf_cif']).max()),float(abs(prior-sh['raw_parent_cif']).max()),float(abs(weight*raw+(1-weight)*prior-node['cif']).max()))
                max_error=max(max_error,error);assert error<=1e-10
                return
            z=X[rows,node['feature']];left=(z<=node['threshold'])|(np.isnan(z)&node['missing_left'])
            visit(node['left'],rows[left],raw);visit(node['right'],rows[~left],raw)
        visit(0,sampled)
    return dict(nodes_checked=checked,max_leaf_parent_error=max_error),np.array(hashable_counts)
