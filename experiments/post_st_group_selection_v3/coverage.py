"""Anchor the original 83-column schedule; repair only removed slots."""
import numpy as np
from crforest.partition import tree_feature_subset


def anchored_sets(keep, seed, n_trees=32, count=25, groups=()):
    keep=tuple(keep); retained=set(keep); inverse={j:i for i,j in enumerate(keep)}
    if len(keep)<count:raise ValueError('fixed count is infeasible')
    result=[];audit=[]
    for tree in range(n_trees):
        base=set(map(int,tree_feature_subset(83,count,seed,tree,'balanced')))
        removed=base-retained; selected=base&retained
        representatives=set().union(*(set(g['members'])&retained for g in groups if set(g['members'])&removed)) if groups else set()
        backup=np.random.default_rng(np.random.SeedSequence([seed,tree,2026110210])).permutation(83)
        order=sorted(representatives-selected)+[int(j) for j in backup if j in retained and j not in representatives]
        for j in order:
            if len(selected)==count:break
            selected.add(j)
        assert len(selected)==count and base&retained<=selected
        result.append([inverse[j] for j in sorted(selected)])
        audit.append(dict(tree=tree,anchor=sorted(base),actual=sorted(selected),removed=sorted(removed),fillers=sorted(selected-base),unchanged=not removed))
    return result,audit


def original_tree(nodes, keep):
    from copy import deepcopy
    result=deepcopy(nodes)
    for node in result:
        node['ancestors']=[keep[j] for j in node['ancestors']]
        if 'feature' in node:node['feature']=keep[node['feature']]
    return result
