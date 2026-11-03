"""Exact tree-index assembly; reuses the existing core grower unchanged."""
from copy import deepcopy
from dataclasses import asdict,replace
import numpy as np
from crforest.core import _grow_tree,cluster_bootstrap
from crforest.partition import training_cut_dictionary
from .common import pairing,array_hash

def grow_range(cfg,tr,start,stop):
    if cfg.global_cell_budget is not None or not 0<=start<stop<=cfg.n_trees:
        raise ValueError('Independent valid tree indices and no global budget required')
    x,t,e,g=[tr[k]for k in ['X','time','event','groups']]
    grid=np.unique(t[e>0]);grid=grid if len(grid)else np.array([0.])
    cuts=training_cut_dictionary(x,cfg);output=[]
    for tid in range(start,stop):
        tree,inbag,evaluations=_grow_tree(cfg,grid,x,t,e,g,tid,shared_cuts=cuts)
        expected=np.bincount(cluster_bootstrap(g,cfg.seed,tid),minlength=len(g))
        np.testing.assert_array_equal(inbag,expected)
        output.append(dict(tree_id=tid,nodes=tree,candidate_evaluations=int(evaluations),
            inbag_sha256=array_hash(inbag)))
    return output

def assemble(base,chunks,total,old_counts):
    first=len(base.trees_)
    if base.config.global_cell_budget is not None or any('leaf_shrinkage'in n for t in base.trees_ for n in t):
        raise ValueError('Only independent raw AJ trees may be extended')
    ordered=sorted(chunks,key=lambda x:x['tree_id'])
    if [x['tree_id']for x in ordered]!=list(range(first,total)):
        raise ValueError('Missing or duplicated original tree indices')
    if len(old_counts)!=first or sum(old_counts)!=base.candidate_evaluations_:
        raise ValueError('Original per-tree search counts must reproduce saved total')
    f=deepcopy(base);f.config=replace(base.config,n_trees=total)
    f.trees_+=deepcopy([x['nodes']for x in ordered])
    counts=list(old_counts)+[x['candidate_evaluations']for x in ordered]
    f.candidate_evaluations_=int(sum(counts));f.audit_=f.audit()
    return f,counts

def prefix(f,count,counts):
    if not 0<count<=len(f.trees_) or len(counts)!=len(f.trees_):raise ValueError('Invalid prefix')
    g=deepcopy(f);g.config=replace(f.config,n_trees=count);g.trees_=g.trees_[:count]
    g.candidate_evaluations_=int(sum(counts[:count]));g.audit_=g.audit()
    return g
