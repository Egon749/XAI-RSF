"""Deterministic independent tree shards; global-budget fits are never sharded."""
from dataclasses import asdict
import numpy as np
from crforest import CompetingRiskForest
from crforest.core import _grow_tree
from crforest.partition import training_cut_dictionary


def fit_shard(cfg, data, start, count):
    if cfg.global_cell_budget is not None:
        raise ValueError('global partition ledger cannot be sharded')
    if not 0<=start<start+count<=cfg.n_trees:
        raise ValueError('invalid tree interval')
    X,t,e,g=(data[k] for k in ['X','time','event','groups'])
    f=CompetingRiskForest(cfg);f.grid_=np.unique(t[e>0]);f.n_features_=X.shape[1]
    if not len(f.grid_):f.grid_=np.array([0.])
    f.shared_cuts_=training_cut_dictionary(X,cfg)
    f.partition_audit_=None;f.trees_=[];f.candidate_evaluations_=0
    for tree_id in range(start,start+count):
        nodes,_,number=_grow_tree(cfg,f.grid_,X,t,e,g,tree_id,shared_cuts=f.shared_cuts_)
        f.trees_.append(nodes);f.candidate_evaluations_+=number
    f.audit_=f.audit()
    return f


def assemble(parts):
    """parts is [(tree_start, partial_forest), ...]; require exactly all trees."""
    parts=sorted(parts,key=lambda x:x[0]);base=parts[0][1];position=0;trees=[];evaluations=0
    for start,f in parts:
        if start!=position or asdict(f.config)!=asdict(base.config) or f.n_features_!=base.n_features_:
            raise ValueError('missing/duplicate/mismatched tree shard')
        np.testing.assert_array_equal(f.grid_,base.grid_)
        if (f.shared_cuts_ is None)!=(base.shared_cuts_ is None):raise ValueError('different cut dictionaries')
        if f.shared_cuts_ is not None:
            for a,b in zip(f.shared_cuts_,base.shared_cuts_):np.testing.assert_array_equal(a,b)
        position+=len(f.trees_);trees.extend(f.trees_);evaluations+=f.candidate_evaluations_
    if position!=base.config.n_trees:raise ValueError('incomplete forest')
    result=CompetingRiskForest(base.config);result.grid_=base.grid_;result.n_features_=base.n_features_
    result.trees_=trees;result.candidate_evaluations_=evaluations;result.shared_cuts_=base.shared_cuts_;result.partition_audit_=None
    result.audit_=result.audit()
    return result
