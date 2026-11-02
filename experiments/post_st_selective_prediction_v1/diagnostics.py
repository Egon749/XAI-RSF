"""Diagnostics use only saved models, their historical training data and X."""
import numpy as np
from crforest.core import cluster_bootstrap
from .common import *

def leaf_diagnostics(forest,train,X):
    out={h:dict(n=[],known=[],causes=[]) for h in [12,24]}
    for tid,tree in enumerate(forest.trees_):
        bootstrap=cluster_bootstrap(train['groups'],forest.config.seed,tid)
        def visit(node_id,indices,query):
            node=tree[node_id];unique=np.unique(indices)
            assert len(np.unique(train['groups'][indices]))==node['groups'] and len(indices)==node['n']
            assert [len(np.unique(train['groups'][indices[train['event'][indices]==k]])) for k in [1,2]]==node['cause_company_support']
            if 'cif' in node:
                for h in [12,24]:
                    at=train['analysis_time'][unique];ev=train['event'][unique];known=(at>=h)|((ev>0)&(at<=h))
                    n=len(unique);cause=[int(((ev==k)&(at<=h)).sum()) for k in [1,2]]
                    out[h]['n'][-1][query]=n;out[h]['known'][-1][query]=int(known.sum());out[h]['causes'][-1][query]=cause
                return
            j=node['feature'];cut=node['threshold'];missing=node['missing_left']
            z=train['X'][indices,j];left=(z<=cut)|(np.isnan(z)&missing)
            q=X[query,j];ql=(q<=cut)|(np.isnan(q)&missing)
            visit(node['left'],indices[left],query[ql]);visit(node['right'],indices[~left],query[~ql])
        for h in [12,24]:
            out[h]['n'].append(np.zeros(len(X),int));out[h]['known'].append(np.zeros(len(X),int));out[h]['causes'].append(np.zeros((len(X),2),int))
        visit(0,bootstrap,np.arange(len(X)))
    return {h:{k:np.asarray(v) for k,v in d.items()} for h,d in out.items()}

def summarize_leaves(parts,h,c):
    d={k:np.concatenate([p[h][k] for p in parts]) for k in ['n','known','causes']};n=d['n'];cause=d['causes'][:,:,c-1]
    assert (n>0).all() and (d['known']<=n).all()
    return dict(leaf_company_penalty=np.mean(1/(1+n),axis=0),leaf_cause_penalty=np.mean(1/(1+cause),axis=0),
        leaf_followup_penalty=np.mean(1-d['known']/n,axis=0),leaf_unique_company_mean=n.mean(0),leaf_cause_unique_mean=cause.mean(0),
        leaf_zero_cause_fraction=(cause==0).mean(0),leaf_known_fraction=(d['known']/n).mean(0))
