"""Lossless scalar-CIF adapters and independent exact Shapley game checks.

TreeSHAP is run by the installed official package. A product reference is
represented exactly by node *region probabilities*, not independently shuffled
finite background rows. This changes explanation covers, never the predictor.
The independent evaluator below enumerates coalitions in each leaf game and
applies the original Shapley and Shapley-interaction factorial weights. It does
not obtain its answer by halving native q2 functions.
"""
from itertools import combinations
from functools import lru_cache
import copy
import math
import json
from pathlib import Path
import numpy as np
from crforest.functions import region_indicator


def q2_adapter(forest, horizon):
    ti=int(np.searchsorted(forest.grid_, horizon, side='right')-1)
    trees=[]
    for nodes in forest.trees_:
        n=len(nodes)
        t=dict(children_left=np.full(n,-1,dtype=np.int32),
               children_right=np.full(n,-1,dtype=np.int32),
               children_default=np.full(n,-1,dtype=np.int32),
               features=np.full(n,-1,dtype=np.int32), thresholds=np.zeros(n),
               values=np.zeros((n,1)),node_sample_weight=np.array([z['n'] for z in nodes],float))
        for i,node in enumerate(nodes):
            if 'cif' in node:
                t['values'][i,0]=(node['cif'][ti][0] if ti>=0 else 0.)/len(forest.trees_)
            else:
                t['children_left'][i]=node['left'];t['children_right'][i]=node['right']
                t['children_default'][i]=node['left'] if node['missing_left'] else node['right']
                t['features'][i]=node['feature'];t['thresholds'][i]=node['threshold']
        trees.append(t)
    return model_dict(trees)


def model_dict(trees):
    return dict(trees=trees,internal_dtype=np.float64,input_dtype=np.float64,
                objective='squared_error',tree_output='raw_value',base_offset=0.)


def load_rsf_export(folder):
    """Do not use rounded JSON thresholds or probabilities for computations."""
    folder=Path(folder);export=json.loads((folder/'forest_export_PRIVATE.json').read_text())
    for b,tree in enumerate(export['trees'],1):
        for field,label in [('threshold','cuts'),('values','values')]:
            values=np.fromfile(folder/f'tree_{b:03d}_{label}_PRIVATE.bin',dtype='<f8')
            if len(values)!=len(tree['feature']):raise ValueError('binary node payload length mismatch')
            tree[field]=values
    return export


def rsf_adapter(export):
    """Native randomForestSRC preorder, exact (unrounded) cuts and terminal CIF."""
    trees=[]
    for raw in export['trees']:
        n=len(raw['feature'])
        t=dict(children_left=np.full(n,-1,dtype=np.int32),children_right=np.full(n,-1,dtype=np.int32),
               children_default=np.full(n,-1,dtype=np.int32),features=np.array(raw['feature'],np.int32),
               thresholds=np.array([float(v) if j>=0 else 0. for j,v in zip(raw['feature'],raw['threshold'])]),
               values=np.array(raw['values'],float)[:,None]/export['ntree'],
               node_sample_weight=np.array(raw['cover'],float),leaf_ids=np.array(raw['leaf_id'],int))
        if not np.isfinite(t['thresholds']).all():raise ValueError('nonfinite active RSF cut')
        def parse(i):
            if t['features'][i]<0:return i+1
            left=i+1;right=parse(left);end=parse(right)
            t['children_left'][i]=left;t['children_right'][i]=right;t['children_default'][i]=left
            return end
        assert parse(0)==n
        validate_tree(t)
        trees.append(t)
    return model_dict(trees)


def validate_tree(t):
    for i,j in enumerate(t['features']):
        if j>=0:
            l,r=t['children_left'][i],t['children_right'][i]
            if not (0<=l<len(t['features']) and 0<=r<len(t['features'])):raise ValueError('invalid child')
            if abs(t['node_sample_weight'][i]-t['node_sample_weight'][l]-t['node_sample_weight'][r])>1e-9:
                raise ValueError('covers do not partition their parent')
    if np.any(t['node_sample_weight']<=0):raise ValueError('all fitted regions must have positive support')


def route(t,X,return_nodes=False):
    X=np.asarray(X,float);at=np.zeros(len(X),int)
    for _ in range(len(t['features'])):
        live=t['features'][at]>=0
        if not live.any():break
        ii=np.flatnonzero(live);n=at[ii];v=X[ii,t['features'][n]]
        at[ii]=np.where(np.isnan(v),t['children_default'][n],
                        np.where(v<=t['thresholds'][n],t['children_left'][n],t['children_right'][n]))
    else:raise ValueError('tree traversal did not terminate')
    return at if return_nodes else t['values'][at,0]


def predict(model,X):return sum(route(t,X) for t in model['trees'])


def walk_regions(t):
    """Yield node region and per-feature unknown-branch probability factors."""
    def visit(i,region,factors):
        yield i,region,factors
        j=int(t['features'][i])
        if j<0:return
        lo,hi,missing=region.get(j,(-np.inf,np.inf,True));cut=t['thresholds'][i]
        left,right=int(t['children_left'][i]),int(t['children_right'][i])
        for child,is_left in [(left,True),(right,False)]:
            reg=dict(region);fac=dict(factors)
            reg[j]=(lo,min(hi,cut),missing and t['children_default'][i]==child) if is_left else (max(lo,cut),hi,missing and t['children_default'][i]==child)
            fac[j]=fac.get(j,1.)*t['node_sample_weight'][child]/t['node_sample_weight'][i]
            yield from visit(child,reg,fac)
    return visit(0,{}, {})


def product_adapter(model,reference):
    """Exact probability of each region under the product of empirical marginals.

Under this reference, branch conditional probabilities are region-mass ratios.
The path-dependent TreeSHAP recurrence therefore evaluates the product-reference
interventional coalition game exactly. No row resampling/Monte Carlo is used.
"""
    ans=copy.deepcopy(model)
    for old,t in zip(model['trees'],ans['trees']):
        masses=np.zeros(len(t['features']))
        for i,region,_ in walk_regions(old):
            masses[i]=math.prod(float(region_indicator(reference[:,j],*bounds).mean()) for j,bounds in region.items())
        t['node_sample_weight']=masses
        validate_tree(t)
    return ans


@lru_cache(maxsize=None)
def shapley_weights(d):
    """Explicit coalition-difference weights, with the global dummy players removed."""
    first=np.zeros((2**d,d));pairs=list(combinations(range(d),2));second=np.zeros((2**d,len(pairs)))
    for j in range(d):
        for s in range(2**d):
            if s>>j&1:continue
            k=s.bit_count();w=math.factorial(k)*math.factorial(d-k-1)/math.factorial(d)
            first[s|1<<j,j]+=w;first[s,j]-=w
    for col,(j,l) in enumerate(pairs):
        for s in range(2**d):
            if s>>j&1 or s>>l&1:continue
            k=s.bit_count();w=math.factorial(k)*math.factorial(d-k-2)/(2*math.factorial(d-1))
            second[s|(1<<j)|(1<<l),col]+=w;second[s|1<<j,col]-=w
            second[s|1<<l,col]-=w;second[s,col]+=w
    return first,second,pairs


def coalition_interactions(model,X):
    """Independent exact per-leaf coalition enumeration, not f_j or f_jl reuse."""
    X=np.asarray(X,float);n,p=X.shape;phi=np.zeros((n,p));Phi=np.zeros((n,p,p));baseline=0.
    for t in model['trees']:
        for i,region,factors in walk_regions(t):
            if t['features'][i]>=0:continue
            c=float(t['values'][i,0]);key=sorted(region);d=len(key)
            if not d:baseline+=c;continue
            if c==0:continue
            indicators=np.column_stack([region_indicator(X[:,j],*region[j]) for j in key]).astype(float)
            rr=np.array([factors[j] for j in key]);v=np.ones((n,2**d))*c
            for a in range(d):
                known=np.array([bool(s>>a&1) for s in range(2**d)])
                v*=np.where(known[None,:],indicators[:,a,None],rr[a])
            baseline+=float(v[0,0])
            w,ww,pairs=shapley_weights(d);phi[:,key]+=v@w;inter=v@ww
            for a,(j,l) in enumerate(pairs):
                Phi[:,key[j],key[l]]+=inter[:,a];Phi[:,key[l],key[j]]+=inter[:,a]
    ii=np.arange(p);Phi[:,ii,ii]=phi-Phi.sum(2)
    return baseline,phi,Phi


def product_components(model,X,reference):
    """Exact centered region expansion at all actual orders (no full-grid materialization)."""
    X=np.asarray(X,float);n,p=X.shape;terms={};base=0.;max_order=0
    for t in model['trees']:
        for i,region,_ in walk_regions(t):
            if t['features'][i]>=0:continue
            c=float(t['values'][i,0]);key=sorted(region);d=len(key);max_order=max(max_order,d)
            if not d:base+=c;continue
            if c==0:continue
            mass=np.array([region_indicator(reference[:,j],*region[j]).mean() for j in key])
            delta=np.column_stack([region_indicator(X[:,j],*region[j]) for j in key])-mass
            base+=float(c*mass.prod())
            for mask in range(1,2**d):
                axes=[a for a in range(d) if mask>>a&1];other=[a for a in range(d) if not mask>>a&1]
                k=tuple(key[a] for a in axes)
                v=c*np.prod(delta[:,axes],axis=1)*np.prod(mass[other])
                if k in terms:terms[k]+=v
                else:terms[k]=v.copy()
    order=np.zeros((n,max_order+1));order[:,0]=base;higher_phi=np.zeros((n,p));shap_from_terms=np.zeros((n,p))
    for key,v in terms.items():
        r=len(key);order[:,r]+=v;shap_from_terms[:,key]+=v[:,None]/r
        if r>=3:higher_phi[:,key]+=v[:,None]/r
    return dict(base=base,terms=terms,order=order,higher_phi=higher_phi,shap_from_terms=shap_from_terms,max_order=max_order)


def default_tree_shap(model,X):
    import shap
    # Explicitly exercise the user's requested default: no data, no override.
    e=shap.TreeExplainer(model)
    assert e.feature_perturbation=='tree_path_dependent'
    phi=np.asarray(e.shap_values(X));Phi=np.asarray(e.shap_interaction_values(X))
    return dict(base=float(np.asarray(e.expected_value).reshape(-1)[0]),phi=phi,Phi=Phi,
                adapter_prediction=np.asarray(e.model.predict(X)),resolved_perturbation=e.feature_perturbation)
