from itertools import combinations
import numpy as np
from scipy.stats import rankdata
from .common import TRIPLES,distributions

LEFT,RIGHT=np.triu_indices(len(TRIPLES),1)
SHARED=np.array([len(set(TRIPLES[i]) & set(TRIPLES[j])) for i,j in zip(LEFT,RIGHT)])
def divide(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)!=0)
def jaccards(mask):
    a=np.asarray(mask,dtype=float);common=a@a.T
    n=a.sum(1)
    return divide(common,n[:,None]+n-common)
def correlations(matrix):
    a=np.asarray(matrix,float);a=a-a.mean(1,keepdims=True)
    norm=np.linalg.norm(a,axis=1)
    return divide(a@a.T,norm[:,None]*norm)
def similarity_rows(values,**info):
    v=np.asarray(values)
    return [dict(**info,shared_members=s,**distributions(v if s=='all' else v[SHARED==s]))
            for s in ('all',0,1,2)]
def midpercent(x,v):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return 100*((x<v).sum()+.5*(x==v).sum())/len(x) if len(x) and np.isfinite(v) else np.nan
