from itertools import combinations
from pathlib import Path
import os,json,hashlib
import numpy as np
import pandas as pd
D=Path(__file__).resolve().parent
REPO=D.parents[1]
PREV=REPO/'experiments/post_st_seed_explanation_stability_v1'
SEEDS=[98,17,503,5001,2570,22,155,150,6720,8679]
ARMS=[*map(str,SEEDS),'reference96','ensemble320']
TRIPLES=np.array(list(combinations(range(10),3)))
COMB=[f'C{i:03d}' for i in range(1,121)]
PAIRS=list(combinations(range(8),2))
LABELS=[f'G{i}' for i in range(1,9)]
INTER=[f'{LABELS[g]}×{LABELS[h]}' for g,h in PAIRS]
LEFT,RIGHT=np.triu_indices(120,1)
SHARED=np.array([len(set(TRIPLES[i])&set(TRIPLES[j])) for i,j in zip(LEFT,RIGHT)])
YEARS=[2023,2024,2025]
def read(p):return json.loads(Path(p).read_text())
def dump(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def run():return Path(read(D/'run_location.json')['private_run'])
def old():return Path(read(PREV/'run_location.json')['private_run'])
def mapping():return pd.read_csv(D/'FEATURE_GROUP_MAPPING.csv').group.str[1:].astype(int).to_numpy()-1
def bins(keys,groups):
    encoded=[]
    for j,k in keys:
        a=groups[j]
        if k<0 or groups[k]==a:encoded.append(a)
        else:encoded.append(8+PAIRS.index(tuple(sorted([a,groups[k]]))))
    return np.asarray(encoded)
def allocate(ai):
    c=ai[...,:8].copy()
    for pos,(g,h) in enumerate(PAIRS):
        c[...,g]+=0.5*ai[...,8+pos];c[...,h]+=0.5*ai[...,8+pos]
    return c
def blocksum(values,indices,size):
    result=np.zeros((*values.shape[:-1],size))
    for j,out in enumerate(indices):result[...,out]+=values[...,j]
    return result
def layout(root,year):
    tr=arrays(root/'folds'/str(year)/'train_PRIVATE.npz');ev=arrays(root/'folds'/str(year)/'eval_PRIVATE.npz')
    lab=arrays(root/'evaluation_input'/f'{year}_test_PRIVATE.npz')
    np.testing.assert_array_equal(ev['sample_ids'],lab['sample_ids'])
    assert len(np.unique(tr['groups']))==len(tr['X']) and len(set(tr['groups'])&set(lab['groups']))==0
    return tr,ev,lab
