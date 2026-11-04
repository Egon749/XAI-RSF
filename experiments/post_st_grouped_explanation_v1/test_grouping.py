from itertools import combinations
import subprocess
import numpy as np
from scipy.stats import rankdata
from . import stats
from .common import D,PAIRS,TRIPLES,LEFT,RIGHT,SHARED,allocate,bins,blocksum,mapping
from .analysis import cancellation,triple

def test_exact_partition_within_and_across_both_orientations():
    groups=np.array([7,7,0,2])
    keys=np.array([(0,-1),(1,-1),(2,-1),(0,1),(0,2),(2,3)])
    vals=np.array([[1.,2.,3.,-4.,5.,-6.]])
    a=blocksum(vals,bins(keys,groups),36)
    assert a[0,7]==-1 and a[0,0]==3 and a[0,8+PAIRS.index((0,7))]==5
    c=allocate(a)
    np.testing.assert_array_equal(c.sum(-1),vals.sum(-1))
    assert c[0,7]==1.5 and c[0,0]==2.5 and c[0,2]==-3

def test_sign_cancellation_after_combining_and_zero_denominator():
    x=np.zeros((10,2,2));x[:,:,0]=1;x[:,:,1]=-1;x[0]=0
    raw=triple(x);net=np.zeros((120,2,8))
    records=cancellation(raw,np.array([0,0]),net,{})
    assert records[0]['cancellation_mean']==1 and np.isnan(records[1]['cancellation_mean'])

def test_same_feature_cardinality_per_random_slot():
    import pandas as pd
    sets=pd.read_csv(D/'RANDOM_MAPS.csv',index_col=0).to_numpy()-1
    assert len(np.unique(sets,axis=0))==200
    for g in [mapping(),*sets]:assert np.bincount(g).tolist()==[7,12,8,8,6,21,7,5]

def test_native_kernel_ties_signs_overlap_and_empty_comparisons(tmp_path,monkeypatch):
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC',str(D/'kernel.cpp'),'-o',str(tmp_path/'kernel.so')],check=True)
    monkeypatch.setattr(stats,'D',tmp_path)
    rng=np.random.default_rng(128)
    for dim in (4,75):
        x=rng.integers(-2,3,size=(1,120,dim))*.001;x[0,:2]=0
        scores,axis=stats.similarity(x,3,True)
        assert np.all(np.isfinite(scores[:,:,:,0]))
        for s in range(3):
            ix=np.where(SHARED==s)[0];a=x[0,LEFT[ix]];b=x[0,RIGHT[ix]]
            ranks=[rankdata(abs(v),axis=1) for v in (a,b)];aa,bb=[z-z.mean(1)[:,None] for z in ranks]
            dn=np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1);valid=dn>0
            assert abs(scores[0,s,0,0]-np.mean(np.sum(aa[valid]*bb[valid],1)/dn[valid]))<1e-12
            def top(v):
                mask=np.zeros_like(v,bool);np.put_along_axis(mask,np.argsort(-abs(v),axis=1,kind='stable')[:,:3],True,axis=1)
                return mask&(abs(v)>=.001)
            ta,tb=top(a),top(b);un=np.sum(ta|tb,axis=1);ok=un>0
            assert abs(scores[0,s,1,0]-np.mean((ta&tb).sum(1)[ok]/un[ok]))<1e-12
            both=(abs(a)>=.001)&(abs(b)>=.001)
            np.testing.assert_array_equal(axis[0,s,:,0],(both&(np.sign(a)==np.sign(b))).sum(0))
            np.testing.assert_array_equal(axis[0,s,:,1],both.sum(0))
            assert abs(scores[0,s,2,0]-np.mean(abs(a-b).sum(1)))<1e-12
