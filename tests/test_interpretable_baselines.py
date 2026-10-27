from itertools import combinations
import math
import numpy as np
from scipy.special import expit
from crforest import CompetingRiskForest,ForestConfig
from experiments.interpretable_baselines_v1.methods import ScalarForest,endpoint_weights,resample_groups,fit_ebm,ebm_complexity,binary_grid,walsh,truth,SCENARIOS


def test_exact_shap_matches_all_coalitions_and_official_shap():
    import shap
    rng=np.random.default_rng(340);X=rng.normal(size=(150,4));X[::9,0]=np.nan
    f=CompetingRiskForest(ForestConfig(n_trees=3,max_depth=4,path_budget=None,min_leaf=8,min_events=1,min_groups=2,mtry=4,n_thresholds=3)).fit(X,rng.integers(1,20,len(X)),rng.integers(0,3,len(X)),np.arange(len(X)))
    fun=ScalarForest(f,12);xx=X[3:7];bg=X[::15]
    np.testing.assert_allclose(fun(X),f.predict(X,[12])[:,0,0],atol=1e-15,rtol=0)
    base,phi=fun.exact_shap(xx,bg);brute=np.zeros_like(phi)
    for i,x in enumerate(xx):
        vals={}
        for mask in range(16):
            z=bg.copy();cols=[j for j in range(4) if mask>>j&1];z[:,cols]=x[cols];vals[mask]=fun(z).mean()
        for j in range(4):
            for mask in range(16):
                if mask>>j&1:continue
                k=mask.bit_count();brute[i,j]+=math.factorial(k)*math.factorial(3-k)/math.factorial(4)*(vals[mask|1<<j]-vals[mask])
    np.testing.assert_allclose(phi,brute,atol=1e-14,rtol=0)
    official=shap.ExactExplainer(fun,shap.maskers.Independent(bg,max_samples=len(bg)))(xx)
    np.testing.assert_allclose(phi,official.values,atol=1e-14,rtol=0)
    approx=shap.PermutationExplainer(fun,bg,seed=42)(xx,max_evals=9*2,silent=True)
    np.testing.assert_allclose(approx.base_values+approx.values.sum(1),fun(xx),atol=1e-14)


def test_ipcw_unknown_competing_and_cluster_clones():
    w=endpoint_weights(np.array([1.,.5,2.,2.]),np.array([1,0,0,2]),2.)
    assert w[1]==0 and w[0]>0 and w[2]>0 and w[3]>0
    groups=np.repeat(np.arange(4),[1,2,3,4]);i,g=resample_groups(groups,42)
    for clone in np.unique(g):
        selected=i[g==clone];original=groups[selected[0]]
        np.testing.assert_array_equal(selected,np.flatnonzero(groups==original))
    assert len(np.unique(g))==4


def test_walsh_known_truth_and_links():
    grid=binary_grid();W=walsh(grid);np.testing.assert_array_equal(W.T@W,256*np.eye(256))
    for scenario in SCENARIOS:
        p=truth(grid,scenario);assert np.all((p>0)&(p<1))
        c=W.T@p/256;np.testing.assert_allclose(W@c,p,atol=1e-14)
        if scenario=='pair_cif':assert set(np.flatnonzero(abs(c)>1e-12))=={0,1,6}
        if scenario=='triple_cif':assert set(np.flatnonzero(abs(c)>1e-12))=={0,1,6,112}


def test_ebm_native_link_reconstruction_and_no_internal_validation():
    rng=np.random.default_rng(77);X=rng.normal(size=(200,4));X[::7,1]=np.nan;e=rng.integers(0,3,len(X))
    m=fit_ebm(X,e,np.ones(len(X)),2,42,rounds=8)
    d=ebm_complexity(m,X);assert d['reconstruction_error']<1e-14
    assert m.validation_size==0 and m.outer_bags==1
    assert len(m.term_features_)==6
