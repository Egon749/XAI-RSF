import numpy as np
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_selective_prediction_v1.common import ecdf,scores,threshold,accept,cell_support,matched
from experiments.post_st_selective_prediction_v1.diagnostics import leaf_diagnostics,summarize_leaves

def test_historical_threshold_is_individual_and_ties_are_reproducible():
    score=np.array([0.,0.,1.,1.,1.]);ids=np.array(list('abcde'));cut=threshold(score,ids,.6)
    assert accept(score,ids,cut).sum()==3
    future=np.array([.2,.8,2.]);fi=np.array(['x','y','z']);old=accept(future,fi,cut)
    np.testing.assert_array_equal(old,accept(np.r_[future,-999.,999.],np.concatenate([fi,['new','newer']]),cut)[:3])
    assert matched(score,ids,2).sum()==2 and matched(score,ids,0).sum()==0

def test_constant_proxy_and_out_of_range_ecdf_do_not_create_confidence():
    np.testing.assert_array_equal(ecdf([2,2,2],[1,2,3]),[0,.5,1])
    r=np.arange(60,dtype=float).reshape(10,6);x=r[[1,5]]
    assert np.allclose(scores(r,x,'combined'),np.mean([ecdf(r[:,j],x[:,j]) for j in range(5)],axis=0))
    np.testing.assert_array_equal(scores(r,x,'seed_sd'),x[:,0])

def test_joint_cell_support_counts_companies_once_and_missing_separately():
    class Stub:
        def cuts(self,key,j):return np.array([0.])
    tr=np.array([[-1,-1],[-1,-1],[1,-1],[np.nan,1],[np.nan,1]],float);g=np.array(['a','a','b','c','d'])
    q=np.array([[-1,-1],[1,-1],[np.nan,1],[1,1]],float)
    np.testing.assert_array_equal(cell_support(Stub(),(0,1),tr,q,g),[1,1,2,0])

def test_saved_shrunk_leaf_support_is_observed_not_borrowed():
    rng=np.random.default_rng(102);X=rng.normal(size=(60,3));t=rng.integers(1,30,60);e=rng.choice([0,1,2],60);groups=np.arange(60)
    raw=CompetingRiskForest(ForestConfig(n_trees=2,max_depth=2,path_budget=2,min_leaf=5,min_groups=3,min_events=1)).fit(X,t,e,groups)
    shrunk=shrink_leaf_cifs(raw,X,t,e,groups,10)
    tr=dict(X=X,time=t,event=e,groups=groups,analysis_time=t.astype(float));a=leaf_diagnostics(raw,tr,X);b=leaf_diagnostics(shrunk,tr,X)
    for h in [12,24]:
        for k in a[h]:np.testing.assert_array_equal(a[h][k],b[h][k])
        d=summarize_leaves([a],h,1);assert np.all((d['leaf_followup_penalty']>=0)&(d['leaf_followup_penalty']<=1))
    assert (a[12]['causes']<=a[24]['causes']).all()


def test_selective_loss_decomposition_and_empty_subset_are_honest():
    from experiments.post_st_selective_prediction_v1.metrics import group_metrics,decomposition
    y=np.array([0,0,0,1,1,1],bool);p=np.array([.02,.03,.04,.1,.2,.3]);ids=np.array(list('abcdef'));years=np.full(6,2024)
    all=group_metrics(y,p,np.ones(6,bool),ids,years);a=group_metrics(y,p,np.array([1,1,1,0,1,0],bool),ids,years)
    d=decomposition(all,a);assert abs(d['delta_brier']-d['prevalence_mix_component']-d['within_status_component'])<1e-14
    empty=group_metrics(y,p,np.zeros(6,bool),ids,years);assert np.isnan(empty['brier']) and empty['accepted']==0 and empty['referred_events']==3
    noevents=group_metrics(y,p,~y,ids,years);assert np.isnan(noevents['ap']) and np.isnan(noevents['balanced_mse'])


def test_script_launcher_does_not_shadow_standard_library():
    import subprocess,sys
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'experiments/post_st_selective_prediction_v1/launch.py'
    result=subprocess.run([sys.executable,str(p),'--help'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert '--year' in result.stdout


def test_resample_and_random_control_use_event_counts_not_boolean_dot():
    from experiments.post_st_selective_prediction_v1.analyze import simple_bootstrap,matched_random_statistics
    y=np.array([1,1,1,0],bool);p=np.array([.1,.2,.3,.4]);mask=np.array([1,1,0,1],bool)
    counts=np.array([[1,1,1,1],[2,1,0,1]])
    b=simple_bootstrap(y,p,mask,counts)
    np.testing.assert_allclose(b['event_retention'],[2/3,1])
    np.testing.assert_allclose(b['brier'],[(.9**2+.8**2+.4**2)/3,(2*.9**2+.8**2+.4**2)/4])
    r=matched_random_statistics(y,p,np.array([[1,1,1,0],[1,1,0,1]],bool),3,np.arange(4))
    np.testing.assert_array_equal(r['original_budget_captured'],[3,2])
    np.testing.assert_allclose(r['event_retention'],[1,2/3])
    empty=matched_random_statistics(y,p,np.zeros((2,4),bool),3,np.arange(4))
    assert np.isnan(empty['brier']).all() and (empty['original_budget_captured']==0).all()
