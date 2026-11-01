import numpy as np
from experiments.post_st_dropcolumn_selection_v1.common import configuration,support_gate,bootstrap_differences,metrics
from experiments.post_st_dropcolumn_selection_v1.selector import eliminate

def test_support_failure_is_not_evidence_for_deletion():
    gate=support_gate([dict(horizon=h,mature=True,supported=True,train_termination=4,termination=3,controls=20) for h in [12,24]])
    assert not gate['passed']
    def forbidden(*args):raise AssertionError('A closed gate must not run a selector')
    chosen,path=eliminate(range(83),[(j,) for j in range(83)],gate,forbidden)
    assert len(chosen)==83 and path[0]['new_candidate_fits']==0

def test_recursive_selection_recomputes_after_each_removal():
    calls=[]
    def evaluate(current,proposals):
        calls.append(current)
        out=[]
        for p in proposals:
            out.append(dict(keep=p,subset_id=str(p),support_pass=True,delta_ap12=.01 if len(p)>=2 else -.2,
                delta_capture12=0,ap12_lower90=0,delta_auc24=0,delta_ap24=0,delta_capture24=0,
                delta_brier12=0,delta_brier24=0,recovery_delta_brier12=0,recovery_delta_brier24=0,
                stable_direction_fraction=1,pair_jaccard_change=0,spatial_cell_change=-1,valid_model=True))
        return out
    keep,path=eliminate([0,1,2],[(0,),(1,),(2,)],{'passed':True},evaluate)
    assert len(keep)==2 and calls[0]==(0,1,2) and calls[1]==keep and path[-1]['action']=='STOP'

def test_count_ratio_and_unsupported_metrics():
    assert configuration(42,82,'ratio').tree_feature_count==25
    assert configuration(42,74,'ratio').tree_feature_count==22
    assert configuration(42,74,'count').tree_feature_count==25
    labels=dict(analysis_time=np.array([4.,8.,9.]),event=np.array([1,0,0]),label_eligible=np.ones(3,bool),sample_ids=np.array(['a','b','c']))
    result=metrics(labels,np.array([.2,.1,.1]),24,1)
    assert not result['supported'] and np.isnan(result['auc']) and np.isnan(result['captured'])

def test_paired_bootstrap_matches_explicit_repeated_companies():
    from experiments.post_st_train2024_h12_v1.validate import direct_metrics
    labels=dict(analysis_time=np.array([4.,12.,12.,8.]),event=np.array([1,0,2,1]),label_eligible=np.ones(4,bool),sample_ids=np.array(['a','b','c','d']))
    a=np.array([.1,.2,.2,.3]);b=np.array([.4,.1,.2,.3]);draws=np.array([[0,1,2,3],[0,0,1,3],[1,2,1,2]])
    d=bootstrap_differences(labels,a,b,12,1,draws)
    for i,ix in enumerate(draws):
        y=labels['event'][ix]==1
        want=np.array(direct_metrics(y,b[ix]))-np.array(direct_metrics(y,a[ix]))
        np.testing.assert_allclose([d[k][i] for k in ['auc','ap','brier']],want,atol=1e-14,equal_nan=True)
