import numpy as np
from experiments.post_st_dropcolumn_selection_h24_v1.selection import gate24,constraints,bootstrap_ap_difference,SELECTION_SEED

def test_h24_support_independent_of_sparse_h12():
    rows=[dict(cohort=y,horizon=24,mature=True,supported=True,train_termination=2,termination=e,controls=20) for y,e in [(2020,6),(2021,7)]]
    assert gate24(rows)['passed']
    assert not gate24(rows[:1])['passed']

def test_h24_guards_reject_undefined_secondary_and_complexity_increase():
    d=dict(delta_ap24=.01,delta_capture24=0,ap24_lower90=0,delta_auc12=0,delta_ap12=0,delta_capture12=0,
        delta_brier12=0,delta_brier24=0,recovery_delta_brier12=0,recovery_delta_brier24=0,seed_gain_fraction=1,pair_jaccard_change=0,spatial_cell_change=-1)
    assert constraints(d)==[]
    d['delta_ap12']=np.nan;d['spatial_cell_change']=1
    assert constraints(d)==['delta_ap12','spatial_cell_change']

def test_h24_resampling_identical_risks_and_group_multiplicity():
    labels=dict(analysis_time=np.array([4.,24.,24.,8.]),event=np.array([1,0,2,1]),label_eligible=np.ones(4,bool),sample_ids=np.array(['a','b','c','d']))
    a=np.array([.1,.2,.2,.3]);b=np.array([.4,.1,.2,.3]);counts=np.array([2,0,1,1])
    same=bootstrap_ap_difference(labels,a,a,counts,2025,2020,0)
    assert np.nanmax(abs(same))==0 and np.isnan(same).any()
    delta=bootstrap_ap_difference(labels,a,b,counts,2025,2020,0)
    assert np.nanmin(delta)>=-1e-14 and np.nanmax(delta)>0

def test_h24_bootstrap_ap_matches_independent_expanded_company_scores():
    from sklearn.metrics import average_precision_score
    labels=dict(analysis_time=np.array([4.,24.,24.,8.,24.]),event=np.array([1,0,2,1,0]),label_eligible=np.ones(5,bool),sample_ids=np.array(['a','b','c','d','e']))
    a=np.array([.1,.2,.2,.3,.4]);b=np.array([.4,.1,.2,.3,.3]);counts=np.array([2,0,1,3,1])
    values=bootstrap_ap_difference(labels,a,b,counts,2025,2021,7)
    active=np.flatnonzero(counts);rng=np.random.default_rng(np.random.SeedSequence([SELECTION_SEED,2025,2021,8]))
    for i,draw in enumerate(rng.integers(len(active),size=(20,len(active)))):
        ix=np.repeat(active[draw],counts[active[draw]]);y=labels['event'][ix]==1
        expected=average_precision_score(y,b[ix])-average_precision_score(y,a[ix]) if y.any() else np.nan
        np.testing.assert_allclose(values[i],expected,atol=1e-14,equal_nan=True)
def test_paired_function_variation_has_known_scale_and_identity():
    import numpy as np
    from experiments.post_st_dropcolumn_selection_h24_v1.stability_comparison import compare_variation
    reference={'X':np.zeros((2,1)),'h12':np.array([365.,365.]),'h24':np.array([730.,730.])}
    shape=np.broadcast_to(np.array([1.,-1.])[:,None,None],(2,4,2)).copy()
    base=[{(0,):r*shape} for r in range(50)];scaled=[{(0,):2*r*shape} for r in range(50)];mu=[np.zeros((4,2)) for _ in range(50)]
    rows,paired,pred=compare_variation(2024,reference,base,scaled,{(0,):shape},{(0,):2*shape},mu,mu,mu[0],mu[0])
    wanted=np.std(np.arange(50),ddof=1)
    for row in rows:
        expected=0 if row['order']==2 else wanted*(2 if row['model']=='selected_h24' else 1)
        assert np.isclose(row['component_spread_l2'],expected,rtol=0,atol=1e-12)
    for row in pred:assert np.isclose(row['root_mean_prediction_variance'],wanted*(2 if row['model']=='selected_h24' else 1))
    _,paired,_=compare_variation(2024,reference,base,base,{(0,):shape},{(0,):shape},mu,mu,mu[0],mu[0])
    assert all(row['difference']==0 and row['full83_jaccard']==row['selected_jaccard'] for row in paired)
