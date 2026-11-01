import numpy as np
from experiments.post_st_dropcolumn_selection_h24_v1.selection import gate24,constraints,bootstrap_ap_difference

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
