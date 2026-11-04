import numpy as np
from experiments.post_st_benchmark_calibration_v1.bootstrap import assessment,FIELDS
from .metrics import metrics,record,position
def test_known_endpoint_ties_and_annual_quota():
    ids=np.array(list('abcde')+list('vwxyz'));yrs=np.repeat([2023,2024],5);p=np.array([.2,.2,.3,.2,.1,.1,.5,.5,.2,.8]);y=np.array([0,1,1,0,0,0,0,1,0,1]);w=np.ones(10)
    data=[record(yr,ids[yrs==yr],y[yrs==yr],w[yrs==yr],p[yrs==yr]) for yr in [2023,2024]]
    a=metrics(data);e=assessment(w[None,:],y,p,yrs,ids)[0]
    for f in ['auc','within_year_auc','ap','top20_captured','top20_coverage']:np.testing.assert_allclose(a[f],e[FIELDS.index(f)])
    assert a['checked']==2 and position(np.array([.1,.2,.2,.3]),.2)==50.
def test_oob_censor_in_top_and_no_positive():
    a=metrics([record(2024,np.arange(4),np.array([-1,0,0,0]),np.array([0.,1,1,1]),np.array([.8,.2,.1,.5]))])
    assert a['unknown_checked']==1 and a['top20_captured']==0 and np.isnan(a['precision20']) and np.isnan(a['auc'])
