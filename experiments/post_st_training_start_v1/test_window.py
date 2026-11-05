import numpy as np
import pandas as pd
import pytest
from .common import filter_history, supported, endpoint

def fixture():
    meta = pd.DataFrame({'sample_id':['a','b','c'], 'origin':pd.to_datetime(['2012-05-01','2019-05-01','2022-05-01']),
        'stop':pd.to_datetime(['2014-05-01','2019-07-01','2022-12-31']), 'entry_year':[2012,2019,2022]})
    data = dict(X=np.array([[1.,np.nan],[2.,3.],[4.,5.]]),time=np.array([730.,61.,244.]),
                event=np.array([0,1,0]),groups=np.array(['aa','bb','cc']),sample_ids=np.array(['a','b','c']))
    return data,meta

def test_filter_identity_and_order():
    data,meta=fixture()
    full,_=filter_history(data,meta,2012,2023)
    for k in data:np.testing.assert_array_equal(full[k],data[k])
    new,mask=filter_history(data,meta,2019,2023)
    np.testing.assert_array_equal(mask,[False,True,True])
    for k in data:np.testing.assert_array_equal(new[k],data[k][1:])
    assert new['event'][-1]==0  # A recent censor remains censored, never relabeled.

def test_future_or_misaligned_snapshot_rejected():
    data,meta=fixture();meta.loc[2,'stop']=pd.Timestamp('2023-02-01')
    with pytest.raises(ValueError,match='cutoff'):filter_history(data,meta,2020,2023)
    data,meta=fixture();data['sample_ids']=data['sample_ids'][::-1]
    with pytest.raises(AssertionError):filter_history(data,meta,2020,2023)

def test_calendar_endpoints_and_unavailable_horizon():
    p=np.arange(24).reshape(3,4,2)
    np.testing.assert_array_equal(endpoint(p,np.array([365.,366.,731.])),p[[0,1,2],[0,1,3]])
    assert not supported(2025,24) and supported(2025,12) and supported(2024,24)
