import numpy as np
import pandas as pd
import pytest
from experiments.post_st_calendar_level_v1.rates import spells, contributions, independent_totals, requested_slope, certified_slopes


def test_cross_year_censor_and_calendar_cap():
    meta=pd.DataFrame(dict(group=['a','b','c'],origin=['2022-12-30','2022-12-31','2020-02-29'],stop=['2023-01-02','2023-01-10','2022-03-01'],event=[1,2,1]))
    m=spells(meta,'2023-01-05')
    days, events=contributions(m,'2023-01-01','2023-01-05')
    np.testing.assert_array_equal(days,[1.,4.,0.]);np.testing.assert_array_equal(events.sum(0),[1,0])
    assert m.loc[2,'stop']==pd.Timestamp('2022-02-28') and m.loc[2,'event']==0
    d,e=independent_totals(meta,'2023-01-05','2023-01-01','2023-01-05')
    assert d==5;np.testing.assert_array_equal(e,[1,0])


def test_boundaries_no_event_window_and_unique_support():
    meta=pd.DataFrame(dict(group=['a','b'],origin=['2021-01-01']*2,stop=['2022-01-01','2023-01-01'],event=[1,1]))
    m=spells(meta,'2023-01-01');d,e=contributions(m,'2022-01-01','2023-01-01')
    assert e[:,0].sum()==1  # Jan1 left included, right/cutoff excluded.
    assert requested_slope(d,e[:,0],d,np.zeros(2,bool))['requested']==1
    # Repeated bootstrap company is never five independent events.
    r=requested_slope(np.ones(6)*10,np.ones(6,bool),np.ones(6),np.ones(6,bool),np.zeros(6,int))
    assert r['distinct_events']==1 and r['requested']==1


def test_cutoff_isolation_future_records_and_future_stop_changes():
    meta=pd.DataFrame(dict(group=list('abcdefgh'),origin=['2021-01-01']*6+['2024-01-01']*2,stop=['2022-06-01']*5+['2025-01-01']*3,event=[1]*8))
    changed=meta.copy();changed.loc[6:,'event']=2;changed.loc[6:,'origin']='2026-02-01';changed.loc[6:,'stop']='2027-01-01';changed.loc[5,'stop']='2029-01-01';changed.loc[5,'event']=2
    results=[]
    for data in [meta,changed]:
        m=spells(data,'2023-01-01');hd,he=contributions(m,'2020-01-01','2023-01-01');wd,we=contributions(m,'2022-01-01','2023-01-01')
        results.append(requested_slope(hd,he[:,0],wd,we[:,0]))
    assert results[0]==results[1]


def test_certificate_preserves_recovery_and_uses_maximum_slope():
    v=np.array([[.2,.7],[.4,.3]])
    b,bound=certified_slopes(v,[4,1]);np.testing.assert_allclose(b,[1.5,1]);assert bound==1
    assert np.max(v@np.array([b[0]+1e-5,b[1]]))>1
    with pytest.raises(ValueError,match='recovery'):certified_slopes(v,[1,2])


def test_hand_calculated_supported_slope_and_bounds():
    hd=np.full(20,10.);wd=np.full(20,4.)
    he=np.arange(20)<10;we=np.arange(20)<6
    # (6/80)/(10/200)=1.5; common days-to-month constant cancels.
    r=requested_slope(hd,he,wd,we)
    assert r['raw_ratio']==pytest.approx(1.5) and r['requested']==pytest.approx(1.5)
    assert not r['sparse_fallback']
    assert requested_slope(hd,he,wd/10,we)['requested']==4
    assert requested_slope(hd,he,wd*10,we)['requested']==.5
