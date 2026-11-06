import numpy as np
import pandas as pd
from experiments.pre_st_warning_v1.clock import stop_at,endpoint,available_values
from experiments.pre_st_warning_v1.raw_scan import normalize,MARKET

def test_future_training_events_are_censored_not_normal_labels():
    t,e,end=stop_at('2022-01-01','2026-08-31','2023-02-01',1,'2023-01-01')
    assert t==364 and e==0 and end==pd.Timestamp('2022-12-31')
    known,y=endpoint([t],[e],[365],1)
    assert not known[0] and not y[0]
    assert stop_at('2022-01-01','2026-08-31','2024-02-01',1,'2023-01-01')==(t,e,end)

def test_calendar_horizon_and_competing_event():
    assert stop_at('2024-01-01','2026-08-31',pd.NaT,0)[0]==731
    assert stop_at('2023-01-01','2026-08-31',pd.NaT,0)[0]==731
    known,y=endpoint([12,364,365],[2,0,0],[365,365,365],1)
    np.testing.assert_array_equal(known,[True,False,True]);assert not y.any()

def test_derived_feature_requires_both_disclosure_dates():
    row=pd.Series(dict(available=pd.Timestamp('2022-10-20'),prev_available=pd.Timestamp('2023-02-01'),roa=.1,revenue_growth=.4))
    v,d=available_values(row,['roa','revenue_growth'],pd.Timestamp('2023-01-01'))
    assert v[0]==.1 and np.isnan(v[1]) and np.isnat(d[1])

def test_raw_header_rows_and_non_a_shares_are_removed():
    rows=[['证券代码','交易日期']+[None]*10,['1','2022-12-30']+[1]*8+[1,1],['2','2022-12-30']+[1]*8+[2,1]]
    d=normalize(pd.DataFrame(rows,columns=MARKET),'market')
    assert d.Stkcd.tolist()==['000001']
