import numpy as np
import pandas as pd
import pytest
from experiments.post_st_prognosis_v1.common import classify_first,horizons,endpoint_weights,horizon_predictions
from experiments.post_st_prognosis_v1.prepare import market_features


def test_calendar_months_not_constant_day_count():
    assert horizons('2023-03-01') == [366,731]
    assert horizons('2024-03-01') == [365,730]
    assert horizons('2024-02-29') == [365,730]


def test_recovery_then_termination_is_first_recovery():
    r=classify_first('2020-01-01',[('2021-01-01',2),('2021-06-01',1)],'2022-01-01')
    assert r['event']==2 and r['time']==366 and not r['conflict']


def test_early_termination_retained_and_unobserved_is_not_recovery():
    r=classify_first('2020-01-01',[('2020-03-01',1)],'2021-01-01')
    assert r['event']==1 and r['time']==60
    r=classify_first('2020-01-01',[],'2021-01-01')
    assert r['event']==0 and r['reason']=='administrative_censoring'


def test_information_cutoff_excludes_later_announcements():
    r=classify_first('2022-01-01',[('2023-01-01',1)],'2022-12-31')
    assert r['event']==0 and r['stop']==pd.Timestamp('2022-12-31')
    r=classify_first('2022-01-01',[('2022-12-31',1)],'2022-12-31')
    assert r['event']==1


def test_same_day_order_and_invalid_event_not_silently_resolved():
    assert classify_first('2020-01-01',[('2020-06-01',1),('2020-06-01',2)],'2022-01-01')['conflict']
    assert classify_first('2020-01-01',[('2020-01-01',2)],'2022-01-01')['conflict']
    with pytest.raises(ValueError):classify_first('2020-01-01',[('2020-06-01',3)],'2022-01-01')


def test_ipcw_post_horizon_events_use_horizon_not_later_event_date():
    w,known,diag=endpoint_weights([1,2,3,4],[1,0,2,0],2.5)
    np.testing.assert_allclose(w,[1,0,1.5,1.5])
    np.testing.assert_array_equal(known,[True,False,True,True])
    assert diag['min_G']==pytest.approx(2/3)


def test_unknown_outcome_has_no_endpoint_weight():
    w,known,_=endpoint_weights([2,2,2],[1,2,0],2,eligible=[False,True,True])
    np.testing.assert_array_equal(w,[0,1,1]);assert not known[0]


def test_calendar_specific_prediction_lookup():
    a=np.arange(2*4*2).reshape(2,4,2)
    np.testing.assert_array_equal(horizon_predictions(a,[365,366,730,731],[366,730]),np.stack([a[0,1],a[1,2]]))
    with pytest.raises(ValueError):horizon_predictions(a,[365,366,730,731],[367,730])


def test_market_cutoff_uses_announcement_previous_day_only():
    dates=pd.date_range('2020-01-01',periods=4)
    d=pd.DataFrame(dict(Stkcd=['test']*4,Trddt=dates,Dretwd=[.01,.02,.90,.95],Clsprc=[10,11,50,90],
        Dsmvtll=[1000]*4,Dsmvosd=[500]*4,Dnvaltrd=[100]*4,Dnshrtrd=[10]*4,Hiprc=[12]*4,Loprc=[9]*4))
    r=market_features(d,dates.to_numpy(dtype='datetime64[D]'),'2020-01-03')
    assert r['market_latest_date']==pd.Timestamp('2020-01-02')
    assert r['ret_cum_1m']==pytest.approx(1.01*1.02-1)
    assert r['log_price_last']==pytest.approx(np.log(11))
