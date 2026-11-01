import numpy as np
import pandas as pd
from experiments.post_st_june_cutoff_v1.common import masks,relabel,evaluation_status
from experiments.post_st_prognosis_v1.common import calendar_elapsed


def test_cutoff_is_inclusive_and_next_entry_window_disjoint():
    d=pd.Series(pd.to_datetime(['2023-06-29','2023-06-30','2023-07-01','2024-06-30','2024-07-01']))
    train,test=masks(d,'2023-06-30')
    np.testing.assert_array_equal(train,[True,True,False,False,False])
    np.testing.assert_array_equal(test,[False,False,True,True,False]);assert not (train&test).any()
    assert evaluation_status(1,False)=='BLOCKED_INCOMPLETE_ENTRY_COHORT'
    assert evaluation_status(1,True)=='INSUFFICIENT_EVALUATION_SAMPLE'


def test_training_outcomes_rebuilt_at_cutoff_and_test_clock_calendar_exact():
    meta=pd.DataFrame(dict(group=['g'],entry_announcement=pd.to_datetime(['2023-05-01']),origin=pd.to_datetime(['2023-05-02'])))
    events={'g':pd.DataFrame(dict(announcement=pd.to_datetime(['2023-07-01']),Chgtype=['DX'],chain_discontinuity=['0']))}
    before=relabel(meta,events,{},'2023-06-30');after=relabel(meta,events,{},'2023-07-01')
    assert before.event.iloc[0]==0 and before.stop.iloc[0]==pd.Timestamp('2023-06-30')
    assert after.event.iloc[0]==1 and after.stop.iloc[0]==pd.Timestamp('2023-07-01')
    later={'g':events['g'].assign(announcement=pd.to_datetime(['2024-05-03']))}
    test=relabel(meta,later,{},'2026-06-30',endpoint_months=12)
    assert test.event.iloc[0]==0 and test.stop.iloc[0]==pd.Timestamp('2024-05-02')
    assert test.time.iloc[0]==366 and calendar_elapsed(meta.origin.iloc[0],test.stop.iloc[0])==12
    zero=relabel(meta.assign(origin=pd.to_datetime(['2023-06-30'])),events,{},'2023-06-30')
    assert zero.time.iloc[0]==0 and not zero.label_eligible.iloc[0]
    assert zero.reason.iloc[0]=='no_positive_verified_followup'
