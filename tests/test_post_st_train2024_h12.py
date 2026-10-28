import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import classify_first,calendar_elapsed,endpoint_weights


def test_twelve_month_training_cap_censors_later_events():
    r=classify_first('2023-04-15',[('2024-04-16',1)],'2024-12-31',months=12)
    assert r['event']==0 and r['stop']==pd.Timestamp('2024-04-15')
    assert r['time']==366 and calendar_elapsed('2023-04-15',r['stop'])==12


def test_2024_entry_cannot_use_2025_outcome_to_fit():
    r=classify_first('2024-04-15',[('2025-03-01',1)],'2024-12-31',months=12)
    assert r['event']==0 and r['reason']=='administrative_censoring'


def test_incomplete_2025_followup_is_not_a_negative():
    r=classify_first('2025-10-01',[],'2026-09-03',months=12)
    assert r['event']==0 and r['reason']=='administrative_censoring'
    time=calendar_elapsed('2025-10-01',r['stop'])
    w,known,_=endpoint_weights([5,time,12],[1,0,0],12)
    np.testing.assert_array_equal(known,[True,False,True])
    assert w[1]==0 and w[2]>1
