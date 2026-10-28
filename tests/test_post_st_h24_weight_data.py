import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import classify_first,calendar_elapsed,horizon_predictions
from experiments.post_st_train2024_h12_v1.evaluate import metric


def test_h24_keeps_second_year_events_but_respects_calendar_fit_cutoff():
    origin=pd.Timestamp('2021-04-30');end=[(pd.Timestamp('2022-10-30'),1)]
    h12=classify_first(origin,end,'2022-12-31',months=12)
    h24=classify_first(origin,end,'2022-12-31',months=24)
    frozen=classify_first(origin,end,'2021-12-31',months=24)
    assert h12['event']==0 and h12['reason']=='complete_horizon'
    assert h24['event']==1 and calendar_elapsed(origin,h24['stop'])==18
    assert frozen['event']==0 and frozen['reason']=='administrative_censoring'
    recovery_first=classify_first(origin,[(pd.Timestamp('2022-04-30'),2),*end],'2022-12-31',months=24)
    assert recovery_first['event']==2


def test_h24_leap_day_predictions_and_endpoint_labels_are_not_h12():
    origins=[pd.Timestamp('2023-04-30'),pd.Timestamp('2024-04-30')]
    horizons=np.array([(o+pd.DateOffset(months=24)-o).days for o in origins])
    np.testing.assert_array_equal(horizons,[731,730])
    for o in origins:assert calendar_elapsed(o,o+pd.DateOffset(months=24))==24
    cif=np.array([[[.01,.02],[.01,.02],[.3,.2],[.4,.2]],[[.02,.03],[.02,.03],[.2,.4],[.2,.4]]])
    risk=horizon_predictions(cif,[365,366,730,731],horizons)[:,0]
    np.testing.assert_array_equal(risk,[.4,.2])
    result,w,y,known=metric(np.array([18.,24.]),np.array([1,0]),np.full(2,24.),risk,1,np.ones(2,bool))
    assert known.all() and y.tolist()==[True,False] and result['auc']==1 and result['ap']==1
    assert result['brier']==.2
