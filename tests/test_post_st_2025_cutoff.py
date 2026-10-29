import pandas as pd
from experiments.post_st_2025_shallow_parent_v1.prepare import horizon_summary
from experiments.post_st_prognosis_v1.common import horizons


def test_partial_2025_followup_cannot_validate_24_months():
    d=pd.DataFrame(dict(time=[100,343,437],event=[1,2,0],label_eligible=[True]*3,h12=[365]*3,h24=[730]*3))
    twelve=horizon_summary(d,12);twenty_four=horizon_summary(d,24)
    assert twelve['endpoint_evaluation_supported'] and twelve['unknown']==0
    assert (twelve['termination'],twelve['recovery'],twelve['known_no_first_event'])==(1,1,1)
    assert not twenty_four['endpoint_evaluation_supported']
    assert twenty_four['unknown']==1 and twenty_four['known_no_first_event']==0


def test_events_after_horizon_are_controls_not_cases_or_unknown():
    d=pd.DataFrame(dict(time=[400,360,365],event=[1,0,0],label_eligible=[True]*3,h12=[365]*3,h24=[730]*3))
    z=horizon_summary(d,12)
    assert z['termination']==0 and z['known_no_first_event']==2 and z['unknown']==1
    assert z['endpoint_evaluation_supported']
    assert horizons('2024-02-29')==[365,730]
