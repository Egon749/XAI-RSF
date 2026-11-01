import numpy as np
from experiments.post_st_drop9_transport_v1.common import horizon_support


def test_unknown_late_horizon_is_not_scored_as_negative():
    data=dict(analysis_time=np.array([4., 10., 17., 13.]),event=np.array([1, 2, 0, 0]),label_eligible=np.ones(4,bool))
    short=horizon_support(data,12);late=horizon_support(data,24)
    assert short==dict(horizon_months=12,companies=4,termination=1,recovery=1,
        known_no_first_event=2,unknown=0,followup_reaches_horizon=2,supported=True)
    assert late['unknown']==2 and late['known_no_first_event']==0 and not late['supported']
    # Full first-event ascertainment can identify an endpoint even without a
    # survivor observed at that horizon; partial follow-up cannot do so.
    data['event']=np.array([1,2,1,2])
    assert horizon_support(data,24)['supported']
    data['label_eligible'][0]=False
    assert horizon_support(data,24)['unknown']==1 and not horizon_support(data,24)['supported']
