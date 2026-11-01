import numpy as np
import pandas as pd
from experiments.post_st_group_selection_v1.selection import *

def test_future_events_do_not_count():
    m=pd.DataFrame({'origin':pd.to_datetime(['2019-02-01','2020-02-01']), 'stop':pd.to_datetime(['2022-01-01','2023-01-01'])})
    d={'event':np.array([1,1]),'groups':np.array(['a','b'])}
    r=support_at(d,m,2021)
    assert all(x['targets']==0 for x in r)
    assert all(x['training_targets']==0 for x in r)

def test_unique_support_not_bootstrap_multiplicity():
    m=pd.DataFrame({'origin':pd.to_datetime(['2019-02-01','2020-02-01']), 'stop':pd.to_datetime(['2019-04-01','2020-04-01'])})
    d={'event':np.array([1,1]),'groups':np.array(['a','b'])}
    assert support_at(d,m,2023,np.array(['a']*20+['b']*20))==support_at(d,m,2023)

def test_confirmation_cannot_use_future_maturity():
    rows=[dict(horizon=24,mature=True,unknown=0,training_targets=2,targets=6,controls=20,cohort=v) for v in [2020,2021]]
    assert discovery_gate(rows)['passed']
    keep,reason=choose({'passed':False,'reason':'NO_HONEST_FORWARD_CONFIRMATION'},[{'keep':[0,1]}])
    assert keep==FULL and reason=='NO_HONEST_FORWARD_CONFIRMATION'

def test_interaction_protection_rejects_flattening_and_reversal():
    a=np.arange(20.)-9.5
    assert interaction_guard(a,a)['passed']
    assert not interaction_guard(a,.1*a)['passed']
    assert not interaction_guard(a,-a)['passed']

def test_candidates_preserve_audit_and_only_one_group():
    rng=np.random.default_rng(5);x=rng.normal(size=(100,83));x[:,1]=x[:,0];x[:,3]=x[:,2]
    d={'X':x,'groups':np.array([str(i) for i in range(100)])}
    m=pd.DataFrame({'origin':pd.to_datetime(['2018-01-01']*100)})
    groups,sets=grouped_candidates(d,m,2024)
    assert len(sets)==3
    for keep in sets:
        assert AUDIT<=set(keep)
        assert len(set(FULL)-set(keep))<=1
