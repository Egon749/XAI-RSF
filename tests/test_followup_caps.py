import pandas as pd
import pytest
from experiments.post_st_followup_cap_v1.labels import rebuild
from experiments.post_st_next_open_info_v1.prepare import label_record


def events(items):
    return pd.DataFrame([dict(announcement=pd.Timestamp(d),Chgtype=k,chain_discontinuity='0',
        event_id=str(i),ShortNamebc='*ST示意',ShortNameac='示意',Content='',all_cause_exit_confirmed='0')
        for i,(d,k) in enumerate(items)],columns=['announcement','Chgtype','chain_discontinuity',
            'event_id','ShortNamebc','ShortNameac','Content','all_cause_exit_confirmed']).astype({'announcement':'datetime64[ns]'})

ROW=dict(origin=pd.Timestamp('2020-02-28'),entry_announcement=pd.Timestamp('2020-02-27'))

def test_caps_recover_later_event_without_looking_beyond_cutoff():
    e=events([('2022-08-01','DX'),('2024-05-01','DA')])
    a=rebuild(ROW,e,{},'2024-01-01',12);b=rebuild(ROW,e,{},'2024-01-01',24)
    c=rebuild(ROW,e,{},'2024-01-01',36);d=rebuild(ROW,e,{},'2024-01-01',None)
    assert a['stop']==pd.Timestamp('2021-02-28') and a['event']==0
    assert b['stop']==pd.Timestamp('2022-02-28') and b['event']==0
    assert c['stop']==d['stop']==pd.Timestamp('2022-08-01') and c['event']==d['event']==1
    assert rebuild(ROW,e,{},'2022-01-01',None)['event']==0
    assert rebuild(ROW,e,{},'2022-01-01',None)['stop']==pd.Timestamp('2021-12-31')

@pytest.mark.parametrize('months',[12,24,36,None])
def test_future_change_invariance_and_first_competing_event(months):
    past=events([('2020-09-01','DA')]);future=events([('2020-09-01','DA'),('2024-01-01','DX')])
    assert rebuild(ROW,past,{},'2024-01-01',months)==rebuild(ROW,future,{},'2024-01-01',months)
    assert rebuild(ROW,past,{},'2024-01-01',months)['event']==2

def test_legacy24_identity_and_boundary_event():
    for e in [events([]),events([('2022-02-28','DX')]),events([('2021-01-01','DA')])]:
        got=rebuild(ROW,e,{},'2024-01-01',24)
        old=label_record(ROW,e,{},pd.Timestamp('2023-12-31'))
        assert got==old
    assert rebuild(ROW,events([('2022-02-28','DX')]),{},'2024-01-01',24)['event']==1

def test_conflict_and_unresolved_gap_are_not_silent_exclusions():
    with pytest.raises(ValueError,match='Simultaneous'):
        rebuild(ROW,events([('2021-01-01','DA'),('2021-01-01','DX')]),{},'2024-01-01',None)
    e=events([('2022-08-01','AZ')]);e.loc[0,'chain_discontinuity']='1'
    with pytest.raises(ValueError,match='chain gap'):
        rebuild(ROW,e,{},'2024-01-01',None)
