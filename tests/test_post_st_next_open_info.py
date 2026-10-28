import numpy as np
import pandas as pd
import pytest
from experiments.post_st_next_open_info_v1.prepare import next_open, financial_record, label_record, announcement_values


def test_next_exchange_open_does_not_wait_for_issuer_to_trade():
    calendar = np.array(['2024-04-30','2024-05-06','2024-05-07'], dtype='datetime64[D]')
    assert next_open('2024-04-30', calendar) == pd.Timestamp('2024-05-06')
    assert next_open('2024-05-04', calendar) == pd.Timestamp('2024-05-06')
    with pytest.raises(ValueError): next_open('2024-05-07', calendar)


def test_future_revision_and_growth_components_cannot_be_backdated():
    f = pd.DataFrame(dict(report_date=pd.to_datetime(['2023-09-30','2023-12-31']),
        available=pd.to_datetime(['2023-10-30','2024-06-01']),
        prev_available=pd.to_datetime(['2025-01-01','2023-04-01']), ratio=[1.,9.], revenue_growth=[2.,8.]))
    x, clocks, m = financial_record(f, ['ratio','revenue_growth'], '2024-05-06')
    assert m['report_date'] == pd.Timestamp('2023-09-30') and x[0] == 1 and np.isnan(x[1])
    assert np.isnat(clocks[1])
    f.loc[1,'available'] = pd.Timestamp('2024-04-30')
    x, clocks, m = financial_record(f, ['ratio','revenue_growth'], '2024-05-06')
    np.testing.assert_array_equal(x,[9.,8.]); assert (clocks < np.datetime64('2024-05-06')).all()


def test_first_recovery_before_shifted_origin_is_not_dropped_then_replaced_by_delisting():
    row = dict(entry_announcement=pd.Timestamp('2024-04-30'), origin=pd.Timestamp('2024-05-06'))
    events = pd.DataFrame(dict(announcement=pd.to_datetime(['2024-05-02','2024-07-01']), Chgtype=['DA','AX']))
    v = label_record(row, events, {}, pd.Timestamp('2025-12-31'))
    assert not v['label_eligible'] and v['time']==0 and v['reason']=='endpoint_at_or_before_new_origin'


def test_new_clock_caps_labels_before_fit_cutoff_and_uses_calendar_months():
    row = dict(entry_announcement=pd.Timestamp('2022-02-25'), origin=pd.Timestamp('2022-02-28'))
    events = pd.DataFrame(dict(announcement=pd.to_datetime(['2023-01-02']), Chgtype=['DX'], chain_discontinuity=['0']))
    v = label_record(row, events, {}, pd.Timestamp('2022-12-31'))
    assert v['event']==0 and v['stop']==pd.Timestamp('2022-12-31') and v['label_eligible']
    v = label_record(row, events.iloc[:0], {}, pd.Timestamp('2026-09-03'))
    assert v['stop']==pd.Timestamp('2024-02-28') and v['time']==730


def test_entry_audit_conflict_remains_unknown_not_standard_opinion():
    row = dict(entry_type='AD', entry_source_ids='a;b', origin=pd.Timestamp('2023-04-29'))
    events = pd.DataFrame(dict(event_id=['a','b'], adjudicated_announcement_date=['',''], Annoudt=['2023-04-29']*2,
        Chgtype=['AD']*2, ShortNameac=['*ST甲']*2, Chgreas=['01']*2,
        TypeAuditOpinbc=['标准无保留意见','无法表示意见'], ISSustManaNonStandExpl=['',''])).set_index('event_id')
    x, meta = announcement_values(row, events)
    assert x[0]==1 and np.isnan(x[1:]).all() and not meta['opinion_known']
