import pandas as pd
from experiments.post_st_train2023_v1.prepare import relabel


def test_extended_cutoff_does_not_import_full_future_2023_labels():
    meta = pd.DataFrame(dict(group=['old','new'], origin=pd.to_datetime(['2022-05-01','2023-05-01'])))
    end = pd.DataFrame(dict(group=['old','new'], announcement=pd.to_datetime(['2023-06-01','2024-01-01']), type=['DX','DX']))
    gaps = pd.DataFrame(dict(group=[], announcement=pd.to_datetime([]), resolved_for_first_A_X=[]))
    baseline = relabel(meta.iloc[:1], end, gaps, '2023-01-01')
    assert baseline.event.tolist() == [0]
    updated = relabel(meta, end, gaps, '2024-01-01')
    assert updated.event.tolist() == [1, 0]
    assert updated.loc[1,'stop'] == pd.Timestamp('2023-12-31')
    assert updated.loc[1,'reason'] == 'administrative_censoring'
    assert updated.label_eligible.all()


def test_chain_conflict_after_first_endpoint_does_not_change_first_event():
    meta = pd.DataFrame(dict(group=['a'], origin=pd.to_datetime(['2022-05-01'])))
    end = pd.DataFrame(dict(group=['a','a'], announcement=pd.to_datetime(['2023-03-01','2023-09-01']), type=['DA','AX']))
    gaps = pd.DataFrame(dict(group=['a'], announcement=pd.to_datetime(['2023-09-01']), resolved_for_first_A_X=[False]))
    result = relabel(meta, end, gaps, '2024-01-01')
    assert result.event.tolist() == [2] and result.label_eligible.all()
