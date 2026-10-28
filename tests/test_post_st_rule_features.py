import pandas as pd
import pytest
from experiments.post_st_rule_features_v1.prepare import recover_entry
from experiments.post_st_rule_features_v1.inspect_raw import RAW_COLUMNS, parse_original, terminal_ground


def example():
    row=dict(entry_type='AD',origin='2024-04-20',entry_source_ids='entry')
    events=pd.DataFrame([dict(event_id='entry',Chgtype='AD',Annoudt='2024-04-20',
        adjudicated_announcement_date='',Execudt='2024-04-22',ShortNameac='*ST测试',
        Chgreas='01,03',TypeAuditOpinbc='无法表示意见',ISSustManaNonStandExpl='1')]).set_index('event_id')
    return row,events


def test_entry_type_uses_announcement_and_is_independent_of_later_outcomes():
    row,events=example();a=recover_entry(row,events)
    row.update(event=1,stop='2025-03-01')
    b=recover_entry(row,events)
    assert a==b and a['entry_is_star_st']==1 and a['available_at']==pd.Timestamp('2024-04-20')


def test_unknown_or_later_warning_is_not_substituted_for_entry():
    row,events=example();row['entry_source_ids']='later'
    with pytest.raises(ValueError,match='Missing entry'):recover_entry(row,events)
    row['entry_source_ids']='entry';events.loc['entry','Annoudt']='2024-05-20'
    with pytest.raises(ValueError,match='date mismatch'):recover_entry(row,events)


def test_warning_name_conflicts_block_export_and_ST_is_not_star_ST():
    row,events=example();events.loc['entry','ShortNameac']='ST测试'
    with pytest.raises(ValueError,match='conflicts'):recover_entry(row,events)
    row['entry_type']='AB';events.loc['entry','Chgtype']='AB'
    assert recover_entry(row,events)['entry_is_star_st']==0


def test_raw_parser_preserves_embedded_newlines_quotes_and_all_records(tmp_path):
    import hashlib
    header=','.join('"'+c+'"' for c in RAW_COLUMNS)+'\r\n'
    values=['000001','AD','03','测试','*ST测试','2024-04-20','2024-04-22',
            '审计否定','第一行含未转义"引号,及逗号\n第二行','无法表示意见','1']
    record=('"'+'","'.join(values)+'"\r\n').encode()
    other=record.replace(b'000001',b'000002',1)
    path=tmp_path/'source.csv';path.write_bytes(b'\xef\xbb\xbf'+header.encode()+record+other)
    data=parse_original(path)
    assert len(data)==2 and data.Symbol.tolist()==['000001','000002']
    assert data.Content.tolist()==[values[8],values[8]]
    assert data.raw_record_sha256.iloc[0]==hashlib.sha256(record).hexdigest()


def test_terminal_ground_keeps_unknown_and_does_not_use_historical_loss_as_cause():
    assert terminal_ground('历史年度亏损，未能在法定期限内披露年度报告。')=='failure_to_file_annual_report'
    assert terminal_ground('本所决定终止其股票上市。')=='not_explicit_in_available_terminal_text'
    with pytest.raises(ValueError,match='explicit review'):
        terminal_ground('收盘价均低于1元，收盘总市值均低于3亿元。')
