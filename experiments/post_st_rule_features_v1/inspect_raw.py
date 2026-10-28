"""Read original source bytes and audit the existing cohort; no new model fits."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from experiments.post_st_prognosis_v1.common import sha, write, arrays

RAW_COLUMNS=['Symbol','Chgtype','Chgreas','ShortNamebc','ShortNameac','Annoudt',
             'Execudt','Chgrsdis','Content','TypeAuditOpinbc','ISSustManaNonStandExpl']


def parse_original(path):
    """Losslessly bound free-text Content; ordinary CSV parsing fails on this export."""
    data=path.read_bytes()
    starts=[m.start() for m in re.finditer(rb'(?m)^"\d{6}","[ABCDSTX]{2}",',data)]
    assert starts and data[:starts[0]].decode('utf-8-sig').strip()==','.join('"'+x+'"' for x in RAW_COLUMNS)
    rebuilt=bytearray(data[:starts[0]]);records=[]
    for i,a in enumerate(starts):
        b=starts[i+1] if i+1<len(starts) else len(data)
        raw=data[a:b];s=raw.rstrip(b'\r\n');first=s.split(b'","',8);last=first[-1].rsplit(b'","',2)
        assert len(first)==9 and len(last)==3 and first[0].startswith(b'"') and last[-1].endswith(b'"')
        v=[first[0][1:]]+first[1:8]+[last[0],last[1],last[2][:-1]]
        restored=b'"'+b'","'.join(v)+b'"'+raw[len(s):];assert restored==raw
        rebuilt.extend(restored)
        records.append(dict(zip(RAW_COLUMNS,[x.decode('utf-8') for x in v]),
                            event_id=f'P{i+1:05d}',raw_record_sha256=hashlib.sha256(raw).hexdigest()))
    assert bytes(rebuilt)==data
    return pd.DataFrame(records)


def terminal_ground(text):
    """Descriptive source-ground labels reviewed for this fixed 45-event cohort.

    These must NEVER be used as prediction inputs. Missing grounds stay unknown.
    Separate failure to file from financial figures quoted as historical context.
    """
    t=re.sub(r'\s+','',text)
    price=re.search(r'收盘价均低于(?:人民币)?(?:股票面值[（(])?1(?:\.0+)?元',t)
    cap=re.search(r'(?:收盘总市值|收盘市值)均低于(?:人民币)?[35]亿元',t)
    if price and cap:raise ValueError('Multiple transaction grounds require explicit review')
    if price:return 'price_below_one_yuan'
    if cap:return 'market_cap_threshold'
    if re.search(r'未(?:能)?在.{0,50}法定期限.{0,20}披露.{0,12}(?:年报|年度报告)',t):
        return 'failure_to_file_annual_report'
    if re.search(r'主动(?:终止上市|撤回)',t):return 'voluntary'
    if re.search(r'第(?:9\.3\.(?:7|11)|10\.3\.(?:10|11))条',t):
        return 'financial_or_audit_ground'
    return 'not_explicit_in_available_terminal_text'


def inspect(base,run):
    out=run/'direct_raw_analysis';out.mkdir(exist_ok=False)
    private=out/'private';private.mkdir();reports=out/'reports';reports.mkdir()
    sources={}
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    raw_path=source('/home/oliver/LLMworkspace/download/BDT_Trdchg.csv')
    raw=parse_original(raw_path);assert len(raw)==4158
    h=pd.read_csv(source('/home/oliver/LLMworkspace/PhD_STARST_Prognosis/runs/phase1_data_freeze_v1/03_event_history.csv'),dtype=str,keep_default_na=False)
    a=h.loc[h.source_role.eq('latest_primary')].set_index('event_id').sort_index()
    b=raw.set_index('event_id').sort_index()
    for col in RAW_COLUMNS+['raw_record_sha256']:assert np.array_equal(a[col],b[col]),col
    source('/home/oliver/LLMworkspace/download/BDT_Trdchg[DES][csv].txt')
    mapping=json.loads(source('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1/assembly/group_mapping.json').read_text())
    h['group']=h.Symbol.map(lambda x:mapping.get(x,x))
    h['announcement']=pd.to_datetime(h.adjudicated_announcement_date.where(h.adjudicated_announcement_date.ne(''),h.Annoudt))
    frames=[]
    for role in ['train','eval']:
        frames.append(pd.read_parquet(source(base/'input'/f'{role}_metadata_PRIVATE.parquet')))
    d=pd.concat(frames,ignore_index=True);hidx=h.set_index('event_id');entry=[];ends=[]
    for row in d.itertuples():
        g=hidx.loc[row.entry_source_ids.split(';')]
        assert g.announcement.eq(row.origin).all() and g.Chgtype.eq(row.entry_type).all()
        text='\n'.join(g.Content+g.supplemental_content)
        entry.append(dict(sample_id=row.sample_id,scope=row.split,opinion=';'.join(sorted(set(g.TypeAuditOpinbc))),
            going_concern=';'.join(sorted(set(g.ISSustManaNonStandExpl))),raw_cause=';'.join(sorted(set(g.Chgrsdis))),
            deducted_revenue_mentioned='营业收入' in text and '扣除' in text,
            deducted_profit_mentioned='扣除非经常性' in text or '扣非' in text,
            audit_mentioned='审计' in text,internal_control_mentioned='内控' in text or '内部控制' in text))
        if row.event==1:
            terminal=h[h.group.eq(row.group)&h.announcement.gt(row.origin)&h.announcement.le(row.stop)&h.Chgtype.str.endswith(('T','X'))]
            text='\n'.join(terminal.Chgtype+':'+terminal.Chgrsdis+':'+terminal.Content+terminal.supplemental_content)
            ends.append(dict(sample_id=row.sample_id,scope=row.split,entry_type=row.entry_type,
                ground=terminal_ground(text),source_ids=';'.join(terminal.event_id),source_text=text))
    e=pd.DataFrame(entry);z=pd.DataFrame(ends)
    e.to_parquet(private/'entry_raw_fields_PRIVATE.parquet',index=False)
    z.to_parquet(private/'termination_grounds_PRIVATE.parquet',index=False)
    assert len(z)==45
    z.groupby(['scope','ground']).size().rename('observed_events').reset_index().to_csv(reports/'termination_ground_counts.csv',index=False)
    for col in ['opinion','going_concern','raw_cause']:
        e.groupby(['scope',col]).size().rename('companies').reset_index().to_csv(reports/f'entry_{col}_counts.csv',index=False)
    e.groupby('scope')[[c for c in e if c.endswith('_mentioned')]].sum().to_csv(reports/'entry_text_coverage.csv')

    # Read the original financial announcement file, not just its prepared clock.
    facts=pd.read_parquet(source('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025/prepared/financial_facts.parquet'))
    f=facts[pd.to_datetime(facts.Accper).dt.month.eq(12)].copy()
    f['annual_year']=pd.to_datetime(f.Accper).dt.year;f=f.rename(columns={'Stkcd':'code','ann':'annual_ann','available':'annual_available'})
    d['annual_year']=d.entry_year-1
    af=d.merge(f[['code','annual_year','annual_ann','annual_available']],on=['code','annual_year'],validate='one_to_one')
    notice=pd.read_csv(source('/home/oliver/pylib/csmar/年、中、季报基本情况文件/IAR_Rept.csv'),
                       usecols=['Stkcd','Accper','Annodt','Reptyp'],dtype={'Stkcd':str})
    notice=notice[pd.to_datetime(notice.Accper).dt.month.eq(12)&notice.Stkcd.isin(d.code)].copy()
    notice['annual_year']=pd.to_datetime(notice.Accper).dt.year
    notice=notice.rename(columns={'Stkcd':'code','Annodt':'raw_annual_announcement'})
    af=af.merge(notice[['code','annual_year','raw_annual_announcement']],on=['code','annual_year'],validate='one_to_one')
    for c in ['annual_ann','annual_available','raw_annual_announcement']:af[c]=pd.to_datetime(af[c])
    assert len(af)==390 and af.annual_ann.eq(af.raw_annual_announcement).all()
    af.to_parquet(private/'annual_clock_PRIVATE.parquet',index=False)
    annual=[]
    for scope,g in af.groupby('split'):
        annual.append(dict(scope=scope,n=len(g),original_annual_notice_available=len(g),
            annual_ann_before_origin=int(g.annual_ann.lt(g.origin).sum()),annual_ann_same_day=int(g.annual_ann.eq(g.origin).sum()),
            annual_ann_after_origin=int(g.annual_ann.gt(g.origin).sum()),latest_vintage_available_after_origin=int(g.annual_available.gt(g.origin).sum()),
            selected_prior_year_annual=int((g.report_date.dt.month.eq(12)&g.report_date.dt.year.eq(g.annual_year)).sum()),
            selected_q3=int(g.report_date.dt.month.eq(9).sum()),median_financial_age_days=float(g.financial_age.median())))
    pd.DataFrame(annual).to_csv(reports/'annual_information_gap.csv',index=False)
    profit_path=source('/home/oliver/pylib/csmar/利润表/FS_Comins.csv')
    profit_schema=pd.read_csv(profit_path,nrows=0).columns.tolist()
    dictionaries={}
    for name,p in [('income','/home/oliver/pylib/csmar/利润表/FS_Comins[DES][csv].txt'),
                   ('profitability','/home/oliver/pylib/csmar/盈利能力/FI_T5[DES][xlsx].txt'),
                   ('daily','/home/oliver/pylib/csmar/日个股回报率文件20210901-20260831/TRD_Dalyr[DES][csv].txt')]:
        dictionaries[name]=source(p).read_text(encoding='utf-8-sig')

    # Directly match all 81 evaluation origins to unadjusted ORIGINAL daily rows.
    ev=frames[1].copy();ev['key']=ev.code+'|'+ev.market_available.dt.strftime('%Y-%m-%d')
    wanted=set(ev.key);found=[];root=Path('/home/oliver/pylib/csmar/日个股回报率文件20210901-20260831')
    for path in sorted(root.glob('*.csv')):
        source(path)
        for chunk in pd.read_csv(path,usecols=['Stkcd','Trddt','Clsprc','Dsmvtll','Markettype','Trdsta'],dtype={'Stkcd':str},chunksize=100000):
            chunk=chunk[chunk.Stkcd.isin(ev.code)].copy()
            if len(chunk):
                chunk['key']=chunk.Stkcd+'|'+chunk.Trddt
                chunk=chunk[chunk.key.isin(wanted)]
                if len(chunk):found.append(chunk)
    daily=pd.concat(found,ignore_index=True);assert len(daily)==81 and daily.key.is_unique
    ev=ev.merge(daily,on='key',validate='one_to_one',sort=False)
    x=arrays(source(base/'input/eval_PRIVATE.npz'))
    assert np.array_equal(x['sample_ids'],ev.sample_id.to_numpy(str))
    names=pd.read_csv(base/'reports/feature_manifest.csv').name.tolist()
    np.testing.assert_allclose(ev.Clsprc,np.exp(x['X'][:,names.index('log_price_last')]),rtol=1e-12)
    np.testing.assert_allclose(ev.Dsmvtll,np.exp(x['X'][:,names.index('log_total_mktcap_last')]),rtol=1e-12)
    ev['market_cap_yuan']=ev.Dsmvtll*1000
    ev.to_parquet(private/'raw_daily_origin_alignment_PRIVATE.parquet',index=False)
    metrics=[];distribution=[]
    for scope,g in [('2023',ev[ev.entry_year.eq(2023)]),('2024',ev[ev.entry_year.eq(2024)]),('pooled',ev)]:
        assert ((g.event>0)|(g.time>=g.h24)).all()
        y=g.event.eq(1)
        scores={'lower_raw_close_price':-g.Clsprc,'lower_raw_market_cap':-g.market_cap_yuan,
                'entry_is_star_st':g.entry_type.eq('AD').astype(float)}
        for name,p in scores.items():
            metrics.append(dict(scope=scope,score=name,n=len(g),events=int(y.sum()),
                auc=roc_auc_score(y,p),ap=average_precision_score(y,p),
                interpretation='post-exposure univariate ranking diagnostic, not a fitted/calibrated model'))
        distribution.append(dict(scope=scope,n=len(g),price_below_1_at_origin=int(g.Clsprc.lt(1).sum()),
            cap_below_3e8_at_origin=int(g.market_cap_yuan.lt(3e8).sum()),
            termination_group_median_price=float(g.loc[y,'Clsprc'].median()),
            other_outcome_group_median_price=float(g.loc[~y,'Clsprc'].median())))
    pd.DataFrame(metrics).to_csv(reports/'univariate_ranking_diagnostic.csv',index=False)
    pd.DataFrame(distribution).to_csv(reports/'raw_market_distribution.csv',index=False)
    ev.groupby(['entry_year','Markettype']).size().rename('companies').reset_index().to_csv(reports/'original_markettype_counts.csv',index=False)
    qa=dict(status='RAW_AUDIT_COMPLETE_NO_TRAINING',new_model_fits=0,raw_event_records=4158,
        raw_11_fields_and_record_hashes_match_existing_audit=True,annual_notice_matches=390,
        original_daily_rows_matched=81,raw_price_and_market_cap_match_original_inputs=True,
        market_cap_unit='CSMAR thousand yuan; multiply by 1000; A-share value excludes B/H shares',
        raw_profit_statement_column_count=len(profit_schema),
        adjusted_revenue_structured_field_in_examined_income_dictionary=False,
        nonrecurring_deducted_profit_field_in_examined_income_dictionary=False,
        terminal_reason_assignment='Explicit terminal T/X announcement text, fixed-cohort manual review; unknown retained',
        terminal_reason_is_outcome_only_never_predictor=True,
        no_claim_original_disclosure_vintages_certified=True,source_sha256=sources,
        inspect_raw_source_sha256=sha(Path(__file__)),
        artifact_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    write(reports/'raw_audit_receipt.json',qa)
    print(json.dumps({k:v for k,v in qa.items() if k not in ['source_sha256','artifact_sha256']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-run',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();inspect(a.base_run,a.run)
