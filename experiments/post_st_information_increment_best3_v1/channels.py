"""Link existing outcome evidence. Never include post-origin case material among fit inputs."""
import re
from experiments.post_st_rule_features_v1.inspect_raw import terminal_ground
from experiments.post_st_prognosis_v1.prepare import OLD,AUDIT
from .common import *
def channels():
    r=run()
    shifted=pd.read_parquet(history(2025)/'private/shifted_cohort_PRIVATE.parquet').set_index('sample_id')
    root=Path(read(ROOT/'experiments/post_st_rule_features_v1/execution_receipt.json')['private_run'])/'direct_raw_analysis'
    path=root/'private/termination_grounds_PRIVATE.parquet'
    assert sha(path)==read(root/'reports/raw_audit_receipt.json')['artifact_sha256']['private/termination_grounds_PRIVATE.parquet']
    reviewed=pd.read_parquet(path).set_index('sample_id')
    events=pd.read_csv(OLD/'phase1_data_freeze_v1/03_event_history.csv',dtype=str,keep_default_na=False)
    mapping=read(AUDIT/'assembly/group_mapping.json');events['group']=events.Symbol.map(lambda c:mapping.get(c,c))
    events['announcement']=pd.to_datetime(events.adjudicated_announcement_date.where(events.adjudicated_announcement_date.ne(''),events.Annoudt))
    ranks=pd.read_parquet(r/'rankings_PRIVATE.parquet').query("variant=='fixed25' and cause==1 and arm in ['M0','M2']")
    rows=[]
    for y in YEARS:
        lab=arrays(r/f'eval/{y}.npz');codeids=lab['sample_ids']
        for i,sid in enumerate(codeids):
            x=shifted.loc[sid];start=pd.Timestamp(x.origin);stop=start+pd.Timedelta(days=int(lab['time'][i]))
            if lab['event'][i]!=1:continue
            end=events[events.group.eq(x.group)&events.announcement.gt(start)&events.announcement.le(stop)&events.Chgtype.str.endswith(('X','T'))]
            oldid=x.old_sample_id;ground='not_explicit_in_available_terminal_text';provenance='not_documented';src_text='';src_ids=''
            if oldid in reviewed.index:
                z=reviewed.loc[oldid];idset=set(z.source_ids.split(';'))
                if idset.issubset(set(end.event_id)) and terminal_ground(z.source_text)==z.ground:
                    ground=z.ground;provenance='existing_review';src_text=z.source_text;src_ids=z.source_ids
            if ground=='not_explicit_in_available_terminal_text':
                # Scope to the saved final exchange/issuer termination notice. Unambiguous operative
                # grounds only; no classification from rank, entry warning, news, or low stock price.
                # Read all T/X decision notices in the actual endpoint chain. These clauses
                # were inspected against the recorded reasons, separately from any forecast.
                final=end
                texts=[' '.join(str(z[col]) for col in ['Content','supplemental_content']) for _,z in final.iterrows()]
                specific=[]
                for t in texts:
                    if not re.search('终止上市|终止.*股票上市',t):continue
                    if '收盘价均低于1元' in t and '收盘总市值均低于5亿元' in t and '上述情形属于' in t:
                        val='dual_price_and_cap'
                    elif '虚假记载' in t and ('重大违法强制退市' in t or ('行政处罚决定书' in t and '第10.5.2条' in t)):
                        val='major_violation'
                    else:val=terminal_ground(t)
                    if val!='not_explicit_in_available_terminal_text':specific.append(val)
                if len(set(specific))==1:ground=specific[0];provenance='explicit_decision_notice_in_terminal_chain';src_text='\n'.join(texts);src_ids=';'.join(final.event_id)
            category={'dual_price_and_cap':'price_below_one','major_violation':'other_known','price_below_one_yuan':'price_below_one','market_cap_threshold':'other_trading_cap','financial_or_audit_ground':'financial_audit','failure_to_file_annual_report':'other_known','voluntary':'other_known','not_explicit_in_available_terminal_text':'unknown'}[ground]
            for h in [12,24]:
                if (y==2025 and h==24) or lab['time'][i]>lab[f'h{h}'][i]:continue
                z=ranks[(ranks.sample_id==sid)&(ranks.months==h)].set_index('arm');assert set(z.index)=={'M0','M2'} and z.event.all()
                rows.append(dict(year=y,months=h,sample_id=sid,ground=category,specific_ground=ground,provenance=provenance,source_ids=src_ids,source_text=src_text,M0=int(z.at['M0','top20']),M2=int(z.at['M2','top20']),p0=z.at['M0','risk'],p2=z.at['M2','risk']))
    df=pd.DataFrame(rows);df.to_parquet(r/'terminal_channel_PRIVATE.parquet',index=False)
    df['pair_numerator_delta']=np.nan;df['pairs_per_event']=np.nan
    for (year,h),part in df.groupby(['year','months']):
        rr=ranks[(ranks.year==year)&(ranks.months==h)]
        controls={a:rr[(rr.arm==a)&~rr.event].risk.to_numpy() for a in ['M0','M2']}
        for idx,z in part.iterrows():
            def concord(risk,vals):return (risk>vals).sum()+.5*(risk==vals).sum()
            df.at[idx,'pair_numerator_delta']=concord(z.p2,controls['M2'])-concord(z.p0,controls['M0'])
            df.at[idx,'pairs_per_event']=len(controls['M0'])
    inc=pd.read_csv(D/'INCREMENTAL_VALUE.csv')
    pub=[]
    for h,block in df.groupby('months'):
        for period,d in [('pooled',block),*[(str(y),g) for y,g in block.groupby('year')]]:
            denom=d.pairs_per_event.sum()
            actual=inc[(inc.year.astype(str)==period)&(inc.months==h)&(inc.cause==1)&(inc.variant=='fixed25')&(inc.metric=='within_year_auc')&(inc.effect=='market_vs_finance')].delta.item()
            assert abs(d.pair_numerator_delta.sum()/denom-actual)<1e-12
            for ground in ['price_below_one','other_trading_cap','financial_audit','other_known','unknown','all_events']:
                g=d if ground=='all_events' else d[d.ground==ground]
                pub.append(dict(scope=period,months=h,ground=ground,all_evaluation_firms=sum(len(arrays(r/f'eval/{y}.npz')['X']) for y in (YEARS if h==12 else YEARS[:2]) if period=='pooled' or int(period)==y),
                                all_target_events=len(d),events=len(g),captured_M0=int(g.M0.sum()),captured_M2=int(g.M2.sum()),net_additional_captured=int(g.M2.sum()-g.M0.sum()),
                                within_year_auc_gain_contribution=g.pair_numerator_delta.sum()/denom if len(g)>=5 or ground=='all_events' else np.nan,mean_increment_in_predicted_risk=float((g.p2-g.p0).mean()) if len(g)>=5 else np.nan,
                                note='event counts are descriptive; no subgroup AUC; individual forecasts suppressed'))
    pd.DataFrame(pub).to_csv(D/'REGULATORY_CHANNEL_AUDIT.csv',index=False)
    counts=df[df.months==12].ground.value_counts().to_dict();assert sum(counts.values())==25
    dump(D/'REGULATORY_SOURCE_CHECK.json',dict(existing_review_sha256=sha(path),event_archive_sha256=sha(OLD/'phase1_data_freeze_v1/03_event_history.csv'),posthoc=True,independent_of_fit=True,no_category_inferred_from_price=True,unknown_retained=True))
if __name__=='__main__':channels()
