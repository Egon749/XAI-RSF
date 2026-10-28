"""Recover a post-ST announcement-clock cohort from existing audited sources."""
from pathlib import Path
import argparse
import datetime
import json
import shutil
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha, write, horizons, classify_first

AUDIT = Path('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1')
LEGACY = Path('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025')
OLD = Path('/home/oliver/LLMworkspace/PhD_STARST_Prognosis/runs')
INDUSTRY = Path('/home/oliver/LLMworkspace/csmar_financial_diagnosis_audit_20261012/industry_raw_cache.parquet')
MARKET = LEGACY/'results/market_stage/2450/task-0/output'


def name_state(s):
    s = str(s).upper()
    return 'D' if '*ST' in s else 'B' if 'ST' in s else 'T' if '退' in s else 'A'


def market_features(g, cal, origin):
    """Same frozen 24 financial-market inputs, with entry announcement cutoff."""
    dt = g.Trddt.to_numpy(dtype='datetime64[D]')
    t = np.datetime64(pd.Timestamp(origin).date())
    end = np.searchsorted(dt, t, side='left'); ci = np.searchsorted(cal, t, side='left')
    z = {'market_latest_date': pd.Timestamp(dt[end-1]) if end else pd.NaT}
    arr = {c: g[c].to_numpy(float) for c in g if c not in ['Stkcd', 'Trddt']}
    def logpos(v): return float(np.log(v)) if np.isfinite(v) and v > 0 else np.nan
    def avg(a):
        a = a[np.isfinite(a)]
        return float(a.mean()) if len(a) else np.nan
    def sd(a):
        a = a[np.isfinite(a)]
        return float(np.std(a, ddof=1)) if len(a) >= 2 else np.nan
    for name, col in [('log_price_last','Clsprc'), ('log_total_mktcap_last','Dsmvtll'), ('log_float_mktcap_last','Dsmvosd')]:
        z[name] = logpos(arr[col][end-1]) if end else np.nan
    for n, label in [(21,'1m'), (63,'3m'), (252,'12m')]:
        start = np.searchsorted(dt, cal[max(0, ci-n)]) if ci else end
        w = {c: a[start:end] for c, a in arr.items()}
        r = w['Dretwd']; r = r[np.isfinite(r) & (r >= -1)]
        z['market_obs_'+label] = len(r)
        z['ret_cum_'+label] = float(np.prod(1+r)-1) if len(r) else np.nan
        z['ret_vol_'+label] = sd(r)
        wealth = np.r_[1., np.cumprod(1+r)]
        z['max_drawdown_'+label] = float(np.min(wealth/np.maximum.accumulate(wealth)-1)) if len(r) else np.nan
        if label in ['3m', '12m']:
            caps = w['Dsmvtll']
            z['total_mktcap_change_'+label] = float(caps[-1]/caps[0]-1) if len(caps)>1 and caps[0]>0 else np.nan
        if label in ['1m', '3m']:
            good = np.isfinite(w['Dretwd']) & (w['Dnvaltrd']>0)
            value = avg(np.abs(w['Dretwd'][good])/w['Dnvaltrd'][good])
            z['log_amihud_illiq_'+label] = float(np.log1p(value*1e9)) if np.isfinite(value) else np.nan
        if label == '3m':
            z['downside_vol_3m'] = sd(r[r<0]); v = float(np.quantile(r,.05)) if len(r) else np.nan
            z['var_5pct_3m'] = v; z['es_5pct_3m'] = avg(r[r<=v])
            z['extreme_negative_days_3m'] = float((r<=-.05).sum()) if len(r) else np.nan
            valid = (w['Clsprc']>0) & (w['Dsmvosd']>0)
            z['turnover_mean_3m'] = avg(w['Dnshrtrd'][valid]*w['Clsprc'][valid]/(1000*w['Dsmvosd'][valid]))
            z['log_amount_mean_3m'] = logpos(avg(w['Dnvaltrd']))
            good = w['Clsprc']>0
            z['intraday_range_mean_3m'] = avg((w['Hiprc'][good]-w['Loprc'][good])/w['Clsprc'][good])
            values = w['Dnvaltrd']; values = values[np.isfinite(values)]
            z['zero_amount_ratio_3m'] = float((values==0).mean()) if len(values) else np.nan
            z['market_coverage_3m'] = len(r)/min(63,ci) if ci else np.nan
    return z


def main(run):
    protocol_path = Path(__file__).with_name('frozen_protocol.json')
    plan = json.loads(protocol_path.read_text())
    shutil.copy2(protocol_path, run/'audit/frozen_protocol.json')
    write(run/'audit/pre_fit_protocol_freeze.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(protocol_path), statement='Frozen before any new model fit or prediction. Existing aggregate outcomes previously exposed; no blind-test claim.'))
    signatures = {}
    def source(p):
        p = Path(p); signatures[str(p)] = sha(p); return p
    scope_path = source(OLD/'method_freeze_input_build_v2/evidence/source_scope_confirmation.json')
    scope = json.loads(scope_path.read_text())
    for key in ['primary_source', 'dictionary', 'parsed_source']:
        source(scope[key]['path']); assert signatures[scope[key]['path']] == scope[key]['sha256']
    assert scope['primary_snapshot_execution_bound'] == '2026-09-03'
    # This is an inherited accepted complete snapshot, not a fabricated array of dates.
    C = pd.Timestamp(scope['primary_snapshot_execution_bound'])
    ep = source(OLD/'phase1_data_freeze_v1/03_event_history.csv')
    events = pd.read_csv(ep, dtype=str, keep_default_na=False)
    raw = pd.read_json(scope['parsed_source']['path'], lines=True, dtype={'Symbol':str})
    pri = events.loc[events.source_role.eq('latest_primary')]
    assert len(pri) == len(raw) == 4158
    raw['key'] = raw.Symbol+'|'+raw.Chgtype+'|'+raw.Execudt
    pri = pri.assign(key=pri.Symbol+'|'+pri.Chgtype+'|'+pri.Execudt)
    assert raw.key.is_unique and pri.key.is_unique
    check = pri.set_index('key').sort_index(); raw = raw.set_index('key').sort_index().fillna('')
    for col in ['Symbol','Chgtype','Annoudt','Execudt','Content']:
        assert np.array_equal(check[col].astype(str), raw[col].astype(str)), col
    reviews_path = source(OLD/'phase1_data_freeze_v1/03_case_review.csv')
    reviews = pd.read_csv(reviews_path, dtype=str, keep_default_na=False).set_index('event_id').to_dict('index')
    mapping = json.loads(source(AUDIT/'assembly/group_mapping.json').read_text())
    events['group'] = events.Symbol.map(lambda x:mapping.get(x,x))
    events['announcement'] = pd.to_datetime(events.adjudicated_announcement_date.where(
        events.adjudicated_announcement_date.ne(''), events.Annoudt), errors='coerce')
    events['execution'] = pd.to_datetime(events.Execudt, errors='raise')
    assert events.announcement.notna().all(), 'Missing announcements require upstream recovery, never execution fallback'
    primary_codes = set(pri.Symbol)
    decisions = pd.DataFrame(json.loads(source(AUDIT/'audit_v4_observed_first/event_decisions.json').read_text()))
    decisions = decisions.loc[decisions.observed_first_accepted].copy()
    base = pd.DataFrame(json.loads(source(AUDIT/'input/quarterly_first_financial_audit_12m.json').read_text())['events'])
    base['event_id'] = base.group_key+'_'+base.date
    decisions = decisions.merge(base[['event_id','group_key','type','category','codes']], on='event_id', validate='one_to_one')
    manual = json.loads(source(AUDIT/'input/review160_event_decisions.json').read_text())['decisions']
    manual = {x['event_id']:x['decision'] for x in manual}
    assert all(x.category=='financial_audit_explicit' or manual.get(x.event_id)=='include' for x in decisions.itertuples())
    event_groups = {k:g.sort_values(['announcement','execution','Chgtype','Symbol']) for k,g in events.groupby('group')}
    entries = []
    for x in decisions.itertuples():
        g = event_groups[x.group_key]
        matched = g.loc[g.Execudt.eq(x.corrected_event_date)&g.Chgtype.eq(x.type)]
        assert len(matched) and matched.announcement.nunique()==1
        t0 = matched.announcement.iloc[0]
        entries.append(dict(code=x.code, group=x.group_key, legacy_event_id=x.event_id,
            origin=t0, execution=pd.Timestamp(x.corrected_event_date), entry_year=t0.year, entry_type=x.type,
            entry_source_ids=';'.join(matched.event_id), entry_announcement_override=bool(matched.adjudicated_announcement_date.ne('').any()),
            prior_history_daily_complete=bool(x.continuous_daily_first_supported),
            primary_code_present=x.code in primary_codes))
    d = pd.DataFrame(entries); d = d.loc[d.entry_year.between(2012,2024)].copy()
    assert d.group.is_unique and d.code.is_unique
    matrix = json.loads(source(LEGACY/'input/model_matrix.json').read_text())
    names = matrix['features']; assert len(names)==72
    finance_names = names[:48]; market_names = names[48:]
    f = pd.read_parquet(source(LEGACY/'prepared/financial_facts.parquet'))
    assert not f.duplicated(['Stkcd','Accper']).any()
    assert np.array_equal(pd.to_datetime(f.available).to_numpy(),
        f[['ann','correction_0','correction_1','correction_2']].apply(pd.to_datetime).max(axis=1).to_numpy())
    f['report_date'] = pd.to_datetime(f.Accper); f['available'] = pd.to_datetime(f.available)
    f['prev_available'] = pd.to_datetime(f.prev_available)
    industry = pd.read_parquet(source(INDUSTRY)).fillna('')
    industry = industry.loc[industry.Typrep.eq('A')].copy()
    assert not industry.duplicated(['Stkcd','Accper']).any()
    industry['industry_field'] = np.where(pd.to_datetime(industry.Accper)<pd.Timestamp('2023-05-01'),'Indcd','Indcd1')
    industry['industry_code'] = np.where(industry.industry_field.eq('Indcd'),industry.Indcd,industry.Indcd1)
    f = f.merge(industry[['Stkcd','Accper','industry_field','industry_code']], on=['Stkcd','Accper'], how='left', validate='one_to_one')
    fg = {k:g.sort_values('report_date') for k,g in f.groupby('Stkcd')}
    align = []; values = []; clocks = []
    for row in d.itertuples():
        g = fg.get(row.code, f.iloc[:0]); g = g.loc[g.available.lt(row.origin)&g.report_date.lt(row.origin)]
        item = dict(code=row.code, report_date=pd.NaT, financial_available=pd.NaT, prev_available=pd.NaT,
                    industry_code='', industry_field='', financial_age=np.nan, financial_period_age=np.nan)
        v = np.full(48,np.nan); dates = np.full(48,np.datetime64('NaT','D'),dtype='datetime64[D]')
        if len(g):
            last = g.iloc[-1]; item.update(report_date=last.report_date, financial_available=last.available,
                prev_available=last.prev_available, industry_code=last.industry_code if pd.notna(last.industry_code) else '',
                industry_field=last.industry_field if pd.notna(last.industry_field) else '',
                financial_age=(row.origin-last.available).days, financial_period_age=(row.origin-last.report_date).days)
            v = last[finance_names].to_numpy(float,copy=True); v[~np.isfinite(v)] = np.nan
            for j,n in enumerate(finance_names):
                date = last.available
                if n.endswith(('_growth','_change')):
                    if pd.isna(last.prev_available) or last.prev_available>=row.origin:
                        v[j]=np.nan; date=pd.NaT
                    else:date=max(date,last.prev_available)
                dates[j] = np.datetime64(date.date(),'D') if pd.notna(date) else np.datetime64('NaT','D')
        align.append(item); values.append(v); clocks.append(dates)
    d = d.merge(pd.DataFrame(align), on='code', validate='one_to_one')
    ic = d.industry_code
    d['industry_class'] = np.select([ic.str.fullmatch(r'C\d{1,3}'),ic.str.fullmatch(r'J\d{1,3}'),ic.str.fullmatch(r'[A-S]\d{1,3}')],
                                    ['manufacturing','financial','other_nonfinancial'], default='unclassified')
    d['selected'] = d.industry_class.isin(['manufacturing','other_nonfinancial'])
    d.to_parquet(run/'private/entry_selection_PRIVATE.parquet',index=False)
    flow = d.groupby(['entry_year','industry_class']).size().reset_index(name='companies')
    flow.to_csv(run/'reports/sample_selection.csv',index=False)
    sel = d.selected.to_numpy(); Xfin=np.asarray(values)[sel]; clocks=np.asarray(clocks)[sel]
    d=d.loc[sel].reset_index(drop=True)
    assert d.primary_code_present.all(), 'Complete-scope evidence required for each selected code'
    assert (np.isfinite(Xfin) <= ~np.isnat(clocks)).all()
    assert not ((~np.isnat(clocks)) & (clocks>=d.origin.to_numpy(dtype='datetime64[D]')[:,None])).any()
    calendar=np.load(source(MARKET/'market_calendar.npy'),allow_pickle=False)
    codes=d.code.tolist(); pieces=[]
    for p in sorted(MARKET.glob('market_shard_*.parquet')):
        source(p); pieces.append(pd.read_parquet(p,filters=[('Stkcd','in',codes)]))
    market=pd.concat(pieces,ignore_index=True).sort_values(['Stkcd','Trddt'])
    assert not market.duplicated(['Stkcd','Trddt']).any()
    market.to_parquet(run/'private/market_source_subset_PRIVATE.parquet',index=False)
    mg={k:g for k,g in market.groupby('Stkcd')}
    marketrows=[dict(code=z.code,**market_features(mg.get(z.code,market.iloc[:0]),calendar,z.origin)) for z in d.itertuples()]
    mf=pd.DataFrame(marketrows); assert np.array_equal(mf.code,d.code)
    Xmarket=mf[market_names].to_numpy(float,copy=True); Xmarket[~np.isfinite(Xmarket)]=np.nan
    d['market_available']=pd.to_datetime(mf.market_latest_date)
    assert ((d.market_available<d.origin)|d.market_available.isna()).all()
    assert not (np.isfinite(Xmarket)&d.market_available.isna().to_numpy()[:,None]).any()
    # Independently reproduce existing quarterly calculations using identical daily rows.
    parity=0
    for p in sorted((LEGACY/'results/market_features').glob('*/*/task-*/output/market_features.parquet')):
        prior=pd.read_parquet(p,filters=[('code','in',codes)])
        prior=prior.loc[prior.quarter.eq('2020Q4')].head(2)
        if len(prior):source(p)
        for row in prior.itertuples():
            computed=market_features(mg[row.code],calendar,pd.Period(row.quarter,freq='Q').end_time.normalize())
            np.testing.assert_allclose([computed[n] for n in market_names],[getattr(row,n) for n in market_names],rtol=1e-12,atol=1e-12,equal_nan=True)
            parity+=1
    assert parity>0
    q=np.column_stack([np.minimum(d.financial_period_age.fillna(730)/365,2),
                       np.minimum(d.financial_age.fillna(730)/365,2),
                       1-mf.market_coverage_3m.fillna(0).clip(0,1)])
    X=np.column_stack([Xfin,Xmarket,q]);assert X.shape==(len(d),75) and not np.isinf(X).any()
    feature_names=names+['fin_period_age_scaled','fin_publication_age_scaled','market_missing_coverage']
    pd.DataFrame(dict(index=range(75),name=feature_names,missing_rate_train=np.isnan(X[d.entry_year.le(2022)]).mean(0))).to_csv(run/'reports/feature_manifest.csv',index=False)
    labels=[]; endpoint_ledger=[]; chain_ledger=[]
    for row in d.itertuples():
        g=event_groups[row.group]; after=g.loc[g.announcement.ge(row.origin)&g.announcement.le(C)].copy()
        # Event order is announcement order; final X avoids requiring an invented missing T.
        terminal=after.loc[after.Chgtype.str.endswith(('A','X'))]
        endpoints=[(z.announcement,1 if z.Chgtype.endswith('X') else 2) for z in terminal.itertuples()]
        for z in terminal.itertuples():
            endpoint_ledger.append(dict(group=row.group,source_event_id=z.event_id,announcement=z.announcement,
                execution=z.execution,type=z.Chgtype,raw_before_state=z.Chgtype[0],name_before_state=name_state(z.ShortNamebc),
                native_chain_discontinuity=z.chain_discontinuity,source_path=z.source_path,raw_record_sha256=z.raw_record_sha256))
        cap=min(C,pd.Timestamp('2022-12-31')) if row.entry_year<=2022 else C
        label=classify_first(row.origin,endpoints,cap)
        first_end=label['stop']; gaps=after.loc[after.announcement.le(first_end)&after.chain_discontinuity.eq('1')]
        unresolved_gap=False
        for z in gaps.itertuples():
            reused=reviews.get(z.event_id,{}).get('decision','').startswith('resolved')
            # A generic AX pre-state with an explicitly ST name is not a recorded A recovery;
            # a missing T before observed X is not a missing endpoint under this X protocol.
            final_x_supported=(z.Chgtype.endswith('X') and
                (name_state(z.ShortNamebc) in ['B','D','T'] or name_state(z.ShortNameac) in ['B','D','T']) and
                ('终止上市' in z.Content or z.all_cause_exit_confirmed=='1'))
            nonterminal_only=z.Chgtype[-1] in ['B','D','C','S','T'] and z.Chgtype[0] in ['B','D','C','S','T']
            resolved=reused or final_x_supported or nonterminal_only
            unresolved_gap |= not resolved
            chain_ledger.append(dict(group=row.group,source_event_id=z.event_id,announcement=z.announcement,
                type=z.Chgtype,reused_adjudication=reused,explicit_final_X_with_ST_name=final_x_supported,
                intermediate_only=nonterminal_only,resolved_for_first_A_X=resolved,
                no_synthetic_recovery_or_T=True,source_path=z.source_path,raw_record_sha256=z.raw_record_sha256))
        label['label_eligible']=not label['conflict'] and not unresolved_gap
        label['chain_gap_present']=bool(len(gaps)); label['unresolved_chain_gap']=unresolved_gap
        label['source_coverage_end']=C; label['effective_followup_cap']=cap
        label['entry_year']=row.entry_year
        labels.append(label)
    labels=pd.DataFrame(labels); d=pd.concat([d,labels.drop(columns='entry_year')],axis=1)
    d['sample_id']=d.code+'_'+d.origin.dt.strftime('%Y-%m-%d')
    d['h12']=[horizons(t)[0] for t in d.origin];d['h24']=[horizons(t)[1] for t in d.origin]
    d['split']=np.where(d.entry_year.le(2022),'train',d.entry_year.astype(str))
    assert not d.duplicated('group').any()
    pd.DataFrame(endpoint_ledger).to_parquet(run/'private/endpoint_ledger_PRIVATE.parquet',index=False)
    pd.DataFrame(chain_ledger).to_parquet(run/'private/chain_diagnostics_PRIVATE.parquet',index=False)
    d.to_parquet(run/'private/cohort_PRIVATE.parquet',index=False)
    train=d.split.eq('train')&d.label_eligible&d.time.gt(0);test=d.entry_year.isin([2023,2024])
    assert not set(d.loc[train,'group'])&set(d.loc[test,'group'])
    assert (d.loc[train,'stop']<pd.Timestamp(plan['fit_information_cutoff_exclusive'])).all()
    for role,mask in [('train',train),('eval',test)]:
        dd=d.loc[mask].reset_index(drop=True)
        np.savez_compressed(run/'input'/f'{role}_PRIVATE.npz',X=X[mask],time=dd.time.to_numpy(float),
            event=dd.event.to_numpy(int),groups=dd.group.to_numpy(str),sample_ids=dd.sample_id.to_numpy(str),
            year=dd.entry_year.to_numpy(int),h12=dd.h12.to_numpy(int),h24=dd.h24.to_numpy(int),
            label_eligible=dd.label_eligible.to_numpy(bool),chain_gap_present=dd.chain_gap_present.to_numpy(bool))
        dd.to_parquet(run/'input'/f'{role}_metadata_PRIVATE.parquet',index=False)
    countrows=[]
    for s,g in d.groupby('split'):
        for months in [12,24]:
            h=g[f'h{months}'];known=g.label_eligible&((g.time>=h)|((g.event>0)&(g.time<=h)))
            hit=known&(g.event==1)&(g.time<=h);rec=known&(g.event==2)&(g.time<=h)
            countrows.append(dict(split=s,horizon_months=months,companies=len(g),termination_events=int(hit.sum()),
                recovery_events=int(rec.sum()),known_no_first_event=int((known&~hit&~rec).sum()),unknown_or_censored=int((~known).sum()),
                chain_gap_records=int(g.chain_gap_present.sum()),unresolved_chain_records=int(g.unresolved_chain_gap.sum())))
    pd.DataFrame(countrows).to_csv(run/'reports/cohort_counts.csv',index=False)
    diagnostics=dict(status=plan['status'],selected_companies=len(d),training_companies=int(train.sum()),
        training_excluded=int((d.split.eq('train')&~train).sum()),evaluation_companies=int(test.sum()),
        entry_candidates_before_industry_filter=int(flow.companies.sum()),industry_selection=flow.to_dict('records'),
        counts=countrows,entry_announcement_year_differs_execution=int((d.entry_year!=d.execution.dt.year).sum()),
        entry_announcement_overrides=int(d.entry_announcement_override.sum()),
        latest_training_observed_end=str(d.loc[train,'stop'].max()),market_quarter_parity_records=parity,
        input_columns=75,financial_fields=48,market_fields=24,quality_fields=3,
        unknown_financial_clock=int(d.financial_available.isna().sum()),unknown_market_clock=int(d.market_available.isna().sum()),
        original_versions_certified=False,industry_versions_certified=False,coverage_basis=scope,
        date_basis='announcement only; audited overrides retained',training_outcomes_after_cutoff_used=0,
        original_source_sha256=signatures)
    assert all(sha(p)==digest for p,digest in signatures.items()),'Source changed'
    write(run/'audit/data_alignment.json',diagnostics)
    plan.update(feature_names=feature_names,source_counts=countrows)
    plan['source_hashes']={p.name:sha(p) for p in (run/'input').iterdir() if p.is_file()}
    write(run/'input/plan.json',plan)
    print(json.dumps({k:diagnostics[k] for k in ['selected_companies','training_companies','evaluation_companies','counts','entry_announcement_year_differs_execution','market_quarter_parity_records']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
