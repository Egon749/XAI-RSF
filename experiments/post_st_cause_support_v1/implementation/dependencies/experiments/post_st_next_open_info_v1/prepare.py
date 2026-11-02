"""Partial, explicitly provisional information repair without backdating values."""
from pathlib import Path
import argparse, datetime, json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizons, calendar_elapsed, classify_first
from experiments.post_st_prognosis_v1.prepare import market_features, name_state, OLD, LEGACY, MARKET, AUDIT
from experiments.post_st_rule_features_v1.prepare import recover_entry

BASE = Path('/home/oliver/MainDiskData/xai_rsf_post_st_prognosis_v1_20261027_u6l00ebk')
AUDIT_LEVELS = ['标准无保留意见', '保留意见', '否定意见', '无法表示意见', '带有解释性说明的无保留意见', '保留意见加事项段']
EXTRA_NAMES = ['entry_is_star_st', 'prior_audit_standard', 'prior_audit_qualified', 'prior_audit_adverse',
               'prior_audit_disclaimer', 'prior_audit_explanatory', 'prior_audit_qualified_matter', 'prior_audit_going_concern']


def next_open(date, calendar):
    cal = np.asarray(calendar, dtype='datetime64[D]')
    if len(cal) < 2 or not np.all(cal[1:] > cal[:-1]):
        raise ValueError('Strictly increasing exchange trading calendar required')
    i = np.searchsorted(cal, np.datetime64(pd.Timestamp(date).date()), side='right')
    if i >= len(cal):
        raise ValueError('No verified next exchange trading day')
    return pd.Timestamp(cal[i])


def financial_record(frame, names, origin):
    """No correction dates are removed; multi-period inputs use both clocks."""
    origin = pd.Timestamp(origin)
    eligible = frame[(frame.available < origin) & (frame.report_date < origin)].sort_values('report_date')
    x = np.full(len(names), np.nan)
    dates = np.full(len(names), np.datetime64('NaT', 'D'), dtype='datetime64[D]')
    meta = dict(report_date=pd.NaT, financial_available=pd.NaT, prev_available=pd.NaT)
    if len(eligible):
        last = eligible.iloc[-1]
        meta.update(report_date=last.report_date, financial_available=last.available, prev_available=last.prev_available)
        x = last[names].to_numpy(float, copy=True)
        x[~np.isfinite(x)] = np.nan
        for j, name in enumerate(names):
            date = last.available
            if name.endswith(('_growth', '_change')):
                if pd.isna(last.prev_available) or last.prev_available >= origin:
                    x[j] = np.nan
                    continue
                date = max(date, last.prev_available)
            dates[j] = np.datetime64(date.date(), 'D')
    return x, dates, meta


def announcement_values(row, events):
    evidence = recover_entry(row, events)
    opinions = set(evidence['raw_prior_year_audit_opinions'].split(';')) - {''}
    flags = set(evidence['raw_going_concern_flags'].split(';')) - {''}
    onehot = np.full(6, np.nan)
    if len(opinions) == 1 and next(iter(opinions)) in AUDIT_LEVELS:
        onehot = np.array([float(x in opinions) for x in AUDIT_LEVELS])
    concern = float(next(iter(flags))) if len(flags) == 1 and flags <= {'0', '1'} else np.nan
    x = np.r_[evidence['entry_is_star_st'], onehot, concern]
    evidence.update(opinion_unique=len(opinions) == 1, opinion_known=bool(np.isfinite(onehot).all()),
                    going_concern_known=bool(np.isfinite(concern)))
    return x, evidence


def label_record(row, events, reviews, cap):
    """Rebuild full first-A/X chain; never skip an endpoint before shifted origin."""
    start = pd.Timestamp(row['entry_announcement'])
    origin = pd.Timestamp(row['origin'])
    ee = events[(events.announcement >= start) & (events.announcement <= cap)]
    terminal = ee[ee.Chgtype.str.endswith(('A', 'X'))]
    if (terminal.announcement <= origin).any():
        return dict(time=0, event=0, stop=origin, reason='endpoint_at_or_before_new_origin', conflict=True,
                    label_eligible=False, chain_gap_present=False, unresolved_chain_gap=False,
                    effective_followup_cap=cap)
    pairs = [(r.announcement, 1 if r.Chgtype.endswith('X') else 2) for r in terminal.itertuples()]
    value = classify_first(origin, pairs, cap, months=24)
    gaps = ee[(ee.announcement <= value['stop']) & (ee.chain_discontinuity == '1')]
    unresolved = False
    for z in gaps.itertuples():
        reused = reviews.get(z.event_id, {}).get('decision', '').startswith('resolved')
        final = (z.Chgtype.endswith('X') and
                 (name_state(z.ShortNamebc) in ['B', 'D', 'T'] or name_state(z.ShortNameac) in ['B', 'D', 'T']) and
                 ('终止上市' in z.Content or z.all_cause_exit_confirmed == '1'))
        intermediate = z.Chgtype[-1] in ['B', 'D', 'C', 'S', 'T'] and z.Chgtype[0] in ['B', 'D', 'C', 'S', 'T']
        unresolved |= not (reused or final or intermediate)
    value.update(label_eligible=not value['conflict'] and not unresolved,
                 chain_gap_present=bool(len(gaps)), unresolved_chain_gap=unresolved, effective_followup_cap=cap)
    return value


def packed(x, meta):
    return dict(X=x, time=meta.time.to_numpy(float), event=meta.event.to_numpy(int),
                groups=meta.group.to_numpy(str), sample_ids=meta.sample_id.to_numpy(str),
                year=meta.entry_year.to_numpy(int), h12=meta.h12.to_numpy(int), h24=meta.h24.to_numpy(int),
                label_eligible=meta.label_eligible.to_numpy(bool), chain_gap_present=meta.chain_gap_present.to_numpy(bool),
                analysis_time=np.array([calendar_elapsed(o, s) for o, s in zip(meta.origin, meta.stop)]))


def prepare(run):
    protocol_path = Path(__file__).with_name('frozen_protocol.json')
    protocol = json.loads(protocol_path.read_text())
    write(run/'audit/protocol.json', protocol)
    write(run/'audit/pre_fit_freeze.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          protocol_sha256=sha(protocol_path), prior_outcomes_exposed=True, new_models_fitted=0))
    signatures = {}
    def source(p):
        p = Path(p); signatures[str(p)] = sha(p); return p
    plan = json.loads(source(BASE/'input/plan.json').read_text())
    old_parts = [arrays(source(BASE/'input'/f'{s}_PRIVATE.npz')) for s in ['train', 'eval']]
    old = np.vstack([x['X'] for x in old_parts])
    meta = pd.concat([pd.read_parquet(source(BASE/'input'/f'{s}_metadata_PRIVATE.parquet')) for s in ['train', 'eval']], ignore_index=True)
    np.testing.assert_array_equal(np.concatenate([x['sample_ids'] for x in old_parts]), meta.sample_id)
    assert meta.group.is_unique and len(meta) == 390
    prior_receipt = json.loads((Path(__file__).resolve().parents[2]/'experiments/post_st_prognosis_v1/execution_receipt.json').read_text())
    for name, h in prior_receipt['model_input_hashes'].items():
        assert sha(BASE/'input'/name) == h
    old_meta = meta.copy()
    names = plan['feature_names']; assert len(names) == 75
    finance = pd.read_parquet(source(LEGACY/'prepared/financial_facts.parquet'))
    assert not finance.duplicated(['Stkcd', 'Accper']).any()
    np.testing.assert_array_equal(pd.to_datetime(finance.available),
        finance[['ann', 'correction_0', 'correction_1', 'correction_2']].apply(pd.to_datetime).max(axis=1))
    for key in ['available', 'prev_available', 'ann']:
        finance[key] = pd.to_datetime(finance[key])
    finance['report_date'] = pd.to_datetime(finance.Accper)
    fg = {k: g for k, g in finance[finance.Stkcd.isin(meta.code)].groupby('Stkcd')}
    market = pd.read_parquet(source(BASE/'private/market_source_subset_PRIVATE.parquet')).sort_values(['Stkcd', 'Trddt'])
    assert not market.duplicated(['Stkcd', 'Trddt']).any()
    mg = {k: g for k, g in market.groupby('Stkcd')}
    cal = np.load(source(MARKET/'market_calendar.npy'), allow_pickle=False)
    events = pd.read_csv(source(OLD/'phase1_data_freeze_v1/03_event_history.csv'), dtype=str, keep_default_na=False)
    source('/home/oliver/LLMworkspace/download/BDT_Trdchg[DES][csv].txt')
    mapping = json.loads(source(AUDIT/'assembly/group_mapping.json').read_text())
    events['group'] = events.Symbol.map(lambda x: mapping.get(x, x))
    events['announcement'] = pd.to_datetime(events.adjudicated_announcement_date.where(events.adjudicated_announcement_date.ne(''), events.Annoudt), errors='raise')
    assert events.announcement.notna().all() and events.event_id.is_unique
    by_event = events.set_index('event_id'); eg = {k: g for k, g in events.groupby('group')}
    reviews = pd.read_csv(source(OLD/'phase1_data_freeze_v1/03_case_review.csv'), dtype=str, keep_default_na=False).set_index('event_id').to_dict('index')
    scope = json.loads(source(OLD/'method_freeze_input_build_v2/evidence/source_scope_confirmation.json').read_text())
    C = pd.Timestamp(scope['primary_snapshot_execution_bound']); assert C == pd.Timestamp('2026-09-03')
    vintage = pd.read_csv(source(OLD/'phase1b_unblock24m_v1/06_financial_pit_vintage_audit.csv'))
    source(OLD/'phase1b2_targeted_recovery_v2/03_financial_source_acceptance.md')
    meta['entry_announcement'] = meta.origin
    meta['old_sample_id'] = meta.sample_id
    meta['origin'] = [next_open(o, cal) for o in meta.entry_announcement]
    meta['entry_year'] = meta.origin.dt.year
    meta['sample_id'] = meta.code+'_'+meta.origin.dt.strftime('%Y-%m-%d')+'_preopen'
    meta['h12'] = [horizons(o)[0] for o in meta.origin]
    meta['h24'] = [horizons(o)[1] for o in meta.origin]
    xbase = old.copy(); xrefresh = old.copy(); extra = []; details = []; clocks = []
    for i, row in meta.iterrows():
        previous = old_meta.iloc[i]
        old_fin, old_dates, old_fmeta = financial_record(fg[row.code], names[:48], previous.origin)
        np.testing.assert_allclose(old_fin, old[i, :48], rtol=0, atol=0, equal_nan=True)
        assert old_fmeta['report_date'] == previous.report_date
        old_market = market_features(mg[row.code], cal, previous.origin)
        np.testing.assert_allclose([old_market[n] for n in names[48:72]], old[i, 48:72], rtol=0, atol=0, equal_nan=True)
        new_fin, dates, fm = financial_record(fg[row.code], names[:48], row.origin)
        updated_market = market_features(mg[row.code], cal, row.origin)
        mv = np.array([updated_market[n] for n in names[48:72]], float); mv[~np.isfinite(mv)] = np.nan
        market_missing = 1 - np.clip(updated_market['market_coverage_3m'] if pd.notna(updated_market['market_coverage_3m']) else 0, 0, 1)
        for xx, ff, fmeta in [(xbase, old_fin, old_fmeta), (xrefresh, new_fin, fm)]:
            xx[i, :48] = ff; xx[i, 48:72] = mv
            xx[i, 72:] = [min((row.origin-fmeta['report_date']).days/365, 2),
                           min((row.origin-fmeta['financial_available']).days/365, 2), market_missing]
        assert (np.isfinite(new_fin) <= ~np.isnat(dates)).all()
        assert not ((~np.isnat(dates)) & (dates >= np.datetime64(row.origin.date(), 'D'))).any()
        announced, evidence = announcement_values(previous.to_dict(), by_event)
        assert evidence['available_at'] < row.origin
        extra.append(announced); clocks.append(dates)
        annual = fg[row.code].loc[fg[row.code].Accper.eq(f'{previous.origin.year-1}-12-31')]
        a = annual.iloc[0] if len(annual) else None
        details.append(dict(sample_id=row.sample_id, entry_year=row.entry_year,
            shift_days=(row.origin-row.entry_announcement).days, old_report=previous.report_date,
            new_report=fm['report_date'], report_updated=fm['report_date'] != previous.report_date,
            report_financial_available=fm['financial_available'], previous_component_available=fm['prev_available'],
            new_market_available=updated_market['market_latest_date'], market_updated=updated_market['market_latest_date'] != previous.market_available,
            annual_snapshot_blocked_by_future_date=bool(a is not None and a.available >= row.origin),
            prior_annual_selected=fm['report_date'] == pd.Timestamp(f'{previous.origin.year-1}-12-31'),
            **evidence))
    details = pd.DataFrame(details)
    details.to_parquet(run/'private/information_alignment_PRIVATE.parquet', index=False)
    np.savez_compressed(run/'private/financial_availability_PRIVATE.npz', dates=np.array(clocks), sample_ids=meta.sample_id.to_numpy(str))
    X = dict(market_clock=xbase, financial_refresh=xrefresh, announcement_fields=np.column_stack([xrefresh, extra]))
    assert X['announcement_fields'].shape[1] == 83
    np.testing.assert_array_equal(X['announcement_fields'][:, :75], xrefresh)
    for a in X.values(): assert not np.isinf(a).any()
    counts = []; exclusions = []; changed = []; summary = []
    for year in protocol['test_years']:
        cutoff = pd.Timestamp(f'{year}-01-01')
        collected = {}
        for role, select in [('fit', meta.entry_year.between(2012, year-1)), ('test', meta.entry_year.eq(year))]:
            dd = meta.loc[select].copy(); cap = min(C, cutoff-pd.Timedelta(days=1)) if role == 'fit' else C
            labels = pd.DataFrame([label_record(row, eg[row['group']], reviews, cap) for row in dd.to_dict('records')], index=dd.index)
            for col in labels: dd[col] = labels[col]
            keep = dd.label_eligible & dd.time.gt(0)
            excluded = dd.loc[~keep].copy(); excluded['test_year'] = year; excluded['role'] = role; exclusions.append(excluded)
            dd = dd.loc[keep].copy()
            assert (dd.stop <= C).all()
            if role == 'fit': assert (dd.origin < cutoff).all() and (dd.stop < cutoff).all()
            assert ((dd.h24 == 730) | (dd.h24 == 731)).all()
            info = details.loc[dd.index]
            summary.append(dict(test_year=year, role=role, companies=len(dd), report_updated=int(info.report_updated.sum()),
                market_updated=int(info.market_updated.sum()), prior_annual_selected=int(info.prior_annual_selected.sum()),
                annual_snapshot_blocked_by_future_date=int(info.annual_snapshot_blocked_by_future_date.sum()),
                audit_metadata_known=int(info.opinion_known.sum()), going_concern_known=int(info.going_concern_known.sum()),
                old_annual_selected=int((old_meta.loc[dd.index,'report_date'].dt.strftime('%m-%d') == '12-31').sum())))
            dd.to_parquet(run/'evaluation_input'/f'{year}_{role}_metadata_PRIVATE.parquet', index=False)
            for months in [12, 24]:
                hit = (dd.event > 0) & (dd.time <= dd[f'h{months}']); known = hit | (dd.time >= dd[f'h{months}'])
                counts.append(dict(test_year=year, role=role, horizon_months=months, companies=len(dd),
                    termination=int((hit & dd.event.eq(1)).sum()), recovery=int((hit & dd.event.eq(2)).sum()),
                    complete_no_event=int((known & ~hit).sum()), unknown=int((~known).sum())))
            collected[role] = dd
        assert not set(collected['fit'].group) & set(collected['test'].group)
        for variant in protocol['variants']:
            folder = run/'input/folds'/str(year)/variant; folder.mkdir(parents=True)
            tr = packed(X[variant][collected['fit'].index], collected['fit'])
            ev = packed(X[variant][collected['test'].index], collected['test'])
            np.savez_compressed(folder/'train_PRIVATE.npz', **tr)
            np.savez_compressed(folder/'eval_PRIVATE.npz', X=ev['X'], sample_ids=ev['sample_ids'])
            np.savez_compressed(run/'evaluation_input'/f'{year}_{variant}_test_PRIVATE.npz', **ev)
            pp = {k:plan[k] for k in ['forest_common','forest_variants','standard_rsf']}
            pp = json.loads(json.dumps(pp)); pp['forest_variants']['xai_q2']['cause_weights'] = [1., 1.]
            pp.update(feature_names=names+(EXTRA_NAMES if variant=='announcement_fields' else []),
                pilot={'trees':4}, variant=variant, test_year=year, calibration_n=0,
                source_hashes={n:sha(folder/n) for n in ['train_PRIVATE.npz','eval_PRIVATE.npz']})
            write(folder/'plan.json', pp)
            pd.DataFrame(dict(index=range(tr['X'].shape[1]), name=pp['feature_names'], missing_rate_train=np.isnan(tr['X']).mean(0))).to_csv(run/'reports'/f'feature_manifest_{year}_{variant}.csv', index=False)
    pd.concat(exclusions, ignore_index=True).to_parquet(run/'private/excluded_PRIVATE.parquet', index=False)
    meta.to_parquet(run/'private/shifted_cohort_PRIVATE.parquet', index=False)
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv', index=False)
    pd.DataFrame(summary).to_csv(run/'reports/information_updates.csv', index=False)
    write(run/'audit/data_alignment.json', dict(status=protocol['status'], unchanged_source_hashes=signatures,
        origin_moved=True, all_390_old_financial_and_market_values_reproduced=True,
        independent_original_value_certifications=int(vintage.original_values_verified.sum()), original_financial_vintages_PIT_certified=False,
        upstream_audit_rows=len(vintage), upstream_original_pdf_pilot='cancelled, not performed',
        financial_snapshot_filter='max(recorded announcement, all corrections); never backdated',
        no_future_correction_override=True, source_coverage_end=str(C.date()),
        cross_year_origins=int((old_meta.entry_year != meta.entry_year).sum()),
        shift_days={str(k):int(v) for k,v in details.shift_days.value_counts().sort_index().items()},
        training_cutoffs_exclusive=['2023-01-01','2024-01-01'], excluded_fold_rows=sum(len(x) for x in exclusions),
        reused_old_predictions=False, registry_lookup='Unavailable: tunnel-client inactive; inspected authoritative local sources',
        counts=counts, information_updates=summary, new2025_evaluation=False,
        event_features_only_from_entry_records=True, algorithm_changed=False))
    assert all(sha(Path(p)) == h for p,h in signatures.items())
    print(pd.DataFrame(counts).to_string(index=False)); print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    prepare(p.parse_args().run)
