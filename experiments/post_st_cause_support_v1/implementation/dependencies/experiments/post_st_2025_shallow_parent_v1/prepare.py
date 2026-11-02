"""Transport the frozen conservative-C recipe, without modifying old artifacts."""
from pathlib import Path
import argparse, datetime, json, shutil
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizons
from experiments.post_st_next_open_info_v1.prepare import (
    next_open, financial_record, announcement_values, label_record, packed, EXTRA_NAMES)
from experiments.post_st_prognosis_v1.prepare import market_features, OLD, LEGACY, AUDIT


def horizon_summary(meta, months):
    h=meta[f'h{months}'].to_numpy()
    t=meta.time.to_numpy();e=meta.event.to_numpy();ok=meta.label_eligible.to_numpy()
    hit=ok&(e>0)&(t<=h);known=ok&(hit|(t>=h))
    return dict(horizon_months=months,companies=len(meta),termination=int((hit&(e==1)).sum()),
        recovery=int((hit&(e==2)).sum()),known_no_first_event=int((known&~hit).sum()),
        unknown=int((~known).sum()),followup_reaches_horizon=int((ok&(t>=h)).sum()),
        endpoint_evaluation_supported=bool(known.all() or (ok&(t>=h)).any()))


def prepare(run):
    repo=Path(__file__).resolve().parents[2]
    protocol_path=Path(__file__).with_name('frozen_protocol.json')
    protocol=json.loads(protocol_path.read_text())
    write(run/'audit/protocol.json',protocol)
    write(run/'audit/pre_fit_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(protocol_path),test_outcomes_previously_exposed=True,new_fits=0,new_predictions=0))
    base=Path(protocol['historical_cohort_run']);old=Path(protocol['old_C_reference_run']);sources={}
    def source(p,expected=None):
        p=Path(p);h=sha(p)
        if expected is not None:assert h==expected,str(p)
        sources[str(p)]=h
        return p
    receipt=json.loads(source(repo/'experiments/post_st_train2024_h12_v1/execution_receipt.json').read_text())
    for name,h in receipt['model_input_hashes'].items():source(base/'input'/name,h)
    old_plan=json.loads(source(base/'input/plan.json').read_text());names=old_plan['feature_names']
    assert len(names)==75
    old_arrays=[arrays(base/'input'/f'{role}_PRIVATE.npz') for role in ['train','eval']]
    original=pd.concat([pd.read_parquet(base/'input'/f'{role}_metadata_PRIVATE.parquet') for role in ['train','eval']],ignore_index=True)
    old_X=np.vstack([x['X'] for x in old_arrays]);assert len(original)==443 and original.group.is_unique
    np.testing.assert_array_equal(original.sample_id,np.concatenate([x['sample_ids'] for x in old_arrays]))
    finance_path=LEGACY/'prepared/financial_facts.parquet'
    finance=pd.read_parquet(source(finance_path,receipt['source_sha256'][str(finance_path)]))
    assert not finance.duplicated(['Stkcd','Accper']).any()
    np.testing.assert_array_equal(pd.to_datetime(finance.available),
        finance[['ann','correction_0','correction_1','correction_2']].apply(pd.to_datetime).max(axis=1))
    for k in ['available','prev_available','ann']:finance[k]=pd.to_datetime(finance[k])
    finance['report_date']=pd.to_datetime(finance.Accper)
    fg={k:g for k,g in finance[finance.Stkcd.isin(original.code)].groupby('Stkcd')}
    market=pd.read_parquet(source(base/'private/market_source_subset_PRIVATE.parquet')).sort_values(['Stkcd','Trddt'])
    assert not market.duplicated(['Stkcd','Trddt']).any()
    mg={k:g for k,g in market.groupby('Stkcd')}
    calendar=np.load(source(base/'private/market_calendar_through2025_PRIVATE.npy'),allow_pickle=False)
    assert calendar.max()==np.datetime64('2025-12-31')
    ep=OLD/'phase1_data_freeze_v1/03_event_history.csv';mp=AUDIT/'assembly/group_mapping.json';rp=ep.with_name('03_case_review.csv')
    for p in [ep,mp,rp]:source(p,receipt['source_sha256'][str(p)])
    events=pd.read_csv(ep,dtype=str,keep_default_na=False);mapping=json.loads(mp.read_text())
    events['group']=events.Symbol.map(lambda c:mapping.get(c,c))
    events['announcement']=pd.to_datetime(events.adjudicated_announcement_date.where(events.adjudicated_announcement_date.ne(''),events.Annoudt),errors='raise')
    assert events.announcement.notna().all() and events.event_id.is_unique
    by_event=events.set_index('event_id');eg={k:g for k,g in events.groupby('group')}
    reviews=pd.read_csv(rp,dtype=str,keep_default_na=False).set_index('event_id').to_dict('index')
    scope_path=OLD/'method_freeze_input_build_v2/evidence/source_scope_confirmation.json'
    scope=json.loads(source(scope_path,receipt['source_sha256'][str(scope_path)]).read_text())
    assert scope['primary_snapshot_execution_bound']=='2026-09-03'
    test_cap=pd.Timestamp(protocol['test_observation_cutoff_inclusive'])
    assert test_cap<=pd.Timestamp(scope['primary_snapshot_execution_bound'])
    meta=original.copy();meta['entry_announcement']=meta.origin;meta['old_sample_id']=meta.sample_id
    meta['origin']=[next_open(o,calendar) for o in meta.entry_announcement]
    meta['entry_year']=meta.origin.dt.year
    meta['sample_id']=meta.code+'_'+meta.origin.dt.strftime('%Y-%m-%d')+'_preopen'
    meta['h12']=[horizons(o)[0] for o in meta.origin];meta['h24']=[horizons(o)[1] for o in meta.origin]
    X=[];clocks=[];details=[]
    for i,row in meta.iterrows():
        previous=original.iloc[i]
        old_fin,_,_=financial_record(fg[row.code],names[:48],previous.origin)
        np.testing.assert_allclose(old_fin,old_X[i,:48],atol=0,rtol=0,equal_nan=True)
        old_market=market_features(mg[row.code],calendar,previous.origin)
        np.testing.assert_allclose([old_market[n] for n in names[48:72]],old_X[i,48:72],atol=0,rtol=0,equal_nan=True)
        ff,dates,fm=financial_record(fg[row.code],names[:48],row.origin)
        mm=market_features(mg[row.code],calendar,row.origin)
        mv=np.array([mm[n] for n in names[48:72]],float);mv[~np.isfinite(mv)]=np.nan
        coverage=mm['market_coverage_3m'];missing=1-np.clip(coverage if pd.notna(coverage) else 0,0,1)
        quality=[min((row.origin-fm['report_date']).days/365,2),min((row.origin-fm['financial_available']).days/365,2),missing]
        extra,evidence=announcement_values(previous.to_dict(),by_event)
        assert evidence['available_at']<row.origin
        assert (np.isfinite(ff)<=~np.isnat(dates)).all()
        assert not ((~np.isnat(dates))&(dates>=np.datetime64(row.origin.date(),'D'))).any()
        assert pd.isna(mm['market_latest_date']) or mm['market_latest_date']<row.origin
        X.append(np.r_[ff,mv,quality,extra]);clocks.append(dates)
        details.append(dict(sample_id=row.sample_id,entry_year=row.entry_year,entry_announcement=row.entry_announcement,
            origin=row.origin,report_date=fm['report_date'],financial_available=fm['financial_available'],
            previous_component_available=fm['prev_available'],market_available=mm['market_latest_date'],
            shift_days=(row.origin-row.entry_announcement).days,**evidence))
    X=np.asarray(X);assert X.shape==(443,83) and not np.isinf(X).any()
    names=names+EXTRA_NAMES
    # Old-C matching covers every pre2025 record, not merely selected examples.
    prior_train=arrays(source(old/'input/folds/2024/announcement_fields/train_PRIVATE.npz'))
    prior_test=arrays(source(old/'evaluation_input/2024_announcement_fields_test_PRIVATE.npz'))
    prior_plan=json.loads(source(old/'input/folds/2024/announcement_fields/plan.json').read_text())
    assert prior_plan['feature_names']==names
    ix=pd.Index(meta.sample_id).get_indexer(np.r_[prior_train['sample_ids'],prior_test['sample_ids']])
    assert len(ix)==390 and (ix>=0).all()
    np.testing.assert_allclose(X[ix],np.vstack([prior_train['X'],prior_test['X']]),atol=0,rtol=0,equal_nan=True)
    pd.DataFrame(details).to_parquet(run/'private/information_alignment_PRIVATE.parquet',index=False)
    np.savez_compressed(run/'private/financial_availability_PRIVATE.npz',dates=np.asarray(clocks),sample_ids=meta.sample_id.to_numpy(str))
    meta.to_parquet(run/'private/shifted_cohort_PRIVATE.parquet',index=False)
    cutoff=pd.Timestamp(protocol['training_information_cutoff_exclusive']);parts={};counts=[];excluded=[]
    for role,mask in [('fit',meta.entry_year.between(2012,2024)),('test',meta.entry_year.eq(2025))]:
        dd=meta.loc[mask].copy();cap=cutoff-pd.Timedelta(days=1) if role=='fit' else test_cap
        labels=pd.DataFrame([label_record(row,eg[row['group']],reviews,cap) for row in dd.to_dict('records')],index=dd.index)
        for c in labels:dd[c]=labels[c]
        good=dd.label_eligible&dd.time.gt(0)
        bad=dd.loc[~good].copy();bad['role']=role;excluded.append(bad)
        dd=dd.loc[good].copy()
        assert dd.group.is_unique and (dd.stop<=cap).all()
        if role=='fit':assert (dd.origin<cutoff).all() and (dd.stop<cutoff).all()
        parts[role]=(packed(X[dd.index],dd),dd)
        dd.to_parquet(run/'evaluation_input'/f'2025_{role}_metadata_PRIVATE.parquet',index=False)
        for m in [12,24]:counts.append(dict(role=role,**horizon_summary(dd,m)))
    tr,tm=parts['fit'];ev,em=parts['test']
    assert not set(tr['groups'])&set(ev['groups'])
    assert len(tm)==390 and len(em)==53,'Any eligibility change requires explicit reconciliation, never silent dropping'
    folder=run/'input/folds/2025';folder.mkdir(parents=True)
    np.savez_compressed(folder/'train_PRIVATE.npz',**tr)
    np.savez_compressed(folder/'eval_PRIVATE.npz',X=ev['X'],sample_ids=ev['sample_ids'])
    np.savez_compressed(run/'evaluation_input/2025_test_PRIVATE.npz',**ev)
    plan={k:protocol[k] for k in ['forest_common','forest_variants','standard_rsf','pilot']}
    plan.update(feature_names=names,test_year=2025,source_status=protocol['status'],
        source_hashes={n:sha(folder/n) for n in ['train_PRIVATE.npz','eval_PRIVATE.npz']})
    write(folder/'plan.json',plan)
    pd.concat(excluded,ignore_index=True).to_parquet(run/'private/excluded_PRIVATE.parquet',index=False)
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv',index=False)
    pd.DataFrame(dict(index=range(83),name=names,missing_rate_train=np.isnan(tr['X']).mean(0))).to_csv(run/'reports/feature_manifest.csv',index=False)
    write(run/'audit/data_alignment.json',dict(status='PASS',data_status=protocol['status'],
        all443_original_financial_market_vectors_reproduced=True,all390_old_C_vectors_exact=True,
        column_order_identical=True,next_open_applied=True,training_cutoff_exclusive=str(cutoff.date()),
        test_cutoff_inclusive=str(test_cap.date()),source_coverage_end=scope['primary_snapshot_execution_bound'],
        fit_companies=len(tm),test_companies=len(em),excluded_records=sum(len(x) for x in excluded),
        counts=counts,unchanged_source_hashes=sources,original_financial_vintages_PIT_certified=False,
        annual_acceptance_override=False,test_labels_absent_from_fit_input=True,new_fits=0))
    assert all(sha(p)==h for p,h in sources.items())
    print(pd.DataFrame(counts).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);prepare(p.parse_args().run)
