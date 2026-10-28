"""Apply a human-confirmed annual-value policy as a separate data-only revision."""
from pathlib import Path
import argparse,datetime,json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_next_open_info_v1.prepare import financial_record,LEGACY

BASE=Path('/home/oliver/MainDiskData/xai_rsf_next_open_info_20261028_eu3undip')


def confirmed_annual_frame(frame):
    """Do not edit source values or infer announcement dates from report periods."""
    if frame.duplicated(['Stkcd','Accper']).any():
        raise ValueError('Company-report keys must be unique')
    f=frame.copy(deep=True)
    for k in ['ann','available','prev_available']:
        f[k]=pd.to_datetime(f[k],errors='raise')
    f['report_date']=pd.to_datetime(f.Accper,errors='raise')
    key=pd.MultiIndex.from_arrays([f.Stkcd,pd.to_datetime(f.prev_period).dt.strftime('%Y-%m-%d')])
    expected=pd.to_datetime(f.Accper)-pd.DateOffset(years=1)
    if not pd.to_datetime(f.prev_period).equals(expected):
        raise ValueError('Expected explicit previous-year component keys')
    lookup=f.set_index(['Stkcd','Accper'])
    previous_original=lookup.available.reindex(key).to_numpy('datetime64[ns]')
    np.testing.assert_array_equal(previous_original,f.prev_available.to_numpy('datetime64[ns]'))
    f['source_available_conservative']=f.available
    f['source_prev_available_conservative']=f.prev_available
    annual=f.report_date.dt.month.eq(12)&f.report_date.dt.day.eq(31)
    f['annual_value_policy']=np.where(annual,'accepted_by_researcher_confirmation','unchanged_conservative_nonannual')
    f.loc[annual,'available']=f.loc[annual,'ann']
    f['prev_available']=f.set_index(['Stkcd','Accper']).available.reindex(key).to_numpy('datetime64[ns]')
    # Every original numeric field and correction date is preserved exactly.
    for name in frame:
        if name not in ['available','prev_available','ann']:
            np.testing.assert_array_equal(frame[name],f[name])
    np.testing.assert_array_equal(f.loc[~annual,'available'],pd.to_datetime(frame.loc[~annual,'available']))
    return f


def prepare(run):
    policy_path=Path(__file__).with_name('source_policy.json');policy=json.loads(policy_path.read_text())
    assert policy['confirmation']['verbatim']=='视为原始版本'
    write(run/'audit/source_policy.json',policy)
    sources={}
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    old_audit=json.loads(source(BASE/'audit/data_alignment.json').read_text())
    frozen=json.loads(source(BASE/'audit/immutable_input_hashes.json').read_text())
    for name,digest in frozen.items():assert sha(BASE/'input'/name)==digest
    fact_path=source(LEGACY/'prepared/financial_facts.parquet')
    assert sources[str(fact_path)]==old_audit['unchanged_source_hashes'][str(fact_path)]
    cohort=pd.read_parquet(source(BASE/'private/shifted_cohort_PRIVATE.parquet'))
    facts=pd.read_parquet(fact_path);facts=facts[facts.Stkcd.isin(cohort.code)].copy()
    revised=confirmed_annual_frame(facts)
    original=facts.assign(report_date=pd.to_datetime(facts.Accper))
    for k in ['ann','available','prev_available']:original[k]=pd.to_datetime(original[k])
    original_groups={k:g for k,g in original.groupby('Stkcd')}
    groups={k:g for k,g in revised.groupby('Stkcd')}
    revised.to_parquet(run/'private/accepted_annual_source_values_and_clocks_PRIVATE.parquet',index=False)
    summaries=[];field_counts=[];cohort_rows=[];hashes=[]
    for year in [2023,2024]:
        folder=run/'input'/str(year)/'announcement_fields';folder.mkdir(parents=True)
        plan=json.loads(source(BASE/'input/folds'/str(year)/'announcement_fields/plan.json').read_text())
        names=plan['feature_names'];assert len(names)==83
        for role in ['fit','test']:
            original_path=BASE/'input/folds'/str(year)/'announcement_fields/train_PRIVATE.npz' if role=='fit' else BASE/'evaluation_input'/f'{year}_announcement_fields_test_PRIVATE.npz'
            old=arrays(source(original_path));new={k:v.copy() for k,v in old.items()}
            meta=pd.read_parquet(source(BASE/'private'/f'{year}_announcement_fields_{role}_feature_metadata_PRIVATE.parquet'))
            np.testing.assert_array_equal(old['sample_ids'],meta.sample_id)
            audits=[];all_clocks=[];annual_released=0;annual_selected=0
            for i,row in meta.iterrows():
                old_fin,_,oldfm=financial_record(original_groups[row.code],names[:48],row.origin)
                np.testing.assert_array_equal(old_fin,old['X'][i,:48])
                assert oldfm['report_date']==row.report_date
                values,clocks,fm=financial_record(groups[row.code],names[:48],row.origin)
                new['X'][i,:48]=values
                new['X'][i,72]=min((row.origin-fm['report_date']).days/365,2)
                new['X'][i,73]=min((row.origin-fm['financial_available']).days/365,2)
                assert (np.isfinite(values)<=~np.isnat(clocks)).all()
                assert not ((~np.isnat(clocks))&(clocks>=np.datetime64(row.origin.date(),'D'))).any()
                assert pd.notna(fm['financial_available']) and fm['financial_available']<row.origin
                annual_period=f'{row.entry_year-1}-12-31'
                ann=groups[row.code][groups[row.code].Accper.eq(annual_period)]
                assert len(ann)==1
                a=ann.iloc[0]
                old_blocked=pd.isna(a.source_available_conservative) or a.source_available_conservative>=row.origin
                eligible=pd.notna(a.available) and a.available<row.origin and a.report_date<row.origin
                annual_released+=bool(old_blocked and eligible)
                is_selected=fm['report_date']==a.report_date;annual_selected+=bool(is_selected)
                audits.append(dict(sample_id=row.sample_id,old_report_date=row.report_date,new_report_date=fm['report_date'],
                    report_period_changed=row.report_date!=fm['report_date'],old_financial_available=row.financial_available,
                    new_financial_available=fm['financial_available'],new_previous_component_available=fm['prev_available'],
                    prior_annual_original_announcement=a.ann,prior_annual_conservative_available=a.source_available_conservative,
                    annual_previously_blocked=bool(old_blocked),annual_now_eligible=bool(eligible),prior_annual_selected=bool(is_selected),
                    later_corrections_recorded=bool(a.source_available_conservative>=row.origin),
                    numeric_source_values_edited=False,source_acceptance='accepted_by_researcher_confirmation',
                    independently_verified_original_values=False))
                all_clocks.append(clocks)
                for col in ['report_date','financial_available','prev_available']:meta.loc[i,col]=fm[col]
                meta.loc[i,'financial_age']=(row.origin-fm['financial_available']).days
                meta.loc[i,'financial_period_age']=(row.origin-fm['report_date']).days
            same=(old['X']==new['X'])|(np.isnan(old['X'])&np.isnan(new['X']))
            allowed=set(range(48))|{72,73}
            for j in range(83):
                if j not in allowed:assert same[:,j].all()
            for name in old:
                if name!='X':np.testing.assert_array_equal(new[name],old[name])
            assert not np.isinf(new['X']).any()
            target=folder/('train_UNTRAINED_PRIVATE.npz' if role=='fit' else 'eval_UNTRAINED_PRIVATE.npz')
            np.savez_compressed(target,**new)
            reread=arrays(target)
            for key in new:np.testing.assert_array_equal(new[key],reread[key])
            meta.to_parquet(run/'private'/f'{year}_{role}_metadata_PRIVATE.parquet',index=False)
            pd.DataFrame(audits).to_parquet(run/'private'/f'{year}_{role}_alignment_PRIVATE.parquet',index=False)
            np.savez_compressed(run/'private'/f'{year}_{role}_availability_PRIVATE.npz',
                sample_ids=new['sample_ids'],financial_available=np.array(all_clocks))
            dd=pd.DataFrame(audits)
            summaries.append(dict(test_year=year,role=role,companies=len(meta),
                previously_blocked_prior_annual=int(dd.annual_previously_blocked.sum()),
                prior_annual_released_by_confirmation=int(annual_released),
                remaining_prior_annual_missing_or_late_announcement=int((~dd.annual_now_eligible).sum()),
                report_period_changed=int(dd.report_period_changed.sum()),prior_annual_selected=int(annual_selected),
                companies_with_changed_financial_values=int((~same[:,:48]).any(axis=1).sum()),
                companies_with_changed_any_input=int((~same).any(axis=1).sum()),
                changed_financial_cells=int((~same[:,:48]).sum()),changed_input_cells=int((~same).sum()),
                source_values_edited=0,labels_changed=0,model_fits=0))
            for j in sorted(allowed):
                field_counts.append(dict(test_year=year,role=role,feature=names[j],
                    changed_cells=int((~same[:,j]).sum()),
                    previously_missing_now_available=int((np.isnan(old['X'][:,j])&np.isfinite(new['X'][:,j])).sum()),
                    previously_available_now_missing=int((np.isfinite(old['X'][:,j])&np.isnan(new['X'][:,j])).sum())))
            hashes.append(dict(test_year=year,role=role,old_input_sha256=sha(original_path),new_untrained_input_sha256=sha(target)))
            for months in [12,24]:
                h=new[f'h{months}'];hit=(new['event']>0)&(new['time']<=h);known=hit|(new['time']>=h)
                cohort_rows.append(dict(test_year=year,role=role,horizon_months=months,companies=len(meta),
                    termination=int((hit&(new['event']==1)).sum()),recovery=int((hit&(new['event']==2)).sum()),
                    known_no_event=int((known&~hit).sum()),unknown=int((~known).sum())))
        future={k:v for k,v in plan.items() if k not in ['code_commit','source_hashes']}
        future.update(status='DATA_ONLY_NOT_TRAINED',source_policy_id=policy['policy_id'],
            source_policy_sha256=sha(policy_path),old_model_training_commit=plan['code_commit'],
            new_fitting_commit=None,new_training_authorized_by_source_confirmation=False)
        write(folder/'prepared_configuration_NOT_RUN.json',future)
    summary=pd.DataFrame(summaries);summary.to_csv(run/'reports/data_changes.csv',index=False)
    pd.DataFrame(field_counts).to_csv(run/'reports/field_change_counts.csv',index=False)
    pd.DataFrame(cohort_rows).to_csv(run/'reports/cohort_counts_UNCHANGED.csv',index=False)
    pd.DataFrame(hashes).to_csv(run/'reports/prepared_input_hashes.csv',index=False)
    tests=summary[summary.role=='test'];assert tests.prior_annual_released_by_confirmation.sum()==25
    assert tests.remaining_prior_annual_missing_or_late_announcement.sum()==0
    for p,h in sources.items():assert sha(Path(p))==h
    for name,h in frozen.items():assert sha(BASE/'input'/name)==h
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/validation.json',dict(status='PASS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_policy=policy['policy_id'],data_status='ACCEPTED_BY_RESEARCHER_CONFIRMATION',
        base_run=str(BASE),private_run=str(run),source_facts_unchanged=True,old_frozen_inputs_unchanged=True,
        raw_numeric_values_edited=False,quarterly_available_dates_unchanged=True,
        same_year_ago_component_keys_rejoined=True,all_available_components_precede_origin=True,
        all_nonfinancial_inputs_unchanged=True,all_outcomes_and_order_unchanged=True,
        evaluation_companies=81,annual_version_blockers_reclassified=25,
        independent_original_PDF_value_verification=False,new_models_fitted=0,
        new_predictions_generated=0,old_reported_auc_0752_recomputed=False,summary=summaries))
    print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);prepare(p.parse_args().run)
