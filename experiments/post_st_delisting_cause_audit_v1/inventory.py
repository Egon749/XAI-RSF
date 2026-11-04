"""Build the label-only register first; no forecasts or model explanations are loaded."""
from experiments.post_st_prognosis_v1.prepare import OLD,AUDIT
from experiments.post_st_paper_features_v1.common import history
from experiments.post_st_prognosis_v1.common import classify_first
from .common import *
def main():
    root=get_run('post_st_seed_explanation_stability_v1')/'input'
    cohort=pd.read_parquet(history(2025)/'private/shifted_cohort_PRIVATE.parquet').set_index('sample_id')
    events=pd.read_csv(OLD/'phase1_data_freeze_v1/03_event_history.csv',dtype=str,keep_default_na=False)
    mapping=load(AUDIT/'assembly/group_mapping.json')
    events['firm']=events.Symbol.map(lambda c:mapping.get(c,c))
    events['ad']=pd.to_datetime(events.adjudicated_announcement_date.where(events.adjudicated_announcement_date.ne(''),events.Annoudt))
    rows=[];counts=[];firms=[];had_later=0
    for y in YEARS:
        labels=arrays(root/f'evaluation_input/{y}_test_PRIVATE.npz');ev=arrays(root/f'folds/{y}/eval_PRIVATE.npz')
        np.testing.assert_array_equal(ev['sample_ids'],labels['sample_ids']);np.testing.assert_array_equal(ev['X'],labels['X'][:,pd.read_csv(ROOT/'experiments/post_st_grouped_explanation_v1/FEATURE_GROUP_MAPPING.csv').original_feature_id])
        for i,sid in enumerate(labels['sample_ids']):
            x=cohort.loc[sid];g=events[events.firm==x.group].sort_values('ad')
            origin=pd.Timestamp(x.origin);stop=origin+pd.Timedelta(days=int(labels['time'][i]));status=int(labels['event'][i])
            term=g[g.ad.gt(origin)&g.ad.le(stop)&g.Chgtype.str.endswith(('X','A'))]
            pairs=[(z.ad,1 if z.Chgtype.endswith('X') else 2) for z in term.itertuples()]
            result=classify_first(origin,pairs,stop,24)
            assert int(result['event'])==status and result['time']==labels['time'][i]
            entries=g[g.Chgtype.isin(['AB','AD'])]
            rec=dict(year=y,row=i,sample_id=str(sid),company_group=str(x.group),code=str(x.code).zfill(6),origin=str(origin.date()),entry_notice=str(x.entry_announcement.date()),stop=str(stop.date()),elapsed_days=int(labels['time'][i]),event=status,h12=int(labels['h12'][i]),h24=int(labels['h24'][i]),raw_observed_entry_changes=len(entries),raw_X_rows=int(g.Chgtype.str.endswith('X').sum()))
            firms.append(rec)
            if status==2 and (g.ad.gt(stop)&g.Chgtype.str.endswith('X')).any():had_later+=1
            if status!=1:continue
            final=term[term.ad.eq(stop)&term.Chgtype.str.endswith('X')];assert len(final)
            rec=rec|dict(board='ChiNext' if x.code.startswith(('300','301')) else ('STAR' if x.code.startswith('688') else 'MAIN'),exchange='SSE' if x.code.startswith('6') else 'SZSE',
                         model_x_date=str(stop.date()),raw_X_execution=';'.join(sorted(set(final.Execudt))),raw_X_type=';'.join(sorted(set(final.Chgtype))),final_event_ids=';'.join(sorted(final.event_id)),
                         year12=bool(labels['time'][i]<=labels['h12'][i]),year24_supported=y!=2025,provider_X_rows_on_date=len(final))
            rows.append(rec)
        for h in (12,24):
            target=(labels['event']==1)&(labels['time']<=labels[f'h{h}']);known=(labels['time']>=labels[f'h{h}'])|(labels['event']>0)
            counts.append(dict(year=y,horizon=h,companies=len(ev['X']),observed_first_terminations=int(target.sum()),unknown=int((~known).sum()),confirmed_evaluation=bool(known.all())))
    f=pd.DataFrame(firms);cases=pd.DataFrame(rows)
    assert f.company_group.is_unique and f.sample_id.is_unique and cases.company_group.is_unique
    f.to_csv(run()/'companies_PRIVATE.csv',index=False);cases.to_csv(run()/'events_PRIVATE.csv',index=False)
    dump(D/'EVENT_DEFINITION_AUDIT.json',dict(model_cause1='First adjudicated announcement date of a change ending in X, before a change ending in A. Includes any documented final exit, including voluntary if present.',model_cause2='first full recovery to A before X',intermediate='B/D warnings and T consolidation/intermediate statuses are not endpoints',origin='pre-open of next exchange trading day after audited first observed financial/audit ST change; day resolution',cap='24 calendar months or administrative data cutoff, whichever earlier; existing labels unchanged',rows_reconciled=len(f),unique_companies=len(f),unique_observed_X=len(cases),single_model_episode_per_company=True,multiple_raw_AB_AD_changes=int((f.raw_observed_entry_changes>1).sum()),multiple_raw_X_rows=int((f.raw_X_rows>1).sum()),recovery_first_with_later_X_in_source=had_later,provider_execution_vs_model_date_diff=int((cases.raw_X_execution!=cases.model_x_date).sum()),cohorts=counts,label_input_sha256={str(y):sha(root/f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}))
if __name__=='__main__':main()
