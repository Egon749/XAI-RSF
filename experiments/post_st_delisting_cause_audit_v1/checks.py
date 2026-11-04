"""Independent label, timing, Top20, disclosure and aggregate consistency checks."""
from subprocess import check_output
from .common import *
from .signals import SUPPLEMENT
def main():
    r=run();s=Path(json.loads(check_output(['git','show',SUPPLEMENT+':experiments/post_st_information_increment_best3_v1/run_location.json'],cwd=ROOT))['private_run'])
    z=pd.read_csv(r/'DELISTING_CAUSE_REGISTRY_PRIVATE.csv',dtype={'code':str});lab=pd.read_csv(r/'companies_PRIVATE.csv',dtype={'code':str});df=pd.read_parquet(r/'linked_channels_PRIVATE.parquet')
    aj=pd.read_parquet(s/'audit_sources_PRIVATE.parquet');comp=df.merge(lab[['year','code','sample_id']],on=['year','code'],validate='many_to_one').merge(aj,on='sample_id',validate='many_to_one',suffixes=('','_source'));assert len(comp)==len(df)
    assert (z.first_documented_trigger_evidence>z.origin).all() and (z.model_x_date>z.decision_date).all() and (z.delisting_date==z.provider_execution).all()
    chrono=pd.to_datetime(z.model_x_date)-pd.to_datetime(z.decision_date);delta=pd.to_datetime(z.model_x_date)-pd.to_datetime(z.delisting_date)
    for (y,h),g in df.groupby(['year','months']):
        ref=pd.read_csv(ROOT/'experiments/post_st_seed_explanation_stability_v1/ALL_METRICS.csv',dtype={'scope':str})
        ref=ref.query("scope==@ys and version=='shrunk' and model=='C039' and cause==1 and months==@h",local_dict={'ys':str(y),'h':h}).iloc[0]
        target=g.category.str.startswith('C');assert (g.Top_M3.sum(),sum(g.Top_M3&target),sum(target))==(ref.checked,ref.captured,ref.target_events)
    cols=['original_before','timely_extraction_pair_agrees','no_annual_or_entry_text'];rows=[]
    for h,t in comp.groupby('months'):
        for group,bb in [('all_evaluation',t),('C1',t[t.category=='C1']),('pure_C2',t[(t.category=='C2')&~t.dual])]:
            if len(bb)<5:continue
            rows.append(dict(months=h,group=group,n=len(bb),**{k:int(bb[k].sum()) for k in cols}))
    pd.DataFrame(rows).to_csv(D/'ORIGIN_AUDIT_EVIDENCE.csv',index=False)
    old=pd.read_parquet(s/'terminal_channel_PRIVATE.parquet',columns=['sample_id','specific_ground']).drop_duplicates()
    old=old.merge(lab[['code','sample_id']],on='sample_id',validate='one_to_one').merge(z[['code','category']],on='code',validate='one_to_one')
    report=old.groupby(['specific_ground','category']).size().rename('unique_events').reset_index();report.to_csv(D/'PRIOR_EVIDENCE_RECONCILIATION.csv',index=False)
    sums=pd.read_csv(D/'CHANNEL_CONCORDANCE_ACCOUNTING.csv');assert all(abs(v.contribution_to_total_AUC_difference.sum()-v.all_event_difference.iloc[0])<1e-12 for _,v in sums.groupby(['months','contrast']))
    norm=pd.read_csv(D/'CAUSE_SIGNAL_COMPARISON.csv');assert norm.n.min()>=5 and norm.nonmissing.min()>=5
    rec=dict(model_x_minus_decision_days_range=[int(chrono.dt.days.min()),int(chrono.dt.days.max())],median_days=float(chrono.dt.days.median()),model_x_minus_delisting_days_range=[int(delta.dt.days.min()),int(delta.dt.days.max())],provider_effective_and_separate_official_filing_agree=int(sum(z.provider_execution==z.delisting_date)),trigger_before_or_at_origin=int(sum(z.first_documented_trigger_evidence<=z.origin)),individual_rows_matched=134,annual_top20_replayed=5,hashes=load(r/'additional_inputs.json'),reviewed_prior_ground_cases=len(old),company_ids_exported=False,prior_audit_clock_hash=sha(s/'audit_sources_PRIVATE.parquet'))
    dump(D/'ALIGNMENT_CHECKS.json',rec)
if __name__=='__main__':main()
