"""Join reviewed evidence to actual frozen quarterly origins and already-frozen lists."""
from pathlib import Path
import json
import pandas as pd
from review_common import write
from verify_source_clocks import before_origin,eligible,asof_versions
MODELS=['q2','unrestricted','M2_allsector']
def information_state(has_opportunity,has_signal):
 return 'NO_ELIGIBLE_ORIGIN' if not has_opportunity else ('EARLY_PUBLIC_AUDIT_INFORMATION_MISSING_FROM_INPUT' if has_signal else 'NO_PREORIGIN_SIGNAL_FOUND_IN_SEARCHED_SOURCES')
def main(run):
 run=Path(run);timeline=pd.read_parquet(run/'private/audit_event_timeline_PRIVATE.parquet');orig=pd.read_parquet(run/'private/frozen_audit_origins_PRIVATE.parquet');cap=pd.read_parquet(run/'private/event_capture_comparison_PRIVATE.parquet');cohort=json.loads((run/'private/cohort.json').read_text());coverage=pd.read_parquet(run/'private/audit_coverage_PRIVATE.parquet');availability=[];eventrows=[];capturerows=[]
 for e in cohort:
  eid=e['event_id'];t=timeline[timeline.event_id.eq(eid)];os=orig[orig.event_id.eq(eid)].sort_values('origin')
  assert not os.empty
  for o in os.to_dict('records'):
   versions=asof_versions(t,o['origin']);ids=set(versions.index)
   for i,row in t.iterrows():
    availability.append(dict(event_id=eid,sample_id=o['sample_id'],origin=o['origin'],document_id=row.document_id,audit_type=row.audit_report_type,fiscal_year=row.audit_report_fiscal_year,available_at=row.available_at,strictly_public_before_origin=before_origin(row.available_at,o['origin']),latest_public_version_of_type_and_year=i in ids,abnormal=row.abnormal,signal_scope=row.signal_scope,latest_available_fiscal_year_for_type=bool(i in ids and (pd.isna(row.audit_report_fiscal_year) or row.audit_report_fiscal_year==versions[versions.audit_report_type.eq(row.audit_report_type)].audit_report_fiscal_year.max())),event_in_12_calendar_months=eligible(o['origin'],o['event_date'],o['horizon'],e['event_date'],0),announcement_eligible_3m=eligible(o['origin'],o['event_date'],o['horizon'],e['announcement_date'],3),announcement_eligible_6m=eligible(o['origin'],o['event_date'],o['horizon'],e['announcement_date'],6),implementation_eligible_3m=eligible(o['origin'],o['event_date'],o['horizon'],e['event_date'],3),implementation_eligible_6m=eligible(o['origin'],o['event_date'],o['horizon'],e['event_date'],6),input_status=row.input_status))
  for clock in ['announcement','implementation']:
   date=e['announcement_date'] if clock=='announcement' else e['event_date']
   for months in [0,3,6]:
    oo=os[os.apply(lambda o:eligible(o.origin,o.event_date,o.horizon,date,months),axis=1)];known=[]
    for o in oo.to_dict('records'):
     z=asof_versions(t,o['origin']);known.append(z[z.abnormal])
    allknown=pd.concat(known).drop_duplicates() if known else t.iloc[:0]
    core=allknown[allknown.signal_scope.eq('CORE_AUDIT')];opinions=core[core.audit_report_type.isin(['FINANCIAL_AUDIT','EXTERNAL_INTERNAL_CONTROL_AUDIT'])]
    current_ic=t[t.is_event_year_report&t.audit_report_type.eq('EXTERNAL_INTERNAL_CONTROL_AUDIT')&t.abnormal]
    assert not current_ic.empty,eid
    current_before=any(before_origin(p,o) for p in current_ic.available_at for o in oo.origin)
    state=information_state(len(oo)>0,len(core)>0)
    incomplete=bool(((coverage.event_id==eid)&coverage.type.isin(['FINANCIAL_AUDIT','EXTERNAL_INTERNAL_CONTROL_AUDIT'])&(coverage.fiscal_year<pd.Timestamp(e['announcement_date']).year-1)&coverage.status.eq('NO_VERIFIED_RECORD_IN_BOUNDED_REVIEW')).any())
    er=dict(prior_opinion_evidence_incomplete=incomplete,event_id=eid,group=e['group'],clock=clock,lead_months=months,eligible_origins=len(oo),opportunity=bool(len(oo)),early_core_signal=bool(len(core)),early_opinion_signal=bool(len(opinions)),early_other_governance=bool(len(allknown[allknown.signal_scope.eq('OTHER_GOVERNANCE')])),current_key_opinion_before_eligible_origin=current_before,key_opinion_too_late=bool(len(oo)) and not current_before,information_state=state,early_documents=json.dumps(sorted(core.document_id.unique().tolist())),current_key_opinion_first_verified_public_at=min(current_ic.available_at),exact_information_in_input=False)
    eventrows.append(er)
    caught={}
    for frac in [.2,.25]:
     for model in MODELS:
      hits=oo[oo[f'{model}_top{int(frac*100)}']];caught[model]=bool(len(hits));first=hits.origin.min() if len(hits) else pd.NaT
      firstinfo=asof_versions(t,first) if pd.notna(first) else t.iloc[:0];firstinfo=firstinfo[firstinfo.abnormal&firstinfo.signal_scope.eq('CORE_AUDIT')]
      c=cap[(cap.event_id==eid)&(cap.model==model)&cap.top_fraction.eq(frac)].iloc[0]
      if months in [3,6]:
       assert bool(len(oo))==bool(c[f'{clock}_opportunity_{months}m'])
       assert bool(len(hits))==bool(c[f'{clock}_detected_{months}m'])
      else:assert bool(len(hits))==pd.notna(c.first_warning)
      capturerows.append(dict(**er,model=model,top_fraction=frac,captured=bool(len(hits)),first_valid_warning_origin=first,audit_signal_before_first_warning=bool(len(firstinfo)),available_audit_document_ids_at_first_warning=json.dumps(sorted(firstinfo.document_id.unique().tolist())),gap_unrestricted_not_q2=False,q2_unique_vs_unrestricted=False,all_three_captured=False,all_three_missed=False))
     for cr in capturerows[-3:]:
      cr['gap_unrestricted_not_q2']=bool(len(oo)) and caught['unrestricted'] and not caught['q2'];cr['q2_unique_vs_unrestricted']=bool(len(oo)) and caught['q2'] and not caught['unrestricted'];cr['all_three_captured']=all(caught.values());cr['all_three_missed']=bool(len(oo)) and not any(caught.values())
      tags=[cr['information_state']]
      if cr['prior_opinion_evidence_incomplete']:tags.append('EVIDENCE_INCOMPLETE')
      if cr['gap_unrestricted_not_q2']:tags.append('EXISTING_INPUT_MODEL_DISCRIMINATION_GAP')
      if cr['key_opinion_too_late']:tags.append('KEY_DISCLOSURE_TOO_LATE_FOR_LEAD')
      cr['mechanism_tags']=json.dumps(tags)
 pd.DataFrame(availability).to_parquet(run/'private/audit_origin_availability_PRIVATE.parquet',index=False)
 er=pd.DataFrame(eventrows);cr=pd.DataFrame(capturerows)
 er.to_parquet(run/'private/event_information_comparison_PRIVATE.parquet',index=False);cr.to_parquet(run/'private/event_capture_comparison_PRIVATE.parquet',index=False)
 er.groupby(['clock','lead_months']).agg(events=('event_id','nunique'),eligible_events=('opportunity','sum'),early_core_signal_events=('early_core_signal','sum'),early_opinion_signal_events=('early_opinion_signal','sum'),other_governance_signal_events=('early_other_governance','sum'),current_key_opinion_early_events=('current_key_opinion_before_eligible_origin','sum'),key_opinion_too_late_events=('key_opinion_too_late','sum'),exact_signal_already_in_input=('exact_information_in_input','sum')).reset_index().to_csv(run/'results/audit_timing_summary.csv',index=False)
 cr.groupby(['clock','lead_months','top_fraction','model','information_state']).agg(events=('event_id','nunique'),captured=('captured','sum'),audit_signal_before_first_warning=('audit_signal_before_first_warning','sum')).reset_index().to_csv(run/'results/model_capture_by_information_status.csv',index=False)
 cr[cr.model.eq('q2')].groupby(['clock','lead_months','top_fraction']).agg(unrestricted_not_q2=('gap_unrestricted_not_q2','sum'),q2_not_unrestricted=('q2_unique_vs_unrestricted','sum'),all_three_captured=('all_three_captured','sum'),all_three_missed=('all_three_missed','sum'),eligible_events=('opportunity','sum')).reset_index().to_csv(run/'results/model_pairwise_capture.csv',index=False)
 write(run/'audit/origin_join.json',{'frozen_event_origin_rows':len(orig),'evidence_origin_rows':len(availability),'capture_event_model_budget_clock_lead_rows':len(cr),'all_frozen_capture_counts_reproduced':True,'calendar_months_not_fixed_days':True,'same_day_not_available':True,'event_key_disclosures_never_backdated':True})
 print((run/'results/audit_timing_summary.csv').read_text());print((run/'results/model_pairwise_capture.csv').read_text())
if __name__=='__main__':
 import sys;main(sys.argv[1])
