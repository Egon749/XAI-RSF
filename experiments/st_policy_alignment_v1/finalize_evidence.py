"""Materialize one-event-many-evidence structure and bounded policy consistency cases."""
from pathlib import Path
import json,re,hashlib
import pandas as pd

def main(run):
 run=Path(run);ev=pd.read_parquet(run/'private/event_rule_evidence.parquet');evidence=[];clocks=[]
 for e in ev.to_dict('records'):
  fams=json.loads(e['verified_families'] if e['verification_status']=='VERIFIED' else e['candidate_trigger_families']) or ['UNRESOLVED']
  for i,fam in enumerate(fams):
   evidence.append(dict(event_id=e['event_id'],evidence_id=e['event_id']+'_'+str(i),trigger_family=fam,verification_status=e['verification_status'],board=e['board'],frozen_event_type=e['frozen_event_type'],frozen_announcement_date=e['announcement_date'],frozen_implementation_date=e['event_date'],verified_fiscal_year=e['verified_fiscal_year'],quoted_rule_version=e['verified_rule_version'],rule_version_unresolved=e.get('rule_version_unresolved',True),official_documents=e['official_documents'],candidate_snippet=json.loads(e['candidate_snippets']).get(fam),is_automated_text_candidate=e['verification_status']!='VERIFIED',use='RETROSPECTIVE_DIAGNOSTIC_ONLY'))
  docs=json.loads(e['official_documents']);dates=sorted(set(re.search(r'finalpage/(\d{4}-\d{2}-\d{2})/',x['url']).group(1) for x in docs if re.search(r'finalpage/(\d{4}-\d{2}-\d{2})/',x['url'])))
  frozen=str(pd.Timestamp(e['announcement_date']).date())
  clocks.append(dict(event_id=e['event_id'],frozen_announcement=frozen,frozen_implementation=str(pd.Timestamp(e['event_date']).date()),source_public_dates=json.dumps(dates),frozen_date_present=frozen in dates,source_clock_status='NO_OFFICIAL_PDF' if not dates else 'EXACT_FROZEN_DATE_IN_SOURCES' if frozen in dates else 'DATE_DIFFERENCE_NEEDS_REVIEW_NOT_AUTOMATIC_CONFLICT'))
 pd.DataFrame(evidence).to_parquet(run/'private/event_rule_evidence_long.parquet',index=False);cl=pd.DataFrame(clocks);cl.to_parquet(run/'private/event_evidence_clock_audit_PRIVATE.parquet',index=False);cl.groupby('source_clock_status').size().rename('events').reset_index().to_csv(run/'results/event_clock_summary.csv',index=False)
 # Private case identities and exact amounts are intentionally not in Git.
 examples=json.loads((run/'private/reviewed_boundary_fact_input_PRIVATE.json').read_text())
 leads=pd.read_parquet(run/'private/event_leads_PRIVATE.parquet');case=[]
 for eid,revenue in examples:
  for _,v in leads[leads.event_id.eq(eid)].iterrows():case.append(dict(event_id=eid,model=v.model,top_fraction=v.top_fraction,actual_deducted_annual_revenue=revenue,actual_new_band_100m_to_300m=1e8<=revenue<3e8,detected=pd.notna(v.first_warning),announcement_detected_3m=v.announcement_detected_3m,announcement_detected_6m=v.announcement_detected_6m,interpretation='NEW_BOUNDARY_CASE_CAPTURED; NOT A MISSED_EVENT; not proof no alternative old trigger',use='RETROSPECTIVE_DIAGNOSTIC_ONLY'))
 ca=pd.DataFrame(case);ca.to_parquet(run/'private/reviewed_new_boundary_cases_PRIVATE.parquet',index=False);ca.groupby(['model','top_fraction']).agg(verified_new_band_events=('event_id','nunique'),detected=('detected','sum'),announcement_detected_3m=('announcement_detected_3m','sum'),announcement_detected_6m=('announcement_detected_6m','sum')).reset_index().to_csv(run/'results/new_boundary_case_summary.csv',index=False)
 ev[ev.verification_status.eq('VERIFIED')].assign(rule_version_gap=lambda d:d.rule_version_unresolved.fillna(True)).groupby(['role','rule_version_gap']).size().rename('events').reset_index().to_csv(run/'results/reviewed_version_gaps.csv',index=False)
 # Label conflicts are not fabricated from implementation PDF typos or publication channel dates.
 (run/'audit/evidence_status_semantics.json').write_text(json.dumps(dict(VERIFIED='official actual trigger and cited clause read; not blanket legal certification',UNRESOLVED='unreviewed trigger OR archive-only evidence; edition gaps additionally flagged',MULTIPLE_TRIGGERS='actual multiple trigger paths, excluding restatement as independent cause',TRANSITION_APPLIES='positive documented fiscal/transition application; not exhaustive count',clock_differences='source-date discrepancies retained separately; no labels changed',private_long_evidence_rows=len(evidence)),ensure_ascii=False,indent=2))
if __name__=='__main__':
 import sys;main(sys.argv[1])
