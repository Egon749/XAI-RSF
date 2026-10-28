"""Build a private timeline only from explicit, source-reviewed annotations.

The annotation file is private because it contains company identities and excerpts.
Candidate extraction is never an automatic VERIFIED decision.
"""
from pathlib import Path
import json
import pandas as pd
from review_common import write,sha
from verify_source_clocks import document_clock
TYPES=['FINANCIAL_AUDIT','EXTERNAL_INTERNAL_CONTROL_AUDIT','SELF_ASSESSMENT','GOING_CONCERN','MATERIAL_CONTROL_DEFICIENCY','REMEDIATION']

def main(run):
 run=Path(run);cohort=json.loads((run/'private/cohort.json').read_text());events={e['event_id']:e for e in cohort}
 receipts=json.loads((run/'private/audit_document_retrieval.json').read_text());docs={d['document_id']:dict(event_id=r['event_id'],**d) for r in receipts for d in r['documents']}
 annotations=json.loads((run/'private/audit_review_annotations.json').read_text());rows=[]
 for a in annotations:
  d=docs[a['document_id']];e=events[d['event_id']];assert a['review_status']=='SOURCE_REVIEWED';assert a['evidence_excerpt'] and a['evidence_locator']
  pub=document_clock(d);typ=a['audit_report_type'];assert typ in TYPES
  version='ORIGINAL_ARCHIVED' if '更新前' in d['title'] else d['version']
  rows.append(dict(event_id=e['event_id'],group=e['group'],target_event_announcement_at=e['announcement_date'],target_event_implementation_at=e['event_date'],other_trigger_families=e['verified_families'],audit_report_fiscal_year=a['fiscal_year'],audit_report_type=typ,audit_signed_at=a.get('signed_at'),audit_actual_published_at=pub,document_version=d['document_id'],original_or_amended=version,available_at=pub,prior_public_signal_type=a['opinion_or_finding'],prior_signal_published_at=pub,source_url_or_archive_id=d['url'],source_hash=d['sha256'],evidence_excerpt=a['evidence_excerpt'],evidence_locator=a['evidence_locator'],timing_verification_status='VERIFIED_OFFICIAL_ARCHIVE_DATE',review_status=a['review_status'],opinion_status=a.get('opinion_status','VERIFIED'),abnormal=a['abnormal'],signal_scope=a.get('signal_scope','CORE_AUDIT'),source_relation=a.get('source_relation','DIRECT_DOCUMENT'),state_note=a.get('state_note','Historical observation; no assumption of unresolved persistence.'),input_status='ONLY_NON_EQUIVALENT_DISTRESS_PROXIES',same_information_in_original_input=False,is_event_year_report=a['fiscal_year']==pd.Timestamp(e['announcement_date']).year-1,document_id=d['document_id']))
 timeline=pd.DataFrame(rows);assert timeline.event_id.nunique()==18;timeline.to_parquet(run/'private/audit_event_timeline_PRIVATE.parquet',index=False)
 coverage=[]
 for e in cohort:
  y=pd.Timestamp(e['announcement_date']).year
  for fy in range(y-3,y):
   for typ in TYPES:
    z=timeline[(timeline.event_id==e['event_id'])&(timeline.audit_report_fiscal_year==fy)&(timeline.audit_report_type==typ)]
    coverage.append(dict(event_id=e['event_id'],fiscal_year=fy,type=typ,status='NO_VERIFIED_RECORD_IN_BOUNDED_REVIEW' if z.empty else ('NOT_APPLICABLE_EXPLICIT' if z.opinion_status.eq('NOT_APPLICABLE_EXPLICIT').all() else 'HAS_REVIEWED_EVIDENCE'),evidence_rows=len(z)))
 pd.DataFrame(coverage).to_parquet(run/'private/audit_coverage_PRIVATE.parquet',index=False)
 write(run/'audit/timeline_build.json',{'cohort_events':18,'evidence_rows':len(timeline),'source_reviewed_documents':timeline.document_id.nunique(),'six_type_coverage_slots':len(coverage),'private_annotations_sha256':sha(run/'private/audit_review_annotations.json'),'normal_not_imputed_for_missing':True,'all_six_types_separate':True})
 return timeline
if __name__=='__main__':
 import sys;main(sys.argv[1])
