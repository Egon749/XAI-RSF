"""Aggregate unresolved coverage without conflating no report, inapplicability or normality."""
from pathlib import Path
import json
import pandas as pd
from review_common import write,sha

def main(run):
 run=Path(run);mapping=pd.read_parquet(run/'private/reason_mapping_scope_PRIVATE.parquet');coverage=pd.read_parquet(run/'private/audit_coverage_PRIVATE.parquet');timeline=pd.read_parquet(run/'private/audit_event_timeline_PRIVATE.parquet');rows=[]
 for role,z in mapping[mapping.outcome.eq('UNRESOLVED')].groupby('role'):
  rows.append(dict(scope='reason_mapping',status='IMPLEMENTATION_OR_CURRENT_CAUSE_UNRESOLVED',role_or_type=role,unit='independent_events',count=z.event_id.nunique(),interpretation='Not confirmed errors; no proposed label change.'))
 for (typ,status),z in coverage.groupby(['type','status']):
  if status=='HAS_REVIEWED_EVIDENCE':continue
  rows.append(dict(scope='audit_type_coverage',status=status,role_or_type=typ,unit='event_fiscal_year_type_slots',count=len(z),interpretation='No verified record is not proof of no signal or report absence; subtype sections may remain unreviewed. Explicit not-applicable is separate.'))
 rows.extend([dict(scope='source_feasibility',status='NO_FULL_RISKSET_CLOCKED_PANEL_VERIFIED',role_or_type='company_year',unit='blocking_requirement',count=1,interpretation='Event-conditioned BDT audit fields cannot supply outcome-independent coverage or actual audit-public clocks.'),dict(scope='source_access',status='MCP_TUNNEL_UNAVAILABLE_LOCAL_REGISTRY_FALLBACK',role_or_type='registry',unit='access_path',count=1,interpretation='Read-only local schema/quality inspection succeeded; no source mutation.'),dict(scope='source_version',status='DUPLICATE_ANNUAL_VERSION_LINKAGE_NOT_CERTIFIED',role_or_type='annual_report',unit='documents',count=1,interpretation='Earlier duplicate not used to establish first availability; independently dated later release used.'),dict(scope='signing_dates',status='SIGNING_DATE_NOT_TRANSCRIBED_OR_UNVERIFIED',role_or_type='all_types',unit='evidence_rows',count=int(timeline.audit_signed_at.isna().sum()),interpretation='Does not substitute for public date; official publication clock separately verified.')])
 pd.DataFrame(rows).to_csv(run/'results/unresolved_evidence.csv',index=False)
 receipts=json.loads((run/'private/audit_document_retrieval.json').read_text());manifest={}
 for r in receipts:
  for q in r['queries']:manifest[q['file']]={'sha256':sha(q['file']),'kind':'official_query_response'}
  for d in r['documents']:
   for p in [Path(d['file']),Path(d['file']).with_suffix('.txt')]:manifest[str(p)]={'sha256':sha(p),'kind':'official_document_or_extracted_text'}
 for p in (run/'private').glob('mapping_supplement*'):
  manifest[str(p)]={'sha256':sha(p),'kind':'single_mapping_gap_supplement'}
 write(run/'audit/new_evidence_manifest.json',manifest)
 write(run/'audit/evidence_counts.json',{'cohort':18,'queries':sum(len(r['queries']) for r in receipts),'cohort_pdf_documents':sum(len(r['documents']) for r in receipts),'new_mapping_pdf_documents':1,'network_failures':sum(len(r['failures']) for r in receipts),'source_reviewed_cohort_documents':timeline.document_id.nunique(),'coded_evidence_rows':len(timeline),'coverage_slots':len(coverage),'fully_coded_every_subtype':False})
if __name__=='__main__':
 import sys;main(sys.argv[1])
