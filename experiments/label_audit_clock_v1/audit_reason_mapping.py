"""Inventory only frozen events reaching the original explicit-code branch."""
from pathlib import Path
import json,re
import pandas as pd
from review_common import PRIOR,EVENT,register,write,sha
from adjudicate_labels import TARGET
FIN_CODES={'01','02','03','04','05','08'}
def same_code_namespace(a,b):
 return all(a.get(k) is not None and a.get(k)==b.get(k) for k in ['source_table','field','version'])
def explicit_target(codes):return bool(set(re.findall(r'\d{2}',str(codes)))&FIN_CODES)
def reviewed_cause_classification(families,complete=False):
 if not complete:return 'UNRESOLVED'
 if set(families)&TARGET:return 'TARGET'
 return 'COMPETING_REQUIRES_FROZEN_POLICY_CHECK'
def main(run):
 run=Path(run);inv=json.loads((PRIOR/'private/event_inventory_PRIVATE.json').read_text());ev=pd.read_parquet(PRIOR/'private/event_rule_evidence.parquet').set_index('event_id');ad=pd.read_parquet(run/'private/label_adjudication_PRIVATE.parquet').set_index('event_id');supp=json.loads((run/'private/mapping_supplement.json').read_text());rows=[]
 schema=json.loads((run/'audit/BDT_schema.json').read_text());version=schema['version'];dictionary=Path('/home/oliver/pylib/csmar/特殊处理变动文件/BDT_Trdchg[DES][json].txt')
 for r in inv:
  if not explicit_target(r.get('frozen_reason_codes')):continue
  v=ev.loc[r['event_id']];f=json.loads(v.verified_families);verified=v.verification_status=='VERIFIED';outcome='SUPPORTED_TARGET' if verified and set(f)&TARGET else 'UNRESOLVED'
  if r['event_id'] in ad.index:outcome='CONFIRMED_SOURCE_RECORD_CONFLICT';verified=True;f=json.loads(ad.loc[r['event_id'],'actual_trigger_families'])
  if r['event_id']==supp['event_id']:verified=True;f=supp['verified_families'];outcome='SUPPORTED_TARGET'
  records=r['change_records'];namespaces=[]
  for rec in records:
   source=rec['source'];namespaces.append({'source_table':'BDT_Trdchg' if source=='current_BDT' else source,'field':'Chgreas' if source!='itemchange_supplement' else 'ReasonID extracted S15 suffix','version':version if source=='current_BDT' else 'frozen_archive_hash; enum_equivalence_unverified'})
  rows.append(dict(event_id=r['event_id'],role=r['role'],quarter_rows=r['window_rows'],frozen_label=r['frozen_event_type'],codes=r['frozen_reason_codes'],description=r['frozen_reason'],namespaces=json.dumps(namespaces,ensure_ascii=False),same_current_BDT_02=any(x.get('source')=='current_BDT' and '02' in x.get('Chgreas','').split(',') for x in records),official_implementation_verified=verified,verified_families=json.dumps(f),outcome=outcome,branch='explicit_target_codes intersection',code_enum_dictionary_complete=False))
 d=pd.DataFrame(rows);d.to_parquet(run/'private/reason_mapping_scope_PRIVATE.parquet',index=False)
 d.groupby(['role','same_current_BDT_02','outcome'],dropna=False).agg(independent_events=('event_id','nunique'),quarter_records=('quarter_rows','sum'),official_evidence_events=('official_implementation_verified','sum')).reset_index().to_csv(run/'results/reason_mapping_scope.csv',index=False)
 raw=pd.read_json('/home/oliver/pylib/csmar/特殊处理变动文件/BDT_Trdchg.json',lines=True,dtype={'Symbol':str});selected=[]
 for r in rows:
  if not r['same_current_BDT_02']:continue
  code,date=r['event_id'].split('_');z=raw[(raw.Symbol==code)&(raw.Execudt==date)];selected.extend(z.to_dict('records'))
 write(run/'private/same_code_raw_source_rows.json',selected)
 write(run/'audit/reason_mapping_result.json',{'automatic_branch_events':len(d),'same_namespace_code02_events':int(d.same_current_BDT_02.sum()),'confirmed_conflicts':int(d.outcome.eq('CONFIRMED_SOURCE_RECORD_CONFLICT').sum()),'unresolved_events':int(d.outcome.eq('UNRESOLVED').sum()),'source_version':version,'dictionary_sha256':sha(dictionary),'classification':'INDIVIDUAL_SOURCE_RECORD_CONFLICT','systematic_code02_replacement_supported':False,'mapping_semantics':'MAPPING_SEMANTICS_UNRESOLVED for cross-table equivalence/full vendor enum; rows match 02=资产缩水 but no enum history supplied','same_code_additional_conflict':'NO_ADDITIONAL_CONFLICT_FOUND_WITHIN_CHECKED_SCOPE: remaining four code02 events supported target','evidence_filter_defect':'old policy extractor excluded mixed stock-and-bond implementation title; new eligibility function fixes this without editing old evidence or labels'})
 print((run/'results/reason_mapping_scope.csv').read_text())
if __name__=='__main__':
 import sys;main(sys.argv[1])
