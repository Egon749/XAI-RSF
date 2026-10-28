"""Apply evidence gates to review decisions, writing an UNAPPLIED patch only."""
from pathlib import Path
import json
import pandas as pd
from review_common import PRIOR,BASE,EVENT,register,sha,write
TARGET={'REVENUE_PROFIT','TWO_YEAR_LOSS','THREE_LOSS_GOING_CONCERN','NEGATIVE_EQUITY','FINANCIAL_AUDIT','INTERNAL_CONTROL','REORGANIZATION'}
COMPETING={'OCCUPATION_GUARANTEE','BANK_FREEZE','OPERATIONS','FRAUD_SANCTION','NONFILING','DIVIDEND'}
def implementation_document_eligible(title):
 if any(w in title for w in ['受托管理','保荐','律师','法律意见','财务顾问','可能被','可能实施','提示性']):return False
 return ('股票' in title or '公司' in title) and any(w in title for w in ['实施','实行']) and any(w in title for w in ['风险警示','特别处理'])
def adjudicate(review,cutoff):
 if not review.get('identity_matched') or not review.get('official_implementation_verified'):return 'UNRESOLVED_IMPLEMENTATION_EVIDENCE'
 if not review.get('historical_clock_verified') or pd.Timestamp(review['published_at'])>=pd.Timestamp(cutoff):return 'HISTORICAL_CLOCK_UNRESOLVED'
 if not review.get('multiple_trigger_checked'):return 'UNRESOLVED_IMPLEMENTATION_EVIDENCE'
 families=set(review['actual_trigger_families'])
 if families&TARGET:return 'RETAIN_TARGET_MULTIPLE_TRIGGERS' if families&COMPETING else 'RETAIN_ORIGINAL_SUPPORTED'
 if not review.get('frozen_policy_supports_competing') or not families or not families<=COMPETING:return 'UNRESOLVED_CLASSIFICATION_POLICY'
 return 'CONFIRMED_TARGET_TO_COMPETING'
def main(run):
 run=Path(run);reviews=json.loads((run/'private/label_review_decisions.json').read_text());inventory={x['event_id']:x for x in json.loads((PRIOR/'private/event_inventory_PRIVATE.json').read_text())};source=BASE/'input/final_train_metadata_PRIVATE.parquet';d=pd.read_parquet(source);arr=BASE/'input/final_train_PRIVATE.npz';rows=[];patch=[]
 for r in reviews:
  eid=r['event_id'];e=inventory[eid];windows=d[d.event_id.eq(eid)];assert len(windows)==e['window_rows'];assert windows.event.eq(1).all()
  status=adjudicate(r,'2022-01-01');r.update(adjudication=status,original_label=1,original_reason_codes=e['frozen_reason_codes'],original_reason=e['frozen_reason'],source_records=e['change_records'],group=e['group'],code=e['code'],announcement_at=e['announcement_date'],implementation_at=e['event_date'],training_cutoff='2022-01-01',affected_sample_ids=windows.sample_id.tolist(),metadata_source=str(source),metadata_sha256=sha(source),array_source=str(arr),array_sha256=sha(arr),applied=False)
  rows.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
  if status=='CONFIRMED_TARGET_TO_COMPETING':
   for w in windows.to_dict('records'):
    patch.append(dict(event_id=eid,group=w['group'],sample_id=w['sample_id'],array_row=int(d.index[d.sample_id.eq(w['sample_id'])][0]),old_event=1,new_event=2,field='event',event_date=w['event_date'],announcement_date=w['announcement_date'],outcome_available=w['outcome_available'],stop_date=w['stop_date'],duration=w['duration'],origin=w['origin'],horizon=w['horizon'],reason=r['rationale'],evidence_url=r['url'],evidence_sha256=r['source_hash'],evidence_published_at=r['published_at'],metadata_source=str(source),metadata_sha256=r['metadata_sha256'],array_source=str(arr),array_sha256=r['array_sha256'],riskset_action='NO_CHANGE_NO_REENTRY_NO_FOLLOWUP_EXTENSION',applied=False))
 pd.DataFrame(rows).to_parquet(run/'private/label_adjudication_PRIVATE.parquet',index=False);p=pd.DataFrame(patch);p.to_parquet(run/'private/label_patch_PRIVATE.parquet',index=False)
 write(run/'private/label_patch_manifest.json',{'patch_sha256':sha(run/'private/label_patch_PRIVATE.parquet'),'metadata_sha256':sha(source),'array_sha256':sha(arr),'applied':False,'patch_kind':'rowwise cause change only; stale derived status fields require separate rebuild before any future training','no_performance_reestimated':True})
 pd.DataFrame(rows).groupby(['adjudication','applied'],dropna=False).agg(independent_events=('event_id','nunique')).reset_index().assign(quarter_records=len(p)).to_csv(run/'results/label_patch_summary.csv',index=False)
 return p
if __name__=='__main__':
 import sys;main(sys.argv[1])
