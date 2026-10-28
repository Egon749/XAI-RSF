"""Every frozen event retained. Automated text tags are candidates, never VERIFIED."""
from pathlib import Path
import json,re
import pandas as pd
from rule_registry import board_of
PATTERNS={
 'REVENUE_PROFIT':r'(?:净利润|利润总额).{0,150}(?:为负|负值|负数|亏损).{0,120}(?:营业收入|营收).{0,80}(?:低于|不足|小于)|营业收入.{0,80}(?:低于|不足|小于).{0,100}(?:净利润|利润总额)',
 'TWO_YEAR_LOSS':r'(?:最近两|连续两|两个会计年度).{0,70}(?:亏损|负值|为负)',
 'THREE_LOSS_GOING_CONCERN':r'(?:三年|三个会计年度).{0,100}(?:为负|负值|亏损).{0,180}持续经营|连续三年亏损',
 'NEGATIVE_EQUITY':r'净资产.{0,70}(?:为负|负值|负数|小于零)',
 'FINANCIAL_AUDIT':r'(?:财务报[表告]|财务会计报告|财务报告|年度报告).{0,65}(?:无法表示意见|否定意见)|(?:无法表示意见|否定意见).{0,35}(?:财务报表|财务会计报告|财务报告审计)',
 'INTERNAL_CONTROL':r'内部控制.{0,60}(?:无法表示|否定意见)|(?:无法表示意见|否定意见).{0,35}内部控制',
 'FRAUD_SANCTION':r'(?:行政处罚|事先告知书).{0,200}(?:虚假记载|虚增)|(?:虚假记载|虚增).{0,200}(?:行政处罚|事先告知书)',
 'OCCUPATION_GUARANTEE':r'非经营性.{0,20}占用|违规担保|违反规定程序对外提供担保',
 'BANK_FREEZE':r'主要银行.{0,15}冻结',
 'OPERATIONS':r'生产经营活动受到严重影响|停产.{0,30}三个月',
 'NONFILING':r'未在法定期限内披露|未能在法定期限内披露',
 'REORGANIZATION':r'法院.{0,60}受理.{0,30}(?:重整|破产|和解)',
 'DIVIDEND':r'累计现金分红.{0,150}(?:低于|不足)',
 'RESTATEMENT':r'追溯重述|追溯调整|前期会计差错|更正.*财务',
}
TARGET={'REVENUE_PROFIT','TWO_YEAR_LOSS','THREE_LOSS_GOING_CONCERN','NEGATIVE_EQUITY','FINANCIAL_AUDIT','INTERNAL_CONTROL','REORGANIZATION'}
def clean(t):return re.sub(r'\s+','',t)
def main(run,out):
 run=Path(run);out=Path(out);events=json.loads((run/'private/event_inventory_PRIVATE.json').read_text());retrieval={r['event_id']:r for r in json.loads((run/'private/official_event_retrieval.json').read_text())};reviews=json.loads((out/'event_manual_review_PRIVATE.json').read_text()) if (out/'event_manual_review_PRIVATE.json').exists() else {}
 rows=[];packets=[]
 for e in events:
  q=retrieval[e['event_id']];docs=[x for x in q['documents'] if not any(w in x['title'] for w in ['受托管理','债券','财务顾问'])];texts=[clean((run/x['file']).with_suffix('.txt').read_text()) for x in docs]
  if not texts:texts=[clean(str(x.get('Comments',''))) for x in e['archived_announcements']]
  text='\n'.join(texts);review=reviews.get(e['event_id']);families=[k for k,p in PATTERNS.items() if re.search(p,text)];clauses=sorted(set(re.findall(r'\b\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\b',text)));years=sorted(set(re.findall(r'(20\d\d)年度',text)));versions=sorted(set(re.findall(r'(20\d\d)年(?:\d{1,2}月)?修订',text)))
  excerpts={}
  for fam in families:
   m=re.search(PATTERNS[fam],text);excerpts[fam]=text[max(0,m.start()-50):m.end()+110]
  base={k:e[k] for k in ['event_id','code','group','frozen_event_type','event_date','announcement_date','role','window_rows']};base.update(board=board_of(e['code']),frozen_reason=e.get('frozen_reason'),evidence_level='OFFICIAL_IMPLEMENTATION_CANDIDATE' if docs else 'ARCHIVE_TRANSCRIPTION_ONLY',official_documents=json.dumps(docs,ensure_ascii=False),candidate_trigger_families=json.dumps(families),clause_candidates=json.dumps(clauses),fiscal_year_candidates=json.dumps(years),version_candidates=json.dumps(versions),candidate_snippets=json.dumps(excerpts,ensure_ascii=False),use='RETROSPECTIVE_DIAGNOSTIC_ONLY',verification_status='UNRESOLVED',verified_families='[]',multiple_triggers=False,transition_applies=False,conflict_with_frozen_label=False,review_reason='未完成逐条法律适用核验；候选关键词不能证明实际触发',verified_fiscal_year=None,verified_rule_version=None)
  if review:
   base.update(review);base['verified_families']=json.dumps(review['verified_families']);base['multiple_triggers']=len(set(review['verified_families'])-{'RESTATEMENT'})>1
  rows.append(base);packets.append(dict(event_id=e['event_id'],label=e['frozen_event_type'],role=e['role'],official=bool(docs),versions=versions,clauses=clauses,families=families,snippets=excerpts,text=text))
 frame=pd.DataFrame(rows);frame.to_parquet(run/'private/event_rule_evidence.parquet',index=False);(run/'private/review_packets_PRIVATE.json').write_text(json.dumps(packets,ensure_ascii=False,indent=2));return frame
if __name__=='__main__':
 import sys;main(sys.argv[1],Path(sys.argv[1])/'private')
