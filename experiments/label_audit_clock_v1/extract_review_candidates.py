"""Quoted annual-report fields for manual confirmation, not automated adjudication."""
from pathlib import Path
import json,re
from review_common import write

def main(run):
 run=Path(run);docs=json.loads((run/'private/audit_review_packets.json').read_text());rows=[]
 for d in docs:
  if d['kind']!='ANNUAL_REPORT':continue
  text=Path(d['file']).with_suffix('.txt').read_text();t=re.sub(r'\s+','',text)
  extracts={}
  for name,pat in [('financial_type',r'(?<!内控)审计意见类型'),('internal_type',r'内控审计报告意见类型'),('signed',r'审计报告签署日期'),('major_financial',r'财务报告重大缺陷数量[（(]个[）)]'),('major_nonfinancial',r'非财务报告重大缺陷数量[（(]个[）)]')]:
   m=list(re.finditer(pat,t));m=m[-1] if m else None
   extracts[name]=t[max(0,m.start()-40):m.end()+90] if m else ''
  rows.append(dict(**{k:d[k] for k in ['event_id','document_id','file','url','sha256','published_at','title','fiscal_year_hint','version']},extracts=extracts,review_status='UNREVIEWED'))
 write(run/'private/annual_field_candidates.json',rows)
 for r in rows:print(r['event_id'],r['document_id'],r['published_at'],r['fiscal_year_hint'],r['extracts']['financial_type'][:145],r['extracts']['internal_type'][:145])
if __name__=='__main__':
 import sys;main(sys.argv[1])
