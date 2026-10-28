"""Candidate excerpts for direct source review. Never certifies a signal automatically."""
from pathlib import Path
import json,re
from review_common import write

def packet(doc):
 text=Path(doc['file']).with_suffix('.txt').read_text();flat=re.sub(r'\s+','',text);hits=[]
 # Include negated statements and context to avoid keyword-only positive labels.
 patterns=[r'一[、.．]?审计意见',r'一[、.．]?否定意见',r'一[、.．]?无法表示意见',r'一[、.．]?保留意见',r'一[、.．]?强调事项',r'我们认为',r'二[、.．]?内部控制评价结论',r'内部控制存在.{0,15}缺陷',r'财务报告内部控制.{0,25}重大缺陷',r'非财务报告内部控制.{0,25}重大缺陷',r'与持续经营相关的重大不确定性',r'无保留意见.{0,12}强调',r'重大缺陷.{0,25}整改',r'上年度.{0,40}审计意见']
 for pat in patterns:
  for m in list(re.finditer(pat,flat))[:4]:hits.append({'offset':m.start(),'excerpt':flat[max(0,m.start()-50):m.end()+420]})
 hits=sorted({x['offset']:x for x in hits}.values(),key=lambda x:x['offset'])
 return dict(**doc,characters=len(flat),excerpt_candidates=hits,review_status='UNREVIEWED')
def main(run):
 run=Path(run);packets=[]
 for receipt in sorted((run/'private/audit_documents').glob('*/receipt.json')):
  r=json.loads(receipt.read_text());packets.extend([dict(event_id=r['event_id'],**packet(d)) for d in r['documents']])
 write(run/'private/audit_review_packets.json',packets);print('candidate documents',len(packets))
if __name__=='__main__':
 import sys;main(sys.argv[1])
