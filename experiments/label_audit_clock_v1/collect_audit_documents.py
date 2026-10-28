"""Bounded official-document search for the frozen 18-event audit cohort only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,urllib.parse,json,re,subprocess
import pandas as pd
from review_common import PRIOR,sha,write

def query(payload):
 req=urllib.request.Request('https://www.cninfo.com.cn/new/hisAnnouncement/query',data=urllib.parse.urlencode(payload).encode(),headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.cninfo.com.cn/'})
 return urllib.request.urlopen(req,timeout=30).read()
def kind(title):
 if any(x in title for x in ['摘要','英文','取消','监事会','独立董事','董事会决议','会议决议','保荐','核查意见']):return None
 if '内部控制' in title:
  if any(w in title for w in ['专项说明','涉及事项','非标准意见的说明']):return 'IC_EXPLANATORY_DOCUMENT'
  if '审计报告' in title or '鉴证报告' in title:return 'EXTERNAL_INTERNAL_CONTROL_AUDIT'
  if '评价报告' in title:return 'SELF_ASSESSMENT'
  if '整改' in title:return 'REMEDIATION'
  if any(x in title for x in ['说明','意见']):return 'IC_EXPLANATORY_DOCUMENT'
 if '审计报告' in title and not any(x in title for x in ['专项','资金','募集','备考']):return 'FINANCIAL_AUDIT_REPORT'
 if re.search(r'20\d{2}年年度报告(?:\(|（|$)',title) or re.search(r'20\d{2}年度报告(?:\(|（|$)',title):return 'ANNUAL_REPORT'
 if '整改' in title:return 'REMEDIATION'
 return None

def main(run):
 run=Path(run);out=run/'private/audit_documents';out.mkdir(exist_ok=True)
 ev=pd.read_parquet(PRIOR/'private/event_rule_evidence.parquet');ev=ev[(ev.role=='eval')&(ev.frozen_event_type==1)&(ev.verification_status=='VERIFIED')&ev.verified_families.str.contains('INTERNAL_CONTROL')];assert len(ev)==18
 lookup={x['code']:x for x in json.loads((PRIOR/'private/cninfo_stock_lookup.json').read_text())['stockList']}
 def one(e):
  folder=out/e['event_id'];folder.mkdir(exist_ok=True);receipt=folder/'receipt.json'
  if receipt.exists():return json.loads(receipt.read_text())
  date=pd.Timestamp(e['announcement_date']);year=date.year;entry=lookup.get(e['code']);row={'event_id':e['event_id'],'queries':[],'documents':[],'failures':[]}
  if not entry:row['failures'].append('NO_STOCK_LOOKUP');write(receipt,row);return row
  seen={}
  for keyword in ['内部控制','审计报告','年度报告','整改']:
   for page in range(1,4):
    payload=dict(pageNum=page,pageSize=100,column='sse' if e['code'].startswith('6') else 'szse',tabName='fulltext',seDate=f'{year-2}-01-01~{date.date()}',isHLtitle='false',stock=e['code']+','+entry['orgId'],searchkey=keyword,plate='')
    try:
     raw=query(payload);qp=folder/f'query_{keyword}_{page}.json';qp.write_bytes(raw);res=json.loads(raw);row['queries'].append({'payload':payload,'file':str(qp),'sha256':sha(qp),'total':res.get('totalAnnouncement'),'truncated':page==3 and res.get('totalAnnouncement',0)>300})
     for x in res.get('announcements') or []:
      if x['secCode']!=e['code']:continue
      title=re.sub('<[^>]+>','',x['announcementTitle']);k=kind(title)
      if not k:continue
      fy=re.findall(r'(20\d{2})年?',title);fiscal=int(fy[0]) if fy else None
      if fiscal and fiscal not in [year-3,year-2,year-1]:continue
      seen[str(x['announcementId'])]=(x,title,k,fiscal)
     if not res.get('hasMore'):break
    except Exception as ex:row['failures'].append({'keyword':keyword,'page':page,'error':str(ex)});break
  # Keep original and amended reports; no selection using model predictions.
  for aid,(x,title,k,fy) in sorted(seen.items()):
   url='https://static.cninfo.com.cn/'+x['adjunctUrl'];p=folder/(aid+'.pdf')
   try:
    b=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30).read();assert b.startswith(b'%PDF');p.write_bytes(b)
    subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True,stderr=subprocess.PIPE)
    published=pd.Timestamp(x['announcementTime'],unit='ms',tz='UTC').tz_convert('Asia/Shanghai').date()
    row['documents'].append({'document_id':aid,'title':title,'kind':k,'fiscal_year_hint':fy,'published_at':str(published),'official_public_timestamp_ms':x['announcementTime'],'url':url,'file':str(p),'sha256':sha(p),'text_sha256':sha(p.with_suffix('.txt')),'version':'ORIGINAL_ARCHIVED' if '更新前' in title else ('AMENDED' if any(w in title for w in ['修订','更新','更正']) else 'ORIGINAL_OR_UNSPECIFIED')})
   except Exception as ex:row['failures'].append({'document_id':aid,'url':url,'error':str(ex)})
  write(receipt,row);print(e['event_id'],'documents',len(row['documents']),'failures',len(row['failures']),flush=True);return row
 with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,ev.to_dict('records')))
 write(run/'private/audit_document_retrieval.json',results);print('COMPLETE',len(results),sum(len(r['documents']) for r in results),flush=True)
if __name__=='__main__':
 import sys;main(sys.argv[1])
