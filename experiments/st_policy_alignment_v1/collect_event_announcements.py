"""Bounded public announcement retrieval for EVERY frozen event, no outcome edits."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta
import urllib.request,urllib.parse,json,re,hashlib,subprocess,time

def main(run):
 run=Path(run);out=run/'private/official_announcements';out.mkdir();events=json.loads((run/'private/event_inventory_PRIVATE.json').read_text());lookup={x['code']:x for x in json.loads((run/'private/cninfo_stock_lookup.json').read_text())['stockList']}
 def one(e):
  eid=e['event_id'];code=e['code'];folder=out/eid;folder.mkdir();date=datetime.fromisoformat(e['event_date']).date();entry=lookup.get(code);payload=dict(pageNum=1,pageSize=100,column='sse' if code.startswith('6') else 'szse',tabName='fulltext',seDate=f'{date-timedelta(days=15)}~{date+timedelta(days=3)}',isHLtitle='true',plate='')
  if entry:payload.update(stock=code+','+entry['orgId'],searchkey='')
  else:
   title=next((x.get('ValueBefore','') for x in e['archived_announcements'] if x.get('ValueBefore')),'');payload.update(stock='',searchkey=title)
  row=dict(event_id=eid,query=payload,status='UNRESOLVED',documents=[])
  try:
   req=urllib.request.Request('https://www.cninfo.com.cn/new/hisAnnouncement/query',data=urllib.parse.urlencode(payload).encode(),headers={'User-Agent':'Mozilla/5.0','Referer':'https://www.cninfo.com.cn/'})
   raw=urllib.request.urlopen(req,timeout=15).read();(folder/'query.json').write_bytes(raw);res=json.loads(raw);row['query_sha256']=hashlib.sha256(raw).hexdigest();row['total_announcement']=res.get('totalAnnouncement');row['truncated']=res.get('totalAnnouncement',0)>100
   items=[x for x in (res.get('announcements') or []) if x['secCode']==code]
   selected=[]
   for x in items:
    title=re.sub('<[^>]+>','',x['announcementTitle'])
    if any(t in title for t in ['撤销','申请','可能','提示性','进展','异动','异常波动','终止上市','问询','回复','年度报告','董事会','监事会','核查']):continue
    if re.search('风险警示|特别处理',title) and re.search('实施|实行|变更|调整|叠加',title):selected.append((x,title))
   for x,title in selected[:3]:
    url='https://static.cninfo.com.cn/'+x['adjunctUrl'];b=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=15).read()
    if not b.startswith(b'%PDF'):raise ValueError('Announcement is not PDF')
    filename=str(x['announcementId'])+'.pdf';p=folder/filename;p.write_bytes(b);subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True,stderr=subprocess.PIPE)
    row['documents'].append(dict(url=url,title=title,official_public_timestamp_ms=x['announcementTime'],file=str(p.relative_to(run)),sha256=hashlib.sha256(b).hexdigest(),text_sha256=hashlib.sha256(p.with_suffix('.txt').read_bytes()).hexdigest()))
   row['status']='RETRIEVED_CANDIDATE' if row['documents'] else 'NO_IMPLEMENTATION_PDF_IN_BOUNDED_QUERY'
  except Exception as ex:row['error']=str(ex)
  (folder/'receipt.json').write_text(json.dumps(row,ensure_ascii=False,indent=2));return row
 with ThreadPoolExecutor(max_workers=4) as pool:
  results=[]
  for i,row in enumerate(pool.map(one,events)):
   results.append(row)
   if (i+1)%40==0:print('Queried',i+1,'events',sum(bool(x['documents']) for x in results),'with candidate PDFs',flush=True)
 (run/'private/official_event_retrieval.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print('COMPLETE',len(results),sum(bool(x['documents']) for x in results),flush=True)
if __name__=='__main__':
 import sys;main(sys.argv[1])
