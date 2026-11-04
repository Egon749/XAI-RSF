"""Archive issuer disclosures separately from exchange decisions; search by security and episode."""
from concurrent.futures import ThreadPoolExecutor
from subprocess import run as call
from .collect_decisions import HEAD,fetch
from .common import *
import requests
from datetime import datetime,timezone,timedelta
def one(e,org):
    folder=run()/'filings'/e.code;folder.mkdir(parents=True,exist_ok=True)
    def query(key,start,end):
        path=folder/(key+'.json')
        if path.exists():return load(path)
        found=[]
        for page in range(1,13):
            d=dict(pageNum=page,pageSize=30,column='sse' if e.exchange=='SSE' else 'szse',tabName='fulltext',stock=e.code+','+org,seDate=start+'~'+end,searchkey=key, isHLtitle='true',plate='')
            res=requests.post('https://www.cninfo.com.cn/new/hisAnnouncement/query',data=d,headers=HEAD|{'Referer':'https://www.cninfo.com.cn/'},timeout=18);res.raise_for_status();j=res.json();found+=j.get('announcements') or []
            if not j.get('hasMore'):break
        dump(path,found);return found
    end=str((pd.Timestamp(e.stop)+pd.Timedelta(days=75)).date());start=str((pd.Timestamp(e.origin)-pd.Timedelta(days=180)).date())
    docs={}; errors=[]
    for key in ['终止上市','摘牌','风险提示']:
        try:
            for d in query(key,start,end):
                if d['secCode']!=e.code:continue
                d['title']=re.sub('<[^>]+>','',d['announcementTitle']);d['pub']=datetime.fromtimestamp(d['announcementTime']/1000,timezone(timedelta(hours=8))).date().isoformat();docs[d['announcementId']]=d
        except Exception as ex:errors.append(str(ex))
    ordered=sorted(docs.values(),key=lambda d:d['pub']);selected={}
    for d in ordered:
        title=d['title']
        if re.search('摘牌|收到.*终止上市.*决定|终止.*上市的决定|终止上市决定|股票终止上市的公告|股票被终止上市的公告|终止上市.*复核决定',title) and not re.search('可能|风险提示|转债.*摘牌(公告)?$',title):
            selected[d['announcementId']]=d
    for pattern in ['低于1元|低于人民币1元|交易类|低于面值','市值','重大违法','财务类','首次.*风险|第一次.*风险|风险提示.*第一次','终止上市.*风险提示|终止上市风险']:
        for d in [v for v in ordered if re.search(pattern,v['title'])][:2]:selected[d['announcementId']]=d
    for d in selected.values():
        try:
            p=folder/(d['announcementId']+'.pdf')
            if not p.exists():
                raw=fetch('https://static.cninfo.com.cn/'+d['adjunctUrl'])
                if raw[:4]!=b'%PDF':raise ValueError('non-PDF')
                p.write_bytes(raw)
            txt=p.with_suffix('.txt')
            if not txt.exists():call(['pdftotext','-layout',str(p),str(txt)],check=True,capture_output=True)
            d['pdf_sha256']=sha(p)
        except Exception as ex:errors.append(str(ex)[:160])
    dump(folder/'index.json',dict(selected=list(selected.values()),failures=errors,archive_titles=len(docs)))
if __name__=='__main__':
    ids={d['code']:d['orgId'] for d in load(Path(os.environ['CAUSE_POLICY_RUN'])/'private/cninfo_stock_lookup.json')['stockList']}
    events=pd.read_csv(run()/'events_PRIVATE.csv',dtype={'code':str})
    with ThreadPoolExecutor(5) as ex:list(ex.map(lambda e:one(e,ids[e.code]),events.itertuples()))
