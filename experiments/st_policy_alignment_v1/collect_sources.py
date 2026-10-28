"""Bounded official rule/notice acquisition; immutable bytes and extracted text."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin
import urllib.request,hashlib,json,re,subprocess,zipfile,io,os,xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
SOURCES={
'SSE_MAIN_2024_NOTICE':'https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20240430_10785259.shtml',
'SSE_STAR_2024_NOTICE':'https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20240430_10777832.shtml',
'SZSE_MAIN_2024_NOTICE':'https://www.szse.cn/lawrules/rule/repeal/rules/t20240430_607069.html',
'SZSE_CHINEXT_2024_NOTICE':'https://www.szse.cn/lawrules/rule/repeal/rules/t20240430_607070.html',
'SSE_MAIN_2020_NOTICE':'https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20220107_5679292.shtml',
'SZSE_MAIN_2020_NOTICE':'https://www.szse.cn/disclosure/notice/general/t20201231_584052.html',
'SZSE_MAIN_2012_NOTICE':'https://www.szse.cn/disclosure/notice/company/t20120718_508731.html',
'SZSE_MAIN_2012_RULE':'https://docs.static.szse.cn/www/disclosure/notice/company/W020180328446393013021.pdf',
'SSE_MAIN_2012_EXPLANATION':'https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988587.shtml',
'SZSE_MAIN_2008_NOTICE':'https://www.szse.cn/disclosure/notice/company/t20080904_508301.html',
'SZSE_MAIN_2014_RULE':'https://docs.static.szse.cn/www/disclosure/notice/company/W020180328446480704524.pdf',
'SSE_MAIN_2014_NOTICE':'https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988787.shtml',
'SSE_MAIN_2018_APR_NOTICE':'https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20180420_4510860.shtml',
'SZSE_MAIN_2018_APR_RULE':'https://www.szse.cn/certificate/maind/mainrules/P020180724361500737638.pdf',
'SZSE_MAIN_2022_NOTICE':'https://www.szse.cn/disclosure/notice/general/t20220107_590541.html',
'SSE_MAIN_2023_FEB_NOTICE':'https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20230904_5726022.shtml',
'SZSE_MAIN_2023_AUG_NOTICE':'https://www.szse.cn/www/disclosure/notice/t20230804_602477.html',
'SZSE_MAIN_2023_AUG_RULE':'https://docs.static.szse.cn/www/lawrules/rule/repeal/rules/W020230804625303228188.pdf',
'SZSE_CHINEXT_2023_AUG_NOTICE':'https://www.szse.cn/lawrules/rule/repeal/rules/t20230804_602478.html',
'SZSE_CHINEXT_2012_NOTICE':'https://www.szse.cn/disclosure/notice/company/t20120420_508712.html',
'SSE_MAIN_2025_NOTICE':'https://www.sse.com.cn/services/listingwithsse/home/policy/supervise/c/c_20250515_10779023.shtml',
'SSE_STAR_2025_NOTICE':'https://star.sse.com.cn/services/listingwithsse/home/policy/supervise/c/c_20250516_10779158.shtml',
}
SOURCES.update({'SZSE_CHINEXT_2020_DEC_NOTICE': 'https://www.szse.cn/disclosure/notice/t20201231_584051.html', 'SZSE_CHINEXT_2020_JUNE_NOTICE': 'https://docs.static.szse.cn/www/lawrules/rule/repeal/announcement/W020210521598995459850.pdf', 'SSE_STAR_2019_APR_NOTICE': 'https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20210601_5478483.shtml', 'SSE_MAIN_2023_AUG_NOTICE': 'https://www.sse.com.cn/lawandrules/sselawsrules/stocks/mainipo/c/c_20230804_5724639.shtml', 'SSE_STAR_2023_AUG_NOTICE': 'https://www.sse.com.cn/lawandrules/sselawsrules/stocks/staripo/c/c_20230804_5724640.shtml', 'SSE_MAIN_2008_EXPLANATION': 'https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988237.shtml', 'SSE_MAIN_2018_NOV_NOTICE': 'https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20181116_10784930.shtml', 'SZSE_MAIN_2025_NOTICE': 'https://www.szse.cn/www/lawrules/rule/allrules/bussiness/t20250425_613256.html', 'SZSE_MAIN_2025_RULE': 'https://docs.static.szse.cn/www/lawrules/rule/stock/supervision/mb/W020250425730780478873.pdf'})

class HTML(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.links=[];self.href=None;self.label=[];self.skip=0
 def handle_starttag(self,tag,attrs):
  if tag in ['script','style']:self.skip+=1
  if tag=='a':self.href=dict(attrs).get('href');self.label=[]
 def handle_endtag(self,tag):
  if tag in ['script','style']:self.skip=max(0,self.skip-1)
  if tag=='a' and self.href:self.links.append((self.href,''.join(self.label)));self.href=None
 def handle_data(self,s):
  if not self.skip and s.strip():self.parts.append(s.strip())
  if self.href:self.label.append(s)
def fetch(item,root):
 key,url=item;target=url
 if 'www.szse.cn/' in url:target=url.replace('https://www.szse.cn/','https://docs.static.szse.cn/www/').replace('/www/www/','/www/')
 row=dict(source_id=key,official_source=url,download_url=target)
 try:
  req=urllib.request.Request(target,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(req,timeout=20) as f:b=f.read();row['resolved_url']=f.url
  ext='.pdf' if b.startswith(b'%PDF') else '.docx' if b.startswith(b'PK') else '.doc' if b.startswith(bytes.fromhex('d0cf11e0')) else '.html';path=root/(key+ext)
  with path.open('xb') as f:f.write(b)
  row.update(source_hash=hashlib.sha256(b).hexdigest(),file=path.name,bytes=len(b),retrieval_status='RETRIEVED')
  if ext=='.pdf':subprocess.run(['pdftotext','-layout',str(path),str(root/(key+'.txt'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);text=(root/(key+'.txt')).read_text();links=[]
  elif ext=='.doc':
   text=subprocess.check_output([os.environ['ANTIWORD_BIN'],'-m','UTF-8.txt',str(path)],env=os.environ,text=True);links=[];(root/(key+'.txt')).write_text(text)
  elif ext=='.docx':
   z=zipfile.ZipFile(io.BytesIO(b));text='\n'.join(''.join(n.itertext()) for n in ET.fromstring(z.read('word/document.xml')).iter() if n.tag.endswith('}p'));links=[];(root/(key+'.txt')).write_text(text)
  else:
   parser=HTML();parser.feed(b.decode('utf-8',errors='replace'));text='\n'.join(parser.parts);links=[dict(url=urljoin(target,h),title=t) for h,t in parser.links if re.search(r'\.(pdf|docx?)(?:$|\?)',h,re.I)];(root/(key+'.txt')).write_text(text)
  row.update(text_sha256=hashlib.sha256(text.encode()).hexdigest(),characters=len(text),attachments=links)
 except Exception as e:row.update(retrieval_status='FAILED',error=str(e))
 return row

def main(run):
 root=Path(run)/'sources';out=list(ThreadPoolExecutor(max_workers=4).map(lambda it:fetch(it,root),SOURCES.items()))
 children=[]
 for r in out:
  matches=[l for l in r.get('attachments',[]) if ('上市规则' in l['title'] and not any(t in l['title'] for t in ['说明','对照','修订条文','清单']))]
  for i,l in enumerate(matches[:2]):children.append((r['source_id']+f'_ATTACH{i+1}',l['url']))
 out+=list(ThreadPoolExecutor(max_workers=4).map(lambda it:fetch(it,root),children))
 (root/'rule_sources_acquired.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print([(x['source_id'],x['retrieval_status'],x.get('characters',0)) for x in out])
if __name__=='__main__':
 import sys;main(sys.argv[1])
