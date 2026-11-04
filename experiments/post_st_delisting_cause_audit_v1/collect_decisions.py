"""Archival retrieval of discovered exchange decisions. No model inputs in this module."""
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse,urljoin
import time
import requests
from bs4 import BeautifulSoup
from .common import *
HEAD={'User-Agent':'Mozilla/5.0','Accept':'*/*'}
def fetch(url):
    errors=[]
    for i in range(3):
        try:
            z=requests.get(url,headers=HEAD|{'Referer':url.split('/')[0]+'//'+urlparse(url).netloc+'/'},timeout=(5,10));z.raise_for_status()
            return z.content
        except requests.RequestException as e:errors.append(type(e).__name__+':'+str(e)[:140]);time.sleep(i*.2)
    raise RuntimeError(str(errors))
def stock_indexes():
    root=run();idx=root/'indexes';idx.mkdir(exist_ok=True)
    def one(i):
        url='https://www.szse.cn/disclosure/notice/company/'+('index.html' if i==0 else f'index_{i}.html');p=idx/f'sz_{i}.html'
        if not p.exists():p.write_bytes(fetch(url))
        s=p.read_bytes().decode('utf-8',errors='replace')
        return [dict(title=t,url=urljoin(url,u)) for u,t in re.findall(r"var curHref\s*=\s*'([^']+)';.{0,200}?curTitle\s*=\s*'([^']+)'",s,re.S) if '终止上市' in t]
    def safe(i):
        try:return one(i)
        except Exception as e:dump(idx/f'fail_{i}.json',str(e));return []
    with ThreadPoolExecutor(4) as ex:return sum(ex.map(safe,range(16)),[])
def usable(url):
    u=urlparse(url)
    return (u.hostname or '').endswith(('.sse.com.cn','.szse.cn')) and u.path.endswith(('.html','.shtml','.htm')) and not re.search('/index(_\\d+)?\\.html',u.path)
def one(a,idx):
    root=run();code=a['code'];name=a['name'];folder=root/'official'/code;folder.mkdir(parents=True,exist_ok=True)
    records=[];attempts=[]
    aliases={name,name.replace('深圳市','深圳').replace('深圳',''),name.replace('控股',''),name.replace('股份','')}
    aliases={x for x in aliases if len(x)>=2}
    def match(t):
        s=canon(t);return any(canon(n) in s for n in aliases)
    opts=[x for x in idx if match(x['title'])]+a['urls']
    seen=set()
    for z in opts:
        url=z['url'].replace('https://big5.sse.com.cn/site/cht/www.sse.com.cn','https://www.sse.com.cn')
        if not usable(url) or url in seen:continue
        seen.add(url);title=z['title']
        if not match(title) or not ('终止上市' in title or '股票上市' in title and '终止' in title or '摘牌' in title):continue
        h=hashlib.sha256(url.encode()).hexdigest()[:12];p=folder/(h+'.html')
        try:
            if not p.exists():p.write_bytes(fetch(url))
            s=BeautifulSoup(p.read_bytes().decode('utf-8',errors='replace'),'html.parser')
            for el in s(['script','style','header','footer']):el.decompose()
            lines=[re.sub(r'\s+',' ',l).strip() for l in s.stripped_strings]
            lines=[l for l in lines if l]
            ix=[i for i,l in enumerate(lines) if match(l) and ('终止' in l or '摘牌' in l)]
            if not ix:continue
            start=next((i for i in ix if i>0),ix[0]);body='\n'.join(lines[start:start+55])
            (folder/(h+'.txt')).write_text(body)
            records.append(dict(title=title,url=url,sha256=sha(p),file=p.name,excerpt_file=h+'.txt'))
            if any(('本所决定' in l or '本所作出' in l) and ('终止' in l) for l in lines) and '终止' in title and ('公告' in title or '决定' in title) and '登记服务' not in title:break
        except Exception as e:attempts.append(dict(url=url,error=str(e)[:250]))
    dump(folder/'index.json',dict(code=code,name=name,records=records,failures=attempts))
if __name__=='__main__':
    idx=stock_indexes();dump(run()/'indexes/sz_termination_titles.json',idx)
    with ThreadPoolExecutor(6) as pool:list(pool.map(lambda a:one(a,idx),[load(p) for p in sorted((run()/'discovery').glob('*.json'))]))
