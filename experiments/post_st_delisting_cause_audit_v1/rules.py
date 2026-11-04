"""Archive historical exchange enactment notices separately from issuer evidence."""
from subprocess import check_output
from .common import *
from .collect_decisions import fetch
from concurrent.futures import ThreadPoolExecutor
POLICY_SHA='847619a177cd3ee51215c846009919d540d46cef'
EXTRA={
'SZSE_MAIN_2023_FEB_NOTICE':'https://www.szse.cn/lawrules/rule/repeal/rules/P020231230545116014930.pdf',
'SZSE_MAIN_2025_NOTICE':'https://www.szse.cn/lawrules/rule/repeal/rules/t20250425_613256.html',
'SZSE_CHINEXT_2025_NOTICE':'https://www.szse.cn/lawrules/rule/repeal/rules/t20250425_613257.html',
'SSE_MAIN_2026_NOTICE':'https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/mainipo/c/c_20260424_10816589.shtml',
'SSE_MAIN_2023_AUG_NOTICE':'https://www.sse.com.cn/lawandrules/ma/c/10648766/files/f588a49a5dcc407aad143a90545d3849.pdf'}
DATES={'2023_FEB':'2023-02-17','2023_AUG':'2023-09-04','2024':'2024-04-30','2025':'2025-04-25','2026':'2026-04-24'}
def context(y,board):
    if y.startswith('2023'):return '原版实体标准；2024换版后依新规衔接条款，必要时保留旧标准，决定程序可能另引新版。'
    if y=='2024':return ('主板5亿元市值条款自2024-10-30适用；连续低于旧3亿元的跨界期间依衔接规定。' if board=='MAIN' else '创业板独立上市规则；不套用主板市值或收入门槛。')+'新增财务标准以2024年报为首个会计年度，既有风险警示另依衔接规定；重大违法按收到处罚事先告知或司法裁判时间衔接。'
    return '风险警示和终止上市的衔接继续参照2024年发布通知；以决定明引的实体条款为准。'
def main():
    prior=json.loads(check_output(['git','show',POLICY_SHA+':experiments/st_policy_alignment_v1/rule_sources.json'],cwd=ROOT));old={x['source_id']:x for x in prior}
    keys=set(EXTRA)|{p for p in old if re.fullmatch(r'(SZSE_(MAIN|CHINEXT)|SSE_MAIN)_(2023_AUG|2024|2025)_NOTICE',p)}
    folder=run()/'rule_copies';folder.mkdir(exist_ok=True)
    def one(k):
        src=old.get(k,{});url=EXTRA.get(k) or src['official_source'];p=folder/(k+('.pdf' if url.endswith('pdf') else '.html'));retired=Path(os.environ['CAUSE_POLICY_RUN'])/'sources'/str(src.get('file',''))
        if not p.exists():
            if k not in EXTRA and retired.is_file() and sha(retired)==src['source_hash']:p.write_bytes(retired.read_bytes())
            else:p.write_bytes(fetch(url))
        ex,bd,part=k.split('_',2);v=part.replace('_NOTICE','')
        if p.suffix=='.pdf':assert p.read_bytes().startswith(b'%PDF')
        else:
            from bs4 import BeautifulSoup
            t=canon(BeautifulSoup(p.read_bytes().decode('utf8',errors='replace'),'html.parser').get_text());assert '上市规则' in t and ('发布' in t or '施行' in t)
        return dict(exchange=ex,board='ChiNext' if bd=='CHINEXT' else bd,version=v,rule_version=v,effective_date=DATES[v],transitional_provisions=context(v,bd),official_source_url=url,notice_sha256=sha(p))
    with ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,sorted(keys)))
    dump(run()/'rules.json',rows)
if __name__=='__main__':main()
