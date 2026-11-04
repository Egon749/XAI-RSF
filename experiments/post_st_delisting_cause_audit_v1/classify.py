"""Blind reconciliation of an evidence-reader's register with operative exchange provisions."""
from .common import *
DATE=r'(20\d\d年\d{1,2}月\d{1,2}日)'
def iso(s):return pd.Timestamp(s.replace('年','-').replace('月','-').replace('日','')).date().isoformat()
def text_from(folder,d):
    return canon((folder/(d['announcementId']+'.txt')).read_text())
def ref(d):return 'https://static.cninfo.com.cn/'+d['adjunctUrl']
def operative(s):
    hits=re.search(r'第((?:9|10)\.([2345])\.\d+)条((?:第?[（(]?[一二三四五六七八九十]+[)）]?[款项]?[、和]?)*(?:、?第(?:9|10)\.[2345]\.\d+条(?:第一款)?第[一二三四五六七八九十]+项(?:和第[一二三四五六七八九十]+项)?)?)',s)
    if not hits:return None
    return dict(article=hits[0],category={'2':'C2','3':'C1','4':'C3','5':'C4'}[hits[2]],offset=hits.start())
def extract_ver(s):
    v=re.findall(r'上市规则\((20\d\d)年(?:(\d+)月)?修订\)',s)
    return [(y+('_AUG' if mo=='8' else '_FEB' if y=='2023' else '')) for y,mo in v]
def risk_match(cat,g,s):
    if cat=='C2':
        return (('price' in g and bool(re.search('收盘价.{0,20}(?:低于|不足).{0,5}(?:1元|一元|面值)',s))) or ('cap' in g and bool(re.search('(?:收盘|总)市值.{0,20}低于',s))))
    if cat=='C4':return '重大违法' in s and ('行政处罚' in s or '虚假记载' in s)
    return bool(re.search(r'财务类|财务指标|9\.3\.|10\.3\.|如公司.{0,18}财务会计报告继续被出具',s))
def main():
    ev=pd.read_csv(run()/'events_PRIVATE.csv',dtype={'code':str,'company_group':str});reader=pd.read_csv(run()/'blind_review_PRIVATE.csv',dtype={'code':str})
    assert set(ev.code)==set(reader.code) and len(reader)==len(ev)==38
    ev=ev.merge(reader,on='code',validate='1:1');rules=load(run()/'rules.json');rows=[];prefixes=[];files=[]
    for e in ev.itertuples():
        f=run()/'filings'/e.code;data=load(f/'index.json');docs=[v for v in data['selected'] if v.get('pdf_sha256')]
        texts={v['announcementId']:text_from(f,v) for v in docs}
        official=run()/'official'/e.code;direct=load(official/'index.json')['records']
        endings=[v for v in docs if re.search('摘牌|决定|股票终止上市的公告',v['title']) and not re.search('可能|风险提示',v['title'])]
        receipt_dates=[];cross=[]
        for v in endings:
            s=texts[v['announcementId']]
            for pattern in ['终止上市决定日期[：:为是]*'+DATE,'于?'+DATE+'(?:，?收到|收到了).{0,55}(?:决定书|股票终止上市的决定)']:
                receipt_dates.extend(iso(z) for z in re.findall(pattern,s))
            pos=re.search('终止上市决定的主要内容|终止上市决定书的主要内容',s)
            quote=s[pos.end():] if pos else s
            op=operative(quote)
            if op and op['category']==e.category:cross.append((v,quote))
        if direct:
            v=direct[0];src=v['url'];digest=v['sha256'];s=canon((official/v['excerpt_file']).read_text());assert sha(official/v['file'])==digest and '本所决定' in s
            dec=re.search(r'20\d\d-\d\d-\d\d',s)[0]
        else:
            assert cross
            v,s=cross[0];src=ref(v);digest=v['pdf_sha256'];dec=sorted(set(receipt_dates))[0] if receipt_dates else ''
        op=operative(s);assert op and op['category']==e.category,(e.code,op)
        ver=extract_ver(s[:op['offset']+10]);assert ver,(e.code,s[:180]);ver=ver[-1]
        rule=next(v for v in rules if (v['exchange'],v['board'],v['version'])==(e.exchange,e.board,ver));assert dec and rule['effective_date']<=dec
        prefix=s[:op['offset']]
        if e.category=='C2':
            span=re.search(DATE+r'至(?:同年)?(?:(20\d\d)年)?(\d{1,2})月(\d{1,2})日',prefix)
            assert span,(e.code,prefix[:400]);trigger=f"{span[2] or span[1][:4]}-{int(span[3]):02}-{int(span[4]):02}"
            assert risk_match('C2',e.grounds,prefix)
        else:
            hits=re.findall(DATE,prefix);assert hits;trigger=iso(hits[-1])
        assert trigger<=dec
        drops=[];warnings=[]
        for v in docs:
            s2=texts[v['announcementId']]
            if '摘牌' in v['title']:
                pats=[r'摘牌(?:日期|日|时间)(?:为|[:：]|是){0,2}'+DATE,DATE+r'(?:被|在|将|起|由|于|公司股票|予以|深圳证券交易所|上海证券交易所|本所|终止上市暨|对公司股票){0,6}摘牌(?!日|日期|时间)']
                for pat in pats:
                    drops.extend((iso(k),ref(v),v['pdf_sha256']) for k in re.findall(pat,s2) if iso(k)>=dec)
            if '风险' in v['title'] and v['pub']<=trigger and risk_match(e.category,e.grounds,s2[:2300]):
                warnings.append((v['pub'],ref(v),v['pdf_sha256'],bool(re.search('第一次|首次',v['title']) or re.search(r'(?:本公告|本次公告|本次为|本风险提示公告)[^。；]{0,30}(?:第一次|首次)(?:披露|终止上市|风险提示)',s2))))
        warnings.sort(key=lambda w:(w[0],not w[3],w[1]));ear=warnings[0] if warnings else ('','','',False)
        warning_series_source=''
        if ear[0] and not ear[3]:
            dt=pd.Timestamp(ear[0]);printed=f'{dt.year}年{dt.month}月{dt.day}日'
            for extra in docs:
                body=texts[extra['announcementId']]
                if extra['pub']>ear[0] and '第二次' in extra['title'] and risk_match(e.category,e.grounds,body[:2300]) and re.search(printed+r'[^。]{0,45}披露[^。]{0,90}《[^》]{0,55}风险提示[^》]*》',body):
                    ear=(*ear[:3],True);warning_series_source=ref(extra);break
        ds=sorted({x[0] for x in drops});dd=ds[0] if len(ds)==1 else ''
        dig=next((x for x in drops if x[0]==dd),('','',''))
        for v in docs:
            p=f/(v['announcementId']+'.pdf');assert sha(p)==v['pdf_sha256'];files.append(sha(p))
        conf='VERIFIED' if direct else 'SUPPORTED';multi='+' in e.grounds
        rows.append(dict(internal_company_id=e.company_group,code=e.code,year=e.year,exchange=e.exchange,board=e.board,origin=e.origin,model_x_date=e.model_x_date,within12=e.year12,within_supported24=e.year24_supported,
          first_documented_trigger_evidence=trigger,trigger_clock={'C2':'last required trading day','C1':'operative annual disclosure or statutory deadline','C4':'formal sanction received'}.get(e.category),first_confirmed_specific_warning_date=ear[0] if ear[3] else '',earliest_located_specific_warning=ear[0],warning_evidence=ear[1],warning_hash=ear[2],warning_series_corroboration=warning_series_source,warning_status='first in documented publication series' if ear[3] else 'absolute first not established',
          decision_date=dec,decision_publication_date=dec if direct else '',issuer_decision_publication_if_secondary='' if direct else cross[0][0]['pub'],
          delisting_date=dd,delisting_date_evidence=dig[1],delisting_date_hash=dig[2],provider_execution=e.raw_X_execution,provider_X_type=e.raw_X_type,all_documented_grounds=e.grounds,multiple_grounds=multi,category=e.category,article=op['article'],version=ver,effective_date=rule['effective_date'],transition=rule['transitional_provisions'],rule_url=rule['official_source_url'],rule_hash=rule['notice_sha256'],evidence_url=src,evidence_hash=digest,operative_excerpt=s[max(0,op['offset']-62):op['offset']+55],confidence=conf,review_status='blinded reading agrees with legal chapter',second_filing_count=len(cross),corroborating_filing=ref(cross[0][0]) if cross else '',corroborating_hash=cross[0][0]['pdf_sha256'] if cross else '',decision_date_discrepancy=bool(receipt_dates and set(receipt_dates)!={dec}),filing_date_alternatives='|'.join(sorted(set(receipt_dates))),date_status='one explicit dated delisting filing' if dd else 'effective date unconfirmed; see private filings',final_exit_confirmed=True))
        prefixes.append(dict(**rule,category=e.category,article=op['article']))
    z=pd.DataFrame(rows);z.to_csv(run()/'DELISTING_CAUSE_REGISTRY_PRIVATE.csv',index=False)
    report=[];counts=[]
    for scope,zz in [('all_observed',z),('H12',z[z.within12]),('supported_H24',z[z.within_supported24])]:
        for cat in ['C1','C2','C3','C4','C5','C6','C7']:
            gg=zz[zz.category==cat]
            report.append(dict(scope=scope,category=cat,n=len(gg),verified=sum(gg.confidence=='VERIFIED'),supported=sum(gg.confidence=='SUPPORTED'),unresolved=sum(gg.confidence=='UNRESOLVED'),multiple=sum(gg.multiple_grounds),issuer_text_matches=sum(gg.second_filing_count>0)))
            for yr in (YEARS[:2] if scope=='supported_H24' else YEARS):
                counts.append(dict(scope=scope,entry_year=yr,category=cat,n=int(sum(gg.year==yr))))
    audit=pd.DataFrame(report);audit.to_csv(D/'CAUSE_CLASSIFICATION_AUDIT.csv',index=False);pd.DataFrame(counts).to_csv(D/'CAUSE_COUNTS_BY_YEAR.csv',index=False)
    audit.assign(official_fraction=(audit.verified+audit.supported)/audit.n.replace(0,np.nan),direct_fraction=audit.verified/audit.n.replace(0,np.nan)).to_csv(D/'EVIDENCE_COVERAGE.csv',index=False)
    pd.DataFrame(prefixes).drop_duplicates().to_csv(D/'DELISTING_RULE_VERSION_REGISTRY.csv',index=False)
    gate=(z.confidence=='VERIFIED').mean()>=.8 and z.confidence.isin(['VERIFIED','SUPPORTED']).mean()>=.9 and all(gg.confidence.isin(['VERIFIED','SUPPORTED']).mean()>=.8 for gg in [z[z.within12],z[z.within_supported24]])
    dump(D/'SOURCE_AND_HASH_AUDIT.json',dict(classification_gate=bool(gate),blinded_reader_hash=sha(run()/'blind_review_PRIVATE.csv'),private_registry_hash=sha(run()/'DELISTING_CAUSE_REGISTRY_PRIVATE.csv'),classified=len(z),direct_exchange=int((z.confidence=='VERIFIED').sum()),issuer_quote=int((z.confidence=='SUPPORTED').sum()),matching_issuer_evidence=int((z.second_filing_count>0).sum()),separate_direct_plus_filing=int(((z.confidence=='VERIFIED')&(z.second_filing_count>0)).sum()),unique_filing_pdfs=len(set(files)),delisting_date_verified=int((z.delisting_date!='').sum()),strict_first_warnings=int((z.first_confirmed_specific_warning_date!='').sum()),earliest_located_warnings=int((z.earliest_located_specific_warning!='').sum()),model_date_differs_decision=int((z.model_x_date!=z.decision_date).sum()),model_date_differs_verified_delisting=int(((z.delisting_date!='')&(z.model_x_date!=z.delisting_date)).sum()),conflicting_filing_decision_dates=int(z.decision_date_discrepancy.sum()),files_checked=True))
    assert gate
if __name__=='__main__':main()
