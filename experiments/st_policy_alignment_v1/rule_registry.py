"""Source-backed, bounded clause registry. Text verification != complete applicability."""
from pathlib import Path
import json,re,hashlib
import pandas as pd

# Publication dates come from document body, never URL upload dates.
VERSIONS=[
('SZSE_MAIN','2008','2008-09-04','2008-10-01','SZSE_MAIN_2008_NOTICE','SZSE_MAIN_2008_NOTICE_ATTACH1','13.2.1','13.3.1'),
('SZSE_MAIN','2012','2012-07-07','2012-07-07','SZSE_MAIN_2012_NOTICE','SZSE_MAIN_2012_RULE','13.2.1','13.3.1'),
('SSE_MAIN','2014','2014-10-17','2014-11-16','SSE_MAIN_2014_NOTICE','SSE_MAIN_2014_NOTICE_ATTACH1','13.2.1','13.3.1'),
('SSE_MAIN','2018APR','2018-04-20','2018-04-20','SSE_MAIN_2018_APR_NOTICE','SSE_MAIN_2018_APR_NOTICE_ATTACH1','13.2.1','13.3.1'),
('SSE_MAIN','2018NOV','2018-11-16','2018-11-16','SSE_MAIN_2018_NOV_NOTICE','SSE_MAIN_2018_NOV_NOTICE_ATTACH1','13.2.1','13.3.1'),
('SSE_MAIN','2020','2020-12-31','2020-12-31','SSE_MAIN_2020_NOTICE','SSE_MAIN_2020_NOTICE_ATTACH1','13.3.2','13.9.1'),
('SZSE_MAIN','2020','2020-12-31','2020-12-31','SZSE_MAIN_2020_NOTICE','SZSE_MAIN_2020_NOTICE_ATTACH1','14.3.1','13.3'),
('SZSE_CHINEXT','2020JUN','2020-06-12','2020-06-12','SZSE_CHINEXT_2020_JUNE_NOTICE','SZSE_CHINEXT_2020_JUNE_NOTICE','10.3.1','9.4'),
('SZSE_CHINEXT','2020DEC','2020-12-31','2020-12-31','SZSE_CHINEXT_2020_DEC_NOTICE','SZSE_CHINEXT_2020_DEC_NOTICE_ATTACH1','10.3.1','9.4'),
('SZSE_MAIN','2022','2022-01-07','2022-01-07','SZSE_MAIN_2022_NOTICE','SZSE_MAIN_2022_NOTICE_ATTACH1','9.3.1','9.8.1'),
('SSE_MAIN','2023FEB','2023-02-17','2023-02-17','SSE_MAIN_2023_FEB_NOTICE','SSE_MAIN_2023_FEB_NOTICE_ATTACH1','9.3.2','9.8.1'),
('SZSE_MAIN','2023AUG','2023-08-04','2023-09-04','SZSE_MAIN_2023_AUG_NOTICE','SZSE_MAIN_2023_AUG_RULE','9.3.1','9.8.1'),
('SZSE_CHINEXT','2023AUG','2023-08-04','2023-09-04','SZSE_CHINEXT_2023_AUG_NOTICE','SZSE_CHINEXT_2023_AUG_NOTICE_ATTACH1','10.3.1','9.4'),
*[(b,'2024','2024-04-30','2024-04-30',b+'_2024_NOTICE',b+'_2024_NOTICE_ATTACH1',c,s) for b,c,s in [('SSE_MAIN','9.3.2','9.8.1'),('SZSE_MAIN','9.3.1','9.8.1'),('SZSE_CHINEXT','10.3.1','9.4'),('SSE_STAR','12.4.2','12.9.1')]],
*[(b,'2025','2025-04-25','2025-04-25',b+'_2025_NOTICE',b+'_2025_NOTICE_ATTACH1',c,s) for b,c,s in [('SSE_MAIN','9.3.2','9.8.1'),('SSE_STAR','12.4.2','12.9.1'),('SZSE_MAIN','9.3.1','9.8.1')]],
]
ISSUES=[
('SSE_MAIN','2008–2014','2008/2012官方说明已取；2012/2013完整附件与日期链不足，不从深市移植'),
('SZSE_MAIN','2014–2020','2014及2018年4月全文已取；2014正式通知与2018年11月版本链未闭合'),
('SSE_MAIN','2019APR、2022、2023AUG','2023年2月正文回溯确认2022发布；2023年8月旧链接失效；不宣称连续版本链完整'),
('SZSE_MAIN','2023FEB','2023年8月通知确认2月版；未独立取得2月版全文'),
('SZSE_CHINEXT','2012–2020JUN、2023FEB、2025','2012/2020年6月通知已取，早期无ST制度不可套主板；相关条款版本链未闭合'),
('SSE_STAR','2019–2023','2019年4月全文已取，但2020/2023完整版本链及研发型上市例外仍须核对'),
('UNRESOLVED_BOARD','61条冻结原点','北交所/新三板相关当前代码，含北交所成立前原点；历史板块不回填，保留风险集且政策适用未决'),
('ALL','全部','重大违法、治理、现金分红例外、既有ST恢复/撤销程序不做完全自动裁决；不适用与未决必须区别'),
]

def board_of(code):
 code=str(code).zfill(6)
 if code.startswith('688'):return 'SSE_STAR'
 if code.startswith('6'):return 'SSE_MAIN'
 if code.startswith(('300','301')):return 'SZSE_CHINEXT'
 if code.startswith('00'):return 'SZSE_MAIN'
 return 'UNRESOLVED_BOARD'

def build(run,out):
 run=Path(run);out=Path(out);sources=[]
 for p in sorted((run/'sources').glob('rule_sources_*.json')):sources+=json.loads(p.read_text())
 sources={r['source_id']:r for r in sources};rows=[]
 for board,version,pub,eff,notice,doc,fincl,stcl in VERSIONS:
  src=sources.get(doc,{});np=sources.get(notice,{})
  if src.get('retrieval_status')!='RETRIEVED' or np.get('retrieval_status')!='RETRIEVED':continue
  text=(run/'sources'/(doc+'.txt')).read_text();compact=re.sub(r'\s+','',text);yr=int(version[:4]);threshold=3e8 if yr>=2024 and board.endswith('MAIN') else 1e8
  # Exact predicates intentionally limited to reviewed financial boundaries.
  definitions=[]
  if yr>=2020:
   profits=['net_profit','net_profit_ex_nonrecurring']+(['profit_total'] if yr>=2024 else [])
   expr={'and':[{'or':[{'field':p,'op':'<','value':0} for p in profits]},{'field':'deducted_revenue','op':'<','value':threshold}]}
   definitions.append(('REVENUE_PROFIT',fincl+'(1)','*ST',expr,profits+['deducted_revenue'],1,'年度经审计；净利润扣非前后孰低；2024新增利润总额；扣除非主营/无商业实质收入',2024 if yr>=2024 else 2020))
  elif yr>=2012:
   definitions += [('TWO_YEAR_LOSS',fincl+'(1)','*ST',{'and':[{'field':'annual_net_profit_t','op':'<','value':0},{'field':'annual_net_profit_t_minus1','op':'<','value':0}]},['annual_net_profit_t','annual_net_profit_t_minus1'],2,'连续两个经审计会计年度；含追溯重述',None),('LOW_REVENUE_STANDALONE',fincl+'(3)','*ST',{'field':'annual_revenue','op':'<','value':1e7},['annual_revenue'],1,'年营业收入单独低于1000万元；不是利润且收入组合',2012)]
  else:
   definitions += [('TWO_YEAR_LOSS',fincl+'(1,2)','*ST',{'and':[{'field':'annual_net_profit_t','op':'<','value':0},{'field':'annual_net_profit_t_minus1','op':'<','value':0}]},['annual_net_profit_t','annual_net_profit_t_minus1'],2,'年度经审计净利润连续亏损，追溯调整另列',None)]
  if yr>=2012:
   definitions += [('NEGATIVE_EQUITY',fincl+'(2)','*ST',{'field':'annual_net_assets','op':'<','value':0},['annual_net_assets'],1,'经审计期末净资产；含追溯重述',2012 if yr<2020 else None),('FINANCIAL_AUDIT',fincl+('(4)' if yr<2020 else '(3)'),'*ST',{'field':'financial_audit_opinion','op':'in','value':['disclaimer','adverse']},['financial_audit_opinion'],1,'财务会计报告意见；不等于内控审计',2012 if yr<2020 else None)]
  # Narrative paths retained without executable truth value (exceptions need facts).
  definitions += [('OTHER_ST_PATHS',stcl,'ST',None,['internal_control_audit','occupation','guarantees','bank_freeze','operations','governance','going_concern','three_year_profit'],None,'各项为或；具体条文与例外留原文核对；不跨版本推定内控/三年亏损路径',None)]
  if yr>=2024:definitions += [('FRAUD_NOTICE',stcl+({'SSE_MAIN':'(7)','SZSE_MAIN':'(8)','SZSE_CHINEXT':'(7)','SSE_STAR':'(7)'}[board]),'ST',None,['pre_penalty_notice','penalty_decision','major_illegal_exception'],None,'年度财务指标虚假记载；排除触及重大违法退市情形；通知前后处理须复核',None),('DIVIDEND',stcl+('(9)' if board=='SZSE_MAIN' else '(8)'),'ST',None,['net_profit','parent_undistributed_profit','consolidated_undistributed_profit','cash_dividends_3years','rd_3years','listing_years'],3,'正利润及未分配利润前提；分红比例<30%且金额不足；主板5000万元/双创3000万元；双创研发等例外须逐项核实',2024)]
  for fam,cl,warn,expr,fields,periods,definition,fy in definitions:
   # For 2008 and OLE auto-number loss, use paragraph anchor, not invented PDF pages.
   status='VERIFIED_TEXT_BOUNDARY' if expr is not None else 'VERIFIED_SOURCE_MANUAL_APPLICABILITY'
   if fam=='REVENUE_PROFIT' and ('营业收入低于' not in compact and '营业收入低于' not in compact.replace('扣除后的','')):status='UNRESOLVED'
   rid=board+'_'+version+'_'+fam
   trans='逐事件核对追溯、既有ST及通知例外；未决不得默认适用'
   if yr>=2024:trans+='；新收入利润标准首个适用2024年报，2023年报沿旧标准；分红2025-01-01；科创板普通ST分项延后'
   effective=eff
   if fam=='DIVIDEND' and version=='2024':effective='2025-01-01'
   if fam=='OTHER_ST_PATHS' and version=='2020JUN':effective='2020-09-12';trans='通知第8项普通ST自2020-09-12；第6项财务警示首FY2020；第7项旧已受理重整不实施*ST'
   rows.append(dict(rule_id=rid,rule_status='FORMAL',exchange=board.split('_')[0],board=board,rule_version=version,clause_id=cl,warning_type=warn,trigger_family=fam,official_published_at=pub,effective_from=effective,effective_to='',first_applicable_fiscal_year=fy,transition_conditions=trans,exception_conditions=('研发型2.1.2(5)上市第4个完整会计年起适用财务(1)(2)(4)；普通ST三年起算2024' if board=='SSE_STAR' else '普通风险集/既有ST及更正事实必须分开；见通知'),required_accounting_periods=periods,required_fields=json.dumps(fields),metric_definition=definition,currency_and_unit='CNY yuan; audited annual; exact consolidation basis must be sourced',logical_expression=json.dumps(expr),strict_or_inclusive_boundary='< excludes equality' if expr else 'manual; preserve original AND/OR and exceptions',official_source=src['official_source'],source_hash=src['source_hash'],source_id=doc,notice_source_id=notice,source_page_or_paragraph=cl+'；正文检索对应触发句；OLE自动编号可能丢失，参见原文件',verification_status=status))
 frame=pd.DataFrame(rows)
 ends={('SZSE_MAIN','2008'):'2012-07-07',('SSE_MAIN','2014'):'2018-04-20',('SSE_MAIN','2018APR'):'2018-11-16',('SSE_MAIN','2018NOV'):'2020-12-31',('SSE_MAIN','2020'):'2022-01-07',('SZSE_MAIN','2020'):'2022-01-07',('SSE_MAIN','2024'):'2025-04-25',('SSE_STAR','2024'):'2025-04-25'}
 frame['effective_to']=[ends.get((b,v),'UNRESOLVED') for b,v in zip(frame.board,frame.rule_version)]
 frame['effective_to_use']='RETROSPECTIVE_VERSION_END_ONLY; never used to truncate as-of knowledge; transitions may retain older fiscal paths'
 frame.to_csv(out/'st_rule_versions.csv',index=False)
 # Keep acquisition failures and provenance; do not publish private paths.
 (out/'rule_sources.json').write_text(json.dumps(list(sources.values()),ensure_ascii=False,indent=2))
 pd.DataFrame(ISSUES,columns=['board','period','unresolved_issue']).to_csv(out/'policy_evidence_issues.csv',index=False)
 return frame
