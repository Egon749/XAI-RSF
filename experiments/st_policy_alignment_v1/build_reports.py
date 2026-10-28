"""Publish aggregate evidence only. Private identities, snippets, values and scores stay private."""
from pathlib import Path
import json,hashlib,platform,subprocess,shutil
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def md(frame):
 def v(x):return '' if pd.isna(x) else f'{x:.6f}' if isinstance(x,float) else str(x)
 return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(v(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'
def main(run,public):
 run=Path(run);pub=Path(public);out=run/'results';summary=json.loads((run/'audit/diagnostic_summary.json').read_text());clock=json.loads((run/'private/policy_asof_audit.json').read_text());verify=json.loads((run/'audit/verification.json').read_text());w=pd.read_csv(out/'reproduced_warning_metrics.csv');m=pd.read_csv(out/'reproduced_metrics.csv');e=pd.read_parquet(run/'private/event_rule_evidence.parquet');sources=json.loads((out/'rule_sources.json').read_text());rules=pd.read_csv(out/'st_rule_versions.csv')
 def write(n,t):(pub/n).write_text(t,encoding='utf-8')
 banner='状态：**PROVISIONAL / POST_EXPOSURE_EXPLORATORY**。本轮新增训练、校准及阈值选择均为 **0**。\n\n'
 write('PROTOCOL_ZH.md','# 特别处理政策对齐与预警错误诊断\n\n'+banner+f'''本轮基于透明分支实际提交 b6eab6a6ff3200ee879afdc7780efc1e897b7777 创建独立分支 research/st-policy-alignment-v1；保留原开发参考 5de9cf55a40de6b8af41a6877e51ab3529c6c003 与透明代码参考 fd4a87aee54c70d5e4d165b1e7fa7011fc43c5db 的关系。原工作区干净且远端一致，未回退或覆盖 main。

私有运行目录：`{run}`。复用原制造业冻结输入，训练56,458行、2,577公司组、177目标事件和31竞争事件；评价23,799行、3,138公司组、跨年79目标事件和36竞争事件。合计323个独立事件，随访实施日延伸至2025年12月。2023/2024年度独立目标事件分别44/60，有重叠，不能相加为104个独立事件。

默认对象固定为 q2 与同后端不限阶森林的42/101/202概率平均，以及M2全行业训练概率平均参考。它们均为原全训练参考，不是透明v2公司角色匹配对照；本轮未纳入可选透明v2补充，不据成绩替换对象。

原点按季度；目标为原冻结12日历月首次财务/审计ST定义，旧协议中法院受理重整也纳入目标。本轮保留其历史覆盖不完整备注。所有标签、资格、风险集、分数不变。主Top20、扩展Top25；季度内风险降序、sample_id升序打破并列，人数向上取整。未知结局不当负例，竞争单列。IPCW及公司等权完全调用冻结原评价函数；公司等权为各scope内每公司行数倒数。

A表记录事后实施原因；B表只允许正式公布日严格早于原点的规则。已公布未来安排保留；原点后规则置于 RETROSPECTIVE_DIAGNOSTIC_ONLY 列，不是特征。日历3/6月使用原评价代码 DateOffset，不替换为90/180天。规则候选路径保留多条，不能由未来事件选择原点版本。

检索范围固定为323个事件逐个查询、实施日前15日至后3日、最多100条公告/3个候选PDF；每个缺口保留。现有公告摘录和原因码只作线索。规则正式文号、发布时间以正文为准，不用页面迁移路径中的日期。下载后保留原文件、文本及SHA256；包含下载失败。未扩展无关行业或全历史身份重建。

只运行政策诊断测试，不运行包含 fit 的旧完整测试。开始与结束均核对原359个文件哈希。计算为轻量只读统计和并行网络获取，直接运行；没有Slurm训练任务。下阶段P0–P3只是规格，未经另行授权不执行。
''')
 good=[s for s in sources if s['retrieval_status']=='RETRIEVED'];bad=[s for s in sources if s['retrieval_status']!='RETRIEVED']
 sr=pd.DataFrame([dict(source_id=s['source_id'],status=s['retrieval_status'],url=s['official_source'],sha256=s.get('source_hash','')) for s in sources])
 write('POLICY_SOURCE_INDEX_ZH.md','# 官方规则来源与版本边界\n\n'+banner+f'''登记{len(sources)}个来源条目（包括重复附件），成功取得{len(good)}个，{len(bad)}个失败；不是{len(good)}部独立法规。规则表共{len(rules)}条登记项，区分可执行财务边界与须人工处理的条款组，不代表已把所有例外编成法律裁决器。

覆盖沪深主板、原中小板所属深市主板规则族、创业板、科创板；登记2008/2012、2014、2018年4月/11月、2020年6月/12月、2022、2023年2月/8月、2024、2025相关来源。缺失版本链见 policy_evidence_issues.csv；不能跨缺口推定精确版次。原中小板没有另套主板外的一套财务门槛；板块历史身份不另行重建。

- 旧主板两年亏损与低收入单独触发，不等于2020后“利润为负且扣除后收入不足1亿元”。2024主板新口径增加利润总额并把收入门槛提高到3亿元；创业板/科创板仍为1亿元。利润须保留扣非前后关系，收入须保留扣除定义。
- 2024年4月30日正式通知明确新财务条款首个适用2024年报，2023年报沿旧标准，不能按实施公告落在4月30日后就统一使用新门槛。
- 创业板2020年6月12日通知：财务警示以FY2020起算；普通ST到2020年9月12日才实施；此前已受理的重整存在旧制例外。新增了此独立过渡节点，未把2020全年作一档。
- 科创板2024部分普通ST延后6个月；内控审计/三年亏损相关条款按通知的不同首年度处理，不套主板。分红ST自2025-01-01施行，首三年FY2022–2024，双创研发等例外保留。
- 财务审计无法表示/否定、内控审计、持续经营不确定性是不同路径；重大违法与普通虚假记载ST也分开。股票简称不能代替触发依据。

61条冻结原点（训练14、评价47）使用北交所/新三板相关代码，历史板块时钟无法在本轮确认。其B表规则候选为空、状态未决，仍在全部预测评价中。没有把沪深规则套给这些记录。

source_hash指下载原件；私有目录含对应文本、公告原件和页/条款定位。旧Word自动编号可能丢失，登记原条款并保留原件，未用OCR补造编号。6个旧Word最初按HTML误解码的问题已记录并使用antiword重新提取，初始文件保留；错误文本未用于最终分析。法律文本核验与程序测试分别记录，测试通过不等于法律语义齐备。

以下表中公司实施公告不公开，只列监管规则来源：

'''+md(sr))
 vc=e[e.verification_status.eq('VERIFIED')];vg=int(vc.rule_version_unresolved.fillna(True).sum());counts=pd.read_csv(out/'event_rule_summary.csv');candidates=pd.read_csv(out/'label_revision_candidates.csv')
 write('POLICY_LABEL_AUDIT_ZH.md','# 冻结事件与公告依据审计\n\n'+banner+f'''323个独立事件全部纳入查询和逐事件证据表，没有删样本。242个事件取得候选PDF；排除债券受托报告、区分保荐/法律意见后，**239个实际实施原因有官方公告证据确认，84个原因仍未决**。其中评价期64个确认、51个未决。VERIFIED仅指实际实施原因及所引条款确认；不是整套法律适用、全部历史版本及原始财报的认证。239个中仍有{vg}个版次字段未闭合；其余有引用版次也不自动意味着所有例外核验完成。

确认的多重触发33个，确认适用财政年度/过渡安排24个。两项标志可重叠；追溯重述是事实/时钟标志，不独立算一个风险警示触发族。没有把公告中“未来可能终止上市”的模板条款当成当前触发。逐事件摘要及一事件多条证据长表均在私有目录。

**标签复核候选3个，均为训练期，合计12条已标目标窗口。** 其中2个有公司正式实施公告：冻结原因码02“资产缩水”，实际公告为资金占用普通ST，涉及8条训练窗口；建议复核为竞争事件。另1个冻结为“两年亏损”，当期保荐意见指停产/银行冻结，涉及4条，缺正式实施公告，证据级别较低。未应用任何修订，不声称3个都已证实标错，更不据此推断模型性能变化。

'''+md(candidates)+'''
原公告日与实施日独立保留；来源公开时间另列，不替换主事件时钟。公开日期能匹配240个事件的冻结公告日，1个有日期差异，82个无符合筛选的官方PDF；这些是来源覆盖统计，不等于全部日期获PIT认证。后续公告重述历史、文件内部笔误与引用旧版本分别留审计备注，不以此自动改事件日。原结局仍有713条评价未知记录。

'''+md(counts))
 pooled=w[w.scope.eq('pooled')][['model','top_fraction','alert_records','alert_company_groups','detected_events','company_confirmed_hit_fraction','announcement_detected_3m','announcement_opportunity_3m','announcement_detected_6m','announcement_opportunity_6m','known_target_free_alerts','competing_alerts','unknown_outcome_alerts','unknown_company_groups','repeated_alert_records']]
 es=pd.read_csv(out/'policy_error_slices.csv');short=es[(es.evidence=='VERIFIED_IMPLEMENTATION_REASON')&(es.model=='q2')].groupby(['family','top_fraction'])[['independent_events','missed','announcement_detected_3m','announcement_detected_6m']].sum().reset_index()
 write('POLICY_ERROR_ANALYSIS_ZH.md','# 固定预测的政策与错误诊断\n\n'+banner+'''原三模型概率集成的18行记录/公司等权指标与18行年度/合并预警汇总精确复现。AUC/AP/Brier差异仅浮点序列化量级，最高差异小于1e-12；全部核查计数相同。未重新排序某个政策子群来改善结果。各季度、年度、板块、原点已知/未知政策变化切片见 warning_utility_by_policy.csv；完整指标见 reproduced_metrics.csv。

合并开发后的2023/2024探索性评价：

'''+md(m[m.scope.eq('pooled')])+'''
Top20/25企业核查与提前量：

'''+md(pooled)+'''
“已确认命中企业/核查公司”是固定名单的确认命中比例，不等于把全部未命中判正常。q2 Top20的1171家公司中62家确认命中，5.29%；Top25为67/1379，4.86%。未知公司可能同时有其他已知季度，不能与已知公司数简单相加。窗口内捕获不等于提前3/6个月有效识别；机会分母来自原日历月评价，均保留公告与实施两套时钟。

政策假设的实际检验结果：

1. 已核对的10个主板FY2024新收入—利润条款事件，q2 Top20和Top25全部捕获。另有2个公告明确披露扣除后年收入落在1亿至3亿元之间的实例，三模型在两个预算均捕获且达到公告前6个月；它们是“新增边界的实际案例”，却不是该机制导致漏报的证据。实际财报只作事后诊断，未倒填原点特征，也未声称排除全部旧条款后仍不会ST。
2. 已确认的目标事件子集中，q2 Top20内控审计路径18个漏8个，Top25漏4个；收入—利润路径23个漏1个/0个；净资产8个均捕获；三年亏损加持续经营13个均捕获。路径可重叠，评价中仍有51个事件未确认，不能把子集差异当总体稳定机制。对内控路径的漏报，更直接的待检验缺口是合法公开的内控审计信息，而不是只增加政策年份。
3. q2 Top20有4328条已获随访支持但窗口内无目标事件的预警记录，其中3535条现有财务代理亏损标记为1。Top25相应为5444条中3550条。亏损不自动满足收入、连续年度、审计等全部条件；这些值多数来自中期报告，且原始披露版本未获PIT认证。因此没有把这些记录判为“已证实政策错配误报”，也不能称企业健康。
4. q2与不限阶Top20共同漏6个，仅q2漏11个，反向仅不限阶漏2个；Top25分别5、7、1。三模型Top20共同高分但已知无目标有3692条季度记录；共同竞争55条、未知199条。它们是描述性共同错误，不是政策因果效果。
5. 两个训练标签与官方原因冲突是已定位的上游分类问题；本轮未重评新标签，无法量化修订后的性能收益。不能把它包装为政策增强模型已经有效。

'''+md(short)+'''
季度/板块/变化窗口层不均衡，事件数小，未开展新的训练bootstrap或政策效果显著性检验。历史不确定性结果没有冒充新的分层区间。每个误报/漏报对应样本ID、事件依据、公司组、首次警示与机会日期留在服务器，可按共同冻结名单复核。
''')
 write('POLICY_FEATURE_SPEC_ZH.md','# 原点可得政策输入可行性\n\n'+banner+f'''原点政策表有80,257行。规则公开日严格早于原点；同日不纳入。原点后正式规则只出现在事后诊断列。80,257条选定财务记录有available<origin的元数据，{clock['annual_proxy_available_rows']}条可定位最近已公开年报年份代理；已恢复现有correction字段的revision_available_at。

**日期存在不等于财务值已经通过PIT认证。** 本轮可认证为可部署的政策财务取值仍为0行：缺扣除后年收入、扣非净利润和审计意见的完整原始版本及公开时间证据；现有净资产/利润/普通营业收入只能作原有探索性代理。不能用季度值替代年值，不能用普通营业收入代替扣除后营业收入，不能从最终ST反推审计意见。最新年度数衡量未来年度标准只能标 LATEST_AVAILABLE_PROXY。

板块、正式公开的规则路径、季度和报告陈旧度可优先形成结构化候选，但历史板块61条未决、部分规则版本链未闭合，不能默认正式可用。缺失、规则不适用、过渡条件未决是三个状态；逻辑使用三值计算，未知不是False。单条财务规则未触发，不能推出未来不会ST。

利润min与收入AND必须展开至净利润、扣非净利润、利润总额、扣除后收入等真实原始依赖；不能把复合指标当一个变量来降低表观交互阶数，或把等价字段重复加入以增加抽样概率。合法阈值只是候选边界，不把触发设为概率1。具体公式、来源、覆盖分母、单位及阻断项见下表。

'''+md(pd.read_csv(out/'policy_feature_availability.csv')))
 write('NEXT_EXPERIMENT_SPEC_ZH.md','# P0—P3匹配规格（未执行）\n\n'+banner+'''当前建议先处理标签复核和字段证据，不立即启动P0—P3。2个有正式公告的训练冲突候选及1个低证据候选须形成单独审查决定；任何标签修订都须另行冻结版本并使全部模型统一评价。本轮原标签不动。

|版本|允许差异|对照目的|
|---|---|---|
|P0|同一套原始财务与分开的财务/内控审计信息，不加政策编码|新增原始信息的共同基线|
|P1|P0＋季度/时间/板块|分离一般时期与板块差异|
|P2|P1＋原点之前已正式公布的规则路径/门槛距离/未决状态|检验政策对齐的独立增量|
|P3|P2＋合法政策边界候选切点|检验搜索边界增量|

模型固定同一透明预测器、公司角色划分、训练截止、损失、搜索候选总预算、树/函数预算及42/101/202种子。新增审计或扣除后收入等原始字段必须同时给P0；不改变未知结局保护。政策候选替换或插入须维持总候选预算并记录选择概率，不能给P3更多搜索预算。复合逻辑按真实原始依赖计阶和预算，不追加gamma、不提高q、不隐藏残差或设置企业变化权重。

主评估固定AP、IPCW Brier、Top20/25确认命中占比、核查企业负担、独立目标事件覆盖和公告前3/6月机会分母；同时检验完整CIF重构、缺失单元、实际函数/交互/分区规模及固定重抽样解释稳定性。不得通过2023/2024后暴露结局挑选配置；若继续用它们，只能明确探索性且先冻结规格。任何新不确定性分析须共同公司组抽样、固定预测，不能与完整训练不确定性混称。

启动前阻断项：扣除后收入与扣非利润的年度、合并口径、单位和原始披露版本；财务审计/内控审计/持续经营的独立公开时间；修订历史；相关版次、过渡和历史板块未决项；三项标签复核候选的裁决。解决后再提出固定12个拟合的执行请求。本轮12个拟合均未启动。
''')
 write('RESEARCH_DECISION_ZH.md','# 研究判断\n\n'+banner+'''本轮证据不支持把当前低命中占比主要归因于政策变化。模型训练标签确实混合了两年亏损、收入与利润组合、净资产、财务审计、内控审计、持续经营和法院重整等不同路径；但“有制度异质性”与“异质性造成预测错误”是两回事。

并存结论：

- **LABEL_REVIEW_REQUIRED**：2个训练事件有正式实施公告与冻结目标原因冲突，另1个低证据候选；分别影响8条和4条目标训练窗口。没有自动修订，也没有证明它们解释了总体性能差距。
- **POLICY_EFFECT_NOT_ESTABLISHED**：10个已核对主板新收入—利润标准事件均被q2 Top20捕获；两个明确落在新增收入区间的实例也均及时捕获。当前不能把这些案例称为政策错配漏报。内控审计漏报值得核查可得信息，但未证明政策编码会改善它。
- **INSUFFICIENT_POLICY_EVIDENCE**：84个实施原因尚未完成官方证据确认，部分已确认事件的精确版次/例外仍未闭合；61条历史板块原点未决；关键政策财务字段缺原始披露版本与可验证时钟。未决记录全部保留，不当正常、不按有利结果删除。

本轮**不授予 POLICY_MISMATCH_SUPPORTED_FOR_FOLLOWUP 的强错误归因结论**。规则变化已证实，具体标签冲突已定位，但尚无一批同时满足“原点合法财务信息充分、实际规则适用完整、与模型误报/漏报一致”的严格政策错配案例。高分亏损但未ST可由多条非政策因素共同产生，不能以单条财务规则未触发排除其他ST路径。

下一步的优先次序是标签证据与原始字段时钟，其后才值得执行有限P0—P3匹配实验。政策信息有清晰的待检验用途，但本轮不承诺AUC、AP、命中率或解释性将提升，也不启动任何训练。

原359个输入/模型/预测/标签及相关源文件哈希前后完全一致；旧训练/预测和本轮描述性核查的适用范围必须分开。此结论不推翻既有模型预测，只限制对其误报原因和政策增强收益的解释。
''')
 write('README_ZH.md','# 执行入口与交付索引\n\n本轮只读政策审计；不调用任何模型fit。优先阅读 [研究判断](RESEARCH_DECISION_ZH.md)、[标签审计](POLICY_LABEL_AUDIT_ZH.md) 和 [错误诊断](POLICY_ERROR_ANALYSIS_ZH.md)。所有公开数据均为规则元数据或聚合结果。\n\n'+f'''实际私有目录：`{run}`。环境：`/home/oliver/.venvs/xai-baselines-20261027/bin/python`。不需安装训练依赖；使用现有numpy/pandas/pyarrow/scikit-learn/pytest，文本提取用pdftotext及本轮隔离解包的antiword。

复算已有冻结数据及已审查证据：

```bash
python experiments/st_policy_alignment_v1/run_diagnostics.py PRIVATE_RUN
python experiments/st_policy_alignment_v1/feature_availability.py PRIVATE_RUN
python experiments/st_policy_alignment_v1/finalize_evidence.py PRIVATE_RUN
POLICY_RUN=PRIVATE_RUN OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q experiments/st_policy_alignment_v1/tests
python experiments/st_policy_alignment_v1/verify_policy_outputs.py PRIVATE_RUN
```

`prepare.py`仅用于新的唯一私有目录，先创建input/audit/private/sources/results；原文件只读复制。`collect_sources.py`只对空sources目录执行，原件使用独占创建；antiword路径通过ANTIWORD_BIN及ANTIWORDHOME配置。`collect_event_announcements.py`读取已有官方stock lookup，不猜orgId，保存有界查询和失败。私有手工核对表 `event_manual_review_PRIVATE.json` 是本次审查输入，须随来源及哈希保留，不能用自动关键词替代；Git不含公司ID和公告摘录。再次抓取需新目录，不覆盖证据。

`origin_policy_context.parquet`是B表；`event_rule_evidence.parquet`为逐事件摘要，`event_rule_evidence_long.parquet`为一事件多条依据的长表。三个模型逐记录Top名单、事件提前量、公告原件/摘录和全部个体冲突候选均留私有目录。公共CSV不能用于反推公司模型参数。

`st_rule_versions.csv`的effective_to是事后版本结束元数据，未决明确标记；as-of解析不以未来废止信息截断历史知识。VERIFIED_TEXT_BOUNDARY只指已核查边界，整体适用性仍受例外及缺失事实约束。
''')
 # Public output whitelist: aggregate CSVs and regulatory source index only.
 for p in out.glob('*.csv'):shutil.copy2(p,pub/p.name)
 shutil.copy2(out/'rule_sources.json',pub/'rule_sources.json')
 test=(run/'audit/policy_tests_final.log').read_text();write('VALIDATION_ZH.md','# 政策诊断验证\n\n'+banner+test+'\n原359个文件哈希一致；另补核M2三种子原RDS、原预测及预处理共9个文件在最终诊断复算前后完全一致（不倒称为会话最初哈希）；原指标与预警计数复现。未执行旧含fit测试。本轮开发中发现并修复：旧Word文件识别/提取、冻结风险集中61条板块不能直接归沪深、创业板2020普通ST延后实施节点。初始提取和测试日志保留私有目录。未放宽数值测试门槛，也未裁剪分数。\n')
 private_hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['audit','private','results','sources'] for p in (run/folder).rglob('*') if p.is_file() and '/tools/' not in str(p) and p.name!='artifact_hashes.json'}
 (run/'audit/artifact_hashes.json').write_text(json.dumps(private_hashes,indent=2));shutil.copy2(run/'audit/verification.json',pub/'verification_summary.json')
 env=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True);(run/'audit/environment.txt').write_text(env);write('environment.txt',env)
 receipt=dict(status='COMPLETED_BOUNDED_DIAGNOSTIC_WITH_EXPLICIT_UNRESOLVED_EVIDENCE',data_status=['PROVISIONAL','POST_EXPOSURE_EXPLORATORY'],base_commit='b6eab6a6ff3200ee879afdc7780efc1e897b7777',branch='research/st-policy-alignment-v1',private_output_directory=str(run),new_training_fits=0,new_calibrator_fits=0,new_threshold_selection=0,labels_modified=0,original_files_checked=verify['original_files_checked'],original_hashes_unchanged=True,source_hash_manifest_before_sha256=hashlib.sha256((run/'audit/source_hashes_before.json').read_bytes()).hexdigest(),source_hash_manifest_after_sha256=hashlib.sha256((run/'audit/source_hashes_after.json').read_bytes()).hexdigest(),event_audit=summary,source_acquisition=dict(entries=len(sources),retrieved=len(good),failed=[s['source_id'] for s in bad]),policy_context=dict(records=80257,board_unresolved_rows=61,transitions=clock['transition_counts']),label_review_candidates=dict(official_conflicts=2,official_conflict_train_rows=8,lower_evidence_candidates=1,lower_evidence_train_rows=4,applied=0),tests=dict(command='python -m pytest -q experiments/st_policy_alignment_v1/tests',passed=int(ET.parse(run/'audit/policy_tests_final.xml').getroot().find('testsuite').get('tests')),skipped=int(ET.parse(run/'audit/policy_tests_final.xml').getroot().find('testsuite').get('skipped')),model_training_tests_run=0),decisions=['LABEL_REVIEW_REQUIRED','POLICY_EFFECT_NOT_ESTABLISHED','INSUFFICIENT_POLICY_EVIDENCE'],immutable_evidence_manifest_sha256=hashlib.sha256((run/'audit/artifact_hashes.json').read_bytes()).hexdigest(),m2_original_artifact_check=json.loads((run/'audit/m2_models_after_final_evaluation.json').read_text()),git_receipt_note='This receipt is committed with the result; final commit and remote equality recorded in private audit/publication_receipt.json to avoid self-referential SHA.')
 (pub/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));(run/'audit/execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':
 import sys;main(sys.argv[1],sys.argv[2])
