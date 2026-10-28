"""Produce aggregate-only evidence package. No fitted parameters or identities published."""
from pathlib import Path
import sys,json,shutil,hashlib,datetime
import numpy as np,pandas as pd
from support import sha,write

def table(d,cols=None):
 if cols is not None:d=d[cols]
 def fmt(x):
  if isinstance(x,float):return f'{x:.6g}'
  return str(x)
 return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'

def main(run):
 dst=Path(__file__).parent;out=run/'reports';inp=run/'input';m=pd.read_csv(out/'metrics_by_year_seed.csv');w=pd.read_csv(out/'warning_utility.csv');c=pd.read_csv(out/'native_complexity.csv');ci=pd.read_csv(out/'paired_metric_differences.csv');trans=pd.read_csv(out/'alert_transition_summary.csv');train=pd.read_csv(out/'training_diagnostics.csv');gates=json.loads((run/'audit/publication_gates.json').read_text())
 # Add semantic hashes for every metadata column, in addition to container/NPZ hashes.
 manifest=json.loads((inp/'data_version_manifest.json').read_text());sem={}
 for version in ['L0_ORIGINAL','L1_ADJUDICATED']:
  dd=pd.read_parquet(inp/version/'final_train_metadata_PRIVATE.parquet');sem[version]={col:hashlib.sha256((str(dd[col].dtype)+'|').encode()+pd.util.hash_pandas_object(dd[col],index=False).to_numpy().tobytes()).hexdigest() for col in dd}
 manifest['metadata_column_semantic_hashes']=sem;manifest['L1_private_input']=str(inp/'L1_ADJUDICATED');manifest['handoff_manifest']=str(run/'handoff_L1/handoff_manifest.json');manifest['frozen_for_future_work']=True;write(inp/'data_version_manifest.json',manifest)
 primary=m[(m.seed==-1)&(m.weighting=='record_equal')];warnings=w[(w.seed==-1)&w.scope.isin(['2023','2024','pooled'])];primaryci=ci[(ci.seed==-1)&(ci.scope=='pooled')&ci.metric.isin(['ap','ipcw_brier','company_hit_fraction','event_coverage','lead3m_detected','lead6m_detected'])&ci.weighting_or_budget.isin(['record_equal','Top20'])]
 def doc(name,text):(out/name).write_text(text.strip()+'\n')
 caveat='数据保持 **PROVISIONAL**；2023/2024为已接触结局后的探索性比较。原始财务披露版本PIT、行业分类历史时钟和47项未决事件没有因本轮补丁获得认证。'
 doc('PROTOCOL_ZH.md',f'''# 已裁决标签的配对重训协议

本轮检验已确定的训练标签修正对同一冻结模型的影响，以及修正后q2与同后端不限阶数森林的差距。不是新算法比较、政策再审计或调参。{caveat}

裁决参考64d373c952e07d2e3dd4f65ed543f3eceee6c86a；实际原制造业训练代码206e414d05e8cd15e13429637bc6efc7f0abfcd4。报告提交5de9cf55a40de6b8af41a6877e51ab3529c6c003不作为训练版本。原代码从归档复制，core/functions/partition/engine逐字对齐提交；原学习核心零修改。

训练2012—2020原点，拟合截止2022-01-01；制造业冻结输入56,458行、2,577组、75列，评价23,799行。L0永久保留；L1只作3事件、12行1→2修正。缺失原生处理、公司bootstrap、logrankCR分裂（两个原因固定等权）、叶AJ估计、等权森林均沿用原实现。

两个版本分别保留原配置：32树、最大深度6、叶最少100行/20公司组/3个任意原因事件、mtry75、5候选阈值、2原因。q2的path_budget=2、每树25个变量、shared_quantile；不限阶数path_budget=null、每树全部变量、local_random。无全森林格数预算，不能称本轮是在相同搜索配置下单独识别q限制。完整配置见frozen_configs.json。

种子固定42/101/202。每树编号固定，公司抽样和标签无关随机流保持一致；原因改变允许叶CIF/分裂变化。六个L0先逐行精确重放，全部通过，故实际新增六个L1拟合，无控制补训、无新增试运行。48个4树分片按原树编号合并，不是48个独立森林。

评价主结果为三种子概率平均后的指标，记录等权；公司等权、各seed及其范围单列。合并AP在合并预测重新计算。Top20/25按完整季度风险集排序（同分按sample_id、ceil预算），未知保留负担且不当正常。提前量沿用公告日减3/6个日历月，机会分母独立报告。

1,000共同公司组bootstrap：固定模型、预测、名单和原IPCW估计，保留公司所有季度及抽样多重性；所有模型、标签版本、种子和年份共用SeedSequence(20261025,spawn_key=(rep,))。五个预设差值完整交付；百分位95%区间只描述固定预测的条件评价，不包含重训不确定性，不是非劣性检验。

运行入口：prepare.py → orchestrate.py prepare/control → train → assemble → explain；evaluate_results.py → bootstrap → collect.py → verify_and_handoff.py → diagnose.py → finalize.py。正式学习由不可变旧代码快照执行。每阶段JSONL及资源/发布收据保留私有运行目录；不得整批重交已成功任务。使用新的唯一目录重现；现有完成目录不可再次应用补丁。

本轮至此停止，不自动启动transparent_cif_v2六方案、审计特征、gamma、q3/q4或额外搜索。
''')
 doc('PATCH_APPLICATION_ZH.md',f'''# 补丁应用和数据一致性

3项裁决已完整应用到新的L1副本，3公司组、3独立首次事件、12训练窗口全部1→2；没有改为0、寻找后续事件、重新入组或全局替换原因02。

{table(pd.read_csv(out/'label_counts_before_after.csv'))}

NPZ只event数组改变。元数据允许改变event、status、previous_label、label_observed_first、label_eligible、three_view_eligible、followup_reason，且只能在这12行；其余列逐值不变。实际变化逐列统计见label_dependency_map.json，逐行旧新值仅保留私有账本。

字段按原来源代码的定义处理：竞争事件的label_observed_first为NA，label_eligible/three_view_eligible为false；它们表示旧目标/存活二元视图，不能照名称解释为所有已知结局。有效端点资格是label_eligible OR event==2，修正前后完全一致。三视图有11行发生改变，另一行本来为false。原origin/first-history未知标记保留。

全部12个monthly_labels原因单元由1变2；target/competing掩码重建。任意事件(event>0)、时间网格、删失分布以及全训练集原端点IPCW权重逐值一致。未知或提前删失资格不变。sample_id顺序、组、X、缺失位置、特征顺序、首次事件ID、公告/实施日期、stop_date、时间长度和365/366日窗口不变。评价输入和元数据容器逐字节一致；没有读取后期结局选择标签。

重复应用、错误来源哈希、缺失窗口和重复窗口会拒绝。container SHA256与数组/元数据逐列语义哈希同时交付，不能用重序列化哈希变化冒充内容变化。补丁来源SHA256：{manifest['patch_sha256']}。

{caveat} L1作为未来工作基线由裁决证据决定；指标下降不构成恢复已确认错误标签的理由。
''')
 doc('CONTROL_REPRODUCTION_ZH.md',f'''# L0控制兼容性与重放

六个原制造业L0模型（q2、不限阶数×42/101/202）全部复用。每个模型的来源、原训练SHA、配置、模型/预测SHA256及新输出位置见model_provenance.csv。模型来源逐项对应原训练receipt，不依赖同名或同seed猜测。

原模型在本轮冻结评价X上重载预测，所有23,799行、两原因、逐行365/366日端点与原存档完全一致，最大绝对误差0。样本身份/组/期限逐项对齐。Python3.13.13、NumPy2.5.4、SciPy1.18.1、pandas3.1.0与原环境版本相同；两节点版本记录见environment.json。

逐对验证32棵树的公司抽样、变量资格、配置、共享切点及253个原生事件时间一致。节点随机流由同一旧代码定义；不强迫原因改变后的实际路径相同。控制及新模型全部采用原company bootstrap与等权聚合。

L0 q2也在同一56,458行X的等权参考分布重新编译全部CIF函数，所有评价行/时间/原因重新核验；没有以旧图代替本轮透明性检查。

M2只读使用原全行业冻结预测，未经本轮制造业补丁重训，不是L1匹配对照。不得从其结果推断全行业标签已修订。
''')
 doc('TRAINING_DIAGNOSTICS_ZH.md',f'''# 实际执行与数值诊断

新增正式企业拟合 **6/6**，复用L0 **6/6**，新增L0补训0，试运行0，正式训练/发布失败0，模型恢复训练0。48个分片共192棵树完整合并为六个森林；无丢弃seed。

{table(pd.DataFrame(gates),['stage','tasks','seconds','hosts','failed_tasks'])}

两节点physical池：oliver可用8P+16E、lemon8物理核，单任务1CPU、BLAS1；不声称单进程跨机器。训练实分配oliver35分片、lemon13分片。六模型合计fit CPU {train.fit_cpu_seconds.sum():.3f}秒；训练阶段墙钟116.991秒，不含准备/排队/解释/评价。合并期最大RSS {train.peak_rss_MiB.max():.3f}MiB。训练分片沿用原6GiB地址空间上限；没有单独采集实际fit峰值RSS，Slurm accounting被禁用，不能把合并RSS冒充训练RSS。

{table(train,['variant','seed','fit_cpu_seconds','peak_rss_MiB','leaf_zero_target_company_support','leaf_zero_competing_company_support'])}

最少3事件是任意原因合计，并不保证每叶都有每个原因支持。上表零原因支持叶如实保留。无优化器/迭代损失，本模型是有限候选生存分裂与非参数叶CIF，因此不虚构数值gap或收敛状态；所有概率/保存重载/分解门槛保持原值。

测试：原训练版本完整45项（含R survival独立AJ、randomForestSRC单树分裂/叶AJ）；相关旧审计/日历/未知结局64项；本轮数据/真实抽样/交接12项，最终全部通过、无skip。初始45项又在明确冻结源码cwd下完整重复一次以消除导入路径歧义，记录未删除。初始补丁7项包含于最终12项。两次原测试合计Python fit调用86次：54次合成拟合成功、32次预期非法输入拒绝；另2次合成R单树拟合。均不计入六个正式企业拟合。

保留两项非训练中止：旧评价测试首次缺POLICY_RUN环境路径而在collection失败，补齐后64通过；发布核验脚本首次按错误层级读取publication_completed，修正为实际收据顶层后通过。无模型重训，原失败日志保留。初版交接plan包含旧协议来源字段，冻结前精简并刷新所有source_hashes，旧草稿保留，不用于任何训练。
''')
 doc('LABEL_SENSITIVITY_RESULTS_ZH.md',f'''# 标签修正后的预测敏感性

结论：确认的修正对本次预测影响较小，没有消除q2与不限阶数森林的差距。标签正确性与预测增益分开判断。{caveat}

主结果为三个seed概率平均、记录等权；每seed、公司等权、均值范围和全部校准分组见CSV。

{table(primary,['scope','model','auc','ap','ipcw_brier','prediction_mean','observed_ipcw_incidence'])}

合并q2 AP：0.092670→0.091838（差−0.000832，条件95%区间[−0.003015,0.000168]）；不限阶数0.144595→0.144792（+0.000197，区间[−0.000291,0.001002]）。跨年不是一致增益：q2 AP两年略降，AUC2023下降/2024微升；不限阶数AP2023微降/2024微升，AUC方向相反。

两者Brier均小幅下降（较好），但幅度仅约8e-6/1e-5。修正后q2相对不限阶数AP差−0.052954，条件区间[−0.087561,−0.011046]；q2 Brier差−0.000984，区间[−0.001624,−0.000425]。排序与平方概率误差的权衡仍存在。两者平均风险仍高于原IPCW观察发生率，不能将微小Brier改善写成已校准。

完整五类差值、年度及公司等权、各seed区间见paired_metric_differences.csv。差距变化的AP区间含0，不是证明无标签影响，更不是模型等效。区间是固定预测下的公司抽样条件区间，不包含重新训练，也没有按显著性改选结论。原配置除q外还含候选变量覆盖和切点策略差异，不能把全部差距归因于q限制。

{table(primaryci[primaryci.metric.isin(['ap','ipcw_brier'])],['contrast','metric','point_difference','conditional_ci_lower','conditional_ci_upper'])}
''')
 doc('WARNING_UTILITY_COMPARISON_ZH.md',f'''# 预警效用与配对名单流转

Top20为主要季度预算，Top25为扩展预算；全部23,799行参加各季度排序，未知结果保留核查负担。企业级命中占比=有已确认目标窗口命中的去重公司/被核查去重公司，不是记录级命中率；未知不能当误报。下表三seed概率集成，跨年目标事件去重79个；可提前3/6个月机会均为60个，年度分别25/35。

{table(warnings,['scope','model','top_fraction','alert_company_groups','confirmed_hit_company_groups','company_hit_fraction','detected_events','independent_events','announcement_detected_3m','announcement_detected_6m','unknown_outcome_alerts'])}

合并Top20：q2由62/79降至61/79、核查1171→1169组，公告前3个月45→44、6个月42→41；不限阶数71/79保持，核查1176→1177组，3个月51保持、6个月50→51。修正后命中企业占比q2约5.22%、不限阶数约6.03%，仍有很大核查负担，不能用AUC较高掩盖。

Top25：q2仍67/79、核查1379→1373组，6个月46→47；不限阶数仍73/79、核查1413→1416组，3个月53→54。窗口捕获不等于及时捕获，名单规模微变不等于实际事件增益。完整逐季度记录、年度/合并去重公司、重复预警、已知无目标（其中竞争单列）及未知见warning_utility.csv。

{table(trans[(trans.seed==-1)&(trans.scope=='pooled')],['model','top_fraction','endpoint','common_captured','new_captured','lost_captured','common_missed','companies_entering','companies_exiting'])}

共同捕获/新增/丢失/共同遗漏按同一事件集合比较，不在内控或任何政策子集重排名单。公司状态可能跨季度重叠，未知/竞争/已知无目标的公司数不能简单相加；记录类别则逐项对账等于全部预警记录。

{table(primaryci[primaryci.weighting_or_budget=='Top20'],['contrast','metric','point_difference','conditional_ci_lower','conditional_ci_upper'])}

1000共同公司bootstrap保持完整季度名单固定，估计对象是冻结名单在公司构成扰动下的效用；不重新排名。相同捕获事件集合可产生[0,0]的条件差值区间，不是总体部署无差异的证明。
''')
 doc('TRANSPARENCY_CHECK_ZH.md',f'''# 完整透明CIF的独立核验

L0/L1各3个q2及各自三seed集成，共8个完整函数预测器已导出。参考分布固定为全部原训练X的56,458行等权乘积边际，避免重中心化制造差异。独立函数执行器只有完整函数参数，不调用森林预测或遍历树。

对每个预测器检查全部23,799评价行、255个检查时点（253个原生事件时点加必要边界）、目标/竞争两原因；每个12,137,490个标量、总97,099,920个标量。原森林与完整函数最大误差{c.reconstruction_error.max():.3g}，完整逐函数显式求和锚点误差最大{c.component_sum_error.max():.3g}，保存重载误差0；原1e-10重构门槛未放宽，概率非负/时间单调/原因概率和≤1均通过，未裁剪输出。

{table(c,['version','seed','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction','reconstruction_error'])}

三个seed集成的完整函数并集为32主函数+67交互、2423共同空间格、1,226,038原生时间×原因标量格；不是拿单seed复杂度冒充集成。factored rectangle表示无损保存全部参数；稀疏存储并不等于逻辑函数表小。每个函数的真实切点、缺失格、逐格联合样本支持和因子参数全部私有保存，低支持/零支持单元未删除。

L0→L1的q2三seed共96棵树分裂结构均未改变，主要是原因标签变化后的叶CIF/函数值改变；不限阶数每seed有2/32棵树结构变化。见structure_changes.csv。两版本q2复杂度完全相同，不能声称标签修正使解释更简洁。固定512训练锚点的函数变化汇总见function_perturbation.csv，只描述一次已裁决标签扰动，不等于普遍重训稳定性。

这是完整预测函数透明与数值可核验，不代表所有函数足够简单、人易理解、统计可辨识或因果。CIF贡献单位是概率（乘100为百分点）；不得混写为对数风险率或因果效应。原始输入依赖/特征语义继承既有75列合同，PIT限制不因函数透明而消失。
''')
 doc('NEXT_STEP_HANDOFF_ZH.md',f'''# 冻结L1及后续数据交接

L1已形成，作为后续透明模型研究工作基线；依据是3项已裁决证据，不是择优模型成绩。L0永久保留。本轮不启动transparent_cif_v2六方案或审计信息增量实验。

- 原始L1输入：`{inp/'L1_ADJUDICATED'}`。
- 完整数据版本manifest：`{inp/'data_version_manifest.json'}`。
- 可直接作为run.py的`--source-input`：`{run/'handoff_L1'}`。
- 特征合同`--features`：`{run/'handoff_L1/features.json'}`。
- 交接SHA256及文件清单：`{run/'handoff_L1/handoff_manifest.json'}`。

映射：final_train_PRIVATE.npz/event及配套metadata使用L1；X/time/groups/sample_ids/horizon与L0一致。final_eval两文件逐字节保持原版。plan.json保留原forest_configs.q2，刷新L1计数及输入哈希，去掉原基线大实验/模拟任务清单，不能由旧plan误触发其它任务。features.json保留72内容+3质量字段及原缺失处理，单位未知继续UNVERIFIED，未追加审计特征或72个独立缺失列。

已实际调用transparent_cif_v2的training_contract和feature_contract，无fit；结果PASS，56,458行、75特征、12日历月365/366天、有效端点39,362行、历史截止2022-01-01。未来按原公司角色分配仍须先核验新原因支持；不能自动复用旧标签权重或拟合模型。

reference_L0_q2_{{seed}}_PRIVATE.npz为旧全训练q2，reference_L1_q2_{{seed}}_PRIVATE.npz为修订全训练q2；两者不能冒充未来80/20公司角色匹配的base_equal。M2_PRIVATE.npz始终是原全行业历史参考。旧server/prepare.py硬编码父输入白名单，应使用本次已验证的交接目录/manifest而非把旧父hash检查强行改成通过。

合并记录等权基准：q2 L0 AUC0.843991/AP0.092670/Brier0.009488；L1 AUC0.843814/AP0.091838/Brier0.009480。年度、单seed、配对区间和Top20/25见本目录CSV。不得只引用合并点估计隐藏不利年度/名单变化。

{caveat} 后续若获授权，必须在新的唯一运行目录执行；不得把本轮交接当成已完成下一阶段研究。
''')
 # Sanitize and publish only named aggregate artifacts.
 for p in out.glob('*.csv'):shutil.copy2(p,dst/p.name)
 for p in out.glob('*.md'):shutil.copy2(p,dst/p.name)
 for name in ['data_version_manifest.json','label_dependency_map.json','patch_application_receipt.json']:shutil.copy2(inp/name,dst/name)
 configs=json.loads((inp/'plan.json').read_text());configs['original_L0_source_counts']=configs.pop('source_counts');configs['L1_source_counts']=json.loads((run/'handoff_L1/plan.json').read_text())['source_counts'];write(dst/'frozen_configs.json',configs)
 environment={h:json.loads((run/f'environment/{h}.json').read_text()) for h in ['oliver','lemon']};write(dst/'environment.json',environment)
 logs=['frozen_tests.log','frozen_exact_tests.log','patch_tests.log','legacy_evaluation_tests.log','legacy_evaluation_tests_recovery.log','final_integration_tests.log','diagnose.log','diagnose_recovery.log']
 receipt=dict(status='COMPLETED_PROVISIONAL_POST_EXPOSURE_EXPLORATORY',private_run=str(run),branch='research/label-revision-sensitivity-v1',label_reference_sha='64d373c952e07d2e3dd4f65ed543f3eceee6c86a',actual_frozen_training_sha='206e414d05e8cd15e13429637bc6efc7f0abfcd4',L1_formal_fits=6,L0_reused=6,L0_refits=0,pilot_fits=0,training_shards=48,trained_trees=192,bootstrap_replicates=1000,bootstrap_tasks=10,formal_failed_tasks=0,formal_fit_recoveries=0,synthetic_test_python_successful_fits=54,synthetic_test_python_rejected_validation_calls=32,synthetic_test_R_fits=2,synthetic_test_suites_executed_twice=True,tests_unique_related_passed=121,tests_skipped=0,nontraining_failures=[dict(kind='test_collection',reason='POLICY_RUN missing',recovery='set original audit paths; 64 passed; no fit'),dict(kind='postprocessing',reason='publication_completed read at wrong nesting',recovery='use actual top-level receipt schema; gates passed; no fit')],test_logs_sha256={n:sha(run/'audit'/n) for n in logs},training_records_unique=56458,training_groups=2577,L0_target_events=177,L1_target_events=174,L0_competing_events=31,L1_competing_events=34,patched_events=3,patched_records=12,evaluation_records=23799,evaluation_events=79,fit_cpu_seconds=float(train.fit_cpu_seconds.sum()),training_peak_rss='NOT_CAPTURED; original 6GiB per-process address-space cap; Slurm accounting disabled',assembly_peak_rss_MiB=float(train.peak_rss_MiB.max()),complete_transparent_predictors=8,complete_function_max_error=float(c.reconstruction_error.max()),function_reload_max_error=0,source_integrity=json.loads((run/'audit/immutability_verification.json').read_text()),publication=gates,bootstrap_validation=json.loads((run/'audit/bootstrap_verification.json').read_text()),handoff_contract=json.loads((run/'audit/handoff_contract.json').read_text()),M2_retrained=False,learning_core_changed=False,additional_features=False,transparent_cif_next_stage_trained=False,L1_frozen=True,default_future_version='L1_ADJUDICATED',selection_basis='adjudicated label evidence, not favorable metrics',limitations=['PROVISIONAL raw financial version PIT','47 unresolved events unchanged','post-exposure exploratory 2023/2024','conditional bootstrap not refit uncertainty','two original configurations differ beyond q'],training_wrapper_snapshot_sha256=sha(run/'audit/worker_training_snapshot.py'),frozen_learning_files_sha256={str(p.relative_to(inp/'frozen_code')):sha(p) for p in [inp/'frozen_code/crforest/core.py',inp/'frozen_code/crforest/functions.py',inp/'frozen_code/crforest/partition.py',inp/'frozen_code/experiments/shared_partition_v1/engine.py']},data_manifest_sha256=sha(inp/'data_version_manifest.json'),handoff_manifest_sha256=sha(run/'handoff_L1/handoff_manifest.json'),aggregate_hashes={p.name:sha(p) for p in dst.glob('*.csv')})
 write(out/'execution_receipt.json',receipt);write(dst/'execution_receipt.json',receipt)
 # Freeze only our newly created data copies, never change source permissions.
 for folder in [inp/'L1_ADJUDICATED',run/'handoff_L1']:
  for p in folder.rglob('*'):
   if p.is_file():p.chmod(0o444)
 print('DELIVERED',len(list(dst.glob('*.md'))),'reports')
if __name__=='__main__':main(Path(sys.argv[1]))
