"""Aggregate-only final evidence; no identities, thresholds or fitted function parameters."""
from pathlib import Path
import json,sys,shutil,collections,xml.etree.ElementTree as ET
import numpy as np,pandas as pd
from crforest.transparent import file_sha256 as sha,write_json
from experiments.transparent_cif_l1_v1.orchestrate import gate,MODELS

def table(d,cols=None):
 if cols is not None:d=d[cols]
 def f(x):return f'{x:.6g}' if isinstance(x,float) else str(x)
 return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(f(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'
def main(run):
 out=run/'reports';dest=Path(__file__).parent;m=pd.read_csv(out/'metrics_by_year_seed.csv');w=pd.read_csv(out/'warning_utility.csv');c=pd.read_csv(out/'native_complexity.csv');ci=pd.read_csv(out/'paired_differences.csv');wd=pd.read_csv(out/'weight_diagnostics.csv');pa=pd.read_csv(out/'candidate_pair_audit_summary.csv');roles=pd.read_csv(out/'role_support.csv');sim=pd.read_csv(out/'simulation_signal_recovery.csv');ebm=pd.read_csv(out/'ebm_selection.csv');er=pd.read_csv(out/'ebm_role_diagnostics.csv');calls=pd.read_csv(out/'real_execution_calls.csv')
 primary=m[(m.seed==-1)&(m.weighting=='record_equal')];warn=w[(w.seed==-1)&w.scope.isin(['2023','2024','pooled'])];pooled=primary[primary.scope=='pooled'];ensemble=c[c.seed==-1];paired=ci[(ci.seed==-1)&(ci.scope=='pooled')&ci.weighting_or_budget.eq('record_equal')&ci.metric.isin(['ap','ipcw_brier'])]
 ledger=json.loads((run/'audit/source_hashes_before.json').read_text());assert all(sha(p)==v for p,v in ledger.items());write_json(run/'audit/source_hashes_after.json',dict(files=len(ledger),all_unchanged=True,sha256=ledger))
 for name,h in json.loads((run/'audit/bootstrap_input_frozen.json').read_text()).items():assert sha(run/'bootstrap_input'/name)==h
 ranges=m[m.seed!=-1].groupby(['scope','model','weighting'])[['auc','ap','ipcw_brier','prediction_mean']].agg(['mean','min','max']);ranges.columns=['_'.join(x) for x in ranges.columns];ranges.reset_index().to_csv(out/'seed_ranges.csv',index=False)
 stages=[];costs=[]
 for stage,n in [('demo',1),('train_oliver',2),('train_lemon',1),('predict',3),('native_audit',24),('bootstrap_recovery',10)]:
  paths=gate(run,stage,n);rep=next((run/'audit').glob(stage+'.jsonl.results-*.json'));r=json.loads(rep.read_text());stages.append(dict(stage=stage,tasks=n,seconds=r['seconds'],hosts=dict(collections.Counter(x['host'] for x in r['completed'])),publication_verified=True,report_sha256=sha(rep)))
  for path in paths:
   q=json.loads((path/'summary.json').read_text())
   if 'peak_rss_MiB' in q:costs.append(dict(stage=stage,**{k:q[k] for k in ['seed','model','host','seconds','peak_rss_MiB','affinity_count']}))
 pd.DataFrame(costs).to_csv(out/'task_costs.csv',index=False)
 bootstrap_failed_report=next((run/'audit').glob('bootstrap.jsonl.results-*.json'));bootstrap_original=json.loads(bootstrap_failed_report.read_text())
 initial=json.loads((run/'audit/test_calls.json').read_text());final=json.loads((run/'audit/final_test_calls.json').read_text());cli0=json.loads((run/'audit/initial_test_cli_receipts.json').read_text());cli1=[x for x in final if x['kind']=='unit_cli_pipeline'];test_counts={}
 for label,rows,cli in [('initial',initial,[x['receipt'] for x in cli0]),('final',final,[x['receipt'] for x in cli1])]:
  count=collections.Counter((x['kind'],x['status']) for x in rows);test_counts[label]=dict(inprocess_successful_forests=count['forest_fit','SUCCESS'],cli_successful_forests=sum(x['base_forest_fits'] for x in cli),rejected_inprocess_fit_calls=count['forest_fit','RAISED'],real_InterpretML_fits=count['ebm_fit','SUCCESS']+sum(x['ebm_candidate_fits'] for x in cli),weight_solves=count['weight_solve','SUCCESS']+sum(x['weight_qp_solves'] for x in cli),independent_R_synthetic_forests=1,cli_receipts=len(cli))
 tests={}
 for label,name in [('initial','tests.xml'),('final','final_tests.xml')]:
  root=ET.parse(run/'audit'/name).getroot();s=root.find('testsuite') if root.tag=='testsuites' else root;tests[label]={k:int(s.get(k,0)) for k in ['tests','failures','errors','skipped']};assert tests[label]['failures']==tests[label]['errors']==tests[label]['skipped']==0
 realfits=int(((calls.kind=='forest_fit')&(calls.status=='SUCCESS')).sum());qps=int(((calls.kind=='weight_solve')&(calls.status=='SUCCESS')).sum());ebmfits=int(((calls.kind=='ebm_fit')&(calls.status=='SUCCESS')).sum());assert (realfits,qps,ebmfits)==(3,9,18)
 demo=gate(run,'demo',1)[0];dc=json.loads((demo/'calls.json').read_text());dcounters=dict(collections.Counter(x['kind'] for x in dc if x['status']=='SUCCESS'))
 frozen=json.loads((run/'audit/code_freeze.json').read_text());source=Path('/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1')
 summary=dict(status='COMPLETED_PROVISIONAL_POST_EXPOSURE_EXPLORATORY',private_run=str(run),branch='research/transparent-cif-l1-v1',actual_training_sha=frozen['training_sha'],algorithm_reference='fd4a87aee54c70d5e4d165b1e7fa7011fc43c5db',L1_reference='526948fd5a11694831bac47090a8fe3d94089004',real_seed_runs=3,real_base_forest_fits=realfits,real_base_trees=96,selected_pair_trees=int(pa.selected_pairs.sum()),ordinary_added_trees=48,constructed_pair_candidate_trees=int(pa.constructed_candidate_trees.sum()),candidate_pairs=int(pa.candidate_pairs.sum()),candidate_partition_evaluations=int(pa.evaluated_partitions.sum()),candidate_supported_partitions=int(pa.supported_partitions.sum()),weight_solves=qps,weight_converged=int(wd.status.eq('CONVERGED').sum()),weight_max_FW_gap=float(wd.frank_wolfe_gap.max()),transparent_seed_models=18,transparent_ensemble_models=6,ebm_candidate_fits=ebmfits,ebm_selected_models=6,ebm_selected_rounds=sorted(ebm.selected_rounds.unique().tolist()),prediction_tasks=3,prediction_sets_including_references=44,formal_training_failures=0,formal_training_retries=0,pilot_fits=0,bootstrap_numeric_replicates=1000,bootstrap_original_batches=10,bootstrap_wrapper_failed_tasks=10,bootstrap_recovery_publication_tasks=10,bootstrap_recomputed_replicates=0,bootstrap_failure='duplicate exclusive-create summary.json after numerical computation; source scratch verified and copied through new pool publication, no resampling',bootstrap_precompletion_collect_attempts=1,bootstrap_failed_job_gate_attempts=1,lemon_submission_refusals=1,lemon_submission_computed_tasks=0,synthetic_demo_successful_calls=dcounters,synthetic_demo_comparisons=27,synthetic_oracle_comparisons=9,unit_test_fit_accounting=test_counts,test_results=tests,full_native_scalar_checks=int(c.scalar_checks.sum()),max_weighted_tree_error=float(c.weighted_tree_max_error.max()),max_explicit_all_function_error=float(c.explicit_all_function_max_error.max()),max_reload_error=float(c.reload_error.max()),independent_deployment_models=24,source_files_unchanged=len(ledger),source_identity_verified=True,training_records=56458,training_groups=2577,target_events=174,competing_events=34,evaluation_records=23799,evaluation_target_events=79,roles=roles.to_dict('records'),no_evaluation_outcomes_in_training=True,all_fits_frozen_before_evaluation=True,core_algorithm_unchanged=True,reference_models_retrained=False,patch_reapplied=False,stages=stages,bootstrap_verification=json.loads((run/'audit/bootstrap_check.json').read_text()),frozen_config_sha256=sha(run/'input/frozen_config.json'),predeclared_inference_sha256=sha(run/'input/inference_protocol.json'),decision='Complete transparent deployment verified; no reliable independent predictive gain for joint pair candidates or global weights; preserve all negative/mixed evidence',no_second_search=True)
 summary['bootstrap_original_seconds']=bootstrap_original['seconds'];summary['bootstrap_failed_report_sha256']=sha(bootstrap_failed_report)
 summary['probability_checks']=dict(minimum_cif=float(c.minimum_cif.min()),maximum_cause_sum=float(c.maximum_cause_sum.max()),minimum_time_increment=float(c.minimum_time_increment.min()),tolerance=1e-10,clipping=False)
 write_json(out/'execution_receipt.json',summary)
 hashes=dict(source=ledger,predictions=pd.read_csv(out/'model_provenance.csv').fillna('').to_dict('records'),code_freeze=frozen,bootstrap_recovery_identity_sha256=sha(run/'recovery_input/identity_verification.json'),standalone_receipt_sha256=sha(run/'standalone_deployment/deployment_receipt.json'))
 write_json(out/'prediction_and_source_hashes.json',hashes)
 def doc(name,text):(out/name).write_text(text.strip()+'\n')
 caveat='数据为PROVISIONAL；2023/2024是后暴露探索性比较，不能称独立盲测。原财务披露版本PIT、行业历史时钟和47项未决事件限制保留。'
 doc('PROTOCOL_ZH.md',f'''# 冻结L1上的透明CIF有限实验协议

本轮完整执行已有六方案×三个种子，检验联合成对候选与全局凸权重是否保留有效信号并改善预测/预警。{caveat}

算法参考fd4a87a；算法分支实际tip b6eab6a；L1交接526948f。核心core/functions/partition/pair_candidates/transparent/run/ebm_control/config逐字节等于算法参考。后续demo仅加主效应幅度与oracle收据记录；评价仅补显式版本化参考读入。实际训练SHA：{frozen['training_sha']}。最终交付提交另见Git历史，不混称训练SHA。

结构/内部留出公司角色固定role_seed=20261028、20%留出。森林参数取L1 plan森林q2，32树、深度6、每树25个声明输入、shared_quantile、mtry75，原叶支持门槛不变。75坐标的低阶约束不等于所有派生指标底层报表量都至多二阶。

固定pair:64对、最多16棵、每轴最多3切点、min_score0、seed20261028；weights:ridge0.001、max_iterations3000、gap1e-8。种子42/101/202；EBM主效应和10交互只在内部留出比较50/200/500轮，未把留出混回重训。无试运行/新增搜索。

六方案：base_equal/base_weighted、ordinary_extra_equal/ordinary_extra_weighted、pair_equal/pair_weighted。基础树完整保留，同银行等权/加权复用完全相同的树；ordinary新增数量匹配实际入选pair树。全局非负权重对企业、时间、原因全部共用。训练目标为结构资料估计删失权重的多状态Brier加固定L2，评价主表是目标原因Brier，二者不同；没有AP损失或新校准器。

拟合信息截止2022-01-01；三个seed全拟合结束、哈希冻结才预测与统一评价。训练入口设有禁止读eval元数据的守卫；原预测入口只读取X/身份/原点/期限，未读结局。评价使用原完整23,799行、原日历月提前量及未知保护。

统计规则已在训练前冻结inference_protocol.json。主口径三seed概率平均后的记录等权；公司等权及seed范围单列。主要机制比较pair_weighted−ordinary_extra_weighted，另报告三项权重作用、等权pair对ordinary、总体变化及与全训练L1 q2实用差距。所有1,000次共同公司bootstrap保持预测、名单、IPCW估计固定，保留抽样多重性。全部区间为未调整、探索性95%百分位区间；不择显著端点宣布成功，不是重训不确定性或非劣性证明。

执行入口prepare.py核验只读L1，orchestrate.py freeze/demo/train/predict/audit/bootstrap组织分阶段任务；postprocess.py调用已有评价器；collect.py汇总、deploy_and_ebm.py独立部署；report.py生成汇总。对现有完成目录不再应用补丁或重交拟合。bootstrap包装器故障的无重算恢复见recover_bootstrap.py和收据。
''')
 doc('DATA_AND_ROLE_AUDIT_ZH.md',f'''# L1与公司角色合同

唯一训练源：`{source}`。没有重新应用补丁、改标签、换样本、添加特征或扩展资料审计。训练56,458条/2,577组/75坐标，目标642窗口/174事件、竞争129窗口/34事件；所有计数由数据核对。X、缺失、顺序、身份、实际随访、365/366日窗口与冻结L0非标签部分逐值一致。

{table(roles)}

六方案、三个seed的structure/weight数组逐项一致，公司无交叉。留出只有32目标、6竞争独立事件，是估计支持限制；有事件不等于估计稳定。这是历史公司隔离留出，不是前推时间验证。未根据后期结果重抽角色。

label_eligible继承目标/存活二元视图，event>0或label_eligible构成端点资格；未知结局保护保留。结构资料估计权重学习的删失分布；所有原因目标/IPCW/权重均由L1本轮重新计算，未复用L0或全训练缓存。

评价记录23,799、跨年去重目标79；2023/2024分别11,686/12,113行。原legacy日历月口径为origin≤公告日−DateOffset(months=m)，不改为固定90/180天。训练前只核验eval容器哈希，全部模型冻结后才统一打开评价结局。

L1全训练q2、不限阶数参考显式读入reference_manifest.json并核对sample_id/group/SHA；M2只读原全行业预测，无参考模型重训。不能将这些全训练模型当作本轮角色匹配控制。源文件{len(ledger)}个哈希全部不变。

{caveat} 特征名称/单位/定义及缺失语义随完整模型保存在服务器，未知单位继续UNVERIFIED；完整透明并不补足历史数据认证。
''')
 doc('TRAINING_DIAGNOSTICS_ZH.md',f'''# 执行、收敛与故障账本

实际3个seed主运行；3次基础森林拟合=96棵基础树；普通新增48棵，入选成对48棵。成对搜索共192候选对、{int(pa.evaluated_partitions.sum())}次分区评价，{int(pa.constructed_candidate_trees.sum())}棵支持合格候选树被构造后按既定上限选择。共9次权重求解、18份单seed透明预测器及6份无重训集成。不是18次独立森林重训。

真实EBM候选18次、选中6个，全部选50轮；没有内部留出再拟合。固定demo另有3基础森林、9标签权重和9真值容量求解，27比较全部保留。单元测试合成fit、子进程CLI fit、R和真实InterpretML计数见execution_receipt.json；初始177项、最终183项全部通过，无skip，均包含独立R/Parquet/真实InterpretML。

{table(pd.DataFrame(stages),['stage','tasks','seconds','hosts'])}

seed42/101在oliver，202在lemon，单seed串行1CPU/BLAS1，未声称单进程跨机或GPU参与。两节点原隔离环境未修改。lemon申请1物理CPU/8GiB，原pool负责oliver affinity。完整原生核验/独立bootstrap批次使用两节点池。各阶段时间是实际阶段墙钟，不等于总端到端或可相加的拟合CPU。原bootstrap数值阶段墙钟{bootstrap_original["seconds"]:.3f}秒，另列恢复发布时间，不能把恢复耗时当作全部bootstrap耗时。逐任务耗时与峰值RSS见task_costs.csv。

{table(pa,['seed','candidate_pairs','evaluated_partitions','supported_partitions','unsupported_pairs','selected_pairs','pair_search_seconds','ordinary_extension_seconds'])}

普通和成对总树数都48，但搜索方式/成本不同；不称相同算力预算。失败支持的候选没有通过放宽门槛补足。

{table(wd,['seed','bank','status','iterations','cleanup_iterations','frank_wolfe_gap','objective','uniform_objective','effective_tree_count','largest_weight','constraint_residual'])}

9/9收敛，最大FW gap {wd.frank_wolfe_gap.max():.6g}≤1e-8。单seed有效树数约8.45—13.02，最大权重约0.207：有集中，但没有单树独占；内优化收敛不保证后期泛化。原显式浮点单纯形修复保留于收据，不裁剪预测。

故障完整保留：lemon-only快捷包装器拒绝1次（未计算），改用单节点Slurm并复用原storage发布协议，未改系统或pool。lemon初始非共享cwd警告后，runner在共享冻结代码目录实际执行。bootstrap10批均数值完成，但外层重复创建summary.json导致退出1；旧报告/FAILED/原scratch保留。逐份验证原节点SHA、100个rep编号、统一列/点估计，再经10个新发布任务复制恢复，重算bootstrap0、重训0。曾有一次过早汇总和一次失败任务gate拒绝，日志均保留；包装器已修复并增加回归测试。
''')
 doc('PAIR_AND_WEIGHT_ABLATION_ZH.md',f'''# 成对候选与全局权重的独立贡献

本轮没有证据支持将联合成对候选或全局权重升级为整体更优机制。所有树/角色匹配及数值收敛已通过，因此当前不利结果不能归咎于未运行或优化失败。{caveat}

{table(pooled[pooled.model.isin(MODELS)],['model','auc','ap','ipcw_brier'])}

{table(paired,['contrast','metric','point_difference','conditional_ci_lower','conditional_ci_upper'])}

主要机制pair_weighted−ordinary_extra_weighted：AP −0.011122，条件区间[−0.029085,0.000721]；Brier +0.000090，区间[0.000070,0.000110]。AP缺乏正向增量证据，Brier方向不利。pair_equal对同树数ordinary_equal的AP也较低，不能将单纯加树收益归为联合成对机制。

权重作用依赖银行：base/pair加权合并AP未改善且Brier更高；ordinary加权AP仅+0.000536、Brier约−0.000006，区间均含0。不能说权重普遍改进；也不能把区间含0解释为等效。pair_weighted相对base_equal的AP−0.008055、Brier+0.000063；不是只有Brier改善、排名没改善，而是该总体方案两项合并点估计均不利。

年度方向混合：pair_equal相对ordinary_equal在2023 AP更高（0.080625对0.071132），2024更低（0.076866对0.099365）。pair_weighted对ordinary_weighted在2023几乎相同，2024较低。三seed概率平均与单seed统计分别交付，未挑seed或年度。

所有区间均按训练前声明未作多重比较调整，只作固定预测的探索性条件评价；保留全部方向，不据单个端点显著性宣布方法首创或总体优越。
''')
 doc('SIMULATION_SIGNAL_RECOVERY_ZH.md',f'''# 固定合成的幅度与容量检查

原demo设置完全沿用：3,000样本、8个二值坐标、种子42/101/202、固定角色，真目标CIF=0.18+0.04*x0+0.05*x1*x2，竞争概率0.06。仅此一个既有DGP，无新场景/种子选择。真实主/交互系数分别0.04/0.05，对应二值反差尺度须一致比较。

{table(sim.groupby(['bank','aggregation'],as_index=False)[['main_amplitude_ratio','pair_amplitude_ratio','uniform_grid_mse','effective_trees']].mean())}

标签学习的成对扩展：主效应恢复约63.56%、交互21.31%；相较普通扩展的61.45%和15.72%，平均两者同时略有改善，并非仅把一种损失转移给另一种。但这远未实现完整幅度恢复。

容量问题与估计问题分开：成对银行即使用真CIF的乐观权重，平均仍只恢复主效应60.32%、交互71.30%；有限凸银行未实现全信号叠加。标签学习的交互恢复又远低于容量诊断，说明可实现容量与有限标签下实际学得的权重存在差距。不能据此唯一归因于某一个噪声/删失因素。

ORACLE_TRUE_CIF_CAPACITY只用模拟真值，不能部署、不能混入真实模型成绩。9个oracle和全部27条seed×bank×aggregation结果在simulation_signal_recovery.csv完整保留，含预测误差与结构树数；原pair支持审计和权重收据留在私有demo目录。合成工程通过后继续完成真实比较，没有为不利幅度更改配置。
''')
 doc('PREDICTION_AND_WARNING_RESULTS_ZH.md',f'''# 真实预测与企业预警效用

主表为三个seed概率平均后重算、记录等权。公司等权/每seed/范围/校准见CSV；合并AP不是年度AP平均。{caveat}

{table(primary,['scope','model','auc','ap','ipcw_brier','prediction_mean','observed_ipcw_incidence'])}

全训练L1 q2、不限阶数及M2只为实用参考。有限内部选择的EBM0在本轮合并AP0.139283、Brier0.009357，优于六个透明CIF方案的这些点估计；但没有竞争风险完整曲线，不能混为同一输出类型。

{table(warn[warn.scope=='pooled'],['model','top_fraction','alert_records','alert_company_groups','hit_company_groups','company_hit_fraction','detected_events','announcement_detected_3m','announcement_detected_6m','unknown_outcome_alerts'])}

Top20各模型季度记录预算合计同为4,761；去重核查公司可不同。base_equal为61/79事件、核查1168家；pair_weighted为60/79、1182家；ordinary_extra_weighted为62/79、1181家。主要机制比较的企业命中占比约5.08%对5.25%，未出现以较少核查获取更好覆盖的实用优势。

公告前3/6个月独立机会分母各60（年度25/35），窗口内总事件分母79。pair_weighted与ordinary_extra_weighted提前3个月43对44、6个月42对41，不能只挑后者称及时覆盖全面提升。Top25分别66/79对66/79，但提前6个月46对48。完整年度/季度、已知无目标、其中竞争、未知与重复负担在warning_utility.csv。

命中企业按去重公司另算；同一首次事件不会因多季度重复算成功。未知结局保留核查负担，不当正常或误报。状态按公司优先分成已命中/未命中但有未知/未命中且预警窗口均已知无目标；记录级类别另行对账。未在事件子集重排Top。

所有1000次共同公司bootstrap保持完整名单固定；年度/合并/AP/Brier/企业命中/事件及3、6月覆盖的七组配对比较完整交付paired_differences.csv。0差值或区间触0不是未来保证。
''')
 doc('TRANSPARENCY_AND_COMPLEXITY_ZH.md',f'''# 完整透明与真实部署规模

18份单seed和6份三seed概率集成均已导出完整主函数/二元交互，无森林即可执行CIF。对全部23,799评价输入、原生网格及365/366日边界、两个原因，逐块比较独立逐树固定权重、原生函数预测、全部函数表显式求和及保存重载。合计{int(c.scalar_checks.sum()):,}标量检查，逐树最大误差{c.weighted_tree_max_error.max():.3g}，显式求和{c.explicit_all_function_max_error.max():.3g}，重载0；未放宽1e-10门槛，未裁剪概率。实际最小CIF={c.minimum_cif.min():.3g}，最大原因和={c.maximum_cause_sum.max():.12g}，最小相邻时间增量={c.minimum_time_increment.min():.3g}；微小负值属于浮点误差，均在原容差内，未改为零。

在不含core.py和树银行的独立python -I目录部署24个函数包，实际端点、全缺失输入及完整解释均通过。模型算术为CIF概率贡献，不是风险率贡献、因果效应或已识别经济机制。

{table(ensemble,['model','trees','effective_trees','main_functions','interaction_functions','retained_main_functions','retained_interaction_functions','spatial_cells','native_time_scalar_cells','function_file_bytes','complete_table_file_bytes'])}

active函数统计使用旧编译器数值定义，retained是完整结构函数并集。加权可使某些贡献数值为零，但本轮未删树、未删函数格，不能把active计数当作实际部署表规模缩小。base集成保留33+79函数/2786格；pair为34+81/2800格，约增加0.5%；ordinary扩展36+100/3546格，约增加27.3%。无数量级爆炸，但仍逾百结构函数与百万时间×原因标量格，不等于易于人工阅读。

每个模型的真实切点、全部缺失格、逐格联合记录/公司/独立目标和竞争事件支持、全部函数值和案例保存在私有目录。低支持/零支持格全部保留；无损因子文件字节数与逻辑表规模分开。完整字段名称、单位、定义与转换合同随包保存；未认证单位不编造。

加权单seed有效树数8.45—13.02、最大权重约20.7%，有明显集中而非单树独占。三seed集成也实际编译共同分区、重做支持和全样本核验，不用平均单seed复杂度冒充集成。仅描述三个固定seed，没有追加重训稳定性实验，也不能从精确重构推断统计可辨识或普遍稳定。
''')
 doc('EBM_CONTROL_VALIDITY_ZH.md',f'''# 有限内部选择EBM的有效比较范围

真实InterpretML 0.7.8执行，主效应/10交互×50/200/500轮×3seed共18候选，选中6个均为50轮。所有候选Brier与选择记录完整保留；不再使用原2000轮无验证结果冒充强基线。与森林共用结构训练/内部公司留出，无交叉，所有结局早于2022-01-01；未知权重0，不混回留出重训，不用2023/2024选轮数。

{table(ebm)}

{table(er.groupby(['model','role'],as_index=False)[['auc','ap','ipcw_brier','prediction_mean','gt99pct']].mean())}

EBM0平均训练AP约0.444、内部约0.314，后期概率集成AP0.139；EBM10约0.538/0.309，后期0.130。存在拟合与泛化落差，交互版更强的训练拟合没有转为后期优势。跨期数据/发生率变化也会影响指标，不能仅据该差距唯一诊断过拟合；更不能说有限选择后EBM仍失败或已充分调优。

纯主效应EBM后期AP/Brier点估计较强，是必须保留的有限可解释端点对照。它用完整log-odds函数经logistic链接重构概率，本轮预测入口对全部评价记录核验该加和；没有完整竞争风险时间曲线，不能冒称与原生CIF概率分解完全相同。InterpretML缺失图展示警告被保留，缺失得分保存在原term_scores模型，未以图形省略代替完整模型。
''')
 doc('METHOD_DECISION_ZH.md',f'''# 研究判断：透明部署成立，新增机制未获独立预测增益支持

1. 六方案×三个seed实际完成且严格配对：3基础森林、48普通新增树、48入选成对树、9收敛权重求解、18单seed透明预测器；不是18次全森林独立拟合。
2. 成对候选相对同树数普通扩展没有可靠独立增量证据。主比较合并AP差−0.011122、Brier差+0.000090，后者条件区间为正；少数年度/AP或提前量的正向变化不构成整体成功。
3. 全局权重没有普遍独立收益。base/pair加权Brier变差；ordinary的微小AP/Brier改善区间均含0。优化已收敛，不能把泛化不佳写成工程失败。
4. 合成中平均主效应与交互均略改善，但只恢复约63.6%和21.3%；真值容量诊断也未全幅恢复。没有解决完整信号保留，更非真实经济机制恢复。
5. pair_weighted没有共同改善AP、企业命中和3/6月提前量。Top20相对ordinary_weighted少2个窗口事件、少1个3月捕获、多1个6月捕获，权衡明确保留。
6. 不能概括为“仅Brier改善、排名未改善”：主要新方案的合并AP与Brier点估计都比匹配ordinary_weighted差。只有普通扩展权重有极小且不确定的Brier改善。
7. 完整透明性与独立部署均保持。pair集成实际保留34主函数+81交互、2800空间格、1,299,200原生时间×原因标量；低支持与零支持单元不删除。
8. 权重集中至单seed有效约8.45—13.02树，最大单树约20.7%；不是单树独占，但与32/48树等权有明显区别。192候选对中10对没有合格分区，均如实保留。表规模未数量级爆炸，但仍不能称人易理解。
9. 有限EBM内部选择有效执行且没有测试调参；纯主效应EBM后期AP/Brier较强。训练/内部/后期差距提示泛化限制，不能宣称EBM全面失败或充分调优。其输出是端点logit模型，不是完整竞争风险CIF。
10. 后续研究保留base_equal作角色匹配控制、全训练L1 q2作实用透明参考，并保留ordinary扩展与EBM0作为必要比较。pair_weighted和全局权重不升级为推荐默认，也不因ordinary_weighted在这次后暴露数据某项最好就正式选定新胜者。是否再研究需另行授权；本轮没有第二轮搜索。

整体框架价值是完整CIF函数可见、可执行和可逐格审计；本轮没有证明新增候选/权重机制的独立方法创新已成功，也没有测试可辨识性惩罚。{caveat} 全部条件区间未调整多重比较，不能作因果、未来部署或非劣性承诺。
''')
 for p in out.iterdir():
  if p.suffix in ['.csv','.md','.json']:shutil.copy2(p,dest/p.name)
 for name in ['frozen_config.json','inference_protocol.json']:assert sha(dest/name)==sha(run/'input'/name)
 write_json(dest/'environment.json',{name:json.loads((run/'environment'/f'{name}.json').read_text()) for name in ['oliver','lemon']})
 write_json(dest/'algorithm_identity.json',json.loads((run/'audit/algorithm_identity.json').read_text()))
 print('Created nine reports, aggregate tables and complete execution receipt.')
if __name__=='__main__':main(Path(sys.argv[1]))
